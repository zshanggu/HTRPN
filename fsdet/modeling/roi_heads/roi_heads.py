# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from typing import Dict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from fsdet.layers import ShapeSpec
from fsdet.structures import Boxes, Instances, pairwise_iou
from fsdet.utils.events import get_event_storage
from fsdet.utils.registry import Registry
import fvcore.nn.weight_init as weight_init

from ..backbone import build_backbone
from ..backbone.resnet import BottleneckBlock, make_stage
from ..box_regression import Box2BoxTransform
from ..matcher import Matcher
from ..poolers import ROIPooler
from ..proposal_generator.proposal_utils import add_ground_truth_to_proposals, add_ternary_ground_truth_to_proposals
from ..sampling import subsample_labels
from .box_head import build_box_head
from .fast_rcnn import (
    FastRCNNOutputLayers,
    FastRCNNOutputs,
    FastRCNNContrastOutputs,
    FastRCNNMoCoOutputs,
    ContrastWithPrototypeOutputs,
    ContrastOutputsWithStorage,
    ROI_HEADS_OUTPUT_REGISTRY,
)
from ..utils import concat_all_gathered, select_all_gather
from ..contrastive_loss import (
    SupConLoss,
    SupConLossV2,
    ContrastiveHead,
    SupConLossWithPrototype,
    SupConLossWithStorage
)
from fsdet.layers import cat
from fsdet.utils.events import EventStorage

from fsdet.modeling.proposal_generator.rpn_outputs import rpn_losses, ternary_rpn_losses

ROI_HEADS_REGISTRY = Registry("ROI_HEADS")
ROI_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""

logger = logging.getLogger(__name__)


def build_roi_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)


def select_foreground_proposals(proposals, bg_label):
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


class ROIHeads(torch.nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It contains logic of cropping the regions, extract per-region features,
    and make per-region predictions.

    It can have many variants, implemented as subclasses of this class.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super(ROIHeads, self).__init__()

        # fmt: off
        self.batch_size_per_image     = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        self.test_score_thresh        = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.test_nms_thresh          = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_detections_per_img  = cfg.TEST.DETECTIONS_PER_IMAGE
        self.in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes              = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.proposal_append_gt       = cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT
        self.feature_strides          = {k: v.stride for k, v in input_shape.items()}
        self.feature_channels         = {k: v.channels for k, v in input_shape.items()}
        self.cls_agnostic_bbox_reg    = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        self.smooth_l1_beta           = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA
        # fmt: on

        # Matcher to assign box proposals to gt boxes
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )

        # Box2BoxTransform for bounding box regression
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)

    def _sample_proposals(self, matched_idxs, matched_labels, gt_classes):
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]  # post_nms_top_k proposals have no matche will be drop here
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_sample_fraction, self.num_classes
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns `self.batch_size_per_image` random samples from proposals and groundtruth boxes,
        with a fraction of positives that is no larger than `self.positive_sample_fraction.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:
                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                   then the ground-truth box is random)
                Other fields such as "gt_classes" that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            # use ground truth bboxes as super-high quality proposals for training
            # with logits = math.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            # matched_idxs in [0, M)
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            iou, _ = match_quality_matrix.max(dim=0)
            # random sample batche_size_per_image proposals with positive fraction
            # NOTE: only matched proposals will be returned
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes
            proposals_per_image.iou = iou[sampled_idxs]

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # will filter the proposals again (by foreground/background,
                # etc), so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes
            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        with EventStorage():
            storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt
        # proposals_with_gt, List[Instances], fields = ['gt_boxes', 'gt_classes', ‘proposal_boxes’, 'objectness_logits']

    def forward(self, images, features, proposals, targets=None):
        """
        Args:
            images (ImageList):
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`s. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:
                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].

        Returns:
            results (list[Instances]): length `N` list of `Instances`s containing the
                detected instances. Returned during inference only; may be []
                during training.
            losses (dict[str: Tensor]): mapping from a named loss to a tensor
                storing the loss. Used during training only.
        """
        raise NotImplementedError()


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where the heads share the
    cropping and the per-region feature computation by a Res5 block.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        assert len(self.in_features) == 1

        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / self.feature_strides[self.in_features[0]], )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.res5, out_channels = self._build_res5_block(cfg)
        output_layer = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg, out_channels, self.num_classes, self.cls_agnostic_bbox_reg
        )

    def _build_res5_block(self, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = make_stage(
            BottleneckBlock,
            3,
            first_stride=2,
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        return self.res5(x)  # RoI Align 之后的 feature 进入 res5

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images

        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        pred_class_logits, pred_proposal_deltas = self.box_predictor(feature_pooled)
        del feature_pooled

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )

        if self.training:
            del features
            losses = outputs.losses()
            return [], losses
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return pred_instances, {}


@ROI_HEADS_REGISTRY.register()
class StandardROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(StandardROIHeads, self).__init__(cfg, input_shape)
        self._init_box_head(cfg)

    def _init_box_head(self, cfg):
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        self.output_layer_name = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(self.output_layer_name)(
            cfg, self.box_head.output_size, self.num_classes, self.cls_agnostic_bbox_reg
        )

    def forward(self, images, features, proposals, targets=None, alg_iter=0):
        """
        See :class:`ROIHeads.forward`.
            proposals (List[Instance]): fields=[proposal_boxes, objectness_logits]
                post_nms_top_k proposals for each image， len = N

            targets (List[Instance]):   fields=[gt_boxes, gt_classes]
                gt_instances for each image, len = N
        """
        del images
        if self.training:
            # label and sample 256 from post_nms_top_k each images
            # has field [proposal_boxes, objectness_logits ,gt_classes, gt_boxes]
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        features_list = [features[f] for f in self.in_features]

        if self.training:
            # FastRCNNOutputs.losses()
            # {'loss_cls':, 'loss_box_reg':}
            losses = self._forward_box(features_list, proposals, alg_iter)  # get losses from fast_rcnn.py::FastRCNNOutputs
            return proposals, losses  # return to rcnn.py line 201
        else:
            pred_instances = self._forward_box(features_list, proposals)
            return pred_instances, {}

    def _forward_box(self, features, proposals, alg_iter=0):
        """
        Forward logic of the box prediction branch.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])  # [None, 256, POOLER_RESOLU, POOLER_RESOLU]
        box_features = self.box_head(box_features)  # [None, FC_DIM]
        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)
        del box_features

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )
        if self.training:
            return outputs.losses()
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return pred_instances


@ROI_HEADS_REGISTRY.register()
class TernaryHiercContraROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(TernaryHiercContraROIHeads, self).__init__(cfg, input_shape)
        self._init_box_head(cfg)

        self.fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        self.mlp_head_dim = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.MLP_FEATURE_DIM
        self.temperature = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.TEMPERATURE
        self.contrast_loss_weight = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.LOSS_WEIGHT
        self.box_reg_weight       = cfg.MODEL.ROI_BOX_HEAD.BOX_REG_WEIGHT
        self.weight_decay         = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.ENABLED
        self.decay_steps          = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.STEPS
        self.decay_rate           = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.RATE
        self.loss_version = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.LOSS_VERSION
        self.contrast_iou_thres = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.IOU_THRESHOLD
        self.reweight_func = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.REWEIGHT_FUNC
        self.loss_weight             = cfg.MODEL.RPN.LOSS_WEIGHT
        self.cl_head_only = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.HEAD_ONLY
        # Contrastive Loss head
        self.encoder = ContrastiveHead(self.fc_dim, self.mlp_head_dim)
        self.RPN_encoder = ContrastiveHead(self.fc_dim, self.mlp_head_dim)
        if self.loss_version == 'V1':
            self.criterion = SupConLoss(self.temperature, self.contrast_iou_thres, self.reweight_func)
            self.RPN_criterion = SupConLoss(self.temperature, self.contrast_iou_thres, self.reweight_func)
        elif self.loss_version == 'V2':
            self.criterion = SupConLossV2(self.temperature, self.contrast_iou_thres)
            self.RPN_criterion = SupConLossV2(self.temperature, self.contrast_iou_thres)
        self.criterion.num_classes = self.num_classes  # to be used in protype version
        self.RPN_criterion.num_classes = 3  # to be used in protype version
        # f = open("/sgzy_fsce_test2/media/zeyu/8T/AI_MODEL/few_shot_learning/CODES/fsod/results/novel_imgs.txt", "r")
        # self.novel_file = f.read()

        self.is_rpn_contra = cfg.MODEL.PROPOSAL_GENERATOR.IS_CONTRAST

    def _init_box_head(self, cfg):
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.proposal_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        self.proposal_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        self.output_layer_name = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(self.output_layer_name)(
            cfg, self.box_head.output_size, self.num_classes, self.cls_agnostic_bbox_reg
        )
        self.potential_threshold = cfg.MODEL.PROPOSAL_GENERATOR.POTENTIAL_THRESHOLD


    @torch.no_grad()
    def ternary_label_and_sample_proposals(self, proposals, targets):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns `self.batch_size_per_image` random samples from proposals and groundtruth boxes,
        with a fraction of positives that is no larger than `self.positive_sample_fraction.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:
                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                   then the ground-truth box is random)
                Other fields such as "gt_classes" that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            # use ground truth bboxes as super-high quality proposals for training
            # with logits = math.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))
            proposals = add_ternary_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            # matched_idxs in [0, M)
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            iou, _ = match_quality_matrix.max(dim=0)
            # random sample batche_size_per_image proposals with positive fraction
            # NOTE: only matched proposals will be returned
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes
            proposals_per_image.iou = iou[sampled_idxs]

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # will filter the proposals again (by foreground/background,
                # etc), so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes
            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        with EventStorage():
            storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt
        # proposals_with_gt, List[Instances], fields = ['gt_boxes', 'gt_classes', ‘proposal_boxes’, 'objectness_logits']

    def forward(self, images, features, proposals, targets=None, alg_iter=0, for_calculate_loss=None, batched_inputs=None):
        """
        See :class:`ROIHeads.forward`.
            proposals (List[Instance]): fields=[proposal_boxes, objectness_logits]
                post_nms_top_k proposals for each image， len = N

            targets (List[Instance]):   fields=[gt_boxes, gt_classes]
                gt_instances for each image, len = N
        """

        if self.training:
            # label and sample 256 from post_nms_top_k each images
            # has field [proposal_boxes, objectness_logits ,gt_classes, iou, gt_boxes]
            proposals = self.ternary_label_and_sample_proposals(proposals, targets)
            # print("roi_heads.py proposals......", proposals)  # IMS_PER_BATCH / num_gpus = images, 256 instances per image
        del targets

        features_list = [features[f] for f in self.in_features]

        if self.training:
            # FastRCNNOutputs.losses()
            # {'loss_cls':, 'loss_box_reg':}
            losses = self._forward_box(features_list, proposals, alg_iter)  # get losses from fast_rcnn.py::FastRCNNOutputs (sub classification)
            rpn_losses = self._forward_rpn_box(features_list, proposals, alg_iter, for_calculate_loss, batched_inputs, images)
            del images
            return proposals, losses, rpn_losses  # return to rcnn.py line 201
        else:
            pred_instances = self._forward_box(features_list, proposals)
            del images
            return pred_instances, {}

    def _forward_box(self, features, proposals, alg_iter=0):
        """
        Forward logic of the box prediction branch.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)  # [None, FC_DIM]
        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)

        box_features_contrast = self.encoder(box_features)
        del box_features

        if self.weight_decay:
            with EventStorage():
                storage = get_event_storage()
            if int(storage.iter) in self.decay_steps:
                self.contrast_loss_weight *= self.decay_rate

        outputs = FastRCNNContrastOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            box_features_contrast,
            self.criterion,
            self.contrast_loss_weight,
            self.box_reg_weight,
            self.cl_head_only,
        )
        if self.training:
            return outputs.losses()
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return pred_instances

    def supervised_contrastive_loss(self, feature_con, gt_classes, ious):
        contrastive_loss = self.RPN_criterion(feature_con, gt_classes, ious)
        return contrastive_loss

    def _forward_rpn_box(self, features, proposals, alg_iter=0, for_calculate_loss=None, batched_inputs=None, images=None):
        """
        Forward logic of the box prediction branch.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """

        # print("roi_heads.py!!!!!!!!!!! num_neg_hierc", for_calculate_loss["num_neg_hierc"]) #  [[50, 50, 50, 51, 51], [50, 50, 50, 51, 51]]
        # print("roi_heads.py!!!!!!!!!!! for_calculate_loss", len(for_calculate_loss["neg_anchor_flatten"]))  # list of 2
        # print("roi_heads.py!!!!!!!!!!! for_calculate_loss", len(for_calculate_loss["neg_anchor_flatten"][0]), len(for_calculate_loss["neg_anchor_flatten"][1]))  # list of 252, batch of 2
        test_box_features = self.box_pooler(features, for_calculate_loss["neg_anchor_flatten"])
        # print("roi_heads.py!!!!!!!!!!! test_box_features1", len(test_box_features))  # 463 batch1 + batch 2
        test_box_features = self.box_head(test_box_features)  # [None, FC_DIM]
        # del proposal_features
        # print("roi_heads.py!!!!!!!!!!! test_box_features2", test_box_features.shape)  # torch.Size([463, 1024])
        test_pred_class_logits, test_pred_proposal_deltas = self.box_predictor(test_box_features)
        # print("roi_heads.py!!!!!!!!!!! test_pred_class_logits", test_pred_class_logits)
        # test_pred_class_logits_zero = torch.zeros_like(test_pred_class_logits)
        # test_pred_class_logits = torch.where(test_pred_class_logits < 0.5, test_pred_class_logits_zero, test_pred_class_logits)
        test_pred_class_concat = []
        for test_pred_class_logit in test_pred_class_logits:
            if len(test_pred_class_logit[test_pred_class_logit>self.potential_threshold]) > 0:  # confidence 0.8
                test_pred_class_concat.append(test_pred_class_logit.argmax(dim=0).item())
            else:
                test_pred_class_concat.append(-1)
        # test_pred_class_concat = torch.tensor(test_pred_class_concat)
        # print("roi_heads.py!!!!!!!!!!! test_pred_class", test_pred_class_concat)  # 487 list
        # print("roi_heads.py!!!!!!!!!!! neg_idx_srtd", for_calculate_loss["neg_idx_srtd"])
        # print("roi_heads.py!!!!!!!!!!! neg_idx_srtd", len(for_calculate_loss["neg_idx_srtd"][0]), len(for_calculate_loss["neg_idx_srtd"][1]))  # 2 = 244 + 250

        test_pred_class_hierc = []
        sum_hierc_start = 0
        sum_hierc_end = 0
        for i in range(len(for_calculate_loss["neg_idx_srtd"])):
            sum_hierc_end += len(for_calculate_loss["neg_idx_srtd"][i])
            test_pred_class_hierc.append(torch.tensor(test_pred_class_concat[sum_hierc_start:sum_hierc_end]))
            sum_hierc_start += len(for_calculate_loss["neg_idx_srtd"][i])
        # print("roi_heads.py!!!!!!!!!!! test_pred_class_hierc", test_pred_class_hierc)  # 2 list
        # print("roi_heads.py!!!!!!!!!!! test_pred_class_hierc", len(test_pred_class_hierc[0]), len(test_pred_class_hierc[1]))  # 2 list

        neg_idx_srtd_fixed = []
        for i in range(len(test_pred_class_hierc)):
            test_pred_class_hierc_i = torch.eq(test_pred_class_hierc[i]>=0, test_pred_class_hierc[i]<self.num_classes)
            # print("roi_heads.py!!!!!!!!!!!!!!!!test_pred_class_hierc_i", test_pred_class_hierc_i)
            neg_idx_srtd_fixed.append(for_calculate_loss["neg_idx_srtd"][i][test_pred_class_hierc_i])
        # print("roi_heads.py!!!!!!!!!!!!!!!!neg_idx_srtd_fixed", neg_idx_srtd_fixed)  # 2 list

        # print("roi_heads.py!!!!!!!!!!!!!!!!gt_objectness_logits", len(for_calculate_loss["gt_objectness_logits"]))  # 2 list = batch 1 + batch 2

        for i in range(len(neg_idx_srtd_fixed)):
            for j in range(len(neg_idx_srtd_fixed[i])):
                # print("roi_heads.py!!!!!!!!!!!!!!!!gt_objectness_logits ij", for_calculate_loss["gt_objectness_logits"][i][neg_idx_srtd_fixed[i][j]])
                assert for_calculate_loss["gt_objectness_logits"][i][neg_idx_srtd_fixed[i][j]] == 0
                for_calculate_loss["gt_objectness_logits"][i][neg_idx_srtd_fixed[i][j]] = 2  # potential object

        # Split to tuple of L tensors, each with shape (N, num_anchors_per_map)
        for_calculate_loss["gt_objectness_logits"] = torch.split(for_calculate_loss["gt_objectness_logits"], for_calculate_loss["num_anchors_per_map"], dim=1)
        # Concat from all feature maps
        for_calculate_loss["gt_objectness_logits"] = cat([x.flatten() for x in for_calculate_loss["gt_objectness_logits"]], dim=0)
        # print("gt_objectness_logits[0] cat([x.flatten()!!!!!!!!!!!!!!!!", for_calculate_loss["gt_objectness_logits"].size())  # batch[0]:p2+p3+p4+p5+p6 + batch[1]:p2+p3+p4+p5+p6 torch.Size([257796])

        # sum_hierc_objectness = 0
        # for i in range(len(neg_idx_srtd_fixed)):
        #     print("roi_heads.py!!!!!!!!!!!!!!!!neg_idx_srtd_fixed[i]", len(neg_idx_srtd_fixed[i]))
        #     for j in range(len(neg_idx_srtd_fixed[i])):
        #         print("roi_heads.py!!!!!!!!!!!!!!!!j", i, j,
        #               for_calculate_loss["gt_objectness_logits"][neg_idx_srtd_fixed[i][j] + sum_hierc_objectness])
        #         assert for_calculate_loss["gt_objectness_logits"][neg_idx_srtd_fixed[i][j] + sum_hierc_objectness] == 0
        #         for_calculate_loss["gt_objectness_logits"][neg_idx_srtd_fixed[i][j] + sum_hierc_objectness] = 1
        #     sum_hierc_objectness += len(neg_idx_srtd_fixed[i])

        if self.is_rpn_contra:
            proposal_features = self.proposal_pooler(features, [p.proposal_boxes for p in proposals])
            proposal_features = self.proposal_head(proposal_features)
            proposal_features_contrast = self.RPN_encoder(proposal_features)
            # print("roi_heads.py!!!!!!!!!!!!!!!!proposals", proposals[0]._fields['gt_classes'])
            # print("roi_heads.py!!!!!!!!!!!!!!!!proposal_features_contrast", proposal_features_contrast.shape)  # torch.Size([[1024, 128]])
            contra_gt_class = torch.cat([x._fields['gt_classes'] for x in proposals], dim=0)
            # print("roi_heads.py!!!!!!!!!!!!!!!!contra_gt_class", contra_gt_class.shape)  # torch.Size([1024])
            contra_iou = torch.cat([x._fields['iou'] for x in proposals], dim=0)
            # print("roi_heads.py!!!!!!!!!!!!!!!!contra_iou.shape", contra_iou.shape)  # torch.Size([1024])
            loss_contrast = self.contrast_loss_weight * self.supervised_contrastive_loss(proposal_features_contrast, contra_gt_class, contra_iou)

        objectness_loss, localization_loss = ternary_rpn_losses(
            for_calculate_loss["gt_objectness_logits"],
            for_calculate_loss["gt_anchor_deltas"],
            for_calculate_loss["ternary_pred_objectness_logits"],
            for_calculate_loss["pred_anchor_deltas"],
            for_calculate_loss["smooth_l1_beta"],
        )

        # print("objectness_loss!!!!!!!!!!!!!!!!", objectness_loss)  # 357.6484
        normalizer = 1.0 / (for_calculate_loss["batch_size_per_image"] * for_calculate_loss["num_images"])
        loss_cls = objectness_loss * normalizer  # cls: classification loss
        loss_loc = localization_loss * normalizer  # loc: localization loss
        if self.is_rpn_contra:
            proposal_losses = {"loss_rpn_cls": loss_cls, "loss_rpn_loc": loss_loc, "loss_contrast": loss_contrast}
        else:
            proposal_losses = {"loss_rpn_cls": loss_cls, "loss_rpn_loc": loss_loc}

        proposal_losses = {k: v * self.loss_weight for k, v in proposal_losses.items()}

        if self.training:
            return proposal_losses



@ROI_HEADS_REGISTRY.register()
class MoCoROIHeadsV1(StandardROIHeads):
    """
    MoCo queue encoder is the roi box head only.
    """
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        # fmt: on
        self.momentum                  = cfg.MODEL.MOCO.MOMENTUM
        self.queue_size                = cfg.MODEL.MOCO.QUEUE_SIZE
        self.tao                       = cfg.MODEL.MOCO.TEMPERATURE
        self.mlp_dims                  = cfg.MODEL.MOCO.MLP_DIMS
        self.warmup_steps              = cfg.MODEL.MOCO.WARM_UP_STEPS
        self.cls_loss_weight           = cfg.MODEL.MOCO.CLS_LOSS_WEIGHT
        self.save_queue_iters          = cfg.MODEL.MOCO.SAVE_QUEUE_ITERS

        self.debug_deque_and_enque     = cfg.MODEL.MOCO.DEBUG_DEQUE_AND_ENQUE
        # fmt: off

        self._init_box_head(cfg)

    def _init_box_head(self, cfg):
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head_q = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        self.box_head_k = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        self.output_layer_name = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(self.output_layer_name)(
            cfg, self.box_head_q.output_size, self.num_classes, self.cls_agnostic_bbox_reg
        )

    def _moco_encoder_init(self, cfg, *args):
        # ROI Box Head
        for param_q, param_k in zip(self.box_head_q.parameters(),
                                    self.box_head_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # MLP head
        self.mlp_q = nn.Sequential(
            nn.Linear(1024, self.mlp_dims[0]),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.mlp_dims[1]),
        )
        for layer in self.mlp_q:
            if isinstance(layer, nn.Linear):
                weight_init.c2_xavier_fill(layer)
        self.mlp_k = nn.Sequential(
            nn.Linear(1024, self.mlp_dims[0]),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.mlp_dims[1]),
        )
        for param_q, param_k in zip(self.mlp_q.parameters(),
                                    self.mlp_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # MoCo queue
        self.register_buffer('queue', torch.randn(self.mlp_dims[1], self.queue_size))
        self.register_buffer('queue_label', torch.empty(self.queue_size).fill_(-1).long())
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer('cycles', torch.zeros(1))

    @torch.no_grad()
    def _momentum_update(self):
        for param_q, param_k in zip(self.box_head_q.parameters(),
                                    self.box_head_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1 - self.momentum)

        for param_q, param_k in zip(self.mlp_q.parameters(),
                                    self.mlp_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1 - self.momentum)


    def _forward_box(self, features, proposals):
        '''Args:
            proposals: 256 * 2 random sampled proposals w/ positive fraction
            features: List of L features
        '''
        q_box_features = self.box_pooler(features, [p.proposal_boxes for p in proposals]) # [None, 256, 7, 7]
        q_box_features = self.box_head_q(q_box_features) # [None, 1024]
        pred_class_logits, pred_proposal_deltas = self.box_predictor(q_box_features)

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )

        if not self.training:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return pred_instances

        # compute query features
        q_embedding = self.mlp_q(q_box_features)
        q = F.normalize(q_embedding)
        del q_box_features

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update() # update key encoder
            k_box_features = self.box_pooler(features, [p.proposal_boxes for p in proposals])
            k_box_features = self.box_head_k(k_box_features)
            k_embedding = self.mlp_k(k_box_features)
            k = F.normalize(k_embedding)
            del k_box_features

        _ = self._dequeue_and_enqueue(k, proposals)
        # q.shape = [None, 128], self.queue.shape = [128, S]
        moco_logits = torch.mm(q, self.queue.clone().detach()) # [None, queue size(S)]
        moco_logits /= self.tao

        with EventStorage():
            storage = get_event_storage()
        self.moco_loss_weight = min(storage.iter / self.warmup_steps, 1.0)
        if self.save_queue_iters and (storage.iter % self.save_queue_iters == 0):
            save_as = '/data/tmp/queue_{}.pth'.format(storage.iter)
            print('save moco queue to ', save_as)
            torch.save(self.queue, save_as)

        outputs = FastRCNNMoCoOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            moco_logits,       # [None, S]
            self.queue_label,  # [1, None]
            self.moco_loss_weight,
            self.cls_loss_weight,
        )
        return outputs.losses()

    @torch.no_grad()
    def _dequeue_and_enqueue(self, key, proposals):
        label = torch.cat([p.gt_classes for p in proposals], dim=0)
        if self.debug_deque_and_enque:
            keys = key
            labels = label
        else:
            keys = concat_all_gathered(key)
            labels = concat_all_gathered(label)

        # 确认能通过
        # assert keys.shape[0] == labels.shape[0]
        # assert self.queue.shape[1] == self.queue_size

        batch_size = keys.shape[0]
        if self.queue_size % batch_size != 0:
            print()
            print(self.queue_ptr, self.cycles, batch_size, self.queue.shape)
            print()

        ptr = int(self.queue_ptr)
        cycles = int(self.cycles)
        if ptr + batch_size <= self.queue.shape[1]:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            self.queue_label[ptr:ptr + batch_size] = labels
        else:
            rem = self.queue.shape[1] - ptr
            self.queue[:, ptr:ptr + rem] = keys[:rem, :].T
            self.queue_label[ptr:ptr + rem] = labels[:rem]

        ptr += batch_size
        if ptr >= self.queue.shape[1]:
            ptr = 0
            cycles += 1
        self.cycles[0] = cycles
        self.queue_ptr[0] = ptr
        return cycles


@ROI_HEADS_REGISTRY.register()
class MoCoROIHeadsV3(MoCoROIHeadsV1):
    """
    MoCo v2: contrastive encoder is composed of backbone and roi
    """
    def _moco_encoder_init(self, cfg, *args):
        # ResNet-FPN backbone
        self.backbone_q, = args
        self.backbone_k = build_backbone(cfg)
        for param_q, param_k in zip(self.backbone_q.parameters(),
                                    self.backbone_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # ROI Box Head
        for param_q, param_k in zip(self.box_head_q.parameters(),
                                    self.box_head_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # MLP head
        self.mlp_q = nn.Sequential(
            nn.Linear(1024, self.mlp_dims[0]),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.mlp_dims[1]),
        )
        for layer in self.mlp_q:
            if isinstance(layer, nn.Linear):
                weight_init.c2_xavier_fill(layer)
        self.mlp_k = nn.Sequential(
            nn.Linear(1024, self.mlp_dims[0]),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.mlp_dims[1]),
        )
        for param_q, param_k in zip(self.mlp_q.parameters(),
                                    self.mlp_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # MoCo queue
        self.register_buffer('queue', torch.randn(self.mlp_dims[1], self.queue_size))
        self.register_buffer('queue_label', torch.empty(self.queue_size).fill_(-1).long())
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer('cycles', torch.zeros(1))

    @torch.no_grad()
    def _momentum_update(self):
        for param_q, param_k in zip(self.backbone_q.parameters(),
                                    self.backbone_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1 - self.momentum)

        for param_q, param_k in zip(self.box_head_q.parameters(),
                                    self.box_head_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1 - self.momentum)

        for param_q, param_k in zip(self.mlp_q.parameters(),
                                    self.mlp_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1 - self.momentum)

    def forward(self, images, features, proposals, targets=None):
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        features_list = [features[f] for f in self.in_features]
        if self.training:
            losses = self._forward_box(images, features_list, proposals)
            return proposals, losses  # return to rcnn.py line 201
        else:
            pred_instances = self._forward_box(images, features_list, proposals)
            return pred_instances, {}

    def _forward_box(self, images, features, proposals):
        '''Args:
            proposals: 256 * 2 random sampled proposals w/ positive fraction
            features: List of L features
        '''
        q_box_features = self.box_pooler(features, [p.proposal_boxes for p in proposals]) # [None, 256, 7, 7]
        q_box_features = self.box_head_q(q_box_features) # [None, 1024]
        pred_class_logits, pred_proposal_deltas = self.box_predictor(q_box_features)

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )

        if not self.training:
            del images
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return pred_instances

        # compute query features
        q_embedding = self.mlp_q(q_box_features)
        q = F.normalize(q_embedding)
        del q_box_features

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update() # update key encoder
            k_features = self.backbone_k(images.tensor)
            k_features = [k_features[fpn_lvl] for fpn_lvl in self.in_features]
            del images
            k_box_features = self.box_pooler(k_features, [p.proposal_boxes for p in proposals])
            del k_features
            k_box_features = self.box_head_k(k_box_features)
            k_embedding = self.mlp_k(k_box_features)
            k = F.normalize(k_embedding)
            del k_box_features

        _ = self._dequeue_and_enqueue(k, proposals)
        # q.shape = [None, 128], self.queue.shape = [128, S]
        moco_logits = torch.mm(q, self.queue.clone().detach()) # [None, queue size(S)]
        moco_logits /= self.tao

        with EventStorage():
            storage = get_event_storage()
        self.moco_loss_weight = min(storage.iter / self.warmup_steps, 1.0)
        if self.save_queue_iters and (storage.iter % self.save_queue_iters == 0):
            save_as = '/data/tmp/queue_{}.pth'.format(storage.iter)
            print('save moco queue to ', save_as)
            torch.save(self.queue, save_as)

        outputs = FastRCNNMoCoOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            moco_logits,       # [None, S]
            self.queue_label,  # [1, None]
            self.moco_loss_weight,
            self.cls_loss_weight,
        )
        return outputs.losses()


@ROI_HEADS_REGISTRY.register()
class DoubleHeadROIHeads(StandardROIHeads):
    """
    Implementation of Double Head Faster-RCNN(https://arxiv.org/pdf/1904.06493.pdf).
    Support supervised contrastive learning (https://arxiv.org/pdf/2004.11362.pdf)

    Components that I implemented for this head are:
        modeling.roi_heads.roi_heads.DoubleHeadROIHeads (this class)
        modeling.roi_heads.box_head.FastRCNNDoubleHead  (specify this name in yaml)
        modeling.fast_rcnn.FastRCNNDoubleHeadOutputLayers
        modeling.backbone.resnet.BasicBlock
    """
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        # fmt: off
        self.contrastive_branch    = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.ENABLED
        self.fc_dim                = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        self.mlp_head_dim          = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.MLP_FEATURE_DIM
        self.temperature           = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.TEMPERATURE

        self.contrast_loss_weight  = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.LOSS_WEIGHT
        self.fg_proposals_only     = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.FG_ONLY
        self.cl_head_only          = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.HEAD_ONLY
        # fmt: on

        # Contrastive Loss head
        if self.contrastive_branch:
            self.encoder = ContrastiveHead(self.fc_dim, self.mlp_head_dim)
            self.criterion = SupConLoss(self.temperature, self.fg_proposals_only)

    def _forward_box(self, features, proposals):
        """
        Forward logic of the box prediction branch.

        Box regression branch: 1Basic -> 4BottleNeck -> GAP
        Box classification branch: flatten -> fc1 -> fc2 (unfreeze fc2 is doen in rcnn.py)
                                                      | self.head (ConstrastiveHead)
                                                      ∨
                                               Contrastive Loss

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_loc_feat, box_cls_feat = self.box_head(box_features)
        del box_features
        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_loc_feat, box_cls_feat)

        if self.contrastive_branch:
            box_cls_feat_contrast = self.encoder(box_cls_feat)  # feature after contrastive head
            outputs = FastRCNNContrastOutputs(
                self.box2box_transform,
                pred_class_logits,
                pred_proposal_deltas,
                proposals,
                self.smooth_l1_beta,
                box_cls_feat_contrast,
                self.criterion,
                self.contrast_loss_weight,
                self.fg_proposals_only,
                self.cl_head_only,
            )
        else:
            outputs = FastRCNNOutputs(
                self.box2box_transform,
                pred_class_logits,  # cls_logits and box_deltas returned from OutputLayer
                pred_proposal_deltas,
                proposals,
                self.smooth_l1_beta,
            )

        if self.training:
            return outputs.losses()
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return pred_instances


@ROI_HEADS_REGISTRY.register()
class NovelRoiHeads(StandardROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        # fmt: off
        self.contrastive_branch    = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.ENABLED
        self.fc_dim                = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        self.mlp_head_dim     = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.MLP_FEATURE_DIM
        self.temperature           = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.TEMPERATURE

        self.contrast_loss_weight  = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.LOSS_WEIGHT
        self.fg_proposals_only     = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.FG_ONLY
        self.cl_head_only          = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.HEAD_ONLY
        # fmt: on

        # Contrastive Loss head
        if self.contrastive_branch:
            self.encoder = ContrastiveHead(self.fc_dim, self.mlp_head_dim)
            self.criterion = SupConLoss(self.temperature, self.fg_proposals_only)

    def _forward_box(self, features, proposals):
        """
        Forward logic of the box prediction branch.

        Box regression branch: 1Basic -> 4BottleNeck -> GAP
        Box classification branch: flatten -> fc1 -> fc2 (unfreeze fc2 is doen in rcnn.py)
                                                      | self.head (ConstrastiveHead)
                                                      ∨
                                               Contrastive Loss

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_loc_feat, box_cls_feat = self.box_head(box_features)
        del box_features
        if self.output_layer_name == 'FastRCNNDoubleHeadCosMarginLayers':
            gt_classes = None
            if proposals[0].has('gt_classes'):
                gt_classes = cat([p.gt_classes for p in proposals], dim=0)
            pred_class_logits, pred_proposal_deltas = self.box_predictor(box_loc_feat, box_cls_feat, gt_classes, self.training)
        else:
            pred_class_logits, pred_proposal_deltas = self.box_predictor(box_loc_feat, box_cls_feat)

        if self.contrastive_branch:
            box_cls_feat_contrast = self.encoder(box_cls_feat)  # feature after contrastive head
            outputs = FastRCNNContrastOutputs(
                self.box2box_transform,
                pred_class_logits,
                pred_proposal_deltas,
                proposals,
                self.smooth_l1_beta,
                box_cls_feat_contrast,
                self.criterion,
                self.contrast_loss_weight,
                self.fg_proposals_only,
                self.cl_head_only,
            )
        else:
            outputs = FastRCNNOutputs(
                self.box2box_transform,
                pred_class_logits,  # cls_logits and box_deltas returned from OutputLayer
                pred_proposal_deltas,
                proposals,
                self.smooth_l1_beta,
            )

        if self.training:
            return outputs.losses()
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return pred_instances


@ROI_HEADS_REGISTRY.register()
class ContrastiveROIHeads(StandardROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        # fmt: on
        self.fc_dim               = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        self.mlp_head_dim         = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.MLP_FEATURE_DIM
        self.temperature          = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.TEMPERATURE
        self.contrast_loss_weight = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.LOSS_WEIGHT
        self.box_reg_weight       = cfg.MODEL.ROI_BOX_HEAD.BOX_REG_WEIGHT
        self.weight_decay         = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.ENABLED
        self.decay_steps          = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.STEPS
        self.decay_rate           = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.RATE

        self.num_classes          = cfg.MODEL.ROI_HEADS.NUM_CLASSES

        self.loss_version         = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.LOSS_VERSION
        self.contrast_iou_thres   = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.IOU_THRESHOLD
        self.reweight_func        = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.REWEIGHT_FUNC

        self.cl_head_only         = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.HEAD_ONLY
        # fmt: off

        self.encoder = ContrastiveHead(self.fc_dim, self.mlp_head_dim)
        if self.loss_version == 'V1':
            self.criterion = SupConLoss(self.temperature, self.contrast_iou_thres, self.reweight_func)
        elif self.loss_version == 'V2':
            self.criterion = SupConLossV2(self.temperature, self.contrast_iou_thres)
        self.criterion.num_classes = self.num_classes  # to be used in protype version

    def _forward_box(self, features, proposals, alg_iter=0):
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)  # [None, FC_DIM]
        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)
        box_features_contrast = self.encoder(box_features)
        del box_features

        if self.weight_decay:
            with EventStorage():
                storage = get_event_storage()
            if int(storage.iter) in self.decay_steps:
                self.contrast_loss_weight *= self.decay_rate

        outputs = FastRCNNContrastOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            box_features_contrast,
            self.criterion,
            self.contrast_loss_weight,
            self.box_reg_weight,
            self.cl_head_only,
        )
        if self.training:
            return outputs.losses()
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return pred_instances


@ROI_HEADS_REGISTRY.register()
class ContrastiveROIHeadsWithStorage(StandardROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        # fmt: on
        self.fc_dim               = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        self.mlp_head_dim         = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.MLP_FEATURE_DIM
        self.temperature          = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.TEMPERATURE
        self.contrast_loss_weight = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.LOSS_WEIGHT
        self.box_reg_weight       = cfg.MODEL.ROI_BOX_HEAD.BOX_REG_WEIGHT
        self.weight_decay         = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.ENABLED
        self.decay_steps          = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.STEPS
        self.decay_rate           = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.RATE

        self.num_classes          = cfg.MODEL.ROI_HEADS.NUM_CLASSES

        self.contrast_iou_thres   = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.IOU_THRESHOLD
        self.reweight_func        = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.REWEIGHT_FUNC

        self.use_storage          = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.STORAGE.ENABLED
        self.queue_size           = cfg.MODEL.MOCO.QUEUE_SIZE
        self.storage_start_iter   = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.STORAGE.START
        self.storage_threshold    = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.STORAGE.IOU_THRESHOLD
        # fmt: off

        # momentum encoding queue
        self.register_buffer('queue', torch.randn(self.queue_size, self.mlp_head_dim))
        self.register_buffer('queue_label', torch.empty(self.queue_size).fill_(-1).long())
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.encoder = ContrastiveHead(self.fc_dim, self.mlp_head_dim)
        self.criterion = SupConLossWithStorage(self.temperature, self.contrast_iou_thres)
        self.criterion.num_classes = self.num_classes  # to be used in protype version

    def _forward_box(self, features, proposals):
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)  # [None, FC_DIM]
        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)
        box_features_contrast = self.encoder(box_features)
        del box_features

        if self.use_storage and self.training:
            event = get_event_storage()
            if int(event.iter) >= self.storage_start_iter:
                iou = cat([p.iou for p in proposals])
                label = cat([p.gt_classes for p in proposals], dim=0)
                idx = (iou >= self.storage_threshold).long()
                self._dequeue_and_enqueue(box_features_contrast, label, idx)

        if self.weight_decay:
            with EventStorage():
                storage = get_event_storage()
            if int(storage.iter) in self.decay_steps:
                self.contrast_loss_weight *= self.decay_rate

        outputs = ContrastOutputsWithStorage(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            box_features_contrast,
            self.criterion,
            self.contrast_loss_weight,
            self.queue,
            self.queue_label,
            self.box_reg_weight,
        )
        if self.training:
            return outputs.losses()
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return pred_instances

    @torch.no_grad()
    def _dequeue_and_enqueue(self, key, label, idx):
        keys = select_all_gather(key, idx)
        labels = select_all_gather(label, idx)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        if ptr + batch_size <= self.queue.shape[0]:
            self.queue[ptr:ptr + batch_size] = keys
            self.queue_label[ptr:ptr + batch_size] = labels
        else:
            rem = self.queue.shape[1] - ptr
            self.queue[ptr:ptr + rem] = keys[:rem, :]
            self.queue_label[ptr:ptr + rem] = labels[:rem]

        ptr += batch_size
        if ptr >= self.queue.shape[0]:
            ptr = 0
        self.queue_ptr[0] = ptr


@ROI_HEADS_REGISTRY.register()
class ContrastiveROIHeadsWithPrototype(StandardROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        # fmt: on
        self.temperature          = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.TEMPERATURE
        self.contrast_loss_weight = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.LOSS_WEIGHT
        self.box_reg_weight       = cfg.MODEL.ROI_BOX_HEAD.BOX_REG_WEIGHT
        self.box_cls_weight       = cfg.MODEL.ROI_BOX_HEAD.BOX_CLS_WEIGHT
        self.weight_decay         = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.ENABLED
        self.decay_steps          = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.STEPS
        self.decay_rate           = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.RATE

        self.num_classes          = cfg.MODEL.ROI_HEADS.NUM_CLASSES

        self.prototype_dataset    = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.PROTOTYPE.DATASET
        self.prototype_path       = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.PROTOTYPE.PATH
        # fmt: off

        # self.encoder = ContrastiveHead(self.fc_dim, self.mlp_head_dim)

        prototype_tensor = torch.load(self.prototype_path)  # [num_classes+1, 1024]
        prototype_tensor = prototype_tensor[:-1, :]  # [num_classes, 1024]
        if self.prototype_dataset == 'PASCAL VOC':
            assert prototype_tensor.shape == (15, 1024)
            prototype_label = torch.arange(15)
        else:
            raise(NotImplementedError, 'prototype not implemented for non-VOC dataset')
        self.register_buffer('prototype', prototype_tensor)
        self.register_buffer('prototype_label', prototype_label)

        self.criterion = SupConLossWithPrototype(self.temperature)
        self.criterion.num_classes = self.num_classes

    def _forward_box(self, features, proposals):
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)  # [None, FC_DIM]
        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)

        box_features_normalized = F.normalize(box_features)
        del box_features

        if self.weight_decay:
            with EventStorage():
                storage = get_event_storage()
            if int(storage.iter) in self.decay_steps:
                self.contrast_loss_weight *= self.decay_rate

        outputs = ContrastWithPrototypeOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            box_features_normalized,
            self.prototype,
            self.prototype_label,
            self.criterion,
            self.contrast_loss_weight,
            self.box_reg_weight,
            self.box_cls_weight,
        )
        if self.training:
            return outputs.losses()
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return pred_instances


@ROI_HEADS_REGISTRY.register()
class ContrastiveROIHeadsPrototypeWithMLP(ContrastiveROIHeadsWithPrototype):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        # fmt: on
        self.fc_dim                      =  cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        self.mlp_head_dim                =  cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.MLP_FEATURE_DIM
        self.temperature                 =  cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.TEMPERATURE

        self.box_reg_weight              =  cfg.MODEL.ROI_BOX_HEAD.BOX_REG_WEIGHT
        self.box_cls_weight              =  cfg.MODEL.ROI_BOX_HEAD.BOX_CLS_WEIGHT

        self.contrast_loss_weight        =  cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.LOSS_WEIGHT
        self.weight_decay                =  cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.ENABLED
        self.decay_steps                 =  cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.STEPS
        self.decay_rate                  =  cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.RATE

        self.num_classes                 =  cfg.MODEL.ROI_HEADS.NUM_CLASSES

        self.prototype_dataset           =  cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.PROTOTYPE.DATASET
        self.prototype_path              =  cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.PROTOTYPE.PATH
        self.disable_prototype_mlp_grad  =  cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.PROTOTYPE.DISABLE_PROTOTYPE_GRAD
        # fmt: off

        # mlp head, return L-2 normalized features
        self.encoder = ContrastiveHead(self.fc_dim, self.mlp_head_dim)

        # prototype obtained from base training
        prototype_tensor = torch.load(self.prototype_path)  # [num_classes+1, 1024]
        prototype_tensor = prototype_tensor[:-1, :]  # [num_classes, 1024]
        if self.prototype_dataset == 'PASCAL VOC':
            assert prototype_tensor.shape == (15, 1024)
            prototype_label = torch.arange(15)
        else:
            raise(NotImplementedError, 'prototype not implemented for non-VOC dataset')
        self.register_buffer('prototype', prototype_tensor)
        self.register_buffer('prototype_label', prototype_label)

        # Supervised Contrastive Loss With Prototype
        self.criterion = SupConLossWithPrototype(self.temperature)
        self.criterion.num_classes = self.num_classes

    def _forward_box(self, features, proposals):
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)  # [None, FC_DIM]
        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)

        box_features = F.normalize(box_features)
        box_features_enc = self.encoder(box_features)
        del box_features

        if self.disable_prototype_mlp_grad:
            with torch.no_grad():
                proto_features_enc = self.encoder(self.prototype)
        else:
            proto_features_enc = self.encoder(self.prototype)

        if self.weight_decay:
            with EventStorage():
                storage = get_event_storage()
            if int(storage.iter) in self.decay_steps:
                self.contrast_loss_weight *= self.decay_rate

        outputs = ContrastWithPrototypeOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            box_features_enc,
            proto_features_enc,
            self.prototype_label,
            self.criterion,
            self.contrast_loss_weight,
            self.box_reg_weight,
            self.box_cls_weight,
        )
        if self.training:
            return outputs.losses()
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return pred_instances
