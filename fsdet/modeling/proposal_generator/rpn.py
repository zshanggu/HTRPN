# Copyrigh:t (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, List
import torch
import torch.nn.functional as F
from torch import nn

from fsdet.layers import ShapeSpec
from fsdet.utils.registry import Registry

from ..anchor_generator import build_anchor_generator
from ..box_regression import Box2BoxTransform
from ..matcher import Matcher
from .build import PROPOSAL_GENERATOR_REGISTRY
from .rpn_outputs import RPNOutputs, HiercOnlyRPNOutputs, TernaryHiercRPNOutputs, find_top_rpn_proposals, find_ternary_top_rpn_proposals

RPN_HEAD_REGISTRY = Registry("RPN_HEAD")
"""
Registry for RPN heads, which take feature maps and perform
objectness classification and bounding box regression for anchors.
"""


def build_rpn_head(cfg, input_shape):
    """
    Build an RPN head defined by `cfg.MODEL.RPN.HEAD_NAME`.
    """
    name = cfg.MODEL.RPN.HEAD_NAME
    return RPN_HEAD_REGISTRY.get(name)(cfg, input_shape)


@RPN_HEAD_REGISTRY.register()
class StandardRPNHead(nn.Module):
    """
    RPN classification and regression heads. Uses a 3x3 conv to produce a shared
    hidden state from which one 1x1 conv predicts objectness logits for each anchor
    and a second 1x1 conv predicts bounding-box deltas specifying how to deform
    each anchor into an object proposal.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()

        # Standard RPN is shared across levels:
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        # RPNHead should take the same input as anchor generator
        # NOTE: it assumes that creating an anchor generator does not have unwanted side effect.
        anchor_generator = build_anchor_generator(cfg, input_shape)
        num_cell_anchors = anchor_generator.num_cell_anchors
        box_dim = anchor_generator.box_dim
        assert (
            len(set(num_cell_anchors)) == 1
        ), "Each level must have the same number of cell anchors"
        num_cell_anchors = num_cell_anchors[0]

        # 3x3 conv for the hidden representation
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 1x1 conv for predicting objectness logits
        self.objectness_logits = nn.Conv2d(in_channels, num_cell_anchors, kernel_size=1, stride=1)
        # 1x1 conv for predicting box2box transform deltas
        self.anchor_deltas = nn.Conv2d(
            in_channels, num_cell_anchors * box_dim, kernel_size=1, stride=1
        )

        for l in [self.conv, self.objectness_logits, self.anchor_deltas]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): list of feature maps
        """
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for x in features:
            t = F.relu6(self.conv(x))
            pred_objectness_logits.append(self.objectness_logits(t))
            anchor_deltas_ = self.anchor_deltas(t)
            # if torch.isfinite(anchor_deltas_).all():
            #     print("rpn.py......", anchor_deltas_)
            anchor_deltas_ = torch.where(torch.isinf(anchor_deltas_).any(), torch.full_like(anchor_deltas_, 1),
                        anchor_deltas_)
            anchor_deltas_ = torch.where(torch.isnan(anchor_deltas_).any(), torch.full_like(anchor_deltas_, 0),
                        anchor_deltas_)

            pred_anchor_deltas.append(anchor_deltas_)
        return pred_objectness_logits, pred_anchor_deltas


@RPN_HEAD_REGISTRY.register()
class TernaryRPNHead(nn.Module):
    """
    RPN classification and regression heads. Uses a 3x3 conv to produce a shared
    hidden state from which one 1x1 conv predicts objectness logits for each anchor
    and a second 1x1 conv predicts bounding-box deltas specifying how to deform
    each anchor into an object proposal.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()

        # Standard RPN is shared across levels:
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        # RPNHead should take the same input as anchor generator
        # NOTE: it assumes that creating an anchor generator does not have unwanted side effect.
        anchor_generator = build_anchor_generator(cfg, input_shape)
        num_cell_anchors = anchor_generator.num_cell_anchors
        box_dim = anchor_generator.box_dim
        assert (
            len(set(num_cell_anchors)) == 1
        ), "Each level must have the same number of cell anchors"
        num_cell_anchors = num_cell_anchors[0]

        # 3x3 conv for the hidden representation
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 1x1 conv for predicting objectness logits
        # !!!!!!!!ternary classification: 0: is an object, 1: not an object, 2: potential object
        self.objectness_logits = nn.Conv2d(in_channels, num_cell_anchors * 3, kernel_size=1, stride=1)
        # 1x1 conv for predicting box2box transform deltas
        self.anchor_deltas = nn.Conv2d(
            in_channels, num_cell_anchors * box_dim, kernel_size=1, stride=1
        )

        for l in [self.conv, self.objectness_logits, self.anchor_deltas]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): list of feature maps
        """
        ternary_pred_objectness_logits = []
        pred_anchor_deltas = []
        for x in features:
            t = F.relu(self.conv(x))
            ternary_pred_objectness_logits.append(self.objectness_logits(t))
            anchor_deltas_ = self.anchor_deltas(t)
            anchor_deltas_ = torch.where(torch.isinf(anchor_deltas_).any(), torch.full_like(anchor_deltas_, 1),
                                         anchor_deltas_)
            anchor_deltas_ = torch.where(torch.isnan(anchor_deltas_).any(), torch.full_like(anchor_deltas_, 0),
                                         anchor_deltas_)
            pred_anchor_deltas.append(anchor_deltas_)

        # print("rpn.py!!!!!!!!!!!!ternary_pred_objectness_logits", len(ternary_pred_objectness_logits))  # 5
        # print("rpn.py!!!!!!!!!!!!ternary_pred_objectness_logits", ternary_pred_objectness_logits[0].shape)  # torch.Size([2, 9, 184, 248])
        # print("rpn.py!!!!!!!!!!!!pred_anchor_deltas", len(pred_anchor_deltas))  # 5
        # print("rpn.py!!!!!!!!!!!!pred_anchor_deltas", pred_anchor_deltas[0].shape)  # torch.Size([2, 12, 184, 248])

        return ternary_pred_objectness_logits, pred_anchor_deltas


@PROPOSAL_GENERATOR_REGISTRY.register()
class RPN(nn.Module):
    """
    Region Proposal Network, introduced by the Faster R-CNN paper.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        # fmt: off
        self.min_box_side_len        = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
        self.in_features             = cfg.MODEL.RPN.IN_FEATURES
        self.nms_thresh              = cfg.MODEL.RPN.NMS_THRESH
        self.batch_size_per_image    = cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE
        self.positive_fraction       = cfg.MODEL.RPN.POSITIVE_FRACTION
        self.smooth_l1_beta          = cfg.MODEL.RPN.SMOOTH_L1_BETA
        self.loss_weight             = cfg.MODEL.RPN.LOSS_WEIGHT

        self.cl_head_only            = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.HEAD_ONLY
        # fmt: on

        # Map from self.training state to train/test settings
        self.pre_nms_topk = {
            True: cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN,
            False: cfg.MODEL.RPN.PRE_NMS_TOPK_TEST,
        }
        self.post_nms_topk = {
            True: cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN,
            False: cfg.MODEL.RPN.POST_NMS_TOPK_TEST,
        }
        self.boundary_threshold = cfg.MODEL.RPN.BOUNDARY_THRESH

        self.anchor_generator = build_anchor_generator(
            cfg, [input_shape[f] for f in self.in_features]
        )
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        self.anchor_matcher = Matcher(
            cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS, allow_low_quality_matches=True
        )
        self.rpn_head = build_rpn_head(cfg, [input_shape[f] for f in self.in_features])

    def forward(self, images, features, gt_instances=None):
        """
        Args:
            images (ImageList): input images of length `N`
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
                Each `Instances` stores ground-truth instances for the corresponding image.

        Returns:
            proposals: list[Instances] or None
            loss: dict[Tensor]
        """
        gt_boxes = [x.gt_boxes for x in gt_instances] if gt_instances is not None else None
        del gt_instances
        features = [features[f] for f in self.in_features]
        # pred_objectness_logits: list of L tensor of shape [N, A, Hi, Wi]
        # pred_anchor_deltas: list of L tensor of shape [N, A*B, Hi, Wi]
        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        # print("pred_anchor_deltas", pred_anchor_deltas)
        anchors = self.anchor_generator(features)
        # TODO: The anchors only depend on the feature map shape; there's probably
        # an opportunity for some optimizations (e.g., caching anchors).
        outputs = RPNOutputs(
            self.box2box_transform,
            self.anchor_matcher,
            self.batch_size_per_image,
            self.positive_fraction,
            images,
            pred_objectness_logits,
            pred_anchor_deltas,
            anchors,
            self.boundary_threshold,
            gt_boxes,
            self.smooth_l1_beta,
        )

        if self.training and not self.cl_head_only:
            losses = {k: v * self.loss_weight for k, v in outputs.losses().items()}
        else:
            losses = {}

        with torch.no_grad():
            # Find the top proposals by applying NMS and removing boxes that
            # are too small. The proposals are treated as fixed for approximate
            # joint training with roi heads. This approach ignores the derivative
            # w.r.t. the proposal boxes’ coordinates that are also network
            # responses, so is approximate.
            proposals = find_top_rpn_proposals(
                outputs.predict_proposals(),  # transform anchors to proposals by applying delta
                outputs.predict_objectness_logits(),
                images,
                self.nms_thresh,
                self.pre_nms_topk[self.training],
                self.post_nms_topk[self.training],
                self.min_box_side_len,
                self.training,
            )
            # For RPN-only models, the proposals are the final output and we return them in
            # high-to-low confidence order.
            # For end-to-end models, the RPN proposals are an intermediate state
            # and this sorting is actually not needed. But the cost is negligible.
            # 但是要注意，end-to-end models 在后面进入 RoI 的 proposals 实际上会在 label_and_sample_proposals 再次被打乱，
            # 所以再以后用到的其实并不是按照 objectness 倒序排列的。
            inds = [p.objectness_logits.sort(descending=True)[1] for p in proposals]
            proposals = [p[ind] for p, ind in zip(proposals, inds)]


        return proposals, losses


@PROPOSAL_GENERATOR_REGISTRY.register()
class HiercOnlyRPN(nn.Module):
    """
    Region Proposal Network, introduced by the Faster R-CNN paper.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        # fmt: off
        self.min_box_side_len        = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
        self.in_features             = cfg.MODEL.RPN.IN_FEATURES
        self.nms_thresh              = cfg.MODEL.RPN.NMS_THRESH
        self.batch_size_per_image    = cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE
        self.positive_fraction       = cfg.MODEL.RPN.POSITIVE_FRACTION
        self.smooth_l1_beta          = cfg.MODEL.RPN.SMOOTH_L1_BETA
        self.loss_weight             = cfg.MODEL.RPN.LOSS_WEIGHT

        self.cl_head_only            = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.HEAD_ONLY
        # fmt: on

        # Map from self.training state to train/test settings
        self.pre_nms_topk = {
            True: cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN,
            False: cfg.MODEL.RPN.PRE_NMS_TOPK_TEST,
        }
        self.post_nms_topk = {
            True: cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN,
            False: cfg.MODEL.RPN.POST_NMS_TOPK_TEST,
        }
        self.boundary_threshold = cfg.MODEL.RPN.BOUNDARY_THRESH

        self.anchor_generator = build_anchor_generator(
            cfg, [input_shape[f] for f in self.in_features]
        )
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        self.anchor_matcher = Matcher(
            cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS, allow_low_quality_matches=True
        )
        self.rpn_head = build_rpn_head(cfg, [input_shape[f] for f in self.in_features])

    def forward(self, images, features, gt_instances=None, batched_inputs=None):
        """
        Args:
            images (ImageList): input images of length `N`
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
                Each `Instances` stores ground-truth instances for the corresponding image.

        Returns:
            proposals: list[Instances] or None
            loss: dict[Tensor]
        """
        gt_boxes = [x.gt_boxes for x in gt_instances] if gt_instances is not None else None
        del gt_instances
        features = [features[f] for f in self.in_features]
        # pred_objectness_logits: list of L tensor of shape [N, A, Hi, Wi]
        # pred_anchor_deltas: list of L tensor of shape [N, A*B, Hi, Wi]
        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        # print("rpn.py pred_objectness_logits!!!!!!!!!", len(pred_objectness_logits))
        # print("rpn.py pred_anchor_deltas!!!!!!!!!", len(pred_anchor_deltas))
        anchors = self.anchor_generator(features)
        # TODO: The anchors only depend on the feature map shape; there's probably
        # an opportunity for some optimizations (e.g., caching anchors).
        outputs = HiercOnlyRPNOutputs(
            self.box2box_transform,
            self.anchor_matcher,
            self.batch_size_per_image,
            self.positive_fraction,
            images,
            pred_objectness_logits,
            pred_anchor_deltas,
            anchors,
            self.boundary_threshold,
            gt_boxes,
            self.smooth_l1_beta,
        )

        if self.training and not self.cl_head_only:
            losses = {k: v * self.loss_weight for k, v in outputs.losses().items()}
        else:
            losses = {}

        with torch.no_grad():
            # Find the top proposals by applying NMS and removing boxes that
            # are too small. The proposals are treated as fixed for approximate
            # joint training with roi heads. This approach ignores the derivative
            # w.r.t. the proposal boxes’ coordinates that are also network
            # responses, so is approximate.
            proposals = find_top_rpn_proposals(
                outputs.predict_proposals(),  # transform anchors to proposals by applying delta
                outputs.predict_objectness_logits(),
                images,
                self.nms_thresh,
                self.pre_nms_topk[self.training],
                self.post_nms_topk[self.training],
                self.min_box_side_len,
                self.training,
            )
            # For RPN-only models, the proposals are the final output and we return them in
            # high-to-low confidence order.
            # For end-to-end models, the RPN proposals are an intermediate state
            # and this sorting is actually not needed. But the cost is negligible.
            # 但是要注意，end-to-end models 在后面进入 RoI 的 proposals 实际上会在 label_and_sample_proposals 再次被打乱，
            # 所以再以后用到的其实并不是按照 objectness 倒序排列的。
            inds = [p.objectness_logits.sort(descending=True)[1] for p in proposals]
            proposals = [p[ind] for p, ind in zip(proposals, inds)]


        return proposals, losses


@PROPOSAL_GENERATOR_REGISTRY.register()
class TernaryHiercRPN(nn.Module):
    """
    Region Proposal Network, introduced by the Faster R-CNN paper.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        # fmt: off
        self.min_box_side_len        = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
        self.in_features             = cfg.MODEL.RPN.IN_FEATURES
        self.nms_thresh              = cfg.MODEL.RPN.NMS_THRESH
        self.batch_size_per_image    = cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE
        self.positive_fraction       = cfg.MODEL.RPN.POSITIVE_FRACTION
        self.smooth_l1_beta          = cfg.MODEL.RPN.SMOOTH_L1_BETA

        self.cl_head_only            = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.HEAD_ONLY
        # fmt: on

        # Map from self.training state to train/test settings
        self.pre_nms_topk = {
            True: cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN,
            False: cfg.MODEL.RPN.PRE_NMS_TOPK_TEST,
        }
        self.post_nms_topk = {
            True: cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN,
            False: cfg.MODEL.RPN.POST_NMS_TOPK_TEST,
        }
        self.boundary_threshold = cfg.MODEL.RPN.BOUNDARY_THRESH

        self.anchor_generator = build_anchor_generator(
            cfg, [input_shape[f] for f in self.in_features]
        )
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        self.anchor_matcher = Matcher(
            cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS, allow_low_quality_matches=True
        )
        self.rpn_head = build_rpn_head(cfg, [input_shape[f] for f in self.in_features])
        self.fine_tuning = cfg.MODEL.PROPOSAL_GENERATOR.IS_FINE_TUNING

    def forward(self, images, features, gt_instances=None, batched_inputs=None):
        """
        Args:
            images (ImageList): input images of length `N`
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
                Each `Instances` stores ground-truth instances for the corresponding image.

        Returns:
            proposals: list[Instances] or None
            loss: dict[Tensor]
        """

        gt_boxes = [x.gt_boxes for x in gt_instances] if gt_instances is not None else None
        del gt_instances
        features = [features[f] for f in self.in_features]
        # ternary_pred_objectness_logits: list of L tensor of shape [N, A, Hi, Wi]
        # pred_anchor_deltas: list of L tensor of shape [N, A*B, Hi, Wi]
        ternary_pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        # print("rpn.py pred_objectness_logits!!!!!!!!!", len(ternary_pred_objectness_logits))
        # print("rpn.py pred_anchor_deltas!!!!!!!!!", len(pred_anchor_deltas))
        anchors = self.anchor_generator(features)

        # print("features[0]!!!!!!!!!!!!!!!!", features[0].size())  # p2 torch.Size([2, 256, 200, 200])
        # print("features[1]!!!!!!!!!!!!!!!!", features[1].size())  # p3 torch.Size([2, 256, 100, 100])
        # print("features[2]!!!!!!!!!!!!!!!!", features[2].size())  # p4 torch.Size([2, 256, 50, 50])
        # print("features[3]!!!!!!!!!!!!!!!!", features[3].size())  # p5 torch.Size([2, 256, 25, 25])
        # print("features[4]!!!!!!!!!!!!!!!!", features[4].size())  # p6 torch.Size([2, 256, 13, 13])

        # 3 anchors per pixel
        # print("anchors!!!!!!!!!!!!!!!!", len(anchors))  # batch: 2
        # print("anchors[0]!!!!!!!!!!!!!!!!", len(anchors[0]))  # layer: 5
        # print("anchors[0]!!!!!!!!!!!!!!!!", anchors[0][0].tensor.size())  # p2 torch.Size([120000, 4]) = features.width * features.height * 3
        # print("anchors[0]!!!!!!!!!!!!!!!!", anchors[0][1].tensor.size())  # p3 torch.Size([30000, 4])
        # print("anchors[0]!!!!!!!!!!!!!!!!", anchors[0][2].tensor.size())  # p4 torch.Size([7500, 4])
        # print("anchors[0]!!!!!!!!!!!!!!!!", anchors[0][3].tensor.size())  # p5 torch.Size([1875, 4])
        # print("anchors[0]!!!!!!!!!!!!!!!!", anchors[0][4].tensor.size())  # p6 torch.Size([507, 4])
        #
        # print("anchors[1]!!!!!!!!!!!!!!!!", anchors[1][0].tensor.size())
        # print("anchors[1]!!!!!!!!!!!!!!!!", anchors[1][1].tensor.size())
        # print("anchors[1]!!!!!!!!!!!!!!!!", anchors[1][2].tensor.size())
        # print("anchors[1]!!!!!!!!!!!!!!!!", anchors[1][3].tensor.size())
        # print("anchors[1]!!!!!!!!!!!!!!!!", anchors[1][4].tensor.size())
        # for i in range(anchors[0][0].tensor.size()[0]):
        #     start_point = (int(anchors[0][0].tensor[i][0].item()),
        #                    int(anchors[0][0].tensor[i][1].item()))
        #     end_point = (int(anchors[0][0].tensor[i][2].item()),
        #                  int(anchors[0][0].tensor[i][3].item()))
        #     cv2.rectangle(mat, start_point, end_point, (0, 0, 255), 2)
        #     cv2.putText(mat, format(gt_instances[0]._fields['gt_classes'][i].item(), '.3f'), start_point, cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)
        # cv2.imwrite("./results/test_potential_anchor/testimg_{}.jpg".format(now.strftime("%Y-%m-%d-%H-%M-%S")), mat)


        # TODO: The anchors only depend on the feature map shape; there's probably
        # an opportunity for some optimizations (e.g., caching anchors).
        outputs = TernaryHiercRPNOutputs(
            self.box2box_transform,
            self.anchor_matcher,
            self.batch_size_per_image,
            self.positive_fraction,
            images,
            ternary_pred_objectness_logits,
            pred_anchor_deltas,
            anchors,
            self.boundary_threshold,
            gt_boxes,
            self.smooth_l1_beta,
        )

        if self.training and not self.cl_head_only:
            # loss = {k: v * self.loss_weight for k, v in outputs.losses(images, batched_inputs).items()}
            for_calculate_loss = outputs.losses(images, batched_inputs)
        else:
            # loss = {}
            for_calculate_loss = {}

        with torch.no_grad():
            # Find the top proposals by applying NMS and removing boxes that
            # are too small. The proposals are treated as fixed for approximate
            # joint training with roi heads. This approach ignores the derivative
            # w.r.t. the proposal boxes’ coordinates that are also network
            # responses, so is approximate.
            proposals = find_ternary_top_rpn_proposals(
                outputs.predict_proposals(),  # transform anchors to proposals by applying delta
                outputs.predict_objectness_logits(),
                images,
                self.nms_thresh,
                self.pre_nms_topk[self.training],
                self.post_nms_topk[self.training],
                self.min_box_side_len,
                self.training,
                self.fine_tuning,
            )
            # For RPN-only models, the proposals are the final output and we return them in
            # high-to-low confidence order.
            # For end-to-end models, the RPN proposals are an intermediate state
            # and this sorting is actually not needed. But the cost is negligible.
            # for p in proposals:
            #     print("rpn.py!!!!!!!!!!!!!!!!!!! proposals", p.objectness_logits.shape)  # torch.Size([1000, 3])
            #     test = torch.randint(1, 100, (5, 3)) #
            #     print("rpn.py!!!!!!!!!!!!!!!!!!!!!!test", test) #
            #     test = test.sort(descending=True, dim=0)
            #     print("rpn.py!!!!!!!!!!!!!!!!!!!!!!test", test, torch.squeeze(test[1][:, 1:2], dim=1)) #

            inds = [torch.squeeze(p.objectness_logits.sort(descending=True, dim=0)[1][:, 1:2], dim=1) for p in proposals]
            # import numpy as np
            # print("rpn.py!!!!!!!!!!!!!!!!!!!!!!test", np.array(inds[0].cpu()).shape) # 2 * 1000
            proposals = [p[ind] for p, ind in zip(proposals, inds)]

        # print("rpn.py!!!!!!!!!!!!!!!!!!! proposals", proposals)
        # print("rpn.py!!!!!!!!!!!!!!!!!!! proposals.shape", len(proposals[0])) # 2 * 1000
        # print("rpn.py!!!!!!!!!!!!!!!!!!! for_calculate_loss", for_calculate_loss)
        # return proposals, loss
        return proposals, for_calculate_loss