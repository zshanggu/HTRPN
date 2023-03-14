# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import itertools
import logging
import numpy as np
import torch
import torch.nn.functional as F
from fvcore.nn import smooth_l1_loss

from fsdet.layers import batched_nms, cat
from fsdet.structures import Boxes, Instances, pairwise_iou
from fsdet.utils.events import get_event_storage

from ..sampling import subsample_labels, subsample_labels_hierc

logger = logging.getLogger(__name__)

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    L: number of feature maps per image on which RPN is run
    A: number of cell anchors (must be the same for all feature maps)
    Hi, Wi: height and width of the i-th feature map
    4: size of the box parameterization

Naming convention:

    objectness: refers to the binary classification of an anchor as object vs. not
    object.

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_objectness_logits: predicted objectness scores in [-inf, +inf]; use
        sigmoid(pred_objectness_logits) to estimate P(object).

    gt_objectness_logits: ground-truth binary classification labels for objectness

    pred_anchor_deltas: predicted box2box transform deltas

    gt_anchor_deltas: ground-truth box2box transform deltas
"""


def find_top_rpn_proposals(
    proposals,
    pred_objectness_logits,
    images,
    nms_thresh,
    pre_nms_topk,
    post_nms_topk,
    min_box_side_len,
    training,
):
    """
    For each feature map, select the `pre_nms_topk` highest scoring proposals,
    apply NMS, clip proposals, and remove small boxes. Return the `post_nms_topk`
    highest scoring proposals among all the feature maps if `training` is True,
    otherwise, returns the highest `post_nms_topk` scoring proposals for each
    feature map.

    Args:
        proposals (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A, 4).
            All proposal predictions on the feature maps.
        pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A).
        images (ImageList): Input images as an :class:`ImageList`.
        nms_thresh (float): IoU threshold to use for NMS
        pre_nms_topk (int): number of top k scoring proposals to keep before applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is per
            feature map.
        post_nms_topk (int): number of top k scoring proposals to keep after applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is total,
            over all feature maps.
        min_box_side_len (float): minimum proposal box side length in pixels (absolute units
            wrt input images).
        training (bool): True if proposals are to be used in training, otherwise False.
            This arg exists only to support a legacy bug; look for the "NB: Legacy bug ..."
            comment.

    Returns:
        proposals (list[Instances]): list of N Instances. The i-th Instances
            stores post_nms_topk object proposals for image i.
    """
    image_sizes = images.image_sizes  # in (h, w) order
    num_images = len(image_sizes)
    device = proposals[0].device

    # 1. Select top-k anchor for every level and every image
    topk_scores = []  # #lvl Tensor, each of shape N x topk
    topk_proposals = []
    level_ids = []  # #lvl Tensor, each of shape (topk,)
    batch_idx = torch.arange(num_images, device=device)
    for level_id, proposals_i, logits_i in zip(
        itertools.count(), proposals, pred_objectness_logits
    ):
        Hi_Wi_A = logits_i.shape[1]
        num_proposals_i = min(pre_nms_topk, Hi_Wi_A)

        # sort is faster than topk (https://github.com/pytorch/pytorch/issues/22812)
        # topk_scores_i, topk_idx = logits_i.topk(num_proposals_i, dim=1)
        logits_i, idx = logits_i.sort(descending=True, dim=1)
        topk_scores_i = logits_i[batch_idx, :num_proposals_i]
        topk_idx = idx[batch_idx, :num_proposals_i]

        # each is N x topk
        topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]  # N x topk x 4

        topk_proposals.append(topk_proposals_i)
        topk_scores.append(topk_scores_i)
        level_ids.append(torch.full((num_proposals_i,), level_id, dtype=torch.int64, device=device))

    # 2. Concat all levels together
    topk_scores = cat(topk_scores, dim=1)
    topk_proposals = cat(topk_proposals, dim=1)
    level_ids = cat(level_ids, dim=0)

    # 3. For each image, run a per-level NMS, and choose topk results.
    results = []
    for n, image_size in enumerate(image_sizes):
        boxes = Boxes(topk_proposals[n])
        scores_per_img = topk_scores[n]
        boxes.clip(image_size)

        # filter empty boxes
        keep = boxes.nonempty(threshold=min_box_side_len)
        lvl = level_ids
        if keep.sum().item() != len(boxes):
            boxes, scores_per_img, lvl = boxes[keep], scores_per_img[keep], level_ids[keep]

        keep = batched_nms(boxes.tensor, scores_per_img, lvl, nms_thresh)
        # In Detectron1, there was different behavior during training vs. testing.
        # (https://github.com/facebookresearch/Detectron/issues/459)
        # During training, topk is over the proposals from *all* images in the training batch.
        # During testing, it is over the proposals for each image separately.
        # As a result, the training behavior becomes batch-dependent,
        # and the configuration "POST_NMS_TOPK_TRAIN" end up relying on the batch size.
        # This bug is addressed in Detectron2 to make the behavior independent of batch size.
        keep = keep[:post_nms_topk]

        res = Instances(image_size)
        res.proposal_boxes = boxes[keep]
        res.objectness_logits = scores_per_img[keep]
        results.append(res)
    return results


def find_ternary_top_rpn_proposals(
    proposals,
    ternary_pred_objectness_logits,
    images,
    nms_thresh,
    pre_nms_topk,
    post_nms_topk,
    min_box_side_len,
    training,
    fine_tuning,
):
    """
    For each feature map, select the `pre_nms_topk` highest scoring proposals,
    apply NMS, clip proposals, and remove small boxes. Return the `post_nms_topk`
    highest scoring proposals among all the feature maps if `training` is True,
    otherwise, returns the highest `post_nms_topk` scoring proposals for each
    feature map.

    Args:
        proposals (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A, 4).
            All proposal predictions on the feature maps.
        ternary_pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A/3, 3).
        images (ImageList): Input images as an :class:`ImageList`.
        nms_thresh (float): IoU threshold to use for NMS
        pre_nms_topk (int): number of top k scoring proposals to keep before applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is per
            feature map.
        post_nms_topk (int): number of top k scoring proposals to keep after applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is total,
            over all feature maps.
        min_box_side_len (float): minimum proposal box side length in pixels (absolute units
            wrt input images).
        training (bool): True if proposals are to be used in training, otherwise False.
            This arg exists only to support a legacy bug; look for the "NB: Legacy bug ..."
            comment.

    Returns:
        proposals (list[Instances]): list of N Instances. The i-th Instances
            stores post_nms_topk object proposals for image i.
    """
    image_sizes = images.image_sizes  # in (h, w) order
    num_images = len(image_sizes)
    device = proposals[0].device

    # 1. Select top-k anchor for every level and every image
    topk_scores = []  # #lvl Tensor, each of shape N x topk
    topk_proposals = []
    # test_topk_scores = []  # #lvl Tensor, each of shape N x topk
    # test_topk_proposals = []
    level_ids = []  # #lvl Tensor, each of shape (topk,)
    batch_idx = torch.arange(num_images, device=device) # tensor([0, 1], device='cuda:0') 2 images in a batch
    test_batch_idx = torch.arange(num_images, device=device)
    for level_id, proposals_i, logits_i in zip(
        itertools.count(), proposals, ternary_pred_objectness_logits
    ):
        # test = torch.randint(1, 100, (2, 9)) # 2: 2 images in a batch, 5 proposals each image, 3 score each proposal
        # # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test", test) #
        # # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test", test.shape) #
        # test1 = test.reshape(2, int(test.shape[1]/3), 3)
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test1", test1) #
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test1.shape", test1.shape) # torch.Size([2, 3, 3])
        # test1_sum = torch.sum(test1[:, :, 1:3], dim=2, keepdim=True)
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test1_sum", test1_sum) #
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test1_sum.shape", test1_sum.shape) # torch.Size([2, 3, 1])
        # test2, test2_idx = test1_sum.sort(descending=True, dim=1)
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test2", test2, test2_idx) #
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test2.shape", test2.shape) # torch.Size([2, 3, 1])
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test2_idx.shape", test2_idx.shape) # torch.Size([2, 3, 1])
        # test2_topk_idx = test2_idx[test_batch_idx, :2]
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test2_topk_idx", test2_topk_idx) #
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test2_topk_idx.shape", test2_topk_idx.shape) # torch.Size([2, 2, 1])
        # test2_topk_idx_slice = test2_topk_idx[:, :, 1:2]
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test2_topk_idx_slice", test2_topk_idx_slice) #
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test2_topk_idx_slice", test2_topk_idx_slice.shape) #
        # test2_scores_i = test1[0, test2_topk_idx_slice]  # 2: num_proposals_i
        # test2_scores_i = torch.squeeze(test2_scores_i, dim=2)  # N x topk x 4
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test2_scores_i", test2_scores_i) #
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test2_scores_i", test2_scores_i.shape) #
        # test_proposals_i = torch.randint(1, 100, (2, 5, 4))
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test_proposals_i", test_proposals_i) #
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test_proposals_i", test_proposals_i.shape) #
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test_batch_idx[:, None]", test_batch_idx[:, None]) # tensor([[0], [1]], device='cuda:0')
        # # test2_proposals_i = test_proposals_i[test_batch_idx[:, None], test2_topk_idx_slice]  # N x topk x 4
        # # test2_proposals_i = test_proposals_i[test_batch_idx[:, None], :]  # N x topk x 4
        # # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test2_proposals_i", test2_proposals_i) #
        # # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test2_proposals_i", test2_proposals_i.shape) #
        # test2_proposals_i = test_proposals_i[0, test2_topk_idx_slice]  # N x topk x 4
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test2_proposals_i", test2_proposals_i) #
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test2_proposals_i", test2_proposals_i.shape) #
        # test2_proposals_i = torch.squeeze(test2_proposals_i, dim=2)  # N x topk x 4
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test2_proposals_i", test2_proposals_i) #
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test2_proposals_i", test2_proposals_i.shape) #
        # test_topk_proposals.append(test2_proposals_i)
        # test_topk_scores.append(test2_scores_i)

        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!logits_i.shape", logits_i.shape) # batch * number of anchors in p level_id
        Hi_Wi_A = int(logits_i.shape[1])
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!Hi_Wi_A", Hi_Wi_A) # number of anchors in p level_id
        num_proposals_i = min(pre_nms_topk, Hi_Wi_A)

        # sort is faster than topk (https://github.com/pytorch/pytorch/issues/22812)
        # topk_scores_i, topk_idx = logits_i.topk(num_proposals_i, dim=1)
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!logits_i", logits_i.shape)  # torch.Size([2, 170496, 3])
        if fine_tuning or not training:
            logits_i_temp, _ = torch.max(logits_i[:, :, 1:3], 2)
            logits_i_temp_unsqueeze = torch.unsqueeze(logits_i_temp, -1)
            _, idx = logits_i_temp_unsqueeze.sort(descending=True, dim=1)  # sort the ternary classification
            del logits_i_temp, logits_i_temp_unsqueeze
            # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!logits_i.shape", logits_i.shape) #
            # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!idx.shape", idx.shape)  # torch.Size([2, 170496, 1])
            topk_idx_slice = idx[batch_idx, :num_proposals_i]  # indexes of the top num_proposals_i scores
            # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!topk_idx", topk_idx.shape) #
        else:
            _, idx = logits_i.sort(descending=True, dim=1)  # sort the ternary classification
            topk_idx = idx[batch_idx, :num_proposals_i]  # indexes of the top num_proposals_i scores
            # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!topk_idx", topk_idx.shape) #
            topk_idx_slice = topk_idx[:, :, 1:2]  # choose second score of the ternary classification: is an object
            # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!topk_idx_slice", topk_idx_slice.shape) #

        # each is N x topk
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!proposals_i", proposals_i.shape) #
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!batch_idx[:, None]", batch_idx[:, None]) #
        # proposals_i = proposals_i[batch_idx[:, None], :]  # left top num_proposals_i elements proposals
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!proposals_i", proposals_i.shape) #
        # proposals_i = torch.squeeze(proposals_i, dim=1)  # N x topk x 4
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!proposals_i", proposals_i.shape) #
        topk_proposals_i = proposals_i[0, topk_idx_slice]  # left top num_proposals_i elements proposals
        topk_proposals_i = torch.squeeze(topk_proposals_i, dim=2)  # N x topk x 4
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!topk_proposals_i", topk_proposals_i.shape) #
        topk_scores_i = logits_i[0, topk_idx_slice]  # left top num_proposals_i scores
        topk_scores_i = torch.squeeze(topk_scores_i, dim=2)  # N x topk x 4
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!topk_scores_i", topk_scores_i.shape) #

        topk_proposals.append(topk_proposals_i)
        topk_scores.append(topk_scores_i)
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!num_proposals_i", num_proposals_i) #
        level_ids.append(torch.full((num_proposals_i,), level_id, dtype=torch.int64, device=device))

    # 2. Concat all levels together
    topk_scores = cat(topk_scores, dim=1)
    # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!topk_scores", topk_scores.shape) #
    topk_proposals = cat(topk_proposals, dim=1)
    # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!topk_proposals", topk_proposals.shape) #
    # test_topk_proposals = cat(test_topk_proposals, dim=1)
    # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test_topk_proposals", test_topk_proposals) #
    # test_topk_scores = cat(test_topk_scores, dim=1)
    # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test_topk_scores", test_topk_scores) #
    level_ids = cat(level_ids, dim=0)
    # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!level_ids", level_ids.shape) #

    # 3. For each image, run a per-level NMS, and choose topk results.
    results = []
    for n, image_size in enumerate(image_sizes):
        boxes = Boxes(topk_proposals[n])
        scores_per_img = topk_scores[n]
        boxes.clip(image_size)

        # filter empty boxes
        keep = boxes.nonempty(threshold=min_box_side_len)
        lvl = level_ids
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!keep", keep.shape) # torch.Size([8576])
        if keep.sum().item() != len(boxes):
            boxes, scores_per_img, lvl = boxes[keep], scores_per_img[keep], level_ids[keep]
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!boxes", boxes.tensor.shape) # torch.Size([8576, 4])
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!scores_per_img", scores_per_img.shape) # torch.Size([8576, 3])
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!lvl", lvl.shape) #  # torch.Size([8576])

        # test = torch.randint(1, 100, (15, 3)) #
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test", test) #
        # test = test[:, 1:2]
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test", test) #
        scores_per_img_is_object = scores_per_img[:, 1:2] # torch.Size([8576, 1])
        scores_per_img_is_object = torch.squeeze(scores_per_img_is_object, dim=1)
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!scores_per_img_is_object", scores_per_img_is_object.shape) #torch.Size([8576])

        # keep = batched_nms(boxes.tensor, scores_per_img, lvl, nms_thresh)
        keep = batched_nms(boxes.tensor, scores_per_img_is_object, lvl, nms_thresh)
        # In Detectron1, there was different behavior during training vs. testing.
        # (https://github.com/facebookresearch/Detectron/issues/459)
        # During training, topk is over the proposals from *all* images in the training batch.
        # During testing, it is over the proposals for each image separately.
        # As a result, the training behavior becomes batch-dependent,
        # and the configuration "POST_NMS_TOPK_TRAIN" end up relying on the batch size.
        # This bug is addressed in Detectron2 to make the behavior independent of batch size.
        keep = keep[:post_nms_topk]

        res = Instances(image_size)
        res.proposal_boxes = boxes[keep]
        res.objectness_logits = scores_per_img[keep]
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!res.proposal_boxes", res.proposal_boxes.tensor.shape) # torch.Size([1000, 4])
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!res.objectness_logits", res.objectness_logits.shape) # scores_per_img torch.Size([1000, 3])
        results.append(res)
    return results


def rpn_losses(
    gt_objectness_logits,
    gt_anchor_deltas,
    pred_objectness_logits,
    pred_anchor_deltas,
    smooth_l1_beta,
):
    """
    Args:
        gt_objectness_logits (Tensor): shape (N,), each element in {-1, 0, 1} representing
            ground-truth objectness labels with: -1 = ignore; 0 = not object; 1 = object.
        gt_anchor_deltas (Tensor): shape (N, box_dim), row i represents ground-truth
            box2box transform targets (dx, dy, dw, dh) or (dx, dy, dw, dh, da) that map anchor i to
            its matched ground-truth box.
        pred_objectness_logits (Tensor): shape (N,), each element is a predicted objectness
            logit.
        pred_anchor_deltas (Tensor): shape (N, box_dim), each row is a predicted box2box
            transform (dx, dy, dw, dh) or (dx, dy, dw, dh, da)
        smooth_l1_beta (float): The transition point between L1 and L2 loss in
            the smooth L1 loss function. When set to 0, the loss becomes L1. When
            set to +inf, the loss becomes constant 0.

    Returns:
        objectness_loss, localization_loss, both unnormalized (summed over samples).
    """
    pos_masks = gt_objectness_logits == 1
    localization_loss = smooth_l1_loss(
        pred_anchor_deltas[pos_masks], gt_anchor_deltas[pos_masks], smooth_l1_beta, reduction="sum"
    )

    valid_masks = gt_objectness_logits >= 0
    objectness_loss = F.binary_cross_entropy_with_logits(
        pred_objectness_logits[valid_masks],
        gt_objectness_logits[valid_masks].to(torch.float32),
        reduction="sum",
    )
    return objectness_loss, localization_loss


def ternary_rpn_losses(
    gt_objectness_logits,
    gt_anchor_deltas,
    ternary_pred_objectness_logits,
    pred_anchor_deltas,
    smooth_l1_beta,
):
    """
    Args:
        gt_objectness_logits (Tensor): shape (N,), each element in {-1, 0, 1} representing
            ground-truth objectness labels with: -1 = ignore; 0 = not object; 1 = object.
        gt_anchor_deltas (Tensor): shape (N, box_dim), row i represents ground-truth
            box2box transform targets (dx, dy, dw, dh) or (dx, dy, dw, dh, da) that map anchor i to
            its matched ground-truth box.
        pred_objectness_logits (Tensor): shape (N,), each element is a predicted objectness
            logit.
        pred_anchor_deltas (Tensor): shape (N, box_dim), each row is a predicted box2box
            transform (dx, dy, dw, dh) or (dx, dy, dw, dh, da)
        smooth_l1_beta (float): The transition point between L1 and L2 loss in
            the smooth L1 loss function. When set to 0, the loss becomes L1. When
            set to +inf, the loss becomes constant 0.

    Returns:
        objectness_loss, localization_loss, both unnormalized (summed over samples).
    """
    pos_masks = gt_objectness_logits == 1  # position
    # print("pos_masks", pos_masks)  # tensor([False, False, False,  ..., False, False, False], device='cuda:0')
    # print("pos_masks.size()!!!!!!!!!!!!!", pos_masks.size())  # same size as gt_objectness_logits
    # print("gt_anchor_deltas.size()!!!!!!!!!!!!!", gt_anchor_deltas.size())  # same size as gt_objectness_logits
    # print("pred_anchor_deltas.size()!!!!!!!!!!!!!", pred_anchor_deltas.size())  # same size as gt_objectness_logits
    localization_loss = smooth_l1_loss(
        pred_anchor_deltas[pos_masks], gt_anchor_deltas[pos_masks], smooth_l1_beta, reduction="sum"
    )

    valid_masks = gt_objectness_logits >= 0  # objectness 0, 1 and 2
    # print("rpn_outputs.py!!!!!!!!!!!!!!!!ternary_pred_objectness_logits", ternary_pred_objectness_logits.shape)  # torch.Size([343776, 3]) torch.float32
    # print("rpn_outputs.py!!!!!!!!!!!!!!!!gt_objectness_logits", gt_objectness_logits.shape)  # torch.Size([343776]) torch.int8
    objectness_loss = F.cross_entropy(
        ternary_pred_objectness_logits[valid_masks],
        gt_objectness_logits[valid_masks].long(),
        reduction="sum",
    )
    return objectness_loss, localization_loss


class RPNOutputs(object):
    def __init__(
        self,
        box2box_transform,
        anchor_matcher,
        batch_size_per_image,
        positive_fraction,
        images,
        pred_objectness_logits,
        pred_anchor_deltas,
        anchors,

        boundary_threshold=0,
        gt_boxes=None,
        smooth_l1_beta=0.0,
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform): :class:`Box2BoxTransform` instance for
                anchor-proposal transformations.
            anchor_matcher (Matcher): :class:`Matcher` instance for matching anchors to
                ground-truth boxes; used to determine training labels.
            batch_size_per_image (int): number of proposals to sample when training
            positive_fraction (float): target fraction of sampled proposals that should be positive
            images (ImageList): :class:`ImageList` instance representing N input images
            pred_objectness_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, A, Hi, Wi) representing
                the predicted objectness logits for anchors.
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, A*4, Hi, Wi) representing the predicted "deltas" used to transform anchors
                to proposals.
            anchors (list[list[Boxes]]): A list of N elements. Each element is a list of L
                Boxes. The Boxes at (n, l) stores the entire anchor array for feature map l in image
                n (i.e. the cell anchors repeated over all locations in feature map (n, l)).
            boundary_threshold (int): if >= 0, then anchors that extend beyond the image
                boundary by more than boundary_thresh are not used in training. Set to a very large
                number or < 0 to disable this behavior. Only needed in training.
            gt_boxes (list[Boxes], optional): A list of N elements. Element i a Boxes storing
                the ground-truth ("gt") boxes for image i.
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        self.box2box_transform = box2box_transform              # get and apply deltas
        self.anchor_matcher = anchor_matcher
        self.batch_size_per_image = batch_size_per_image        # 256
        self.positive_fraction = positive_fraction              # 0.5
        self.pred_objectness_logits = pred_objectness_logits    # [p2, p3, p4, p5, p6]
        self.pred_anchor_deltas = pred_anchor_deltas

        self.anchors = anchors                                  # List[List[Boxes], len=5], len=2
        self.gt_boxes = gt_boxes
        self.num_feature_maps = len(pred_objectness_logits)
        self.num_images = len(images)
        self.image_sizes = images.image_sizes
        self.boundary_threshold = boundary_threshold            # threshold to remove anchors outside the image
        self.smooth_l1_beta = smooth_l1_beta

    def _get_ground_truth(self):
        """
        Returns:
            gt_objectness_logits: list of N tensors. Tensor i is a vector whose length is the
                total number of anchors in image i (i.e., len(anchors[i])). Label values are
                in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative class; 1 = positive class.
            gt_anchor_deltas: list of N tensors. Tensor i has shape (len(anchors[i]), 4).
        """
        gt_objectness_logits = []
        gt_anchor_deltas = []
        # Concatenate anchors from all feature maps into a single Boxes per image
        anchors = [Boxes.cat(anchors_i) for anchors_i in self.anchors]
        for image_size_i, anchors_i, gt_boxes_i in zip(self.image_sizes, anchors, self.gt_boxes):
            """
            image_size_i: (h, w) for the i-th image
            anchors_i: anchors for i-th image
            gt_boxes_i: ground-truth boxes for i-th image
            """
            match_quality_matrix = pairwise_iou(gt_boxes_i, anchors_i)
            # matched_idxs is the ground-truth index in [0, M)
            # gt_objectness_logits_i is [0, -1, 1] indicating proposal is true positive, ignored or false positive
            matched_idxs, gt_objectness_logits_i = self.anchor_matcher(match_quality_matrix)

            if self.boundary_threshold >= 0:
                # Discard anchors that go out of the boundaries of the image
                # NOTE: This is legacy functionality that is turned off by default in Detectron2
                anchors_inside_image = anchors_i.inside_box(image_size_i, self.boundary_threshold)
                gt_objectness_logits_i[~anchors_inside_image] = -1

            if len(gt_boxes_i) == 0:
                # These values won't be used anyway since the anchor is labeled as background
                gt_anchor_deltas_i = torch.zeros_like(anchors_i.tensor)
            else:
                # TODO wasted computation for ignored boxes
                matched_gt_boxes = gt_boxes_i[matched_idxs]
                gt_anchor_deltas_i = self.box2box_transform.get_deltas(
                    anchors_i.tensor, matched_gt_boxes.tensor
                )

            gt_objectness_logits.append(gt_objectness_logits_i)
            gt_anchor_deltas.append(gt_anchor_deltas_i)

        return gt_objectness_logits, gt_anchor_deltas

    def losses(self):
        """
        Return the losses from a set of RPN predictions and their associated ground-truth.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
                Loss names are: `loss_rpn_cls` for objectness classification and
                `loss_rpn_loc` for proposal localization.
        """

        def resample(label):
            """
            Randomly sample a subset of positive and negative examples by overwriting
            the label vector to the ignore value (-1) for all elements that are not
            included in the sample.
            """
            pos_idx, neg_idx = subsample_labels(
                label, self.batch_size_per_image, self.positive_fraction, 0
            )
            # Fill with the ignore label (-1), then set positive and negative labels
            label.fill_(-1)
            label.scatter_(0, pos_idx, 1)
            label.scatter_(0, neg_idx, 0)
            return label

        """
        gt_objectness_logits: list of N tensors. Tensor i is a vector whose length is the
            total number of anchors in image i (i.e., len(anchors[i]))
        gt_anchor_deltas: list of N tensors. Tensor i has shape (len(anchors[i]), B),
            where B is the box dimension
        """
        # 这一步是对未经筛选所有 anchors，找到对应的 gt （MxN，N 是非常大的数目，所有 p_level 合并的）
        # gt_objectness_logits in [0, -1, 1]
        gt_objectness_logits, gt_anchor_deltas = self._get_ground_truth()

        # Collect all objectness labels and delta targets over feature maps and images
        # The final ordering is L, N, H, W, A from slowest to fastest axis.
        num_anchors_per_map = [np.prod(x.shape[1:]) for x in self.pred_objectness_logits]
        num_anchors_per_image = sum(num_anchors_per_map)

        # Stack to: (N, num_anchors_per_image), e.g., torch.Size([2, 247086])
        gt_objectness_logits = torch.stack(
            # resample +1/-1 to fraction 0.5, inplace modify other laberls to -1
            # -1 will be ingored in loss calculation function
            # NOTE: in VOC, not enough positive sample pairs, 12-24 out of 256 are positive.
            # NOTE: 负样本是从 247086 里面随机抽出来 512 - pos 的
            [resample(label) for label in gt_objectness_logits], dim=0
        )

        # Log the number of positive/negative anchors per-image that's used in training
        num_pos_anchors = (gt_objectness_logits == 1).sum().item()
        num_neg_anchors = (gt_objectness_logits == 0).sum().item()
        storage = get_event_storage()
        storage.put_scalar("rpn/num_pos_anchors", num_pos_anchors / self.num_images)
        storage.put_scalar("rpn/num_neg_anchors", num_neg_anchors / self.num_images)

        assert gt_objectness_logits.shape[1] == num_anchors_per_image
        # Split to tuple of L tensors, each with shape (N, num_anchors_per_map)
        gt_objectness_logits = torch.split(gt_objectness_logits, num_anchors_per_map, dim=1)
        # Concat from all feature maps
        gt_objectness_logits = cat([x.flatten() for x in gt_objectness_logits], dim=0)

        # Stack to: (N, num_anchors_per_image, B)
        gt_anchor_deltas = torch.stack(gt_anchor_deltas, dim=0)
        assert gt_anchor_deltas.shape[1] == num_anchors_per_image
        B = gt_anchor_deltas.shape[2]  # box dimension (4 or 5)

        # Split to tuple of L tensors, each with shape (N, num_anchors_per_image)
        gt_anchor_deltas = torch.split(gt_anchor_deltas, num_anchors_per_map, dim=1)
        # Concat from all feature maps
        gt_anchor_deltas = cat([x.reshape(-1, B) for x in gt_anchor_deltas], dim=0)

        # Collect all objectness logits and delta predictions over feature maps
        # and images to arrive at the same shape as the labels and targets
        # The final ordering is L, N, H, W, A from slowest to fastest axis.
        pred_objectness_logits = cat(
            [
                # Reshape: (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N*Hi*Wi*A, )
                x.permute(0, 2, 3, 1).flatten()
                for x in self.pred_objectness_logits
            ],
            dim=0,
        )
        pred_anchor_deltas = cat(
            [
                # Reshape: (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B)
                #          -> (N*Hi*Wi*A, B)
                x.view(x.shape[0], -1, B, x.shape[-2], x.shape[-1])
                .permute(0, 3, 4, 1, 2)
                .reshape(-1, B)
                for x in self.pred_anchor_deltas
            ],
            dim=0,
        )
        objectness_loss, localization_loss = rpn_losses(
            gt_objectness_logits,
            gt_anchor_deltas,
            pred_objectness_logits,
            pred_anchor_deltas,
            self.smooth_l1_beta,
        )
        normalizer = 1.0 / (self.batch_size_per_image * self.num_images)
        loss_cls = objectness_loss * normalizer  # cls: classification loss
        loss_loc = localization_loss * normalizer  # loc: localization loss
        losses = {"loss_rpn_cls": loss_cls, "loss_rpn_loc": loss_loc}

        return losses

    def predict_proposals(self):
        """
        Transform anchors into proposals by applying the predicted anchor deltas.

        Returns:
            proposals (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A, B), where B is box dimension (4 or 5).
        """
        proposals = []
        # Transpose anchors from images-by-feature-maps (N, L) to feature-maps-by-images (L, N)
        anchors = list(zip(*self.anchors))
        # For each feature map
        for anchors_i, pred_anchor_deltas_i in zip(anchors, self.pred_anchor_deltas):
            # print("pred_anchor_deltas_i", pred_anchor_deltas_i)
            B = anchors_i[0].tensor.size(1)
            N, _, Hi, Wi = pred_anchor_deltas_i.shape
            # Reshape: (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N*Hi*Wi*A, B)
            pred_anchor_deltas_i = (
                pred_anchor_deltas_i.view(N, -1, B, Hi, Wi).permute(0, 3, 4, 1, 2).reshape(-1, B)
            )
            # Concatenate all anchors to shape (N*Hi*Wi*A, B)
            # type(anchors_i[0]) is Boxes (B = 4) or RotatedBoxes (B = 5)
            anchors_i = type(anchors_i[0]).cat(anchors_i)
            proposals_i = self.box2box_transform.apply_deltas(
                pred_anchor_deltas_i, anchors_i.tensor
            )
            # Append feature map proposals with shape (N, Hi*Wi*A, B)
            proposals.append(proposals_i.view(N, -1, B))
        return proposals

    def predict_objectness_logits(self):
        """
        Return objectness logits in the same format as the proposals returned by
        :meth:`predict_proposals`.

        Returns:
            pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A).
        """
        pred_objectness_logits = [
            # Reshape: (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).reshape(self.num_images, -1)
            for score in self.pred_objectness_logits
        ]
        return pred_objectness_logits


class HiercOnlyRPNOutputs(object):
    def __init__(
        self,
        box2box_transform,
        anchor_matcher,
        batch_size_per_image,
        positive_fraction,
        images,
        pred_objectness_logits,
        pred_anchor_deltas,
        anchors,

        boundary_threshold=0,
        gt_boxes=None,
        smooth_l1_beta=0.0,
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform): :class:`Box2BoxTransform` instance for
                anchor-proposal transformations.
            anchor_matcher (Matcher): :class:`Matcher` instance for matching anchors to
                ground-truth boxes; used to determine training labels.
            batch_size_per_image (int): number of proposals to sample when training
            positive_fraction (float): target fraction of sampled proposals that should be positive
            images (ImageList): :class:`ImageList` instance representing N input images
            pred_objectness_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, A, Hi, Wi) representing
                the predicted objectness logits for anchors.
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, A*4, Hi, Wi) representing the predicted "deltas" used to transform anchors
                to proposals.
            anchors (list[list[Boxes]]): A list of N elements. Each element is a list of L
                Boxes. The Boxes at (n, l) stores the entire anchor array for feature map l in image
                n (i.e. the cell anchors repeated over all locations in feature map (n, l)).
            boundary_threshold (int): if >= 0, then anchors that extend beyond the image
                boundary by more than boundary_thresh are not used in training. Set to a very large
                number or < 0 to disable this behavior. Only needed in training.
            gt_boxes (list[Boxes], optional): A list of N elements. Element i a Boxes storing
                the ground-truth ("gt") boxes for image i.
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        self.box2box_transform = box2box_transform              # get and apply deltas
        self.anchor_matcher = anchor_matcher
        self.batch_size_per_image = batch_size_per_image        # 256
        self.positive_fraction = positive_fraction              # 0.5
        self.pred_objectness_logits = pred_objectness_logits    # [p2, p3, p4, p5, p6]
        self.pred_anchor_deltas = pred_anchor_deltas

        self.anchors = anchors                                  # List[List[Boxes], len=5], len=2
        self.gt_boxes = gt_boxes
        self.num_feature_maps = len(pred_objectness_logits)
        self.num_images = len(images)
        self.image_sizes = images.image_sizes
        self.boundary_threshold = boundary_threshold            # threshold to remove anchors outside the image
        self.smooth_l1_beta = smooth_l1_beta

    def _get_ground_truth(self):
        """
        Returns:
            gt_objectness_logits: list of N tensors. Tensor i is a vector whose length is the
                total number of anchors in image i (i.e., len(anchors[i])). Label values are
                in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative class; 1 = positive class.
            gt_anchor_deltas: list of N tensors. Tensor i has shape (len(anchors[i]), 4).
        """
        gt_objectness_logits = []
        gt_anchor_deltas = []
        # Concatenate anchors from all feature maps into a single Boxes per image
        anchors = [Boxes.cat(anchors_i) for anchors_i in self.anchors]
        for image_size_i, anchors_i, gt_boxes_i in zip(self.image_sizes, anchors, self.gt_boxes):
            """
            image_size_i: (h, w) for the i-th image
            anchors_i: anchors for i-th image
            gt_boxes_i: ground-truth boxes for i-th image
            """
            match_quality_matrix = pairwise_iou(gt_boxes_i, anchors_i)
            # matched_idxs is the ground-truth index in [0, M)
            # gt_objectness_logits_i is [0, -1, 1] indicating proposal is true negative, ignored, true positive
            matched_idxs, gt_objectness_logits_i = self.anchor_matcher(match_quality_matrix)

            if self.boundary_threshold >= 0:
                # Discard anchors that go out of the boundaries of the image
                # NOTE: This is legacy functionality that is turned off by default in Detectron2
                anchors_inside_image = anchors_i.inside_box(image_size_i, self.boundary_threshold)
                gt_objectness_logits_i[~anchors_inside_image] = -1

            if len(gt_boxes_i) == 0:
                # These values won't be used anyway since the anchor is labeled as background
                gt_anchor_deltas_i = torch.zeros_like(anchors_i.tensor)
            else:
                # TODO wasted computation for ignored boxes
                matched_gt_boxes = gt_boxes_i[matched_idxs]
                gt_anchor_deltas_i = self.box2box_transform.get_deltas(
                    anchors_i.tensor, matched_gt_boxes.tensor
                )

            gt_objectness_logits.append(gt_objectness_logits_i)
            gt_anchor_deltas.append(gt_anchor_deltas_i)

        return gt_objectness_logits, gt_anchor_deltas

    def losses(self):
        """
        Return the losses from a set of RPN predictions and their associated ground-truth.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
                Loss names are: `loss_rpn_cls` for objectness classification and
                `loss_rpn_loc` for proposal localization.
        """

        anchors_for_hierc = [p.tensor.size()[0] for p in self.anchors[0]]
        # print("anchors_for_hierc!!!!!!!!!!!!!!!!", anchors_for_hierc)  # [103680, 25920, 6480, 1620, 420]
        # self.num_neg_hierc = []
        # self.neg_idx_srtd = []
        def resample(label):
            """
            Randomly sample a subset of positive and negative examples by overwriting
            the label vector to the ignore value (-1) for all elements that are not
            included in the sample.
            """
            # print("anchors_for_hierc2!!!!!!!!!!!!!!!!", anchors_for_hierc)  # [103680, 25920, 6480, 1620, 420]
            pos_idx, neg_idx, num_neg_hierc = subsample_labels_hierc(
                label, self.batch_size_per_image, self.positive_fraction, 0, anchors_for_hierc
            )
            # Fill with the ignore label (-1), then set positive and negative labels
            label.fill_(-1)
            label.scatter_(0, pos_idx, 1)
            label.scatter_(0, neg_idx, 0)
            # self.num_neg_hierc.append(num_neg_hierc)
            # neg_idx_srtd, _ = neg_idx.sort()
            # self.neg_idx_srtd.append(neg_idx_srtd)
            return label

        """
        gt_objectness_logits: list of N tensors. Tensor i is a vector whose length is the
            total number of anchors in image i (i.e., len(anchors[i]))
        gt_anchor_deltas: list of N tensors. Tensor i has shape (len(anchors[i]), B),
            where B is the box dimension
        """
        # 这一步是对未经筛选所有 anchors，找到对应的 gt （MxN，N 是非常大的数目，所有 p_level 合并的）
        # gt_objectness_logits in [0, -1, 1]
        gt_objectness_logits, gt_anchor_deltas = self._get_ground_truth()

        # Collect all objectness labels and delta targets over feature maps and images
        # The final ordering is L, N, H, W, A from slowest to fastest axis.
        num_anchors_per_map = [np.prod(x.shape[1:]) for x in self.pred_objectness_logits]
        num_anchors_per_image = sum(num_anchors_per_map)

        # Stack to: (N, num_anchors_per_image), e.g., torch.Size([2, 247086])
        gt_objectness_logits = torch.stack(
            # resample +1/-1 to fraction 0.5, inplace modify other laberls to -1
            # -1 will be ingored in loss calculation function
            # NOTE: in VOC, not enough positive sample pairs, 12-24 out of 256 are positive.
            # NOTE: 负样本是从 247086 里面随机抽出来 512 - pos 的
            [resample(label) for label in gt_objectness_logits], dim=0
        )

        # Log the number of positive/negative anchors per-image that's used in training
        num_pos_anchors = (gt_objectness_logits == 1).sum().item()
        num_neg_anchors = (gt_objectness_logits == 0).sum().item()
        storage = get_event_storage()
        storage.put_scalar("rpn/num_pos_anchors", num_pos_anchors / self.num_images)
        storage.put_scalar("rpn/num_neg_anchors", num_neg_anchors / self.num_images)

        assert gt_objectness_logits.shape[1] == num_anchors_per_image
        # Split to tuple of L tensors, each with shape (N, num_anchors_per_map)
        gt_objectness_logits = torch.split(gt_objectness_logits, num_anchors_per_map, dim=1)
        # Concat from all feature maps
        gt_objectness_logits = cat([x.flatten() for x in gt_objectness_logits], dim=0)

        # Stack to: (N, num_anchors_per_image, B)
        gt_anchor_deltas = torch.stack(gt_anchor_deltas, dim=0)
        assert gt_anchor_deltas.shape[1] == num_anchors_per_image
        B = gt_anchor_deltas.shape[2]  # box dimension (4 or 5)

        # Split to tuple of L tensors, each with shape (N, num_anchors_per_image)
        gt_anchor_deltas = torch.split(gt_anchor_deltas, num_anchors_per_map, dim=1)
        # Concat from all feature maps
        gt_anchor_deltas = cat([x.reshape(-1, B) for x in gt_anchor_deltas], dim=0)

        # Collect all objectness logits and delta predictions over feature maps
        # and images to arrive at the same shape as the labels and targets
        # The final ordering is L, N, H, W, A from slowest to fastest axis.
        pred_objectness_logits = cat(
            [
                # Reshape: (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N*Hi*Wi*A, )
                x.permute(0, 2, 3, 1).flatten()
                for x in self.pred_objectness_logits
            ],
            dim=0,
        )
        pred_anchor_deltas = cat(
            [
                # Reshape: (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B)
                #          -> (N*Hi*Wi*A, B)
                x.view(x.shape[0], -1, B, x.shape[-2], x.shape[-1])
                .permute(0, 3, 4, 1, 2)
                .reshape(-1, B)
                for x in self.pred_anchor_deltas
            ],
            dim=0,
        )
        objectness_loss, localization_loss = rpn_losses(
            gt_objectness_logits,
            gt_anchor_deltas,
            pred_objectness_logits,
            pred_anchor_deltas,
            self.smooth_l1_beta,
        )
        normalizer = 1.0 / (self.batch_size_per_image * self.num_images)
        loss_cls = objectness_loss * normalizer  # cls: classification loss
        loss_loc = localization_loss * normalizer  # loc: localization loss
        losses = {"loss_rpn_cls": loss_cls, "loss_rpn_loc": loss_loc}

        return losses

    def predict_proposals(self):
        """
        Transform anchors into proposals by applying the predicted anchor deltas.

        Returns:
            proposals (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A, B), where B is box dimension (4 or 5).
        """
        proposals = []
        # Transpose anchors from images-by-feature-maps (N, L) to feature-maps-by-images (L, N)
        anchors = list(zip(*self.anchors))
        # For each feature map
        for anchors_i, pred_anchor_deltas_i in zip(anchors, self.pred_anchor_deltas):
            B = anchors_i[0].tensor.size(1)
            N, _, Hi, Wi = pred_anchor_deltas_i.shape
            # Reshape: (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N*Hi*Wi*A, B)
            pred_anchor_deltas_i = (
                pred_anchor_deltas_i.view(N, -1, B, Hi, Wi).permute(0, 3, 4, 1, 2).reshape(-1, B)
            )
            # Concatenate all anchors to shape (N*Hi*Wi*A, B)
            # type(anchors_i[0]) is Boxes (B = 4) or RotatedBoxes (B = 5)
            anchors_i = type(anchors_i[0]).cat(anchors_i)
            proposals_i = self.box2box_transform.apply_deltas(
                pred_anchor_deltas_i, anchors_i.tensor
            )
            # Append feature map proposals with shape (N, Hi*Wi*A, B)
            proposals.append(proposals_i.view(N, -1, B))
        return proposals

    def predict_objectness_logits(self):
        """
        Return objectness logits in the same format as the proposals returned by
        :meth:`predict_proposals`.

        Returns:
            pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A).
        """
        pred_objectness_logits = [
            # Reshape: (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).reshape(self.num_images, -1)
            for score in self.pred_objectness_logits
        ]
        return pred_objectness_logits


class TernaryHiercRPNOutputs(object):
    def __init__(
        self,
        box2box_transform,
        anchor_matcher,
        batch_size_per_image,
        positive_fraction,
        images,
        ternary_pred_objectness_logits,
        pred_anchor_deltas,
        anchors,

        boundary_threshold=0,
        gt_boxes=None,
        smooth_l1_beta=0.0,
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform): :class:`Box2BoxTransform` instance for
                anchor-proposal transformations.
            anchor_matcher (Matcher): :class:`Matcher` instance for matching anchors to
                ground-truth boxes; used to determine training labels.
            batch_size_per_image (int): number of proposals to sample when training
            positive_fraction (float): target fraction of sampled proposals that should be positive
            images (ImageList): :class:`ImageList` instance representing N input images
            ternary_pred_objectness_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, A*3, Hi, Wi) representing
                the predicted objectness logits for anchors. A represents 3 anchors at each point of
                the feature map. A*3 refers to the ternary classification.
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, A*4, Hi, Wi) representing the predicted "deltas" used to transform anchors
                to proposals.
            anchors (list[list[Boxes]]): A list of N elements. Each element is a list of L
                Boxes. The Boxes at (n, l) stores the entire anchor array for feature map l in image
                n (i.e. the cell anchors repeated over all locations in feature map (n, l)).
            boundary_threshold (int): if >= 0, then anchors that extend beyond the image
                boundary by more than boundary_thresh are not used in training. Set to a very large
                number or < 0 to disable this behavior. Only needed in training.
            gt_boxes (list[Boxes], optional): A list of N elements. Element i a Boxes storing
                the ground-truth ("gt") boxes for image i.
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        self.box2box_transform = box2box_transform              # get and apply deltas
        self.anchor_matcher = anchor_matcher
        self.batch_size_per_image = batch_size_per_image        # 256
        self.positive_fraction = positive_fraction              # 0.5
        self.ternary_pred_objectness_logits = ternary_pred_objectness_logits    # [p2, p3, p4, p5, p6]
        self.pred_anchor_deltas = pred_anchor_deltas

        self.anchors = anchors                                  # List[List[Boxes], len=5], len=2
        self.gt_boxes = gt_boxes
        self.num_feature_maps = len(ternary_pred_objectness_logits)
        self.num_images = len(images)
        self.image_sizes = images.image_sizes
        self.boundary_threshold = boundary_threshold            # threshold to remove anchors outside the image
        self.smooth_l1_beta = smooth_l1_beta


    def _get_ground_truth(self):
        """
        Returns:
            gt_objectness_logits: list of N tensors. Tensor i is a vector whose length is the
                total number of anchors in image i (i.e., len(anchors[i])). Label values are
                in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative class; 1 = positive class.
            gt_anchor_deltas: list of N tensors. Tensor i has shape (len(anchors[i]), 4).
        """
        gt_objectness_logits = []
        gt_anchor_deltas = []
        # Concatenate anchors from all feature maps into a single Boxes per image
        anchors = [Boxes.cat(anchors_i) for anchors_i in self.anchors]
        for image_size_i, anchors_i, gt_boxes_i in zip(self.image_sizes, anchors, self.gt_boxes):
            """
            image_size_i: (h, w) for the i-th image
            anchors_i: anchors for i-th image
            gt_boxes_i: ground-truth boxes for i-th image
            """
            match_quality_matrix = pairwise_iou(gt_boxes_i, anchors_i)
            # matched_idxs is the ground-truth index in [0, M)
            # gt_objectness_logits_i is [0, -1, 1, 2] indicating proposal is true negative, ignored, true positive or potential novel
            matched_idxs, gt_objectness_logits_i = self.anchor_matcher(match_quality_matrix)

            if self.boundary_threshold >= 0:
                # Discard anchors that go out of the boundaries of the image
                # NOTE: This is legacy functionality that is turned off by default in Detectron2
                anchors_inside_image = anchors_i.inside_box(image_size_i, self.boundary_threshold)
                gt_objectness_logits_i[~anchors_inside_image] = -1

            if len(gt_boxes_i) == 0:
                # These values won't be used anyway since the anchor is labeled as background
                gt_anchor_deltas_i = torch.zeros_like(anchors_i.tensor)
            else:
                # TODO wasted computation for ignored boxes
                matched_gt_boxes = gt_boxes_i[matched_idxs]
                gt_anchor_deltas_i = self.box2box_transform.get_deltas(
                    anchors_i.tensor, matched_gt_boxes.tensor
                )

            gt_objectness_logits.append(gt_objectness_logits_i)
            gt_anchor_deltas.append(gt_anchor_deltas_i)

        return gt_objectness_logits, gt_anchor_deltas

    def losses(self, images=None, batched_inputs=None):
        """
        Return the losses from a set of RPN predictions and their associated ground-truth.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
                Loss names are: `loss_rpn_cls` for objectness classification and
                `loss_rpn_loc` for proposal localization.
        """

        # now = datetime.datetime.now()

        # for i in range(gt_instances[0]._fields['gt_boxes'].tensor.size()[0]):
        #     start_point = (int(gt_instances[0]._fields['gt_boxes'].tensor[i][0].item()),
        #                    int(gt_instances[0]._fields['gt_boxes'].tensor[i][1].item()))
        #     end_point = (int(gt_instances[0]._fields['gt_boxes'].tensor[i][2].item()),
        #                  int(gt_instances[0]._fields['gt_boxes'].tensor[i][3].item()))
        #     cv2.rectangle(mat, start_point, end_point, (0, 0, 255), 2)
        #     cv2.putText(mat, format(gt_instances[0]._fields['gt_classes'][i].item(), '.3f'), start_point, cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)
        # cv2.imwrite("./results/test_potential_anchor/testimg_{}.jpg".format(now.strftime("%Y-%m-%d-%H-%M-%S")), mat)
        # print("batched_inputs[0]['file_name']----------------", batched_inputs[0]['file_name'])

        # print("self.anchors[0]!!!!!!!!!!!!!!!!", self.anchors[0][0].tensor.size())  # p2 torch.Size([120000, 4]) = features.width * features.height * 3
        # print("self.anchors[0]!!!!!!!!!!!!!!!!", self.anchors[0][1].tensor.size())  # p3 torch.Size([30000, 4])
        # print("self.anchors[0]!!!!!!!!!!!!!!!!", self.anchors[0][2].tensor.size())  # p4 torch.Size([7500, 4])
        # print("self.anchors[0]!!!!!!!!!!!!!!!!", self.anchors[0][3].tensor.size())  # p5 torch.Size([1875, 4])
        # print("self.anchors[0]!!!!!!!!!!!!!!!!", self.anchors[0][4].tensor.size())  # p6 torch.Size([507, 4])
        anchors_for_hierc = [p.tensor.size()[0] for p in self.anchors[0]]
        # print("anchors_for_hierc!!!!!!!!!!!!!!!!", anchors_for_hierc)  # [103680, 25920, 6480, 1620, 420]
        self.num_neg_hierc = []
        self.neg_idx_srtd = []  # sorted
        def resample(label):
            """
            Randomly sample a subset of positive and negative examples by overwriting
            the label vector to the ignore value (-1) for all elements that are not
            included in the sample.
            """
            # print("anchors_for_hierc2!!!!!!!!!!!!!!!!", anchors_for_hierc)  # [103680, 25920, 6480, 1620, 420]
            pos_idx, neg_idx, num_neg_hierc = subsample_labels_hierc(
                label, self.batch_size_per_image, self.positive_fraction, 0, anchors_for_hierc
            )
            # Fill with the ignore label (-1), then set positive and negative labels
            label.fill_(-1)
            label.scatter_(0, pos_idx, 1)
            label.scatter_(0, neg_idx, 0)
            self.num_neg_hierc.append(num_neg_hierc)
            neg_idx_srtd, _ = neg_idx.sort()
            self.neg_idx_srtd.append(neg_idx_srtd)
            return label

        """
        gt_objectness_logits: list of N tensors. Tensor i is a vector whose length is the
            total number of anchors in image i (i.e., len(anchors[i]))
        gt_anchor_deltas: list of N tensors. Tensor i has shape (len(anchors[i]), B),
            where B is the box dimension
        """
        # 这一步是对未经筛选所有 anchors，找到对应的 gt （MxN，N 是非常大的数目，所有 p_level 合并的）
        # gt_objectness_logits in [0, -1, 1, 2]
        gt_objectness_logits, gt_anchor_deltas = self._get_ground_truth()

        # print("gt_objectness_logits[0]!!!!!!!!!!!!!!!!", gt_objectness_logits[0].size())  # p2+p3+p4+p5+p6 torch.Size([128898])
        # print("gt_objectness_logits[1]!!!!!!!!!!!!!!!!", gt_objectness_logits[1].size())  # p2+p3+p4+p5+p6 torch.Size([128898])
        # Collect all objectness labels and delta targets over feature maps and images
        # The final ordering is L, N, H, W, A from slowest to fastest axis.
        # print("self.ternary_pred_objectness_logits!!!!!!!!!!!!!!!!", len(self.ternary_pred_objectness_logits))  # 5
        # print("self.ternary_pred_objectness_logits!!!!!!!!!!!!!!!!", self.ternary_pred_objectness_logits[0].shape)  # torch.Size([2, 9, 160, 336]): 2 images, 3 logits for 3 anchors each pixel position
        num_anchors_per_map = [int(np.prod(x.shape[1:]) / 3) for x in self.ternary_pred_objectness_logits]
        num_anchors_per_image = sum(num_anchors_per_map)

        # print("num_anchors_per_image!!!!!!!!!!!!!!!!", num_anchors_per_image)  # p2+p3+p4+p5+p6 128898
        # Stack to: (N, num_anchors_per_image), e.g., torch.Size([2, 247086])
        gt_objectness_logits = torch.stack(
            # resample +1/-1 to fraction 0.5, inplace modify other labels to -1
            # -1 will be ingored in loss calculation function
            # NOTE: in VOC, not enough positive sample pairs, 12-24 out of 256 are positive.
            # NOTE: 负样本是从 247086 里面随机抽出来 256 - pos 的 (? 512 - pos)
            [resample(label) for label in gt_objectness_logits], dim=0
        )

        neg_anchor_flatten = []
        for i in range(len(self.anchors)):
            anchor_ = self.anchors[i][0].clone()
            # for anchor_p in anchor:
            #     print("rpn_outputs.py!!!!!!!!!!!!!!!!anchor", anchor_p.tensor)  # p2+p3+p4+p5+p6 128898
            anchor_.tensor = torch.cat([anchor_p.tensor for anchor_p in self.anchors[i]], dim=0)
            anchor_.tensor = anchor_.tensor[gt_objectness_logits[i]==0]
            neg_anchor_flatten.append(anchor_)
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!anchor_i", len(anchor_i[0]))  # p2+p3+p4+p5+p6 244
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!anchor_i", len(anchor_i[1]))  # p2+p3+p4+p5+p6 247

        # print("rpn_outputs.py!!!!!!!!!!!!!!!!self.neg_idx_srtd", self.neg_idx_srtd)  # p2+p3+p4+p5+p6 247

        # gt_neg_objectness_logits = torch.cat(
        #     [(gt_objectness_logit==0).nonzero() for gt_objectness_logit in gt_objectness_logits], dim=0
        # )
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!gt_objectness_logits", gt_objectness_logits)  # p2+p3+p4+p5+p6 128898
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!gt_neg_objectness_logits", gt_neg_objectness_logits)  # p2+p3+p4+p5+p6 128898

        # print("gt_objectness_logits[0] torch.stack!!!!!!!!!!!!!!!!", gt_objectness_logits[0][gt_objectness_logits[0]==0])  # p2+p3+p4+p5+p6 128898
        # print("gt_objectness_logits[0] torch.stack!!!!!!!!!!!!!!!!", gt_objectness_logits[0].size()[0])  # p2+p3+p4+p5+p6 128898
        # print("gt_objectness_logits[1] torch.stack!!!!!!!!!!!!!!!!", gt_objectness_logits[1].size()[0])  # p2+p3+p4+p5+p6 128898

        # Log the number of positive/negative anchors per-image that's used in training
        num_pos_anchors = (gt_objectness_logits == 1).sum().item()
        num_neg_anchors = (gt_objectness_logits == 0).sum().item()
        # print("num_pos_anchors!!!!!!!!!!!!!!!!", num_pos_anchors)  # batch of 2 :p2+p3+p4+p5+p6  21
        # print("num_neg_anchors!!!!!!!!!!!!!!!!", num_neg_anchors)  # batch of 2 :p2+p3+p4+p5+p6  491 = 512 - 21
        storage = get_event_storage()
        storage.put_scalar("rpn/num_pos_anchors", num_pos_anchors / self.num_images)
        storage.put_scalar("rpn/num_neg_anchors", num_neg_anchors / self.num_images)

        # print(gt_objectness_logits.shape[1], "!=", num_anchors_per_image)
        assert gt_objectness_logits.shape[1] == num_anchors_per_image
        # # Split to tuple of L tensors, each with shape (N, num_anchors_per_map)
        # gt_objectness_logits = torch.split(gt_objectness_logits, num_anchors_per_map, dim=1)
        # # Concat from all feature maps
        # gt_objectness_logits = cat([x.flatten() for x in gt_objectness_logits], dim=0)
        # print("gt_objectness_logits[0] cat([x.flatten()!!!!!!!!!!!!!!!!", gt_objectness_logits.size())  # batch[0]:p2+p3+p4+p5+p6 + batch[1]:p2+p3+p4+p5+p6 torch.Size([257796])

        # Stack to: (N, num_anchors_per_image, B)
        gt_anchor_deltas = torch.stack(gt_anchor_deltas, dim=0)
        assert gt_anchor_deltas.shape[1] == num_anchors_per_image
        B = gt_anchor_deltas.shape[2]  # box dimension (4 or 5)

        # Split to tuple of L tensors, each with shape (N, num_anchors_per_image)
        gt_anchor_deltas = torch.split(gt_anchor_deltas, num_anchors_per_map, dim=1)
        # Concat from all feature maps
        gt_anchor_deltas = cat([x.reshape(-1, B) for x in gt_anchor_deltas], dim=0)

        # Collect all objectness logits and delta predictions over feature maps
        # and images to arrive at the same shape as the labels and targets
        # The final ordering is L, N, H, W, A from slowest to fastest axis.

        # test = torch.randint(1, 100, (2, 9, 2, 3)) # 2: 2 images in a batch, 4 base anchors each image, 3 ex-anchors each base anchor, 3 score each ex-anchor， 2*3 base anchor
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test", test)  #
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test", test.shape)  # torch.Size([2, 9, 2, 3])
        # test_permute = test.permute(0, 2, 3, 1)
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test_permute", test_permute)  #
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test_permute", test_permute.shape)  # torch.Size([2, 2, 3, 9])
        # test_reshape = test_permute.reshape(-1, 3)
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test_flatten", test_reshape)  #
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test_flatten", test_reshape.shape)  # torch.Size([2, 2, 3, 9])
        ternary_pred_objectness_logits = cat(
            [
                # Reshape: (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N*Hi*Wi*A/3, 3)
                x.permute(0, 2, 3, 1).reshape(-1, 3)
                for x in self.ternary_pred_objectness_logits
            ],
            dim=0,
        )
        # print("ternary_pred_objectness_logits!!!!!!!!!!!!!!", len(ternary_pred_objectness_logits), len(ternary_pred_objectness_logits[0]))  # batch[0]:p2+p3+p4+p5+p6 + batch[1]:p2+p3+p4+p5+p6 torch.Size([427614, 3])
        pred_anchor_deltas = cat(
            [
                # Reshape: (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B)
                #          -> (N*Hi*Wi*A, B)
                x.view(x.shape[0], -1, B, x.shape[-2], x.shape[-1])
                .permute(0, 3, 4, 1, 2)
                .reshape(-1, B)
                for x in self.pred_anchor_deltas
            ],
            dim=0,
        )

        # objectness_loss, localization_loss = rpn_losses(
        #     gt_objectness_logits,
        #     gt_anchor_deltas,
        #     ternary_pred_objectness_logits,
        #     pred_anchor_deltas,
        #     self.smooth_l1_beta,
        # )
        # # print("objectness_loss!!!!!!!!!!!!!!!!", objectness_loss)  # 357.6484
        # normalizer = 1.0 / (self.batch_size_per_image * self.num_images)
        # loss_cls = objectness_loss * normalizer  # cls: classification loss
        # loss_loc = localization_loss * normalizer  # loc: localization loss
        # losses = {"loss_rpn_cls": loss_cls, "loss_rpn_loc": loss_loc}

        for_calculate_loss = {
            "gt_objectness_logits": gt_objectness_logits,
            "gt_anchor_deltas": gt_anchor_deltas,
            "ternary_pred_objectness_logits": ternary_pred_objectness_logits,
            "pred_anchor_deltas": pred_anchor_deltas,
            "smooth_l1_beta": self.smooth_l1_beta,
            "batch_size_per_image": self.batch_size_per_image,
            "num_images": self.num_images,
            "neg_anchor_flatten": neg_anchor_flatten,
            "num_neg_hierc": self.num_neg_hierc,
            "neg_idx_srtd": self.neg_idx_srtd,
            "num_anchors_per_map": num_anchors_per_map,
        }

        # return losses
        return for_calculate_loss

    def predict_proposals(self):
        """
        Transform anchors into proposals by applying the predicted anchor deltas.

        Returns:
            proposals (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A, B), where B is box dimension (4 or 5).
        """
        proposals = []
        # Transpose anchors from images-by-feature-maps (N, L) to feature-maps-by-images (L, N)
        anchors = list(zip(*self.anchors))
        # For each feature map
        for anchors_i, pred_anchor_deltas_i in zip(anchors, self.pred_anchor_deltas):
            B = anchors_i[0].tensor.size(1)
            N, _, Hi, Wi = pred_anchor_deltas_i.shape
            # Reshape: (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N*Hi*Wi*A, B)
            pred_anchor_deltas_i = (
                pred_anchor_deltas_i.view(N, -1, B, Hi, Wi).permute(0, 3, 4, 1, 2).reshape(-1, B)
            )
            # Concatenate all anchors to shape (N*Hi*Wi*A, B)
            # type(anchors_i[0]) is Boxes (B = 4) or RotatedBoxes (B = 5)
            anchors_i = type(anchors_i[0]).cat(anchors_i)
            proposals_i = self.box2box_transform.apply_deltas(
                pred_anchor_deltas_i, anchors_i.tensor
            )
            # Append feature map proposals with shape (N, Hi*Wi*A, B)
            proposals.append(proposals_i.view(N, -1, B))
        return proposals

    def predict_objectness_logits(self):
        """
        Return objectness logits in the same format as the proposals returned by
        :meth:`predict_proposals`.

        Returns:
            ternary_pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A).
        """
        # test = torch.randint(1, 100, (2, 9, 2, 3)) # 2: 2 images in a batch, 4 base anchors each image, 3 ex-anchors each base anchor, 3 score each ex-anchor， 2*3 base anchor
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test", test)  #
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test", test.shape)  # torch.Size([2, 9, 2, 3])
        # test_permute = test.permute(0, 2, 3, 1)
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test_permute", test_permute)  #
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test_permute", test_permute.shape)  # torch.Size([2, 2, 3, 9])
        # test_reshape = test_permute.reshape(2, -1, 3)
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test_flatten", test_reshape)  #
        # print("rpn_outputs.py!!!!!!!!!!!!!!!!!!!!!!test_flatten", test_reshape.shape)  # torch.Size([2, 2, 3, 9])
        ternary_pred_objectness_logits = [
            # Reshape: (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A/3, 3)
            score.permute(0, 2, 3, 1).reshape(self.num_images, -1, 3)
            for score in self.ternary_pred_objectness_logits
        ]
        # print("rpn_outputs.py!!!!!!!!!!!!!ternary_pred_objectness_logits[0]", ternary_pred_objectness_logits[0].size())  # number of anchors in p2 torch.Size([2, 145728, 3])
        # print("rpn_outputs.py!!!!!!!!!!!!!ternary_pred_objectness_logits[1]", ternary_pred_objectness_logits[1].size())  # number of anchors in p3 torch.Size([2, 36432, 3])
        # print("rpn_outputs.py!!!!!!!!!!!!!ternary_pred_objectness_logits[2]", ternary_pred_objectness_logits[2].size())  # number of anchors in p4 torch.Size([2, 9108, 3])
        # print("rpn_outputs.py!!!!!!!!!!!!!ternary_pred_objectness_logits[3]", ternary_pred_objectness_logits[3].size())  # number of anchors in p5 torch.Size([2, 2277, 3])
        # print("rpn_outputs.py!!!!!!!!!!!!!ternary_pred_objectness_logits[4]", ternary_pred_objectness_logits[4].size())  # number of anchors in p6 torch.Size([2, 612, 3])
        return ternary_pred_objectness_logits