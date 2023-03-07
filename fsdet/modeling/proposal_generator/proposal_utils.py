# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import torch

from fsdet.structures import Instances


def add_ground_truth_to_proposals(gt_boxes, proposals):
    """Augment proposals with ground-truth boxes.
        In the case of learned proposals (e.g., RPN), when training starts
        the proposals will be low quality due to random initialization.
        It's possible that none of these initial
        proposals have high enough overlap with the gt objects to be used
        as positive examples for the second stage components (box head,
        cls head). Adding the gt boxes to the set of proposals
        ensures that the second stage components will have some positive
        examples from the start of training. For RPN, this augmentation improves
        convergence and empirically improves box AP on COCO by about 0.5
        points (under one tested configuration).

    Call `add_ground_truth_to_proposals_single_image` for all images.

    Args:
        gt_boxes(list[Boxes]): list of N elements. Element i is a Boxes
            representing the gound-truth for image i.
        proposals (list[Instances]): list of N elements. Element i is a Instances
            representing the proposals for image i.

    Returns:
        list[Instances]: list of N Instances. Each is the proposals for the image,
            with field "proposal_boxes" and "objectness_logits".
    """
    assert gt_boxes is not None

    assert len(proposals) == len(gt_boxes)
    if len(proposals) == 0:
        return proposals

    return [
        add_ground_truth_to_proposals_single_image(gt_boxes_i, proposals_i)
        for gt_boxes_i, proposals_i in zip(gt_boxes, proposals)
    ]


def add_ground_truth_to_proposals_single_image(gt_boxes, proposals):
    """
    Augment `proposals` with ground-truth boxes from `gt_boxes`.

    Args:
        Same as `add_ground_truth_to_proposals`, but with gt_boxes and proposals
        per image.

    Returns:
        Same as `add_ground_truth_to_proposals`, but for only one image.
    """
    device = proposals.objectness_logits.device
    # Concatenating gt_boxes with proposals requires them to have the same fields
    # Assign all ground-truth boxes an objectness logit corresponding to P(object) \approx 1.
    gt_logit_value = math.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))

    gt_logits = gt_logit_value * torch.ones(len(gt_boxes), device=device)
    gt_proposal = Instances(proposals.image_size)

    gt_proposal.proposal_boxes = gt_boxes
    gt_proposal.objectness_logits = gt_logits
    new_proposals = Instances.cat([proposals, gt_proposal])

    return new_proposals


def add_ternary_ground_truth_to_proposals(gt_boxes, proposals):
    """Augment proposals with ground-truth boxes.
        In the case of learned proposals (e.g., RPN), when training starts
        the proposals will be low quality due to random initialization.
        It's possible that none of these initial
        proposals have high enough overlap with the gt objects to be used
        as positive examples for the second stage components (box head,
        cls head). Adding the gt boxes to the set of proposals
        ensures that the second stage components will have some positive
        examples from the start of training. For RPN, this augmentation improves
        convergence and empirically improves box AP on COCO by about 0.5
        points (under one tested configuration).

    Call `add_ground_truth_to_proposals_single_image` for all images.

    Args:
        gt_boxes(list[Boxes]): list of N elements. Element i is a Boxes
            representing the gound-truth for image i.
        proposals (list[Instances]): list of N elements. Element i is a Instances
            representing the proposals for image i.

    Returns:
        list[Instances]: list of N Instances. Each is the proposals for the image,
            with field "proposal_boxes" and "objectness_logits".
    """
    assert gt_boxes is not None

    assert len(proposals) == len(gt_boxes)
    if len(proposals) == 0:
        return proposals

    return [
        add_ground_truth_to_ternary_proposals_single_image(gt_boxes_i, proposals_i)
        for gt_boxes_i, proposals_i in zip(gt_boxes, proposals)
    ]

def add_ground_truth_to_ternary_proposals_single_image(gt_boxes, proposals):
    """
    Augment `proposals` with ground-truth boxes from `gt_boxes`.

    Args:
        Same as `add_ground_truth_to_proposals`, but with gt_boxes and proposals
        per image.

    Returns:
        Same as `add_ground_truth_to_proposals`, but for only one image.
    """
    device = proposals.objectness_logits.device
    # Concatenating gt_boxes with proposals requires them to have the same fields
    # Assign all ground-truth boxes an objectness logit corresponding to P(object) \approx 1.
    gt_logit_value_not_object = math.log(1e-10 / (1 - 1e-10))  # 0
    gt_logit_value_is_object = math.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))  # 1
    gt_logit_value_potential_object = math.log(1e-10 / (1 - 1e-10))  # 2
    # print("proposal_utils.py!!!!!!!!!!!!!!!!!!!!gt_logit_value_not_object", gt_logit_value_not_object)  # -23.025850847100088
    # print("proposal_utils.py!!!!!!!!!!!!!!!!!!!!gt_logit_value_is_object", gt_logit_value_is_object)  # 23.025850847100088
    # print("proposal_utils.py!!!!!!!!!!!!!!!!!!!!gt_logit_value_potential_object", gt_logit_value_potential_object)  # -23.025850847100088
    gt_logit_value = torch.tensor([gt_logit_value_not_object, gt_logit_value_is_object, gt_logit_value_potential_object])

    gt_logits = torch.ones(len(gt_boxes), 3, device=device)
    gt_logits = torch.mul(gt_logit_value.cuda(), gt_logits)
    # print("proposal_utils.py!!!!!!!!!!!!!!!!!!!!len(gt_boxes)", len(gt_logits))  # 5: number of ground truth
    # print("proposal_utils.py!!!!!!!!!!!!!!!!!!!!len(proposals)", len(proposals))  # 1000
    gt_proposal = Instances(proposals.image_size)

    gt_proposal.proposal_boxes = gt_boxes
    gt_proposal.objectness_logits = gt_logits
    new_proposals = Instances.cat([proposals, gt_proposal])

    return new_proposals

