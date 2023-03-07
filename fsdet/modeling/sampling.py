# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch

__all__ = ["subsample_labels"]


def subsample_labels(labels, num_samples, positive_fraction, bg_label):
    """
    Return `num_samples` random samples from `labels`, with a fraction of
    positives no larger than `positive_fraction`.

    Args:
        labels (Tensor): (N, ) label vector with values:
            * -1: ignore
            * bg_label: background ("negative") class
            * otherwise: one or more foreground ("positive") classes
        num_samples (int): The total number of labels with value >= 0 to return.
            Values that are not sampled will be filled with -1 (ignore).
        positive_fraction (float): The number of subsampled labels with values > 0
            is `min(num_positives, int(positive_fraction * num_samples))`. The number
            of negatives sampled is `min(num_negatives, num_samples - num_positives_sampled)`.
            In order words, if there are not enough positives, the sample is filled with
            negatives. If there are also not enough negatives, then as many elements are
            sampled as is possible.
        bg_label (int): label index of background ("negative") class.

    Returns:
        pos_idx, neg_idx (Tensor):
            1D indices. The total number of indices is `num_samples` if possible.
            The fraction of positive indices is `positive_fraction` if possible.
    """
    positive = torch.nonzero((labels != -1) & (labels != bg_label)).squeeze(1)
    negative = torch.nonzero(labels == bg_label).squeeze(1)

    num_pos = int(num_samples * positive_fraction)
    # protect against not enough positive examples
    num_pos = min(positive.numel(), num_pos)
    num_neg = num_samples - num_pos
    # protect against not enough negative examples
    num_neg = min(negative.numel(), num_neg)

    # randomly select positive and negative examples
    perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

    pos_idx = positive[perm1]
    neg_idx = negative[perm2]
    return pos_idx, neg_idx


def subsample_labels_hierc(labels, num_samples, positive_fraction, bg_label, anchors_hierc):
    """
    Return `num_samples` random samples from `labels`, with a fraction of
    positives no larger than `positive_fraction`.

    Args:
        labels (Tensor): (N, ) label vector with values:
            * -1: ignore
            * bg_label: background ("negative") class
            * otherwise: one or more foreground ("positive") classes
        num_samples (int): The total number of labels with value >= 0 to return.
            Values that are not sampled will be filled with -1 (ignore).
        positive_fraction (float): The number of subsampled labels with values > 0
            is `min(num_positives, int(positive_fraction * num_samples))`. The number
            of negatives sampled is `min(num_negatives, num_samples - num_positives_sampled)`.
            In order words, if there are not enough positives, the sample is filled with
            negatives. If there are also not enough negatives, then as many elements are
            sampled as is possible.
        bg_label (int): label index of background ("negative") class.

    Returns:
        pos_idx, neg_idx (Tensor):
            1D indices. The total number of indices is `num_samples` if possible.
            The fraction of positive indices is `positive_fraction` if possible.
    """
    anchors_hierc_idx = [0]
    tmp_num = 0
    for p_num in anchors_hierc:
        anchors_hierc_idx.append(p_num + tmp_num)
        tmp_num += p_num
    # print("anchors_hierc_idx!!!!!!!!!!!!!!!!", anchors_hierc_idx)
    labels_hierc = [labels[anchors_hierc_idx[i]:anchors_hierc_idx[i+1]] for i in range(len(anchors_hierc_idx) -1)]
    # print("labels_hierc!!!!!!!!!!!!!!!!", len(labels_hierc[0]), len(labels_hierc[1]), len(labels_hierc[2]), len(labels_hierc[3]), len(labels_hierc[4]))  # p6 torch.Size([507, 4])

    positive = torch.nonzero((labels != -1) & (labels != bg_label)).squeeze(1)
    # print("positive!!!!!!!!!!!!!!!!", positive)
    positive_hierc = [torch.nonzero((labels_hierc[i] != -1) & (labels_hierc[i] != bg_label)).squeeze(1) + anchors_hierc_idx[i] for i in range(len(labels_hierc))]
    # print("positive_hierc!!!!!!!!!!!!!!!!", positive_hierc)
    negative = torch.nonzero(labels == bg_label).squeeze(1)
    # print("negative!!!!!!!!!!!!!!!!", negative)
    negative_hierc = [torch.nonzero(labels_hierc[i] == bg_label).squeeze(1) + anchors_hierc_idx[i] for i in range(len(labels_hierc))]
    # print("negative_hierc!!!!!!!!!!!!!!!!", negative_hierc)

    num_pos = int(num_samples * positive_fraction)
    # protect against not enough positive examples
    num_pos = min(positive.numel(), num_pos)
    num_neg = num_samples - num_pos
    # protect against not enough negative examples
    num_neg = min(negative.numel(), num_neg)
    # print("num_neg!!!!!!!!!!!!!!", num_neg)

    num_neg_hierc = [-1, -1, -1, -1, -1]
    neg_to_minus = 0
    hierc_to_minus = 0
    for i in range(len(negative_hierc)):
        # print("negative_hierc[i]!!!!!!!!!!!!!!", len(negative_hierc[i]))
        if len(negative_hierc[i]) < num_neg / 5:
            num_neg_hierc[i] = len(negative_hierc[i])
            neg_to_minus += len(negative_hierc[i])
            hierc_to_minus += 1
    num_neg_remain_per_p_quotident = (num_neg - neg_to_minus) // (5 - hierc_to_minus)
    num_neg_remain_per_p_remainder = (num_neg - neg_to_minus) % (5 - hierc_to_minus)
    idx_remainder = 0
    for i in range(len(negative_hierc)):
        if num_neg_hierc[i] == -1:
            if 5 - hierc_to_minus - idx_remainder <= num_neg_remain_per_p_remainder:
                num_neg_hierc[i] = num_neg_remain_per_p_quotident + 1
            else:
                num_neg_hierc[i] = num_neg_remain_per_p_quotident
            idx_remainder += 1
    # print("sum(num_neg_hierc)!!!!!!!!!!!!!!", sum(num_neg_hierc), num_neg_hierc)
    assert sum(num_neg_hierc) == num_neg

    # randomly select positive and negative examples
    perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    perm2 = [torch.randperm(negative_hierc[i].numel(), device=negative.device)[:num_neg_hierc[i]] for i in range(len(negative_hierc))]
    # print("perm2_!!!!!!!!!!!!!!", perm2)

    pos_idx = positive[perm1]
    # for i in range(len(negative_hierc)):
        # print("negative_hierc[i].numel()", negative_hierc[i].numel(), num_neg_hierc[i])
        # print("negative_hierc[i]!!!!!!!!!!!!!!", negative_hierc[i])
        # print("i", i, negative_hierc[i][perm2[i]])
    neg_idx = torch.cat([negative_hierc[i][perm2[i]] for i in range(len(negative_hierc))], 0)
    # print("neg_idx_!!!!!!!!!!!!!!", neg_idx)
    return pos_idx, neg_idx, num_neg_hierc