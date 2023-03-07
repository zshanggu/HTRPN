# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Common data processing utilities that are used in a
typical object detection data pipeline.
"""
import logging
import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image, ImageOps

from fsdet.structures import (
    Boxes,
    BoxMode,
    Instances,
    RotatedBoxes,
)

from . import transforms as T
from .catalog import MetadataCatalog


class SizeMismatchError(ValueError):
    """
    When loaded image has difference width/height compared with annotation.
    """


def read_image(file_name, format=None):
    """
    Read an image into the given format.
    Will apply rotation and flipping if the image has such exif information.

    Args:
        file_name (str): image file path
        format (str): one of the supported image modes in PIL, or "BGR"

    Returns:
        image (np.ndarray): an HWC image
    """
    with PathManager.open(file_name, "rb") as f:
        image = Image.open(f)

        # capture and ignore this bug: https://github.com/python-pillow/Pillow/issues/3973
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass

        if format is not None:
            # PIL only supports RGB, so convert to RGB and flip channels over below
            conversion_format = format
            if format == "BGR":
                conversion_format = "RGB"
            image = image.convert(conversion_format)
        image = np.asarray(image)
        if format == "BGR":
            # flip channels if needed
            image = image[:, :, ::-1]
        # PIL squeezes out the channel dimension for "L", so make it HWC
        if format == "L":
            image = np.expand_dims(image, -1)
        return image


def convert_image_to_rgb(image, format):
    """
    Convert an image from given format to RGB.
    Args:
        image (np.ndarray or Tensor): an HWC image
        format (str): the format of input image, also see `read_image`
    Returns:
        (np.ndarray): (H,W,3) RGB image in 0-255 range, can be either float or uint8
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if format == "BGR":
        image = image[:, :, [2, 1, 0]]
    elif format == "YUV-BT.601":
        image = np.dot(image, np.array(_M_YUV2RGB).T)
        image = image * 255.0
    else:
        if format == "L":
            image = image[:, :, 0]
        image = image.astype(np.uint8)
        image = np.asarray(Image.fromarray(image, mode=format).convert("RGB"))
    return image


def check_image_size(dataset_dict, image):
    """
    Raise an error if the image does not match the size specified in the dict.
    """
    if "width" in dataset_dict or "height" in dataset_dict:
        image_wh = (image.shape[1], image.shape[0])
        expected_wh = (dataset_dict["width"], dataset_dict["height"])
        if not image_wh == expected_wh:
            raise SizeMismatchError(
                "Mismatched (W,H){}, got {}, expect {}".format(
                    " for image " + dataset_dict["file_name"]
                    if "file_name" in dataset_dict
                    else "",
                    image_wh,
                    expected_wh,
                )
            )

    # To ensure bbox always remap to original image size
    if "width" not in dataset_dict:
        dataset_dict["width"] = image.shape[1]
    if "height" not in dataset_dict:
        dataset_dict["height"] = image.shape[0]


def transform_proposals(dataset_dict, image_shape, transforms, min_box_side_len, proposal_topk):
    """
    Apply transformations to the proposals in dataset_dict, if any.

    Args:
        dataset_dict (dict): a dict read from the dataset, possibly
            contains fields "proposal_boxes", "proposal_objectness_logits", "proposal_bbox_mode"
        image_shape (tuple): height, width
        transforms (TransformList):
        min_box_side_len (int): keep proposals with at least this size
        proposal_topk (int): only keep top-K scoring proposals

    The input dict is modified in-place, with abovementioned keys removed. A new
    key "proposals" will be added. Its value is an `Instances`
    object which contains the transformed proposals in its field
    "proposal_boxes" and "objectness_logits".
    """
    if "proposal_boxes" in dataset_dict:
        # Transform proposal boxes
        boxes = transforms.apply_box(
            BoxMode.convert(
                dataset_dict.pop("proposal_boxes"),
                dataset_dict.pop("proposal_bbox_mode"),
                BoxMode.XYXY_ABS,
            )
        )
        boxes = Boxes(boxes)
        objectness_logits = torch.as_tensor(
            dataset_dict.pop("proposal_objectness_logits").astype("float32")
        )

        boxes.clip(image_shape)
        keep = boxes.nonempty(threshold=min_box_side_len)
        boxes = boxes[keep]
        objectness_logits = objectness_logits[keep]

        proposals = Instances(image_shape)
        proposals.proposal_boxes = boxes[:proposal_topk]
        proposals.objectness_logits = objectness_logits[:proposal_topk]
        dataset_dict["proposals"] = proposals


def transform_instance_annotations(annotation, transforms, image_size):
    """
    Apply transforms to box of annotations of a single instance.

    It will use `transforms.apply_box` for the box,.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.

    Args:
        annotation (dict): dict of instance annotations for a single instance.
        transforms (TransformList):
        image_size (tuple): the height, width of the transformed image

    Returns:
        dict: the same input dict with fields "bbox" transformed according to
              `transforms`. The "bbox_mode" field will be set to XYXY_ABS.
    """
    bbox = BoxMode.convert(
        annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    # Note that bbox is 1d (per-instance bounding box)
    annotation["bbox"] = transforms.apply_box([bbox])[0]
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    return annotation


def annotations_to_instances(annos, image_size):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes",
            if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    boxes = target.gt_boxes = Boxes(boxes)
    boxes.clip(image_size)

    classes = [obj["category_id"] for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    return target


def annotations_to_instances_rotated(annos, image_size):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.
    Compared to `annotations_to_instances`, this function is for rotated boxes only

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            Containing fields "gt_boxes", "gt_classes",
            if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = [obj["bbox"] for obj in annos]
    target = Instances(image_size)
    boxes = target.gt_boxes = RotatedBoxes(boxes)
    boxes.clip(image_size)

    classes = [obj["category_id"] for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    return target


def filter_empty_instances(instances, by_box=True):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes

    Returns:
        Instances: the filtered instances.
    """
    assert by_box
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty())

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x
    return instances[m]


def gen_crop_transform_with_instance(crop_size, image_size, instance):
    """
    Generate a CropTransform so that the cropping region contains
    the center of the given instance.

    Args:
        crop_size (tuple): h, w in pixels
        image_size (tuple): h, w
        instance (dict): an annotation dict of one instance, in Detectron2's
            dataset format.
    """
    crop_size = np.asarray(crop_size, dtype=np.int32)
    bbox = BoxMode.convert(instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS)
    center_yx = (bbox[1] + bbox[3]) * 0.5, (bbox[0] + bbox[2]) * 0.5

    min_yx = np.maximum(np.floor(center_yx).astype(np.int32) - crop_size, 0)
    max_yx = np.maximum(np.asarray(image_size, dtype=np.int32) - crop_size, 0)
    max_yx = np.minimum(max_yx, np.ceil(center_yx).astype(np.int32))

    y0 = np.random.randint(min_yx[0], max_yx[0] + 1)
    x0 = np.random.randint(min_yx[1], max_yx[1] + 1)
    return T.CropTransform(x0, y0, crop_size[1], crop_size[0])


def check_metadata_consistency(key, dataset_names):
    """
    Check that the datasets have consistent metadata.

    Args:
        key (str): a metadata key
        dataset_names (list[str]): a list of dataset names

    Raises:
        AttributeError: if the key does not exist in the metadata
        ValueError: if the given datasets do not have the same metadata values defined by key
    """
    if len(dataset_names) == 0:
        return
    logger = logging.getLogger(__name__)
    entries_per_dataset = [getattr(MetadataCatalog.get(d), key) for d in dataset_names]
    for idx, entry in enumerate(entries_per_dataset):
        if entry != entries_per_dataset[0]:
            logger.error(
                "Metadata '{}' for dataset '{}' is '{}'".format(key, dataset_names[idx], str(entry))
            )
            logger.error(
                "Metadata '{}' for dataset '{}' is '{}'".format(
                    key, dataset_names[0], str(entries_per_dataset[0])
                )
            )
            raise ValueError("Datasets have different metadata '{}'!".format(key))


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Now it includes resizing and flipping.

    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"

    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(
            len(min_size)
        )

    logger = logging.getLogger(__name__)
    tfm_gens = []
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))

    if is_train:
        tfm_gens.append(T.RandomFlip())
        if cfg.INPUT.USE_TRANSFORM_AUG:
            # Newer detectron2 has a RandomApply https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/transforms/augmentation_impl.py
            # currently, every augmentation is about to happen definitely
            tfm_gens.append(T.RandomBrightness(0.85, 1.15))
            tfm_gens.append(T.RandomContrast(0.85, 1.15))
            tfm_gens.append(T.RandomSaturation(0.85, 1.15))
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens
