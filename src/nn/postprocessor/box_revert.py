"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from enum import Enum

import torch
import torchvision
from torch import Tensor
from torchvision.transforms.v2.functional import convert_bounding_box_format


class BoxProcessFormat(Enum):
    """Box process format

    Available formats are
    * ``RESIZE``
    * ``RESIZE_KEEP_RATIO``
    * ``RESIZE_KEEP_RATIO_PADDING``
    """

    RESIZE = 1
    RESIZE_KEEP_RATIO = 2
    RESIZE_KEEP_RATIO_PADDING = 3


def box_revert(
    boxes: Tensor,
    orig_sizes: Tensor = None,
    eval_sizes: Tensor = None,
    inpt_sizes: Tensor = None,
    inpt_padding: Tensor = None,
    normalized: bool = True,
    in_fmt: str = "CXCYWH",
    out_fmt: str = "XYXY",
    process_fmt=BoxProcessFormat.RESIZE,
) -> Tensor:
    """
    Args:
        boxes(Tensor), [N, :, 4], (x1, y1, x2, y2), pred boxes.
        inpt_sizes(Tensor), [N, 2], (w, h). input sizes.
        orig_sizes(Tensor), [N, 2], (w, h). origin sizes.
        inpt_padding (Tensor), [N, 2], (w_pad, h_pad, ...).
        (inpt_sizes + inpt_padding) == eval_sizes
    """
    assert in_fmt in ("CXCYWH", "XYXY"), ""

    if normalized and eval_sizes is not None:
        boxes = boxes * eval_sizes.repeat(1, 2).unsqueeze(1)

    if inpt_padding is not None:
        if in_fmt == "XYXY":
            boxes -= inpt_padding[:, :2].repeat(1, 2).unsqueeze(1)
        elif in_fmt == "CXCYWH":
            boxes[..., :2] -= inpt_padding[:, :2].repeat(1, 2).unsqueeze(1)

    if orig_sizes is not None:
        orig_sizes = orig_sizes.repeat(1, 2).unsqueeze(1)
        if inpt_sizes is not None:
            inpt_sizes = inpt_sizes.repeat(1, 2).unsqueeze(1)
            boxes = boxes * (orig_sizes / inpt_sizes)
        else:
            boxes = boxes * orig_sizes

    boxes = convert_bounding_box_format(boxes, old_format=in_fmt, new_format=out_fmt)
    return boxes
