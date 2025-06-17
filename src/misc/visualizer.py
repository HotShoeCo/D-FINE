""" "
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import PIL
import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.transforms.v2.functional as F

from pathlib import Path
from PIL import ImageFont
from torchvision.tv_tensors import wrap
from torchvision.utils import draw_bounding_boxes, draw_keypoints
from typing import Optional, Tuple, List, Dict, Union


__all__ = ["save_samples"]

BOX_COLORS = [
    "red", "blue", "green", "orange", "purple",
    "cyan", "magenta", "yellow", "lime", "pink",
    "teal", "lavender", "brown", "beige", "maroon",
    "navy", "olive", "coral", "turquoise", "gold"
]


def _get_alternating_colors(count):
    return [
        BOX_COLORS[i % len(BOX_COLORS)]
        for i in range(count)
    ]


def save_samples(
    output_dir: str,
    split: str,
    images: torch.Tensor,
    annotations: List[Dict],
    threshold: float = 0.50,
):
    os.makedirs(Path(output_dir) / Path(f"{split}_samples"), exist_ok=True)

    font = ImageFont.load_default()
    font.size = 32

    for sample, anno in zip(images, annotations):
        label_strings = [str(int(l)) for l in anno["labels"]]
        gt_boxes = anno["boxes"]
        gt_kps = anno.get("keypoints")
        scores = anno.get("scores")

        if scores is not None:
            keep = scores > threshold
            if keep.sum() == 0:
                continue
            # filter boxes, label_strings, colors, and keypoints
            gt_boxes = gt_boxes[keep]
            label_strings = [label_strings[i] for i in range(len(label_strings)) if keep[i]]
            if gt_kps is not None:
                gt_kps = [gt_kps[i] for i in range(len(gt_kps)) if keep[i]]

        colors = _get_alternating_colors(gt_boxes.shape[0])
        out_img = sample.clone().cpu()
        out_img = draw_bounding_boxes(out_img, gt_boxes, labels=label_strings, colors=colors, width=3)

        if gt_kps is not None:
            for k, c in zip(gt_kps, colors):
                k = k.unsqueeze(0)
                out_img = draw_keypoints(out_img, k, colors=c, radius=2)

        save_path = Path(output_dir) / f"{split}_samples" / f"{Path(anno['image_path']).stem}.webp"
        pil_img = F.to_pil_image(out_img)
        pil_img.save(save_path)
