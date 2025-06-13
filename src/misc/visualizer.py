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


__all__ = ["show_sample", "save_samples"]

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


def show_sample(sample, pred=None):
    """for coco dataset/dataloader"""
    raise "TODO: not fixed after code changes. Possibly unused code?"
    import matplotlib.pyplot as plt

    image, target = sample
    if isinstance(image, PIL.Image.Image):
        image = F.to_image_tensor(image)

    image = F.convert_dtype(image, torch.uint8)
    image_size = F.get_size(image)

    label_strings = [str(int(l)) for l in target.get("labels", torch.tensor([])).clone().cpu().tolist()]

    # draw predictions if provided
    if pred is not None:
        pred_boxes = wrap(pred["boxes"].clone().cpu(), like=pred["boxes"])
        pred_kps = pred.get("keypoints")
        rendered_img = _draw_annotations(
            image, pred_boxes, pred_kps, None,
            False, image_size,
            colors=_get_alternating_colors(pred_boxes.shape[0]),
            width=3, radius=5
        )
        # draw ground truth in lime
        gt_boxes = wrap(target["boxes"].clone().cpu(), like=target["boxes"])
        gt_kps = target.get("keypoints")
        rendered_img = _draw_annotations(
            rendered_img,
            gt_boxes,
            gt_kps,
            label_strings,
            False,
            image_size,
            colors="lime",
            width=3,
            radius=2
        )
    else:
        # no preds: rotate GT colors
        gt_boxes = wrap(target["boxes"].clone().cpu(), like=target["boxes"])
        gt_kps = target.get("keypoints")
        rendered_img = _draw_annotations(
            image,
            gt_boxes,
            gt_kps,
            label_strings,
            False,
            image_size,
            colors=_get_alternating_colors(gt_boxes.shape[0]),
            width=3,
            radius=2
        )

    fig, ax = plt.subplots()
    ax.imshow(rendered_img.permute(1, 2, 0).numpy())
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.tight_layout()
    fig.show()
    plt.show()
