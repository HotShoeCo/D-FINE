""" "
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import PIL
import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.transforms.v2.functional as F

from PIL import ImageColor
from torchvision.tv_tensors import wrap, Image
from torchvision.utils import draw_bounding_boxes, draw_keypoints
from typing import Optional, Tuple, List, Dict, Union

torchvision.disable_beta_transforms_warning()

__all__ = ["show_sample", "save_samples"]

BOX_COLORS = [
    "red", "blue", "green", "orange", "purple",
    "cyan", "magenta", "yellow", "lime", "pink",
    "teal", "lavender", "brown", "beige", "maroon",
    "navy", "olive", "coral", "turquoise", "gold"
]

def _draw_annotations(
    image: torch.Tensor,
    boxes: torch.Tensor,
    keypoints: Optional[torch.Tensor],
    labels: Optional[List[str]],
    normalized: bool,
    image_size: Tuple[int, int],
    colors: Union[str, List[str]],
    width: int = 3,
    radius: float = 2
) -> torch.Tensor:
    """
    Draw boxes and keypoints on image. Rotates colors if a list is provided.
    """
    img = image
    # normalize to pixel space
    if normalized:
        img = F.to_dtype(img, torch.uint8, scale=True)
        boxes = F.resize(boxes, image_size)
        if keypoints is not None:
            keypoints = F.resize(wrap(keypoints.clone().cpu(), like=keypoints), image_size)
    # convert boxes to XYXY and clamp
    boxes = F.convert_bounding_box_format(boxes, new_format="XYXY")
    boxes = F.clamp_bounding_boxes(boxes)
    # prepare color list
    num = boxes.shape[0]
    if isinstance(colors, str):
        color_list = [colors] * num
    else:
        color_list = colors
    # draw one by one to rotate colors
    for i in range(num):
        box = boxes[i].unsqueeze(0)
        c = color_list[i % len(color_list)]
        lbl = [labels[i]] if labels is not None else None
        img = draw_bounding_boxes(img, box, labels=lbl, colors=c, width=width)
        if keypoints is not None:
            kp = keypoints[i].unsqueeze(0)
            img = draw_keypoints(img, kp, colors=c, radius=radius)
    return img


def _get_alternating_colors(count):
    return [
        BOX_COLORS[i % len(BOX_COLORS)]
        for i in range(count)
    ]


def save_samples(
    output_dir: str,
    split: str,
    samples: torch.Tensor,
    targets: List[Dict],
    preds: Optional[List[Dict]]=None,
    min_score: Optional[float]=0.5,
    normalized: bool=False
):
    '''
    normalized: whether the boxes are normalized to [0, 1]
    preds: optional list of prediction dicts with "boxes" and optionally "keypoints"
    '''
    from pathlib import Path
    from PIL import ImageFont
    import os

    os.makedirs(Path(output_dir) / Path(f"{split}_samples"), exist_ok=True)

    font = ImageFont.load_default()
    font.size = 32

    for i, (sample, target) in enumerate(zip(samples, targets)):
        pred = preds[i] if preds is not None else None
        sample_visualization = sample.clone().cpu()
        sample_size = F.get_size(sample_visualization)

        if normalized:
            sample_visualization = F.to_dtype(sample_visualization, torch.uint8, scale=True)

        # draw predictions if provided
        if pred is not None:
            filter = pred["scores"] > min_score

            pred_labels = pred["labels"][filter]
            pred_label_strings = [str(int(l)) for l in pred_labels.clone().cpu().tolist()]

            pred_boxes = pred["boxes"][filter].clone().cpu()
            pred_boxes = wrap(pred_boxes, like=pred["boxes"])

            if "keypoints" in pred:
                pred_kps = pred["keypoints"][filter].clone().cpu()
                pred_kps = wrap(pred_kps, like=pred["keypoints"])

            rendered_img = _draw_annotations(
                sample_visualization,
                pred_boxes,
                pred_kps,
                pred_label_strings,
                normalized,
                sample_size,
                colors=_get_alternating_colors(pred_boxes.shape[0]),
                width=3,
                radius=5
            )

            # draw ground truth in lime
            gt_label_strings = [str(int(l)) for l in target["labels"].clone().cpu().tolist()]
            gt_boxes = wrap(target["boxes"].clone().cpu(), like=target["boxes"])
            gt_kps = target.get("keypoints")
            if gt_kps is not None:
                gt_kps = wrap(gt_kps.clone().cpu(), like=gt_kps)
            
            rendered_img = _draw_annotations(
                rendered_img,
                gt_boxes,
                gt_kps,
                gt_label_strings,
                normalized,
                sample_size,
                colors="lime",
                width=3,
                radius=2
            )
        else:
            # no preds: rotate GT colors
            label_strings = [str(int(l)) for l in target["labels"].clone().cpu().tolist()]
            gt_boxes = wrap(target["boxes"].clone().cpu(), like=target["boxes"])
            gt_kps = target.get("keypoints")
            rendered_img = _draw_annotations(
                sample_visualization,
                gt_boxes,
                gt_kps,
                label_strings,
                normalized,
                sample_size,
                colors=_get_alternating_colors(gt_boxes.shape[0]),
                width=3,
                radius=2
            )

        save_path = Path(output_dir) / f"{split}_samples" / f"{target['image_id'].item()}_{Path(target['image_path']).stem}.webp"
        pil_img = F.to_pil_image(rendered_img)
        pil_img.save(save_path)

def show_sample(sample, pred=None):
    """for coco dataset/dataloader"""
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
