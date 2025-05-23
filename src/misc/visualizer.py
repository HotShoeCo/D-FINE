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

from torchvision.tv_tensors import wrap, Image
from torchvision.utils import draw_bounding_boxes
from typing import List, Dict

torchvision.disable_beta_transforms_warning()

__all__ = ["show_sample", "save_samples"]

def save_samples(samples: torch.Tensor, targets: List[Dict], output_dir: str, split: str, normalized: bool):
    '''
    normalized: whether the boxes are normalized to [0, 1]
    '''
    from torchvision.transforms.v2.functional import to_pil_image
    from pathlib import Path
    from PIL import ImageFont
    import os

    os.makedirs(Path(output_dir) / Path(f"{split}_samples"), exist_ok=True)
    # Predefined colors (standard color names recognized by PIL)
    BOX_COLORS = [
        "red", "blue", "green", "orange", "purple",
        "cyan", "magenta", "yellow", "lime", "pink",
        "teal", "lavender", "brown", "beige", "maroon",
        "navy", "olive", "coral", "turquoise", "gold"
    ]

    LABEL_TEXT_COLOR = "white"

    font = ImageFont.load_default()
    font.size = 32

    for i, (sample, target) in enumerate(zip(samples, targets)):
        sample_visualization = sample.clone().cpu()
        sample_size = F.get_size(sample_visualization)

        target_boxes = wrap(target["boxes"].clone().cpu(), like=target["boxes"])
        target_labels = target["labels"].clone().cpu()
        target_image_id = target["image_id"].item()
        target_image_path = target["image_path"]
        target_image_path_stem = Path(target_image_path).stem

        # normalized to pixel space
        if normalized:
            sample_visualization = F.to_dtype(sample_visualization, torch.uint8, scale=True)
            target_boxes = F.resize(target_boxes, sample_size)

        # any box format -> xyxy
        target_boxes = F.convert_bounding_box_format(target_boxes, new_format="XYXY")

        # clip to image size
        target_boxes = F.clamp_bounding_boxes(target_boxes)

        label_strings = [str(int(l)) for l in target_labels.tolist()]
        box_colors = [BOX_COLORS[int(label) % len(BOX_COLORS)] for label in target_labels]
        rendered_img = draw_bounding_boxes(
            sample_visualization,
            target_boxes,
            labels=label_strings,
            colors=box_colors,
            width=3
        )

        save_path = Path(output_dir) / f"{split}_samples" / f"{target_image_id}_{target_image_path_stem}.webp"
        pil_img = to_pil_image(rendered_img)
        pil_img.save(save_path)

def show_sample(sample):
    """for coco dataset/dataloader"""
    import matplotlib.pyplot as plt

    image, target = sample
    if isinstance(image, PIL.Image.Image):
        image = F.to_image_tensor(image)

    image = F.convert_dtype(image, torch.uint8)
    annotated_image = draw_bounding_boxes(image, target["boxes"], colors="yellow", width=3)

    fig, ax = plt.subplots()
    ax.imshow(annotated_image.permute(1, 2, 0).numpy())
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.tight_layout()
    fig.show()
    plt.show()
