""" "
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import PIL
import numpy as np
import torch
import torchvision.transforms.v2.functional as F
import torch.utils.data
import torchvision

from typing import List, Dict
from torchvision.utils import draw_bounding_boxes, draw_keypoints
from torchvision.tv_tensors import wrap, BoundingBoxes

torchvision.disable_beta_transforms_warning()

__all__ = ["show_sample", "save_samples"]


def save_samples(samples: torch.Tensor, targets: List[Dict], output_dir: str, split: str, normalized: bool):
    '''
    normalized: whether the boxes are normalized to [0, 1]
    '''
    
    from pathlib import Path
    import os

    os.makedirs(Path(output_dir) / Path(f"{split}_samples"), exist_ok=True)

    for i, (sample, target) in enumerate(zip(samples, targets)):
        sample_visualization = (sample.clone().cpu() * 255).clamp(0, 255).to(torch.uint8)
        target_boxes = wrap(target["boxes"].clone(), like=target["boxes"]) # Need to maintain BoundingBoxes type; clone() reverts to tensor type.
        target_labels = target["labels"].clone().cpu()
        target_image_id = target["image_id"].item()
        target_image_path = target["image_path"]
        target_image_path_stem = Path(target_image_path).stem
        annotated = sample_visualization
        h, w = F.get_size(sample_visualization)

        if len(target_boxes) > 0:
            if normalized:
                rescaled_boxes, canvas_size = F.resize_bounding_boxes(target_boxes, canvas_size=[1,1], size=[h, w])
                target_boxes = BoundingBoxes(rescaled_boxes, format=target_boxes.format, canvas_size=canvas_size)

            target_boxes = F.convert_bounding_box_format(target_boxes, new_format="XYXY")

            # Draw bounding boxes
            annotated = draw_bounding_boxes(
                sample_visualization,
                target_boxes,
                labels=[str(lbl.item()) for lbl in target_labels],
                colors="green",
                width=3
            )

        # Draw keypoints if present
        if "keypoints" in target:
            target_keypoints = wrap(target["keypoints"].clone(), like=target["keypoints"]) # Need to maintain KeyPoints type; clone() reverts to tensor type.
            if normalized:
                target_keypoints[..., 0] *= w
                target_keypoints[..., 1] *= h

            annotated = draw_keypoints(
                annotated,
                target_keypoints,
                colors="cyan",
                radius=3
            )

        annotated = F.to_pil_image(annotated)

        save_path = Path(output_dir) / f"{split}_samples" / f"{target_image_id}_{target_image_path_stem}.webp"
        annotated.save(save_path)

def show_sample(sample):
    """for coco dataset/dataloader"""
    import matplotlib.pyplot as plt
    from torchvision.transforms.v2 import functional as F
    from torchvision.utils import draw_bounding_boxes

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
