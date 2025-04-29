""" "
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import PIL
import numpy as np
import torch
import torch.utils.data
import torchvision
from typing import List, Dict

torchvision.disable_beta_transforms_warning()

__all__ = ["show_sample", "save_samples"]

def save_samples(samples: torch.Tensor, targets: List[Dict], output_dir: str, split: str, normalized: bool, box_fmt: str):
    '''
    normalized: whether the boxes are normalized to [0, 1]
    box_fmt: 'xyxy', 'xywh', 'cxcywh', D-FINE uses 'cxcywh' for training, 'xyxy' for validation
    '''
    from torchvision.transforms.v2.functional import to_pil_image
    from torchvision.ops import box_convert
    from torchvision.utils import draw_bounding_boxes, draw_keypoints
    from pathlib import Path
    import os

    os.makedirs(Path(output_dir) / Path(f"{split}_samples"), exist_ok=True)

    for i, (sample, target) in enumerate(zip(samples, targets)):
        sample_visualization = (sample.clone().cpu() * 255).clamp(0, 255).to(torch.uint8)
        target_boxes = target["boxes"].clone().cpu()
        target_labels = target["labels"].clone().cpu()
        target_image_id = target["image_id"].item()
        target_image_path = target["image_path"]
        target_image_path_stem = Path(target_image_path).stem

        if normalized:
            w, h = sample_visualization.shape[-1], sample_visualization.shape[-2]
            target_boxes[:, [0, 2]] *= w
            target_boxes[:, [1, 3]] *= h

        target_boxes = box_convert(target_boxes, in_fmt=box_fmt, out_fmt="xyxy")
        target_boxes = target_boxes.to(torch.int32)
        target_labels = target_labels.to(torch.int64)

        # Draw bounding boxes
        annotated = draw_bounding_boxes(
            sample_visualization,
            target_boxes,
            labels=[str(lbl.item()) for lbl in target_labels],
            colors="red",
            width=3,
            font_size=12
        )

        # Draw keypoints if present
        if "keypoints" in target:
            keypoints = target["keypoints"].clone().cpu()
            if normalized:
                keypoints[..., 0] *= w
                keypoints[..., 1] *= h
            keypoints = keypoints[..., :2]
            keypoints = keypoints.to(torch.int32)

            annotated = draw_keypoints(
                annotated,
                keypoints,
                colors="cyan",
                radius=3
            )

        annotated = to_pil_image(annotated)

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
