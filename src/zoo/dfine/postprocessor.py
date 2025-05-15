"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.tv_tensors import BoundingBoxes, KeyPoints

from torchvision.transforms.v2.functional import convert_bounding_box_format
from ...core import register

__all__ = ["DFINEPostProcessor"]


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class DFINEPostProcessor(nn.Module):
    __share__ = ["num_classes", "use_focal_loss", "num_top_queries", "remap_mscoco_category"]

    def __init__(
        self, num_classes=80, use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False

    def extra_repr(self) -> str:
        return f"use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}"

    def forward(self, outputs):
        boxes_list = outputs["pred_boxes"]
        keypoints_list = outputs["pred_keypoints"]
        logits = outputs["pred_logits"]

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            # TODO for older tensorrt
            # labels = index % self.num_classes
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            # Select top queries per sample, then index each TVTensor
            boxes = [
                BoundingBoxes(b.data[idx], format=b.format, canvas_size=b.canvas_size)
                for b, idx in zip(boxes_list, index)
            ]
            keypoints = [
                KeyPoints(kpts.data[idx], canvas_size=kpts.canvas_size)
                for kpts, idx in zip(keypoints_list, index)
            ]
        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = [
                    BoundingBoxes(b.data[idx], format=b.format, canvas_size=b.canvas_size)
                    for b, idx in zip(boxes_list, index)
                ]
                keypoints = [
                    KeyPoints(kpts.data[idx], canvas_size=kpts.canvas_size)
                    for kpts, idx in zip(keypoints_list, index)
                ]

        # TODO for onnx export
        if self.deploy_mode:
            return labels, boxes, scores, keypoints

        # TODO
        if self.remap_mscoco_category:
            from ...data.dataset import mscoco_label2category

            labels = (
                torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])
                .to(boxes[0].device)
                .reshape(labels.shape)
            )

        results = [dict(labels=lab, boxes=box, scores=sco, keypoints=kpt) for lab, box, sco, kpt in zip(labels, boxes, scores, keypoints)]
        return results

    def deploy(
        self,
    ):
        self.eval()
        self.deploy_mode = True
        return self
