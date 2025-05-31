"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchvision.tv_tensors import BoundingBoxes, KeyPoints
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

    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits = outputs["pred_logits"]
        boxes = outputs["pred_boxes"]
        keypoints = outputs.get("pred_keypoints", None)

        if keypoints is not None:
            batch_size, _, num_kpts, _ = keypoints.shape
            # keypoints are [B, N, K, 2], normalized; scale by image width/height
            # orig_target_sizes is [B, 2] (width, height)
            scale = orig_target_sizes.view(batch_size, 1, 1, 2)
            keypoints *= scale

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            # TODO for older tensorrt
            # labels = index % self.num_classes
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = bbox_pred.gather(
                dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1])
            )
            if keypoints is not None:
                keypoints = keypoints.gather(
                    dim=1,
                    index=index.view(batch_size, self.num_top_queries, 1, 1)
                          .expand(batch_size, self.num_top_queries, num_kpts, 2)
                )

        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(
                    boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1])
                )
                if keypoints is not None:
                    keypoints = keypoints.gather(
                        dim=1,
                        index=index.view(batch_size, self.num_top_queries, 1, 1)
                              .expand(batch_size, self.num_top_queries, num_kpts, 2)
                    )

        # TODO for onnx export
        if self.deploy_mode:
            return labels, boxes, scores, keypoints

        # TODO
        if self.remap_mscoco_category:
            from ...data.dataset import mscoco_label2category

            labels = (
                torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])
                .to(boxes.device)
                .reshape(labels.shape)
            )

        # Package results and wrap tv tensor types.
        results = []
        for i, (lab, box, sco, size) in enumerate(zip(labels, boxes, scores, orig_target_sizes)):
            tvbox = BoundingBoxes(box, format="XYXY", canvas_size=size)
            result = dict(labels=lab, boxes=tvbox, scores=sco)
            if keypoints is not None:
                result["keypoints"] = KeyPoints(keypoints[i], canvas_size=size)
            results.append(result)

        return results

    def deploy(
        self,
    ):
        self.eval()
        self.deploy_mode = True
        return self
