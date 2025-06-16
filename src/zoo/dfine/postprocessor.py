"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TVF

from torchvision.tv_tensors import BoundingBoxes, Image, KeyPoints, set_return_type
from ...core import register
from ...data.transforms.container import Compose

__all__ = ["DFINEPostProcessor"]


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class DFINEPostProcessor(nn.Module):
    __share__ = ["num_classes", "use_focal_loss", "num_top_queries", "remap_mscoco_category"]

    def __init__(
        self,
        num_classes=80,
        use_focal_loss=True,
        num_top_queries=300,
        remap_mscoco_category=False,
        transforms=None,
    ) -> None:

        super().__init__()
        self.deploy_mode = False
        self.num_classes = int(num_classes)
        self.num_top_queries = num_top_queries
        self.remap_mscoco_category = remap_mscoco_category
        self.transforms = Compose(ops=transforms)
        self.use_focal_loss = use_focal_loss

    def extra_repr(self) -> str:
        return f"use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}"

    def forward(self, samples, outputs, targets):
        with set_return_type("TVTensor"):
            logits = outputs["pred_logits"]
            boxes = outputs["pred_boxes"]
            keypoints = outputs.get("pred_keypoints", None)

            if self.use_focal_loss:
                scores = F.sigmoid(logits)
                scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
                # TODO for older tensorrt
                # labels = index % self.num_classes
                labels = mod(index, self.num_classes)
                index = index // self.num_classes
                boxes = boxes.gather(
                    dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1])
                )
                if keypoints is not None:
                    batch_size, _, num_kpts, _ = keypoints.shape
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
                        batch_size, _, num_kpts, _ = keypoints.shape
                        keypoints = keypoints.gather(
                            dim=1,
                            index=index.view(batch_size, self.num_top_queries, 1, 1)
                                .expand(batch_size, self.num_top_queries, num_kpts, 2)
                        )
            
            boxes, keypoints = self._wrap_outputs(boxes, keypoints)

            # TODO for onnx export
            if self.deploy_mode:
                raise "TODO: check this return type. Perhaps just make it fit with normal return type."
                return labels, boxes, scores, keypoints

            # TODO
            if self.remap_mscoco_category:
                from ...data.dataset import mscoco_label2category

                labels = (
                    torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])
                    .to(boxes[0].device)
                    .reshape(labels.shape)
                )

            # Add the image to the packed values for sample saving methods.
            image_paths = [t["image_path"] for t in targets]
            results = self._pack_results(boxes=boxes, labels=labels, scores=scores, keypoints=keypoints, image_path=image_paths)
            samples, results, targets = self._run_transforms(samples, results, targets)
            return samples, results, targets

    def _pack_results(self, **kwargs):
        # Filter out None values
        valid_items = {k: v for k, v in kwargs.items() if v is not None}
        field_names = list(valid_items.keys())
        field_values = list(valid_items.values())
        return [dict(zip(field_names, values)) for values in zip(*field_values)]
    
    def _wrap_outputs(self, boxes, keypoints=None):
        boxes = [
                BoundingBoxes(b, format="CXCYWH", canvas_size=(1, 1))
                for b in boxes
            ]

        if keypoints is not None:
            keypoints = [
                KeyPoints(k, canvas_size=(1, 1))
                for k in keypoints
            ]

        return boxes, keypoints

    def _run_transforms(self, samples, results, targets):
        if self.transforms is None:
            return samples, results, targets

        transformed_results = []
        transformed_samples = []
        transformed_targets = []

        for smpl, res, tgt in zip(samples, results, targets):
            # orig_size was stored (w,h) while transforms expect (h,w).
            content_size = tgt["orig_size"].flip(dims=[0])

            # Transforms need images to get the size and function properly. We do not want to hold references to
            # the images in the results/targets, hence the push/pop procedures here.
            res["orig_canvas_size"] = content_size
            res["image"] = smpl
            t_res = self.transforms(res)
            t_res = t_res[0]
            t_res.pop("image")
            transformed_results.append(t_res)

            tgt["orig_canvas_size"] = content_size
            tgt["image"] = smpl
            t_tgt = self.transforms(tgt)
            t_tgt = t_tgt[0]
            t_smpl = t_tgt.pop("image")
            transformed_samples.append(t_smpl)
            transformed_targets.append(t_tgt)

        return transformed_samples, transformed_results, transformed_targets

    def deploy(
        self,
    ):
        self.eval()
        self.deploy_mode = True
        return self
