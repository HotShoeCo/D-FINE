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
from ...data.transforms import DenormalizeAnnotations, ConvertBoundingBoxFormat
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
        self.post_transforms = Compose(ops=transforms)
        self.use_focal_loss = use_focal_loss

        self._raw_output_transforms = T.Compose([
            DenormalizeAnnotations(),
            ConvertBoundingBoxFormat("XYXY"),
        ])

    def extra_repr(self) -> str:
        return f"use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}"

    def forward(self, samples, outputs, targets):
        with set_return_type("TVTensor"):
            logits = outputs["pred_logits"]
            boxes = outputs["pred_boxes"]
            keypoints = outputs.get("pred_keypoints", None)
            boxes, keypoints = self._prepare_raw_outputs(samples, boxes, keypoints)

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

            # Add the image to the packed values for sample saving methods.
            image_paths = [t["image_path"] for t in targets]
            results = self._pack_results(boxes=boxes, labels=labels, scores=scores, keypoints=keypoints, image_path=image_paths)
            results, targets = self._post_transforms(results, targets)
            return results, targets

    def _prepare_raw_outputs(self, samples, boxes, keypoints=None):
        # Wrap with TV Tensors.
        samples = Image(samples)
        boxes = [
            BoundingBoxes(b, format="CXCYWH", canvas_size=(1, 1))
            for b in boxes
        ]

        if keypoints is not None:
            keypoints = [
                KeyPoints(k, canvas_size=(1, 1))
                for k in keypoints
            ]

        # Transform
        results = self._pack_results(image=samples, boxes=boxes, keypoints=keypoints)
        results = self._raw_output_transforms(results)

        # Merge types again for scoring procedures.
        boxes = [r["boxes"] for r in results]
        boxes = torch.stack(boxes)

        keypoints = [r["keypoints"] for r in results if r.get("keypoints") is not None]
        if len(keypoints) > 0:
            keypoints = torch.stack(keypoints)
        else:
            keypoints = None

        return boxes, keypoints

    def _pack_results(self, **kwargs):
        # Filter out None values
        valid_items = {k: v for k, v in kwargs.items() if v is not None}
        field_names = list(valid_items.keys())
        field_values = list(valid_items.values())
        return [dict(zip(field_names, values)) for values in zip(*field_values)]

    def _post_transforms(self, results, targets):
        if self.post_transforms is None:
            return results, targets

        transformed_results = []
        transformed_targets = []
        for res, tgt in zip(results, targets):
            # orig_size was stored (w,h) while transforms expect (h,w).
            content_size = tgt["orig_size"].flip(dims=[0])

            res["orig_canvas_size"] = content_size
            t_res = self.post_transforms(res)
            transformed_results.append(t_res[0])
            res.pop("orig_canvas_size")

            tgt["orig_canvas_size"] = content_size
            t_tgt = self.post_transforms(tgt)
            transformed_targets.append(t_tgt[0])
            tgt.pop("orig_canvas_size")

        return transformed_results, transformed_targets

    def deploy(
        self,
    ):
        self.eval()
        self.deploy_mode = True
        return self
