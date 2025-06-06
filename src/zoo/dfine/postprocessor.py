"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
            samples = [Image(s) for s in samples]
            boxes = outputs["pred_boxes"]
            keypoints = outputs.get("pred_keypoints", None)
            boxes, keypoints = self._denormalize(samples, boxes, keypoints)

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

            results = self._package_results(samples, labels, boxes, scores, keypoints)
            targets, results = self._run_transforms(targets, results)
            return targets, results

    def _denormalize(self, samples, boxes, keypoints=None):
        denorm_boxes = []
        denorm_keypoints = None

        for box, sample in zip(boxes, samples):
            sample_size = TVF.get_size(sample)
            box = BoundingBoxes(box, format="CXCYWH", canvas_size=(1,1))
            box = TVF.resize(box, size=sample_size)
            box = TVF.convert_bounding_box_format(box, new_format="XYXY")
            denorm_boxes.append(box)

        denorm_boxes = torch.stack(denorm_boxes) if len(denorm_boxes) > 0 else boxes

        if keypoints is not None:
            denorm_keypoints = []
            for kpts, sample in zip(keypoints, samples):
                sample_size = TVF.get_size(sample)
                kpts = KeyPoints(kpts, canvas_size=(1,1))
                kpts = TVF.resize(kpts, size=sample_size)
                denorm_keypoints.append(kpts)

            denorm_keypoints = torch.stack(denorm_keypoints) if len(denorm_keypoints) > 0 else keypoints

        return denorm_boxes, denorm_keypoints

    def _package_results(self, samples, labels, boxes, scores, keypoints):
        if keypoints is not None:
            results = [
                dict(image=img, labels=lab, boxes=box, scores=sco, keypoints=kpt)
                for (img, lab, box, sco, kpt) in zip(samples, labels, boxes, scores, keypoints)
            ]
        else:
            results = [
                dict(image=img, labels=lab, boxes=box, scores=sco)
                for (img, lab, box, sco) in zip(samples, labels, boxes, scores)
            ]

        return results

    def _run_transforms(self, targets, results):
        if self.transforms is None:
            return results

        transformed_results = []
        transformed_targets = []
        for tgt, res in zip(targets, results):
            # orig_size was stored (w,h) while transforms expect (h,w).
            content_size = tgt["orig_size"].flip(dims=[0])

            res["orig_canvas_size"] = content_size
            t_res = self.transforms(res)
            transformed_results.append(t_res[0])
            res.pop("orig_canvas_size")

            tgt["orig_canvas_size"] = content_size
            t_tgt = self.transforms(tgt)
            transformed_targets.append(t_tgt[0])
            tgt.pop("orig_canvas_size")

        return transformed_targets, transformed_results

    def deploy(
        self,
    ):
        self.eval()
        self.deploy_mode = True
        return self
