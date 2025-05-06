"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
COCO evaluator that works in distributed mode.
Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""

import contextlib
import copy
import os

import faster_coco_eval.core.mask as mask_util
import numpy as np
import torch

from .coco_dataset import category_keypoint_layouts

from faster_coco_eval import COCO, COCOeval_faster
from ...core import register
from ...misc import dist_utils

__all__ = [
    "CocoEvaluator",
]


@register()
class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt: COCO = coco_gt
        self.iou_types = iou_types

        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval_faster(
                coco_gt, iouType=iou_type, print_function=print, separate_eval=True
            )

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def cleanup(self):
        self.coco_eval = {}
        for iou_type in self.iou_types:
            self.coco_eval[iou_type] = COCOeval_faster(
                self.coco_gt, iouType=iou_type, print_function=print, separate_eval=True
            )
        self.img_ids = []
        self.eval_imgs = {k: [] for k in self.iou_types}

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            coco_eval = self.coco_eval[iou_type]

            # suppress pycocotools prints
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = self.coco_gt.loadRes(results) if results else COCO()
                    coco_eval.cocoDt = coco_dt
                    coco_eval.params.imgIds = list(img_ids)
                    coco_eval.evaluate()

            self.eval_imgs[iou_type].append(
                np.array(coco_eval._evalImgs_cpp).reshape(
                    len(coco_eval.params.catIds),
                    len(coco_eval.params.areaRng),
                    len(coco_eval.params.imgIds),
                )
            )

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            img_ids, eval_imgs = merge(self.img_ids, self.eval_imgs[iou_type])

            coco_eval = self.coco_eval[iou_type]
            coco_eval.params.imgIds = img_ids
            coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
            coco_eval._evalImgs_cpp = eval_imgs

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            # Filter out predictions for classes not annotated in this image.
            # This is necessary as keypoint annotations may only be for people, but the model can predict bboxes for all categories.
            gt_anns = self.coco_gt.imgToAnns.get(original_id, [])
            image_gt_label_set = {ann["category_id"] for ann in gt_anns}

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            for k, box in enumerate(boxes):
                label_k = labels[k]
                # only score classes present in GT for this image
                if label_k not in image_gt_label_set:
                    continue
                coco_results.append({
                    "image_id": original_id,
                    "category_id": label_k,
                    "bbox": box,
                    "score": scores[k],
                })
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            # Only evaluate classes present in GT with non-empty keypoints.
            gt_anns = self.coco_gt.imgToAnns.get(original_id, [])
            image_gt_label_set = {
                ann["category_id"]
                for ann in gt_anns
                if "keypoints" in ann and any(ann["keypoints"])
            }

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            # KeyPoints TV types are dim 2; coco is dim 3. 
            # Add the third column and set to 2 (visible).
            keypoints = prediction["keypoints"]
            x = keypoints[..., 0]
            y = keypoints[..., 1]
            visible = ((x != 0) & (y != 0)).float().unsqueeze(-1) * 2.0
            keypoints = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), visible], dim=-1)

            for k, label_k in enumerate(labels):
                if label_k not in image_gt_label_set:
                    # Skip if this category isn't in GT (e.g., not 'person').
                    continue

                layout = category_keypoint_layouts[label_k]
                n_kpt = layout["num_keypoints"]
                kpt = keypoints[k]

                # Zero out everything after n_kpt
                if n_kpt < kpt.shape[1]:
                    kpt[:, n_kpt:] = 0

                num_visible_kpts = int(((kpt[:n_kpt, 0] != 0) | (kpt[:n_kpt, 1] != 0)).sum().item())
                kpt = kpt.reshape(1, -1).flatten().tolist()

                coco_results.append({
                    "bbox": boxes[k],
                    "image_id": original_id,
                    "category_id": label_k,
                    "keypoints": kpt,
                    "num_keypoints": num_visible_kpts,
                    "score": scores[k],
                })

        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def merge(img_ids, eval_imgs):
    all_img_ids = dist_utils.all_gather(img_ids)
    all_eval_imgs = dist_utils.all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.extend(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, axis=2).ravel()
    # merged_eval_imgs = np.array(merged_eval_imgs).T.ravel()

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)

    return merged_img_ids.tolist(), merged_eval_imgs.tolist()
