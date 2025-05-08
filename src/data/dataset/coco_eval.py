"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
COCO evaluator that works in distributed mode.
Mostly copy-paste from https://github.com/pytorch/vision/blob/main/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""

from contextlib import redirect_stdout
import copy
import faster_coco_eval.core.mask as mask_util
import numpy as np
import io
import torch

from .coco_dataset import category_keypoint_layouts

from faster_coco_eval import COCO, COCOeval_faster
from ...core import register
from ...misc import dist_utils

__all__ = [
    "CocoEvaluator",
]


@register()
class CocoEvaluator:
    def __init__(self, coco_gt, iou_types):
        if not isinstance(iou_types, (list, tuple)):
            raise TypeError(f"This constructor expects iou_types of type list or tuple, instead  got {type(iou_types)}")
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt
        self.iou_types = iou_types
        self.reset()

    def reset(self):
        """
        Called between batches to reset the evaluator data.
        """
        self.coco_eval = {}
        self.img_ids = []
        self.eval_imgs = {k: [] for k in self.iou_types}
        
        # Standard COCO sigmas (for 17 keypoints)
        oks_sigmas = [
            0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62,
            0.62, 1.07, 1.07, 0.87, 0.89, 0.79, 0.96
        ] 

        for iou_type in self.iou_types:
            if iou_type == "keypoints":
                self.coco_eval[iou_type] = COCOeval_faster(
                    self.coco_gt,
                    iouType=iou_type,
                    print_function=print,
                    separate_eval=True,
                    extra_calc=True,
                    kpt_oks_sigmas=oks_sigmas,
                )
            else:
                self.coco_eval[iou_type] = COCOeval_faster(
                    self.coco_gt,
                    iouType=iou_type,
                    print_function=print,
                    separate_eval=True,
                    extra_calc=True,
                )

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            coco_eval = self.coco_eval[iou_type]
            
            # with redirect_stdout(io.StringIO()):
            print("evaluate 1")
            coco_dt = self.coco_gt.loadRes(results) if results else COCO()
            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            print("evaluate 2")
            coco_eval.evaluate()
            print("evaluate 3")

            self.eval_imgs[iou_type].append(
                np.array(coco_eval._evalImgs_cpp).reshape(
                    len(coco_eval.params.catIds),
                    len(coco_eval.params.areaRng),
                    len(coco_eval.params.imgIds),
                )
            )
            print("evaluate 4")

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print(f"IoU metric: {iou_type}")
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        if iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        if iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        raise ValueError(f"Unknown iou type {iou_type}")

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
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
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0] for mask in masks
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

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            # KeyPoints TV types are dim 2; coco is dim 3.
            # Add the third column and set to 2 (visible).
            keypoints = prediction["keypoints"]
            x = keypoints[..., 0]
            y = keypoints[..., 1]
            visible = ((x >= 0) & (y >= 0)).float().unsqueeze(-1)
            keypoints = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), visible], dim=-1)
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "keypoints": keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
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
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
