"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""

import math
import sys
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.amp
import torchvision.transforms.v2.functional as F

from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from ..data import CocoEvaluator
from ..data.dataset import mscoco_category2label
from ..misc import MetricLogger, SmoothedValue, dist_utils, save_samples
from ..optim import ModelEMA, Warmup
from .validator import Validator


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_wandb: bool,
    max_norm: float = 0,
    **kwargs,
):
    if use_wandb:
        import wandb

    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

    epochs = kwargs.get("epochs", None)
    header = "Epoch: [{}]".format(epoch) if epochs is None else "Epoch: [{}/{}]".format(epoch, epochs)

    print_freq = kwargs.get("print_freq", 10)
    writer: SummaryWriter = kwargs.get("writer", None)

    ema: ModelEMA = kwargs.get("ema", None)
    scaler: GradScaler = kwargs.get("scaler", None)
    lr_warmup_scheduler: Warmup = kwargs.get("lr_warmup_scheduler", None)
    losses = []

    output_dir = kwargs.get("output_dir", None)
    num_visualization_sample_batch = kwargs.get("num_visualization_sample_batch", 1)

    for i, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        if global_step < num_visualization_sample_batch and output_dir is not None and dist_utils.is_main_process():
            save_samples(samples, targets, output_dir, "train", normalized=True)

        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)

            if any(torch.isnan(b.data).any() or torch.isinf(b.data).any() for b in outputs["pred_boxes"]):
                state = model.state_dict()
                new_state = {}
                for key, value in model.state_dict().items():
                    # Replace 'module' with 'model' in each key
                    new_key = key.replace("module.", "")
                    # Add the updated key-value pair to the state dictionary
                    state[new_key] = value
                new_state["model"] = state
                dist_utils.save_on_master(new_state, "./NaN.pth")

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)

            loss: torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        # ema
        if ema is not None:
            ema.update(model)

        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())
        losses.append(loss_value.detach().cpu().numpy())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar("Loss/total", loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f"Lr/pg_{j}", pg["lr"], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f"Loss/{k}", v.item(), global_step)
    
    if use_wandb:
        wandb.log(
            {"lr": optimizer.param_groups[0]["lr"], "epoch": epoch, "train/loss": np.mean(losses)}
        )
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessor,
    data_loader,
    coco_evaluator: CocoEvaluator,
    device,
    epoch: int,
    use_wandb: bool,
    **kwargs,
):
    if use_wandb:
        import wandb

    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = "Test:"
    iou_types = coco_evaluator.iou_types

    gt: List[Dict[str, torch.Tensor]] = []
    preds: List[Dict[str, torch.Tensor]] = []

    output_dir = kwargs.get("output_dir", None)
    num_visualization_sample_batch = kwargs.get("num_visualization_sample_batch", 1)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        global_step = epoch * len(data_loader) + i

        if global_step < num_visualization_sample_batch and output_dir is not None and dist_utils.is_main_process():
            save_samples(samples, targets, output_dir, "val", normalized=False)

        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        outputs = model(samples)
        # with torch.autocast(device_type=str(device)):
        #     outputs = model(samples)

        # TODO (lyuwenyu), fix dataset converted using `convert_to_coco_api`?
        results = postprocessor(outputs)
        denormalize_results(samples, targets, results)

        if i < 20:
            render_results(samples, targets, results)
        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)
        
        if coco_evaluator is not None:
            coco_result = {target["image_id"].item(): output for target, output in zip(targets, results)}
            coco_evaluator.update(coco_result)

        gt_i, preds_i = format_metrics(targets, results, postprocessor.remap_mscoco_category)
        gt.extend(gt_i)
        preds.extend(preds_i)

    # Conf matrix, F1, Precision, Recall, box IoU
    # metrics = Validator(gt, preds).compute_metrics(extended=True)
    # print("Metrics:")
    # for k, v in sorted(metrics.items()):
    #     print(f"  {k:30}: {v:.4f}" if isinstance(v, float) else f"  {k:30}: {v}")

    # if use_wandb:
    #     metrics = {f"metrics/{k}": v for k, v in metrics.items()}
    #     metrics["epoch"] = epoch
    #     wandb.log(metrics)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    stats = {}
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

        if "bbox" in iou_types:
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in iou_types:
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()
        if "keypoints" in iou_types:
            stats["coco_eval_keypoints"] = coco_evaluator.coco_eval["keypoints"].stats.tolist()

    return stats, coco_evaluator


def denormalize_results(samples, targets, results):
    for result in results:
        size = F.get_image_size(samples)

        result["boxes"] = F.resize(result["boxes"], size)

        if "keypoints" in result:
            result["keypoints"] = F.resize(result["keypoints"], size)


def render_results(samples, targets, results):
    import torch
    from torchvision.utils import draw_bounding_boxes, draw_keypoints, save_image
    from pathlib import Path

    try:
        pred_dir = Path("/workspace/training/output/predictions")
        pred_dir.mkdir(parents=True, exist_ok=True)


        # Loop over each image in the batch
        for idx in range(samples.size(0)):
            img = samples[idx]
            # Convert to uint8 image tensor for drawing
            if img.dtype != torch.uint8:
                img_disp = (img * 255).clamp(0, 255).to(torch.uint8)
            else:
                img_disp = img

            # Draw ground-truth boxes and keypoints
            gt = targets[idx]
            gt_boxes = F.convert_bounding_box_format(gt["boxes"], new_format="XYXY")
            gt_labels = [str(int(l.item())) for l in gt["labels"]]
            img_overlay = draw_bounding_boxes(img_disp, gt_boxes, labels=gt_labels, colors="#00FF00", width=2)

            # Draw predicted boxes, scores, and keypoints
            pred = results[idx]
            pred_boxes = F.convert_bounding_box_format(pred["boxes"], new_format="XYXY")
            pred_scores = pred.get("scores", torch.zeros(pred_boxes.size(0)))
            pred_labels = [
                f"{int(l.item())}:{s:.2f}" for l, s in zip(pred["labels"].flatten(), pred_scores.flatten())
            ]
            img_overlay = draw_bounding_boxes(img_overlay, pred_boxes, labels=pred_labels, colors="#00FFFF", width=2)
            # Now draw ground-truth keypoints on top of boxes
            if "keypoints" in gt:
                img_overlay = draw_keypoints(img_overlay, gt["keypoints"], colors="#00FF00", radius=3)
            # Then draw predicted keypoints in bright yellow
            if "keypoints" in pred:
                img_overlay = draw_keypoints(img_overlay, pred["keypoints"], colors="#FFFF00", radius=3)

            # Save annotated image to disk
            save_image(img_overlay.float() / 255.0, pred_dir / f"render_{idx}.png")
    except Exception as e:
        print(f"render_results error: {e}")

def format_metrics(targets, results, remap_mscoco_category):
    gt: List[Dict[str, torch.Tensor]] = []
    preds: List[Dict[str, torch.Tensor]] = []

    for idx, (target, result) in enumerate(zip(targets, results)):
        # Targets
        gt_i = {
            "boxes": target["boxes"],
            "labels": target["labels"],
        }

        if "keypoints" in target:
            gt_i["keypoints"] = target["keypoints"]

        gt.append(gt_i)

        # Predictions
        pred_labels = (
            torch.tensor([mscoco_category2label[int(x.item())] for x in result["labels"].flatten()])
            .to(result["labels"].device)
            .reshape(result["labels"].shape)
        ) if remap_mscoco_category else result["labels"]

        pred_i = {
            "boxes": result["boxes"], 
            "labels": pred_labels, 
            "scores": result["scores"]
        }

        if "keypoints" in result:
            pred_i["keypoints"] = result["keypoints"]

        preds.append(pred_i)

    return gt, preds
