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
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from ..data import CocoEvaluator
from ..data.dataset import mscoco_category2label
from ..misc import MetricLogger, SmoothedValue, dist_utils, save_samples
from ..optim import ModelEMA, Warmup
from .validator import Validator, scale_boxes, scale_keypoints


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
            save_samples(samples, targets, output_dir, "train", normalized=True, box_fmt="cxcywh")

        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)

            if torch.isnan(outputs["pred_boxes"]).any() or torch.isinf(outputs["pred_boxes"]).any():
                print(outputs["pred_boxes"])
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
            save_samples(samples, targets, output_dir, "val", normalized=False, box_fmt="xyxy")

        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        outputs = model(samples)
        # with torch.autocast(device_type=str(device)):
        #     outputs = model(samples)

        # TODO (lyuwenyu), fix dataset converted using `convert_to_coco_api`?
        results = postprocessor(outputs)
        normalize_and_scale(samples, targets, results)
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
    metrics = Validator(gt, preds).compute_metrics(extended=True)
    print("Metrics:")
    for k, v in sorted(metrics.items()):
        print(f"  {k:30}: {v:.4f}" if isinstance(v, float) else f"  {k:30}: {v}")

    if use_wandb:
        metrics = {f"metrics/{k}": v for k, v in metrics.items()}
        metrics["epoch"] = epoch
        wandb.log(metrics)

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


def normalize_and_scale(samples, targets, results):
    for idx, (target, result) in enumerate(zip(targets, results)):
        # Scales
        w_target, h_target = target["orig_size"]
        h_sample, w_sample = samples[idx].shape[-2:]


        # Prediction Results
        # Denormalize predictions from [0,1] space relative to the model input dim.
        result["boxes"][:, [0, 2]] *= w_sample
        result["boxes"][:, [1, 3]] *= h_sample
        
        # Scale back to original image size. (e.g, model outputs in 640x640 but sample image is 640x480).
        result["boxes"] = scale_boxes(result["boxes"], w_target, h_target, w_sample, h_sample)
        
        if "keypoints" in result:
            # Same denorm, scale stuff.
            k = result["keypoints"]
            k[..., 0] *= w_sample
            k[..., 1] *= h_sample
            result["keypoints"] = scale_keypoints(k, w_target, h_target, w_sample, h_sample)


        # Targets
        # Targets are not normalized; just need scaling.
        target["boxes"] = scale_boxes(target["boxes"], w_target, h_target, w_sample, h_sample)

        if "keypoints" in target:
            target["keypoints"] = scale_keypoints(target["keypoints"], w_target, h_target, w_sample, h_sample)

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

        # --- DEBUG VISUALIZATION ---
        image_id = target["image_id"].item()

        if idx > 20:
            continue 
        
        from pathlib import Path
        from torchvision.utils import draw_keypoints, draw_bounding_boxes
        from torchvision.transforms.functional import to_pil_image, to_tensor
        from PIL import Image

        if dist_utils.is_main_process():
            try:
                pred_dir = Path("/workspace/training/output/predictions")
                pred_dir.mkdir(parents=True, exist_ok=True)

                im_pil = Image.open(target["image_path"]).convert("RGB")
                img = (to_tensor(im_pil) * 255).to(torch.uint8)

                from torchvision.ops import box_convert
                gt_boxes_xyxy = box_convert(gt_i["boxes"].cpu(), in_fmt="cxcywh", out_fmt="xyxy") if gt_i["boxes"].shape[-1] == 4 else gt_i["boxes"].cpu()
                pred_boxes_xyxy = box_convert(pred_i["boxes"].cpu(), in_fmt="cxcywh", out_fmt="xyxy") if pred_i["boxes"].shape[-1] == 4 else pred_i["boxes"].cpu()

                img_out = draw_bounding_boxes(img.clone(), gt_boxes_xyxy, colors="green")
                img_out = draw_bounding_boxes(img_out, pred_boxes_xyxy, colors="red")

                if "keypoints" in gt_i:
                    img_out = draw_keypoints(img_out, gt_i["keypoints"].cpu(), colors="green", radius=5)
                if "keypoints" in pred_i:
                    img_out = draw_keypoints(img_out, pred_i["keypoints"].cpu(), colors="red", radius=5)

                out_path = pred_dir / f"{image_id}_eval.png"
                to_pil_image(img_out).save(out_path)
            except Exception as e:
                print(f"Error while visualizing prediction for image {image_id}: {e}")
        # --- END DEBUG VISUALIZATION ---

    return gt, preds
