"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import datetime
import json
import sys
import time
import torch

from ..misc import dist_utils, stats
from ._solver import BaseSolver
from .best_stat import BestStatTracker
from .det_engine import evaluate, train_one_epoch


class DetSolver(BaseSolver):
    def fit(self):
        self.train()
        args = self.cfg
        metric_names = ["AP50:95", "AP50", "AP75", "APsmall", "APmedium", "APlarge"]

        if self.use_wandb:
            import wandb

            wandb.init(
                project=args.yaml_cfg["project_name"],
                name=args.yaml_cfg["exp_name"],
                config=args.yaml_cfg,
            )
            wandb.watch(self.model)

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-" * 42 + "Start training" + "-" * 43)
        best_stat_tracker = BestStatTracker(mode="pareto")

        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                self.last_epoch,
                self.use_wandb
            )

        best_stat_tracker = BestStatTracker(mode="pareto")  # Or "single"
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epochs):
            self.train_dataloader.set_epoch(epoch)

            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                # Stage transition: load best stage 1 model, adjust EMA, reset best_stat
                self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
                if self.ema:
                    self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                    print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")

            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                epochs=args.epochs,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                use_wandb=self.use_wandb,
                output_dir=self.output_dir,
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir and epoch < self.train_dataloader.collate_fn.stop_epoch:
                checkpoint_paths = [self.output_dir / "last.pth"]
                # Extra checkpoint before LR drop and every checkpoint_freq epochs.
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f"checkpoint{epoch:04}.pth")
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                epoch,
                self.use_wandb,
                output_dir=self.output_dir,
            )

            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f"Test/{k}_{i}".format(k), v, epoch)



            # Update best stats
            current_stats = {
                "epoch": epoch,
                "coco_eval_bbox": test_stats["coco_eval_bbox"][0],
                "coco_eval_keypoints": test_stats["coco_eval_keypoints"][0] if "coco_eval_keypoints" in test_stats else 0,
            }

            best_stat_tracker.update(current_stats)

            if best_stat_tracker.is_current_best(epoch):
                best_stat = best_stat_tracker.get_best_model_stats()
                print(f"New best model (epoch {best_stat.get('epoch', -1)}): {best_stat}")
                if self.output_dir:
                    if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / "best_stg2.pth")
                    else:
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / "best_stg1.pth")
            else:
                best_stat = best_stat_tracker.get_best_model_stats()
                best_epoch = best_stat.get('epoch', -1)
                print(f"Epoch {epoch} did not become the best. Best epoch: {best_epoch}")
                for k in test_stats:
                    current_value = test_stats[k][0]
                    best_value = best_stat.get(k, None)
                    print(f"  {k}: current = {current_value:.4f}, best = {best_value if best_value is None else f'{best_value:.4f}'}")

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

            if "coco_eval_keypoints" in test_stats:
                for idx, metric_name in enumerate(metric_names): # Add keypoint metrics to log_stats
                    log_stats[f"test_{metric_name}kp"] = test_stats["coco_eval_keypoints"][idx]

            if self.use_wandb:
                wandb_logs = {}
                for idx, metric_name in enumerate(metric_names):
                    wandb_logs[f"metrics/bbox_{metric_name}"] = test_stats["coco_eval_bbox"][idx]
                
                if "coco_eval_keypoints" in test_stats:
                    for idx, metric_name in enumerate(metric_names):
                        wandb_logs[f"metrics/keypoints_{metric_name}kp"] = test_stats["coco_eval_keypoints"][idx]
                
                wandb_logs["epoch"] = epoch
                wandb.log(wandb_logs)

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / "eval").mkdir(exist_ok=True)
                    filenames = ["latest.pth"]
                    if epoch % 50 == 0:
                        filenames.append(f"{epoch:03}.pth")
                    for iou_type, coco_eval in coco_evaluator.coco_eval.items():
                        for name in filenames:
                            torch.save(
                                coco_eval.eval,
                                self.output_dir / "eval" / f"{iou_type}_{name}",
                            )
            sys.stdout.flush()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    def val(self):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            self.evaluator,
            self.device,
            epoch=-1,
            use_wandb=False,
        )

        if self.output_dir:
            dist_utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval_bbox.pth"
            )
            if "keypoints" in coco_evaluator.coco_eval:
                dist_utils.save_on_master(
                    coco_evaluator.coco_eval["keypoints"].eval, self.output_dir / "eval_keypoints.pth"
                )

        return