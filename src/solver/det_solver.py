"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import datetime
import json
import time

import torch

from ..misc import dist_utils, stats
from ._solver import BaseSolver
from .best_stat import BestStat
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
        best_stat = BestStat()

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
            best_stat.update(test_stats)

        start_time = time.time()
        start_epoch = self.last_epoch + 1
        
        for epoch in range(start_epoch, args.epochs):
            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
                if self.ema:
                    self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                    print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")

            freeze_layers = self.cfg.freeze_schedule.get(epoch)
            if freeze_layers:
                if freeze_layers < 0:
                    self.model.backbone.unfreeze_layers()
                    self.model.backbone.freeze_norm(False)
                else:
                    self.model.backbone.freeze_layers(freeze_layers)

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
                num_visualization_sample_batch=self.cfg.num_visualization_sample_batch,
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir and epoch < self.train_dataloader.collate_fn.stop_epoch:
                checkpoint_paths = [self.output_dir / "last.pth"]
                # extra checkpoint before LR drop and every 100 epochs
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
                num_visualization_sample_batch=self.cfg.num_visualization_sample_batch,
                prediction_render_freq=self.cfg.prediction_render_freq,
            )

            best_stat.update(epoch, test_stats)
            current_best_stat = best_stat.get_best()

            if dist_utils.is_main_process():
                print(f"Current Best Stat:\n  epoch: {current_best_stat['epoch']}")
                for k, v in current_best_stat.items():
                    if k != "epoch":
                        print(f"  {k}: {v}")

                # Save best checkpoint for stage 1 or 2
                stop_epoch = self.train_dataloader.collate_fn.stop_epoch
                if self.output_dir and best_stat.is_current_best(epoch):
                    ckpt_name = "best_stg1.pth" if epoch < stop_epoch else "best_stg2.pth"
                    dist_utils.save_on_master(self.state_dict(), self.output_dir / ckpt_name)

            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f"Test/{k}_{i}".format(k), v, epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                best_stat = BestStat()
                if self.ema:
                    self.ema.decay -= 0.0001
                    self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
                    print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

            if self.use_wandb:
                wandb_logs = {}
                for idx, metric_name in enumerate(metric_names):
                    wandb_logs[f"metrics/{metric_name}"] = test_stats["coco_eval_bbox"][idx]
                wandb_logs["epoch"] = epoch
                wandb.log(wandb_logs)

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs (bbox and keypoints)
                if coco_evaluator is not None:
                    (self.output_dir / "eval").mkdir(exist_ok=True)
                    for eval_type, eval_obj in coco_evaluator.coco_eval.items():
                        filenames = ["latest.pth"]
                        if epoch % 50 == 0:
                            filenames.append(f"{epoch:03}.pth")
                        for name in filenames:
                            torch.save(
                                eval_obj.eval,
                                self.output_dir / "eval" / f"{eval_type}_{name}",
                            )

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
                coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth"
            )

        return
