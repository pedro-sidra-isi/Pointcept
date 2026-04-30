import wandb
import os
import pandas as pd
from pathlib import Path
import torch
from collections import OrderedDict
import pointcept.utils.comm as comm
from pointcept.engines.hooks import CheckpointLoader, CheckpointSaver, is_main_process, HOOKS, HookBase
from pointcept.utils.scheduler import CosineScheduler

@HOOKS.register_module()
class CheckpointLoaderAllowMismatch(CheckpointLoader):
    def __init__(self, keywords="", replacement=None, strict=False):
        self.keywords = keywords
        self.replacement = replacement if replacement is not None else keywords
        self.strict = strict

    def before_train(self):
        self.trainer.logger.info("=> Loading checkpoint & weight ...")
        if self.trainer.cfg.weight and os.path.isfile(self.trainer.cfg.weight):
            self.trainer.logger.info(f"Loading weight at: {self.trainer.cfg.weight}")
            checkpoint = torch.load(
                self.trainer.cfg.weight,
                map_location=lambda storage, loc: storage.cuda(),
                weights_only=False,
            )
            self.trainer.logger.info(
                f"Loading layer weights with keyword: {self.keywords}, "
                f"replace keyword with: {self.replacement}"
            )
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if not key.startswith("module."):
                    key = "module." + key  # xxx.xxx -> module.xxx.xxx
                # Now all keys contain "module." no matter DDP or not.
                if self.keywords in key:
                    key = key.replace(self.keywords, self.replacement, 1)
                if comm.get_world_size() == 1:
                    key = key[7:]  # module.xxx.xxx -> xxx.xxx
                weight[key] = value

            current_model_dict = self.trainer.model.state_dict()
            # allows size mismatch
            removed_keys = {
                k
                for k, v in weight.items()
                if k in current_model_dict and v.size() != current_model_dict[k].size()
            }
            for key in removed_keys:
                weight.pop(key)
            self.trainer.logger.info(f"Removed keys due to size mismatch: {removed_keys}")


            load_state_info = self.trainer.model.load_state_dict(
                weight, strict=self.strict
            )
            self.trainer.logger.info(f"Missing keys: {load_state_info[0]}")
            if self.trainer.cfg.resume:
                self.trainer.logger.info(
                    f"Resuming train at eval epoch: {checkpoint['epoch']}"
                )
                self.trainer.start_epoch = checkpoint["epoch"]
                self.trainer.best_metric_value = checkpoint["best_metric_value"]
                self.trainer.optimizer.load_state_dict(checkpoint["optimizer"])
                self.trainer.scheduler.load_state_dict(checkpoint["scheduler"])
                if self.trainer.cfg.enable_amp:
                    self.trainer.scaler.load_state_dict(checkpoint["scaler"])
        else:
            self.trainer.logger.info(f"No weight found at: {self.trainer.cfg.weight}")

@HOOKS.register_module()
class CheckpointSaverWandb(CheckpointSaver):
    def __init__(self, save_freq=None):
        self.save_freq = save_freq  # None or int, None indicate only save model last

        # best model stat-time, dummy value to start with
        self.last_model_best_stat = os.stat(__file__)

    def delete_oldest_artifact_version(self, name_part):
        logged_artifacts = wandb.Api().run(wandb.run._get_path()).logged_artifacts()

        interest_artifacts = [a for a in logged_artifacts if name_part in a.name]
        dates = pd.to_datetime([a.created_at for a in interest_artifacts])

        # Don't delete my only artifact!
        if len(dates) > 1:
            idx = dates.argmin()
            interest_artifacts[idx].delete()

    def after_epoch(self):
        super().after_epoch()
        if is_main_process() and (self.save_freq and (self.trainer.epoch + 1) % self.save_freq == 0):
            # our super() saved it here
            filename = os.path.join(
                self.trainer.cfg.save_path, "model", "model_last.pth"
            )

            # Log to wandb and remove old versions for storage
            wandb.log_model(filename, name=f"{wandb.run.id}-model_last")
            self.delete_oldest_artifact_version(name_part="model_last")

            # Check if a new best model exists
            model_best = os.path.join(
                self.trainer.cfg.save_path, "model", "model_best.pth"
            )

            # use os.stat to check modified time because i don't wanna actually check the file contents
            if Path(model_best).is_file():
                if os.stat(model_best).st_atime != self.last_model_best_stat.st_atime:
                    wandb.log_model(model_best, name=f"{wandb.run.id}-model_best")
                    self.delete_oldest_artifact_version(name_part="model_best")

                self.last_model_best_stat = os.stat(model_best)


@HOOKS.register_module()
class MaskSizeScheduler(HookBase):
    def __init__(
        self,
        mask_size_start=0.1,
        mask_size_end=0.1,
        mask_size_base=0.4,
        mask_size_warmup_ratio=0.05,
        mask_ratio_start=0.3,
        mask_ratio_end=0.1,
        mask_ratio_base=0.7,
        mask_ratio_warmup_ratio=0.05,
    ):
        # masking and scheduler
        self.mask_size = mask_size_start
        self.mask_size_start = mask_size_start
        self.mask_size_end = mask_size_end
        self.mask_size_base = mask_size_base
        self.mask_size_warmup_ratio = mask_size_warmup_ratio
        self.mask_size_scheduler = None

        self.mask_ratio = mask_ratio_start
        self.mask_ratio_start = mask_ratio_start
        self.mask_ratio_end = mask_ratio_end
        self.mask_ratio_base = mask_ratio_base
        self.mask_ratio_warmup_ratio = mask_ratio_warmup_ratio
        self.mask_ratio_scheduler = None

    def before_train(self):
        # make ModelHook after CheckPointLoader
        total_steps = self.trainer.cfg.scheduler.total_steps
        curr_step = self.trainer.start_epoch * len(self.trainer.train_loader)
        # mask size scheduler
        self.mask_size_scheduler = CosineScheduler(
            start_value=self.mask_size_start,
            base_value=self.mask_size_base,
            final_value=self.mask_size_end,
            warmup_iters=int(total_steps * self.mask_size_warmup_ratio),
            total_iters=total_steps,
        )
        self.mask_size_scheduler.iter = curr_step

        # mask ratio scheduler
        self.mask_ratio_scheduler = CosineScheduler(
            start_value=self.mask_ratio_start,
            base_value=self.mask_ratio_base,
            final_value=self.mask_ratio_end,
            warmup_iters=int(total_steps * self.mask_ratio_warmup_ratio),
            total_iters=total_steps,
        )
        self.mask_ratio_scheduler.iter = curr_step

    def before_step(self):
        # update parameters from schedulers
        self.mask_size = self.mask_size_scheduler.step()
        self.mask_ratio = self.mask_ratio_scheduler.step()

        self.trainer.comm_info["input_dict"]["mask_size"] = self.mask_size
        self.trainer.comm_info["input_dict"]["mask_ratio"] = self.mask_ratio


        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar(
                "params/mask_size",
                self.mask_size,
                self.mask_size_scheduler.iter,
            )
            self.trainer.writer.add_scalar(
                "params/mask_ratio",
                self.mask_ratio,
                self.mask_ratio_scheduler.iter,
            )
