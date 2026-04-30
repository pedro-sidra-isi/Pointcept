"""
Sculptor v1m1 Base

Author: Pedro Sidra (pedrosidra0@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch_scatter
from timm.layers import trunc_normal_

import pointops
from pointcept.models.utils.structure import Point
from pointcept.models.builder import MODELS, build_model
from pointcept.models.modules import PointModel
from pointcept.models.utils import offset2batch, offset2bincount, batch2offset
from pointcept.utils.comm import get_world_size, all_gather
from pointcept.utils.scheduler import CosineScheduler
from pointcept.models.default import DefaultSegmentorV2

from pointcept.datasets.sculpting_ops import get_cube


@MODELS.register_module("Sculptor-v1m1")
class Sculptor(DefaultSegmentorV2):
    def __init__(
        self,
        backbone_out_channels,
        num_classes=2,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
        cube_cell_size=0.02,
        mask_size_start=0.1,
        mask_size_base=0.4,
        mask_size_warmup_ratio=0.05,
        mask_ratio_start=0.3,
        mask_ratio_base=0.7,
        mask_ratio_warmup_ratio=0.05,
        mask_jitter=None,
        **kwargs
    ):
        super().__init__(
            num_classes,
            backbone_out_channels,
            backbone,
            criteria,
            freeze_backbone,
        )

        self.cube_cell_size = cube_cell_size

        # masking and scheduler
        self.mask_size = mask_size_start
        self.mask_size_start = mask_size_start
        self.mask_size_base = mask_size_base
        self.mask_size_warmup_ratio = mask_size_warmup_ratio
        self.mask_size_scheduler = None

        self.mask_ratio = mask_ratio_start
        self.mask_ratio_start = mask_ratio_start
        self.mask_ratio_base = mask_ratio_base
        self.mask_ratio_warmup_ratio = mask_ratio_warmup_ratio
        self.mask_ratio_scheduler = None

        self.mask_jitter = mask_jitter

        self.backbone = build_model(backbone)
        # turn off parameters like drop path for teacher model

    def before_train(self):
        # make ModelHook after CheckPointLoader
        total_steps = self.trainer.cfg.scheduler.total_steps
        curr_step = self.trainer.start_epoch * len(self.trainer.train_loader)
        # mask size scheduler
        self.mask_size_scheduler = CosineScheduler(
            start_value=self.mask_size_start,
            base_value=self.mask_size_base,
            final_value=self.mask_size_base,
            warmup_iters=int(total_steps * self.mask_size_warmup_ratio),
            total_iters=total_steps,
        )
        self.mask_size_scheduler.iter = curr_step

        # mask ratio scheduler
        self.mask_ratio_scheduler = CosineScheduler(
            start_value=self.mask_ratio_start,
            base_value=self.mask_ratio_base,
            final_value=self.mask_ratio_base,
            warmup_iters=int(total_steps * self.mask_ratio_warmup_ratio),
            total_iters=total_steps,
        )
        self.mask_ratio_scheduler.iter = curr_step

    def before_step(self):
        # update parameters from schedulers
        self.mask_size = self.mask_size_scheduler.step()
        self.mask_ratio = self.mask_ratio_scheduler.step()
        self.teacher_temp = self.teacher_temp_scheduler.step()
        self.momentum = self.momentum_scheduler.step()

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
            self.trainer.writer.add_scalar(
                "params/teacher_temp",
                self.teacher_temp,
                self.teacher_temp_scheduler.iter,
            )
            self.trainer.writer.add_scalar(
                "params/momentum",
                self.momentum,
                self.momentum_scheduler.iter,
            )

    def after_step(self):
        pass

    def generate_mask(self, coord, offset):
        batch = offset2batch(offset)
        mask_size = self.mask_size
        mask_ratio = self.mask_ratio

        # Grouping points with grid patch
        min_coord = torch_scatter.segment_coo(coord, batch, reduce="min")
        grid_coord = ((coord - min_coord[batch]) // mask_size).int()
        grid_coord = torch.cat([batch.unsqueeze(-1), grid_coord], dim=-1)
        unique, point_cluster, counts = torch.unique(
            grid_coord, dim=0, sorted=True, return_inverse=True, return_counts=True
        )
        patch_num = unique.shape[0]
        mask_patch_num = int(patch_num * mask_ratio)
        patch_index = torch.randperm(patch_num, device=coord.device)
        mask_patch_index = patch_index[:mask_patch_num]
        point_mask = torch.isin(point_cluster, mask_patch_index)
        return point_mask, point_cluster

    def add_sculpting_mask(self, input_dict):
        coord = input_dict["coord"]
        offset = input_dict["offset"]

        batch = offset2batch(offset)
        mask_size = self.mask_size
        mask_ratio = self.mask_ratio

        # Grouping points with grid patch
        min_coord = torch_scatter.segment_coo(coord, batch, reduce="min")
        grid_coord = ((coord - min_coord[batch]) // mask_size).int()
        grid_coord = torch.cat([batch.unsqueeze(-1), grid_coord], dim=-1)

        unique_cells = torch.unique(grid_coord, dim=0)

        ncells = unique_cells.shape[0]
        ncubes = int(ncells * mask_ratio)
        picked_cells = torch.randint(0, ncells, size=ncubes)

        p0s = (
            unique_cells[picked_cells, 1:] * mask_size
            + min_coord[unique_cells[picked_cells, 0]]
        )

        c = get_cube(0.2, 0.02, point_sampling="dense") + 0.2 / 2
        offsetted = p0s[np.newaxis, ...] + c[:, np.newaxis, :]
        offsetted = offsetted.reshape(-1, 3)

        out = torch.cat([coord, offsetted])
        print(offsetted.shape, coord.shape, out.shape)

        # p0s = (coord[torch.isin(point_cluster, mask_patch_index)] + min_coord)

        # point_mask =
        # return point_mask, point_cluster

    def forward(self, input_dict):

        input_dict = self.add_sculpting_mask(input_dict)
        return super().forward(input_dict)
