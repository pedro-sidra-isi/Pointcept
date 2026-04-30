"""
Sonata v1m1 Base

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from itertools import chain
from packaging import version
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch_scatter
from timm.layers import trunc_normal_

import pointops
from pointcept.models.sonata.sonata_v1m1_base import Sonata, OnlineCluster
from pointcept.models.utils.structure import Point
from pointcept.models.builder import MODELS, build_model
from pointcept.models.modules import PointModel
from pointcept.models.utils import offset2batch, offset2bincount, batch2offset
from pointcept.utils.comm import get_world_size, all_gather
from pointcept.utils.scheduler import CosineScheduler
from pointcept.datasets.transform import TRANSFORMS

from timm.layers import trunc_normal_

from pointcept.models.modules import PointModel


@MODELS.register_module("Sculptor-v1m1")
class Sculptor(PointModel):
    def __init__(
        self,
        backbone,
        head_in_channels,
        head_hidden_channels=256,
        mask_size_start=0.1,
        mask_size_base=0.4,
        mask_size_warmup_ratio=0.05,
        mask_ratio_start=0.3,
        mask_ratio_base=0.7,
        mask_ratio_warmup_ratio=0.05,
        # Sculpting
        sculpt_loss_weight=1 / 2,
        reconstruct_loss_weight=1 / 2,
        sculpt_original_point_weight=1,
        sculpt_block_point_weight=1,
        sculpt_mask_point_weight=1,
    ):
        super(Sculptor, self).__init__()

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

        self.backbone = build_model(backbone)

        # Sculpting additions
        self.features_to_reconstruct = backbone["in_channels"]
        self.sculpt_loss_weight = sculpt_loss_weight
        self.reconstruct_loss_weight = reconstruct_loss_weight
        self.mask_token = nn.Parameter(torch.zeros(1, self.features_to_reconstruct))
        trunc_normal_(self.mask_token, mean=0.0, std=0.02)

        self.sculpt_head = nn.Sequential(
            nn.Linear(head_in_channels, head_hidden_channels),
            nn.GELU(),
            nn.Linear(head_hidden_channels, 2),
        )

        self.reconstruct_head = nn.Sequential(
            nn.Linear(head_in_channels, head_hidden_channels),
            nn.GELU(),
            nn.Linear(head_hidden_channels, self.features_to_reconstruct),
        )

        self.sculpt_weights = [
            sculpt_original_point_weight,
            sculpt_block_point_weight,
            sculpt_mask_point_weight,
        ]

    def before_step(self):
        super().before_step()
        self.trainer.comm_info["input_dict"]["mask_size"] = self.mask_size
        self.trainer.comm_info["input_dict"]["mask_ratio"] = self.mask_ratio

        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar(
                "params/mask_size", self.mask_size, current_epoch
            )
            self.trainer.writer.add_scalar(
                "params/mask_ratio", self.mask_ratio, current_epoch
            )

    def forward(self, data_dict):

        # prepare global_point, mask_global_point, local_point
        with torch.no_grad():

            # global_point & masking
            feat = data_dict["feat"]
            mask = data_dict["mask"]
            coord = data_dict["coord"]
            batch = offset2batch(data_dict["offset"])
            grid_size = data_dict["grid_size"][0]

            masked_feats = feat.clone()
            masked_feats[mask != 0] = self.mask_token  # zero-out when masked or cube

            mask_global_point = Point(
                feat=masked_feats,
                coord=coord,
                offset=data_dict["offset"],
                grid_size=grid_size,
                mask=mask,  # masked points
            )

            # create result dictionary for return
            result_dict = dict(loss=[])

        mask_global_point_ = self.backbone(mask_global_point)

        sculpt_pred = self.sculpt_head(mask_global_point_.feat)
        reconstruct_pred = self.reconstruct_head(mask_global_point_.feat)

        # predictions outside of sculpting blocks
        pred_original_points = sculpt_pred[mask_global_point.mask == 0]
        pred_block_points = sculpt_pred[mask_global_point.mask == 1]
        pred_masked_points = sculpt_pred[mask_global_point.mask == 2]

        lossfunc = nn.CrossEntropyLoss()

        result_dict["sculpt_loss"] = (
            self.sculpt_weights[0]
            * lossfunc(pred_original_points, torch.ones_like(pred_original_points))
            + self.sculpt_weights[1]
            * lossfunc(pred_block_points, torch.ones_like(pred_block_points))
            + self.sculpt_weights[2]
            * lossfunc(pred_masked_points, torch.ones_like(pred_masked_points))
        )
        result_dict["loss"].append(result_dict["sculpt_loss"] * self.sculpt_loss_weight)

        # result_dict["reconstruct_loss"] = (
        #     torch.sum((reconstruct_pred[masked_points] - feat[masked_points]) ** 2)
        #     + torch.sum((reconstruct_pred[block_points] - self.mask_token) ** 2)
        # ) / (masked_points.nonzero() + block_points.nonzero())
        # result_dict["loss"].append(
        #     result_dict["reconstruct_loss"] * self.reconstruct_loss_weight
        # )

        result_dict["loss"] = sum(result_dict["loss"])

        if get_world_size() > 1:
            for loss in result_dict.values():
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)

        return result_dict
