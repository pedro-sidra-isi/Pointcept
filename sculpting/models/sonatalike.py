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


@MODELS.register_module("SonataSculptor-v1m1")
class SonataLikeSculptor(Sonata):
    def __init__(
        self,
        backbone,
        head_in_channels,
        head_hidden_channels=4096,
        head_embed_channels=512,
        head_num_prototypes=4096,
        teacher_custom=None,
        num_global_view=2,
        num_local_view=4,
        mask_size_start=0.1,
        mask_size_base=0.4,
        mask_size_warmup_ratio=0.05,
        mask_ratio_start=0.3,
        mask_ratio_base=0.7,
        mask_ratio_warmup_ratio=0.05,
        mask_jitter=None,
        teacher_temp_start=0.04,
        teacher_temp_base=0.07,
        teacher_temp_warmup_ratio=0.05,
        student_temp=0.1,
        mask_loss_weight=2 / 12,
        roll_mask_loss_weight=2 / 12,
        unmask_loss_weight=4 / 12,
        momentum_base=0.996,
        momentum_final=1,
        match_max_k=8,
        match_max_r=0.08,
        up_cast_level=2,
        # Sculpting
        sculpt_loss_weight=4 / 12,
        reconstruct_loss_weight=4 / 12,
    ):
        super(Sonata, self).__init__()
        # Sonata stuff
        self.mask_loss_weight = mask_loss_weight
        self.roll_mask_loss_weight = roll_mask_loss_weight
        self.unmask_loss_weight = unmask_loss_weight

        self.num_global_view = num_global_view
        self.num_local_view = num_local_view

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

        # temperature and scheduler
        self.teacher_temp = teacher_temp_start
        self.teacher_temp_start = teacher_temp_start
        self.teacher_temp_base = teacher_temp_base
        self.teacher_temp_warmup_ratio = teacher_temp_warmup_ratio
        self.teacher_temp_scheduler = None
        self.student_temp = student_temp

        # momentum and scheduler
        self.momentum = momentum_base
        self.momentum_base = momentum_base
        self.momentum_final = momentum_final
        self.momentum_scheduler = None

        # dynamic matching
        self.match_max_k = match_max_k
        self.match_max_r = match_max_r

        # up cast level
        self.up_cast_level = up_cast_level

        # one of unmask, mask, roll mask loss enable
        assert unmask_loss_weight + mask_loss_weight + roll_mask_loss_weight > 0
        # roll mask loss need more than one global view
        assert num_global_view > 1 or roll_mask_loss_weight == 0
        # current roll mask only support two global views
        assert num_global_view == 1 or num_global_view == 2

        student_model_dict = dict()
        teacher_model_dict = dict()
        if teacher_custom is None:
            teacher_custom = {}
        student_backbone = build_model(backbone)
        # turn off parameters like drop path for teacher model
        backbone.update(teacher_custom)

        teacher_backbone = build_model(backbone)
        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone

        head = partial(
            OnlineCluster,
            in_channels=head_in_channels,
            hidden_channels=head_hidden_channels,
            embed_channels=head_embed_channels,
            num_prototypes=head_num_prototypes,
        )
        if self.mask_loss_weight > 0 or self.roll_mask_loss_weight > 0:
            student_model_dict["mask_head"] = head()
            teacher_model_dict["mask_head"] = head()
        if self.unmask_loss_weight > 0:
            student_model_dict["unmask_head"] = head()
            teacher_model_dict["unmask_head"] = head()

        # Sculpting additions
        features_to_reconstruct = backbone["in_channels"]
        self.sculpt_loss_weight = sculpt_loss_weight
        self.mask_token = nn.Parameter(torch.zeros(1, features_to_reconstruct ))
        trunc_normal_(self.mask_token, mean=0.0, std=0.02)

        student_model_dict["sculpt_head"] = nn.Sequential(
            nn.Linear(head_in_channels, head_hidden_channels),
            nn.GELU(),
            nn.Linear(head_hidden_channels, 2),
        )
        teacher_model_dict["sculpt_head"] = nn.Sequential(
            nn.Linear(head_in_channels, head_hidden_channels),
            nn.GELU(),
            nn.Linear(head_hidden_channels, 2),
        )
        student_model_dict["reconstruct_head"] = nn.Sequential(
            nn.Linear(head_in_channels, head_hidden_channels),
            nn.GELU(),
            nn.Linear(head_hidden_channels, features_to_reconstruct),
        )
        teacher_model_dict["reconstruct_head"] = nn.Sequential(
            nn.Linear(head_in_channels, head_hidden_channels),
            nn.GELU(),
            nn.Linear(head_hidden_channels, features_to_reconstruct),
        )

        # Sonata stuff
        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)
        for k, v in self.student.items():
            self.teacher[k].load_state_dict(self.student[k].state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False

    def before_step(self):
        super().before_step()
        self.trainer.comm_info["input_dict"]["mask_size"] = self.mask_size
        self.trainer.comm_info["input_dict"]["mask_ratio"] = self.mask_ratio

    def forward(self, data_dict, return_point=False):
        if return_point:
            point = self.teacher.backbone(data_dict)
            for _ in range(self.up_cast_level):
                assert "pooling_parent" in point.keys()
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            return dict(point=point)

        # prepare global_point, mask_global_point, local_point
        with torch.no_grad():
            # global_point & masking

            feat = data_dict["global_feat"]
            mask = data_dict["global_mask"]
            coord = data_dict["global_coord"]
            origin_coord = data_dict["global_origin_coord"]
            batch = offset2batch(data_dict["global_offset"])
            grid_size = data_dict["grid_size"][0]

            # i.e. not sculpting blocks
            clean_indexes = mask != 1

            global_point = Point(
                feat=feat[clean_indexes],
                coord=coord[clean_indexes],
                origin_coord=origin_coord[clean_indexes],
                offset=batch2offset(batch[clean_indexes]),
                grid_size=grid_size,
                mask=mask[clean_indexes],
            )

            gt_noblock = global_point.feat

            masked_feats = feat.clone()
            masked_feats[mask != 0] = self.mask_token  # zero-out when masked or cube

            mask_global_point = Point(
                feat=masked_feats,
                coord=coord,
                origin_coord=origin_coord,
                offset=data_dict["global_offset"],
                grid_size=grid_size,
                mask=mask,  # masked points
            )

            # local point & matching
            clean_indexes = data_dict["local_mask"] != 1  # not blocks
            local_point = Point(
                feat=data_dict["local_feat"][clean_indexes],
                mask=data_dict["local_mask"][clean_indexes],
                coord=data_dict["local_coord"][clean_indexes],
                origin_coord=data_dict["local_origin_coord"][clean_indexes],
                offset=batch2offset(
                    offset2batch(data_dict["local_offset"])[clean_indexes]
                ),
                grid_size=data_dict["grid_size"][0],
            )

            # create result dictionary for return
            result_dict = dict(loss=[])
            # teacher backbone forward (shared with mask and unmask)
            global_point_ = self.teacher.backbone(global_point)
            global_point_ = self.up_cast(global_point_)
            global_feat = global_point_.feat

        # Local
        if self.unmask_loss_weight > 0:
            # teacher head forward
            with torch.no_grad():
                global_point_.feat = self.teacher.unmask_head(global_feat)
            # student forward
            local_point_ = self.student.backbone(local_point)
            local_point_ = self.up_cast(local_point_)

            unmask_pred_sim = self.student.unmask_head(local_point_.feat)
            with torch.no_grad():
                principal_view_mask = global_point_.batch % self.num_global_view == 0
                principal_view_batch = (
                    global_point_.batch[principal_view_mask] // self.num_global_view
                )
                match_index = self.match_neighbour(
                    local_point_.origin_coord,
                    local_point_.offset[self.num_local_view - 1 :: self.num_local_view],
                    global_point_.origin_coord[principal_view_mask],
                    batch2offset(principal_view_batch),
                )
                # teacher forward
                unmask_target_sim = self.sinkhorn_knopp(
                    global_point_.feat[principal_view_mask][match_index[:, 1]],
                    self.teacher_temp,
                )
            # loss
            unmask_loss = -torch.sum(
                unmask_target_sim
                * F.log_softmax(
                    unmask_pred_sim[match_index[:, 0]] / self.student_temp, dim=-1
                ),
                dim=-1,
            )
            unmask_loss = torch_scatter.segment_coo(
                unmask_loss,
                index=local_point_.batch[match_index[:, 0]],
                reduce="mean",
            ).mean()
            result_dict["unmask_loss"] = unmask_loss
            result_dict["loss"].append(unmask_loss * self.unmask_loss_weight)

        # Global
        if (
            self.mask_loss_weight > 0
            or self.roll_mask_loss_weight > 0
            or self.sculpt_loss_weight > 0
        ):
            # teacher head forward
            with torch.no_grad():
                global_point_.feat = self.teacher.mask_head(global_feat)
            # student forward
            mask_global_point_ = self.student.backbone(mask_global_point)
            mask_global_point_ = self.up_cast(mask_global_point_)

            if self.sculpt_loss_weight > 0:
                # teacher head forward
                sculpt_pred = self.student.sculpt_head(mask_global_point_.feat)

                # predictions outside of sculpting blocks
                pred_noblock = sculpt_pred[mask_global_point.mask != 1]
                # predictions of sculpting blocks
                pred_block = sculpt_pred[mask_global_point.mask == 1]

                sculpt_loss = (
                    torch.sum((pred_noblock - gt_noblock) ** 2)
                    + torch.sum((pred_block - 0) ** 2)
                ) / (pred_noblock.shape[0] + pred_block.shape[0])

                result_dict["sculpt_loss"] = sculpt_loss
                result_dict["loss"].append(sculpt_loss * self.sculpt_loss_weight)

            if self.mask_loss_weight > 0 or self.roll_mask_loss_weight > 0:
                mask_pred_sim = self.student.mask_head(mask_global_point_.feat)

            if self.mask_loss_weight > 0:
                with torch.no_grad():
                    match_index = self.match_neighbour(
                        mask_global_point_.origin_coord,
                        mask_global_point_.offset,
                        global_point_.origin_coord,
                        global_point_.offset,
                    )
                    # teacher forward
                    mask_target_sim = self.sinkhorn_knopp(
                        global_point_.feat[match_index[:, 1]],
                        self.teacher_temp,
                    )

                # loss
                mask_loss = -torch.sum(
                    mask_target_sim
                    * F.log_softmax(
                        mask_pred_sim[match_index[:, 0]] / self.student_temp, dim=-1
                    ),
                    dim=-1,
                )
                mask_loss = torch_scatter.segment_coo(
                    mask_loss,
                    index=mask_global_point_.batch[match_index[:, 0]],
                    reduce="mean",
                ).mean()
                result_dict["mask_loss"] = mask_loss
                result_dict["loss"].append(mask_loss * self.mask_loss_weight)

            if self.roll_mask_loss_weight > 0:
                roll_global_point_ = self.roll_point(global_point_)
                with torch.no_grad():
                    # match index for pred and roll target
                    match_index = self.match_neighbour(
                        mask_global_point_.origin_coord,
                        mask_global_point_.offset,
                        roll_global_point_.origin_coord,
                        roll_global_point_.offset,
                    )
                    # teacher forward
                    roll_mask_target_sim = self.sinkhorn_knopp(
                        roll_global_point_.feat[match_index[:, 1]],
                        self.teacher_temp,
                    )

                roll_mask_loss = -torch.sum(
                    roll_mask_target_sim
                    * F.log_softmax(
                        mask_pred_sim[match_index[:, 0]] / self.student_temp, dim=-1
                    ),
                    dim=-1,
                )
                roll_mask_loss = torch_scatter.segment_coo(
                    roll_mask_loss,
                    index=mask_global_point_.batch[match_index[:, 0]],
                    reduce="mean",
                ).mean()
                result_dict["roll_mask_loss"] = roll_mask_loss
                result_dict["loss"].append(roll_mask_loss * self.roll_mask_loss_weight)

        result_dict["loss"] = sum(result_dict["loss"])

        if get_world_size() > 1:
            for loss in result_dict.values():
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        return result_dict
