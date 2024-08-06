"""
Masked Scene Contrast
https://arxiv.org/abs/2303.14191

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import random
from itertools import chain
import torch
import torch.nn as nn
import torch.distributed as dist
from torch_geometric.nn.pool import voxel_grid

from timm.models.layers import trunc_normal_
import pointops

from pointcept.models.builder import MODELS, build_model
from pointcept.models.utils import offset2batch
from pointcept.utils.comm import get_world_size


@MODELS.register_module("MSC-v1m1")
class MaskedSceneContrast(nn.Module):
    def __init__(
        self,
        backbone,
        backbone_in_channels,
        backbone_out_channels,
        mask_grid_size=0.1,
        mask_rate=0.4,
        view1_mix_prob=0,
        view2_mix_prob=0,
        matching_max_k=8,
        matching_max_radius=0.03,
        matching_max_pair=8192,
        nce_t=0.4,
        contrast_weight=1,
        reconstruct_weight=1,
        reconstruct_color=True,
        reconstruct_normal=True,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.mask_grid_size = mask_grid_size
        self.mask_rate = mask_rate
        self.view1_mix_prob = view1_mix_prob
        self.view2_mix_prob = view2_mix_prob
        self.matching_max_k = matching_max_k
        self.matching_max_radius = matching_max_radius
        self.matching_max_pair = matching_max_pair
        self.nce_t = nce_t
        self.contrast_weight = contrast_weight
        self.reconstruct_weight = reconstruct_weight
        self.reconstruct_color = reconstruct_color
        self.reconstruct_normal = reconstruct_normal

        # trainable token 
        self.mask_token = nn.Parameter(torch.zeros(1, backbone_in_channels))
        trunc_normal_(self.mask_token, mean=0.0, std=0.02)

        self.color_head = (
            nn.Linear(backbone_out_channels, 3) if reconstruct_color else None
        )
        self.normal_head = (
            nn.Linear(backbone_out_channels, 3) if reconstruct_normal else None
        )

        self.nce_criteria = torch.nn.CrossEntropyLoss(reduction="mean")

    @torch.no_grad()
    def generate_cross_masks(
        self, view1_original_coord, view1_offset, view2_original_coord, view2_offset
    ):
        # split tensors by batch
        view1_batch = offset2batch(view1_offset)
        view2_batch = offset2batch(view2_offset)
        view1_batch_count = view1_batch.bincount()
        view2_batch_count = view2_batch.bincount()
        view1_original_coord_split = view1_original_coord.split(list(view1_batch_count))
        view2_original_coord_split = view2_original_coord.split(list(view2_batch_count))

        # Concatenate coordinates in sequence:
        # view1_b1
        # view2_b1
        # view1_b2
        # view2_b2
        # where b1, b2, etc. are the scenes from this batch, up to batch_size
        union_original_coord = torch.cat(
            list(
                chain.from_iterable(
                    zip(view1_original_coord_split, view2_original_coord_split)
                )
            )
        )
        union_offset = torch.cat(
            [view1_offset.unsqueeze(-1), view2_offset.unsqueeze(-1)], dim=-1
        ).sum(-1)
        union_batch = offset2batch(union_offset)

        # get grid indexes for a grid with cells of size `self.mask_grid_size`
        mask_patch_coord = union_original_coord.div(self.mask_grid_size)
        mask_patch_grid_coord = torch.floor(mask_patch_coord)

        # voxelize constrastive view coordinates
        # mask_patch_cluster = for each point, the ID of which patch that point belongs to
        mask_patch_cluster = voxel_grid(
            pos=mask_patch_grid_coord, size=1, batch=union_batch, start=0
        )
        # Get unique patch coordinates with inverse mapping
        # unique: each unique patch ID
        # cluster: mapping of points to patch IDs
        # counts: count of voxels that fall into each patch ID
        unique, cluster, counts = torch.unique(
            mask_patch_cluster, sorted=True, return_inverse=True, return_counts=True
        )

        # Number of voxels
        patch_num = unique.shape[0]
        # Max point count of a patch
        patch_max_point = counts.max().item()
        # mapping : f(voxel_ID) -> list of point indexes
        patch2point_map = cluster.new_zeros(patch_num, patch_max_point)
        # mask of indexes in patch2point_map that are 'valid'
        patch2point_mask = torch.lt(
            torch.arange(patch_max_point).cuda().unsqueeze(0), counts.unsqueeze(-1)
        )
        # sorted list of patch IDs will have the indices of each cluster in order
        sorted_cluster_value, sorted_cluster_indices = torch.sort(cluster)
        # populate patch2point_map. Now we have a mapping from voxel_ID -> original indexes
        patch2point_map[patch2point_mask] = sorted_cluster_indices

        # generate cross masks
        assert self.mask_rate <= 0.5

        # mark each patch with a tag: 1 for view1, 2 for view2
        patch_mask = torch.zeros(patch_num, device=union_original_coord.device).int()
        
        # random list of patch IDs
        rand_perm = torch.randperm(patch_num)
        # fraction of patches chosen to be masked-out
        mask_patch_num = int(patch_num * self.mask_rate)

        # mask1 tag with 1, mask2 tag with 2
        # `mask_patch_num` patches used for view1
        patch_mask[rand_perm[0:mask_patch_num]] = 1
        # `mask_patch_num` patches used for view2
        patch_mask[rand_perm[mask_patch_num : mask_patch_num * 2]] = 2
        # (note: there may be some 'unpatched' voxels that both views can see since mask_rate<=0.5)

        # Translate from patch mask to point mask
        point_mask = torch.zeros(
            union_original_coord.shape[0], device=union_original_coord.device
        ).int()

        # points with 1 are visible only on view1, 2 only on view2
        # (there are some zeros left!)
        point_mask[
            patch2point_map[patch_mask == 1][patch2point_mask[patch_mask == 1]]
        ] = 1
        point_mask[
            patch2point_map[patch_mask == 2][patch2point_mask[patch_mask == 2]]
        ] = 2

        # separate mask to view1 and view2
        point_mask_split = point_mask.split(
            list(
                torch.cat(
                    [view1_batch_count.unsqueeze(-1), view2_batch_count.unsqueeze(-1)],
                    dim=-1,
                ).flatten()
            )
        )
        # point_mask_split is in the same order as union_original_coord
        # each mask has values of 0, 1 or 2. view1 only sees 1s, view2 only sees 2s
        view1_point_mask = torch.cat(point_mask_split[0::2]) == 1
        view2_point_mask = torch.cat(point_mask_split[1::2]) == 2
        return view1_point_mask, view2_point_mask

    @torch.no_grad()
    def match_contrastive_pair(
        self, view1_coord, view1_offset, view2_coord, view2_offset, max_k, max_radius
    ):

        # K-nearest neighbors to each point
        index, distance = pointops.knn_query(
            max_k,
            view2_coord.float(),
            view2_offset.int(),
            view1_coord.float(),
            view1_offset.int(),
        )

        # Yikes...
        # I - think - this flattens `index` out into a list of neighbors
        # like: point 0 from view1 is neighbors with points a b c of view2, etc.
        index = torch.cat(
            [
                torch.arange(index.shape[0], device=index.device, dtype=torch.long)
                .view(-1, 1, 1)
                .expand(-1, max_k, 1),
                index.view(-1, max_k, 1),
            ],
            dim=-1,
        )[distance.squeeze(-1) < max_radius]

        # number of neighbors within `max_radius` distance for each point
        unique, count = index[:, 0].unique(return_counts=True)
        # Select one random neighbor from each 'neighborhood'

        select = (
            torch.cumsum(count, dim=0) # index of first point within each neighborhood
             - torch.randint(count.max(), count.shape, device=count.device) % count # random offset into the 'neighborhood' indexes
            - 1 # starts at 0
        )
        # Select random neighbors
        index = index[select]

        # enforme maximum number of matching pairs
        if index.shape[0] > self.matching_max_pair:
            index = index[torch.randperm(index.shape[0])[: self.matching_max_pair]]
        return index

    def compute_contrastive_loss(
        self, view1_feat, view1_offset, view2_feat, view2_offset, match_index
    ):
        assert view1_offset.shape == view2_offset.shape

        # features from the positive matching pairs
        # (points in 1 & their neighbors in 2)
        view1_feat = view1_feat[match_index[:, 0]]
        view2_feat = view2_feat[match_index[:, 1]]

        # Normalize features to [0,1]
        view1_feat = view1_feat / ( torch.norm(view1_feat, p=2, dim=1, keepdim=True) + 1e-7)
        view2_feat = view2_feat / ( torch.norm(view2_feat, p=2, dim=1, keepdim=True) + 1e-7)

        # dot between features => similarity between features i and features j
        # (row-> view1, col->view2, i.e. diagonal is positive match, off-diagonal is negative)
        sim = torch.mm(view1_feat, view2_feat.transpose(1, 0))

        with torch.no_grad():
            # Positive matches along the diagonal
            pos_sim = torch.diagonal(sim).mean()
            # negative matches are off the diagonal
            neg_sim = sim.mean(dim=-1).mean() - pos_sim / match_index.shape[0]

        # 0, 1, 2... num_matches
        # each row should maximize the element along the diagonal and minimize the rest
        # labels are one-hot encoded, so if you have 10 rows, labels would be arange(10)
        labels = torch.arange(sim.shape[0], device=view1_feat.device).long()

        # nce - normalized cross-entropy
        # nce_t = temperature parameter
        loss = self.nce_criteria(torch.div(sim, self.nce_t), labels)

        if get_world_size() > 1:
            dist.all_reduce(loss)
            dist.all_reduce(pos_sim)
            dist.all_reduce(neg_sim)
        return (
            loss / get_world_size(),
            pos_sim / get_world_size(),
            neg_sim / get_world_size(),
        )

    def forward(self, data_dict):
        # original_coord = original coordinate before augmentation
        view1_original_coord = data_dict["view1_original_coord"]
        view1_coord = data_dict["view1_coord"]
        view1_feat = data_dict["view1_feat"]
        view1_offset = data_dict["view1_offset"].int()

        # original_coord = original coordinate before augmentation
        view2_original_coord = data_dict["view2_original_coord"]
        view2_coord = data_dict["view2_coord"]
        view2_feat = data_dict["view2_feat"]
        view2_offset = data_dict["view2_offset"].int()

        # `point_mask` is True for points that are masked-out for each view
        # set of indexes with True values is non-intersecting between mask1 and mask2.
        # But they share indexes with False values
        # (True on mask1 => False on mask2) but not necessarily vice-versa
        # (True on mask2 => False on mask1) but not necessarily vice-versa
        # (False on mask1 => can be true or false on mask2)
        # (False on mask2 => can be true or false on mask1)
        view1_point_mask, view2_point_mask = self.generate_cross_masks(
            view1_original_coord, view1_offset, view2_original_coord, view2_offset
        )

        # Basically just take `view1_feat` and substitute the features with `mask_token`
        # wherever `view1_point_mask`=True
        view1_mask_tokens = self.mask_token.expand(view1_coord.shape[0], -1)
        
        view1_weight = view1_point_mask.unsqueeze(-1).type_as(view1_mask_tokens)
        view1_feat = view1_feat * (1 - view1_weight) + view1_mask_tokens * view1_weight

        # same as above for view2
        view2_mask_tokens = self.mask_token.expand(view2_coord.shape[0], -1)
        view2_weight = view2_point_mask.unsqueeze(-1).type_as(view2_mask_tokens)
        view2_feat = view2_feat * (1 - view2_weight) + view2_mask_tokens * view2_weight

        view1_data_dict = dict(
            original_coord=view1_original_coord,
            coord=view1_coord,
            feat=view1_feat,
            offset=view1_offset,
        )
        view2_data_dict = dict(
            original_coord=view2_original_coord,
            coord=view2_coord,
            feat=view2_feat,
            offset=view2_offset,
        )

        # SparseConv based method need grid coord
        if "view1_grid_coord" in data_dict.keys():
            view1_data_dict["grid_coord"] = data_dict["view1_grid_coord"]
        if "view2_grid_coord" in data_dict.keys():
            view2_data_dict["grid_coord"] = data_dict["view2_grid_coord"]

        # view mixing strategy (mix3d)
        # use half of the normal batch size, mix every two scenes together
        if random.random() < self.view1_mix_prob:
            view1_data_dict["offset"] = torch.cat(
                [view1_offset[1:-1:2], view1_offset[-1].unsqueeze(0)], dim=0
            )
        # use half of the normal batch size, mix every two scenes together
        if random.random() < self.view2_mix_prob:
            view2_data_dict["offset"] = torch.cat(
                [view2_offset[1:-1:2], view2_offset[-1].unsqueeze(0)], dim=0
            )

        # Forward pass for both views (siamese net)
        view1_feat = self.backbone(view1_data_dict)
        view2_feat = self.backbone(view2_data_dict)

        # Get positive match indices from KNN query neighborhood
        match_index = self.match_contrastive_pair(
            view1_original_coord,
            view1_offset,
            view2_original_coord,
            view2_offset,
            max_k=self.matching_max_k,
            max_radius=self.matching_max_radius,
        )
        
        # Contrastive loss -> all positive samples generate similar features between eachother
        # and unique between eachother
        nce_loss, pos_sim, neg_sim = self.compute_contrastive_loss(
            view1_feat, view1_offset, view2_feat, view2_offset, match_index
        )

        loss = nce_loss * self.contrast_weight
        result_dict = dict(nce_loss=nce_loss, pos_sim=pos_sim, neg_sim=neg_sim)

        # Predict rgb
        if self.color_head is not None:
            assert "view1_color" in data_dict.keys()
            assert "view2_color" in data_dict.keys()
            view1_color = data_dict["view1_color"]
            view2_color = data_dict["view2_color"]

            # Predict only for masked points
            view1_color_pred = self.color_head(view1_feat[view1_point_mask])
            view2_color_pred = self.color_head(view2_feat[view2_point_mask])

            # L2?
            color_loss = (
                torch.sum((view1_color_pred - view1_color[view1_point_mask]) ** 2)
                + torch.sum((view2_color_pred - view2_color[view2_point_mask]) ** 2)
            ) / (view1_color_pred.shape[0] + view2_color_pred.shape[0])
            loss = loss + color_loss * self.reconstruct_weight
            result_dict["color_loss"] = color_loss

        # Predict normals
        if self.normal_head is not None:
            assert "view1_normal" in data_dict.keys()
            assert "view2_normal" in data_dict.keys()
            view1_normal = data_dict["view1_normal"]
            view2_normal = data_dict["view2_normal"]

            # Predict only for masked points
            view1_normal_pred = self.normal_head(view1_feat[view1_point_mask])
            view2_normal_pred = self.normal_head(view2_feat[view2_point_mask])

            # normalize
            view1_normal_pred = view1_normal_pred / (
                torch.norm(view1_normal_pred, p=2, dim=1, keepdim=True) + 1e-10
            )
            view2_normal_pred = view2_normal_pred / (
                torch.norm(view2_normal_pred, p=2, dim=1, keepdim=True) + 1e-10
            )

            normal_loss = (
                torch.sum(view1_normal_pred * view1_normal[view1_point_mask])
                + torch.sum(view2_normal_pred * view2_normal[view2_point_mask])
            ) / (view1_normal_pred.shape[0] + view2_normal_pred.shape[0])
            loss = loss + normal_loss * self.reconstruct_weight
            result_dict["normal_loss"] = normal_loss

        result_dict["loss"] = loss
        return result_dict
