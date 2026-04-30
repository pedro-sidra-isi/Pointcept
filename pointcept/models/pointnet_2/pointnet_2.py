"""PointNet++ Implementation for Pointcept Framework"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.builder import MODELS
from pointcept.models.utils.structure import Point
from pointcept.models.utils import offset2batch

try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../..', 'libs', 'pointops'))
    from pointops.functions import farthest_point_sampling, ball_query, grouping2
except ImportError:
    farthest_point_sampling = None


def simple_fps(xyz, npoint):
    """Simple FPS fallback implementation."""
    N = xyz.shape[0]
    device = xyz.device
    
    if npoint >= N:
        return torch.arange(N, device=device)
    
    idx_list = [torch.randint(0, N, (1,), device=device)]
    distances = torch.full((N,), float('inf'), device=device)
    
    for i in range(1, npoint):
        last_point = xyz[idx_list[-1]]
        new_distances = torch.sum((xyz - last_point) ** 2, dim=-1)
        distances = torch.minimum(distances, new_distances)
        farthest_idx = torch.argmax(distances)
        idx_list.append(farthest_idx.unsqueeze(0))
    
    return torch.cat(idx_list)

def simple_ball_query(xyz, new_xyz, radius, nsample):
    """Simple Ball Query fallback implementation."""
    M, N = new_xyz.shape[0], xyz.shape[0]
    device = xyz.device
    
    dist = torch.cdist(new_xyz, xyz)
    ball_idx = torch.zeros((M, nsample), dtype=torch.long, device=device)
    
    for i in range(M):
        valid_mask = dist[i] <= radius
        valid_indices = torch.where(valid_mask)[0]
        
        if len(valid_indices) >= nsample:
            perm = torch.randperm(len(valid_indices))[:nsample]
            ball_idx[i] = valid_indices[perm]
        else:
            if len(valid_indices) > 0:
                repeat_times = (nsample + len(valid_indices) - 1) // len(valid_indices)
                repeated = valid_indices.repeat(repeat_times)[:nsample]
                ball_idx[i] = repeated
            else:
                closest_idx = torch.argmin(dist[i])
                ball_idx[i] = closest_idx.repeat(nsample)
    
    return ball_idx

def fps_and_ball_query(xyz, points, offset, npoint, radius, nsample):
    """FPS + Ball Query with fallback implementation."""
    if farthest_point_sampling is not None:
        new_offset = torch.tensor([npoint], device=xyz.device, dtype=torch.int)
        idx = farthest_point_sampling(xyz, offset, new_offset)
        new_xyz = xyz[idx]
        ball_idx, _ = ball_query(nsample, radius, 0.0, xyz, offset, new_xyz, new_offset)
        
        if points is not None:
            grouped_points = grouping2(points, ball_idx)
            grouped_xyz = grouping2(xyz, ball_idx)
            grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(1)
            grouped_features = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        else:
            grouped_xyz = grouping2(xyz, ball_idx)
            grouped_features = grouped_xyz - new_xyz.unsqueeze(1)
    else:
        idx = simple_fps(xyz, npoint)
        new_xyz = xyz[idx]
        ball_idx = simple_ball_query(xyz, new_xyz, radius, nsample)
        
        if points is not None:
            grouped_points = points[ball_idx]
            grouped_xyz = xyz[ball_idx]
            grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(1)
            grouped_features = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        else:
            grouped_xyz = xyz[ball_idx]
            grouped_features = grouped_xyz - new_xyz.unsqueeze(1)
    
    return new_xyz, grouped_features


def interpolate_features(xyz1, xyz2, feat2, k=3):
    """Interpolate features using k nearest neighbors with memory optimization."""
    B, N, _ = xyz1.shape
    _, M, C = feat2.shape
    
    if M == 1:
        return feat2.expand(B, N, C)
    
    chunk_size = min(1000, N)
    interpolated_list = []
    
    for i in range(0, N, chunk_size):
        end_idx = min(i + chunk_size, N)
        xyz1_chunk = xyz1[:, i:end_idx, :]
        dist_chunk = torch.cdist(xyz1_chunk, xyz2)
        k_actual = min(k, M)
        dists, indices = torch.topk(dist_chunk, k_actual, dim=-1, largest=False)
        dists = dists + 1e-8
        
        weights = 1.0 / dists
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        gathered_feat = torch.gather(feat2.unsqueeze(1).expand(-1, end_idx - i, -1, -1),
                                     2,
                                     indices.unsqueeze(-1).expand(-1, -1, -1, C))
        
        interpolated_chunk = (gathered_feat * weights.unsqueeze(-1)).sum(dim=2)
        interpolated_list.append(interpolated_chunk)
    
    return torch.cat(interpolated_list, dim=1)


class PointNetSetAbstraction(nn.Module):
    """Set Abstraction using FPS + Ball Query."""
    
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = in_channel + 3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
    def forward(self, xyz, points, offset):
        new_xyz, grouped_features = fps_and_ball_query(
            xyz, points, offset, self.npoint, self.radius, self.nsample
        )
        
        grouped_features = grouped_features.permute(2, 1, 0).unsqueeze(0)
        
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            grouped_features = F.relu(bn(conv(grouped_features)))
        
        new_points = torch.max(grouped_features, 2)[0]
        new_points = new_points.squeeze(0).permute(1, 0)
        new_offset = torch.tensor([self.npoint], device=xyz.device, dtype=torch.int)
        
        return new_xyz, new_points, new_offset


class PointNetFeaturePropagation(nn.Module):
    """Feature Propagation layer."""
    
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
    
    def forward(self, xyz1, xyz2, points1, points2, offset1, offset2):
        try:
            from pointops.functions import interpolation2 as interp2
            interpolated_points = interp2(points2, offset2, xyz2, offset1, xyz1)
        except:
            interpolated_points = interpolate_features(xyz1.unsqueeze(0), xyz2.unsqueeze(0), points2.unsqueeze(0))
            interpolated_points = interpolated_points.squeeze(0)
        
        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
        
        new_points = new_points.permute(1, 0).unsqueeze(0)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        new_points = new_points.squeeze(0).permute(1, 0)
        return new_points


@MODELS.register_module("PointNet2")
class PointNetPlusPlusBackbone(nn.Module):
    """PointNet++ backbone."""
    
    def __init__(
        self,
        in_channels=3,
        out_channels=128,
        npoints=[1024, 256, 64, 16],
        radius=[0.1, 0.2, 0.4, 0.8],
        nsample=[32, 32, 32, 32],
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        
        self.sa1 = PointNetSetAbstraction(
            npoint=npoints[0], radius=radius[0], nsample=nsample[0],
            in_channel=in_channels, mlp=[64, 64, 128]
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=npoints[1], radius=radius[1], nsample=nsample[1],
            in_channel=128, mlp=[128, 128, 256]
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=npoints[2], radius=radius[2], nsample=nsample[2],
            in_channel=256, mlp=[256, 256, 512]
        )
        self.sa4 = PointNetSetAbstraction(
            npoint=npoints[3], radius=radius[3], nsample=nsample[3],
            in_channel=512, mlp=[512, 512, 1024]
        )
        
        self.fp4 = PointNetFeaturePropagation(in_channel=1024 + 512, mlp=[512, 512])
        self.fp3 = PointNetFeaturePropagation(in_channel=512 + 256, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[128, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + in_channels, mlp=[128, 128, out_channels])
        
    def forward(self, point):
        l0_xyz = point.coord
        l0_points = point.feat
        l0_offset = point.offset
        
        l1_xyz, l1_points, l1_offset = self.sa1(l0_xyz, l0_points, l0_offset)
        l2_xyz, l2_points, l2_offset = self.sa2(l1_xyz, l1_points, l1_offset)
        l3_xyz, l3_points, l3_offset = self.sa3(l2_xyz, l2_points, l2_offset)
        l4_xyz, l4_points, l4_offset = self.sa4(l3_xyz, l3_points, l3_offset)
        
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points, l3_offset, l4_offset)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points, l2_offset, l3_offset)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points, l1_offset, l2_offset)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points, l0_offset, l1_offset)
        
        point.feat = l0_points
        
        return point