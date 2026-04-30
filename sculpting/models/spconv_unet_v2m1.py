from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn

import spconv.pytorch as spconv
from torch_geometric.utils import scatter

from timm.layers import trunc_normal_

from pointcept.models.builder import MODELS
from pointcept.models.utils import offset2batch

from pointcept.models.utils.structure import Point
from pointcept.models.sparse_unet.spconv_unet_v1m1_base import BasicBlock, SpUNetBase


@MODELS.register_module("SpUNet-v2m1")
class SpUNetWithPointStructure(SpUNetBase):
    def forward(self, point):

        if not isinstance(point, Point):
            out_point = Point(point)
        else:
            out_point=Point(point)


        point.sparsify()

        feat = super().forward(point)
        out_point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(feat)
        out_point.feat = feat

        return out_point
