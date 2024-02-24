import torch
from torch import nn
from copy import deepcopy

from mmcv.runner import BaseModule, Linear
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS

from .lidar_utils import clamp_to_lidar_range
from ...constants import *

@PLUGIN_LAYERS.register_module()
class AnchorRefinement(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 output_dim=10,
                 num_cls=10,
                 with_quality_estimation=False,
                 refine_center_only=True,
                 limit=False,
                 num_reg_fcs=2,
                 ):
        super().__init__()
        self.embed_dims=embed_dims
        self.with_quality_estimation=with_quality_estimation
        if with_quality_estimation:
            raise NotImplementedError()
        self.refine_center_only=refine_center_only
        self.limit=limit
        self.refine_state = [X, Y, Z] if refine_center_only else [X,Y,Z,W,L,H, SIN_YAW, COS_YAW]

        reg_branch = []
        for _ in range(num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, output_dim))
        self.reg_branch = nn.Sequential(*reg_branch)

        cls_branch = []
        for _ in range(num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, num_cls))
        self.cls_branch = nn.Sequential(*cls_branch)

    def forward(self,
                query,
                anchors,
                query_pos,
                time_interval=None,
                return_cls=True,
                ):
        # TODO: use query + query_pos
        feature = query

        reg_out = self.reg_branch(feature)
        reg_out[..., self.refine_state] = reg_out[..., self.refine_state] \
                                          + anchors[..., self.refine_state]
        if self.limit:
            print("WARNING: USING LIMIT")
            reg_out = clamp_to_lidar_range(reg_out, PC_RANGE)
        
        if not self.refine_center_only:
            raise NotImplementedError()
        
        if return_cls:
            # TODO: try use query + query_pos
            cls_out = self.cls_branch(query)
        else:
            cls_out=None
        
        if self.with_quality_estimation:
            raise NotImplementedError()
        else:
            qt_out = None

        return reg_out, cls_out, qt_out