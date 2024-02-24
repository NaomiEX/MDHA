import torch
from torch import nn
from copy import deepcopy
from mmcv.runner import BaseModule
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from .misc import linear_relu_ln
from .positional_encoding import pos2posemb3d
from ...constants import *

@PLUGIN_LAYERS.register_module()
class AnchorEmbedding(BaseModule):
    def __init__(self,
                 mode=0,
                 embed_center_only=True,
                 output_dims=256,
                 ):
        super().__init__()
        self.mode = mode
        self.embed_center_only=embed_center_only
        if not embed_center_only: 
            raise NotImplementedError()
        
        def embedding_layer(input_dims, output_dims):
            return nn.Sequential(
                *linear_relu_ln(output_dims, 1, 2, input_dims)
            )

        if self.mode == ANCHOR_SINCOS_LIN:
            self.query_embedding = nn.Sequential(
                nn.Linear(output_dims*3//2, output_dims),
                nn.ReLU(),
                nn.Linear(output_dims, output_dims),
            )
    
        elif self.mode == ANCHOR_LIN:
            self.center_emb = embedding_layer(3, output_dims)
        
    
    def forward(self,
                anchors,
                ):
        assert self.embed_center_only
        if self.mode == ANCHOR_SINCOS_LIN:
            query_embedding = self.query_embedding(pos2posemb3d(anchors[..., :3]))
        elif self.mode == ANCHOR_LIN:
            query_embedding = self.center_emb(anchors[..., :3])
        return query_embedding
