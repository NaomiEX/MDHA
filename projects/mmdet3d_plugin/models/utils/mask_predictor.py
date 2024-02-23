import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_
from .dam import attn_map_to_flat_grid
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmcv.runner import BaseModule
from mmdet.models import build_loss

@PLUGIN_LAYERS.register_module()
class MaskPredictor(BaseModule):
    def __init__(self, in_dim, hidden_dim, loss_type="multilabel_soft_margin_loss",
                 sigmoid_on_output=False, loss_cfg=None, loss_weight=1.0):
        self._iter = 0
        super().__init__()
        # self.cams_as_global=cams_as_global
        self.hidden_dim = hidden_dim
        self.layer1 = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        assert loss_type.lower() in ["multilabel_soft_margin_loss", "l1_loss"]
        self.loss_type=loss_type.lower()
        self.loss_weight=loss_weight
        if self.loss_type in ["l1_loss"]:
            loss_cfg = dict(type="L1Loss", loss_weight=0.25) if loss_cfg is None else loss_cfg
            self.mask_loss = build_loss(loss_cfg)
        self.sigmoid_on_output=sigmoid_on_output

    def init_weights(self):
        for name, param in self.named_parameters():
            if param.requires_grad and param.dim() > 1:
                # param_type = name.split('.')[-1]
                # if param_type == 'bias':
                #     constant_(param, 0.)
                # else:
                xavier_uniform_(param)

    def forward(self, x):
        """
        Args:
            x (_type_): [B, h0*N*w0+..., C]
            B (_type_): _description_
            N (_type_): _description_
        """
        z = self.layer1(x) # [B, h0*N*w0+..., C]
        z_local, z_global = torch.split(z, self.hidden_dim // 2, dim=-1) # [B, h0*N*w0+..., C//2]
        # [B, 1, C//2] -> [B, h0*N*w0+..., C//2]
        z_global = z_global.mean(dim=1, keepdim=True).expand(-1, z_local.shape[1], -1)
        z = torch.cat([z_local, z_global], dim=-1) # [B, h0*N*w0+..., C]
        out = self.layer2(z) # [B, h0*N*w0+..., 1]
        if self.sigmoid_on_output:
            out = out.sigmoid()
        return out
    
    def loss(self, mask_prediction, sampling_locations, attn_weights,
             flattened_spatial_shapes, flattened_level_start_index, 
             sparse_token_nums):
        """
        Args:
            mask_prediction (_type_): [B, h0*N*w0+...]
            sampling_locations (_type_): [B, num_dec_layers, Q, n_heads, n_levels, n_points, 2] | 
                                         [B, Q, n_heads, n_levels, n_points, 2]
            attn_weights (_type_): [B, num_dec_layers, Q, n_heads, n_levels, n_points] | 
                                   [B, Q, n_heads, n_levels, n_points]
            flattened_spatial_shapes (_type_): [n_levels, 2]
            flattened_level_start_index (_type_): [n_levels]
            sparse_token_nums (_type_): rho points from backbone

        Returns:
            _type_: _description_
        """
        # [B, h0*N*w0+...]
        # with torch.no_grad():
        flat_grid_attn_map = attn_map_to_flat_grid(flattened_spatial_shapes, flattened_level_start_index,
                                                    sampling_locations, attn_weights, sum_out=True)
        # flat_grid_attn_map_dec=flat_grid_attn_map_dec.sum(dim=(1,2)) # [B, h0*N*w0+...]
        topk_idx_tgt = torch.topk(flat_grid_attn_map, sparse_token_nums, dim=-1)[1] # [B, p]
        if self.loss_type == "multilabel_soft_margin_loss":
            target = torch.zeros_like(mask_prediction) # [B, h0*N*w0+...]
            target.scatter_(1, topk_idx_tgt, 1)
            mask_pred_loss = F.multilabel_soft_margin_loss(mask_prediction, target)
        elif self.loss_type == "l1_loss":
            assert hasattr(self, "loss")
            num_tokens = mask_prediction.numel()
            mask_pred_loss = self.mask_loss(mask_prediction, flat_grid_attn_map, avg_factor=num_tokens)
        else:
            raise Exception(f"loss type: {self.loss_type} is not supported")
        mask_pred_loss=torch.nan_to_num(mask_pred_loss, nan=1e-16, posinf=100.0, neginf=-100.0)
        mask_loss_dict= dict(mask_loss=mask_pred_loss * self.loss_weight)
        if self._iter < 10:
            print("USING NAN TO NUM IN MASK PRED")
        self._iter += 1
        return mask_loss_dict