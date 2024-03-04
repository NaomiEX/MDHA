from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import math
import warnings
import torch
from torch import nn
from mmcv.cnn.bricks.registry import ATTENTION
from ..models.utils.projections import Projections
from projects.mmdet3d_plugin.attentions.ops.functions import MSDeformAttnFunction
from projects.mmdet3d_plugin.models.utils.positional_encoding import posemb2d
from projects.mmdet3d_plugin.models.utils.misc import MLN
from projects.mmdet3d_plugin.models.utils.debug import *

from mmcv.runner.base_module import BaseModule
from .ops.modules.ms_deform_attn import _is_power_of_2

import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

@ATTENTION.register_module()
class CustomDeformAttn(BaseModule):
    def __init__(self, embed_dims, num_heads, proj_drop=0.0, 
                 n_levels=4, n_points=4*6, with_wrap_around=False, key_weight_modulation=False, 
                 div_sampling_offset_x=False, mlvl_feats_format=0, 
                 ref_pts_mode="single", encode_2d_ref_pts_into_query_pos=False,
                 query_pos_2d_ref_pts_encoding_method="mln", test_mode=False,
                 residual_mode="add",
                   **kwargs):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError('embed_dims must be divisible by num_heads, but got {} and {}'.format(embed_dims, num_heads))
        _d_per_head = embed_dims // num_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set embed_dims in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")
        self.im2col_step = 64

        self.embed_dims = embed_dims
        self.n_levels = n_levels
        self.num_heads = num_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(embed_dims, num_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * n_levels * n_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.python_ops_for_test = False
        self.residual_mode=residual_mode

        self.proj_drop = nn.Dropout(proj_drop)

        self.wrap_around = with_wrap_around
        self.key_weight_modulation = key_weight_modulation
        if self.key_weight_modulation:
            raise ValueError("not using key weight modulation rn")
        self.div_sampling_offset_x=div_sampling_offset_x
        self.mlvl_feats_format=mlvl_feats_format
        # self.allow_multi_point=allow_multi_point
        self.ref_pts_mode=ref_pts_mode.lower()
        assert self.ref_pts_mode in ["single", "multiple"]

        self.encode_2d_ref_pts_into_query_pos=encode_2d_ref_pts_into_query_pos

        if self.encode_2d_ref_pts_into_query_pos:
            self.query_pos_2d_ref_pts_encoding_method = query_pos_2d_ref_pts_encoding_method.lower()
            assert self.query_pos_2d_ref_pts_encoding_method in ["mln", "linear"]
            if self.query_pos_2d_ref_pts_encoding_method == "mln":
                self.query_pos_2d_ref_pts = MLN(self.embed_dims, f_dim=self.embed_dims)
            elif self.query_pos_2d_ref_pts_encoding_method == "linear":
                self.query_pos_2d_ref_pts = nn.Sequential(
                    nn.Linear(self.embed_dims, self.embed_dims),
                    nn.LayerNorm(self.embed_dims)
                )
           
        self._is_init=test_mode

    @property
    def is_init(self):
        return self._is_init

    def reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.num_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)
        self._is_init = True
        print("CIRCULAR DEFORMABLE ATTENION RESET")


    def preprocess(self, query, key, value, query_pos, key_pos, spatial_shapes, reference_points,
                flattened_spatial_shapes=None, flattened_lvl_start_index=None, num_cameras=6,
                num_second_matches=None, second_matches_valid_idxs=None, idx_with_second_match=None,
                **kwargs):
        """
        Args:
            query (_type_): [B, Q, C]
            value (_type_): [B, n_tokens, C]
            query_pos (_type_): [B, Q, C]
            reference_points (_type_): [B, R, n_levels,C], if self.ref_pts_mode == "multiple", its possible that R != Q
            num_cameras (int, optional): _description_. Defaults to 6.
            num_second_matches (_type_, optional): _description_. Defaults to None.
            second_matches_valid_idxs (_type_, optional): [B, num_second_matches] indexes the valid second matches
                because diff images have varying number of second matches. Defaults to None.
            idx_with_second_match (_type_, optional): [B, Q], indexes the queries with a second match. Defaults to None.
        """
        assert query_pos is not None

        B, Q, C = query.shape
        if self.ref_pts_mode == "multiple" and num_second_matches > 0:
            assert second_matches_valid_idxs is not None 
            assert idx_with_second_match is not None
            second_match_queries = query.new_zeros([B, num_second_matches, C])
            second_match_query_pos = second_match_queries.clone()
            second_match_queries[second_matches_valid_idxs] = query[idx_with_second_match]
            query = torch.cat([query, second_match_queries], 1) # [B, R, C]
            second_match_query_pos[second_matches_valid_idxs] = query_pos[idx_with_second_match]
            query_pos = torch.cat([query_pos, second_match_query_pos], 1) # [B, R, C]

        assert query.shape == query_pos.shape, \
            f"NOT EQUAL: ({query.shape}, {query_pos.shape})"
        assert query.shape[:2] == reference_points.shape[:2]

        if self.encode_2d_ref_pts_into_query_pos:
            if do_debug_process(self): print("ENCODING 2D REF PTS INTO MULTI QUERY")
            ref_pts_2d_emb = posemb2d(reference_points[:,:, 0]) # [B, R, 256]

            if self.query_pos_2d_ref_pts_encoding_method == "mln":
                query_pos = self.query_pos_2d_ref_pts(query_pos, ref_pts_2d_emb) # [B, Q, 256]
            elif self.query_pos_2d_ref_pts_encoding_method == "linear":
                query_pos = query_pos+ self.query_pos_2d_ref_pts(ref_pts_2d_emb)

        if query_pos is not None:
            assert query.shape == query_pos.shape
            query = query + query_pos
        # assert key_pos is not None
        if key_pos is not None:
            assert key is not None
            key=key+key_pos

        input_spatial_shapes = flattened_spatial_shapes
        input_level_start_index = flattened_lvl_start_index

        if reference_points.dim() == 3:
            assert list(reference_points.shape) == [B, Q, 2]
            reference_points = reference_points.unsqueeze(2).repeat(1, 1, self.n_levels, 1) # [B, Q, n_lvls, 2]

        return query, key, value, input_spatial_shapes, input_level_start_index, reference_points
    
    def postprocess(self, output, num_second_matches=None, second_matches_valid_idxs=None, idx_with_second_match=None,
                    **kwargs):
        if self.ref_pts_mode == "multiple" and num_second_matches > 0:
            assert second_matches_valid_idxs is not None
            assert idx_with_second_match is not None

            out_main = output[:, :-num_second_matches] # [B, Q, C]
            out_second = output[:, -num_second_matches:]
            # TODO: RIGHT NOW JUST AVERAGES OVER RESULTS OF PROJECTED POINTS TRY OTHER METHODS OF COMBINING THEM
            out_main[idx_with_second_match] = (out_main[idx_with_second_match] + out_second[second_matches_valid_idxs]) / 2
            return out_main
        else:
            return output
            

    def forward(self, query, value, key=None, identity=None, query_pos=None, key_pos=None, 
                reference_points=None, spatial_shapes=None, flattened_spatial_shapes=None,
                flattened_lvl_start_index=None, num_cameras=6, return_query_only=False,
                **kwargs):
        assert self._is_init

        if do_debug_process(self): print(f"CUSTOM DEFORM ATTN: USING {self.n_points} POINTS")

        if identity is None:
            identity = query

        query, key, value, input_spatial_shapes, input_level_start_index, reference_points = \
            self.preprocess(query, key, value, query_pos, key_pos, spatial_shapes, reference_points, 
                            flattened_spatial_shapes, flattened_lvl_start_index, **kwargs)

        N, Len_q, _ = query.shape
        N, Len_in, _ = value.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in
        value = self.value_proj(value)
        value = value.view(N, Len_in, self.num_heads, self.embed_dims // self.num_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.num_heads, self.n_levels, self.n_points, 2)
        if self.div_sampling_offset_x:
            sampling_offsets[..., 0] = sampling_offsets[..., 0] / num_cameras
        
        assert reference_points.dim() == 4
        if reference_points.shape[-1] == 2:
            # (W,H)
            # [n_levels, 2]
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            # reference_points: [B, R, 1, n_levels, 1, 2]
            # offset_normalizer: [1, 1, 1, n_levels, 1, 2]
            # sampling_offsets/offset_normalizer: [B, R, n_heads, n_levels, n_points, 2]
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        assert sampling_locations.shape == sampling_offsets.shape
        if self.wrap_around:
            if do_debug_process(self): print(f"CDA: USING WRAP AROUND")
            sampling_locations = sampling_locations % 1.0

        # [B, p, num_heads * n_levels * n_points]
        attention_weights = self.attention_weights(query) \
                            .view(N, Len_q, self.num_heads, self.n_levels * self.n_points)
        
        assert list(attention_weights.shape) == [N, Len_q, self.num_heads, self.n_levels * self.n_points]
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.num_heads, self.n_levels, self.n_points)

        # [B, R, C]
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)

        output = self.postprocess(output, **kwargs)
        assert output.shape == identity.shape

        if self.residual_mode == "cat":
            output = torch.cat([output, identity], dim=-1)
        else:
            output = identity + self.proj_drop(output)

        if return_query_only:
            return output
        else:
            raise NotImplementedError()
            # !WARNING: be careful here because for mult ref pts some of the sampling locations and attention weights are invalid (i.e. masked)
            # (cont.): can use num_second_matches and second_matches_valid_idxs to get valid ones similar to pre and post process here
            return output, sampling_locations, attention_weights

            
    def flatten_memory(self,tensor, spatial_shapes):
        """
        Args:
            tensor (_type_): [B, N, h0*w0+..., C]
            shapes (_type_): [n_levels, 2]
        Returns:
            Tensor [B, h0*N*w0 + h1*N*w1+..., C]
        """
        assert self.mlvl_feats_format == 0
        split_levels = tensor.split([h*w.item() for h,w in spatial_shapes], dim=2) # [B, N, H_i*W_i, C]
        split_hw = [x.unflatten(2, (shape.tolist())) for x, shape in zip(split_levels, spatial_shapes)] # [B, N, H_i, W_i, C]
        split_flipped = [x.transpose(1, 2).contiguous() for x in split_hw] # [B, H_i, N, W_i, C]
        # [B, H_i*N*W_i, C] -> [B, h0*N*w0 + h1*N*w1+..., C]
        flattened = torch.cat([x.flatten(1, 3) for x in split_flipped], dim=1)
        return flattened
