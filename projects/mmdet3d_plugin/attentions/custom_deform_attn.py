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
# from projects.mmdet3d_plugin.attentions.ops.modules import MSDeformAttn

from mmcv.runner.base_module import BaseModule
from .ops.modules.ms_deform_attn import _is_power_of_2

import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

@ATTENTION.register_module()
class CustomDeformAttn(BaseModule):
    #! WARNING: RIGHT NOW ONLY WORKS FOR BATCH_FIRST=TRUE
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
        self._iter = 0
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

        # self._reset_parameters()
        self.proj_drop = nn.Dropout(proj_drop)

        self.wrap_around = with_wrap_around
        self.key_weight_modulation = key_weight_modulation
        if self.key_weight_modulation:
            raise ValueError("not using key weight modulation rn")
            self.key_proj = nn.Linear(embed_dims, 1)
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
            if self._iter == 0: print("ENCODING 2D REF PTS INTO MULTI QUERY")
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


        # [B, h0*N*w0 + h1*N*w1+..., C]
        if self.mlvl_feats_format == 0: # this should be [B, N, HW, C]
            value = self.flatten_memory(value.clone(), spatial_shapes)
            if key is not None:
                key = self.flatten_memory(key.clone(), spatial_shapes)
                assert key.shape == value.shape
        else:
            assert self.mlvl_feats_format == 1 # this should be [B, H*N*W, C]
            assert value.dim() == 3
            if key is not None:
                assert key.dim() == 3
        if flattened_spatial_shapes is None:
            print("warning: flattened_spatial_shapes is not given")
            assert spatial_shapes is not None
            input_spatial_shapes = spatial_shapes.clone()
            input_spatial_shapes[:, 1] = input_spatial_shapes[:, 1] * num_cameras
        else:
            input_spatial_shapes = flattened_spatial_shapes
        if flattened_lvl_start_index is None:
            print("warning: flattened_lvl_start_index is not given")
            input_level_start_index = torch.cat((input_spatial_shapes.new_zeros((1, )), 
                                        input_spatial_shapes.prod(1).cumsum(0)[:-1]))
        else:
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
        """
        Args:
            query: [B, Q, 256]
            value: [B, N, h0*w0+..., C] | [B, h0*N*w0+..., C]
            key: either same as value or NONE if two_stage
            query_pos: Tensor [B, Q, 256]
            key_pos: 3d pos embeds; Tensor [B, N, h0*w0+..., C] or NONE if two_stage
            identity (optional): 
            reference_points (optional): projected cam reference points. Tensor [B, Q, n_levels, 2]
            spatial_shapes = orig spatial shapes [n_levels, 2]
            flattened_spatial_shapes = flattened transformed spatial shapes [n_levels, 2]
            flattened_lvl_start_index = flattened lvl start index
        """
        # TODO: INVESTIGATE WHETHER ENCODING THE 2D REF PTS INFO INTO QUERY TO 
        # TODO(cont.): DIFFERENTIATE QUERIES WHICH ARE MAPPED TO TWO POINTS ARE BENEFICIAL
        # TODO2: INVESTIGATE WHETHER ENCODING THE CAMERA INFO INTO QUERY FOR FURTHER DIFFERENTIATION HELPS
        assert self._is_init

        if self._iter == 0: print(f"CUSTOM DEFORM ATTN: USING {self.n_points} POINTS")

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
        # if self._iter % 50 == 0:
        #     num_sampling_locs_out_of_range = torch.logical_or(sampling_locations > 1, sampling_locations < 0).sum()
        #     print(f"proportion of sampling locs out of range [0,1]: {num_sampling_locs_out_of_range/sampling_locations.numel()}")
        if self.wrap_around:
            sampling_locations = sampling_locations % 1.0
            # isnotnan=sampling_locations.isfinite()
            # assert (sampling_locations[isnotnan] >= 0.0).all()

            if self._iter % 50 == 0:
                num_sampling_locs_out_of_range = torch.logical_or(sampling_locations > 1, sampling_locations < 0).sum()
                print(f"(AFTER WRAP AROUND)proportion of sampling locs out of range [0,1]: {num_sampling_locs_out_of_range/sampling_locations.numel()}")

        attention_weights = self.attention_weights(query) # [B, p, num_heads * n_levels * n_points]

        if self.key_weight_modulation:
            attention_weights = attention_weights.view(N, Len_q, self.num_heads, self.n_levels, self.n_points)
            assert key is not None
            key_projected = self.key_proj(key).squeeze(-1) # [B, h0*N*w0 + ...]
            assert list(key_projected.shape) == [N, Len_in]
            try:
                attention_weights = self.modulate_weights_by_key(sampling_locations, attention_weights, key_projected,
                                                                    spatial_shapes, input_level_start_index)
            except:
                import pickle
                modulate_inp = (sampling_locations, attention_weights, key_projected,
                                                                    spatial_shapes, input_level_start_index)
                with open("modulate_input.pkl", "wb") as f:
                    pickle.dump(modulate_inp, f)
                raise Exception()
            assert list(attention_weights.shape) == [N, Len_q, self.num_heads, self.n_points, self.n_levels]
            attention_weights = attention_weights.transpose(-2, -1).flatten(-2, -1)
        else:
            attention_weights=attention_weights.view(N, Len_q, self.num_heads, self.n_levels * self.n_points)
        
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

        self._iter += 1

        if return_query_only:
            return output
        else:
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

    @staticmethod
    def modulate_weights_by_key(sampling_locs, attn_weights, key, spatial_shapes, level_start_index):
        # print("new3 func")
        B, Q, n_heads, n_levels, n_points, _ = sampling_locs.shape

        with torch.no_grad():
            valid_margins, idx_flattened_all = CustomDeformAttn.get_sampling_locs_weights_and_idx(
                sampling_locs, spatial_shapes, level_start_index)
        key_gathered_all = []
        for idx_flattened in idx_flattened_all:
            key_gathered = key.gather(1, idx_flattened) # [B, Q*n_heads*n_points*n_levels]
            key_gathered_all.append(key_gathered)
        key_gathered_all_stack = torch.stack(key_gathered_all) # [4, B, Q*n_heads*n_points*n_levels]
        assert list(key_gathered_all_stack.shape) == [4, B, Q*n_heads*n_points*n_levels]
        # weights_all_stack = torch.stack(weights_all) # [4, B, Q, n_heads, n_points, n_levels]
        weights_all_stack = attn_weights.transpose(3, 4).unsqueeze(0) * valid_margins
        assert list(weights_all_stack.shape) == [4, B, Q, n_heads, n_points, n_levels]
        key_gathered_all_stack = key_gathered_all_stack.unflatten(-1, (Q, n_heads, n_points, n_levels))
        modulated_weights = key_gathered_all_stack * weights_all_stack # [4, B, Q, n_heads, n_points, n_levels]
        modulated_weights = modulated_weights.sum(dim=0) # [B, Q, n_heads, n_points, n_levels]
        return modulated_weights
    
    @staticmethod
    def get_sampling_locs_weights_and_idx(sampling_locs, spatial_shapes, level_start_index):
        """
        Args:
            sampling_locs (_type_): [B, Q, n_heads, n_levels, n_points, 2]
            attn_weights (_type_): [B, Q, n_heads, n_levels, n_points]
            key (_type_): [B, h0*w0+...]
            spatial_shapes (_type_): [n_levels, 2]
            level_start_index: [n_levels]
        """
        B, Q, n_heads, n_levels, n_points, _ = sampling_locs.shape

        # W,H
        rev_spatial_shapes = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], dim=-1)
        sampling_locations = sampling_locs.transpose(3, 4) # [B, Q, n_heads, n_points, n_levels, 2]
        assert list(sampling_locations.shape) == [B, Q, n_heads, n_points, n_levels, 2]
        colrow_f = sampling_locations * rev_spatial_shapes # [B, Q, n_heads, n_points, n_levels, 2]
        colrow_ll = colrow_f.floor().to(torch.int32)

        colrow_lh = colrow_ll.detach().clone()
        colrow_lh[..., 1] = colrow_lh[..., 1] + 1

        colrow_hl = colrow_ll.detach().clone()
        colrow_hl[..., 0] = colrow_hl[..., 0] + 1

        colrow_hh = colrow_ll.detach().clone()
        colrow_hh = colrow_hh + 1

        margin_ll = (colrow_f - colrow_ll).prod(dim=-1) # [B, Q, n_heads, n_points, n_levels]
        margin_lh = -(colrow_f - colrow_lh).prod(dim=-1)
        margin_hl = -(colrow_f - colrow_hl).prod(dim=-1)
        margin_hh = (colrow_f - colrow_hh).prod(dim=-1)

        valid_margins=[]
        idx_flattened_all = []
        zipped = [(colrow_ll, margin_hh), (colrow_lh, margin_hl), (colrow_hl, margin_lh), (colrow_hh, margin_ll)]
        for colrow, margin in zipped:
            valid_mask = torch.logical_and(
                torch.logical_and(colrow[..., 0] >= 0, colrow[..., 0] < rev_spatial_shapes[..., 0]),
                torch.logical_and(colrow[..., 1] >= 0, colrow[..., 1] < rev_spatial_shapes[..., 1]),
            )
            idx = colrow[..., 1] * spatial_shapes[..., 1] + colrow[..., 0] + level_start_index
            idx = idx * valid_mask # [B, Q, n_heads, n_points, n_levels]
            idx_flatten = idx.flatten(1, 4) # [B, Q*n_heads*n_points*n_levels]
            valid_margins.append(valid_mask * margin)
            idx_flattened_all.append(idx_flatten)
        return torch.stack(valid_margins), idx_flattened_all