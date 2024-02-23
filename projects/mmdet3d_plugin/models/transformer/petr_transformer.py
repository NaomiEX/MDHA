# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
import warnings
import torch
# import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence,
                                         build_attention,
                                         build_feedforward_network)
from mmcv.cnn.bricks.drop import build_dropout
from mmdet.models.utils.builder import TRANSFORMER
from mmdet.models.utils.transformer import inverse_sigmoid
from mmcv.cnn import build_norm_layer, xavier_init
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.registry import (ATTENTION,TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.utils import deprecated_api_warning, ConfigDict
import copy
from torch.nn import ModuleList
from ..utils.attention import FlashMHA
import torch.utils.checkpoint as cp

from mmcv.runner import auto_fp16

from ..utils.projections import Projections, convert_3d_to_2d_global_cam_ref_pts, convert_3d_to_mult_2d_global_cam_ref_pts
from ..utils.lidar_utils import denormalize_lidar, normalize_lidar, clamp_to_lidar_range, not_in_lidar_range
from projects.mmdet3d_plugin.attentions.custom_deform_attn import CustomDeformAttn
@ATTENTION.register_module()
class PETRMultiheadFlashAttention(BaseModule):
    """A wrapper for ``torch.nn.MultiheadAttention``.
    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 batch_first=True,
                 **kwargs):
        super(PETRMultiheadFlashAttention, self).__init__(init_cfg)
        if 'dropout' in kwargs:
            warnings.warn(
                'The arguments `dropout` in MultiheadAttention '
                'has been deprecated, now you can separately '
                'set `attn_drop`(float), proj_drop(float), '
                'and `dropout_layer`(dict) ', DeprecationWarning)
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = True

        self.attn = FlashMHA(embed_dims, num_heads, attn_drop, dtype=torch.float16, device='cuda',
                                          **kwargs)

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiheadAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `MultiheadAttention`.
        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.
        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
                If None, the ``query`` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            identity (Tensor): This tensor, with the same shape as x,
                will be used for the identity link.
                If None, `x` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. If not None, it will
                be added to `x` before forward function. Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.
        Returns:
            Tensor: forwarded results with shape
            [num_queries, bs, embed_dims]
            if self.batch_first is False, else
            [bs, num_queries embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        out = self.attn(
            q=query,
            k=key,
            v=value,
            key_padding_mask=None)[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))


class MultiheadAttentionWrapper(nn.MultiheadAttention):
    def __init__(self, *args, **kwargs):
        super(MultiheadAttentionWrapper, self).__init__(*args, **kwargs)
        self.fp16_enabled = True

    @auto_fp16(out_fp32=True)
    def forward_fp16(self, *args, **kwargs):
        return super(MultiheadAttentionWrapper, self).forward(*args, **kwargs)

    def forward_fp32(self, *args, **kwargs):
        return super(MultiheadAttentionWrapper, self).forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_fp16(*args, **kwargs)
        else:
            return self.forward_fp32( *args, **kwargs)

@ATTENTION.register_module()
class PETRMultiheadAttention(BaseModule):
    """A wrapper for ``torch.nn.MultiheadAttention``.
    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 batch_first=False,
                 fp16 = False,
                 **kwargs):
        super(PETRMultiheadAttention, self).__init__(init_cfg)
        if 'dropout' in kwargs:
            warnings.warn(
                'The arguments `dropout` in MultiheadAttention '
                'has been deprecated, now you can separately '
                'set `attn_drop`(float), proj_drop(float), '
                'and `dropout_layer`(dict) ', DeprecationWarning)
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.fp16_enabled = True
        if fp16:
            self.attn = MultiheadAttentionWrapper(embed_dims, num_heads, attn_drop, 
                                                  batch_first=batch_first, **kwargs)
        else:
            self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop, 
                                              batch_first=batch_first, **kwargs)

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiheadAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                inp_batch_first=True,
                **kwargs):
        """
        Args:
            query (_type_): current queries|prev queries; Tensor [B, Nq+num_propagated, 256]
            key (_type_, optional):# [B, Nq + num_propagated + rest of memory, 256]
            value (_type_, optional): # [B, Nq + num_propagated + rest of memory, 256]
            identity (_type_, optional): _description_. Defaults to None.
            query_pos (_type_, optional): query pos | prev query pos; Tensor [B, Nq+num_propagated, 256]
            key_pos (_type_, optional): # [B, Nq + num_propagated + rest of memory, 256]
            attn_mask (_type_, optional): temporal attention mask; [pad_size + Nq + num_propagated,  pad_size + Nq + memory_len]
            key_padding_mask (_type_, optional): _description_. Defaults to None.
            inp_batch_first (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        # todo: code to convert instead of assert
        assert (self.batch_first and inp_batch_first) or (self.batch_first == False and inp_batch_first == False)
        
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        # if self.batch_first:
        #     query = query.transpose(0, 1).contiguous()
        #     key = key.transpose(0, 1).contiguous()
        #     value = value.transpose(0, 1).contiguous()

        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)[0]

        # if self.batch_first:
        #     out = out.transpose(0, 1).contiguous()

        return identity + self.dropout_layer(self.proj_drop(out))



@TRANSFORMER_LAYER_SEQUENCE.register_module()
class PETRTransformerEncoder(TransformerLayerSequence):
    """TransformerEncoder of DETR.
    Args:
        post_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`. Only used when `self.pre_norm` is `True`
    """

    def __init__(self, *args, post_norm_cfg=dict(type='LN'), **kwargs):
        super(PETRTransformerEncoder, self).__init__(*args, **kwargs)
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(
                post_norm_cfg, self.embed_dims)[1] if self.pre_norm else None
        else:
            assert not self.pre_norm, f'Use prenorm in ' \
                                      f'{self.__class__.__name__},' \
                                      f'Please specify post_norm_cfg'
            self.post_norm = None

    def forward(self, *args, **kwargs):
        """Forward function for `TransformerCoder`.
        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        x = super(PETRTransformerEncoder, self).forward(*args, **kwargs)
        if self.post_norm is not None:
            x = self.post_norm(x)
        return x


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class PETRTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        post_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self,
                 *args,
                 post_norm_cfg=dict(type='LN'),
                 return_intermediate=False,
                 pc_range=None,
                 two_stage=False,
                 ref_pts_mode="single",
                 use_inv_sigmoid=False,
                 use_sigmoid_on_attn_out=False,
                 mask_pred_target=False,
                 **kwargs):
        kwargs['transformerlayers']['mask_pred_target'] = mask_pred_target
        super(PETRTransformerDecoder, self).__init__(*args, **kwargs)
        self._iter = 0
        self.pc_range = nn.Parameter(torch.tensor(
            pc_range), requires_grad=False)
        self.return_intermediate = return_intermediate
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg,
                                              self.embed_dims)[1]
        else:
            self.post_norm = None
        self.ref_pts_mode = ref_pts_mode
        assert self.ref_pts_mode in ["single", "multiple"]
        self.two_stage=two_stage
        self.use_inv_sigmoid=use_inv_sigmoid
        self.use_sigmoid_on_attn_out=use_sigmoid_on_attn_out
        self.mask_pred_target=mask_pred_target
        

    def forward(self, bbox_embed, query, *args, reference_points=None, lidar2img=None, extrinsics=None, 
                orig_spatial_shapes=None, img_metas=None, num_cameras=6, **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
        NOTE: Q = pad size | cur. queries | propagated
            bbox_embed (nn.ModuleList): same as StreamPETR's reg branch 
            query (Tensor): current queries|prev queries; Tensor [B, Q, 256]
            reference_points (Tensor): current ref pts|prev ref pts Tensor [B, Q, 3]
            lidar2img (Tensor): current frame's lidar2img [B, N, 4, 4]
            extrinsics (Tensor): current frame's extrinsics [B, N, 4, 4]
            spatial_shapes: [n_levels, 2]
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        if not self.return_intermediate:
            x = super().forward(query, *args, reference_points=None, **kwargs)
            if self.post_norm:
                x = self.post_norm(x)[None]
            return x

        intermediate = []
        intermediate_reference_points = []
        sampling_locs_all = []
        attn_weights_all = []

        cam_transformations = dict(lidar2img=lidar2img, lidar2cam=extrinsics)
        assert lidar2img.dim() == 4, f"got lidar2img shape: {lidar2img.shape}"
        assert extrinsics.dim()==4, f"got extrinsics shape: {extrinsics.shape}"

        # init_reference_point = reference_points.clone()
        for lid, layer in enumerate(self.layers):
            if self.use_inv_sigmoid:
                ref_pts_unnormalized = inverse_sigmoid(reference_points.clone()) # R: lidar space
            else:
                ref_pts_unnormalized = denormalize_lidar(reference_points.clone(), self.pc_range) # R:lidar space
            # print(f"petr transformer decoder ref pts unnormalized shape: {ref_pts_unnormalized.shape}")
            if lid == 0:
                init_reference_point =ref_pts_unnormalized.clone()
            # [B, Q, 2]
            with torch.no_grad():
                if self.ref_pts_mode == "single":
                    reference_points_2d_cam, num_non_matches = convert_3d_to_2d_global_cam_ref_pts(cam_transformations,
                                                                        ref_pts_unnormalized, orig_spatial_shapes,
                                                                        img_metas, ret_num_non_matches=True)
                    
                    # if self.debug and self._iter %50 == 0:
                    # if self._iter % 50 == 0:
                    #     num_non_match_prop = num_non_matches.sum(1) / num_non_matches.size(-1)
                    #     debug_msg =f"3d->2d @ decoder layer {lid}, proportion of non match ref pts: {num_non_match_prop}"
                    #     print(debug_msg)
                    #     if self.debug: self.debug_logger.info(debug_msg)
                    num_second_matches, second_matches_valid_idxs, idx_with_second_match = [None]*3
                elif self.ref_pts_mode == "multiple":
                    if self._iter < 10: print("DECODER: USING MULTIPLE REF PTS")
                    ref_pts_mult_outs = convert_3d_to_mult_2d_global_cam_ref_pts(cam_transformations,
                                                                    ref_pts_unnormalized, orig_spatial_shapes,
                                                                    img_metas, ret_num_non_matches=self.debug)
                    reference_points_2d_cam, num_second_matches, second_matches_valid_idxs, idx_with_second_match = \
                        ref_pts_mult_outs[:4]
                    if self.debug and self._iter % 50 == 0:
                        non_matches = ref_pts_mult_outs[4]
                        num_non_match_prop = non_matches.sum(1) / non_matches.size(-1)
                        num_second_matches_prop = second_matches_valid_idxs[1].size(0) / reference_points_2d_cam.size(1)
                        debug_msg=f"3d->2d @ decoder layer {lid}, non matches prop.: {num_non_match_prop}, second match prop.: {num_second_matches_prop}"
                        print(f"num second matches @ decoder layer {lid}: {num_second_matches}")
                        self.debug_logger.info(debug_msg)
                        print(debug_msg)

            # query: [B, Q, C]
            # sampling_locs: [B, Q, n_heads, n_levels, n_points, 2]
            # attn_weights: [B, Q, n_heads, n_levels, n_points]
            layer_out = layer(query, *args, reference_points=reference_points_2d_cam, 
                                orig_spatial_shapes=orig_spatial_shapes, num_cameras=6, 
                                num_second_matches=num_second_matches, second_matches_valid_idxs=second_matches_valid_idxs,
                                idx_with_second_match=idx_with_second_match,
                                **kwargs)
            query = layer_out[0]
            if self.mask_pred_target: ##
                sampling_locs, attn_weights = layer_out[1:3]
                # assert sampling_locs is not None and attn_weights is not None
                sampling_locs_all.append(sampling_locs)
                attn_weights_all.append(attn_weights)
            # hack implementation for iterative bounding box refinement
            assert bbox_embed is not None
            query_out=torch.nan_to_num(query)
            if bbox_embed is not None:
                assert ref_pts_unnormalized is not None, "box refinement needs reference points!"
                coord_offset = bbox_embed[lid](query_out) # [B, Q, 10]
                if self.use_sigmoid_on_attn_out:
                    if self._iter == 0: print("PETR_TRANSFORMER: using sigmoid on attn out")
                    coord_offset[..., :3] = F.sigmoid(coord_offset[..., :3])
                    coord_offset[..., :3] = denormalize_lidar(coord_offset[..., :3], self.pc_range)
                coord_pred = coord_offset
                assert ref_pts_unnormalized.shape[-1] == 3
                coord_pred[..., 0:3] = coord_pred[..., 0:3] + ref_pts_unnormalized[..., 0:3]
                if self.use_inv_sigmoid:
                    coord_pred[..., 0:3] = coord_pred[..., 0:3].sigmoid()
                    next_ref_pts = coord_pred[..., 0:3].detach().clone()
                    unnormalized_ref_pts = inverse_sigmoid(coord_pred[..., 0:3].detach().clone())
                else:
                    if self.debug and self._iter % 50 == 0:
                        prop_out_of_range = not_in_lidar_range(coord_pred[..., 0:3], self.pc_range).sum().item() / coord_pred.size(1)
                        print(f"coord prediction within decoder layer {lid} out of range: {prop_out_of_range}")
                    coord_pred[..., 0:3] = clamp_to_lidar_range(coord_pred[..., 0:3], self.pc_range)
                    unnormalized_ref_pts = coord_pred[..., 0:3].detach().clone()
                    next_ref_pts = normalize_lidar(coord_pred[..., 0:3].detach().clone(), self.pc_range)

                # ref_pts_unnormalized = coord_pred[..., 0:3].detach() # [B, Q, 3]

            if self.return_intermediate:
                intermediate_reference_points.append(unnormalized_ref_pts)
                if self.post_norm is not None:
                    intermediate.append(self.post_norm(query))
                else:
                    intermediate.append(query)
            reference_points = next_ref_pts

        self._iter += 1
        # raise Exception
        if self.mask_pred_target:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points), init_reference_point, \
                torch.stack(sampling_locs_all, dim=1), torch.stack(attn_weights_all, dim=1)
        else:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points), init_reference_point



@TRANSFORMER.register_module()
class PETRTemporalTransformer(BaseModule):
    """Implements the DETR transformer.
    Following the official DETR implementation, this module copy-paste
    from torch.nn.Transformer with modifications:
        * positional encodings are passed in MultiheadAttention
        * extra LN at the end of encoder is removed
        * decoder returns a stack of activations from all decoding layers
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        encoder (`mmcv.ConfigDict` | Dict): Config of
            TransformerEncoder. Defaults to None.
        decoder ((`mmcv.ConfigDict` | Dict)): Config of
            TransformerDecoder. Defaults to None
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Defaults to None.
    """

    def __init__(self, encoder=None, decoder=None, init_cfg=None, two_stage=False):
        super(PETRTemporalTransformer, self).__init__(init_cfg=init_cfg)
        if encoder is not None:
            self.encoder = build_transformer_layer_sequence(encoder)
        else:
            self.encoder = None
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.two_stage=two_stage

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        # custom init for attention
        for i in range(self.decoder.num_layers):
            for attn in self.decoder.layers[i].attentions:
                if isinstance(attn, CustomDeformAttn):
                    assert attn.is_init == False
                    attn.reset_parameters()
        
        self._is_init = True


    def forward(self, bbox_embed, memory, tgt, query_pos, attn_masks, pos_embed=None,
                temp_memory=None, temp_pos=None, mask=None, reference_points=None, 
                lidar2img=None, extrinsics=None, orig_spatial_shapes=None,
                flattened_spatial_shapes=None, flattened_level_start_index=None,
                img_metas=None, prev_exists=None):
        """
        Args:
            bbox_embed (ModuleList): StreamPETR's reg_branches 
                (note: need to pass this way because setting the same modules as attributes 
                in two separate modules causes error in the running code) 
            memory (_type_): [B, N, h0*w0+..., C] | [B, h0*N*w0+..., C]
            tgt (_type_): dn queries|current queries|prev queries; Tensor [B, Nq+num_propagated, 256]
            query_pos (_type_): query pos | prev query pos; Tensor [B, Nq+num_propagated, 256]
            pos_embed (_type_): 3d pos embeds; Tensor [B, N, h0*w0+..., C] | None if it is integrated in encoder
            attn_masks (_type_): temporal attention mask; [pad_size + Nq + num_propagated,  pad_size + Nq + memory_len]
            temp_memory (_type_, optional): rest of memory queue Tensor [B, memory_len - num_propagated, 256]
            temp_pos (_type_, optional): rest of query pos in queue Tensor [B, memory_len - num_propagated, 256]
            mask (_type_, optional): _description_. Defaults to None.
            reg_branch (_type_, optional): _description_. Defaults to None.
            reference_points (_type_, optional): current ref pts|prev ref pts Tensor [B, Nq+num_propagated, 3]
            lidar2img (_type_, optional): current frame's lidar2img [B, N, 4, 4]
            extrinsics (_type_, optional): current frame's lidar2cam [B, N, 4, 4]
            orig_spatial_shapes (_type_, optional): [n_levels, 2]
            flattened_spatial_shapes: (H, W*num_cams)
            flattened_level_start_index: [n_levels]
        """
        # ! WARNING: NEED TO ENSURE THAT REF_PTS[NUM_PROPAGATED:] are aligned with current frame
        assert self.two_stage
        if tgt is None:
            tgt = torch.zeros_like(query_pos)

        # out_dec: [num_layers, B, Q, C]
        # out_ref_pts: [num_layers, B, Q, 3]
        # init_ref_pts: [B, Q, 3]
        outs_decoder = self.decoder(
            bbox_embed=bbox_embed,
            query=tgt,
            key=memory if not self.two_stage else None,
            value=memory,
            key_pos=pos_embed, # pos_embed is None if two_stage
            query_pos=query_pos,
            temp_memory=temp_memory,
            temp_pos=temp_pos,
            key_padding_mask=mask,
            attn_masks=[attn_masks, None],
            reference_points=reference_points, 
            lidar2img=lidar2img, 
            extrinsics=extrinsics, 
            orig_spatial_shapes=orig_spatial_shapes,
            flattened_spatial_shapes=flattened_spatial_shapes, 
            flattened_level_start_index=flattened_level_start_index,
            img_metas=img_metas,
            prev_exists=prev_exists,
            )
        return outs_decoder


@TRANSFORMER_LAYER.register_module()
class PETRTemporalDecoderLayer(BaseModule):
    """Base `TransformerLayer` for vision transformer.

    It can be built from `mmcv.ConfigDict` and support more flexible
    customization, for example, using any number of `FFN or LN ` and
    use different kinds of `attention` by specifying a list of `ConfigDict`
    named `attn_cfgs`. It is worth mentioning that it supports `prenorm`
    when you specifying `norm` as the first element of `operation_order`.
    More details about the `prenorm`: `On Layer Normalization in the
    Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for `self_attention` or `cross_attention` modules,
            The order of the configs in the list should be consistent with
            corresponding attentions in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config. Default: None.
        ffn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for FFN, The order of the configs in the list should be
            consistent with corresponding ffn in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Support `prenorm` when you specifying first element as `norm`.
            Default：None.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape
            of (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
    """

    def __init__(self,
                 attn_cfgs=None,
                 ffn_cfgs=dict(
                     type='FFN',
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 batch_first=False,
                 with_cp=True,
                 skip_first_frame_self_attn=False,
                 mask_pred_target=False,
                 **kwargs):

        deprecated_args = dict(
            feedforward_channels='feedforward_channels',
            ffn_dropout='ffn_drop',
            ffn_num_fcs='num_fcs')
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                warnings.warn(
                    f'The arguments `{ori_name}` in BaseTransformerLayer '
                    f'has been deprecated, now you should set `{new_name}` '
                    f'and other FFN related arguments '
                    f'to a dict named `ffn_cfgs`. ', DeprecationWarning)
                ffn_cfgs[new_name] = kwargs[ori_name]

        super().__init__(init_cfg)

        self.batch_first = batch_first

        assert set(operation_order) & {
            'self_attn', 'norm', 'ffn', 'cross_attn'} == \
            set(operation_order), f'The operation_order of' \
            f' {self.__class__.__name__} should ' \
            f'contains all four operation type ' \
            f"{['self_attn', 'norm', 'ffn', 'cross_attn']}"

        num_attn = operation_order.count('self_attn') + operation_order.count(
            'cross_attn')
        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), f'The length ' \
                f'of attn_cfg {num_attn} is ' \
                f'not consistent with the number of attention' \
                f'in operation_order {operation_order}.'

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == 'norm'
        self.attentions = ModuleList()

        index = 0
        for operation_name in operation_order:
            if operation_name in ['self_attn', 'cross_attn']:
                if 'batch_first' in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]['batch_first']
                else:
                    attn_cfgs[index]['batch_first'] = self.batch_first
                attention = build_attention(attn_cfgs[index])
                # Some custom attentions used as `self_attn`
                # or `cross_attn` can have different behavior.
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

        self.ffns = ModuleList()
        num_ffns = operation_order.count('ffn')
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = ConfigDict(ffn_cfgs)
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns
        for ffn_index in range(num_ffns):
            if 'embed_dims' not in ffn_cfgs[ffn_index]:
                ffn_cfgs[ffn_index]['embed_dims'] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]['embed_dims'] == self.embed_dims
            self.ffns.append(
                build_feedforward_network(ffn_cfgs[ffn_index],
                                          dict(type='FFN')))

        self.norms = ModuleList()
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])

        self.use_checkpoint = with_cp

        self.skip_first_frame_self_attn=skip_first_frame_self_attn
        self.mask_pred_target=mask_pred_target

    def _forward(self,
                query,
                key=None,
                value=None,
                key_pos=None,
                query_pos=None,
                temp_memory=None,
                temp_pos=None,
                key_padding_mask=None,
                attn_masks=None,
                reference_points=None,
                query_key_padding_mask=None,
                orig_spatial_shapes=None,
                flattened_spatial_shapes=None,
                flattened_level_start_index=None,
                num_cameras=6,
                num_second_matches=None,
                second_matches_valid_idxs=None,
                idx_with_second_match=None,
                prev_exists=None, # Tensor [B]
                ):
        """
        Args:
        NOTE: Q = padding + current queries + propagated
            query (_type_): current queries|prev queries; Tensor [B, Q, 256]
            key (_type_, optional):[B, N, h0*w0+..., C]
            value (_type_, optional): [B, N, h0*w0+..., C]
            query_pos (_type_, optional): query pos | prev query pos; Tensor [B, Q, 256]
            key_pos (_type_, optional): 3d pos embeds; Tensor [B, N, h0*w0+..., C]
            temp_memory (_type_, optional): rest of memory queue Tensor [B, memory_len - num_propagated, 256]
            temp_pos (_type_, optional): rest of query pos in queue Tensor [B, memory_len - num_propagated, 256]
            attn_masks (_type_, optional): _description_. Defaults to None.
            query_key_padding_mask (_type_, optional): _description_. Defaults to None.
            key_padding_mask (_type_, optional): _description_. Defaults to None.
            reference_points (_type_, optional): projected cam reference points. Tensor [B, Q, n_levels, 2]
            spatial_shapes (_type_, optional): [n_levels, 2]
            num_cameras (int, optional): _description_. Defaults to 6.
        """
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        if prev_exists is not None:
            valid_mask = (prev_exists.long() | (not self.skip_first_frame_self_attn)).bool()
        else:
            valid_mask=torch.arange(0, query.size(0))

        for layer in self.operation_order:
            if layer == 'self_attn':
                if temp_memory is not None:
                    # [B, padding + curr. queries + num_propagated + rest of memory, 256]
                    temp_key = temp_value = torch.cat([query, temp_memory], dim=1)
                    # [B, padding + curr. queries + num_propagated + rest of memory, 256]
                    temp_pos = torch.cat([query_pos, temp_pos], dim=1)
                else:
                    temp_key = temp_value = query
                    temp_pos = query_pos

                # [B, Q, 256]
                query_out = self.attentions[attn_index](
                    query,
                    key=temp_key,
                    value=temp_value,
                    identity=identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=temp_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    inp_batch_first=True,
                    )
                
                if not valid_mask.all():
                    invalid_mask = ~valid_mask
                    query_out[invalid_mask] = query[invalid_mask]
                
                query=query_out

                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                cross_attn_out = self.attentions[attn_index](
                    query,
                    value,
                    key=key, # None if two_stage
                    identity=identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos, # None if two stage
                    reference_points=reference_points,
                    spatial_shapes=orig_spatial_shapes,
                    flattened_spatial_shapes=flattened_spatial_shapes,
                    flattened_lvl_start_index=flattened_level_start_index,
                    num_cameras=num_cameras,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    return_query_only=not self.mask_pred_target,
                    num_second_matches=num_second_matches,
                    second_matches_valid_idxs=second_matches_valid_idxs,
                    idx_with_second_match=idx_with_second_match,
                    )
                if self.mask_pred_target:
                    assert isinstance(cross_attn_out, (list, tuple))
                    # query: [B, Q, C]
                    # sampling_locs: [B, Q, n_heads, n_levels, n_points, 2]
                    # attn_weights: [B, Q, n_heads, n_levels, n_points]
                    query, sampling_locs, attn_weights = cross_attn_out
                else:
                    # print("warning: not returning sampling locs and attn weights")
                    query = cross_attn_out
                    sampling_locs, attn_weights = None, None

                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query, sampling_locs, attn_weights

    def forward(self, 
                query,
                key=None,
                value=None,
                key_pos=None,
                query_pos=None,
                temp_memory=None,
                temp_pos=None,
                key_padding_mask=None,
                attn_masks=None,
                reference_points=None,
                query_key_padding_mask=None,
                orig_spatial_shapes=None,
                flattened_spatial_shapes=None,
                flattened_level_start_index=None,
                num_cameras=6,
                num_second_matches=None,
                second_matches_valid_idxs=None,
                idx_with_second_match=None,
                prev_exists=None,
                ):
        """Forward function for `TransformerCoder`.
        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if self.use_checkpoint and self.training:
            x = cp.checkpoint(
                self._forward, 
                query,
                key,
                value,
                key_pos,
                query_pos,
                temp_memory,
                temp_pos,
                key_padding_mask,
                attn_masks,
                reference_points,
                query_key_padding_mask,
                orig_spatial_shapes,
                flattened_spatial_shapes,
                flattened_level_start_index,
                num_cameras,
                num_second_matches,
                second_matches_valid_idxs,
                idx_with_second_match,
                prev_exists,
                )
        else:
            x = self._forward(
            query,
            key,
            value,
            key_pos,
            query_pos,
            temp_memory,
            temp_pos,
            key_padding_mask,
            attn_masks,
            reference_points,
            query_key_padding_mask,
            orig_spatial_shapes,
            flattened_spatial_shapes,
            flattened_level_start_index,
            num_cameras,
            num_second_matches,
            second_matches_valid_idxs,
            idx_with_second_match,
            prev_exists,
        )
        return x

