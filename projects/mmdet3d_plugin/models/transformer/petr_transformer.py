from copy import deepcopy
import warnings
import torch
# import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import Linear, bias_init_with_prob
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
import torch.utils.checkpoint as cp

from mmcv.runner import auto_fp16

from ..utils.projections import Projections, convert_3d_to_2d_global_cam_ref_pts, convert_3d_to_mult_2d_global_cam_ref_pts
from ..utils.lidar_utils import denormalize_lidar, normalize_lidar, clamp_to_lidar_range, not_in_lidar_range
from projects.mmdet3d_plugin.attentions.custom_deform_attn import CustomDeformAttn
from projects.mmdet3d_plugin.models.utils.debug import *

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

        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)[0]


        return identity + self.dropout_layer(self.proj_drop(out))

@TRANSFORMER.register_module()
class PETRTemporalTransformer(BaseModule):

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


    def forward(self, anchor_refinements, memory, tgt, query_pos, attn_masks, pos_embed=None,
                temp_memory=None, temp_pos=None, mask=None, reference_points=None, 
                lidar2img=None, extrinsics=None, orig_spatial_shapes=None,
                flattened_spatial_shapes=None, flattened_level_start_index=None,
                img_metas=None):
        
        assert self.two_stage

        # out_dec: [num_layers, B, Q, C]
        # out_ref_pts: [num_layers, B, Q, 3]
        # init_ref_pts: [B, Q, 3]
        outs_decoder = self.decoder(
            anchor_refinements=anchor_refinements,
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
            )
        return outs_decoder

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
                 num_classes=10,
                 anchor_dims=10,
                 num_reg_fcs=2,
                 limit_3d_pts_to_pc_range=False,
                 **kwargs):
        super(PETRTransformerDecoder, self).__init__(*args, **kwargs)
        self.limit_3d_pts_to_pc_range=limit_3d_pts_to_pc_range
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

        # cls_branch = []
        # for _ in range(num_reg_fcs):
        #     cls_branch.append(Linear(self.embed_dims, self.embed_dims))
        #     cls_branch.append(nn.LayerNorm(self.embed_dims))
        #     cls_branch.append(nn.ReLU(inplace=True))
        # cls_branch.append(Linear(self.embed_dims, num_classes))
        # cls_branch = nn.Sequential(*cls_branch)

        # reg_branch = []
        # for _ in range(num_reg_fcs):
        #     reg_branch.append(Linear(self.embed_dims, self.embed_dims))
        #     reg_branch.append(nn.ReLU())
        # reg_branch.append(Linear(self.embed_dims, anchor_dims))
        # reg_branch = nn.Sequential(*reg_branch)

        # self.cls_branches = nn.ModuleList(
        #     [deepcopy(cls_branch) for _ in range(self.num_layers)])
        # self.reg_branches = nn.ModuleList(
        #     [deepcopy(reg_branch) for _ in range(self.num_layers)])

    # def init_weights(self):
    #     bias_init = bias_init_with_prob(0.01)
    #     for m in self.cls_branches:
    #         nn.init.constant_(m[-1].bias, bias_init)
        

    def forward(self, anchor_refinements, query, *args, reference_points=None, lidar2img=None, extrinsics=None, 
                orig_spatial_shapes=None, img_metas=None, num_cameras=6, **kwargs):
        if not self.return_intermediate:
            x = super().forward(query, *args, reference_points=None, **kwargs)
            if self.post_norm:
                x = self.post_norm(x)[None]
            return x

        intermediate = []
        intermediate_reference_points = []

        cam_transformations = dict(lidar2img=lidar2img, lidar2cam=extrinsics)

        for lid, layer in enumerate(self.layers):
            if self.limit_3d_pts_to_pc_range:
                if self.use_inv_sigmoid:
                    ref_pts_unnormalized = inverse_sigmoid(reference_points.clone()) # R: lidar space
                else:
                    ref_pts_unnormalized = denormalize_lidar(reference_points.clone(), self.pc_range) # R:lidar space
            else:
                ref_pts_unnormalized = reference_points

            if lid == 0:
                init_reference_point =ref_pts_unnormalized.clone()
            # [B, Q, 2]
            with torch.no_grad():
                if self.ref_pts_mode == "single":
                    reference_points_2d_cam, _ = convert_3d_to_2d_global_cam_ref_pts(cam_transformations,
                                                                        ref_pts_unnormalized, orig_spatial_shapes,
                                                                        img_metas, ret_num_non_matches=True)
                    
                    
                    num_second_matches, second_matches_valid_idxs, idx_with_second_match = [None]*3
                elif self.ref_pts_mode == "multiple":
                    if do_debug_process(self): print("DECODER: USING MULTIPLE REF PTS")
                    ref_pts_mult_outs = convert_3d_to_mult_2d_global_cam_ref_pts(cam_transformations,
                                                                    ref_pts_unnormalized, orig_spatial_shapes,
                                                                    img_metas, 
                                                                    ret_num_non_matches=with_debug(self))
                    reference_points_2d_cam, num_second_matches, second_matches_valid_idxs, idx_with_second_match = \
                        ref_pts_mult_outs[:4]
                    if do_debug_process(self, repeating=True):
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
            query = layer(query, *args, reference_points=reference_points_2d_cam, 
                                orig_spatial_shapes=orig_spatial_shapes, num_cameras=6, 
                                num_second_matches=num_second_matches, second_matches_valid_idxs=second_matches_valid_idxs,
                                idx_with_second_match=idx_with_second_match,
                                **kwargs)
            # query_out = self.post_norm(query) if self.post_norm else query
            query_out=torch.nan_to_num(query)
            
            # cls_pred = self.cls_branches[lid](query_out)
            # TODO: integrate query pos
            reg_out, _ = anchor_refinements[lid](query_out, ref_pts_unnormalized, query_pos=None, return_cls=False) # [B, Q, 10]
            reference_points = unnormalized_ref_pts = reg_out[..., :3].detach().clone()
            # if self.use_sigmoid_on_attn_out:
            #     assert self.limit_3d_pts_to_pc_range, "sigmoid on attn output assumes output is limited to pc range"
            #     if do_debug_process(self): print("PETR_TRANSFORMER: using sigmoid on attn out")
            #     coord_offset[..., :3] = F.sigmoid(coord_offset[..., :3])
            #     coord_offset[..., :3] = denormalize_lidar(coord_offset[..., :3], self.pc_range)

            # coord_pred = coord_offset
            # coord_pred[..., 0:3] = coord_pred[..., 0:3] + ref_pts_unnormalized[..., 0:3]

            # if self.use_inv_sigmoid:
            #     coord_pred[..., 0:3] = coord_pred[..., 0:3].sigmoid()
            #     reference_points = coord_pred[..., 0:3].detach().clone()
            # elif self.limit_3d_pts_to_pc_range:
            #     if do_debug_process(self, repeating=True, interval=500):
            #         prop_out_of_range = not_in_lidar_range(coord_pred[..., 0:3], self.pc_range).sum().item() / coord_pred.size(1)
            #         print(f"coord prediction within decoder layer {lid} out of range: {prop_out_of_range}")
            #     coord_pred[..., 0:3] = clamp_to_lidar_range(coord_pred[..., 0:3], self.pc_range)
            #     unnormalized_ref_pts = coord_pred[..., 0:3].detach().clone()
            #     reference_points = normalize_lidar(coord_pred[..., 0:3].detach().clone(), self.pc_range)
            # else:
            #     reference_points = unnormalized_ref_pts = coord_pred[..., 0:3].detach().clone()

            intermediate_reference_points.append(unnormalized_ref_pts)
            if self.post_norm is not None:
                intermediate.append(self.post_norm(query))
            else:
                intermediate.append(query)

        return torch.stack(intermediate), torch.stack(intermediate_reference_points), init_reference_point

@TRANSFORMER_LAYER.register_module()
class PETRTemporalDecoderLayer(BaseModule):
    
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
                ):
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
                query = self.attentions[attn_index](
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
            
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
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
                    return_query_only=True,
                    num_second_matches=num_second_matches,
                    second_matches_valid_idxs=second_matches_valid_idxs,
                    idx_with_second_match=idx_with_second_match,
                    )
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query

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
        )
        return x

