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
import torch
import os
import copy
from torch import nn
from torch.nn.init import xavier_uniform_, constant_

from mmcv.runner import force_fp32, auto_fp16, get_dist_info
from mmcv.cnn import Linear
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.cnn.bricks.plugin import build_plugin_layer
from mmdet.models.utils import NormedLinear
from mmdet.models import DETECTORS
from mmdet3d.models.builder import build_head
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.models.utils.misc import locations
from projects.mmdet3d_plugin.models.utils.positional_encoding import posemb2d_from_spatial_shapes
from ..utils.lidar_utils import normalize_lidar
from mmdet.models.utils.transformer import inverse_sigmoid

@DETECTORS.register_module()
class Petr3D(MVXTwoStageDetector):
    """Petr3D."""

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 num_frame_head_grads=2,
                 num_frame_backbone_grads=2,
                 num_frame_losses=2,
                 stride=16,
                 strides=[4, 8, 16, 32],
                 position_level=0,
                 aux_2d_only=True,
                 single_test=False,
                 pretrained=None,
                 embed_dims=256,
                 use_xy_embed=True,
                 use_cam_embed=False,
                 use_lvl_embed=False,
                 mlvl_feats_format=0,
                 encoder=None,
                 num_cameras=6,
                 pc_range=None,
                 extra_module=None,
                 debug_args=None,
                 focal_layers=[0,1,2,3],
                 depth_net=None,
                 depth_pred_position=0,
                 calc_depth_pred_loss=False,
                 share_pos_3d_encoding_method=False,
                 use_spatial_alignment=False,
                 pos_embed3d=None,):
        if depth_net is not None:
            depth_net['depth_pred_position'] = depth_pred_position
            depth_net['mlvl_feats_format'] = mlvl_feats_format
            depth_net['n_levels'] = len(strides)

        if encoder is not None:
            encoder['pc_range']=pc_range
            encoder['depth_pred_position']=depth_pred_position
            encoder['pred_ref_pts_depth']=depth_net is not None
            
            if depth_net is not None and depth_pred_position == 1:
                encoder['depth_net'] = depth_net
        if pts_bbox_head is not None:
            pts_bbox_head['pc_range'] = pc_range


        super(Petr3D, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.iter_num = 0

        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.prev_scene_token = None
        self.num_frame_head_grads = num_frame_head_grads
        self.num_frame_backbone_grads = num_frame_backbone_grads
        self.num_frame_losses = num_frame_losses
        self.single_test = single_test
        self.stride = stride # NOTE: this is used for generation location for single feature level
        self.position_level = position_level
        self.aux_2d_only = aux_2d_only
        self.test_flag = False

        ## new params
        self.strides=strides
        # self.embed_dims=embed_dims
        if self.with_pts_bbox:
            self.embed_dims = self.pts_bbox_head.embed_dims
        else:
            self.embed_dims=embed_dims
        self.n_levels=len(strides)
        self.use_xy_embed=use_xy_embed
        self.use_cam_embed = use_cam_embed
        if self.use_cam_embed:
            self.cam_embed = torch.nn.Parameter(torch.Tensor(num_cameras, self.embed_dims))
        self.use_lvl_embed = use_lvl_embed
        if self.use_lvl_embed:
            self.lvl_embed = torch.nn.Parameter(torch.Tensor(
                self.n_levels, self.embed_dims))
        self.mlvl_feats_format = mlvl_feats_format
        self.use_encoder=encoder is not None
        if encoder is not None:
            if train_cfg is not None and 'encoder' in train_cfg:
                encoder['train_cfg'] = train_cfg['encoder']
            self.encoder=build_transformer_layer_sequence(encoder)
        
        self.num_cameras=num_cameras

        self.pc_range=nn.Parameter(torch.tensor(pc_range), requires_grad=False)

        if debug_args is not None:
            self.debug = debug_args['debug']
            if self.use_encoder:
                self.encoder.debug = self.debug
            if self.with_pts_bbox:
                from projects.mmdet3d_plugin.models.transformer.petr_transformer import PETRTemporalTransformer, PETRTransformerDecoder, PETRTemporalDecoderLayer
                self.pts_bbox_head.debug = self.debug
                for m in self.pts_bbox_head.modules():
                    if isinstance(m, (PETRTemporalTransformer, PETRTransformerDecoder, PETRTemporalDecoderLayer)):
                        m.debug = self.debug

            if self.debug:
                debug_log_filename = debug_args['log_file']
                dirname = os.path.dirname(debug_log_filename)
                if not os.path.exists(dirname):
                    os.mkdir(dirname)
                import logging
                logger = logging.getLogger("debug_log")
                logger.setLevel(logging.INFO)
                fh = logging.FileHandler(debug_log_filename)
                fh.setLevel(logging.INFO)
                logger.addHandler(fh)
                self.debug_log_filename=debug_log_filename
                self.debug_logger = logger
                self.debug_collect_stats = debug_args['collect_stats'] if 'collect_stats' in debug_args else False
                if self.use_encoder:
                    self.encoder.debug_logger = self.debug_logger
                    self.encoder.debug_log_filename = self.debug_log_filename
                    self.encoder.debug_collect_stats=self.debug_collect_stats
                if self.with_pts_bbox:
                    self.pts_bbox_head.debug_logger = self.debug_logger
                    self.pts_bbox_head.debug_log_filename = self.debug_log_filename
                    self.pts_bbox_head.debug_collect_stats=self.debug_collect_stats
                    for m in self.pts_bbox_head.modules():
                        if isinstance(m, (PETRTemporalTransformer, PETRTransformerDecoder, PETRTemporalDecoderLayer)):
                            m.debug_logger = self.debug_logger
                            m.debug_log_filename = self.debug_log_filename
                            m.debug_collect_stats=self.debug_collect_stats
        
        ## extra modules
        self.has_extra_mod = extra_module is not None
        if self.has_extra_mod:
            self.extra_module = build_head(extra_module)

        self.focal_layers=focal_layers

        self.depth_pred_position=depth_pred_position
        if depth_net is not None and depth_pred_position==0:
            self.depth_net=build_plugin_layer(depth_net)[1]
        else:
            self.depth_net=None
        # if depth_pred_position != 0:
        #     raise NotImplementedError(f"depth_pred_position {depth_pred_position} not currently supported")
        self.calc_depth_pred_loss=calc_depth_pred_loss
        if calc_depth_pred_loss: 
            assert self.depth_net is not None or (self.use_encoder and self.encoder.depth_net is not None)
        self.share_pos_3d_encoding_method=share_pos_3d_encoding_method
        if share_pos_3d_encoding_method:
            assert self.use_encoder and self.encoder.encode_3dpos_method in ["mln"]
        self.use_spatial_alignment=use_spatial_alignment
        self.use_pos_embed3d=pos_embed3d is not None
        if self.use_pos_embed3d:
            self.pos_embed3d = build_plugin_layer(pos_embed3d)[1]

    def init_weights(self):
        if self.has_extra_mod:
            self.extra_module.init_weights()
        if self.use_encoder:
            self.encoder.init_weights()
            print("initialized encoder")
        if self.with_pts_bbox:
            self.pts_bbox_head.init_weights()
            print("initialized pts bbox head")
        if self.depth_net is not None:
            self.depth_net.init_weights()
            print("initialized depth net")
        ## use init_cfg
        assert not self.img_backbone.is_init
        self.img_backbone.init_weights()
        assert not self.img_neck.is_init
        self.img_neck.init_weights()

        already_init = ['pts_bbox_head', 'img_backbone', 'img_neck', 'mask_predictor', 'encoder', 'depth_net']

        for name, param in self.named_parameters():
            if param.requires_grad and param.dim() > 1:
                module_name = name.split('.')[0]
                if module_name not in already_init:
                    print(name)
                    # assert module_name in to_init, f"got {module_name} which is not in {to_init}"
                    param_type = name.split('.')[-1]
                    # if param_type == 'bias':
                    #     constant_(param, 0.)
                    # else:
                    xavier_uniform_(param)

        self._is_init = True


    def extract_img_feat(self, img, len_queue=1, training_mode=False):
        """Extract features of images.
        train: img: [B, Q, N, C, H, W]
        test: img: [B, N, C, H, W]
        """
        B = img.size(0)
        n_cams, C, H, W = img.shape[-4:]
        # B, q_len, n_cams, C, H, W = img.size()
        if img is not None:
            if img.dim() == 6:
                img = img.flatten(1, 2)
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_() # [BN, C, H, W]
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                # NOTE: during training, grid_mask's forward just returns img unaltered
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        if self.training or training_mode:
            # ([B, len_queue, num_cams, 256, H/8, W/8]
            #  [B, len_queue, num_cams, 256, H/16, W/16],
            #  [B, len_queue, num_cams, 256, H/32, W/32],
            #  [B, len_queue, num_cams, 256, H/64, W/64])
            img_feats_neck_out = [feat.view(B, len_queue, n_cams, *feat.shape[-3:]) for feat in img_feats]
        else:
            # ([B, num_cams, 256, H/8, W/8]
            #  [B, num_cams, 256, H/16, W/16],
            #  [B, num_cams, 256, H/32, W/32],
            #  [B, num_cams, 256, H/64, W/64])
            img_feats_neck_out = [feat.view(B, n_cams, *feat.shape[-3:]) for feat in img_feats]
        return img_feats_neck_out


    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, T, training_mode=False):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, T, training_mode) # []
        return img_feats

    def obtain_history_memory(self,
                            gt_bboxes_3d=None,
                            gt_labels_3d=None,
                            gt_bboxes=None,
                            gt_labels=None,
                            img_metas=None,
                            centers2d=None,
                            depths=None,
                            gt_bboxes_ignore=None,
                            **data):
        losses = dict()
        T = data['img'].size(1)
        num_nograd_frames = T - self.num_frame_head_grads
        num_grad_losses = T - self.num_frame_losses
        for i in range(T):
            requires_grad = False
            return_losses = False
            data_t = dict()
            for key in data:
                if key in ['img_feats']: continue
                data_t[key] = data[key][:, i] 

            data_t['img_feats'] = [d[:, i] for d in data['img_feats']]
            if i >= num_nograd_frames:
                requires_grad = True
            if i >= num_grad_losses:
                return_losses = True
            loss = self.forward_pts_train(gt_bboxes_3d[i],
                                        gt_labels_3d[i], gt_bboxes[i],
                                        gt_labels[i], img_metas[i], centers2d[i], depths[i], 
                                        requires_grad=requires_grad,return_losses=return_losses,**data_t)
            if loss is not None:
                for key, value in loss.items():
                    losses['frame_'+str(i)+"_"+key] = value
        return losses


    def prepare_location(self, img_metas, **data):
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        bs, n = data['img_feats'].shape[:2]
        x = data['img_feats'].flatten(0, 1)
        location = locations(x, self.stride, pad_h, pad_w)[None].repeat(bs*n, 1, 1, 1)
        return location
    
    def prepare_location_multiscale(self, img_metas, spatial_shapes, mode=None, **data):
        mode = self.mlvl_feats_format if mode is None else mode
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        B, N, *_ = data['img_feats'][0].shape
        device = data['img_feats_flatten'].device

        locations_all=[]
        locations_flattened=[]
        for lvl, (H_i, W_i) in enumerate(spatial_shapes):
            lvl_stride = self.strides[lvl]
            shifts_x = (torch.arange(0, lvl_stride*W_i, step=lvl_stride, dtype=torch.float32, 
                                     device=device) + lvl_stride//2) / pad_w
            shifts_y = (torch.arange(0, H_i*lvl_stride, step=lvl_stride, dtype=torch.float32,
                                    device=device) + lvl_stride//2) / pad_h
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            locations = torch.stack((shift_x, shift_y), dim=1).reshape(H_i, W_i, 2) # [H_i,W_i, 2]
            if mode == 0:
                locations = locations[None, None].repeat(B, N, 1, 1, 1) # [B, N, H_i, W_i, 2]
                locations_flat = locations.flatten(2, 3) # [B, N, H_i*W_i, 2]
            elif mode == 1:
                locations = locations[None, :, None].repeat(B, 1, N, 1, 1) # [B, H_i, N, W_i, 2] 
                locations_flat = locations.flatten(1, 3) # [B, H_i*N*W_i, 2]
            locations_flattened.append(locations_flat)
            locations_all.append(locations)

        if mode == 0:
            locations_flattened = torch.cat(locations_flattened, dim=2) # [B, N, h_0*w_0+..., 2]
        elif mode == 1:
            locations_flattened = torch.cat(locations_flattened, dim=1) # [B, h0*N*w0+..., 2]
        # locations_flattened = torch.cat(locations_flattened, dim=2) 
        # locations = torch.cat(locations_all, dim=0) # [h_0*w_0+..., 2]
        # locations = locations[None].repeat(B, N, 1, 1) # [B, N, h_0*w_0+..., 2]
        return locations_all, locations_flattened

    def forward_roi_head(self, location, **data):
        if (self.aux_2d_only and not self.training) or not self.with_img_roi_head:
            return {'topk_indexes':None}
        else:
            src = [data['img_feats'][i] for i in self.focal_layers]
            locations = [location[i] for i in self.focal_layers]
            outs_roi = self.img_roi_head(locations, src=src, **data)
            return outs_roi
        
    def forward_pts_bbox(self, locations_flattened, img_metas, 
                         enc_pred_dict=None, out_memory=None,
                         **kwargs):
        if self.use_encoder:
            # ref_pts_init: [B, Q, 3] in lidar R range
            # query_dec_init: [B, Q, ]
            decoder_inps = self.prepare_decoder_inputs(enc_pred_dict, out_memory)
            memory=out_memory
        else:
            decoder_inps = dict()
            memory = kwargs['data']['img_feats_flatten']
        # reference_points_dec_init: [B, Q, 3] in [0,1] range normalized R range
        reference_points_dec_init = decoder_inps.get("reference_points_dec_init")
        query_dec_init = decoder_inps.get("query_dec_init")
        init_pseudo_ref_pts = decoder_inps.get("ref_pts_propagated")
        pos_dec_init = decoder_inps.get("pos_dec_init")
        pos3d_dec_init = decoder_inps.get("pos_3d_dec_init")
        remaining_outs_dec_init=decoder_inps.get("remaining_outs_dec_init")
        outs = self.pts_bbox_head(memory, locations_flattened, img_metas, 
                    query_init=query_dec_init, reference_points=reference_points_dec_init,
                    init_pseudo_ref_pts=init_pseudo_ref_pts, pos_init=pos_dec_init,
                    pos_3d_init=pos3d_dec_init,
                    pos_encoding_method=self.encoder.pos3d_encoding if self.share_pos_3d_encoding_method else None,
                    remaining_outs_init=remaining_outs_dec_init,
                    **kwargs)
        return outs
        
    def prepare_decoder_inputs(self, enc_preds_dict, enc_out_memory):
        """
        Args:
            enc_preds_dict (_type_): _description_
            enc_out_memory (_type_): [B, h0*N*w0+..., C]
        """
        num_dec_queries = self.pts_bbox_head.num_query
        # [B, p, num_classes]
        enc_out_cls = enc_preds_dict['cls_scores_enc']
        enc_out_coord = enc_preds_dict['bbox_preds_enc']
        enc_out_repr_cls = enc_out_cls.max(-1).values # [B, p]
        num_topk = num_dec_queries
        if self.pts_bbox_head.init_pseudo_ref_pts_from_encoder_out:
            num_topk += self.pts_bbox_head.num_propagated
        topq_inds = torch.topk(enc_out_repr_cls, num_dec_queries, dim=1)[1] # [B, q]
        if self.pts_bbox_head.init_pseudo_ref_pts_from_encoder_out:
            propagated_inds=topq_inds[:, -self.pts_bbox_head.num_propagated:]
            ref_pts_propagated = torch.gather(enc_out_coord, 1,
                                    propagated_inds.unsqueeze(-1).repeat(1, 1, enc_out_coord.size(-1)))
            topq_inds = topq_inds[:, :num_dec_queries]
        # [B, q, 10]
        topq_coords = torch.gather(enc_out_coord, 1, 
                                   topq_inds.unsqueeze(-1).repeat(1,1,enc_out_coord.size(-1)))
        reference_points_dec_init = topq_coords.detach()[..., :3] # [B, q, 3]
        reference_points_dec_init = normalize_lidar(reference_points_dec_init, self.pc_range)
        query_dec_init = torch.gather(enc_out_memory, 1, 
                                      topq_inds.unsqueeze(-1).repeat(1, 1, enc_out_memory.size(-1)))
        outs = dict(reference_points_dec_init=reference_points_dec_init, query_dec_init=query_dec_init)
        if self.pts_bbox_head.refine_all:
            remaining_outs_dec_init = topq_coords.detach()[..., 3:] # [B, q, 7]
            outs.update(dict(remaining_outs_dec_init=remaining_outs_dec_init))

        # outs = [reference_points_dec_init, query_dec_init]
        if self.encoder.propagate_pos:
            propagated_pos = enc_preds_dict['pos_enc']
            pos_dec_init = torch.gather(propagated_pos, 1, 
                                        topq_inds.unsqueeze(-1).repeat(1, 1, propagated_pos.size(-1)))
            # outs += [pos_dec_init]
            outs.update(dict(pos_dec_init=pos_dec_init))
        if self.encoder.propagate_3d_pos:
            propagated_3d_pos = enc_preds_dict['pos_3d_enc']
            pos_3d_dec_init = torch.gather(propagated_3d_pos, 1, 
                                           topq_inds.unsqueeze(-1).repeat(1, 1, propagated_3d_pos.size(-1)))
            outs.update(dict(pos_3d_dec_init=pos_3d_dec_init))
        if self.pts_bbox_head.init_pseudo_ref_pts_from_encoder_out:
            # outs += [ref_pts_propagated]
            outs.update(dict(ref_pts_propagated=ref_pts_propagated))
        return outs
        
    def prepare_mlvl_feats(self, img_feats):
        cur_mlvl_feats = img_feats
        cur_mlvl_feats_flatten = []
        pos_all = []
        spatial_shapes=[]

        if self.mlvl_feats_format == 0:
            raise Exception("mlvl feats format 0 not guaranteed to be correct")
            token_dim = 2
            for lvl, lvl_feat in enumerate(cur_mlvl_feats):
                B, num_cams, embed_dims, H_i, W_i = lvl_feat.shape
                lvl_feat = lvl_feat.flatten(-2).transpose(-2, -1) # [B, N, C, H_i*W_i] -> [B, N, H_i*W_i, C]
                pos = torch.zeros_like(lvl_feat) # [B, N, H_i*W_i, C] 
                if self.use_cam_embed:
                    pos = pos + self.cam_embed[None, :, None, :] # [N, C] -> [1, N, 1, C]
                if self.use_lvl_embed:
                    pos = pos + self.lvl_embed[None, lvl:lvl+1,None,:] # [n_levels, C] -> [1, 1, 1, C]
                spatial_shapes.append((H_i, W_i))
                cur_mlvl_feats_flatten.append(lvl_feat)
                pos_all.append(pos)

        elif self.mlvl_feats_format == 1:
            token_dim=1
            for lvl, lvl_feat in enumerate(cur_mlvl_feats):
                B, num_cams, embed_dims, H_i, W_i = lvl_feat.shape
                lvl_feat = lvl_feat.permute(0, 3, 1, 4, 2) # [B, N, C, H_i, W_i] -> [B, H_i, N, W_i, C]
                if self.use_xy_embed:
                    # if self.iter_num == 0: print("use xy embed")
                    # [B, H_i, N*W_i, C]
                    pos = posemb2d_from_spatial_shapes((H_i, W_i*num_cams), lvl_feat.device, B, normalize=True)
                    pos = pos.flatten(1, 2) # [B, H_i*N*W_i, C]
                else:
                    pos = lvl_feat.new_zeros([B, H_i*num_cams*W_i, embed_dims])
                # NOTE: when flattening images from all 6 cams, the (x,y) should implicitly encodes camera too
                # if self.use_cam_embed:
                #     pos = pos + self.cam_embed[None, None, :, None, :] # [1, 1, N, 1, C]

                if self.use_lvl_embed:
                    # [B, H_i*N*W_i, C]
                    pos = pos + self.lvl_embed[lvl:lvl+1, None, :] # [1, 1, C]
                spatial_shapes.append((H_i, W_i))
                lvl_feat = lvl_feat.flatten(1,3) # [B, H_i*N*W_i, C]
                assert pos.shape == lvl_feat.shape
                assert list(lvl_feat.shape) == [B, H_i*num_cams*W_i, embed_dims], f"got: {lvl_feat.shape}"
                cur_mlvl_feats_flatten.append(lvl_feat)
                pos_all.append(pos)
        cur_mlvl_feats_flatten = torch.cat(cur_mlvl_feats_flatten, token_dim) # [B, N, h0*w0+..., C] | [B, h0*N*w0+..., C]
        pos_flatten = torch.cat(pos_all, token_dim) # [B, N, h_0*w_0+..., C] | [B, h0*N*w0+..., C]
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=cur_mlvl_feats[0].device) # [n_levels, 2]
        flattened_spatial_shapes = spatial_shapes.detach().clone()
        flattened_spatial_shapes[:, 1] = flattened_spatial_shapes[:, 1]*self.num_cameras
        flattened_level_start_index = torch.cat((flattened_spatial_shapes.new_zeros(
            (1,)), flattened_spatial_shapes.prod(1).cumsum(0)[:-1]))
        return cur_mlvl_feats_flatten, pos_flatten, spatial_shapes, flattened_spatial_shapes, flattened_level_start_index

    def forward_pts_train(self,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_bboxes,
                          gt_labels,
                          img_metas,
                          centers2d,
                          depths,
                          requires_grad=True,
                          return_losses=False,
                          **data):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """

        # multi-level img processing
        # data['img_feats'] = list of len 4
        #   [B, 6, 256, H_0, W_0],
        #   [B, 6, 256, H_1, W_1],
        #   [B, 6, 256, H_2, W_2],
        #   [B, 6, 256, H_3, W_3]
        B, num_cams = data['img_feats'][0].shape[:2]
        if self.depth_net is not None and self.depth_pred_position == 0:
            # [B, h0*N*w0+..., 1]
            depth_pred = self.depth_net(data['img_feats'], return_flattened=True)
        else:
            depth_pred=None

        cur_mlvl_feats_flatten, pos_flatten, spatial_shapes, \
            flattened_spatial_shapes, flattened_level_start_index=self.prepare_mlvl_feats(data['img_feats'])
        data['img_feats_flatten'] = cur_mlvl_feats_flatten
        
        # location: List[num_levels], with each elem being a Tensor[B, H_i, N, W_i, 2]
        # locations_flattened: Tensor[B, h0*N*w0+..., 2]
        location, locations_flattened = self.prepare_location_multiscale(img_metas, spatial_shapes, **data)
        if self.with_img_roi_head:
            location_roihead = [loc.permute(0, 2, 1, 3, 4) for loc in location] # [B, N, H_i, W_i, 2]
            for idx, (H_i, W_i) in enumerate(spatial_shapes):
                assert list(location_roihead[idx].shape) == [B, num_cams, H_i,W_i, 2], f"got {location_roihead[idx].shape}"
        assert list(locations_flattened.shape)== list(cur_mlvl_feats_flatten.shape[:-1]) + [2]
        if not requires_grad:
            self.eval()
            with torch.no_grad():
                outs = self.pts_bbox_head(location, img_metas, None, **data)
            self.train()
        else:
            if self.with_img_roi_head:
                # assert self.mlvl_feats_format == 0, f"img roi head is only compatible with mlvl_feats_format:0"
                outs_roi = self.forward_roi_head(location_roihead, **data)
                topk_indexes = outs_roi['topk_indexes']
            else:
                topk_indexes=None

            if self.has_extra_mod:
                outs_extra = self.extra_module(img_metas, spatial_shapes, flattened_spatial_shapes, flattened_level_start_index,
                                               **data)

            if self.use_encoder:
                assert hasattr(self, "encoder")
                assert self.mlvl_feats_format == 1, f"encoder is only compatible with mlvl_feats_format:1"
                # enc_pred_dict = {
                #   "cls_scores_enc": Tensor[B, p, num_classes] # NOTE: p != num_query,
                #   "bbox_preds_enc": Tensor[B, p, num_classes] # NOTE: [..., :3] are in lidar range unnormalized
                #   "sparse_token_num": int
                #   "src_mask_prediction": Tensor[B, h0*N*w0+...]
                # }
                # output: encoder enhanced memory Tensor[B, h0*N*w0+..., C]
                enc_pred_dict, out_memory = self.encoder(data['img_feats_flatten'], spatial_shapes, 
                                                flattened_spatial_shapes, flattened_level_start_index, 
                                                pos_flatten, img_metas, locations_flattened, depth_pred=depth_pred,
                                                **data)
            else:
                enc_pred_dict, out_memory=None, None
                
            if self.with_pts_bbox:
                # outs = {
                #   - all_cls_scores: [n_dec_layers, B, Q, num_classes]
                #   - all_bbox_preds: [n_dec_layers, B, Q, 10]
                #   - dn_mask_dict
                #   - all_sampling_locs_dec: [B, num_dec_layers, Q, n_heads, n_levels, n_points, 2]
                #   - all_attn_weights_dec: [B, num_dec_layers, Q, n_heads, n_levels, n_points]
                # }
                outs = self.forward_pts_bbox(locations_flattened, img_metas, enc_pred_dict=enc_pred_dict,
                                      out_memory=out_memory, topk_indexes=topk_indexes, orig_spatial_shapes=spatial_shapes,
                                      flattened_spatial_shapes=flattened_spatial_shapes,
                                      flattened_level_start_index=flattened_level_start_index, **data)
            
        if return_losses:
            losses=dict()
            if self.with_pts_bbox:
                loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
                losses = self.pts_bbox_head.loss(*loss_inputs)
            if self.has_extra_mod:
                loss_inputs = [outs_extra,cur_mlvl_feats_flatten.device]
                losses_extra = self.extra_module.loss(*loss_inputs)
                losses.update(losses_extra)
                # if self.iter_num % 10 == 0:
                #     print(losses_extra)
            if self.calc_depth_pred_loss:
                if self.depth_pred_position == 0:
                    depthnet = self.depth_net
                elif self.depth_pred_position == 1:
                    depthnet = self.encoder.depth_net
                assert depthnet is not None
                losses_depth = depthnet.loss(gt_bboxes_3d, enc_pred_dict, 
                                             data['lidar2img'], data['extrinsics'], spatial_shapes)
                losses.update(losses_depth)
            if self.use_encoder:
                loss_enc_inputs = [gt_bboxes_3d, gt_labels_3d, enc_pred_dict, 
                                   flattened_spatial_shapes, flattened_level_start_index]
                # if self.encoder.pred_ref_pts_loss:
                #     loss_enc_inputs += [data['lidar2img'], data['extrinsics'], img_metas,spatial_shapes]
                losses.update(self.encoder.loss(*loss_enc_inputs))
                if self.encoder.use_mask_predictor:
                    loss_mask = self.encoder.mask_predictor.loss(
                        mask_prediction=enc_pred_dict['src_mask_prediction'],
                        sampling_locations=outs['all_sampling_locs_dec'],
                        attn_weights=outs['all_attn_weights_dec'],
                        flattened_spatial_shapes=flattened_spatial_shapes,
                        flattened_level_start_index=flattened_level_start_index,
                        sparse_token_nums=enc_pred_dict['sparse_token_num']
                    )
                    losses.update(loss_mask)
            if self.with_img_roi_head:
                loss2d_inputs = [gt_bboxes, gt_labels, centers2d, depths, outs_roi, img_metas, self.focal_layers]
                losses2d = self.img_roi_head.loss(*loss2d_inputs)
                losses.update(losses2d) 
            return losses
        else:
            return None


    @force_fp32(apply_to=('img'))
    def forward(self, return_loss=True, **data):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            # print("train iter: ", self.iter_num)
            for key in ['gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'img_metas']:
                data[key] = list(zip(*data[key]))
            if self.iter_num % 50 == 0:
                rank, world_size=get_dist_info()
                print(f"GPU {rank} memory allocated: {torch.cuda.memory_allocated(rank)/1e9} GB")
                print(f"GPU {rank} memory reserved: {torch.cuda.memory_reserved(rank)/1e9} GB")
                print(f"GPU {rank} max memory reserved: {torch.cuda.max_memory_reserved(rank)/1e9} GB")
            self.iter_num += 1
            out= self.forward_train(**data)
        else:
            out= self.forward_test(**data)

        return out

    def forward_train(self,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      depths=None,
                      centers2d=None,
                      **data):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        if self.iter_num < 10:
            print(f"img shape: {data['img'].shape}")
        if self.test_flag: #for interval evaluation
            self.pts_bbox_head.reset_memory()
            self.test_flag = False
        T = data['img'].size(1)

        prev_img = data['img'][:, :-self.num_frame_backbone_grads]
        rec_img = data['img'][:, -self.num_frame_backbone_grads:]
        # ([B, len_queue, num_cams, 256, H/8, W/8]
        #  [B, len_queue, num_cams, 256, H/16, W/16],
        #  [B, len_queue, num_cams, 256, H/32, W/32],
        #  [B, len_queue, num_cams, 256, H/64, W/64])
        rec_img_feats = self.extract_feat(rec_img, self.num_frame_backbone_grads)

        if T-self.num_frame_backbone_grads > 0:
            self.eval()
            with torch.no_grad():
                prev_img_feats = self.extract_feat(prev_img, T-self.num_frame_backbone_grads, True)
            self.train()
            data['img_feats'] = torch.cat([prev_img_feats, rec_img_feats], dim=1)
        else:
            data['img_feats'] = rec_img_feats

        losses = self.obtain_history_memory(gt_bboxes_3d,
                        gt_labels_3d, gt_bboxes,
                        gt_labels, img_metas, centers2d, depths, gt_bboxes_ignore, **data)
        return losses
  
  
    def forward_test(self, img_metas, rescale, **data):
        """
        Args:
            img_metas (_type_): _description_
            rescale (_type_): _description_
            data: dict(
                - img_metas: List[1] with element List[1] with element img metas dict
                - img: List[1] with element Tensor[B=1, N=6, C=3, H=256, W=704]
                - lidar2img: List[1] with element List[1] with element [N=6, 4, 4]
                ...
            )

        Raises:
            TypeError: _description_

        Returns:
            _type_: _description_
        """
        self.test_flag = True
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        for key in data:
            if key != 'img':
                data[key] = data[key][0][0].unsqueeze(0)
            else:
                data[key] = data[key][0]
        return self.simple_test(img_metas[0], **data)

    def simple_test_pts(self, img_metas, **data):
        """Test function of point cloud branch.
            - img_metas: List[B] with element img metas dict
        """
        # data['img_feats'] = [
        #       ([B, num_cams, 256, H/8, W/8]
        #        [B, num_cams, 256, H/16, W/16],
        #        [B, num_cams, 256, H/32, W/32],
        #        [B, num_cams, 256, H/64, W/64])]
        if self.depth_net is not None and self.depth_pred_position == 0:
            # [B, h0*N*w0+..., 1]
            depth_pred = self.depth_net(data['img_feats'], return_flattened=True)
        else:
            depth_pred=None
        cur_mlvl_feats_flatten, pos_flatten, spatial_shapes, \
            flattened_spatial_shapes, flattened_level_start_index=self.prepare_mlvl_feats(data['img_feats'])
        data['img_feats_flatten'] = cur_mlvl_feats_flatten

        location, locations_flattened = self.prepare_location_multiscale(img_metas, spatial_shapes, **data)
        if self.with_img_roi_head:
            # location_roihead = [loc.permute(0, 2, 1, 3, 4) for loc in location] # [B, N, H_i, W_i, 2]
            outs_roi = self.forward_roi_head(None, **data) # location doesn't matter cus in test it just returns None
            topk_indexes = outs_roi['topk_indexes']
        else:
            topk_indexes=None

        if self.has_extra_mod:
            raise NotImplementedError()
        
        if self.use_encoder:
            enc_pred_dict, out_memory = self.encoder(data['img_feats_flatten'], spatial_shapes, 
                                                flattened_spatial_shapes, flattened_level_start_index, 
                                                pos_flatten, img_metas, locations_flattened, depth_pred=depth_pred,
                                                ensure_no_print=True,
                                                **data)
        else:
            enc_pred_dict, out_memory=None, None
        if self.with_pts_bbox:
            B = data['img_feats_flatten'].size(0)
            if img_metas[0]['scene_token'] != self.prev_scene_token: # different scene, reset memory
                self.prev_scene_token = img_metas[0]['scene_token']
                data['prev_exists'] = data['img'].new_zeros(B) # zero Tensor[B]
                self.pts_bbox_head.reset_memory()
            else:
                data['prev_exists'] = data['img'].new_ones(B) # one Tensor[B]
            outs = self.forward_pts_bbox(locations_flattened, img_metas, enc_pred_dict=enc_pred_dict,
                                    out_memory=out_memory, topk_indexes=topk_indexes, orig_spatial_shapes=spatial_shapes,
                                    flattened_spatial_shapes=flattened_spatial_shapes,
                                    flattened_level_start_index=flattened_level_start_index, **data)

            bbox_list = self.pts_bbox_head.get_bboxes(
                outs, img_metas)
            bbox_results = [
                bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in bbox_list
            ]
        return bbox_results
    
    def simple_test(self, img_metas, **data):
        """Test function without augmentaiton.
            img_metas: List[B] with element img metas dict
        """
        # data['img'] = [B, N, C, H, W]
        # data['img_feats'] = [
        #       ([B, num_cams, 256, H/8, W/8]
        #        [B, num_cams, 256, H/16, W/16],
        #        [B, num_cams, 256, H/32, W/32],
        #        [B, num_cams, 256, H/64, W/64])]
        # data['img_feats'] = self.extract_img_feat(data['img'], 1)
        data['img_feats'] = self.extract_feat(data['img'], 1, training_mode=False)

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(
            img_metas, **data)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    