import time
import torch
from torch import nn
from torch.nn.init import xavier_uniform_, constant_

from mmcv.runner import force_fp32, auto_fp16, get_dist_info
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.cnn.bricks.plugin import build_plugin_layer
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.models.utils.misc import locations
from projects.mmdet3d_plugin.models.utils.positional_encoding import posemb2d_from_spatial_shapes
from projects.mmdet3d_plugin.constants import *
from ..utils.lidar_utils import normalize_lidar
from ..utils.projections import Projections
from ..utils.debug import *

@DETECTORS.register_module()
class Petr3D(MVXTwoStageDetector):
    """Petr3D."""

    def __init__(self,
                 use_grid_mask=False,
                 img_backbone=None,
                 img_neck=None,
                 pts_bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 num_frame_backbone_grads=2,
                 strides=[4, 8, 16, 32],
                 pretrained=None,
                 embed_dims=256,
                 use_xy_embed=True,
                 use_cam_embed=False,
                 use_lvl_embed=False,
                 mlvl_feats_format=0,
                 encoder=None,
                 num_cameras=6,
                 pc_range=None,
                 limit_3d_pts_to_pc_range=False,
                 ## depth
                 depth_net=None,
                 depth_pred_position=0,
                 calc_depth_pred_loss=True,
                 ## debug
                 debug_args=None,
                 ):
        if depth_net is not None:
            depth_net['depth_pred_position'] = depth_pred_position
            depth_net['mlvl_feats_format'] = mlvl_feats_format
            depth_net['n_levels'] = len(strides)

        if encoder is not None:
            encoder['pc_range']=pc_range
            encoder['depth_pred_position']=depth_pred_position
            
            if depth_net is not None and depth_pred_position == DEPTH_PRED_IN_ENCODER:
                encoder['depth_net'] = depth_net
        if pts_bbox_head is not None:
            pts_bbox_head['pc_range'] = pc_range

        super(Petr3D, self).__init__(img_backbone=img_backbone, img_neck=img_neck, 
                                     pts_bbox_head=pts_bbox_head, train_cfg=train_cfg, 
                                     test_cfg=test_cfg, pretrained=pretrained)

        self.use_grid_mask = use_grid_mask
        if self.use_grid_mask:
            self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.prev_scene_token = None
        self.num_frame_backbone_grads = num_frame_backbone_grads
        self.test_flag = False
        self.limit_3d_pts_to_pc_range=limit_3d_pts_to_pc_range

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
        self.depth_pred_position=depth_pred_position
        if depth_net is not None and depth_pred_position==DEPTH_PRED_BEFORE_ENCODER:
            self.depth_net=build_plugin_layer(depth_net)[1]
        else:
            self.depth_net=None
        self.calc_depth_pred_loss= calc_depth_pred_loss and \
            (self.depth_net is not None or (self.use_encoder and self.encoder.depth_net is not None))

        self.cached_locations=None
        
        ## debug init
        self.debug = Debug(**debug_args)
        debug_modules=debug_args.get('debug_modules', [])
        if len(debug_modules) > 0:
            for m in self.modules():
                if any([type(m).__name__ == debug_mod for debug_mod in debug_modules]):
                    m.debug=self.debug

    def init_weights(self):
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
        img_feats = self.extract_img_feat(img, T, training_mode) # []
        return img_feats

    def obtain_history_memory(self,
                            gt_bboxes_3d=None,
                            gt_labels_3d=None,
                            img_metas=None,
                            **data):
        i=0
        data_t = dict()
        for key in data:
            if key in ['img_feats']: continue
            data_t[key] = data[key][:, i] 

        data_t['img_feats'] = [d[:, i] for d in data['img_feats']]
        loss = self.forward_pts_train(gt_bboxes_3d[i], gt_labels_3d[i], img_metas[i], 
                                      return_losses=True, **data_t)
        return loss

    
    def prepare_location_multiscale(self, img_metas, spatial_shapes, **data):
        if self.cached_locations is not None:
            return self.cached_locations.clone()
        
        assert self.mlvl_feats_format == MLVL_HNW
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        B, N, *_ = data['img_feats'][0].shape
        device = data['img_feats_flatten'].device

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
            locations = locations[None, :, None].repeat(B, 1, N, 1, 1) # [B, H_i, N, W_i, 2] 
            locations_flat = locations.flatten(1, 3) # [B, H_i*N*W_i, 2]
            locations_flattened.append(locations_flat)

        locations_flattened = torch.cat(locations_flattened, dim=1) # [B, h0*N*w0+..., 2]
        self.cached_locations = locations_flattened.clone()
        return locations_flattened
        
    def forward_pts_bbox(self, img_metas, enc_pred_dict=None, out_memory=None, dn_known_bboxs=None, 
                         dn_known_labels=None, **kwargs):
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
        outs = self.pts_bbox_head(
            memory, img_metas, query_init=query_dec_init, reference_points=reference_points_dec_init, 
            dn_known_bboxs=dn_known_bboxs, dn_known_labels=dn_known_labels, **kwargs)
        return outs
        
    def prepare_decoder_inputs(self, enc_preds_dict, enc_out_memory):
        num_dec_queries = self.pts_bbox_head.num_query
        # [B, p, num_classes]
        enc_out_cls = enc_preds_dict['cls_scores_enc']
        enc_out_coord = enc_preds_dict['bbox_preds_enc']
        enc_out_repr_cls = enc_out_cls.max(-1).values # [B, p]
        
        topq_inds = torch.topk(enc_out_repr_cls, num_dec_queries, dim=1)[1] # [B, q]
        
        # [B, q, 10]
        topq_coords = torch.gather(enc_out_coord, 1, 
                                   topq_inds.unsqueeze(-1).repeat(1,1,enc_out_coord.size(-1)))
        reference_points_dec_init = topq_coords.detach()[..., :3] # [B, q, 3]
        if self.limit_3d_pts_to_pc_range:
            reference_points_dec_init = normalize_lidar(reference_points_dec_init, self.pc_range)
        query_dec_init = torch.gather(enc_out_memory, 1, 
                                      topq_inds.unsqueeze(-1).repeat(1, 1, enc_out_memory.size(-1)))
        outs = dict(reference_points_dec_init=reference_points_dec_init, query_dec_init=query_dec_init)
        return outs
        
    def prepare_mlvl_feats(self, img_feats):
        cur_mlvl_feats = img_feats
        cur_mlvl_feats_flatten = []
        pos_all = []
        spatial_shapes=[]

        if self.mlvl_feats_format == 0:
            raise Exception("mlvl feats format 0 not guaranteed to be correct")

        elif self.mlvl_feats_format == MLVL_HNW:
            token_dim=1
            for lvl, lvl_feat in enumerate(cur_mlvl_feats):
                B, num_cams, embed_dims, H_i, W_i = lvl_feat.shape
                lvl_feat = lvl_feat.permute(0, 3, 1, 4, 2) # [B, N, C, H_i, W_i] -> [B, H_i, N, W_i, C]
                if self.use_xy_embed:
                    # [B, H_i, N*W_i, C]
                    pos = posemb2d_from_spatial_shapes((H_i, W_i*num_cams), lvl_feat.device, B, normalize=True)
                    pos = pos.flatten(1, 2) # [B, H_i*N*W_i, C]
                else:
                    pos = lvl_feat.new_zeros([B, H_i*num_cams*W_i, embed_dims])
                if self.use_lvl_embed:
                    # [B, H_i*N*W_i, C]
                    pos = pos + self.lvl_embed[lvl:lvl+1, None, :] # [1, 1, C]
                spatial_shapes.append((H_i, W_i))
                lvl_feat = lvl_feat.flatten(1,3) # [B, H_i*N*W_i, C]
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
                          img_metas,
                          return_losses=False,
                          **data):

        # multi-level img processing
        # data['img_feats'] = list of len 4
        #   [B, 6, 256, H_0, W_0],
        #   [B, 6, 256, H_1, W_1],
        #   [B, 6, 256, H_2, W_2],
        #   [B, 6, 256, H_3, W_3]
        if self.depth_net is not None and self.depth_pred_position == DEPTH_PRED_BEFORE_ENCODER:
            # [B, h0*N*w0+..., 1]
            depth_pred = self.depth_net(data['img_feats'], focal=data['focal'], return_flattened=True)
        else:
            depth_pred=None

        cur_mlvl_feats_flatten, pos_flatten, spatial_shapes, \
            flattened_spatial_shapes, flattened_level_start_index=self.prepare_mlvl_feats(data['img_feats'])
        data['img_feats_flatten'] = cur_mlvl_feats_flatten
        
        # locations_flattened: Tensor[B, h0*N*w0+..., 2]
        locations_flattened = self.prepare_location_multiscale(img_metas, spatial_shapes, **data)
        
        if self.use_encoder:
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
            outs = self.forward_pts_bbox(img_metas, enc_pred_dict=enc_pred_dict,
                                    out_memory=out_memory, orig_spatial_shapes=spatial_shapes,
                                    flattened_spatial_shapes=flattened_spatial_shapes,
                                    flattened_level_start_index=flattened_level_start_index, 
                                    dn_known_bboxs=gt_bboxes_3d, dn_known_labels=gt_labels_3d,
                                    **data)
        
        if return_losses:
            losses=dict()
            if self.with_pts_bbox:
                loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
                losses = self.pts_bbox_head.loss(*loss_inputs)
            
            if self.calc_depth_pred_loss and (self.depth_net is not None or self.encoder.depth_net is not None):
                if self.depth_pred_position == DEPTH_PRED_BEFORE_ENCODER:
                    depthnet = self.depth_net
                elif self.depth_pred_position == DEPTH_PRED_IN_ENCODER:
                    depthnet = self.encoder.depth_net
                losses_depth = depthnet.loss(gt_bboxes_3d, enc_pred_dict, 
                                             data['lidar2img'], data['extrinsics'], spatial_shapes)
                losses.update(losses_depth)
            if self.use_encoder:
                loss_enc_inputs = [gt_bboxes_3d, gt_labels_3d, enc_pred_dict, 
                                   flattened_spatial_shapes, flattened_level_start_index]
                losses.update(self.encoder.loss(*loss_enc_inputs))
                if self.encoder.use_mask_predictor:
                    raise NotImplementedError()
            
            return losses
        else:
            return None


    @force_fp32(apply_to=('img'))
    def forward(self, return_loss=True, **data):
        # with open("./experiments/data_forward_clean.pkl", "wb") as f:
        #     pickle.dump(data, f)
        # time.sleep(5)
        # raise Exception()
        if return_loss:
            for key in ['gt_bboxes_3d', 'gt_labels_3d', 'img_metas']:
                data[key] = list(zip(*data[key]))
            rank, _=get_dist_info()
            
            if do_debug_process(self, repeating=True):
                print(f"GPU {rank} memory allocated: {torch.cuda.memory_allocated(rank)/1e9} GB")
                print(f"GPU {rank} memory reserved: {torch.cuda.memory_reserved(rank)/1e9} GB")
                print(f"GPU {rank} max memory reserved: {torch.cuda.max_memory_reserved(rank)/1e9} GB")
            self.debug.iter += 1
            out= self.forward_train(**data)
        else:
            out= self.forward_test(**data)

        return out

    def forward_train(self,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      **data):
        if do_debug_process(self):
            print(f"img shape: {data['img'].shape}")
            assert Projections.IMG_SIZE[0] == data['img'].size(-1), \
                f"Projection's expected input W to be {Projections.IMG_SIZE[0]} but got: {data['img'].size(-1)}"
            assert Projections.IMG_SIZE[1] == data['img'].size(-2), \
                f"Projection's expected input H to be {Projections.IMG_SIZE[1]} but got {data['img'].size(-2)}"
        if self.test_flag: #for interval evaluation
            self.pts_bbox_head.reset_memory()
            self.test_flag = False

        rec_img = data['img'][:, -self.num_frame_backbone_grads:]
        # ([B, len_queue, num_cams, 256, H/8, W/8]
        #  [B, len_queue, num_cams, 256, H/16, W/16],
        #  [B, len_queue, num_cams, 256, H/32, W/32],
        #  [B, len_queue, num_cams, 256, H/64, W/64])
        rec_img_feats = self.extract_feat(rec_img, self.num_frame_backbone_grads)

        data['img_feats'] = rec_img_feats

        losses = self.obtain_history_memory(gt_bboxes_3d,
                        gt_labels_3d, img_metas, **data)
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

    