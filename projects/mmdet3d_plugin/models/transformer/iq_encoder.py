import torch
import os
import pickle
import copy
import numpy as np
from torch import nn
import torch.nn.functional as F
from mmcv.cnn import Linear, xavier_init, build_norm_layer
from mmcv.cnn.bricks.transformer import (TransformerLayerSequence, BaseTransformerLayer, 
                                         TRANSFORMER_LAYER_SEQUENCE, TRANSFORMER_LAYER)
from mmcv.cnn.bricks.plugin import build_plugin_layer
from mmdet.core import (build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models import HEADS, build_loss
from mmdet.models.utils.transformer import inverse_sigmoid
from projects.mmdet3d_plugin.models.utils.positional_encoding import pos2posemb3d, pos2posemb1d, nerf_positional_encoding, posemb2d
from projects.mmdet3d_plugin.models.utils.misc import MLN, SELayer_Linear
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox, clamp_to_rot_range
from projects.mmdet3d_plugin.attentions.custom_deform_attn import CustomDeformAttn
from ..utils.projections import convert_3d_to_2d_global_cam_ref_pts, Projections, project_to_matching_2point5d_cam_points
from ..utils.mask_predictor import MaskPredictor
from ..utils.lidar_utils import normalize_lidar, denormalize_lidar, clamp_to_lidar_range, not_in_lidar_range
from ..utils.misc import flatten_mlvl, groupby_agg_mean
from ..utils.debug import *
from ..detectors.depthnet import DepthNet

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class IQTransformerEncoder(TransformerLayerSequence):
    # modified from SparseDETR
    def __init__(self, *args, return_intermediate=False, aux_heads=False, aux_layers=[],
                 code_size=10, mlvl_feats_formats=1, pc_range=None, learn_ref_pts_type="linear", 
                 learn_ref_pts_num_layers=None, use_spatial_alignment=True, spatial_alignment_all_memory=True, 
                 pos_embed3d=None, encode_query_with_ego_pos=False, encode_query_pos_with_ego_pos=False, 
                 use_inv_sigmoid=False,use_sigmoid_on_attn_out=False, num_classes=10, match_with_velo=False, 
                 code_weights=None, match_costs=None, sync_cls_avg_factor=False, train_cfg=None, 
                 loss_cls=None, loss_bbox=None, loss_iou=None, 
                 ## sparsification
                 mask_predictor=None, mask_pred_before_encoder=False, mask_pred_target=['decoder'], sparse_rho=1.0, 
                 process_backbone_mem=True, process_encoder_mem=True, post_norm_cfg=dict(type="LN"),
                 ## depth
                 depth_start = 1, position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0], depth_pred_position=0, 
                 depth_net=None, pred_ref_pts_depth=False, div_depth_loss_by_target_count=True, 
                 reference_point_generator=None, encode_ref_pts_depth_into_query_pos=False, ref_pts_depth_encoding_method=None,
                 encode_3dpos_method="add", propagate_pos=False, propagate_3d_pos=False, rot_post_process=None, wlhclamp=None,
                 **kwargs):
        self._iter = 0
        self.fp16_enabled=False
        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']
            self.assigner = build_assigner(assigner)
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        else:
            print("ENCODER: !! train_cfg is None, IN TEST MODE !!")
            transformerlayers=kwargs.get('transformerlayers')
            assert transformerlayers is not None
            for attn_cfg in transformerlayers['attn_cfgs']:
                if attn_cfg['type'] == 'CustomDeformAttn':
                    attn_cfg['test_mode'] = True
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor

        super(IQTransformerEncoder, self).__init__(*args, **kwargs)
        self.pc_range = nn.Parameter(torch.tensor(pc_range), requires_grad=False)
        self.return_intermediate = return_intermediate
        if mlvl_feats_formats != 1:
            raise NotImplementedError("encoder only supports mlvl feats format 1")
        self.mlvl_feats_formats=mlvl_feats_formats
        self.use_pos_embed3d=pos_embed3d is not None
        self.use_spatial_alignment=use_spatial_alignment
        self.spatial_alignment_all_memory=spatial_alignment_all_memory
        if self.use_spatial_alignment:
            self.spatial_alignment = MLN(8)

        self.position_range=nn.Parameter(torch.tensor(position_range), requires_grad=False)
        self.depth_start=depth_start # 1
        self.depth_range=self.position_range[3] - self.depth_start
        if self.use_pos_embed3d:
            self.featurized_pe = SELayer_Linear(self.embed_dims)
        if self.use_pos_embed3d:
            self.pos_embed3d = build_plugin_layer(pos_embed3d)[1]

        self.encode_query_with_ego_pos=encode_query_with_ego_pos 
        self.encode_query_pos_with_ego_pos=encode_query_pos_with_ego_pos
        if encode_query_with_ego_pos or encode_query_pos_with_ego_pos:
            self.rec_ego_motion=None
        if encode_query_with_ego_pos:
            self.ego_pose_memory = MLN(180)
        if encode_query_pos_with_ego_pos:
            self.ego_pose_pe=MLN(180)

        self.code_size=code_size

        self.learn_ref_pts_type=learn_ref_pts_type.lower()
        assert self.learn_ref_pts_type in ['embedding', 'linear', 'mlp', 'anchor']
        if self.learn_ref_pts_type in ["mlp"]:
            assert learn_ref_pts_num_layers is not None
            self.learn_ref_pts_num_layers = learn_ref_pts_num_layers
        if self.learn_ref_pts_type == "anchor":
            assert reference_point_generator is not None
            reference_point_generator['mlvl_feats_format'] = self.mlvl_feats_formats
            self.reference_points = build_plugin_layer(reference_point_generator)[1]
        self.created_reference_points=False
        
        self.use_mask_predictor = mask_predictor is not None
        if self.use_mask_predictor:
            self.mask_predictor = build_plugin_layer(mask_predictor)[1]

        self.sparse_rho = sparse_rho

        self.mask_pred_before_encoder=mask_pred_before_encoder
        self.mask_pred_target=mask_pred_target

        self.src_embed = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        self.process_backbone_mem=process_backbone_mem
        self.process_encoder_mem=process_encoder_mem
        
        if self.use_mask_predictor and (process_backbone_mem or process_encoder_mem):
            self.enc_output = nn.Linear(self.embed_dims, self.embed_dims)
            self.enc_output_norm = nn.LayerNorm(self.embed_dims)
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg, self.embed_dims)[1]
        else:
            self.post_norm = None

        self.use_sigmoid_on_attn_out=use_sigmoid_on_attn_out
        
        self.encode_ref_pts_depth_into_query_pos=encode_ref_pts_depth_into_query_pos
        if encode_ref_pts_depth_into_query_pos:
            self.ref_pts_depth_encoding_method = ref_pts_depth_encoding_method.lower()
            assert self.ref_pts_depth_encoding_method in ["mln", "linear"]
            if self.ref_pts_depth_encoding_method == "mln":
                self.query_pos_2p5d_ref_pts_depth = MLN(self.embed_dims, f_dim=self.embed_dims)
            elif self.ref_pts_depth_encoding_method == "linear":
                self.query_pos_2p5d_ref_pts_depth = nn.Sequential(
                    nn.Linear(self.embed_dims, self.embed_dims),
                    nn.LayerNorm(self.embed_dims)
                )
        self.pred_ref_pts_depth=pred_ref_pts_depth
        self.depth_pred_position=depth_pred_position
        if depth_net is not None and self.depth_pred_position == 1:
            self.depth_net=build_plugin_layer(depth_net)[1]
        else:
            self.depth_net=None
        
        ## loss cfg
        self.num_classes=num_classes
        self.match_with_velo=match_with_velo
        self.code_weights=nn.Parameter(torch.tensor(code_weights), requires_grad=False)
        self.match_costs=nn.Parameter(torch.tensor(code_weights), requires_grad=False) if match_costs is None else match_costs
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)
        # self.loss_depth=build_loss(loss_depth) if loss_depth is not None else None

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.div_depth_loss_by_target_count=div_depth_loss_by_target_count

        self.use_inv_sigmoid=use_inv_sigmoid

        self.encode_3dpos_method=encode_3dpos_method.lower()
        assert self.encode_3dpos_method in ["add", "mln"]
        if self.encode_3dpos_method == "mln":
            self.pos3d_encoding = MLN(self.embed_dims)
        self.propagate_pos=propagate_pos
        self.propagate_3d_pos=propagate_3d_pos
        self.rot_post_process=rot_post_process
        self.wlhclamp=wlhclamp

        ## pred branches
        cls_branch = []
        for _ in range(2):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.num_classes))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(2):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)
        self.cls_branches = nn.ModuleList(
            [copy.deepcopy(fc_cls) for _ in range(self.num_layers)])
        self.reg_branches = nn.ModuleList(
            [copy.deepcopy(reg_branch) for _ in range(self.num_layers)])
        
        if self.num_layers > 1:
            raise NotImplementedError("right now only using 1 encoder layer")
    
    def init_weights(self):
        # print("iq_encoder init weights")
        for m in self.modules():
            # print(m)
            if isinstance(m, CustomDeformAttn):
                assert m.is_init == False
                m.reset_parameters()
            elif isinstance(m, (MaskPredictor, DepthNet)):
                m.init_weights()
            if hasattr(m, 'weight') and m.weight is not None and \
                m.weight.requires_grad and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        
        self._is_init = True


    def create_reference_points_method(self, device, n_sparse_tokens=None):
        assert not self.created_reference_points
        # assert not hasattr(self, "reference_points") or \
        #     (self.learn_ref_pts_type == "anchor" and not self.reference_points.already_init)
        if self.learn_ref_pts_type == "embedding":
            self.reference_points = nn.Embedding(n_sparse_tokens, 3)
            ## ref pts init weights
            # TODO: uniformly distribute in range [0,1]
            nn.init.uniform_(self.reference_points.weight.data, 0, 1)

            # ! WARNING: THIS IS NOT IN RANGE [0,1]
        elif self.learn_ref_pts_type == "linear":
            self.reference_points = nn.Sequential(nn.Linear(self.embed_dims, 3), 
                                                  nn.Sigmoid())
            # ! TODO: INTEGRATE REF PTS INIT HERE
        elif self.learn_ref_pts_type == "mlp":
            ref_pts_branch = []
            for _ in range(self.learn_ref_pts_num_layers):
                ref_pts_branch.append(Linear(self.embed_dims, self.embed_dims))
                ref_pts_branch.append(nn.ReLU())
            ref_pts_branch.append(Linear(self.embed_dims, 3))
            ref_pts_branch.append(nn.Sigmoid())
            self.reference_points = nn.Sequential(*ref_pts_branch)
        
        # NOTE: for "anchor" type, it is initialized in init
        elif self.learn_ref_pts_type == "anchor":
            assert hasattr(self, "reference_points")
            self.reference_points.init_coords_depth(device)

        if self.learn_ref_pts_type != "anchor":
            self.reference_points.to(device)

        self.created_reference_points = True

    def get_reference_points(self, query, orig_spatial_shapes, depths=None, n_tokens=None, top_rho_inds=None):
        # depths: [B, h0*N*w0+..., 1] unnormalized depths
        assert hasattr(self, "reference_points")
        if self.learn_ref_pts_type == "embedding":
            # TODO: either clip results to [0,1] or [pc_min, pc_max]
            reference_points = self.reference_points.weight # [p, 3]
            reference_points = reference_points.unsqueeze(0).expand(query.size(0), -1, -1)
        elif self.learn_ref_pts_type == "linear":
            reference_points = self.reference_points(query) # [B, p, 3]
        elif self.learn_ref_pts_type == "mlp":
            reference_points = self.reference_points(query)
            if self._iter % 50 == 0 and self.debug:
                assert hasattr(self, "debug_log_file") is not None
                with torch.no_grad():
                    ref_pts_cpu = reference_points.detach().cpu().numpy()
                    self.debug_logger.info(f"reference_points min: {reference_points.min()}, max: {reference_points.max()}, "
                        f"25%={np.percentile(ref_pts_cpu, 0.25)} | 50%={np.percentile(ref_pts_cpu, 0.5)} | "
                        f"75%={np.percentile(ref_pts_cpu, 0.75)}")
        elif self.learn_ref_pts_type == "anchor":
            # if "learnable" depths, it is expected to be unnormalized
            if depths is not None: self.reference_points.set_coord_depths(depths, n_tokens=n_tokens, top_rho_inds=top_rho_inds)
            # if self._iter == 0 and self.reference_points.coords_depth_type=="fixed": print(f"using anchor ref pts with depth : {self.reference_points.coords_depth}")
            
            # ref_pts_2d_norm: 2d img ref points normalized in range [0,1] Tensor[B, h0*N*w0+..., 2]
            # ref_pts_2p5d_norm: 2.5d img ref points unnormalized Tensor[B, h0*N*w0+..., 3]
            ref_pts_2d_norm, ref_pts_2p5d_unnorm = self.reference_points.get_enc_out_proposals_and_ref_pts(
                    query.size(0), orig_spatial_shapes, query.device, center_depth_group=True)
                
            assert torch.logical_and(ref_pts_2d_norm >= 0.0, ref_pts_2d_norm <= 1.0).all()
            reference_points = [ref_pts_2d_norm, ref_pts_2p5d_unnorm]
        else:
            raise Exception(f"{self.learn_ref_pts_type} is not supported")
        
        if self.learn_ref_pts_type in ["linear", "mlp"]:
            assert torch.logical_and(reference_points >= 0.0, reference_points <= 1.0).all()
        
        return reference_points

    
    def sparsify_inputs(self, src, pos, n_sparse_tokens):
        """
        Args:
            src (_type_): [B, h0*N*w0+..., C]
            pos (_type_): [B, h0*N*w0+..., C]
        """
        src_mask_pred=None
        if self.use_mask_predictor:
            src_with_pos = src + pos
            if self.process_backbone_mem:
                src_with_pos = self.enc_output(src_with_pos)
                src_with_pos = self.enc_output_norm(src_with_pos)
            # [B, h0*N*w0+...]
            src_mask_pred = self.mask_predictor(src_with_pos).squeeze(-1)
            top_rho_inds = torch.topk(src_mask_pred, n_sparse_tokens, dim=1)[1]
        else:
            top_rho_inds = torch.arange(0, src.size(1),dtype=torch.int64, device=src.device)
            top_rho_inds=top_rho_inds[None].repeat(src.size(0), 1) # [B, H*N*W]

        top_rho_inds_rep_d = top_rho_inds.unsqueeze(-1).repeat(1, 1, self.embed_dims) # [B, p, C]

        query = torch.gather(src, 1, top_rho_inds_rep_d) # [B, p, C]
        pos = torch.gather(pos, 1, top_rho_inds_rep_d) # [B, p, C]

        return query, pos, top_rho_inds, src_mask_pred


    def forward(self, src, orig_spatial_shapes, flattened_spatial_shapes, 
                flattened_level_start_index, pos, img_metas, locations_flatten, 
                depth_pred=None, lidar2img=None, extrinsics=None, ensure_no_print=False,
                **data):
        """
        Args:
            src (_type_): [B, h0*N*w0+..., C]
            orig_spatial_shapes (torch.Tensor): [n_levels, 2]
            flattened_spatial_shapes (torch.Tensor): [n_levels, 2]
            flattened_level_start_index (torch.Tensor): [n_levels]
            pos (torch.Tensor): pos embeds [B, h0*N*w0+..., C]
            img_metas (dict),
            locations_flatten (torch.Tensor): [B, h0*N*w0+..., 2]
            depth_pred (torch.Tensor) [B, h0*N*w0+...,1] (unnormalized)
            lidar2img (torch.Tensor): [B, N, 4, 4],
            extrinsics (torch.Tensor): [B, N, 4, 4]
        """
        assert self.use_pos_embed3d
        assert self.use_spatial_alignment

        B = src.size(0)
        n_feat_levels = orig_spatial_shapes.size(0)
        n_tokens = flattened_spatial_shapes.prod(1).sum().item()
        n_sparse_tokens = int(self.sparse_rho * n_tokens) + 1 if self.use_mask_predictor else n_tokens # round up

        if self._iter == 0 and n_sparse_tokens != n_tokens:
            print(f"using {n_sparse_tokens} / {n_tokens} tokens (proportion: {n_sparse_tokens/n_tokens})")
        src = self.src_embed(src) # [B, h0*N*w0+..., C] # M:0.55GB

        if self.mask_pred_before_encoder:
            pos_orig=pos.detach().clone()
            query, pos, top_rho_inds, mask_pred = self.sparsify_inputs(src, pos, n_sparse_tokens)

        if not self.created_reference_points:
            self.create_reference_points_method(src.device, n_sparse_tokens)

        # if self.pred_ref_pts_depth and self.pred_ref_pts_depth_before_spatial_alignment:
            # ref_pts_depth_norm = self.depth_branch(src) # [B, h0*N*w0+..., 1]
            # assert torch.logical_and(ref_pts_depth_norm >= 0.0, ref_pts_depth_norm <= 1.0).all()
            # ref_pts_depth_unnorm = ref_pts_depth_norm * self.depth_range + self.depth_start # in range [1, depth_max]
        
        if self.use_pos_embed3d:
            # pos_embed3d: [B, h0*N*w0+..., C],  cone: [B, h0*N*w0+..., 8], img2lidar: # [B, h0*N*w0+..., 4, 4]
            # mem:1.8GB
            pos_embed3d, cone, img2lidar = self.pos_embed3d(data, locations_flatten, img_metas, 
                                                        orig_spatial_shapes, lidar2img=lidar2img)
            if self.use_spatial_alignment and self.spatial_alignment_all_memory:
                assert pos_embed3d.shape == src.shape
                assert list(cone.shape) == list(src.shape[:2]) + [8]
                # mem: 0.92GB
                src = self.spatial_alignment(src, cone) # [B, h0*N*w0+..., C]
            # [B, h0*N*w0+..., C]
            # mem: 0.738GB
            # ! TODO: NEED TO PULL THIS OUT TO TAKE INTO ACCOUNT OF CASE WHERE POS_EMBED3D IS GENERATED FROM PETR3D
            pos_embed3d = self.featurized_pe(pos_embed3d, src)
            if pos.size(1) < pos_embed3d.size(1):
                pos_embed3d=torch.gather(pos_embed3d, 1, top_rho_inds.unsqueeze(-1).repeat(1, 1, pos_embed3d.size(-1)))
            assert pos.shape == pos_embed3d.shape
            if self.encode_3dpos_method == "add":
                pos = pos + pos_embed3d
            elif self.encode_3dpos_method == "mln":
                if self._iter == 0: print("using mln to encode 3d query pos")
                pos = self.pos3d_encoding(pos, pos_embed3d)
        
        if not self.mask_pred_before_encoder:
            pos_orig=pos.detach().clone()
            query, pos, top_rho_inds, mask_pred = self.sparsify_inputs(src, pos, n_sparse_tokens) # M: 1.977 GB

        if depth_pred is None and self.depth_pred_position == 1 and self.depth_net is not None:
            # TODO: predict depths here after spatial alignment
            # TODO: try predicting depths not only with src but with src + pos
            # TODO: explore ways to use multi-scale features meaningfully to predict depth
            # TODO: explore using sparseconv
            if self.depth_net.depth_net_type in ["conv", "residual"] or not self.depth_net.shared:
                depth_pred_inp = src+pos_orig
            else:
                depth_pred_inp=query+pos
            # [B, p, 1] | [B, H*N*W, 1]
            depth_pred = self.depth_net(depth_pred_inp, flattened_spatial_shapes=flattened_spatial_shapes,
                                        orig_spatial_shapes=orig_spatial_shapes, return_flattened=True)

        ref_pts_out_props = self.get_reference_points(query, orig_spatial_shapes, depths=depth_pred, n_tokens=n_tokens, 
                                                      top_rho_inds=top_rho_inds if n_tokens != n_sparse_tokens else None) # M: 0.59GB
        # ! TODO: ENCODE DEPTH CHOICE INTO QUERY POS
        if self.learn_ref_pts_type == "anchor":
            assert isinstance(ref_pts_out_props, (tuple, list)) and len(ref_pts_out_props) == 2
            reference_points_2d_cam_orig = ref_pts_out_props[0] # [B, h_0*N*w_0+..., 2]
            # NOTE: (x,y) is normalized, depth is unnormalized
            ref_pts_2p5d_unnorm = ref_pts_out_props[1] # [B, h_0*N*w_0+..., 3]
            ref_pts_2p5d_unnorm = torch.gather(ref_pts_2p5d_unnorm, 1, top_rho_inds.unsqueeze(-1).repeat(1, 1, ref_pts_2p5d_unnorm.size(-1))) # [B, p, 3]
            
            # [B, h0*N*w0+..., 1] | [B, p, 1]
            ref_pts_depth_norm = (ref_pts_2p5d_unnorm[..., 2:3] - self.depth_start) / (self.depth_range)
            reference_points_2d_cam_orig = torch.gather(reference_points_2d_cam_orig, 1, top_rho_inds.unsqueeze(-1).repeat(1, 1, reference_points_2d_cam_orig.size(-1))) # [B, p, 2]
            reference_points_2d_cam = reference_points_2d_cam_orig.unsqueeze(2).repeat(1, 1, orig_spatial_shapes.size(0), 1)
            img2lidar = torch.gather(img2lidar, 1, top_rho_inds[...,None,None].repeat(1, 1, *img2lidar.shape[-2:]))
            
            # [B, h0*N*w0+..., 3]
            output_proposals = Projections.project_2p5d_cam_to_3d_lidar(ref_pts_2p5d_unnorm, img2lidar, 
                                                     pc_range=self.pc_range)
  
            # output_proposals = torch.gather(all_output_proposals, 1, 
            #                                 top_rho_inds.unsqueeze(-1).repeat(1, 1, all_output_proposals.size(-1)))
            if not ensure_no_print and self._iter % 50 == 0: 
                prop_out_of_range=not_in_lidar_range(output_proposals, self.pc_range).sum().item()/output_proposals.numel()
                print(f"proportion of output proposals out of range: {prop_out_of_range}")
            output_proposals = clamp_to_lidar_range(output_proposals, self.pc_range)
            output_proposals = normalize_lidar(output_proposals, self.pc_range)
        else:
            reference_points = ref_pts_out_props
            output_proposals = ref_pts_out_props
            assert list(reference_points.shape) == [*query.shape[:-1], 3]
            assert torch.logical_and(reference_points >= 0.0, reference_points <= 1.0).all()
            if self.use_inv_sigmoid:
                reference_points_unnormalized = inverse_sigmoid(reference_points.clone())
            else:
                reference_points_unnormalized = denormalize_lidar(reference_points.clone(), self.pc_range)
            cam_transformations = dict(lidar2img=lidar2img, lidar2cam=extrinsics)
            assert lidar2img.dim() == 4, f"got lidar2img shape: {lidar2img.shape}"
            assert extrinsics.dim()==4, f"got extrinsics shape: {extrinsics.shape}"
            with torch.no_grad():
                # [B, p, n_levels, 2]
                reference_points_2d_cam, non_matches, proj2dpts, projcams = convert_3d_to_2d_global_cam_ref_pts(
                    cam_transformations, reference_points_unnormalized, orig_spatial_shapes, img_metas,
                    debug=True)
                if self.debug and self._iter % 50 == 0:
                    num_non_match_prop = non_matches.sum(1) / non_matches.size(-1)
                    proj2dpts_cpu=proj2dpts.detach().cpu().numpy()
                    projcam_count=torch.unique(projcams, return_counts=True, sorted=True)
                    self.debug_logger.info(f"ENCODER: ref pts non match proportion: {num_non_match_prop}\n"
                                f"\t proj2dpts: (min=({proj2dpts[..., 0].min()},{proj2dpts[..., 1].min()}), max=({proj2dpts[..., 0].max()},{proj2dpts[..., 1].max()}), "
                                f"25%=({np.percentile(proj2dpts_cpu[..., 0], 0.25), np.percentile(proj2dpts_cpu[..., 1], 0.25)})|"
                                f"50%=({np.percentile(proj2dpts_cpu[..., 0], 0.50), np.percentile(proj2dpts_cpu[..., 1], 0.50)})|"
                                f"75%=({np.percentile(proj2dpts_cpu[..., 0], 0.75)},{np.percentile(proj2dpts_cpu[..., 1], 0.75)}))\n"
                                f"\tprojcams: (cams={projcam_count[0]}, counts={projcam_count[1]})")

        if self.encode_ref_pts_depth_into_query_pos:
            if self._iter == 0: print("encoding depth into query pos")
            assert ((ref_pts_depth_norm >= 0.0) & (ref_pts_depth_norm <= 1.0)).all()
            # query_pos=pos.clone()
            ref_pts_depth_norm_emb = pos2posemb1d(ref_pts_depth_norm) # [B, h0*N*w0+..., 256]
            if self.ref_pts_depth_encoding_method == "mln":
                if self._iter == 0: print("using mln method of encoding depth into query pos")
                pos = self.query_pos_2p5d_ref_pts_depth(pos, ref_pts_depth_norm_emb)
            elif self.ref_pts_depth_encoding_method == "linear":
                pos += self.query_pos_2p5d_ref_pts_depth(ref_pts_depth_norm_emb)
            else:
                raise Exception()

        if (self.encode_query_with_ego_pos or self.encode_query_pos_with_ego_pos) and self.rec_ego_motion is None:
            # [1, 1, 4, 4] -> [B, p, 4, 4]
            rec_ego_pose = torch.eye(4, device=query.device)[None, None].repeat(*query.shape[:2], 1, 1)
            # cat [B, p, 3] - [B, p, 3*4] -> [B, p, 15]
            rec_ego_motion = torch.cat([torch.zeros_like(reference_points[...,:3]), 
                                        rec_ego_pose[..., :3, :].flatten(-2)], dim=-1)
            rec_ego_motion = nerf_positional_encoding(rec_ego_motion) # [B, p, 180]
            self.rec_ego_motion=rec_ego_motion
        if self.encode_query_with_ego_pos:
            assert query.shape[:2] == self.rec_ego_motion.shape[:2]
            query = self.ego_pose_memory(query, self.rec_ego_motion)

        if self.encode_query_pos_with_ego_pos:
            pos = self.ego_pose_pe(pos, rec_ego_motion)
        
        intermediate = []
        intermediate_reference_points = []

        # TODO: IF NOT USING ITERATIVE BBOX REFINEMENT, CAN EXPERIMENT JUST USING FIXED REF PTS
        # TODO(cont): NO NEED TO DO 3D LIDAR - 2D CAM PROJECTION AT ALL, JUST UNIFORMLY DISTRIBUTE IN RANGE [0,1]
        # TODO(cont): MIMICKING THE FIXED REF PTS IN SPARSE DETR
        

        # init_ref_pts_unnormalized=None
        for lid, layer in enumerate(self.layers):
            if lid > 0:
                raise NotImplementedError("doesn't support > 1 encoder layer yet")
            assert list(reference_points_2d_cam.shape) == [B, n_sparse_tokens, n_feat_levels, 2]
            # query_out: [B, Q, C]
            # sampling_locations: [B, Q, n_heads, n_feature_lvls, n_points, 2]
            # attn_weights: [B, Q, n_heads, n_feature_lvls, n_points]
            query_out, sampling_locations, attn_weights = layer(query=query, src=src, pos=pos,
                                                            reference_points=reference_points_2d_cam, 
                                                            flattened_spatial_shapes=flattened_spatial_shapes, 
                                                            flattened_level_start_index=flattened_level_start_index)
            if "encoder" not in self.mask_pred_target:
                del sampling_locations
                del attn_weights
            if self.post_norm:
                query_out = self.post_norm(query_out)
            output = src.scatter(1, top_rho_inds.unsqueeze(-1).repeat(1,1, src.size(-1)), query_out)
        
        ## prediction
        # ! WARNING: only supports one layer rn
        level=0
        out_cls=self.cls_branches[level](query_out) # [B, p, num_classes]
        out_coord_offset = self.reg_branches[level](query_out) # [B, p, num_classes]
        if self.use_sigmoid_on_attn_out:
            if self._iter == 0: print("ENCODER: using sigmoid on attention out")
            out_coord_offset[..., :3] = F.sigmoid(out_coord_offset[..., :3])
            out_coord_offset[..., :3] = denormalize_lidar(out_coord_offset[..., :3], self.pc_range)
        out_coord = out_coord_offset
        assert ((output_proposals >= 0.0) & (output_proposals <= 1.0)).all()
        if self.use_inv_sigmoid:
            if self._iter == 0: print("using inverse sigmoid")
            output_proposals_unnormalized = inverse_sigmoid(output_proposals)
        else:
            if self._iter == 0: print("NOT using inverse sigmoid")
            output_proposals_unnormalized = denormalize_lidar(output_proposals, self.pc_range)
            assert not_in_lidar_range(output_proposals_unnormalized, self.pc_range).sum() == 0
        
        if level == 0:
            # out_coord[..., :3] += init_ref_pts_unnormalized[..., :3]
            assert out_coord[..., :3].shape == output_proposals_unnormalized.shape
            out_coord[..., :3] += output_proposals_unnormalized[..., :3]
        else:
            raise NotImplementedError()
        if self.use_inv_sigmoid:
            out_coord[..., :3]=out_coord[..., :3].sigmoid()
            out_coord[..., :3]=denormalize_lidar(out_coord[..., :3], self.pc_range)
            out_coord_final=out_coord
        else:
            out_coord_final = clamp_to_lidar_range(out_coord.clone(), self.pc_range)
        

        enc_pred_dict = {
            "cls_scores_enc": out_cls,
            "bbox_preds_enc": out_coord_final,
            "sparse_token_num": n_sparse_tokens,
            "src_mask_prediction": mask_pred
        }
        if self.pred_ref_pts_depth:
            # reference_points_2d_cam: [B, h0*N*w0+..., 2] (x,y) is normalized in level range [0,1]
            # depth_pred: [B, h0*N*w0+..., 1] is unnormalized in range [depth_start, depth_max]
            enc_pred_dict.update({
                "ref_pts_2point5d": torch.cat([reference_points_2d_cam_orig, depth_pred], -1),
            })
        if "encoder" in self.mask_pred_target:
            enc_pred_dict.update({
                "sampling_locations_enc": sampling_locations,
                "attention_weights_enc": attn_weights
            })
        if self.propagate_pos:
            enc_pred_dict.update({
                "pos_enc": pos
            })
        if self.propagate_3d_pos:
            enc_pred_dict.update({
                "pos_3d_enc": pos_embed3d
            })
        self._iter += 1
        return enc_pred_dict, output


    def assign_gt_bboxs_to_2d_grid_cells(self,gt_bboxes, lidar2img, lidar2cam, orig_spatial_shapes):
        orig_h, orig_w = orig_spatial_shapes
        # gt_bboxes: Tensor[n_objs, 10]
        # lidar2img: Tensor[6, 4, 4]
        # lidar2cam: Tensor[6, 4, 4]
        gt_3d_xyz = gt_bboxes.unsqueeze(0)[..., :3]
        point_2p5d_cam = Projections.project_lidar_points_to_all_2point5d_cams_batch(gt_3d_xyz, lidar2img.unsqueeze(0)).transpose(1,2) # [B, 6, n_objs, 2]
        point_3d_cam = Projections.project_lidar_points_to_all_3d_cams_batch(gt_3d_xyz, lidar2cam.unsqueeze(0)).transpose(1, 2) # [B, 6, n_objs, 3]

        in_front = point_3d_cam[..., 2] > 0
        in_2d_range = (point_2p5d_cam[..., 0] >= 0) & (point_2p5d_cam[..., 0] < Projections.IMG_SIZE[0]) & \
                    (point_2p5d_cam[..., 1] >= 0) & (point_2p5d_cam[..., 1] < Projections.IMG_SIZE[1])
        matches = in_front & in_2d_range # [B, n_objs, 6]
        # ! HUGE BUG: THE MAPPING OF GT BBOXES TO 2D POINTS NEEDS TO ACCOUNT WHICH CAM THE POINT GOT PROJECTED TO (SIMILAR TO convert_3d_to_mult_2d_global_cam_ref_pts IN projections.py)
        # ! AND THE LABEL1D SHOULD BE GLOBAL, I.E. INSTEAD OF ORIG_W USE ORIG_W*NUM_CAMERAS
        valid_2p5d_cam = point_2p5d_cam[matches] # [n_valid_objs, 3]
        label1d = valid_2p5d_cam[..., 1] * orig_w + valid_2p5d_cam[..., 0]
        label1d = label1d.long()
        gt_depths = valid_2p5d_cam[..., 2:3] # Tensor[n_valid_objs, 1]
        agg_depths = groupby_agg_mean(gt_depths, label1d, orig_w*orig_h).squeeze(-1) # [n_valid_objs]
        return agg_depths

    

    # def get_depth_target_single(self, gt_bboxs_3d, lidar2img=None, lidar2cam=None, 
    #                             orig_spatial_shapes=None, ref_pts_pred_norm=None, num_cameras=6):
    #     """
    #     Args:
    #         gt_bboxs_3d (_type_): [n_objs, 9]
    #         lidar2img (_type_, optional): _description_. Defaults to None.
    #         lidar2cam (_type_, optional): _description_. Defaults to None.
    #         orig_spatial_shapes (_type_, optional): _description_. Defaults to None.
    #         img_metas (_type_, optional): _description_. Defaults to None.
    #         ref_pts_pred_norm (_type_, optional): [h0*N*w0+..., 3], expecting the (x,y) values for each level to be unnormalized to 
    #                                                     [width_level, height_level], the depth is expected to be unnormalized too. 
    #                                                     Defaults to None.
    #         num_cameras (int, optional): _description_. Defaults to 6.
    #     """
    #     gt_bbox_centers_3d= gt_bboxs_3d[..., :3].unsqueeze(0) # [1, n_objs, 3]
    #     def get_depth_weights(x, lvl=0):
    #         # exponential_decay
    #         lam = 10
    #         lam = lam / (lvl + 1)
    #         w = torch.exp(-x/lam)

    #         return w
    #     n_levels=orig_spatial_shapes.size(0)
    #     cam_transformations=dict(lidar2img=lidar2img.unsqueeze(0), lidar2cam=lidar2cam.unsqueeze(0))
    #     # global_ref_pts: [n_matches, n_levels, 3]
    #     # ! WARNING: since a point can be projected to multiple cams, n_matches >= n_objs
    #     global_2p5d_pts_norm = project_to_matching_2point5d_cam_points(gt_bbox_centers_3d, cam_transformations, 
    #                                                                    orig_spatial_shapes, num_cameras=num_cameras)
    #     flattened_spatial_shapes = orig_spatial_shapes.clone()
    #     flattened_spatial_shapes[:, 1] = flattened_spatial_shapes[:, 1] * num_cameras
    #     flattened_spatial_shapes_xy = torch.stack([flattened_spatial_shapes[..., 1], flattened_spatial_shapes[..., 0]], -1)
    #     global_2p5d_pts = global_2p5d_pts_norm.clone() # [n_matches, n_levels, 3]
    #     global_2p5d_pts[..., :2] = global_2p5d_pts[..., :2] * flattened_spatial_shapes_xy 
    #     levels = [h_i*w_i for (h_i, w_i) in flattened_spatial_shapes]
    #     split_grids_xy = torch.split(ref_pts_pred_norm[..., :2], levels, dim=0)
    #     # global_2p5d_pts=global_2p5d_pts.contiguous()
    #     # ref_pts_pred_unnorm = ref_pts_pred_norm.clone().unsqueeze(1) # [n_preds, 1, 3]
    #     # ref_pts_pred_unnorm[..., :2] = ref_pts_pred_unnorm[..., :2].repeat(1,n_levels,1) * flattened_spatial_shapes_xy[None] 
    #     all_targets = []
    #     all_dists=[]
    #     all_weights = []
    #     all_target_depths = []
    #     for lvl in range(n_levels):
    #         unnorm_pts = split_grids_xy[lvl] * flattened_spatial_shapes_xy[lvl] # [n_pred_lvl, 2]
    #         # unnorm_pts = unnorm_pts.contiguous()
    #         global_lvl_pts = global_2p5d_pts[:,lvl,:2]
    #         # global_lvl_pts=global_lvl_pts.contiguous()
    #         l1dist_lvl = (unnorm_pts.unsqueeze(1) -global_lvl_pts[None]).abs().sum(-1)
    #         # l1dist_lvl_old = torch.cdist(unnorm_pts, global_lvl_pts, p=1.0)
    #         # (l1dist_lvl == l1dist_lvl_old).all()
    #         l1_loss, targets = l1dist_lvl.min(-1) # [n_matches]
    #         w = get_depth_weights(l1_loss, lvl=lvl)
    #         if self.depth_weight_bound is not None:
    #             w[w<self.depth_weight_bound]=0.0
    #         target_depths = global_2p5d_pts[targets, lvl, 2]
    #         all_dists.append(l1_loss)
    #         all_targets.append(targets)
    #         all_weights.append(w)
    #         all_target_depths.append(target_depths)
    #     all_targets = torch.cat(all_targets, 0)
    #     all_dists = torch.cat(all_dists, 0)
    #     all_weights = torch.cat(all_weights, 0)
    #     all_target_depths = torch.cat(all_target_depths, 0)
    #     assert all_target_depths.shape == ref_pts_pred_norm[..., 2].shape
    #     if self.div_depth_loss_by_target_count:
    #         selected_targets, target_counts = torch.unique(all_targets, return_counts=True,sorted=True)
    #         full_target_counts = target_counts.new_zeros([global_2p5d_pts_norm.size(0)])
    #         full_target_counts[selected_targets] = target_counts
    #         # print(f"unique targets: {selected_targets}")
    #         # print(f"target counts: {target_counts}")
    #         # print(f"full target counts: {full_target_counts}")
    #     elif self.depth_weight_bound is not None:
    #         full_target_counts=global_2p5d_pts_norm.size(0)
    #     else:
    #         full_target_counts=None
    #     return all_target_depths, all_targets, all_weights, full_target_counts
        
    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                        #    lidar2img=None,
                        #    lidar2cam=None,
                        #    orig_spatial_shapes=None,
                        #    ref_pts_depth_norm=None, # [h0*N*w0+..., 3]
                           ):
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler

        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes, gt_labels, 
                                             code_weights=self.match_costs, with_velo=self.match_with_velo)
        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        code_size = gt_bboxes.size(1)
        bbox_targets = torch.zeros_like(bbox_pred)[..., :code_size]
        bbox_weights = torch.zeros_like(bbox_pred)
        # print(gt_bboxes.size(), bbox_pred.size())
        # DETR
        if sampling_result.num_gts > 0:
            bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
            bbox_weights[pos_inds] = 1.0
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]    

        # depth 
        # if self.pred_ref_pts_loss:
        #     # ref_pts_depth_unnorm_lvl = ref_pts_depth_norm.clone()
        #     # ref_pts_depth_unnorm_lvl[..., :2] = ref_pts_depth_unnorm_lvl[..., :2] * orig_spatial_shapes
        #     target_depths, target_depth_gt_inds, depth_weights, depth_target_counts = \
        #         self.get_depth_target_single(gt_bboxes,lidar2img,lidar2cam,orig_spatial_shapes,
        #                                      ref_pts_depth_norm)
        #     if self.div_depth_loss_by_target_count:
        #         try:
        #             depth_weights = depth_weights / depth_target_counts[target_depth_gt_inds]
        #         except:
        #             with open("./experiments/depth_target_counts.pkl", "wb") as f:
        #                 pickle.dump(depth_target_counts, f)
        #             with open("./experiments/target_depth_gt_inds.pkl", "wb") as f:
        #                 pickle.dump(target_depth_gt_inds, f)
        #             with open("./experiments/depth_weights.pkl", "wb") as f:
        #                 pickle.dump(depth_weights, f)
        #             raise Exception()

        #         num_total_depth_pos = depth_target_counts.numel()
        #     elif self.depth_weight_bound is not None:
        #         num_total_depth_pos = depth_target_counts
        #     else:
        #         num_total_depth_pos = ref_pts_depth_norm.numel()
        #         # print(num_total_depth_pos)

        #     return (labels, label_weights, bbox_targets, bbox_weights, 
        #             pos_inds, neg_inds, target_depths, depth_weights, 
        #             num_total_depth_pos)
        # else:
        return (labels, label_weights, bbox_targets, bbox_weights, 
                pos_inds, neg_inds)
    
    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    # lidar2img_list=None,
                    # lidar2cam_list=None,
                    # orig_spatial_shapes_list=None,
                    # ref_pts_depth_norm_list=None
                    ):
        # if self.pred_ref_pts_loss:
        #     (labels_list, label_weights_list, bbox_targets_list,
        #     bbox_weights_list, pos_inds_list, neg_inds_list, 
        #     depth_targets_list, depth_weights_list, num_valid_depths_list) = multi_apply(
        #         self._get_target_single, cls_scores_list, bbox_preds_list,
        #         gt_labels_list, gt_bboxes_list, lidar2img_list, lidar2cam_list, 
        #         orig_spatial_shapes_list, ref_pts_depth_norm_list)
        # else:
        (labels_list, label_weights_list, bbox_targets_list,
        bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        # if self.pred_ref_pts_loss: 
        #     num_total_valid_depths = sum(num_valid_depths_list)
        #     return (labels_list, label_weights_list, bbox_targets_list,
        #         bbox_weights_list, num_total_pos, num_total_neg, depth_targets_list, 
        #         depth_weights_list, num_total_valid_depths)
        # else:
        return (labels_list, label_weights_list, bbox_targets_list,
            bbox_weights_list, num_total_pos, num_total_neg)
    
    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    # lidar2img=None,
                    # lidar2cam=None,
                    # orig_spatial_shapes=None,
                    # ref_pts_pred_norm=None
                    ):
        """
        Args:
            cls_scores (_type_): Tensor[B, p, num_classes]
            bbox_preds (_type_): Tensor[B, p, 10]
            gt_bboxes_list (_type_): List[B] each element is a Tensor[num_gt_i, 9]
            gt_labels_list (_type_): List[B] each element is a Tensor[num_gt_i]
            orig_spatial_shapes: Tensor[n_levels, 2]
            ref_pts_pred_norm: Tensor[B, h0*N*w0+..., 3]
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        # if self.pred_ref_pts_loss:
        #     lidar2img_list = [lidar2img[i] for i in range(num_imgs)]
        #     lidar2cam_list = [lidar2cam[i] for i in range(num_imgs)]
        #     orig_spatial_shapes_list = [orig_spatial_shapes for _ in range(num_imgs)]
        #     ref_pts_depth_norm_list = [ref_pts_pred_norm[i] for i in range(num_imgs)]


        # ref_pts_depth_preds_list = [ref_pts_depth_preds[i] for i in range(num_imgs)]
        # if self.pred_ref_pts_loss:
        #     cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
        #                                     gt_bboxes_list, gt_labels_list, 
        #                                     lidar2img_list=lidar2img_list, lidar2cam_list=lidar2cam_list,
        #                                     orig_spatial_shapes_list=orig_spatial_shapes_list,
        #                                     ref_pts_depth_norm_list=ref_pts_depth_norm_list)
        # else:
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                        gt_bboxes_list, gt_labels_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
        num_total_pos, num_total_neg) = cls_reg_targets[:6]
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)


        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
            
        cls_avg_factor = max(cls_avg_factor, 1)

        cls_info=dict(cls_scores=cls_scores, labels=labels, label_weights=label_weights,
                      avg_factor=cls_avg_factor)
        with open("./experiments/cls_info_iq.pkl", "wb") as f:
            pickle.dump(cls_info, f)

        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)
        

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
                bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], 
                bbox_weights[isnotnan, :10], avg_factor=num_total_pos)
        
        loss_cls = torch.nan_to_num(loss_cls, nan=1e-16, posinf=100.0, neginf=-100.0)
        loss_bbox = torch.nan_to_num(loss_bbox, nan=1e-16, posinf=100.0, neginf=-100.0)

        # depth loss
        # if self.pred_ref_pts_loss:
        #     (depth_targets_list, depth_weights_list, num_valid_depths)= cls_reg_targets[6:9]
        #     depth_targets = torch.cat(depth_targets_list, 0)
        #     depth_weights = torch.cat(depth_weights_list, 0)
        #     num_total_valid_depths = loss_cls.new_tensor([num_valid_depths])
        #     num_total_valid_depths = torch.clamp(reduce_mean(num_total_valid_depths), min=1).item()
        #     # ref_pts_depth_pred = ref_pts_pred_norm[..., 2]
        #     ref_pts_depth_pred = torch.cat([ref_pts[..., 2] for ref_pts in  ref_pts_depth_norm_list],0)
        #     loss_depth = self.loss_depth(
        #         ref_pts_depth_pred, depth_targets, depth_weights, avg_factor=num_total_valid_depths
        #     )
        #     loss_depth = torch.nan_to_num(loss_depth)
        #     return loss_cls,loss_bbox,loss_depth

        return loss_cls, loss_bbox
    
    def loss(self, gt_bboxes_list, gt_labels_list, preds_dicts, flattened_spatial_shapes,
             flattened_level_start_index, lidar2img=None, lidar2cam=None, img_metas=None,
             orig_spatial_shapes=None):
        """
        Args:
            gt_bboxes_list (List): ground truth 3d gt bboxes
            gt_labels_list (List): ground truth labels for each bbox
            preds_dicts (_type_): 
            lidar2img: [B, N, 4, 4]
            lidar2cam: [B, N, 4, 4]

        Returns:
            _type_: _description_
        """
        # print([b.tensor.shape for b in gt_bboxes_list])
        assert self.num_layers == 1
        cls_scores = preds_dicts['cls_scores_enc']
        bbox_preds = preds_dicts['bbox_preds_enc']

        device = gt_labels_list[0].device
        # list of len B where each element is a Tensor[num_gts, 9]
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]
        
        loss_dict=dict()
        # if self.pred_ref_pts_loss:
        #     # ref_pts_depth_preds = preds_dicts['ref_pts_depth_pred'] # [B, h0*N*w0+..., 1]
        #     ref_pts_2point5d_norm = preds_dicts['ref_pts_2point5d_norm'] # [B, h0*N*w0+..., 3]
        #     # orig_spatial_shape = img_metas[0]['pad_shape'][0][:2]
        #     loss_cls, loss_bbox, loss_depth = self.loss_single(cls_scores, bbox_preds, gt_bboxes_list, gt_labels_list,
        #                                            lidar2img=lidar2img, lidar2cam=lidar2cam, 
        #                                            orig_spatial_shapes=orig_spatial_shapes, 
        #                                            ref_pts_pred_norm=ref_pts_2point5d_norm)
        #     loss_dict['enc.loss_depth'] = loss_depth
        # else:
        loss_cls, loss_bbox = self.loss_single(cls_scores, bbox_preds, gt_bboxes_list, gt_labels_list)

        loss_dict['enc.loss_cls'] = loss_cls
        loss_dict['enc.loss_bbox'] = loss_bbox

        ## mask loss
        if self.use_mask_predictor is not None and "encoder" in self.mask_pred_target:
            loss_mask = self.mask_predictor.loss(
                mask_prediction=preds_dicts['src_mask_prediction'],
                sampling_locations=preds_dicts['sampling_locations_enc'],
                attn_weights=preds_dicts['attention_weights_enc'],
                flattened_spatial_shapes=flattened_spatial_shapes,
                flattened_level_start_index=flattened_level_start_index,
                sparse_token_nums=preds_dicts['sparse_token_num']
            )

            loss_dict.update({
                "loss_mask": loss_mask
            })

        return loss_dict
        

@TRANSFORMER_LAYER.register_module()
class IQTransformerEncoderLayer(BaseTransformerLayer):
    def __init__(self, **kwargs):
        super(IQTransformerEncoderLayer, self).__init__(**kwargs)
        

    def forward(self, query, src, pos, reference_points, flattened_spatial_shapes,
                flattened_level_start_index):
        """
        Args:
            query (_type_): top-rho feature tokens [B, p, C]
            src (_type_): [B, h0*N*w0+..., C]
            pos (_type_): 3d pos embeddings [B, p, C]
            reference_points (_type_): 2d cam ref pts [B, p, 2]
            orig_spatial_shapes (_type_): [n_levels, 2]
            flattened_spatial_shapes (_type_): [n_levels, 2]
            flattened_level_start_index (_type_): [n_levels]
        """
        norm_index=0
        attn_index=0
        ffn_index=0
        for layer in self.operation_order:
            if layer == "self_attn":
                query, sampling_locations, attn_weights = self.attentions[attn_index](
                    query=query, value=src, query_pos=pos, 
                    reference_points=reference_points, flattened_spatial_shapes=flattened_spatial_shapes, 
                    flattened_lvl_start_index=flattened_level_start_index, return_query_only=False)
                attn_index += 1
            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index+=1
            elif layer == "ffn":
                query = self.ffns[ffn_index](query, None)
                ffn_index += 1
        return query, sampling_locations, attn_weights