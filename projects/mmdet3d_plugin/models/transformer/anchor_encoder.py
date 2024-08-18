import torch
from torch import nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import (TransformerLayerSequence, BaseTransformerLayer, 
                                         TRANSFORMER_LAYER_SEQUENCE, TRANSFORMER_LAYER)
from mmcv.cnn.bricks.plugin import build_plugin_layer
from mmdet.core import (build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models import build_loss
from projects.mmdet3d_plugin.models.utils.positional_encoding import pos2posemb3d, pos2posemb1d
from projects.mmdet3d_plugin.models.utils.misc import MLN, SELayer_Linear
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from projects.mmdet3d_plugin.attentions.custom_deform_attn import CustomDeformAttn
from ..utils.lidar_utils import normalize_lidar, clamp_to_lidar_range, not_in_lidar_range
from ..utils.debug import *
from ..utils.anchor_refine import AnchorRefinement
from ..utils.positional_encoding import pos2posemb3d

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class AnchorEncoder(TransformerLayerSequence):
    # modified from SparseDETR
    def __init__(self, 
                 *args,
                 code_size=10, 
                 mlvl_feats_formats=1, 
                 pc_range=None, 
                 learn_ref_pts_type="linear", 
                 use_spatial_alignment=True, 
                 pos_embed3d=None,
                 use_inv_sigmoid=False, 
                 use_sigmoid_on_attn_out=False, 
                 num_classes=10, 
                 limit_3d_pts_to_pc_range=False,
                 anchor_refinement=None,
                 use_anchor_pos=False,
                 ## sparsification
                 mask_predictor=None,
                 sparse_rho=1.0, 
                 post_norm_cfg=dict(type="LN"),
                 ## depth
                 depth_start = 1, 
                 position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0], 
                 depth_pred_position=0, 
                 depth_net=None, 
                 reference_point_generator=None, 
                 encode_ref_pts_depth_into_query_pos=False, 
                 ref_pts_depth_encoding_method=None,
                 encode_3dpos_method="add",
                 ## loss
                 match_with_velo=False, 
                 code_weights=None, 
                 match_costs=None, 
                 sync_cls_avg_factor=False, 
                 train_cfg=None, 
                 loss_cls=None, 
                 loss_bbox=None, 
                 loss_iou=None, 
                 **kwargs):
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

        super(AnchorEncoder, self).__init__(*args, **kwargs)

        self.use_anchor_pos=use_anchor_pos
        if use_anchor_pos:
            self.anchor_embedding=nn.Sequential(
                nn.Linear(self.embed_dims*3//2, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims),
            )

        self.limit_3d_pts_to_pc_range=limit_3d_pts_to_pc_range
        self.pc_range = nn.Parameter(torch.tensor(pc_range), requires_grad=False)
        if mlvl_feats_formats != 1:
            raise NotImplementedError("encoder only supports mlvl feats format 1")
        self.mlvl_feats_formats=mlvl_feats_formats
        self.use_pos_embed3d=pos_embed3d is not None
        self.use_spatial_alignment=use_spatial_alignment
        if self.use_spatial_alignment:
            self.spatial_alignment = MLN(8)
            if self.use_pos_embed3d:
                self.featurized_pe = SELayer_Linear(self.embed_dims)

        self.depth_start=depth_start # 1
        self.depth_range=position_range[3] - self.depth_start
        if self.use_pos_embed3d:
            self.pos_embed3d = build_plugin_layer(pos_embed3d)[1]

        self.code_size=code_size

        self.learn_ref_pts_type=learn_ref_pts_type.lower()
        assert self.learn_ref_pts_type in ['anchor']
        if self.learn_ref_pts_type == "anchor":
            assert reference_point_generator is not None
            reference_point_generator['mlvl_feats_format'] = self.mlvl_feats_formats
            self.reference_points = build_plugin_layer(reference_point_generator)[1]
        
        self.use_mask_predictor = mask_predictor is not None
        if self.use_mask_predictor:
            raise NotImplementedError()
            self.mask_predictor = build_plugin_layer(mask_predictor)[1]
        self.anchor_refinement=build_plugin_layer(anchor_refinement)[1]
        self.sparse_rho = sparse_rho

        self.src_embed = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

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

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self.use_inv_sigmoid=use_inv_sigmoid

        self.encode_3dpos_method=encode_3dpos_method.lower()
        assert self.encode_3dpos_method in ["add", "mln"]
        if self.encode_3dpos_method == "mln":
            self.pos3d_encoding = MLN(self.embed_dims)

        self.anchor_refinement = build_plugin_layer(anchor_refinement)[1]
        if self.num_layers > 1:
            raise NotImplementedError("right now only using 1 encoder layer")
    
    def init_weights(self):
        to_skip=['attention']
        for name, p in self.named_parameters():
            if any([n in name for n in to_skip]):
                continue
            if p is not None and p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p)

        for m in self.modules():
            if m == self: continue
            if isinstance(m, AnchorRefinement):
                continue
            if isinstance(m, CustomDeformAttn):
                # assert m.is_init == False
                m.reset_parameters()
            elif hasattr(m, "init_weights"):
                m.init_weights()
        
        self._is_init = True

    def get_reference_points(self, query, orig_spatial_shapes, depths=None, n_tokens=None, top_rho_inds=None):
        # depths: [B, h0*N*w0+..., 1] unnormalized depths
        
        if self.learn_ref_pts_type == "anchor":
            # if "learnable" depths, it is expected to be unnormalized
            if depths is not None: self.reference_points.set_coord_depths(depths, n_tokens=n_tokens, top_rho_inds=top_rho_inds)
            
            # ref_pts_2d_norm: 2d img ref points normalized in range [0,1] Tensor[B, h0*N*w0+..., 2]
            # ref_pts_2p5d_norm: 2.5d img ref points unnormalized Tensor[B, h0*N*w0+..., 3]
            ref_pts_2d_norm, ref_pts_2p5d_unnorm = self.reference_points.get_enc_out_proposals_and_ref_pts(
                    query.size(0), orig_spatial_shapes, query.device)
                
            assert torch.logical_and(ref_pts_2d_norm >= 0.0, ref_pts_2d_norm <= 1.0).all()
            reference_points = [ref_pts_2d_norm, ref_pts_2p5d_unnorm]
        else:
            raise Exception(f"{self.learn_ref_pts_type} is not supported")
        
        return reference_points

    
    def sparsify_inputs(self, src, pos, n_sparse_tokens):
        """
        Args:
            src (_type_): [B, h0*N*w0+..., C]
            pos (_type_): [B, h0*N*w0+..., C]
        """
        src_mask_pred=None
        if self.use_mask_predictor:
            raise NotImplementedError()
        else:
            top_rho_inds = torch.arange(0, src.size(1),dtype=torch.int64, device=src.device)
            top_rho_inds=top_rho_inds[None].repeat(src.size(0), 1) # [B, H*N*W]

        top_rho_inds_rep_d = top_rho_inds.unsqueeze(-1).repeat(1, 1, self.embed_dims) # [B, p, C]
        query = torch.gather(src, 1, top_rho_inds_rep_d) # [B, p, C]
        pos = torch.gather(pos, 1, top_rho_inds_rep_d) # [B, p, C]
        return query, pos, top_rho_inds, src_mask_pred


    def forward(self, src, orig_spatial_shapes, flattened_spatial_shapes, 
                flattened_level_start_index, pos, img_metas, locations_flatten, 
                depth_pred=None, lidar2img=None, **data):

        B = src.size(0)
        n_feat_levels = orig_spatial_shapes.size(0)
        n_tokens = flattened_spatial_shapes.prod(1).sum().item()
        n_sparse_tokens = int(self.sparse_rho * n_tokens) + 1 if self.use_mask_predictor else n_tokens # round up

        src = self.src_embed(src) # [B, h0*N*w0+..., C] # M:0.55GB
        
        img2lidar=None
        if self.use_pos_embed3d or self.use_spatial_alignment:
            # pos_embed3d: [B, h0*N*w0+..., C],  cone: [B, h0*N*w0+..., 8], img2lidar: # [B, h0*N*w0+..., 4, 4]
            # mem:1.8GB
            pos_embed3d, cone, img2lidar = self.pos_embed3d(data, locations_flatten, img_metas, 
                                                        orig_spatial_shapes, lidar2img=lidar2img)
            if self.use_spatial_alignment:
                # mem: 0.92GB
                src = self.spatial_alignment(src, cone) # [B, h0*N*w0+..., C]
                if self.use_pos_embed3d:
                    # [B, h0*N*w0+..., C]
                    # mem: 0.738GB
                    pos_embed3d = self.featurized_pe(pos_embed3d, src)
            if self.encode_3dpos_method == "add":
                pos = pos + pos_embed3d
            elif self.encode_3dpos_method == "mln":
                pos = self.pos3d_encoding(pos, pos_embed3d)
        else:
            if do_debug_process(self): print("ENCODER: NOT USING 3D POS IN ENCODER")
            img2lidar = []
            i2l = torch.inverse(lidar2img.to('cpu')).to('cuda') # [B, N, 4, 4]
            for (h, w) in orig_spatial_shapes:
                i2l_i = i2l[:, None, :, None, :, :].expand(-1, h, -1, w, -1, -1)
                img2lidar.append(i2l_i.flatten(1, 3)) # [B, h*N*w, 4, 4]
            img2lidar = torch.cat(img2lidar, 1) # [B, h0*N*w0+..., 4, 4]
        
        pos_orig=pos.detach().clone()
        query, pos, top_rho_inds, mask_pred = self.sparsify_inputs(src, pos, n_sparse_tokens) # M: 1.977 GB

        if depth_pred is None and self.depth_pred_position == 1 and self.depth_net is not None:
            if self.depth_net.depth_net_type in ["conv", "residual"] or not self.depth_net.shared:
                depth_pred_inp = src+pos_orig
            else:
                depth_pred_inp=query+pos
            # [B, p, 1] | [B, H*N*W, 1]
            depth_pred = self.depth_net(depth_pred_inp, flattened_spatial_shapes=flattened_spatial_shapes,
                                        orig_spatial_shapes=orig_spatial_shapes, return_flattened=True)

        ref_pts_out_props = self.get_reference_points(query, orig_spatial_shapes, depths=depth_pred, n_tokens=n_tokens, 
                                                      top_rho_inds=top_rho_inds if n_tokens != n_sparse_tokens else None) # M: 0.59GB
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
            output_proposals = self.projections.project_2p5d_cam_to_3d_lidar(ref_pts_2p5d_unnorm, img2lidar)
  
            if do_debug_process(self, repeating=True, interval=500): 
                prop_out_of_range=not_in_lidar_range(output_proposals, self.pc_range).sum().item()/output_proposals.numel()
                print(f"proportion of output proposals out of range: {prop_out_of_range}")
            if self.limit_3d_pts_to_pc_range:
                output_proposals = clamp_to_lidar_range(output_proposals, self.pc_range)
                output_proposals = normalize_lidar(output_proposals, self.pc_range)
        else:
            raise NotImplementedError()
        
        if self.use_anchor_pos:
            if do_debug_process(self): print("ENCODER: USING ANCHOR POS EMBEDDING")
            pos += self.anchor_embedding(pos2posemb3d(output_proposals))
            
        elif self.encode_ref_pts_depth_into_query_pos:
            if do_debug_process(self): print("ENCODER: ENCODING DEPTH REF PTS INTO QUERY POS")
            # assert ((ref_pts_depth_norm >= 0.0) & (ref_pts_depth_norm <= 1.0)).all() # NOTE: from depthnet, depths can be out of range
            ref_pts_depth_norm_emb = pos2posemb1d(ref_pts_depth_norm) # [B, h0*N*w0+..., 256]
            if self.ref_pts_depth_encoding_method == "mln":
                if do_debug_process(self): print("using mln method of encoding depth into query pos")
                pos = self.query_pos_2p5d_ref_pts_depth(pos, ref_pts_depth_norm_emb)
            elif self.ref_pts_depth_encoding_method == "linear":
                pos += self.query_pos_2p5d_ref_pts_depth(ref_pts_depth_norm_emb)
            else:
                raise Exception()

        for lid, layer in enumerate(self.layers):
            if lid > 0:
                raise NotImplementedError("doesn't support > 1 encoder layer yet")
            assert list(reference_points_2d_cam.shape) == [B, n_sparse_tokens, n_feat_levels, 2]
            # query_out: [B, Q, C]
            # sampling_locations: [B, Q, n_heads, n_feature_lvls, n_points, 2]
            # attn_weights: [B, Q, n_heads, n_feature_lvls, n_points]
            query_out = layer(query=query, src=src, pos=pos, reference_points=reference_points_2d_cam, 
                              flattened_spatial_shapes=flattened_spatial_shapes, 
                              flattened_level_start_index=flattened_level_start_index)
            if self.post_norm:
                query_out = self.post_norm(query_out)
            output = src.scatter(1, top_rho_inds.unsqueeze(-1).repeat(1,1, src.size(-1)), query_out)
        
        ## prediction
        # level=0
        out_coord, out_cls = self.anchor_refinement(query_out, output_proposals, pos, 
                                                    time_interval=None, return_cls=True)
        
        enc_pred_dict = {
            "cls_scores_enc": out_cls,
            "bbox_preds_enc": out_coord,
            "sparse_token_num": n_sparse_tokens,
            "src_mask_prediction": mask_pred
        }
        if depth_pred is not None:
            # reference_points_2d_cam: [B, h0*N*w0+..., 2] (x,y) is normalized in level range [0,1]
            # depth_pred: [B, h0*N*w0+..., 1] is unnormalized in range [depth_start, depth_max]
            enc_pred_dict.update({
                "ref_pts_2point5d_pred": torch.cat([reference_points_2d_cam_orig, depth_pred], -1),
            })
        
        return enc_pred_dict, output
    
    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
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
        # DETR
        if sampling_result.num_gts > 0:
            bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
            bbox_weights[pos_inds] = 1.0
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]    

        return (labels, label_weights, bbox_targets, bbox_weights, 
                pos_inds, neg_inds)
    
    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    ):
        
        (labels_list, label_weights_list, bbox_targets_list,
        bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        
        return (labels_list, label_weights_list, bbox_targets_list,
            bbox_weights_list, num_total_pos, num_total_neg)
    
    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    ):
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        
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

        return loss_cls, loss_bbox
    
    def loss(self, gt_bboxes_list, gt_labels_list, preds_dicts, flattened_spatial_shapes,
             flattened_level_start_index):
        assert self.num_layers == 1
        cls_scores = preds_dicts['cls_scores_enc']
        bbox_preds = preds_dicts['bbox_preds_enc']

        device = gt_labels_list[0].device
        # list of len B where each element is a Tensor[num_gts, 9]
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]
        
        loss_dict=dict()
        
        loss_cls, loss_bbox = self.loss_single(cls_scores, bbox_preds, gt_bboxes_list, gt_labels_list)

        loss_dict['enc.loss_cls'] = loss_cls
        loss_dict['enc.loss_bbox'] = loss_bbox

        return loss_dict
        
@TRANSFORMER_LAYER.register_module()
class IQTransformerEncoderLayer(BaseTransformerLayer):
    def __init__(self, **kwargs):
        super(IQTransformerEncoderLayer, self).__init__(**kwargs)

    def forward(self, query, src, pos, reference_points, flattened_spatial_shapes,
                flattened_level_start_index):
        
        norm_index=0
        attn_index=0
        ffn_index=0
        for layer in self.operation_order:
            if layer == "self_attn":
                query= self.attentions[attn_index](
                    query=query, value=src, query_pos=pos, 
                    reference_points=reference_points, flattened_spatial_shapes=flattened_spatial_shapes, 
                    flattened_lvl_start_index=flattened_level_start_index, return_query_only=True)
                attn_index += 1
            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index+=1
            elif layer == "ffn":
                query = self.ffns[ffn_index](query, None)
                ffn_index += 1
        return query