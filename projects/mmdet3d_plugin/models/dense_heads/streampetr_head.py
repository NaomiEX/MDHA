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
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from mmcv.cnn import Linear, bias_init_with_prob

from mmcv.runner import force_fp32, get_dist_info
from mmcv.cnn.bricks.plugin import build_plugin_layer
from mmdet.core import (build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.utils import build_transformer
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox, clamp_to_rot_range
from ..utils.lidar_utils import normalize_lidar

from mmdet.models.utils import NormedLinear
from projects.mmdet3d_plugin.models.utils.positional_encoding import pos2posemb3d, pos2posemb1d, nerf_positional_encoding
from projects.mmdet3d_plugin.models.utils.misc import MLN, topk_gather, transform_reference_points, memory_refresh, SELayer_Linear
from projects.mmdet3d_plugin.models.utils.lidar_utils import denormalize_lidar, clamp_to_lidar_range

@HEADS.register_module()
class StreamPETRHead(AnchorFreeHead):
    """Implements the DETR transformer head.
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """
    _version = 2

    def __init__(self,
                 num_classes,
                 in_channels=256,
                 embed_dims=256,
                 num_query=100,
                 num_reg_fcs=2,
                 memory_len=1024,
                 topk_proposals=256,
                 num_propagated=256,
                 with_dn=True,
                 with_ego_pos=True,
                 match_with_velo=True,
                 match_costs=None,
                 transformer=None,
                 sync_cls_avg_factor=False,
                 code_weights=None,
                 bbox_coder=None,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 train_cfg=dict(
                     assigner=dict(
                         type='HungarianAssigner3D',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(
                             type='IoUCost', iou_mode='giou', weight=2.0)),),
                 test_cfg=dict(max_per_img=100),
                 depth_step=0.8,
                 depth_num=64,
                 LID=False,
                 depth_start = 1,
                 position_range=[-65, -65, -8.0, 65, 65, 8.0],
                 scalar = 5,
                 noise_scale = 0.4,
                 noise_trans = 0.0,
                 dn_weight = 1.0,
                 split = 0.5,
                 init_cfg=None,
                 normedlinear=False,

                 ## new params
                 init_ref_pts = False, # TODO: TRY THESE TWO
                 init_pseudo_ref_pts=False,
                 pos_embed3d=None,
                 use_spatial_alignment=False,
                 use_own_reference_points=False,
                 two_stage=False,
                 mlvl_feats_format=0,
                 pc_range=None,
                 skip_first_frame_self_attn=False,
                 init_pseudo_ref_pts_from_encoder_out=False,
                 use_inv_sigmoid=True,
                 mask_pred_target=False,
                 encode_3d_ref_pts_as_query_pos=True,
                 use_sigmoid_on_attn_out=False,
                 refine_all=False,
                 rot_post_process=None,
                 wlhclamp=None,
                 **kwargs):
        self._iter = 0
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.code_weights = self.code_weights[:self.code_size]

        if match_costs is not None:
            self.match_costs = match_costs
        else:
            self.match_costs = self.code_weights

        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is StreamPETRHead):
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']


            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.memory_len = memory_len
        self.topk_proposals = topk_proposals
        self.num_propagated = num_propagated
        self.with_dn = with_dn
        self.with_ego_pos = with_ego_pos
        self.match_with_velo = match_with_velo
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.embed_dims = embed_dims
        if not two_stage:
            self.depth_step = depth_step
            self.depth_num = depth_num
            self.position_dim = depth_num * 3
            self.LID = LID
            self.depth_start = depth_start

        self.use_pos_embed3d=pos_embed3d is not None
        if self.use_pos_embed3d:
            self.pos_embed3d=build_plugin_layer(pos_embed3d)[1]
        
        self.use_spatial_alignment=use_spatial_alignment
        self.use_own_reference_points=use_own_reference_points

        self.scalar = scalar
        self.bbox_noise_scale = noise_scale
        self.bbox_noise_trans = noise_trans
        self.dn_weight = dn_weight
        self.split = split 
        self.skip_first_frame_self_attn=skip_first_frame_self_attn
        self.init_pseudo_ref_pts_from_encoder_out=init_pseudo_ref_pts_from_encoder_out


        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        # self.act_cfg=transformer.get('act_cfg', dict(type="ReLU", inplace=False))
        self.normedlinear = normedlinear
        
        transformer['two_stage'] = two_stage
        transformer['decoder']['two_stage'] = two_stage
        transformer['decoder']['pc_range'] = pc_range
        transformer['decoder']['use_sigmoid_on_attn_out'] = use_sigmoid_on_attn_out

        self.num_decoder_layers = transformer['decoder']['num_layers']
        self.two_stage=two_stage
        # self.encode_query_with_ego_pos=encode_query_with_ego_pos
        transformer['decoder']['use_inv_sigmoid'] = use_inv_sigmoid
        transformer['decoder']['mask_pred_target'] = mask_pred_target
        if train_cfg is None:
            print("DECODER: !! IN TEST MODE !!")
            for attn_cfg in transformer['decoder']['transformerlayers']['attn_cfgs']:
                if attn_cfg['type'] == 'CustomDeformAttn':
                    attn_cfg['test_mode'] = True
        super(StreamPETRHead, self).__init__(num_classes, in_channels, init_cfg = init_cfg)

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self.transformer = build_transformer(transformer)

        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights), requires_grad=False)

        self.match_costs = nn.Parameter(torch.tensor(
            self.match_costs), requires_grad=False)

        self.bbox_coder = build_bbox_coder(bbox_coder)

        # self.pc_range = nn.Parameter(torch.tensor(
        #     pointcloud_range), requires_grad=False)
        self.pc_range = nn.Parameter(torch.tensor(pc_range), requires_grad=False)
        
        # self.pointcloud_range = nn.Parameter(torch.tensor(
        #     pointcloud_range), requires_grad=False)

        self.position_range = nn.Parameter(torch.tensor(
            position_range), requires_grad=False)
        
        # if self.use_spatial_alignment or self.use_pos_embed3d:
        #     if self.LID:
        #         index  = torch.arange(start=0, end=self.depth_num, step=1).float()
        #         index_1 = index + 1
        #         bin_size = (self.position_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
        #         coords_d = self.depth_start + bin_size * index * index_1
        #     else:
        #         index  = torch.arange(start=0, end=self.depth_num, step=1).float()
        #         bin_size = (self.position_range[3] - self.depth_start) / self.depth_num
        #         coords_d = self.depth_start + bin_size * index

        #     self.coords_d = nn.Parameter(coords_d, requires_grad=False) # Tensor[self.depth_num] = Tensor[64]

        self.init_ref_pts = init_ref_pts
        self.init_pseudo_ref_pts = init_pseudo_ref_pts
        self.mlvl_feats_format=mlvl_feats_format
        for i in range(self.transformer.decoder.num_layers):
            self.transformer.decoder.layers[i].skip_first_frame_self_attn = skip_first_frame_self_attn
        self.use_inv_sigmoid=use_inv_sigmoid
        self.mask_pred_target=mask_pred_target
        self.encode_3d_ref_pts_as_query_pos=encode_3d_ref_pts_as_query_pos
        self.use_sigmoid_on_attn_out=use_sigmoid_on_attn_out
        self.refine_all=refine_all
        self.rot_post_process=rot_post_process
        self.wlhclamp=wlhclamp
        # self.transformer.decoder.use_inv_sigmoid=use_inv_sigmoid

        self._init_layers()
        self.reset_memory()

        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
            # cls_branch.append(nn.ReLU(inplace=False))
        if self.normedlinear:
            cls_branch.append(NormedLinear(self.embed_dims, self.cls_out_channels))
        else:
            cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        num_branches = self.num_decoder_layers
        self.cls_branches = nn.ModuleList(
            [deepcopy(fc_cls) for _ in range(num_branches)])
        self.reg_branches = nn.ModuleList(
            [deepcopy(reg_branch) for _ in range(num_branches)])

        ## initialize reference points to be equally spaced points across pc range
        if self.init_ref_pts:
            print("warning:init ref pts")
            n_x, n_y, n_z = 23, 14, 2
            assert n_x*n_y*n_z == self.num_query, f"{[n_x, n_y, n_z]} do not multiply to {self.num_query}"
            pc_range = self.pc_range.tolist()
            step_x = (pc_range[3]-(pc_range[0]+4))/n_x
            step_y = (pc_range[4]-(pc_range[1]+7))/n_y
            step_z = (pc_range[5]-(pc_range[2]+3))/n_z
            x = torch.arange(pc_range[0]+4, pc_range[3], step=step_x, dtype=torch.float32)
            y = torch.arange(pc_range[1]+7, pc_range[4], step=step_y, dtype=torch.float32)
            z = torch.arange(pc_range[2]+3, pc_range[5], step=step_z, dtype=torch.float32)
            xm, ym, zm = torch.meshgrid([x, y, z])
            xm=xm.reshape(-1).unsqueeze(-1)
            ym=ym.reshape(-1).unsqueeze(-1)
            zm=zm.reshape(-1).unsqueeze(-1)
            # init_ref_pts_weight = torch.cat([xm, ym, zm], dim=-1).sigmoid()
            init_ref_pts_weight = normalize_lidar(torch.cat([xm, ym, zm], dim=-1), self.pc_range)
            print(f"num_query: {self.num_query} pc_range: {pc_range}")
            assert init_ref_pts_weight.shape == self.reference_points.weight.shape, f"init_ref_pts_weight shape: {init_ref_pts_weight.shape} != self.reference_points weight shape: {self.reference_points.weight.shape}"
            self.reference_points.weight = nn.Parameter(init_ref_pts_weight)

        ## initialize pseudo reference points to be equally spaced points across pc range
        if not self.skip_first_frame_self_attn and not self.init_pseudo_ref_pts_from_encoder_out and self.init_pseudo_ref_pts:
            print("warning: init pseudo ref pts")
            n_x, n_y, n_z = 16, 8, 2
            assert n_x*n_y*n_z == self.num_propagated, f"{[n_x, n_y, n_z]} do not multiply to {self.num_propagated}"
            pc_range = self.pc_range.tolist()
            
            step_x = (pc_range[3]-(pc_range[0]+6))/n_x
            step_y = (pc_range[4]-(pc_range[1]+12.5))/n_y
            step_z = (pc_range[5]-(pc_range[2]+4))/n_z
            x = torch.arange(pc_range[0]+6, pc_range[3], step=step_x, dtype=torch.float32)
            y = torch.arange(pc_range[1]+12.5, pc_range[4], step=step_y, dtype=torch.float32)
            z = torch.arange(pc_range[2]+4, pc_range[5], step=step_z, dtype=torch.float32)
            xm, ym, zm = torch.meshgrid([x, y, z])
            xm=xm.reshape(-1).unsqueeze(-1)
            ym=ym.reshape(-1).unsqueeze(-1)
            zm=zm.reshape(-1).unsqueeze(-1)
            # pseudo_init_ref_pts_weights = torch.cat([xm, ym, zm], dim=-1).sigmoid()
            pseudo_init_ref_pts_weights = normalize_lidar(torch.cat([xm, ym, zm], dim=-1), self.pc_range)
            assert pseudo_init_ref_pts_weights.shape == self.pseudo_reference_points.weight.shape
            self.pseudo_reference_points.weight = nn.Parameter(pseudo_init_ref_pts_weights)
        
        # print("-----self.reference_points.weight-----")
        # print(self.reference_points.weight)
        # print("-----self.pseudo_reference_points.weight-----")
        # print(self.pseudo_reference_points.weight)
        # add bbox embed for iterative bbox refinement
        # self.transformer.decoder.bbox_embed = deepcopy(self.reg_branches)

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        # if self.use_pos_embed3d:
        #     self.position_encoder = nn.Sequential(
        #             nn.Linear(self.position_dim, self.embed_dims*4),
        #             nn.ReLU(),
        #             nn.Linear(self.embed_dims*4, self.embed_dims),
        #         )

        self.memory_embed = nn.Sequential(
                nn.Linear(self.in_channels, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims),
            )
        
        if self.use_pos_embed3d:
            # can be replaced with MLN
            self.featurized_pe = SELayer_Linear(self.embed_dims)
        if self.use_own_reference_points:
            self.reference_points = nn.Embedding(self.num_query, 3)
        if self.num_propagated > 0 and not self.skip_first_frame_self_attn and not self.init_pseudo_ref_pts_from_encoder_out:
            self.pseudo_reference_points = nn.Embedding(self.num_propagated, 3)

        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims*3//2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        if self.use_spatial_alignment:
            self.spatial_alignment = MLN(8)

        self.time_embedding = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims)
        )

        # encoding ego pose
        if self.with_ego_pos:
            self.ego_pose_pe = MLN(180)
            self.ego_pose_memory = MLN(180)

        

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # TODO: TRY TO INITIALIZE IT AS THE TOP K MOST COMMON OBJECT (X,Y,Z)
        # The initialization for transformer is important
        if self.use_own_reference_points and not self.init_ref_pts:
            nn.init.uniform_(self.reference_points.weight.data, 0, 1)
        if hasattr(self, "pseudo_reference_points"):
            if not self.init_pseudo_ref_pts:
                nn.init.uniform_(self.pseudo_reference_points.weight.data, 0, 1)
            self.pseudo_reference_points.weight.requires_grad = False

        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

       
    def reset_memory(self):
        self.memory_embedding = None
        self.memory_reference_point = None
        self.memory_timestamp = None
        self.memory_egopose = None
        self.memory_velo = None
        if self.refine_all:
            self.prev_wlhrot=None

    def pre_update_memory(self, data, init_pseudo_ref_pts=None):
        x = data['prev_exists'] # Tensor[B]
        B = x.size(0)
        # refresh the memory when the scene changes
        if self.memory_embedding is None:
            self.memory_embedding = x.new_zeros(B, self.memory_len, self.embed_dims)
            self.memory_reference_point = x.new_zeros(B, self.memory_len, 3)
            self.memory_timestamp = x.new_zeros(B, self.memory_len, 1)
            self.memory_egopose = x.new_zeros(B, self.memory_len, 4, 4)
            self.memory_velo = x.new_zeros(B, self.memory_len, 2)
            if self.refine_all:
                self.prev_wlhrot = x.new_zeros(B, self.num_propagated, 5)
        else:
            self.memory_timestamp += data['timestamp'].unsqueeze(-1).unsqueeze(-1)
            self.memory_egopose = data['ego_pose_inv'].unsqueeze(1) @ self.memory_egopose
            self.memory_reference_point = transform_reference_points(self.memory_reference_point, data['ego_pose_inv'], reverse=False)
            self.memory_timestamp = memory_refresh(self.memory_timestamp[:, :self.memory_len], x)
            self.memory_reference_point = memory_refresh(self.memory_reference_point[:, :self.memory_len], x)
            self.memory_embedding = memory_refresh(self.memory_embedding[:, :self.memory_len], x)
            self.memory_egopose = memory_refresh(self.memory_egopose[:, :self.memory_len], x)
            self.memory_velo = memory_refresh(self.memory_velo[:, :self.memory_len], x)
            if self.refine_all:
                self.prev_wlhrot = memory_refresh(self.prev_wlhrot, x)
        if x.sum() > 0:
            prev_exists_mask = x.bool()
            # ensure that for data which has a prev, that memory is not empty
            assert (self.memory_reference_point[prev_exists_mask].flatten(1) != 0).any(dim=1).all()
    
        # for the first frame, padding pseudo_reference_points (non-learnable)
        if self.num_propagated > 0 and not self.skip_first_frame_self_attn:
            if self.init_pseudo_ref_pts_from_encoder_out:
                assert init_pseudo_ref_pts is not None
                pseudo_reference_points = init_pseudo_ref_pts
            else:
                pseudo_reference_points = self.pseudo_reference_points.weight * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3]
            self.memory_reference_point[:, :self.num_propagated]  = self.memory_reference_point[:, :self.num_propagated] + (1 - x).view(B, 1, 1) * pseudo_reference_points
            self.memory_egopose[:, :self.num_propagated]  = self.memory_egopose[:, :self.num_propagated] + (1 - x).view(B, 1, 1, 1) * torch.eye(4, device=x.device)

    def post_update_memory(self, data, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec, mask_dict):
        """
        Args:
            data (_type_): _description_
            rec_ego_pose (_type_): _description_
            all_cls_scores (_type_): [n_dec_layers, B, pad_size+Q+propagated, num_classes=10]
            all_bbox_preds (_type_): _description_
            outs_dec (_type_): _description_
            mask_dict (_type_): _description_
        """
        if self.training and mask_dict and mask_dict['pad_size'] > 0:
            rec_reference_points = all_bbox_preds[:, :, mask_dict['pad_size']:, :3][-1]
            rec_velo = all_bbox_preds[:, :, mask_dict['pad_size']:, -2:][-1]
            rec_memory = outs_dec[:, :, mask_dict['pad_size']:, :][-1]
            # [B, pad_size+Q+propagated, 1]
            rec_score = all_cls_scores[:, :, mask_dict['pad_size']:, :][-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
            rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)
            if self.refine_all: 
                rec_wlhrot = all_bbox_preds[:, :, mask_dict['pad_size']:, 3:-2][-1]
        else:
            rec_reference_points = all_bbox_preds[..., :3][-1]
            rec_velo = all_bbox_preds[..., -2:][-1]
            rec_memory = outs_dec[-1]
            rec_score = all_cls_scores[-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
            rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)
            if self.refine_all:
                rec_wlhrot = all_bbox_preds[..., 3:-2][-1]
        
        # topk proposals
        _, topk_indexes = torch.topk(rec_score, self.topk_proposals, dim=1)
        rec_timestamp = topk_gather(rec_timestamp, topk_indexes)
        rec_reference_points = topk_gather(rec_reference_points, topk_indexes).detach()
        rec_memory = topk_gather(rec_memory, topk_indexes).detach()
        rec_ego_pose = topk_gather(rec_ego_pose, topk_indexes)
        rec_velo = topk_gather(rec_velo, topk_indexes).detach()
        if self.refine_all:
            rec_wlhrot = topk_gather(rec_wlhrot, topk_indexes).detach()
            self.prev_wlhrot = rec_wlhrot
        self.memory_embedding = torch.cat([rec_memory, self.memory_embedding], dim=1)
        self.memory_timestamp = torch.cat([rec_timestamp, self.memory_timestamp], dim=1)
        self.memory_egopose= torch.cat([rec_ego_pose, self.memory_egopose], dim=1)
        self.memory_reference_point = torch.cat([rec_reference_points, self.memory_reference_point], dim=1)
        self.memory_velo = torch.cat([rec_velo, self.memory_velo], dim=1)
        self.memory_reference_point = transform_reference_points(self.memory_reference_point, data['ego_pose'], reverse=False)
        self.memory_timestamp -= data['timestamp'].unsqueeze(-1).unsqueeze(-1)
        self.memory_egopose = data['ego_pose'].unsqueeze(1) @ self.memory_egopose


    def position_embeding(self, data, locations, img_metas):
        """
        Args:
            data (_type_): dict:
                - img_feats: list of len n_features levels, with elements Tensor[B, N, C, h_i, w_i]
                - img_feats_flatten: concatenated flattened features Tensor[B, N, h_0*w_0+..., C]
            locations (_type_): flattened locations Tensor [B, N, h_0*w_0+..., 2]
            img_metas (_type_): _description_
        """
        eps = 1e-5
        # BN, H, W, _ = locations.shape
        B, N, HW, _ = data['img_feats_flatten'].shape
        # B = data['intrinsics'].size(0)

        # [B, N, 2]
        intrinsic = torch.stack([data['intrinsics'][..., 0, 0], data['intrinsics'][..., 1, 1]], dim=-1)
        intrinsic = torch.abs(intrinsic) / 1e3
        # [B, N, 1, 2] -> [B, N, HW, 2]
        intrinsic = intrinsic.unsqueeze(2).repeat(1, 1, HW, 1)
        # intrinsic = intrinsic.repeat(1, HW, 1).view(B, -1, 2) # [B, N*HW, 2]
        assert intrinsic.shape[:-1] == data['img_feats_flatten'].shape[:-1]

        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        locations[..., 0] = locations[..., 0] * pad_w
        locations[..., 1] = locations[..., 1] * pad_h

        D = self.coords_d.shape[0]

        assert list(locations.shape) == [B, N, HW, 2]
        locations = locations.detach().unsqueeze(3).repeat(1, 1, 1, D, 1) # [B, N, HW, 1, 2] -> [B, N, HW, D, 2]
        # topk_centers = topk_gather(locations, topk_indexes).repeat(1, 1, D, 1)
        # coords_d = self.coords_d.view(1, 1, D, 1).repeat(B, num_sample_tokens, 1 , 1)

        # [D] -> [1, 1, 1, D, 1] -> [B, N, HW, D, 1]
        coords_d = self.coords_d.view(1, 1, 1, D, 1).repeat(B, N, HW, 1, 1)
        # coords = torch.cat([topk_centers, coords_d], dim=-1)
        coords = torch.cat([locations, coords_d], dim=-1) # [B, N, HW, D, 3]
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1) # [B,N,HW,D,4]
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)

        coords = coords.unsqueeze(-1) # [B, N, HW, D, 4, 1]

        # ! WARNING: WORKAROUND, NEED TO DO THE INVERSE IN CPU 
        img2lidars=torch.inverse(data['lidar2img'].to('cpu')).to('cuda') # [B, N, 4, 4]
        # [B, N, 4, 4] -> [B, N, 1, 1, 4, 4] -> [B, N, HW, D, 4, 4]
        img2lidars = img2lidars[:, :, None, None].repeat(1, 1, HW, D, 1, 1) 
        # img2lidars=img2lidars[:,:,None,None, 4, 4].repeat(1, 1, HW, D, 1, 1) # [B, N, 1, 1, 4, 4]->[B, N, HW, D, 4, 4]
        # img2lidars = topk_gather(img2lidars, topk_indexes)

        # convert generated locations to lidar coords
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3] # [B, N, HW, D, 3]
        # normalize to range [0,1]
        coords3d[..., 0:3] = (coords3d[..., 0:3] - self.position_range[0:3]) / (self.position_range[3:6] - self.position_range[0:3])
        # coords3d = coords3d.reshape(B, -1, D*3) # [B, N*HW, D*3]
        coords3d = coords3d.reshape(B, N, HW, D*3)
      
        pos_embed  = inverse_sigmoid(coords3d) # convert back to R range
        coords_position_embeding = self.position_encoder(pos_embed) # [B,N,HW, C]
        # intrinsic = topk_gather(intrinsic, topk_indexes)

        # for spatial alignment in focal petr
        # intrinsic: [B,N,HW,2]
        # coords3d: [B,N,HW,D*3]
        cone = torch.cat([intrinsic, coords3d[..., -3:], coords3d[..., -90:-87]], dim=-1) # [B, N,HW, 8]

        return coords_position_embeding, cone

    def temporal_alignment(self, query_pos, tgt, reference_points, data, attn_mask, remaining_outs_init=None):
        """
        Args:
            query_pos (_type_): [B, pad_size+Q, 256]
            tgt (_type_): query init [B, pad_size+Q, 256]
            reference_points (_type_): [B, pad_size+Q, 3]
        """
        assert query_pos.shape == tgt.shape
        B = query_pos.size(0)
        
        # [B, pad_size+Q, 4, 4]
        rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, query_pos.size(1), 1, 1)
        
        if self.with_ego_pos:
            ## encode current query and query_pos with current ego motion
            # [B, pad_size+Q, 15]
            rec_ego_motion = torch.cat([torch.zeros_like(reference_points[...,:3]), rec_ego_pose[..., :3, :].flatten(-2)], dim=-1)
            rec_ego_motion = nerf_positional_encoding(rec_ego_motion) # [B, pad_size+Q, 180]
            # if self.encode_query_with_ego_pos:
            tgt = self.ego_pose_memory(tgt, rec_ego_motion)
            query_pos = self.ego_pose_pe(query_pos, rec_ego_motion) 

        query_pos += self.time_embedding(pos2posemb1d(torch.zeros_like(reference_points[...,:1])))

        prev_exists = data['prev_exists'] # Tensor[B]
        valid_mask = (prev_exists.long() | (not self.skip_first_frame_self_attn)).bool()
        # print(f"valid mask shape: {valid_mask}")

        if not self.skip_first_frame_self_attn:
            assert valid_mask.all()

        all_temp_memory = torch.zeros_like(self.memory_embedding) # [B, memory_len, 256]
        all_temp_pos = torch.zeros_like(self.memory_embedding) # [B, memory_len, 256]
        all_temp_reference_points = torch.zeros_like(self.memory_reference_point) # [B, memory_len, 3]

        if self.num_propagated > 0:
            # normalize memory ref pts
            # [B, memory_len, 3] in range[0,1]
            temp_reference_point = (self.memory_reference_point[valid_mask] - self.pc_range[:3]) / (self.pc_range[3:6] - self.pc_range[0:3])
            temp_pos = self.query_embedding(pos2posemb3d(temp_reference_point)) # [B, memory_len, 256]
            temp_memory = self.memory_embedding[valid_mask] # memory queries [B, memory_len, 256]
            if self.with_ego_pos:
                ## encode memory embedding and query pos with memory ego motion
                memory_ego_motion = torch.cat([self.memory_velo[valid_mask], self.memory_timestamp[valid_mask], 
                                               self.memory_egopose[valid_mask,..., :3, :].flatten(-2)], dim=-1).float()
                memory_ego_motion = nerf_positional_encoding(memory_ego_motion)
                temp_pos = self.ego_pose_pe(temp_pos, memory_ego_motion)
                temp_memory = self.ego_pose_memory(temp_memory, memory_ego_motion)
            temp_pos += self.time_embedding(pos2posemb1d(self.memory_timestamp[valid_mask]).float())

            all_temp_memory[valid_mask] = temp_memory
            all_temp_pos[valid_mask] = temp_pos
            all_temp_reference_points[valid_mask] = temp_reference_point
            if remaining_outs_init is not None:
                assert list(self.prev_wlhrot.shape) == [B, self.num_propagated, 5]
                assert self.prev_wlhrot.size(1) == self.memory_velo[:, :self.num_propagated].size(1)
                # [B, num_propagated, 7]
                prev_remaining_outs = torch.cat([self.prev_wlhrot, self.memory_velo[:, :self.num_propagated]], dim=-1)
                assert prev_remaining_outs.size(-1) == remaining_outs_init.size(-1)
                # [B, num_masked+Q+num_propagated, 7]
                remaining_outs_init = torch.cat([remaining_outs_init, prev_remaining_outs], 1)
            
        tgt = torch.cat([tgt, all_temp_memory[:, :self.num_propagated]], dim=1) # [B, Q + propagated, 256]
        query_pos = torch.cat([query_pos, all_temp_pos[:, :self.num_propagated]], dim=1)
        reference_points = torch.cat([reference_points, all_temp_reference_points[:, :self.num_propagated]], dim=1)
        # [B, Q + propagated, 4, 4]
        rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, query_pos.shape[1], 1, 1)
        temp_memory = all_temp_memory[:, self.num_propagated:]
        temp_pos = all_temp_pos[:, self.num_propagated:]

        if not valid_mask.all() and attn_mask is not None:
            invalid_mask = ~valid_mask
            # attn_mask: [padding + Q + propagated, padding + Q + memory_len]
            attn_mask = attn_mask.unsqueeze(0).repeat(B, 1, 1) # [B, padding + Q + propagated, padding + Q + memory_len]
            # since the attention mask is only used in self attention, 
            # if it's the first frame mask everything from that batch
            attn_mask[invalid_mask, :] = True # note: true = mask

        outs=[tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose, attn_mask]
        if remaining_outs_init is not None:
            outs += [remaining_outs_init]
        return outs

    def prepare_for_dn(self, batch_size, reference_points, img_metas):
        # reference_points: [B, Q, 3]
        if self.training and self.with_dn:
            # [num_obj_i, 3] (x,y,z) gravity center
            # [num_obj_i, 6] (w, l, h, rot, vx, vy)
            # list of len num_queue
            targets = [torch.cat((img_meta['gt_bboxes_3d']._data.gravity_center, 
                                  img_meta['gt_bboxes_3d']._data.tensor[:, 3:]),dim=1) 
                                  for img_meta in img_metas ]
            labels = [img_meta['gt_labels_3d']._data for img_meta in img_metas ]
            known = [(torch.ones_like(t)).cuda() for t in labels]
            know_idx = known
            unmask_bbox = unmask_label = torch.cat(known)
            #gt_num
            known_num = [t.size(0) for t in targets]
        
            labels = torch.cat([t for t in labels])
            boxes = torch.cat([t for t in targets])
            batch_idx = torch.cat([torch.full((t.size(0), ), i) for i, t in enumerate(targets)])
        
            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)
            # add noise
            known_indice = known_indice.repeat(self.scalar, 1).view(-1)
            known_labels = labels.repeat(self.scalar, 1).view(-1).long().to(reference_points.device)
            known_bid = batch_idx.repeat(self.scalar, 1).view(-1)
            known_bboxs = boxes.repeat(self.scalar, 1).to(reference_points.device)
            known_bbox_center = known_bboxs[:, :3].clone()
            known_bbox_scale = known_bboxs[:, 3:6].clone()

            if self.bbox_noise_scale > 0:
                diff = known_bbox_scale / 2 + self.bbox_noise_trans
                rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0
                known_bbox_center += torch.mul(rand_prob,
                                            diff) * self.bbox_noise_scale
                known_bbox_center[..., 0:3] = (known_bbox_center[..., 0:3] - self.pc_range[0:3]) / (self.pc_range[3:6] - self.pc_range[0:3])

                known_bbox_center = known_bbox_center.clamp(min=0.0, max=1.0)
                mask = torch.norm(rand_prob, 2, 1) > self.split
                known_labels[mask] = self.num_classes
            
            single_pad = int(max(known_num))
            pad_size = int(single_pad * self.scalar)
            padding_bbox = torch.zeros(pad_size, 3).to(reference_points.device)
            if reference_points.dim() == 2:
                padded_reference_points = torch.cat([padding_bbox, reference_points], dim=0).unsqueeze(0).repeat(batch_size, 1, 1)
            elif reference_points.dim() == 3: # [B, Q, 3]
                padding_bbox = padding_bbox[None].repeat(batch_size, 1, 1) # [B, pad_size, 3]
                padded_reference_points = torch.cat([padding_bbox, reference_points], dim=1) # [B, pad_size + Q, 3]
            else:
                raise ValueError(f"reference points shape is not supported: {reference_points.shape}")

            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(self.scalar)]).long()
            if len(known_bid):
                padded_reference_points[(known_bid.long(), map_known_indice)] = known_bbox_center.to(reference_points.device)

            tgt_size = pad_size + self.num_query
            attn_mask = torch.ones(tgt_size, tgt_size).to(reference_points.device) < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(self.scalar):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == self.scalar - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
             
            # update dn mask for temporal modeling
            query_size = pad_size + self.num_query + self.num_propagated
            tgt_size = pad_size + self.num_query + self.memory_len
            temporal_attn_mask = torch.ones(query_size, tgt_size).to(reference_points.device) < 0
            temporal_attn_mask[:attn_mask.size(0), :attn_mask.size(1)] = attn_mask 
            temporal_attn_mask[pad_size:, :pad_size] = True
            attn_mask = temporal_attn_mask

            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'know_idx': know_idx,
                'pad_size': pad_size
            }
            
        else:
            if reference_points.dim() == 2:
                padded_reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
            elif reference_points.dim() == 3:
                padded_reference_points = reference_points
            attn_mask = None
            mask_dict = None

        return padded_reference_points, attn_mask, mask_dict

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """load checkpoints."""
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since `AnchorFreeHead._load_from_state_dict` should not be
        # called here. Invoking the default `Module._load_from_state_dict`
        # is enough.

        # Names of some parameters in has been changed.
        version = local_metadata.get('version', None)
        if (version is None or version < 2) and self.__class__ is StreamPETRHead:
            convert_dict = {
                '.self_attn.': '.attentions.0.',
                # '.ffn.': '.ffns.0.',
                '.multihead_attn.': '.attentions.1.',
                '.decoder.norm.': '.decoder.post_norm.'
            }
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                for ori_key, convert_key in convert_dict.items():
                    if ori_key in k:
                        convert_key = k.replace(ori_key, convert_key)
                        state_dict[convert_key] = state_dict[k]
                        del state_dict[k]

        super(AnchorFreeHead,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)
    

    def forward(self, memory, locations, img_metas, topk_indexes=None,
                orig_spatial_shapes=None, flattened_spatial_shapes=None, 
                flattened_level_start_index=None, query_init=None, reference_points=None,
                init_pseudo_ref_pts=None, pos_init=None, pos_3d_init=None, pos_encoding_method=None,
                remaining_outs_init=None,
                **data):
        """Forward function.
        Args:
            memory: [B, N, h_0*w_0+..., 2] | [B, h0*N*w0+..., C]
            locations: flattened locations Tensor [B, N, h_0*w_0+..., 2] | [B, h0*N*w0+..., 2]
            img_metas
            topk_indexes: topk indices (from focal head NOT from encoder)
            orig_spatial_shapes: Tensor[n_levels, 2]
            flattened_spatial_shapes: [n_levels, 2]
            flattened_level_start_index: [n_levels]
            query_init: Tensor[B, Q, C]
            reference_points: 3d ref pts Tensor[B, Q, 3]
            data:
                - img_feats: list of len n_features levels, with elements Tensor[B, N, C, h_i, w_i]
                - img_feats_flatten: concatenated flattened features Tensor[B, N, h_0*w_0+..., C] | [B, h0*N*w0+..., C]
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        # zero init the memory bank
        self.pre_update_memory(data, init_pseudo_ref_pts)

        B=data['img_feats_flatten'].size(0)

        memory = self.memory_embed(memory) # [B, N, HW, C] | [B, h0*N*w0+..., C]

        if not self.two_stage:
            print("WARNING: not using two stage")
            assert self.mlvl_feats_format == 0
            # pos_embed: [B,N,HW,C]
            # cone: [B,N,HW,8]
            pos_embed, cone = self.position_embeding(data, locations, img_metas)

            # spatial_alignment in focal petr
            assert memory.shape[:-1] == cone.shape[:-1]
            memory = self.spatial_alignment(memory, cone) # [B, N, HW, C]
            pos_embed = self.featurized_pe(pos_embed, memory) # [B, N, HW, C]

            # TODO: CAN TRY AND USE CENTERS2D PREDICTION FROM FOCAL HEAD AS REFERENCE POINT INIT
            # TODO: BUT IF THIS IS IMPLEMENTED NEED TO TAKE THE QUERIES FROM THE ENCODER AS WELL (AND ENSURE CONSISTENCY)
            reference_points = self.reference_points.weight # [Nq, 3]
        else:
            reference_points = torch.clamp(reference_points.clone(),min=0.0, max=1.0)
            assert self.mlvl_feats_format == 1
        # reference_points: [B, pad_size+Nq, 3] normalized in lidar range [0,1]
        # attn_mask: [pad_size + Nq + num_propagated,  pad_size + Nq + memory_len] | None
        # mask_dict: dict | None
        reference_points, attn_mask, mask_dict = self.prepare_for_dn(B, reference_points, img_metas)
    
        if self.encode_3d_ref_pts_as_query_pos:
            query_pos = self.query_embedding(pos2posemb3d(reference_points)) # [B, pad_size+Nq, 256]
            if pos_init is not None:
                query_pos = query_pos + pos_init
            if pos_3d_init is not None:
                if self._iter == 0: print("using both 3d ref pts encoding and pos 3d init from encoder")
                assert pos_3d_init.size(1) == self.num_query
                if pos_encoding_method is not None:
                    if self._iter == 0: print("using shared mln for query pos encoding")
                    # ! WARNING ONLY WORKS WITH MLN METHOD
                    query_pos[:, -self.num_query:] = pos_encoding_method(query_pos[:,-self.num_query:].clone(), pos_3d_init)
                else:
                    if self._iter == 0: print("using addition for query pos encoding")
                    query_pos[:, -self.num_query:] = query_pos[:, -self.num_query:].clone() + pos_3d_init
        elif pos_init is not None:
            if mask_dict is not None and mask_dict['pad_size'] > 0:
                pad_query_pos = self.query_embedding(pos2posemb3d(reference_points[:, :mask_dict['pad_size']]))
                query_pos = torch.cat([pad_query_pos,pos_init], 1)
            else:
                query_pos = pos_init
        else:
            query_pos = reference_points.new_zeros([reference_points.shape[:-1], self.embed_dims])

        # query init
        if self.two_stage and mask_dict is not None:
            # [B, pad_size, C]
            zero_pad = query_pos.new_zeros([query_pos.size(0), mask_dict['pad_size'], query_pos.size(2)])
            # [B, pad_size+Q, C]
            tgt = torch.cat([zero_pad, query_init], dim=1)
            if self.refine_all:
                # [B, pad_size, 7]
                zero_pad = query_pos.new_zeros([query_pos.size(0), mask_dict['pad_size'], remaining_outs_init.size(-1)])
                remaining_outs_init = torch.cat([zero_pad, remaining_outs_init], dim=1) # [B, pad_size+Q, 7]
        elif self.two_stage:
            tgt=query_init # [B, Q, C]
            # print(f"\ntgt shape in decoder: {tgt.shape}\n")
        else:
            tgt = torch.zeros_like(query_pos)

        assert tgt.shape == query_pos.shape

        # prepare for the tgt and query_pos using mln.
        # tgt: current queries|prev queries Tensor[B, pad_size + Q + num_propagated, 256]
        # query_pos: query pos|prev query pos Tensor [B, pad_size + Q + num_propagated, 256]
        # reference_points: current ref pts|prev ref pts Tensor [B, pad_size + Q + num_propagated, 3]
        # temp_memory: rest of memory queue Tensor [B, memory_len - num_propagated, 256]
        # temp_pos: rest of query pos in queue Tensor [B, memory_len - num_propagated, 256]
        # rec_ego_pose: eye tensor [B, pad_size+Q+num_propagated*2, 4, 4]
        # attn_mask: [B, padding+Q+propagated, padding+Q+memory_len] | None
        temporal_alignment_out = self.temporal_alignment(query_pos, tgt, reference_points, data, attn_mask, 
                                                         remaining_outs_init=remaining_outs_init)
        tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose, attn_mask = temporal_alignment_out[:7]
        if self.refine_all:
            remaining_outs_init = temporal_alignment_out[7]

        # out_dec: [num_layers, B, Q, C]
        # out_ref_pts: [num_layers, B, Q, 3]
        # init_ref_pts: [B, Q, 3]
        outs_decoder = self.transformer(self.reg_branches, memory, tgt, query_pos, attn_mask, 
                                        pos_embed=pos_embed if not self.two_stage else None, 
                                        temp_memory=temp_memory, temp_pos=temp_pos,
                                        reference_points = reference_points.clone(), lidar2img=data['lidar2img'],
                                        extrinsics=data['extrinsics'], orig_spatial_shapes=orig_spatial_shapes, 
                                        flattened_spatial_shapes=flattened_spatial_shapes, 
                                        flattened_level_start_index=flattened_level_start_index,
                                        img_metas=img_metas, prev_exists=data['prev_exists'])
        # outs_dec: [num_dec_layers, B, Q, C]
        # out_ref_pts: sigmoided bbox predictions [num_dec_layers, B, Q, 3]
        # init_ref_pts: initial ref pts (in [0,1] range) [B, Q, 3]
        # NOTE: expecting all ref_pts to already be unnormalized
        outs_dec, out_ref_pts, init_ref_pts = outs_decoder[:3]
        if self.mask_pred_target:
            # sampling_locs_all: [B, num_dec_layers, Q, n_heads, n_levels, n_points, 2]
            # attn_weights_all: [B, num_dec_layers, Q, n_heads, n_levels, n_points]
            sampling_locs_all, attn_weights_all = outs_decoder[3:5]

        outs_dec = torch.nan_to_num(outs_dec)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(outs_dec.shape[0]):
            outputs_class = self.cls_branches[lvl](outs_dec[lvl]) # [B, Q, num_classes]
            out_coord_offset = self.reg_branches[lvl](outs_dec[lvl]) # [B, Q, 10]
            if self.use_sigmoid_on_attn_out:
                if self._iter == 0: print("DECODER: using sigmoid on attention out")
                out_coord_offset[..., :3] = F.sigmoid(out_coord_offset[..., :3])
                out_coord_offset[..., :3] = denormalize_lidar(out_coord_offset[..., :3], self.pc_range)
            out_coord = out_coord_offset
            if lvl == 0:
                out_coord[..., 0:3] += init_ref_pts[..., 0:3]
                if self.refine_all:
                    if self._iter < 10: print("refining all")
                    assert remaining_outs_init is not None
                    assert out_coord[..., 3:].shape == remaining_outs_init.shape
                    out_coord[..., 3:] += remaining_outs_init
            else:
                out_coord[..., 0:3] += out_ref_pts[lvl-1][..., 0:3]
                if self.refine_all:
                    out_coord[..., 3:] += outputs_coords[-1][..., 3:]
            

            outputs_classes.append(outputs_class)
            outputs_coords.append(out_coord)

        all_cls_scores = torch.stack(outputs_classes) # [n_dec_layers, B, Q, num_classes]
        all_bbox_preds = torch.stack(outputs_coords) # [n_dec_layers, B, Q, 10]
        
        ## wlh postprocessing
        if self._iter %50 == 0:
            invalid_wlh = all_bbox_preds[..., 3:6] < 0.0
            prop_invalid_wlh = invalid_wlh.sum() / invalid_wlh.numel()
            # print(f"StreamPETR: wlh out of range: {prop_invalid_wlh}")
        if self.wlhclamp =="abs":
            # if self._iter < 2: print("StreamPETR: using abs wlh clamp")
            all_bbox_preds[..., 3:6] = all_bbox_preds[..., 3:6].clone().abs()
        if self._iter %50 == 0:
            invalid_wlh = all_bbox_preds[..., 3:6] < 0.0
            prop_invalid_wlh = invalid_wlh.sum() / invalid_wlh.numel()
            # print(f"StreamPETR: wlh out of range: {prop_invalid_wlh}")

        ## center point post processing
        if self.use_inv_sigmoid:
            out_coord[..., 0:3] = out_coord[..., 0:3].sigmoid()
        if self.use_inv_sigmoid:
            assert (torch.logical_and(all_bbox_preds[..., :3] >= 0.0, all_bbox_preds[..., :3] <= 1.0)).all()
            all_bbox_preds[..., 0:3] = denormalize_lidar(all_bbox_preds[..., 0:3], self.pc_range)
        else:
            all_bbox_preds = clamp_to_lidar_range(all_bbox_preds, self.pc_range)
        
        if self.debug and self._iter % 20 == 0:
            invalid_rots = torch.logical_or(all_bbox_preds[..., 6:8] > 1.0, all_bbox_preds[..., 6:8] < -1.0)
            prop_invalid_rots = invalid_rots.sum() / invalid_rots.numel()
            print(f"StreamPETR: prop invalid rotations: {prop_invalid_rots}")
        if self.rot_post_process == "clamp":
            all_bbox_preds[..., 6:8] = clamp_to_rot_range(all_bbox_preds)
        
        ## check wlh
        # invalid_wlh = all_bbox_preds[..., 3:6] < 0.0
        # prop_invalid_wlh = invalid_wlh.sum() / invalid_wlh.numel()
        # print(f"StreamPETR: wlh out of range: {prop_invalid_wlh}")

        # update the memory bank
        self.post_update_memory(data, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec, mask_dict)

        if mask_dict and mask_dict['pad_size'] > 0:
            output_known_class = all_cls_scores[:, :, :mask_dict['pad_size'], :]
            output_known_coord = all_bbox_preds[:, :, :mask_dict['pad_size'], :]
            outputs_class = all_cls_scores[:, :, mask_dict['pad_size']:, :]
            outputs_coord = all_bbox_preds[:, :, mask_dict['pad_size']:, :]
            mask_dict['output_known_lbs_bboxes']=(output_known_class, output_known_coord)
            outs = {
                'all_cls_scores': outputs_class,
                'all_bbox_preds': outputs_coord,
                'dn_mask_dict':mask_dict,
            }
        else:
            outs = {
                'all_cls_scores': all_cls_scores,
                'all_bbox_preds': all_bbox_preds,
                'dn_mask_dict':None,
            }

        if self.mask_pred_target:
            # sampling_locs_all: [B, num_dec_layers, Q, n_heads, n_levels, n_points, 2]
            # attn_weights_all: [B, num_dec_layers, Q, n_heads, n_levels, n_points]
            outs.update({
                'all_sampling_locs_dec': sampling_locs_all,
                'all_attn_weights_dec': attn_weights_all,
            })
        self._iter += 1
        return outs
    
    def prepare_for_loss(self, mask_dict):
        """
        prepare dn components to calculate loss
        Args:
            mask_dict: a dict that contains dn information
        """
        output_known_class, output_known_coord = mask_dict['output_known_lbs_bboxes']
        known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
        map_known_indice = mask_dict['map_known_indice'].long()
        known_indice = mask_dict['known_indice'].long().cpu()
        batch_idx = mask_dict['batch_idx'].long()
        bid = batch_idx[known_indice]
        if len(output_known_class) > 0:
            output_known_class = output_known_class.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
            output_known_coord = output_known_coord.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
        num_tgt = known_indice.numel()
        return known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt


    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indexes for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indexes for each image.
                - neg_inds (Tensor): Sampled negative indexes for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler

        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                                gt_labels, gt_bboxes_ignore, self.match_costs, self.match_with_velo)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
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
        return (labels, label_weights, bbox_targets, bbox_weights, 
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list,
             gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list, 
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
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
                bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls, nan=1e-16, posinf=100.0, neginf=-100.0)
        loss_bbox = torch.nan_to_num(loss_bbox, nan=1e-16, posinf=100.0, neginf=-100.0)
        return loss_cls, loss_bbox

   
    def dn_loss_single(self,
                    cls_scores,
                    bbox_preds,
                    known_bboxs,
                    known_labels,
                    num_total_pos=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 3.14159 / 6 * self.split * self.split  * self.split ### positive rate
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        bbox_weights = torch.ones_like(bbox_preds)
        label_weights = torch.ones_like(known_labels)
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, known_labels.long(), label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(known_bboxs, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)

        bbox_weights = bbox_weights * self.code_weights

        
        loss_bbox = self.loss_bbox(
                bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls,nan=1e-16, posinf=100.0, neginf=-100.0)
        loss_bbox = torch.nan_to_num(loss_bbox, nan=1e-16, posinf=100.0, neginf=-100.0)
        
        return self.dn_weight * loss_cls, self.dn_weight * loss_bbox
    
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None):
        """"Loss function.
        Args:
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, 
            all_gt_bboxes_ignore_list)

        loss_dict = dict()

        # loss_dict['size_loss'] = size_loss
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        
        if preds_dicts['dn_mask_dict'] is not None:
            known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt = self.prepare_for_loss(preds_dicts['dn_mask_dict'])
            all_known_bboxs_list = [known_bboxs for _ in range(num_dec_layers)]
            all_known_labels_list = [known_labels for _ in range(num_dec_layers)]
            all_num_tgts_list = [
                num_tgt for _ in range(num_dec_layers)
            ]
            
            dn_losses_cls, dn_losses_bbox = multi_apply(
                self.dn_loss_single, output_known_class, output_known_coord,
                all_known_bboxs_list, all_known_labels_list, 
                all_num_tgts_list)
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(dn_losses_cls[:-1],
                                            dn_losses_bbox[:-1]):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                num_dec_layer += 1
                
        elif self.with_dn:
            dn_losses_cls, dn_losses_bbox = multi_apply(
                self.loss_single, all_cls_scores, all_bbox_preds,
                all_gt_bboxes_list, all_gt_labels_list, 
                all_gt_bboxes_ignore_list)
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1].detach()
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1].detach()     
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(dn_losses_cls[:-1],
                                            dn_losses_bbox[:-1]):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i.detach()     
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i.detach()     
                num_dec_layer += 1

        return loss_dict


    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        # List[B] with elem dict(bboxes, scores, labels)
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)

        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            # reduce z center prediction by half the depth because lidarinstancebbox z center is 0 whereas nuscenes is 0.5 (see NuScenesDataset get_ann_info())
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list
