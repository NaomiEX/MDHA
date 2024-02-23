# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
from copy import deepcopy
from mmcv.cnn import bias_init_with_prob
from mmcv.runner import force_fp32
from mmdet.core import (build_assigner, build_sampler, multi_apply,
                        reduce_mean, bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh)
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from projects.mmdet3d_plugin.models.utils.misc import draw_heatmap_gaussian, apply_center_offset, apply_ltrb
from mmdet.core import bbox_overlaps
from mmdet3d.models.utils import clip_sigmoid
import random

@HEADS.register_module()
class FocalHead(AnchorFreeHead):
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
    def __init__(self,
                 num_classes,
                 in_channels=256,
                 embed_dims=256,
                 stride=16,
                 use_hybrid_tokens=False,
                 train_ratio=1.0,
                 infer_ratio=1.0,
                 sync_cls_avg_factor=False,
                 loss_cls2d=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_centerness=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_bbox2d=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou2d=dict(type='GIoULoss', loss_weight=2.0),
                 loss_centers2d=dict(type='L1Loss', loss_weight=5.0),
                 train_cfg=dict(
                     assigner2d=dict(
                         type='HungarianAssigner2D',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                         centers2d_cost=dict(type='BBox3DL1Cost', weight=1.0))),
                 test_cfg=dict(max_per_img=100),
                 init_cfg=None,
                 num_layers=4,
                 strides=[4, 8, 16, 32],
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor

        if train_cfg:
            assert 'assigner2d' in train_cfg, 'assigner2d should be provided '\
                'when train_cfg is set.'
            assigner2d = train_cfg['assigner2d']

            self.assigner2d = build_assigner(assigner2d)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.embed_dims = embed_dims

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.stride=stride
        self.use_hybrid_tokens=use_hybrid_tokens
        self.train_ratio=train_ratio
        self.infer_ratio=infer_ratio

        self.n_layers = num_layers

        self.strides=strides

        super(FocalHead, self).__init__(num_classes, in_channels, init_cfg = init_cfg)

        self.loss_cls2d = build_loss(loss_cls2d)
        self.loss_bbox2d = build_loss(loss_bbox2d)
        self.loss_iou2d = build_loss(loss_iou2d)
        self.loss_centers2d = build_loss(loss_centers2d)
        self.loss_centerness = build_loss(loss_centerness)


        self._init_layers()

    def _init_layers(self):
        cls_conv = nn.Conv2d(self.embed_dims, self.num_classes, kernel_size=1)
        self.cls_all = nn.ModuleList([deepcopy(cls_conv) for _ in range(self.n_layers)])

        shared_reg= nn.Sequential(
                                 nn.Conv2d(self.in_channels, self.embed_dims, kernel_size=(3, 3), padding=1),
                                 nn.GroupNorm(32, num_channels=self.embed_dims),
                                 nn.ReLU(),)
        self.shared_reg_all = nn.ModuleList([deepcopy(shared_reg) for _ in range(self.n_layers)])

        shared_cls = nn.Sequential(
                                 nn.Conv2d(self.in_channels, self.embed_dims, kernel_size=(3, 3), padding=1),
                                 nn.GroupNorm(32, num_channels=self.embed_dims),
                                 nn.ReLU(),)
        self.shared_cls_all = nn.ModuleList([deepcopy(shared_cls) for _ in range(self.n_layers)])

        centerness = nn.Conv2d(self.embed_dims, 1, kernel_size=1)
        self.centerness = nn.ModuleList([deepcopy(centerness) for _ in range(self.n_layers)])

        ltrb = nn.Conv2d(self.embed_dims, 4, kernel_size=1)
        self.ltrb = nn.ModuleList([deepcopy(ltrb) for _ in range(self.n_layers)])

        center2d = nn.Conv2d(self.embed_dims, 2, kernel_size=1)
        self.center2d = nn.ModuleList([deepcopy(center2d) for _ in range(self.n_layers)])

        bias_init = bias_init_with_prob(0.01)
        for i in range(self.n_layers):
            nn.init.constant_(self.cls_all[i].bias, bias_init)
            nn.init.constant_(self.centerness[i].bias, bias_init)

    def forward(self, locations, src=None, **data):
        if src is None:
            src = data['img_feats']
        B, N = data['img'].shape[:2]

        # bs, n, c, h, w= src.shape
        # B, N, HW, C = src.shape
        # num_tokens = N*HW
        
        # focal sampling
        # if self.training:
        #     if self.use_hybrid_tokens:
        #         sample_ratio = random.uniform(0.2, 1.0)
        #     else:
        #         sample_ratio = self.train_ratio 
        #     num_sample_tokens = int(num_tokens * sample_ratio)
           
        # else:
        #     sample_ratio = self.infer_ratio
        #     num_sample_tokens = int(num_tokens * sample_ratio)

        enc_cls_scores=[]
        enc_bbox_preds=[]
        enc_pred_centers2d=[]
        enc_centerness=[]
        for lvl,lvl_feat in enumerate(src):

            x = lvl_feat.flatten(0, 1) # [B*N, C, H, W]

            cls_feat = self.shared_cls_all[lvl](x) # [BN, C, H, W]
            cls = self.cls_all[lvl](cls_feat) # [BN, n_classes, H, W]
            centerness = self.centerness[lvl](cls_feat) # [BN, 1, H, W]
            cls_logits = cls.permute(0,2,3,1).reshape(B*N,-1,self.num_classes) # [BN, HW, num_classes]
            centerness = centerness.permute(0,2,3,1).reshape(B*N,-1,1) # [BN, HW, 1]

            pred_bboxes = None
            pred_centers2d = None
        
            reg_feat = self.shared_reg_all[lvl](x) # [BN, C, H, W]
            ltrb = self.ltrb[lvl](reg_feat).permute(0,2,3,1).contiguous() # [BN, H, W, 4]
            ltrb = ltrb.sigmoid()
            centers2d_offset = self.center2d[lvl](reg_feat).permute(0,2,3,1).contiguous() # [BN, H, W, 2]

            centers2d = apply_center_offset(locations[lvl], centers2d_offset) # [BN, H, W, 2]
            bboxes = apply_ltrb(locations[lvl], ltrb) # [BN, H, W, 4]
            
            pred_bboxes = bboxes.view(B*N,-1,4) # [BN, HW, 4]
            pred_centers2d = centers2d.view(B*N,-1,2) # [BN, HW, 2]

            enc_cls_scores.append(cls_logits)
            enc_bbox_preds.append(pred_bboxes)
            enc_pred_centers2d.append(pred_centers2d)
            enc_centerness.append(centerness)

            # cls_score = cls_logits.topk(1, dim=2).values[..., 0].view(B, -1, 1)
            # sample_weight = cls_score.detach().sigmoid() * centerness.detach().view(bs,-1,1).sigmoid()
            # _, topk_indexes = torch.topk(sample_weight, num_sample_tokens, dim=1)

        outs = {
                'enc_cls_scores': enc_cls_scores, # list of len n_levels [BN, HW, num_classes]
                'enc_bbox_preds': enc_bbox_preds, # list of len n_levels [BN, HW, 4] in format (cx cy w h) in range [0,1]
                'pred_centers2d': enc_pred_centers2d, # list of len n_levels [BN, HW, 2] in format cx cy in range [0,1]
                'centerness':enc_centerness, # list of len n_levels [BN, HW, 1]
                # 'topk_indexes':topk_indexes,
                'topk_indexes':None,
            }

        return outs
    
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes2d_list, # list of len B where each element is a list of size n_cams=6 where each element is a Tensor [n_obj_i, 4]
             gt_labels2d_list, # list of len B where each element is a list of size n_cams=6 where each element is a Tensor [n_obj_i]
             centers2d, # list of len B where each element is a list of size n_cams=6 where each element is a Tensor [n_obj_i, 2]
             depths, # list of len B where each element is a list of size n_cams=6 where each element is a Tensor [n_obj_i]
             preds_dicts,
             img_metas, # tuple of len B where each element is an img_meta dict
             focal_layers,
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

        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        pred_centers2d = preds_dicts['pred_centers2d']
        centerness = preds_dicts['centerness']


        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        # list of len BN where each element is a LiDARInstance3DBoxes with Tensor [n_obj_i, ...]
        all_gt_bboxes2d_list = [bboxes2d for i in gt_bboxes2d_list for bboxes2d in i]
        all_gt_labels2d_list = [labels2d for i in gt_labels2d_list for labels2d in i]
        all_centers2d_list = [center2d for i in centers2d for center2d in i]
        all_depths_list = [depth for i in depths for depth in i]

        # to accomodate for multiscale
        all_gt_bboxes2d_list = [all_gt_bboxes2d_list for _ in range(self.n_layers)]
        all_gt_labels2d_list = [all_gt_labels2d_list for _ in range(self.n_layers)]
        all_centers2d_list = [all_centers2d_list for _ in range(self.n_layers)]
        all_depths_list = [all_depths_list for _ in range(self.n_layers)]

        all_img_metas_list = [img_metas for _ in range(self.n_layers)]
        
        # levels=list(range(self.n_layers))

        enc_loss_cls, enc_losses_bbox, enc_losses_iou, centers2d_losses, centerness_losses = \
            multi_apply(
                self.loss_single, enc_cls_scores, enc_bbox_preds, pred_centers2d, centerness,
                all_gt_bboxes2d_list, all_gt_labels2d_list, all_centers2d_list,
                all_depths_list, all_img_metas_list, focal_layers,
            )

        lvl = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i, \
            loss_centers2d_i, loss_centerness_i in zip(enc_loss_cls,enc_losses_bbox,enc_losses_iou,
                                                       centers2d_losses,centerness_losses):
            loss_dict[f'{lvl}.focal_head_loss_cls'] = loss_cls_i
            loss_dict[f'{lvl}.focal_head_loss_bbox'] = loss_bbox_i
            loss_dict[f'{lvl}.focal_head_loss_iou'] = loss_iou_i
            loss_dict[f'{lvl}.focal_head_centers2d_losses'] = loss_centers2d_i
            loss_dict[f'{lvl}.focal_head_centerness_losses'] = loss_centerness_i
            lvl+=1
        
        return loss_dict


    def loss_single(self,
                    cls_scores, # [BN, H_i*W_i, 10]
                    bbox_preds, # [BN, H_i*W_i, 4] in format (cx cy w h) in range[0,1]
                    pred_centers2d, # [BN, H_i*W_i, 2] in range[0,1]
                    centerness, # [BN, H_i*W_i, 1]
                    gt_bboxes_list,
                    gt_labels_list,
                    all_centers2d_list,
                    all_depths_list,
                    img_metas,
                    focal_layer,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single feature level

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
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        centers2d_preds_list = [pred_centers2d[i] for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list, centers2d_preds_list,
                                             gt_bboxes_list, gt_labels_list, all_centers2d_list,
                                             all_depths_list, img_metas, gt_bboxes_ignore_list)
        # labels_list: list of len BN with Tensor [n_pred,] 
        #       where tensors with matches have the matched gt label, otherwise 10
        # label_weights_list: list of len BN with Tensor [n_pred] with all 1s
        # bbox_targets_list: list of len BN with Tensor [n_pred,4]  
        #       where tensors with matches have the matched gt bboxes in format (cx cy w h) in range [0,1], 
        #       otherwise 10
        # bbox_weights_list: list of len BN with Tensor [n_pred,4] 
        #       where matched bboxs are given 1 weightage and unmatched bbox preds are given 0 weightage
        # centers2d_targets_list: list of len BN with Tensor [n_pred, 2]
        #       where matched preds have the matched gt centers in range [0,1], otherwise 0.0
        # num_total_pos: total match indices across all BN
        # num_total_neg: total unmatched indices across all BN
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, centers2d_targets_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        labels = torch.cat(labels_list, 0) # [BN*n_pred]
        label_weights = torch.cat(label_weights_list, 0) # [BN*n_pred]
        bbox_targets = torch.cat(bbox_targets_list, 0) # [BN*n_pred, 4]
        bbox_weights = torch.cat(bbox_weights_list, 0) # [BN*n_pred, 4]
        centers2d_targets = torch.cat(centers2d_targets_list, 0) # [BN*n_pred, 2]


        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        # construct factors used for rescale bboxes
        img_h, img_w, _ = img_metas[0]['pad_shape'][0]

        factors = []

        for bbox_pred in  bbox_preds:
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1) # [H_i*W_i, 4]
            factors.append(factor)
        factors = torch.cat(factors, 0) # [BN*n_pred, 4]
        bbox_preds = bbox_preds.reshape(-1, 4) # [BN*n_pred, 4]
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors # (x1 y1 x2 y2) in range R
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors # (x1 y1 x2 y2) in range R

        # regression IoU loss, defaultly GIoU loss
        # averaged by number of total matches across BN
        loss_iou = self.loss_iou2d(bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos) # single val

        # iou score for each prediction
        iou_score = bbox_overlaps(bboxes_gt, bboxes, is_aligned=True).reshape(-1) # [BN*n_pred]

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels) # [BN*n_rped, 10]
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        # quality focal loss
        # averaged over number of matches
        loss_cls = self.loss_cls2d(
            cls_scores, (labels, iou_score.detach()), label_weights, avg_factor=cls_avg_factor)
        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

         #centerness BCE loss
        img_shape = [img_metas[0]['pad_shape'][0]] * num_imgs
        level = [focal_layer] * num_imgs
        # list of BN where the heatmap is a tensor of shape [H_i, W_i]
        # where a gaussian heatmap of the various 2d centers are located, otherwise they are just 0s
        (heatmaps, ) = multi_apply(self._get_heatmap_single, all_centers2d_list, 
                                   gt_bboxes_list, img_shape, level)
        
        heatmaps = torch.stack(heatmaps, dim=0) # [BN, H_i, W_i]
        centerness = clip_sigmoid(centerness) # take the sigmoid and clip to [1e-4, 1-1e-4]
        # gaussian focal loss
        loss_centerness = self.loss_centerness(
                centerness, # [BN, H_i*W_i, 1]
                heatmaps.view(num_imgs, -1, 1), # [BN, H_i*W_i, 1]
                avg_factor=max(num_total_pos, 1))

        # regression L1 loss
        loss_bbox = self.loss_bbox2d(
            bbox_preds, # [BN*n_pred, 4]
            bbox_targets, # [BN*n_pred, 4]
            bbox_weights, # where matched bboxs are given 1 weightage and unmatched bbox preds are given 0 weightage 
            avg_factor=num_total_pos)

        pred_centers2d = pred_centers2d.view(-1, 2) # [BN*n_pred, 2]
        # centers2d L1 loss
        loss_centers2d = self.loss_centers2d(
            pred_centers2d, 
            centers2d_targets, # [BN*n_pred, 2]
            bbox_weights[:, 0:2], # [BN*n_pred, 2] where matched bboxs are given 1 weightage and unmatched bboxs are given 0 weightage
            avg_factor=num_total_pos)
        loss_cls=torch.nan_to_num(loss_cls)
        loss_bbox=torch.nan_to_num(loss_bbox)
        loss_iou=torch.nan_to_num(loss_iou)
        loss_centers2d=torch.nan_to_num(loss_centers2d)
        loss_centerness=torch.nan_to_num(loss_centerness)

        return loss_cls, loss_bbox, loss_iou, loss_centers2d, loss_centerness

    def _get_heatmap_single(self, 
                            obj_centers2d, # [n_obj_i, 2]
                            obj_bboxes, # [n_obj_i, 4]
                            img_shape, # [704, 256]
                            lvl):
        
        img_h, img_w, _ = img_shape
        # zero tensor of shape [H_i, W_i]
        heatmap = torch.zeros(img_h // self.strides[lvl], img_w // self.strides[lvl], device=obj_centers2d.device)
        if len(obj_centers2d) != 0:
            # get left, top, right, bottom
            l = obj_centers2d[..., 0:1] - obj_bboxes[..., 0:1]
            t = obj_centers2d[..., 1:2] - obj_bboxes[..., 1:2]
            r = obj_bboxes[..., 2:3] - obj_centers2d[..., 0:1]
            b = obj_bboxes[..., 3:4] - obj_centers2d[..., 1:2]
            bound = torch.cat([l, t, r, b], dim=-1)
            radius = torch.ceil(torch.min(bound, dim=-1)[0] / 16)
            radius = torch.clamp(radius, 1.0).cpu().numpy().tolist()
            for center, r in zip(obj_centers2d, radius):
                heatmap = draw_heatmap_gaussian(heatmap, center / 16, radius=int(r), k=1)
        return (heatmap, )

    def get_targets(self,
                    ## predictions
                    cls_scores_list, # list of len BN each with Tensor [H_i*W_i, 10]
                    bbox_preds_list, # list of len BN each with Tensor [H_i*W_i, 4]
                    centers2d_preds_list, # list of len BN each with Tensor [H_i*W_i, 2]
                    ## ground truth
                    gt_bboxes_list, # list of len BN each wth Tensor [n_obj_i, 4]
                    gt_labels_list, # list of len BN each with Tensor [n_obj_i]
                    all_centers2d_list, # list of len BN each with Tensor [n_obj_i, 2]
                    all_depths_list, # list of len BN each with Tensor [n_obj_i]
                    img_metas, # list of len B with dict elements
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
            img_metas (list[dict]): List of image meta information.
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
        num_imgs = len(cls_scores_list) # BN
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]
        img_meta = {'pad_shape':img_metas[0]['pad_shape'][0]}
        img_meta_list = [img_meta for _ in range(num_imgs)]
        # labels_list: list of len BN with Tensor [n_pred,] 
        #       where tensors with matches have the matched gt label, otherwise 10
        # label_weights_list: list of len BN with Tensor [n_pred] with all 1s
        # bbox_targets_list: list of len BN with Tensor [n_pred,4]  
        #       where tensors with matches have the matched gt bboxes in format (cx cy w h) in range [0,1], 
        #       otherwise 10
        # bbox_weights_list: list of len BN with Tensor [n_pred,4] 
        #       where matched bboxs are given 1 weightage and unmatched bbox preds are given 0 weightage
        # centers2d_targets_list: list of len BN with Tensor [n_pred, 2]
        #       where matched preds have the matched gt centers in range [0,1], otherwise 0.0
        # pos_inds_list: list of len BN with the match indices Tensor[n_obj_i]
        # neg_inds_list: list of len BN with the unmatched indices Tensor[n_obj_i]
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         centers2d_targets_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list, centers2d_preds_list,
            gt_bboxes_list, gt_labels_list, all_centers2d_list,
            all_depths_list, img_meta_list, gt_bboxes_ignore_list)
        
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
                centers2d_targets_list, num_total_pos, num_total_neg)

    def _get_target_single(self,
                           cls_score, # [H_i*W_i, 10]
                           bbox_pred, # [H_i*W_i, 4]
                           pred_centers2d, # [H_i*W_i, 2]

                           gt_bboxes, # [n_obj_i, 4]
                           gt_labels, # [n_obj_i]
                           centers2d, # [n_obj_i, 2]
                           depths, 
                           img_meta, # single dict
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
            img_meta (dict): Meta information for one image.
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
        assign_result = self.assigner2d.assign(bbox_pred, cls_score, pred_centers2d, gt_bboxes,
                                               gt_labels, centers2d, img_meta, gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        pos_inds = sampling_result.pos_inds # [n_obj_i]
        neg_inds = sampling_result.neg_inds # [n_preds-n_obj_i]

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds].long()
        label_weights = gt_bboxes.new_ones(num_bboxes)


        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta['pad_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        # [n_obj_i, 4]
        pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
        # [n_obj_i, 4] in format (cx cy w h) in range [0,1]
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized) 
        bbox_targets[pos_inds] = pos_gt_bboxes_targets

        #centers2d target
        centers2d_targets = bbox_pred.new_full((num_bboxes, 2), 0.0, dtype=torch.float32)
        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert sampling_result.pos_assigned_gt_inds.numel() == 0
            centers2d_labels = torch.empty_like(gt_bboxes).view(-1, 2)
        else:
            # gt centers2d in the correct order
            # [n_obj_i, 2]
            centers2d_labels = centers2d[sampling_result.pos_assigned_gt_inds.long(), :]
        # [n_obj_i, 2] in range [0,1]
        centers2d_labels_normalized = centers2d_labels / factor[:, 0:2]
        centers2d_targets[pos_inds] = centers2d_labels_normalized
        return (labels, label_weights, bbox_targets, bbox_weights, centers2d_targets,
                pos_inds, neg_inds)