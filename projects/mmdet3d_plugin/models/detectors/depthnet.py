import torch
import pickle
from copy import deepcopy
from torch import nn
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmcv.cnn import xavier_init
from mmcv.runner.base_module import BaseModule
from mmdet.core import multi_apply, reduce_mean
from mmdet.models import build_loss
from projects.mmdet3d_plugin.constants import *
from ..utils.debug import *

@PLUGIN_LAYERS.register_module()
class DepthNet(BaseModule):
    def __init__(self, in_channels, depth_net_type="conv", depth_start=1.0, depth_max=61.2,
                 depth_pred_position=0, mlvl_feats_format=None, n_levels=4, loss_depth=None,
                 depth_weight_bound=False, depth_weight_limit=0.01, use_focal=True, equal_focal=100,
                 single_target=False,
                 **kwargs):
        super().__init__()
        self.in_channels=in_channels
        self.depth_pred_position=depth_pred_position
        self.mlvl_feats_format=mlvl_feats_format
        assert self.mlvl_feats_format == 1
        self.depth_net_type = depth_net_type.lower()
        self.sigmoid_out = kwargs.get("sigmoid_out", False)
        self.depth_start=depth_start
        self.depth_max=depth_max
        self.depth_range=depth_max - depth_start
        self.single_target=single_target
        self.n_levels=n_levels
        self.use_focal=use_focal
        if use_focal:
            self.equal_focal=equal_focal
        mid_channels=kwargs.get("mid_channels",self.in_channels)
        if self.depth_net_type == "conv":
            depth_branch = [
                nn.Conv2d(self.in_channels, mid_channels, kernel_size=(3,3), padding=1),
                nn.GroupNorm(32, num_channels=self.in_channels),
                nn.ReLU(),
                nn.Conv2d(self.in_channels, 1, kernel_size=1),
            ]
        else:
            raise ValueError(f"{self.depth_net_type} not supported")


        if self.sigmoid_out:
            assert self.use_focal is False
            depth_branch.append(nn.Sigmoid())
        
        net = nn.Sequential(*depth_branch)
        self.shared = kwargs.get("shared", False)
        if not self.shared:
            self.net = nn.ModuleList([deepcopy(net) for _ in range(self.n_levels)])
        else:
            self.net = net
        self.loss_depth=build_loss(loss_depth) if loss_depth is not None else None
        self.depth_weight_bound=depth_weight_bound
        self.depth_weight_limit=depth_weight_limit
        self.div_depth_loss_by_target_count=False
    
    # i dont think need to init
    def init_weights(self):
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init=True


    def get_lvl_depths(self, x, lvl):
        net = self.net[lvl] if not self.shared else self.net
        return net(x)

    def forward(self, 
                x, 
                focal=None, # [B, N]
                return_flattened=True,
                ):
        focal=focal.reshape(-1)
        all_pred_depths = []
        if self.depth_pred_position == DEPTH_PRED_BEFORE_ENCODER:
            for lvl, lvl_feat in enumerate(x[:self.n_levels]):
                # lvl_feat: [B, N, C, H_i, W_i]
                B, N = lvl_feat.shape[:2]
                inp_feats = lvl_feat.flatten(0,1) # [B*N, C, H_i, W_i]
                
                pred_depths = self.get_lvl_depths(inp_feats, lvl) # [B*N, 1, H_i, W_i]

                if self.sigmoid_out:
                    if do_debug_process(self): print("DEPTHNET: USING SIGMOID OUT")
                    # unnormalize output
                    pred_depths = pred_depths * self.depth_range + self.depth_start
                elif self.use_focal:
                    pred_depths = pred_depths.exp() * focal[..., None, None, None] / self.equal_focal
                    pred_depths = torch.clip(pred_depths.clone(), min=0.0, max=self.depth_max)
                if return_flattened:
                    # [B, N, 1, H_i, W_i] -> [B, H_i, N, W_i, 1] -> [B, H_i*N*W_i, 1]
                    pred_depths = pred_depths.unflatten(0,(B,N)).permute(0,3,1,4,2).flatten(1, 3)
                all_pred_depths.append(pred_depths)

            if return_flattened:
                all_pred_depths = torch.cat(all_pred_depths, 1) # [B, h0*N*w0+..., 1]
        else:
            raise NotImplementedError()
                
        return all_pred_depths
    
    def _get_target_single(self,
                           gt_bboxs_3d,
                           lidar2img=None,
                           lidar2cam=None,
                           orig_spatial_shapes=None,
                           ref_pts_2point5d_pred=None,
                           num_cameras=6):
        if gt_bboxs_3d.numel() == 0:
            # gt_bboxs_3d: [0, 9]
            all_depth_preds = ref_pts_2point5d_pred.new_tensor([])
            all_target_depths=gt_bboxs_3d.new_tensor([]) # shape: [0]
            all_weights=gt_bboxs_3d.new_tensor([])
            num_total_depth_pos=0
        else:
            gt_bbox_centers_3d= gt_bboxs_3d[..., :3].unsqueeze(0) # [1, n_objs, 3]
            def get_depth_weights(x, lvl=0):
                # exponential_decay
                lam = 10
                lam = lam / (lvl + 1)
                w = torch.exp(-x/lam)
                if self.depth_weight_bound:
                    mask=w < self.depth_weight_limit
                    if do_debug_process(self, repeating=True, interval=500):
                        print(f"masked proportion (lvl {lvl}): {mask.sum()/mask.numel()}")
                    w = w.masked_fill(mask, 0.0)
                return w
            n_levels=orig_spatial_shapes.size(0)
            # global_ref_pts: [n_matches, n_levels, 3]
            # ! WARNING: since a point can be projected to multiple cams, n_matches >= n_objs
            global_2p5d_pts_norm = self.projections.project_to_matching_2point5d_cam_points(
                gt_bbox_centers_3d, lidar2img.unsqueeze(0), lidar2cam.unsqueeze(0), 
                orig_spatial_shapes, num_cameras=num_cameras)
            if global_2p5d_pts_norm.size(0) == 0: # no matches
                all_depth_preds = ref_pts_2point5d_pred.new_tensor([])
                all_target_depths=gt_bboxs_3d.new_tensor([]) # shape: [0]
                all_weights=gt_bboxs_3d.new_tensor([])
                num_total_depth_pos=0
                return all_depth_preds, all_target_depths, all_weights, num_total_depth_pos
            flattened_spatial_shapes = orig_spatial_shapes.clone()
            flattened_spatial_shapes[:, 1] = flattened_spatial_shapes[:, 1] * num_cameras
            flattened_spatial_shapes_xy = torch.stack([flattened_spatial_shapes[..., 1], flattened_spatial_shapes[..., 0]], -1)
            global_2p5d_pts = global_2p5d_pts_norm.clone() # [n_matches, n_levels, 3]
            global_2p5d_pts[..., :2] = global_2p5d_pts[..., :2] * flattened_spatial_shapes_xy 
            if self.single_target:
                global_2p5d_pts[..., :2] = global_2p5d_pts[..., :2].floor()
            levels = [h_i*w_i for (h_i, w_i) in flattened_spatial_shapes]

            # List[n_levels] with elements Tensor[B, h_i*N*w_i, 2] containing the globally normalized (x,y) points
            split_grids_xy = torch.split(ref_pts_2point5d_pred[..., :2], levels, dim=0)
            all_targets = []
            all_dists=[]
            all_weights = []
            all_target_depths = []
            for lvl in range(n_levels):
                # ! WARNING: EXPECTING REF_PTS_2POINT5D_PRED TO BE (X,Y) NORMALIZED WHERE 1=w_i*num_cams, AND UNNORMALIZED DEPTH
                unnorm_pts = split_grids_xy[lvl] * flattened_spatial_shapes_xy[lvl] # [n_pred_lvl, 2]
                global_lvl_pts = global_2p5d_pts[:,lvl,:2] # [n_matches, 2]
                # get the distance between unnormalized xy points and gt bbox projected xy pts
                # [n_pred_lvl_i, 1, 2] - [1, n_matches, 2] = [n_pred_lvl_i, n_matches, 2] -> [n_pred_lvl_i, n_matches]
                l1dist_lvl = (unnorm_pts.unsqueeze(1) - global_lvl_pts[None]).abs().sum(-1)

                # for each prediction get the closest gt
                l1_dist, targets = l1dist_lvl.min(-1) # [n_matches]

                if self.single_target:
                    w = torch.zeros_like(l1_dist) # [n_matches]
                    w[targets] = 1.0

                else:
                    # get weights for each pred based on distance to gt, points far away from gt gets masked to 0 weight
                    w = get_depth_weights(l1_dist, lvl=lvl)

                # gets the closest gt corresponding to each pred
                target_depths = global_2p5d_pts[targets, lvl, 2]

                all_dists.append(l1_dist)
                all_targets.append(targets)
                all_weights.append(w)

                all_target_depths.append(target_depths)

            all_depth_preds = ref_pts_2point5d_pred[..., 2]
            all_targets = torch.cat(all_targets, 0)
            all_dists = torch.cat(all_dists, 0)
            all_weights = torch.cat(all_weights, 0)
            all_target_depths = torch.cat(all_target_depths, 0)

            assert all_target_depths.shape == all_depth_preds.shape

            if self.div_depth_loss_by_target_count:
                selected_targets, target_counts = torch.unique(all_targets, return_counts=True,sorted=True)
                full_target_counts = target_counts.new_zeros([global_2p5d_pts_norm.size(0)])
                full_target_counts[selected_targets] = target_counts
            else:
                full_target_counts=global_2p5d_pts_norm.size(0) * global_2p5d_pts_norm.size(1) # n_matches * n_levels
            
            if self.div_depth_loss_by_target_count:
                depth_weights = depth_weights / full_target_counts[all_targets]

                num_total_depth_pos = full_target_counts.numel()
            elif self.depth_weight_bound:
                num_total_depth_pos = full_target_counts
            else:
                num_total_depth_pos = ref_pts_2point5d_pred.numel()
        return all_depth_preds, all_target_depths, all_weights, num_total_depth_pos
    
    def get_targets(self,
                    gt_bboxes_list,
                    lidar2img_list=None,
                    lidar2cam_list=None,
                    orig_spatial_shapes_list=None,
                    ref_pts_2point5d_pred_list=None):
        all_depth_preds_list, depth_targets_list, depth_weights_list, num_valid_depths_list = multi_apply(
            self._get_target_single, gt_bboxes_list, lidar2img_list, lidar2cam_list,
            orig_spatial_shapes_list, ref_pts_2point5d_pred_list
        )
        num_total_valid_depths = sum(num_valid_depths_list)
        return all_depth_preds_list, depth_targets_list, depth_weights_list, num_total_valid_depths
    
    def loss_single(self,
                    gt_bboxes_list,
                    lidar2img=None,
                    lidar2cam=None,
                    orig_spatial_shapes=None,
                    ref_pts_2point5d_pred=None, # [B, h0*N*w0+...,, 3]
                    ):
        
        num_imgs = ref_pts_2point5d_pred.size(0)
        
        lidar2img_list = [lidar2img[i] for i in range(num_imgs)]
        lidar2cam_list = [lidar2cam[i] for i in range(num_imgs)]
        orig_spatial_shapes_list = [orig_spatial_shapes for _ in range(num_imgs)]
        ref_pts_2point5d_pred_list = [ref_pts_2point5d_pred[i] for i in range(num_imgs)]

        depth_pred_list, depth_targets_list, depth_weights_list, num_valid_depths=\
            self.get_targets(gt_bboxes_list, lidar2img_list, lidar2cam_list, 
                             orig_spatial_shapes_list, ref_pts_2point5d_pred_list)
        depth_preds = torch.cat(depth_pred_list, 0)
        depth_targets = torch.cat(depth_targets_list, 0)
        depth_weights = torch.cat(depth_weights_list, 0)
        num_total_valid_depths = gt_bboxes_list[0].new_tensor([num_valid_depths])
        num_total_valid_depths = torch.clamp(reduce_mean(num_total_valid_depths), min=1).item()
        # ! ENSURE BOTH depth_targets AND ref_pts_depth_pred ARE UNNORMALIZED

        if depth_preds.size(0) != depth_targets.size(0):
            out = dict(depth_preds=depth_preds, depth_targets=depth_targets)
            with open("./experiments/depth_coords_pred_target_unequal.pkl", "wb") as f:
                pickle.dump(out, f)
            raise Exception(f"depth_preds.size(0) != depth_targets.size(0)\ndepth_preds: {depth_preds}"
                            f"\ndepth_targets: {depth_targets}")

        avg_factor = num_total_valid_depths
        loss_depth = self.loss_depth(
            depth_preds, depth_targets, depth_weights, avg_factor=avg_factor
        )
        if loss_depth == 0.0:
            print(f"WARNING: LOSS DEPTH IS 0")
        loss_depth = torch.nan_to_num(loss_depth, nan=1e-16, posinf=100.0, neginf=-100.0)
        return loss_depth

    def loss(self, gt_bboxes_list, preds_dicts, 
             lidar2img=None, lidar2cam=None, orig_spatial_shapes=None):
        ref_pts_2point5d_pred = preds_dicts['ref_pts_2point5d_pred'] # [B, h0*N*w0+..., 3] NOTE: all 4 levels have the same (x,y) points since they are global, but the depth preds are diff
        device = ref_pts_2point5d_pred.device
        # list of len B where each element is a Tensor[num_gts, 9]
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]
        loss_dict=dict()
        loss_depth = self.loss_single(gt_bboxes_list, lidar2img=lidar2img, lidar2cam=lidar2cam,
                                      orig_spatial_shapes=orig_spatial_shapes,
                                      ref_pts_2point5d_pred=ref_pts_2point5d_pred)
        loss_dict['loss_depth'] = loss_depth
        return loss_dict