import torch
import pickle
from copy import deepcopy
from torch import nn
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmcv.cnn import xavier_init, Linear
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.resnet import BasicBlock
from mmdet.core import multi_apply, reduce_mean
from mmdet.models import build_loss
from ..utils.projections import project_to_matching_2point5d_cam_points

@PLUGIN_LAYERS.register_module()
class DepthNet(BaseModule):
    def __init__(self, in_channels, depth_net_type="mlp", depth_start=1.0, depth_max=61.2,
                 depth_pred_position=0, mlvl_feats_format=None, n_levels=4, loss_depth=None,
                 depth_weight_bound=False, depth_weight_limit=0.01, 
                 **kwargs):
        super().__init__()
        self.in_channels=in_channels
        self.depth_pred_position=depth_pred_position
        self.mlvl_feats_format=mlvl_feats_format
        assert self.mlvl_feats_format == 1
        self.depth_net_type = depth_net_type.lower()
        self.sigmoid_out = kwargs.get("sigmoid_out", True)
        self.depth_start=depth_start
        self.depth_range=depth_max - depth_start
        self.n_levels=n_levels
        mid_channels=kwargs.get("mid_channels",self.in_channels)
        if self.depth_net_type == "mlp":
            num_layers=kwargs.get("num_layers", 2)
            depth_branch = []
            for _ in range(num_layers):
                depth_branch.append(Linear(self.in_channels, self.in_channels))
                depth_branch.append(nn.ReLU(inplace=True))
            depth_branch.append(Linear(self.in_channels, 1))
        elif self.depth_net_type == "conv":
            depth_branch = [
                nn.Conv2d(self.in_channels, mid_channels, kernel_size=(3,3), padding=1),
                nn.GroupNorm(32, num_channels=self.in_channels),
                nn.ReLU(),
                nn.Conv2d(self.in_channels, 1, kernel_size=1),
            ]

        elif self.depth_net_type == "residual":
            block_additions = kwargs.get("block_additions", [])
            depth_branch = [
                BasicBlock(self.in_channels, mid_channels, stride=1),
            ]
            for a in block_additions:
                addition = a.lower()
                # assert addition in ["avg_pool", "fc", "conv"]
                if addition == "avg_pool":
                    depth_branch.append(nn.AdaptiveAvgPool2d((1, 1)))
                elif addition == "fc":
                    depth_branch += [
                        nn.Flatten(),
                        nn.Linear(mid_channels, 1)
                    ]
        else:
            raise ValueError(f"{self.depth_net_type} not supported")


        if self.sigmoid_out:
            depth_branch.append(nn.Sigmoid())
        
        net = nn.Sequential(*depth_branch)
        self.shared = kwargs.get("shared", True)
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

    def forward(self, x, flattened_spatial_shapes=None, orig_spatial_shapes=None, num_cams=6, 
                return_flattened=True):
        all_pred_depths = []
        if self.depth_pred_position == 0:
            assert isinstance(x, list)
            for lvl, lvl_feat in enumerate(x):
                # lvl_feat: [B, N, C, H_i, W_i]
                B, N = lvl_feat.shape[:2]
                if self.depth_net_type == "mlp":
                    inp_feats = lvl_feat.permute(0, 1, 3, 4, 2) # [B, N, H_i, W_i, C]
                elif self.depth_net_type in ["conv", "residual"]:
                    inp_feats = lvl_feat.flatten(0,1) # [B*N, C, H_i, W_i]
                pred_depths = self.get_lvl_depths(inp_feats, lvl) 
                # if mlp: [B, N, H_i, W_i, 1]
                # if conv, pred depths: [B*N, 1, H_i, W_i]

                if self.sigmoid_out:
                    # unnormalize output
                    pred_depths = pred_depths * self.depth_range + self.depth_start

                if return_flattened:
                    if self.depth_net_type == "mlp":
                        # [B, N, H_i, W_i, 1] -> [B, H_i, N, W_i, 1]
                        pred_depths = pred_depths.permute(0,2,1,3,4).flatten(1, 3)
                    elif self.depth_net_type in ["conv", "residual"]:
                        # [B, N, 1, H_i, W_i] -> [B, H_i, N, W_i, 1]
                        pred_depths = pred_depths.unflatten(0,(B,N)).permute(0,3,1,4,2).flatten(1, 3)
                    # all_pred_depths.append(pred_depths.flatten(1, 3))
                all_pred_depths.append(pred_depths)

            if return_flattened:
                all_pred_depths = torch.cat(all_pred_depths, 1) # [B, h0*N*w0+..., 1]
        else:
            # x: [B, p, C] (only for mlp & shared) | [B, H*N*W, C]
            if self.depth_net_type == "mlp":
                if not self.shared:
                    inp_feats = x.split((flattened_spatial_shapes[..., 0] * flattened_spatial_shapes[..., 1]).tolist(), dim=1)
                    all_pred_depths = [self.get_lvl_depths(lvl_feat, lvl) for lvl, lvl_feat in enumerate(inp_feats)]
                    if return_flattened:
                        all_pred_depths = torch.cat(all_pred_depths, 1)
                else:
                    all_pred_depths = self.get_lvl_depths(x, None)

                if self.sigmoid_out:
                    if return_flattened or self.shared:
                        all_pred_depths = all_pred_depths* self.depth_range + self.depth_start
                    else:
                        all_pred_depths = [d * self.depth_range + self.depth_start for d in all_pred_depths]
                
            elif self.depth_net_type in ["conv", "residual"]:
                # List[n_feat_levels] with elements [B, h_i*N*w_i, C]
                inp_feats = x.split((flattened_spatial_shapes[..., 0] * flattened_spatial_shapes[..., 1]).tolist(), dim=1)
                all_pred_depths = []
                for lvl, lvl_feat in enumerate(inp_feats):
                    B = lvl_feat.size(0)
                    # [B, h_i, N, w_i, C]
                    lvl_inp = lvl_feat.unflatten(1, [orig_spatial_shapes[lvl][0], num_cams, orig_spatial_shapes[lvl][1]])
                    lvl_inp = lvl_inp.permute(0, 2, 4, 1, 3).flatten(0,1) # [B, N, C, h_i, w_i] -> [B*N, C, h_i, w_i]
                    pred_depths = self.get_lvl_depths(lvl_inp, lvl) # [B*N, C, h_i, w_i]

                    if self.sigmoid_out:
                        # unnormalize output
                        pred_depths = pred_depths * self.depth_range + self.depth_start
                    if return_flattened:
                        # [B, N, C, h_i, w_i] -> [B, h_i, N, w_i, C] -> [B, h_i*N*w_i, C]
                        pred_depths=pred_depths.unflatten(0, (B, num_cams)).permute(0, 3, 1, 4, 2).flatten(1, 3)
                    all_pred_depths.append(pred_depths)
                if return_flattened:
                    all_pred_depths=torch.cat(all_pred_depths, 1) # [B, H*N*W, C]
                
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
                    w = w.masked_fill(mask, 0.0)
                return w
            n_levels=orig_spatial_shapes.size(0)
            cam_transformations=dict(lidar2img=lidar2img.unsqueeze(0), lidar2cam=lidar2cam.unsqueeze(0))
            # global_ref_pts: [n_matches, n_levels, 3]
            # ! WARNING: since a point can be projected to multiple cams, n_matches >= n_objs
            global_2p5d_pts_norm = project_to_matching_2point5d_cam_points(gt_bbox_centers_3d, cam_transformations, 
                                                                        orig_spatial_shapes, num_cameras=num_cameras)
            if global_2p5d_pts_norm.size(0) == 0: # no matches
                all_target_depths=gt_bboxs_3d.new_tensor([]) # shape: [0]
                all_weights=gt_bboxs_3d.new_tensor([])
                num_total_depth_pos=0
                return all_target_depths, all_weights, num_total_depth_pos
            flattened_spatial_shapes = orig_spatial_shapes.clone()
            flattened_spatial_shapes[:, 1] = flattened_spatial_shapes[:, 1] * num_cameras
            flattened_spatial_shapes_xy = torch.stack([flattened_spatial_shapes[..., 1], flattened_spatial_shapes[..., 0]], -1)
            global_2p5d_pts = global_2p5d_pts_norm.clone() # [n_matches, n_levels, 3]
            global_2p5d_pts[..., :2] = global_2p5d_pts[..., :2] * flattened_spatial_shapes_xy 
            levels = [h_i*w_i for (h_i, w_i) in flattened_spatial_shapes]

            # take level 0 only because its the same (x,y) for each level
            split_grids_xy = torch.split(ref_pts_2point5d_pred[..., :2], levels, dim=0)
            # global_2p5d_pts=global_2p5d_pts.contiguous()
            # ref_pts_pred_unnorm = ref_pts_pred_norm.clone().unsqueeze(1) # [n_preds, 1, 3]
            # ref_pts_pred_unnorm[..., :2] = ref_pts_pred_unnorm[..., :2].repeat(1,n_levels,1) * flattened_spatial_shapes_xy[None] 
            all_targets = []
            all_dists=[]
            all_weights = []
            all_target_depths = []
            for lvl in range(n_levels):
                # ! WARNING: EXPECTING REF_PTS_2POINT5D_PRED TO BE (X,Y) NORMALIZED WHERE 1=w_i*num_cams, AND UNNORMALIZED DEPTH
                unnorm_pts = split_grids_xy[lvl] * flattened_spatial_shapes_xy[lvl] # [n_pred_lvl, 2]
                # unnorm_pts = unnorm_pts.contiguous()
                global_lvl_pts = global_2p5d_pts[:,lvl,:2]
                # global_lvl_pts=global_lvl_pts.contiguous()
                l1dist_lvl = (unnorm_pts.unsqueeze(1) -global_lvl_pts[None]).abs().sum(-1)
                # l1dist_lvl_old = torch.cdist(unnorm_pts, global_lvl_pts, p=1.0)
                # (l1dist_lvl == l1dist_lvl_old).all()
                l1_dist, targets = l1dist_lvl.min(-1) # [n_matches]
                w = get_depth_weights(l1_dist, lvl=lvl)
                # if self.depth_weight_bound is not None:
                #     w[w<self.depth_weight_bound]=0.0
                target_depths = global_2p5d_pts[targets, lvl, 2]
                all_dists.append(l1_dist)
                all_targets.append(targets)
                all_weights.append(w)
                all_target_depths.append(target_depths)
            all_targets = torch.cat(all_targets, 0)
            all_dists = torch.cat(all_dists, 0)
            all_weights = torch.cat(all_weights, 0)
            all_target_depths = torch.cat(all_target_depths, 0)
            # all_weights=all_weights.unsqueeze(1).expand(-1, ref_pts_2point5d_pred.size(1))
            # all_target_depths = all_target_depths.unsqueeze(1).expand(-1, ref_pts_2point5d_pred.size(1))
            assert all_target_depths.shape == ref_pts_2point5d_pred[..., 2].shape
            if self.div_depth_loss_by_target_count:
                selected_targets, target_counts = torch.unique(all_targets, return_counts=True,sorted=True)
                full_target_counts = target_counts.new_zeros([global_2p5d_pts_norm.size(0)])
                full_target_counts[selected_targets] = target_counts
                # print(f"unique targets: {selected_targets}")
                # print(f"target counts: {target_counts}")
                # print(f"full target counts: {full_target_counts}")
            else:
                full_target_counts=global_2p5d_pts_norm.size(0)
            
            if self.div_depth_loss_by_target_count:
                try:
                    depth_weights = depth_weights / full_target_counts[all_targets]
                except:
                    with open("./experiments/depth_target_counts.pkl", "wb") as f:
                        pickle.dump(full_target_counts, f)
                    with open("./experiments/target_depth_gt_inds.pkl", "wb") as f:
                        pickle.dump(all_targets, f)
                    with open("./experiments/depth_weights.pkl", "wb") as f:
                        pickle.dump(depth_weights, f)
                    raise Exception()

                num_total_depth_pos = full_target_counts.numel()
            elif self.depth_weight_bound is not None:
                num_total_depth_pos = full_target_counts
            else:
                num_total_depth_pos = ref_pts_2point5d_pred.numel()
        return all_target_depths, all_weights, num_total_depth_pos
    
    def get_targets(self,
                    gt_bboxes_list,
                    lidar2img_list=None,
                    lidar2cam_list=None,
                    orig_spatial_shapes_list=None,
                    ref_pts_2point5d_pred_list=None):
        depth_targets_list, depth_weights_list, num_valid_depths_list = multi_apply(
            self._get_target_single, gt_bboxes_list, lidar2img_list, lidar2cam_list,
            orig_spatial_shapes_list, ref_pts_2point5d_pred_list
        )
        num_total_valid_depths = sum(num_valid_depths_list)
        return depth_targets_list, depth_weights_list, num_total_valid_depths
    
    def loss_single(self,
                    gt_bboxes_list,
                    lidar2img=None,
                    lidar2cam=None,
                    orig_spatial_shapes=None,
                    ref_pts_2point5d_pred=None, # [B, n_tokens, 3]
                    ):
        
        num_imgs = ref_pts_2point5d_pred.size(0)
        
        lidar2img_list = [lidar2img[i] for i in range(num_imgs)]
        lidar2cam_list = [lidar2cam[i] for i in range(num_imgs)]
        orig_spatial_shapes_list = [orig_spatial_shapes for _ in range(num_imgs)]
        ref_pts_2point5d_pred_list = [ref_pts_2point5d_pred[i] for i in range(num_imgs)]

        depth_targets_list, depth_weights_list, num_valid_depths=\
            self.get_targets(gt_bboxes_list, lidar2img_list, lidar2cam_list, 
                             orig_spatial_shapes_list, ref_pts_2point5d_pred_list)
        depth_targets = torch.cat(depth_targets_list, 0)
        depth_weights = torch.cat(depth_weights_list, 0)
        num_total_valid_depths = gt_bboxes_list[0].new_tensor([num_valid_depths])
        num_total_valid_depths = torch.clamp(reduce_mean(num_total_valid_depths), min=1).item()
        ref_pts_depth_pred = torch.cat([ref_pts[..., 2] for ref_pts in ref_pts_2point5d_pred_list],0)
        # ! ENSURE BOTH depth_targets AND ref_pts_depth_pred ARE UNNORMALIZED
        loss_depth = self.loss_depth(
            ref_pts_depth_pred, depth_targets, depth_weights, avg_factor=num_total_valid_depths*4
        )
        if loss_depth == 0.0:
            print(f"LOSS DEPTH IS 0")
        loss_depth = torch.nan_to_num(loss_depth, nan=1e-16, posinf=100.0, neginf=-100.0)
        return loss_depth

    def loss(self, gt_bboxes_list, preds_dicts, 
             lidar2img=None, lidar2cam=None, orig_spatial_shapes=None):
        ref_pts_2point5d_pred = preds_dicts['ref_pts_2point5d'] # [B, h0*N*w0+..., 4, 3] NOTE: all 4 levels have the same (x,y) points since they are global, but the depth preds are diff
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