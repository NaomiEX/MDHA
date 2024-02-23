import pickle
import torch
from torch import nn
from copy import deepcopy
from mmcv.runner import BaseModule
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from .projections import Projections
from projects.mmdet3d_plugin.models.utils.misc import MLN
from ..utils.misc import flatten_mlvl

@PLUGIN_LAYERS.register_module()
class ReferencePoints(BaseModule):
    def __init__(self, coords_depth_type="fixed", single_coords_depth=None, coords_depth_files=None, n_levels=None, coords_depth_bucket_size=None,
                 coords_depth_file_format="xy", coords_depth_bucket_format="xy", mlvl_feats_format=1, depth_start=1.0, depth_max=61.2):
        super().__init__()

        self.coords_depth_type = coords_depth_type.lower()
        self.mlvl_feats_format=mlvl_feats_format
        assert self.coords_depth_type in ["fixed", "learnable", "constant"]
        if self.coords_depth_type == "fixed":
            self.n_levels=n_levels
            if isinstance(coords_depth_files, dict):
                assert all([lvl in coords_depth_files for lvl in range(n_levels)])
                print(coords_depth_files)
                cam_depths_lvl = dict()
                for lvl in range(n_levels):
                    with open(coords_depth_files[lvl], "rb") as f:
                        cam_depths = pickle.load(f)
                    if coords_depth_file_format == "xy": # convert to y,x format
                        # cam_depths = [d.T for d in cam_depths]
                        cam_depths = [torch.clamp(d.T, min=depth_start, max=depth_max) for d in cam_depths]

                    cam_depths_lvl[lvl] = cam_depths
            else:
                with open(coords_depth_files, "rb") as f:
                    cam_depths=pickle.load(f)
                if coords_depth_file_format == "xy": # convert to y,x format
                    cam_depths = [torch.clamp(d.T, min=depth_start, max=depth_max) for d in cam_depths]
                    # cam_depths = [d.T for d in cam_depths]

                cam_depths_lvl = {lvl:deepcopy(cam_depths) for lvl in range(n_levels)}
            self.coords_depth = cam_depths_lvl
            # self.coords_depth = cam_depths
            if not isinstance(coords_depth_bucket_size[0], list):
                coords_depth_bucket_size = [coords_depth_bucket_size for _ in range(n_levels)]
            if coords_depth_bucket_format == "xy":
                coords_depth_bucket_size = [[b[1], b[0]] for b in coords_depth_bucket_size]
            self.coords_depth_bucket_size = coords_depth_bucket_size
            
            self.already_init=False
        elif self.coords_depth_type == "learnable":
            self.already_init = True
            self.coords_depth = None
        elif self.coords_depth_type == "constant":
            assert single_coords_depth is not None
            self.already_init = True
            self.coords_depth = single_coords_depth
        self.img_size = [Projections.IMG_SIZE[1], Projections.IMG_SIZE[0]] # [H, W]

    def init_coords_depth(self, device):
        if self.coords_depth_type == "fixed":
            # self.coords_depth = [d.to(device) for d in self.coords_depth]   
            self.coords_depth = {lvl: [d.to(device) for d in cam_depths] for lvl, cam_depths in self.coords_depth.items()}
            self.already_init=True
        
    def set_coord_depths(self, depths, n_tokens=None, top_rho_inds=None):
        assert self.coords_depth_type == "learnable"
        if top_rho_inds is None or depths.size(1) == n_tokens:
            self.coords_depth = depths # [B, h0*N*w0+..., 1]
        else:
            self.coords_depth = torch.zeros([depths.size(0), n_tokens, depths.size(-1)])
            self.coords_depth.scatter_(1, top_rho_inds, depths)
    
    
    def get_enc_out_proposals_and_ref_pts(self, batch_size, spatial_shapes, device, center_depth_group=True, num_cams=6,
                                          ret_coord_groups=False, 
                                        #   ret_orig_coords=False, debug=False
                                          ):
        assert self.already_init
        assert self.coords_depth is not None 
        all_ref_pts_2d_norm = []
        all_ref_pts_2p5d_unnorm = []
        all_ref_pts_2p5d_unnorm_orig = []
        all_coord_groups = []

        for lvl, (h_i, w_i) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.linspace(0, h_i-1, h_i, dtype=torch.float32, device=device),
                                            torch.linspace(0, w_i-1, w_i, dtype=torch.float32, device=device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1) # [H, W, 2]
            
            ##### get coord depths #####
            if self.coords_depth_type == "fixed":
                if self.coords_depth_bucket_size[lvl] == [h_i, w_i]:
                    coord_depths_all_cams = torch.stack(self.coords_depth[lvl]) # [N, H, W]
                    assert list(coord_depths_all_cams.shape) == [num_cams, h_i, w_i]
                elif self.coords_depth_bucket_size[lvl] == [h_i, 1]:
                    coord_depths_all_cams = torch.stack([d.repeat(1, w_i) for d in self.coords_depth[lvl]])
                    assert list(coord_depths_all_cams.shape) == [num_cams, h_i, w_i]
                else:
                    coords_depth_bucket_factor= [[self.img_size[0]//d[0],self.img_size[1]//d[1]] for d in self.coords_depth_bucket_size]
                    size_factor = [self.img_size[0] // h_i, self.img_size[1] // w_i]
                    depth_lvl_bucket_size = [coords_depth_bucket_factor[lvl][0] / size_factor[0],
                                            coords_depth_bucket_factor[lvl][1] / size_factor[1]] # [y, x]
                    if center_depth_group:
                        half_step_size = [(1/depth_lvl_bucket_size[0])/2, (1/depth_lvl_bucket_size[1])/2]
                        # print(f"bucket size: {depth_lvl_bucket_size}, step size: {half_step_size}")
                    # else:
                        # print(f"bucket size: {depth_lvl_bucket_size}")
                    coord_groups = torch.stack([grid_y/depth_lvl_bucket_size[0], grid_x/depth_lvl_bucket_size[1]],
                                                dim=-1)
                    # center coord groups
                    if center_depth_group:
                        coord_groups = coord_groups + coord_groups.new_tensor(half_step_size)
                    coord_groups = coord_groups.floor().long()
                    all_coord_groups.append(coord_groups)
                    coord_depths = []
                    for cam_idx in range(num_cams):
                        coord_cam_depths = self.coords_depth[lvl][cam_idx][coord_groups[..., 0], coord_groups[..., 1]]
                        coord_depths.append(coord_cam_depths)
                    coord_depths_all_cams = torch.stack(coord_depths) # [N, H, W]
            ##############################

            scale = grid.new_tensor([w_i, h_i])
            grid = grid.unsqueeze(0).repeat(num_cams, 1, 1, 1) + 0.5 # [N, H, W, 2]
            ind = torch.arange(0, num_cams, device=device)[:, None, None, None] # [N, 1, 1, 1]
            offset = torch.cat([ind, torch.zeros_like(ind)], -1) * scale # [N, 1, 1, 2]
            ref_pts_2d_norm = grid + offset # [N, H, W, 2]

            global_scale = grid.new_tensor([w_i*num_cams, h_i])
            ref_pts_2d_norm = (ref_pts_2d_norm/ global_scale).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1) # [B, N, H, W, 2]
            grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1) # [B, N, H, W, 2]
            # assert grid.dim() == coord_depths_all_cams.dim()
            # assert grid.shape[:-1] == coord_depths_all_cams.shape[:-1], f"{grid.shape[:-1]} != {coord_depths_all_cams.shape[:-1]}"

            grid_norm = grid / scale 
            assert ((grid_norm >= 0.0) & (grid_norm <= 1.0)).all()
            if self.coords_depth_type == "fixed":
                coord_depths_all_cams = coord_depths_all_cams[None, ..., None].repeat(batch_size, 1, 1, 1, 1)
                # if debug and ret_orig_coords:
                #     ref_pts_2p5d_unnorm_orig = torch.cat([grid, coord_depths_all_cams], dim=-1)
                #     all_ref_pts_2p5d_unnorm_orig.append(ref_pts_2p5d_unnorm_orig)
            grid_global_unnorm = grid_norm * grid.new_tensor([self.img_size[1], self.img_size[0]])
            if self.coords_depth_type == "fixed":
                ref_pts_2p5d = torch.cat([grid_global_unnorm, coord_depths_all_cams], dim=-1) # [B, N, H, W, 3]
            else:
                ref_pts_2p5d = grid_global_unnorm # [B, N, H, W, 2]
            all_ref_pts_2d_norm.append(ref_pts_2d_norm)
            all_ref_pts_2p5d_unnorm.append(ref_pts_2p5d)

        all_ref_pts_2p5d_unnorm = flatten_mlvl(all_ref_pts_2p5d_unnorm, spatial_shapes,
                                               self.mlvl_feats_format) # [B, h0*N*w0+..., 2|3]
        if self.coords_depth_type == "learnable":
            assert all_ref_pts_2p5d_unnorm.dim() == self.coords_depth.dim()
            assert all_ref_pts_2p5d_unnorm.shape[:2] == self.coords_depth.shape[:2]
            all_ref_pts_2p5d_unnorm = torch.cat([all_ref_pts_2p5d_unnorm, self.coords_depth], -1) # [B, h0*N*w0+..., 3]
        elif self.coords_depth_type == "constant":
            coords_depth = torch.full_like(all_ref_pts_2p5d_unnorm[..., :1], self.coords_depth)
            assert list(coords_depth.shape) == [all_ref_pts_2p5d_unnorm.size(0), all_ref_pts_2p5d_unnorm.size(1), 1]
            all_ref_pts_2p5d_unnorm = torch.cat([all_ref_pts_2p5d_unnorm, coords_depth], -1)
        # if debug:
        #     debug_msg=""
        #     # debug_msg = f"normalized 2d ref pts: \n{ref_pts_2d_norm}"
        #     for lvl in range(len(all_ref_pts_2d_norm)):
        #         lvl_ref_pts_2d_norm = all_ref_pts_2d_norm[lvl]
        #         debug_msg += f"\nlvl {lvl} range: \n\tmin ({lvl_ref_pts_2d_norm[..., 0].min()}, {lvl_ref_pts_2d_norm[..., 1].min()}) " \
        #                         f"max ({lvl_ref_pts_2d_norm[..., 0].max()}, {lvl_ref_pts_2d_norm[..., 1].max()})"
            
        all_ref_pts_2d_norm = flatten_mlvl(all_ref_pts_2d_norm, spatial_shapes, self.mlvl_feats_format)
        out = [all_ref_pts_2d_norm, all_ref_pts_2p5d_unnorm]

        if ret_coord_groups:
            out += [all_coord_groups]
        # if debug:
        #     if self.coords_depth_type == "fixed" and ret_orig_coords:
        #         out += [all_ref_pts_2p5d_unnorm_orig]
        #     out += [debug_msg]

        return out