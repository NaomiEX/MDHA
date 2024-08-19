import pickle
import torch
from torch import nn
from mmcv.runner import BaseModule
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from ..utils.misc import flatten_mlvl
from ..utils.debug import *
from ..utils.proj import Projections

@PLUGIN_LAYERS.register_module()
class ReferencePoints(BaseModule):
    def __init__(self, 
                 coords_depth_type="fixed", 
                 coords_depth_files=None, 
                 n_levels=None, 
                 coords_depth_bucket_size=None,
                 coords_depth_file_format="xy",
                 coords_depth_bucket_format="xy", 
                 mlvl_feats_format=1, 
                 depth_start=1.0, 
                 depth_max=61.2,
                 coords_depth_grad=False,
                 ):
        super().__init__()

        self.coords_depth_type = coords_depth_type.lower()
        self.mlvl_feats_format=mlvl_feats_format
        assert self.coords_depth_type in ["fixed", "learnable", "constant"]
        if self.coords_depth_type == "fixed":
            print("USING FIXED DEPTH COORDS")
            self.n_levels=n_levels
            if isinstance(coords_depth_files, dict):
                assert all([lvl in coords_depth_files for lvl in range(n_levels)])
                print(coords_depth_files)
                for lvl in range(n_levels):
                    with open(coords_depth_files[lvl], "rb") as f:
                        cam_depths = pickle.load(f)
                    if coords_depth_file_format == "xy": # convert to y,x format
                        cam_depths = [torch.clamp(d.T, min=depth_start, max=depth_max) for d in cam_depths]
                    setattr(self, f"cam_depths_lvl{lvl}", nn.Parameter(
                        torch.stack(cam_depths), requires_grad=coords_depth_grad
                    ))
            
            self.coords_depth = {lvl: getattr(self, f"cam_depths_lvl{lvl}") 
                                 for lvl in range(n_levels)}
            if not isinstance(coords_depth_bucket_size[0], list):
                coords_depth_bucket_size = [coords_depth_bucket_size for _ in range(n_levels)]
            if coords_depth_bucket_format == "xy":
                coords_depth_bucket_size = [[b[1], b[0]] for b in coords_depth_bucket_size]
            self.coords_depth_bucket_size = coords_depth_bucket_size
            
        elif self.coords_depth_type == "learnable":
            self.coords_depth = None
        
        self.img_size = [Projections.IMG_SIZE[1], Projections.IMG_SIZE[0]] # [H, W]

        self.cached_ref_pts_2d_norm = None
        self.cached_ref_pts_2p5d_unnorm = None
        
    def set_coord_depths(self, depths, n_tokens=None, top_rho_inds=None):
        assert self.coords_depth_type == "learnable"
        if top_rho_inds is None or depths.size(1) == n_tokens:
            self.coords_depth = depths # [B, h0*N*w0+..., 1]
        else:
            self.coords_depth = torch.zeros([depths.size(0), n_tokens, depths.size(-1)])
            self.coords_depth.scatter_(1, top_rho_inds, depths)
    
    
    def get_enc_out_proposals_and_ref_pts(self, batch_size, spatial_shapes, device, num_cams=6,
                                          ):
        assert self.coords_depth is not None 

        if self.cached_ref_pts_2d_norm is not None and self.cached_ref_pts_2p5d_unnorm is not None:
            # NOTE: assumes spatial shapes are uniform across all inputs
            if do_debug_process(self): print("USING CACHE IN REF PTS")
            all_ref_pts_2d_norm = self.cached_ref_pts_2d_norm.clone()
            all_ref_pts_2p5d_unnorm = self.cached_ref_pts_2p5d_unnorm.clone()
            if self.coords_depth_type == "learnable":
                all_ref_pts_2p5d_unnorm = torch.cat([
                    all_ref_pts_2p5d_unnorm, self.coords_depth
                ], -1)
            return [all_ref_pts_2d_norm, all_ref_pts_2p5d_unnorm]

        all_ref_pts_2d_norm = []
        all_ref_pts_2p5d_unnorm = []


        for lvl, (h_i, w_i) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.linspace(0, h_i-1, h_i, dtype=torch.float32, device=device),
                                            torch.linspace(0, w_i-1, w_i, dtype=torch.float32, device=device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1) # [H, W, 2]
            
            ##### get coord depths #####
            if self.coords_depth_type == "fixed":
                print("USING FIXED DEPTH COORDS")

                if self.coords_depth_bucket_size[lvl] == [h_i, w_i]:
                    coord_depths_all_cams = self.coords_depth[lvl] # [N, H, W]
                    assert list(coord_depths_all_cams.shape) == [num_cams, h_i, w_i]
                else:
                    raise NotImplementedError()
            elif do_debug_process(self): 
                print("REFERENCE POINTS: USING LEARNABLE DEPTH PTS")
            ##############################

            scale = grid.new_tensor([w_i, h_i])
            grid = grid.unsqueeze(0).repeat(num_cams, 1, 1, 1) + 0.5 # [N, H, W, 2]
            ind = torch.arange(0, num_cams, device=device)[:, None, None, None] # [N, 1, 1, 1]
            offset = torch.cat([ind, torch.zeros_like(ind)], -1) * scale # [N, 1, 1, 2]
            ref_pts_2d_norm = grid + offset # [N, H, W, 2]

            global_scale = grid.new_tensor([w_i*num_cams, h_i])
            ref_pts_2d_norm = (ref_pts_2d_norm/ global_scale).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1) # [B, N, H, W, 2]
            grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1) # [B, N, H, W, 2]
            
            grid_norm = grid / scale 
            assert ((grid_norm >= 0.0) & (grid_norm <= 1.0)).all()
            if self.coords_depth_type == "fixed":
                coord_depths_all_cams = coord_depths_all_cams[None, ..., None].repeat(batch_size, 1, 1, 1, 1)
                
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
        
        all_ref_pts_2d_norm = flatten_mlvl(all_ref_pts_2d_norm, spatial_shapes, self.mlvl_feats_format)

        ## cache ref points
        self.cached_ref_pts_2d_norm = all_ref_pts_2d_norm.detach().clone()
        self.cached_ref_pts_2p5d_unnorm = all_ref_pts_2p5d_unnorm.detach().clone() if self.coords_depth_type == "fixed" \
                                          else all_ref_pts_2p5d_unnorm.detach().clone()[..., :2]

        return all_ref_pts_2d_norm, all_ref_pts_2p5d_unnorm