import torch

from .debug import *

class Projections():
    IMG_SIZE=[704, 256]
    def __init__(self,
                 img_size, # format: [W, H]
                 point_selection="first",
                 ):
        super().__init__()
        Projections.IMG_SIZE=img_size # hack way to let other modules access img size
        self.img_center_coords = [img_size[0] // 2, img_size[1] // 2] # [cx, cy]
        self.point_selection=point_selection
        assert self.point_selection in [None, "first", "center"]

    @staticmethod
    def lidar_to_all_2d_img(lidar_pts, lidar2imgs, ret_2p5d=False):
        """
        Args:
            lidar_points (Tensor): [B, x, 3]
            lidar2imgs (Tensor): [B, N, 4, 4]
        """
        # [B, x, 4]
        l = torch.cat([lidar_pts,
                       lidar_pts.new_ones(lidar_pts[..., :1].shape)],
                       dim=-1)
        l = l[:, :, None, :, None] # [B, x, 1, 4, 1]
        l2is = lidar2imgs[:, None] # [B, 1, N, 4, 4]
        i = (l2is @ l).squeeze(-1)[..., :3] # [B, x, N, 3]
        normalized_i = i[..., 0:2] / torch.clamp(i[..., 2:3], min=1e-5)
        if ret_2p5d:
            normalized_i = torch.cat([normalized_i, i[..., 2:3]], -1)
        return normalized_i
    
    @staticmethod
    def lidar_to_all_3d_cam(lidar_pts, lidar2cams):
        """
        Args:
            lidar_points (Tensor): [B, x, 3]
            lidar2cams (Tensor): [B, N, 4, 4]
        """
        # [B, x, 4]
        l = torch.cat([lidar_pts,
                       lidar_pts.new_ones(lidar_pts[..., :1].shape)],
                       dim=-1)
        l = l[:, :, None, :, None] # [B, x, 1, 4, 1]
        l2cs = lidar2cams[:, None] # [B, 1, N, 4, 4]
        c = (l2cs @ l).squeeze(-1)[..., :3] # [B, x, N, 3]
        return c
    
    def get_valid_projections(self,
                               img2d_pts, # [B, ..., 2|3]
                               cam3d_pts, # [B, ..., 3]
                               ):
        in_front = cam3d_pts[..., 2] >= 0.1
        in_img_bounds = (img2d_pts[..., 0] >= 0) & (img2d_pts[..., 0] < self.IMG_SIZE[0]) \
                        & (img2d_pts[..., 1] >= 0) & (img2d_pts[..., 1] < self.IMG_SIZE[1])
        matches = in_front & in_img_bounds
        return matches, in_front
    
    def lidar_to_single_valid_2d_img(self, 
                                     lidar_pts, # [B, x, 3]
                                     lidar2imgs, # [B, N, 4, 4]
                                     lidar2cams, # [B, N, 4, 4]
                                     ):
        """
        Returns:
            res: [B, x, 2|3]
            chosen_cam: [B, x]
        """
        img2d_pts = self.lidar_to_all_2d_img(lidar_pts, lidar2imgs) # [B, x, N, 2|3]
        cam3d_pts = self.lidar_to_all_3d_cam(lidar_pts, lidar2cams) # [B, x, N, 3]
        # [B, x, N]
        matches_mask, in_front_mask = self.get_valid_projections(img2d_pts, cam3d_pts)
        matches_int = matches_mask.to(torch.int32)
        idx_with_match = matches_int.sum(-1) > 0

        pts_dim = 2

        if self.point_selection == "first":
            res = lidar_pts.new_full([*lidar_pts.shape[:2], pts_dim], -1) # [B, x, 2|3]
            chosen_cam = torch.full(lidar_pts.shape[:2], -1, 
                                    dtype=torch.long, device=lidar_pts.device) # [B, x]
            ## get first matches
            first_match_idx = torch.argmax(matches_int[idx_with_match], dim=-1)
            res[idx_with_match] = img2d_pts[idx_with_match, first_match_idx]
            chosen_cam[idx_with_match] = first_match_idx

        elif self.point_selection == "center":
            if do_debug_process(self): print("using CENTER selection for projection single")
            ## get the point which is closest to the center of the img
            img2d_pts_copy = img2d_pts.new_full(img2d_pts.shape[:-1], float('inf')) # [B, x, N]
            img2d_pts_copy[matches_mask] = torch.linalg.norm(
                img2d_pts[matches_mask] - lidar_pts.new_tensor(self.img_center_coords), dim=-1)
            chosen_cam = torch.min(img2d_pts_copy, dim=-1).indices
            idx = chosen_cam[..., None, None].expand(-1, -1, *img2d_pts.shape[-2:])
            res = img2d_pts.gather(dim=2, index=idx)[..., 0,:]
            chosen_cam = chosen_cam.clone()
            res = res.clone()

        else:
            raise ValueError(f"projection to single valid 2d img point only supports "
                             f"'first' and 'center' point selection but got {self.point_selection}")
        ## project invalid points to closest image bounds
        non_matches = img2d_pts[~idx_with_match]
        non_matches_xy = non_matches[..., :2]

        ## debug ##
        if with_debug(self):
            if not hasattr(self, "total_non_matches"): 
                self.total_non_matches = 0
                self.total_pts = 0
            self.total_non_matches +=non_matches.size(0)
            self.total_pts += lidar_pts.size(0) * lidar_pts.size(1)

            if do_debug_process(self, repeating=True):
                print(f"proportion non matches so far: {self.total_non_matches / self.total_pts}")
                self.total_non_matches = 0
                self.total_pts = 0

        ## end debug ##

        if non_matches.size(0) > 0:
            dists = torch.max(
                torch.stack([-non_matches_xy, 
                             torch.zeros_like(non_matches_xy), 
                             non_matches_xy - non_matches_xy.new_tensor(self.IMG_SIZE)],
                             dim=-1),
                dim=-1).values
            dists_sum = dists.sum(-1).double()
            non_match_in_front = in_front_mask[~idx_with_match]
            # minus by a suitably large number to ensure the ones which are in front of the cam get chosen first
            dists_sum[non_match_in_front] = dists_sum[non_match_in_front] - 1e10
            min_inds = torch.min(dists_sum.nan_to_num(), dim=-1).indices
            bid = torch.arange(non_matches_xy.size(0), device=non_matches.device)
            projected_non_match_xy = non_matches_xy[bid, min_inds].abs() - dists[bid, min_inds]
            
            res[~idx_with_match] = projected_non_match_xy
            chosen_cam[~idx_with_match] = min_inds
        
        res = res.nan_to_num()
        res[..., 0] = torch.clamp(res[..., 0].clone(), min=0, max=self.IMG_SIZE[0])
        res[..., 1] = torch.clamp(res[..., 1].clone(), min=0, max=self.IMG_SIZE[1])

        return res, chosen_cam
    
    def lidar_to_mult_valid_2d_img(self,
                                   lidar_pts, # [B, x, 3]
                                   lidar2imgs, # [B, N, 4, 4]
                                   lidar2cams, # [B, N, 4, 4]
                                   ):
        B=lidar_pts.size(0)
        img2d_pts = self.lidar_to_all_2d_img(lidar_pts, lidar2imgs) # [B, x, N, 2|3]
        cam3d_pts = self.lidar_to_all_3d_cam(lidar_pts, lidar2cams) # [B, x, N, 3]
        # [B, x, N]
        matches_mask, in_front_mask = self.get_valid_projections(img2d_pts, cam3d_pts)
        matches_int = matches_mask.to(torch.int32)
        matches_int_sum = matches_int.sum(-1) # [B, x]
        idx_with_match = matches_int_sum > 0 # [B, x]
        idx_with_mult_match = matches_int_sum > 1 # [B, x]

        pts_dim = 2

        res = lidar_pts.new_full([*lidar_pts.shape[:2], pts_dim], -1) # [B, x, 2|3]
        chosen_cam = torch.full(lidar_pts.shape[:2], -1, 
                                dtype=torch.long, device=lidar_pts.device) # [B, x]
        ## get first matches
        first_match_idx = torch.argmax(matches_int[idx_with_match], dim=-1)
        res[idx_with_match] = img2d_pts[idx_with_match, first_match_idx]
        chosen_cam[idx_with_match] = first_match_idx

        ## get second matches (if any)
        nobjs = idx_with_mult_match.sum(dim=-1)
        max_num_second_matches = nobjs.max().item()
        second_res = lidar_pts.new_full([B, max_num_second_matches, 2], -1)
        second_cams = lidar_pts.new_full([B, max_num_second_matches], -1, dtype=torch.int64)

        matches_int[idx_with_match, first_match_idx] = 0
        second_match_idx = torch.argmax(matches_int[idx_with_mult_match], dim=-1)
        point_2d_cam_second_matches = img2d_pts[idx_with_mult_match, second_match_idx]

        bids = torch.cat([torch.full((t, ), i) for i, t in enumerate(nobjs)])
        item_idxs = torch.cat([torch.arange(t) for t in nobjs])
        second_match_mask = (bids, item_idxs)
        second_res[bids, item_idxs] = point_2d_cam_second_matches
        second_cams[bids, item_idxs] = second_match_idx

        ## project invalid points to closest image bounds
        non_matches = img2d_pts[~idx_with_match]
        non_matches_xy = non_matches[..., :2]

        ## debug ##
        if with_debug(self):
            if not hasattr(self, "total_non_matches"): 
                self.total_non_matches = 0
                self.total_pts = 0
                self.num_second_matches = 0
            self.total_non_matches +=non_matches.size(0)
            self.total_pts += lidar_pts.size(1)
            self.num_second_matches += nobjs.sum() # ! CHECK THIS

            if do_debug_process(self, repeating=True):
                print(f"proportion non matches so far: {self.total_non_matches / self.total_pts}")
                print(f"proportion of second matches: {self.num_second_matches/self.total_pts}")
                self.total_non_matches = 0
                self.total_pts = 0
                self.num_second_matches = 0
        ## end debug ##
                
        if non_matches.size(0) > 0:
            dists = torch.max(
                torch.stack([-non_matches_xy, 
                             torch.zeros_like(non_matches_xy), 
                             non_matches_xy - non_matches_xy.new_tensor(self.IMG_SIZE)],
                             dim=-1),
                dim=-1).values
            dists_sum = dists.sum(-1).double()
            non_match_in_front = in_front_mask[~idx_with_match]
            # minus by a suitably large number to ensure the ones which are in front of the cam get chosen first
            dists_sum[non_match_in_front] = dists_sum[non_match_in_front] - 1e10
            min_inds = torch.min(dists_sum.nan_to_num(), dim=-1).indices
            bid = torch.arange(non_matches_xy.size(0), device=non_matches.device)
            projected_non_match_xy = non_matches_xy[bid, min_inds].abs() - dists[bid, min_inds]
            res[~idx_with_match] = projected_non_match_xy
            chosen_cam[~idx_with_match] = min_inds

        res = res.nan_to_num()
        res[..., 0] = torch.clamp(res[..., 0].clone(), min=0, max=self.IMG_SIZE[0])
        res[..., 1] = torch.clamp(res[..., 1].clone(), min=0, max=self.IMG_SIZE[1])

        return res, chosen_cam, second_res, second_cams, second_match_mask, idx_with_mult_match
    
    @staticmethod
    def project_2p5d_cam_to_3d_lidar(cam_pts_2p5d, img2lidar):
        """
        Args:
            cam_pts_2p5d (_type_): [B, ..., 3]
            img2lidar (_type_): [B, ..., 4, 4]
        """
        eps=1e-5
            
        B, *rest, D = cam_pts_2p5d.shape

        assert img2lidar.size(0) == B and list(img2lidar.shape[-2:]) == [4,4]
        assert D == 3
        assert list(img2lidar.shape[1:-2]) == rest

        coords = torch.cat([cam_pts_2p5d, torch.ones_like(cam_pts_2p5d[..., :1])], -1) # [B, ..., 4]
        coords[..., :2] = coords[..., :2].clone() * torch.maximum(coords[..., 2:3].clone(),
                                                        torch.ones_like(coords[..., 2:3]) * eps)
        coords = coords.unsqueeze(-1) # [B, ..., 4, 1]
        proj_pts = torch.matmul(img2lidar, coords).squeeze(-1)[..., :3] # [B, ..., 3]
        # collect % of values out of range
        # collect distribution of (x,y,z) values
        return proj_pts
    
    def convert_3d_to_2d_global_cam_ref_pts(self, lidar2imgs, lidar2cams, ref_pts_3d, orig_spatial_shapes, 
                                            num_cameras=6, ref_pts_mode="single"):
        proj_w, proj_h = self.IMG_SIZE
        if ref_pts_mode == "single":
            # local_ref_pts_2d: [B, R, 2]
            # chosen_cams: [B, R]
            local_ref_pts_2d, chosen_cams = self.lidar_to_single_valid_2d_img(
                ref_pts_3d, lidar2imgs, lidar2cams)
        else:
            if do_debug_process(self): print("USING MULTIPLE PROJECTED REFERENCE POINTS")
            # loc_ref_pts_2d_fst: [B, R, 2]
            # cams_fst: [B, R]
            # loc_ref_pts_2d_sec: [B, max_num_second_matches, 2]
            # cams_sec: [B, max_num_second_matches]
            # second_match_mask: (bid, valid_item_idxs)
            # idx_with_mult_match: [B, R]
            loc_ref_pts_2d_fst, cams_fst, loc_ref_pts_2d_sec, cams_sec, second_match_mask, idx_with_mult_match =\
                self.lidar_to_mult_valid_2d_img(ref_pts_3d, lidar2imgs, lidar2cams)
            num_second_matches = loc_ref_pts_2d_sec.size(1)
            local_ref_pts_2d = torch.cat([loc_ref_pts_2d_fst, loc_ref_pts_2d_sec], 1) # [B, R+S, 2]
            chosen_cams = torch.cat([cams_fst, cams_sec], 1) # [B, R+S]
        
        # [B, R]
        global_ref_x = (local_ref_pts_2d[..., 0] + (chosen_cams * proj_w)) / \
                       (proj_w * num_cameras)
        # [B, R]
        global_ref_y = local_ref_pts_2d[..., 1] / proj_h
        global_ref_pts = torch.stack([global_ref_x, global_ref_y], -1) # [B, R, 2]
        global_ref_pts = global_ref_pts.unsqueeze(2).repeat(1, 1, orig_spatial_shapes.size(0), 1) # [B, R, n_levels, 2]

        if ref_pts_mode == "single":
            return global_ref_pts, chosen_cams
        else:
            return global_ref_pts, chosen_cams, num_second_matches, second_match_mask, idx_with_mult_match
        
    def project_to_matching_2point5d_cam_points(self, lidar_pts, lidar2imgs, lidar2cams, orig_spatial_shapes,
                                                num_cameras=6):
        proj_w, proj_h = self.IMG_SIZE
        
        # [B, R, N, 3]
        # point_2p5d_cam = self.project_lidar_points_to_all_2point5d_cams_batch(lidar_pts, lidar2imgs)
        point_2p5d_cam = self.lidar_to_all_2d_img(lidar_pts, lidar2imgs, ret_2p5d=True)
        point_3d_cam = self.lidar_to_all_3d_cam(lidar_pts, lidar2cams) # [B, R, N, 3]

        matches, in_front = self.get_valid_projections(point_2p5d_cam, point_3d_cam)
        matches_int = matches.to(torch.int32)

        res = point_2p5d_cam[matches] # [n_matches, 3]

        chosen_cams = matches_int.nonzero()[..., -1] # [n_objs]

        global_ref_x = (res[..., 0] + (chosen_cams * proj_w)) / \
                       (proj_w * num_cameras)
        global_ref_y = res[..., 1] / proj_h

        global_2p5d_ref_pts = torch.stack([
            global_ref_x, global_ref_y, res[..., 2]
        ], -1) # [n_matches, 3]
        global_2p5d_ref_pts = global_2p5d_ref_pts.unsqueeze(1)\
                              .expand(-1, orig_spatial_shapes.size(0), -1) # [n_matches, n_levels, 3]
        return global_2p5d_ref_pts