import torch
# import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from .lidar_utils import clamp_to_lidar_range, not_in_lidar_range

class Projections:
    IMG_SIZE = [1408, 512] # [W, H]
    # IMG_SIZE_IN_BOUNDS = [703, 255]
    IMG_CENTER_COORDS=[IMG_SIZE[0]//2, IMG_SIZE[1]//2]
    # # NOTE: IT'S IMPORTANT THAT THIS ORDERING MATCHES THE ORDERING OF THE NUSCENES_CONVERTER
    CAMERA_TYPES = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_RIGHT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_FRONT_LEFT',
    ]

    @staticmethod
    def project_lidar_points_to_all_2d_cams_batch(lidar_points, lidar2imgs):
        """
        Args:
            lidar_points (Tensor): [B, n_objs, 3]
            lidar2imgs (Tensor): [B, 6, 4, 4]
        """
        assert lidar_points.dim() == 3
        assert lidar2imgs.dim() == 4
        l2 = torch.cat([lidar_points, lidar_points.new_ones([*lidar_points.shape[:-1], 1])], dim=-1) # [B, n_objs, 4]
        l2 = l2[:, None, ..., None] # [B, 1, n_objs, 4, 1]
        l2is = lidar2imgs[:, :, None, ...] # [B, 6, 1, 4, 4]
        projected_pts = (l2is @ l2).squeeze(-1) # [B, 6, n_objs, 4]
        projected_pts=projected_pts[..., :3]
        # normalize
        return projected_pts[..., 0:2] / projected_pts[..., 2:3] # [B, 6, n_objs, 2]
    
    @staticmethod
    def project_lidar_points_to_all_2point5d_cams_batch(lidar_points, lidar2imgs):
        """
        Args:
            lidar_points (Tensor): [B, n_objs, 3]
            lidar2imgs (Tensor): [B, 6, 4, 4]
        """
        assert lidar_points.dim() == 3
        assert lidar2imgs.dim() == 4
        l2 = torch.cat([lidar_points, lidar_points.new_ones([*lidar_points.shape[:-1], 1])], dim=-1) # [B, n_objs, 4]
        l2 = l2[:, None, ..., None] # [B, 1, n_objs, 4, 1]
        l2is = lidar2imgs[:, :, None, ...] # [B, 6, 1, 4, 4]
        projected_pts = (l2is @ l2).squeeze(-1) # [B, 6, n_objs, 4]
        projected_pts=projected_pts[..., :3] # [B, 6, n_objs, 3]
        # normalize
        projected_pts[..., 0:2] = projected_pts[..., 0:2] / projected_pts[..., 2:3] # [B, 6, n_objs, 3]
        return projected_pts

    @staticmethod
    def project_lidar_points_to_all_3d_cams_batch(lidar_points, lidar2cams):
        # lidar_points: [B, n_objs, 3]
        # lidar2cams: [B, 6, 4, 4]
        assert lidar_points.dim() == 3
        assert lidar2cams.dim() == 4
        l2 = torch.cat([lidar_points, lidar_points.new_ones([*lidar_points.shape[:-1], 1])], dim=-1) # [B, n_objs, 4]
        l2 = l2[:, None, ..., None] # [B, 1, n_objs, 4, 1]
        l2cs = lidar2cams[:, :, None] # [B, 6, 1, 4, 4]
        projected_pts = (l2cs @ l2).squeeze(-1) # [B, 6, n_objs, 4]
        return projected_pts[..., :3] # [B, 6, n_objs, 3]

    @staticmethod
    def project_to_2d_cam_point(lidar_points, cam_transformations, ret_orig_2d_projs=False, ret_orig_3d_projs=False,
                                ret_non_matches=False, ret_matches=False):
        """
        lidar_points: [B, n_objs, 3]
        cam_transformations: {"lidar2img": [B, 6, 4, 4], "lidar2cam": [B, 6, 4, 4]}
        return:
            - projected points are in range[0, IMG_SIZE-1]
        """
        assert isinstance(cam_transformations, dict) and 'lidar2img' in cam_transformations and 'lidar2cam' in cam_transformations
        # print("using new project func")
        point_2d_cam = Projections.project_lidar_points_to_all_2d_cams_batch(lidar_points, cam_transformations['lidar2img']).transpose(1,2) # [B, n_objs, 6, 2]
        point_3d_cam = Projections.project_lidar_points_to_all_3d_cams_batch(lidar_points, cam_transformations['lidar2cam']).transpose(1, 2) # [B, n_objs, 6, 3]
        
        in_front = point_3d_cam[..., 2] > 0
        in_2d_range = (point_2d_cam[..., 0] >= 0) & (point_2d_cam[..., 0] < Projections.IMG_SIZE[0]) & \
                    (point_2d_cam[..., 1] >= 0) & (point_2d_cam[..., 1] < Projections.IMG_SIZE[1])
        matches = in_front & in_2d_range
        pts_with_match = matches.any(dim=-1)

        # for matches, choose the ones that are closest to the center
        # NOTE: vast majority of points will fall in this category
        # TODO: TRY TO JUST PICK THE FIRST ONE THAT MATCHES 
        point_2d_cam_copy = point_2d_cam.new_full([*point_2d_cam.shape[:-1]], float('inf'))
        point_2d_cam_copy[matches] = torch.linalg.norm(point_2d_cam[matches] - lidar_points.new_tensor(Projections.IMG_CENTER_COORDS), dim=-1)
        closest = torch.min(point_2d_cam_copy, dim=-1)[1]
        res = point_2d_cam.gather(dim=2, index=closest[..., :, None, None].expand(-1, -1, *point_2d_cam.shape[-2:]))[..., 0, :]
        
        # for non matches, choose the ones that are closest to the img boundaries, and project the point to the 2d img boundaries
        non_matches = point_2d_cam[~pts_with_match] # [num_non_matches, num_cameras, 2]
        if non_matches.size(0) > 0:
            dists = torch.max(torch.stack([-non_matches, torch.zeros_like(non_matches), non_matches-non_matches.new_tensor(Projections.IMG_SIZE)], dim=-1), -1)[0]
            dists_sum=dists.sum(-1)
            in_front_not_in_2d_range = in_front[~pts_with_match] & (~in_2d_range)[~pts_with_match]
            # in_front_not_in_2d_range = in_front_not_in_2d_range[~pts_with_match]
            dists_sum[in_front_not_in_2d_range] = dists_sum[in_front_not_in_2d_range] - torch.max(dists_sum)
            min_inds = torch.min(dists_sum.nan_to_num(), dim=-1)[1]
            bid = torch.arange(non_matches.shape[0], device=non_matches.device)
            res[~pts_with_match] = non_matches[bid, min_inds].abs() - dists[bid, min_inds]

            closest[(~pts_with_match).nonzero(as_tuple=True)] = min_inds
        res=res.nan_to_num()
        res[..., 0]=torch.clamp(res[..., 0].clone(), min=0, max=Projections.IMG_SIZE[0])
        res[..., 1]=torch.clamp(res[..., 1].clone(), min=0, max=Projections.IMG_SIZE[1])
        rets=[res, closest]
        if ret_orig_2d_projs:
            rets.append(point_2d_cam)
        if ret_orig_3d_projs:
            rets.append(point_3d_cam)
        if ret_non_matches:
            rets.append(~pts_with_match)
        if ret_matches:
            rets.append(matches)
        return rets
    
    @staticmethod
    def project_to_mult_2point5d_cam_points(lidar_points, cam_transformations, ret_non_matches=False):
        """
        lidar_points: [B, n_objs, 3]
        cam_transformations: {"lidar2img": [B, 6, 4, 4], "lidar2cam": [B, 6, 4, 4]}
        return:
            - projected points are in range[0, IMG_SIZE-1]
        """
        assert isinstance(cam_transformations, dict) and 'lidar2img' in cam_transformations and 'lidar2cam' in cam_transformations
        B = lidar_points.size(0)
        point_2p5d_cam = Projections.project_lidar_points_to_all_2point5d_cams_batch(lidar_points, cam_transformations['lidar2img']).transpose(1,2) # [B, 6, n_objs, 3]
        point_3d_cam = Projections.project_lidar_points_to_all_3d_cams_batch(lidar_points, cam_transformations['lidar2cam']).transpose(1, 2) # [B, 6, n_objs, 3]
        
        in_front = point_3d_cam[..., 2] > 0
        in_2d_range = (point_2p5d_cam[..., 0] >= 0) & (point_2p5d_cam[..., 0] < Projections.IMG_SIZE[0]) & \
                    (point_2p5d_cam[..., 1] >= 0) & (point_2p5d_cam[..., 1] < Projections.IMG_SIZE[1])
        matches = in_front & in_2d_range # [B, n_objs, 6]
        matches_int = matches.type(torch.int32)
        matches_int_sum = matches_int.sum(-1)
        idx_with_match = matches_int_sum > 0
        idx_with_mult_match = matches_int_sum > 1
        assert not (matches_int_sum > 2).any()

        res = torch.full([*lidar_points.shape[:-1], 3], -1, dtype=torch.float32, device=lidar_points.device)
        chosen_cam = torch.full([*lidar_points.shape[:2]], -1, dtype=torch.int64, device=lidar_points.device)
        
        first_match_idx = torch.argmax(matches_int[idx_with_match], dim=-1)
        point_2p5d_cam_first_matches = point_2p5d_cam[idx_with_match, first_match_idx]
        res[idx_with_match] = point_2p5d_cam_first_matches
        chosen_cam[idx_with_match] = first_match_idx

        ## get second matches (if any)
        nobjs = idx_with_mult_match.sum(dim=-1)
        max_num_second_matches = nobjs.max().item()
        second_res = lidar_points.new_full([B, max_num_second_matches, 3], -1)
        second_cams = lidar_points.new_full([B, max_num_second_matches], -1, dtype=torch.int64)

        matches_int[idx_with_match, first_match_idx] = 0
        second_match_idx = torch.argmax(matches_int[idx_with_mult_match], dim=-1)
        point_2p5d_cam_second_matches = point_2p5d_cam[idx_with_mult_match, second_match_idx]

        bids = torch.cat([torch.full((t, ), i) for i, t in enumerate(nobjs)])
        item_idxs = torch.cat([torch.arange(t) for t in nobjs])
        second_match_valid_idxs = (bids, item_idxs)
        second_res[bids, item_idxs] = point_2p5d_cam_second_matches
        second_cams[bids, item_idxs] = second_match_idx

        non_matches = point_2p5d_cam[~idx_with_match] # [num_non_matches, num_cameras, 3]
        non_matches_xy = non_matches[..., :2]
        if non_matches.size(0) > 0:
            dists = torch.max(torch.stack([-non_matches_xy, torch.zeros_like(non_matches_xy), non_matches_xy-non_matches_xy.new_tensor(Projections.IMG_SIZE)], dim=-1), -1)[0]
            dists_sum=dists.sum(-1)
            in_front_not_in_2d_range = in_front[~idx_with_match] & (~in_2d_range)[~idx_with_match]
            dists_sum[in_front_not_in_2d_range] = dists_sum[in_front_not_in_2d_range] - torch.max(dists_sum)
            min_inds = torch.min(dists_sum.nan_to_num(), dim=-1)[1]
            bid = torch.arange(non_matches_xy.shape[0], device=non_matches_xy.device)
            res[~idx_with_match] = torch.cat([non_matches_xy[bid, min_inds].abs() - dists[bid, min_inds], non_matches[..., 2:3]], -1)
            chosen_cam[(~idx_with_match).nonzero(as_tuple=True)] = min_inds
        assert not (res == -1.0).any()
        res=res.nan_to_num()
        res[..., 0]=torch.clamp(res[..., 0].clone(), min=0, max=Projections.IMG_SIZE[0])
        res[..., 1]=torch.clamp(res[..., 1].clone(), min=0, max=Projections.IMG_SIZE[1])
        
        rets=[res, chosen_cam, second_res, second_cams, second_match_valid_idxs, idx_with_mult_match]
        if ret_non_matches:
            rets.append(~idx_with_match)
        return rets

    @staticmethod
    def project_to_mult_2d_cam_points(lidar_points, cam_transformations, ret_orig_2d_projs=False, ret_orig_3d_projs=False,
                                ret_non_matches=False):
        """
        lidar_points: [B, n_objs, 3]
        cam_transformations: {"lidar2img": [B, 6, 4, 4], "lidar2cam": [B, 6, 4, 4]}
        return:
            - projected points are in range[0, IMG_SIZE-1]
        """
        assert isinstance(cam_transformations, dict) and 'lidar2img' in cam_transformations and 'lidar2cam' in cam_transformations
        B = lidar_points.size(0)
        point_2d_cam = Projections.project_lidar_points_to_all_2d_cams_batch(lidar_points, cam_transformations['lidar2img']).transpose(1,2) # [B, 6, n_objs, 2]
        point_3d_cam = Projections.project_lidar_points_to_all_3d_cams_batch(lidar_points, cam_transformations['lidar2cam']).transpose(1, 2) # [B, 6, n_objs, 3]
        
        in_front = point_3d_cam[..., 2] > 0
        in_2d_range = (point_2d_cam[..., 0] >= 0) & (point_2d_cam[..., 0] < Projections.IMG_SIZE[0]) & \
                    (point_2d_cam[..., 1] >= 0) & (point_2d_cam[..., 1] < Projections.IMG_SIZE[1])
        matches = in_front & in_2d_range # [B, n_objs, 6]
        matches_int = matches.type(torch.int32)
        matches_int_sum = matches_int.sum(-1)
        idx_with_match = matches_int_sum > 0
        idx_with_mult_match = matches_int_sum > 1
        assert not (matches_int_sum > 2).any()

        res = torch.full([*lidar_points.shape[:-1], 2], -1, dtype=torch.float32, device=lidar_points.device)
        chosen_cam = torch.full([*lidar_points.shape[:2]], -1, dtype=torch.int64, device=lidar_points.device)
        
        first_match_idx = torch.argmax(matches_int[idx_with_match], dim=-1)
        point_2d_cam_first_matches = point_2d_cam[idx_with_match, first_match_idx]
        res[idx_with_match] = point_2d_cam_first_matches
        chosen_cam[idx_with_match] = first_match_idx

        ## get second matches (if any)
        nobjs = idx_with_mult_match.sum(dim=-1)
        max_num_second_matches = nobjs.max().item()
        second_res = lidar_points.new_full([B, max_num_second_matches, 2], -1)
        second_cams = lidar_points.new_full([B, max_num_second_matches], -1, dtype=torch.int64)

        matches_int[idx_with_match, first_match_idx] = 0
        second_match_idx = torch.argmax(matches_int[idx_with_mult_match], dim=-1)
        point_2d_cam_second_matches = point_2d_cam[idx_with_mult_match, second_match_idx]

        bids = torch.cat([torch.full((t, ), i) for i, t in enumerate(nobjs)])
        item_idxs = torch.cat([torch.arange(t) for t in nobjs])
        second_match_valid_idxs = (bids, item_idxs)
        second_res[bids, item_idxs] = point_2d_cam_second_matches
        second_cams[bids, item_idxs] = second_match_idx

        non_matches = point_2d_cam[~idx_with_match] # [num_non_matches, num_cameras, 2]
        if non_matches.size(0) > 0:
            dists = torch.max(torch.stack([-non_matches, torch.zeros_like(non_matches), non_matches-non_matches.new_tensor(Projections.IMG_SIZE)], dim=-1), -1)[0]
            dists_sum=dists.sum(-1)
            in_front_not_in_2d_range = in_front[~idx_with_match] & (~in_2d_range)[~idx_with_match]
            dists_sum[in_front_not_in_2d_range] = dists_sum[in_front_not_in_2d_range] - torch.max(dists_sum)
            min_inds = torch.min(dists_sum.nan_to_num(), dim=-1)[1]
            bid = torch.arange(non_matches.shape[0], device=non_matches.device)
            res[~idx_with_match] = non_matches[bid, min_inds].abs() - dists[bid, min_inds]

            chosen_cam[(~idx_with_match).nonzero(as_tuple=True)] = min_inds
        res=res.nan_to_num()
        res[..., 0]=torch.clamp(res[..., 0].clone(), min=0, max=Projections.IMG_SIZE[0])
        res[..., 1]=torch.clamp(res[..., 1].clone(), min=0, max=Projections.IMG_SIZE[1])
        
        rets=[res, chosen_cam, second_res, second_cams, second_match_valid_idxs, idx_with_mult_match]
        if ret_orig_2d_projs:
            rets.append(point_2d_cam)
        if ret_orig_3d_projs:
            rets.append(point_3d_cam)
        if ret_non_matches:
            rets.append(~idx_with_match)
        return rets

    @staticmethod
    def project_2p5d_cam_to_3d_lidar(cam_pts_2p5d, img2lidar, collect_stats=False, logger=None, pc_range=None):
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
        # ones_shape = [1] * len(rest)
        # img2lidar = torch.inverse(lidar2img.cpu()).cuda().view(B, N, *ones_shape, 4, 4).repeat(1, 1, *rest ,1, 1)
        # assert coords.dim() == img2lidar.dim() and coords.shape[:-1] == img2lidar.shape[:-1]
        proj_pts = torch.matmul(img2lidar, coords).squeeze(-1)[..., :3] # [B, ..., 3]
        # collect % of values out of range
        # collect distribution of (x,y,z) values
        return proj_pts
    

def convert_3d_to_mult_2d_global_cam_ref_pts(cam_transformations, reference_points, 
                                        orig_spatial_shapes, img_metas, num_cameras=6,
                                        ret_num_non_matches=False):
    pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
    assert pad_h == Projections.IMG_SIZE[1]
    assert pad_w == Projections.IMG_SIZE[0]
    attn_ref_pts = reference_points.clone()
    try:
        outs =  Projections.project_to_mult_2d_cam_points(reference_points, cam_transformations, ret_non_matches=ret_num_non_matches)
        proj_2d_pts_orig, cams, proj_2d_pts_second_matches, cams_second_matches, \
            second_matches_valid_idxs, idx_with_second_match = outs[:6]
        if ret_num_non_matches:
            non_matches = outs[6]
        num_second_matches = proj_2d_pts_second_matches.size(1)
        all_proj_2d_pts = torch.cat([proj_2d_pts_orig, proj_2d_pts_second_matches], 1) # [B, num_objs + second_matches, 2]
        all_cams = torch.cat([cams, cams_second_matches], 1) # [B, num_objs + second_matches]
        projected_2d_pts = all_proj_2d_pts.clone()
        projected_2d_pts[..., 0] = projected_2d_pts[..., 0] / pad_w
        projected_2d_pts[..., 1] = projected_2d_pts[..., 1] / pad_h
        assert ((projected_2d_pts[:, :-num_second_matches] >= 0) & 
                (projected_2d_pts[:, :-num_second_matches] <= 1)).all()
        assert ((projected_2d_pts[:, -num_second_matches:][second_matches_valid_idxs] >= 0) & 
                (projected_2d_pts[:, -num_second_matches:][second_matches_valid_idxs] <= 1)).all()
    except:
        import pickle
        pickle.dump(attn_ref_pts, open("./experiments/failed_to_convert_ref_pts.pkl", "wb"))
        pickle.dump(cam_transformations, open("./experiments/failed_to_convert_ref_pts_cam_trans.pkl", "wb"))

        raise Exception("failed to convert ref pts")
    assert projected_2d_pts.shape[:2] == all_cams.shape[:2]
    global_x_inds = projected_2d_pts[..., 0:1] * (orig_spatial_shapes[:, 1])[None, None].expand(*projected_2d_pts.shape[:2], -1)
    global_x_inds = global_x_inds + (all_cams.unsqueeze(-1) * orig_spatial_shapes[:, 1][None, None].expand(*projected_2d_pts.shape[:2], -1))
    global_ref_x = global_x_inds / (orig_spatial_shapes[:, 1] * num_cameras) # [B, Q, N_levels]
    global_ref_x = global_ref_x.unsqueeze(-1) # [B, Q, N_levels, 1]

    n_levels = orig_spatial_shapes.shape[0]
    # repeat y ratio for all levels (same ratio)
    global_ref_y = projected_2d_pts[..., 1:2].unsqueeze(2).expand(-1, -1, n_levels, -1) # [B, Q, n_levels, 1]

    assert global_ref_x.shape == global_ref_y.shape
    global_ref_pts = torch.cat([global_ref_x, global_ref_y], dim=-1) # [B, Q, n_levels, 2]
    assert list(global_ref_pts.shape) == [projected_2d_pts.size(0), projected_2d_pts.size(1), orig_spatial_shapes.size(0),
                                            projected_2d_pts.size(2)]
    if ret_num_non_matches:
        return global_ref_pts, num_second_matches, second_matches_valid_idxs, idx_with_second_match, non_matches
    return global_ref_pts, num_second_matches, second_matches_valid_idxs, idx_with_second_match  # [B, Q, n_levels, 2]
    
    
def convert_3d_to_2d_global_cam_ref_pts(cam_transformations, reference_points, 
                                        orig_spatial_shapes, img_metas, num_cameras=6,
                                        debug=False, ret_num_non_matches=False):
    """
    Args:
        cam_transformations (_type_): dict:
            - "lidar2img": Tensor[B, N, 4, 4]
            - "lidar2cam": Tensor[B, N, 4, 4]
        reference_points (_type_): Tensor[B, Q, 3]
        orig_spatial_shapes (_type_): [n_levels, 2]
        num_cameras (int, optional): . Defaults to 6.
    """
    pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
    assert pad_h == Projections.IMG_SIZE[1]
    assert pad_w == Projections.IMG_SIZE[0]
    attn_ref_pts = reference_points.clone()
    try:
        # projected_2d_pts_orig: [B, L, 2]
        # projected_cams: [B, L]
        outs = Projections.project_to_2d_cam_point(attn_ref_pts, cam_transformations, 
                                                   ret_non_matches=debug or ret_num_non_matches)
        projected_2d_pts_orig, projected_cams = outs[:2]
        if debug or ret_num_non_matches:
            non_matches = outs[2]
        projected_2d_pts = projected_2d_pts_orig.clone()
        projected_2d_pts[..., 0] = projected_2d_pts[..., 0] / pad_w
        projected_2d_pts[..., 1] = projected_2d_pts[..., 1] / pad_h
        assert ((projected_2d_pts >= 0) & (projected_2d_pts <= 1)).all()
    except:
        import pickle
        pickle.dump(attn_ref_pts, open("./experiments/failed_to_convert_ref_pts.pkl", "wb"))
        pickle.dump(cam_transformations, open("./experiments/failed_to_convert_ref_pts_cam_trans.pkl", "wb"))

        raise Exception("failed to convert ref pts")
    # attn_ref_pts= convert_2d_cam_ref_pts_to_global_ref_pts(projected_2d_pts,
    #                                         projected_cams, orig_spatial_shapes, num_cameras)
    assert projected_2d_pts.shape[0] == projected_cams.shape[0]
    global_x_inds = projected_2d_pts[..., 0:1] * ((orig_spatial_shapes[:, 1])[None, None].expand(*projected_2d_pts.shape[:2], -1))
    global_x_inds = global_x_inds + (projected_cams.unsqueeze(-1) * orig_spatial_shapes[:, 1][None, None].expand(*projected_2d_pts.shape[:2], -1))
    global_ref_x = global_x_inds / (orig_spatial_shapes[:, 1] * num_cameras) # [B, Q, N_levels]
    global_ref_x = global_ref_x.unsqueeze(-1) # [B, Q, N_levels, 1]

    n_levels = orig_spatial_shapes.shape[0]
    # repeat y ratio for all levels (same ratio)
    global_ref_y = projected_2d_pts[..., 1:2].unsqueeze(2).expand(-1, -1, n_levels, -1) # [B, Q, n_levels, 1]

    assert global_ref_x.shape == global_ref_y.shape
    global_ref_pts = torch.cat([global_ref_x, global_ref_y], dim=-1) # [B, Q, n_levels, 2]
    assert list(global_ref_pts.shape) == [projected_2d_pts.size(0), projected_2d_pts.size(1), orig_spatial_shapes.size(0),
                                            projected_2d_pts.size(2)]
    if debug:
        return global_ref_pts, non_matches, projected_2d_pts_orig, projected_cams
    elif ret_num_non_matches:
        return global_ref_pts, non_matches
    
    return global_ref_pts # [B, Q, n_levels, 2]


def project_to_matching_2point5d_cam_points(lidar_points, cam_transformations, orig_spatial_shapes, num_cameras=6):
    assert isinstance(cam_transformations, dict) and 'lidar2img' in cam_transformations and 'lidar2cam' in cam_transformations
    point_2p5d_cam = Projections.project_lidar_points_to_all_2point5d_cams_batch(lidar_points, cam_transformations['lidar2img']).transpose(1,2) # [B, n_objs, 6, 3]
    point_3d_cam = Projections.project_lidar_points_to_all_3d_cams_batch(lidar_points, cam_transformations['lidar2cam']).transpose(1, 2) # [B, n_objs, 6, 3]

    in_front = point_3d_cam[..., 2] > 0
    in_2d_range = (point_2p5d_cam[..., 0] >= 0) & (point_2p5d_cam[..., 0] < Projections.IMG_SIZE[0]) & \
                (point_2p5d_cam[..., 1] >= 0) & (point_2p5d_cam[..., 1] < Projections.IMG_SIZE[1])
    matches = in_front & in_2d_range # [B, n_objs, 6]
    matches_int = matches.type(torch.int32)

    res = point_2p5d_cam[matches] # [n_objs, 3]
    match_cams = matches_int.nonzero()[..., -1] # [n_objs]

    res_norm=res.clone()
    res_norm[..., 0] = res_norm[..., 0] / Projections.IMG_SIZE[0]
    res_norm[..., 1] = res_norm[..., 1] / Projections.IMG_SIZE[1]

    global_x_inds = res_norm[..., 0:1] * (orig_spatial_shapes[:, 1])[None].expand(*res_norm.shape[:1], -1) # [n_objs, n_levels]
    global_x_inds = global_x_inds + (match_cams.unsqueeze(-1) * orig_spatial_shapes[:, 1][None].expand(*res_norm.shape[:1], -1))
    global_ref_x = global_x_inds / (orig_spatial_shapes[:, 1] * num_cameras) # [n_objs, n_levels]
    global_ref_x = global_ref_x.unsqueeze(-1) # [n_objs, n_levels, 1]
    n_levels = orig_spatial_shapes.shape[0]
    # repeat y ratio for all levels (same ratio)
    global_ref_y = res_norm[..., 1:2].unsqueeze(1).expand(-1, n_levels, -1) # [n_objs, n_levels, 1]

    assert global_ref_x.shape == global_ref_y.shape
    global_ref_pts = torch.cat([global_ref_x, global_ref_y, res_norm[..., 2:3].unsqueeze(1).expand(-1, n_levels, -1)], 
                               dim=-1) # [n_objs, n_levels, 3]
    return global_ref_pts