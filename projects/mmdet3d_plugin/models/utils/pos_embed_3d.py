import torch
from torch import nn
from torch.nn.init import constant_, xavier_uniform_
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmcv.runner import BaseModule
from mmdet.models.utils.transformer import inverse_sigmoid


@PLUGIN_LAYERS.register_module()
class PositionEmbedding3d(BaseModule):
    def __init__(self, embed_dims=256,spatial_alignment_all_memory=True, depth_start = 1.0, depth_step=0.8, depth_num=64,
                 position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0], use_inv_sigmoid_in_pos_embed=True,
                 use_norm_input_in_pos_embed=False,flattened_inp=True):
        super().__init__()
        self.embed_dims=embed_dims
        self.spatial_alignment_all_memory=spatial_alignment_all_memory
        self.depth_num=depth_num # 64
        self.depth_step = depth_step # 0.8
        self.position_dim = depth_num * 3 # 192
        self.position_range=nn.Parameter(torch.tensor(position_range), requires_grad=False)
        self.depth_start=depth_start
        self.depth_range=position_range[3] - self.depth_start
        index  = torch.arange(start=0, end=self.depth_num, step=1).float()
        index_1 = index + 1
        bin_size = (self.depth_range) / (self.depth_num * (1 + self.depth_num))
        coords_d = self.depth_start + bin_size * index * index_1
        self.position_encoder = nn.Sequential(
                nn.Linear(self.position_dim, self.embed_dims*4),
                nn.ReLU(),
                nn.Linear(self.embed_dims*4, self.embed_dims),
            )
        self.coords_d = nn.Parameter(coords_d, requires_grad=False) # Tensor[64]
        self.use_inv_sigmoid_in_pos_embed=use_inv_sigmoid_in_pos_embed
        self.use_norm_input_in_pos_embed=use_norm_input_in_pos_embed
        assert (self.use_inv_sigmoid_in_pos_embed and not self.use_norm_input_in_pos_embed) or \
            (self.use_norm_input_in_pos_embed and not self.use_inv_sigmoid_in_pos_embed) or \
            (not self.use_inv_sigmoid_in_pos_embed and not self.use_norm_input_in_pos_embed)
        # if flattened_inp, expecting input to be [B, H*N*W, ...], otherwise expecting [B, N, C, H, W]
        self.flattened_inp=flattened_inp
        

    def position_embeding_flattened(self, data, locations, img_metas, orig_spatial_shapes, lidar2img):
        """generate 3d pos embeds for img tokens

        Args:
            data (_type_): dict:
                - img_feats: list of len n_features levels, with elements Tensor[B, N, C, h_i, w_i]
                - img_feats_flatten: concatenated flattened features Tensor[B, h0*N*w0+..., C]
            locations (_type_): flattened locations Tensor [B, h0*N*w0+..., 2]
            img_metas (_type_): _description_
            flattened_spatial_shapes: [n_levels, 2]
        Returs:
            coords_position_embeding: [B, k, C]
            cone: [B, k, 8]
        """
        # assert self.with_pts_bbox
        if self.spatial_alignment_all_memory == False:
            raise NotImplementedError()
        eps=1e-5
        D = self.coords_d.shape[0]
        # [B, N, 2]
        intrinsic = torch.stack([data['intrinsics'][..., 0, 0], data['intrinsics'][..., 1, 1]], dim=-1)
        intrinsic = torch.abs(intrinsic) / 1e3
        intrinsic_all = []
        img2lidars_all = []
        img2lidars = torch.inverse(lidar2img.to('cpu')).to('cuda') # [B, N, 4, 4]
        for (h, w) in orig_spatial_shapes:
            intrinsic_rep = intrinsic[:, None, :, None, :].expand(-1, h, -1, w, -1)
            img2lidars_rep=img2lidars.clone()
            img2lidars_rep = img2lidars_rep[:, None, :, None, :, :].expand(-1, h, -1, w, -1, -1)
            intrinsic_all.append(intrinsic_rep.flatten(1, 3)) # [B, h*N*w, 2]
            img2lidars_all.append(img2lidars_rep.flatten(1, 3)) # [B, h*N*w, 4, 4]
        intrinsic_all = torch.cat(intrinsic_all, 1) # [B, h0*N*w0+..., 2]
        img2lidars_all = torch.cat(img2lidars_all, 1) # [B, h0*N*w0+..., 4, 4]
        assert list(intrinsic_all.shape) == list(data['img_feats_flatten'].shape[:-1]) + [2], \
            f"intrinsic_all shape: {intrinsic_all.shape} expecting: {data['img_feats_flatten'].shape}"
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        locations[..., 0] = locations[..., 0] * pad_w
        locations[..., 1] = locations[..., 1] * pad_h

        locations = locations.detach().unsqueeze(2).expand(-1, -1, D, -1) # [B, h0*N*w0+..., D, 2]

        # [B, h0*N*w0+..., D, 1]
        coords_d = self.coords_d.view(1, 1, D, 1).expand(*locations.shape[:2], -1, -1)
        # [B, h0*N*w0+..., D, 4]
        coords = torch.cat([locations, coords_d, torch.ones_like(locations[..., :1])], dim=-1)
        assert list(coords.shape) == list(locations.shape[:-1]) + [4]
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)
        
        coords = coords.unsqueeze(-1) # [B, h0*N*w0+..., D, 4, 1]

        img2lidars = img2lidars_all.unsqueeze(2).expand(-1, -1, D, -1, -1) # [B, h0*N*w0+..., D, 4, 4]
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3] # [B, h0*N*w0+..., D, 3]

        if not self.use_norm_input_in_pos_embed and not self.use_inv_sigmoid_in_pos_embed:
            pos_embed = coords3d.clone() # [B, h0*N*w0+..., D, 3]
            pos_embed = pos_embed.flatten(-2, -1) # [B, h0*N*w0+..., D*3]

        coords3d[..., 0:3] = (coords3d[..., 0:3] - self.position_range[0:3]) / (self.position_range[3:6] - self.position_range[0:3])

        coords3d = coords3d.flatten(-2, -1) # [B, h0*N*w0+..., D*3]
        # [B, k, D*3]
        # topk_coords3d = torch.gather(coords3d, 1, topk_inds.unsqueeze(-1).expand(-1, -1, coords3d.size(-1)))
        if self.use_inv_sigmoid_in_pos_embed:
            pos_embed = inverse_sigmoid(coords3d) # [B, h0*N*w0+..., D*3]
        elif self.use_norm_input_in_pos_embed:
            pos_embed=coords3d
        coords_position_embeding = self.position_encoder(pos_embed) # [B, h0*N*w0+..., C]

        assert intrinsic_all.shape[:-1] == coords3d.shape[:-1]
        # [B, h0*N*w0+..., 8]
        cone = torch.cat([intrinsic_all, coords3d[..., -3:], coords3d[..., -90:-87]], dim=-1) 

        return coords_position_embeding, cone, img2lidars_all
    
    def position_embeding(self, data, locations, img_metas, orig_spatial_shapes, lidar2img):
        pass
    
    def forward(self, data, locations, img_metas, orig_spatial_shapes, lidar2img):
        if self.flattened_inp:
            outs = self.position_embeding_flattened(data, locations, img_metas, orig_spatial_shapes, lidar2img)
        else:
            outs = self.position_embeding(data, locations, img_metas, orig_spatial_shapes, lidar2img)
        return outs