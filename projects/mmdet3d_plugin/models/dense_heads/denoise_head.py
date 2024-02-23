import torch
import copy
import warnings
from torch import nn

from mmdet.models.utils import build_transformer
from mmcv.cnn import Linear, bias_init_with_prob, xavier_init, build_norm_layer
from projects.mmdet3d_plugin.models.utils.projections import Projections, convert_3d_to_2d_global_cam_ref_pts
from projects.mmdet3d_plugin.attentions.custom_deform_attn import CustomDeformAttn
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from projects.mmdet3d_plugin.models.utils.lidar_utils import clamp_to_lidar_range, normalize_lidar, denormalize_lidar
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmdet.models.utils.transformer import inverse_sigmoid
from projects.mmdet3d_plugin.models.utils.positional_encoding import pos2posemb3d, pos2posemb1d, nerf_positional_encoding
from mmcv.cnn.bricks.transformer import TransformerLayerSequence, TRANSFORMER_LAYER_SEQUENCE, BaseTransformerLayer, TRANSFORMER_LAYER
from mmdet.core import (build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models import HEADS, build_loss

@HEADS.register_module()
class DenoiseHead(nn.Module):
    def __init__(self, embed_dims=256, scalar=10, bbox_noise_scale=0.5, 
                 bbox_noise_trans=0.0, pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], split=0.75, 
                 num_classes=10, transformer=None, code_weights=None, use_inv_sigmoid=False,
                 loss_cls=None, loss_bbox=None):
        super().__init__()
        self.embed_dims=embed_dims
        self.with_dn = True
        self.scalar=scalar
        self.bbox_noise_scale=bbox_noise_scale
        self.bbox_noise_trans=bbox_noise_trans
        self.pc_range=nn.Parameter(torch.tensor(pc_range), False)
        self.split=split
        self.num_classes=num_classes
        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims*3//2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        self.transformer = build_transformer_layer_sequence(transformer)

        cls_branch = []
        for _ in range(2):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
            # cls_branch.append(nn.ReLU(inplace=False))
        # if self.normedlinear:
        #     cls_branch.append(NormedLinear(self.embed_dims, 10))
        # else:
        cls_branch.append(Linear(self.embed_dims, 10))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(2):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 10))
        reg_branch = nn.Sequential(*reg_branch)
        self.cls_branches = nn.ModuleList(
            [copy.deepcopy(fc_cls) for _ in range(self.transformer.num_layers)])
        self.reg_branches = nn.ModuleList(
            [copy.deepcopy(reg_branch) for _ in range(self.transformer.num_layers)])

        ## los cfg
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        
        self.cls_out_channels = num_classes
        self.sync_cls_avg_factor=False
        self.code_weights = nn.Parameter(torch.tensor(code_weights), False)
        self.use_inv_sigmoid=use_inv_sigmoid

    def prepare_for_dn(self, batch_size, img_metas, device):
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
            known_labels = labels.repeat(self.scalar, 1).view(-1).long().to(device)
            known_bid = batch_idx.repeat(self.scalar, 1).view(-1)
            known_bboxs = boxes.repeat(self.scalar, 1).to(device)
            known_bbox_center = known_bboxs[:, :3].clone()
            known_bbox_scale = known_bboxs[:, 3:6].clone()

            if self.bbox_noise_scale > 0:
                diff = known_bbox_scale / 2 + self.bbox_noise_trans
                rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0 # in range[-1,1]
                known_bbox_center += torch.mul(rand_prob, diff) * self.bbox_noise_scale
                known_bbox_center[..., 0:3] = (known_bbox_center[..., 0:3] - self.pc_range[0:3]) / (self.pc_range[3:6] - self.pc_range[0:3])

                known_bbox_center = known_bbox_center.clamp(min=0.0, max=1.0)
                mask = torch.norm(rand_prob, 2, 1) > self.split
                known_labels[mask] = self.num_classes
            
            single_pad = int(max(known_num))
            pad_size = int(single_pad * self.scalar)
            padded_reference_points = torch.zeros([batch_size, pad_size, 3], device=device)

            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(self.scalar)]).long()
            if len(known_bid):
                padded_reference_points[(known_bid.long(), map_known_indice)] = known_bbox_center.to(device)

            tgt_size = pad_size
            attn_mask = torch.ones(tgt_size, tgt_size).to(device) < 0
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
            # query_size = pad_size + self.num_query + self.num_propagated
            query_size = pad_size
            # tgt_size = pad_size + self.num_query + self.memory_len
            tgt_size = pad_size

            temporal_attn_mask = torch.ones(query_size, tgt_size).to(device) < 0
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

            padded_reference_points,attn_mask, mask_dict = None, None, None

        return padded_reference_points, attn_mask, mask_dict
    
    def init_weights(self):
        self.transformer.init_weights()
        bias_init = bias_init_with_prob(0.01)
        for m in self.cls_branches:
            nn.init.constant_(m[-1].bias, bias_init)
    
    def forward(self, img_metas, orig_spatial_shapes, flattened_spatial_shapes, 
                flattened_level_start_index, **data):
        memory=data['img_feats_flatten']
        n_tokens = flattened_spatial_shapes.prod(1).sum().item()
        B = data['img'].size(0)
        assert list(memory.shape) == [B, n_tokens, self.embed_dims]

        # padded_reference_points: in range [0,1]
        # NOTE: attn_mask not required if not doing temporal self-attention
        
        padded_reference_points,attn_mask, mask_dict=self.prepare_for_dn(B, img_metas, data['img'].device)
        if mask_dict is not None and mask_dict['pad_size'] > 0:
            assert list(padded_reference_points.shape) == [B, mask_dict['pad_size'], 3]
        query_pos = self.query_embedding(pos2posemb3d(padded_reference_points)) # [B, pad_size, 3]
        tgt = torch.zeros_like(query_pos)
        # ref_pts_out: unnormalized in pc_range
        all_outs_dec, all_ref_pts_dec, init_ref_pts = self.transformer(bbox_embed=self.reg_branches,
                                                  query=tgt, key=memory, value=memory,
                                                  query_pos=query_pos, attn_masks=attn_mask,
                                                  reference_points=padded_reference_points,
                                                  orig_spatial_shapes=orig_spatial_shapes,
                                                  flattened_spatial_shapes=flattened_spatial_shapes,
                                                  flattened_level_start_index=flattened_level_start_index,
                                                  img_metas=img_metas,
                                                  **data)
        all_query_out = torch.nan_to_num(all_outs_dec)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(all_outs_dec.shape[0]):
            # NOTE:expecting all ref pts to already be unnormalized i.e. in lidar R range
            out_known_cls = self.cls_branches[lvl](all_query_out[lvl]) # [B, Q, num_classes]
            out_coord_offset = self.reg_branches[lvl](all_query_out[lvl])
            out_known_coord = out_coord_offset
            if lvl == 0:
                out_known_coord[..., :3] += init_ref_pts[..., :3]
            else:
                out_known_coord[..., :3] += all_ref_pts_dec[lvl-1][..., :3]
            if self.use_inv_sigmoid:
                # cannot sigmoid the output if denormalizing into lidar range because
                # for ex. [34.10, 7.85, -0.2], the first two (x,y) values are perfectly normal values
                # but when sigmoided turns into basically 1.0 (in fact 1 is 0.7311, and 3 is already 0.95)
                # so 99% of everything will hit bounds
                out_known_coord[..., :3] =out_known_coord[..., :3].sigmoid()
            outputs_classes.append(out_known_cls)
            outputs_coords.append(out_known_coord)
        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)
        if self.use_inv_sigmoid:
            assert (torch.logical_and(all_bbox_preds[..., :3] >= 0.0, all_bbox_preds[..., :3] <= 1.0)).all()
            all_bbox_preds[..., :3] = denormalize_lidar(all_bbox_preds[..., :3], self.pc_range)
        else:
            all_bbox_preds = clamp_to_lidar_range(all_bbox_preds, self.pc_range)


        if mask_dict and mask_dict['pad_size'] > 0:
            mask_dict['output_known_lbs_bboxes'] = (all_cls_scores, all_bbox_preds)
            outs = {
                'dn_mask_dict': mask_dict
            }
        else:
            outs = {
                'dn_mask_dict': None
            }
        return outs

    def dn_loss_single(self,cls_scores,
                    bbox_preds,
                    known_bboxs,
                    known_labels,
                    num_total_pos=None):
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

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        
        return loss_cls, loss_bbox
    
    def loss_single(self, mask_dict, device):
        if mask_dict is not None:
            ## prepare_for_loss for dn loss
            # output_known_class: [num_layers, B, Q, num_classes]
            output_known_class, output_known_coord = mask_dict['output_known_lbs_bboxes']
            known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
            map_known_indice = mask_dict['map_known_indice'].long()
            known_indice = mask_dict['known_indice'].long().cpu()
            batch_idx = mask_dict['batch_idx'].long()
            bid = batch_idx[known_indice]
            if len(output_known_class) > 0:
                # output_known_class = output_known_class.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
                # output_known_coord = output_known_coord.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
                output_known_class = output_known_class[(bid, map_known_indice)]
                output_known_coord = output_known_coord[(bid, map_known_indice)]
            num_tgt = known_indice.numel()
            dn_losses_cls, dn_losses_bbox = self.dn_loss_single(output_known_class, output_known_coord,
                                                                known_bboxs, known_labels, num_tgt)
            loss_dict = {
                "dn_loss_cls": dn_losses_cls,
                "dn_loss_bbox": dn_losses_bbox
            }
        
        return loss_dict

    def loss(self, preds_dicts, device):
        mask_dict = preds_dicts['dn_mask_dict']
        if mask_dict is not None:
            output_known_class, output_known_coord = mask_dict['output_known_lbs_bboxes']
            num_layers=output_known_coord.size(0)
            known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
            map_known_indice = mask_dict['map_known_indice'].long()
            known_indice = mask_dict['known_indice'].long().cpu()
            batch_idx = mask_dict['batch_idx'].long()
            bid = batch_idx[known_indice]
            if len(output_known_class) > 0:
                # [num_layers, B, Q, 10] -> [B, Q, num_layers, 10] -> [num_gt, num_layers, 10] -> [num_layers, num_gt, 10]
                output_known_class = output_known_class.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
                output_known_coord = output_known_coord.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
            num_tgt = known_indice.numel()
            all_known_bboxs_list = [known_bboxs for _ in range(num_layers)]
            all_known_labels_list = [known_labels for _ in range(num_layers)]
            all_num_tgts_list = [num_tgt for _ in range(num_layers)]
            dn_losses_cls, dn_losses_bbox = multi_apply(
                self.dn_loss_single, output_known_class, output_known_coord,
                all_known_bboxs_list, all_known_labels_list, 
                all_num_tgts_list)
            loss_dict = {
                "dn_loss_cls": dn_losses_cls[-1],
                "dn_loss_bbox": dn_losses_bbox[-1]
            }
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(dn_losses_cls[:-1],
                                            dn_losses_bbox[:-1]):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                num_dec_layer += 1
        else:
            print("NO GT BBOX")
            loss_dict = {
                "dn_loss_cls":torch.tensor(0., device=device),
                "dn_loss_bbox": torch.tensor(0., device=device)
            }
        return loss_dict

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DenoiseTransformerDecoder(TransformerLayerSequence):
    def __init__(self, mlvl_feats_formats=1, pc_range=None, use_inv_sigmoid=False, 
                 return_intermediate=True, post_norm_cfg=dict(type='LN'), **kwargs):
        self._iter=0
        super(DenoiseTransformerDecoder, self).__init__(**kwargs)
        self.pc_range = nn.Parameter(torch.tensor(pc_range), requires_grad=False)
        self.mlvl_feats_formats=mlvl_feats_formats
        self.use_inv_sigmoid=use_inv_sigmoid
        self.return_intermediate=return_intermediate
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg,
                                              self.embed_dims)[1]
        else:
            self.post_norm = None
        # ## loss cfg
        # self.num_classes=num_classes
        # self.match_with_velo=match_with_velo
        # self.code_weights=nn.Parameter(torch.tensor(code_weights), requires_grad=False)
        # self.match_costs=nn.Parameter(torch.tensor(code_weights), requires_grad=False) if match_costs is None else match_costs
        # self.loss_cls = build_loss(loss_cls)
        # self.loss_bbox = build_loss(loss_bbox)
        # self.loss_iou = build_loss(loss_iou)

        # if self.loss_cls.use_sigmoid:
        #     self.cls_out_channels = num_classes
        # else:
        #     self.cls_out_channels = num_classes + 1

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight is not None and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        for i in range(self.num_layers):
            for attn in self.layers[i].attentions:
                if isinstance(attn, CustomDeformAttn):
                    assert attn.is_init == False
                    # print("init iq encoder custom deform attn")
                    attn.reset_parameters()
        self._is_init = True

    def forward(self, bbox_embed=None, query=None, 
                orig_spatial_shapes=None, 
                reference_points=None, 
                img_metas=None, 
                lidar2img=None, extrinsics=None, attn_masks=None, **kwargs):
        assert query.shape[:-1] == reference_points.shape[:-1]
        cam_transformations = dict(lidar2img=lidar2img, lidar2cam=extrinsics)
        assert lidar2img.dim() == 4, f"got lidar2img shape: {lidar2img.shape}"
        assert extrinsics.dim()==4, f"got extrinsics shape: {extrinsics.shape}"

        intermediate=[]
        intermediate_reference_points=[]
        init_ref_pts = None
        if self.num_layers > 1: assert bbox_embed is not None
        for lid, layer in enumerate(self.layers):
            if self.use_inv_sigmoid:
                if self._iter == 0: print("using inverse sigmoid")
                ref_pts_unnormalized = inverse_sigmoid(reference_points.clone())
            else:
                if self._iter == 0: print("NOT using inverse sigmoid")
                ref_pts_unnormalized = denormalize_lidar(reference_points.clone(), self.pc_range)
            if lid == 0:
                init_ref_pts = ref_pts_unnormalized
            with torch.no_grad():
                ref_pts_2d_cam, non_matches = convert_3d_to_2d_global_cam_ref_pts(cam_transformations, 
                                                ref_pts_unnormalized, orig_spatial_shapes, img_metas,
                                                ret_num_non_matches=True)
            num_non_match_prop = non_matches.sum(1) / non_matches.size(-1)
            if self._iter % 20 == 0:
                print(f"non match proportion @ layer {lid}: {num_non_match_prop}")
            query = layer(query, reference_points=ref_pts_2d_cam, orig_spatial_shapes=orig_spatial_shapes, **kwargs)
            query_out = torch.nan_to_num(query)
            coord_offset = bbox_embed[lid](query_out)[..., :3] # [B, Q, 3]
            coord_pred = coord_offset
            coord_pred[..., :3] += ref_pts_unnormalized[..., :3]
            if self.use_inv_sigmoid:
                coord_pred[..., :3] = coord_pred[..., :3].sigmoid()
                next_ref_pts = coord_pred[..., :3].detach().clone()
                unnormalized_ref_pts = inverse_sigmoid(coord_pred[..., :3].detach().clone())
                # coord_pred[..., :3] = denormalize_lidar(coord_pred[..., :3], self.pc_range)
            else:
                coord_pred = clamp_to_lidar_range(coord_pred, self.pc_range)
                unnormalized_ref_pts = coord_pred.detach().clone()
                next_ref_pts = normalize_lidar(coord_pred.detach().clone(), self.pc_range)
            if self.return_intermediate:
                intermediate_reference_points.append(unnormalized_ref_pts)
                if self.post_norm is not None:
                    intermediate.append(self.post_norm(query))
                else:
                    intermediate.append(query)
            reference_points = next_ref_pts

        self._iter += 1
        return torch.stack(intermediate), torch.stack(intermediate_reference_points), init_ref_pts

@TRANSFORMER_LAYER.register_module()
class DenoiseTransformerDecoderLayer(BaseTransformerLayer):
    def __init__(self,
                 **kwargs):
        super(DenoiseTransformerDecoderLayer, self).__init__(**kwargs)

    def forward(self, query, 
                key=None,
                value=None,
                key_pos=None,
                query_pos=None,
                key_padding_mask=None,
                attn_masks=None,
                reference_points=None,
                orig_spatial_shapes=None,
                flattened_spatial_shapes=None,
                flattened_level_start_index=None,
                num_cameras=6,
                **kwargs):
        norm_index=0
        attn_index=0
        ffn_index=0
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'
        for layer in self.operation_order:
            if layer == "cross_attn":
                query = self.attentions[attn_index](
                    query,
                    value,
                    key=key, # None if two_stage
                    identity=None,
                    query_pos=query_pos,
                    key_pos=key_pos, # None if two stage
                    reference_points=reference_points,
                    spatial_shapes=orig_spatial_shapes,
                    flattened_spatial_shapes=flattened_spatial_shapes,
                    flattened_lvl_start_index=flattened_level_start_index,
                    num_cameras=num_cameras,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    return_query_only=True)
                attn_index += 1
            elif layer == 'ffn':
                query = self.ffns[ffn_index](query, None)
                ffn_index += 1
            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1
        return query