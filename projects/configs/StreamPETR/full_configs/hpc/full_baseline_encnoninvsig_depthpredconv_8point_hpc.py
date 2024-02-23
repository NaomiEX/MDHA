_base_ = [
    "../../default.py"
]
backbone_norm_cfg = dict(type='LN', requires_grad=True)
plugin=True
plugin_dir='projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
num_gpus=2
batch_size=8
num_iters_per_epoch = 28130 // (num_gpus * batch_size)
num_epochs = 200

cfg_name = "full_baseline_depthpredconv_8point_hpc"
debug_args = dict(
    debug=False,
    log_file=f"./experiments/debug/{cfg_name}_log.log"
)

embed_dims=256
strides=[4, 8, 16, 32]
num_levels=len(strides)

queue_length = 1
num_frame_losses = 1
collect_keys=['lidar2img', 'intrinsics', 'extrinsics','timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv']
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

mlvl_feats_formats = {
    "[B, N, HW, C]": 0,
    "[B, HNW, C]": 1
}

mlvl_feats_format = mlvl_feats_formats["[B, HNW, C]"]

depth_pred_position_types = {
    "before_feature_flattening": 0,
    "within_encoder": 1,
}

depth_pred_position=depth_pred_position_types["before_feature_flattening"]

spatial_alignment="encoder"
pos_embed3d="encoder"

use_inv_sigmoid = {
    "encoder": False,
    "decoder": False 
}
global_deform_attn_wrap_around=False

position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
depth_start=1.0

## modules
modules = dict(
    encoder = True,
    pts_bbox_head=True,
    dn_head = False,
    img_roi_head=False,
    mask_predictor=False,
    depth_net = True,
)

mask_pred_target=[]

obj_det3d_loss_cfg = dict(
    num_classes=10,
    match_with_velo=False,
    code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    loss_cls=dict(
        type='FocalLoss',
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=2.0),
    loss_bbox=dict(type='L1Loss', loss_weight=0.25),
    loss_iou=dict(type='GIoULoss', loss_weight=0.0),
)

enc_obj_det3d_loss_cfg = dict(
    num_classes=10,
    match_with_velo=False,
    code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    loss_cls=dict(
        type='FocalLoss',
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=2.0),
    loss_bbox=dict(type='L1Loss', loss_weight=0.25),
    loss_iou=dict(type='GIoULoss', loss_weight=0.0),
)

train_cfg_obj_det3d=dict(
    grid_size=[512, 512, 1],
    voxel_size=voxel_size,
    point_cloud_range=point_cloud_range,
    out_size_factor=4,
    assigner=dict(
        type='HungarianAssigner3D',
        cls_cost=dict(type='FocalLossCost', weight=2.0),
        reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
        iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head. 
        pc_range=point_cloud_range),)

# NOTE: the encoder config is from encoder_anchor_fixedfull_xyrefptsqencoding_1gpu
encoder = dict(
    type="IQTransformerEncoder",
    return_intermediate=False,
    num_layers=1,

    mlvl_feats_formats=mlvl_feats_format,
    # pc range is set by petr3d
    learn_ref_pts_type="anchor",
    use_spatial_alignment=spatial_alignment == "encoder",
    spatial_alignment_all_memory=True, # if False only topk
    use_pos_embed3d=pos_embed3d == "encoder",
    encode_query_with_ego_pos=False,
    encode_query_pos_with_ego_pos=False,

    encode_ref_pts_depth_into_query_pos=True,
    ref_pts_depth_encoding_method="mln",

    use_inv_sigmoid=use_inv_sigmoid["encoder"],
    use_inv_sigmoid_in_pos_embed=True, # !
    use_norm_input_in_pos_embed=False,

    sync_cls_avg_factor=False,
    
    mask_predictor=dict(
        type="MaskPredictor",
        in_dim=embed_dims,
        hidden_dim=embed_dims,
        loss_type="multilabel_soft_margin_loss",
    ) if modules['mask_predictor'] else None,
    mask_pred_before_encoder=False,
    mask_pred_target=mask_pred_target, # ! IF USING DECODER SHOULD INCLUDE DECODER AS WELL
    sparse_rho=0.8,
    
    process_backbone_mem=True, 
    process_encoder_mem=True,

    position_range=position_range,

    reference_point_generator = dict(
        type="ReferencePoints",
        coords_depth_type="learnable",
    ),

    transformerlayers=dict(
        type="IQTransformerEncoderLayer",
        batch_first=True,
        attn_cfgs=[
            dict(
                type="CustomDeformAttn",
                embed_dims=256,
                num_heads=8,
                proj_drop=0.1,
                n_levels=num_levels,
                n_points=4*2, # num_points * num_cams
                with_wrap_around=global_deform_attn_wrap_around,
                key_weight_modulation=False,
                div_sampling_offset_x=False,
                mlvl_feats_format=mlvl_feats_format,
                ref_pts_mode="single",
                encode_2d_ref_pts_into_query_pos=False,
            )
        ],
        feedforward_channels=1024,
        ffn_dropout=0.1,
        operation_order=('self_attn', 'norm', 'ffn', 'norm')
    ),
    **enc_obj_det3d_loss_cfg,
)

ref_pts_mode = "single" # either single or multiple

pts_bbox_head=dict(
        type='StreamPETRHead',
        in_channels=256,
        num_query=644,
        memory_len=1024,
        topk_proposals=256,
        num_propagated=256,
        with_ego_pos=True,
        with_dn=True,
        scalar=10, ##noise groups
        noise_scale = 1.0, 
        dn_weight= 1.0, ##dn loss weight
        split = 0.75, ###positive rate
        LID=True,
        with_position=True,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],

        **obj_det3d_loss_cfg,

        ##new##
        init_ref_pts=False,
        init_pseudo_ref_pts=False,
        use_pos_embed3d=pos_embed3d=="decoder",
        use_spatial_alignment=spatial_alignment=="decoder",
        use_own_reference_points=False,
        two_stage=modules['encoder'],
        mlvl_feats_format=mlvl_feats_format,
        skip_first_frame_self_attn=False,
        init_pseudo_ref_pts_from_encoder_out=False,
        use_inv_sigmoid=use_inv_sigmoid["decoder"],
        ##
        transformer=dict(
            type='PETRTemporalTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                ref_pts_mode=ref_pts_mode,
                transformerlayers=dict(
                    type='PETRTemporalDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='PETRMultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            attn_drop=0.1,
                            proj_drop=0.1),
                        dict(
                            type="CustomDeformAttn",
                            embed_dims=256,
                            num_heads=8,
                            proj_drop=0.1,
                            n_levels=num_levels,
                            n_points=4*2, # num_points * num_cams
                            with_wrap_around=global_deform_attn_wrap_around,
                            key_weight_modulation=False,
                            div_sampling_offset_x=False,
                            mlvl_feats_format=mlvl_feats_format,
                            ref_pts_mode=ref_pts_mode,
                            encode_2d_ref_pts_into_query_pos=False,
                        )
                        ],
                    feedforward_channels=2048, # TODO: TRY WITH JUST 1024
                    ffn_dropout=0.1,
                    with_cp=True,  ###use checkpoint to save memory
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'),
                    batch_first=True,
                ),
            )),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10), 
        )

    # model training and testing settings

img_roi_head=dict(
        type='FocalHead',
        num_classes=10,
        in_channels=256,
        loss_cls2d=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=2.0),
        loss_centerness=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        loss_bbox2d=dict(type='L1Loss', loss_weight=5.0),
        loss_iou2d=dict(type='GIoULoss', loss_weight=2.0),
        loss_centers2d=dict(type='L1Loss', loss_weight=10.0),
        train_cfg=dict(
            assigner2d=dict(
                type='HungarianAssigner2D',
                cls_cost=dict(type='FocalLossCost', weight=2.),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                centers2d_cost=dict(type='BBox3DL1Cost', weight=10.0))),
        strides=strides,
    )

model = dict(
    type='Petr3D',
    num_frame_head_grads=num_frame_losses,
    num_frame_backbone_grads=num_frame_losses,
    num_frame_losses=num_frame_losses,
    strides=strides,
    use_grid_mask=True,
    ##new##
    mlvl_feats_format=mlvl_feats_format,
    encoder=encoder if modules['encoder'] else None,
    num_cameras=6,
    pc_range=point_cloud_range,
    use_xy_embed=True,
    use_cam_embed=False,
    use_lvl_embed=modules['encoder'], # ! IMPORTANT: MAKE THIS TRUE IF USING ENCODER
    debug_args=debug_args,
    depth_pred_position=depth_pred_position,
    depth_net=dict(
        type="DepthNet",
        in_channels=embed_dims,
        depth_net_type="conv",
        depth_start=depth_start,
        depth_max = position_range[3],
        # extra args
        sigmoid_out=True,
        num_layers=2,
        shared=False,
        depth_weight_bound=True,
        depth_weight_limit=0.01,
        loss_depth=dict(type='L1Loss', loss_weight=0.01),
    ) if modules["depth_net"] else None,
    calc_depth_pred_loss=True,
    ##
    img_backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        frozen_stages=-1,
        norm_eval=False,
        style="pytorch",
        with_cp=True,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type="BN", requires_grad=True),
        pretrained="ckpt/resnet50-19c8e357.pth",
    ),
    img_neck=dict(
        type="FPN",
        num_outs=num_levels,
        start_level=0,
        out_channels=embed_dims,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
        in_channels=[256, 512, 1024, 2048],
    ),
    img_roi_head=img_roi_head if modules['img_roi_head'] else None,
    pts_bbox_head=pts_bbox_head if modules['pts_bbox_head'] else None,
    train_cfg=dict(
        pts=train_cfg_obj_det3d if modules['pts_bbox_head'] else None,
        encoder=train_cfg_obj_det3d if modules['encoder'] else None,
        )
    )


dataset_type = 'CustomNuScenesDataset'
data_root = './data/nuscenes/'

file_client_args = dict(backend='disk')


ida_aug_conf = {
        "resize_lim": (0.38, 0.55),
        "final_dim": (256, 704),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": True,
        # "rand_flip": False
    }

# TODO: ENSURE RESIZE DOESN'T CUT OFF BBOX CENTERS, OR FILTER OUT BBOXES WHICH ARE OUT OF CAMERA FRAME
# TODO: CURRENTLY I BELIEVE THAT ObjectRangeFilter just filters out the objects out of lidar range not cam range
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True,
        with_label=True, with_bbox_depth=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipRotImage', data_aug_conf = ida_aug_conf, training=True),
    dict(type='GlobalRotScaleTransImage',
            rot_range=[-0.3925, 0.3925],
            translation_std=[0, 0, 0],
            scale_ratio_range=[0.95, 1.05],
            reverse_angle=True,
            training=True,
            ),
    dict(type='FilterObjectOutOfFrame'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='PETRFormatBundle3D', class_names=class_names, collect_keys=collect_keys + ['prev_exists']),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'prev_exists'] + collect_keys,
             meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token', 'gt_bboxes_3d','gt_labels_3d'))
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipRotImage', data_aug_conf = ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PETRFormatBundle3D',
                collect_keys=collect_keys,
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img'] + collect_keys,
            meta_keys=('filename', 'ori_shape', 'img_shape','pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token'))
        ])
]

train_ann_filename='nuscenes2d_with_circular_cams_temporal_infos_train.pkl'
val_ann_filename='nuscenes2d_with_circular_cams_temporal_infos_val.pkl'
test_ann_filename='nuscenes2d_with_circular_cams_temporal_infos_val.pkl'

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + train_ann_filename,
        num_frame_losses=num_frame_losses,
        seq_split_num=2, # streaming video training
        seq_mode=True, # streaming video training
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        collect_keys=collect_keys + ['img', 'prev_exists', 'img_metas'],
        queue_length=queue_length,
        test_mode=False,
        use_valid_flag=True,
        filter_empty_gt=True,
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type, pipeline=test_pipeline, collect_keys=collect_keys + ['img', 'img_metas'], queue_length=queue_length, ann_file=data_root + val_ann_filename, classes=class_names, modality=input_modality),
    test=dict(type=dataset_type, pipeline=test_pipeline, collect_keys=collect_keys + ['img', 'img_metas'], queue_length=queue_length, ann_file=data_root + test_ann_filename, classes=class_names, modality=input_modality),
    shuffler_sampler=dict(type='InfiniteGroupEachSampleInBatchSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
    )


optimizer = dict(
    type='AdamW', 
    lr=4e-4, # bs 8: 2e-4 || bs 16: 4e-4
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.25), # 0.25 only for Focal-PETR with R50-in1k pretrained weights
        }),
    weight_decay=0.01)

optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic', grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
    )

evaluation = dict(interval=num_iters_per_epoch*num_epochs, pipeline=test_pipeline)
find_unused_parameters=False #### when use checkpoint, find_unused_parameters must be False
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=10)
runner = dict(
    type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
load_from=None
resume_from=None
