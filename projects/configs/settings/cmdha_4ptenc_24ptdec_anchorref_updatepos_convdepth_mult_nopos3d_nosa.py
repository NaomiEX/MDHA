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

enc_attn_num_points=4
dec_attn_num_points=24
embed_dims=256
strides=[4, 8, 16, 32]
num_levels=len(strides)
update_pos=True

queue_length = 1
num_frame_losses = 1
collect_keys=['lidar2img', 'intrinsics', 'extrinsics', 'timestamp', 'ego_pose', 'ego_pose_inv', 'focal']
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

spatial_alignment=None
pos_embed3d=None

use_inv_sigmoid = {
    "encoder": False,
    "decoder": False 
}
global_deform_attn_wrap_around=False

anchor_dim = 10
enc_with_quality_estimation=False
dec_with_quality_estimation=False

## modules
modules = dict(
    encoder = True,
    pts_bbox_head=True,
    dn_head = False,
    img_roi_head=False,
    mask_predictor=False,
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

position_embedding_3d = dict(
    type="PositionEmbedding3d",
    embed_dims=256,
    spatial_alignment_all_memory=True, 
    depth_start = 1.0, 
    # depth_step=0.8, 
    # depth_num=64,
    position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0], 
    use_inv_sigmoid_in_pos_embed=True,
    use_norm_input_in_pos_embed=False,
    flattened_inp=spatial_alignment!="mdha"
)

# ENSURE CONSISTENT WITH CONSTANTS.PY
depth_pred_positions = {
    "before_encoder": 0,
    "in_encoder": 1,
}
depth_pred_pos = depth_pred_positions["before_encoder"]

depthnet = dict(
    type="DepthNet",
    in_channels=embed_dims,
    depth_pred_position=depth_pred_pos,
    mlvl_feats_format=mlvl_feats_format,
    n_levels=len(strides),
    loss_depth = dict(type='L1Loss', loss_weight=0.01),
    depth_weight_bound=True,
    depth_weight_limit=0.4,
    use_focal=False,
    single_target=False,
    sigmoid_out=True,
)

encoder_anchor_refinement = dict(
    type="AnchorRefinement",
    embed_dims=embed_dims,
    output_dim=anchor_dim,
    num_cls=len(class_names),
    with_quality_estimation=enc_with_quality_estimation,
    refine_center_only=True,
    limit=False,
)

# NOTE: the encoder config is from encoder_anchor_fixedfull_xyrefptsqencoding_1gpu
encoder = dict(
    type="AnchorEncoder",
    num_layers=1,
    mlvl_feats_formats=mlvl_feats_format,
    # pc range is set by mdha
    learn_ref_pts_type="anchor",
    use_spatial_alignment=spatial_alignment == "encoder",
    encode_ref_pts_depth_into_query_pos=True,
    ref_pts_depth_encoding_method="mln",
    use_inv_sigmoid=use_inv_sigmoid["encoder"],
    sync_cls_avg_factor=False,
    position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],

    ## modules

    anchor_refinement=encoder_anchor_refinement,
    pos_embed3d= position_embedding_3d if pos_embed3d == "encoder" else None,
    mask_predictor=None,
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
                n_points=enc_attn_num_points, # num_points * num_cams
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

dec_ref_pts_mode = "single" # either single or multiple

dn_args = dict(
    scalar=10, ##noise groups
    noise_scale = 1.0, 
    dn_weight= 1.0, ##dn loss weight
    split = 0.75, ###positive rate
)

decoder_anchor_refinement = dict(
    type="AnchorRefinement",
    embed_dims=embed_dims,
    output_dim=anchor_dim,
    num_cls=len(class_names),
    with_quality_estimation=dec_with_quality_estimation,
    refine_center_only=True,
    limit=False,
)

pts_bbox_head=dict(
        type='StreamPETRHead',
        num_query=644,
        memory_len=1024,
        num_propagated=256,
        with_ego_pos=True,
        with_dn=True,

        **dn_args,

        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],

        **obj_det3d_loss_cfg,

        ##new##
        anchor_refinement=decoder_anchor_refinement,
        pos_embed3d= position_embedding_3d if pos_embed3d=="decoder" else None,
        use_spatial_alignment=spatial_alignment=="decoder",
        two_stage=modules['encoder'],
        mlvl_feats_format=mlvl_feats_format,
        use_inv_sigmoid=use_inv_sigmoid["decoder"],
        mask_pred_target="pts_bbox_head" in mask_pred_target,
        ##
        transformer=dict(
            type='MDHATemporalTransformer',
            decoder=dict(
                type='MDHATransformerDecoder',
                update_pos=update_pos,
                return_intermediate=True,
                num_layers=6,
                ref_pts_mode=dec_ref_pts_mode,
                transformerlayers=dict(
                    type='MDHATemporalDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MDHAMultiheadAttention',
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
                            n_points=dec_attn_num_points, # num_points * num_cams
                            with_wrap_around=global_deform_attn_wrap_around,
                            key_weight_modulation=False,
                            div_sampling_offset_x=False,
                            mlvl_feats_format=mlvl_feats_format,
                            ref_pts_mode=dec_ref_pts_mode,
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
model = dict(
    type='MDHA',
    num_frame_backbone_grads=num_frame_losses,
    strides=strides,
    use_grid_mask=True,
    ##new##
    depth_net=depthnet,
    depth_pred_position=depth_pred_pos,
    mlvl_feats_format=mlvl_feats_format,
    encoder=encoder if modules['encoder'] else None,
    num_cameras=6,
    pc_range=point_cloud_range,
    use_xy_embed=True,
    use_cam_embed=False,
    use_lvl_embed=modules['encoder'], # ! IMPORTANT: MAKE THIS TRUE IF USING ENCODER
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
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=False,
        with_label=False, with_bbox_depth=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipRotImage', data_aug_conf = ida_aug_conf, with_2d=False, training=True),
    dict(type='GlobalRotScaleTransImage',
            rot_range=[-0.3925, 0.3925],
            translation_std=[0, 0, 0],
            scale_ratio_range=[0.95, 1.05],
            reverse_angle=True,
            training=True,
            ),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='PETRFormatBundle3D', class_names=class_names, collect_keys=collect_keys + ['prev_exists']),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'prev_exists'] + collect_keys,
             meta_keys=['pad_shape'])
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipRotImage', data_aug_conf = ida_aug_conf, with_2d=False, training=False),
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
            meta_keys=['filename', 'ori_shape', 'img_shape','pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token'])
        ])
]

train_ann_filename='nuscenes2d_with_circular_cams_temporal_infos_train.pkl'
val_ann_filename='nuscenes2d_with_circular_cams_temporal_infos_val.pkl'
test_ann_filename='nuscenes2d_with_circular_cams_temporal_infos_val.pkl'
val_data_root='./data/nuscenes/'
data = dict(
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
        filter_empty_gt=False,
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type, pipeline=test_pipeline, data_root=val_data_root, 
             collect_keys=collect_keys + ['img', 'img_metas'], queue_length=queue_length, 
             ann_file=data_root + val_ann_filename, classes=class_names, modality=input_modality),
    test=dict(type=dataset_type, pipeline=test_pipeline, data_root=val_data_root, 
              collect_keys=collect_keys + ['img', 'img_metas'], queue_length=queue_length, 
              ann_file=data_root + test_ann_filename, classes=class_names, modality=input_modality),
    shuffler_sampler=dict(type='InfiniteGroupEachSampleInBatchSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
    )

# evaluation = dict(interval=num_iters_per_epoch*num_epochs, pipeline=test_pipeline)
evaluation = dict(pipeline=test_pipeline)