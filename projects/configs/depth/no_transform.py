dataset_type = 'CustomNuScenesDataset'
data_root = './data/nuscenes/'
trainval_ann_filename='nuscenes2d_with_circular_cams_temporal_infos_trainval.pkl'
collect_keys=['lidar2img', 'intrinsics', 'extrinsics','timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv']
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)
queue_length = 1

ida_aug_conf = {
    "resize_lim": (0.38, 0.55),
    "final_dim": (256, 704),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    # "rand_flip": True,
    "rand_flip": False
}
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True,
        with_label=True, with_bbox_depth=True),
    # dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeCropFlipRotImage', data_aug_conf = ida_aug_conf, training=False),
    dict(type='PETRFormatBundle3D', class_names=class_names, collect_keys=collect_keys + ['prev_exists']),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'prev_exists'] + collect_keys,
             meta_keys=('filename', 'ori_shape', 'img_shape', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token', 'gt_bboxes_3d','gt_labels_3d'))
]

data = dict(
    type=dataset_type,
    pipeline=pipeline,
    seq_split_num=2, # streaming video training
    seq_mode=True, # streaming video training
    collect_keys=collect_keys + ['img', 'prev_exists', 'img_metas'], 
    queue_length=queue_length,
    ann_file=data_root + trainval_ann_filename,
    classes=class_names, 
    modality=input_modality,
    filter_empty_gt=False,
    use_valid_flag=True,
    box_type_3d='LiDAR',
    test_mode=False # to load gt bboxes
)