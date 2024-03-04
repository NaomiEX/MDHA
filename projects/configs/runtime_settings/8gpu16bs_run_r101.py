_base_ = [
    "./default.py"
]
num_gpus=8
batch_size=2
num_iters_per_epoch = 28130 // (num_gpus * batch_size)
num_epochs = 100


BS_LR_MAP = {
    16: 4e-4,
    8: 2e-4,
    2: 5e-5
}
optimizer = dict(
    type='AdamW', 
    lr=2e-4, # bs 8: 2e-4 || bs 16: 4e-4
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1), # 0.25 only for Focal-PETR with R50-in1k pretrained weights
        }),
    weight_decay=0.01)

optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic', grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
    )

evaluation_interval=num_iters_per_epoch*num_epochs
find_unused_parameters=False #### when use checkpoint, find_unused_parameters must be False
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=20)
runner = dict(
    type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
load_from ="ckpt/cascade_mask_rcnn_r101_fpn_1x_nuim_20201024_134804-45215b1e.pth"
resume_from=None