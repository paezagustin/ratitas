# --- Modelo más simple ---
model = dict(
    type='EncoderDecoder',
    backbone=dict(type='ResNet18', depth=18, num_stages=4, out_indices=(0,1,2,3),
                  norm_cfg=dict(type='BN', requires_grad=True),
                  style='pytorch'),
    decode_head=dict(
        type='FCNHead',
        in_channels=512,
        channels=64,
        num_classes=2,  # cambia según tus clases
        loss_decode=dict(type='CrossEntropyLoss', loss_weight=1.0)),
    auxiliary_head=None
)

# --- Dataset ---
dataset_type = 'CustomDataset'
data_root = 'dataset/'  # ruta a tu dataset

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img','gt_semantic_seg'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(512, 512)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/train',
        ann_dir='ann_dir/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/test',
        ann_dir='ann_dir/test',
        pipeline=test_pipeline)
)

# --- Entrenamiento mínimo ---
optimizer = dict(type='Adam', lr=0.0001)
lr_config = dict(policy='step', step=[50])
runner = dict(type='EpochBasedRunner', max_epochs=5)

log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
checkpoint_config = dict(interval=5)
