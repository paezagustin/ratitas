custom_imports = dict(
    imports=['mmseg.datasets.custom_dataset'],
    allow_failed_imports=False
)

_base_ = [
    '/workspace/mmsegmentation/configs/_base_/models/fcn_unet_s5-d16.py',
    '/workspace/mmsegmentation/configs/_base_/default_runtime.py',
    '/workspace/mmsegmentation/configs/_base_/schedules/schedule_20k.py'
]

data_root = '/workspace/GitHub/ratitas/mmsegmentation_project/datasets/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(512, 1024), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 1024), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='PackSegInputs')
]

dataset_type = 'CustomDataset'

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/train',
            seg_map_path='masks/train'
        ),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/val',
            seg_map_path='masks/val'
        ),
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

model = dict(
    data_preprocessor=dict(
        size=(512, 512),
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
    ),
    decode_head=dict(num_classes=4),
    auxiliary_head=dict(num_classes=4)
)

work_dir = './work_dirs/unet_custom'