# model_config_base.py
_base_ = [
    '/workspace/mmsegmentation/configs/_base_/models/fcn_unet_s5-d16.py',
    '/workspace/mmsegmentation/configs/_base_/default_runtime.py',
    '/workspace/mmsegmentation/configs/_base_/schedules/schedule_20k.py'
]

# Dataset settings
dataset_type = 'BaseSegDataset'
data_root = '/workspace/GitHub/ratitas/mmsegmentation_project/datasets/'

# Metainfo
metainfo = {
    'classes': ('0', '1', '2', '3'),
    'palette': [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]]
}

# Pipelines
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=crop_size, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size, keep_ratio=True),
    dict(type='PackSegInputs')
]

# Data loaders
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/train',
            seg_map_path='masks/train'),
        metainfo=metainfo,
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/valid', 
            seg_map_path='masks/valid'),
        metainfo=metainfo,
        pipeline=test_pipeline))

test_dataloader = val_dataloader

# Evaluator
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# Model
model = dict(
    decode_head=dict(num_classes=4),
    auxiliary_head=dict(num_classes=4))

# Work directory
work_dir = './work_dirs/unet_base'