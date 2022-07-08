# dataset settings
dataset_type = 'CocoDataset'
data_root = '../../../data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    test=dict(
        type=dataset_type,
        #ann_file=data_root + 'annotations/instances_val2014_sample0.005.json',
        #ann_file=data_root + 'annotations/instances_val2014.json',
        #img_prefix=data_root + 'val2014/',
        #ann_file=data_root + 'annotations/instances_val2017_sample1.0.json',
        ann_file=data_root+'annotations/instances_val2017_sample0.25.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
