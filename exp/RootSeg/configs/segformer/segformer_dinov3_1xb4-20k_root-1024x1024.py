# ======================== 1. 全局基础配置 ========================
# 数据集基础信息
dataset_type = 'RootSystemDataset'
num_classes = 2
data_root = 'J:/dataset/own/segdino'
train_img_path = 'train/image'
train_seg_map_path = 'train/label'
val_img_path = 'test/image'
val_seg_map_path = 'test/label'

# 训练基础参数
# DINOv3 使用 timm 预训练权重，无需单独指定 checkpoint
crop_size = (1024, 1024)
max_iters = 20000
val_interval = 1000
log_interval = 50

# 性能与硬件相关
batch_size = 2
num_workers = 2

# ======================== 2. 模型配置 (Model) ========================
norm_cfg = dict(type='BN', requires_grad=True)  # 单卡训练使用 BN

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

# DINOv3 vit_base_patch16 配置
# embed_dim=768, depth=12, patch_size=16
# out_indices=(2, 5, 8, 11) 输出4个尺度的特征，每个都是 768 channels
dinov3_embed_dim = 768

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='DINOv3MultiScale',
        model_name='vit_base_patch16_dinov3',
        pretrained=True,  # 使用 timm 预训练权重
        # out_indices 会自动设置为 (2, 5, 8, 11) 对于 12 层模型
        frozen_stages=11,  # 冻结所有 12 层 (0-11)，backbone 完全不训练
        freeze_patch_embed=True,  # 冻结 patch embedding
        drop_path_rate=0.0),  # 冻结时不需要 drop path
    decode_head=dict(
        type='SegformerHead',
        in_channels=[dinov3_embed_dim] * 4,  # [768, 768, 768, 768]
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # 训练与测试逻辑配置
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(768, 768)))

# ======================== 3. 数据流水线 (Pipeline) ========================
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2024, 3400),
        ratio_range=(0.5, 1.25),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(2024, 3400), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# TTA (Test Time Augmentation) 配置
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0.0, direction='horizontal'),
                dict(type='RandomFlip', prob=1.0, direction='horizontal')
            ],
            [dict(type='LoadAnnotations')],
            [dict(type='PackSegInputs')]
        ])
]

# ======================== 4. 数据加载器 (DataLoader) ========================
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path=train_img_path, 
            seg_map_path=train_seg_map_path),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path=val_img_path, 
            seg_map_path=val_seg_map_path),
        pipeline=test_pipeline))

test_dataloader = val_dataloader

# ======================== 5. 优化与调度 (Optimizer & Schedule) ========================
# backbone 已冻结，只训练 decode_head，可以使用较大的学习率
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01),  # 学习率提高
    paramwise_cfg=dict(
        custom_keys=dict(
            norm=dict(decay_mult=0.0))))

# 学习率调度器
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-06, by_epoch=False, begin=0, end=500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=500,
        end=max_iters,
        by_epoch=False)
]

# 训练循环配置
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=val_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 评估器
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# ======================== 6. 其他运行时配置 (Runtime) ========================
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', name='visualizer', vis_backends=vis_backends)

log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=log_interval, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', 
        by_epoch=False, 
        interval=val_interval, 
        max_keep_ckpts=5, 
        save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

# 断点保存与工作目录
work_dir = 'exp/root_seg/work_dirs/segformer_dinov3_1xb4-20k_root-1024x1024'
