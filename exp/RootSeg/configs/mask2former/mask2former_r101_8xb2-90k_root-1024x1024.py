# ======================== 1. 全局基础配置 ========================
# 数据集基础信息
dataset_type = 'RootSystemDataset'  # 数据集类型, 对应 mmseg.datasets.CityscapesDataset
num_classes = 2  # 语义分割类别数 (不含ignore)
data_root = 'J:/dataset/own/segdino'  # 数据集根目录
train_img_path = 'train/image'  # 训练图像相对路径
train_seg_map_path = 'train/label'  # 训练标注相对路径
val_img_path = 'test/image'  # 验证图像相对路径
val_seg_map_path = 'test/label'  # 验证标注相对路径

# 训练基础参数
pretrained = 'torchvision://resnet101'
crop_size = (1024, 1024)
max_iters = 90000
val_interval = 5000
log_interval = 50

# 性能与硬件相关
batch_size = 2
num_workers = 2

# ======================== 2. 模型配置 (Model) ========================
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    test_cfg=dict(size_divisor=32))

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='ResNet',
        depth=101,
        deep_stem=False,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='SyncBN', requires_grad=False),
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    decode_head=dict(
        type='Mask2FormerHead',
        in_channels=[256, 512, 1024, 2048],
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_classes=num_classes,
        num_queries=100,
        num_transformer_feat_level=3,
        align_corners=False,
        pixel_decoder=dict(
            type='mmdet.MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                num_layers=6,
                layer_cfg=dict(
                    self_attn_cfg=dict(
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=True),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)))),
            positional_encoding=dict(num_feats=128, normalize=True)),
        enforce_decoder_input_project=False,
        transformer_decoder=dict(
            num_layers=9,
            return_intermediate=True,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    batch_first=True),
                cross_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type='ReLU', inplace=True),
                    add_identity=True))),
        positional_encoding=dict(num_feats=128, normalize=True),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),  # todo：各类权重, no-object 类权重为 0.1
        loss_mask=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='mmdet.DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='mmdet.HungarianAssigner',
                match_costs=[
                    dict(type='mmdet.ClassificationCost', weight=2.0),
                    dict(type='mmdet.CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                    dict(type='mmdet.DiceCost', weight=5.0, pred_act=True, eps=1.0)
                ]),
            sampler=dict(type='mmdet.MaskPseudoSampler'))),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(768, 768)))  # todo：测试模式'whole'整图推理, 'slide'滑窗推理

# ======================== 3. 数据流水线 (Pipeline) ========================
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomChoiceResize',
        scales=[int(x * 0.1 * 1024) for x in range(5, 21)],  # todo：随机选择缩放因子
        resize_type='ResizeShortestEdge',
        max_size=4096),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(2048, 1024), keep_ratio=True),  # todo：测试时是否需要缩放到固定尺寸
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
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
        data_prefix=dict(img_path=train_img_path, seg_map_path=train_seg_map_path),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path=val_img_path, seg_map_path=val_seg_map_path),
        pipeline=test_pipeline))

test_dataloader = val_dataloader

# ======================== 5. 优化与调度 (Optimizer & Schedule) ========================
# 优化器配置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05, eps=1e-08),  # todo：lr调整
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': dict(lr_mult=1.0, decay_mult=0.0),
            'query_feat': dict(lr_mult=1.0, decay_mult=0.0),
            'level_embed': dict(lr_mult=1.0, decay_mult=0.0),
        },
        norm_decay_mult=0.0))

# 学习率调度器
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=max_iters,
        by_epoch=False)
]

# 训练循环配置
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=val_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 评估器
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])  # todo：评估指标
test_evaluator = val_evaluator

# ======================== 6. 其他运行时配置 (Runtime) ========================
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]  # todo：tensorboard可视化
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
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=val_interval, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

# 断点保存与工作目录
work_dir = 'exp/root_seg/work_dirs/mask2former_r101_8xb2-90k_root-1024x1024'  # todo：工作目录
