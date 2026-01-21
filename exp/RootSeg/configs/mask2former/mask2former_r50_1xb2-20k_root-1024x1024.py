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
pretrained = 'torchvision://resnet50'  # 预训练权重路径 (torchvision://, https://, 或本地路径)
crop_size = (1024, 1024)  # 训练时的裁剪尺寸 (height, width)
max_iters = 20000  # 最大训练迭代次数 (基于迭代的训练)
val_interval = 100  # 每隔多少次迭代进行一次验证
log_interval = 50  # 每隔多少次迭代打印一次日志

# 性能与硬件相关
batch_size = 2  # 每个 GPU 的 batch size, 8xb2 表示 8卡 x 2batch = 有效batch 16
num_workers = 2  # DataLoader 的工作进程数

# ======================== 2. 模型配置 (Model) ========================
# 数据预处理器配置, 继承自 mmseg.models.data_preprocessor.SegDataPreProcessor
data_preprocessor = dict(
    type='SegDataPreProcessor',  # 预处理器类型
    mean=[123.675, 116.28, 103.53],  # ImageNet RGB 均值, 用于图像归一化
    std=[58.395, 57.12, 57.375],  # ImageNet RGB 标准差
    bgr_to_rgb=True,  # OpenCV 读取为 BGR, 需转换为 RGB
    pad_val=0,  # 图像 padding 填充值
    seg_pad_val=255,  # 分割标注 padding 填充值 (255 通常表示 ignore)
    size=crop_size,  # 训练时将图像 pad 到此尺寸
    test_cfg=dict(size_divisor=32))  # 测试时确保尺寸能被 32 整除 (适配 backbone 下采样)

# 模型总体配置, 继承自 mmseg.models.segmentors.EncoderDecoder
model = dict(
    type='EncoderDecoder',  # 分割器类型: 编码器-解码器架构
    data_preprocessor=data_preprocessor,  # 绑定数据预处理器
    # -------------------- Backbone 配置 --------------------
    # ResNet-50 骨干网络, 继承自 mmpretrain/mmcls 的 ResNet
    backbone=dict(
        type='ResNet',  # 骨干网络类型
        depth=50,  # ResNet 深度: 18, 34, 50, 101, 152 可选
        deep_stem=False,  # 是否使用 deep stem (3 个 3x3 卷积代替 7x7)
        num_stages=4,  # ResNet 有 4 个 stage (layer1-4)
        out_indices=(0, 1, 2, 3),  # 输出哪些 stage 的特征 (0-indexed)
        frozen_stages=-1,  # 冻结到第几个 stage (-1 表示不冻结, 0 表示仅冻结 stem)
        norm_cfg=dict(type='SyncBN', requires_grad=False),  # 归一化层配置: SyncBN 用于多卡同步
        style='pytorch',  # 'pytorch' 或 'caffe' (主要影响 stride 位置)
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),  # 初始化配置: 加载预训练权重
    # -------------------- Decode Head 配置 --------------------
    # Mask2Former 解码头, 继承自 mmdet.models.dense_heads.Mask2FormerHead
    decode_head=dict(
        type='Mask2FormerHead',  # 解码头类型
        in_channels=[256, 512, 1024, 2048],  # 各 stage 输入通道数 (ResNet50 的输出通道)
        strides=[4, 8, 16, 32],  # 各 stage 相对于原图的下采样率
        feat_channels=256,  # Transformer 内部特征通道数
        out_channels=256,  # Pixel Decoder 输出通道数
        num_classes=num_classes,  # 类别数 (不含背景, Mask2Former 会自动加 1 个 no-object 类)
        num_queries=100,  # Query 数量, 每个 query 对应一个 mask 预测
        num_transformer_feat_level=3,  # Transformer Encoder 使用的特征层级数 (通常为 3)
        align_corners=False,  # F.interpolate 的对齐方式, 一般设为 False
        # Pixel Decoder 配置 (MSDeformAttn 多尺度可变形注意力)
        pixel_decoder=dict(
            type='mmdet.MSDeformAttnPixelDecoder',  # 像素解码器类型
            num_outs=3,  # 输出特征层数
            norm_cfg=dict(type='GN', num_groups=32),  # Group Normalization
            act_cfg=dict(type='ReLU'),  # 激活函数
            encoder=dict(  # Deformable Transformer Encoder 配置
                num_layers=6,  # Encoder 层数
                layer_cfg=dict(  # 每层的配置
                    self_attn_cfg=dict(  # Multi-Scale Deformable Self-Attention
                        embed_dims=256,  # 嵌入维度
                        num_heads=8,  # 注意力头数
                        num_levels=3,  # 特征层级数
                        num_points=4,  # 每个注意力头的采样点数
                        im2col_step=64,  # CUDA 实现的批处理步长
                        dropout=0.0,  # Dropout 率
                        batch_first=True),  # Batch 维度是否在第一位
                    ffn_cfg=dict(  # Feed-Forward Network 配置
                        embed_dims=256,  # 嵌入维度
                        feedforward_channels=1024,  # FFN 隐藏层通道数 (通常为 4x embed_dims)
                        num_fcs=2,  # FFN 中的全连接层数
                        ffn_drop=0.0,  # FFN Dropout 率
                        act_cfg=dict(type='ReLU', inplace=True)))),  # 激活函数
            positional_encoding=dict(num_feats=128, normalize=True)),  # 位置编码 (正弦编码)
        enforce_decoder_input_project=False,  # 是否强制使用输入投影
        # Transformer Decoder 配置
        transformer_decoder=dict(
            num_layers=9,  # Decoder 层数
            return_intermediate=True,  # 是否返回中间层结果 (用于辅助损失)
            layer_cfg=dict(  # 每层配置
                self_attn_cfg=dict(  # Self-Attention
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,  # Attention Dropout
                    proj_drop=0.0,  # Projection Dropout
                    batch_first=True),
                cross_attn_cfg=dict(  # Cross-Attention (Masked Attention)
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    batch_first=True),
                ffn_cfg=dict(  # FFN
                    embed_dims=256,
                    feedforward_channels=2048,  # Decoder FFN 通常为 8x embed_dims
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type='ReLU', inplace=True),
                    add_identity=True))),  # 是否添加残差连接
        positional_encoding=dict(num_feats=128, normalize=True),  # Query 位置编码
        # -------------------- 损失函数配置 --------------------
        loss_cls=dict(  # 分类损失 (CE Loss)
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,  # 使用 Softmax 而非 Sigmoid
            loss_weight=2.0,  # 损失权重
            reduction='mean',  # 归约方式
            class_weight=[1.0] * num_classes + [0.1]),  # 各类权重, no-object 类权重为 0.1
        loss_mask=dict(  # Mask 二分类损失 (BCE Loss)
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,  # 使用 Sigmoid
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(  # Dice Loss (区域重叠度)
            type='mmdet.DiceLoss',
            use_sigmoid=True,
            activate=True,  # 是否先 sigmoid
            reduction='mean',
            naive_dice=True,  # 使用简化版 Dice
            eps=1.0,  # 数值稳定性常数
            loss_weight=5.0),
        # -------------------- 训练配置 --------------------
        train_cfg=dict(
            num_points=12544,  # 点采样数 (用于计算 mask loss, 112*112=12544)
            oversample_ratio=3.0,  # 过采样比例
            importance_sample_ratio=0.75,  # 重要性采样比例
            assigner=dict(  # 匈牙利匹配器 (二分匹配)
                type='mmdet.HungarianAssigner',
                match_costs=[  # 匹配代价函数
                    dict(type='mmdet.ClassificationCost', weight=2.0),  # 分类代价
                    dict(type='mmdet.CrossEntropyLossCost', weight=5.0, use_sigmoid=True),  # BCE 代价
                    dict(type='mmdet.DiceCost', weight=5.0, pred_act=True, eps=1.0)  # Dice 代价
                ]),
            sampler=dict(type='mmdet.MaskPseudoSampler'))),  # 伪采样器 (Mask2Former 不需要真正采样)
    train_cfg=dict(),  # 分割器级别的训练配置 (这里为空, 实际配置在 decode_head 内)
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(768, 768)))  # 测试模式: 'whole' 整图推理, 'slide' 滑窗推理

# ======================== 3. 数据流水线 (Pipeline) ========================
# 训练数据增强流水线
train_pipeline = [
    dict(type='LoadImageFromFile'),  # 从文件加载图像
    dict(type='LoadAnnotations'),  # 加载分割标注
    dict(
        type='RandomChoiceResize',  # 随机选择一个尺度进行 resize
        scales=[int(x * 0.1 * 2048) for x in range(5, 15)],  # 短边 1024-3072 像素
        resize_type='ResizeShortestEdge',  # 按短边缩放
        max_size=4096),  # 长边最大值限制
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),  # 随机裁剪, 单类最大占比 75%
    dict(type='RandomFlip', prob=0.5),  # 随机水平翻转, 概率 50%
    # dict(type='RandomFlip', prob=0.5, direction='vertical'),  # 垂直翻转
    # dict(type='RandomRotate', prob=0.5, degree=30),  # 随机旋转（需要mmseg支持)
    dict(type='PhotoMetricDistortion'),  # 光度畸变 (亮度/对比度/饱和度/色相)
    dict(type='PackSegInputs')  # 打包为模型输入格式
]

# 测试/验证数据流水线
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(2024, 3400), keep_ratio=True),  # 缩放到固定尺寸
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations'),
#     dict(
#         type='RandomResize',
#         scale=(2024, 3400),
#         ratio_range=(0.5, 1.25),
#         keep_ratio=True),
#     dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PhotoMetricDistortion'),
#     dict(type='PackSegInputs')
# ]  # todo

# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='Resize', scale=(2024, 3400), keep_ratio=True),
#     dict(type='LoadAnnotations'),
#     dict(type='PackSegInputs')
# ]  # todo

# ======================== 4. 数据加载器 (DataLoader) ========================
# 训练数据加载器
train_dataloader = dict(
    batch_size=batch_size,  # 每个 GPU 的 batch size
    num_workers=num_workers,  # 加载数据的子进程数
    persistent_workers=True,  # 保持 worker 进程不销毁, 加速数据加载
    sampler=dict(type='InfiniteSampler', shuffle=True),  # 无限采样器 (迭代训练)
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path=train_img_path, seg_map_path=train_seg_map_path),
        pipeline=train_pipeline))

# 验证数据加载器
val_dataloader = dict(
    batch_size=1,  # 验证时 batch size 通常为 1
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),  # 默认采样器, 不打乱
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path=val_img_path, seg_map_path=val_seg_map_path),
        pipeline=test_pipeline))

test_dataloader = val_dataloader  # 测试使用与验证相同的加载器

# ======================== 5. 优化与调度 (Optimizer & Schedule) ========================
# 优化器配置
optim_wrapper = dict(
    type='OptimWrapper',  # 优化器包装器类型
    optimizer=dict(
        type='AdamW',  # AdamW 优化器 (比 Adam 更好的权重衰减)
        lr=0.0000125,  # 基础学习率
        betas=(0.9, 0.999),  # Adam 的动量参数
        weight_decay=0.05,  # 权重衰减 (L2 正则化)
        eps=1e-08),  # 数值稳定性常数
    clip_grad=dict(max_norm=0.01, norm_type=2),  # 梯度裁剪, 最大范数 0.01
    paramwise_cfg=dict(  # 参数级别的配置
        custom_keys={  # 自定义参数学习率/衰减
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),  # backbone 学习率乘 0.1
            'query_embed': dict(lr_mult=1.0, decay_mult=0.0),  # Query 嵌入不权重衰减
            'query_feat': dict(lr_mult=1.0, decay_mult=0.0),
            'level_embed': dict(lr_mult=1.0, decay_mult=0.0),
        },
        norm_decay_mult=0.0))  # 所有 Norm 层不进行权重衰减

# 学习率调度器
param_scheduler = [
    dict(
        type='PolyLR',  # 多项式学习率衰减
        eta_min=0,  # 最小学习率
        power=0.9,  # 衰减幂次
        begin=0,  # 开始迭代
        end=max_iters,  # 结束迭代
        by_epoch=False)  # 基于迭代而非 epoch
]

# 训练循环配置
train_cfg = dict(
    type='IterBasedTrainLoop',  # 基于迭代的训练循环 (而非 epoch)
    max_iters=max_iters,  # 最大迭代次数
    val_interval=val_interval)  # 验证间隔
val_cfg = dict(type='ValLoop')  # 验证循环类型
test_cfg = dict(type='TestLoop')  # 测试循环类型

# 评估器
val_evaluator = dict(
    type='IoUMetric',  # IoU 评估指标
    iou_metrics=['mIoU'])  # 计算 mIoU
test_evaluator = val_evaluator

# ======================== 6. 其他运行时配置 (Runtime) ========================
default_scope = 'mmseg'  # 默认模块注册域
env_cfg = dict(
    cudnn_benchmark=True,  # 启用 cuDNN benchmark 加速
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),  # 多进程启动方式
    dist_cfg=dict(backend='nccl'))  # 分布式训练后端

# 可视化配置
vis_backends = [
    dict(type='LocalVisBackend'),  # 本地可视化
    dict(type='TensorboardVisBackend')  # Tensorboard 可视化
]
visualizer = dict(
    type='SegLocalVisualizer',  # 分割可视化器
    name='visualizer',
    vis_backends=vis_backends)

log_processor = dict(by_epoch=False)  # 日志按迭代记录而非 epoch
log_level = 'INFO'  # 日志级别
load_from = None  # 加载检查点路径 (用于 finetune)
resume = False  # 是否从断点恢复训练

# 钩子函数配置
default_hooks = dict(
    timer=dict(type='IterTimerHook'),  # 计时钩子
    logger=dict(type='LoggerHook', interval=log_interval, log_metric_by_epoch=False),  # 日志钩子
    param_scheduler=dict(type='ParamSchedulerHook'),  # 学习率调度钩子
    checkpoint=dict(
        type='CheckpointHook',  # 检查点保存钩子
        by_epoch=False,  # 按迭代保存
        interval=val_interval,  # 保存间隔
        save_best='mIoU'),  # 保存最佳模型的指标
    sampler_seed=dict(type='DistSamplerSeedHook'),  # 分布式采样器种子钩子
    visualization=dict(type='SegVisualizationHook'))  # 分割可视化钩子

# 断点保存与工作目录
work_dir = 'exp/root_seg/work_dirs/mask2former_r50_1xb2-20k_root-1024x1024'  # 工作目录 (保存日志和检查点)
