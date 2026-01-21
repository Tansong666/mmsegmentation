"""
测试 DINOv3_Adapter 是否正确注册为 mmseg backbone
"""
import sys
sys.path.insert(0, "exp/RootSeg")

import torch
from mmseg.registry import MODELS

# 导入自定义模型以触发注册
from mmseg_custom.models.backbones import DINOv3_Adapter, DINOv3VisionBackbone

def test_registration():
    """测试模型是否正确注册"""
    print("=" * 60)
    print("Testing DINOv3 Backbone Registration")
    print("=" * 60)
    
    # 检查是否已注册
    print("\n1. 检查注册状态:")
    print(f"   DINOv3VisionBackbone 已注册: {'DINOv3VisionBackbone' in MODELS.module_dict}")
    print(f"   DINOv3_Adapter 已注册: {'DINOv3_Adapter' in MODELS.module_dict}")
    
    # 测试 DINOv3VisionBackbone
    print("\n2. 测试 DINOv3VisionBackbone 创建:")
    backbone_cfg = dict(
        type='DINOv3VisionBackbone',
        size='base',  # 使用较小的模型测试
        patch_size=16,
        img_size=224,
        freeze_vit=True,
    )
    
    try:
        backbone = MODELS.build(backbone_cfg)
        print(f"   ✓ DINOv3VisionBackbone 创建成功!")
        print(f"   - embed_dim: {backbone.embed_dim}")
        print(f"   - patch_size: {backbone.patch_size}")
        print(f"   - n_blocks: {backbone.n_blocks}")
        print(f"   - out_indices: {backbone.out_indices}")
    except Exception as e:
        print(f"   ✗ DINOv3VisionBackbone 创建失败: {e}")
        return False
    
    # 测试 DINOv3_Adapter
    print("\n3. 测试 DINOv3_Adapter 创建:")
    adapter_cfg = dict(
        type='DINOv3_Adapter',
        backbone_cfg=dict(
            type='DINOv3VisionBackbone',
            size='base',
            patch_size=16,
            img_size=512,
            freeze_vit=True,
        ),
        interaction_indexes=[2, 5, 8, 11],  # 对应 base 模型的 12 层
        pretrain_size=512,
        freeze_backbone=True,
    )
    
    try:
        adapter = MODELS.build(adapter_cfg)
        print(f"   ✓ DINOv3_Adapter 创建成功!")
        print(f"   - backbone.embed_dim: {adapter.backbone.embed_dim}")
        print(f"   - patch_size: {adapter.patch_size}")
        print(f"   - interaction_indexes: {adapter.interaction_indexes}")
    except Exception as e:
        print(f"   ✗ DINOv3_Adapter 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试前向传播
    print("\n4. 测试前向传播:")
    try:
        adapter.eval()
        with torch.no_grad():
            # 使用 CPU 避免 CUDA 依赖问题
            x = torch.randn(1, 3, 512, 512)
            # 注意：实际测试可能需要 CUDA 因为使用了 SyncBatchNorm
            print(f"   输入形状: {x.shape}")
            # output = adapter(x)
            # print(f"   输出 keys: {list(output.keys())}")
            # for k, v in output.items():
            #     print(f"   - {k}: {v.shape}")
            print("   (跳过实际前向传播测试，需要 CUDA 环境)")
    except Exception as e:
        print(f"   ✗ 前向传播测试失败: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    # 切换到项目根目录
    import os
    os.chdir("e:/Project/mmsegmentation")
    
    test_registration()
