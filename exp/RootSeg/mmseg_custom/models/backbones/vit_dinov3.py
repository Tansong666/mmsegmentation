# Copyright (c) Meta Platforms, Inc. and affiliates.
# Wrapper for DINOv3 Vision Transformer backbone compatible with mmsegmentation >= 1.0

import copy
import sys
import warnings
from typing import Optional, Literal

import torch
from mmengine.model import BaseModule
from mmseg.registry import MODELS

# Add DINOv3 to path
sys.path.insert(0, "exp/RootSeg")
from dinov3.models.vision_transformer import DinoVisionTransformer


@MODELS.register_module()
class DINOv3VisionBackbone(BaseModule):
    """MMSegmentation compatible DINOv3 Vision Transformer backbone.
    
    This wrapper makes DINOv3's DinoVisionTransformer compatible with MMSeg's
    configuration system and initialization mechanism.
    
    Args:
        size (str): Size of ViT backbone. Options: 'small', 'base', 'large', 
            'so400m', 'huge2', 'giant2', '7b'. Default: 'giant2'.
        patch_size (int): Patch size. Default: 16.
        img_size (int): Input image size. Default: 224.
        freeze_vit (bool): Whether to freeze the backbone. Default: False.
        pretrained (str, optional): Deprecated. Use init_cfg instead.
        init_cfg (dict, optional): Initialization config dict.
        **kwargs: Additional arguments passed to DinoVisionTransformer.
    """
    
    # Model configurations for different sizes
    MODEL_CONFIGS = {
        'small': dict(embed_dim=384, depth=12, num_heads=6, ffn_ratio=4.0),
        'base': dict(embed_dim=768, depth=12, num_heads=12, ffn_ratio=4.0),
        'large': dict(embed_dim=1024, depth=24, num_heads=16, ffn_ratio=4.0),
        'so400m': dict(embed_dim=1152, depth=27, num_heads=18, ffn_ratio=3.777777778),
        'huge2': dict(embed_dim=1280, depth=32, num_heads=20, ffn_ratio=4.0),
        'giant2': dict(embed_dim=1536, depth=40, num_heads=24, ffn_ratio=4.0),
        '7b': dict(embed_dim=4096, depth=40, num_heads=32, ffn_ratio=3.0),
    }
    
    # Default output indices for different model depths
    OUT_INDICES = {
        12: [2, 5, 8, 11],       # small, base
        24: [4, 11, 17, 23],     # large
        27: [5, 12, 19, 26],     # so400m
        32: [7, 15, 23, 31],     # huge2
        40: [9, 19, 29, 39],     # giant2, 7b
    }
    
    def __init__(
        self,
        size: str = 'giant2',
        patch_size: int = 16,
        img_size: int = 224,
        freeze_vit: bool = False,
        pretrained: Optional[str] = None,
        init_cfg: Optional[dict] = None,
        out_indices: Optional[list] = None,
        # RoPE parameters
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: Optional[float] = None,
        pos_embed_rope_max_period: Optional[float] = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_dtype: str = "bf16",
        # FFN and Norm
        ffn_layer: str = "mlp",
        norm_layer: str = "layernorm",
        # Other
        n_storage_tokens: int = 0,
        **kwargs
    ):
        # Handle deprecated pretrained argument
        assert not (init_cfg and pretrained), \
            f'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        
        super().__init__(init_cfg=init_cfg)
        
        # Get model config based on size
        if size not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model size: {size}. "
                           f"Choose from: {list(self.MODEL_CONFIGS.keys())}")
        
        model_cfg = self.MODEL_CONFIGS[size].copy()
        model_cfg.update(kwargs)  # Allow overriding
        
        # Create the DINOv3 ViT
        self.vit = DinoVisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            pos_embed_rope_base=pos_embed_rope_base,
            pos_embed_rope_min_period=pos_embed_rope_min_period,
            pos_embed_rope_max_period=pos_embed_rope_max_period,
            pos_embed_rope_normalize_coords=pos_embed_rope_normalize_coords,
            pos_embed_rope_dtype=pos_embed_rope_dtype,
            ffn_layer=ffn_layer,
            norm_layer=norm_layer,
            n_storage_tokens=n_storage_tokens,
            **model_cfg
        )
        
        # Initialize weights
        self.vit.init_weights()
        
        # Store attributes for external access
        self.embed_dim = model_cfg['embed_dim']
        self.patch_size = patch_size
        self.n_blocks = model_cfg['depth']
        
        # Set output indices
        if out_indices is not None:
            self.out_indices = out_indices
        else:
            depth = model_cfg['depth']
            self.out_indices = self.OUT_INDICES.get(depth, [depth - 1])
        
        self._is_init = False
        self.init_cfg = copy.deepcopy(init_cfg)
        
        # Freeze if specified
        if freeze_vit:
            for param in self.parameters():
                param.requires_grad = False
    
    def get_intermediate_layers(self, x, n=None, reshape=False, return_class_token=False, **kwargs):
        """Get intermediate layer outputs.
        
        This method delegates to the underlying ViT's get_intermediate_layers.
        """
        if n is None:
            n = self.out_indices
        return self.vit.get_intermediate_layers(
            x, n=n, reshape=reshape, return_class_token=return_class_token, **kwargs
        )
    
    def forward(self, x, **kwargs):
        """Forward pass returning intermediate features."""
        return self.get_intermediate_layers(
            x, n=self.out_indices, reshape=True, return_class_token=False, **kwargs
        )
