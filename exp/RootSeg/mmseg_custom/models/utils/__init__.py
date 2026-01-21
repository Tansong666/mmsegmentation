# Multi-Scale Deformable Attention module
# Copied from dinov3 to avoid import chain issues

from .ms_deform_attn import MSDeformAttn, MSDeformAttnFunction, ms_deform_attn_core_pytorch

__all__ = ['MSDeformAttn', 'MSDeformAttnFunction', 'ms_deform_attn_core_pytorch']
