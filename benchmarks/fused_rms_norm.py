"""
Benchmark Preset 1: Fused RMSNorm (LLaMA-3 Core Primitive).
"""

PRESET_RMS_NORM = {
    "operation_name": "fused_rms_norm",
    "description": "LLaMA-3 Root Mean Square Layer Normalization",
    "pytorch_source": '''def rms_norm_eager(x, weight, eps=1e-6):
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight
''',
    "input_tensor_shapes": [[32, 4096]],
    "dtype": "float16",
    "fault_shape": [[32, 4097]]
}
