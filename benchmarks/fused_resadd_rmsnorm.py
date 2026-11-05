"""
Benchmark Preset 4 (Stanford KernelBench Level-2): Fused Residual-Add + RMSNorm.
Eliminates 4 sequential HBM memory roundtrips into a single SRAM register pass.
"""

PRESET_RESADD_RMSNORM = {
    "operation_name": "fused_resadd_rmsnorm",
    "description": "LLaMA-3 Core Transformer Layer: Fused Residual Add + Layer Normalization (Level-2)",
    "pytorch_source": '''def fused_residual_rmsnorm_eager(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    # PyTorch Eager Bottleneck: 4 roundtrips to HBM memory
    # 1. Residual accumulation
    x_res = x + residual
    # 2. Mean square variance reduction across hidden dimension
    variance = x_res.pow(2).mean(-1, keepdim=True)
    # 3. Normalization and scale
    normed = x_res * torch.rsqrt(variance + eps) * weight
    # Returns both the new residual state AND the normalized output for the next layer
    return normed, x_res
''',
    "input_tensor_shapes": [[32, 4096]],
    "dtype": "float16",
    "fault_shape": [[32, 4097]]
}

