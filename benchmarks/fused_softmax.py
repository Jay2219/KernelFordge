"""
Benchmark Preset 3: Fused Online Softmax (FlashAttention Subroutine).
"""

PRESET_SOFTMAX = {
    "operation_name": "fused_softmax",
    "description": "Safe Softmax Online Reduction across Attention Heads",
    "pytorch_source": '''def softmax_eager(scores):
    return torch.softmax(scores, dim=-1)
''',
    "input_tensor_shapes": [[32, 4096]],
    "dtype": "float16",
    "fault_shape": [[32, 4097]]
}
