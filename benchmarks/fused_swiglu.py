"""
Benchmark Preset 2: Fused SwiGLU (LLaMA-3 Feed-Forward Layer).
"""

PRESET_SWIGLU = {
    "operation_name": "fused_swiglu",
    "description": "LLaMA-3 Swish-Gated Linear Unit Activation",
    "pytorch_source": '''def swiglu_eager(gate, up):
    return torch.nn.functional.silu(gate) * up
''',
    "input_tensor_shapes": [[32, 11008]],
    "dtype": "float16",
    "fault_shape": [[32, 11009]]
}
