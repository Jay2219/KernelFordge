"""
Analytical Roofline and Hardware Arithmetic Intensity Calculators.
"""

from typing import List, Dict, Any, Tuple
from ..schemas.compiler import HardwareSpec


def calculate_op_characteristics(
    operation_name: str,
    shapes: List[List[int]],
    dtype: str = "float16",
    hardware: HardwareSpec = HardwareSpec()
) -> Dict[str, Any]:
    """
    Computes analytical FLOPs, HBM bytes transferred, operational intensity,
    and identifies memory vs compute bound status relative to the target GPU knee point.
    """
    bytes_per_elem = 2 if dtype in ("float16", "bfloat16") else 4
    
    # Primary input tensor dimensions
    primary_shape = shapes[0] if shapes else [32, 4096]
    total_elements = 1
    for dim in primary_shape:
        total_elements *= dim

    op_lower = operation_name.lower()
    
    if "rms" in op_lower or "norm" in op_lower:
        # RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight
        # FLOPs:
        # 1. x^2: total_elements
        # 2. mean reduction: total_elements
        # 3. + eps: rows
        # 4. rsqrt: rows
        # 5. x * rstd: total_elements
        # 6. * weight: total_elements
        # Total FLOPs ~ 4 * total_elements
        flops = 4 * total_elements
        
        # Bytes transferred in PyTorch eager (3 intermediate buffers written/read from HBM):
        # Read x, Write x^2, Read x^2, Write mean, Read mean, Write rsqrt, Read rsqrt, Read x, Write y
        # In Fused Triton:
        # Read x (once), Read weight (broadcast once), Write y (once)
        fused_bytes = (2 * total_elements + primary_shape[-1]) * bytes_per_elem
        eager_bytes = fused_bytes + (3 * total_elements * bytes_per_elem)
        op_type = "Reduction"
        fusion_rationale = (
            "Fusing sum-of-squares reduction and elementwise normalization keeps the accumulator "
            "entirely in on-chip SRAM registers, eliminating 3 intermediate HBM global roundtrips."
        )

    elif "swiglu" in op_lower or "silu" in op_lower:
        # SwiGLU: silu(gate) * up = gate * sigmoid(gate) * up
        # FLOPs:
        # 1. sigmoid(gate): ~4 FLOPs
        # 2. gate * sigmoid: 1 FLOP
        # 3. * up: 1 FLOP
        # Total ~ 6 * total_elements
        flops = 6 * total_elements
        # Bytes: Read gate, Read up, Write y
        fused_bytes = 3 * total_elements * bytes_per_elem
        eager_bytes = fused_bytes + (2 * total_elements * bytes_per_elem)
        op_type = "Elementwise"
        fusion_rationale = (
            "Fusing the non-linear SiLU activation and Hadamard product avoids allocating intermediate "
            "activation buffers, evaluating the gating equation in a single SRAM streaming pass."
        )

    elif "softmax" in op_lower:
        # Softmax: exp(x - max) / sum(exp(x - max))
        # Total FLOPs ~ 5 * total_elements
        flops = 5 * total_elements
        fused_bytes = 2 * total_elements * bytes_per_elem
        eager_bytes = fused_bytes + (3 * total_elements * bytes_per_elem)
        op_type = "Reduction"
        fusion_rationale = (
            "Online safe-softmax evaluates max tracking, exponential shift, and normalizer summation "
            "directly in thread-block shared memory without materializing un-normalized exponent buffers."
        )

    else:
        # Generic elementwise / pointwise fallback
        flops = 2 * total_elements
        fused_bytes = 2 * total_elements * bytes_per_elem
        eager_bytes = 4 * total_elements * bytes_per_elem
        op_type = "Fused"
        fusion_rationale = "Kernel fusion consolidates pointwise memory access into a single load-store kernel."

    # Arithmetic intensity: FLOPs / Byte
    operational_intensity = round(flops / max(fused_bytes, 1), 3)

    # Hardware Roofline Knee: Peak TFLOPS / Peak Bandwidth GB/s
    # e.g., A10G: 31,200 GFLOPS / 600 GB/s = 52.0 FLOPs/Byte
    roofline_knee = (hardware.peak_fp16_tflops * 1000.0) / hardware.peak_bandwidth_gbs
    is_memory_bound = operational_intensity < roofline_knee

    return {
        "operation_type": op_type,
        "total_elements": total_elements,
        "total_flops": flops,
        "fused_bytes_transferred": fused_bytes,
        "eager_bytes_transferred": eager_bytes,
        "operational_intensity": operational_intensity,
        "roofline_knee": round(roofline_knee, 2),
        "is_memory_bound": is_memory_bound,
        "fusion_rationale": fusion_rationale
    }


def compute_bandwidth_saturation(
    total_bytes: int,
    latency_us: float,
    peak_bandwidth_gbs: float
) -> Tuple[float, float]:
    """
    Calculates achieved memory bandwidth in GB/s and % saturation against peak hardware roofline.
    """
    if latency_us <= 0.0:
        return 0.0, 0.0
    
    latency_seconds = latency_us / 1e6
    achieved_gbs = (total_bytes / 1e9) / latency_seconds
    saturation_pct = min(100.0, (achieved_gbs / peak_bandwidth_gbs) * 100.0)
    
    return round(achieved_gbs, 2), round(saturation_pct, 1)
