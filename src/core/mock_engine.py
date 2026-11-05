"""
Local Mock Execution Simulator for Zero-GPU Windows Development and UI Testing.
"""

import time
from typing import Dict, Any, List
from ..schemas.compiler import BenchmarkResult, HardwareSpec
from .roofline import compute_bandwidth_saturation


class MockExecutionEngine:
    """
    Simulates remote Modal GPU execution deterministically without requiring
    local CUDA hardware or Triton compilation.
    """

    def __init__(self):
        self.iteration_history: Dict[str, int] = {}

    def execute_payload(
        self,
        session_id: str,
        operation_name: str,
        shapes: List[List[int]],
        dtype: str = "float16",
        hardware: HardwareSpec = HardwareSpec(),
        trigger_fault_demo: bool = False
    ) -> Dict[str, Any]:
        """
        Simulates kernel execution, parity check, and tri-comparison benchmarking.
        """
        current_iter = self.iteration_history.get(session_id, 1)

        # Simulate GPU runtime delay (0.6s)
        time.sleep(0.6)

        # If fault demo is requested and it's the first iteration, simulate an illegal memory access
        if trigger_fault_demo and current_iter == 1:
            self.iteration_history[session_id] = 2
            return {
                "status": "FAILED",
                "error_category": "CUDARuntimeError",
                "benchmark": None,
                "traceback": (
                    "Traceback (most recent call last):\n"
                    '  File "/root/sandbox_exec.py", line 28, in main\n'
                    "    results = verify_fn()\n"
                    '  File "<string>", line 42, in run_verification_and_benchmark\n'
                    "    triton_out = triton_fn(x, w)\n"
                    '  File "<string>", line 19, in triton_rms_norm\n'
                    "    _rms_norm_kernel[grid](x, y, w, ...)\n"
                    "RuntimeError: CUDA error: an illegal memory access was encountered\n"
                    "CUDA kernel failed: Device-side assert triggered at offset 4096.\n"
                    "Reason: Thread tried to write beyond buffer boundary without 'mask = cols < N'."
                ),
                "max_abs_err": None,
                "max_rel_err": None,
                "mismatch_coord": None
            }

        # Otherwise, return realistic Tri-Comparison Benchmark telemetry
        op_lower = operation_name.lower()
        primary_shape = shapes[0] if shapes else [32, 4096]
        M = primary_shape[0]
        N = primary_shape[1] if len(primary_shape) > 1 else primary_shape[0]
        total_elements = M * N
        bytes_per_elem = 2 if dtype in ("float16", "bfloat16") else 4
        total_bytes = 2 * total_elements * bytes_per_elem  # Read X + Write Y

        peak_bw = hardware.peak_bandwidth_gbs if hardware and hardware.peak_bandwidth_gbs else 300.0

        if "resadd" in op_lower or "residual" in op_lower:
            saturation_pct = 91.4
            achieved_gbs = round(peak_bw * (saturation_pct / 100.0), 1)
            triton_us = round((total_bytes / (achieved_gbs * 1e9)) * 1e6 * 2.0, 1)
            compile_us = round(triton_us * 1.42, 1)
            eager_us = round(triton_us * 3.78, 1)
        elif "rms" in op_lower or "norm" in op_lower:
            saturation_pct = 86.8
            achieved_gbs = round(peak_bw * (saturation_pct / 100.0), 1)
            triton_us = round((total_bytes / (achieved_gbs * 1e9)) * 1e6 * 2.1, 1)
            compile_us = round(triton_us * 1.25, 1)
            eager_us = round(triton_us * 2.85, 1)
        elif "swiglu" in op_lower or "silu" in op_lower:
            saturation_pct = 84.5
            achieved_gbs = round(peak_bw * (saturation_pct / 100.0), 1)
            triton_us = round((total_bytes / (achieved_gbs * 1e9)) * 1e6 * 2.3, 1)
            compile_us = round(triton_us * 1.20, 1)
            eager_us = round(triton_us * 2.45, 1)
        else:
            saturation_pct = 82.0
            achieved_gbs = round(peak_bw * (saturation_pct / 100.0), 1)
            triton_us = round((total_bytes / (achieved_gbs * 1e9)) * 1e6 * 2.5, 1)
            compile_us = round(triton_us * 1.22, 1)
            eager_us = round(triton_us * 2.40, 1)

        speedup_eager = round(eager_us / triton_us, 2)
        speedup_compile = round(compile_us / triton_us, 2)

        benchmark = BenchmarkResult(
            pytorch_eager_us=eager_us,
            pytorch_compile_us=compile_us,
            triton_kernel_us=triton_us,
            speedup_vs_eager=speedup_eager,
            speedup_vs_compile=speedup_compile,
            achieved_bandwidth_gbs=achieved_gbs,
            bandwidth_saturation_pct=saturation_pct,
            numerical_parity_verified=True,
            max_absolute_error=4.76e-5,
            max_relative_error=1.12e-4
        )

        return {
            "status": "SUCCESS",
            "benchmark": benchmark.model_dump(),
            "traceback": None
        }


mock_engine = MockExecutionEngine()

