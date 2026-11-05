"""
Child Subprocess Worker for Isolated CUDA Execution and Microsecond Benchmarking.
This script is executed inside an isolated subprocess to guarantee that CUDA
illegal memory accesses or segmentation faults do not poison the host container daemon.
"""

import os
import sys
import json
import linecache
import tempfile
import traceback


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("Usage: python sandbox_exec.py <payload_path> <result_path>\n")
        sys.exit(1)

    payload_path = sys.argv[1]
    result_path = sys.argv[2]

    try:
        with open(payload_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        code_str = payload["combined_executable_code"]
        local_scope = {}

        # Write to a real .py file so inspect.findsource() can locate source lines for @triton.jit
        kernel_file = os.path.join(tempfile.gettempdir(), "kernel_runner.py")
        with open(kernel_file, "w", encoding="utf-8") as kf:
            kf.write(code_str)

        # Register in linecache to guarantee inspect.getsourcelines(fn) succeeds
        linecache.cache[kernel_file] = (
            len(code_str),
            None,
            [line + "\n" for line in code_str.splitlines()],
            kernel_file
        )

        # Compile and execute in a unified module namespace
        # Pre-import torch and triton so type hints (e.g. x: torch.Tensor) resolve instantly
        import torch
        import triton
        import triton.language as tl
        import math

        execution_scope = {
            "__file__": kernel_file,
            "__name__": "__main__",
            "torch": torch,
            "triton": triton,
            "tl": tl,
            "math": math,
        }
        compiled_code = compile(code_str, kernel_file, "exec")
        exec(compiled_code, execution_scope, execution_scope)

        # Auto-bind mutual wrapper aliases so Synthesizer and Auditor naming conventions never mismatch
        triton_named = [
            v for k, v in execution_scope.items() 
            if callable(v) and "triton" in k.lower() and not k.startswith("_")
        ]
        if triton_named:
            primary_triton = triton_named[0]
        else:
            eager_names = {
                "rms_norm", "swiglu", "softmax", "fused_residual_rmsnorm", 
                "resadd_rmsnorm", "eager_fn", "pytorch_fn",
                "rms_norm_eager", "swiglu_eager", "softmax_eager", "fused_residual_rmsnorm_eager"
            }
            other_candidates = [
                v for k, v in execution_scope.items()
                if callable(v)
                and not k.startswith("_")
                and k != "run_verification_and_benchmark"
                and not k.endswith("_eager")
                and k.lower() not in eager_names
            ]
            primary_triton = other_candidates[0] if other_candidates else None

        if primary_triton:
            universal_aliases = [
                "triton_rms_norm", "triton_fused_rms_norm", "rms_norm_triton", "fused_rms_norm_triton",
                "triton_swiglu", "triton_fused_swiglu", "swiglu_triton", "fused_swiglu_triton",
                "triton_softmax", "triton_fused_softmax", "softmax_triton", "fused_softmax_triton",
                "triton_fused_residual_rmsnorm", "triton_resadd_rmsnorm", "fused_residual_rmsnorm_triton",
                "triton_kernel", "kernel_triton"
            ]
            for alias in universal_aliases:
                execution_scope[alias] = primary_triton

        verify_fn = execution_scope.get("run_verification_and_benchmark")
        if not verify_fn:
            sys.stderr.write("Verification harness missing 'run_verification_and_benchmark' entrypoint.\n")
            sys.exit(2)

        benchmark_results = verify_fn()

        with open(result_path, "w", encoding="utf-8") as rf:
            json.dump(benchmark_results, rf)

        sys.exit(0)

    except Exception as e:
        error_trace = traceback.format_exc()
        sys.stderr.write(error_trace)
        sys.exit(3)


if __name__ == "__main__":
    main()

