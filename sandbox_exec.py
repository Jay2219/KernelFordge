"""
Child Subprocess Worker for Isolated CUDA Execution and Microsecond Benchmarking.
This script is executed inside an isolated subprocess to guarantee that CUDA
illegal memory accesses or segmentation faults do not poison the host container daemon.
"""

import sys
import json
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

        # Execute synthesized definitions in isolated process globals
        exec(code_str, globals(), local_scope)

        if "run_verification_and_benchmark" not in local_scope:
            sys.stderr.write("Verification harness missing 'run_verification_and_benchmark' entrypoint.\n")
            sys.exit(2)

        verify_fn = local_scope["run_verification_and_benchmark"]
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
