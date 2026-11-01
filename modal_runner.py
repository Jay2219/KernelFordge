"""
Modal Serverless GPU Sandbox and Subprocess Spawner.
Guarantees clean CUDA driver isolation, zero context poisoning, and instant warm pools.
"""

import os
import sys
import json
import subprocess
import tempfile
import modal

app = modal.App("kernelforge-sandbox")

# CUDA 12.2 environment with PyTorch 2.4 and OpenAI Triton 3.0
kernel_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "triton==3.0.0",
        "numpy>=1.26.0",
        "pydantic>=2.9.0"
    )
    .copy_local_file("sandbox_exec.py", "/root/sandbox_exec.py")
)


@app.function(
    image=kernel_image,
    gpu="A10G",
    timeout=45,
    min_containers=1,       # Eliminates 20s cold start for live video demo
    scaledown_window=300    # Keeps container warm for 5 minutes between queries
)
def execute_kernel_payload(bundle_json: str) -> str:
    """
    Receives synthesized kernel bundle, writes payload to tempfile, and spawns
    isolated child subprocess. Captures crash telemetry without poisoning host context.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        payload_path = os.path.join(tmpdir, "payload.json")
        result_path = os.path.join(tmpdir, "result.json")

        with open(payload_path, "w", encoding="utf-8") as f:
            f.write(bundle_json)

        env = {
            **os.environ,
            "CUDA_LAUNCH_BLOCKING": "1",
            "TORCH_USE_CUDA_DSA": "1",
            "PYTHONUNBUFFERED": "1"
        }

        try:
            # Spawn isolated process with 25s watchdog timeout
            proc = subprocess.run(
                [sys.executable, "/root/sandbox_exec.py", payload_path, result_path],
                capture_output=True,
                text=True,
                env=env,
                timeout=25
            )

            if proc.returncode == 0 and os.path.exists(result_path):
                with open(result_path, "r", encoding="utf-8") as rf:
                    benchmark_data = json.load(rf)
                return json.dumps({
                    "status": "SUCCESS",
                    "benchmark": benchmark_data,
                    "traceback": None
                })
            else:
                stderr_output = proc.stderr or proc.stdout or "Subprocess exited with unknown error"
                return json.dumps({
                    "status": "FAILED",
                    "benchmark": None,
                    "traceback": stderr_output
                })

        except subprocess.TimeoutExpired:
            return json.dumps({
                "status": "FAILED",
                "benchmark": None,
                "traceback": "Subprocess execution timed out after 25s (Deadlock or Infinite Loop detected)"
            })
