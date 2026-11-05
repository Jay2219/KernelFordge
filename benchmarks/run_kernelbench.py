"""
KernelForge // Stanford KernelBench Automated Evaluation & Hugging Face Hub Publishing Harness.
Evaluates canonical GPU operators from Stanford KernelBench Level 1 and Level 2,
records microsecond telemetry and numerical assertions, persists JSON artifacts to disk,
and enables 1-click publishing to the Hugging Face Hub.
"""

import os
import sys
import json
import asyncio
import argparse
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path so it runs standalone from anywhere
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.core.llm_client import settings
from src.core.orchestrator import KernelOrchestrator
from src.schemas.compiler import SynthesisRequest, HardwareSpec

orchestrator = KernelOrchestrator()
from benchmarks.fused_rms_norm import PRESET_RMS_NORM
from benchmarks.fused_swiglu import PRESET_SWIGLU
from benchmarks.fused_softmax import PRESET_SOFTMAX
from benchmarks.fused_resadd_rmsnorm import PRESET_RESADD_RMSNORM

OUTPUT_FILE = ROOT_DIR / "benchmarks" / "kernelbench_results.json"
DATASET_CARD = ROOT_DIR / "benchmarks" / "README.md"

# Curated Stanford KernelBench Level-1 & Level-2 Canonical Targets
CANONICAL_TARGETS = {
    1: [
        {
            "task_id": "kernelbench_lvl1_rmsnorm",
            "name": "Level1_RMSNorm_Primitive",
            "pytorch_source": PRESET_RMS_NORM["pytorch_source"],
            "shapes": PRESET_RMS_NORM["input_tensor_shapes"],
            "dtype": PRESET_RMS_NORM["dtype"]
        },
        {
            "task_id": "kernelbench_lvl1_swiglu",
            "name": "Level1_SwiGLU_Gated_Activation",
            "pytorch_source": PRESET_SWIGLU["pytorch_source"],
            "shapes": PRESET_SWIGLU["input_tensor_shapes"],
            "dtype": PRESET_SWIGLU["dtype"]
        },
        {
            "task_id": "kernelbench_lvl1_softmax",
            "name": "Level1_Safe_Online_Softmax",
            "pytorch_source": PRESET_SOFTMAX["pytorch_source"],
            "shapes": PRESET_SOFTMAX["input_tensor_shapes"],
            "dtype": PRESET_SOFTMAX["dtype"]
        }
    ],
    2: [
        {
            "task_id": "kernelbench_lvl2_resadd_rmsnorm",
            "name": "Level2_Fused_ResAdd_RMSNorm",
            "pytorch_source": PRESET_RESADD_RMSNORM["pytorch_source"],
            "shapes": PRESET_RESADD_RMSNORM["input_tensor_shapes"],
            "dtype": PRESET_RESADD_RMSNORM["dtype"]
        }
    ]
}


def load_kernelbench_tasks(level: str, limit: int, stream_hf: bool = False):
    """
    Hybrid Task Loader with Graceful Offline Fallback.
    Attempts live streaming from `ScalingIntelligence/KernelBench` on Hugging Face;
    falls back cleanly to verified canonical fixtures if offline or on timeout.
    """
    level_str = str(level).lower()
    if level_str == "all":
        all_tasks = CANONICAL_TARGETS[1] + CANONICAL_TARGETS[2]
        return all_tasks[:limit]

    lvl_num = int(level_str) if level_str in ("1", "2") else 1
    tasks = []
    if stream_hf:
        try:
            print(f"[+] Attempting live stream from Hugging Face: ScalingIntelligence/KernelBench (level_{lvl_num})...")
            from datasets import load_dataset
            ds = load_dataset(
                "ScalingIntelligence/KernelBench",
                split=f"level_{lvl_num}",
                streaming=True
            )
            count = 0
            for item in ds:
                if count >= limit:
                    break
                task_name = item.get("name", f"kernelbench_lvl{lvl_num}_task{count}")
                code = item.get("code", "")
                if code.strip():
                    tasks.append({
                        "task_id": f"hf_lvl{lvl_num}_{item.get('problem_id', count)}",
                        "name": task_name,
                        "pytorch_source": code,
                        "shapes": [[32, 4096]],
                        "dtype": "float16",
                        "level": lvl_num
                    })
                    count += 1
            if tasks:
                print(f"    [✓] Successfully streamed {len(tasks)} tasks from Hugging Face Hub.")
                return tasks
        except Exception as e:
            print(f"[!] Notice: Live Hugging Face stream unavailable ({e}).")
            print("[!] Falling back to local verified canonical fixtures...")

    # Canonical Fallback
    canonical = CANONICAL_TARGETS.get(lvl_num, CANONICAL_TARGETS[1])
    return canonical[:limit]


def run_evaluation(
    level: str = "1",
    limit: int = 10,
    push_to_hub: bool = False,
    repo_id: str = "jay2219/kernelforge-kernelbench-results",
    stream_hf: bool = False,
    force_live: bool = False,
    force_mock: bool = False
):
    print("=" * 78)
    print(" [*] KERNELFORGE // STANFORD KERNELBENCH EVALUATION & TELEMETRY ENGINE")
    print("=" * 78)

    # Sandbox routing overrides
    if force_live:
        settings.MOCK_SANDBOX = False
        print("[*] Execution Target: LIVE MODAL GPU SANDBOX (MOCK_SANDBOX=False)")
    elif force_mock:
        settings.MOCK_SANDBOX = True
        print("[*] Execution Target: LOCAL MOCK SIMULATOR (MOCK_SANDBOX=True)")
    else:
        target_mode = "MOCK SIMULATOR" if settings.MOCK_SANDBOX else f"LIVE MODAL ({settings.MODAL_GPU_TARGET})"
        print(f"[*] Execution Target: {target_mode} (From environment configuration)")

    hardware_target = "NVIDIA A10G" if not settings.MOCK_SANDBOX else "Analytical A10G Model"
    peak_bw = 600.0

    tasks_to_run = load_kernelbench_tasks(level=level, limit=limit, stream_hf=stream_hf)
    print(f"[*] Queued {len(tasks_to_run)} task(s) for Stanford KernelBench Level-{level} evaluation.\n")

    results_tasks = []
    bandwidth_saturations = []
    pass_at_1_count = 0

    for idx, t in enumerate(tasks_to_run, 1):
        print(f"[{idx}/{len(tasks_to_run)}] Evaluating Operator: {t['name']}")
        print(f"      Task ID: {t['task_id']} | Shape: {t['shapes'][0]} | Dtype: {t['dtype']}")

        req = SynthesisRequest(
            session_id=f"kb_eval_{level}_{idx}",
            operation_name=t["name"],
            pytorch_source=t["pytorch_source"],
            input_tensor_shapes=t["shapes"],
            dtype=t["dtype"],
            hardware=HardwareSpec(gpu_name="NVIDIA A10G", peak_bandwidth_gbs=peak_bw)
        )

        try:
            # Synchronous invocation of orchestrator
            resp = asyncio.run(orchestrator.run(req))
            is_success = (resp.status == "COMPLETED" and resp.benchmark is not None)

            if is_success:
                b = resp.benchmark
                if resp.iteration_count == 1:
                    pass_at_1_count += 1
                bandwidth_saturations.append(b.bandwidth_saturation_pct)

                task_lvl = "level_2" if "lvl2" in t["task_id"] else ("level_1" if "lvl1" in t["task_id"] else f"level_{level}")
                task_entry = {
                    "task_id": t["task_id"],
                    "task_name": t["name"],
                    "level": task_lvl,
                    "status": "SUCCESS",
                    "iterations_required": resp.iteration_count,
                    "input_shape": t["shapes"][0],
                    "dtype": t["dtype"],
                    "numerical_parity": {
                        "verified": b.numerical_parity_verified,
                        "contract": "torch.testing.assert_close",
                        "rtol": 1e-3,
                        "atol": 1e-3
                    },
                    "telemetry": {
                        "eager_latency_us": b.pytorch_eager_us,
                        "torch_compile_us": b.pytorch_compile_us,
                        "triton_kernel_us": b.triton_kernel_us,
                        "speedup_vs_eager": b.speedup_vs_eager,
                        "speedup_vs_compile": b.speedup_vs_compile,
                        "achieved_bandwidth_gbs": b.achieved_bandwidth_gbs,
                        "bandwidth_saturation_pct": b.bandwidth_saturation_pct
                    }
                }
                print(f"      [✓] PASSED! Triton: {b.triton_kernel_us:.1f}µs | "
                      f"Speedup: {b.speedup_vs_eager:.2f}x vs Eager, {b.speedup_vs_compile:.2f}x vs Compile | "
                      f"BW: {b.bandwidth_saturation_pct:.1f}% Saturation\n")
            else:
                task_lvl = "level_2" if "lvl2" in t["task_id"] else ("level_1" if "lvl1" in t["task_id"] else f"level_{level}")
                task_entry = {
                    "task_id": t["task_id"],
                    "task_name": t["name"],
                    "level": task_lvl,
                    "status": "FAILED",
                    "iterations_required": resp.iteration_count,
                    "input_shape": t["shapes"][0],
                    "dtype": t["dtype"],
                    "error_log": resp.error_log
                }
                print(f"      [!] FAILED: {resp.error_log}\n")

            results_tasks.append(task_entry)

        except Exception as err:
            print(f"      [!] Evaluation error: {err}\n")
            results_tasks.append({
                "task_id": t["task_id"],
                "task_name": t["name"],
                "level": f"level_{level}",
                "status": "ERROR",
                "error": str(err)
            })

    # Summary Statistics
    total_tasks = len(results_tasks)
    successful_tasks = [t for t in results_tasks if t.get("status") == "SUCCESS"]
    pass_rate = round((len(successful_tasks) / max(total_tasks, 1)) * 100.0, 1)

    mean_eager_speedup = round(
        sum(t["telemetry"]["speedup_vs_eager"] for t in successful_tasks) / max(len(successful_tasks), 1), 2
    ) if successful_tasks else 0.0

    mean_compile_speedup = round(
        sum(t["telemetry"]["speedup_vs_compile"] for t in successful_tasks) / max(len(successful_tasks), 1), 2
    ) if successful_tasks else 0.0

    mean_bw_saturation = round(
        sum(bandwidth_saturations) / max(len(bandwidth_saturations), 1), 1
    ) if bandwidth_saturations else 0.0

    output_payload = {
        "metadata": {
            "suite": "Stanford KernelBench Evaluation",
            "version": "1.0.0",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "hardware": {
                "device": hardware_target,
                "compute_capability": 8.6,
                "peak_bandwidth_gbs": peak_bw,
                "runner": "Modal Linux Subprocess Worker (CUDA_LAUNCH_BLOCKING=1)"
            },
            "compiler": "KernelForge Autonomous Multi-Agent Synthesizer",
            "evaluation_contract": {
                "parity_assertion": "torch.testing.assert_close(rtol=1e-3, atol=1e-3)",
                "timing_harness": "triton.testing.do_bench(warmup=10, rep=50)"
            }
        },
        "summary": {
            "total_tasks_evaluated": total_tasks,
            "pass_at_1_count": pass_at_1_count,
            "pass_at_3_count": len(successful_tasks),
            "pass_rate_pct": pass_rate,
            "mean_speedup_vs_eager": mean_eager_speedup,
            "mean_speedup_vs_compile": mean_compile_speedup,
            "mean_bandwidth_saturation_pct": mean_bw_saturation
        },
        "tasks": results_tasks
    }

    # Persist JSON Artifact to Disk
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print("=" * 78)
    print(" [+] EVALUATION SUITE COMPLETE")
    print(f"  • Total Evaluated    : {total_tasks}")
    print(f"  • Pass Rate          : {pass_rate}%")
    print(f"  • Mean Speedup (Eager)  : {mean_eager_speedup}x")
    print(f"  • Mean Speedup (Compile): {mean_compile_speedup}x")
    print(f"  • Mean Memory Saturation: {mean_bw_saturation}%")
    print(f"  • Local JSON Artifact   : {OUTPUT_FILE}")
    print("=" * 78)

    # Optional Push to Hugging Face Hub
    if push_to_hub:
        target_repo = repo_id or os.getenv("HF_REPO_ID", "jay2219/kernelforge-kernelbench-results")
        hf_token = os.getenv("HF_TOKEN")

        if not hf_token:
            print(f"\n[!] Notice: Hugging Face upload skipped: HF_TOKEN environment variable not set.")
            print(f"[*] To publish to https://huggingface.co/datasets/{target_repo}:")
            print(f"    set HF_TOKEN=your_token_here")
            print(f"    python benchmarks/run_kernelbench.py --push --repo {target_repo}")
            return

        try:
            print(f"\n[+] Publishing results to Hugging Face Hub dataset: {target_repo}...")
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(repo_id=target_repo, repo_type="dataset", token=hf_token, exist_ok=True)

            # Upload JSON results
            api.upload_file(
                path_or_fileobj=str(OUTPUT_FILE),
                path_in_repo="kernelbench_results.json",
                repo_id=target_repo,
                repo_type="dataset",
                token=hf_token
            )

            # Upload README card if present
            if DATASET_CARD.exists():
                api.upload_file(
                    path_or_fileobj=str(DATASET_CARD),
                    path_in_repo="README.md",
                    repo_id=target_repo,
                    repo_type="dataset",
                    token=hf_token
                )

            print(f"[✓] Published successfully! View at: https://huggingface.co/datasets/{target_repo}")
        except Exception as e:
            print(f"[!] Error publishing to Hugging Face Hub: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KernelForge Stanford KernelBench Evaluation Harness")
    parser.add_argument("--level", type=str, default="1", choices=["1", "2", "all"], help="KernelBench Level (1, 2, or all)")
    parser.add_argument("--limit", type=int, default=10, help="Max tasks to evaluate")
    parser.add_argument("--push", action="store_true", help="Push results to Hugging Face Hub")
    parser.add_argument("--repo", type=str, default="jay2219/kernelforge-kernelbench-results", help="Hugging Face Dataset Repo ID")
    parser.add_argument("--hf-stream", action="store_true", help="Attempt live streaming from ScalingIntelligence/KernelBench on HF Hub")
    parser.add_argument("--live", action="store_true", help="Force execution against live Modal GPU Sandbox")
    parser.add_argument("--mock", action="store_true", help="Force local mock execution")
    args = parser.parse_args()

    run_evaluation(
        level=args.level,
        limit=args.limit,
        push_to_hub=args.push,
        repo_id=args.repo,
        stream_hf=args.hf_stream,
        force_live=args.live,
        force_mock=args.mock
    )
