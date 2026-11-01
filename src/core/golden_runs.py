"""
Verified Golden Runs Loader.
Loads empirical NVIDIA A10G benchmark executions generated on Modal serverless hardware.
"""

import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("kernelforge.golden_runs")

GOLDEN_RUNS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "artifacts", "golden_runs")


def load_golden_run(operation_name: str) -> Optional[Dict[str, Any]]:
    """
    Loads verified golden run data for an operation from disk.
    """
    clean_op = operation_name.lower().strip()
    file_path = os.path.join(GOLDEN_RUNS_DIR, f"{clean_op}.json")
    
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading golden run from {file_path}: {e}")
            
    # Check alternate direct path (e.g. inside Docker container /app/artifacts/golden_runs)
    alt_path = os.path.join("/app", "artifacts", "golden_runs", f"{clean_op}.json")
    if os.path.exists(alt_path):
        try:
            with open(alt_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading golden run from {alt_path}: {e}")

    logger.warning(f"Golden run for '{operation_name}' not found on disk at {file_path}")
    return None


def list_golden_runs() -> Dict[str, Dict[str, Any]]:
    """
    Enumerates all available verified golden runs.
    """
    runs = {}
    known_ops = ["fused_rms_norm", "fused_swiglu", "fused_softmax"]
    for op in known_ops:
        data = load_golden_run(op)
        if data:
            runs[op] = data
    return runs
