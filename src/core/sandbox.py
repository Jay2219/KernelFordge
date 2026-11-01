"""
Sandbox Dispatcher & Remote Execution Interface (Modal & Local Mock).
"""

import json
import logging
from typing import Dict, Any, List
from ..schemas.compiler import HardwareSpec
from .llm_client import settings
from .mock_engine import mock_engine

logger = logging.getLogger("kernelforge.sandbox")


class SandboxDispatcher:
    """
    Dispatches compilation bundles to either the isolated Modal GPU sandbox
    or the local mock engine.
    """

    def __init__(self):
        self.settings = settings

    def execute(
        self,
        session_id: str,
        operation_name: str,
        pytorch_code: str,
        triton_code: str,
        test_harness_code: str,
        shapes: List[List[int]],
        dtype: str = "float16",
        hardware: HardwareSpec = HardwareSpec(),
        trigger_fault_demo: bool = False
    ) -> Dict[str, Any]:
        """
        Bundles code and dispatches to the appropriate execution backend.
        """
        combined_code = f"""# --- PyTorch Eager Reference ---
{pytorch_code}

# --- KernelForge Synthesized Triton Kernel ---
{triton_code}

# --- Auditor Test & Benchmark Harness ---
{test_harness_code}
"""
        # Route to Mock Engine if enabled
        if self.settings.MOCK_SANDBOX:
            logger.info(f"[{session_id}] Executing in Local Mock Sandbox...")
            return mock_engine.execute_payload(
                session_id=session_id,
                operation_name=operation_name,
                shapes=shapes,
                dtype=dtype,
                hardware=hardware,
                trigger_fault_demo=trigger_fault_demo
            )

        # Route to Real Modal NVIDIA A10G Sandbox
        logger.info(f"[{session_id}] Dispatching payload to Modal GPU Sandbox ({self.settings.MODAL_GPU_TARGET})...")
        try:
            import modal
            execute_fn = modal.Function.from_name(
                self.settings.MODAL_APP_NAME,
                "execute_kernel_payload"
            )
            bundle = json.dumps({"combined_executable_code": combined_code})
            raw_response = execute_fn.remote(bundle)
            return json.loads(raw_response)
        except Exception as e:
            logger.error(f"Modal execution failed: {e}")
            return {
                "status": "FAILED",
                "benchmark": None,
                "traceback": f"Modal Container Dispatch Error: {str(e)}"
            }


sandbox_dispatcher = SandboxDispatcher()

