"""
Orchestration Engine: Finite State Machine Controller for Multi-Agent Synthesis and Self-Healing.
"""

import logging
from typing import Callable, Awaitable, Optional, Dict, Any
from ..schemas.compiler import (
    SynthesisRequest,
    SynthesisResponse,
    ExecutionStep,
    BenchmarkResult,
    TilingSpecification
)
from .llm_client import LLMClient
from ..agents.architect import ArchitectAgent
from ..agents.synthesizer import SynthesizerAgent
from ..agents.auditor import AuditorAgent
from ..agents.reflection import ReflectionAgent
from .sandbox import sandbox_dispatcher

logger = logging.getLogger("kernelforge.orchestrator")


class KernelOrchestrator:
    """
    Coordinates the 4-agent compiler lifecycle:
    Architect -> Synthesizer -> Auditor -> Sandbox -> (Reflection Loop)
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
        self.architect = ArchitectAgent(self.llm)
        self.synthesizer = SynthesizerAgent(self.llm)
        self.auditor = AuditorAgent(self.llm)
        self.reflection = ReflectionAgent(self.llm)

    async def run(
        self,
        request: SynthesisRequest,
        on_step_event: Optional[Callable[[ExecutionStep], Awaitable[None]]] = None
    ) -> SynthesisResponse:
        """
        Executes the closed-loop compilation and benchmarking pipeline.
        """
        session_id = request.session_id
        steps: list[ExecutionStep] = []

        async def emit_step(
            step_name: str,
            status: str,
            agent_name: str,
            message: str,
            code_snippet: Optional[str] = None,
            data: Optional[Dict[str, Any]] = None
        ):
            step = ExecutionStep(
                step_name=step_name,
                status=status,
                agent_name=agent_name,
                message=message,
                code_snippet=code_snippet,
                data=data
            )
            steps.append(step)
            if on_step_event:
                await on_step_event(step)

        # -------------------------------------------------------------
        # Step 1: Roofline & SRAM Tiling Strategy (muse-wonder)
        # -------------------------------------------------------------
        await emit_step(
            step_name="ROOFLINE_ANALYSIS",
            status="IN_PROGRESS",
            agent_name="Muse Wonder (Architect)",
            message="Deconstructing tensor graph arithmetic intensity against target GPU roofline..."
        )

        try:
            tiling_plan: TilingSpecification = self.architect.plan_tiling(
                operation_name=request.operation_name,
                pytorch_code=request.pytorch_source,
                shapes=request.input_tensor_shapes,
                dtype=request.dtype,
                hardware=request.hardware
            )
            await emit_step(
                step_name="TILING_PLAN",
                status="SUCCESS",
                agent_name="Muse Wonder (Architect)",
                message=f"Tiling formulated: BLOCK_M={tiling_plan.BLOCK_SIZE_M}, {tiling_plan.num_warps} warps. {tiling_plan.fusion_rationale}",
                data=tiling_plan.model_dump()
            )
        except Exception as e:
            await emit_step(
                step_name="TILING_PLAN",
                status="FAILED",
                agent_name="Muse Wonder (Architect)",
                message=f"Tiling planning failed: {str(e)}"
            )
            return SynthesisResponse(
                session_id=session_id,
                status="FAILED",
                iteration_count=1,
                error_log=str(e),
                steps=steps
            )

        # -------------------------------------------------------------
        # Step 2: Kernel Code Synthesis (muse-spark-1.3)
        # -------------------------------------------------------------
        await emit_step(
            step_name="KERNEL_SYNTHESIS",
            status="IN_PROGRESS",
            agent_name="Muse Code (Synthesizer)",
            message="Synthesizing memory-coalesced @triton.jit kernel with FP32 promotion and autotuning..."
        )

        try:
            current_triton_code = self.synthesizer.synthesize_kernel(
                operation_name=request.operation_name,
                pytorch_code=request.pytorch_source,
                tiling_plan=tiling_plan,
                shapes=request.input_tensor_shapes,
                dtype=request.dtype,
                hardware=request.hardware,
                force_fault=request.trigger_fault_demo
            )
            await emit_step(
                step_name="KERNEL_SYNTHESIS",
                status="SUCCESS",
                agent_name="Muse Code (Synthesizer)",
                message="Triton kernel successfully synthesized.",
                code_snippet=current_triton_code
            )
        except Exception as e:
            await emit_step(
                step_name="KERNEL_SYNTHESIS",
                status="FAILED",
                agent_name="Muse Code (Synthesizer)",
                message=f"Kernel synthesis failed: {str(e)}"
            )
            return SynthesisResponse(
                session_id=session_id,
                status="FAILED",
                iteration_count=1,
                tiling_plan=tiling_plan,
                error_log=str(e),
                steps=steps
            )

        # -------------------------------------------------------------
        # Step 3: Test Harness Synthesis (Muse Spark Open on Baseten)
        # -------------------------------------------------------------
        await emit_step(
            step_name="TEST_SYNTHESIS",
            status="IN_PROGRESS",
            agent_name="Muse Spark Open (Baseten Auditor)",
            message="Generating deterministic assert_close test fixtures and microsecond benchmark harness..."
        )

        try:
            test_harness_code = self.auditor.synthesize_test_harness(
                operation_name=request.operation_name,
                pytorch_code=request.pytorch_source,
                triton_code=current_triton_code,
                shapes=request.input_tensor_shapes,
                dtype=request.dtype,
                peak_bandwidth_gbs=request.hardware.peak_bandwidth_gbs
            )
            await emit_step(
                step_name="TEST_SYNTHESIS",
                status="SUCCESS",
                agent_name="Muse Spark Open (Baseten Auditor)",
                message="Verification harness and do_bench runner generated.",
                code_snippet=test_harness_code
            )
        except Exception as e:
            await emit_step(
                step_name="TEST_SYNTHESIS",
                status="FAILED",
                agent_name="Muse Spark Open (Baseten Auditor)",
                message=f"Auditor test generation failed: {str(e)}"
            )
            return SynthesisResponse(
                session_id=session_id,
                status="FAILED",
                iteration_count=1,
                tiling_plan=tiling_plan,
                triton_source_code=current_triton_code,
                error_log=str(e),
                steps=steps
            )

        # -------------------------------------------------------------
        # Step 4: Sandbox Execution & Autonomous Self-Healing Loop
        # -------------------------------------------------------------
        max_iterations = 3
        iteration = 1
        final_benchmark: Optional[BenchmarkResult] = None
        error_log = None

        while iteration <= max_iterations:
            await emit_step(
                step_name="SANDBOX_DISPATCH",
                status="IN_PROGRESS",
                agent_name="Modal Subprocess Sandbox",
                message=f"[Iteration {iteration}/{max_iterations}] Executing in isolated A10G container..."
            )

            execution_output = sandbox_dispatcher.execute(
                session_id=session_id,
                operation_name=request.operation_name,
                pytorch_code=request.pytorch_source,
                triton_code=current_triton_code,
                test_harness_code=test_harness_code,
                shapes=request.input_tensor_shapes,
                dtype=request.dtype,
                hardware=request.hardware,
                trigger_fault_demo=request.trigger_fault_demo if iteration == 1 else False
            )

            if execution_output["status"] == "SUCCESS":
                benchmark_data = execution_output["benchmark"]
                final_benchmark = BenchmarkResult(**benchmark_data)
                await emit_step(
                    step_name="BENCHMARK_COMPLETE",
                    status="SUCCESS",
                    agent_name="Modal Subprocess Sandbox",
                    message=(
                        f"Numerical Parity PASSED! Triton: {final_benchmark.triton_kernel_us:.1f}µs "
                        f"({final_benchmark.speedup_vs_eager:.2f}x vs Eager, {final_benchmark.speedup_vs_compile:.2f}x vs Compile) | "
                        f"Bandwidth: {final_benchmark.achieved_bandwidth_gbs:.1f} GB/s ({final_benchmark.bandwidth_saturation_pct:.1f}% Saturation)"
                    ),
                    data=final_benchmark.model_dump()
                )
                return SynthesisResponse(
                    session_id=session_id,
                    status="COMPLETED",
                    iteration_count=iteration,
                    tiling_plan=tiling_plan,
                    triton_source_code=current_triton_code,
                    test_harness_code=test_harness_code,
                    benchmark=final_benchmark,
                    steps=steps
                )

            # Failure occurred: Self-Healing Reflection Engine triggered
            error_log = execution_output.get("traceback", "Unknown error")
            await emit_step(
                step_name="SANDBOX_DISPATCH",
                status="FAILED",
                agent_name="Modal Subprocess Sandbox",
                message=f"Runtime exception intercepted: {error_log[:160]}..."
            )

            if iteration >= max_iterations:
                break

            # Trigger Reflection Agent
            await emit_step(
                step_name="REFLECTION_REPAIR",
                status="IN_PROGRESS",
                agent_name="Muse Code (Reflection Engine)",
                message="Analyzing stack trace and repairing boundary masks..."
            )

            try:
                current_triton_code = self.reflection.refactor_kernel(
                    broken_triton_code=current_triton_code,
                    pytorch_code=request.pytorch_source,
                    error_category=execution_output.get("error_category", "CUDARuntimeError"),
                    traceback_log=error_log
                )
                await emit_step(
                    step_name="REFLECTION_REPAIR",
                    status="SUCCESS",
                    agent_name="Muse Code (Reflection Engine)",
                    message=f"Kernel patched on Iteration {iteration + 1}. Re-dispatching to clean subprocess...",
                    code_snippet=current_triton_code
                )
            except Exception as ref_err:
                await emit_step(
                    step_name="REFLECTION_REPAIR",
                    status="FAILED",
                    agent_name="Muse Code (Reflection Engine)",
                    message=f"Self-healing reflection failed: {str(ref_err)}"
                )
                break

            iteration += 1

        return SynthesisResponse(
            session_id=session_id,
            status="FAILED",
            iteration_count=iteration,
            tiling_plan=tiling_plan,
            triton_source_code=current_triton_code,
            test_harness_code=test_harness_code,
            error_log=error_log,
            steps=steps
        )

