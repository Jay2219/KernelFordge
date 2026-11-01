"""
Automated Test Suite for KernelForge Backend Pipeline.
"""

import pytest
from src.schemas.compiler import SynthesisRequest, HardwareSpec
from src.core.orchestrator import KernelOrchestrator
from src.core.roofline import calculate_op_characteristics, compute_bandwidth_saturation
from benchmarks import PRESET_RMS_NORM, PRESET_SWIGLU


def test_roofline_calculations():
    # Test RMSNorm characteristics
    chars = calculate_op_characteristics("fused_rms_norm", [[32, 4096]], "float16")
    assert chars["operation_type"] == "Reduction"
    assert chars["is_memory_bound"] is True
    assert chars["operational_intensity"] < 10.0

    # Test SwiGLU characteristics
    chars_swi = calculate_op_characteristics("fused_swiglu", [[32, 11008]], "float16")
    assert chars_swi["operation_type"] == "Elementwise"
    assert chars_swi["is_memory_bound"] is True

    # Test Bandwidth Saturation
    bw_gbs, sat_pct = compute_bandwidth_saturation(
        total_bytes=100_000_000,
        latency_us=200.0,
        peak_bandwidth_gbs=600.0
    )
    assert bw_gbs == 500.0
    assert abs(sat_pct - 83.3) < 0.2


@pytest.mark.asyncio
async def test_orchestrator_happy_path():
    orchestrator = KernelOrchestrator()
    request = SynthesisRequest(
        session_id="test-happy-path",
        operation_name=PRESET_RMS_NORM["operation_name"],
        pytorch_source=PRESET_RMS_NORM["pytorch_source"],
        input_tensor_shapes=PRESET_RMS_NORM["input_tensor_shapes"],
        dtype="float16",
        trigger_fault_demo=False
    )

    recorded_steps = []
    async def on_step(step):
        recorded_steps.append(step.step_name)

    response = await orchestrator.run(request, on_step_event=on_step)

    assert response.status == "COMPLETED"
    assert response.iteration_count == 1
    assert response.benchmark is not None
    assert response.benchmark.speedup_vs_eager > 1.5
    assert response.benchmark.speedup_vs_compile >= 1.0
    assert response.benchmark.numerical_parity_verified is True
    assert "ROOFLINE_ANALYSIS" in recorded_steps
    assert "BENCHMARK_COMPLETE" in recorded_steps


@pytest.mark.asyncio
async def test_orchestrator_self_healing_recovery():
    orchestrator = KernelOrchestrator()
    request = SynthesisRequest(
        session_id="test-self-healing",
        operation_name=PRESET_RMS_NORM["operation_name"],
        pytorch_source=PRESET_RMS_NORM["pytorch_source"],
        input_tensor_shapes=PRESET_RMS_NORM["fault_shape"],  # N=4097 unaligned shape
        dtype="float16",
        trigger_fault_demo=True  # Injects CUDA illegal address on iteration 1
    )

    recorded_steps = []
    async def on_step(step):
        recorded_steps.append(step.step_name)

    response = await orchestrator.run(request, on_step_event=on_step)

    # Verifies that iteration 1 failed, reflection triggered, and iteration 2 succeeded!
    assert response.status == "COMPLETED"
    assert response.iteration_count == 2
    assert "REFLECTION_REPAIR" in recorded_steps
    assert response.benchmark is not None
    assert response.benchmark.numerical_parity_verified is True

