"""
Pydantic v2 Data Contracts for KernelForge Compilation and Benchmarking.
"""

from typing import List, Dict, Optional, Literal, Any
from pydantic import BaseModel, Field


class HardwareSpec(BaseModel):
    gpu_name: str = Field(default="NVIDIA A10G", description="Target GPU accelerator")
    compute_capability: float = Field(default=8.6, description="CUDA Major.Minor version")
    total_sm_count: int = Field(default=80, description="Streaming Multiprocessors")
    sram_per_sm_kb: int = Field(default=128, description="Shared memory per SM in Kilobytes")
    peak_bandwidth_gbs: float = Field(default=600.0, description="Theoretical peak memory bandwidth in GB/s")
    peak_fp16_tflops: float = Field(default=31.2, description="Theoretical peak FP16/BF16 compute throughput in TFLOPS")


class SynthesisRequest(BaseModel):
    session_id: str = Field(description="Unique client session identifier")
    operation_name: str = Field(default="fused_rms_norm", description="Human-readable operation title")
    pytorch_source: str = Field(description="Executable PyTorch eager function implementation")
    input_tensor_shapes: List[List[int]] = Field(
        default=[[32, 4096]], 
        description="Input tensor shapes for testing and benchmark verification"
    )
    dtype: Literal["float16", "bfloat16", "float32"] = Field(default="float16")
    hardware: HardwareSpec = Field(default_factory=HardwareSpec)
    trigger_fault_demo: bool = Field(
        default=False, 
        description="If true, intentionally seeds an unmasked out-of-bounds access (e.g. N=4097) to demonstrate self-healing"
    )


class TilingSpecification(BaseModel):
    operation_type: Literal["Reduction", "Elementwise", "GEMM", "Fused"] = Field(default="Reduction")
    operational_intensity: float = Field(description="Calculated FLOPs / Byte transferred")
    memory_bound: bool = Field(default=True)
    BLOCK_SIZE_M: int = Field(default=1024)
    BLOCK_SIZE_N: Optional[int] = Field(default=None)
    num_warps: int = Field(default=4)
    num_stages: int = Field(default=3)
    masking_logic: str = Field(description="Description of boundary check condition")
    fusion_rationale: str = Field(description="Theoretical rationale for why HBM roundtrips are avoided")


class BenchmarkResult(BaseModel):
    pytorch_eager_us: float = Field(description="PyTorch eager execution time in microseconds")
    pytorch_compile_us: float = Field(description="torch.compile (Inductor) execution time in microseconds")
    triton_kernel_us: float = Field(description="KernelForge synthesized Triton execution time in microseconds")
    speedup_vs_eager: float = Field(description="Speedup ratio over PyTorch eager mode")
    speedup_vs_compile: float = Field(description="Speedup ratio over torch.compile mode")
    achieved_bandwidth_gbs: float = Field(description="Empirical memory bandwidth achieved in GB/s")
    bandwidth_saturation_pct: float = Field(description="Percentage of theoretical peak bandwidth saturated")
    numerical_parity_verified: bool = Field(description="Whether assert_close passed across all test shapes")
    max_absolute_error: float = Field(default=0.0, description="Max absolute diff between PyTorch and Triton output")
    max_relative_error: float = Field(default=0.0, description="Max relative diff between PyTorch and Triton output")


class ExecutionStep(BaseModel):
    step_name: Literal[
        "ROOFLINE_ANALYSIS", 
        "TILING_PLAN", 
        "KERNEL_SYNTHESIS", 
        "TEST_SYNTHESIS", 
        "SANDBOX_DISPATCH", 
        "BENCHMARK_COMPLETE",
        "REFLECTION_REPAIR"
    ]
    status: Literal["PENDING", "IN_PROGRESS", "SUCCESS", "FAILED"]
    agent_name: str
    message: str
    code_snippet: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class SynthesisResponse(BaseModel):
    session_id: str
    status: Literal["COMPLETED", "FAILED", "ITERATING"]
    iteration_count: int = Field(default=1)
    tiling_plan: Optional[TilingSpecification] = None
    triton_source_code: Optional[str] = None
    test_harness_code: Optional[str] = None
    benchmark: Optional[BenchmarkResult] = None
    error_log: Optional[str] = None
    steps: List[ExecutionStep] = Field(default_factory=list)

