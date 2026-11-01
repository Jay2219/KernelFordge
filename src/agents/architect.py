"""
Agent 1: The Architect (muse-wonder)
Deconstructs tensor operations, determines operational intensity, and designs the SRAM tiling plan.
"""

from typing import List, Dict, Any
from ..schemas.compiler import HardwareSpec, TilingSpecification
from ..core.llm_client import LLMClient
from ..core.roofline import calculate_op_characteristics


ARCHITECT_SYSTEM_PROMPT = """You are an expert GPU Systems Architect specialized in high-performance computer architecture and OpenAI Triton memory layouts.
Your task is to analyze an incoming PyTorch operation and target GPU hardware specifications, then formulate an optimal shared-memory tiling strategy.

CRITICAL INSTRUCTIONS:
1. Analyze the arithmetic intensity (FLOPs / Byte).
2. Determine power-of-two block sizes (BLOCK_SIZE_M, BLOCK_SIZE_N) that fit within the target GPU's Shared Memory (SRAM) per SM.
3. Select num_warps (2, 4, or 8) and num_stages (2, 3, or 4) to maximize warp occupancy and memory pipeline hiding.
4. Specify the boundary masking condition (e.g. mask = cols < N) required to prevent illegal memory accesses for arbitrary non-power-of-two tensor dimensions.

OUTPUT FORMAT:
Emit ONLY a valid JSON payload matching this exact schema:
{
  "operation_type": "Reduction" | "Elementwise" | "GEMM" | "Fused",
  "operational_intensity": float,
  "memory_bound": boolean,
  "tiling_plan": {
    "BLOCK_SIZE_M": int,
    "BLOCK_SIZE_N": int | null,
    "num_warps": int,
    "num_stages": int
  },
  "masking_logic": "string describing pointer boundary condition",
  "fusion_rationale": "string explaining how intermediate HBM roundtrips are avoided"
}
"""


class ArchitectAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def plan_tiling(
        self,
        operation_name: str,
        pytorch_code: str,
        shapes: List[List[int]],
        dtype: str = "float16",
        hardware: HardwareSpec = HardwareSpec()
    ) -> TilingSpecification:
        """
        Invokes muse-wonder to formulate the architectural tiling plan.
        """
        # Calculate theoretical baseline values
        characteristics = calculate_op_characteristics(operation_name, shapes, dtype, hardware)

        user_prompt = f"""Target Hardware: {hardware.gpu_name}
- Total SMs: {hardware.total_sm_count}
- SRAM per SM: {hardware.sram_per_sm_kb} KB
- Peak Bandwidth: {hardware.peak_bandwidth_gbs} GB/s
- Peak FP16 Compute: {hardware.peak_fp16_tflops} TFLOPS

Operation: {operation_name}
Data Type: {dtype}
Input Shapes: {shapes}

Reference PyTorch Code:
```python
{pytorch_code}
```

Calculated Theoretical FLOPs: {characteristics['total_flops']}
Calculated Fused Bytes: {characteristics['fused_bytes_transferred']}
Calculated Arithmetic Intensity: {characteristics['operational_intensity']} FLOPs/Byte

Formulate the optimal tiling specification and emit JSON."""

        response_text = self.llm.generate(
            system_prompt=ARCHITECT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            agent_role="ARCHITECT"
        )

        plan_data = self.llm.extract_json(response_text)
        tiling_inner = plan_data.get("tiling_plan", {})

        return TilingSpecification(
            operation_type=plan_data.get("operation_type", characteristics["operation_type"]),
            operational_intensity=plan_data.get("operational_intensity", characteristics["operational_intensity"]),
            memory_bound=plan_data.get("memory_bound", characteristics["is_memory_bound"]),
            BLOCK_SIZE_M=tiling_inner.get("BLOCK_SIZE_M", 1024),
            BLOCK_SIZE_N=tiling_inner.get("BLOCK_SIZE_N", None),
            num_warps=tiling_inner.get("num_warps", 4),
            num_stages=tiling_inner.get("num_stages", 3),
            masking_logic=plan_data.get("masking_logic", "mask = cols < N on load and store"),
            fusion_rationale=plan_data.get("fusion_rationale", characteristics["fusion_rationale"])
        )

