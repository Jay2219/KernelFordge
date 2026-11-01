"""
Agent 2: The Synthesizer (muse-spark-1.3 / muse-code)
Translates the architectural tiling plan into watertight, memory-coalesced @triton.jit kernels.
"""

from typing import List
from ..schemas.compiler import TilingSpecification, HardwareSpec
from ..core.llm_client import LLMClient


SYNTHESIZER_SYSTEM_PROMPT = """You are an elite Triton Kernel Compiler Engineer.
Your task is to implement a production-ready, highly optimized OpenAI Triton kernel based on an architectural tiling plan.

CRITICAL TRITON IMPLEMENTATION RULES:
1. Always decorate the kernel with @triton.jit.
2. All block sizes passed to tl.load or tl.store must be tl.constexpr.
3. Every tl.load AND tl.store on non-power-of-two boundaries MUST have a mask argument: e.g., mask = offsets < total_elements, other=0.0 on load, and mask=mask on store.
4. NUMERICAL STABILITY CONTRACT: For any reduction (sum, mean, variance, RMS, softmax), always promote inputs to tl.float32 before computing squares, sum, or rsqrt to prevent FP16 overflow (max 65,504):
   x_f32 = tl.load(...).to(tl.float32)
   Downcast back to target dtype (tl.float16 or tl.bfloat16) only when writing to output via tl.store!
5. STRIDE CONTRACT: Utilize dynamic stride arguments extracted from PyTorch:
   ptr + (pid * stride_m + offsets * stride_k). Do NOT assume contiguous row-major strides.
6. MICRO-AUTOTUNING: Wrap the kernel invocation in a clean @triton.autotune decorator evaluating the top 3 candidate configs (sweeping num_warps in [2, 4, 8] and num_stages in [2, 3, 4]).
7. Provide a clean Python wrapper function that sets up the 1D/2D grid, allocates output tensors via torch.empty_like(), and invokes the autotuned kernel.

DO NOT import triton.language as anything other than tl.
Emit PURE Python code wrapped inside ```python ... ``` markdown blocks.
"""


class SynthesizerAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def synthesize_kernel(
        self,
        operation_name: str,
        pytorch_code: str,
        tiling_plan: TilingSpecification,
        shapes: List[List[int]],
        dtype: str = "float16",
        hardware: HardwareSpec = HardwareSpec(),
        force_fault: bool = False
    ) -> str:
        """
        Synthesizes the Triton kernel code based on the tiling plan.
        """
        user_prompt = f"""Target Accelerator: {hardware.gpu_name} (Capability {hardware.compute_capability})
Operation Name: {operation_name}
Target Dtype: {dtype}
Input Shapes: {shapes}

Tiling Plan:
- Operation Type: {tiling_plan.operation_type}
- BLOCK_SIZE_M: {tiling_plan.BLOCK_SIZE_M}
- BLOCK_SIZE_N: {tiling_plan.BLOCK_SIZE_N}
- num_warps: {tiling_plan.num_warps}
- num_stages: {tiling_plan.num_stages}
- Masking Logic: {tiling_plan.masking_logic}
- Rationale: {tiling_plan.fusion_rationale}

Reference PyTorch Implementation:
```python
{pytorch_code}
```

INSTRUCTIONS:
Synthesize the complete, compilable OpenAI Triton kernel and wrapper function.
Make sure to include the micro-autotune decorator and dynamic PyTorch stride arguments."""

        if force_fault:
            user_prompt += "\n\nNOTE FOR DEMO FAULT: Omit the boundary mask on tl.store to intentionally demonstrate self-healing."

        raw_response = self.llm.generate(
            system_prompt=SYNTHESIZER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            agent_role="SYNTHESIZER"
        )

        return self.llm.extract_python_code(raw_response)
