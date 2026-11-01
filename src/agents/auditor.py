"""
Agent 3: The Auditor (meta-models/muse-spark-open-1026 on Baseten)
Synthesizes unit test fixtures, numerical assertions, and tri-comparison benchmark runners.
"""

from typing import List
from ..core.llm_client import LLMClient


AUDITOR_SYSTEM_PROMPT = """You are an elite QA and Numerical Precision Compiler Auditor.
Your task is to generate a deterministic Python test and benchmark harness that evaluates
a synthesized OpenAI Triton kernel against its reference PyTorch eager and torch.compile implementations.

HARNESS REQUIREMENTS:
1. Provide a single entrypoint function: `run_verification_and_benchmark() -> dict`.
2. Generate deterministic random tensors on the active CUDA device with explicit seeds.
3. Test across multiple shapes including power-of-two (e.g. [32, 4096]) and unaligned edge cases (e.g. [16, 4097]).
4. Verify numerical parity using `torch.testing.assert_close(triton_out, eager_out, rtol=1e-3, atol=1e-3)`.
5. Measure Tri-Comparison execution latency using `triton.testing.do_bench()`:
   - Warm-up PyTorch eager 10 times, benchmark eager_us.
   - Warm-up torch.compile(eager_fn) 15 times, benchmark compile_us.
   - Warm-up Triton kernel 10 times, benchmark triton_us.
6. Calculate speedup ratios and memory bandwidth saturation against the hardware specs.
7. Return a clean Python dictionary containing the results.

Emit PURE Python code wrapped inside ```python ... ``` markdown blocks.
"""


class AuditorAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def synthesize_test_harness(
        self,
        operation_name: str,
        pytorch_code: str,
        triton_code: str,
        shapes: List[List[int]],
        dtype: str = "float16",
        peak_bandwidth_gbs: float = 600.0
    ) -> str:
        """
        Invokes Muse Spark Open on Baseten to generate the verification and benchmark harness.
        """
        user_prompt = f"""Operation Name: {operation_name}
Target Dtype: {dtype}
Primary Input Shapes: {shapes}
Target GPU Peak Bandwidth: {peak_bandwidth_gbs} GB/s

PyTorch Reference Code:
```python
{pytorch_code}
```

Synthesized Triton Code:
```python
{triton_code}
```

Synthesize the complete, executable `run_verification_and_benchmark()` function."""

        raw_response = self.llm.generate(
            system_prompt=AUDITOR_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            agent_role="AUDITOR"
        )

        return self.llm.extract_python_code(raw_response)
