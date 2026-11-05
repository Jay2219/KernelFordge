"""
Agent 4: The Self-Healing Reflection Engine (muse-spark-1.3)
Parses isolated subprocess stack traces and refactors memory misalignment or numerical bugs.
"""

from typing import Optional
from ..core.llm_client import LLMClient


REFLECTION_SYSTEM_PROMPT = """You are an elite Triton Kernel Compiler Engineer specializing in CUDA error recovery and memory bounds debugging.
A candidate OpenAI Triton kernel failed during isolated GPU sandbox execution.

DIAGNOSTIC STRATEGY:
1. If the error mentions 'illegal memory access' or 'device-side assert':
   - An out-of-bounds pointer write occurred.
   - Enforce boundary masks on BOTH tl.load AND tl.store: e.g. `mask = cols < N`.
2. If the error mentions 'CompilationError' or 'constexpr':
   - A block size or layout parameter was passed dynamically instead of as a `tl.constexpr`.
3. If the error is a NumericalMismatch or NaN values:
   - FP16 overflow occurred. Enforce intermediate accumulation in `tl.float32`.
4. Ensure reduction axis matches tensor layout (for 1D block reduction across a row, use axis=0).

Output ONLY the fully corrected, executable Python source code wrapped inside ```python ... ``` blocks.
"""


class ReflectionAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def refactor_kernel(
        self,
        broken_triton_code: str,
        pytorch_code: str,
        error_category: str,
        traceback_log: str,
        max_abs_err: Optional[float] = None,
        max_rel_err: Optional[float] = None
    ) -> str:
        """
        Submits the broken kernel and error traceback to muse-spark-1.3 for automated repair.
        """
        user_prompt = f"""FAILED TRITON SOURCE CODE:
```python
{broken_triton_code}
```

REFERENCE PYTORCH CODE:
```python
{pytorch_code}
```

ERROR CATEGORY: {error_category}
SUBPROCESS STDERR TRACEBACK:
{traceback_log}
"""
        if max_abs_err is not None:
            user_prompt += f"\nASSERTION MISMATCH:\n- Max Absolute Error: {max_abs_err}\n- Max Relative Error: {max_rel_err}\n"

        user_prompt += "\nAnalyze the failure and provide the fully corrected, revised Python implementation."

        raw_response = self.llm.generate(
            system_prompt=REFLECTION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            agent_role="REFLECTION",
            max_tokens=8192
        )

        extracted_code = self.llm.extract_python_code(raw_response)
        if not extracted_code or ("def " not in extracted_code and "@triton.jit" not in extracted_code):
            extracted_code = failed_triton_code

        return extracted_code

