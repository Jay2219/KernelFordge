"""
Unified LLM Provider Client with Resilient Fallback for Meta Model API and Modal Open-Weights.
"""

import os
import json
import re
import logging
from typing import Dict, Any, Optional, List
import httpx
from openai import OpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("kernelforge.llm")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    META_MODEL_API_KEY: str = "mock_key"
    META_API_BASE_URL: str = "https://api.meta.ai/v1"
    MODEL_ARCHITECT: str = "muse-wonder-1.0"
    MODEL_SYNTHESIZER: str = "muse-spark-1.3"
    MODEL_REFLECTION: str = "muse-spark-1.3"

    # Modal Open-Weights Serverless Endpoint (qualifies for $10K prize)
    MODAL_OPEN_WEIGHTS_URL: Optional[str] = None
    MODEL_AUDITOR: str = "meta-models/muse-spark-open-1026"

    FALLBACK_MODE: bool = False
    FALLBACK_API_KEY: Optional[str] = None
    FALLBACK_BASE_URL: str = "https://api.openai.com/v1"
    FALLBACK_MODEL: str = "gpt-4o"

    MOCK_SANDBOX: bool = False
    MODAL_GPU_TARGET: str = "A10G"
    MODAL_APP_NAME: str = "kernelforge-sandbox"
    MODAL_TOKEN_ID: Optional[str] = None
    MODAL_TOKEN_SECRET: Optional[str] = None



settings = Settings()


class LLMClient:
    """
    Unified client coordinating Meta Model API and Modal Open-Weights endpoints
    with automatic failover handling.
    """

    def __init__(self):
        self.settings = settings
        
        # Meta Model API Client
        self.meta_client = OpenAI(
            base_url=self.settings.META_API_BASE_URL,
            api_key=self.settings.META_MODEL_API_KEY or "none"
        )

        # Optional Fallback Proxy Client
        self.fallback_client = None
        if self.settings.FALLBACK_API_KEY:
            self.fallback_client = OpenAI(
                base_url=self.settings.FALLBACK_BASE_URL,
                api_key=self.settings.FALLBACK_API_KEY
            )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        agent_role: str = "SYNTHESIZER",
        temperature: float = 0.2,
        max_tokens: int = 8192
    ) -> str:
        """
        Routes the request to the appropriate model based on agent role.
        """
        # Route AUDITOR to Modal Open-Weights Serverless Engine
        if agent_role == "AUDITOR":
            modal_ow_url = (self.settings.MODAL_OPEN_WEIGHTS_URL or "").strip()
            if modal_ow_url and not modal_ow_url.startswith("https://your-workspace"):
                try:
                    with httpx.Client(timeout=60.0) as http_client:
                        resp = http_client.post(
                            modal_ow_url,
                            json={
                                "prompt": user_prompt,
                                "system_prompt": system_prompt,
                                "operation_name": "kernel_audit"
                            }
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            return data.get("test_harness") or data.get("response") or resp.text
                except Exception as e:
                    logger.warning(f"Modal Open-Weights endpoint call failed: {e}. Falling back to Meta Model API...")
                    if self.settings.MOCK_SANDBOX and self.settings.META_MODEL_API_KEY in ("mock_key", "", "none"):
                        return self._mock_agent_response("AUDITOR", user_prompt)
                    # Gracefully fall through to Meta Model API below
            elif self.settings.MOCK_SANDBOX:
                return self._mock_agent_response("AUDITOR", user_prompt)

        # Determine model for Meta Model API
        if agent_role == "ARCHITECT":
            client = self.meta_client
            model = self.settings.MODEL_ARCHITECT
        elif agent_role in ("SYNTHESIZER", "REFLECTION", "AUDITOR"):
            client = self.meta_client
            model = self.settings.MODEL_SYNTHESIZER
        else:
            client = self.meta_client
            model = self.settings.MODEL_SYNTHESIZER

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # In Mock Mode without valid live keys, return deterministic templates
        if self.settings.MOCK_SANDBOX and (
            self.settings.META_MODEL_API_KEY in ("mock_key", "", "none")
        ):
            return self._mock_agent_response(agent_role, user_prompt)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"Primary endpoint failed for {agent_role} ({model}): {e}")
            if self.settings.FALLBACK_MODE and self.fallback_client:
                logger.info("Executing resilient fallback to secondary proxy...")
                fb_resp = self.fallback_client.chat.completions.create(
                    model=self.settings.FALLBACK_MODEL,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return fb_resp.choices[0].message.content or ""
            elif self.settings.MOCK_SANDBOX:
                logger.info("Falling back to local deterministic mock response...")
                return self._mock_agent_response(agent_role, user_prompt)
            raise e

    def extract_json(self, text: str) -> Dict[str, Any]:
        """
        Robustly extracts JSON from an LLM response even if wrapped in markdown blocks.
        """
        # Check for ```json ... ``` blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if json_match:
            clean_str = json_match.group(1).strip()
        else:
            clean_str = text.strip()

        try:
            return json.loads(clean_str)
        except json.JSONDecodeError:
            # Fallback: extract substring between first { and last }
            first_brace = clean_str.find("{")
            last_brace = clean_str.rfind("}")
            if first_brace != -1 and last_brace != -1:
                try:
                    return json.loads(clean_str[first_brace : last_brace + 1])
                except json.JSONDecodeError:
                    pass
            logger.warning(f"Could not parse valid JSON from LLM response. Falling back to default spec.")
            return {}

    def extract_python_code(self, text: str) -> str:
        """
        Extracts executable Python code from markdown blocks, or raw python code if not fenced.
        """
        code_match = re.search(r"```(?:python)?\s*([\s\S]*?)\s*```", text)
        if code_match:
            return code_match.group(1).strip()
        cleaned = text.strip()
        if "@triton.jit" in cleaned or "def " in cleaned or "import " in cleaned:
            return cleaned
        return ""

    def _mock_agent_response(self, role: str, prompt: str) -> str:
        """
        Provides deterministic, production-grade responses for offline development.
        """
        if role == "ARCHITECT":
            return json.dumps({
                "operation_type": "Reduction",
                "operational_intensity": 1.25,
                "memory_bound": True,
                "tiling_plan": {
                    "BLOCK_SIZE_M": 1024,
                    "BLOCK_SIZE_N": None,
                    "num_warps": 4,
                    "num_stages": 3
                },
                "masking_logic": "mask = cols < N on both load and store",
                "fusion_rationale": "Keeps row sum-of-squares in SRAM registers, eliminating 3 intermediate HBM buffers."
            }, indent=2)

        elif role == "SYNTHESIZER":
            return """```python
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024, 'num_warps': 4, 'num_stages': 3}),
        triton.Config({'BLOCK_SIZE': 2048, 'num_warps': 8, 'num_stages': 4}),
        triton.Config({'BLOCK_SIZE': 512,  'num_warps': 2, 'num_stages': 2}),
    ],
    key=['N']
)
@triton.jit
def _rms_norm_kernel(
    x_ptr, y_ptr, w_ptr,
    stride_x_row, stride_y_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    x_row = x_ptr + row_idx * stride_x_row
    y_row = y_ptr + row_idx * stride_y_row

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # Load and promote to float32 accumulator for numerical stability
    x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    sum_sq = tl.sum(x * x, axis=0)
    variance = sum_sq / N
    rstd = tl.rsqrt(variance + eps)

    y = (x * rstd * w).to(tl.float16)
    tl.store(y_row + cols, y, mask=mask)

def triton_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty_like(x)
    grid = (M,)
    _rms_norm_kernel[grid](
        x, y, weight,
        x.stride(0), y.stride(0),
        N, eps
    )
    return y
```"""

        elif role == "AUDITOR":
            return """```python
import torch
import triton.testing

def run_verification_and_benchmark():
    # Deterministic test harness generated by Muse Spark Open on Modal Open-Weights Engine
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Parity Audit across unaligned and standard shapes
    test_shapes = [[16, 4096], [32, 1024]]
    for M, N in test_shapes:
        x = torch.randn(M, N, dtype=torch.float16, device=device)
        w = torch.ones(N, dtype=torch.float16, device=device)
        
        # Ground truth
        eager_out = rms_norm_eager(x, w)
        triton_out = triton_rms_norm(x, w)
        
        torch.testing.assert_close(triton_out, eager_out, rtol=1e-3, atol=1e-3)
        
    return {"parity_verified": True}
```"""

        elif role == "REFLECTION":
            return """```python
# Refactored patch applying boundary mask
import torch
import triton
import triton.language as tl

@triton.jit
def _rms_norm_kernel(x_ptr, y_ptr, w_ptr, stride_x_row, stride_y_row, N, eps, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N  # Enforce boundary mask to prevent illegal memory address
    x = tl.load(x_ptr + row_idx * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / N + eps)
    tl.store(y_ptr + row_idx * stride_y_row + cols, (x * rstd * w).to(tl.float16), mask=mask)
```"""

        return ""
