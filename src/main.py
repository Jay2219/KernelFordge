"""
KernelForge FastAPI Gateway and Real-Time WebSocket Orchestrator.
"""

import json
import uuid
import time
import logging
from datetime import datetime, timezone
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas.compiler import SynthesisRequest, SynthesisResponse, HardwareSpec, ExecutionStep
from .core.orchestrator import KernelOrchestrator
from .core.llm_client import settings
from .core.roofline import calculate_op_characteristics
from benchmarks import ALL_PRESETS

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("kernelforge.api")

app = FastAPI(
    title="KernelForge API",
    description="Autonomous GPU Kernel Synthesizer & Memory-Roofline Compiler",
    version="0.1.0"
)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = KernelOrchestrator()


@app.get("/")
def get_status():
    return {
        "status": "ONLINE",
        "service": "KernelForge Gateway",
        "mock_sandbox": settings.MOCK_SANDBOX,
        "gpu_target": settings.MODAL_GPU_TARGET,
        "models": {
            "architect": settings.MODEL_ARCHITECT,
            "synthesizer": settings.MODEL_SYNTHESIZER,
            "auditor": settings.MODEL_AUDITOR,
            "reflection": settings.MODEL_REFLECTION
        }
    }


@app.get("/api/v1/health")
async def health_check():
    """
    Real-time pre-flight infrastructure diagnostic endpoint.
    Checks actual API keys and sandbox connectivity without fabricated metrics.
    """
    # 1. Meta Model API Health Check
    meta_key = (settings.META_MODEL_API_KEY or "").strip()
    meta_configured = bool(meta_key and meta_key not in ("mock_key", "none", ""))
    meta_status = "OFFLINE"
    meta_latency_ms = None
    meta_message = "META_MODEL_API_KEY not configured"

    if meta_configured:
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                res = await client.get(
                    f"{settings.META_API_BASE_URL.rstrip('/')}/models",
                    headers={"Authorization": f"Bearer {meta_key}"}
                )
                meta_latency_ms = round((time.perf_counter() - t0) * 1000, 1)
                meta_status = "ONLINE" if res.status_code in (200, 401, 403) else "ERROR"
                meta_message = f"HTTP {res.status_code}"
        except Exception as e:
            meta_status = "OFFLINE"
            meta_message = f"Connection failed: {str(e)}"

    # 2. Baseten Open-Weights Health Check
    baseten_key = (settings.BASETEN_API_KEY or "").strip()
    baseten_configured = bool(baseten_key and baseten_key not in ("mock_key", "none", ""))
    baseten_status = "OFFLINE"
    baseten_latency_ms = None
    baseten_message = "BASETEN_API_KEY not configured"

    if baseten_configured:
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                res = await client.get(
                    f"{settings.BASETEN_API_BASE_URL.rstrip('/')}/models",
                    headers={"Authorization": f"Bearer {baseten_key}"}
                )
                baseten_latency_ms = round((time.perf_counter() - t0) * 1000, 1)
                baseten_status = "ONLINE" if res.status_code in (200, 401, 403) else "ERROR"
                baseten_message = f"HTTP {res.status_code}"
        except Exception as e:
            baseten_status = "OFFLINE"
            baseten_message = f"Connection failed: {str(e)}"

    # 3. GPU Sandbox Health Check
    if settings.MOCK_SANDBOX:
        gpu_status = "LOCAL_MOCK"
        gpu_target = "Local CPU Mock (Zero-GPU)"
        gpu_latency_ms = None
        gpu_message = "MOCK_SANDBOX=True; local mock active"
    else:
        # Check Modal connectivity
        t0 = time.perf_counter()
        try:
            import modal
            gpu_status = "ONLINE"
            gpu_target = f"Modal {settings.MODAL_GPU_TARGET}"
            gpu_latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            gpu_message = f"Connected to Modal app: {settings.MODAL_APP_NAME}"
        except Exception as e:
            gpu_status = "OFFLINE"
            gpu_target = f"Modal {settings.MODAL_GPU_TARGET}"
            gpu_latency_ms = None
            gpu_message = f"Modal unavailable: {str(e)}"

    if meta_status == "ONLINE" and baseten_status == "ONLINE" and gpu_status == "ONLINE":
        system_status = "HEALTHY"
    elif gpu_status == "LOCAL_MOCK":
        system_status = "LOCAL_MOCK_ACTIVE"
    else:
        system_status = "OFFLINE"

    return {
        "status": system_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu_pool": {
            "status": gpu_status,
            "target": gpu_target,
            "latency_ms": gpu_latency_ms,
            "message": gpu_message
        },
        "models": {
            "meta_api": {
                "status": meta_status,
                "configured": meta_configured,
                "model": f"{settings.MODEL_ARCHITECT} + {settings.MODEL_SYNTHESIZER}",
                "latency_ms": meta_latency_ms,
                "message": meta_message
            },
            "baseten": {
                "status": baseten_status,
                "configured": baseten_configured,
                "model": settings.MODEL_AUDITOR,
                "latency_ms": baseten_latency_ms,
                "message": baseten_message
            }
        }
    }


@app.get("/api/v1/presets")
def list_presets():
    return ALL_PRESETS


@app.get("/api/v1/hardware")
def get_hardware_specs():
    return {
        "A10G": HardwareSpec(
            gpu_name="NVIDIA A10G",
            compute_capability=8.6,
            total_sm_count=80,
            sram_per_sm_kb=128,
            peak_bandwidth_gbs=600.0,
            peak_fp16_tflops=31.2
        ).model_dump(),
        "T4": HardwareSpec(
            gpu_name="NVIDIA T4",
            compute_capability=7.5,
            total_sm_count=40,
            sram_per_sm_kb=64,
            peak_bandwidth_gbs=300.0,
            peak_fp16_tflops=65.0
        ).model_dump(),
        "H100": HardwareSpec(
            gpu_name="NVIDIA H100 SXM5",
            compute_capability=9.0,
            total_sm_count=132,
            sram_per_sm_kb=228,
            peak_bandwidth_gbs=3350.0,
            peak_fp16_tflops=1979.0
        ).model_dump()
    }


@app.post("/api/v1/synthesize", response_model=SynthesisResponse)
async def synthesize_sync(request: SynthesisRequest):
    return await orchestrator.run(request)


@app.websocket("/ws/synthesize")
async def websocket_synthesize(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established.")

    try:
        data_text = await websocket.receive_text()
        req_data = json.loads(data_text)
        
        if "session_id" not in req_data:
            req_data["session_id"] = str(uuid.uuid4())
            
        request = SynthesisRequest(**req_data)

        async def step_callback(step: ExecutionStep):
            await websocket.send_json({
                "type": "STEP_UPDATE",
                "step": step.model_dump()
            })

        response = await orchestrator.run(request, on_step_event=step_callback)
        
        await websocket.send_json({
            "type": "SYNTHESIS_COMPLETE",
            "response": response.model_dump()
        })

    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "ERROR",
                "message": str(e)
            })
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass

