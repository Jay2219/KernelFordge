"""
KernelForge FastAPI Gateway and Real-Time WebSocket Orchestrator.
"""

import json
import uuid
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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

