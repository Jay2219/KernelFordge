"""
API Integration Tests for KernelForge FastAPI Endpoints.
"""

from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_root_status():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ONLINE"
    assert data["service"] == "KernelForge Gateway"
    assert data["mock_sandbox"] is True
    assert "models" in data


def test_presets_endpoint():
    response = client.get("/api/v1/presets")
    assert response.status_code == 200
    presets = response.json()
    assert "fused_rms_norm" in presets
    assert "fused_swiglu" in presets
    assert "fused_softmax" in presets


def test_hardware_endpoint():
    response = client.get("/api/v1/hardware")
    assert response.status_code == 200
    hardware = response.json()
    assert "A10G" in hardware
    assert hardware["A10G"]["peak_bandwidth_gbs"] == 600.0


def test_synthesize_endpoint():
    payload = {
        "session_id": "api-test-session",
        "operation_name": "fused_rms_norm",
        "pytorch_source": "def rms_norm_eager(x, w): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w",
        "input_tensor_shapes": [[32, 4096]],
        "dtype": "float16",
        "trigger_fault_demo": False
    }
    response = client.post("/api/v1/synthesize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "COMPLETED"
    assert data["benchmark"]["speedup_vs_eager"] > 1.5
    assert len(data["steps"]) >= 4


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "gpu_pool" in data
    assert "models" in data
    assert "meta_api" in data["models"]
    assert "baseten" in data["models"]
    # Verify honest reporting: with mock/unconfigured keys, models report not configured
    assert data["models"]["meta_api"]["configured"] is False
    assert data["models"]["baseten"]["configured"] is False



