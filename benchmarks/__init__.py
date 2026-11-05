from .fused_rms_norm import PRESET_RMS_NORM
from .fused_swiglu import PRESET_SWIGLU
from .fused_softmax import PRESET_SOFTMAX
from .fused_resadd_rmsnorm import PRESET_RESADD_RMSNORM

ALL_PRESETS = {
    "fused_rms_norm": PRESET_RMS_NORM,
    "fused_swiglu": PRESET_SWIGLU,
    "fused_softmax": PRESET_SOFTMAX,
    "fused_resadd_rmsnorm": PRESET_RESADD_RMSNORM
}

__all__ = [
    "PRESET_RMS_NORM",
    "PRESET_SWIGLU",
    "PRESET_SOFTMAX",
    "PRESET_RESADD_RMSNORM",
    "ALL_PRESETS"
]

