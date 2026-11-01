from .fused_rms_norm import PRESET_RMS_NORM
from .fused_swiglu import PRESET_SWIGLU
from .fused_softmax import PRESET_SOFTMAX

ALL_PRESETS = {
    "fused_rms_norm": PRESET_RMS_NORM,
    "fused_swiglu": PRESET_SWIGLU,
    "fused_softmax": PRESET_SOFTMAX
}

__all__ = ["PRESET_RMS_NORM", "PRESET_SWIGLU", "PRESET_SOFTMAX", "ALL_PRESETS"]

