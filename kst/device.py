"""Compute-device selection, shared by every model in the app.

NVIDIA GPU (CUDA) is the default everywhere; CPU is used only when no usable GPU is found.
  * PyTorch / Ultralytics (vehicle detector)  -> TORCH_DEVICE + QUANTIZE (FP16 on GPU)
  * ONNX Runtime (plate detector, plate OCR, RapidOCR, ONNX vehicle model) -> ort_providers()

Override with the KST_DEVICE environment variable: "cpu" forces CPU, "cuda:1" picks a GPU.
Check what was picked:  python -m kst.device
"""
from __future__ import annotations

import os

FORCE = os.environ.get("KST_DEVICE", "").strip().lower()
# Accepted: "" / "auto" / "gpu" / "cuda" (GPU if usable), "cuda:N" (that GPU), "cpu".
# Anything else is ignored (treated as auto) rather than breaking startup.
if FORCE in ("auto", "gpu", "cuda") or (FORCE and FORCE != "cpu" and not FORCE.startswith("cuda:")):
    FORCE = ""


def _cuda_index() -> int:
    if FORCE.startswith("cuda:"):
        try:
            return max(0, int(FORCE.split(":", 1)[1]))
        except ValueError:
            pass
    return 0


GPU_INDEX = _cuda_index()
FALLBACK_REASON: str | None = None      # why a GPU was wanted but CPU is used

# ---------------------------------------------------------------- PyTorch
USE_CUDA = False
GPU_NAME: str | None = None
GPU_MEM_GB: float | None = None
try:
    import torch

    if FORCE != "cpu" and torch.cuda.is_available():
        GPU_INDEX = min(GPU_INDEX, torch.cuda.device_count() - 1)
        dev = f"cuda:{GPU_INDEX}"
        # Smoke test: a CUDA build without kernels for this GPU (e.g. pre-CUDA-12.8 torch on an
        # RTX 50-series) passes is_available() but fails on the first real op.
        x = torch.ones(1, 3, 16, 16, device=dev)
        torch.nn.functional.conv2d(x, torch.ones(4, 3, 3, 3, device=dev)).sum().item()
        torch.nn.functional.conv2d(x.half(), torch.ones(4, 3, 3, 3, device=dev).half()).sum().item()
        USE_CUDA = True
        props = torch.cuda.get_device_properties(GPU_INDEX)
        GPU_NAME, GPU_MEM_GB = props.name, round(props.total_memory / 1024 ** 3, 1)
        torch.cuda.set_device(GPU_INDEX)
        torch.backends.cudnn.benchmark = True          # fastest conv kernels for a fixed input size
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif FORCE != "cpu":
        FALLBACK_REASON = ("PyTorch is a CPU-only build (install requirements-gpu.txt)"
                           if torch.version.cuda is None else "CUDA not available (driver / GPU?)")
except Exception as e:   # torch missing / broken CUDA install: fall back to CPU
    USE_CUDA = False
    FALLBACK_REASON = f"CUDA failed: {type(e).__name__}: {str(e).splitlines()[0][:120]}"

if not USE_CUDA:
    GPU_INDEX = 0                         # ONNX Runtime must not get a GPU index torch rejected

TORCH_DEVICE = f"cuda:{GPU_INDEX}" if USE_CUDA else "cpu"
HALF = USE_CUDA                                        # FP16 on GPU: ~2x faster, half the VRAM
QUANTIZE = 16 if HALF else None                        # Ultralytics precision arg (16 = FP16)


# ---------------------------------------------------------------- ONNX Runtime
_ort_providers = None


def ort_providers() -> list:
    """Providers for every onnxruntime session: CUDA first when available, CPU otherwise."""
    global _ort_providers
    if _ort_providers is not None:
        return _ort_providers
    import onnxruntime as ort

    if FORCE != "cpu" and hasattr(ort, "preload_dlls"):
        try:   # loads CUDA / cuDNN DLLs from the nvidia-* pip wheels (requirements-gpu.txt)
            ort.preload_dlls()
        except Exception:
            pass
    if FORCE != "cpu" and "CUDAExecutionProvider" in ort.get_available_providers():
        _ort_providers = [("CUDAExecutionProvider", {"device_id": GPU_INDEX}), "CPUExecutionProvider"]
    else:
        _ort_providers = ["CPUExecutionProvider"]
    return _ort_providers


def ort_uses_cuda() -> bool:
    p = ort_providers()[0]
    return (p[0] if isinstance(p, tuple) else p) == "CUDAExecutionProvider"


def session_providers(obj, depth: int = 4) -> list[str]:
    """Providers actually in use by the first onnxruntime session found inside `obj`.
    (A session silently falls back to CPU if the CUDA DLLs fail to load.)"""
    import onnxruntime as ort

    seen = set()

    def walk(o, d):
        if o is None or id(o) in seen or d < 0:
            return None
        seen.add(id(o))
        if isinstance(o, ort.InferenceSession):
            return o.get_providers()
        for v in getattr(o, "__dict__", {}).values():
            r = walk(v, d - 1)
            if r:
                return r
        return None

    return walk(obj, depth) or []


def force_ort_cpu() -> None:
    """Called when creating a CUDA session failed: every later session uses the CPU."""
    global _ort_providers
    _ort_providers = ["CPUExecutionProvider"]


def summary() -> str:
    if USE_CUDA:
        return f"GPU · {GPU_NAME} ({GPU_MEM_GB} GB) · {TORCH_DEVICE} · FP16"
    if FORCE == "cpu":
        return "CPU (forced by KST_DEVICE=cpu)"
    return f"CPU ({FALLBACK_REASON or 'no CUDA GPU found'})"


if __name__ == "__main__":
    print("PyTorch      :", summary())
    print("ONNX Runtime :", ort_providers())
