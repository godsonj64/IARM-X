import torch

PRECISIONS = ("auto", "bf16", "fp16", "fp32")


def native_bf16(device) -> bool:
    """True when a CUDA device has bf16 tensor cores (compute capability >= 8.0).

    ``torch.cuda.is_bf16_supported()`` also reports emulated support, which is
    true on a T4 but very slow, so it is not used here.
    """
    device = torch.device(device)
    return device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8


def resolve_precision(precision: str, device) -> str:
    """Map a configured precision to the one actually used on ``device``.

    Mixed precision is CUDA-only in the trainer, so CPUs run fp32. ``auto`` and
    ``bf16`` fall back to fp16 (with loss scaling) on GPUs without native bf16,
    such as the T4 in free Colab and Kaggle notebooks.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    device = torch.device(device)
    if device.type != "cuda":
        return "fp32"
    if precision in ("auto", "bf16"):
        return "bf16" if native_bf16(device) else "fp16"
    return precision


def inference_dtype(device) -> torch.dtype:
    device = torch.device(device)
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if native_bf16(device) else torch.float16
