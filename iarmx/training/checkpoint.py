import os
import re
from pathlib import Path
import torch

_STEP_FILE = re.compile(r"step-(\d+)\.pt")


def unwrap_model(model):
    while True:
        if hasattr(model, "module"):
            model = model.module
            continue
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
            continue
        return model


def save_checkpoint(path, model, optimizer, scheduler, step, extra=None):
    """Write atomically: a preemption mid-write leaves the previous checkpoint intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    target = unwrap_model(model)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "model": target.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "step": step,
            "extra": extra or {},
        },
        tmp,
    )
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def step_checkpoints(out_dir) -> list[Path]:
    """``step-N.pt`` files in ``out_dir``, oldest first."""
    found = []
    for p in Path(out_dir).glob("step-*.pt"):
        m = _STEP_FILE.fullmatch(p.name)
        if m:
            found.append((int(m.group(1)), p))
    return [p for _, p in sorted(found)]


def latest_checkpoint(out_dir) -> Path | None:
    """The checkpoint a rerun should resume from: ``last.pt`` (a finished run)
    or else the highest ``step-N.pt``; ``None`` when there is nothing to resume."""
    last = Path(out_dir) / "last.pt"
    if last.exists():
        return last
    steps = step_checkpoints(out_dir)
    return steps[-1] if steps else None


def prune_checkpoints(out_dir, keep_last: int):
    """Delete all but the newest ``keep_last`` step checkpoints (0 keeps everything)."""
    if keep_last > 0:
        for p in step_checkpoints(out_dir)[:-keep_last]:
            p.unlink(missing_ok=True)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # older PyTorch
        ckpt = torch.load(path, map_location=map_location)
    target = unwrap_model(model)
    target.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt
