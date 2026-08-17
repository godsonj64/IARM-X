from pathlib import Path
import torch


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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    target = unwrap_model(model)
    torch.save(
        {
            "model": target.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "step": step,
            "extra": extra or {},
        },
        path,
    )


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
