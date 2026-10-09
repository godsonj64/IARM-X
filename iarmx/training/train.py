import argparse
from contextlib import nullcontext
import math
import os
import time
from functools import partial
from pathlib import Path
import numpy as np
import yaml
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchdata.stateful_dataloader import StatefulDataLoader

from ..config import IARMXConfig
from ..model.model import IARMXForCausalLM
from ..data.tokenizer import load_tokenizer
from ..data.memmap import TokenWindows, WindowSampler
from ..data.pretrain import build_pretrain_dataset, collate_rows
from ..data.sft import build_sft_dataset, collate_sft
from ..utils.device import resolve_precision
from .checkpoint import (
    latest_checkpoint,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
    unwrap_model,
)


def warmup_cosine(step, warmup, progress, min_ratio):
    """Linear warmup over ``warmup`` steps, then cosine decay over ``progress`` in [0, 1]."""
    if step < warmup:
        return max(step / max(warmup, 1), 1e-8)
    p = min(1.0, max(0.0, progress))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))


def cosine_with_warmup(step, warmup, total, min_ratio):
    return warmup_cosine(step, warmup, (step - warmup) / max(total - warmup, 1), min_ratio)


def clear_stream_resume_point(dataset):
    """A Hugging Face IterableDataset re-applies a loaded resume state on every new
    pass of the same epoch; once the resumed pass ends, the next must start at 0."""
    try:
        from datasets import IterableDataset
    except ImportError:
        return
    if isinstance(dataset, IterableDataset):
        dataset.load_state_dict(None)


def setup_dist():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend)
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        return world, local_rank, device
    return 1, 0, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parameter_groups(model, weight_decay: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and "embed" not in name:
            decay.append(p)
        else:
            no_decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def shard_dataset(ds, world: int, rank: int, streaming: bool, seed: int):
    if world == 1:
        return ds, None
    if streaming:
        if not hasattr(ds, "shard"):
            raise TypeError("streaming dataset must provide .shard(num_shards, index)")
        try:
            ds = ds.shard(num_shards=world, index=rank)
        except TypeError:
            ds = ds.shard(world, rank)
        return ds, None
    return ds, DistributedSampler(
        ds, num_replicas=world, rank=rank, shuffle=True, seed=seed
    )


def build_loader(ds, collate, train_cfg, world, rank, streaming, seed, sampler=None):
    if sampler is None:
        ds, sampler = shard_dataset(ds, world, rank, streaming, seed)
    # A drop-in DataLoader whose state_dict() records exactly which samples were
    # consumed, including inside worker processes and streaming datasets.
    loader = StatefulDataLoader(
        ds,
        batch_size=train_cfg["micro_batch_size"],
        collate_fn=collate,
        num_workers=train_cfg.get("num_workers", 0),
        sampler=sampler,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, sampler


def capture_data_state(loader, sampler_epoch, memmap_position, world):
    """Snapshot every rank's data position and RNG streams. A collective when world > 1.

    RNG states are stored as numpy arrays so ``torch.load(map_location=...)``
    cannot move them off the CPU.
    """
    local = {
        "rng_cpu": torch.get_rng_state().numpy(),
        "rng_cuda": torch.cuda.get_rng_state().numpy() if torch.cuda.is_available() else None,
    }
    if memmap_position is None:
        local["loader"] = loader.state_dict()
        local["epoch"] = sampler_epoch
        local["num_workers"] = loader.num_workers
    ranks = [local]
    if world > 1:
        ranks = [None] * world
        torch.distributed.all_gather_object(ranks, local)
    state = {"world": world, "ranks": ranks}
    if memmap_position is not None:
        state["memmap"] = memmap_position
    return state


def restore_rng(local):
    torch.set_rng_state(torch.from_numpy(np.array(local["rng_cpu"], dtype=np.uint8)))
    if torch.cuda.is_available() and local.get("rng_cuda") is not None:
        torch.cuda.set_rng_state(torch.from_numpy(np.array(local["rng_cuda"], dtype=np.uint8)))


def apply_overrides(spec: dict, overrides) -> dict:
    """Apply ``section.key=value`` overrides; values are parsed as YAML.

    ``training.target_tokens=100000000`` sets an int, ``training.epochs=null``
    removes a setting, ``model.gradient_checkpointing=false`` sets a bool.
    """
    for item in overrides or ():
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"override must look like section.key=value, got {item!r}")
        value = yaml.safe_load(raw)
        if isinstance(value, str):
            try:  # YAML reads 1e8 as a string; accept scientific notation for numbers
                number = float(value)
                value = int(number) if number.is_integer() else number
            except ValueError:
                pass
        node = spec
        *parents, leaf = key.strip().split(".")
        for name in parents:
            node = node.setdefault(name, {})
        node[leaf] = value
    return spec


def allreduce_int(value: int, device: torch.device, world: int) -> int:
    if world == 1:
        return int(value)
    t = torch.tensor([value], device=device, dtype=torch.long)
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    return int(t.item())


def estimate_total_steps(train_cfg, data_cfg, world: int, dataset_len=None) -> int:
    """Estimate scheduler horizon independently of GPU count for budgeted runs."""
    micro = int(train_cfg.get("micro_batch_size", 1))
    accum = int(train_cfg.get("grad_accum", 1))
    global_sequences = max(1, micro * accum * world)

    if train_cfg.get("target_tokens") is not None:
        seq_len = int(data_cfg["seq_len"])
        per_step = max(1, global_sequences * seq_len)
        return max(1, math.ceil(int(train_cfg["target_tokens"]) / per_step))
    if train_cfg.get("epochs") is not None:
        if dataset_len is None:
            raise ValueError("epochs requires a finite non-streaming dataset")
        target_examples = int(dataset_len) * float(train_cfg["epochs"])
        return max(1, math.ceil(target_examples / global_sequences))
    if train_cfg.get("max_steps") is not None:
        return max(1, int(train_cfg["max_steps"]))
    raise ValueError("training requires target_tokens, epochs, or max_steps")


def should_stop(step, tokens_seen, examples_seen, train_cfg, dataset_len=None):
    if train_cfg.get("target_tokens") is not None and tokens_seen >= int(train_cfg["target_tokens"]):
        return True
    if train_cfg.get("epochs") is not None:
        if dataset_len is None:
            raise ValueError("epochs requires dataset_len")
        if examples_seen >= math.ceil(float(train_cfg["epochs"]) * dataset_len):
            return True
    if train_cfg.get("max_steps") is not None and step >= int(train_cfg["max_steps"]):
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--resume",
        help="training checkpoint .pt to resume, or 'auto' for the newest checkpoint in "
        "output_dir (starts fresh when there is none)",
    )
    ap.add_argument(
        "--init-from",
        help="model directory from save_pretrained(); starts a new optimizer/scheduler",
    )
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override a config value, e.g. --set training.target_tokens=100000000 (repeatable)",
    )
    args = ap.parse_args()

    spec = apply_overrides(yaml.safe_load(Path(args.config).read_text()), args.set)
    cfg = IARMXConfig.from_dict(spec["model"])
    train = spec["training"]
    data_cfg = spec["data"]
    world, rank, device = setup_dist()
    seed = train.get("seed", 42)
    torch.manual_seed(seed + rank)
    if args.resume == "auto":
        found = latest_checkpoint(train.get("output_dir", "checkpoints"))
        args.resume = str(found) if found is not None else None
        if rank == 0:
            print(f"--resume auto: {args.resume or 'no checkpoint found, starting fresh'}", flush=True)

    tok = load_tokenizer(data_cfg.get("tokenizer", "gpt2"))
    cfg.vocab_size = len(tok)

    stage = data_cfg.get("stage", "pretrain")
    memmap = stage == "pretrain" and data_cfg.get("dataset") == "memmap"
    streaming = (
        bool(data_cfg.get("streaming", False)) if stage == "pretrain" and not memmap else False
    )
    data_files = data_cfg.get("data_files")
    if stage == "sft":
        ds = build_sft_dataset(
            tok,
            data_cfg.get("dataset", "HuggingFaceH4/ultrachat_200k"),
            data_cfg.get("split", "train_sft"),
            data_cfg["seq_len"],
            data_files=data_files,
        )
        collate = partial(collate_sft, pad_id=tok.pad_token_id)
    elif memmap:
        ds = TokenWindows(data_cfg["path"], data_cfg["seq_len"])
        if ds.vocab_size > len(tok):
            raise ValueError(
                f"token shards use a vocabulary of {ds.vocab_size} but the tokenizer has {len(tok)}"
            )
        collate = None  # windows are already fixed-length tensors
        target = train.get("target_tokens")
        if target is not None and int(target) > ds.n_windows * ds.seq_len and rank == 0:
            print(
                f"warning: target_tokens={int(target):,} exceeds the {ds.n_windows * ds.seq_len:,} "
                f"tokens in {data_cfg['path']}; the data will be repeated "
                f"{int(target) / (ds.n_windows * ds.seq_len):.1f} times",
                flush=True,
            )
    else:
        ds = build_pretrain_dataset(
            tok,
            data_cfg.get("dataset", "HuggingFaceFW/fineweb-edu"),
            data_cfg.get("dataset_config", "sample-10BT"),
            data_cfg.get("split", "train"),
            data_cfg["seq_len"],
            streaming,
            data_files=data_files,
            pack=data_cfg.get("pack", True),
        )
        collate = collate_rows

    dataset_len = None if streaming else len(ds)

    if args.resume and args.init_from:
        raise ValueError("use either --resume or --init-from, not both")
    # A config-level init_from only seeds a fresh stage; a resumed checkpoint
    # already holds the weights.
    init_from = args.init_from or (None if args.resume else train.get("init_from"))
    if init_from:
        base_model = IARMXForCausalLM.from_pretrained(init_from, device=device)
        loaded = base_model.cfg.to_dict()
        requested = cfg.to_dict()
        # max_seq_len can safely differ only if RoPE buffers are regenerated; this
        # reference implementation keeps configs identical to avoid silent drift.
        if loaded != requested:
            mismatches = {k: (loaded.get(k), requested.get(k)) for k in requested if loaded.get(k) != requested.get(k)}
            raise ValueError(f"init_from config mismatch: {mismatches}")
    else:
        base_model = IARMXForCausalLM(cfg).to(device)

    optimizer = AdamW(
        parameter_groups(base_model, train.get("weight_decay", 0.1)),
        lr=train["lr"],
        betas=tuple(train.get("betas", [0.9, 0.95])),
    )
    total = estimate_total_steps(train, data_cfg, world, dataset_len)
    min_ratio = train.get("min_lr", train["lr"] * 0.1) / train["lr"]
    warmup = int(train.get("warmup_steps", 0))
    target_tokens = train.get("target_tokens")
    # With a token budget the decay follows tokens, not steps: tokens per step vary
    # with document packing and with the number of GPUs, so a step-based horizon
    # would end the run before (or after) the schedule does.
    schedule = {"tokens": 0, "warmup_tokens": 0 if warmup == 0 else None}

    def lr_lambda(s):
        if target_tokens is None or s < warmup:
            return cosine_with_warmup(s, warmup, total, min_ratio)
        start = schedule["warmup_tokens"] or 0
        progress = (schedule["tokens"] - start) / max(int(target_tokens) - start, 1)
        return warmup_cosine(s, warmup, progress, min_ratio)

    sched = LambdaLR(optimizer, lr_lambda)

    step = 0
    tokens_seen = 0
    examples_seen = 0
    data_state = None
    resumed_extra = {}
    if args.resume:
        ckpt = load_checkpoint(args.resume, base_model, optimizer, sched, map_location=device)
        step = int(ckpt.get("step", 0))
        extra = resumed_extra = ckpt.get("extra") or {}
        tokens_seen = int(extra.get("tokens_seen", 0))
        examples_seen = int(extra.get("examples_seen", 0))
        data_state = extra.get("data")
        schedule["tokens"] = tokens_seen
        if extra.get("warmup_tokens") is not None:
            schedule["warmup_tokens"] = int(extra["warmup_tokens"])
        elif step >= warmup:  # checkpoint predates token-based decay
            schedule["warmup_tokens"] = tokens_seen * warmup // max(step, 1)
        saved_batch = extra.get("global_batch")
        global_batch = int(train["micro_batch_size"]) * int(train.get("grad_accum", 1)) * world
        if saved_batch is not None and saved_batch != global_batch and rank == 0:
            print(
                f"warning: global batch changed from {saved_batch} to {global_batch} sequences "
                "per step (micro_batch_size x grad_accum x GPUs); set grad_accum to keep it",
                flush=True,
            )
        if rank == 0:
            print(
                f"resumed={args.resume} step={step} tokens={tokens_seen} examples={examples_seen}",
                flush=True,
            )

    # The loader is built after resume so it can start at the saved data position.
    sampler = None
    memmap_spec = None
    memmap_base = 0
    if memmap:
        # Every row is one window, so the global data position is examples_seen.
        memmap_spec = {
            "seq_len": ds.seq_len,
            "n_windows": ds.n_windows,
            "shuffle": bool(data_cfg.get("shuffle", True)),
            "seed": int(seed),
        }
        start = 0
        saved = (data_state or {}).get("memmap")
        if saved is not None:
            changed = {k: (saved.get(k), v) for k, v in memmap_spec.items() if saved.get(k) != v}
            if changed:
                raise ValueError(
                    f"cannot resume the token-window order; changed (saved, now): {changed}. "
                    "Start a new stage with --init-from instead."
                )
            start = int(saved["position"])
        elif args.resume and rank == 0:
            print("checkpoint has no token-window position; data starts at window 0", flush=True)
        # examples_seen may include rows from an earlier non-memmap stage; the
        # saved position must count only windows drawn from this sampler.
        memmap_base = examples_seen - start
        sampler = WindowSampler(
            ds.n_windows, start, world, rank, memmap_spec["shuffle"], memmap_spec["seed"]
        )
    loader, sampler = build_loader(ds, collate, train, world, rank, streaming, seed, sampler)

    sampler_epoch = 0
    rng_state = None
    saved_world = None if data_state is None else data_state.get("world")
    if data_state is not None and not memmap and "loader" not in data_state["ranks"][0]:
        if rank == 0:
            print(
                "checkpoint came from a token-shard (memmap) run; this run's data starts "
                "from the beginning",
                flush=True,
            )
        if saved_world == world:
            rng_state = data_state["ranks"][rank]
    elif data_state is not None and saved_world == world:
        local = data_state["ranks"][rank]
        rng_state = local
        if not memmap:
            saved_workers = local.get("num_workers", loader.num_workers)
            if saved_workers != loader.num_workers:
                raise ValueError(
                    f"checkpoint data-loader state was saved with num_workers={saved_workers}, "
                    f"now {loader.num_workers}: set num_workers back to {saved_workers} to resume "
                    "exactly, or start a new stage with --init-from."
                )
            sampler_epoch = int(local["epoch"])
            if sampler is not None:
                sampler.set_epoch(sampler_epoch)
            loader.load_state_dict(local["loader"])
    elif data_state is not None and not memmap:
        raise ValueError(
            f"checkpoint data state was saved with world size {data_state.get('world')}, "
            f"now {world}: exact resume of a streaming or map-style dataset needs the same "
            "world size. Use dataset: memmap to resume across world sizes, or start a new "
            "stage with --init-from."
        )
    elif data_state is not None and rank == 0:
        print(
            f"world size changed ({data_state.get('world')} -> {world}): token-window position "
            "restored exactly; RNG streams are reseeded",
            flush=True,
        )
    elif args.resume and not memmap and rank == 0:
        print(
            "checkpoint has no data-loader state; the data stream restarts from the beginning",
            flush=True,
        )

    def data_extra():
        position = None
        if memmap_spec is not None:
            position = {**memmap_spec, "position": examples_seen - memmap_base}
        return {
            "tokens_seen": tokens_seen,
            "examples_seen": examples_seen,
            "warmup_tokens": schedule["warmup_tokens"],
            "global_batch": int(train["micro_batch_size"]) * grad_accum * world,
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "data": capture_data_state(loader, sampler_epoch, position, world),
        }

    model = base_model
    if train.get("compile", False) and hasattr(torch, "compile"):
        model = torch.compile(model)
    if world > 1:
        ddp_kwargs = {"device_ids": [device.index]} if device.type == "cuda" else {}
        model = torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)

    requested_precision = str(train.get("precision", "bf16"))
    precision = resolve_precision(requested_precision, device)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
    # fp16 needs loss scaling; GPUs without native bf16 (T4, V100, P100) use it.
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    if scaler.is_enabled() and resumed_extra.get("scaler"):
        scaler.load_state_dict(resumed_extra["scaler"])
    if rank == 0 and device.type == "cuda" and requested_precision not in ("auto", precision):
        print(
            f"{requested_precision} is not natively supported on this GPU; training in {precision} "
            "with loss scaling",
            flush=True,
        )
    grad_accum = int(train.get("grad_accum", 1))
    save_every = int(train.get("save_every", 1000))
    keep_last = int(train.get("keep_last", 0))
    out_dir = Path(train.get("output_dir", "checkpoints"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        print(
            f"params={base_model.num_parameters():,} stage={stage} world={world} "
            f"precision={precision} scheduler_steps={total} "
            f"target_tokens={train.get('target_tokens')} epochs={train.get('epochs')}",
            flush=True,
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(loader)
    if rng_state is not None:
        # After iter(): creating an iterator draws a base seed from the global RNG,
        # which the uninterrupted run did before the checkpointed step, not after.
        restore_rng(rng_state)
    log_time, log_tokens = time.perf_counter(), tokens_seen
    while not should_stop(step, tokens_seen, examples_seen, train, dataset_len):
        running = 0.0
        local_tokens = 0
        local_examples = 0
        for micro in range(grad_accum):
            try:
                batch = next(iterator)
            except StopIteration:
                sampler_epoch += 1
                if sampler is not None:
                    sampler.set_epoch(sampler_epoch)
                clear_stream_resume_point(loader.dataset)
                iterator = iter(loader)
                batch = next(iterator)

            # Count on the CPU copy: a .item() on a device tensor would block the host
            # from queueing the next kernels on every micro-step.
            local_tokens += int((batch["labels"] != -100).sum())
            local_examples += int(batch["input_ids"].size(0))
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            sync_ctx = (
                model.no_sync()
                if world > 1 and micro < grad_accum - 1 and hasattr(model, "no_sync")
                else nullcontext()
            )
            with sync_ctx:
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype or torch.float32,
                    enabled=amp_dtype is not None,
                ):
                    out = model(**batch)
                    loss = out.loss / grad_accum
                scaler.scale(loss).backward()
            running = running + loss.detach()  # synchronizes only when logged

        scaler.unscale_(optimizer)  # no-op unless fp16
        torch.nn.utils.clip_grad_norm_(model.parameters(), train.get("grad_clip", 1.0))
        scaler.step(optimizer)  # skips the update if fp16 gradients overflowed
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        tokens_seen += allreduce_int(local_tokens, device, world)
        examples_seen += allreduce_int(local_examples, device, world)
        schedule["tokens"] = tokens_seen
        if schedule["warmup_tokens"] is None and step >= warmup:
            schedule["warmup_tokens"] = tokens_seen
        sched.step()  # sets the learning rate for the next step

        if rank == 0 and step % int(train.get("log_every", 10)) == 0:
            now = time.perf_counter()
            rate = (tokens_seen - log_tokens) / max(now - log_time, 1e-9)
            log_time, log_tokens = now, tokens_seen
            print(
                f"step={step} loss={running:.4f} lr={sched.get_last_lr()[0]:.3e} "
                f"tokens={tokens_seen:,} examples={examples_seen:,} tok/s={rate:,.0f}",
                flush=True,
            )
        if step % save_every == 0:
            extra = data_extra()  # collective: every rank must reach it
            if rank == 0:
                save_checkpoint(out_dir / f"step-{step}.pt", model, optimizer, sched, step, extra)
                prune_checkpoints(out_dir, keep_last)

    extra = data_extra()
    if rank == 0:
        unwrap_model(model).save_pretrained(out_dir / "final")
        save_checkpoint(out_dir / "last.pt", model, optimizer, sched, step, extra)
        print(
            f"finished step={step} tokens={tokens_seen:,} examples={examples_seen:,}",
            flush=True,
        )
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
