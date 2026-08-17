import argparse
from contextlib import nullcontext
import math
import os
from functools import partial
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from ..config import IARMXConfig
from ..model.model import IARMXForCausalLM
from ..data.tokenizer import load_tokenizer
from ..data.pretrain import build_pretrain_dataset, collate_pretrain
from ..data.sft import build_sft_dataset, collate_sft
from .checkpoint import save_checkpoint, load_checkpoint, unwrap_model


def cosine_with_warmup(step, warmup, total, min_ratio):
    if step < warmup:
        return max(step / max(warmup, 1), 1e-8)
    p = min(1.0, (step - warmup) / max(total - warmup, 1))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))


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


def build_loader(ds, collate, train_cfg, world, rank, streaming, seed):
    ds, sampler = shard_dataset(ds, world, rank, streaming, seed)
    loader = DataLoader(
        ds,
        batch_size=train_cfg["micro_batch_size"],
        collate_fn=collate,
        num_workers=train_cfg.get("num_workers", 0),
        sampler=sampler,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, sampler


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
    ap.add_argument("--resume", help="training checkpoint .pt to resume")
    ap.add_argument(
        "--init-from",
        help="model directory from save_pretrained(); starts a new optimizer/scheduler",
    )
    args = ap.parse_args()

    spec = yaml.safe_load(Path(args.config).read_text())
    cfg = IARMXConfig.from_dict(spec["model"])
    train = spec["training"]
    data_cfg = spec["data"]
    world, rank, device = setup_dist()
    seed = train.get("seed", 42)
    torch.manual_seed(seed + rank)

    tok = load_tokenizer(data_cfg.get("tokenizer", "gpt2"))
    cfg.vocab_size = len(tok)

    stage = data_cfg.get("stage", "pretrain")
    streaming = bool(data_cfg.get("streaming", False)) if stage == "pretrain" else False
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
    else:
        ds = build_pretrain_dataset(
            tok,
            data_cfg.get("dataset", "HuggingFaceFW/fineweb-edu"),
            data_cfg.get("dataset_config", "sample-10BT"),
            data_cfg.get("split", "train"),
            data_cfg["seq_len"],
            streaming,
            data_files=data_files,
        )
        collate = partial(
            collate_pretrain,
            pad_id=tok.pad_token_id,
            seq_len=data_cfg["seq_len"],
            pack=data_cfg.get("pack", True),
        )

    dataset_len = None if streaming else len(ds)
    loader, sampler = build_loader(ds, collate, train, world, rank, streaming, seed)

    init_from = args.init_from or train.get("init_from")
    if args.resume and init_from:
        raise ValueError("use either --resume or --init-from, not both")
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
    sched = LambdaLR(
        optimizer,
        lambda s: cosine_with_warmup(
            s, train.get("warmup_steps", 0), total, min_ratio
        ),
    )

    step = 0
    tokens_seen = 0
    examples_seen = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume, base_model, optimizer, sched, map_location=device)
        step = int(ckpt.get("step", 0))
        extra = ckpt.get("extra") or {}
        tokens_seen = int(extra.get("tokens_seen", 0))
        examples_seen = int(extra.get("examples_seen", 0))
        if rank == 0:
            print(
                f"resumed={args.resume} step={step} tokens={tokens_seen} examples={examples_seen}",
                flush=True,
            )

    model = base_model
    if train.get("compile", False) and hasattr(torch, "compile"):
        model = torch.compile(model)
    if world > 1:
        ddp_kwargs = {"device_ids": [device.index]} if device.type == "cuda" else {}
        model = torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)

    amp_dtype = (
        torch.bfloat16
        if train.get("precision", "bf16") == "bf16"
        else torch.float16
    )
    grad_accum = int(train.get("grad_accum", 1))
    save_every = int(train.get("save_every", 1000))
    out_dir = Path(train.get("output_dir", "checkpoints"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        print(
            f"params={base_model.num_parameters():,} stage={stage} world={world} "
            f"scheduler_steps={total} target_tokens={train.get('target_tokens')} "
            f"epochs={train.get('epochs')}",
            flush=True,
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(loader)
    sampler_epoch = 0
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
                iterator = iter(loader)
                batch = next(iterator)

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            local_tokens += int((batch["labels"] != -100).sum().item())
            local_examples += int(batch["input_ids"].size(0))
            sync_ctx = (
                model.no_sync()
                if world > 1 and micro < grad_accum - 1 and hasattr(model, "no_sync")
                else nullcontext()
            )
            with sync_ctx:
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=device.type == "cuda",
                ):
                    out = model(**batch)
                    loss = out.loss / grad_accum
                loss.backward()
            running += float(loss.detach())

        torch.nn.utils.clip_grad_norm_(model.parameters(), train.get("grad_clip", 1.0))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        sched.step()
        step += 1

        tokens_seen += allreduce_int(local_tokens, device, world)
        examples_seen += allreduce_int(local_examples, device, world)

        if rank == 0 and step % int(train.get("log_every", 10)) == 0:
            print(
                f"step={step} loss={running:.4f} lr={sched.get_last_lr()[0]:.3e} "
                f"tokens={tokens_seen:,} examples={examples_seen:,}",
                flush=True,
            )
        if rank == 0 and step % save_every == 0:
            save_checkpoint(
                out_dir / f"step-{step}.pt",
                model,
                optimizer,
                sched,
                step,
                extra={"tokens_seen": tokens_seen, "examples_seen": examples_seen},
            )

    if rank == 0:
        unwrap_model(model).save_pretrained(out_dir / "final")
        save_checkpoint(
            out_dir / "last.pt",
            model,
            optimizer,
            sched,
            step,
            extra={"tokens_seen": tokens_seen, "examples_seen": examples_seen},
        )
        print(
            f"finished step={step} tokens={tokens_seen:,} examples={examples_seen:,}",
            flush=True,
        )
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
