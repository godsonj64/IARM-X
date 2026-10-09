"""End-to-end resume tests: a run resumed from a step-k checkpoint must finish
with the same weights as the uninterrupted run (bit-exact with the same world
size, dropout included)."""

import os
import random
import socket
import sys

import pytest
import torch
import torch.multiprocessing as mp
import yaml
from safetensors.torch import load_file

import iarmx.training.train as train_mod
from iarmx.data.memmap import TokenShardWriter

VOCAB = 97


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __len__(self):
        return VOCAB

    def __call__(self, texts, add_special_tokens=False, split_special_tokens=False):
        return {"input_ids": [[1 + ord(c) % (VOCAB - 1) for c in t] for t in texts]}


def make_texts(n=40, seed=0):
    rng = random.Random(seed)
    letters = "abcdefghijklmnopqrstuvwxyz"
    return ["".join(rng.choice(letters) for _ in range(rng.randrange(3, 120))) for _ in range(n)]


def write_shards(path):
    tok = FakeTokenizer()
    writer = TokenShardWriter(path, vocab_size=VOCAB, eos_id=0, shard_tokens=500)
    for ids in tok(make_texts())["input_ids"]:
        writer.add_document(ids)
    writer.close()


def _fake_stream(shards, n):
    # Module level so DataLoader workers can pickle it under spawn/forkserver too.
    texts = make_texts(n)
    for s in shards:
        for i in range(s, len(texts), 4):
            yield {"text": texts[i], "id": i}


def fake_load_dataset(name, config=None, split=None, streaming=False, data_files=None):
    """``name`` is "fake" (40 documents) or "fake:N" (N documents)."""
    from datasets import Dataset, IterableDataset

    n = int(name.split(":")[1]) if ":" in name else 40
    if not streaming:
        return Dataset.from_list([{"text": t, "id": i} for i, t in enumerate(make_texts(n))])
    return IterableDataset.from_generator(
        _fake_stream, gen_kwargs={"shards": [0, 1, 2, 3], "n": n}
    )


def write_config(path, out_dir, data, steps=6, save_every=3, dropout=0.1, workers=0,
                 micro=2, accum=2, **training):
    spec = {
        "model": dict(vocab_size=VOCAB, dim=32, n_layers=2, n_heads=4, ffn_hidden=64,
                      max_seq_len=64, n_operators=2, operator_rank=4, fast_memory_rank=4,
                      attention_every=2, dropout=dropout, scan_chunk_size=8),
        "data": dict(stage="pretrain", seq_len=16, **data),
        "training": dict(seed=0, micro_batch_size=micro, grad_accum=accum, max_steps=steps,
                         lr=3e-3, min_lr=3e-4, warmup_steps=2, compile=False, log_every=100,
                         save_every=save_every, num_workers=workers, output_dir=str(out_dir),
                         **training),
    }
    path.write_text(yaml.safe_dump(spec))
    return path


def load_ckpt(path):
    return torch.load(path, weights_only=False)


def run(config, resume=None, overrides=()):
    import datasets

    saved = train_mod.load_tokenizer, datasets.load_dataset, sys.argv
    train_mod.load_tokenizer = lambda name: FakeTokenizer()
    datasets.load_dataset = fake_load_dataset
    sys.argv = ["train", "--config", str(config)] + (["--resume", str(resume)] if resume else [])
    for item in overrides:
        sys.argv += ["--set", item]
    try:
        train_mod.main()
    finally:
        train_mod.load_tokenizer, datasets.load_dataset, sys.argv = saved


def final_weights(out_dir):
    return load_file(str(out_dir / "final" / "model.safetensors"))


def assert_same_weights(a, b, exact=True):
    assert a.keys() == b.keys()
    for k in a:
        if exact:
            assert torch.equal(a[k], b[k]), k
        else:
            assert torch.allclose(a[k], b[k], atol=1e-5, rtol=1e-4), k


DATA = {
    "memmap": lambda tmp: {"dataset": "memmap", "path": str(tmp / "tokens")},
    "streaming": lambda tmp: {"dataset": "fake", "streaming": True},
    # 1 document packs into 7 rows = 1.75 steps per epoch (2 on two ranks), so the
    # step-3 checkpoint falls inside the second epoch; the tests assert this.
    "map-style, epoch wrap": lambda tmp: {"dataset": "fake:1", "streaming": False},
}


def resume_and_compare(tmp_path, data, resume_step=3, **kw):
    """Run a to completion, resume b from a's step checkpoint, compare finals."""
    run(write_config(tmp_path / "a.yaml", tmp_path / "a", data, **kw))
    run(write_config(tmp_path / "b.yaml", tmp_path / "b", data, **kw),
        resume=tmp_path / "a" / f"step-{resume_step}.pt")
    # b really resumed: it never re-ran (and so never re-saved) step resume_step.
    assert not (tmp_path / "b" / f"step-{resume_step}.pt").exists()
    assert_same_weights(final_weights(tmp_path / "a"), final_weights(tmp_path / "b"))


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("kind", list(DATA))
def test_resume_matches_uninterrupted_run(tmp_path, kind, workers):
    write_shards(tmp_path / "tokens")
    data = DATA[kind](tmp_path)
    if kind == "map-style, epoch wrap":
        run(write_config(tmp_path / "probe.yaml", tmp_path / "probe", data, workers=workers))
        assert load_ckpt(tmp_path / "probe" / "step-3.pt")["extra"]["data"]["ranks"][0]["epoch"] >= 1
    resume_and_compare(tmp_path, data, workers=workers)


def test_streaming_without_workers_starts_later_passes_at_the_beginning(tmp_path):
    # 8 documents = ~8 steps per pass, so the stream wraps twice after the resume.
    resume_and_compare(tmp_path, {"dataset": "fake:8", "streaming": True}, steps=20, workers=0)


def test_switching_from_streaming_to_memmap_saves_only_memmap_windows(tmp_path):
    write_shards(tmp_path / "tokens")
    run(write_config(tmp_path / "a.yaml", tmp_path / "a", DATA["streaming"](tmp_path)))
    run(write_config(tmp_path / "b.yaml", tmp_path / "b", DATA["memmap"](tmp_path)),
        resume=tmp_path / "a" / "step-3.pt")
    ckpt = load_ckpt(tmp_path / "b" / "step-6.pt")
    # Steps 4-6 drew 3 x (2 x 2) = 12 windows; examples_seen also counts stage a's rows.
    assert ckpt["extra"]["data"]["memmap"]["position"] == 12
    assert ckpt["extra"]["examples_seen"] > 12


def test_changing_num_workers_is_refused_with_a_clear_error(tmp_path):
    data = DATA["streaming"](tmp_path)
    run(write_config(tmp_path / "a.yaml", tmp_path / "a", data, workers=0))
    with pytest.raises(ValueError, match="num_workers"):
        run(write_config(tmp_path / "b.yaml", tmp_path / "b", data, workers=2),
            resume=tmp_path / "a" / "step-3.pt")


def test_resume_auto_continues_a_stage_that_has_a_config_init_from(tmp_path):
    write_shards(tmp_path / "tokens")
    run(write_config(tmp_path / "a.yaml", tmp_path / "a", DATA["memmap"](tmp_path)))
    stage2 = write_config(tmp_path / "b.yaml", tmp_path / "b", DATA["streaming"](tmp_path),
                          init_from=str(tmp_path / "a" / "final"), keep_last=2)
    run(stage2)
    reference = final_weights(tmp_path / "b")
    # Simulate a preemption right after step 3.
    for name in ("step-6.pt", "last.pt"):
        (tmp_path / "b" / name).unlink()
    run(stage2, resume="auto")
    assert_same_weights(reference, final_weights(tmp_path / "b"))
    run(stage2, resume="auto")  # a finished stage resumes from last.pt and stops at once
    assert_same_weights(reference, final_weights(tmp_path / "b"))


def test_checkpoints_are_atomic_and_pruned(tmp_path):
    write_shards(tmp_path / "tokens")
    run(write_config(tmp_path / "a.yaml", tmp_path / "a", DATA["memmap"](tmp_path),
                     save_every=2, keep_last=1))
    assert sorted(p.name for p in (tmp_path / "a").glob("*.pt*")) == ["last.pt", "step-6.pt"]


def test_token_budget_decays_to_min_lr_exactly_at_the_target(tmp_path):
    write_shards(tmp_path / "tokens")
    tokens_per_step = 2 * 2 * 16
    cfg = write_config(tmp_path / "a.yaml", tmp_path / "a", DATA["memmap"](tmp_path), steps=None)
    spec = yaml.safe_load(cfg.read_text())
    spec["training"].pop("max_steps")
    spec["training"]["target_tokens"] = 7 * tokens_per_step
    cfg.write_text(yaml.safe_dump(spec))
    run(cfg)
    last = load_ckpt(tmp_path / "a" / "last.pt")
    assert last["step"] == 7
    assert last["optimizer"]["param_groups"][0]["lr"] == pytest.approx(3e-4)


def test_resume_without_data_state_diverges(tmp_path):
    # Control: the equality above is not vacuous. Dropping the saved data state
    # (the old behavior) replays data from the start and changes the result.
    write_shards(tmp_path / "tokens")
    data = DATA["memmap"](tmp_path)
    run(write_config(tmp_path / "a.yaml", tmp_path / "a", data))
    ckpt = torch.load(tmp_path / "a" / "step-3.pt", weights_only=False)
    del ckpt["extra"]["data"]
    torch.save(ckpt, tmp_path / "stale.pt")
    run(write_config(tmp_path / "b.yaml", tmp_path / "b", data), resume=tmp_path / "stale.pt")
    a, b = final_weights(tmp_path / "a"), final_weights(tmp_path / "b")
    assert any(not torch.equal(a[k], b[k]) for k in a)


def test_memmap_resume_rejects_a_changed_window_order(tmp_path):
    write_shards(tmp_path / "tokens")
    data = DATA["memmap"](tmp_path)
    run(write_config(tmp_path / "a.yaml", tmp_path / "a", data))
    changed = dict(data, shuffle=False)
    with pytest.raises(ValueError, match="token-window order"):
        run(write_config(tmp_path / "b.yaml", tmp_path / "b", changed),
            resume=tmp_path / "a" / "step-3.pt")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ddp_worker(rank, world, port, config, resume):
    os.environ.update(WORLD_SIZE=str(world), RANK=str(rank), LOCAL_RANK=str(rank),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(1)
    run(config, resume)


def run_ddp(world, config, resume=None):
    mp.spawn(_ddp_worker, args=(world, _free_port(), config, resume), nprocs=world, join=True)


def test_ddp_resume_is_exact_and_memmap_survives_a_world_size_change(tmp_path):
    write_shards(tmp_path / "tokens")
    data = DATA["memmap"](tmp_path)
    # 2 ranks x 1 accumulation step, saving at step 2 (collective all_gather of data state).
    run_ddp(2, write_config(tmp_path / "a.yaml", tmp_path / "a", data, steps=4, save_every=2,
                            dropout=0.0, accum=1))
    # Same world size: bit-exact.
    run_ddp(2, write_config(tmp_path / "b.yaml", tmp_path / "b", data, steps=4, save_every=2,
                            dropout=0.0, accum=1), resume=tmp_path / "a" / "step-2.pt")
    assert_same_weights(final_weights(tmp_path / "a"), final_weights(tmp_path / "b"))
    # One rank x 2 accumulation steps sees the same global windows per step, so it
    # matches up to floating-point summation order.
    run(write_config(tmp_path / "c.yaml", tmp_path / "c", data, steps=4, save_every=2,
                     dropout=0.0, accum=2), resume=tmp_path / "a" / "step-2.pt")
    assert_same_weights(final_weights(tmp_path / "a"), final_weights(tmp_path / "c"), exact=False)


def test_ddp_map_style_resume_restores_the_sampler_epoch(tmp_path):
    data = DATA["map-style, epoch wrap"](tmp_path)
    kw = dict(steps=6, save_every=3, accum=1)
    run_ddp(2, write_config(tmp_path / "a.yaml", tmp_path / "a", data, **kw))
    assert load_ckpt(tmp_path / "a" / "step-3.pt")["extra"]["data"]["ranks"][1]["epoch"] >= 1
    run_ddp(2, write_config(tmp_path / "b.yaml", tmp_path / "b", data, **kw),
            resume=tmp_path / "a" / "step-3.pt")
    assert not (tmp_path / "b" / "step-3.pt").exists()
    assert_same_weights(final_weights(tmp_path / "a"), final_weights(tmp_path / "b"))


def test_ddp_streaming_resume_is_exact_and_refuses_a_world_size_change(tmp_path):
    data = DATA["streaming"](tmp_path)
    run_ddp(2, write_config(tmp_path / "a.yaml", tmp_path / "a", data, steps=4, save_every=2,
                            accum=1))
    run_ddp(2, write_config(tmp_path / "b.yaml", tmp_path / "b", data, steps=4, save_every=2,
                            accum=1), resume=tmp_path / "a" / "step-2.pt")
    assert_same_weights(final_weights(tmp_path / "a"), final_weights(tmp_path / "b"))
    with pytest.raises(ValueError, match="same world size"):
        run(write_config(tmp_path / "c.yaml", tmp_path / "c", data, steps=4, accum=2),
            resume=tmp_path / "a" / "step-2.pt")


def test_command_line_overrides_reach_the_run(tmp_path):
    write_shards(tmp_path / "tokens")
    cfg = write_config(tmp_path / "a.yaml", tmp_path / "unused", DATA["memmap"](tmp_path))
    run(cfg, overrides=[f"training.output_dir={tmp_path / 'b'}", "training.max_steps=2",
                        "training.precision=auto"])
    assert load_ckpt(tmp_path / "b" / "last.pt")["step"] == 2
    assert not (tmp_path / "unused").exists()
