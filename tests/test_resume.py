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

    def __call__(self, texts, add_special_tokens=False):
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


def fake_load_dataset(name, config=None, split=None, streaming=False, data_files=None):
    from datasets import Dataset, IterableDataset

    texts = make_texts()
    if not streaming:
        # 6 documents split into 9 pieces = 2.25 optimizer steps per epoch, so the
        # step-3 checkpoint falls inside the second epoch.
        return Dataset.from_list([{"text": t, "id": i} for i, t in enumerate(texts[:6])])

    def gen(shards):
        for s in shards:
            for i in range(s, len(texts), 4):
                yield {"text": texts[i], "id": i}

    return IterableDataset.from_generator(gen, gen_kwargs={"shards": [0, 1, 2, 3]})


def write_config(path, out_dir, data, steps=6, save_every=3, dropout=0.1, workers=0,
                 micro=2, accum=2):
    spec = {
        "model": dict(vocab_size=VOCAB, dim=32, n_layers=2, n_heads=4, ffn_hidden=64,
                      max_seq_len=64, n_operators=2, operator_rank=4, fast_memory_rank=4,
                      attention_every=2, dropout=dropout, scan_chunk_size=8),
        "data": dict(stage="pretrain", seq_len=16, **data),
        "training": dict(seed=0, micro_batch_size=micro, grad_accum=accum, max_steps=steps,
                         lr=3e-3, min_lr=3e-4, warmup_steps=2, compile=False, log_every=100,
                         save_every=save_every, num_workers=workers, output_dir=str(out_dir)),
    }
    path.write_text(yaml.safe_dump(spec))
    return path


def run(config, resume=None):
    import datasets

    saved = train_mod.load_tokenizer, datasets.load_dataset, sys.argv
    train_mod.load_tokenizer = lambda name: FakeTokenizer()
    datasets.load_dataset = fake_load_dataset
    sys.argv = ["train", "--config", str(config)] + (["--resume", str(resume)] if resume else [])
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
    "map-style, epoch wrap": lambda tmp: {"dataset": "fake", "streaming": False},
}


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("kind", list(DATA))
def test_resume_matches_uninterrupted_run(tmp_path, kind, workers):
    write_shards(tmp_path / "tokens")
    data = DATA[kind](tmp_path)
    run(write_config(tmp_path / "a.yaml", tmp_path / "a", data, workers=workers))
    if kind == "map-style, epoch wrap":
        ckpt = torch.load(tmp_path / "a" / "step-3.pt", weights_only=False)
        assert ckpt["extra"]["data"]["ranks"][0]["epoch"] >= 1
    run(write_config(tmp_path / "b.yaml", tmp_path / "b", data, workers=workers),
        resume=tmp_path / "a" / "step-3.pt")
    assert_same_weights(final_weights(tmp_path / "a"), final_weights(tmp_path / "b"))


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
