import importlib.util
import multiprocessing
import sys
from pathlib import Path

import numpy as np
import pytest

from iarmx.data.memmap import TokenWindows, read_index

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pretokenize.py"
TEXTS = ["hello world", "", "x" * 5000, "ab"] * 30
VOCAB = 300


class FakeTokenizer:
    eos_token_id = 299
    pad_token_id = 299

    def __len__(self):
        return VOCAB

    def __call__(self, texts, add_special_tokens=False, split_special_tokens=False):
        return {"input_ids": [[ord(c) for c in t] for t in texts]}


def fake_load_dataset(*args, streaming=False, **kwargs):
    from datasets import IterableDataset

    return IterableDataset.from_generator(lambda: ({"text": t} for t in TEXTS))


def run_script(out, num_proc, monkeypatch, extra=()):
    spec = importlib.util.spec_from_file_location("pretokenize_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Pool pickles worker functions by module name.
    monkeypatch.setitem(sys.modules, spec.name, module)
    monkeypatch.setattr(module, "load_tokenizer", lambda name: FakeTokenizer())
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(sys, "argv", ["pretokenize", "--out", str(out), "--num-proc", str(num_proc),
                                      "--batch-size", "7", "--shard-tokens", "1000", *extra])
    module.main()
    return read_index(out)


def shard_tokens(out, index):
    return np.concatenate([np.fromfile(out / s["file"], dtype=index["dtype"]) for s in index["shards"]])


@pytest.mark.parametrize("num_proc", [1, 2])
def test_pretokenize_keeps_whole_documents_in_order(tmp_path, monkeypatch, num_proc):
    if num_proc > 1 and multiprocessing.get_start_method() != "fork":
        pytest.skip("worker processes only inherit the fake tokenizer under fork")
    index = run_script(tmp_path, num_proc, monkeypatch)
    expected = [tok for t in TEXTS if t for tok in [ord(c) for c in t] + [299]]
    assert shard_tokens(tmp_path, index).tolist() == expected  # 5000-char docs are not truncated
    assert index["documents"] == 90 and index["eos_id"] == 299 and index["vocab_size"] == VOCAB
    assert TokenWindows(tmp_path, seq_len=64)[0]["input_ids"].tolist() == expected[:64]


@pytest.mark.parametrize("num_proc", [1, 2])
def test_pretokenize_max_tokens_stops_early(tmp_path, monkeypatch, num_proc):
    # With worker processes this also checks that stopping early releases the
    # in-flight throttle, so terminating the pool cannot hang.
    if num_proc > 1 and multiprocessing.get_start_method() != "fork":
        pytest.skip("worker processes only inherit the fake tokenizer under fork")
    index = run_script(tmp_path, num_proc, monkeypatch, extra=("--max-tokens", "6000"))
    assert 6000 <= index["tokens"] < 6000 + 7 * 5001
