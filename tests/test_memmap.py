import json
import pickle
from itertools import islice

import numpy as np
import pytest

from iarmx.data.memmap import TokenShardWriter, TokenWindows, WindowSampler

EOS = 0


def write_docs(path, docs, vocab=97, shard_tokens=7):
    writer = TokenShardWriter(path, vocab_size=vocab, eos_id=EOS, shard_tokens=shard_tokens)
    for doc in docs:
        writer.add_document(doc)
    return writer.close()


def stream_of(docs):
    out = []
    for doc in docs:
        if doc:
            out.extend(doc)
            out.append(EOS)
    return out


DOCS = [[5, 6, 7], [], list(range(10, 40)), [41], [50, 51, 52, 53, 54, 55, 56, 57]]


def test_writer_keeps_every_token_and_one_eos_per_document(tmp_path):
    index = write_docs(tmp_path, DOCS)
    expected = stream_of(DOCS)
    assert index["tokens"] == len(expected)
    assert index["documents"] == 4  # the empty document is skipped
    assert index["dtype"] == "uint16"
    assert len(index["shards"]) > 3  # the 30-token document spans several 7-token shards
    assert all(s["tokens"] == 7 for s in index["shards"][:-1])
    on_disk = np.concatenate([np.fromfile(tmp_path / s["file"], dtype=np.uint16) for s in index["shards"]])
    assert on_disk.tolist() == expected
    assert json.loads((tmp_path / "index.json").read_text()) == index
    assert not list(tmp_path.glob("*.tmp"))


def test_writer_validates_ids_and_refuses_to_overwrite(tmp_path):
    writer = TokenShardWriter(tmp_path, vocab_size=10, eos_id=EOS)
    with pytest.raises(ValueError):
        writer.add_document([3, 10])
    writer.close()
    with pytest.raises(FileExistsError):
        TokenShardWriter(tmp_path, vocab_size=10, eos_id=EOS)


def test_large_vocabulary_uses_uint32(tmp_path):
    index = write_docs(tmp_path, [[70_000, 1]], vocab=100_000)
    assert index["dtype"] == "uint32"
    assert TokenWindows(tmp_path, seq_len=2)[0]["input_ids"].tolist() == [70_000, 1]


@pytest.mark.parametrize("seq_len", [1, 4, 9])
def test_windows_are_shifted_next_token_pairs_across_shards(tmp_path, seq_len):
    write_docs(tmp_path, DOCS)
    stream = stream_of(DOCS)
    ds = TokenWindows(tmp_path, seq_len)
    assert len(ds) == (len(stream) - 1) // seq_len
    targets = []
    for i in range(len(ds)):
        item = ds[i]
        assert item["input_ids"].tolist() == stream[i * seq_len : (i + 1) * seq_len]
        assert item["labels"].tolist() == stream[i * seq_len + 1 : (i + 1) * seq_len + 1]
        targets.extend(range(i * seq_len + 1, (i + 1) * seq_len + 1))
    # Every stream position after the first is a target exactly once, up to the
    # final partial window.
    assert targets == list(range(1, len(ds) * seq_len + 1))
    with pytest.raises(IndexError):
        ds[len(ds)]


def test_windows_pickle_without_copying_token_data(tmp_path):
    write_docs(tmp_path, [list(range(1, 90))] * 50, shard_tokens=1000)
    ds = TokenWindows(tmp_path, seq_len=16)
    first = ds[3]["input_ids"].tolist()  # opens the memory maps
    blob = pickle.dumps(ds)
    assert len(blob) < 4096
    assert pickle.loads(blob)[3]["input_ids"].tolist() == first


def global_order(n, count, shuffle=True, seed=3):
    return list(islice(iter(WindowSampler(n, shuffle=shuffle, seed=seed)), count))


def take(sampler, k):
    return list(islice(iter(sampler), k))


@pytest.mark.parametrize("world", [1, 2, 3])
def test_ranks_partition_the_global_order(world):
    n, k = 17, 9
    order = global_order(n, world * k)
    per_rank = [take(WindowSampler(n, 0, world, r, seed=3), k) for r in range(world)]
    merged = [per_rank[r][i] for i in range(k) for r in range(world)]
    assert merged == order


def test_resume_continues_the_global_order_even_with_a_new_world_size():
    n = 23
    order = global_order(n, 60)
    # Run A: 2 ranks draw 7 windows each, then a checkpoint records position 14.
    first = [take(WindowSampler(n, 0, 2, r, seed=3), 7) for r in range(2)]
    position = 14
    # Run B resumes on 3 ranks from that position.
    second = [take(WindowSampler(n, position, 3, r, seed=3), 5) for r in range(3)]
    consumed = [first[r][i] for i in range(7) for r in range(2)]
    consumed += [second[r][i] for i in range(5) for r in range(3)]
    assert consumed == order[: position + 15]


def test_epochs_reshuffle_deterministically():
    n = 10
    order = global_order(n, 3 * n)
    epochs = [order[i * n : (i + 1) * n] for i in range(3)]
    assert all(sorted(e) == list(range(n)) for e in epochs)
    assert epochs[0] != epochs[1]
    assert order == global_order(n, 3 * n)
    assert global_order(n, n, shuffle=False) == list(range(n))
