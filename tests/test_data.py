import torch

from iarmx.data.pretrain import build_pretrain_dataset, encode_texts, pack_windows
from iarmx.data.sft import render_ultrachat, collate_sft


class FakeTokenizer:
    ids = {"<|user|>": 1, "<|assistant|>": 2, "<|end|>": 3}

    def convert_tokens_to_ids(self, token):
        return self.ids[token]

    def encode(self, content, add_special_tokens=False, split_special_tokens=False):
        table = {
            "hello": [10, 11],
            "answer": [20, 21],
            "short": [30],
        }
        return table[content]


def test_sft_is_shifted_next_token_supervision_without_leakage():
    tok = FakeTokenizer()
    ids, labels = render_ultrachat(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "answer"},
        ],
        tok,
    )
    assert ids == [1, 10, 11, 3, 2, 20, 21]
    assert labels == [-100, -100, -100, -100, 20, 21, 3]
    supervised = [(i, ids[i], labels[i]) for i in range(len(ids)) if labels[i] != -100]
    assert supervised == [(4, 2, 20), (5, 20, 21), (6, 21, 3)]


def test_sft_collator_masks_padding():
    batch = [
        {"input_ids": [1, 2, 3], "labels": [-100, 8, 9]},
        {"input_ids": [1], "labels": [-100]},
    ]
    out = collate_sft(batch, pad_id=0)
    assert out["input_ids"].tolist() == [[1, 2, 3], [1, 0, 0]]
    assert out["labels"].tolist() == [[-100, 8, 9], [-100, -100, -100]]


def test_pack_windows_targets_every_token_once_and_masks_tail_padding():
    stream = list(range(10, 21))  # 11 tokens -> 10 targets
    out = pack_windows(stream, seq_len=4, pad_id=0)
    assert out["input_ids"] == [[10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 0, 0]]
    assert out["labels"] == [[11, 12, 13, 14], [15, 16, 17, 18], [19, 20, -100, -100]]
    targets = [t for row in out["labels"] for t in row if t != -100]
    assert targets == stream[1:]


class FakeTextTokenizer:
    eos_token_id = 9
    pad_token_id = 9

    def __call__(self, texts, add_special_tokens=False, split_special_tokens=False):
        return {"input_ids": [[int(c) for c in t] for t in texts]}


def fake_load_dataset(*args, streaming=False, **kwargs):
    from datasets import Dataset, IterableDataset

    texts = ["123", "", "45678123456781234", "5"]
    if streaming:
        return IterableDataset.from_generator(_fake_texts, gen_kwargs={"texts": texts})
    return Dataset.from_list([{"text": t} for t in texts])


def _fake_texts(texts):
    for t in texts:
        yield {"text": t}


def test_packed_rows_keep_whole_documents_with_one_eos_each(monkeypatch):
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    stream = [1, 2, 3, 9] + [4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 9] + [5, 9]
    for streaming in (True, False):
        ds = build_pretrain_dataset(FakeTextTokenizer(), seq_len=4, streaming=streaming)
        rows = list(ds)
        assert all(len(r["input_ids"]) == len(r["labels"]) == 4 for r in rows)
        targets = [t for r in rows for t in r["labels"] if t != -100]
        assert targets == stream[1:]  # nothing truncated, no padding trained on


def test_unpacked_rows_are_one_truncated_document_each(monkeypatch):
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    rows = list(build_pretrain_dataset(FakeTextTokenizer(), seq_len=4, streaming=False, pack=False))
    assert [r["input_ids"] for r in rows] == [[1, 2, 9, 9], [4, 5, 6, 7]]
    assert [r["labels"] for r in rows] == [[2, 3, -100, -100], [5, 6, 7, 8]]


def test_special_token_strings_in_text_stay_ordinary_text():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz<>|_")}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="_"))
    backend.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="_")
    tok.add_special_tokens({"eos_token": "<|endoftext|>", "additional_special_tokens": ["<|user|>"]})
    specials = {tok.eos_token_id, tok.convert_tokens_to_ids("<|user|>")}
    ids = encode_texts(tok, ["a<|endoftext|>b<|user|>c"])[0]
    assert not specials & set(ids)
