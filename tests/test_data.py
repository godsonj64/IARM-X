import torch

from iarmx.data.pretrain import collate_pretrain
from iarmx.data.sft import render_ultrachat, collate_sft


class FakeTokenizer:
    ids = {"<|user|>": 1, "<|assistant|>": 2, "<|end|>": 3}

    def convert_tokens_to_ids(self, token):
        return self.ids[token]

    def encode(self, content, add_special_tokens=False):
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


def test_pretrain_masks_synthetic_trailing_padding():
    out = collate_pretrain([{"input_ids": [1, 2, 3]}], pad_id=0, seq_len=4, pack=False)
    assert out["input_ids"].tolist() == [[1, 2, 3, 0]]
    assert out["labels"].tolist() == [[2, 3, -100, -100]]


def test_pretrain_packing_uses_document_boundary_but_masks_tail():
    out = collate_pretrain(
        [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6]}], pad_id=9, seq_len=4, pack=True
    )
    assert out["input_ids"].shape[1] == 4
    # flattened stream begins 1,2,EOS,3,4,... so first next-token row is exact.
    assert out["input_ids"][0].tolist() == [1, 2, 3, 9]
    assert out["labels"][0].tolist() == [2, 3, 9, 4]
    assert out["labels"][1].tolist() == [6, 9, -100, -100]
