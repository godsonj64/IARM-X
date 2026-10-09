# Review status: data-pipeline fix

Adversarial review of commit `41b56d9` (truncation fix, exact resume): four
reviewers (resume/checkpoint, data correctness, scale/performance,
tests/docs/packaging), each followed by a skeptic who tried to refute every
finding by running code. The fixes are in the commit that adds this table.
Delete this file once you have read it.

## Findings and resolutions

| # | Finding | Skeptic | Resolution |
|---|---|---|---|
| P1 | `pretokenize.py` main process buffered worker results without limit (2.7 GB in 11 s with 24 fast workers) | confirmed | Workers return `uint16` arrays with EOS inserted; at most 2× `--num-proc` batches in flight; reads only the text column. The same 24-worker repro now peaks at 1.2 GB with a backlog of 0 |
| R1 | Streaming checkpoint resumed into a memmap config saved a wrong window position | confirmed | The saved position counts only windows drawn from the sampler; a memmap checkpoint resumed with a streaming config says so |
| R2 | Resuming on a different GPU count made the LR jump (step-based horizon) | confirmed | With `target_tokens`, cosine decay follows tokens seen; warns when the global batch changes |
| D2 | Streaming runs hit `target_tokens` at ~56% of the LR schedule (pre-existing) | confirmed | Token-based decay (as R2), and streaming now packs fixed `seq_len` rows, so every micro-batch has the same shape and token count |
| D3 | Map-style data under DDP joined document pieces without EOS | confirmed | Documents are packed into rows inside the dataset (no pieces to reorder) |
| R3 | Streaming with `num_workers: 0` re-applied the resume point at every new pass | confirmed | The HF resume state is cleared before each new pass; test fails without the fix |
| R4 | Changing `num_workers` crashed with a bare `AssertionError` | confirmed | Clear error naming the saved value |
| R5 | `--resume` refused when the config sets `training.init_from` (SFT configs) | confirmed | A config-level `init_from` is ignored when resuming |
| R6 | Checkpoints written non-atomically and never pruned (~180 GB for 10B tokens) | confirmed | Atomic write (`.tmp` + fsync + rename); `keep_last: 3` in the 100M pretraining configs |
| R7 | `train_100m.sh` never resumed | confirmed | `--resume auto` picks the newest checkpoint; the script uses it |
| D4 | Literal `<\|endoftext\|>` / `<\|user\|>` in text became EOS / control tokens | confirmed | `split_special_tokens=True` in pre-tokenization, streaming and SFT rendering |
| D5 | Memmap silently repeats data when `target_tokens` exceeds the shards; README pilot pointed at the 10B path | confirmed | Startup warning with the repeat factor; README pilot uses its own directory and `target_tokens` |
| T1 | Bit-exact resume test passed even if `--resume` were ignored | pending | Each resumed run must not re-save the checkpoint it resumed from |
| T2 | Multi-GPU map-style resume (sampler epoch) untested | pending | 2-process DDP map-style test resuming inside epoch 2 |
| T3 | Streaming with workers crashed on Python 3.14 (forkserver) | pending | Tokenization is a module-level function; streaming and map-style resume tests pass under forced forkserver |
| T4 | Docs overclaimed "bit-for-bit" (GPU kernels, GPU-count change) | pending | Reworded: bit-for-bit on CPU with the same GPU count; same data and RNG on GPU |
| S1 | Scale review: P1 depends on reader speed; read only the text column | pending | Covered by P1 |

The scale/performance review found no other issue at 10B-token scale. The
sampler's permutation takes 1.6 s and 156 MB per rank per epoch, and window
reads run at ~41M tokens/s per process, about 100× what a GPU consumes.

## Test status

70/70 tests pass. CI-equivalent compile, smoke-test and parameter-count checks
pass, and all changed files parse as Python 3.11.
