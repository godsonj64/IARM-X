# Review status: data-pipeline fix (commit 41b56d9)

Snapshot saved 2026-10-09 while the adversarial review was still running.
Delete this file once the findings below are resolved.

## What is already done and pushed

| Commit | Content |
|---|---|
| `ed727f3` | Exact chunkwise-parallel recurrent scan (81–89× fewer GPU ops per step) |
| `38e65ef` | Cost analysis `docs/TRAINING_BUDGET.md`, `scripts/estimate_cost.py`, `scripts/bench_scan.py` |
| `41b56d9` | Data pipeline: no truncation, `scripts/pretokenize.py`, `dataset: memmap`, exact resume (62/62 tests) |

## Review progress

Four reviewers (data correctness, resume/checkpoint, scale/performance,
tests/docs/packaging), each followed by a skeptic who tries to refute the
findings by running code.

| Reviewer | State |
|---|---|
| resume-checkpoint | **finished**: 7 findings below (not yet through the skeptic) |
| data-correctness | was still running; one finding reproduced in its notes (P1) |
| scale-performance | was running |
| tests-docs-packaging | not started |

None of the findings below affects the parallel scan or the cost estimates.

## Findings so far, in planned fix order

Line numbers refer to commit `41b56d9`.

**P1. `scripts/pretokenize.py` can run out of RAM** (data-correctness, medium, reproduced by the reviewer).
Worker processes return Python lists of ints and nothing bounds how many
results queue in the main process. With a fast tokenizer the reviewer measured
memory growing ~120 MB/s, peaking at 9 GB for 300M tokens, which would exhaust
RAM long before 10B.
*Fix:* workers return compact `uint16` arrays with EOS already inserted, and
cap in-flight batches (a semaphore around the `imap` input) at about 2× `--num-proc`.

**R1. Switching a run from streaming to memmap corrupts the saved position** (medium, reproduced). `train.py:330`
saves `position = examples_seen`. After resuming a streaming checkpoint into a
memmap config, the sampler starts at 0 but `examples_seen` doesn't, so the
*next* resume skips windows.
*Fix:* save the sampler's own position (`examples_seen - base`, where `base`
is the resumed `examples_seen` minus the saved start), or refuse cross-pipeline
resume. Also print a message when a memmap checkpoint is resumed with a
streaming config.

**R2. Resuming on a different GPU count changes the learning-rate schedule** (medium, reproduced). `train.py:246`
The scheduler horizon depends on world size. The data position is exact, but
with the same per-GPU batch settings the global batch and the cosine LR position
jump (1 → 2 GPUs at step 50k: LR 4.8e-4 → 2.1e-4).
*Fix:* keep the global batch constant by adjusting `grad_accum` automatically
(error if it doesn't divide), or store the scheduler horizon in the checkpoint.
Fix the README/TRAINING_BUDGET wording "exactly even on a different number of GPUs".

**R3. Streaming with `num_workers: 0` re-applies the resume point at every new pass** (low, reproduced). `train.py:380`
The HF dataset keeps the loaded state, so after an epoch wrap the stream restarts
from the checkpoint position instead of the beginning. `num_workers >= 1` (the
shipped configs use 2) is unaffected.
*Fix:* clear the dataset's start state (`loader.dataset.load_state_dict(None)`)
before re-iterating, and add a `num_workers=0` wrap-after-resume test.

**R4. Changing `num_workers` between runs crashes with a bare `AssertionError`** (low, reproduced). `train.py:309`
*Fix:* store `num_workers` in the data state and raise a clear message.

**R5. README says SFT resumes, but SFT configs set `training.init_from`, which makes `--resume` raise** (low, reproduced). `train.py:227`
*Fix:* with `--resume`, ignore a config-level `init_from`; reject only the CLI flag combination.

**R6. Checkpoints are written non-atomically and never pruned** (low, pre-existing). `checkpoint.py:20`
A preemption mid-write leaves a corrupt newest checkpoint. Keeping every
`step-N.pt` is ~180 GB for a 10B-token 100M run.
*Fix:* write to `.tmp`, then `os.replace`. Keep the last K step checkpoints.

**R7. `scripts/train_100m.sh` never passes `--resume`** (low). Rerunning it after a preemption restarts at step 0 and overwrites checkpoints.
*Fix:* auto-detect the newest intact checkpoint in `output_dir` and pass `--resume`.

## Open question for the user

Raise the default training context from 512 to 2048 tokens (pretraining and
SFT)? It costs about 2% more compute for IARM-X and lets chats run to roughly
1,500 words instead of ~380.

## How to continue in a new session

Check out branch `claude/eloquent-dirac-9cqy7j` and ask Claude to "fix the
findings in docs/REVIEW_STATUS.md, then re-run an adversarial review of the
remaining lenses (scale/performance, tests/docs/packaging)". Run the suite
offline with `pip install -e '.[dev]' && pytest -q`.
