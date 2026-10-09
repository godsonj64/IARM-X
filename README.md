# IARM-X

**Inductive Algebraic Resonance Attention Memory** is a research language-model architecture that combines algebraic resonance operators, constant-state recurrent memory, query-addressable associative memory, and sparse exact attention.

[![CI](https://github.com/godsonj64/IARM-X/actions/workflows/ci.yml/badge.svg)](https://github.com/godsonj64/IARM-X/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](pyproject.toml)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5%2B-ee4c2c.svg)](pyproject.toml)

> **Research status.** IARM-X is a correctness-validated reference implementation intended for controlled architecture experiments. It is not yet a claim of state-of-the-art performance. The recurrent scan runs as an exact chunkwise-parallel algorithm (`scan_impl: chunk`, the default) and keeps the per-token loop (`scan_impl: loop`) as its correctness oracle; a fused Triton/CUDA kernel is still needed before making systems-performance claims. See [`docs/TRAINING_BUDGET.md`](docs/TRAINING_BUDGET.md) for the compute and cost analysis of the 10B-token recipe.

## Quickstart: Colab, Kaggle or any Jupyter GPU

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/godsonj64/IARM-X/blob/claude/eloquent-dirac-9cqy7j/notebooks/iarmx_colab.ipynb)

[`notebooks/iarmx_colab.ipynb`](notebooks/iarmx_colab.ipynb) runs the whole recipe from a fresh notebook:

1. clone and install;
2. check the GPU (bf16, or fp16 with loss scaling on T4/V100/P100);
3. sanity tests and a GPU benchmark;
4. pretraining on streamed FineWeb-Edu at a 2048-token context;
5. the full 10B-token run's time and cost, projected from the speed measured on your GPU;
6. UltraChat chat-tuning;
7. a chat with the model.

Checkpoints go to Google Drive on Colab (only the newest is kept), and rerunning after a disconnect resumes exactly. The default `PLAN = "pilot"` (20M tokens plus 50 chat-tuning steps) takes about 1–2 hours on a free T4 and 15–30 minutes on an A100/H100, including downloads and data preparation. `PLAN = "full"` is the 10B-token run for an A100/H100-class GPU. A failed step stops the notebook with its error rather than running on.

Until this work is merged, the notebook clones the `claude/eloquent-dirac-9cqy7j` branch; set `BRANCH = "main"` after merging.

The same flow from a terminal:

```bash
git clone -b claude/eloquent-dirac-9cqy7j https://github.com/godsonj64/IARM-X.git && cd IARM-X
pip install -e '.[dev]'
# a 20M-token pilot: short warmup, frequent checkpoints, prints tokens/s
python -m iarmx.training.train --config configs/iarmx_100m_pretrain.yaml --resume auto \
  --set training.target_tokens=20000000 --set training.warmup_steps=30 \
  --set training.save_every=100 --set training.precision=auto \
  --set training.output_dir=checkpoints/pilot
```

`--set section.key=value` overrides any config value without editing the YAML.

## Why IARM-X

The original IARM design replaces pairwise self-attention with softmax-gated low-rank algebraic resonance operators and a coordinatewise normalized causal memory. IARM-X keeps that efficient slow-memory mechanism and adds two capabilities that are difficult for a purely compressed recurrent state:

1. **Query-dependent associative retrieval** through a finite-rank fast memory with explicit decay, erase, write, and read operations.
2. **Sparse exact retrieval** through resonance-conditioned attention layers interleaved with recurrent layers.

The default hybrid schedule is:

```text
R -> R -> R -> A -> R -> R -> R -> A
```

where `R` is an IARM-X recurrent block and `A` is a resonance-conditioned exact-attention block.

## 100M research model

The recommended first full experiment is the **100,202,256-parameter** model trained in two stages:

```text
FineWeb-Edu sample-10BT
        |
        v
IARM-X 100.202M pretraining
        |
        v
UltraChat-200k supervised fine-tuning
```

### Architecture

```text
model width              768
layers                     8
heads                     12
head width                64
FFN hidden              1984
resonance operators/head   4
operator rank             16
fast-memory rank          32
schedule              R,R,R,A x 2
training context        2048
max configured context   2048
parameters        100,202,256
```

Verify the count without allocating the model:

```bash
python scripts/count_params.py --config configs/iarmx_100m_pretrain.yaml --meta
```

## Architecture overview

```text
Token embedding
      |
      v
+---------------- IARM-X recurrent block ----------------+
| RMSNorm                                                |
| q / k / v projections + RoPE                          |
| resonance(q), resonance(k)                            |
| slow coordinatewise IARM memory       [persistent FP32]|
| associative fast memory               [persistent FP32]|
| cached causal depthwise local path                    |
| adaptive slow / fast / local fusion                   |
| SwiGLU + residual                                     |
+--------------------------------------------------------+
      |
      v
       ... R -> R -> R ...
      |
      v
+---------- resonance-conditioned attention block -------+
| resonant Q/K                                           |
| exact causal SDPA                                      |
| recurrent historical-importance bias                  |
| K/V + importance cache                                |
| SwiGLU + residual                                     |
+--------------------------------------------------------+
```

## Core equations

### Slow IARM memory

For each head and coordinate,

\[
\tilde q_t=q_t+\frac{1}{\sqrt R}\sum_o g_{t,o}U_oV_o^\top q_t,
\qquad
\phi_t=\operatorname{ELU}(\tilde q_t)+1.
\]

The persistent causal state is

\[
N_t=N_{t-1}+\phi_t\odot v_t,
\qquad
D_t=D_{t-1}+\phi_t,
\qquad
m_t^{\mathrm{slow}}=\frac{N_t}{D_t+\epsilon}.
\]

`N_t` and `D_t` remain FP32 under BF16/FP16 execution to avoid cumulative-state quantization failure at long context.

### Associative fast memory

Each head additionally maintains a finite-rank state \(S_t\in\mathbb R^{r\times d_h}\):

\[
\hat v_t=(k_t^m)^\top S_{t-1},
\]

\[
S_t=\gamma_tS_{t-1}-e_t k_t^m\hat v_t^\top+w_t k_t^m v_t^\top,
\]

\[
m_t^{\mathrm{fast}}=(q_t^m)^\top S_t.
\]

This introduces learned **decay, erase, write, and current-query read** operations while keeping recurrent-state size independent of sequence length.

### Adaptive fusion

\[
[\pi_s,\pi_f,\pi_l]=\operatorname{softmax}(W_\pi h_t),
\]

\[
z_t=\pi_s m_t^{\mathrm{slow}}+\pi_fm_t^{\mathrm{fast}}+\pi_l\ell_t.
\]

### Resonance-conditioned exact attention

Attention layers use resonantly transformed queries and keys. Recurrent blocks also emit token-level historical importance \(\rho_s\), aggregated between exact-attention layers:

\[
a_{t,s}=\frac{q_t^\top k_s}{\sqrt{d_h}}+\beta_h\log(\rho_s+\epsilon).
\]

The implementation retains PyTorch SDPA without explicitly materializing a dense importance-bias tensor. A key-only additive bias \(b_s\) is encoded by augmenting Q/K with one coordinate:

\[
\frac{[q,1]^\top[k,b_s\sqrt d]}{\sqrt d}
=
\frac{q^\top k}{\sqrt d}+b_s.
\]

## What is retained from IARM

- decoder-only RMSNorm / RoPE / SwiGLU stack;
- softmax-gated low-rank algebraic resonance operators;
- positive feature map `ELU(q_tilde) + 1`;
- numerator/denominator coordinatewise causal memory;
- causal depthwise local path;
- residual output gating.

## IARM-X extensions

- separate resonance transforms for queries and keys;
- finite-rank associative fast memory;
- learned decay, erase, and write controls;
- current-query-dependent fast-memory reads;
- adaptive slow/fast/local fusion;
- exact rolling convolution state;
- resonance-conditioned sparse exact-attention layers;
- recurrent historical-importance bias with cache parity;
- grouped importance aggregation across recurrent layers;
- FP32 persistent memory under mixed precision;
- full-sequence / cached-decoding numerical parity tests.

## Training data

The provided research recipe uses:

- **Pretraining:** `HuggingFaceFW/fineweb-edu`, configuration `sample-10BT`.
- **SFT:** `HuggingFaceH4/ultrachat_200k`, split `train_sft`.

### Pre-tokenize once (recommended for paid or preemptible GPUs)

```bash
python scripts/pretokenize.py --out data/tokens/fineweb-edu-10bt-gpt2 --num-proc 8
MEMMAP=1 bash scripts/train_100m.sh
```

`scripts/pretokenize.py` keeps every document whole (no truncation), appends one EOS per document and writes about 20 GB of `uint16` token shards plus an `index.json`. It runs on CPU, so do it before renting a GPU. `configs/iarmx_100m_pretrain_memmap.yaml` trains on fixed 2048-token windows read from those shards; every token is a training target exactly once per epoch and no compute is spent on padding. Add `--data-files "data/raw/fineweb-edu-10bt/sample/10BT/*.parquet"` to tokenize the local copy from `scripts/download_datasets.py` instead of streaming from the Hub.

For a pilot, write a small shard set to its own directory (`--max-tokens 100000000 --out data/tokens/pilot`) and point a copy of the config at it with `path: data/tokens/pilot` and a matching `target_tokens` (for example `100000000`). If `target_tokens` exceeds the tokens in the shard set, training repeats the data and prints a warning at startup saying how many times.

### Stream FineWeb-Edu directly

```bash
python -m iarmx.training.train --config configs/iarmx_100m_pretrain.yaml
```

Streaming tokenizes on the fly and packs documents into fixed `seq_len` rows, one EOS after each document; nothing is truncated, and `micro_batch_size` counts rows, so every micro-batch has the same shape. Literal special-token strings in the text (`<|endoftext|>`, `<|user|>`, ...) stay ordinary text in both pipelines.

### Download datasets once

```bash
python scripts/download_datasets.py
```

The default local layout is:

```text
data/raw/
├── fineweb-edu-10bt/
└── ultrachat_200k/
```

Then run the complete two-stage pipeline:

```bash
LOCAL_DATA=1 bash scripts/train_100m.sh
```

Two GPUs:

```bash
NPROC=2 LOCAL_DATA=1 bash scripts/train_100m.sh
```

Eight GPUs:

```bash
NPROC=8 LOCAL_DATA=1 bash scripts/train_100m.sh
```

Stage 1 stops by **global supervised-token count**, so `target_tokens: 10000000000` remains approximately a 10B-token experiment when world size changes. Stage 2 initializes from the final pretraining checkpoint and runs two UltraChat SFT epochs.

## Configuration files

```text
configs/
├── debug.yaml
├── sft_debug.yaml
├── iarmx_100m_pretrain.yaml
├── iarmx_100m_pretrain_local.yaml
├── iarmx_100m_pretrain_memmap.yaml
├── iarmx_100m_sft.yaml
├── iarmx_100m_sft_local.yaml
└── iarmx_1.3b.yaml
```

The larger research configuration is parameter-verified at:

```text
1,309,384,768 parameters = 1.309B
```

Use the meta device to inspect it without allocating the weights:

```bash
python scripts/count_params.py --config configs/iarmx_1.3b.yaml --meta
```

## Installation

```bash
git clone https://github.com/godsonj64/IARM-X.git
cd IARM-X
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[dev]'
```

## Training

### 100M pretraining

```bash
iarmx-train --config configs/iarmx_100m_pretrain.yaml
```

### 100M SFT

```bash
iarmx-train \
  --config configs/iarmx_100m_sft.yaml \
  --init-from checkpoints/iarmx-100m-pretrain/final
```

### Distributed training

```bash
torchrun --standalone --nproc_per_node=8 -m iarmx.training.train \
  --config configs/iarmx_100m_pretrain.yaml
```

The trainer includes BF16 autocast (`precision: auto` or `bf16` falls back to FP16 with loss scaling on GPUs without native BF16, such as the T4), FP32 persistent recurrent state, packed fixed-length rows, gradient accumulation, AdamW parameter groups, cosine decay with warmup (with `target_tokens` the decay follows tokens seen, so it ends exactly at the token budget whatever the GPU count), gradient clipping, DDP data partitioning, DDP `no_sync()` during accumulation, gradient checkpointing, checkpointing, optional `torch.compile`, and resumable optimizer/scheduler/step state.

### Resume a run

```bash
iarmx-train \
  --config configs/iarmx_100m_pretrain.yaml \
  --resume checkpoints/iarmx-100m-pretrain/step-10000.pt
```

`--resume auto` picks the newest checkpoint in the config's `output_dir` (or starts fresh if there is none); `scripts/train_100m.sh` uses it, so rerunning the script after a preemption continues where it stopped. Checkpoints are written atomically, and `keep_last: 3` in the 100M pretraining configs deletes older `step-N.pt` files (about 1.2 GB each).

`--resume` restores model, optimizer, scheduler, step, tracked training counters, each rank's RNG streams and the exact data position, so a resumed run continues with the next unseen batch instead of replaying data:

- `dataset: memmap` stores the global window position, so the data continues exactly even on a different number of GPUs. Keep `micro_batch_size x grad_accum x GPUs` constant in that case (the trainer warns if it changes); the learning rate follows tokens seen, so it does not jump.
- Streaming and map-style datasets (including SFT) store each rank's `StatefulDataLoader` state; they resume exactly with the same number of GPUs and `num_workers`, and refuse to resume otherwise rather than replay data.

With the same GPU count and settings, a resumed run reproduces the uninterrupted run bit-for-bit on CPU (`tests/test_resume.py`). On GPUs it sees the same data in the same order with the same RNG streams, but CUDA kernels are not bitwise deterministic, so weights match only approximately (as between any two GPU runs); after a GPU-count change the RNG streams are reseeded. `--init-from` loads pretrained model weights into a fresh training stage, starting its data from the beginning; with `--resume`, a config-level `init_from` is ignored because the checkpoint already holds the weights.

## Generation

```bash
iarmx-generate \
  --model checkpoints/iarmx-100m-sft/final \
  --chat \
  --prompt "Explain associative memory in simple terms." \
  --max-new-tokens 128
```

`--chat` wraps the prompt in the UltraChat turn format used for fine-tuning and stops at the end of the reply. Leave it out to continue plain text with a pretrained-only model.

Cached generation maintains:

- constant-size slow/fast recurrent memory per recurrent layer;
- an exact rolling cache for the causal depthwise convolution;
- K/V caches only in sparse exact-attention layers;
- historical-importance caches aligned with attention keys.

`model.generate(..., attention_mask=...)` also supports padded variable-length prompt batches.

## Evaluation

```bash
iarmx-eval --model checkpoints/iarmx-100m-sft/final --pairs 16
```

The bundled evaluator contains a minimal key-value recall test. A publishable IARM-X comparison should additionally include held-out perplexity, MQAR, RULER, Needle-in-a-Haystack, LongBench, reasoning/code benchmarks, throughput, memory, and parameter/FLOP-matched Transformer and original-IARM baselines.

See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## QA

Run:

```bash
pytest -q
python scripts/smoke_test.py
python -m compileall -q iarmx scripts tests
```

The current documented regression suite passes **83/83 tests** and covers:

- chunkwise-parallel vs per-token recurrent scan parity (outputs, state, gradients, cached decoding);
- bit-exact resume on CPU for token-shard, streaming and map-style data (0 and 2 loader workers, epoch wraps before and after the resume point, 2-process DDP), token-shard resume on a different world size, `--resume auto`, atomic and pruned checkpoints, and a token-budget schedule that ends at `min_lr`;
- notebook support: `--set` config overrides, fp16 fallback on GPUs without native bf16, the chat prompt format, and a static check of `notebooks/iarmx_colab.ipynb`;
- pre-tokenization without truncation (bounded memory, ordered multiprocessing), token-window/label alignment across shards, fixed-length packed rows with one EOS per document, and special-token strings kept as text;
- full-vs-cached and arbitrary-chunk parity;
- zero future-token leakage;
- exact convolution-cache behavior;
- historical-importance cache parity;
- SDPA/manual-attention equivalence;
- FP32 persistent state under BF16 weights;
- assistant-only shifted SFT targets;
- pretraining/SFT padding masks and pretraining packing;
- tied embedding/output safetensors round-trip;
- recurrent-state scaling;
- gradient-checkpointing correctness;
- gradient coverage;
- padded batched generation;
- checkpoint restoration;
- DDP map-dataset partitioning;
- exact 1.309B configuration verification.

See [`docs/QA.md`](docs/QA.md) for the recorded numerical checks and limitations.

## Repository structure

```text
IARM-X/
├── .github/
│   ├── ABOUT.md
│   └── workflows/ci.yml
├── configs/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── EXPERIMENTS.md
│   ├── QA.md
│   └── TRAINING_BUDGET.md
├── notebooks/
│   └── iarmx_colab.ipynb
├── iarmx/
│   ├── config.py
│   ├── generate.py
│   ├── data/
│   ├── eval/
│   ├── model/
│   ├── training/
│   └── utils/
├── scripts/
│   ├── bench_scan.py
│   ├── count_params.py
│   ├── download_datasets.py
│   ├── estimate_cost.py
│   ├── pretokenize.py
│   ├── profile_model.py
│   ├── smoke_test.py
│   └── train_100m.sh
├── tests/
├── CITATION.cff
├── Dockerfile
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Reproducibility and claims

This repository distinguishes architectural implementation from empirical claims. The code and regression tests establish implementation consistency; they do **not** establish superiority over Transformers, Mamba-family models, linear attention, or other hybrid architectures. Comparative claims should only be made from released, controlled experiments with matched data, parameters, training tokens/FLOPs, hardware, precision, and evaluation protocols.

## Original IARM paper

IARM-X extends the original IARM mechanism described in:

> Godson Johnson. **Inductive Algebraic Resonance Memory for Attention-Free Language Modelling.** Preprints.org, 2026. DOI: `10.20944/preprints202607.1228.v1`.

Paper: https://doi.org/10.20944/preprints202607.1228.v1

## Citation

Software citation metadata is provided in [`CITATION.cff`](CITATION.cff). If IARM-X is used in research, cite the software repository and distinguish the IARM-X hybrid extensions from the original IARM paper.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
