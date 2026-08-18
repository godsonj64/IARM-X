# IARM-X

**Inductive Algebraic Resonance Attention Memory** is a research language-model architecture that combines algebraic resonance operators, constant-state recurrent memory, query-addressable associative memory, and sparse exact attention.

[![CI](https://github.com/godsonj64/IARM-X/actions/workflows/ci.yml/badge.svg)](https://github.com/godsonj64/IARM-X/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](pyproject.toml)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5%2B-ee4c2c.svg)](pyproject.toml)

> **Research status.** IARM-X is a correctness-validated reference implementation intended for controlled architecture experiments. It is not yet a claim of state-of-the-art performance. The recurrent scan is deliberately written as a clear PyTorch reference implementation; serious large-scale throughput work should replace it with a fused chunkwise Triton/CUDA kernel before making systems-performance claims.

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
training context          512
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

### Stream FineWeb-Edu directly

```bash
python -m iarmx.training.train --config configs/iarmx_100m_pretrain.yaml
```

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

The trainer includes BF16 autocast, FP32 persistent recurrent state, packed document collation, gradient accumulation, AdamW parameter groups, cosine decay with warmup, gradient clipping, DDP data partitioning, DDP `no_sync()` during accumulation, gradient checkpointing, checkpointing, optional `torch.compile`, and resumable optimizer/scheduler/step state.

### Resume a run

```bash
iarmx-train \
  --config configs/iarmx_100m_pretrain.yaml \
  --resume checkpoints/iarmx-100m-pretrain/step-10000.pt
```

`--resume` restores model, optimizer, scheduler, step, and tracked training counters. `--init-from` loads pretrained model weights into a fresh training stage.

## Generation

```bash
iarmx-generate \
  --model checkpoints/iarmx-100m-sft/final \
  --prompt "Explain associative memory in simple terms." \
  --max-new-tokens 128
```

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

The current documented regression suite passes **21/21 tests** and covers:

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
│   └── QA.md
├── iarmx/
│   ├── config.py
│   ├── generate.py
│   ├── data/
│   ├── eval/
│   ├── model/
│   ├── training/
│   └── utils/
├── scripts/
│   ├── count_params.py
│   ├── download_datasets.py
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
