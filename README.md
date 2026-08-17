# IARM-X

**Inductive Algebraic Resonance Attention Memory** — a PyTorch research implementation that extends IARM with query-addressable finite-state memory and sparse exact attention.

> **Status:** correctness-validated research reference implementation. **This branch includes a parameter-verified 100.202M two-stage FineWeb-Edu 10BT → UltraChat-200k training recipe.** The repository includes cached autoregressive decoding, assistant-only SFT, packed pretraining collation, DDP-aware training, checkpoint resume, gradient checkpointing, SDPA attention, tests, profiling, and a parameter-verified 1.309B research configuration. The recurrent scan remains a clear Python reference implementation; large-scale throughput work should replace it with a fused chunkwise Triton/CUDA kernel before making systems-performance claims.


## 100M research model: FineWeb-Edu 10BT → UltraChat

The recommended first full architecture comparison is the **100,202,256-parameter** IARM-X model:

```text
width              768
layers               8
heads                12
head width            64
FFN hidden          1984
resonance ops/head     4
operator rank          16
fast-memory rank       32
schedule        R,R,R,A × 2
training context      512
parameters    100,202,256
```

Parameter count:

```bash
python scripts/count_params.py --config configs/iarmx_100m_pretrain.yaml --meta
```

### Dataset options

**Stream directly from Hugging Face** (no full upfront FineWeb download):

```bash
python -m iarmx.training.train --config configs/iarmx_100m_pretrain.yaml
```

**Download both datasets first** for repeatable/local training:

```bash
python scripts/download_datasets.py
```

This stores FineWeb-Edu `sample-10BT` under `data/raw/fineweb-edu-10bt` and UltraChat-200k under `data/raw/ultrachat_200k`. Then run:

```bash
LOCAL_DATA=1 bash scripts/train_100m.sh
```

For two GPUs:

```bash
NPROC=2 LOCAL_DATA=1 bash scripts/train_100m.sh
```

Stage 1 stops by **global supervised token count**, not a GPU-dependent step count, so `target_tokens: 10000000000` remains approximately 10B tokens when world size changes. Stage 2 uses `epochs: 2` over `train_sft` and initializes from `checkpoints/iarmx-100m-pretrain/final`.

The default 100M recipe uses BF16, persistent FP32 recurrent state, gradient checkpointing, packed pretraining, AdamW, cosine decay, and the `R,R,R,A` 3:1 hybrid schedule. The recurrent scan remains the exact PyTorch reference implementation; a fused kernel is still recommended before treating 10B-token wall-clock throughput as production-grade.

## Architecture

The original IARM design uses softmax-gated low-rank resonance operators and a coordinatewise normalized causal memory. IARM-X preserves that slow-memory mechanism and adds current-query-dependent retrieval.

```text
embedding
   │
   ├─ R: IARM-X recurrent block
   │    ├─ RMSNorm
   │    ├─ q/k/v projections + RoPE
   │    ├─ resonance(q), resonance(k)
   │    ├─ slow IARM coordinatewise memory      [persistent FP32]
   │    ├─ selective associative fast memory    [persistent FP32]
   │    ├─ cached causal depthwise local path
   │    └─ adaptive 3-way fusion + SwiGLU
   │
   ├─ R
   ├─ R
   ├─ A: resonance-conditioned exact attention + SwiGLU
   │       ├─ cached K/V
   │       └─ cached recurrent-importance signal
   │
   └─ repeat R,R,R,A ...
```

The standard schedule is **3 recurrent layers : 1 exact-attention layer**.

## Core equations

### Original IARM slow memory

For each head and coordinate,

\[
\tilde q_t=q_t+\frac{1}{\sqrt R}\sum_o g_{t,o}U_oV_o^\top q_t,
\qquad
\phi_t=\mathrm{ELU}(\tilde q_t)+1,
\]

\[
N_t=N_{t-1}+\phi_t\odot v_t,
\qquad
D_t=D_{t-1}+\phi_t,
\qquad
m_t^{slow}=\frac{N_t}{D_t+\epsilon}.
\]

`N_t` and `D_t` are stored in FP32 even under BF16/FP16 inference so cumulative updates do not quantize onto long-context plateaus.

### IARM-X associative memory

Each head additionally maintains \(S_t\in\mathbb R^{r\times d_h}\):

\[
\hat v_t=(k_t^m)^\top S_{t-1},
\]

\[
S_t=\gamma_tS_{t-1}-e_t k_t^m\hat v_t^\top+w_t k_t^m v_t^\top,
\]

\[
m_t^{fast}=(q_t^m)^\top S_t.
\]

This adds explicit **decay / erase / write / query-read** operations while retaining context-independent recurrent-state size.

### Fusion

\[
[\pi_s,\pi_f,\pi_l]=\operatorname{softmax}(W_\pi h_t),
\]

\[
z_t=\pi_s m_t^{slow}+\pi_fm_t^{fast}+\pi_l\ell_t.
\]

### Resonance-conditioned exact attention

The attention layers apply separate resonance transforms to queries and keys. Recurrent layers also emit per-token historical importance \(\rho_s\), aggregated across every recurrent layer since the previous attention layer:

\[
a_{t,s}=\frac{q_t^\top k_s}{\sqrt{d_h}}+\beta_h\log(\rho_s+\epsilon).
\]

The implementation keeps the optimized PyTorch SDPA path without constructing a dense importance-bias matrix. A key-only additive bias \(b_s\) is encoded exactly by augmenting Q/K with one coordinate:

\[
\frac{[q,1]^\top[k,b_s\sqrt d]}{\sqrt d}
=
\frac{q^\top k}{\sqrt d}+b_s.
\]

## Repository structure

```text
iarm-x/
├── configs/
│   ├── debug.yaml
│   ├── sft_debug.yaml
│   └── iarmx_1.3b.yaml
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
│   ├── profile_model.py
│   └── smoke_test.py
├── tests/
├── Dockerfile
├── pyproject.toml
└── README.md
```

## Installation

```bash
git clone <your-repo-url>
cd iarm-x
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[dev]'
```

## QA

```bash
pytest -q
python scripts/smoke_test.py
python -m compileall -q iarmx scripts tests
```

The current QA revision passes **21/21 tests**. The suite covers full-vs-cached parity, chunked-vs-full parity, causal leakage, convolution-cache correctness, historical-importance cache correctness, SDPA/manual attention equivalence, BF16-safe FP32 persistent memory, SFT target alignment, pretraining padding masks and packing, tied safetensors save/load, gradient coverage, gradient checkpointing, checkpoint restore, distributed sampler partitioning, padding-aware generation, recurrent-state scaling, and exact research-config parameter count. See `docs/QA.md`.

## Smoke test

```bash
python scripts/smoke_test.py
```

## Parameter count

For small configurations:

```bash
python scripts/count_params.py --config configs/debug.yaml
```

For the large research configuration, use the meta device to avoid allocating ~1.3B parameters:

```bash
python scripts/count_params.py --config configs/iarmx_1.3b.yaml --meta
```

Expected count:

```text
1,309,384,768 parameters (1.309B)
```

## Pretraining

Small debug run:

```bash
iarmx-train --config configs/debug.yaml
```

Distributed:

```bash
torchrun --standalone --nproc_per_node=8 -m iarmx.training.train \
  --config configs/iarmx_1.3b.yaml
```

The training harness includes BF16 autocast, packed document collation, gradient accumulation, AdamW parameter groups, cosine decay with warmup, clipping, DDP dataset partitioning, DDP `no_sync()` during accumulation, gradient checkpointing, checkpointing, optional `torch.compile`, and resumable optimizer/scheduler/step state.

Resume:

```bash
iarmx-train \
  --config configs/iarmx_1.3b.yaml \
  --resume checkpoints/iarmx-1.3b/step-10000.pt
```

For streaming datasets this restores model/optimizer/scheduler/step state; exact data-iterator position is not serialized.

## Supervised fine-tuning

```bash
iarmx-train --config configs/sft_debug.yaml
```

The UltraChat renderer performs assistant-only **next-token** supervision. It does not supervise a token with itself at the same sequence position.

## Generation

```bash
iarmx-generate \
  --model checkpoints/debug/final \
  --prompt "Explain associative memory in simple terms." \
  --max-new-tokens 128
```

Cached generation maintains:

- constant-size recurrent slow/fast memory per recurrent layer;
- an exact rolling cache for the causal depthwise convolution;
- K/V caches only in sparse attention layers;
- historical-importance caches aligned one-to-one with attention keys.

`model.generate(..., attention_mask=...)` also supports padded variable-length prompt batches by compacting each recurrent prompt before decoding.

## Evaluation

```bash
iarmx-eval --model checkpoints/debug/final --pairs 16
```

The bundled evaluator includes a minimal key-value recall test. Publishable comparisons should additionally include MQAR, RULER, Needle-in-a-Haystack, LongBench, held-out perplexity, and parameter/FLOP-matched Transformer and original-IARM baselines. See `docs/EXPERIMENTS.md`.

## IARM versus IARM-X

Retained from the original IARM design:

- decoder-only RMSNorm / RoPE / SwiGLU stack;
- softmax-gated low-rank algebraic resonance operators;
- positive feature map `ELU(q_tilde) + 1`;
- numerator/denominator coordinatewise causal memory;
- causal depthwise local path;
- residual output gating.

IARM-X extensions in this repository:

- a separate resonance transform for keys;
- finite-rank associative fast memory;
- learned decay, erase and write controls;
- query-dependent fast-memory reads;
- adaptive fusion across slow, fast and local paths;
- exact rolling convolution state;
- resonance-conditioned exact-attention layers;
- recurrent historical-importance bias with cache parity;
- grouped importance aggregation across recurrent layers.

## Performance note

The Python token scan in `IARMXRecurrentBlock` is the semantic reference implementation, not the final 1B+ training kernel. Before claiming throughput superiority over Transformers or other linear/recurrent architectures, implement a fused chunkwise recurrent kernel and benchmark identical hardware, precision, tokens, batch size, context length, parameter count, and training FLOPs.

## Citation

If this architecture is used in a paper, cite the original IARM work separately from the IARM-X hybrid extension. Restrict empirical claims to benchmark results actually reproduced and released with the code.
