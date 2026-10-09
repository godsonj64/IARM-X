# IARM-X on 10B tokens for under $50

A multidisciplinary cost analysis of the IARM-X pretraining recipe: physics,
numerical analysis, computational mathematics, statistics, economics and game
theory. Numbers marked **measured** come from this repository on this branch;
numbers marked **estimate** are model-based and must be checked on the target
GPU (section 12 lists which is which).

## 0. Verdict

| Model | Training compute (10B tokens) | Energy floor | Estimated rental cost | Under $50? |
|---|---:|---:|---:|---|
| IARM-X 100M, ctx 512 | 6.15e18 FLOP | 2–19 kWh ≈ **$0.3–3** | **$5–15** | **Yes, with ample margin**, once the recurrent scan is parallel (done on this branch) |
| IARM-X 1.3B, ctx 2048 | 8.11e19 FLOP | 23–46 kWh on H100s ≈ $3.5–7 | $60–113 baseline | Only by stacking several levers (section 10.2), or with free research TPUs |

10B tokens on a 100M model is not "theoretically impossible". The physics
floor (electricity) is about a dollar. Karpathy's llm.c already trained a 124M
GPT-2 on 10B FineWeb tokens for about $20 at 2024 prices. What made IARM-X
*look* impossible was the reference implementation of the recurrence. It runs a
Python loop over sequence positions, issuing about 1,100 small GPU operations
per position per training step. That caps throughput at a few thousand tokens
per second, which puts the 10B-token run at roughly $300–3,500 (section 2).
This branch replaces that loop with an exact chunkwise-parallel algorithm
(section 3).

---

## 1. First-principles cost model

$$
\text{cost} \;=\; \underbrace{\frac{F_{\text{tok}}\cdot D}{P_{\text{peak}}\cdot\text{MFU}}}_{\text{GPU-seconds}}\cdot\frac{\text{price}}{3600}
\;+\;\text{experiments}\;+\;\text{preemption waste}\;+\;\text{data/storage}
$$

where $F_{\text{tok}}$ is training FLOPs per token, $D=10^{10}$ tokens,
$P_{\text{peak}}$ the dense BF16 tensor peak, and MFU the achieved fraction.
`scripts/estimate_cost.py` computes $F_{\text{tok}}$ analytically from any config. Its
forward count agrees with PyTorch's `FlopCounterMode` to 0.5% (8.12M vs 8.08M
MACs/token per recurrent block, **measured**).

**Where the compute goes** (forward MACs per token, `--checkpointing off`):

| Component | 100M (ctx 512) | share | 1.3B (ctx 2048) | share |
|---|---:|---:|---:|---:|
| LM head (50,260 vocab) | 38.6M | **37.7%** | 102.9M | 7.6% |
| SwiGLU FFN | 36.6M | 35.7% | 660.6M | 48.8% |
| q/k/v/out/gate projections | 22.4M | 21.9% | 478.2M | 35.4% |
| resonance operators | 2.2M | 2.1% | 62.9M | 4.7% |
| recurrent scan (chunk 64) | 1.3M | 1.2% | 13.6M | 1.0% |
| exact attention | 0.8M | 0.8% | 25.2M | 1.9% |
| memory/gate heads | 0.7M | 0.7% | 9.0M | 0.7% |
| **Total** | **102.5M** | | **1,352M** | |

Training FLOPs per token are $6\times$ forward MACs ($2\times$ for the forward
pass, $4\times$ for the backward pass). Gradient checkpointing adds another
$2\times$ the layer MACs. That gives 0.615 GFLOP/token for 100M and 8.11
GFLOP/token for 1.3B.

**Two structural facts drive the whole plan:**

1. For the 100M model the **vocabulary projection is the single largest cost**
   (38%). The novel IARM-X machinery (resonance, scan, gates, attention) is
   under 5% of compute once the scan is parallel. The architecture is cheap; the
   GPT-2 vocabulary is not.
2. Recurrent layers cost the same per token at any context length, and only 2
   of the 8 layers are attention. Training at 2048 instead of 512 context adds
   only about 2% FLOPs to the 100M model (102.5M → 104.8M MACs/token), so context length is a quality knob,
   not a cost knob.

**Throughput required for $50** ($= D \cdot \text{price} / (3600\cdot 50)$):

| GPU (representative price, Sept–Oct 2026) | $/h | tok/s needed for $50 | 100M tok/s at 25–40% MFU (estimate) | 100M cost |
|---|---:|---:|---:|---:|
| RTX 3090 | 0.15 | 8,333 | 29k–46k | $9–14 |
| RTX 4090 | 0.30 | 16,667 | 67k–107k | $8–12 |
| RTX 5090 | 0.35 | 19,444 | 85k–136k | $7–11 |
| H100 PCIe, interruptible | 0.90 | 50,000 | 307k–492k | $5–8 |
| H100 SXM, on-demand | 1.74 | 96,667 | 402k–643k | $8–12 |

Reproduce: `python scripts/estimate_cost.py --config configs/iarmx_100m_pretrain.yaml --checkpointing off`
(pass `--price NAME=USD` with live prices; marketplace prices vary about 3× within one GPU model).

---

## 2. Why the shipped code cannot hit $50

**Measured** non-view ATen operations per training step on the 100M config
(T=512, fwd+bwd). On a GPU each is roughly one kernel launch.

| Scan | checkpointing off | checkpointing on (shipped config) |
|---|---:|---:|
| per-token loop (`scan_impl: loop`) | 459,475 (897 per position) | 564,691 (1,103 per position) |
| chunkwise (`scan_impl: chunk`, this branch) | 5,161 (10 per position) | 6,943 (14 per position) |
| reduction | **89×** | **81×** |

The loop's operations are *sequential in the sequence dimension*, so a
512-token micro-step issues about 565k launches regardless of batch size. At
5–10 µs of host overhead per eager-mode op, that is 2.8–5.6 s per micro-step.
For the shipped micro-batch (8 packed documents, about 8k tokens), that works
out to **≈1.4–2.9k tokens/s**, or 40–80 days for 10B tokens: about $300–600 on a
4090 or $1,700–3,500 on an H100 (**estimate**). `torch.compile` cannot rescue
this, because it would have to unroll 512 iterations × 6 layers.

**Measured on CPU** (4 threads; CPU is GEMM-bound, so these understate the GPU gap):

| Benchmark | loop | chunk | ratio |
|---|---:|---:|---:|
| One 100M recurrent block, fwd+bwd, B=2, T=512 | 1,693–1,834 ms | 168–187 ms | **9.8–10.1×** |
| ops per block step | 79,577 | 856 | **93×** fewer |
| matmul FLOPs per block step | 48.9 GFLOP | 49.8 GFLOP | +1.8% |
| Full 100M model, fwd+bwd, B=2, T=512 | 12.1 s | 2.55 s | **4.7×** |
| Max output difference | — | 4.8e-7 | identical loss |

Reproduce with `python scripts/bench_scan.py --config configs/iarmx_100m_pretrain.yaml`.

**Other cost leaks found in the pipeline:**

| Issue | Where | Effect | Status |
|---|---|---|---|
| Per-micro-step GPU→CPU syncs (`.item()` on device labels, `float(loss)`) | `training/train.py` | The host cannot queue the next kernels while the GPU works; small models become latency-bound | **Fixed** |
| Gradient checkpointing on the 100M model | `iarmx_100m_*.yaml` | +33% layer FLOPs. A 100M model probably fits a 24 GB card without it (untested here) | Recommend `gradient_checkpointing: false` after a memory test |
| Attention Q/K augmented to head dim 65 for the importance bias | `model/attention.py` | PyTorch's flash/memory-efficient SDPA kernels require head dims that are multiples of 8 (and flash requires equal Q/K/V dims), so these layers likely fall back to the math kernel | Pad the extra coordinate to 72, or use FlexAttention `score_mod`. Verify with `torch.nn.attention.sdpa_kernel` |
| Documents truncated at `4 × seq_len` = 2048 tokens | `data/pretrain.py` | Long-document tails are discarded, so `sample-10BT` cannot supply 10B unique tokens and the loader silently starts a second pass | **Fixed**: `scripts/pretokenize.py` keeps whole documents; streaming splits long documents into pieces |
| `--resume` restarts the data stream from the beginning | `training/train.py` | On preemptible instances every restart replays already-seen data | **Fixed**: exact data position and RNG state in every checkpoint (section 7.4) |

---

## 3. Computational mathematics: an exact parallel recurrence (implemented)

### 3.1 Slow memory is a prefix sum

$N_t=N_{t-1}+\phi_t\odot v_t$ and $D_t=D_{t-1}+\phi_t$ have no decay, so
$N_t = N_0 + \sum_{i\le t}\phi_i\odot v_i$: a `cumsum` in FP32. That is a single
parallel prefix-sum operation.

### 3.2 Fast memory is a gated delta rule with a decoupled write

The fast memory update

$$S_t=\gamma_tS_{t-1}-e_t\,k_t(k_t^\top S_{t-1})+w_t\,k_tv_t^\top$$

can be rewritten with a "pseudo-value" $\delta_t$:

$$S_t=\gamma_tS_{t-1}+k_t\delta_t^\top,\qquad \delta_t=w_tv_t-e_t\,S_{t-1}^\top k_t .$$

This is the gated DeltaNet recurrence with one difference: the write strength
$w_t$ is independent of the erase strength $e_t$. Inside a chunk of length $C$
starting from $S_0$, let $G_t=\sum_{i\le t}\log\gamma_i$ (so the decay product
is $\Gamma_t=e^{G_t}$). Substituting the unrolled $S_{t-1}$ into $\delta_t$
gives a unit lower-triangular linear system. This is the UT/WY representation
of the Householder-like product $\prod_t(\gamma_tI-e_tk_tk_t^\top)$:

$$(I+L)\,\Delta = U-\operatorname{diag}(e_t\Gamma_{t-1})\,K S_0,\qquad L_{tj}=e_t\,e^{G_{t-1}-G_j}\,(k_t\!\cdot\!k_j)\ \ (j<t).$$

With $\tilde U=(I+L)^{-1}U$ and $\tilde W=(I+L)^{-1}\operatorname{diag}(e\Gamma_{\text{prev}})K$,
both independent of $S_0$ and therefore computed for **all chunks in parallel**:

$$\Delta=\tilde U-\tilde W S_0,\qquad
O=\operatorname{diag}(\Gamma)\,QS_0+(M\odot QK^\top)\,\Delta,\qquad
S_C=\Gamma_CS_0+\big(K\odot\tfrac{\Gamma_C}{\Gamma}\big)^{\!\top}\Delta ,$$

with $M_{tj}=e^{G_t-G_j}$ for $j\le t$. Only these last three lines run
sequentially, once per chunk. The sequential depth falls from $T$ to $T/C$
(512 → 8), and everything else is batched matmuls. No division by $e_t$ is
needed, unlike mapping onto an existing gated-delta kernel. The implementation
is `iarmx/model/scan.py`, selected by `scan_impl: chunk` (default) with
`scan_chunk_size: 64`.

**Stability lemma (control theory).** The transition
$\gamma_tI-e_tk_tk_t^\top$ with $\|k_t\|=1$ has eigenvalues $\gamma_t$
(multiplicity $r-1$) and $\gamma_t-e_t$. For any $\gamma_t\in[0,1]$ and
$e_t\in(0,1)$ they lie in $(-1,1]$; with the default bounds
$\gamma_t\in[0.9,0.9999]$ they lie in $(-0.1,1)$. The state is therefore
non-expansive, every decay ratio in the algorithm is at most 1, and forward
substitution mirrors a contractive recurrence. The worst-case test
(identical keys, $e=0.999$, $\gamma=0.9$, $T=256$) matches the loop to 1e-4.

**Cost of the extra work.** The intra-chunk $O(C^2)$ terms add
$h\,[2Cr + C(d_h+r)/2 + Cd_h + 3rd_h]$ MACs per token: 1.2% of the 100M
model's compute (**measured** +1.8% matmul FLOPs per block). For contexts much
longer than 2k, the remaining $T/C$-step loop is an affine recurrence and can
be replaced by a Blelloch associative scan with $O(\log(T/C))$ depth.

### 3.3 Fused resonance contraction

$\sum_o g_oU_oV_o^\top x$ was computed by materializing all $O$ per-operator
responses $[B,T,H,O,D]$. By associativity it is one contraction over the joint
index $(o,r)$: `einsum("bthor,hodr->bthd", low * gates, U)`. This avoids an
activation $O$ times the size of $x$ (4× here) in both forward and saved-for-backward.

### 3.4 Verification

`tests/test_scan.py` (11 tests):

- the chunk scan against a literal per-token recurrence, for chunk sizes {1, 3, 8, 16, 64}, $T=37$ (not divisible), and a nonzero initial state, with outputs, final state and gradients for all seven inputs;
- the worst-case conditioning test above;
- the prefix-sum slow memory;
- full-model loop vs chunk forward and backward, with all parameter gradients;
- cached decoding (token-by-token and 5-token pieces) against a full-sequence pass with multi-chunk scans;
- the fused resonance contraction.

The full suite is 35/35 passing.

### 3.5 Next step: a fused kernel

A Triton kernel modeled on flash-linear-attention's `chunk_gated_delta_rule`
would fuse the per-chunk solve and products. The only change from the
upstream kernel is the right-hand side of the triangular solve ($w_tv_t$
instead of $\beta_tv_t$). Reusing the upstream kernel unchanged is possible via
$\beta_t=e_t/\gamma_t$, $v'_t=v_tw_t\gamma_t/e_t$, but that divides by $e_t$
and is ill-conditioned when the erase gate closes. Fork the kernel rather than
reparametrize.

---

## 4. Numerical analysis

| Finding | Evidence | Consequence | Status |
|---|---|---|---|
| `torch.einsum` on FP32 inputs returns **BF16** under autocast | **measured** on CPU autocast. einsum decomposes into `bmm`, which is also on CUDA's autocast list, so the same is expected on GPU | The loop's "FP32 persistent state" receives BF16-rounded products during mixed-precision training | Chunk path disables autocast for the scan, so it is truly FP32 |
| Decay gate $0.9+0.0999\,\sigma(z)$ evaluated in BF16 takes only **27 distinct values** | **measured** | Memory timescales $1/(1-\gamma)$ jump 128 → 256 → ∞; the configured max of 0.9999 is unreachable | `_decay` now squashes in FP32 |
| FP8 matmuls | NVIDIA forum: RTX 4090 measures ~330 TFLOP/s FP8 with FP32 accumulate vs 165 BF16 | Up to 2× on GEMMs on consumer cards. End-to-end gains are smaller for $d=768$ GEMMs (scaling/casting overhead) | Try `torchao` float8 on linear layers only. Keep the scan, norms and LM head in BF16/FP32. Expect ~1.1–1.3× (**estimate**) |

The stability lemma (3.2) is also why BF16 *inputs* to a future fused scan
kernel are safe: errors are not amplified by the recurrence. Only the
accumulators need FP32.

---

## 5. Physics

### 5.1 The energy floor: what you are actually paying for

Landauer's bound ($kT\ln2\approx2.9\times10^{-21}$ J per erased bit at 300 K)
puts the thermodynamic floor for all $6\times10^{18}$ FLOPs of the 100M run at
a few joules, even allowing ~100 bit erasures per FLOP. An H100 SXM spends
about $7\times10^{-13}$ J per FLOP at peak (989 TFLOP/s at 700 W), roughly 8
orders of magnitude more per FLOP. The binding physical limit is therefore
device efficiency, not thermodynamics. At 40% MFU the 100M run needs **3.0 kWh** (H100) to **11.6 kWh**
(4090), which is **$0.45–1.75 of electricity** at $0.15/kWh. The 1.3B run
needs 23–46 kWh on H100s.

Everything above that is *capital rent* (GPU depreciation and the
provider's margin) multiplied by *inefficiency* (MFU, preemption, failed
runs). The plan attacks those two terms, not physics. If you **own** a GPU,
your marginal cost is the energy line alone: about $2–3 for the 100M run on a
4090.

### 5.2 Roofline and latency: why small models are launch-bound

The chunk path issues 5.2–6.9k ops per 512-token micro-step regardless of
batch size. At 5–10 µs each, that is 26–69 ms of host time. The GPU time for
$n$ tokens is $n\cdot F_{\text{tok}}/(P\cdot\text{MFU})$: about 1.55 µs/token
on an H100 at 40%, but 9.3 µs/token on a 4090. Keeping the GPU busy needs
**≥ ~17–45k tokens per micro-step on an H100, but only ~3–7k on a 4090**.
Consumer cards are therefore *less* sensitive to framework overhead for this
model size. On an H100, use `torch.compile`, which fuses the elementwise ops,
plus large micro-batches, ideally with CUDA graphs.

### 5.3 Statistical mechanics of SGD: temperature and critical batch size

SGD behaves like Langevin dynamics at temperature $\propto\eta/B$. The
gradient noise scale defines a critical batch size above which extra batch
buys no extra progress. Kaplan et al.'s fit is
$B_{\text{crit}}(L)\approx2\times10^8/L^{4.76}$ tokens: ≈0.27M tokens at
$L=4.0$ and ≈0.68M at $L=3.3$. The shipped 65k-token step is far below that,
which is token-efficient. You can raise it to about 0.25M tokens for
throughput (fewer optimizer steps, larger micro-batches) without wasting data.
Do not go above about 0.5M.

### 5.4 Annealing: use a warmup-stable-decay schedule

The final learning-rate decay is literally annealing: most of the loss drop
happens in the cooldown. A warmup-stable-decay (WSD) schedule (constant LR,
then a 10–20% linear cooldown) yields a finished model from *any* stable-phase
checkpoint by running a short cooldown branch. It also combines well with
checkpoint averaging. Cosine decay, as shipped, ties quality to completing the
planned horizon. Section 7.5 explains why that matters economically.

### 5.5 Mean-field theory: tune once, cheaply (μP)

Maximal-update parametrization (Tensor Programs) makes optimal learning rates
and initializations width-invariant. Tune on a $d=256$ proxy (≈20M MACs/token,
half of it the LM head; 0.2B tokens ≈ 2.4e16 FLOP ≈ 8–10 minutes on a 4090 ≈
**$0.05 per trial**), then transfer zero-shot to $d=768$ and $d=2048$. A
20-trial sweep costs about $1. Without this, hyperparameter search, not the
final run, is what usually blows a $50 budget.

### 5.6 Renormalization and multigrid: progressive growth

Coarse-to-fine training (train a shallow model, then stack its layers into a
deep one, as in G_stack) is the multigrid idea applied to depth. Du et al.
(NeurIPS 2024) report reaching a 7B baseline's loss with 194B instead of 300B
tokens (54.6% speedup). Under a *literal* "process 10B tokens" constraint, the
saving is only the cheap early phase: about 8% for 100M, where 38% of compute
is the LM head, and 10–17% for 1.3B. Under a "match 10B-token quality"
constraint, it is up to about 1.5×.

---

## 6. Information theory and linguistics

**Zipf's law and the LM head.** Token frequencies are heavy-tailed, so the top
~8k GPT-2 tokens carry roughly 80–90% of occurrences (measure on your shards).
An adaptive softmax (a full head over the frequent tokens plus low-rank tail
clusters) cuts the 100M model's LM head from 38.6M to about 7–8M MACs/token.
That is **~1.4× cheaper per token overall** (**estimate**), and the largest
single algorithmic lever for this model size. Trade-offs: rare-token quality,
and it changes the model, so baselines must match. A fused linear-cross-entropy
kernel (e.g. Liger, Cut Cross-Entropy) gives no FLOP savings but removes the
[tokens × 50k] logits memory, which is what lets you drop gradient checkpointing.

**Pre-tokenize once, on CPU.** 10B GPT-2 tokens as `uint16` take 20 GB.
`scripts/pretokenize.py --num-proc N` does it in one parallel pass on any
multi-core machine, and CPU hours cost a few cents (or nothing on a free
notebook), so the GPU never waits on tokenization. Training then reads fixed
windows (`dataset: memmap`): no truncation, no padding, fixed-shape batches
(no `torch.compile` recompiles from the packing collator's variable row
count), and a deterministic order.

**Soft labels carry more bits per token, but the teacher costs too much.**
Distillation gives the student a full distribution per position instead of one
index. But a teacher forward pass costs $2N_{\text{teacher}}$ FLOPs per token.
For any teacher larger than the student, labeling 10B tokens costs more than
training the student. Rejected for this budget.

---

## 7. Economics

### 7.1 Unit economics

$\$/\text{token}=\text{price}/(3600\cdot\text{tok/s})$. The target is
$\$5\times10^{-9}$ per token. Compare GPUs by **peak TFLOP-hours per dollar**,
then discount by measured MFU:

| Market (representative 2026 prices) | peak BF16 PFLOP-h per $ |
|---|---:|
| RTX 4090 at $0.14 (marketplace floor) | **1.18** |
| H100 PCIe interruptible at $0.67–0.90 | **0.84–1.13** |
| RTX 5090 at $0.30–0.46 | 0.46–0.70 |
| H100 SXM on-demand at $1.74 | 0.57 |
| RTX 4090 at $0.30 (typical) | 0.55 |
| RTX 3090 at $0.15 | 0.47 |

Consumer cards trade in a competitive market where prices sit near marginal
cost (power plus depreciation of a ~$1.6–2k card). Datacenter GPUs carry a
scarcity rent, except in their own interruptible tier. The cheapest FLOP is
whichever of those two is cheapest *today*: check live prices and re-run
`estimate_cost.py --price`.

### 7.2 Price discrimination and self-selection

Providers sell the same silicon at different prices by interruptibility:
Vast.ai documents interruptible instances as 50–80% cheaper. In
mechanism-design terms this is screening: customers whose jobs tolerate
interruption self-select into the cheap tier. Making the training job
preemption-tolerant (atomic checkpoints, exact data-position resume,
idempotent restarts) is therefore **a purchase of access to the cheap market**
for an engineering cost of about one afternoon.

### 7.3 Optimal checkpointing under preemption (Young–Daly)

With checkpoint cost $\delta$, mean time between interruptions $M$ and restart
cost $R$, the optimal interval is $\tau^*\approx\sqrt{2\delta M}$ and the waste
fraction is $\approx\sqrt{2\delta/M}+R/M$.

For 100M: a checkpoint with AdamW state is about 1.2 GB, so $\delta\approx5$ s
on NVMe. Assuming $M=2$ h and $R=5$ min, $\tau^*\approx4.5$ min and the waste
is about **8%**, against a 50–80% discount. That is strongly positive expected
value. Set `save_every` to $\tau^*$ × steps/s, and write checkpoints from a
background thread to push $\delta$ toward 0.

### 7.4 Exact resume is part of the price

Section 2's `--resume` used to replay the stream from the start. Under
preemption that is a hidden tax: you pay for duplicate tokens and never see the
tail of the data. Checkpoints now carry the exact data position and every
rank's RNG state, are written atomically, and `--resume auto` (used by
`scripts/train_100m.sh`) picks the newest one, so rerunning the same command
after a preemption continues the run. With pre-tokenized shards the position is
one integer (global windows consumed), so the data continues exactly even on a
different number of GPUs, and the learning rate follows tokens seen, so it does
not jump. Streaming and SFT runs store `StatefulDataLoader` state and resume
exactly with the same GPU count. With the same GPU count a resumed run matches
the uninterrupted one bit-for-bit on CPU (`tests/test_resume.py`); on GPUs it
sees the same data and RNG streams but is subject to ordinary CUDA kernel
nondeterminism. Either way a preemption costs only the work since the last
checkpoint.

### 7.5 Real options and the value of information

- **Pilot (≈1% of budget).** 100M tokens (~$0.10–0.50) measures the real tok/s,
  MFU, memory headroom and early loss slope before committing. This is the
  cheapest information you will buy.
- **Kill rule.** Fit $L(D)=E+B\,D^{-\beta}$ to the first ~10% of the run.
  Stop if the extrapolated final loss is worse than a matched baseline's
  extrapolation.
- **WSD as an American option.** With a constant-LR stable phase you can stop
  at any time (budget exhausted, preempted for good) and still get a fully
  annealed model after a short cooldown. Cosine forfeits that option.

### 7.6 Allocating a fixed compute budget between N and D

Chinchilla-optimal allocation for the 100M@10B compute ($6.15\times10^{18}$ FLOP,
$D\approx20N$) is $N\approx230$M on ≈4.5B tokens, which gives lower loss for
the same money. 100M@10B is a deliberately over-trained (inference-cheap) model;
1.3B@10B is under-trained (Chinchilla wants ~26B tokens). If the goal is the
best model for $50, rather than "10B tokens" specifically, a ~200M IARM-X on
~5B tokens is the economically efficient point.

### 7.7 Subsidized compute (zero marginal price)

- **Google TPU Research Cloud (TRC).** Free Cloud TPU quota for accepted
  researchers. The grant is temporary, and participants are expected to publish
  or open-source the work, which matches a project that already has a preprint.
  Only small GCP VM and storage charges apply. The chunk scan matters here too:
  static shapes and matmuls are XLA-friendly, while the per-token loop is not.
  This is the realistic route to 1.3B under $50.
- **Kaggle and Colab notebooks.** Free GPU/TPU time with weekly quotas
  (published figures disagree; check your account). A Kaggle TPU v5e-8 would
  be about 1.6 PFLOP/s peak, enough for the 100M run in a few sessions *if*
  ported to `torch_xla` or JAX. Use one account; multi-accounting violates the
  terms of service.

### 7.8 A $50 portfolio for the 100M run

| Item | Budget |
|---|---:|
| Pre-tokenization (CPU) | $0–0.50 |
| μP proxy sweep (~20 trials) | ~$1–2 |
| Pilot run (1%) | ~$0.50 |
| Main pretraining run (interruptible 4090/5090/H100) | ~$6–15 |
| UltraChat SFT (~0.2B tokens) | ~$0.20–0.50 |
| Matched Transformer baseline (for a publishable comparison) | ~$6–15 |
| **Reserve** (failed runs, price spikes) | ≥ $15 |

---

## 8. Game theory

**8.1 Bidding in the interruptible auction.** On Vast.ai, among interruptible
renters the highest bid runs, on-demand renters outrank all bids, and outbid
instances pause. Your preemption rate $\lambda(b)$ falls as your bid $b$ rises,
so the effective price is

$$p_{\text{eff}}(b)=\frac{b}{1-\big(\sqrt{2\delta\lambda(b)}+\lambda(b)R\big)} .$$

The best response is to minimize $p_{\text{eff}}$, not $b$. When $\delta$ and
$R$ are small (async checkpoints, exact resume) the denominator stays near 1
even at high preemption rates, so the optimal bid sits just above the low
quantile of recent clearing prices. Raise the bid only when measured
preemptions push the waste term toward the discount. The cheaper your
interruptions, the lower you can bid: engineering for preemption is what turns
the auction in your favor.

**8.2 Host selection as a multi-armed bandit.** Marketplace hosts with the same
GPU differ in real throughput (PCIe lanes, CPU, disk, thermal throttling). Use
explore-then-commit: rent 3–5 candidate hosts for 5 minutes, run the pilot
step loop, keep the best tok/s per dollar, and release the rest.

**8.3 Hyperparameter search as best-arm identification.** Successive halving
(Hyperband) on μP proxies spends the sweep budget on promising configurations
and stops the rest early. It is near-optimal for a fixed budget.

**8.4 Escaping the interconnect premium (DiLoCo).** Fast interconnect is a
complementary good that datacenters price in. DiLoCo-style training
(independent inner steps, outer synchronization every few hundred steps)
communicates hundreds of times less. That lets you pool the cheapest *isolated*
GPUs, even across hosts. It is irrelevant for 100M (one GPU suffices) but
useful for 1.3B on consumer cards.

---

## 9. Optimization theory: more quality per token

These do not lower the cost of *processing* 10B tokens. They raise what the
10B tokens buy, or reach the same quality with fewer tokens.

- **Muon** orthogonalizes each matrix update via a Newton–Schulz polar
  iteration (steepest descent under the spectral norm). Moonshot reports about
  2× compute efficiency vs AdamW at scale. It has also driven several
  nanoGPT-speedrun records. Use it for 2-D weights, including the per-$(h,o)$
  resonance $U,V$ factors, and AdamW for embeddings, gains and biases. Its
  Newton–Schulz overhead is about 1–4% of step FLOPs at $d=768$, depending on
  tokens per step. It also keeps
  one optimizer buffer instead of two, which frees memory for larger batches.
- **Checkpoint/weight averaging** (EMA, latest-weight averaging) gives a free
  late-training loss reduction. It pairs naturally with WSD.
- **Warm start of the tied embedding from GPT-2-small** (same tokenizer, same
  $d=768$, MIT license) skips learning 38.6M parameters. Useful for a product,
  but it confounds architecture comparisons, so give baselines the same start
  or skip it.

---

## 10. Concrete plans

### 10.1 100M on 10B tokens: expected total $10–20 including experiments

1. Use this branch (`scan_impl: chunk`). Set `gradient_checkpointing: false`
   if the memory test passes, and keep `compile: true`.
2. Pre-tokenize FineWeb-Edu `sample-10BT` on a CPU machine
   (`scripts/pretokenize.py`), then train with
   `configs/iarmx_100m_pretrain_memmap.yaml` (`MEMMAP=1 bash scripts/train_100m.sh`).
3. Fix the attention head dim (pad 65 → 72 or use FlexAttention) and confirm
   which SDPA backend runs.
4. Run a μP proxy sweep (~$1–2), then a pilot (1%). Pick the GPU by measured
   tok/s per dollar (8.2).
5. Main run on an interruptible instance, with a WSD schedule and checkpoints
   every $\tau^*$ (7.3). Optional: Muon, FP8, adaptive softmax (each changes
   the experiment; ablate).

### 10.2 1.3B on 10B tokens: possible only by stacking levers

Starting from $60–113 (H100 at 35–45% MFU, no checkpointing, 80 GB), the
factors multiply:

| Lever | Factor (estimate) | Running cost |
|---|---:|---:|
| Baseline: H100 PCIe interruptible at $0.90, 45% MFU | — | ~$60 |
| FP8 linear layers (d=2048 GEMMs) | ×0.80–0.85 | ~$50 |
| Progressive depth growth (first ~15–25% of tokens at 1/4 depth) | ×0.83–0.90 | ~$40–46 |
| Bid near the interruptible floor ($0.67) | ×0.75 | ~$30–34 |
| Preemption waste (7.3) | ×1.05–1.10 | **~$32–38** |

This lands under $50 only if the 45% MFU and the low interruptible price both
materialize. Budget no slack for failed runs. The **robust** routes are
TRC TPUs (~$0–10), or owned hardware: energy only, about $20–26 on a 4090,
but 12–16 days of wall clock and an 8-bit or single-buffer optimizer to fit 24 GB.

---

## 11. Considered and rejected

| Idea | Why not |
|---|---|
| Knowledge distillation | Labeling 10B tokens with any larger teacher costs more than training the student (section 6) |
| Selective backprop / token dropping | Causal recurrence and attention still need a full trunk backward. Only the LM-head backward shrinks |
| 2:4 structured sparsity | ~1.1–1.3× on FFN GEMMs at this width, and it changes the model |
| MoE / mixture-of-depths | Better quality per FLOP, but changes the 100M spec and adds routing risk |
| Reversible/symplectic layers, Landauer-limited computing | Memory or energy savings far below anything that matters at this scale |
| Multiple free-tier accounts | Violates provider terms. Not a legitimate cost reduction |
| Data curriculum by difficulty | Weak, inconsistent evidence at this scale. Not worth budget |

---

## 12. What is measured and what is estimated

| Claim | Status |
|---|---|
| Chunk scan equals loop scan (outputs, state, gradients, cached decoding) | **Measured**: 11 new tests, 35/35 suite |
| 81–93× fewer ops, ~10× block speedup, 4.7× model speedup on CPU | **Measured** (CPU, 4 threads) |
| FLOPs/token model | **Measured** against `FlopCounterMode` (0.5%) |
| Autocast einsum → BF16; 27-value BF16 decay | **Measured** |
| GPU tok/s for the loop path (≈1.4–2.9k) and chunk path (25–40% MFU) | **Estimate**: run `scripts/bench_scan.py` on the GPU, then a pilot |
| SDPA fallback for head dim 65 | **Estimate** from PyTorch kernel constraints; verify with `sdpa_kernel` |
| FP8, adaptive softmax, growth, Muon factors | **Estimates** from the literature; ablate |
| Prices | Third-party snapshots, Sept–Oct 2026; they move daily |

## References

- Karpathy, llm.c GPT-2 (124M) reproduction on 10B FineWeb tokens, 90 min / ~$20 on 8×A100 (2024).
- Yang, Wang, Zhang, Shen, Kim. *Parallelizing Linear Transformers with the Delta Rule over Sequence Length.* NeurIPS 2024.
- Yang, Kautz, Hatamizadeh. *Gated Delta Networks: Improving Mamba2 with Delta Rule.* ICLR 2025.
- Bischof & Van Loan, *The WY representation for products of Householder matrices* (1987); Joffrain et al., *Accumulating Householder transformations, revisited* (2006).
- Kaplan et al. *Scaling Laws for Neural Language Models* (2020); McCandlish et al. *An Empirical Model of Large-Batch Training* (2018).
- Hoffmann et al. *Training Compute-Optimal Large Language Models* (2022).
- Yang & Hu et al. *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer* (μP, 2022).
- Du et al. *Stacking Your Transformers: A Closer Look at Model Growth for Efficient LLM Pre-Training.* NeurIPS 2024.
- Liu et al. (Moonshot AI). *Muon is Scalable for LLM Training.* arXiv:2502.16982 (2025).
- Hägele et al. *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations* (WSD, 2024).
- Young (1974); Daly (2006): optimal checkpoint intervals.
- Grave et al. *Efficient softmax approximation for GPUs* (adaptive softmax, 2017).
- Douillard et al. *DiLoCo: Distributed Low-Communication Training of Language Models* (2023).
- Li et al. *Hyperband* (2018).
- Vast.ai documentation: rental types and interruptible instances.
- Google TPU Research Cloud: sites.research.google/trc.
