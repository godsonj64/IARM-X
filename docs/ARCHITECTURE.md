# IARM-X Architecture

## Source-derived IARM core

The original IARM mechanism uses a softmax-gated mixture of low-rank operators to correct a query, then applies a strictly positive feature map and a coordinatewise normalized causal memory. In per-head notation:

\[
\tilde q_t=q_t+\frac{1}{\sqrt R}\sum_o g_{t,o}U_oV_o^\top q_t,
\]

\[
\phi_t=\mathrm{ELU}(\tilde q_t)+1,
\qquad
N_t=N_{t-1}+\phi_t\odot v_t,
\qquad
D_t=D_{t-1}+\phi_t,
\]

\[
m_t^{slow}=N_t/(D_t+\epsilon).
\]

The repository keeps that slow-memory path intact.

## IARM-X additions

### 1. Resonance-conditioned keys

A second resonance bank transforms keys. This lets the same algebraic operator family shape the geometry used by both associative memory and exact attention.

### 2. Selective associative fast memory

Each head maintains \(S_t\in\mathbb R^{r\times d_h}\). A normalized memory key \(k_t^m\) retrieves the current prediction, and learned erase/write strengths update the matrix:

\[
\hat v_t=(k_t^m)^\top S_{t-1},
\]

\[
S_t=\gamma_tS_{t-1}-e_t k_t^m\hat v_t^\top+w_t k_t^m v_t^\top.
\]

A separately projected query reads it:

\[
m_t^{fast}=(q_t^m)^\top S_t.
\]

The state remains constant in sequence length.

### 3. Three-timescale fusion

Each recurrent block combines slow IARM memory, fast associative memory, and a local depthwise convolution through a learned per-head simplex gate.

### 4. Sparse exact attention

Every `attention_every` layers, a resonance-conditioned causal attention block performs exact token retrieval. With `attention_every=4`, the schedule is `R,R,R,A`.

## Complexity

For fixed model width, recurrent blocks are linear in sequence length and maintain context-independent recurrent state. Exact attention blocks remain quadratic during full-sequence training and carry a growing KV cache during decoding. Because only a fraction of layers use attention, cache and attention compute are reduced relative to a full Transformer with the same depth.

## Parallel training scan

Slow memory is a prefix sum. Fast memory is a gated delta rule with a write strength decoupled from its erase strength, so each chunk of `scan_chunk_size` tokens reduces to a unit lower-triangular solve (the UT/WY transform) plus batched matrix products; only one small state update per chunk is sequential. `iarmx/model/scan.py` derives and implements this; `scan_impl: loop` keeps the per-token recurrence as the correctness oracle, and `tests/test_scan.py` checks outputs, states and gradients against it. Derivation, stability argument and measurements: [`TRAINING_BUDGET.md`](TRAINING_BUDGET.md#3-computational-mathematics-an-exact-parallel-recurrence-implemented).

## Research warning

The chunkwise scan is exact but is still composed from PyTorch operators rather than a fused production kernel. Benchmarking the architecture fairly at large scale requires a fused Triton/CUDA kernel, parameter/FLOP matching, and controlled training against Transformer and original-IARM baselines.
