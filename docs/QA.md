# IARM-X QA Report

This report records the correctness regression suite for the post-QA repository revision.

## Automated suite

```bash
pytest -q
```

Result:

```text
21 passed
```

The suite validates:

1. forward shape and finite causal-LM loss;
2. full-sequence versus one-token cached-logit equivalence;
3. full-sequence versus arbitrary chunked cached-logit equivalence;
4. zero future-token leakage into prefix logits;
5. exact depthwise-convolution rolling cache;
6. historical-importance state cached alongside attention K/V;
7. SDPA execution when historical importance is active;
8. SDPA importance-bias numerics against the explicit manual-attention implementation;
9. FP32 slow/fast persistent states under BF16 model weights;
10. assistant-only shifted SFT targets with no same-position target leakage;
11. SFT padding label masking;
12. pretraining padding masking;
13. pretraining document packing and EOS boundaries;
14. tied embedding/output safetensors round-trip;
15. constant recurrent-state size versus context length;
16. growing sparse-attention cache versus context length;
17. gradient-checkpointed forward/backward equivalence;
18. nonzero gradients for every recurrent importance predictor and attention importance scale;
19. padded variable-length generation equivalence to independent prompt decoding;
20. checkpoint model/optimizer/scheduler/step restoration;
21. exact 1.309B research configuration and DDP map-dataset partitioning.

## Numerical parity measurements

A separate deterministic QA run measured full-vs-cached decoding on three random seeds:

```text
seed 11: max |Δlogit| = 2.3842e-07, mean = 3.7938e-08
seed 12: max |Δlogit| = 2.1234e-07, mean = 3.6690e-08
seed 13: max |Δlogit| = 2.9802e-07, mean = 3.5706e-08
```

Arbitrary chunking produced:

```text
max |Δlogit| = 2.3842e-07
mean |Δlogit| = 2.7366e-08
```

Causal leakage check:

```text
max prefix difference after changing only future tokens = 0.0
```

These differences are at normal FP32 floating-point roundoff scale.

## Persistent-state numerical stability

With model weights converted to BF16, persistent recurrent state tensors remain FP32:

```text
slow_num: float32
slow_den: float32
fast:     float32
```

A controlled 300-token test with `q=0`, hence `phi=1`, produced exactly:

```text
slow_den = 300.0
```

This directly regression-tests the prior BF16 cumulative-sum failure mode.

## State scaling

For the tiny 3:1 hybrid QA configuration:

```text
context T=1:  recurrent state = 2688 elements, attention state = 132 elements
context T=8:  recurrent state = 2688 elements, attention state = 1056 elements
context T=32: recurrent state = 2688 elements, attention state = 4224 elements
```

The recurrent state is context-independent; sparse attention state grows linearly with cached context.

## Synthetic learnability

A 25-step deterministic overfit check on a fixed synthetic next-token batch reduced loss from:

```text
4.5973 -> 0.6800
```

This is not a quality benchmark; it is a gradient/optimization sanity test.

## Serialization

Tied input/output embeddings now save as one safetensors tensor and are re-tied on load.

Measured round-trip:

```text
max logit difference = 0.0
shared embedding/head storage after load = true
```

## Gradient coverage

A full backward pass through an `R,R,R,A` block group produced no trainable parameter with missing or exactly-zero gradient in the QA model:

```text
dead_grad_params = []
```

Historical importance is aggregated across all recurrent layers before the next attention layer, and the attention importance scale starts nonzero so those predictors receive gradient from the first update.

## Research configuration

`configs/iarmx_1.3b.yaml` is parameter-verified on the PyTorch meta device:

```text
1,309,384,768 parameters = 1.309B
```

## Packaging and static checks

The repository also passes:

```bash
python -m compileall -q iarmx scripts tests
python scripts/smoke_test.py
python scripts/count_params.py --config configs/iarmx_1.3b.yaml --meta
```

Editable installation was verified with:

```bash
pip install -e . --no-deps --no-build-isolation
```

The QA environment did not have network access, so a fresh dependency download from PyPI and live FineWeb/UltraChat integration run could not be executed. Dataset/tokenizer imports are lazy, and the data-rendering/collation logic is covered by local unit tests. GPU-specific NCCL, FlashAttention backend selection, and multi-node DDP remain environment-dependent integration tests rather than CPU QA claims.
