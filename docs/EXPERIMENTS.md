# Controlled Experiments

Use identical token streams, optimizer hyperparameters, effective batch size, training tokens, and approximately matched non-embedding parameter counts.

Recommended ablations:

1. Full causal Transformer.
2. Original IARM: resonance + coordinatewise slow memory + local convolution.
3. Original IARM with 3:1 exact attention layers.
4. IARM with fast associative memory but no exact attention.
5. IARM-X without resonance on keys.
6. IARM-X without erase term.
7. Full IARM-X.

Report held-out next-token loss/perplexity, MQAR-style associative recall, Needle-in-a-Haystack, RULER, LongBench where feasible, generation throughput, prefill throughput, peak memory, KV-state bytes/token, and recurrent-state bytes/layer.


## 100M controlled experiment

Use `configs/iarmx_100m_pretrain.yaml` followed by `configs/iarmx_100m_sft.yaml`.
The model has 100,202,256 parameters and an exact `R,R,R,A,R,R,R,A` mixer schedule.
Pretraining terminates at a global 10B supervised-token budget, independent of DDP world size.
UltraChat SFT runs for two dataset epochs and must initialize from the pretraining final model.
For a publishable claim, train a parameter/FLOP-matched Transformer and an original-IARM baseline on the same tokenizer, sample order, token budget, optimizer family, and evaluation harness.
