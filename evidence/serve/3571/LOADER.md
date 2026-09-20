# #3571, layer 1 — `apr serve` can load a Qwen3.5 model

Host `noah-Lambda-Vector`, RTX 4090, published-equivalent build from this branch with
`--features cuda`, `apr serve run ./Qwen3.5-0.8B-Q4_K_M.gguf --gpu --port 18751`.

## Before (origin/main)

```
=== APR Serve ===
Detected format: GGUF
GGUF loaded: 320 tensors, 46 metadata entries
Building quantized inference model...
error: Model load failed: Failed to build quantized model: Format error:
  Architecture 'qwen35' is the Qwen3.5 hybrid (Gated Delta Net, detected tensor 'blk.0.ssm_a'):
  it runs through `Qwen35Model` (re…
```

## After (this branch)

```
Building quantized inference model...
Model ready: 0 layers, vocab_size=248320, hidden_dim=1024
gpu-layers: requested=all resolved=0 total=0 (backend=cuda)
CUDA optimized model ready
error: Inference failed: Failed to create state: Operation 'create_bpe_tokenizer'
  not supported: Unknown token '<unk>' not in vocabulary
```

`0 layers` is correct and not a regression: `create_base_model` builds the shared base — embeddings,
final norm, `lm_head` — because the hybrid's Gated-DeltaNet layers live in `Qwen35Model`, not in the
dense layer list. It is the same base `apr run` builds for this architecture.

**The load defect in #3571's title is fixed and the error that replaces it is a different one, in a
different component, further along.** That second failure is filed separately: the serve path
hardcodes `"<unk>"` as the BPE unknown token and Qwen3.5's vocabulary has none.

## Why this was invisible

`run_gguf_inference` has carried this exact branch since #3091. The serve path never got it, and no
gate would have noticed: `grep -nE 'batch|serve' scripts/check_model_parity.sh scripts/check_model_ladder.sh`
returns nothing (#3555). Every gate we own is single-stream `apr run`. **No ladder rung, parity
record or dogfood row has ever asked `apr serve` to load a model at all.**
