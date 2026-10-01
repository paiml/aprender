---
spike: 017
idea: llm-decision-classifier
name: kev-rust-forward-parity
type: standard
validates: "Given Kev-0.8B trained in Python with its LoRA folded into the base and exported to GGUF, when a decision row runs through upstream's Qwen35Model extended with a hidden-state readout plus the pointer head in Rust, then option probabilities match Python Kev fp32 to ~1e-5 and CPU latency is measured"
verdict: VALIDATED
related: [015, 016]
tags: [kev, qwen3.5, deltanet, parity, gguf, handoff, latency]
---

# Spike 017: Kev forward in Rust, parity with Python

## What This Validates
The Python-trains / Rust-serves handoff end to end: a Kev checkpoint (adapter + head) becomes an artifact the Rust
path loads, and the Rust decision forward reproduces Python's fp32 probabilities.

## Research
- Kev's row form (`kev/model.py::forward_rows_batch`): each question is one causal row `state ids + branch ids`,
  positions `0..L-1`, read from `last_hidden_state` (after the final RMS norm) at `<decide>` and each `</opt>`;
  pointer logits `k(h_opt)·q(h_decide)/16`, served probabilities `softmax(z / T)` with the checkpoint's T (2.406).
- Upstream `Qwen35Model` (`aprender-serve/src/gguf/inference/forward/forward_qwen35.rs`) is GGUF-only,
  token-at-a-time, logits-only; its parity bar is llama.cpp argmax with a 0.25-logit near-tie allowance. Its own
  comment: *"there is no batched prefill for the hybrid"*. Fused matmul dispatches F32/F16/BF16/Q8_0 weights.
- llama.cpp `convert_hf_to_gguf.py` (master `d2e5458`) registers `Qwen3_5ForCausalLM`.

| Handoff step | Where | Tool | Output |
|---|---|---|---|
| merge LoRA (fp32, exact) | Python back office | PEFT `merge_and_unload` | `Qwen3_5ForCausalLM` safetensors, 2.8 GB |
| export head | Python | `safetensors` | `head.safetensors` 2 MB, temperature in metadata |
| convert | Python | llama.cpp converter, 22 s | GGUF: f32 3.0 GB / bf16, f16 1.5 GB / q8_0 0.8 GB |
| forward + head | Rust | upstream `Qwen35Model` + 2 small patches + 40-line head | probabilities |

## How to Run
```bash
cd ../015-kev-vs-setfit-few-shot/vendor/kev && uv run python ../../../017-kev-rust-forward-parity/tools/export_and_oracle.py
cd ../../../017-kev-rust-forward-parity/vendor/llama.cpp   # git clone --depth 1 https://github.com/ggml-org/llama.cpp
uv run --python 3.12 --with ./gguf-py --with "transformers>=5" --with torch --with numpy --with sentencepiece \
  --with safetensors --with protobuf python convert_hf_to_gguf.py ../../models/kev-0.8b-merged --outtype f32 \
  --outfile ../../models/kev-0.8b-merged-f32.gguf
cd ../.. && CARGO_TARGET_DIR=<repo>/target cargo build --release     # needs the spike-016 worktree + upstream-qwen35-hidden-and-mtp.patch
<repo>/target/release/kev-rust-forward-parity models/kev-0.8b-merged-f32.gguf models/kev-0.8b-head.safetensors fixtures/kev-0.8b_fixture.json
<repo>/target/release/prefill_bound
```

## Investigation Trail
1. **Two upstream patches were needed** (`upstream-qwen35-hidden-and-mtp.patch`, committed `12976da0a` on the local
   `spike/016-upstream-sync` branch):
   - `forward_single_qwen35_hidden`: the forward stopped at the final RMS norm; `forward_single_qwen35` now calls it
     and applies `lm_head`, so existing callers are unchanged (41/41 qwen35 tests pass).
   - **MTP depth defect**: llama.cpp writes `qwen35.block_count = 25` and `nextn_predict_layers = 1` for the
     24-layer 0.8B (tensors stop at `blk.23`); llama.cpp's runtime subtracts the MTP block, upstream's loader did not
     and failed with `Tensor 'blk.24.attn_norm.weight' not found`. Any Qwen3.5 GGUF freshly converted from HF weights
     hits this — an upstream-worthy fix independent of Kev.
2. **Parity on the first run** (f32), no numerical debugging needed.
3. **Latency was the surprise**: 79 ms per token, so 4.8 s for an 85-token decision and 72 s for a 915-token state.
4. **Weight precision is not the lever**: BF16 halves the bytes and buys 1.8×; Q8_0 the same 43 ms/token while its
   RSS rises to **10.9 GB** (above the 3 GB f32 file — the loader appears to hold dequantised owned copies).
   High `sys` time (368 s of 1183 s CPU on the f32 run) points at per-token thread dispatch overhead.
5. **A batched prefill is the lever**: timing the model's 13 projection shapes (498M params) as GEMMs at Kev's row
   lengths with the same trueno BLIS kernels gives the lower bound in `PREFILL-BOUND.md`.

## Results

**Parity (f32 GGUF vs torch fp32 CPU, 10 records / 12 rows, incl. 3-question ticket, delimiter-injection probe,
mixed-script unicode, 915-token state)**

| quantity | worst |
|---|---|
| final hidden, all 87 positions of row 0 | 1.2e-4 abs (hidden rms 2.1) |
| hidden at readouts | 6.3e-5 abs, 5.2e-5 relative |
| raw pointer logits | 2.0e-5 |
| served probabilities | **1.5e-6** |
| argmax | **12/12** |

**Weight dtype (11 rows ≤ 100 tokens, M4 Pro)**

| GGUF | file | peak RSS | ms/token | worst \|Δp\| | argmax |
|---|---|---|---|---|---|
| f32 | 3.0 GB | 7.2 GB | 79 | 1.5e-6 | 11/11 |
| f16 | 1.5 GB | 4.2 GB | 64 | 3.4e-4 | 11/11 |
| bf16 | 1.5 GB | 4.2 GB | 43 | 1.9e-3 | 11/11 |
| q8_0 | 0.8 GB | 10.9 GB | 43 | 1.3e-2 | 11/11 |

**Batched-prefill lower bound (projection GEMMs only; recurrence/attention/norms excluded)**

| row tokens | token-at-a-time today | batched, 1 thread | batched, 14 threads |
|---|---|---|---|
| 85 | 6.7 s | 1.27 s (67 GFLOP/s) | **0.73 s** (116 GFLOP/s) |
| 300 | 23.7 s | 3.92 s (76 GFLOP/s) | **1.80 s** (166 GFLOP/s) |
| 915 | 72.2 s | 11.5 s (79 GFLOP/s) | **1.92 s** (474 GFLOP/s) |

**Verdict: VALIDATED ✓ — for correctness and the handoff; latency is the build's main work item.**
- The Python→Rust handoff works with off-the-shelf tools: PEFT merge, llama.cpp converter, upstream loader.
  The only Kev-specific Rust is a 40-line pointer head. Parity 1.5e-6 on probabilities.
- As shipped upstream, the Qwen3.5 CPU path is **not servable for Kev**: 5–7 s for a short decision on an M4 Pro.
- A batched prefill (projections as one GEMM per layer over the row, DeltaNet recurrence stays sequential — it is
  16 heads × 128² per token) is the required work. Its lower bound is 0.7–1.9 s on 14 cores; M-only parallel
  splitting caps short rows at 116 GFLOP/s. MLX on the same machine's GPU does the whole decision in 33 ms.
- **Feed-forward to 019**: CPU inference of Kev-0.8B is on the order of 1 s per decision, and Kev-4B (the size that
  beats SetFit in 015) is ~5× that. Lambda has ≤ 6 vCPU and no GPU.

**Not done**: the batched prefill itself; the Qwen tokenizer in Rust (fixture carries ids; upstream has
`gguf/byte_level_bpe.rs` and Kev's `<|x|>` → `<¦x¦>` escaping must be ported); the state-prefix cache; Kev-4B.
