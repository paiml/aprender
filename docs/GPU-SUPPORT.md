# GPU vs CPU: which models get real GPU inference

<!-- GENERATED. Do not edit by hand.
     Architectures rendered from crates/aprender-serve/src/capability.rs;
     quantizations read from contracts/apr-model-capability-v1.yaml (#3856).
     Asserted byte-for-byte by the test
     `gpu_support_doc::the_committed_doc_matches_the_capability_gate`.
     Regenerate: APR_WRITE_GPU_SUPPORT_DOC=1 cargo test -p aprender-serve --lib gpu_support_doc
     Issue: #3077 (alfredodeza) -->

A model that is not GPU-eligible is **not broken** — it runs on the CPU. What this
table exists to prevent is the surprise: assuming an RTX 4090 makes any GGUF fast.

## Architectures

| architecture | models | GPU | why not |
|---|---|---|---|
| `llama` | Llama 2/3, TinyLlama, CodeLlama | yes | — |
| `mistral` | Mistral, Mixtral (dense) | yes | — |
| `qwen2` | Qwen2, Qwen2.5 (incl. Coder) | yes | — |
| `qwen3` | Qwen3 dense | yes | — |
| `qwen35` | Qwen3.5 hybrid (Gated DeltaNet) | yes | — |
| `qwen3_moe` | Qwen3 MoE (Qwen3-30B-A3B, Qwen3-Coder-30B-A3B) | yes | — |
| `qwen3_5_moe` | Qwen3.5 MoE (A3B, hybrid Gated DeltaNet) | **refused** | no CUDA forward: hybrid SSM MoE, not run by the qwen3moe forward (#3714) |
| `gemma2` | Gemma 2 | CPU fallback | missing `AttnFinalSoftcap`, `PostAttnFfnNorm` |
| `gemma3` | Gemma 3 | CPU fallback | missing `AttnFinalSoftcap`, `PostAttnFfnNorm` |
| `phi2` | Phi-2 | CPU fallback | missing `GeluMlp`, `LayerNorm` |
| `phi3` | Phi-3 | CPU fallback | missing `LayerNorm` |
| `gpt2` | GPT-2 | CPU fallback | missing `AbsolutePos`, `GeluMlp`, `LayerNorm` |

**`refused` is not `CPU fallback`.** A refusal names the architecture and exits
rather than loading; a fallback runs on the CPU. `apr run` without `--gpu` uses the
CPU path deliberately and works for every row above.

## Quantizations, on the GPU path

| quantization | GPU | note |
|---|---|---|
| F32 | yes | ggml type 0 |
| F16 | yes | ggml type 1 — Opened in the GPU whitelist on the 0.69.1 release branch. This row read `false` with reason "no GPU GEMV kernel" until FALSIFY-CAP-002 caught the disagreement with `gpu_unsupported_quant_qtype` after the kernel landed. |
| Q4_0 | yes | ggml type 2 |
| Q4_1 | yes | ggml type 3 |
| Q5_0 | yes | ggml type 6 |
| Q5_1 | yes | ggml type 7 |
| Q8_0 | yes | ggml type 8 |
| Q8_1 | no | ggml type 9 — no GPU GEMV kernel |
| Q2_K | no | ggml type 10 — no GPU GEMV kernel |
| Q3_K | no | ggml type 11 — no GPU GEMV kernel |
| Q4_K | yes | ggml type 12 — the release matrix's quant |
| Q5_K | yes | ggml type 13 |
| Q6_K | yes | ggml type 14 |
| Q8_K | no | ggml type 15 — no GPU GEMV kernel |
| IQ2_XXS | yes | ggml type 16 |
| IQ2_XS | no | ggml type 17 — no GPU GEMV kernel; IQ also fails the CPU dequant path |
| IQ3_XXS | no | ggml type 18 — no GPU GEMV kernel; IQ also fails the CPU dequant path |
| IQ1_S | no | ggml type 19 — no GPU GEMV kernel; IQ also fails the CPU dequant path |
| IQ4_NL | yes | ggml type 20 |
| IQ3_S | yes | ggml type 21 |
| IQ2_S | no | ggml type 22 — no GPU GEMV kernel; IQ also fails the CPU dequant path |
| IQ4_XS | yes | ggml type 23 — Opened in the GPU whitelist on the 0.69.1 release branch. The CPU dequant path has existed all along (`iq_dispatch.rs`); only the GPU side was shut. |
| BF16 | yes | ggml type 30 — #3908: GEMV kernel measured BIT-EXACT against the CPU decoder on device (RTX 4090 sm_89) - 0 ULP over 64 rows on integer-exact data, where every partial sum is exact in f32 so summation order cannot matter, and 1.468e-5 worst relative on ordinary bf16 values. The only type in this table measured exactly rather than within a tolerance, because bf16 decoding is `bits << 16` reinterpreted and rounds nothing. Planted faults RED first: shift 8 not 16, byte-swapped halfword, row stride k not k*2. |
| IQ1_M | no | ggml type 29 — no GPU GEMV kernel; IQ also fails the CPU dequant path |
| I8 | no | ggml type 24 — integer storage type, not a weight quantisation for inference |
| I16 | no | ggml type 25 — integer storage type, not a weight quantisation for inference |
| I32 | no | ggml type 26 — integer storage type, not a weight quantisation for inference |
| I64 | no | ggml type 27 — integer storage type, not a weight quantisation for inference |
| F64 | no | ggml type 28 — not named by the GPU whitelist |
| TQ1_0 | no | ggml type 34 — not named by the GPU whitelist |
| TQ2_0 | no | ggml type 35 — not named by the GPU whitelist |
| MXFP4 | no | ggml type 39 — not named by the GPU whitelist |
| NVFP4 | no | ggml type 40 — not named by the GPU whitelist |
| Q1_0 | no | ggml type 41 — not named by the GPU whitelist |
| Q2_0 | no | ggml type 42 — not named by the GPU whitelist |

An unsupported quantization on an otherwise GPU-eligible architecture falls back
to the CPU. See #3850 for the case where it was silently reinterpreted instead.
