# GPU vs CPU: which models get real GPU inference

<!-- GENERATED. Do not edit by hand.
     Rendered from crates/aprender-serve/src/capability.rs by the test
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
| `qwen3_moe` | Qwen3 / Qwen3.5 MoE (A3B) | **refused** | no CUDA forward at all (#3714) |
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
| F16 | no | ggml type 1 — #3846/#3850 |
| Q4_0 | yes | ggml type 2 |
| Q4_1 | yes | ggml type 3 |
| Q5_0 | yes | ggml type 6 |
| Q8_0 | yes | ggml type 8 |
| Q4_K | yes | ggml type 12 — the release matrix's quant |
| Q5_K | yes | ggml type 13 |
| Q6_K | yes | ggml type 14 |
| Q5_1 / Q8_1 / Q2_K / Q3_K / Q8_K | no | no GPU GEMV kernel |
| IQ2_XXS / IQ3_* / IQ4_NL / IQ4_XS | no | no GPU GEMV kernel; IQ also fails the CPU dequant path |

An unsupported quantization on an otherwise GPU-eligible architecture falls back
to the CPU. See #3850 for the case where it was silently reinterpreted instead.
