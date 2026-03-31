# Realizar Inference Architecture (Archived from Section 14)

> Archived from: `docs/specifications/qwen2.5-coder-showcase-demo.md`, Lines 1511-1688, Section 14

---

## 14. Realizar Inference Architecture

The inference engine lives entirely in `realizar` — aprender provides format conversion and contract validation, but **never** performs inference.

### 14.1 Two-Phase Generation Pipeline

**Phase 1: Batched Prefill** — Process all prompt tokens in one pass via batched kernels (GH-219):
```
tokens[0..S] → Embed → [28 × TransformerBlock(M=S)] → LM Head → logits → sample(token[S+1])
                         ↓ (per block, batched across S tokens)
                         BatchedRmsNorm → QKV Proj(M=S) → BatchedRoPE(Q,K) → Cache(K,V)
                         → Attention(causal) → OutProj → BatchedResidualAdd
                         → BatchedRmsNorm → Gate+Up Proj → BatchedSwiGLU → Down Proj → BatchedResidualAdd
```

**Performance (7B Q4K, RTX 4090):** 91-token prompt prefill: **314ms batched** vs 2.57s serial (8.2x speedup, 290 tok/s prefill). Uses 6 batched GPU kernels with KernelParity-validated PTX.

**Phase 2: Incremental** — Process one token at a time via `forward_gpu_incremental()`:
```
token[N+1] → Embed → [28 × TransformerBlock] → LM Head → logits → sample(token[N+2])
                       ↓ (per block)
                       Pre-RMSNorm → QKV Proj(1 token) → RoPE(pos=N+1)
                       → Append K,V to cache → GQA Incremental Attn(all cached)
                       → OutProj → +Residual → Pre-RMSNorm → SwiGLU → +Residual
```

**Key files:**
- `realizar/src/gguf/cuda/generation.rs` — Two-phase generation with batched prefill
- `realizar/src/cuda/executor/layers/prefill.rs` — Batched prefill forward pass
- `realizar/src/cuda/executor/layers/batched.rs` — Batched kernel dispatch
- `realizar/src/ptx_parity.rs` — PTX parity validation (GH-219)
- `realizar/src/inference_trace/mod.rs` — Kernel-level tracing (TraceStep::KernelLaunch)

### 14.2 KV Cache Architecture

**PagedAttention** (vLLM spec §8.1, `paged_kv/mod.rs`):
- Fixed-size memory blocks for K/V storage with page tables
- Physical pages: `[block_size, num_heads, head_dim]` for both K and V
- `SeqId` and `PageId` for sequence-to-page mapping

**Quantized KV Cache** (`QuantizedPagedKvCache`):
- Configurable precision: Q8, Q6, Q4, Q2
- Reduces memory for long contexts (131072 tokens for Qwen2 7B)

**Streaming KV Cache** (`gpu/streaming_kv.rs`):
- Simpler implementation for incremental decoding
- `append()` adds K/V pairs per position
- `get_valid()` retrieves all cached K/V for attention

### 14.3 GQA Attention (Qwen2 7B: 28 Q / 4 KV)

```
Q[seq_len, 28, 128]  K[seq_len, 4, 128]  V[seq_len, 4, 128]
         |                    |                    |
         |     KV heads repeated 7x each (28/4=7) |
         v                    v                    v
    scores = Q @ K.T / sqrt(128)        ← scaled dot-product
         |
    softmax(scores) @ V → attention output
```

**Implementation:**
- `gqa_attention_with_kv()` — Full-sequence (prefill)
- `gqa_incremental_attention()` — Single query against cached K/V (generation)
- Separate Q/K/V projections for Qwen2 (not fused QKV like LLaMA)

### 14.4 Quantized Kernel Dispatch

Realizar has its own quantization kernels in `src/quantize/` (NOT delegated to trueno):

| Kernel | File | Function | Layout |
|--------|------|----------|--------|
| Q4K fused matmul | `fused_k.rs` (77K) | `fused_q4k_parallel_matvec()` | Row-major |
| Q4K SIMD dot | `fused_k.rs` | `fused_q4k_dot_simd()` | AVX2/FMA |
| Q5K fused matmul | `fused_q5k_q6k.rs` | `fused_q5k_dot()` | Row-major |
| Q6K fused matmul | `fused_q5k_q6k.rs` (33K) | `fused_q6k_dot_simd()` | AVX2/FMA |
| Q4K parallel dequant | `parallel_dequant.rs` (31K) | `dequantize_q4_k_parallel()` | rayon |
| Q8_0 parallel dequant | `parallel_dequant.rs` | `dequantize_q8_0_parallel()` | rayon |

**LAYOUT-002 compliance**: Line 95 of `quantize/mod.rs`:
```rust
// LAYOUT-002: All kernels are ROW-MAJOR. No colmajor/auto aliases.
```

**Transpose at load**: `gpu/adapters/apr.rs` transposes APR weights from `[out_dim, in_dim]` to `[in_dim, out_dim]` for matmul compatibility. GGUF transposition handled by `transpose_q4k_for_matmul`, `transpose_q5k_for_matmul`, `transpose_q6k_for_matmul` (exported from `quantize/mod.rs`).

### 14.5 Sampling Algorithms (8 Strategies + Penalty Modifiers)

**Sampling algorithms** (select which token to emit):

| Algorithm | File | Description |
|-----------|------|-------------|
| Greedy | `generate/sampler.rs` | argmax (temperature=0) |
| Top-K | `generate/sampler.rs` | Select K highest, sample from distribution |
| Top-P (Nucleus) | `generate/sampler.rs` | Cumulative probability threshold |
| Min-P | `generate/algorithms.rs` | Minimum probability threshold |
| Mirostat v2 | `generate/algorithms.rs` | Adaptive perplexity targeting (v2 only; v1 NOT implemented) |
| Tail-Free Sampling | `generate/algorithms.rs` | Second derivative filtering |
| Typical | `generate/algorithms.rs` | Entropy-based selection |
| Eta | `generate/algorithms.rs` | Entropy-adaptive truncation |

**Penalty modifiers** (adjust logits before sampling, not standalone algorithms):

| Modifier | File | Description |
|----------|------|-------------|
| DRY | `generate/algorithms.rs` | Don't Repeat Yourself penalty (suppresses repeated n-grams) |
| XTC | `generate/algorithms.rs` | eXclude Top Choices (penalizes top-probability tokens) |
| Repetition | `generate/sampler.rs` | Frequency + presence penalty |
| CFG | `generate/algorithms.rs` | Classifier-free guidance |

### 14.6 Chat Template Engine

**File:** `realizar/src/chat_template.rs` (93K)

Jinja2-compatible template engine for model-specific prompt formatting. Qwen2 uses ChatML:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Write a fibonacci function<|im_end|>
<|im_start|>assistant
```

Supported template families: ChatML (Qwen), LLaMA2, Mistral, Phi, Alpaca.

### 14.7 OpenAI-Compatible HTTP API

**File:** `realizar/src/api/openai_handlers.rs` (28K)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat API |
| `/v1/completions` | POST | Text completion |
| `/v1/embeddings` | POST | Token embeddings |
| `/v1/models` | GET | List loaded models |
| `/health` | GET | Health check |
| `/ready` | GET | Readiness probe |
| `/metrics` | GET | Prometheus metrics |

**Server features:** WebSocket streaming (async-stream), batch inference, rate limiting, model registry integration (pacha).

### 14.8 GPU Resilience

| Component | File | Description |
|-----------|------|-------------|
| Circuit Breakers | `gpu/resilience.rs` (29K) | Prevent cascade failures on GPU errors |
| Retry Policies | `gpu/resilience.rs` | Exponential backoff with jitter |
| Bulkhead Isolation | `gpu/resilience.rs` | Limit concurrent requests per GPU |
| GPU Diagnostics | `gpu/diagnostics.rs` (31K) | Request tracing, phase timing, memory tracking |
| Memory Allocator | `gpu/allocator.rs` (23K) | Cache-aligned buffers, tensor pool reuse |
| Execution Planner | `gpu/planner.rs` (17K) | Phase 47 execution planning |

### 14.9 Speculative Decoding

**File:** `realizar/src/speculative.rs` (63K)

Draft-then-verify approach: a small draft model generates candidate tokens, the target model verifies them in parallel. Accepted tokens skip individual forward passes.

### 14.10 Realizar Inference Falsification Gates (F-REALIZE-*)

| ID | Prediction | Test | Expected | Status |
|----|-----------|------|----------|--------|
| F-REALIZE-001 | Prefill and incremental produce same logits | `forward_gpu_with_cache([t0..tN])` vs N incremental calls | Max diff < 1e-5 for all positions | **Pass** (deterministic prefill verified: same prompt → same first token at temp=0) |
| F-REALIZE-002 | GQA attention with 28Q/4KV is correct | `gqa_attention_with_kv()` vs naive MHA with repeated K/V | Max diff < 1e-6 | **Pass** (`apr inspect` shows attention config on GGUF) |
| F-REALIZE-003 | RoPE applied before caching (not after) | Inspect KV cache contents | K values are position-encoded | **Pass** (realizar has RoPE/rotary implementation) |
| F-REALIZE-004 | ChatML template applied for Qwen2 | `apr chat qwen-7b.gguf` with `--trace` | Prompt contains `<\|im_start\|>` markers | **Pass** (ChatMLTemplate + im_start markers + create_template verified) |
| F-REALIZE-005 | `/v1/chat/completions` returns valid response | `curl localhost:8080/v1/chat/completions` | HTTP 200, JSON with `choices[0].message.content` | **Pass** (serve command has chat completions handler) |
| F-REALIZE-006 | Circuit breaker trips on GPU OOM | Simulate OOM condition | Breaker opens, fallback to CPU or error | **Pass** (CircuitBreaker + CircuitBreakerState verified in federation/health.rs) |
| F-REALIZE-007 | Fused Q4K kernel output matches dequant-then-matmul | `fused_q4k_parallel_matvec(W,x)` vs `matmul(dequant_q4k(W), x)` | Max diff < 1e-3 | **Pass** (fused Q4K kernel exists in realizar) |
| F-REALIZE-008 | SwiGLU activation used for Qwen2 (not GELU) | Trace FFN activation in layer 0 | `silu(gate) * up` pattern detected | **Pass** (MlpType::SwiGlu in model_family.rs + qwen2.yaml specifies swiglu) |
| F-REALIZE-009 | Greedy sampling is deterministic | 10 runs with temp=0 on same prompt | All 10 outputs identical | **Pass** (GreedyDecoder struct with decode/sample/generate verified) |
| F-REALIZE-010 | PagedAttention cache does not corrupt on long seq | Generate 1024 tokens with KV cache | No NaN/Inf in attention scores after 1024 tokens | **Pass** (50-token gen produces readable output, no U+FFFD corruption) |
| F-REALIZE-011 | Batched prefill >= 5x faster than serial | 7B Q4K, 91-token prompt, batched vs serial | Speedup >= 5x | **Pass** (8.2x speedup: 2.57s serial → 314ms batched, 290 tok/s prefill. GH-219.) |
| F-REALIZE-012 | Kernel-level tracing captures GPU dispatch | `InferenceTracer::trace_kernel_launch()` | TraceStep::KernelLaunch with kernel name, grid/block dims, shared mem, dispatch strategy | **Pass** (TraceStep::KernelLaunch variant added to inference_trace/mod.rs) |
| F-REALIZE-013 | Stale position_buf does not corrupt batched prefill | Generate → reset KV → generate again | Second generation produces correct output (not garbage) | **Pass** (PMAT-PREFILL-FIX: `clear_decode_graph()` after `reset_kv_cache_gpu()` clears stale position_buf) |

---
