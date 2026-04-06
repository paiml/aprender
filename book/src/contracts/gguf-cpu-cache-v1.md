# gguf-cpu-cache-v1

**Version:** 1.0.0

GGUF CPU inference must use KV cache for O(n) autoregressive generation

## References

- realizar#95: GGUF CPU inference 11x slower than APR CPU
- qwen-coder-deploy/contracts/inference-showdown-v1.yaml (GAP-CPU-001)

## Equations

### autoregressive_generation

```
Without KV cache (current bug):
  work(n) = Σ_{i=1}^{n} (i × L × M)  =  n(n+1)/2 × L × M  ∈ O(n²)

With KV cache (correct):
  work(n) = n × L × M  ∈ O(n)

Where:
  n = number of generated tokens
  L = number of transformer layers (28 for Qwen2.5-1.5B)
  M = matmul cost per layer (fused_q4k_parallel_matvec)

Speedup ratio = (n+1)/2
  n=20 tokens → 10.5x (matches measured 11x gap)

```

**Domain:** $n \in ℕ, L \in ℕ, M > 0$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | equivalence | KV cache output matches no-cache output | `generate_with_cache(prompt, config) ≡ generate(prompt, config) for all prompts` |
| 2 | invariant | KV cache reduces work from O(n²) to O(n) | `forward_single_with_cache processes exactly 1 token per call` |
| 3 | bound | GGUF CPU throughput matches APR CPU | $tok/s(GGUF CPU) \geq 0.8 × tok/s(APR CPU)$ |
| 4 | invariant | No regression in generation quality | `argmax(logits_cached) == argmax(logits_uncached) for greedy decoding` |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-CC-001 | Output equivalence | generate_with_cache produces identical tokens to generate for greedy (temp=0) | KV cache corrupts hidden state across autoregressive steps |
| FALSIFY-CC-002 | Single-token forward | forward_single_with_cache processes exactly 1 token per call | Cache miss causes full-sequence recomputation |
| FALSIFY-CC-003 | Throughput bound | GGUF CPU tok/s >= 0.8 * APR CPU tok/s after cache fix | KV cache overhead negates O(n) benefit at short sequences |
| FALSIFY-CC-004 | Greedy decoding parity | argmax(logits_cached) == argmax(logits_uncached) for all steps | Numerical divergence in cached attention computation |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-CC-001 | KV cache output matches no-cache output | 4 | stub_float |
| KANI-CC-002 | KV cache reduces work from O(n^2) to O(n) | 8 | bounded_int |
| KANI-CC-003 | GGUF CPU throughput matches APR CPU | 4 | bounded_int |
| KANI-CC-004 | No regression in generation quality | 8 | stub_float |
| KANI-GGUF_C-005 | KV cache reduces work from O(n²) to O(n) | 8 | exhaustive |

## QA Gate

**GGUF CPU KV Cache Contract** (F-GGUF-CACHE-001)

**Checks:** try_quantized_generate calls generate_with_cache, not generate, Output tokens match between cached and uncached for greedy decoding, GGUF CPU throughput >= 5.0 tok/s on Qwen2.5-Coder-1.5B, No regression in GPU path

**Pass criteria:** All checks pass

