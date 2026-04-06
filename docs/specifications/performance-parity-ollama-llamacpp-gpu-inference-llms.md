# Performance Parity: Ollama/llama.cpp GPU Inference

**Status:** SUPERSEDED
**Superseded By:** [decoder-throughput-specification-llama-mistral-phi-qwen.md](./decoder-throughput-specification-llama-mistral-phi-qwen.md)
**Date:** 2025-12-16

---

## Summary

This specification has been closed and superseded by the Decoder Throughput Specification.

### Root Cause Identified

The 190x performance gap (1.2 tok/s vs 228 tok/s Ollama) was traced to:

> **Non-coalesced memory access in M=1 GEMV operations during token generation.**

The original approach of using tiled GEMM for M=1 matmuls caused 16KB strided reads, achieving only 1.4% of theoretical memory bandwidth.

### Solution

The successor specification defines:
- Coalesced GEMV kernel via trueno-gpu PTX generation
- 256 threads/block with shared memory caching
- Target: <0.1ms per 1×4096×4096 GEMV (vs 4.41ms current)

### Migration

All work continues in the successor specification with:
- 35 peer-reviewed references
- 100-point QA checklist
- Toyota Way / Popper methodology

---

## Historical Context

Original code pattern (for reference):

```rust
// OLD - NO CACHE (O(n³) per token):
let logits = transformer.forward(&tokens);

// NEW - WITH KV CACHE (O(n²) per token):
let quantized = OwnedQuantizedModel::from_mapped(&mapped)?;
let tokens = quantized.generate_with_cache(&prompt, &config)?;
```

KV cache was implemented but GEMV remained the bottleneck.

---

**Next:** See [Decoder Throughput Specification](./decoder-throughput-specification-llama-mistral-phi-qwen.md)
