# Phase 2 GPU Benchmark Verification Report

**Date:** 2026-01-08
**Hardware:** NVIDIA RTX 4090 (24GB VRAM), Driver 570.195.03, CUDA 12.8
**Model:** Qwen2.5-Coder-1.5B-Instruct Q4_K_M (1.1GB)
**Software:** apr-cli v0.2.2, realizar v0.5.0, trueno v0.11.0

---

## Executive Summary

This document provides verified benchmark results for the PAIML Sovereign AI Stack showcase.
**Important distinction:** Results measure **decode throughput** (token generation), not prefill.

---

## Verified Benchmark Results

### APR vs Ollama (Decode Throughput)

| System | Throughput | Std Dev | TTFT | Runs | Backend |
|--------|------------|---------|------|------|---------|
| **APR (realizar)** | 8.7 tok/s | ±0.1 | 114.6ms | 30 | GPU (CUDA) |
| **Ollama** | 32.2 tok/s | N/A | 158.8ms | 1 | GPU (llama.cpp) |

### Statistical Quality (PMAT Points)

| Metric | Requirement | Measured | Status |
|--------|-------------|----------|--------|
| Coefficient of Variation | <5% | 1.1% | **PASS** |
| Benchmark Runs | ≥30 | 30 | **PASS** |
| Speedup vs Ollama | ≥25% | -72.9% | **FAIL** |
| Minimum Throughput | ≥60 tok/s | 8.7 tok/s | **FAIL** |

---

## Falsification Point Status

### Point 42: APR ≥60 tok/s minimum threshold
**Status: FAIL (Decode)**
**Measured:** 8.7 tok/s (GPU decode)
**Note:** The spec's 150.3 tok/s figure is for **batched prefill only**, not decode.

### Point 49: Performance consistent across runs (CV <5%)
**Status: PASS**
**Measured:** CV = 1.1% (0.1/8.7)

### Point 50: Sufficient benchmark runs (≥30)
**Status: PASS**
**Measured:** 30 runs

---

## Performance Gap Analysis

### Why APR is slower than Ollama

1. **Kernel Launch Overhead**: Each token requires multiple CUDA kernel launches
2. **Memory Bandwidth**: Suboptimal memory access patterns in decode phase
3. **KV Cache Management**: Per-token cache updates vs batched operations
4. **No Speculative Decoding**: Single token generation per iteration

### Ollama Advantages

- llama.cpp's highly optimized CUDA kernels
- Years of community optimization
- Speculative decoding enabled by default
- Continuous batching for multiple requests

---

## Reproduce Benchmark

```bash
# Build with CUDA feature
cargo build --release -p apr-cli --features cuda

# Run benchmark
cargo run --release -p apr-cli --features cuda -- showcase \
  --step bench --tier small --runs 30 --gpu --model-dir models
```

---

## Recommendations

1. **Do not claim** 150 tok/s without specifying "prefill only"
2. **Document** decode vs prefill distinction in all claims
3. **Set realistic targets**: 15-20 tok/s decode is achievable short-term
4. **Focus optimization** on decode kernels (PAR-041 blocking)

---

## Raw Output

```
═══ Benchmark Results ═══

┌─────────────────┬────────────┬────────────┬──────────┐
│ System          │ Tokens/sec │ TTFT (ms)  │ Runs     │
├─────────────────┼────────────┼────────────┼──────────┤
│ APR (ours)     │     8.7±0.1 │      114.6 │       30 │
│ Ollama          │       32.2 │      158.8 │      N/A │
└─────────────────┴────────────┴────────────┴──────────┘

Speedup vs Ollama: -72.9% FAIL (target: 25%)
```

---

## Test Matrix Status

| Format | Backend | Tier | Status | Notes |
|--------|---------|------|--------|-------|
| GGUF | CPU | small (1.5B) | ✅ Works | 1.2 tok/s |
| GGUF | GPU | small (1.5B) | ✅ Works | 8.7 tok/s |
| GGUF | CPU | tiny (0.5B) | [ ] Untested | Model not available |
| GGUF | GPU | tiny (0.5B) | [ ] Untested | Model not available |
| GGUF | CPU | medium (7B) | [ ] Untested | Model not available |
| GGUF | GPU | medium (7B) | [ ] Untested | Model not available |
| GGUF | CPU | large (32B) | [ ] Untested | Model available but requires testing |
| GGUF | GPU | large (32B) | [ ] Untested | Model available but requires testing |
| APR | CPU | any | ❌ N/A | APR format for traditional ML only |

---

## Conclusion

The GPU inference pipeline is **functional** but does not meet performance targets.
The 8.7 tok/s decode throughput is approximately **3.7x slower** than Ollama (32.2 tok/s).

Further optimization of the decode path in realizar is required before claiming
Ollama-parity performance.
