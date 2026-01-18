# Showcase Benchmark

This example demonstrates the Qwen2.5-Coder showcase benchmark harness for measuring inference performance against baselines like Ollama and llama.cpp.

## 🏆 SHOWCASE COMPLETE (2026-01-18)

**CORRECTNESS-012 fixed! Both GGUF and APR formats exceed 2X Ollama on GPU.**

### Qwen2.5-Coder-1.5B Results

| Format | M=8 | M=16 | M=32 | Status |
|--------|-----|------|------|--------|
| **GGUF** | 770.0 tok/s (2.65x) | **851.8 tok/s (2.93x)** | 812.8 tok/s (2.79x) | ✅ PASS |
| **Target** | 582 tok/s (2X) | 582 tok/s (2X) | 582 tok/s (2X) | - |

### Key Achievements

- **GGUF GPU**: 851.8 tok/s = **2.93x Ollama** (291 tok/s baseline)
- **CPU/GPU Parity**: Verified - outputs match exactly
- **APR Format**: Quantization preserved (Q4_K, Q6_K) through GGUF → APR conversion
- **File Size**: 1.9GB APR file with full model fidelity

### Run the Showcase

```bash
# APR GPU Benchmark (FEATURED)
MODEL_PATH=/path/to/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
  cargo run --example apr_gpu_benchmark --release --features cuda

# Full showcase benchmark suite
cargo run --release --example showcase_benchmark
```

## Overview

The `showcase_benchmark` example implements:

- Automated model downloading from Hugging Face
- Side-by-side benchmarking against Ollama
- Performance visualization
- Regression detection
- GGUF → APR conversion with quantization preservation

## Test Matrix

| Model | Size | GPU Target | GPU Achieved | CPU Target |
|-------|------|------------|--------------|------------|
| Qwen2.5-Coder-0.5B | 490MB | 500+ tok/s | TBD | 150+ tok/s |
| Qwen2.5-Coder-1.5B | 1.1GB | 350+ tok/s | **824.7 tok/s** ✅ | 75+ tok/s |
| Qwen2.5-Coder-7B | 4.4GB | 150+ tok/s | TBD | 25+ tok/s |
| Qwen2.5-Coder-32B | 19GB | 40+ tok/s | TBD | 6+ tok/s |

## Metrics

- **Throughput**: Tokens per second (decode phase)
- **Prefill**: Prompt processing speed
- **TTFT**: Time to first token
- **Memory**: Peak VRAM/RAM usage

## See Also

- [Qwen2.5-Coder Showcase Spec](../../specifications/qwen2.5-coder-showcase-demo.md)
- [Benchmark Comparison](./bench-comparison.md)
