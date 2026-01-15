# Showcase Benchmark

This example demonstrates the Qwen2.5-Coder showcase benchmark harness for measuring inference performance against baselines like Ollama and llama.cpp.

## 🏆 MILESTONE ACHIEVED (2026-01-15)

**Both GGUF and APR formats now exceed 2X Ollama on GPU!**

### Qwen2.5-Coder-1.5B Results

| Format | M=8 | M=16 | M=32 | Status |
|--------|-----|------|------|--------|
| **GGUF** | - | 824.7 tok/s (2.83x) | - | ✅ EXCEEDED |
| **APR** | 723.8 tok/s (2.49x) | 799.9 tok/s (2.75x) | 763.9 tok/s (2.63x) | ✅ EXCEEDED |
| **Target** | 582 tok/s (2X) | 582 tok/s (2X) | 582 tok/s (2X) | - |

### Key Achievements

- **GGUF GPU**: 824.7 tok/s = **2.83x Ollama** (291 tok/s baseline)
- **APR GPU**: 799.9 tok/s = **2.75x Ollama** at M=16
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
