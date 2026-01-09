# Showcase Benchmark

This example demonstrates the Qwen2.5-Coder showcase benchmark harness for measuring inference performance against baselines like Ollama and llama.cpp.

## Overview

The `showcase_benchmark` example implements:

- Automated model downloading from Hugging Face
- Side-by-side benchmarking against Ollama
- Performance visualization
- Regression detection

## Running the Showcase

```bash
# Run full showcase benchmark suite
cargo run --release --example showcase_benchmark

# Or via the CLI
apr showcase --model qwen2.5-coder-0.5b --format gguf
```

## Test Matrix

| Model | Size | GPU Target | CPU Target |
|-------|------|------------|------------|
| Qwen2.5-Coder-0.5B | 490MB | 500+ tok/s | 150+ tok/s |
| Qwen2.5-Coder-1.5B | 1.1GB | 350+ tok/s | 75+ tok/s |
| Qwen2.5-Coder-7B | 4.4GB | 150+ tok/s | 25+ tok/s |
| Qwen2.5-Coder-32B | 19GB | 40+ tok/s | 6+ tok/s |

## Metrics

- **Throughput**: Tokens per second (decode phase)
- **Prefill**: Prompt processing speed
- **TTFT**: Time to first token
- **Memory**: Peak VRAM/RAM usage

## See Also

- [Qwen2.5-Coder Showcase Spec](../../specifications/qwen2.5-coder-showcase-demo.md)
- [Benchmark Comparison](./bench-comparison.md)
