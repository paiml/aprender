# Benchmark Comparison

This example demonstrates how to compare performance across different implementations and configurations in the aprender ecosystem.

## Overview

The `bench_comparison` example provides a standardized way to measure and compare:

- CPU vs GPU performance
- Different quantization levels (Q4_K, Q8, F16, F32)
- Inference throughput (tokens per second)
- Memory bandwidth utilization

## Running the Benchmark

```bash
cargo run --release --example bench_comparison
```

## Key Metrics

| Metric | Description |
|--------|-------------|
| **tok/s** | Tokens generated per second |
| **Bandwidth** | Memory throughput (GB/s) |
| **Latency** | Time per token (ms) |
| **Efficiency** | % of theoretical peak |

## See Also

<!-- Performance Profiling and Quantization Guide chapters not yet written -->
