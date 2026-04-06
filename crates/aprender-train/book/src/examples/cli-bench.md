# CLI Benchmark Example

This example demonstrates latency benchmarking for model inference.

## Running the Example

```bash
cargo run --example cli_bench
```

## Code

```rust
{{#include ../../../examples/cli_bench.rs}}
```

## Expected Output

```
Latency Benchmark Example
=========================

Batch size: 1
  p50: 0.11ms
  p95: 0.12ms
  p99: 0.12ms
  mean: 0.11ms
  throughput: 9012.4 samples/sec

Batch size: 8
  p50: 0.18ms
  p95: 0.19ms
  p99: 0.19ms
  mean: 0.18ms
  throughput: 43586.8 samples/sec

Batch size: 32
  p50: 0.45ms
  p95: 0.47ms
  p99: 0.48ms
  mean: 0.45ms
  throughput: 71111.1 samples/sec
```

## CLI Usage

```bash
# Benchmark with warmup and iterations
entrenar bench config.yaml --warmup 5 --iterations 100

# Benchmark specific batch sizes
entrenar bench model.safetensors --batch-sizes 1,8,32,64
```
