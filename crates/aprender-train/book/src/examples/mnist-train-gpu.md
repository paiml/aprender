# MNIST Training (GPU)

This example trains MNIST with GPU acceleration via CUDA when available.

## Running the Example

```bash
cargo run --example mnist_train_gpu
```

## Code

```rust
{{#include ../../../examples/mnist_train_gpu.rs}}
```

## GPU Detection

The example automatically detects CUDA availability:

```rust
let backend = if cuda_available() {
    Backend::Cuda
} else {
    Backend::Cpu
};
```

## Performance Comparison

| Backend | Training Time | Throughput |
|---------|---------------|------------|
| CPU | ~60s/epoch | ~1000 samples/s |
| GPU (CUDA) | ~5s/epoch | ~12000 samples/s |

## Requirements

- NVIDIA GPU with CUDA support
- CUDA toolkit installed
- `cuda` feature enabled in Cargo.toml
