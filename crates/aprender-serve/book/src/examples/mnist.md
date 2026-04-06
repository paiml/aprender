# MNIST Examples

Realizar includes several examples demonstrating MNIST digit classification with the .apr format.

## Prerequisites

These examples require the `aprender-serve` feature:

```bash
cargo run --example <name> --features "aprender-serve"
```

## 1. Build MNIST Model

Create and save an MNIST classifier:

```bash
cargo run --example build_mnist_model --features "aprender-serve"
```

This creates a neural network for digit classification and saves it as `mnist_classifier.apr`.

### Architecture

```
Input (784) -> Hidden (128, ReLU) -> Output (10, Softmax)
```

### Output

```
=== Building MNIST Classifier ===
  Architecture: 784 -> 128 -> 10
  Activation: ReLU (hidden), Softmax (output)
  Format: .apr (Aprender native)

Saving to mnist_classifier.apr...
  Model size: 103,562 bytes
  Quantization: None (F32)

=== Build Complete ===
```

## 2. Serve MNIST Model

Serve the model via HTTP API:

```bash
cargo run --example serve_mnist --features "aprender-serve"
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Single digit classification |
| `/batch` | POST | Batch classification |

### Example Request

```bash
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"pixels": [0.0, 0.0, ..., 0.5, 1.0, ...]}'
```

### Response

```json
{
  "digit": 7,
  "confidence": 0.94,
  "probabilities": [0.01, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.94, 0.01, 0.01]
}
```

## 3. MNIST Benchmark

Compare .apr inference against PyTorch:

```bash
cargo run --example mnist_apr_benchmark --features "aprender-serve"
```

### Metrics Measured

| Metric | .apr | PyTorch | Improvement |
|--------|------|---------|-------------|
| Cold Start | ~2ms | ~150ms | 75x faster |
| Inference (single) | ~15µs | ~100µs | 6.7x faster |
| Inference (batch 100) | ~1.2ms | ~5ms | 4x faster |
| Memory | ~400KB | ~50MB | 125x smaller |

### Output Example

```
=== MNIST APR vs PyTorch Benchmark ===

Model: mnist_classifier.apr (103KB)

Cold Start Latency:
  .apr:     1.8ms ± 0.2ms
  PyTorch:  148ms ± 12ms
  Speedup:  82x

Single Inference:
  .apr:     14.2µs ± 1.1µs
  PyTorch:  98µs ± 8µs
  Speedup:  6.9x

Batch Inference (n=100):
  .apr:     1.15ms ± 0.08ms
  PyTorch:  4.8ms ± 0.3ms
  Speedup:  4.2x

Memory Usage:
  .apr:     412KB
  PyTorch:  48.2MB
  Reduction: 117x

=== Benchmark Complete ===
```

## Integration with single-shot-eval

For SLM Pareto Frontier evaluation with .apr models:

```bash
cd ../single-shot-eval
cargo run -- --model mnist_classifier.apr --task mnist
```

See: [single-shot-eval](https://github.com/paiml/single-shot-eval)

## Why .apr for Edge Deployment?

1. **Zero Dependencies**: No Python, PyTorch, or CUDA required
2. **Tiny Footprint**: 100KB vs 50MB for equivalent PyTorch model
3. **Instant Cold Start**: 2ms vs 150ms - critical for Lambda
4. **ARM64 Native**: Perfect for Graviton, Apple Silicon, embedded
5. **Reproducible**: Deterministic inference across platforms
