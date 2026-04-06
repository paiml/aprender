# Neural Networks

This chapter demonstrates Trueno's neural network primitives using the `activation_functions` example.

## Running the Example

```bash
cargo run --example activation_functions
```

## Activation Functions

Trueno provides 11 activation functions commonly used in neural networks:

### Basic Activations

```rust
use trueno::Vector;

let x = Vector::from_slice(&[0.5, -0.2, 1.2, -0.8, 2.1]);

// ReLU - Rectified Linear Unit
let relu = x.relu()?;  // max(0, x)

// Sigmoid - Logistic function
let sigmoid = x.sigmoid()?;  // 1 / (1 + exp(-x))

// Tanh - Hyperbolic tangent
let tanh_result = x.tanh_activation()?;  // (exp(x) - exp(-x)) / (exp(x) + exp(-x))
```

### Advanced Activations

```rust
// GELU - Gaussian Error Linear Unit (Transformer default)
let gelu = x.gelu()?;

// Swish/SiLU - x * sigmoid(x) (EfficientNet)
let swish = x.swish()?;

// Mish - x * tanh(softplus(x)) (YOLOv4)
let mish = x.mish()?;

// SELU - Self-Normalizing ELU
let selu = x.selu()?;

// Hardswish - Efficient approximation (MobileNetV3)
let hardswish = x.hardswish()?;

// Softplus - Smooth ReLU approximation
let softplus = x.softplus()?;

// ELU - Exponential Linear Unit
let elu = x.elu(1.0)?;  // alpha = 1.0

// Leaky ReLU - ReLU with negative slope
let leaky = x.leaky_relu(0.01)?;  // alpha = 0.01
```

### Softmax (Probability Distribution)

```rust
let logits = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
let probs = logits.softmax()?;

// Properties:
// - All values in [0, 1]
// - Sum = 1.0
```

## When to Use Each Activation

| Network Type | Recommended Activation | Example |
|--------------|----------------------|---------|
| CNN | ReLU | ResNet, VGG |
| Transformer | GELU | BERT, GPT |
| EfficientNet | Swish | EfficientNet-B0 to B7 |
| MobileNet | Hardswish | MobileNetV3 |
| Object Detection | Mish | YOLOv4 |
| Self-Normalizing | SELU | Deep autoencoders |
| Output Layer (classification) | Softmax | Most classifiers |
| Output Layer (regression) | None (linear) | Regression tasks |

## Building a Simple MLP

```rust
use trueno::{Vector, Matrix};

fn mlp_forward(
    input: &Vector,
    weights1: &Matrix,
    bias1: &Vector,
    weights2: &Matrix,
    bias2: &Vector,
) -> Result<Vector, TruenoError> {
    // Layer 1: Linear + ReLU
    let h1 = weights1.matvec(input)?;
    let h1 = h1.add(bias1)?;
    let h1 = h1.relu()?;

    // Layer 2: Linear + Softmax
    let h2 = weights2.matvec(&h1)?;
    let h2 = h2.add(bias2)?;
    h2.softmax()
}
```

## Transformer Building Blocks

### Layer Normalization

```rust
fn layer_norm(x: &Vector, gamma: &Vector, beta: &Vector) -> Result<Vector, TruenoError> {
    let mean = x.mean()?;
    let centered = x.sub_scalar(mean)?;
    let var = centered.dot(&centered)? / x.len() as f32;
    let std = (var + 1e-5).sqrt();

    let normalized = centered.mul_scalar(1.0 / std)?;
    let scaled = normalized.mul(gamma)?;
    scaled.add(beta)
}
```

### Attention Scores

```rust
fn attention_scores(query: &Vector, key: &Vector) -> Result<f32, TruenoError> {
    let d_k = query.len() as f32;
    let score = query.dot(key)?;
    Ok(score / d_k.sqrt())
}
```

## Performance Tips

### Batching for Efficiency

```rust
// Process multiple samples together
let batch: Vec<Vector> = inputs
    .iter()
    .map(|x| x.relu().unwrap())
    .collect();
```

### Fused Operations

```rust
// Fusing reduces memory bandwidth
// Instead of:
let h = x.relu()?.mul_scalar(scale)?;

// Use pre-scaled weights when possible
```

### GPU Acceleration

For large batch sizes, use GPU:

```bash
cargo run --release --features gpu --example activation_functions
```

### Fused Bias + Activation (GPU PTX)

For GPU inference, `trueno-gpu` provides a fused bias+activation kernel that combines bias addition with activation in a single kernel pass:

```rust
use trueno_gpu::kernels::{BiasActivationKernel, Kernel};

// Bias + GELU (common in Transformers)
let kernel = BiasActivationKernel::new(4096, 256).with_gelu();

// Bias + ReLU (common in CNNs)
let kernel = BiasActivationKernel::new(4096, 256).with_relu();

let ptx = kernel.emit_ptx();
```

This is typically used as an epilogue after GEMM operations, reducing memory bandwidth by avoiding intermediate writes.

```bash
cargo run -p trueno-gpu --example bias_activation
```

## See Also

- [Activation Functions Example](https://github.com/paiml/trueno/blob/main/examples/activation_functions.rs) - Full source
- [ML Similarity](https://github.com/paiml/trueno/blob/main/examples/ml_similarity.rs) - k-NN example
- [Performance Demo](../performance/benchmarks.md) - SIMD speedups
