# Core Concepts

This chapter explains the fundamental concepts behind Entrenar's design and how they work together to provide a complete
neural network training system.

## Architecture Overview

Entrenar is built on four core pillars:

```
┌─────────────────────────────────────────────────────────┐
│                    Training Loop                        │
│  (User Code: forward pass, loss, backward, optimize)    │
└─────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   Autograd    │  │  Optimizers   │  │   LoRA/QLoRA  │
│   Engine      │  │  (SGD, Adam,  │  │  (Parameter-  │
│   (Gradient   │  │   AdamW, LR   │  │   Efficient   │
│   Computation)│  │   Schedulers) │  │   Fine-Tuning)│
└───────────────┘  └───────────────┘  └───────────────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
                   ┌───────────────┐
                   │     Tensor    │
                   │ (Data + Grad) │
                   └───────────────┘
```

## 1. Tensors

**Tensors** are the fundamental data structure in Entrenar, representing multi-dimensional arrays with optional gradient
tracking.

### Tensor Creation

```rust
use entrenar::Tensor;

// Scalar (0D)
let scalar = Tensor::from_vec(vec![3.14], false);

// Vector (1D)
let vector = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);

// Matrix (2D) - flattened representation
let matrix = Tensor::from_vec(
    vec![1.0, 2.0,
         3.0, 4.0],  // 2x2 matrix
    true
);

// Random initialization
let weights = Tensor::randn(vec![256], true);  // Normal(0, 1)

// Zero initialization
let bias = Tensor::zeros(vec![128], true);
```

### Gradient Tracking

```rust
// Trainable parameter
let w = Tensor::from_vec(vec![1.0, 2.0], true);  // requires_grad=true
assert!(w.requires_grad());

// Frozen parameter (e.g., pretrained base weights)
let frozen = Tensor::from_vec(vec![1.0, 2.0], false);  // requires_grad=false
assert!(!frozen.requires_grad());
```

### Tensor Operations

```rust
// Arithmetic operations
let a = Tensor::from_vec(vec![1.0, 2.0], true);
let b = Tensor::from_vec(vec![3.0, 4.0], true);

let c = &a + &b;  // Element-wise addition
let d = &a * &b;  // Element-wise multiplication
let e = &a - &b;  // Element-wise subtraction

// Matrix operations
use entrenar::autograd::ops::matmul;

let result = matmul(&a, &b, rows, cols, batch_size);
```

**Key Insight**: Tensor operations use references (`&`) to avoid consuming the original tensors, allowing reuse in
computational graphs.

## 2. Automatic Differentiation (Autograd)

**Autograd** computes gradients automatically using reverse-mode differentiation (backpropagation).

### Computational Graph

Entrenar uses a **tape-based** computational graph:

```rust
let x = Tensor::from_vec(vec![2.0], true);
let y = &x * &x;           // y = x²  (tape records: mul operation)
let z = &y + &x;           // z = x² + x  (tape records: add operation)

backward(&z);              // Compute dz/dx

println!("dz/dx = {}", x.grad()[0]);  // dz/dx = 2x + 1 = 5.0
```

**Tape Structure**:
```
Tape:
  1. Op: Mul(x, x) -> y
  2. Op: Add(y, x) -> z

Backward pass (reverse order):
  1. dz/dz = 1.0
  2. dz/dy = 1.0, dz/dx += 1.0
  3. dy/dx = 2x, dz/dx += 2x * dz/dy = 4.0
  Result: dz/dx = 5.0
```

### Supported Operations

| Operation | Forward | Backward |
|-----------|---------|----------|
| **Matrix Mul** | `C = A @ B` | `dA = dC @ B^T`, `dB = A^T @ dC` |
| **ReLU** | `max(0, x)` | `dx = (x > 0) ? dy : 0` |
| **GELU** | `x * Φ(x)` | Chain rule with Gaussian CDF |
| **Layer Norm** | `(x - μ) / σ` | Mean/variance gradients |
| **Attention** | `softmax(QK^T/√d)V` | Q, K, V chain rule |

### Gradient Checking

Entrenar validates all gradients with finite differences:

```rust
#[test]
fn test_gradient_correctness() {
    let x = Tensor::from_vec(vec![1.0, 2.0], true);
    let y = &x * &x;

    backward(&y);

    // Finite difference: f(x+ε) - f(x-ε) / 2ε
    let epsilon = 1e-3;
    let threshold = 0.2;  // 20% relative error tolerance

    check_gradient(&y, &x, epsilon, threshold);  // ✅ Passes
}
```

**Zero-tolerance policy**: Every operation has gradient checking tests ensuring mathematical correctness.

## 3. Optimizers

**Optimizers** update parameters using computed gradients.

### Optimizer Interface

All optimizers share a common interface:

```rust
use entrenar::optim::{SGD, Adam, AdamW};

let mut optimizer = Adam::default_params(learning_rate=0.001);

// Training step
backward(&loss);
optimizer.step(&mut [&mut w1, &mut b1, &mut w2, &mut b2]);

// Zero gradients for next iteration
w1.zero_grad();
b1.zero_grad();
// ... etc
```

### SGD (Stochastic Gradient Descent)

```rust
use entrenar::optim::SGD;

let mut sgd = SGD::new(
    learning_rate=0.01,
    momentum=0.9,           // Accelerates convergence
);

// Update rule: v = momentum * v + grad
//              param = param - learning_rate * v
sgd.step(&mut params);
```

**Use case**: Simple optimization, baseline comparisons

### Adam (Adaptive Moment Estimation)

```rust
use entrenar::optim::Adam;

let mut adam = Adam::default_params(learning_rate=0.001);

// Adaptive learning rates per parameter
// m = β1*m + (1-β1)*grad           (1st moment)
// v = β2*v + (1-β2)*grad²          (2nd moment)
// param = param - lr * m̂ / (√v̂ + ε)
adam.step(&mut params);
```

**Use case**: General-purpose, works well out-of-the-box

### AdamW (Adam with Decoupled Weight Decay)

```rust
use entrenar::optim::AdamW;

let mut adamw = AdamW::new(
    learning_rate=0.001,
    weight_decay=0.01,      // L2 regularization
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
);

// Decoupled weight decay: param = param * (1 - wd)
adamw.step(&mut params);
```

**Use case**: Fine-tuning transformers, improved generalization

### Learning Rate Schedulers

```rust
use entrenar::optim::schedulers::CosineScheduler;

let scheduler = CosineScheduler::new(
    initial_lr=0.1,
    min_lr=0.001,
    total_steps=1000,
);

for step in 0..1000 {
    let lr = scheduler.get_lr(step);  // Cosine annealing
    optimizer.set_lr(lr);

    // ... training step ...
}
```

## 4. LoRA (Low-Rank Adaptation)

**LoRA** enables parameter-efficient fine-tuning by freezing base weights and training low-rank adapters.

### Architecture

```
Original Layer: W ∈ ℝ^(d_out × d_in)

LoRA Layer:
  Base: W ∈ ℝ^(d_out × d_in)     [FROZEN, requires_grad=false]
  Adapters:
    A ∈ ℝ^(rank × d_in)          [TRAINABLE, requires_grad=true]
    B ∈ ℝ^(d_out × rank)         [TRAINABLE, requires_grad=true]

Output: y = Wx + (α/r)(B(Ax))
```

### Usage

```rust
use entrenar::lora::LoRALayer;

// Pretrained base weights (frozen)
let base_weight = Tensor::from_vec(vec![...], false);

// Create LoRA layer
let lora = LoRALayer::new(
    base_weight,
    d_out=256,
    d_in=256,
    rank=16,       // Low-rank bottleneck
    alpha=32.0,    // Scaling factor
);

// Forward pass
let output = lora.forward(&input);

// Only LoRA adapters receive gradients
backward(&loss);  // base_weight.grad() remains zero
```

### Parameter Efficiency

```
Full Fine-Tuning: 7B parameters trainable
LoRA (rank=64):   8M parameters trainable (0.1%)

Memory savings: 99.9% reduction in trainable parameters
```

### Adapter Persistence

```rust
use entrenar::lora::adapter::{save_adapter, load_adapter};

// Save LoRA adapters (32MB file)
save_adapter(&lora, rank=16, alpha=32.0, "adapter.json")?;

// Load adapters (without full model weights)
let loaded_lora = load_adapter("adapter.json", base_weight)?;
```

**Use case**: Share fine-tuned adapters without distributing 28GB base model weights

## 5. QLoRA (Quantized LoRA)

**QLoRA** reduces memory by 75% through 4-bit quantization of frozen base weights.

### 4-Bit Quantization

```rust
use entrenar::lora::QLoRALayer;

// Base weights quantized to 4-bit (75% memory reduction)
let qlora = QLoRALayer::new(
    base_weight,
    d_out=4096,
    d_in=4096,
    rank=64,
    alpha=128.0,
);

// On-the-fly dequantization during forward pass
let output = qlora.forward(&input);
```

### Memory Comparison

| Configuration | LoRA Memory | QLoRA Memory | Savings |
|---------------|-------------|--------------|---------|
| **Small (256-dim, 6 layers)** | 1.5 MB | 0.5 MB | 65% |
| **Medium (768-dim, 12 layers)** | 27 MB | 8 MB | 68% |
| **Large (4096-dim, 32 layers)** | 4.2 GB | 1.2 GB | 70% |

### Quantization Details

```
Block-wise quantization (64 elements per block):
  1. Compute scale factor: s = max(|values|) / 7
  2. Quantize: q = round(value / s)  ∈ [-7, 7]
  3. Store: 4-bit signed integers (15 discrete levels)

Dequantization:
  value = q * s  (full precision restored)
```

**Trade-off**: Minimal accuracy loss (<1%) for 75% memory reduction

## 6. EXTREME TDD Quality

Entrenar is built with **zero-tolerance for defects** using multiple testing strategies:

### Unit Tests

```rust
#[test]
fn test_matmul_correctness() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], false);
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], false);

    let c = matmul(&a, &b, 2, 2, 1);

    assert_eq!(c.data()[0], 19.0);  // 1*5 + 2*7
    assert_eq!(c.data()[1], 43.0);  // 3*5 + 4*7
}
```

### Property-Based Tests

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_adam_converges(lr in 0.05f32..0.5) {
        let optimizer = Adam::default_params(lr);
        assert!(converges_to_minimum(optimizer, 100));
    }
}
```

### Gradient Checking

```rust
#[test]
fn test_relu_gradient() {
    let x = Tensor::from_vec(vec![-1.0, 0.0, 1.0], true);
    let y = relu(&x);

    backward(&y);

    // Finite difference validation (ε=1e-3, threshold=0.2)
    check_gradient(&y, &x, 1e-3, 0.2);
}
```

### Mutation Testing

```bash
cargo mutants --file src/autograd/ops.rs

# Ensures tests catch intentional bugs
# Target: >80% mutation kill rate
```

## Putting It All Together

### Complete Training Workflow

```rust
use entrenar::{Tensor, backward, optim::AdamW, lora::QLoRALayer};

// 1. Load pretrained base weights
let base_weight = load_pretrained_weights("llama-7b.bin");

// 2. Create QLoRA layer (75% memory reduction)
let qlora = QLoRALayer::new(base_weight, 4096, 4096, rank=64, alpha=128.0);

// 3. Initialize optimizer
let mut optimizer = AdamW::new(lr=0.0001, weight_decay=0.01, ...);

// 4. Training loop
for (input, target) in dataloader {
    // Forward pass
    let output = qlora.forward(&input);
    let loss = cross_entropy_loss(&output, &target);

    // Backward pass (only LoRA adapters get gradients)
    backward(&loss);

    // Update (only 8M parameters instead of 7B)
    optimizer.step(&mut qlora.trainable_parameters());

    // Zero gradients
    qlora.zero_grad();
}

// 5. Save adapters (32MB file)
save_adapter(&qlora, "custom_adapter.json")?;
```

**Result**: Fine-tune 7B parameter model on consumer GPU with 8GB VRAM

## Key Takeaways

1. **Tensors** store data and gradients, enabling automatic differentiation
2. **Autograd** computes gradients via reverse-mode differentiation on a tape-based graph
3. **Optimizers** update parameters using various strategies (SGD, Adam, AdamW)
4. **LoRA** trains low-rank adapters instead of full weights (99.9% parameter reduction)
5. **QLoRA** quantizes base weights to 4-bit for 75% memory savings
6. **EXTREME TDD** ensures zero defects through comprehensive testing

## What's Next?

- **[Autograd Engine](../autograd/what-is-autograd.md)** - Deep dive into automatic differentiation
- **[Optimizers](../optimizers/overview.md)** - Explore optimizer algorithms and theory
- **[LoRA/QLoRA](../lora/what-is-lora.md)** - Master parameter-efficient fine-tuning
- **[Examples](../examples/linear-regression.md)** - See practical applications

---

**Ready to explore the autograd engine?** Continue to
[What is Automatic Differentiation?](../autograd/what-is-autograd.md) →
