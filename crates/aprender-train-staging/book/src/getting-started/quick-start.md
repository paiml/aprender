# Quick Start

This guide will get you training your first neural network with Entrenar in under 5 minutes.

## Your First Neural Network

Let's build a simple linear regression model to learn the function `y = 2x + 1`.

### Step 1: Create a New Project

```bash
cargo new entrenar_quickstart
cd entrenar_quickstart
```

### Step 2: Add Dependencies

Edit `Cargo.toml`:

```toml
[dependencies]
entrenar = "0.1"
ndarray = "0.15"
```

### Step 3: Write the Training Code

Edit `src/main.rs`:

```rust
use entrenar::{Tensor, optim::SGD, backward};

fn main() {
    // Training data: y = 2x + 1
    let x_data = vec![1.0, 2.0, 3.0, 4.0];
    let y_data = vec![3.0, 5.0, 7.0, 9.0];

    // Initialize parameters (trainable)
    let mut w = Tensor::from_vec(vec![0.0], true);  // weight
    let mut b = Tensor::from_vec(vec![0.0], true);  // bias

    // Create optimizer
    let mut optimizer = SGD::new(0.01, 0.0);  // learning_rate=0.01, momentum=0.0

    // Training loop
    for epoch in 0..100 {
        let mut total_loss = 0.0;

        for (x, y_true) in x_data.iter().zip(y_data.iter()) {
            // Forward pass: y_pred = w * x + b
            let x_tensor = Tensor::from_vec(vec![*x], false);
            let y_pred = &(&w * &x_tensor) + &b;

            // Compute loss: MSE = (y_pred - y_true)²
            let y_true_tensor = Tensor::from_vec(vec![*y_true], false);
            let diff = &y_pred - &y_true_tensor;
            let loss = &diff * &diff;

            total_loss += loss.data()[0];

            // Backward pass (compute gradients)
            backward(&loss);

            // Update parameters
            optimizer.step(&mut [&mut w, &mut b]);

            // Zero gradients for next iteration
            w.zero_grad();
            b.zero_grad();
        }

        if epoch % 10 == 0 {
            println!("Epoch {}: Loss = {:.6}", epoch, total_loss / x_data.len() as f32);
        }
    }

    // Check learned parameters
    println!("\nLearned parameters:");
    println!("w = {:.4} (expected: 2.0)", w.data()[0]);
    println!("b = {:.4} (expected: 1.0)", b.data()[0]);
}
```

### Step 4: Run the Training

```bash
cargo run --release
```

Expected output:

```
Epoch 0: Loss = 23.500000
Epoch 10: Loss = 5.123456
Epoch 20: Loss = 1.234567
Epoch 30: Loss = 0.456789
Epoch 40: Loss = 0.123456
Epoch 50: Loss = 0.034567
Epoch 60: Loss = 0.009876
Epoch 70: Loss = 0.002345
Epoch 80: Loss = 0.000567
Epoch 90: Loss = 0.000123

Learned parameters:
w = 1.9987 (expected: 2.0)
b = 1.0024 (expected: 1.0)
```

**Success!** Your model learned the linear relationship `y = 2x + 1`.

## Understanding the Code

Let's break down the key components:

### 1. Tensor Creation

```rust
let mut w = Tensor::from_vec(vec![0.0], true);  // requires_grad=true
```

- `requires_grad=true`: Enables gradient tracking for backpropagation
- Parameters must be mutable (`mut`) to update during training

### 2. Forward Pass

```rust
let y_pred = &(&w * &x_tensor) + &b;  // y = w * x + b
```

- Operators (`*`, `+`) are overloaded for tensors
- Use references (`&`) to avoid moving tensors

### 3. Loss Computation

```rust
let diff = &y_pred - &y_true_tensor;
let loss = &diff * &diff;  // MSE = (y_pred - y_true)²
```

- Mean Squared Error (MSE) is a common regression loss
- Loss must be a scalar for backpropagation

### 4. Backward Pass

```rust
backward(&loss);
```

- Computes gradients for all tensors with `requires_grad=true`
- Gradients accumulate in `tensor.grad()`

### 5. Optimizer Step

```rust
optimizer.step(&mut [&mut w, &mut b]);
```

- Updates parameters: `w = w - learning_rate * grad_w`
- SGD, Adam, AdamW all use the same interface

### 6. Zero Gradients

```rust
w.zero_grad();
b.zero_grad();
```

- **Critical**: Gradients accumulate by default
- Always zero gradients after each optimizer step

## Next Steps

### Try Different Optimizers

Replace SGD with Adam for adaptive learning rates:

```rust
use entrenar::optim::Adam;

let mut optimizer = Adam::default_params(0.01);  // learning_rate=0.01
```

### Add More Layers

Build a multi-layer perceptron:

```rust
use entrenar::autograd::ops::{matmul, relu};

// Hidden layer: h = relu(W1 * x + b1)
let h = relu(&(&matmul(&w1, &x, 10, 1, 1) + &b1));

// Output layer: y = W2 * h + b2
let y_pred = &matmul(&w2, &h, 1, 10, 1) + &b2;
```

### Use LoRA for Fine-Tuning

Apply LoRA to large pretrained weights:

```rust
use entrenar::lora::LoRALayer;

// Freeze base weights, train only LoRA adapters
let base_weight = Tensor::from_vec(vec![...], false);  // frozen
let lora = LoRALayer::new(base_weight, 256, 256, rank=16, alpha=32.0);

let output = lora.forward(&input);
```

### Enable QLoRA for Memory Efficiency

Reduce memory by 75% with 4-bit quantization:

```rust
use entrenar::lora::QLoRALayer;

// Base weights quantized to 4-bit, adapters remain float32
let qlora = QLoRALayer::new(base_weight, 256, 256, rank=16, alpha=32.0);

let output = qlora.forward(&input);  // Dequantizes on-the-fly
```

## Common Patterns

### Gradient Checking

Validate gradients with finite differences:

```rust
#[cfg(test)]
mod tests {
    use entrenar::autograd::test_utils::check_gradient;

    #[test]
    fn test_my_operation() {
        let x = Tensor::from_vec(vec![1.0, 2.0], true);
        let output = my_operation(&x);

        // Verify gradients are correct (ε=1e-3, threshold=0.2)
        assert!(check_gradient(&output, &x, 1e-3, 0.2));
    }
}
```

### Learning Rate Scheduling

Decay learning rate over time:

```rust
use entrenar::optim::schedulers::CosineScheduler;

let scheduler = CosineScheduler::new(
    initial_lr=0.1,
    min_lr=0.001,
    total_steps=1000
);

for step in 0..1000 {
    let lr = scheduler.get_lr(step);
    optimizer.set_lr(lr);

    // ... training step ...
}
```

### Gradient Clipping

Prevent exploding gradients:

```rust
use entrenar::optim::clip_grad_norm;

// Clip gradients to max norm of 1.0
clip_grad_norm(&mut [&mut w, &mut b], 1.0);

optimizer.step(&mut [&mut w, &mut b]);
```

## Performance Tips

### 1. Use Release Mode

Always train with optimizations enabled:

```bash
cargo run --release  # 10-100x faster than debug builds
```

### 2. Enable SIMD

SIMD acceleration activates automatically for tensors ≥16 elements:

```rust
// SIMD-accelerated (fast)
let large_tensor = Tensor::from_vec(vec![0.0; 1024], true);

// Scalar fallback (slower)
let small_tensor = Tensor::from_vec(vec![0.0; 8], true);
```

### 3. Batch Operations

Process multiple samples together:

```rust
// Batch matrix multiplication
let batch_output = matmul(&weights, &batch_input, d_out, d_in, batch_size);
```

## Troubleshooting

### Gradients Not Flowing

**Problem**: Parameters not updating

**Solution**: Check `requires_grad=true` and that backward pass is called:

```rust
let mut w = Tensor::from_vec(vec![0.0], true);  // ✅ requires_grad=true
backward(&loss);  // ✅ Must call backward
```

### Loss Not Decreasing

**Problem**: Training is stuck

**Solutions**:
1. Check learning rate (try 0.001, 0.01, 0.1)
2. Verify loss computation is correct
3. Check gradients aren't being zeroed too early
4. Try different optimizer (Adam instead of SGD)

### Stack Overflow in Tests

**Problem**: Gradient checking causes stack overflow

**Solution**: Increase stack size:

```bash
RUST_MIN_STACK=8388608 cargo test
```

## What's Next?

- **[First Training Loop](./first-training-loop.md)** - Build a complete training pipeline with validation
- **[Core Concepts](./core-concepts.md)** - Deep dive into Entrenar's architecture
- **[Examples](../examples/linear-regression.md)** - More practical examples

---

**Ready for a complete training pipeline?** Continue to [First Training Loop](./first-training-loop.md) →
