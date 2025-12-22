# Case Study: Automatic Differentiation for Neural Network Training

This case study demonstrates Aprender's autograd engine for computing gradients and training neural networks.

## Overview

The `autograd` module provides:
- **Tensor**: Gradient-tracking tensor type
- **Computation Graph**: Tape-based recording of operations
- **Backward Pass**: Automatic gradient computation via chain rule
- **No-Grad Context**: Disable tracking for inference

## Basic Gradient Computation

```rust
use aprender::autograd::{Tensor, no_grad, clear_graph};

fn main() {
    // Create tensors with gradient tracking
    let x = Tensor::from_slice(&[1.0, 2.0, 3.0]).requires_grad();
    let w = Tensor::from_slice(&[0.5, 0.5, 0.5]).requires_grad();

    // Forward pass: y = sum(x * w)
    let z = x.mul(&w);
    let y = z.sum();

    // Backward pass
    y.backward();

    // Access gradients
    // ∂y/∂x = w (element-wise)
    // ∂y/∂w = x (element-wise)
    println!("x.grad = {:?}", x.grad());  // [0.5, 0.5, 0.5]
    println!("w.grad = {:?}", w.grad());  // [1.0, 2.0, 3.0]

    // Clear graph for next iteration
    clear_graph();
}
```

## Tensor Operations

### Element-wise Operations

```rust
use aprender::autograd::Tensor;

let a = Tensor::from_slice(&[1.0, 2.0, 3.0]).requires_grad();
let b = Tensor::from_slice(&[4.0, 5.0, 6.0]).requires_grad();

// Arithmetic
let c = a.add(&b);      // [5, 7, 9]
let d = a.sub(&b);      // [-3, -3, -3]
let e = a.mul(&b);      // [4, 10, 18]
let f = a.div(&b);      // [0.25, 0.4, 0.5]

// Unary
let g = a.neg();        // [-1, -2, -3]
let h = a.exp();        // [e¹, e², e³]
let i = a.log();        // [0, ln(2), ln(3)]
let j = a.sqrt();       // [1, √2, √3]
let k = a.pow(2.0);     // [1, 4, 9]
```

### Reduction Operations

```rust
use aprender::autograd::Tensor;

let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).requires_grad();

let sum_all = x.sum();           // 10.0
let mean_all = x.mean();         // 2.5
let sum_axis0 = x.sum_axis(0);   // [4.0, 6.0]
let sum_axis1 = x.sum_axis(1);   // [3.0, 7.0]
```

### Matrix Operations

```rust
use aprender::autograd::Tensor;

let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).requires_grad();
let b = Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]).requires_grad();

// Matrix multiplication
let c = a.matmul(&b);

// Transpose
let at = a.transpose();

// View/reshape
let flat = a.view(&[4]);
```

### Activation Functions

```rust
use aprender::autograd::Tensor;

let x = Tensor::from_slice(&[-1.0, 0.0, 1.0]).requires_grad();

let relu_out = x.relu();           // [0, 0, 1]
let sigmoid_out = x.sigmoid();     // [0.27, 0.5, 0.73]
let tanh_out = x.tanh();           // [-0.76, 0, 0.76]
let gelu_out = x.gelu();           // [-0.16, 0, 0.84]
let leaky_relu = x.leaky_relu(0.01); // [-0.01, 0, 1]

// Softmax (normalizes to probability distribution)
let logits = Tensor::from_slice(&[1.0, 2.0, 3.0]).requires_grad();
let probs = logits.softmax();      // [0.09, 0.24, 0.67]
```

## Training Loop Example

```rust
use aprender::autograd::{Tensor, clear_graph, no_grad};

fn train_linear_regression() {
    // Model parameters
    let mut w = Tensor::from_slice(&[0.0]).requires_grad();
    let mut b = Tensor::from_slice(&[0.0]).requires_grad();

    // Training data: y = 2x + 1
    let x_train = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let y_train = Tensor::from_slice(&[3.0, 5.0, 7.0, 9.0]);

    let learning_rate = 0.01;
    let epochs = 100;

    for epoch in 0..epochs {
        // Forward pass
        let y_pred = x_train.mul(&w).add(&b);

        // Loss: MSE
        let diff = y_pred.sub(&y_train);
        let loss = diff.mul(&diff).mean();

        // Backward pass
        loss.backward();

        // Gradient descent update (no_grad to avoid tracking)
        no_grad(|| {
            let w_grad = w.grad().unwrap();
            let b_grad = b.grad().unwrap();

            // w = w - lr * grad
            w = w.sub(&w_grad.mul(&Tensor::from_slice(&[learning_rate])));
            b = b.sub(&b_grad.mul(&Tensor::from_slice(&[learning_rate])));

            // Re-enable gradient tracking
            w = w.requires_grad();
            b = b.requires_grad();
        });

        // Clear graph for next iteration
        clear_graph();

        if epoch % 10 == 0 {
            println!("Epoch {}: loss = {:.4}", epoch, loss.item());
        }
    }

    println!("Learned: w = {:.2}, b = {:.2}", w.item(), b.item());
    // Expected: w ≈ 2.0, b ≈ 1.0
}
```

## Neural Network Layer

```rust
use aprender::autograd::Tensor;

struct Linear {
    weight: Tensor,
    bias: Tensor,
}

impl Linear {
    fn new(in_features: usize, out_features: usize) -> Self {
        // Xavier initialization
        let scale = (2.0 / (in_features + out_features) as f32).sqrt();
        let weight_data: Vec<f32> = (0..in_features * out_features)
            .map(|_| rand::random::<f32>() * scale - scale / 2.0)
            .collect();
        let bias_data = vec![0.0; out_features];

        Self {
            weight: Tensor::new(&weight_data, &[in_features, out_features]).requires_grad(),
            bias: Tensor::new(&bias_data, &[out_features]).requires_grad(),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        // y = x @ W + b
        x.matmul(&self.weight).add(&self.bias)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight, &self.bias]
    }
}
```

## Multi-Layer Perceptron

```rust
use aprender::autograd::Tensor;

struct MLP {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

impl MLP {
    fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        Self {
            fc1: Linear::new(input_dim, hidden_dim),
            fc2: Linear::new(hidden_dim, hidden_dim),
            fc3: Linear::new(hidden_dim, output_dim),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let h1 = self.fc1.forward(x).relu();
        let h2 = self.fc2.forward(&h1).relu();
        self.fc3.forward(&h2)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        params.extend(self.fc1.parameters());
        params.extend(self.fc2.parameters());
        params.extend(self.fc3.parameters());
        params
    }
}
```

## Gradient Checking

Verify autograd correctness with numerical gradients:

```rust
use aprender::autograd::{Tensor, clear_graph};

fn numerical_gradient(f: impl Fn(&Tensor) -> Tensor, x: &Tensor, eps: f32) -> Vec<f32> {
    let mut grads = Vec::with_capacity(x.len());

    for i in 0..x.len() {
        let mut x_plus = x.data().to_vec();
        let mut x_minus = x.data().to_vec();
        x_plus[i] += eps;
        x_minus[i] -= eps;

        let y_plus = f(&Tensor::from_slice(&x_plus)).item();
        let y_minus = f(&Tensor::from_slice(&x_minus)).item();

        grads.push((y_plus - y_minus) / (2.0 * eps));
    }

    grads
}

fn test_gradient() {
    let x = Tensor::from_slice(&[1.0, 2.0, 3.0]).requires_grad();

    // f(x) = sum(x^2) = x₁² + x₂² + x₃²
    let f = |t: &Tensor| t.pow(2.0).sum();

    // Autograd gradient
    let y = f(&x);
    y.backward();
    let autograd_grad = x.grad().unwrap();

    // Numerical gradient
    let numerical_grad = numerical_gradient(f, &x, 1e-5);

    println!("Autograd: {:?}", autograd_grad.data());
    println!("Numerical: {:?}", numerical_grad);

    // Should be close: [2, 4, 6]
    for (ag, ng) in autograd_grad.data().iter().zip(numerical_grad.iter()) {
        assert!((ag - ng).abs() < 1e-4, "Gradient mismatch!");
    }

    clear_graph();
}
```

## No-Grad for Inference

```rust
use aprender::autograd::{Tensor, no_grad, is_grad_enabled};

fn inference(model: &MLP, input: &Tensor) -> Tensor {
    // Disable gradient tracking for inference
    no_grad(|| {
        assert!(!is_grad_enabled());

        let output = model.forward(input);

        // No tape is recorded, saves memory
        output
    })
}

fn validate(model: &MLP, val_data: &[(Tensor, Tensor)]) -> f32 {
    let mut total_loss = 0.0;

    no_grad(|| {
        for (x, y) in val_data {
            let pred = model.forward(x);
            let loss = pred.sub(y).pow(2.0).mean();
            total_loss += loss.item();
        }
    });

    total_loss / val_data.len() as f32
}
```

## Broadcasting

```rust
use aprender::autograd::Tensor;

let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).requires_grad();
let bias = Tensor::from_slice(&[10.0, 20.0]).requires_grad();

// Bias is broadcast across rows
let y = x.add_broadcast(&bias);
// [[11, 22], [13, 24]]

y.sum().backward();

// Gradient is summed across broadcast dimension
println!("bias.grad = {:?}", bias.grad());  // [2.0, 2.0]
```

## Memory Management

```rust
use aprender::autograd::{Tensor, clear_graph, clear_grad};

fn training_loop() {
    let mut model = MLP::new(10, 64, 2);

    for batch in 0..1000 {
        // Forward + backward
        let loss = compute_loss(&model);
        loss.backward();

        // Update parameters
        update_params(&mut model, 0.01);

        // IMPORTANT: Clear graph after each iteration
        clear_graph();

        // Optionally clear individual gradients
        for param in model.parameters() {
            clear_grad(param.id());
        }
    }
}
```

## Running Examples

```bash
# Basic autograd demo
cargo run --example autograd_basics

# Train a simple model
cargo run --example autograd_training

# Gradient checking
cargo run --example gradient_check
```

## References

- Baydin et al. (2018). "Automatic differentiation in machine learning: a survey." JMLR.
- Rumelhart et al. (1986). "Learning representations by back-propagating errors." Nature.
- Griewank & Walther (2008). "Evaluating derivatives." SIAM.
