# Case Study: XOR Neural Network

The XOR problem is the "Hello World" of deep learning - a classic benchmark that proves a neural network can learn non-linear patterns through backpropagation.

## Why XOR Matters

XOR (exclusive or) is **not linearly separable**. No single straight line can separate the classes:

```
    X2
    │
  1 │  ●(0,1)=1     ○(1,1)=0
    │
    ├───────────────────── X1
    │
  0 │  ○(0,0)=0     ●(1,0)=1
    │
        0           1
```

This means:
- **Perceptrons fail** (single-layer networks)
- **Hidden layers required** to create non-linear decision boundaries
- **Proves backpropagation works** when the network learns XOR

## The Mathematics

### Truth Table

| X1 | X2 | XOR Output |
|----|----|----|
| 0  | 0  | 0  |
| 0  | 1  | 1  |
| 1  | 0  | 1  |
| 1  | 1  | 0  |

### Network Architecture

```
Input(2) → Linear(2→8) → ReLU → Linear(8→1) → Sigmoid
```

- **Input layer**: 2 features (X1, X2)
- **Hidden layer**: 8 neurons with ReLU activation
- **Output layer**: 1 neuron with Sigmoid (outputs probability)

Total parameters: `2×8 + 8 + 8×1 + 1 = 33`

## Implementation

```rust
use aprender::autograd::{clear_graph, Tensor};
use aprender::nn::{
    loss::MSELoss, optim::SGD, Linear, Module, Optimizer,
    ReLU, Sequential, Sigmoid,
};

fn main() {
    // XOR dataset
    let x = Tensor::new(&[
        0.0, 0.0,  // → 0
        0.0, 1.0,  // → 1
        1.0, 0.0,  // → 1
        1.0, 1.0,  // → 0
    ], &[4, 2]);

    let y = Tensor::new(&[0.0, 1.0, 1.0, 0.0], &[4, 1]);

    // Build network
    let mut model = Sequential::new()
        .add(Linear::with_seed(2, 8, Some(42)))
        .add(ReLU::new())
        .add(Linear::with_seed(8, 1, Some(43)))
        .add(Sigmoid::new());

    // Setup training
    let mut optimizer = SGD::new(model.parameters_mut(), 0.5);
    let loss_fn = MSELoss::new();

    // Training loop
    for epoch in 0..1000 {
        clear_graph();

        // Forward pass
        let x_grad = x.clone().requires_grad();
        let output = model.forward(&x_grad);

        // Compute loss
        let loss = loss_fn.forward(&output, &y);

        // Backward pass
        loss.backward();

        // Update weights
        let mut params = model.parameters_mut();
        optimizer.step_with_params(&mut params);
        optimizer.zero_grad();

        if epoch % 100 == 0 {
            println!("Epoch {}: Loss = {:.6}", epoch, loss.item());
        }
    }

    // Evaluate
    let final_output = model.forward(&x);
    println!("Predictions: {:?}", final_output.data());
}
```

## Training Dynamics

### Loss Curve

```
Epoch     Loss        Accuracy
─────────────────────────────
    0     0.304618      50%
  100     0.081109     100%
  200     0.013253     100%
  300     0.005368     100%
  500     0.002103     100%
 1000     0.000725     100%
```

The network:
1. **Starts random** (50% accuracy = random guessing)
2. **Learns quickly** (100% by epoch 100)
3. **Refines confidence** (loss continues decreasing)

### Final Predictions

| Input | Target | Prediction | Confidence |
|-------|--------|------------|------------|
| (0,0) | 0 | 0.034 | 96.6% |
| (0,1) | 1 | 0.977 | 97.7% |
| (1,0) | 1 | 0.974 | 97.4% |
| (1,1) | 0 | 0.023 | 97.7% |

## Key Concepts Demonstrated

### 1. Automatic Differentiation

```rust
loss.backward();  // Computes ∂L/∂w for all weights
```

The autograd engine:
- Records operations during forward pass
- Computes gradients in reverse (backpropagation)
- Handles chain rule automatically

### 2. Non-Linear Activation

```rust
.add(ReLU::new())  // f(x) = max(0, x)
```

ReLU enables the network to learn non-linear decision boundaries. Without it, stacking linear layers would still be linear.

### 3. Gradient Descent

```rust
optimizer.step_with_params(&mut params);
```

Updates weights: `w = w - lr × ∂L/∂w`

With learning rate 0.5, the network converges in ~100 epochs.

## Running the Example

```bash
cargo run --example xor_training
```

## Exercises

1. **Change hidden size**: Try 4 or 16 neurons instead of 8
2. **Change learning rate**: What happens with lr=0.1 or lr=1.0?
3. **Use Adam optimizer**: Replace SGD with Adam
4. **Add another hidden layer**: Does it help or hurt?

## Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| Loss stuck at ~0.25 | Vanishing gradients | Increase learning rate |
| Loss oscillates | Learning rate too high | Decrease learning rate |
| 50% accuracy | Not learning | Check gradient flow |

## Theory: Universal Approximation

The XOR example demonstrates the **Universal Approximation Theorem**: a neural network with one hidden layer can approximate any continuous function, given enough neurons.

XOR requires learning a function like:
```
f(x1, x2) ≈ x1(1-x2) + x2(1-x1)
```

The hidden layer learns intermediate features that make this separable.

## Next Steps

- [Classification Training](./classification-training.md) - Multi-class with CrossEntropy
- MNIST Digits - Real image classification (planned)
