# Case Study: Neural Network Classification

Train a multi-class classifier using aprender's neural network module.

## Problem: Quadrant Classification

Classify 2D points into 4 quadrants:
- Q1: (+x, +y) → Class 0
- Q2: (-x, +y) → Class 1
- Q3: (-x, -y) → Class 2
- Q4: (+x, -y) → Class 3

## Architecture

```text
Input (2) → Linear(16) → ReLU → Linear(16) → ReLU → Linear(4) → Softmax
```

## Implementation

```rust,ignore
use aprender::autograd::Tensor;
use aprender::nn::{
    loss::CrossEntropyLoss, optim::Adam,
    Linear, Module, Optimizer, ReLU, Sequential, Softmax,
};

fn main() {
    // Build classifier
    let mut model = Sequential::new()
        .add(Linear::new(2, 16))
        .add(ReLU::new())
        .add(Linear::new(16, 16))
        .add(ReLU::new())
        .add(Linear::new(16, 4))
        .add(Softmax::new(1));

    // Training data: points in each quadrant
    let x_data = vec![
        vec![1.0, 1.0], vec![0.5, 0.8],   // Q1
        vec![-1.0, 1.0], vec![-0.7, 0.9], // Q2
        vec![-1.0, -1.0], vec![-0.8, -0.5], // Q3
        vec![1.0, -1.0], vec![0.6, -0.7], // Q4
    ];
    let y_labels = vec![0, 0, 1, 1, 2, 2, 3, 3]; // One-hot encoded

    let mut optimizer = Adam::new(model.parameters(), 0.01);
    let loss_fn = CrossEntropyLoss::new();

    // Training loop
    for epoch in 0..1000 {
        let x = Tensor::from_vec(x_data.clone(), &[8, 2]);
        let y = one_hot_encode(&y_labels, 4);

        let pred = model.forward(&x);
        let loss = loss_fn.forward(&pred, &y);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        if epoch % 100 == 0 {
            println!("Epoch {}: loss = {:.4}", epoch, loss.data()[0]);
        }
    }
}
```

## Key Concepts

1. **CrossEntropyLoss**: Multi-class classification loss
2. **Softmax**: Converts logits to probabilities
3. **One-hot encoding**: Target format for multi-class

## Run

```bash
cargo run --example classification_training
```
