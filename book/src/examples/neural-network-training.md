# Case Study: Neural Network Training Pipeline

Complete deep learning workflow with aprender's nn module.

## Features Demonstrated

- Multi-layer perceptron (MLP)
- Backpropagation training
- Optimizers (Adam, SGD)
- Learning rate schedulers
- Model serialization

## Problem: XOR Function

Learn the classic non-linearly separable XOR:

| X1 | X2 | Output |
|----|----|----|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

## Architecture

```text
Input (2) → Linear(8) → ReLU → Linear(8) → ReLU → Linear(1) → Sigmoid
```

## Implementation

```rust,ignore
use aprender::autograd::Tensor;
use aprender::nn::{
    loss::MSELoss,
    optim::{Adam, Optimizer},
    scheduler::{LRScheduler, StepLR},
    serialize::{save_model, load_model},
    Linear, Module, ReLU, Sequential, Sigmoid,
};

fn main() {
    // Build network
    let mut model = Sequential::new()
        .add(Linear::new(2, 8))
        .add(ReLU::new())
        .add(Linear::new(8, 8))
        .add(ReLU::new())
        .add(Linear::new(8, 1))
        .add(Sigmoid::new());

    // XOR data
    let x_data = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let y_data = vec![0.0, 1.0, 1.0, 0.0];

    let mut optimizer = Adam::new(model.parameters(), 0.1);
    let mut scheduler = StepLR::new(&mut optimizer, 500, 0.5);
    let loss_fn = MSELoss::new();

    // Train
    for epoch in 0..2000 {
        let x = Tensor::from_vec(x_data.clone(), &[4, 2]);
        let y = Tensor::from_vec(y_data.clone(), &[4, 1]);

        let pred = model.forward(&x);
        let loss = loss_fn.forward(&pred, &y);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        scheduler.step();

        if epoch % 500 == 0 {
            println!("Epoch {}: loss = {:.6}", epoch, loss.data()[0]);
        }
    }

    // Save model
    save_model(&model, "xor_model.bin").unwrap();

    // Load and verify
    let loaded: Sequential = load_model("xor_model.bin").unwrap();
    println!("Model loaded, params: {}", count_parameters(&loaded));
}
```

## Key Concepts

1. **StepLR**: Decay learning rate every N epochs
2. **save_model/load_model**: Binary serialization
3. **ReLU activation**: Enables non-linear learning

## Run

```bash
cargo run --example neural_network_training
```
