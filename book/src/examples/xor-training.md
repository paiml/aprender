# Case Study: XOR Neural Network Training

The "Hello World" of deep learning - proving non-linear learning works.

## Why XOR?

XOR is **not linearly separable**:

```text
    X2
    │
  1 │  ●         ○
    │
  0 │  ○         ●
    └──────────────── X1
       0         1

● = Output 1
○ = Output 0
```

No single line can separate the classes. A neural network with hidden layers can learn this.

## Implementation

```rust,ignore
use aprender::autograd::{clear_graph, Tensor};
use aprender::nn::{
    loss::MSELoss, optim::SGD,
    Linear, Module, Optimizer, ReLU, Sequential, Sigmoid,
};

fn main() {
    // XOR truth table
    let x_data = vec![
        vec![0.0, 0.0],  // → 0
        vec![0.0, 1.0],  // → 1
        vec![1.0, 0.0],  // → 1
        vec![1.0, 1.0],  // → 0
    ];
    let y_data = vec![0.0, 1.0, 1.0, 0.0];

    // Network: 2 → 4 → 4 → 1
    let mut model = Sequential::new()
        .add(Linear::new(2, 4))
        .add(ReLU::new())
        .add(Linear::new(4, 4))
        .add(ReLU::new())
        .add(Linear::new(4, 1))
        .add(Sigmoid::new());

    let mut optimizer = SGD::new(model.parameters(), 0.5);
    let loss_fn = MSELoss::new();

    // Training
    for epoch in 0..5000 {
        clear_graph();

        let x = Tensor::from_vec(x_data.clone().concat(), &[4, 2]);
        let y = Tensor::from_vec(y_data.clone(), &[4, 1]);

        let pred = model.forward(&x);
        let loss = loss_fn.forward(&pred, &y);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        if epoch % 1000 == 0 {
            println!("Epoch {}: loss = {:.6}", epoch, loss.data()[0]);
        }
    }

    // Test
    println!("\nResults:");
    for (input, expected) in x_data.iter().zip(y_data.iter()) {
        let x = Tensor::from_vec(input.clone(), &[1, 2]);
        let pred = model.forward(&x);
        let output = pred.data()[0];
        println!(
            "  ({}, {}) → {:.3} (expected {})",
            input[0], input[1], output, expected
        );
    }
}
```

## Expected Output

```text
Epoch 0: loss = 0.250000
Epoch 1000: loss = 0.045123
Epoch 2000: loss = 0.008234
Epoch 3000: loss = 0.002156
Epoch 4000: loss = 0.000891

Results:
  (0, 0) → 0.012 (expected 0)
  (0, 1) → 0.987 (expected 1)
  (1, 0) → 0.991 (expected 1)
  (1, 1) → 0.008 (expected 0)
```

## Key Takeaways

1. **Hidden layers enable non-linear decision boundaries**
2. **ReLU activation** introduces non-linearity
3. **Sigmoid output** squashes to [0, 1] for binary classification
4. **SGD with momentum** works well for small networks

## Run

```bash
cargo run --example xor_training
```
