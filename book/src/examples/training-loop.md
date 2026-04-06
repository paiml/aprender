# Basic Training Loop

This example demonstrates the fundamental training loop with gradient clipping, learning rate scheduling, and
epoch-based training.

## Running the Example

```bash
cargo run --example training_loop
```

## Code

```rust
{{#include ../../../examples/training_loop.rs}}
```

## Expected Output

```
=== Training Loop Example ===

Initial learning rate: 0.010000
Gradient clipping: enabled (max_norm=1.0)

Training data:
  Batches: 3
  Batch size: 3

Starting training...

Epoch 1: loss=1.0000, lr=0.010000
Epoch 2: loss=1.0000, lr=0.010000
...
Epoch 6: loss=1.0000, lr=0.010000
  → Reducing learning rate
Epoch 7: loss=1.0000, lr=0.001000
...

=== Training Complete ===

Training Metrics:
  Total epochs: 10
  Total steps: 30
```

## Key Concepts

### Training Configuration

```rust
let config = TrainConfig::new()
    .with_lr(0.01)
    .with_grad_clip(1.0)
    .with_log_interval(1);
```

### Epoch-Based Training

```rust
for epoch in 1..=num_epochs {
    let loss = trainer.train_epoch(&batches, |x| model.forward(x));

    // Learning rate scheduling
    if epoch % 6 == 0 {
        trainer.reduce_lr(0.1);
    }
}
```
