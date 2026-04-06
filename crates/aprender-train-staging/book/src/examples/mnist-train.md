# MNIST Training (CPU)

This example trains a neural network on the MNIST handwritten digit dataset using CPU computation.

## Running the Example

```bash
cargo run --example mnist_train
```

## Code

```rust
{{#include ../../../examples/mnist_train.rs}}
```

## Key Features

- **Multi-layer perceptron** architecture
- **Cross-entropy loss** for classification
- **Adam optimizer** with learning rate scheduling
- **Batch processing** for efficient training
- **Validation metrics** (accuracy, loss)

## Architecture

```
Input (784) → Hidden1 (256) → ReLU → Hidden2 (128) → ReLU → Output (10)
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch size | 64 |
| Learning rate | 0.001 |
| Epochs | 10 |
| Hidden layers | 2 |
| Optimizer | Adam |
