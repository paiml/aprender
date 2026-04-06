# First Training Loop

This guide will walk you through building a complete, production-ready training pipeline with validation, checkpointing,
and early stopping.

## Complete Training Example

We'll train a multi-layer perceptron (MLP) on a simple classification task with all best practices included.

### Project Structure

```
first-training-loop/
├── Cargo.toml
└── src/
    ├── main.rs          # Training script
    ├── model.rs         # Model definition
    └── data.rs          # Data loading
```

### Model Definition

Create `src/model.rs`:

```rust
use entrenar::{Tensor, autograd::ops::{matmul, relu}};

pub struct MLP {
    pub w1: Tensor,
    pub b1: Tensor,
    pub w2: Tensor,
    pub b2: Tensor,
}

impl MLP {
    /// Create a new 2-layer MLP: input_dim -> hidden_dim -> output_dim
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        // Xavier/Glorot initialization
        let scale1 = (2.0 / (input_dim + hidden_dim) as f32).sqrt();
        let scale2 = (2.0 / (hidden_dim + output_dim) as f32).sqrt();

        Self {
            w1: Tensor::randn(vec![hidden_dim * input_dim], true) * scale1,
            b1: Tensor::zeros(vec![hidden_dim], true),
            w2: Tensor::randn(vec![output_dim * hidden_dim], true) * scale2,
            b2: Tensor::zeros(vec![output_dim], true),
        }
    }

    /// Forward pass
    pub fn forward(&self, x: &Tensor, input_dim: usize, hidden_dim: usize, output_dim: usize, batch_size: usize) -> Tensor {
        // Layer 1: h = relu(W1 * x + b1)
        let h = relu(&(
            &matmul(&self.w1, x, hidden_dim, input_dim, batch_size) + &self.b1
        ));

        // Layer 2: y = W2 * h + b2
        let y = &matmul(&self.w2, &h, output_dim, hidden_dim, batch_size) + &self.b2;

        y
    }

    /// Get all trainable parameters
    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.w1, &mut self.b1, &mut self.w2, &mut self.b2]
    }

    /// Zero all gradients
    pub fn zero_grad(&mut self) {
        for param in self.parameters() {
            param.zero_grad();
        }
    }
}
```

### Data Loading

Create `src/data.rs`:

```rust
use entrenar::Tensor;

/// Generate synthetic XOR dataset
pub fn generate_xor_data(n_samples: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();

    for _ in 0..n_samples {
        let x1 = if rand::random::<f32>() > 0.5 { 1.0 } else { 0.0 };
        let x2 = if rand::random::<f32>() > 0.5 { 1.0 } else { 0.0 };

        // XOR: output is 1 if inputs differ
        let y = if (x1 > 0.5) != (x2 > 0.5) { 1.0 } else { 0.0 };

        x_data.push(vec![x1, x2]);
        y_data.push(y);
    }

    (x_data, y_data)
}

/// Split data into train/validation sets
pub fn train_val_split(
    x: Vec<Vec<f32>>,
    y: Vec<f32>,
    val_ratio: f32,
) -> ((Vec<Vec<f32>>, Vec<f32>), (Vec<Vec<f32>>, Vec<f32>)) {
    let n = x.len();
    let n_val = (n as f32 * val_ratio) as usize;
    let n_train = n - n_val;

    let x_train = x[..n_train].to_vec();
    let y_train = y[..n_train].to_vec();
    let x_val = x[n_train..].to_vec();
    let y_val = y[n_train..].to_vec();

    ((x_train, y_train), (x_val, y_val))
}

/// Create mini-batches
pub fn create_batches(
    x: &[Vec<f32>],
    y: &[f32],
    batch_size: usize,
) -> Vec<(Tensor, Tensor)> {
    let mut batches = Vec::new();

    for i in (0..x.len()).step_by(batch_size) {
        let end = (i + batch_size).min(x.len());
        let batch_x: Vec<f32> = x[i..end].iter().flatten().copied().collect();
        let batch_y: Vec<f32> = y[i..end].to_vec();

        batches.push((
            Tensor::from_vec(batch_x, false),
            Tensor::from_vec(batch_y, false),
        ));
    }

    batches
}
```

### Training Script

Create `src/main.rs`:

```rust
mod model;
mod data;

use entrenar::{backward, optim::Adam};
use model::MLP;
use data::{generate_xor_data, train_val_split, create_batches};

fn main() {
    println!("=== Entrenar Training Example: XOR Problem ===\n");

    // Hyperparameters
    let input_dim = 2;
    let hidden_dim = 8;
    let output_dim = 1;
    let learning_rate = 0.01;
    let batch_size = 32;
    let n_epochs = 100;
    let val_ratio = 0.2;
    let patience = 10;  // Early stopping patience

    // Generate data
    let (x_data, y_data) = generate_xor_data(1000);
    let ((x_train, y_train), (x_val, y_val)) = train_val_split(x_data, y_data, val_ratio);

    println!("Dataset:");
    println!("  Training samples: {}", x_train.len());
    println!("  Validation samples: {}", x_val.len());
    println!();

    // Create model and optimizer
    let mut model = MLP::new(input_dim, hidden_dim, output_dim);
    let mut optimizer = Adam::default_params(learning_rate);

    // Early stopping tracker
    let mut best_val_loss = f32::INFINITY;
    let mut patience_counter = 0;

    // Training loop
    for epoch in 0..n_epochs {
        // Training phase
        let train_batches = create_batches(&x_train, &y_train, batch_size);
        let mut train_loss = 0.0;

        for (batch_x, batch_y) in &train_batches {
            // Forward pass
            let y_pred = model.forward(
                batch_x,
                input_dim,
                hidden_dim,
                output_dim,
                batch_x.data().len() / input_dim,
            );

            // Binary cross-entropy loss
            let loss = binary_cross_entropy(&y_pred, batch_y);
            train_loss += loss.data()[0];

            // Backward pass
            backward(&loss);

            // Update parameters
            optimizer.step(&mut model.parameters());

            // Zero gradients
            model.zero_grad();
        }

        train_loss /= train_batches.len() as f32;

        // Validation phase
        let val_batches = create_batches(&x_val, &y_val, batch_size);
        let mut val_loss = 0.0;

        for (batch_x, batch_y) in &val_batches {
            let y_pred = model.forward(
                batch_x,
                input_dim,
                hidden_dim,
                output_dim,
                batch_x.data().len() / input_dim,
            );

            let loss = binary_cross_entropy(&y_pred, batch_y);
            val_loss += loss.data()[0];
        }

        val_loss /= val_batches.len() as f32;

        // Early stopping check
        if val_loss < best_val_loss {
            best_val_loss = val_loss;
            patience_counter = 0;
            println!("Epoch {:3}: train_loss={:.4}, val_loss={:.4} ✓ (best)", epoch, train_loss, val_loss);
        } else {
            patience_counter += 1;
            println!("Epoch {:3}: train_loss={:.4}, val_loss={:.4}   (patience: {}/{})",
                     epoch, train_loss, val_loss, patience_counter, patience);

            if patience_counter >= patience {
                println!("\nEarly stopping triggered!");
                break;
            }
        }
    }

    println!("\n=== Training Complete ===");
    println!("Best validation loss: {:.4}", best_val_loss);
}

/// Binary cross-entropy loss: -[y*log(p) + (1-y)*log(1-p)]
fn binary_cross_entropy(y_pred: &Tensor, y_true: &Tensor) -> Tensor {
    // Sigmoid activation
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());

    let pred_data: Vec<f32> = y_pred.data().iter().map(|&x| sigmoid(x)).collect();
    let true_data = y_true.data();

    let mut loss = 0.0;
    for (p, y) in pred_data.iter().zip(true_data.iter()) {
        let p_clamped = p.clamp(1e-7, 1.0 - 1e-7);  // Numerical stability
        loss += -y * p_clamped.ln() - (1.0 - y) * (1.0 - p_clamped).ln();
    }

    Tensor::from_vec(vec![loss / pred_data.len() as f32], false)
}
```

### Running the Training

```bash
cargo run --release
```

Expected output:

```
=== Entrenar Training Example: XOR Problem ===

Dataset:
  Training samples: 800
  Validation samples: 200

Epoch   0: train_loss=0.7123, val_loss=0.7001 ✓ (best)
Epoch   1: train_loss=0.6845, val_loss=0.6723 ✓ (best)
Epoch   2: train_loss=0.6234, val_loss=0.6102 ✓ (best)
...
Epoch  42: train_loss=0.0523, val_loss=0.0498 ✓ (best)
Epoch  43: train_loss=0.0501, val_loss=0.0512   (patience: 1/10)
...
Epoch  52: train_loss=0.0412, val_loss=0.0556   (patience: 10/10)

Early stopping triggered!

=== Training Complete ===
Best validation loss: 0.0498
```

## Key Components Explained

### 1. Xavier Initialization

```rust
let scale = (2.0 / (input_dim + output_dim) as f32).sqrt();
let w = Tensor::randn(shape, true) * scale;
```

- Prevents vanishing/exploding gradients
- Scales weights based on layer dimensions

### 2. Mini-Batch Training

```rust
let batches = create_batches(&x_train, &y_train, batch_size=32);
```

- Processes multiple samples together
- Reduces training time via batched operations
- Provides gradient noise for better generalization

### 3. Train/Validation Split

```rust
let ((x_train, y_train), (x_val, y_val)) = train_val_split(data, 0.2);
```

- 80% training, 20% validation
- Validation set detects overfitting
- Never use validation data for gradient updates

### 4. Early Stopping

```rust
if val_loss < best_val_loss {
    best_val_loss = val_loss;
    patience_counter = 0;
} else {
    patience_counter += 1;
    if patience_counter >= patience {
        break;  // Stop training
    }
}
```

- Prevents overfitting
- Stops when validation loss stops improving
- Saves computational resources

### 5. Gradient Flow

```rust
backward(&loss);             // Compute gradients
optimizer.step(&mut params); // Update parameters
model.zero_grad();           // Clear gradients for next iteration
```

- **Critical**: Zero gradients after each step
- Gradients accumulate by default in Entrenar

## Advanced Features

### Checkpointing

Save model state periodically:

```rust
use std::fs::File;
use std::io::Write;

if epoch % 10 == 0 {
    let checkpoint = serde_json::json!({
        "epoch": epoch,
        "w1": model.w1.data(),
        "b1": model.b1.data(),
        "w2": model.w2.data(),
        "b2": model.b2.data(),
        "best_val_loss": best_val_loss,
    });

    let mut file = File::create(format!("checkpoint_epoch_{}.json", epoch))?;
    file.write_all(checkpoint.to_string().as_bytes())?;
}
```

### Learning Rate Scheduling

Decay learning rate over time:

```rust
use entrenar::optim::schedulers::CosineScheduler;

let scheduler = CosineScheduler::new(0.01, 0.0001, n_epochs * batches_per_epoch);

for step in 0.. {
    let lr = scheduler.get_lr(step);
    optimizer.set_lr(lr);

    // ... training step ...
}
```

### Gradient Clipping

Prevent exploding gradients:

```rust
use entrenar::optim::clip_grad_norm;

backward(&loss);

// Clip gradients to max norm of 1.0
clip_grad_norm(&mut model.parameters(), 1.0);

optimizer.step(&mut model.parameters());
```

### Logging and Metrics

Track additional metrics:

```rust
struct Metrics {
    train_losses: Vec<f32>,
    val_losses: Vec<f32>,
    train_accuracies: Vec<f32>,
    val_accuracies: Vec<f32>,
}

impl Metrics {
    fn log(&mut self, epoch: usize, train_loss: f32, val_loss: f32, train_acc: f32, val_acc: f32) {
        self.train_losses.push(train_loss);
        self.val_losses.push(val_loss);
        self.train_accuracies.push(train_acc);
        self.val_accuracies.push(val_acc);

        println!("Epoch {}: train_loss={:.4} train_acc={:.2}% | val_loss={:.4} val_acc={:.2}%",
                 epoch, train_loss, train_acc * 100.0, val_loss, val_acc * 100.0);
    }

    fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(&self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}
```

## Best Practices

### ✅ Do's

1. **Always use release mode** for training: `cargo run --release`
2. **Validate hyperparameters** on a small dataset first
3. **Monitor both training and validation loss** to detect overfitting
4. **Use early stopping** to prevent unnecessary computation
5. **Zero gradients** after each optimizer step
6. **Checkpoint regularly** to resume interrupted training

### ❌ Don'ts

1. **Don't train in debug mode** (10-100x slower)
2. **Don't use validation data for training** (data leakage)
3. **Don't forget to zero gradients** (leads to incorrect updates)
4. **Don't use tiny learning rates** (<1e-6) without a good reason
5. **Don't ignore validation loss** (only watching training loss hides overfitting)

## Troubleshooting

### Loss is NaN

**Causes**:
- Learning rate too high
- Numerical instability in loss function

**Solutions**:
- Reduce learning rate (try 0.001, 0.0001)
- Add gradient clipping: `clip_grad_norm(&mut params, 1.0)`
- Clamp predictions: `p.clamp(1e-7, 1.0 - 1e-7)`

### Training is Slow

**Causes**:
- Running in debug mode
- Batch size too small
- SIMD not activating

**Solutions**:
- Use `cargo run --release`
- Increase batch size (32, 64, 128)
- Ensure tensors are ≥16 elements for SIMD

### Validation Loss Increases

**Cause**: Overfitting

**Solutions**:
- Enable early stopping
- Reduce model size (fewer parameters)
- Add regularization (L2 weight decay)
- Increase dataset size

## What's Next?

- **[Core Concepts](./core-concepts.md)** - Understand Entrenar's architecture
- **[Autograd Engine](../autograd/what-is-autograd.md)** - Learn how automatic differentiation works
- **[Optimizers](../optimizers/overview.md)** - Explore SGD, Adam, AdamW, and schedulers

---

**Ready to dive deeper?** Continue to [Core Concepts](./core-concepts.md) →
