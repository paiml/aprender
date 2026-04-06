# Trainer API

The `Trainer` struct provides a high-level abstraction for training
neural networks with full callback support, automatic metrics
tracking, and gradient management.

## Overview

```rust
use entrenar::train::{Trainer, TrainConfig, Batch, MSELoss, EarlyStopping};
use entrenar::optim::Adam;
use entrenar::Tensor;

// Create trainer
let params = vec![Tensor::zeros(784 * 128, true)];
let optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);
let config = TrainConfig::default();

let mut trainer = Trainer::new(params, Box::new(optimizer), config);
trainer.set_loss(Box::new(MSELoss));

// Add callbacks
trainer.add_callback(EarlyStopping::new(5, 0.001));

// Train
let result = trainer.train(100, || batches.clone(), |x| model.forward(x));
```

## Trainer Struct

```rust
pub struct Trainer {
    params: Vec<Tensor>,           // Model parameters
    optimizer: Box<dyn Optimizer>, // Optimizer instance
    loss_fn: Option<Box<dyn LossFn>>, // Loss function
    config: TrainConfig,           // Training configuration
    pub metrics: MetricsTracker,   // Metrics tracking
    callbacks: CallbackManager,    // Callback system
    best_loss: Option<f32>,        // Best loss achieved
    start_time: Option<Instant>,   // Training start time
}
```

## Creating a Trainer

```rust
let trainer = Trainer::new(params, optimizer, config);
```

**Parameters:**
- `params: Vec<Tensor>` - Model parameters to optimize (must have `requires_grad = true`)
- `optimizer: Box<dyn Optimizer>` - Optimizer instance (SGD, Adam, AdamW)
- `config: TrainConfig` - Training configuration

## Setting the Loss Function

```rust
trainer.set_loss(Box::new(MSELoss));
// or
trainer.set_loss(Box::new(CrossEntropyLoss));
```

The loss function must be set before calling `train()` or `train_step()`.

## Adding Callbacks

```rust
use entrenar::train::{EarlyStopping, CheckpointCallback, ProgressCallback, MonitorCallback};

trainer.add_callback(EarlyStopping::new(5, 0.001));
trainer.add_callback(CheckpointCallback::new("./checkpoints"));
trainer.add_callback(ProgressCallback::new(10));
trainer.add_callback(MonitorCallback::new());
```

See [Callback System](#callback-system) for details on available callbacks.

## Training Methods

### `train()` - Full Training Loop

The primary method for training with full callback support:

```rust
pub fn train<F, B, I>(
    &mut self,
    max_epochs: usize,
    batch_fn: B,
    forward_fn: F,
) -> TrainResult
where
    F: Fn(&Tensor) -> Tensor,
    B: Fn() -> I,
    I: IntoIterator<Item = Batch>,
```

**Parameters:**
- `max_epochs` - Maximum number of epochs to train
- `batch_fn` - Function that returns batches for each epoch
- `forward_fn` - Model forward pass (inputs → predictions)

**Returns:** `TrainResult` with training outcome

**Example:**
```rust
let batches = vec![
    Batch::new(inputs1, targets1),
    Batch::new(inputs2, targets2),
];

let result = trainer.train(
    100,                          // max epochs
    || batches.clone(),           // batch function
    |x| model.forward(x),         // forward function
);

println!("Final epoch: {}", result.final_epoch);
println!("Final loss: {:.4}", result.final_loss);
println!("Best loss: {:.4}", result.best_loss);
println!("Stopped early: {}", result.stopped_early);
println!("Elapsed: {:.2}s", result.elapsed_secs);
```

### `train_epoch()` - Single Epoch

Train for one epoch without callback overhead:

```rust
pub fn train_epoch<F, I>(&mut self, batches: I, forward_fn: F) -> f32
where
    F: Fn(&Tensor) -> Tensor,
    I: IntoIterator<Item = Batch>,
```

**Returns:** Average loss for the epoch

### `train_step()` - Single Batch

Train on a single batch:

```rust
pub fn train_step<F>(&mut self, batch: &Batch, forward_fn: F) -> f32
where
    F: FnOnce(&Tensor) -> Tensor,
```

**Returns:** Loss for this batch

## TrainResult

```rust
#[derive(Debug, Clone)]
pub struct TrainResult {
    pub final_epoch: usize,    // Last epoch completed
    pub final_loss: f32,       // Loss at final epoch
    pub best_loss: f32,        // Best loss achieved
    pub stopped_early: bool,   // Whether early stopping triggered
    pub elapsed_secs: f64,     // Total training time
}
```

## Callback System

The trainer fires callbacks at six points in the training lifecycle:

| Event | Method | When |
|-------|--------|------|
| `on_train_begin` | `CallbackAction` | Before first epoch |
| `on_train_end` | `()` | After training completes |
| `on_epoch_begin` | `CallbackAction` | Before each epoch |
| `on_epoch_end` | `CallbackAction` | After each epoch |
| `on_step_begin` | `CallbackAction` | Before each batch |
| `on_step_end` | `CallbackAction` | After each batch |

### CallbackAction

Callbacks return an action that controls training flow:

```rust
pub enum CallbackAction {
    Continue,   // Continue training normally
    Stop,       // Stop training immediately
    SkipEpoch,  // Skip to next epoch (epoch_begin only)
}
```

### CallbackContext

Callbacks receive context with current training state:

```rust
pub struct CallbackContext {
    pub epoch: usize,           // Current epoch (0-indexed)
    pub max_epochs: usize,      // Maximum epochs
    pub step: usize,            // Current step in epoch
    pub steps_per_epoch: usize, // Total steps per epoch
    pub global_step: usize,     // Total steps across all epochs
    pub loss: f32,              // Current loss
    pub lr: f32,                // Current learning rate
    pub best_loss: Option<f32>, // Best loss so far
    pub val_loss: Option<f32>,  // Validation loss (if available)
    pub elapsed_secs: f64,      // Time since training started
}
```

### Built-in Callbacks

#### EarlyStopping

Stop training when loss stops improving:

```rust
let es = EarlyStopping::new(
    5,      // patience: epochs without improvement
    0.001,  // min_delta: minimum improvement threshold
);
trainer.add_callback(es);
```

#### CheckpointCallback

Save model checkpoints:

```rust
let ckpt = CheckpointCallback::new("./checkpoints")
    .save_every(5)      // Save every 5 epochs
    .save_best(true);   // Also save best model
trainer.add_callback(ckpt);
```

#### ProgressCallback

Log training progress:

```rust
let progress = ProgressCallback::new(10);  // Log every 10 steps
trainer.add_callback(progress);
```

#### MonitorCallback

Real-time monitoring with NaN/Inf detection:

```rust
let monitor = MonitorCallback::new();
trainer.add_callback(monitor);
// Automatically stops training on NaN/Inf loss
```

### Custom Callbacks

Implement `TrainerCallback` for custom behavior:

```rust
use entrenar::train::{TrainerCallback, CallbackContext, CallbackAction};

struct CustomCallback {
    // your state
}

impl TrainerCallback for CustomCallback {
    fn on_epoch_end(&mut self, ctx: &CallbackContext) -> CallbackAction {
        println!("Epoch {} complete, loss: {:.4}", ctx.epoch, ctx.loss);

        if ctx.loss > 100.0 {
            CallbackAction::Stop  // Loss exploded
        } else {
            CallbackAction::Continue
        }
    }

    fn name(&self) -> &str {
        "CustomCallback"
    }

    // Other methods have default implementations that return Continue
}

trainer.add_callback(CustomCallback { /* ... */ });
```

## Accessing Trainer State

```rust
// Learning rate
let lr = trainer.lr();
trainer.set_lr(0.0001);

// Parameters
let params = trainer.params();
let params_mut = trainer.params_mut();

// Callbacks
let callbacks = trainer.callbacks();
let callbacks_mut = trainer.callbacks_mut();
```

## Complete Example

```rust
use entrenar::train::{
    Trainer, TrainConfig, TrainResult, Batch, MSELoss,
    EarlyStopping, CheckpointCallback, ProgressCallback, MonitorCallback,
};
use entrenar::optim::Adam;
use entrenar::Tensor;

fn main() {
    // Model parameters
    let params = vec![
        Tensor::randn(784 * 256, true),  // Layer 1
        Tensor::randn(256 * 10, true),   // Layer 2
    ];

    // Optimizer
    let optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);

    // Config
    let config = TrainConfig::new()
        .with_max_grad_norm(1.0)
        .with_log_interval(100);

    // Create trainer
    let mut trainer = Trainer::new(params, Box::new(optimizer), config);
    trainer.set_loss(Box::new(MSELoss));

    // Add callbacks
    trainer.add_callback(EarlyStopping::new(10, 0.0001));
    trainer.add_callback(CheckpointCallback::new("./ckpt").save_every(5));
    trainer.add_callback(ProgressCallback::new(50));
    trainer.add_callback(MonitorCallback::new());

    // Training data
    let batches: Vec<Batch> = load_training_data();

    // Train
    let result: TrainResult = trainer.train(
        100,
        || batches.clone(),
        |x| forward_pass(x, trainer.params()),
    );

    // Results
    println!("Training complete!");
    println!("  Epochs: {}", result.final_epoch);
    println!("  Final loss: {:.6}", result.final_loss);
    println!("  Best loss: {:.6}", result.best_loss);
    println!("  Early stopped: {}", result.stopped_early);
    println!("  Time: {:.1}s", result.elapsed_secs);
}
```

## See Also

- [Train Config](./train-config.md) - Configuration options
- [Early Stopping](./early-stopping.md) - Early stopping details
- [Checkpointing](./checkpointing.md) - Checkpoint management
- [Loss Functions](./loss-functions.md) - Available loss functions
