# Callback System

The callback system provides extensible hooks into the training loop,
enabling behaviors like early stopping, checkpointing, progress
logging, and real-time monitoring without modifying the core trainer.

## Overview

```rust
use entrenar::train::{
    TrainerCallback, CallbackContext, CallbackAction, CallbackManager,
    EarlyStopping, CheckpointCallback, ProgressCallback, MonitorCallback,
};

// Add multiple callbacks
trainer.add_callback(EarlyStopping::new(5, 0.001));
trainer.add_callback(CheckpointCallback::new("./ckpt"));
trainer.add_callback(ProgressCallback::new(10));
trainer.add_callback(MonitorCallback::new());
```

## Callback Lifecycle

Callbacks fire at six points during training:

```
train()
  │
  ├─► on_train_begin
  │
  ├─► for epoch in 0..max_epochs:
  │     │
  │     ├─► on_epoch_begin
  │     │
  │     ├─► for batch in batches:
  │     │     ├─► on_step_begin
  │     │     ├─► train_step()
  │     │     └─► on_step_end
  │     │
  │     └─► on_epoch_end
  │
  └─► on_train_end
```

## CallbackAction

Callbacks return an action that controls training flow:

```rust
pub enum CallbackAction {
    Continue,   // Continue training normally
    Stop,       // Stop training immediately
    SkipEpoch,  // Skip to next epoch (epoch_begin only)
}
```

**Behavior:**
- `Continue` - Training proceeds normally
- `Stop` - Training stops, `TrainResult.stopped_early = true`
- `SkipEpoch` - Skip remaining steps in current epoch (only valid in `on_epoch_begin`)

## CallbackContext

Every callback receives context with current training state:

```rust
pub struct CallbackContext {
    pub epoch: usize,           // Current epoch (0-indexed)
    pub max_epochs: usize,      // Maximum epochs configured
    pub step: usize,            // Current step within epoch
    pub steps_per_epoch: usize, // Total steps in epoch
    pub global_step: usize,     // Total steps across all epochs
    pub loss: f32,              // Current/latest loss
    pub lr: f32,                // Current learning rate
    pub best_loss: Option<f32>, // Best loss achieved so far
    pub val_loss: Option<f32>,  // Validation loss (if provided)
    pub elapsed_secs: f64,      // Seconds since training started
}
```

## TrainerCallback Trait

```rust
pub trait TrainerCallback: Send {
    /// Called before training begins
    fn on_train_begin(&mut self, ctx: &CallbackContext) -> CallbackAction {
        CallbackAction::Continue
    }

    /// Called after training ends
    fn on_train_end(&mut self, ctx: &CallbackContext) {}

    /// Called at the start of each epoch
    fn on_epoch_begin(&mut self, ctx: &CallbackContext) -> CallbackAction {
        CallbackAction::Continue
    }

    /// Called at the end of each epoch
    fn on_epoch_end(&mut self, ctx: &CallbackContext) -> CallbackAction {
        CallbackAction::Continue
    }

    /// Called before each training step
    fn on_step_begin(&mut self, ctx: &CallbackContext) -> CallbackAction {
        CallbackAction::Continue
    }

    /// Called after each training step
    fn on_step_end(&mut self, ctx: &CallbackContext) -> CallbackAction {
        CallbackAction::Continue
    }

    /// Callback name for logging
    fn name(&self) -> &str;
}
```

## Built-in Callbacks

### EarlyStopping

Stops training when loss stops improving:

```rust
pub struct EarlyStopping {
    patience: usize,      // Epochs to wait before stopping
    min_delta: f32,       // Minimum improvement threshold
    best_loss: f32,       // Best loss seen
    epochs_without_improvement: usize,
}

// Usage
let es = EarlyStopping::new(5, 0.001);
// Stops if loss doesn't improve by at least 0.001 for 5 epochs
```

**Behavior:**
- Tracks best loss seen during training
- Counts epochs without improvement (loss not decreasing by `min_delta`)
- Returns `CallbackAction::Stop` when patience exhausted

### CheckpointCallback

Saves model checkpoints periodically and/or when best loss achieved:

```rust
pub struct CheckpointCallback {
    save_dir: PathBuf,
    save_every: Option<usize>,  // Save every N epochs
    save_best: bool,            // Save when best loss achieved
    best_loss: f32,
}

// Usage
let ckpt = CheckpointCallback::new("./checkpoints")
    .save_every(5)      // Save every 5 epochs
    .save_best(true);   // Also save best model

// Creates files like:
// ./checkpoints/checkpoint_epoch_5.json
// ./checkpoints/checkpoint_epoch_10.json
// ./checkpoints/checkpoint_best.json
```

### ProgressCallback

Logs training progress to stdout:

```rust
pub struct ProgressCallback {
    log_interval: usize,  // Log every N steps
}

// Usage
let progress = ProgressCallback::new(10);
// Logs: "Epoch 1/100 [========>  ] Step 50/500: loss: 0.1234"
```

**Output format:**
```
Epoch 1/100 [========>  ] loss: 0.2345, lr: 0.001, elapsed: 12.3s
  Step 10/100: loss: 0.2456
  Step 20/100: loss: 0.2234
```

### MonitorCallback

Real-time monitoring with NaN/Inf detection and metrics collection:

```rust
pub struct MonitorCallback {
    collector: MetricsCollector,
    andon: AndonSystem,
}

// Usage
let monitor = MonitorCallback::new();
// Automatically:
// - Records loss, learning rate metrics
// - Detects NaN/Inf and triggers Stop
// - Integrates with Andon alerting system
```

**Automatic detection:**
- NaN loss → `CallbackAction::Stop`
- Inf loss → `CallbackAction::Stop`
- Triggers Andon alert for investigation

## Custom Callbacks

### Basic Example

```rust
use entrenar::train::{TrainerCallback, CallbackContext, CallbackAction};

struct LossLogger {
    losses: Vec<f32>,
}

impl LossLogger {
    fn new() -> Self {
        Self { losses: Vec::new() }
    }
}

impl TrainerCallback for LossLogger {
    fn on_epoch_end(&mut self, ctx: &CallbackContext) -> CallbackAction {
        self.losses.push(ctx.loss);
        println!("Epoch {}: loss = {:.6}", ctx.epoch, ctx.loss);
        CallbackAction::Continue
    }

    fn name(&self) -> &str {
        "LossLogger"
    }
}
```

### Learning Rate Warmup

```rust
struct WarmupCallback {
    warmup_epochs: usize,
    target_lr: f32,
}

impl TrainerCallback for WarmupCallback {
    fn on_epoch_begin(&mut self, ctx: &CallbackContext) -> CallbackAction {
        if ctx.epoch < self.warmup_epochs {
            let warmup_lr = self.target_lr * (ctx.epoch + 1) as f32
                          / self.warmup_epochs as f32;
            // Would need trainer access to set LR
            println!("Warmup LR: {:.6}", warmup_lr);
        }
        CallbackAction::Continue
    }

    fn name(&self) -> &str {
        "WarmupCallback"
    }
}
```

### Gradient Explosion Detector

```rust
struct GradientMonitor {
    max_loss: f32,
    loss_history: Vec<f32>,
}

impl TrainerCallback for GradientMonitor {
    fn on_step_end(&mut self, ctx: &CallbackContext) -> CallbackAction {
        self.loss_history.push(ctx.loss);

        // Detect sudden loss spike
        if self.loss_history.len() > 1 {
            let prev = self.loss_history[self.loss_history.len() - 2];
            if ctx.loss > prev * 10.0 {
                eprintln!("WARNING: Loss spike detected! {} -> {}", prev, ctx.loss);
                return CallbackAction::Stop;
            }
        }

        if ctx.loss > self.max_loss {
            eprintln!("ERROR: Loss exceeded threshold: {} > {}", ctx.loss, self.max_loss);
            return CallbackAction::Stop;
        }

        CallbackAction::Continue
    }

    fn name(&self) -> &str {
        "GradientMonitor"
    }
}
```

## CallbackManager

The `CallbackManager` orchestrates multiple callbacks:

```rust
pub struct CallbackManager {
    callbacks: Vec<Box<dyn TrainerCallback>>,
}

impl CallbackManager {
    pub fn new() -> Self;
    pub fn add<C: TrainerCallback + 'static>(&mut self, callback: C);
    pub fn is_empty(&self) -> bool;
    pub fn len(&self) -> usize;

    // Event dispatchers (called by Trainer)
    pub fn on_train_begin(&mut self, ctx: &CallbackContext) -> CallbackAction;
    pub fn on_train_end(&mut self, ctx: &CallbackContext);
    pub fn on_epoch_begin(&mut self, ctx: &CallbackContext) -> CallbackAction;
    pub fn on_epoch_end(&mut self, ctx: &CallbackContext) -> CallbackAction;
    pub fn on_step_begin(&mut self, ctx: &CallbackContext) -> CallbackAction;
    pub fn on_step_end(&mut self, ctx: &CallbackContext) -> CallbackAction;
}
```

**Dispatch behavior:**
- Callbacks fire in order they were added
- If any callback returns `Stop`, remaining callbacks don't fire
- `on_train_end` always fires (even after early stop)

## Best Practices

### Callback Order

Add callbacks in order of priority:

```rust
// Critical monitoring first
trainer.add_callback(MonitorCallback::new());     // NaN detection
trainer.add_callback(EarlyStopping::new(5, 0.001)); // Early stopping

// Logging/checkpointing after
trainer.add_callback(ProgressCallback::new(10));
trainer.add_callback(CheckpointCallback::new("./ckpt"));
```

### Stateful Callbacks

Callbacks can maintain state across training:

```rust
struct StatefulCallback {
    epoch_losses: Vec<f32>,
    best_epoch: usize,
}

impl TrainerCallback for StatefulCallback {
    fn on_epoch_end(&mut self, ctx: &CallbackContext) -> CallbackAction {
        self.epoch_losses.push(ctx.loss);

        if ctx.best_loss == Some(ctx.loss) {
            self.best_epoch = ctx.epoch;
        }

        CallbackAction::Continue
    }

    fn on_train_end(&mut self, ctx: &CallbackContext) {
        println!("Best epoch: {} with loss {:.6}",
            self.best_epoch,
            self.epoch_losses[self.best_epoch]);
    }

    fn name(&self) -> &str {
        "StatefulCallback"
    }
}
```

### Thread Safety

Callbacks must be `Send` to support potential future parallelism:

```rust
// Good: Uses Arc for shared state
struct ThreadSafeCallback {
    counter: Arc<AtomicUsize>,
}

// Bad: Uses Rc (not Send)
struct NotSendCallback {
    counter: Rc<RefCell<usize>>,  // Won't compile!
}
```

## See Also

- [Trainer API](./trainer-api.md) - Main trainer documentation
- [Early Stopping](./early-stopping.md) - Detailed early stopping guide
- [Checkpointing](./checkpointing.md) - Checkpoint management
- [Curriculum Learning](./curriculum-learning.md) - Progressive difficulty training (CITL)
- [Explainability](./explainability.md) - Feature attribution callbacks
- [Real-Time Monitoring](../monitor/overview.md) - Monitor integration
