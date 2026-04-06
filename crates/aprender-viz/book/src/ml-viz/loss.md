# Loss Curves

Loss curves visualize model training progress, showing how loss decreases
over epochs or iterations.

## Basic Loss Curve

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::LossCurve;

let epochs: Vec<f32> = (1..=100).map(|e| e as f32).collect();
let train_loss: Vec<f32> = epochs.iter()
    .map(|e| 1.0 / (1.0 + e * 0.05))
    .collect();

let loss = LossCurve::new()
    .epochs(&epochs)
    .train_loss(&train_loss)
    .build();
```

**Test Reference**: `src/plots/loss.rs::test_loss_curve_basic`

## Training and Validation Loss

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::LossCurve;

let loss = LossCurve::new()
    .epochs(&epochs)
    .train_loss(&train_loss)
    .val_loss(&val_loss)
    .build();
```

## Customization

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::LossCurve;

let loss = LossCurve::new()
    .epochs(&epochs)
    .train_loss(&train_loss)
    .val_loss(&val_loss)
    .train_color(Rgba::BLUE)
    .val_color(Rgba::RED)
    .line_width(2.0)
    .title("Training Progress")
    .xlabel("Epoch")
    .ylabel("Loss")
    .build();
```

## Log Scale

For losses spanning multiple orders of magnitude:

```rust
use trueno_viz::plots::LossCurve;

let loss = LossCurve::new()
    .epochs(&epochs)
    .train_loss(&train_loss)
    .y_log_scale(true)
    .build();
```

## Early Stopping Marker

```rust
use trueno_viz::plots::LossCurve;

let loss = LossCurve::new()
    .epochs(&epochs)
    .train_loss(&train_loss)
    .val_loss(&val_loss)
    .mark_early_stopping(75)  // Stopped at epoch 75
    .build();
```

## Multiple Metrics

```rust
use trueno_viz::plots::LossCurve;

let loss = LossCurve::new()
    .epochs(&epochs)
    .add_metric("Train Loss", &train_loss, Rgba::BLUE)
    .add_metric("Val Loss", &val_loss, Rgba::RED)
    .add_metric("Train Acc", &train_acc, Rgba::new(0, 150, 0, 255))
    .add_metric("Val Acc", &val_acc, Rgba::new(0, 100, 0, 255))
    .build();
```

## Complete Example

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::LossCurve;

fn main() -> Result<()> {
    let epochs: Vec<f32> = (1..=50).map(|e| e as f32).collect();

    // Simulated training progress
    let train_loss: Vec<f32> = epochs.iter()
        .map(|e| 2.0 * (-e * 0.08).exp() + 0.1)
        .collect();
    let val_loss: Vec<f32> = epochs.iter()
        .map(|e| 2.2 * (-e * 0.06).exp() + 0.15)
        .collect();

    let loss = LossCurve::new()
        .epochs(&epochs)
        .train_loss(&train_loss)
        .val_loss(&val_loss)
        .train_color(Rgba::new(66, 133, 244, 255))
        .val_color(Rgba::new(234, 67, 53, 255))
        .title("Model Training Progress")
        .xlabel("Epoch")
        .ylabel("Loss")
        .show_legend(true)
        .build();

    loss.render_to_file("training_loss.png")?;

    Ok(())
}
```

## Next Chapter

Continue to [Confusion Matrices](./confusion-matrix.md) for classification evaluation.
