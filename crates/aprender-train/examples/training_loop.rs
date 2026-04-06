//! Example: Training Loop
//!
//! Demonstrates the high-level Trainer abstraction for training models.

use entrenar::optim::Adam;
use entrenar::train::{Batch, MSELoss, TrainConfig, Trainer};
use entrenar::Tensor;

fn main() {
    println!("=== Training Loop Example ===\n");

    // Setup model parameters (simple linear model)
    let params = vec![
        Tensor::from_vec(vec![0.1, 0.2, 0.3], true), // Weights
    ];

    // Create optimizer
    let optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);

    // Configure training
    let config = TrainConfig::new().with_grad_clip(1.0).with_log_interval(5);

    // Create trainer
    let mut trainer = Trainer::new(params, Box::new(optimizer), config);
    trainer.set_loss(Box::new(MSELoss));

    println!("Initial learning rate: {:.6}", trainer.lr());
    println!("Gradient clipping: enabled (max_norm=1.0)\n");

    // Create training data
    let batches = vec![
        Batch::new(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], false),
            Tensor::from_vec(vec![2.0, 3.0, 4.0], false),
        ),
        Batch::new(
            Tensor::from_vec(vec![2.0, 3.0, 4.0], false),
            Tensor::from_vec(vec![3.0, 4.0, 5.0], false),
        ),
        Batch::new(
            Tensor::from_vec(vec![3.0, 4.0, 5.0], false),
            Tensor::from_vec(vec![4.0, 5.0, 6.0], false),
        ),
    ];

    println!("Training data:");
    println!("  Batches: {}", batches.len());
    println!("  Batch size: {}", batches[0].size());
    println!();

    // Training loop
    println!("Starting training...\n");

    for epoch in 0..10 {
        let avg_loss = trainer.train_epoch(batches.clone(), |x| {
            // Simple identity function for demo
            x.clone()
        });

        println!("Epoch {}: loss={:.4}, lr={:.6}", epoch + 1, avg_loss, trainer.lr());

        // Optional: Learning rate schedule
        if epoch == 5 {
            println!("  → Reducing learning rate");
            trainer.set_lr(trainer.lr() * 0.1);
        }
    }

    println!("\n=== Training Complete ===\n");

    // Show metrics
    println!("Training Metrics:");
    println!("  Total epochs: {}", trainer.metrics.epoch);
    println!("  Total steps: {}", trainer.metrics.steps);
    println!("  Best loss: {:.4}", trainer.metrics.best_loss().expect("operation should succeed"));
    println!("  Avg loss (last 3 epochs): {:.4}", trainer.metrics.avg_loss(3));

    // Check if training improved
    if trainer.metrics.is_improving(3) {
        println!("\n✓ Training is improving!");
    } else {
        println!("\n⚠ Training may have plateaued");
    }

    println!("\nFinal parameters:");
    for (i, param) in trainer.params().iter().enumerate() {
        println!("  param[{}]: {:?}", i, param.data());
    }
}
