#![allow(clippy::expect_used, clippy::unwrap_used)]
//! ML Training Loss Curves Example
//!
//! Demonstrates visualizing training progress with loss curves,
//! including train/validation loss with smoothing.
//!
//! Run with: `cargo run --example loss_training`

use trueno_viz::output::PngEncoder;
use trueno_viz::plots::{LossCurve, MetricSeries};
use trueno_viz::prelude::*;

fn main() {
    println!("ML Training Loss Curves Example");
    println!("================================\n");

    // Step 1: Simulate training metrics
    println!("Step 1: Simulating training run (50 epochs)...");
    let (train_losses, val_losses) = simulate_training(50);

    println!("  Epochs: {}", train_losses.len());
    println!("  Initial train loss: {:.4}", train_losses[0]);
    println!(
        "  Final train loss: {:.4}",
        train_losses.last().expect("collection should not be empty")
    );

    // Step 2: Create loss curve visualization
    println!("\nStep 2: Creating loss curve visualization...");

    // Create series with different colors
    let train_series =
        MetricSeries::new("Train Loss", Rgba::BLUE).smoothing(0.6).raw(true).smooth(true);

    let val_series =
        MetricSeries::new("Val Loss", Rgba::rgb(255, 128, 0)).smoothing(0.6).raw(true).smooth(true);

    let mut loss_curve = LossCurve::new()
        .add_series(train_series)
        .add_series(val_series)
        .dimensions(800, 400)
        .margin(40)
        .best_markers(true)
        .lower_is_better(true)
        .build()
        .expect("Failed to build loss curve");

    // Step 3: Stream the data (simulating real-time training)
    println!("\nStep 3: Streaming epoch data...");
    for (epoch, (&train_loss, &val_loss)) in train_losses.iter().zip(val_losses.iter()).enumerate()
    {
        loss_curve.push_all(&[train_loss, val_loss]);

        // Print progress every 10 epochs
        if epoch % 10 == 0 || epoch == train_losses.len() - 1 {
            println!("  Epoch {epoch:>3}: train={train_loss:.4}, val={val_loss:.4}");
        }
    }

    // Step 4: Get summary statistics
    println!("\nStep 4: Computing statistics...");
    let summaries = loss_curve.summary();

    for summary in &summaries {
        println!(
            "  {}: min={:.4} (epoch {}), last={:.4}",
            summary.name,
            summary.min.unwrap_or(0.0),
            summary.best_epoch.unwrap_or(0),
            summary.last.unwrap_or(0.0)
        );
    }

    // Step 5: Render and save
    println!("\nStep 5: Rendering to PNG...");
    let fb = loss_curve.to_framebuffer().expect("Failed to render");

    let output_path = "loss_training.png";
    PngEncoder::write_to_file(&fb, output_path).expect("Failed to write PNG");

    println!("  Saved to: {output_path}");

    // Final summary
    println!("\n--- Training Summary ---");
    println!("Total epochs: {}", train_losses.len());
    println!(
        "Best train loss: {:.4} at epoch {}",
        summaries[0].min.unwrap_or(0.0),
        summaries[0].best_epoch.unwrap_or(0)
    );
    println!(
        "Best val loss: {:.4} at epoch {}",
        summaries[1].min.unwrap_or(0.0),
        summaries[1].best_epoch.unwrap_or(0)
    );

    // Check for overfitting
    let train_final = summaries[0].last.unwrap_or(0.0);
    let val_final = summaries[1].last.unwrap_or(0.0);
    if val_final > train_final * 1.5 {
        println!("\nWarning: Possible overfitting detected!");
        println!("  Train/Val gap: {:.2}%", (val_final / train_final - 1.0) * 100.0);
    }

    println!("\nLoss curves successfully generated!");
}

/// Simulate a typical neural network training run.
///
/// Returns (`train_losses`, `val_losses`) for each epoch.
fn simulate_training(epochs: usize) -> (Vec<f32>, Vec<f32>) {
    let mut train_losses = Vec::with_capacity(epochs);
    let mut val_losses = Vec::with_capacity(epochs);

    for epoch in 0..epochs {
        let t = epoch as f32 / epochs as f32;

        // Training loss: exponential decay with noise
        let base_train = 2.5 * (-3.0 * t).exp() + 0.1;
        let noise = ((epoch * 7919 + 104_729) % 1000) as f32 / 5000.0 - 0.1;
        let train_loss = (base_train + noise).max(0.05);

        // Validation loss: similar but with slight overfitting at the end
        let overfit_factor = if t > 0.7 { (t - 0.7) * 0.5 } else { 0.0 };
        let val_noise = ((epoch * 6971 + 7723) % 1000) as f32 / 4000.0 - 0.125;
        let val_loss = (base_train * 1.1 + val_noise + overfit_factor).max(0.08);

        train_losses.push(train_loss);
        val_losses.push(val_loss);
    }

    (train_losses, val_losses)
}
