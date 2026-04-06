//! Experiment Tracking Example
//!
//! Demonstrates tracking training runs:
//! - Creating experiment runs
//! - Logging metrics during training
//! - Finding the best run by metric
//!
//! Run with: cargo run --example experiment_tracking

use pacha::prelude::*;
use tempfile::TempDir;

fn main() -> Result<()> {
    println!("=== Experiment Tracking Example ===\n");

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = RegistryConfig::new(temp_dir.path());
    let registry = Registry::open(config)?;

    // Register a recipe first
    println!("1. Registering training recipe...");
    let recipe = TrainingRecipe::builder()
        .name("fraud-training")
        .version(RecipeVersion::new(1, 0, 0))
        .description("Fraud detection model training recipe")
        .hyperparameters(
            Hyperparameters::builder().learning_rate(1e-3).batch_size(32).epochs(10).build(),
        )
        .build();

    registry.register_recipe(&recipe)?;
    println!("   ✓ Recipe registered: {}:{}", recipe.name, recipe.version);

    // Simulate multiple training runs with different hyperparameters
    println!("\n2. Running experiments with different learning rates...\n");

    let learning_rates = [1e-2, 1e-3, 1e-4];
    let mut run_ids = Vec::new();

    for (i, &lr) in learning_rates.iter().enumerate() {
        println!("   Run {} (lr={lr}):", i + 1);

        // Create hyperparameters for this run
        let hyperparams =
            Hyperparameters::builder().learning_rate(lr).batch_size(32).epochs(10).build();

        // Create and start the run
        let mut run = ExperimentRun::from_recipe(recipe.reference(), hyperparams);
        run.start();

        // Simulate training with metrics
        // In real usage, these would be logged during actual training
        let final_loss = simulate_training(lr);
        let final_auc = 0.80 + (1.0 - final_loss) * 0.15;

        // Log metrics at different steps
        for step in (0..=100).step_by(10) {
            let loss = final_loss + (1.0 - final_loss) * (1.0 - step as f64 / 100.0);
            let auc = final_auc * (step as f64 / 100.0);
            run.log_metric("loss", loss, step as u64);
            run.log_metric("auc", auc, step as u64);
        }

        // Complete the run
        run.complete();

        println!("     Final loss: {:.4}", run.get_metric("loss").unwrap_or(0.0));
        println!("     Final AUC:  {:.4}", run.get_metric("auc").unwrap_or(0.0));
        println!("     Duration:   {} seconds", run.duration_secs().unwrap_or(0));

        // Save the run
        let run_id = registry.start_run(run)?;
        run_ids.push(run_id);
    }

    // Find the best run
    println!("\n3. Finding best run by AUC...");
    let runs = registry.list_runs(&recipe.reference())?;

    let best_run = runs.iter().filter(|r| r.status == RunStatus::Completed).max_by(|a, b| {
        let auc_a = a.get_metric("auc").unwrap_or(0.0);
        let auc_b = b.get_metric("auc").unwrap_or(0.0);
        auc_a.partial_cmp(&auc_b).unwrap()
    });

    if let Some(run) = best_run {
        println!("   Best run: {}", run.run_id);
        println!("   AUC: {:.4}", run.get_metric("auc").unwrap_or(0.0));
        println!("   Learning rate: {}", run.hyperparameters.learning_rate);
    }

    // Compare runs
    println!("\n4. Run comparison:");
    println!("   {:<12} {:<10} {:<10} {:<10}", "Run", "LR", "Loss", "AUC");
    println!("   {}", "-".repeat(42));

    for run in &runs {
        println!(
            "   {:<12} {:<10.0e} {:<10.4} {:<10.4}",
            &run.run_id.to_string()[..8],
            run.hyperparameters.learning_rate,
            run.get_metric("loss").unwrap_or(0.0),
            run.get_metric("auc").unwrap_or(0.0),
        );
    }

    println!("\n✅ Experiment tracking example complete!");
    Ok(())
}

/// Simulate training loss based on learning rate
fn simulate_training(lr: f64) -> f64 {
    // Simulate that lr=1e-3 is optimal
    let optimal_lr = 1e-3_f64;
    let distance = (lr.log10() - optimal_lr.log10()).abs();
    0.1 + 0.3 * distance // Lower is better, optimal around 0.1
}
