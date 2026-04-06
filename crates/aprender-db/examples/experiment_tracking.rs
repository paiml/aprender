//! Experiment Tracking Example
//!
//! Demonstrates the experiment schema for ML experiment tracking,
//! designed to integrate with entrenar's `ExperimentStorage`.
//!
//! Run with: cargo run --example `experiment_tracking`

use trueno_db::experiment::{
    ArtifactRecord, ExperimentRecord, ExperimentStore, MetricRecord, RunRecord, RunStatus,
};

fn main() {
    println!("=== Trueno-DB Experiment Tracking ===\n");

    // Create the experiment store
    let mut store = ExperimentStore::new();

    // -------------------------------------------------------------------------
    // 1. Create an experiment with configuration
    // -------------------------------------------------------------------------
    println!("1. Creating experiment...");

    let config = serde_json::json!({
        "model": "resnet50",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10,
        "optimizer": "adam"
    });

    let experiment = ExperimentRecord::builder("exp-resnet-001", "ResNet50 ImageNet Training")
        .config(config)
        .build();

    println!("   Experiment ID: {}", experiment.experiment_id());
    println!("   Name: {}", experiment.name());
    println!("   Created: {}", experiment.created_at());
    println!("   Config: {}", serde_json::to_string_pretty(experiment.config().unwrap()).unwrap());

    store.add_experiment(experiment.clone());

    // -------------------------------------------------------------------------
    // 2. Start a training run
    // -------------------------------------------------------------------------
    println!("\n2. Starting training run...");

    let mut run = RunRecord::builder("run-001", experiment.experiment_id())
        .renacer_span_id("span-abc-123-def-456")
        .build();

    run.start();
    store.add_run(run.clone());

    println!("   Run ID: {}", run.run_id());
    println!("   Status: {:?}", run.status());
    println!("   Started: {:?}", run.started_at());
    println!("   Renacer Span: {:?}", run.renacer_span_id());

    // -------------------------------------------------------------------------
    // 3. Simulate training loop with metric logging
    // -------------------------------------------------------------------------
    println!("\n3. Simulating training (10 epochs)...");

    let epochs = 10;
    for epoch in 0..epochs {
        // Simulate decreasing loss
        let loss = 2.5 / (epoch as f64 + 1.0) + 0.1;
        let accuracy = 0.5 + 0.05 * epoch as f64;
        let learning_rate = 0.001 * (0.95_f64).powi(epoch as i32);

        // Log metrics
        store.add_metric(MetricRecord::new(run.run_id(), "loss", epoch, loss));
        store.add_metric(MetricRecord::new(run.run_id(), "accuracy", epoch, accuracy));
        store.add_metric(MetricRecord::new(run.run_id(), "learning_rate", epoch, learning_rate));

        println!("   Epoch {epoch}: loss={loss:.4}, accuracy={accuracy:.4}, lr={learning_rate:.6}");
    }

    // -------------------------------------------------------------------------
    // 4. Save model artifact
    // -------------------------------------------------------------------------
    println!("\n4. Saving model artifact...");

    let model_artifact = ArtifactRecord::new(
        run.run_id(),
        "model_final.pt",
        "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        104_857_600, // 100MB
    );

    println!("   Artifact: {}", model_artifact.key());
    println!("   CAS Hash: {}", model_artifact.cas_hash());
    println!("   Size: {} MB", model_artifact.size_bytes() / 1024 / 1024);

    // -------------------------------------------------------------------------
    // 5. Complete the run
    // -------------------------------------------------------------------------
    println!("\n5. Completing run...");

    run.complete(RunStatus::Success);
    println!("   Final Status: {:?}", run.status());
    println!("   Ended: {:?}", run.ended_at());

    // -------------------------------------------------------------------------
    // 6. Query metrics for visualization
    // -------------------------------------------------------------------------
    println!("\n6. Querying metrics...");

    let loss_metrics = store.get_metrics_for_run(run.run_id(), "loss");
    let accuracy_metrics = store.get_metrics_for_run(run.run_id(), "accuracy");

    println!("   Loss curve ({} points):", loss_metrics.len());
    for m in &loss_metrics {
        print!("     Step {}: {:.4}", m.step(), m.value());
        if m.step() < 9 {
            print!(" → ");
        }
    }
    println!();

    println!("\n   Accuracy curve ({} points):", accuracy_metrics.len());
    println!(
        "     Start: {:.4} → End: {:.4}",
        accuracy_metrics.first().map_or(0.0, trueno_db::experiment::MetricRecord::value),
        accuracy_metrics.last().map_or(0.0, trueno_db::experiment::MetricRecord::value)
    );

    // -------------------------------------------------------------------------
    // 7. Store statistics
    // -------------------------------------------------------------------------
    println!("\n7. Store statistics:");
    println!("   Experiments: {}", store.experiment_count());
    println!("   Runs: {}", store.run_count());
    println!("   Metrics: {}", store.metric_count());

    // -------------------------------------------------------------------------
    // 8. Serialization demonstration
    // -------------------------------------------------------------------------
    println!("\n8. JSON serialization:");

    let metric = &loss_metrics[0];
    let json = serde_json::to_string_pretty(metric).unwrap();
    println!("   MetricRecord JSON:\n{json}");

    println!("\n=== Experiment Tracking Complete ===");
}
