//! Quick Start Example
//!
//! Demonstrates basic Pacha registry operations:
//! - Registering models
//! - Registering datasets
//! - Querying artifacts
//!
//! Run with: cargo run --example quick_start

use pacha::prelude::*;
use tempfile::TempDir;

fn main() -> Result<()> {
    println!("=== Pacha Quick Start ===\n");

    // Create a temporary registry for this example
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = RegistryConfig::new(temp_dir.path());
    let registry = Registry::open(config)?;

    // 1. Register a model
    println!("1. Registering a model...");
    let model_data = b"pretrained model weights binary data";
    let card = ModelCard::builder()
        .description("Fraud detection model trained on transaction data")
        .metrics([("auc", 0.95), ("f1", 0.88), ("precision", 0.92)])
        .primary_uses(["Fraud detection in payment transactions"])
        .limitations(["May have reduced accuracy on international transactions"])
        .build();

    let model_id =
        registry.register_model("fraud-detector", &ModelVersion::new(1, 0, 0), model_data, card)?;
    println!("   Registered model ID: {model_id}");

    // 2. Register a dataset
    println!("\n2. Registering a dataset...");
    let dataset_data = b"transaction_id,amount,is_fraud\n1,100.00,0\n2,5000.00,1";
    let datasheet = Datasheet::builder()
        .purpose("Transaction data for fraud detection training")
        .creators(["Data Engineering Team"])
        .instance_count(1_000_000)
        .license("Internal Use Only")
        .build();

    let dataset_id = registry.register_dataset(
        "transactions",
        &DatasetVersion::new(1, 0, 0),
        dataset_data,
        datasheet,
    )?;
    println!("   Registered dataset ID: {dataset_id}");

    // 3. Query the model
    println!("\n3. Querying the model...");
    let model = registry.get_model("fraud-detector", &ModelVersion::new(1, 0, 0))?;
    println!("   Model: {}:{}", model.name, model.version);
    println!("   Stage: {}", model.stage);
    println!("   Description: {}", model.card.description);
    println!("   Metrics:");
    for (name, value) in &model.card.metrics {
        println!("     - {name}: {value}");
    }

    // 4. Transition model stage
    println!("\n4. Promoting model to staging...");
    registry.transition_model_stage(
        "fraud-detector",
        &ModelVersion::new(1, 0, 0),
        ModelStage::Staging,
    )?;
    let model = registry.get_model("fraud-detector", &ModelVersion::new(1, 0, 0))?;
    println!("   New stage: {}", model.stage);

    // 5. Get storage statistics
    println!("\n5. Registry statistics:");
    let stats = registry.storage_stats()?;
    println!("   Models: {}", stats.model_count);
    println!("   Datasets: {}", stats.dataset_count);
    println!("   Objects: {}", stats.object_count);
    println!("   Total size: {} bytes", stats.total_size_bytes);

    println!("\n✅ Quick start complete!");
    Ok(())
}
