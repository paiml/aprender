//! Model Versioning Example
//!
//! Demonstrates semantic versioning for ML models:
//! - MAJOR: Architecture changes (incompatible inputs/outputs)
//! - MINOR: Retraining with new data (backward compatible)
//! - PATCH: Bug fixes, quantization, optimization
//!
//! Run with: cargo run --example model_versioning

use pacha::prelude::*;
use tempfile::TempDir;

fn main() -> Result<()> {
    println!("=== Model Versioning Example ===\n");

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config = RegistryConfig::new(temp_dir.path());
    let registry = Registry::open(config)?;

    // Version 1.0.0 - Initial release
    println!("1. Registering v1.0.0 (initial release)...");
    let v1_card = ModelCard::builder()
        .description("Initial fraud detector - logistic regression")
        .metrics([("auc", 0.85)])
        .build();

    registry.register_model(
        "fraud-detector",
        &ModelVersion::new(1, 0, 0),
        b"v1.0.0 model weights",
        v1_card,
    )?;
    println!("   ✓ v1.0.0 registered");

    // Version 1.0.1 - Patch: Quantized to INT8
    println!("\n2. Registering v1.0.1 (patch: INT8 quantization)...");
    let v1_0_1_card = ModelCard::builder()
        .description("INT8 quantized version for edge deployment")
        .metrics([("auc", 0.84), ("inference_ms", 5.0)])
        .build();

    registry.register_model(
        "fraud-detector",
        &ModelVersion::new(1, 0, 1),
        b"v1.0.1 quantized weights",
        v1_0_1_card,
    )?;
    println!("   ✓ v1.0.1 registered (patch version)");

    // Version 1.1.0 - Minor: Retrained with new data
    println!("\n3. Registering v1.1.0 (minor: retrained with Q3 data)...");
    let v1_1_card = ModelCard::builder()
        .description("Retrained with Q3 2024 transaction data")
        .metrics([("auc", 0.88)])
        .build();

    registry.register_model(
        "fraud-detector",
        &ModelVersion::new(1, 1, 0),
        b"v1.1.0 retrained weights",
        v1_1_card,
    )?;
    println!("   ✓ v1.1.0 registered (minor version)");

    // Version 2.0.0 - Major: New architecture (transformer)
    println!("\n4. Registering v2.0.0 (major: transformer architecture)...");
    let v2_card = ModelCard::builder()
        .description("Transformer-based fraud detector - new input format")
        .metrics([("auc", 0.95), ("f1", 0.91)])
        .limitations(["Requires new feature preprocessing pipeline"])
        .build();

    registry.register_model(
        "fraud-detector",
        &ModelVersion::new(2, 0, 0),
        b"v2.0.0 transformer weights",
        v2_card,
    )?;
    println!("   ✓ v2.0.0 registered (major version)");

    // List all versions
    println!("\n5. All registered versions:");
    let versions = registry.list_model_versions("fraud-detector")?;
    for version in &versions {
        let model = registry.get_model("fraud-detector", version)?;
        let auc = model.card.metrics.get("auc").unwrap_or(&0.0);
        println!("   - v{version}: AUC={auc:.2} - {}", model.card.description);
    }

    // Demonstrate version comparison
    println!("\n6. Version comparison:");
    let v1 = ModelVersion::new(1, 0, 0);
    let v1_1 = ModelVersion::new(1, 1, 0);
    let v2 = ModelVersion::new(2, 0, 0);

    println!("   v1.0.0 < v1.1.0: {}", v1 < v1_1);
    println!("   v1.1.0 < v2.0.0: {}", v1_1 < v2);
    println!("   v2.0.0.is_stable(): {}", v2.is_stable());

    // Pre-release versions
    println!("\n7. Pre-release versions:");
    let beta = ModelVersion::new(2, 1, 0).with_prerelease("beta.1");
    println!("   v2.1.0-beta.1: {beta}");
    println!("   is_prerelease: {}", beta.is_prerelease());
    println!("   is_stable: {}", beta.is_stable());

    println!("\n✅ Model versioning example complete!");
    Ok(())
}
