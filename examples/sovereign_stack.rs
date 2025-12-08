//! Sovereign AI Stack Integration Example
//!
//! Demonstrates the Pragmatic AI Labs Sovereign AI Stack integration:
//! - Stack components (alimentar, aprender, pacha, realizar, presentar, batuta)
//! - Model versioning and lifecycle management
//! - Derivation tracking (fine-tuning, distillation, quantization)
//! - Inference configuration
//! - Health monitoring
//!
//! Stack Architecture:
//! ```text
//! alimentar → aprender → pacha → realizar
//!     ↓           ↓          ↓         ↓
//!              presentar (WASM viz)
//!                    ↓
//!              batuta (orchestration)
//! ```
//!
//! Run with: `cargo run --example sovereign_stack`

use aprender::stack::{
    ComponentHealth, DerivationType, FormatCompatibility, HealthStatus, InferenceConfig,
    ModelStage, ModelVersion, QuantizationType, StackComponent, StackHealth,
};

fn main() {
    println!("=== Sovereign AI Stack Demo ===\n");

    // Part 1: Stack Components
    stack_components_demo();

    // Part 2: Model Lifecycle
    model_lifecycle_demo();

    // Part 3: Model Derivation (Lineage)
    derivation_demo();

    // Part 4: Inference Configuration
    inference_config_demo();

    // Part 5: Stack Health Monitoring
    health_monitoring_demo();

    // Part 6: Format Compatibility
    format_compatibility_demo();

    println!("\n=== Sovereign Stack Demo Complete! ===");
}

fn stack_components_demo() {
    println!("--- Part 1: Stack Components ---\n");

    println!(
        "{:<15} {:<15} {:<40} {:<8}",
        "Component", "Spanish", "Description", "Format"
    );
    println!("{}", "-".repeat(85));

    for component in StackComponent::all() {
        println!(
            "{:<15} {:<15} {:<40} {:<8}",
            component.name(),
            format!("({})", component.english()),
            component.description(),
            component.format().unwrap_or("-")
        );
    }

    println!("\nMagic Bytes (for format detection):");
    for component in StackComponent::all() {
        if let Some(magic) = component.magic() {
            println!(
                "  {}: {:02X} {:02X} {:02X} {:02X} (\"{}\")",
                component.name(),
                magic[0],
                magic[1],
                magic[2],
                magic[3],
                String::from_utf8_lossy(&magic)
            );
        }
    }

    println!("\nDisplay Format:");
    println!("  {}", StackComponent::Aprender);
    println!("  {}", StackComponent::Realizar);
    println!();
}

fn model_lifecycle_demo() {
    println!("--- Part 2: Model Lifecycle ---\n");

    // Show stage transitions
    println!("Model Stages and Valid Transitions:");
    let stages = [
        ModelStage::Development,
        ModelStage::Staging,
        ModelStage::Production,
        ModelStage::Archived,
    ];

    for from in &stages {
        let valid_to: Vec<_> = stages
            .iter()
            .filter(|to| from.can_transition_to(**to) && from != *to)
            .map(|s| s.name())
            .collect();
        println!("  {} -> {:?}", from, valid_to);
    }

    // Create and manage model versions
    println!("\nModel Version Examples:");

    let v1 = ModelVersion::new("1.0.0", [0xAB; 32])
        .with_stage(ModelStage::Production)
        .with_size(5_000_000)
        .with_quality_score(92.5)
        .with_tag("classification")
        .with_tag("iris");

    println!("\nVersion 1.0.0 (Production):");
    println!("  Version: {}", v1.version);
    println!("  Stage: {}", v1.stage);
    println!("  Size: {} bytes", v1.size_bytes);
    println!("  Quality: {:?}", v1.quality_score);
    println!("  Tags: {:?}", v1.tags);
    println!("  Hash: {}...", &v1.hash_hex()[..16]);
    println!("  Production Ready: {}", v1.is_production_ready());

    // Version not production ready
    let v2_dev = ModelVersion::new("2.0.0-beta", [0xCD; 32])
        .with_stage(ModelStage::Development)
        .with_quality_score(78.0);

    println!("\nVersion 2.0.0-beta (Development):");
    println!("  Stage: {}", v2_dev.stage);
    println!("  Quality: {:?}", v2_dev.quality_score);
    println!(
        "  Production Ready: {} (stage=dev)",
        v2_dev.is_production_ready()
    );

    let v2_low_quality = ModelVersion::new("2.0.0", [0xEF; 32])
        .with_stage(ModelStage::Production)
        .with_quality_score(70.0);

    println!("\nVersion 2.0.0 (Production, Low Quality):");
    println!("  Quality: {:?}", v2_low_quality.quality_score);
    println!(
        "  Production Ready: {} (quality < 85)",
        v2_low_quality.is_production_ready()
    );
    println!();
}

fn derivation_demo() {
    println!("--- Part 3: Model Derivation (Lineage) ---\n");

    let parent_hash = [0x11; 32];
    let teacher_hash = [0x22; 32];

    let derivations = [
        ("Original Training", DerivationType::Original),
        (
            "Fine-Tuned",
            DerivationType::FineTune {
                parent_hash,
                epochs: 10,
            },
        ),
        (
            "Distilled",
            DerivationType::Distillation {
                teacher_hash,
                temperature: 3.0,
            },
        ),
        (
            "Merged (TIES)",
            DerivationType::Merge {
                parent_hashes: vec![[0x33; 32], [0x44; 32]],
                method: "TIES".into(),
            },
        ),
        (
            "Quantized (INT8)",
            DerivationType::Quantize {
                parent_hash,
                quant_type: QuantizationType::Int8,
            },
        ),
        (
            "Pruned (50%)",
            DerivationType::Prune {
                parent_hash,
                sparsity: 0.5,
            },
        ),
    ];

    println!(
        "{:<20} {:<15} {:<10} {:>12}",
        "Description", "Type", "Derived?", "Parents"
    );
    println!("{}", "-".repeat(60));

    for (desc, deriv) in &derivations {
        println!(
            "{:<20} {:<15} {:<10} {:>12}",
            desc,
            deriv.type_name(),
            deriv.is_derived(),
            deriv.parent_hashes().len()
        );
    }

    // Quantization types
    println!("\nQuantization Types:");
    let quant_types = [
        QuantizationType::Int8,
        QuantizationType::Int4,
        QuantizationType::Float16,
        QuantizationType::BFloat16,
        QuantizationType::Dynamic,
        QuantizationType::QAT,
    ];

    println!("{:<12} {:>8} {:>20}", "Type", "Bits", "Name");
    println!("{}", "-".repeat(45));

    for qt in &quant_types {
        println!(
            "{:<12} {:>8} {:>20}",
            format!("{:?}", qt),
            qt.bits(),
            qt.name()
        );
    }

    // Model version with derivation
    println!("\nModel with Derivation Info:");
    let derived = ModelVersion::new("1.1.0", [0x55; 32])
        .with_stage(ModelStage::Staging)
        .with_derivation(DerivationType::FineTune {
            parent_hash,
            epochs: 5,
        })
        .with_quality_score(88.0)
        .with_tag("fine-tuned");

    println!("  Version: {}", derived.version);
    println!("  Derivation: {}", derived.derivation.type_name());
    println!(
        "  Parent Count: {}",
        derived.derivation.parent_hashes().len()
    );
    println!();
}

fn inference_config_demo() {
    println!("--- Part 4: Inference Configuration ---\n");

    // Default configuration
    let default = InferenceConfig::default();
    println!("Default Configuration:");
    println!("  Model: {:?}", default.model_path);
    println!("  Port: {}", default.port);
    println!("  Max Batch: {}", default.max_batch_size);
    println!("  Timeout: {} ms", default.timeout_ms);
    println!("  CORS: {}", default.enable_cors);
    println!("  Metrics: {:?}", default.metrics_path);
    println!("  Health: {:?}", default.health_path);

    // Custom configuration
    let custom = InferenceConfig::new("/models/iris_rf.apr")
        .with_port(9000)
        .with_batch_size(64)
        .with_timeout_ms(50)
        .without_cors();

    println!("\nCustom Configuration:");
    println!("  Model: {:?}", custom.model_path);
    println!("  Port: {}", custom.port);
    println!("  Max Batch: {}", custom.max_batch_size);
    println!("  Timeout: {} ms", custom.timeout_ms);
    println!("  CORS: {}", custom.enable_cors);

    // Endpoint URLs
    println!("\nInference Endpoints:");
    println!("  Predict: {}", custom.predict_url());
    println!("  Batch:   {}", custom.batch_predict_url());
    println!();
}

fn health_monitoring_demo() {
    println!("--- Part 5: Stack Health Monitoring ---\n");

    // Create stack health monitor
    let mut health = StackHealth::new();

    // Simulate component health checks
    health.set_component(
        StackComponent::Aprender,
        ComponentHealth::healthy("0.15.0").with_response_time(5),
    );

    health.set_component(
        StackComponent::Pacha,
        ComponentHealth::healthy("1.0.0").with_response_time(12),
    );

    health.set_component(
        StackComponent::Realizar,
        ComponentHealth::degraded("0.8.0", "high latency").with_response_time(250),
    );

    health.set_component(
        StackComponent::Presentar,
        ComponentHealth::unhealthy("connection refused"),
    );

    // Display health status
    println!("Stack Health Status:");
    println!(
        "  Overall: {} (operational: {})",
        health.overall,
        health.overall.is_operational()
    );

    println!("\nComponent Status:");
    println!(
        "{:<15} {:<12} {:<10} {:>12} {}",
        "Component", "Status", "Version", "Latency", "Error"
    );
    println!("{}", "-".repeat(70));

    for component in StackComponent::all() {
        if let Some(ch) = health.components.get(component) {
            println!(
                "{:<15} {:<12} {:<10} {:>10} {}",
                component.name(),
                ch.status.name(),
                ch.version.as_deref().unwrap_or("-"),
                ch.response_time_ms
                    .map(|ms| format!("{} ms", ms))
                    .unwrap_or_else(|| "-".into()),
                ch.error.as_deref().unwrap_or("")
            );
        }
    }

    // Health status types
    println!("\nHealth Status Types:");
    let statuses = [
        HealthStatus::Healthy,
        HealthStatus::Degraded,
        HealthStatus::Unhealthy,
        HealthStatus::Unknown,
    ];

    for status in &statuses {
        println!(
            "  {:<12}: operational={}",
            status.name(),
            status.is_operational()
        );
    }
    println!();
}

fn format_compatibility_demo() {
    println!("--- Part 6: Format Compatibility ---\n");

    let compat = FormatCompatibility::current();

    println!("Current Format Versions:");
    println!("  APR: {}.{}", compat.apr_version.0, compat.apr_version.1);
    println!("  ALD: {}.{}", compat.ald_version.0, compat.ald_version.1);
    println!("  Compatible: {}", compat.compatible);

    // Version compatibility checks
    println!("\nAPR Version Compatibility:");
    let apr_versions = [(1, 0), (1, 1), (2, 0)];
    for (major, minor) in apr_versions {
        println!(
            "  APR {}.{}: {}",
            major,
            minor,
            if compat.is_apr_compatible(major, minor) {
                "compatible"
            } else {
                "NOT compatible"
            }
        );
    }

    println!("\nALD Version Compatibility:");
    let ald_versions = [(1, 0), (1, 2), (1, 3), (2, 0)];
    for (major, minor) in ald_versions {
        println!(
            "  ALD {}.{}: {}",
            major,
            minor,
            if compat.is_ald_compatible(major, minor) {
                "compatible"
            } else {
                "NOT compatible"
            }
        );
    }

    // Stack architecture summary
    println!("\n=== Stack Architecture Summary ===\n");
    println!("Data Flow:");
    println!("  alimentar (.ald) - Load and transform data");
    println!("       ↓");
    println!("  aprender (.apr) - Train ML models");
    println!("       ↓");
    println!("  pacha - Registry with versioning and lineage");
    println!("       ↓");
    println!("  realizar - Pure Rust inference engine");
    println!("       ↓");
    println!("  presentar - WASM visualization");
    println!("       ↓");
    println!("  batuta (.bat) - Orchestration and oracle mode");

    println!("\nDesign Principles:");
    println!("  - Pure Rust: Zero cloud dependencies");
    println!("  - Format Independence: Each tool has its own binary format");
    println!("  - Toyota Way: Jidoka, Muda elimination, Kaizen");
    println!();
}
