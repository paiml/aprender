//! APR Model Inspection Example
//!
//! Demonstrates the inspection tooling for .apr model files:
//! - Header inspection with magic, version, flags
//! - Metadata extraction with hyperparameters and provenance
//! - Weight statistics with health assessment
//! - Model diff for version comparison
//!
//! Toyota Way Alignment:
//! - **Genchi Genbutsu**: Go and see - inspect actual model data
//! - **Visualization**: Make problems visible for debugging
//!
//! Run with: `cargo run --example apr_inspection`

use aprender::inspect::{
    DiffItem, DiffResult, HeaderFlags, HeaderInspection, InspectOptions, InspectionError,
    InspectionResult, InspectionWarning, LicenseInfo, MetadataInspection, TrainingInfo, WeightDiff,
    WeightStats,
};
use std::time::Duration;

fn main() {
    println!("=== APR Model Inspection Demo ===\n");

    // Part 1: Header Inspection
    header_inspection_demo();

    // Part 2: Metadata Inspection
    metadata_inspection_demo();

    // Part 3: Weight Statistics
    weight_stats_demo();

    // Part 4: Model Diff
    model_diff_demo();

    // Part 5: Full Inspection
    full_inspection_demo();

    println!("\n=== Inspection Demo Complete! ===");
}

fn header_inspection_demo() {
    println!("--- Part 1: Header Inspection ---\n");

    let mut header = HeaderInspection::new();
    header.version = (1, 2);
    header.model_type = 3; // RandomForest
    header.compressed_size = 5 * 1024 * 1024;
    header.uncompressed_size = 12 * 1024 * 1024;
    header.checksum = 0xDEADBEEF;
    header.flags = HeaderFlags {
        compressed: true,
        signed: true,
        encrypted: false,
        streaming: false,
        licensed: false,
        quantized: false,
    };

    println!("Header Information:");
    println!(
        "  Magic: {} (valid: {})",
        header.magic_string(),
        header.magic_valid
    );
    println!(
        "  Version: {} (supported: {})",
        header.version_string(),
        header.version_supported
    );
    println!("  Model Type ID: {}", header.model_type);
    println!(
        "  Compressed Size: {} MB",
        header.compressed_size / (1024 * 1024)
    );
    println!(
        "  Uncompressed Size: {} MB",
        header.uncompressed_size / (1024 * 1024)
    );
    println!("  Compression Ratio: {:.2}x", header.compression_ratio());
    println!("  Checksum: 0x{:08X}", header.checksum);

    println!("\nHeader Flags:");
    let flags = header.flags.flag_list();
    if flags.is_empty() {
        println!("  (none)");
    } else {
        for flag in flags {
            println!("  - {}", flag);
        }
    }

    println!("\nFlags Byte: 0x{:02X}", header.flags.to_byte());
    println!("Header Valid: {}", header.is_valid());
    println!();
}

fn metadata_inspection_demo() {
    println!("--- Part 2: Metadata Inspection ---\n");

    let mut meta = MetadataInspection::new("RandomForestClassifier");
    meta.n_parameters = 50_000;
    meta.n_features = 13;
    meta.n_outputs = 3;

    meta.hyperparameters
        .insert("n_estimators".to_string(), "100".to_string());
    meta.hyperparameters
        .insert("max_depth".to_string(), "10".to_string());
    meta.hyperparameters
        .insert("random_state".to_string(), "42".to_string());

    meta.training_info = Some(TrainingInfo {
        trained_at: Some("2024-12-08T10:30:00Z".to_string()),
        duration: Some(Duration::from_secs(120)),
        dataset_name: Some("iris_extended".to_string()),
        n_samples: Some(10000),
        final_loss: Some(0.0234),
        framework: Some("aprender".to_string()),
        framework_version: Some("0.15.0".to_string()),
    });

    meta.license_info = Some(LicenseInfo {
        license_type: "Apache-2.0".to_string(),
        licensee: Some("Acme Corp".to_string()),
        expires_at: None,
        restrictions: vec![],
    });

    println!("Model Metadata:");
    println!("  Model Type: {}", meta.model_type_name);
    println!("  Parameters: {}", meta.n_parameters);
    println!("  Features: {}", meta.n_features);
    println!("  Outputs: {}", meta.n_outputs);

    println!("\nHyperparameters:");
    for (key, value) in &meta.hyperparameters {
        println!("  {}: {}", key, value);
    }

    if let Some(training) = &meta.training_info {
        println!("\nTraining Info:");
        if let Some(trained_at) = &training.trained_at {
            println!("  Trained At: {}", trained_at);
        }
        if let Some(duration) = &training.duration {
            println!("  Duration: {:?}", duration);
        }
        if let Some(dataset) = &training.dataset_name {
            println!("  Dataset: {}", dataset);
        }
        if let Some(n_samples) = training.n_samples {
            println!("  Samples: {}", n_samples);
        }
        if let Some(loss) = training.final_loss {
            println!("  Final Loss: {:.4}", loss);
        }
    }

    if let Some(license) = &meta.license_info {
        println!("\nLicense Info:");
        println!("  Type: {}", license.license_type);
        if let Some(licensee) = &license.licensee {
            println!("  Licensee: {}", licensee);
        }
        println!("  Has Restrictions: {}", license.has_restrictions());
    }
    println!();
}

fn weight_stats_demo() {
    println!("--- Part 3: Weight Statistics ---\n");

    // Normal weights
    let weights: Vec<f32> = (0..1000).map(|i| (i as f32 / 1000.0 - 0.5) * 2.0).collect();
    let stats = WeightStats::from_slice(&weights);

    println!("Weight Statistics (normal distribution):");
    println!("  Count: {}", stats.count);
    println!("  Min: {:.4}", stats.min);
    println!("  Max: {:.4}", stats.max);
    println!("  Mean: {:.4}", stats.mean);
    println!("  Std: {:.4}", stats.std);
    println!("  Zero Count: {}", stats.zero_count);
    println!("  Sparsity: {:.2}%", stats.sparsity * 100.0);
    println!("  L1 Norm: {:.4}", stats.l1_norm);
    println!("  L2 Norm: {:.4}", stats.l2_norm);
    println!(
        "  Health: {:?} - {}",
        stats.health_status(),
        stats.health_status().description()
    );

    // Weights with issues
    let bad_weights: Vec<f32> = vec![1.0, f32::NAN, 3.0, f32::INFINITY, 0.0, 0.0, 0.0];
    let bad_stats = WeightStats::from_slice(&bad_weights);

    println!("\nWeight Statistics (with issues):");
    println!("  Count: {}", bad_stats.count);
    println!("  NaN Count: {} (CRITICAL)", bad_stats.nan_count);
    println!("  Inf Count: {} (CRITICAL)", bad_stats.inf_count);
    println!("  Has Issues: {}", bad_stats.has_issues());
    println!(
        "  Health: {:?} - {}",
        bad_stats.health_status(),
        bad_stats.health_status().description()
    );

    // Sparse weights
    let sparse_weights: Vec<f32> = (0..100)
        .map(|i| if i % 50 == 0 { 1.0 } else { 0.0 })
        .collect();
    let sparse_stats = WeightStats::from_slice(&sparse_weights);

    println!("\nWeight Statistics (sparse):");
    println!("  Count: {}", sparse_stats.count);
    println!("  Zero Count: {}", sparse_stats.zero_count);
    println!("  Sparsity: {:.0}%", sparse_stats.sparsity * 100.0);
    println!("  Health: {:?}", sparse_stats.health_status());
    println!();
}

fn model_diff_demo() {
    println!("--- Part 4: Model Diff ---\n");

    // Create diff result
    let mut diff = DiffResult::new("model_v1.apr", "model_v2.apr");

    // Add header differences
    diff.header_diff
        .push(DiffItem::new("version", "1.0", "1.1"));
    diff.header_diff
        .push(DiffItem::new("flags", "COMPRESSED", "COMPRESSED|SIGNED"));

    // Add metadata differences
    diff.metadata_diff
        .push(DiffItem::new("n_estimators", "100", "150"));
    diff.metadata_diff
        .push(DiffItem::new("max_depth", "10", "12"));

    // Add weight differences
    let weights_a: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
    let weights_b: Vec<f32> = (0..100).map(|i| (i as f32 / 100.0) + 0.01).collect();
    diff.weight_diff = Some(WeightDiff::from_slices(&weights_a, &weights_b));
    diff.similarity = 0.95;

    println!("Model Diff: {} vs {}", diff.model_a, diff.model_b);
    println!("  Identical: {}", diff.is_identical());
    println!("  Similarity: {:.1}%", diff.similarity * 100.0);
    println!("  Total Differences: {}", diff.diff_count());

    println!("\nHeader Differences:");
    for item in &diff.header_diff {
        println!("  {}", item);
    }

    println!("\nMetadata Differences:");
    for item in &diff.metadata_diff {
        println!("  {}", item);
    }

    if let Some(weight_diff) = &diff.weight_diff {
        println!("\nWeight Differences:");
        println!("  Changed Count: {}", weight_diff.changed_count);
        println!("  Max Diff: {:.6}", weight_diff.max_diff);
        println!("  Mean Diff: {:.6}", weight_diff.mean_diff);
        println!("  L2 Distance: {:.6}", weight_diff.l2_distance);
        println!("  Cosine Similarity: {:.4}", weight_diff.cosine_similarity);
    }
    println!();
}

fn full_inspection_demo() {
    println!("--- Part 5: Full Inspection Result ---\n");

    // Create header and metadata
    let header = HeaderInspection::new();
    let meta = MetadataInspection::new("LinearRegression");

    // Create inspection result
    let mut result = InspectionResult::new(header, meta);
    result.quality_score = Some(85);
    result.duration = Duration::from_millis(42);

    // Add some warnings
    result.warnings.push(
        InspectionWarning::new("W001", "No signature found")
            .with_recommendation("Sign model for production use"),
    );
    result.warnings.push(
        InspectionWarning::new("W002", "High sparsity detected (95%)")
            .with_recommendation("Consider quantization or pruning"),
    );

    // Add an error
    result.errors.push(InspectionError::new(
        "E001",
        "Missing training provenance",
        false,
    ));

    println!("Inspection Result:");
    println!("  Valid: {}", result.is_valid());
    println!("  Has Issues: {}", result.has_issues());
    println!("  Issue Count: {}", result.issue_count());
    println!("  Quality Score: {:?}", result.quality_score);
    println!("  Duration: {:?}", result.duration);

    println!("\nWarnings ({}):", result.warnings.len());
    for warning in &result.warnings {
        println!("  {}", warning);
    }

    println!("\nErrors ({}):", result.errors.len());
    for error in &result.errors {
        println!("  {}", error);
    }

    // Inspection options
    println!("\nInspection Options:");

    let quick = InspectOptions::quick();
    println!(
        "  Quick: weights={}, quality={}",
        quick.include_weights, quick.include_quality
    );

    let full = InspectOptions::full();
    println!(
        "  Full: weights={}, quality={}, verbose={}",
        full.include_weights, full.include_quality, full.verbose
    );

    let default = InspectOptions::default();
    println!(
        "  Default: weights={}, quality={}, max_weights={}",
        default.include_weights, default.include_quality, default.max_weights
    );
    println!();
}
