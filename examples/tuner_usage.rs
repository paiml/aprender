#![allow(clippy::disallowed_methods)]
//! ML Tuner Usage Examples
//!
//! This file contains examples that are included in the book documentation
//! using mdbook's `{{#include}}` directive.
//!
//! Run: `cargo run --example tuner_usage --features hardware-detect`

fn main() {
    println!("=== ML Tuner Usage Examples ===\n");

    // Run all examples
    basic_features();
    throughput_prediction();
    full_recommendation();
}

// ANCHOR: basic_features
/// Create a 42-dimension feature vector for ML tuning
fn basic_features() {
    use trueno::tuner::{QuantType, TunerFeatures};

    let features = TunerFeatures::builder()
        .model_params_b(1.5) // 1.5B parameters
        .hidden_dim(1536)
        .num_layers(28)
        .num_heads(12)
        .batch_size(4) // M=4 concurrent sequences
        .seq_len(512)
        .quant_type(QuantType::Q4K)
        .gpu_mem_bw_gbs(1000.0) // RTX 4090: ~1 TB/s
        .gpu_sm_count(128) // RTX 4090: 128 SMs
        .cuda_graphs(true)
        .build();

    // Validate and convert to vector
    assert!(features.validate().is_ok());
    let vec = features.to_vector();
    assert_eq!(vec.len(), 42);

    println!("Features created: {} dimensions", vec.len());
}
// ANCHOR_END: basic_features

// ANCHOR: throughput_prediction
/// Predict throughput with roofline model clamping
fn throughput_prediction() {
    use trueno::tuner::{QuantType, ThroughputRegressor, TunerFeatures};

    let regressor = ThroughputRegressor::new();

    // Create features for RTX 4090 with 1.5B Q4_K model
    let features = TunerFeatures::builder()
        .model_params_b(1.5)
        .batch_size(4)
        .quant_type(QuantType::Q4K)
        .gpu_mem_bw_gbs(1000.0)
        .cuda_graphs(true)
        .build();

    let prediction = regressor.predict(&features);

    println!("Predicted: {:.1} tok/s", prediction.predicted_tps);
    println!("Confidence: {:.1}%", prediction.confidence * 100.0);
    println!("Top features:");
    for (name, importance) in prediction.top_features.iter().take(3) {
        println!("  - {}: {:.1}%", name, importance * 100.0);
    }
}
// ANCHOR_END: throughput_prediction

// ANCHOR: full_recommendation
/// Get full tuning recommendation with kernel and bottleneck analysis
fn full_recommendation() {
    use trueno::tuner::{BrickTuner, QuantType, TunerFeatures};

    let tuner = BrickTuner::new();

    let features = TunerFeatures::builder()
        .model_params_b(7.0) // 7B model
        .hidden_dim(4096)
        .num_layers(32)
        .batch_size(1) // Single sequence
        .quant_type(QuantType::Q4K)
        .gpu_mem_bw_gbs(1000.0)
        .build();

    let rec = tuner.recommend(&features);

    println!("\n=== BrickTuner Recommendation ===");
    println!("Throughput: {:.1} tok/s", rec.throughput.predicted_tps);
    println!("Kernel: {:?}", rec.kernel.top_kernel);
    println!("Bottleneck: {}", rec.bottleneck.class);
    println!("Confidence: {:.0}%", rec.confidence_overall * 100.0);

    println!("\nSuggested experiments:");
    for (i, exp) in rec.suggested_experiments.iter().enumerate() {
        println!("  {}. {}", i + 1, exp);
    }
}
// ANCHOR_END: full_recommendation
