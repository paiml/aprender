//! Tests for calibration module.

use super::*;

// =========================================================================
// CalibrationConfig Tests
// =========================================================================

#[test]
fn test_config_default_values() {
    // TEST_ID: CAL-001
    let config = CalibrationConfig::default();
    assert_eq!(config.num_samples(), 128, "CAL-001 FALSIFIED: Default num_samples should be 128");
    assert_eq!(
        config.sequence_length(),
        2048,
        "CAL-001 FALSIFIED: Default sequence_length should be 2048"
    );
    assert_eq!(config.dataset(), "c4", "CAL-001 FALSIFIED: Default dataset should be c4");
    assert_eq!(config.batch_size(), 1, "CAL-001 FALSIFIED: Default batch_size should be 1");
    assert!(config.normalize(), "CAL-001 FALSIFIED: Default normalize should be true");
}

#[test]
fn test_config_builder_pattern() {
    // TEST_ID: CAL-002
    let config = CalibrationConfig::new()
        .with_num_samples(256)
        .with_sequence_length(1024)
        .with_dataset("wikitext")
        .with_batch_size(4)
        .with_normalize(false);

    assert_eq!(config.num_samples(), 256);
    assert_eq!(config.sequence_length(), 1024);
    assert_eq!(config.dataset(), "wikitext");
    assert_eq!(config.batch_size(), 4);
    assert!(!config.normalize());
}

#[test]
fn test_config_serialize_json() {
    // TEST_ID: CAL-003
    let config = CalibrationConfig::new().with_num_samples(64);
    let json = serde_json::to_string(&config).expect("JSON serialization should succeed");
    let deserialized: CalibrationConfig =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");
    assert_eq!(
        config.num_samples(),
        deserialized.num_samples(),
        "CAL-003 FALSIFIED: Serialization roundtrip failed"
    );
}

// =========================================================================
// LayerActivationStats Tests
// =========================================================================

#[test]
fn test_layer_stats_new() {
    // TEST_ID: CAL-010
    let stats = LayerActivationStats::new(512);
    assert_eq!(stats.input_dim(), 512, "CAL-010 FALSIFIED: input_dim should be 512");
    assert_eq!(stats.count(), 0, "CAL-010 FALSIFIED: initial count should be 0");
    assert!(stats.is_empty(), "CAL-010 FALSIFIED: should be empty initially");
}

#[test]
fn test_layer_stats_update_single_sample() {
    // TEST_ID: CAL-011
    let mut stats = LayerActivationStats::new(4);
    let sample = vec![1.0, 2.0, 3.0, 4.0];
    stats.update(&[sample]);

    assert_eq!(stats.count(), 1, "CAL-011 FALSIFIED: count should be 1 after one sample");
    assert!(!stats.is_empty(), "CAL-011 FALSIFIED: should not be empty after update");
}

#[test]
fn test_layer_stats_input_norms_single_sample() {
    // TEST_ID: CAL-012
    // For a single sample [1, 2, 3, 4]:
    // L2 norm per channel = sqrt(x^2 / 1) = |x|
    let mut stats = LayerActivationStats::new(4);
    let sample = vec![1.0, 2.0, 3.0, 4.0];
    stats.update(&[sample]);

    let norms = stats.input_norms();
    assert!((norms[0] - 1.0).abs() < 1e-6, "CAL-012 FALSIFIED: norm[0] should be 1.0");
    assert!((norms[1] - 2.0).abs() < 1e-6, "CAL-012 FALSIFIED: norm[1] should be 2.0");
    assert!((norms[2] - 3.0).abs() < 1e-6, "CAL-012 FALSIFIED: norm[2] should be 3.0");
    assert!((norms[3] - 4.0).abs() < 1e-6, "CAL-012 FALSIFIED: norm[3] should be 4.0");
}

#[test]
fn test_layer_stats_input_norms_multiple_samples() {
    // TEST_ID: CAL-013
    // Two samples: [1, 2] and [3, 4]
    // squared_sum[0] = 1 + 9 = 10, mean = 5, sqrt = sqrt(5) ≈ 2.236
    // squared_sum[1] = 4 + 16 = 20, mean = 10, sqrt = sqrt(10) ≈ 3.162
    let mut stats = LayerActivationStats::new(2);
    stats.update(&[vec![1.0, 2.0], vec![3.0, 4.0]]);

    let norms = stats.input_norms();
    let expected_0 = (5.0_f32).sqrt();
    let expected_1 = (10.0_f32).sqrt();

    assert!(
        (norms[0] - expected_0).abs() < 1e-5,
        "CAL-013 FALSIFIED: norm[0] should be sqrt(5), got {}",
        norms[0]
    );
    assert!(
        (norms[1] - expected_1).abs() < 1e-5,
        "CAL-013 FALSIFIED: norm[1] should be sqrt(10), got {}",
        norms[1]
    );
}

#[test]
fn test_layer_stats_mean_abs() {
    // TEST_ID: CAL-014
    let mut stats = LayerActivationStats::new(2);
    stats.update(&[vec![1.0, -2.0], vec![3.0, -4.0]]);

    let mean_abs = stats.mean_abs();
    // mean_abs[0] = (1 + 3) / 2 = 2.0
    // mean_abs[1] = (2 + 4) / 2 = 3.0
    assert!((mean_abs[0] - 2.0).abs() < 1e-6, "CAL-014 FALSIFIED: mean_abs[0] should be 2.0");
    assert!((mean_abs[1] - 3.0).abs() < 1e-6, "CAL-014 FALSIFIED: mean_abs[1] should be 3.0");
}

#[test]
fn test_layer_stats_empty_batch() {
    // TEST_ID: CAL-015
    let mut stats = LayerActivationStats::new(4);
    stats.update(&[]);
    assert!(stats.is_empty(), "CAL-015 FALSIFIED: Empty batch should not update stats");
}

#[test]
fn test_layer_stats_reset() {
    // TEST_ID: CAL-016
    let mut stats = LayerActivationStats::new(4);
    stats.update(&[vec![1.0, 2.0, 3.0, 4.0]]);
    assert!(!stats.is_empty());

    stats.reset();
    assert!(stats.is_empty(), "CAL-016 FALSIFIED: Should be empty after reset");
    assert_eq!(stats.count(), 0, "CAL-016 FALSIFIED: Count should be 0 after reset");
}

#[test]
fn test_layer_stats_empty_returns_zeros() {
    // TEST_ID: CAL-017
    let stats = LayerActivationStats::new(3);
    let norms = stats.input_norms();
    assert_eq!(norms.len(), 3);
    assert!(norms.iter().all(|&x| x == 0.0));

    let mean_abs = stats.mean_abs();
    assert!(mean_abs.iter().all(|&x| x == 0.0));
}

// =========================================================================
// CalibrationCollector Tests
// =========================================================================

#[test]
fn test_collector_new() {
    // TEST_ID: CAL-020
    let config = CalibrationConfig::new().with_num_samples(64);
    let collector = CalibrationCollector::new(config);

    assert!(!collector.is_complete());
    assert_eq!(collector.samples_processed(), 0);
    assert!(collector.layer_names().is_empty());
}

#[test]
fn test_collector_register_layer() {
    // TEST_ID: CAL-021
    let config = CalibrationConfig::default();
    let mut collector = CalibrationCollector::new(config);

    collector.register_layer("layer.0.attn", 512);
    collector.register_layer("layer.0.mlp", 2048);

    assert_eq!(
        collector.layer_names().len(),
        2,
        "CAL-021 FALSIFIED: Should have 2 registered layers"
    );
    assert!(collector.get_layer_stats("layer.0.attn").is_some());
    assert!(collector.get_layer_stats("layer.0.mlp").is_some());
}

#[test]
fn test_collector_duplicate_registration() {
    // TEST_ID: CAL-022
    let mut collector = CalibrationCollector::new(CalibrationConfig::default());

    collector.register_layer("layer.0", 512);
    collector.register_layer("layer.0", 1024); // Different dim, should not override

    let stats = collector.get_layer_stats("layer.0").expect("operation should succeed");
    assert_eq!(
        stats.input_dim(),
        512,
        "CAL-022 FALSIFIED: Duplicate registration should not override"
    );
}

#[test]
fn test_collector_record_activations() {
    // TEST_ID: CAL-023
    let mut collector = CalibrationCollector::new(CalibrationConfig::default());
    collector.register_layer("layer.0", 4);

    let activations = vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];
    collector.record_activations("layer.0", &activations);

    let stats = collector.get_layer_stats("layer.0").expect("operation should succeed");
    assert_eq!(stats.count(), 2, "CAL-023 FALSIFIED: Should have recorded 2 samples");
}

#[test]
fn test_collector_unregistered_layer() {
    // TEST_ID: CAL-024
    let mut collector = CalibrationCollector::new(CalibrationConfig::default());

    // Recording to unregistered layer should be a no-op
    collector.record_activations("nonexistent", &[vec![1.0, 2.0]]);

    assert!(
        collector.get_layer_stats("nonexistent").is_none(),
        "CAL-024 FALSIFIED: Unregistered layer should not exist"
    );
}

#[test]
fn test_collector_batch_complete() {
    // TEST_ID: CAL-025
    let config = CalibrationConfig::new().with_num_samples(10);
    let mut collector = CalibrationCollector::new(config);

    collector.batch_complete(5);
    assert_eq!(collector.samples_processed(), 5);
    assert!(!collector.is_complete());

    collector.batch_complete(5);
    assert_eq!(collector.samples_processed(), 10);
    assert!(collector.is_complete());
}

#[test]
fn test_collector_progress() {
    // TEST_ID: CAL-026
    let config = CalibrationConfig::new().with_num_samples(100);
    let mut collector = CalibrationCollector::new(config);

    assert!(collector.progress().abs() < 1e-6, "CAL-026 FALSIFIED: Initial progress should be 0");

    collector.batch_complete(25);
    assert!(
        (collector.progress() - 0.25).abs() < 1e-6,
        "CAL-026 FALSIFIED: Progress should be 0.25"
    );

    collector.batch_complete(75);
    assert!((collector.progress() - 1.0).abs() < 1e-6, "CAL-026 FALSIFIED: Progress should be 1.0");
}

#[test]
fn test_collector_progress_exceeds_target() {
    // TEST_ID: CAL-027
    let config = CalibrationConfig::new().with_num_samples(10);
    let mut collector = CalibrationCollector::new(config);

    collector.batch_complete(20);
    assert!(
        (collector.progress() - 1.0).abs() < 1e-6,
        "CAL-027 FALSIFIED: Progress should be clamped to 1.0"
    );
}

#[test]
fn test_collector_stops_recording_when_complete() {
    // TEST_ID: CAL-028
    let config = CalibrationConfig::new().with_num_samples(2);
    let mut collector = CalibrationCollector::new(config);
    collector.register_layer("layer.0", 2);

    // Complete calibration
    collector.record_activations("layer.0", &[vec![1.0, 2.0], vec![3.0, 4.0]]);
    collector.batch_complete(2);
    assert!(collector.is_complete());

    // Try to record more (should be ignored)
    collector.record_activations("layer.0", &[vec![5.0, 6.0]]);

    let stats = collector.get_layer_stats("layer.0").expect("operation should succeed");
    assert_eq!(stats.count(), 2, "CAL-028 FALSIFIED: Recording should stop when complete");
}

#[test]
fn test_collector_reset() {
    // TEST_ID: CAL-029
    let config = CalibrationConfig::new().with_num_samples(10);
    let mut collector = CalibrationCollector::new(config);
    collector.register_layer("layer.0", 4);

    collector.record_activations("layer.0", &[vec![1.0, 2.0, 3.0, 4.0]]);
    collector.batch_complete(10);
    assert!(collector.is_complete());

    collector.reset();
    assert!(!collector.is_complete());
    assert_eq!(collector.samples_processed(), 0);
    let stats = collector.get_layer_stats("layer.0").expect("operation should succeed");
    assert!(stats.is_empty(), "CAL-029 FALSIFIED: Layer stats should be reset");
}

#[test]
fn test_collector_zero_samples_config() {
    // TEST_ID: CAL-030
    let config = CalibrationConfig::new().with_num_samples(0);
    let collector = CalibrationCollector::new(config);
    assert!(
        (collector.progress() - 1.0).abs() < 1e-6,
        "CAL-030 FALSIFIED: Progress with 0 samples should be 1.0"
    );
}

#[test]
fn test_collector_config_access() {
    // TEST_ID: CAL-031
    let config = CalibrationConfig::new().with_num_samples(256).with_dataset("custom");
    let collector = CalibrationCollector::new(config);

    assert_eq!(collector.config().num_samples(), 256);
    assert_eq!(collector.config().dataset(), "custom");
}

// =========================================================================
// Welford's Algorithm Numerical Stability Tests
// =========================================================================

#[test]
fn test_welford_large_values() {
    // TEST_ID: CAL-040
    // Test with large values that could cause overflow in naive implementations
    let mut stats = LayerActivationStats::new(2);

    // Large values
    stats.update(&[vec![1e6, 1e6]]);
    stats.update(&[vec![1e6 + 1.0, 1e6 + 1.0]]);

    let norms = stats.input_norms();
    // Should not overflow or produce NaN/Inf
    assert!(norms[0].is_finite(), "CAL-040 FALSIFIED: Large values should produce finite results");
}

#[test]
fn test_welford_small_values() {
    // TEST_ID: CAL-041
    let mut stats = LayerActivationStats::new(2);

    // Very small values
    stats.update(&[vec![1e-10, 1e-10]]);
    stats.update(&[vec![1e-10, 1e-10]]);

    let norms = stats.input_norms();
    assert!(norms[0].is_finite(), "CAL-041 FALSIFIED: Small values should produce finite results");
    assert!(
        norms[0] > 0.0,
        "CAL-041 FALSIFIED: Small positive values should produce positive norms"
    );
}

#[test]
fn test_welford_mixed_sign() {
    // TEST_ID: CAL-042
    let mut stats = LayerActivationStats::new(2);

    // Mixed positive and negative values
    stats.update(&[vec![1.0, -2.0], vec![-3.0, 4.0]]);

    let norms = stats.input_norms();
    // L2 norm should be positive regardless of sign
    assert!(norms[0] > 0.0, "CAL-042 FALSIFIED: L2 norm should be positive");
    assert!(norms[1] > 0.0, "CAL-042 FALSIFIED: L2 norm should be positive");
}
