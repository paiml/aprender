//! Tests for benchmarks module

use super::super::granularity::{
    calibrate_per_channel, calibrate_per_tensor, dequantize_with_params, quantization_mse,
    quantize_with_params, QuantGranularity, QuantMode,
};
use super::generators::{
    generate_gaussian_weights, generate_multi_channel_weights, generate_uniform_weights,
    generate_weights_with_outliers,
};
use super::runners::{
    accuracy_retention, compare_bit_width_degradation, run_benchmark, run_full_benchmark_suite,
};
use super::types::{BenchmarkSuite, QuantBenchmarkResult};
use proptest::prelude::*;

#[test]
fn test_run_benchmark() {
    let values = generate_gaussian_weights(1000, 0.0, 1.0, 42);
    let result =
        run_benchmark("test", &values, 8, QuantGranularity::PerTensor, QuantMode::Symmetric);

    assert_eq!(result.name, "test");
    assert_eq!(result.num_elements, 1000);
    assert_eq!(result.bits, 8);
    assert!(result.mse >= 0.0);
    assert!(result.sqnr_db > 0.0);
    assert!(result.compression_ratio > 1.0);
}

#[test]
fn test_benchmark_suite() {
    let mut suite = BenchmarkSuite::default();

    suite.add(QuantBenchmarkResult {
        name: "a".to_string(),
        num_elements: 100,
        bits: 8,
        granularity: QuantGranularity::PerTensor,
        mode: QuantMode::Symmetric,
        mse: 0.01,
        max_error: 0.1,
        sqnr_db: 40.0,
        compression_ratio: 4.0,
    });

    suite.add(QuantBenchmarkResult {
        name: "b".to_string(),
        num_elements: 100,
        bits: 4,
        granularity: QuantGranularity::PerTensor,
        mode: QuantMode::Symmetric,
        mse: 0.1,
        max_error: 0.5,
        sqnr_db: 20.0,
        compression_ratio: 8.0,
    });

    assert_eq!(suite.results.len(), 2);
    assert_eq!(suite.best_by_sqnr().expect("operation should succeed").name, "a");
    assert_eq!(suite.best_by_mse().expect("operation should succeed").name, "a");
}

#[test]
fn test_quality_score() {
    let result = QuantBenchmarkResult {
        name: "test".to_string(),
        num_elements: 100,
        bits: 8,
        granularity: QuantGranularity::PerTensor,
        mode: QuantMode::Symmetric,
        mse: 0.01,
        max_error: 0.1,
        sqnr_db: 40.0,
        compression_ratio: 4.0,
    };

    assert_eq!(result.quality_score(), 10.0); // 40 / 4
}

#[test]
fn test_generate_gaussian_weights() {
    let weights = generate_gaussian_weights(1000, 0.0, 1.0, 42);

    assert_eq!(weights.len(), 1000);

    // Mean should be approximately 0
    let mean: f32 = weights.iter().sum::<f32>() / weights.len() as f32;
    assert!(mean.abs() < 0.2, "Mean {mean} should be close to 0");

    // Std dev should be approximately 1
    let variance: f32 =
        weights.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / weights.len() as f32;
    let std_dev = variance.sqrt();
    assert!((std_dev - 1.0).abs() < 0.3, "Std dev {std_dev} should be close to 1");
}

#[test]
fn test_generate_uniform_weights() {
    let weights = generate_uniform_weights(1000, -1.0, 1.0, 42);

    assert_eq!(weights.len(), 1000);

    for &w in &weights {
        assert!((-1.0..=1.0).contains(&w), "Weight {w} out of range");
    }
}

#[test]
fn test_generate_weights_with_outliers() {
    let weights = generate_weights_with_outliers(1000, 0.01, 10.0, 42);

    assert_eq!(weights.len(), 1000);

    // Should have some large values
    let large_count = weights.iter().filter(|&&w| w.abs() > 5.0).count();
    assert!(large_count > 0, "Should have outliers");
}

#[test]
fn test_generate_multi_channel_weights() {
    let weights = generate_multi_channel_weights(16, 64, 5.0, 42);

    assert_eq!(weights.len(), 16 * 64);
}

#[test]
fn test_full_benchmark_suite() {
    let suite = run_full_benchmark_suite(256);

    // Should have multiple results
    assert!(suite.results.len() >= 5);

    // All results should be valid
    for result in &suite.results {
        assert!(result.mse >= 0.0);
        assert!(result.compression_ratio >= 1.0);
    }
}

#[test]
fn test_bit_width_comparison() {
    let values = generate_gaussian_weights(1000, 0.0, 1.0, 42);
    let results = compare_bit_width_degradation(&values);

    assert_eq!(results.len(), 2);

    // 8-bit should have lower MSE than 4-bit
    let (_, mse_4bit, _) =
        results.iter().find(|(b, _, _)| *b == 4).expect("operation should succeed");
    let (_, mse_8bit, _) =
        results.iter().find(|(b, _, _)| *b == 8).expect("operation should succeed");

    assert!(mse_8bit <= mse_4bit, "8-bit MSE ({mse_8bit}) should be <= 4-bit MSE ({mse_4bit})");
}

#[test]
fn test_accuracy_retention() {
    assert_eq!(accuracy_retention(0.0, 0.0), 100.0);
    assert!(accuracy_retention(0.0, 0.01) >= 0.0);
    assert!(accuracy_retention(0.0, 0.01) <= 100.0);
}

// Property tests

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_benchmark_compression_positive(
        size in 100usize..500,
        bits in prop::sample::select(vec![4u8, 8])
    ) {
        let values = generate_gaussian_weights(size, 0.0, 1.0, 42);
        let result = run_benchmark(
            "test",
            &values,
            bits,
            QuantGranularity::PerTensor,
            QuantMode::Symmetric,
        );

        prop_assert!(result.compression_ratio > 0.0);
    }

    #[test]
    fn prop_8bit_better_than_4bit(size in 100usize..500) {
        let values = generate_gaussian_weights(size, 0.0, 1.0, 42);

        let result_4bit = run_benchmark(
            "4bit",
            &values,
            4,
            QuantGranularity::PerTensor,
            QuantMode::Symmetric,
        );
        let result_8bit = run_benchmark(
            "8bit",
            &values,
            8,
            QuantGranularity::PerTensor,
            QuantMode::Symmetric,
        );

        prop_assert!(
            result_8bit.mse <= result_4bit.mse * 1.01,
            "8-bit MSE ({}) should be <= 4-bit MSE ({})",
            result_8bit.mse,
            result_4bit.mse
        );
    }

    #[test]
    fn prop_per_channel_helps_multi_scale(
        num_channels in 4usize..16,
        scale_variance in 5.0f32..20.0 // Higher variance to ensure per-channel helps
    ) {
        let features = 64;
        let values = generate_multi_channel_weights(num_channels, features, scale_variance, 42);

        // Use calibration directly with correct num_channels
        let params_tensor = calibrate_per_tensor(&values, 8, QuantMode::Symmetric);
        let params_channel = calibrate_per_channel(&values, num_channels, 8, QuantMode::Symmetric);

        let q_tensor = quantize_with_params(&values, &params_tensor);
        let q_channel = quantize_with_params(&values, &params_channel);

        let d_tensor = dequantize_with_params(&q_tensor, &params_tensor);
        let d_channel = dequantize_with_params(&q_channel, &params_channel);

        let mse_tensor = quantization_mse(&values, &d_tensor);
        let mse_channel = quantization_mse(&values, &d_channel);

        // Per-channel should be at least as good when scales vary significantly
        prop_assert!(
            mse_channel <= mse_tensor * 1.01,
            "Per-channel MSE ({}) should be <= per-tensor MSE ({})",
            mse_channel,
            mse_tensor
        );
    }

    #[test]
    fn prop_benchmark_deterministic(size in 100usize..500) {
        let values1 = generate_gaussian_weights(size, 0.0, 1.0, 42);
        let values2 = generate_gaussian_weights(size, 0.0, 1.0, 42);

        // Same seed should produce same weights
        prop_assert_eq!(values1, values2);
    }

    #[test]
    fn prop_sqnr_positive_for_signal(size in 100usize..500) {
        let values = generate_gaussian_weights(size, 0.0, 1.0, 42);
        let result = run_benchmark(
            "test",
            &values,
            8,
            QuantGranularity::PerTensor,
            QuantMode::Symmetric,
        );

        prop_assert!(result.sqnr_db > 0.0, "SQNR must be positive");
    }
}
