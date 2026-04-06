//! Benchmark runner functions
//!
//! Functions for running quantization benchmarks.

use super::super::error_analysis::analyze_error;
use super::super::granularity::{
    calibrate_per_channel, calibrate_per_group, calibrate_per_tensor, dequantize_with_params,
    quantization_mse, quantize_with_params, QuantGranularity, QuantMode,
};
use super::generators::{
    generate_gaussian_weights, generate_multi_channel_weights, generate_uniform_weights,
    generate_weights_with_outliers,
};
use super::types::{BenchmarkSuite, QuantBenchmarkResult};

/// Run benchmark on given values with specified configuration
pub fn run_benchmark(
    name: &str,
    values: &[f32],
    bits: u8,
    granularity: QuantGranularity,
    mode: QuantMode,
) -> QuantBenchmarkResult {
    let params = match granularity {
        QuantGranularity::PerTensor => calibrate_per_tensor(values, bits, mode),
        QuantGranularity::PerChannel => {
            // Assume square-ish shape for simplicity
            let num_channels = (values.len() as f32).sqrt() as usize;
            calibrate_per_channel(values, num_channels.max(1), bits, mode)
        }
        QuantGranularity::PerGroup(size) => calibrate_per_group(values, size, bits, mode),
    };

    let stats = analyze_error(values, &params, 0.1);

    // Calculate compression ratio
    let original_bytes = values.len() * 4; // f32 = 4 bytes
    let scale_bytes = params.scales.len() * 4;
    let zp_bytes = params.zero_points.len() * 4;
    let data_bytes = if bits == 4 { values.len().div_ceil(2) } else { values.len() };
    let compressed_bytes = scale_bytes + zp_bytes + data_bytes;
    let compression_ratio = original_bytes as f32 / compressed_bytes.max(1) as f32;

    QuantBenchmarkResult {
        name: name.to_string(),
        num_elements: values.len(),
        bits,
        granularity,
        mode,
        mse: stats.mse,
        max_error: stats.max_error,
        sqnr_db: stats.sqnr_db,
        compression_ratio,
    }
}

/// Run full benchmark suite on various weight patterns
pub fn run_full_benchmark_suite(size: usize) -> BenchmarkSuite {
    let mut suite = BenchmarkSuite::default();

    // Gaussian weights
    let gaussian = generate_gaussian_weights(size, 0.0, 1.0, 42);

    // Test different configurations
    for bits in [4u8, 8] {
        for granularity in [
            QuantGranularity::PerTensor,
            QuantGranularity::PerChannel,
            QuantGranularity::PerGroup(32),
        ] {
            let name = format!(
                "gaussian_{}bit_{:?}",
                bits,
                match granularity {
                    QuantGranularity::PerTensor => "tensor",
                    QuantGranularity::PerChannel => "channel",
                    QuantGranularity::PerGroup(_) => "group",
                }
            );
            suite.add(run_benchmark(&name, &gaussian, bits, granularity, QuantMode::Symmetric));
        }
    }

    // Uniform weights
    let uniform = generate_uniform_weights(size, -1.0, 1.0, 43);
    suite.add(run_benchmark(
        "uniform_8bit_tensor",
        &uniform,
        8,
        QuantGranularity::PerTensor,
        QuantMode::Symmetric,
    ));

    // Weights with outliers
    let outliers = generate_weights_with_outliers(size, 0.01, 10.0, 44);
    suite.add(run_benchmark(
        "outliers_8bit_tensor",
        &outliers,
        8,
        QuantGranularity::PerTensor,
        QuantMode::Symmetric,
    ));
    suite.add(run_benchmark(
        "outliers_8bit_group32",
        &outliers,
        8,
        QuantGranularity::PerGroup(32),
        QuantMode::Symmetric,
    ));

    // Multi-channel weights
    let multi_ch = generate_multi_channel_weights(16, size / 16, 5.0, 45);
    suite.add(run_benchmark(
        "multi_channel_8bit_tensor",
        &multi_ch,
        8,
        QuantGranularity::PerTensor,
        QuantMode::Symmetric,
    ));
    suite.add(run_benchmark(
        "multi_channel_8bit_channel",
        &multi_ch,
        8,
        QuantGranularity::PerChannel,
        QuantMode::Symmetric,
    ));

    suite
}

/// Compare accuracy degradation across bit widths
pub fn compare_bit_width_degradation(values: &[f32]) -> Vec<(u8, f32, f32)> {
    let mut results = Vec::new();

    for bits in [4u8, 8] {
        let params = calibrate_per_tensor(values, bits, QuantMode::Symmetric);
        let quantized = quantize_with_params(values, &params);
        let dequantized = dequantize_with_params(&quantized, &params);
        let mse = quantization_mse(values, &dequantized);

        let compression = if bits == 4 { 8.0 } else { 4.0 }; // vs f32
        results.push((bits, mse, compression));
    }

    results
}

/// Calculate accuracy retention percentage
pub fn accuracy_retention(original_mse: f32, quantized_mse: f32) -> f32 {
    if quantized_mse > 1e-10 {
        (1.0 - (quantized_mse - original_mse).abs() / quantized_mse.max(original_mse)) * 100.0
    } else if original_mse > 1e-10 {
        0.0
    } else {
        100.0
    }
}
