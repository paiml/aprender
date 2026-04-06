//! Quantization Error Analysis and Property Tests
//!
//! Provides comprehensive error analysis for quantization:
//! - Error bounds validation
//! - Error distribution analysis
//! - Outlier impact measurement
//! - Scale sensitivity analysis
//! - Numerical stability tests

use super::granularity::{
    calibrate_per_tensor, dequantize_with_params, quantization_mse, quantize_with_params,
    QuantMode, QuantParams,
};
use serde::{Deserialize, Serialize};

/// Error statistics for quantization analysis
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct QuantErrorStats {
    /// Mean Squared Error
    pub mse: f32,
    /// Mean Absolute Error
    pub mae: f32,
    /// Maximum absolute error
    pub max_error: f32,
    /// Signal-to-Quantization-Noise Ratio (SQNR) in dB
    pub sqnr_db: f32,
    /// Percentage of values with error > threshold
    pub outlier_rate: f32,
    /// Number of samples
    pub num_samples: usize,
}

impl QuantErrorStats {
    /// Root Mean Squared Error
    pub fn rmse(&self) -> f32 {
        contract_pre_rmse!();
        self.mse.sqrt()
    }
}

/// Analyze quantization error for given values and parameters
///
/// # Arguments
/// * `original` - Original f32 values
/// * `params` - Quantization parameters
/// * `outlier_threshold` - Error threshold for outlier detection
pub fn analyze_error(
    original: &[f32],
    params: &QuantParams,
    outlier_threshold: f32,
) -> QuantErrorStats {
    if original.is_empty() {
        return QuantErrorStats::default();
    }

    let quantized = quantize_with_params(original, params);
    let dequantized = dequantize_with_params(&quantized, params);

    let errors: Vec<f32> =
        original.iter().zip(dequantized.iter()).map(|(o, d)| (o - d).abs()).collect();

    let mse = quantization_mse(original, &dequantized);
    let mae = errors.iter().sum::<f32>() / errors.len().max(1) as f32;
    let max_error = errors.iter().copied().fold(0.0f32, f32::max);

    let outlier_count = errors.iter().filter(|&&e| e > outlier_threshold).count();
    let outlier_rate = outlier_count as f32 / errors.len().max(1) as f32;

    // SQNR = 10 * log10(signal_power / noise_power)
    let signal_power: f32 =
        original.iter().map(|x| x * x).sum::<f32>() / original.len().max(1) as f32;
    let noise_power = mse;
    let sqnr_db = if noise_power > 1e-10 {
        10.0 * (signal_power / noise_power).max(f32::MIN_POSITIVE).log10()
    } else {
        f32::INFINITY
    };

    QuantErrorStats { mse, mae, max_error, sqnr_db, outlier_rate, num_samples: original.len() }
}

/// Calculate theoretical maximum error for given quantization parameters
///
/// For symmetric quantization: max_error = scale / 2 (half quantization step)
/// For asymmetric: max_error = scale / 2
pub fn theoretical_max_error(params: &QuantParams) -> f32 {
    let max_scale = params.scales.iter().copied().fold(0.0f32, f32::max);
    max_scale / 2.0
}

/// Calculate expected SQNR for uniform quantization
///
/// Theoretical SQNR for b-bit quantization: 6.02 * b + 1.76 dB
/// This assumes uniform distribution of input values
pub fn theoretical_sqnr(bits: u8) -> f32 {
    6.02 * f32::from(bits) + 1.76
}

/// Check if error is within expected bounds
pub fn error_within_bounds(stats: &QuantErrorStats, params: &QuantParams, tolerance: f32) -> bool {
    let theoretical_max = theoretical_max_error(params);
    stats.max_error <= theoretical_max * (1.0 + tolerance)
}

/// Analyze sensitivity of error to scale perturbation
///
/// Returns (original_mse, perturbed_mse, sensitivity)
pub fn scale_sensitivity(
    values: &[f32],
    params: &QuantParams,
    perturbation: f32,
) -> (f32, f32, f32) {
    // Original error
    let quantized = quantize_with_params(values, params);
    let dequantized = dequantize_with_params(&quantized, params);
    let original_mse = quantization_mse(values, &dequantized);

    // Perturbed scales
    let perturbed_scales: Vec<f32> =
        params.scales.iter().map(|s| s * (1.0 + perturbation)).collect();

    let perturbed_params = QuantParams {
        scales: perturbed_scales,
        zero_points: params.zero_points.clone(),
        granularity: params.granularity,
        mode: params.mode,
        bits: params.bits,
    };

    let perturbed_quantized = quantize_with_params(values, &perturbed_params);
    let perturbed_dequantized = dequantize_with_params(&perturbed_quantized, &perturbed_params);
    let perturbed_mse = quantization_mse(values, &perturbed_dequantized);

    let sensitivity = if perturbation.abs() > 1e-10 {
        (perturbed_mse - original_mse).abs() / (perturbation.abs() * original_mse.max(1e-10))
    } else {
        0.0
    };

    (original_mse, perturbed_mse, sensitivity)
}

/// Compare error between different bit widths
///
/// Returns (mse_4bit, mse_8bit, improvement_ratio)
pub fn compare_bit_widths(values: &[f32]) -> (f32, f32, f32) {
    let params_4bit = calibrate_per_tensor(values, 4, QuantMode::Symmetric);
    let params_8bit = calibrate_per_tensor(values, 8, QuantMode::Symmetric);

    let q4 = quantize_with_params(values, &params_4bit);
    let q8 = quantize_with_params(values, &params_8bit);

    let d4 = dequantize_with_params(&q4, &params_4bit);
    let d8 = dequantize_with_params(&q8, &params_8bit);

    let mse_4bit = quantization_mse(values, &d4);
    let mse_8bit = quantization_mse(values, &d8);

    let improvement = if mse_8bit > 1e-10 {
        mse_4bit / mse_8bit
    } else if mse_4bit > 1e-10 {
        f32::INFINITY
    } else {
        1.0
    };

    (mse_4bit, mse_8bit, improvement)
}

/// Analyze impact of outliers on quantization error
///
/// Returns (original_mse, clipped_mse, outlier_impact)
pub fn analyze_outlier_impact(values: &[f32], percentile: f32) -> (f32, f32, f32) {
    if values.is_empty() || percentile <= 0.0 || percentile >= 100.0 {
        return (0.0, 0.0, 0.0);
    }

    // Sort values to find percentile thresholds
    let mut sorted: Vec<f32> = values.iter().map(|v| v.abs()).collect();
    sorted.sort_by(f32::total_cmp);

    let upper_idx = (percentile / 100.0 * sorted.len() as f32) as usize;
    let threshold = *sorted.get(upper_idx.min(sorted.len() - 1)).unwrap_or(&0.0);

    let lower_threshold = -threshold;
    let upper_threshold = threshold;

    // Clipped values
    let clipped: Vec<f32> =
        values.iter().map(|&v| v.clamp(lower_threshold, upper_threshold)).collect();

    // Quantize both
    let params_original = calibrate_per_tensor(values, 8, QuantMode::Symmetric);
    let params_clipped = calibrate_per_tensor(&clipped, 8, QuantMode::Symmetric);

    let q_orig = quantize_with_params(values, &params_original);
    let q_clip = quantize_with_params(&clipped, &params_clipped);

    let d_orig = dequantize_with_params(&q_orig, &params_original);
    let d_clip = dequantize_with_params(&q_clip, &params_clipped);

    let mse_original = quantization_mse(values, &d_orig);
    let mse_clipped = quantization_mse(&clipped, &d_clip);

    let outlier_impact = if mse_clipped > 1e-10 {
        mse_original / mse_clipped
    } else if mse_original > 1e-10 {
        f32::INFINITY
    } else {
        1.0
    };

    (mse_original, mse_clipped, outlier_impact)
}

#[cfg(test)]
mod tests {
    use super::super::granularity::{calibrate_per_channel, QuantGranularity};
    use super::*;
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;

    #[test]
    fn test_error_stats_basic() {
        let values: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let params = calibrate_per_tensor(&values, 8, QuantMode::Symmetric);
        let stats = analyze_error(&values, &params, 0.01);

        assert!(stats.mse >= 0.0);
        assert!(stats.mae >= 0.0);
        assert!(stats.max_error >= 0.0);
        assert!(stats.sqnr_db > 0.0);
        assert_eq!(stats.num_samples, 100);
    }

    #[test]
    fn test_rmse_calculation() {
        let stats = QuantErrorStats { mse: 4.0, ..Default::default() };
        assert_abs_diff_eq!(stats.rmse(), 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_theoretical_max_error() {
        let params = QuantParams {
            scales: vec![0.1, 0.2],
            zero_points: vec![],
            granularity: QuantGranularity::PerChannel,
            mode: QuantMode::Symmetric,
            bits: 8,
        };

        let max_err = theoretical_max_error(&params);
        assert_abs_diff_eq!(max_err, 0.1, epsilon = 1e-6); // max scale / 2
    }

    #[test]
    fn test_theoretical_sqnr() {
        // 8-bit: 6.02 * 8 + 1.76 = 49.92 dB
        let sqnr_8bit = theoretical_sqnr(8);
        assert_abs_diff_eq!(sqnr_8bit, 49.92, epsilon = 0.01);

        // 4-bit: 6.02 * 4 + 1.76 = 25.84 dB
        let sqnr_4bit = theoretical_sqnr(4);
        assert_abs_diff_eq!(sqnr_4bit, 25.84, epsilon = 0.01);
    }

    #[test]
    fn test_error_within_bounds() {
        let values: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        let params = calibrate_per_tensor(&values, 8, QuantMode::Symmetric);
        let stats = analyze_error(&values, &params, 0.1);

        // Error should be within bounds with some tolerance
        assert!(error_within_bounds(&stats, &params, 0.1));
    }

    #[test]
    fn test_scale_sensitivity() {
        let values: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let params = calibrate_per_tensor(&values, 8, QuantMode::Symmetric);

        let (orig_mse, pert_mse, sensitivity) = scale_sensitivity(&values, &params, 0.1);

        assert!(orig_mse >= 0.0);
        assert!(pert_mse >= 0.0);
        assert!(sensitivity >= 0.0);
    }

    #[test]
    fn test_compare_bit_widths() {
        let values: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();

        let (mse_4bit, mse_8bit, improvement) = compare_bit_widths(&values);

        // 8-bit should be better than 4-bit
        assert!(mse_8bit <= mse_4bit);
        assert!(improvement >= 1.0);
    }

    #[test]
    fn test_outlier_impact() {
        // Values with outliers
        let mut values: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01).sin()).collect();
        values.push(100.0); // Add outlier
        values.push(-100.0); // Add outlier

        let (mse_orig, mse_clip, impact) = analyze_outlier_impact(&values, 99.0);

        // Clipping should generally help when there are outliers
        assert!(mse_orig >= 0.0);
        assert!(mse_clip >= 0.0);
        assert!(impact >= 0.0);
    }

    #[test]
    fn test_empty_values() {
        let values: Vec<f32> = vec![];
        let params = calibrate_per_tensor(&values, 8, QuantMode::Symmetric);
        let stats = analyze_error(&values, &params, 0.1);

        assert_eq!(stats.num_samples, 0);
    }

    #[test]
    fn test_zeros_error() {
        let values = vec![0.0; 100];
        let params = calibrate_per_tensor(&values, 8, QuantMode::Symmetric);
        let stats = analyze_error(&values, &params, 0.001);

        // Zeros should quantize perfectly
        assert!(stats.mse < 1e-10);
        assert!(stats.mae < 1e-10);
    }

    // Property tests

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_mse_non_negative(values in proptest::collection::vec(-100.0f32..100.0, 10..100)) {
            let params = calibrate_per_tensor(&values, 8, QuantMode::Symmetric);
            let stats = analyze_error(&values, &params, 0.1);

            prop_assert!(stats.mse >= 0.0, "MSE must be non-negative");
            prop_assert!(stats.mae >= 0.0, "MAE must be non-negative");
            prop_assert!(stats.max_error >= 0.0, "Max error must be non-negative");
        }

        #[test]
        fn prop_8bit_better_than_4bit(values in proptest::collection::vec(-100.0f32..100.0, 10..100)) {
            let (mse_4bit, mse_8bit, _) = compare_bit_widths(&values);

            prop_assert!(
                mse_8bit <= mse_4bit * 1.01, // Small tolerance for edge cases
                "8-bit MSE ({}) should be <= 4-bit MSE ({})",
                mse_8bit,
                mse_4bit
            );
        }

        #[test]
        fn prop_error_bounded(values in proptest::collection::vec(-100.0f32..100.0, 10..100)) {
            let params = calibrate_per_tensor(&values, 8, QuantMode::Symmetric);
            let stats = analyze_error(&values, &params, 0.1);

            // Error should be bounded by theoretical max (with tolerance)
            let theoretical_max = theoretical_max_error(&params);
            prop_assert!(
                stats.max_error <= theoretical_max * 1.5,
                "Max error ({}) should be <= theoretical max * 1.5 ({})",
                stats.max_error,
                theoretical_max * 1.5
            );
        }

        #[test]
        fn prop_sqnr_positive_for_nonzero_signal(
            values in proptest::collection::vec(
                prop_oneof![
                    -100.0f32..-1.0,
                    1.0f32..100.0,
                ],
                10..100
            )
        ) {
            let params = calibrate_per_tensor(&values, 8, QuantMode::Symmetric);
            let stats = analyze_error(&values, &params, 0.1);

            // SQNR should be positive for non-zero signal
            prop_assert!(stats.sqnr_db > 0.0, "SQNR must be positive for non-zero signal");
        }

        #[test]
        fn prop_outlier_rate_bounded(
            values in proptest::collection::vec(-100.0f32..100.0, 10..100),
            threshold in 0.001f32..10.0
        ) {
            let params = calibrate_per_tensor(&values, 8, QuantMode::Symmetric);
            let stats = analyze_error(&values, &params, threshold);

            prop_assert!(
                stats.outlier_rate >= 0.0 && stats.outlier_rate <= 1.0,
                "Outlier rate must be in [0, 1], got {}",
                stats.outlier_rate
            );
        }

        #[test]
        fn prop_per_channel_lower_error(
            num_channels in 2usize..5,
            features_per_channel in 5usize..20,
            scale_multiplier in 2.0f32..20.0
        ) {
            // Create values where channels have very different scales
            let values: Vec<f32> = (0..num_channels)
                .flat_map(|ch| {
                    let scale = (ch as f32 + 1.0) * scale_multiplier;
                    (0..features_per_channel).map(move |i| {
                        (i as f32 / features_per_channel as f32 - 0.5) * scale
                    })
                })
                .collect();

            let params_pt = calibrate_per_tensor(&values, 8, QuantMode::Symmetric);
            let params_pc = calibrate_per_channel(&values, num_channels, 8, QuantMode::Symmetric);

            let stats_pt = analyze_error(&values, &params_pt, 0.1);
            let stats_pc = analyze_error(&values, &params_pc, 0.1);

            prop_assert!(
                stats_pc.mse <= stats_pt.mse * 1.01,
                "Per-channel MSE ({}) should be <= per-tensor MSE ({})",
                stats_pc.mse,
                stats_pt.mse
            );
        }

        #[test]
        fn prop_scale_sensitivity_finite(
            values in proptest::collection::vec(-100.0f32..100.0, 10..100),
            perturbation in 0.01f32..0.5
        ) {
            let params = calibrate_per_tensor(&values, 8, QuantMode::Symmetric);
            let (orig, pert, sens) = scale_sensitivity(&values, &params, perturbation);

            prop_assert!(orig.is_finite(), "Original MSE must be finite");
            prop_assert!(pert.is_finite(), "Perturbed MSE must be finite");
            prop_assert!(sens.is_finite(), "Sensitivity must be finite");
        }
    }
}
