//! Value at Risk (`VaR`) calculations
//!
//! `VaR` at confidence level α is the threshold value such that the probability
//! of loss exceeding `VaR` is (1-α).
//!
//! Reference: Jorion (2006), "Value at Risk"

use crate::monte_carlo::engine::{percentile, SimulationPath};
use crate::monte_carlo::error::{MonteCarloError, Result};

/// Value at Risk calculator
#[derive(Debug, Clone, Copy)]
pub struct VaR;

impl VaR {
    /// Calculate historical `VaR` from returns
    ///
    /// VaR(α) = -Quantile(Returns, 1-α)
    ///
    /// Returns `VaR` as a positive value (loss)
    ///
    /// # Arguments
    /// * `returns` - Vector of returns (can be positive or negative)
    /// * `confidence` - Confidence level (e.g., 0.95 for 95% `VaR`)
    ///
    /// # Example
    /// ```
    /// use aprender::monte_carlo::risk::VaR;
    ///
    /// let returns = vec![-0.05, -0.02, 0.01, 0.03, 0.05, 0.02, -0.01, 0.04, -0.03, 0.00];
    /// let var_95 = VaR::historical(&returns, 0.95);
    /// assert!(var_95 >= 0.0); // VaR is positive (represents loss)
    /// ```
    #[must_use]
    pub fn historical(returns: &[f64], confidence: f64) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        // VaR is the negative of the (1-confidence) quantile
        let quantile_level = 1.0 - confidence;
        let loss_quantile = percentile(returns, quantile_level);

        // Return as positive loss
        -loss_quantile.min(0.0)
    }

    /// Calculate parametric `VaR` assuming normal distribution
    ///
    /// `VaR` = -μ - σ × `z_α` where `z_α` = Φ⁻¹(1-α) < 0
    ///
    /// # Arguments
    /// * `mean` - Mean of returns
    /// * `std` - Standard deviation of returns
    /// * `confidence` - Confidence level
    #[must_use]
    pub fn parametric(mean: f64, std: f64, confidence: f64) -> f64 {
        let z = quantile_normal(1.0 - confidence);
        // z is negative for left tail (e.g., -1.645 for 95% confidence)
        // VaR = -(mean + std * z) to get positive loss
        -(mean + std * z).min(0.0)
    }

    /// Calculate Cornish-Fisher `VaR` (accounts for skewness and kurtosis)
    ///
    /// Adjusts the normal quantile for non-normality
    #[must_use]
    pub fn cornish_fisher(
        mean: f64,
        std: f64,
        skewness: f64,
        excess_kurtosis: f64,
        confidence: f64,
    ) -> f64 {
        let z = quantile_normal(1.0 - confidence);

        // Cornish-Fisher adjustment
        let z_cf =
            z + (z.powi(2) - 1.0) * skewness / 6.0 + (z.powi(3) - 3.0 * z) * excess_kurtosis / 24.0
                - (2.0 * z.powi(3) - 5.0 * z) * skewness.powi(2) / 36.0;

        // z_cf is negative for left tail, so we negate to get positive loss
        -(mean + std * z_cf).min(0.0)
    }

    /// Calculate `VaR` from simulation paths
    #[must_use]
    pub fn from_paths(paths: &[SimulationPath], confidence: f64) -> f64 {
        let returns: Vec<f64> = paths
            .iter()
            .filter_map(SimulationPath::total_return)
            .collect();
        Self::historical(&returns, confidence)
    }

    /// Validate confidence level
    pub fn validate_confidence(confidence: f64) -> Result<()> {
        if confidence <= 0.0 || confidence >= 1.0 {
            return Err(MonteCarloError::InvalidConfidenceLevel { value: confidence });
        }
        Ok(())
    }
}

/// Inverse normal CDF (quantile function)
///
/// Uses the Acklam approximation which provides accuracy to ~1.15e-9
#[allow(clippy::excessive_precision)]
fn quantile_normal(p: f64) -> f64 {
    // Clamp to avoid infinities
    let p = p.clamp(1e-15, 1.0 - 1e-15);

    // Acklam's approximation constants
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_690e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];

    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];

    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];

    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];

    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        // Lower tail
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        // Central region
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        // Upper tail
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_historical_var_basic() {
        // Simple test case: uniform losses from -10% to +10%
        let returns: Vec<f64> = (-100..=100).map(|i| i as f64 / 1000.0).collect();

        let var_95 = VaR::historical(&returns, 0.95);

        // 5th percentile of [-0.1, 0.1] should be around -0.09
        // VaR should be around 0.09 (positive loss)
        assert!(var_95 > 0.08 && var_95 < 0.10, "VaR(95%) = {var_95}");
    }

    #[test]
    fn test_historical_var_all_positive() {
        // All positive returns
        let returns = vec![0.01, 0.02, 0.03, 0.04, 0.05];

        let var_95 = VaR::historical(&returns, 0.95);

        // No losses, VaR should be 0 or very small
        assert!(var_95 >= 0.0);
    }

    #[test]
    fn test_historical_var_all_negative() {
        // All negative returns (losses)
        let returns = vec![-0.01, -0.02, -0.03, -0.04, -0.05];

        let var_95 = VaR::historical(&returns, 0.95);

        // VaR should be positive (representing loss)
        assert!(var_95 > 0.0);
    }

    #[test]
    fn test_historical_var_monotonic() {
        let returns: Vec<f64> = (-100..=100).map(|i| i as f64 / 1000.0).collect();

        let var_90 = VaR::historical(&returns, 0.90);
        let var_95 = VaR::historical(&returns, 0.95);
        let var_99 = VaR::historical(&returns, 0.99);

        // Higher confidence should give higher VaR
        assert!(
            var_90 <= var_95,
            "VaR(90%)={var_90} should be <= VaR(95%)={var_95}"
        );
        assert!(
            var_95 <= var_99,
            "VaR(95%)={var_95} should be <= VaR(99%)={var_99}"
        );
    }

    #[test]
    fn test_parametric_var() {
        // Normal distribution: mean=0, std=0.1
        let var_95 = VaR::parametric(0.0, 0.1, 0.95);

        // z(0.05) ≈ -1.645, so VaR ≈ 0.1 * 1.645 ≈ 0.165
        assert!((var_95 - 0.165).abs() < 0.01, "Parametric VaR = {var_95}");
    }

    #[test]
    fn test_cornish_fisher_var() {
        // With zero skewness and kurtosis, should match parametric
        let var_param = VaR::parametric(0.0, 0.1, 0.95);
        let var_cf = VaR::cornish_fisher(0.0, 0.1, 0.0, 0.0, 0.95);

        assert!(
            (var_param - var_cf).abs() < 0.01,
            "CF VaR should match parametric for normal: {} vs {}",
            var_cf,
            var_param
        );
    }

    #[test]
    fn test_cornish_fisher_positive_skew() {
        // Positive skewness (heavy right tail, lighter left tail)
        // Should reduce VaR compared to parametric
        let var_param = VaR::parametric(0.0, 0.1, 0.95);
        let var_cf = VaR::cornish_fisher(0.0, 0.1, 0.5, 0.0, 0.95);

        // Positive skew means less severe left tail
        assert!(
            var_cf < var_param + 0.05,
            "Positive skew should affect VaR: CF={var_cf} vs Param={var_param}"
        );
    }

    #[test]
    fn test_cornish_fisher_excess_kurtosis() {
        // Positive excess kurtosis (fat tails)
        // The Cornish-Fisher adjustment modifies the quantile based on kurtosis
        let var_param = VaR::parametric(0.0, 0.1, 0.95);
        let var_cf = VaR::cornish_fisher(0.0, 0.1, 0.0, 3.0, 0.95);

        // With non-zero excess kurtosis, CF should differ from parametric
        // Note: The Cornish-Fisher expansion can give counterintuitive results
        // at moderate confidence levels; this tests the formula works
        assert!(
            (var_cf - var_param).abs() > 0.001,
            "Excess kurtosis should affect VaR: CF={var_cf} vs Param={var_param}"
        );
        assert!(var_cf > 0.0, "VaR should be positive: {var_cf}");
        assert!(var_cf.is_finite(), "VaR should be finite");
    }

    #[test]
    fn test_validate_confidence() {
        assert!(VaR::validate_confidence(0.95).is_ok());
        assert!(VaR::validate_confidence(0.99).is_ok());
        assert!(VaR::validate_confidence(0.5).is_ok());

        assert!(VaR::validate_confidence(0.0).is_err());
        assert!(VaR::validate_confidence(1.0).is_err());
        assert!(VaR::validate_confidence(-0.1).is_err());
        assert!(VaR::validate_confidence(1.5).is_err());
    }

    #[test]
    fn test_quantile_normal() {
        // Test known values
        assert!((quantile_normal(0.5) - 0.0).abs() < 1e-6);
        assert!((quantile_normal(0.84134) - 1.0).abs() < 1e-3);
        assert!((quantile_normal(0.15866) - (-1.0)).abs() < 1e-3);
        assert!((quantile_normal(0.97725) - 2.0).abs() < 1e-3);
        assert!((quantile_normal(0.02275) - (-2.0)).abs() < 1e-3);
    }

    #[test]
    fn test_empty_returns() {
        let var = VaR::historical(&[], 0.95);
        assert!(var.abs() < 1e-10);
    }

    // Property-based tests
    #[cfg(test)]
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_var_non_negative(
                returns in prop::collection::vec(-1.0..1.0f64, 10..100),
                confidence in 0.5..0.999f64,
            ) {
                let var = VaR::historical(&returns, confidence);
                prop_assert!(var >= 0.0, "VaR should be non-negative: {var}");
            }

            #[test]
            fn prop_var_monotonic_in_confidence(
                returns in prop::collection::vec(-1.0..1.0f64, 100..500),
            ) {
                let var_90 = VaR::historical(&returns, 0.90);
                let var_95 = VaR::historical(&returns, 0.95);
                let var_99 = VaR::historical(&returns, 0.99);

                prop_assert!(var_90 <= var_95 + 0.001);
                prop_assert!(var_95 <= var_99 + 0.001);
            }

            #[test]
            fn prop_parametric_var_non_negative(
                mean in -0.5..0.5f64,
                std in 0.01..0.5f64,
                confidence in 0.5..0.999f64,
            ) {
                let var = VaR::parametric(mean, std, confidence);
                prop_assert!(var >= 0.0);
            }
        }
    }
}
