//! Conditional Value at Risk (`CVaR` / Expected Shortfall)
//!
//! `CVaR` is the expected loss given that loss exceeds `VaR`.
//! It is a coherent risk measure (unlike `VaR`).
//!
//! Reference: Artzner et al. (1999), "Coherent Measures of Risk"

use crate::monte_carlo::engine::{percentile, SimulationPath};

/// Conditional Value at Risk (Expected Shortfall) calculator
#[derive(Debug, Clone, Copy)]
pub struct CVaR;

impl CVaR {
    /// Calculate `CVaR` from returns
    ///
    /// CVaR(α) = E[Loss | Loss > VaR(α)]
    ///
    /// Returns `CVaR` as a positive value (expected loss in tail)
    ///
    /// # Arguments
    /// * `returns` - Vector of returns
    /// * `confidence` - Confidence level (e.g., 0.95)
    ///
    /// # Example
    /// ```
    /// use aprender::monte_carlo::risk::CVaR;
    ///
    /// let returns = vec![-0.10, -0.05, -0.02, 0.01, 0.03, 0.05, 0.02, -0.01, 0.04, -0.03];
    /// let cvar_95 = CVaR::from_returns(&returns, 0.95);
    /// assert!(cvar_95 >= 0.0); // CVaR is positive (expected loss)
    /// ```
    #[must_use]
    pub fn from_returns(returns: &[f64], confidence: f64) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        // Find VaR threshold
        let quantile_level = 1.0 - confidence;
        let var_threshold = percentile(returns, quantile_level);

        // Calculate average of returns below VaR threshold
        let tail_returns: Vec<f64> = returns
            .iter()
            .filter(|&&r| r <= var_threshold)
            .copied()
            .collect();

        if tail_returns.is_empty() {
            // If no returns in tail, CVaR equals VaR
            return -var_threshold.min(0.0);
        }

        // Expected shortfall is average of tail losses
        let avg_tail = tail_returns.iter().sum::<f64>() / tail_returns.len() as f64;

        // Return as positive loss
        -avg_tail.min(0.0)
    }

    /// Calculate `CVaR` from simulation paths
    #[must_use]
    pub fn from_paths(paths: &[SimulationPath], confidence: f64) -> f64 {
        let returns: Vec<f64> = paths
            .iter()
            .filter_map(SimulationPath::total_return)
            .collect();
        Self::from_returns(&returns, confidence)
    }

    /// Calculate `CVaR` using continuous distribution approximation
    ///
    /// For continuous distributions:
    /// CVaR(α) = (1/(1-α)) × ∫_{-∞}^{VaR(α)} x × f(x) dx
    ///
    /// This uses the empirical approximation.
    #[must_use]
    pub fn continuous_approximation(returns: &[f64], confidence: f64) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mut sorted = returns.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        let tail_size = ((1.0 - confidence) * n as f64).ceil() as usize;
        let tail_size = tail_size.max(1).min(n);

        // Average of the tail_size worst returns
        let tail_sum: f64 = sorted[..tail_size].iter().sum();
        let avg = tail_sum / tail_size as f64;

        -avg.min(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monte_carlo::risk::VaR;

    #[test]
    fn test_cvar_basic() {
        let returns: Vec<f64> = (-100..=100).map(|i| i as f64 / 1000.0).collect();

        let cvar_95 = CVaR::from_returns(&returns, 0.95);

        // CVaR should be positive (loss)
        assert!(cvar_95 > 0.0, "CVaR should be positive: {cvar_95}");
    }

    #[test]
    fn test_cvar_geq_var() {
        let returns: Vec<f64> = (-100..=100).map(|i| i as f64 / 1000.0).collect();

        let var_95 = VaR::historical(&returns, 0.95);
        let cvar_95 = CVaR::from_returns(&returns, 0.95);

        // CVaR should be >= VaR
        assert!(
            cvar_95 >= var_95 - 0.001,
            "CVaR({cvar_95}) should be >= VaR({var_95})"
        );
    }

    #[test]
    fn test_cvar_monotonic() {
        let returns: Vec<f64> = (-100..=100).map(|i| i as f64 / 1000.0).collect();

        let cvar_90 = CVaR::from_returns(&returns, 0.90);
        let cvar_95 = CVaR::from_returns(&returns, 0.95);
        let cvar_99 = CVaR::from_returns(&returns, 0.99);

        // Higher confidence should give higher CVaR (looking at worse tail)
        assert!(cvar_90 <= cvar_95 + 0.01);
        assert!(cvar_95 <= cvar_99 + 0.01);
    }

    #[test]
    fn test_cvar_all_positive_returns() {
        let returns = vec![0.01, 0.02, 0.03, 0.04, 0.05];

        let cvar_95 = CVaR::from_returns(&returns, 0.95);

        // No losses, CVaR should be 0 or very small
        assert!(cvar_95 >= 0.0);
    }

    #[test]
    fn test_cvar_all_negative_returns() {
        let returns = vec![-0.05, -0.04, -0.03, -0.02, -0.01];

        let cvar_95 = CVaR::from_returns(&returns, 0.95);
        let var_95 = VaR::historical(&returns, 0.95);

        // CVaR should be >= VaR
        assert!(cvar_95 >= var_95 - 0.001);
        assert!(cvar_95 > 0.0);
    }

    #[test]
    fn test_cvar_continuous_approximation() {
        let returns: Vec<f64> = (-100..=100).map(|i| i as f64 / 1000.0).collect();

        let cvar_standard = CVaR::from_returns(&returns, 0.95);
        let cvar_continuous = CVaR::continuous_approximation(&returns, 0.95);

        // Should be similar
        assert!(
            (cvar_standard - cvar_continuous).abs() < 0.01,
            "Standard: {cvar_standard}, Continuous: {cvar_continuous}"
        );
    }

    #[test]
    fn test_cvar_extreme_confidence() {
        let returns: Vec<f64> = (-100..=100).map(|i| i as f64 / 1000.0).collect();

        let cvar_99 = CVaR::from_returns(&returns, 0.99);
        let cvar_999 = CVaR::from_returns(&returns, 0.999);

        // Both should be valid
        assert!(cvar_99.is_finite());
        assert!(cvar_999.is_finite());
        assert!(cvar_999 >= cvar_99 - 0.01);
    }

    #[test]
    fn test_cvar_empty() {
        let cvar = CVaR::from_returns(&[], 0.95);
        assert!(cvar.abs() < 1e-10);
    }

    #[test]
    fn test_cvar_single_value() {
        let returns = vec![-0.05];
        let cvar = CVaR::from_returns(&returns, 0.95);
        // Single negative return: CVaR should be 0.05
        assert!((cvar - 0.05).abs() < 0.01);
    }

    // Property-based tests
    #[cfg(test)]
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_cvar_geq_var(
                returns in prop::collection::vec(-1.0..1.0f64, 100..500),
                confidence in 0.8..0.99f64,
            ) {
                let var = VaR::historical(&returns, confidence);
                let cvar = CVaR::from_returns(&returns, confidence);

                // CVaR should be >= VaR (with small tolerance for numerical issues)
                prop_assert!(
                    cvar >= var - 0.01,
                    "CVaR({cvar}) should be >= VaR({var})"
                );
            }

            #[test]
            fn prop_cvar_non_negative(
                returns in prop::collection::vec(-1.0..1.0f64, 10..100),
                confidence in 0.5..0.999f64,
            ) {
                let cvar = CVaR::from_returns(&returns, confidence);
                prop_assert!(cvar >= 0.0, "CVaR should be non-negative: {cvar}");
            }

            #[test]
            fn prop_cvar_finite(
                returns in prop::collection::vec(-1.0..1.0f64, 10..100),
                confidence in 0.5..0.999f64,
            ) {
                let cvar = CVaR::from_returns(&returns, confidence);
                prop_assert!(cvar.is_finite(), "CVaR should be finite: {cvar}");
            }
        }
    }
}
