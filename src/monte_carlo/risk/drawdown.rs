//! Maximum Drawdown Analysis
//!
//! Measures peak-to-trough decline during a specific period.
//! Critical for understanding tail risk and recovery requirements.
//!
//! Reference: Magdon-Ismail & Atiya (2004), "Maximum Drawdown"

use crate::monte_carlo::engine::{percentile, SimulationPath};

/// Maximum drawdown calculator for a single path
#[derive(Debug, Clone)]
pub struct DrawdownAnalysis;

impl DrawdownAnalysis {
    /// Calculate maximum drawdown from a value series
    ///
    /// Maximum drawdown = max((peak - trough) / peak)
    ///
    /// # Arguments
    /// * `values` - Time series of portfolio values
    ///
    /// # Returns
    /// Maximum drawdown as a positive fraction (0.1 = 10% drawdown)
    ///
    /// # Example
    /// ```
    /// use aprender::monte_carlo::risk::DrawdownAnalysis;
    ///
    /// let values = vec![100.0, 110.0, 90.0, 95.0, 85.0, 100.0];
    /// let max_dd = DrawdownAnalysis::max_drawdown(&values);
    /// // Peak was 110, trough was 85: (110-85)/110 ≈ 0.227
    /// assert!(max_dd > 0.22 && max_dd < 0.24);
    /// ```
    #[must_use]
    pub fn max_drawdown(values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let mut max_drawdown = 0.0;
        let mut peak = values[0];

        for &value in values.iter().skip(1) {
            if value > peak {
                peak = value;
            } else if peak > 0.0 {
                let drawdown = (peak - value) / peak;
                if drawdown > max_drawdown {
                    max_drawdown = drawdown;
                }
            }
        }

        max_drawdown
    }

    /// Calculate drawdown series from values
    ///
    /// Returns the drawdown at each time point relative to the running peak
    #[must_use]
    pub fn drawdown_series(values: &[f64]) -> Vec<f64> {
        if values.is_empty() {
            return Vec::new();
        }

        let mut drawdowns = Vec::with_capacity(values.len());
        let mut peak = values[0];

        for &value in values {
            if value > peak {
                peak = value;
            }
            let dd = if peak > 0.0 {
                (peak - value) / peak
            } else {
                0.0
            };
            drawdowns.push(dd);
        }

        drawdowns
    }

    /// Calculate drawdown duration (time to recovery)
    ///
    /// Returns the maximum number of periods spent in drawdown
    #[must_use]
    pub fn max_drawdown_duration(values: &[f64]) -> usize {
        if values.len() < 2 {
            return 0;
        }

        let mut max_duration = 0;
        let mut current_duration = 0;
        let mut peak = values[0];

        for &value in values.iter().skip(1) {
            if value >= peak {
                peak = value;
                current_duration = 0;
            } else {
                current_duration += 1;
                if current_duration > max_duration {
                    max_duration = current_duration;
                }
            }
        }

        max_duration
    }

    /// Calculate Ulcer Index (quadratic mean of drawdowns)
    ///
    /// UI = sqrt(mean(drawdown^2))
    ///
    /// Penalizes deep drawdowns more than shallow ones
    #[must_use]
    pub fn ulcer_index(values: &[f64]) -> f64 {
        let drawdowns = Self::drawdown_series(values);
        if drawdowns.is_empty() {
            return 0.0;
        }

        let sum_sq: f64 = drawdowns.iter().map(|d| d * d).sum();
        (sum_sq / drawdowns.len() as f64).sqrt()
    }

    /// Calculate Pain Index (average drawdown)
    #[must_use]
    pub fn pain_index(values: &[f64]) -> f64 {
        let drawdowns = Self::drawdown_series(values);
        if drawdowns.is_empty() {
            return 0.0;
        }

        drawdowns.iter().sum::<f64>() / drawdowns.len() as f64
    }

    /// Calculate drawdown statistics from simulation paths
    #[must_use]
    pub fn from_paths(paths: &[SimulationPath]) -> DrawdownStatistics {
        if paths.is_empty() {
            return DrawdownStatistics::default();
        }

        let max_drawdowns: Vec<f64> = paths
            .iter()
            .map(|p| Self::max_drawdown(&p.values))
            .collect();

        DrawdownStatistics::from_drawdowns(&max_drawdowns)
    }

    /// Calculate recovery factor
    ///
    /// Recovery Factor = Total Return / Max Drawdown
    ///
    /// Higher is better - shows how well returns compensate for risk
    #[must_use]
    pub fn recovery_factor(values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let first = values[0];
        let last = values[values.len() - 1];

        if first <= 0.0 {
            return 0.0;
        }

        let total_return = (last - first) / first;
        let max_dd = Self::max_drawdown(values);

        if max_dd > 0.0 {
            total_return / max_dd
        } else if total_return > 0.0 {
            f64::INFINITY
        } else {
            0.0
        }
    }
}

/// Statistics on maximum drawdowns across simulation paths
#[derive(Debug, Clone, Default)]
pub struct DrawdownStatistics {
    /// Mean of maximum drawdowns
    pub mean: f64,
    /// Median of maximum drawdowns
    pub median: f64,
    /// Standard deviation of maximum drawdowns
    pub std: f64,
    /// 5th percentile (best case)
    pub p5: f64,
    /// 25th percentile
    pub p25: f64,
    /// 75th percentile
    pub p75: f64,
    /// 95th percentile (stress scenario)
    pub p95: f64,
    /// 99th percentile (extreme stress)
    pub p99: f64,
    /// Maximum (worst case observed)
    pub worst: f64,
    /// Minimum (best case observed)
    pub best: f64,
}

impl DrawdownStatistics {
    /// Create statistics from a collection of max drawdowns
    #[must_use]
    pub fn from_drawdowns(drawdowns: &[f64]) -> Self {
        if drawdowns.is_empty() {
            return Self::default();
        }

        let n = drawdowns.len() as f64;
        let mean = drawdowns.iter().sum::<f64>() / n;

        let variance = drawdowns.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();

        let worst = drawdowns.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let best = drawdowns.iter().copied().fold(f64::INFINITY, f64::min);

        Self {
            mean,
            median: percentile(drawdowns, 0.5),
            std,
            p5: percentile(drawdowns, 0.05),
            p25: percentile(drawdowns, 0.25),
            p75: percentile(drawdowns, 0.75),
            p95: percentile(drawdowns, 0.95),
            p99: percentile(drawdowns, 0.99),
            worst,
            best,
        }
    }

    /// Check if drawdown exceeds threshold at given confidence
    #[must_use]
    pub fn exceeds_threshold(&self, threshold: f64, confidence: f64) -> bool {
        let percentile_value = match confidence {
            c if c >= 0.99 => self.p99,
            c if c >= 0.95 => self.p95,
            c if c >= 0.75 => self.p75,
            c if c >= 0.50 => self.median,
            _ => self.p25,
        };
        percentile_value > threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monte_carlo::engine::PathMetadata;

    #[test]
    fn test_max_drawdown_basic() {
        let values = vec![100.0, 110.0, 90.0, 95.0, 85.0, 100.0];
        let max_dd = DrawdownAnalysis::max_drawdown(&values);

        // Peak 110, trough 85: (110-85)/110 = 0.2273
        assert!(
            (max_dd - 0.2273).abs() < 0.01,
            "Max drawdown = {max_dd}, expected ~0.227"
        );
    }

    #[test]
    fn test_max_drawdown_no_drawdown() {
        let values = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let max_dd = DrawdownAnalysis::max_drawdown(&values);

        assert!(max_dd.abs() < 1e-10, "No drawdown for monotonic increase");
    }

    #[test]
    fn test_max_drawdown_complete_loss() {
        let values = vec![100.0, 50.0, 25.0, 10.0, 0.0];
        let max_dd = DrawdownAnalysis::max_drawdown(&values);

        assert!(
            (max_dd - 1.0).abs() < 1e-10,
            "100% drawdown expected: {max_dd}"
        );
    }

    #[test]
    fn test_drawdown_series() {
        let values = vec![100.0, 110.0, 90.0, 100.0];
        let series = DrawdownAnalysis::drawdown_series(&values);

        assert_eq!(series.len(), 4);
        assert!(series[0].abs() < 1e-10); // No drawdown at start
        assert!(series[1].abs() < 1e-10); // New peak
        assert!((series[2] - (110.0 - 90.0) / 110.0).abs() < 1e-10); // ~0.182
        assert!((series[3] - (110.0 - 100.0) / 110.0).abs() < 1e-10); // ~0.091
    }

    #[test]
    fn test_max_drawdown_duration() {
        let values = vec![100.0, 110.0, 90.0, 95.0, 100.0, 110.0, 115.0];
        let duration = DrawdownAnalysis::max_drawdown_duration(&values);

        // In drawdown from index 2 to 5 (3 periods: 90->95->100->110 recovery)
        // Peak=110 at index 1, drawdown starts at index 2, recovers at index 5
        assert_eq!(duration, 3);
    }

    #[test]
    fn test_ulcer_index() {
        let values = vec![100.0, 110.0, 90.0, 100.0, 110.0];
        let ui = DrawdownAnalysis::ulcer_index(&values);

        assert!(ui >= 0.0);
        assert!(ui.is_finite());
    }

    #[test]
    fn test_pain_index() {
        let values = vec![100.0, 110.0, 90.0, 100.0, 110.0];
        let pi = DrawdownAnalysis::pain_index(&values);

        assert!(pi >= 0.0);
        assert!(pi.is_finite());
    }

    #[test]
    fn test_recovery_factor() {
        // 50% return with 20% max drawdown = 2.5 recovery factor
        let values = vec![100.0, 120.0, 100.0, 150.0];
        let rf = DrawdownAnalysis::recovery_factor(&values);

        // Return: (150-100)/100 = 0.5
        // Max DD: (120-100)/120 = 0.167
        // RF = 0.5 / 0.167 ≈ 3.0
        assert!(rf > 2.0 && rf < 4.0, "Recovery factor = {rf}");
    }

    #[test]
    fn test_drawdown_statistics() {
        let drawdowns = vec![0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.12, 0.08, 0.18, 0.22];
        let stats = DrawdownStatistics::from_drawdowns(&drawdowns);

        assert!(stats.mean > 0.0);
        assert!(stats.median > 0.0);
        assert!(stats.std >= 0.0);
        assert!(stats.worst >= stats.p99);
        assert!(stats.best <= stats.p5);
    }

    #[test]
    fn test_from_paths() {
        let paths: Vec<SimulationPath> = (0..100)
            .map(|i| {
                let values = vec![100.0, 105.0, 95.0, 100.0 + (i as f64 * 0.5)];
                SimulationPath::new(
                    vec![0.0, 0.25, 0.5, 1.0],
                    values,
                    PathMetadata {
                        path_id: i,
                        seed: 42,
                        is_antithetic: false,
                    },
                )
            })
            .collect();

        let stats = DrawdownAnalysis::from_paths(&paths);

        assert!(stats.mean > 0.0);
        assert!(stats.mean < 1.0);
    }

    #[test]
    fn test_exceeds_threshold() {
        let drawdowns = vec![0.05, 0.10, 0.15, 0.20, 0.25];
        let stats = DrawdownStatistics::from_drawdowns(&drawdowns);

        // At 95% confidence, check if drawdown exceeds 0.10
        // p95 should be around 0.24
        assert!(stats.exceeds_threshold(0.10, 0.95));
        assert!(!stats.exceeds_threshold(0.30, 0.95));
    }

    #[test]
    fn test_empty_inputs() {
        assert!(DrawdownAnalysis::max_drawdown(&[]).abs() < 1e-10);
        assert!(DrawdownAnalysis::drawdown_series(&[]).is_empty());
        assert_eq!(DrawdownAnalysis::max_drawdown_duration(&[]), 0);
        assert!(DrawdownAnalysis::ulcer_index(&[]).abs() < 1e-10);
    }

    #[test]
    fn test_single_value() {
        let values = vec![100.0];
        assert!(DrawdownAnalysis::max_drawdown(&values).abs() < 1e-10);
        assert_eq!(DrawdownAnalysis::drawdown_series(&values).len(), 1);
    }

    // Property-based tests
    #[cfg(test)]
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_max_drawdown_bounded(values in prop::collection::vec(1.0..1000.0f64, 10..100)) {
                let dd = DrawdownAnalysis::max_drawdown(&values);
                prop_assert!(dd >= 0.0 && dd <= 1.0, "Drawdown must be in [0, 1]: {dd}");
            }

            #[test]
            fn prop_drawdown_series_same_length(values in prop::collection::vec(1.0..1000.0f64, 1..100)) {
                let series = DrawdownAnalysis::drawdown_series(&values);
                prop_assert_eq!(series.len(), values.len());
            }

            #[test]
            fn prop_drawdown_series_non_negative(values in prop::collection::vec(1.0..1000.0f64, 1..100)) {
                let series = DrawdownAnalysis::drawdown_series(&values);
                for dd in series {
                    prop_assert!(dd >= 0.0, "Drawdown must be non-negative: {dd}");
                }
            }

            #[test]
            fn prop_ulcer_index_non_negative(values in prop::collection::vec(1.0..1000.0f64, 10..100)) {
                let ui = DrawdownAnalysis::ulcer_index(&values);
                prop_assert!(ui >= 0.0 && ui.is_finite());
            }

            #[test]
            fn prop_pain_leq_ulcer(values in prop::collection::vec(1.0..1000.0f64, 10..100)) {
                let pi = DrawdownAnalysis::pain_index(&values);
                let ui = DrawdownAnalysis::ulcer_index(&values);
                // Pain (linear mean) <= Ulcer (quadratic mean) by Jensen's inequality
                // With small tolerance for numerical errors
                prop_assert!(pi <= ui + 0.001, "Pain {pi} should be <= Ulcer {ui}");
            }
        }
    }
}
