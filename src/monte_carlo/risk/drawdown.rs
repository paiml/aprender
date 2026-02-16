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

#[path = "drawdown_part_02.rs"]
mod drawdown_part_02;
