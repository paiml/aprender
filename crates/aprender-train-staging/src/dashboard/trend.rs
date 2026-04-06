//! Trend analysis for dashboard metrics.

use serde::{Deserialize, Serialize};

/// Trend direction for a metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Trend {
    /// Metric is increasing
    Rising,
    /// Metric is decreasing
    Falling,
    /// Metric is relatively stable
    Stable,
}

impl Trend {
    /// Compute trend from a series of values.
    ///
    /// Uses linear regression slope to determine trend direction.
    pub fn from_values(values: &[f64]) -> Self {
        if values.len() < 2 {
            return Self::Stable;
        }

        // Simple linear regression slope
        let n = values.len() as f64;
        let sum_x: f64 = (0..values.len()).map(|i| i as f64).sum();
        let sum_y: f64 = values.iter().sum();
        let sum_xy: f64 = values.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let sum_x2: f64 = (0..values.len()).map(|i| (i as f64).powi(2)).sum();

        let denominator = n * sum_x2 - sum_x.powi(2);
        if denominator.abs() < f64::EPSILON {
            return Self::Stable;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denominator;

        // Normalize slope by mean to get relative change
        let mean = sum_y / n;
        if mean.abs() < f64::EPSILON {
            return Self::Stable;
        }

        let relative_slope = slope / mean;

        // Thresholds for trend detection (5% relative change)
        const THRESHOLD: f64 = 0.05;

        if relative_slope > THRESHOLD {
            Self::Rising
        } else if relative_slope < -THRESHOLD {
            Self::Falling
        } else {
            Self::Stable
        }
    }

    /// Get emoji representation.
    pub fn emoji(&self) -> &'static str {
        match self {
            Self::Rising => "↑",
            Self::Falling => "↓",
            Self::Stable => "→",
        }
    }
}

impl std::fmt::Display for Trend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rising => write!(f, "rising"),
            Self::Falling => write!(f, "falling"),
            Self::Stable => write!(f, "stable"),
        }
    }
}
