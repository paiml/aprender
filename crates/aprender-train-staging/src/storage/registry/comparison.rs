//! Version comparison and metric comparison types

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Comparison between two model versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionComparison {
    /// First version
    pub v1: u32,
    /// Second version
    pub v2: u32,
    /// Metric differences (positive = v2 is better for maximizing metrics)
    pub metric_diffs: HashMap<String, f64>,
    /// Whether v2 is better overall
    pub v2_is_better: bool,
    /// Summary of changes
    pub summary: String,
}

/// Metric requirement for promotion policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricRequirement {
    /// Metric name
    pub name: String,
    /// Comparison operator
    pub comparison: Comparison,
    /// Threshold value
    pub threshold: f64,
}

/// Comparison operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Comparison {
    Gt,
    Gte,
    Lt,
    Lte,
    Eq,
}

impl Comparison {
    /// Check if value satisfies comparison with threshold
    pub fn check(&self, value: f64, threshold: f64) -> bool {
        match self {
            Comparison::Gt => value > threshold,
            Comparison::Gte => value >= threshold,
            Comparison::Lt => value < threshold,
            Comparison::Lte => value <= threshold,
            Comparison::Eq => (value - threshold).abs() < f64::EPSILON,
        }
    }

    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            Comparison::Gt => ">",
            Comparison::Gte => ">=",
            Comparison::Lt => "<",
            Comparison::Lte => "<=",
            Comparison::Eq => "==",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comparison_gt() {
        assert!(Comparison::Gt.check(0.96, 0.95));
        assert!(!Comparison::Gt.check(0.95, 0.95));
    }

    #[test]
    fn test_comparison_gte() {
        assert!(Comparison::Gte.check(0.95, 0.95));
        assert!(Comparison::Gte.check(0.96, 0.95));
    }

    #[test]
    fn test_comparison_lt() {
        assert!(Comparison::Lt.check(0.5, 1.0));
        assert!(!Comparison::Lt.check(1.0, 1.0));
    }

    #[test]
    fn test_comparison_eq() {
        assert!(Comparison::Eq.check(0.95, 0.95));
        assert!(!Comparison::Eq.check(0.95, 0.96));
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_comparison_consistent(value in -1000.0f64..1000.0, threshold in -1000.0f64..1000.0) {
            // Gt and Lte are complementary
            let gt = Comparison::Gt.check(value, threshold);
            let lte = Comparison::Lte.check(value, threshold);
            prop_assert!(gt != lte || value == threshold);
        }
    }
}
