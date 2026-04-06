//! Type definitions for drift detection.

use std::collections::HashMap;

/// Statistical test for drift detection
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DriftTest {
    /// Kolmogorov-Smirnov test (continuous features)
    KS { threshold: f64 },
    /// Chi-square test (categorical features)
    ChiSquare { threshold: f64 },
    /// Population Stability Index (standard industry metric)
    PSI { threshold: f64 },
}

impl DriftTest {
    /// Get the name of this test
    pub fn name(&self) -> &'static str {
        match self {
            DriftTest::KS { .. } => "Kolmogorov-Smirnov",
            DriftTest::ChiSquare { .. } => "Chi-Square",
            DriftTest::PSI { .. } => "PSI",
        }
    }

    /// Get the threshold for this test
    pub fn threshold(&self) -> f64 {
        match self {
            DriftTest::KS { threshold }
            | DriftTest::ChiSquare { threshold }
            | DriftTest::PSI { threshold } => *threshold,
        }
    }
}

/// Severity levels for drift
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    /// No drift detected
    None,
    /// Warning: potential drift, log warning and continue
    Warning,
    /// Critical: stop inference or trigger retrain
    Critical,
}

/// Drift detection result
#[derive(Clone, Debug)]
pub struct DriftResult {
    /// Feature name or index
    pub feature: String,
    /// Test used for detection
    pub test: DriftTest,
    /// Test statistic value
    pub statistic: f64,
    /// P-value (for KS and Chi-sq) or PSI value
    pub p_value: f64,
    /// Whether drift was detected
    pub drifted: bool,
    /// Severity of the drift
    pub severity: Severity,
}

/// Summary of drift detection results
#[derive(Debug, Clone)]
pub struct DriftSummary {
    /// Total number of features checked
    pub total_features: usize,
    /// Number of features with detected drift
    pub drifted_features: usize,
    /// Number of warning-level drifts
    pub warnings: usize,
    /// Number of critical-level drifts
    pub critical: usize,
}

impl DriftSummary {
    /// Whether any critical drift was detected
    pub fn has_critical(&self) -> bool {
        self.critical > 0
    }

    /// Whether any drift was detected (warning or critical)
    pub fn has_drift(&self) -> bool {
        self.drifted_features > 0
    }

    /// Percentage of features that drifted
    pub fn drift_percentage(&self) -> f64 {
        if self.total_features == 0 {
            0.0
        } else {
            100.0 * self.drifted_features as f64 / self.total_features as f64
        }
    }
}

/// Callback type for drift events (Andon Cord)
pub type DriftCallback = Box<dyn Fn(&[DriftResult]) + Send + Sync>;

/// Baseline data for categorical features (histogram per feature)
pub type CategoricalBaseline = Vec<HashMap<usize, usize>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drift_test_ks_name() {
        let test = DriftTest::KS { threshold: 0.05 };
        assert_eq!(test.name(), "Kolmogorov-Smirnov");
    }

    #[test]
    fn test_drift_test_chi_square_name() {
        let test = DriftTest::ChiSquare { threshold: 0.05 };
        assert_eq!(test.name(), "Chi-Square");
    }

    #[test]
    fn test_drift_test_psi_name() {
        let test = DriftTest::PSI { threshold: 0.1 };
        assert_eq!(test.name(), "PSI");
    }

    #[test]
    fn test_drift_test_ks_threshold() {
        let test = DriftTest::KS { threshold: 0.05 };
        assert!((test.threshold() - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_drift_test_chi_square_threshold() {
        let test = DriftTest::ChiSquare { threshold: 0.01 };
        assert!((test.threshold() - 0.01).abs() < 1e-9);
    }

    #[test]
    fn test_drift_test_psi_threshold() {
        let test = DriftTest::PSI { threshold: 0.25 };
        assert!((test.threshold() - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_drift_test_clone() {
        let test = DriftTest::KS { threshold: 0.05 };
        let cloned = test;
        assert_eq!(test, cloned);
    }

    #[test]
    fn test_drift_test_debug() {
        let test = DriftTest::KS { threshold: 0.05 };
        let debug_str = format!("{test:?}");
        assert!(debug_str.contains("KS"));
        assert!(debug_str.contains("threshold"));
    }

    #[test]
    fn test_severity_none() {
        let sev = Severity::None;
        assert_eq!(sev, Severity::None);
    }

    #[test]
    fn test_severity_warning() {
        let sev = Severity::Warning;
        assert_eq!(sev, Severity::Warning);
    }

    #[test]
    fn test_severity_critical() {
        let sev = Severity::Critical;
        assert_eq!(sev, Severity::Critical);
    }

    #[test]
    fn test_severity_clone() {
        let sev = Severity::Warning;
        let cloned = sev;
        assert_eq!(sev, cloned);
    }

    #[test]
    fn test_severity_debug() {
        assert_eq!(format!("{:?}", Severity::None), "None");
        assert_eq!(format!("{:?}", Severity::Warning), "Warning");
        assert_eq!(format!("{:?}", Severity::Critical), "Critical");
    }

    #[test]
    fn test_drift_summary_has_critical() {
        let summary =
            DriftSummary { total_features: 10, drifted_features: 3, warnings: 2, critical: 1 };
        assert!(summary.has_critical());

        let no_critical =
            DriftSummary { total_features: 10, drifted_features: 2, warnings: 2, critical: 0 };
        assert!(!no_critical.has_critical());
    }

    #[test]
    fn test_drift_summary_has_drift() {
        let summary =
            DriftSummary { total_features: 10, drifted_features: 3, warnings: 3, critical: 0 };
        assert!(summary.has_drift());

        let no_drift =
            DriftSummary { total_features: 10, drifted_features: 0, warnings: 0, critical: 0 };
        assert!(!no_drift.has_drift());
    }

    #[test]
    fn test_drift_summary_drift_percentage() {
        let summary =
            DriftSummary { total_features: 10, drifted_features: 3, warnings: 2, critical: 1 };
        assert!((summary.drift_percentage() - 30.0).abs() < 1e-9);
    }

    #[test]
    fn test_drift_summary_drift_percentage_zero_features() {
        let summary =
            DriftSummary { total_features: 0, drifted_features: 0, warnings: 0, critical: 0 };
        assert!((summary.drift_percentage() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_drift_result_clone() {
        let result = DriftResult {
            feature: "age".to_string(),
            test: DriftTest::KS { threshold: 0.05 },
            statistic: 0.15,
            p_value: 0.02,
            drifted: true,
            severity: Severity::Warning,
        };
        let cloned = result.clone();
        assert_eq!(result.feature, cloned.feature);
        assert_eq!(result.drifted, cloned.drifted);
    }

    #[test]
    fn test_drift_summary_clone() {
        let summary =
            DriftSummary { total_features: 10, drifted_features: 3, warnings: 2, critical: 1 };
        let cloned = summary.clone();
        assert_eq!(summary.total_features, cloned.total_features);
    }

    #[test]
    fn test_drift_summary_debug() {
        let summary =
            DriftSummary { total_features: 10, drifted_features: 3, warnings: 2, critical: 1 };
        let debug_str = format!("{summary:?}");
        assert!(debug_str.contains("DriftSummary"));
        assert!(debug_str.contains("total_features"));
    }
}
