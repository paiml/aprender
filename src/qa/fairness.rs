//! Fairness Metrics for Model Quality Assurance
//!
//! Implements EEOC Four-Fifths Rule and other fairness metrics.
//!
//! # References
//! - Feldman et al. (2015) "Certifying and Removing Disparate Impact"
//! - Barocas et al. (2019) "Fairness and Machine Learning"
//!
//! # Toyota Way Alignment
//! - **Respect for People**: Ensuring ML systems respect human dignity

use super::{CategoryScore, QaCategory, QaIssue, Severity, TestResult};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Fairness configuration
#[derive(Debug, Clone)]
pub struct FairnessConfig {
    /// Minimum acceptable disparate impact ratio (EEOC: 0.8)
    pub disparate_impact_threshold: f32,
    /// Maximum acceptable equal opportunity difference
    pub equal_opportunity_threshold: f32,
    /// Maximum acceptable demographic parity difference
    pub demographic_parity_threshold: f32,
    /// Protected attributes to test
    pub protected_attributes: Vec<String>,
}

impl Default for FairnessConfig {
    fn default() -> Self {
        Self {
            disparate_impact_threshold: 0.8, // Four-Fifths Rule
            equal_opportunity_threshold: 0.1,
            demographic_parity_threshold: 0.1,
            protected_attributes: vec!["gender".to_string(), "race".to_string(), "age".to_string()],
        }
    }
}

/// Fairness metrics result
#[derive(Debug, Clone)]
pub struct FairnessMetrics {
    /// Disparate Impact Ratio (should be >= 0.8)
    pub disparate_impact_ratio: f32,
    /// Equal Opportunity Difference (should be <= 0.1)
    pub equal_opportunity_diff: f32,
    /// Demographic Parity Difference
    pub demographic_parity_diff: f32,
    /// Per-group metrics
    pub group_metrics: HashMap<String, GroupMetrics>,
    /// Overall fairness passed
    pub passed: bool,
}

impl FairnessMetrics {
    /// Create new fairness metrics
    #[must_use]
    pub fn new(
        disparate_impact_ratio: f32,
        equal_opportunity_diff: f32,
        demographic_parity_diff: f32,
        config: &FairnessConfig,
    ) -> Self {
        let passed = disparate_impact_ratio >= config.disparate_impact_threshold
            && equal_opportunity_diff <= config.equal_opportunity_threshold
            && demographic_parity_diff <= config.demographic_parity_threshold;

        Self {
            disparate_impact_ratio,
            equal_opportunity_diff,
            demographic_parity_diff,
            group_metrics: HashMap::new(),
            passed,
        }
    }

    /// Add metrics for a specific group
    pub fn add_group(&mut self, group_name: String, metrics: GroupMetrics) {
        self.group_metrics.insert(group_name, metrics);
    }
}

/// Metrics for a specific demographic group
#[derive(Debug, Clone)]
pub struct GroupMetrics {
    /// Group name (e.g., "male", "female")
    pub name: String,
    /// Sample size
    pub count: usize,
    /// Positive prediction rate
    pub positive_rate: f32,
    /// True positive rate (recall)
    pub true_positive_rate: f32,
    /// False positive rate
    pub false_positive_rate: f32,
    /// Accuracy for this group
    pub accuracy: f32,
}

impl GroupMetrics {
    /// Create new group metrics
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        count: usize,
        positive_rate: f32,
        true_positive_rate: f32,
        false_positive_rate: f32,
        accuracy: f32,
    ) -> Self {
        Self {
            name: name.into(),
            count,
            positive_rate,
            true_positive_rate,
            false_positive_rate,
            accuracy,
        }
    }
}

/// Calculate Disparate Impact Ratio
///
/// DIR = (`positive_rate_unprivileged`) / (`positive_rate_privileged`)
///
/// Should be >= 0.8 (Four-Fifths Rule) and <= 1.25
#[must_use]
pub fn calculate_disparate_impact(
    privileged_positive_rate: f32,
    unprivileged_positive_rate: f32,
) -> f32 {
    if privileged_positive_rate == 0.0 {
        if unprivileged_positive_rate == 0.0 {
            return 1.0; // Both zero = equal
        }
        return f32::INFINITY;
    }
    unprivileged_positive_rate / privileged_positive_rate
}

/// Calculate Equal Opportunity Difference
///
/// EOD = `TPR_privileged` - `TPR_unprivileged`
///
/// Should be close to 0 (absolute value <= 0.1)
#[must_use]
pub fn calculate_equal_opportunity_diff(privileged_tpr: f32, unprivileged_tpr: f32) -> f32 {
    (privileged_tpr - unprivileged_tpr).abs()
}

/// Calculate Demographic Parity Difference
///
/// DPD = P(Y=1|privileged) - P(Y=1|unprivileged)
#[must_use]
pub fn calculate_demographic_parity_diff(
    privileged_positive_rate: f32,
    unprivileged_positive_rate: f32,
) -> f32 {
    (privileged_positive_rate - unprivileged_positive_rate).abs()
}

/// Check if a model satisfies the Four-Fifths Rule
#[must_use]
pub fn satisfies_four_fifths_rule(dir: f32) -> bool {
    (0.8_f32..=1.25_f32).contains(&dir)
}

/// Run fairness tests on model predictions
#[must_use] 
pub fn run_fairness_tests(config: &FairnessConfig) -> (CategoryScore, Vec<QaIssue>) {
    let start = Instant::now();
    let mut score = CategoryScore::new(15); // 15 points for fairness
    let mut issues = Vec::new();

    // Test each protected attribute
    for attr in &config.protected_attributes {
        let attr_start = Instant::now();
        let metrics = compute_fairness_for_attribute(attr);

        // Test 1: Disparate Impact Ratio
        if satisfies_four_fifths_rule(metrics.disparate_impact_ratio) {
            score.add_result(TestResult::pass(
                format!("DIR ({attr})"),
                attr_start.elapsed(),
            ));
        } else {
            let msg = format!(
                "DIR = {:.2}, threshold = [{:.2}, {:.2}]",
                metrics.disparate_impact_ratio, config.disparate_impact_threshold, 1.25
            );
            score.add_result(TestResult::fail(
                format!("DIR ({attr})"),
                msg.clone(),
                attr_start.elapsed(),
            ));

            let severity = if metrics.disparate_impact_ratio < 0.5 {
                Severity::Blocker // Severe disparity
            } else {
                Severity::Critical
            };

            issues.push(QaIssue::new(
                QaCategory::Fairness,
                severity,
                format!("Disparate impact violation for '{attr}': {msg}"),
                "Review training data balance, consider reweighting or resampling",
            ));
        }

        // Test 2: Equal Opportunity Difference
        if metrics.equal_opportunity_diff <= config.equal_opportunity_threshold {
            score.add_result(TestResult::pass(
                format!("EOD ({attr})"),
                Duration::from_millis(1),
            ));
        } else {
            let msg = format!(
                "EOD = {:.2}, threshold = {:.2}",
                metrics.equal_opportunity_diff, config.equal_opportunity_threshold
            );
            score.add_result(TestResult::fail(
                format!("EOD ({attr})"),
                msg.clone(),
                Duration::from_millis(1),
            ));
            issues.push(QaIssue::new(
                QaCategory::Fairness,
                Severity::Warning,
                format!("Equal opportunity difference violation for '{attr}': {msg}"),
                "Consider equalized odds post-processing",
            ));
        }
    }

    score.finalize();

    let _elapsed = start.elapsed();
    (score, issues)
}

/// Compute fairness metrics for a single attribute (mock implementation)
fn compute_fairness_for_attribute(attribute: &str) -> FairnessMetrics {
    // Mock implementation - in real code, would compute from predictions
    let (dir, eod, dpd) = match attribute {
        "gender" => (0.85, 0.08, 0.05), // Passes
        "race" => (0.72, 0.15, 0.12),   // Fails DIR and EOD
        "age" => (0.90, 0.05, 0.03),    // Passes
        _ => (0.80, 0.10, 0.10),        // Borderline
    };

    let config = FairnessConfig::default();
    let mut metrics = FairnessMetrics::new(dir, eod, dpd, &config);

    // Add mock group metrics
    metrics.add_group(
        format!("{attribute}_privileged"),
        GroupMetrics::new(
            format!("{attribute}_privileged"),
            500,
            0.6,
            0.75,
            0.15,
            0.85,
        ),
    );
    metrics.add_group(
        format!("{attribute}_unprivileged"),
        GroupMetrics::new(
            format!("{attribute}_unprivileged"),
            300,
            0.6 * dir,
            0.75 - eod,
            0.15 + dpd,
            0.82,
        ),
    );

    metrics
}

/// Domain-specific fairness thresholds
#[derive(Debug, Clone, Copy)]
pub enum FairnessDomain {
    /// Healthcare: Stricter requirements
    Healthcare,
    /// Finance/Credit: ECOA regulated
    Finance,
    /// Employment/HR: EEOC regulated
    Employment,
    /// Criminal Justice: Highest scrutiny
    CriminalJustice,
    /// General: Default thresholds
    General,
}

impl FairnessDomain {
    /// Get recommended thresholds for this domain
    #[must_use]
    pub const fn thresholds(&self) -> (f32, f32, f32) {
        match self {
            Self::Healthcare => (0.85, 0.05, 0.05),
            Self::CriminalJustice => (0.90, 0.03, 0.03),
            // EEOC Four-Fifths Rule applies to Finance, Employment, and General
            Self::Finance | Self::Employment | Self::General => (0.80, 0.10, 0.10),
        }
    }

    /// Get regulatory basis description
    #[must_use]
    pub const fn regulatory_basis(&self) -> &'static str {
        match self {
            Self::Healthcare => "HIPAA, FDA AI/ML guidance",
            Self::Finance => "ECOA, Fair Lending Act",
            Self::Employment => "EEOC Guidelines, Title VII",
            Self::CriminalJustice => "Pretrial Justice Institute standards",
            Self::General => "No specific regulation",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disparate_impact_calculation() {
        // Equal rates
        assert!((calculate_disparate_impact(0.5, 0.5) - 1.0).abs() < 1e-6);

        // Four-Fifths Rule boundary
        assert!((calculate_disparate_impact(0.5, 0.4) - 0.8).abs() < 1e-6);

        // Violation
        assert!(calculate_disparate_impact(0.5, 0.3) < 0.8);

        // Both zero
        assert!((calculate_disparate_impact(0.0, 0.0) - 1.0).abs() < 1e-6);

        // Privileged zero, unprivileged non-zero
        assert!(calculate_disparate_impact(0.0, 0.5).is_infinite());
    }

    #[test]
    fn test_equal_opportunity_diff() {
        assert!((calculate_equal_opportunity_diff(0.8, 0.8) - 0.0).abs() < 1e-6);
        assert!((calculate_equal_opportunity_diff(0.8, 0.7) - 0.1).abs() < 1e-6);
        assert!((calculate_equal_opportunity_diff(0.7, 0.8) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_demographic_parity_diff() {
        assert!((calculate_demographic_parity_diff(0.5, 0.5) - 0.0).abs() < 1e-6);
        assert!((calculate_demographic_parity_diff(0.5, 0.4) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_four_fifths_rule() {
        assert!(satisfies_four_fifths_rule(0.8));
        assert!(satisfies_four_fifths_rule(1.0));
        assert!(satisfies_four_fifths_rule(1.25));
        assert!(!satisfies_four_fifths_rule(0.79));
        assert!(!satisfies_four_fifths_rule(1.26));
        assert!(!satisfies_four_fifths_rule(0.5));
    }

    #[test]
    fn test_fairness_metrics_creation() {
        let config = FairnessConfig::default();
        let metrics = FairnessMetrics::new(0.85, 0.08, 0.05, &config);

        assert!(metrics.passed);
        assert!((metrics.disparate_impact_ratio - 0.85).abs() < 1e-6);
    }

    #[test]
    fn test_fairness_metrics_failing() {
        let config = FairnessConfig::default();
        let metrics = FairnessMetrics::new(0.7, 0.15, 0.05, &config);

        assert!(!metrics.passed); // DIR < 0.8 and EOD > 0.1
    }

    #[test]
    fn test_group_metrics() {
        let metrics = GroupMetrics::new("male", 100, 0.6, 0.8, 0.1, 0.9);

        assert_eq!(metrics.name, "male");
        assert_eq!(metrics.count, 100);
        assert!((metrics.positive_rate - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_fairness_config_default() {
        let config = FairnessConfig::default();

        assert!((config.disparate_impact_threshold - 0.8).abs() < 1e-6);
        assert!((config.equal_opportunity_threshold - 0.1).abs() < 1e-6);
        assert!(config.protected_attributes.contains(&"gender".to_string()));
    }

    #[test]
    fn test_run_fairness_tests() {
        let config = FairnessConfig::default();
        let (score, issues) = run_fairness_tests(&config);

        assert_eq!(score.points_possible, 15);
        assert!(score.tests_passed + score.tests_failed > 0);
        // Race attribute should fail in mock
        assert!(!issues.is_empty());
    }

    #[test]
    fn test_fairness_domain_thresholds() {
        let (dir, eod, dpd) = FairnessDomain::CriminalJustice.thresholds();
        assert!((dir - 0.90).abs() < 1e-6);
        assert!((eod - 0.03).abs() < 1e-6);
        assert!((dpd - 0.03).abs() < 1e-6);

        let (dir, eod, _dpd) = FairnessDomain::Employment.thresholds();
        assert!((dir - 0.80).abs() < 1e-6);
        assert!((eod - 0.10).abs() < 1e-6);
    }

    #[test]
    fn test_fairness_domain_regulatory_basis() {
        assert!(FairnessDomain::Employment
            .regulatory_basis()
            .contains("EEOC"));
        assert!(FairnessDomain::Finance.regulatory_basis().contains("ECOA"));
        assert!(FairnessDomain::Healthcare
            .regulatory_basis()
            .contains("FDA"));
    }
}
