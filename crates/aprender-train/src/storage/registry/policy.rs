//! Promotion policies for stage transitions (Poka-yoke)

use serde::{Deserialize, Serialize};

use super::comparison::{Comparison, MetricRequirement};
use super::stage::ModelStage;
use super::version::ModelVersion;

/// Promotion policy for stage transitions (Poka-yoke)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromotionPolicy {
    /// Required metrics with thresholds
    pub required_metrics: Vec<MetricRequirement>,
    /// Minimum test coverage
    pub min_test_coverage: Option<f64>,
    /// Required number of approvals
    pub required_approvals: u32,
    /// Auto-promote if all requirements pass
    pub auto_promote_on_pass: bool,
    /// Target stage this policy applies to
    pub target_stage: ModelStage,
}

impl PromotionPolicy {
    /// Create a new promotion policy for a target stage
    pub fn new(target_stage: ModelStage) -> Self {
        Self {
            required_metrics: Vec::new(),
            min_test_coverage: None,
            required_approvals: 0,
            auto_promote_on_pass: false,
            target_stage,
        }
    }

    /// Add a metric requirement
    pub fn require_metric(mut self, name: &str, comparison: Comparison, threshold: f64) -> Self {
        self.required_metrics.push(MetricRequirement {
            name: name.to_string(),
            comparison,
            threshold,
        });
        self
    }

    /// Set minimum test coverage
    pub fn require_coverage(mut self, coverage: f64) -> Self {
        self.min_test_coverage = Some(coverage);
        self
    }

    /// Set required approvals
    pub fn require_approvals(mut self, count: u32) -> Self {
        self.required_approvals = count;
        self
    }

    /// Enable auto-promotion
    pub fn auto_promote(mut self) -> Self {
        self.auto_promote_on_pass = true;
        self
    }

    /// Check if a model version meets the policy requirements
    pub fn check(&self, model: &ModelVersion, approvals: u32) -> PolicyCheckResult {
        let mut failed_requirements = Vec::new();

        // Check metrics
        for req in &self.required_metrics {
            if let Some(&value) = model.metrics.get(&req.name) {
                if !req.comparison.check(value, req.threshold) {
                    failed_requirements.push(format!(
                        "Metric '{}' = {} does not satisfy {} {}",
                        req.name,
                        value,
                        req.comparison.as_str(),
                        req.threshold
                    ));
                }
            } else {
                failed_requirements.push(format!("Missing required metric '{}'", req.name));
            }
        }

        // Check test coverage
        if let Some(min_coverage) = self.min_test_coverage {
            if let Some(&coverage) = model.metrics.get("test_coverage") {
                if coverage < min_coverage {
                    failed_requirements
                        .push(format!("Test coverage {coverage} < required {min_coverage}"));
                }
            } else {
                failed_requirements.push("Missing test_coverage metric".to_string());
            }
        }

        // Check approvals
        if approvals < self.required_approvals {
            failed_requirements
                .push(format!("Approvals {} < required {}", approvals, self.required_approvals));
        }

        PolicyCheckResult { passed: failed_requirements.is_empty(), failed_requirements }
    }
}

/// Result of policy check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyCheckResult {
    /// Whether all requirements passed
    pub passed: bool,
    /// List of failed requirements
    pub failed_requirements: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_promotion_policy_new() {
        let policy = PromotionPolicy::new(ModelStage::Production);
        assert_eq!(policy.target_stage, ModelStage::Production);
        assert!(policy.required_metrics.is_empty());
    }

    #[test]
    fn test_promotion_policy_require_metric() {
        let policy = PromotionPolicy::new(ModelStage::Production).require_metric(
            "accuracy",
            Comparison::Gte,
            0.95,
        );

        assert_eq!(policy.required_metrics.len(), 1);
        assert_eq!(policy.required_metrics[0].name, "accuracy");
    }

    #[test]
    fn test_promotion_policy_check_pass() {
        let policy = PromotionPolicy::new(ModelStage::Production).require_metric(
            "accuracy",
            Comparison::Gte,
            0.95,
        );

        let model = ModelVersion::new("test", 1, "/path").with_metric("accuracy", 0.96);

        let result = policy.check(&model, 0);
        assert!(result.passed);
    }

    #[test]
    fn test_promotion_policy_check_fail_metric() {
        let policy = PromotionPolicy::new(ModelStage::Production).require_metric(
            "accuracy",
            Comparison::Gte,
            0.95,
        );

        let model = ModelVersion::new("test", 1, "/path").with_metric("accuracy", 0.90);

        let result = policy.check(&model, 0);
        assert!(!result.passed);
        assert!(!result.failed_requirements.is_empty());
    }

    #[test]
    fn test_promotion_policy_check_fail_missing_metric() {
        let policy = PromotionPolicy::new(ModelStage::Production).require_metric(
            "accuracy",
            Comparison::Gte,
            0.95,
        );

        let model = ModelVersion::new("test", 1, "/path");

        let result = policy.check(&model, 0);
        assert!(!result.passed);
        assert!(result.failed_requirements[0].contains("Missing"));
    }

    #[test]
    fn test_promotion_policy_check_approvals() {
        let policy = PromotionPolicy::new(ModelStage::Production).require_approvals(2);

        let model = ModelVersion::new("test", 1, "/path");

        // Not enough approvals
        let result = policy.check(&model, 1);
        assert!(!result.passed);

        // Enough approvals
        let result = policy.check(&model, 2);
        assert!(result.passed);
    }

    #[test]
    fn test_promotion_policy_check_coverage() {
        let policy = PromotionPolicy::new(ModelStage::Production).require_coverage(0.90);

        let model = ModelVersion::new("test", 1, "/path").with_metric("test_coverage", 0.85);

        let result = policy.check(&model, 0);
        assert!(!result.passed);
        assert!(result.failed_requirements[0].contains("coverage"));
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_policy_check_deterministic(
            accuracy in 0.0f64..1.0,
            threshold in 0.0f64..1.0,
            approvals in 0u32..10,
            required_approvals in 0u32..10
        ) {
            let policy = PromotionPolicy::new(ModelStage::Production)
                .require_metric("accuracy", Comparison::Gte, threshold)
                .require_approvals(required_approvals);

            let model = ModelVersion::new("test", 1, "/path")
                .with_metric("accuracy", accuracy);

            let result1 = policy.check(&model, approvals);
            let result2 = policy.check(&model, approvals);

            prop_assert_eq!(result1.passed, result2.passed);
        }
    }
}
