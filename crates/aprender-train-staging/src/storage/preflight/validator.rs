//! Main Preflight validation system.

use super::{
    CheckMetadata, CheckResult, PreflightCheck, PreflightContext, PreflightError, PreflightResults,
};

/// Preflight validation system
///
/// Runs a series of checks before training to catch common issues early.
#[derive(Debug, Default)]
pub struct Preflight {
    /// List of checks to run
    checks: Vec<PreflightCheck>,
    /// Context for checks
    context: PreflightContext,
}

impl Preflight {
    /// Create a new preflight validator
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with standard data integrity checks
    pub fn standard() -> Self {
        Self::new()
            .add_check(PreflightCheck::no_nan_values())
            .add_check(PreflightCheck::no_inf_values())
            .add_check(PreflightCheck::consistent_dimensions())
            .add_check(PreflightCheck::no_constant_features())
    }

    /// Create with all checks (data + environment)
    pub fn comprehensive() -> Self {
        Self::standard()
            .add_check(PreflightCheck::min_samples(10))
            .add_check(PreflightCheck::min_features(1))
            .add_check(PreflightCheck::disk_space_mb(100))
            .add_check(PreflightCheck::memory_mb(256))
            .add_check(PreflightCheck::gpu_available())
    }

    /// Add a check
    pub fn add_check(mut self, check: PreflightCheck) -> Self {
        self.checks.push(check);
        self
    }

    /// Set context
    pub fn with_context(mut self, context: PreflightContext) -> Self {
        self.context = context;
        self
    }

    /// Get the number of checks
    pub fn check_count(&self) -> usize {
        self.checks.len()
    }

    /// Run all checks
    pub fn run(&self, data: &[Vec<f64>]) -> PreflightResults {
        let mut results = Vec::new();
        let mut passed_count = 0;
        let mut failed_count = 0;
        let mut warning_count = 0;
        let mut skipped_count = 0;
        let mut all_required_passed = true;

        for check in &self.checks {
            let result = check.run(data, &self.context);

            match &result {
                CheckResult::Passed { .. } => passed_count += 1,
                CheckResult::Failed { .. } => {
                    failed_count += 1;
                    if check.required {
                        all_required_passed = false;
                    }
                }
                CheckResult::Warning { .. } => warning_count += 1,
                CheckResult::Skipped { .. } => skipped_count += 1,
            }

            results.push((CheckMetadata::from(check), result));
        }

        PreflightResults::new(
            results,
            all_required_passed,
            passed_count,
            failed_count,
            warning_count,
            skipped_count,
        )
    }

    /// Run checks and return error if any required check fails
    pub fn validate(&self, data: &[Vec<f64>]) -> Result<PreflightResults, PreflightError> {
        let results = self.run(data);

        if results.all_passed() {
            Ok(results)
        } else {
            Err(PreflightError::ValidationFailed {
                checks_failed: results.failed_count(),
                total_checks: self.checks.len(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Preflight Tests
    // =========================================================================

    #[test]
    fn test_preflight_new() {
        let preflight = Preflight::new();
        assert_eq!(preflight.check_count(), 0);
    }

    #[test]
    fn test_preflight_add_check() {
        let preflight = Preflight::new()
            .add_check(PreflightCheck::no_nan_values())
            .add_check(PreflightCheck::no_inf_values());
        assert_eq!(preflight.check_count(), 2);
    }

    #[test]
    fn test_preflight_standard() {
        let preflight = Preflight::standard();
        assert!(preflight.check_count() >= 3);
    }

    #[test]
    fn test_preflight_comprehensive() {
        let preflight = Preflight::comprehensive();
        assert!(preflight.check_count() >= 5);
    }

    #[test]
    fn test_preflight_run_all_pass() {
        let preflight = Preflight::new()
            .add_check(PreflightCheck::no_nan_values())
            .add_check(PreflightCheck::no_inf_values())
            .add_check(PreflightCheck::min_samples(2));

        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let results = preflight.run(&data);

        assert!(results.all_passed());
        assert_eq!(results.passed_count(), 3);
        assert_eq!(results.failed_count(), 0);
    }

    #[test]
    fn test_preflight_run_with_failure() {
        let preflight = Preflight::new()
            .add_check(PreflightCheck::no_nan_values())
            .add_check(PreflightCheck::min_samples(10));

        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let results = preflight.run(&data);

        assert!(!results.all_passed());
        assert_eq!(results.passed_count(), 1);
        assert_eq!(results.failed_count(), 1);
    }

    #[test]
    fn test_preflight_optional_check_doesnt_fail() {
        let preflight = Preflight::new()
            .add_check(PreflightCheck::no_nan_values())
            .add_check(PreflightCheck::no_constant_features()); // Optional

        let data = vec![vec![1.0, 2.0], vec![1.0, 4.0]]; // First column constant
        let results = preflight.run(&data);

        // Should pass because constant features check is optional
        assert!(results.all_passed());
        assert_eq!(results.warning_count(), 1);
    }

    #[test]
    fn test_preflight_validate_success() {
        let preflight = Preflight::new().add_check(PreflightCheck::no_nan_values());
        let data = vec![vec![1.0, 2.0]];
        let result = preflight.validate(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_preflight_validate_failure() {
        let preflight = Preflight::new().add_check(PreflightCheck::min_samples(100));
        let data = vec![vec![1.0, 2.0]];
        let result = preflight.validate(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_preflight_results_failed_checks() {
        let preflight = Preflight::new()
            .add_check(PreflightCheck::no_nan_values())
            .add_check(PreflightCheck::min_samples(10));

        let data = vec![vec![1.0]];
        let results = preflight.run(&data);

        let failed = results.failed_checks();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].0.name, "min_samples");
    }

    #[test]
    fn test_preflight_results_warnings() {
        let preflight = Preflight::new().add_check(PreflightCheck::no_constant_features());

        let data = vec![vec![1.0, 2.0], vec![1.0, 3.0]];
        let results = preflight.run(&data);

        let warnings = results.warnings();
        assert_eq!(warnings.len(), 1);
    }

    #[test]
    fn test_preflight_results_report() {
        let preflight = Preflight::new().add_check(PreflightCheck::no_nan_values());

        let data = vec![vec![1.0, 2.0]];
        let results = preflight.run(&data);
        let report = results.report();

        assert!(report.contains("Preflight Check Results"));
        assert!(report.contains("PASSED"));
        assert!(report.contains("no_nan_values"));
    }

    #[test]
    fn test_preflight_with_context() {
        let ctx = PreflightContext::new().with_min_samples(5);
        let preflight =
            Preflight::new().add_check(PreflightCheck::min_samples(1)).with_context(ctx);

        let data = vec![vec![1.0], vec![2.0], vec![3.0]];
        let results = preflight.run(&data);

        // Context min_samples=5 should override check's default of 1
        assert!(!results.all_passed());
    }

    // =========================================================================
    // Property Tests
    // =========================================================================

    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_preflight_results_counts_consistent(
            n_checks in 1usize..10
        ) {
            let mut preflight = Preflight::new();
            for _ in 0..n_checks {
                preflight = preflight.add_check(PreflightCheck::no_nan_values());
            }

            let data = vec![vec![1.0, 2.0]];
            let results = preflight.run(&data);

            let total = results.passed_count()
                + results.failed_count()
                + results.warning_count()
                + results.skipped_count();

            prop_assert_eq!(total, n_checks);
        }
    }
}
