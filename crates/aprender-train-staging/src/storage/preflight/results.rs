//! Preflight validation results.

use std::borrow::Cow;

use super::{CheckMetadata, CheckResult};

/// Results from running preflight checks
#[derive(Debug, Clone)]
pub struct PreflightResults {
    /// Individual check results
    results: Vec<(CheckMetadata, CheckResult)>,
    /// Overall pass/fail
    passed: bool,
    /// Number of checks that passed
    passed_count: usize,
    /// Number of checks that failed
    failed_count: usize,
    /// Number of warnings
    warning_count: usize,
    /// Number of skipped checks
    skipped_count: usize,
}

impl PreflightResults {
    /// Create a new PreflightResults instance
    pub(crate) fn new(
        results: Vec<(CheckMetadata, CheckResult)>,
        passed: bool,
        passed_count: usize,
        failed_count: usize,
        warning_count: usize,
        skipped_count: usize,
    ) -> Self {
        Self { results, passed, passed_count, failed_count, warning_count, skipped_count }
    }

    /// Check if all required checks passed
    pub fn all_passed(&self) -> bool {
        self.passed
    }

    /// Get number of passed checks
    pub fn passed_count(&self) -> usize {
        self.passed_count
    }

    /// Get number of failed checks
    pub fn failed_count(&self) -> usize {
        self.failed_count
    }

    /// Get number of warnings
    pub fn warning_count(&self) -> usize {
        self.warning_count
    }

    /// Get number of skipped checks
    pub fn skipped_count(&self) -> usize {
        self.skipped_count
    }

    /// Get all results
    pub fn results(&self) -> &[(CheckMetadata, CheckResult)] {
        &self.results
    }

    /// Get failed checks only
    pub fn failed_checks(&self) -> Vec<(&CheckMetadata, &CheckResult)> {
        self.results
            .iter()
            .filter(|(check, result)| check.required && result.is_failed())
            .map(|(c, r)| (c, r))
            .collect()
    }

    /// Get warnings only
    pub fn warnings(&self) -> Vec<(&CheckMetadata, &CheckResult)> {
        self.results.iter().filter(|(_, result)| result.is_warning()).map(|(c, r)| (c, r)).collect()
    }

    /// Format results as a report
    pub fn report(&self) -> String {
        let mut lines = Vec::new();
        lines.push("=== Preflight Check Results ===".to_string());
        lines.push(format!("Status: {}", if self.passed { "PASSED" } else { "FAILED" }));
        lines.push(format!(
            "Passed: {}, Failed: {}, Warnings: {}, Skipped: {}",
            self.passed_count, self.failed_count, self.warning_count, self.skipped_count
        ));
        lines.push(String::new());

        for (check, result) in &self.results {
            let status = match result {
                CheckResult::Passed { .. } => "✓",
                CheckResult::Failed { .. } => "✗",
                CheckResult::Warning { .. } => "⚠",
                CheckResult::Skipped { .. } => "○",
            };

            let message: Cow<'_, str> = match result {
                CheckResult::Passed { message } => Cow::Borrowed(message),
                CheckResult::Failed { message, details } => {
                    if let Some(d) = details {
                        Cow::Owned(format!("{message} ({d})"))
                    } else {
                        Cow::Borrowed(message)
                    }
                }
                CheckResult::Warning { message } => Cow::Borrowed(message),
                CheckResult::Skipped { reason } => Cow::Borrowed(reason),
            };

            lines.push(format!("{status} {}: {message}", check.name));
        }

        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::preflight::types::CheckType;

    fn make_check(name: &str, required: bool) -> CheckMetadata {
        CheckMetadata {
            name: name.to_string(),
            check_type: CheckType::Configuration,
            description: format!("{name} description"),
            required,
        }
    }

    #[test]
    fn test_preflight_results_new() {
        let results = PreflightResults::new(vec![], true, 5, 0, 1, 2);
        assert!(results.all_passed());
        assert_eq!(results.passed_count(), 5);
        assert_eq!(results.failed_count(), 0);
        assert_eq!(results.warning_count(), 1);
        assert_eq!(results.skipped_count(), 2);
    }

    #[test]
    fn test_preflight_results_all_passed_true() {
        let results = PreflightResults::new(vec![], true, 3, 0, 0, 0);
        assert!(results.all_passed());
    }

    #[test]
    fn test_preflight_results_all_passed_false() {
        let results = PreflightResults::new(vec![], false, 2, 1, 0, 0);
        assert!(!results.all_passed());
    }

    #[test]
    fn test_preflight_results_results_accessor() {
        let check = make_check("test", true);
        let result = CheckResult::Passed { message: "ok".to_string() };
        let results = PreflightResults::new(vec![(check, result)], true, 1, 0, 0, 0);
        assert_eq!(results.results().len(), 1);
    }

    #[test]
    fn test_preflight_results_failed_checks() {
        let check1 = make_check("pass", true);
        let result1 = CheckResult::Passed { message: "ok".to_string() };
        let check2 = make_check("fail", true);
        let result2 = CheckResult::Failed { message: "error".to_string(), details: None };
        let check3 = make_check("optional_fail", false);
        let result3 = CheckResult::Failed { message: "not required".to_string(), details: None };

        let results = PreflightResults::new(
            vec![(check1, result1), (check2, result2), (check3, result3)],
            false,
            1,
            2,
            0,
            0,
        );

        let failed = results.failed_checks();
        // Only required failed checks
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].0.name, "fail");
    }

    #[test]
    fn test_preflight_results_warnings() {
        let check1 = make_check("pass", true);
        let result1 = CheckResult::Passed { message: "ok".to_string() };
        let check2 = make_check("warn", false);
        let result2 = CheckResult::Warning { message: "heads up".to_string() };

        let results =
            PreflightResults::new(vec![(check1, result1), (check2, result2)], true, 1, 0, 1, 0);

        let warnings = results.warnings();
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].0.name, "warn");
    }

    #[test]
    fn test_preflight_results_report_passed() {
        let check = make_check("test_check", true);
        let result = CheckResult::Passed { message: "All good".to_string() };
        let results = PreflightResults::new(vec![(check, result)], true, 1, 0, 0, 0);

        let report = results.report();
        assert!(report.contains("PASSED"));
        assert!(report.contains("test_check"));
        assert!(report.contains("All good"));
        assert!(report.contains("✓"));
    }

    #[test]
    fn test_preflight_results_report_failed() {
        let check = make_check("failing_check", true);
        let result = CheckResult::Failed {
            message: "Something went wrong".to_string(),
            details: Some("extra info".to_string()),
        };
        let results = PreflightResults::new(vec![(check, result)], false, 0, 1, 0, 0);

        let report = results.report();
        assert!(report.contains("FAILED"));
        assert!(report.contains("failing_check"));
        assert!(report.contains("Something went wrong"));
        assert!(report.contains("extra info"));
        assert!(report.contains("✗"));
    }

    #[test]
    fn test_preflight_results_report_warning() {
        let check = make_check("warn_check", false);
        let result = CheckResult::Warning { message: "Be careful".to_string() };
        let results = PreflightResults::new(vec![(check, result)], true, 0, 0, 1, 0);

        let report = results.report();
        assert!(report.contains("warn_check"));
        assert!(report.contains("Be careful"));
        assert!(report.contains("⚠"));
    }

    #[test]
    fn test_preflight_results_report_skipped() {
        let check = make_check("skipped_check", false);
        let result = CheckResult::Skipped { reason: "Not applicable".to_string() };
        let results = PreflightResults::new(vec![(check, result)], true, 0, 0, 0, 1);

        let report = results.report();
        assert!(report.contains("skipped_check"));
        assert!(report.contains("Not applicable"));
        assert!(report.contains("○"));
    }

    #[test]
    fn test_preflight_results_clone() {
        let results = PreflightResults::new(vec![], true, 1, 0, 0, 0);
        let cloned = results.clone();
        assert_eq!(results.passed_count(), cloned.passed_count());
    }

    #[test]
    fn test_preflight_results_debug() {
        let results = PreflightResults::new(vec![], true, 1, 0, 0, 0);
        let debug = format!("{results:?}");
        assert!(debug.contains("PreflightResults"));
    }
}
