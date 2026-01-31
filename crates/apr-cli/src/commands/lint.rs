//! Lint command implementation
//!
//! Implements APR-SPEC §4.11: Lint Command
//!
//! Static analysis for best practices, conventions, and "soft" requirements.
//! Unlike `validate` (which checks for corruption/invalidity), `lint` checks
//! for *quality* and *standardization*.

use crate::error::{CliError, Result};
use aprender::format::{lint_apr_file, LintCategory, LintLevel, LintReport};
use colored::Colorize;
use std::path::Path;

/// Run the lint command
pub(crate) fn run(file: &Path) -> Result<()> {
    // Validate input exists
    if !file.exists() {
        return Err(CliError::FileNotFound(file.to_path_buf()));
    }

    println!("{}", "=== APR Lint ===".cyan().bold());
    println!();
    println!("Checking: {}", file.display());
    println!();

    // Run lint
    let report = lint_apr_file(file).map_err(|e| CliError::ValidationFailed(e.to_string()))?;

    // Display results
    display_report(&report);

    // Return success/failure based on lint result
    if report.passed() {
        Ok(())
    } else {
        Err(CliError::ValidationFailed(format!(
            "Lint failed with {} warning(s) and {} error(s)",
            report.warn_count, report.error_count
        )))
    }
}

/// Format lint level as colored string.
fn format_level(level: LintLevel) -> colored::ColoredString {
    match level {
        LintLevel::Info => format!("[{}]", level.as_str()).blue(),
        LintLevel::Warn => format!("[{}]", level.as_str()).yellow(),
        LintLevel::Error => format!("[{}]", level.as_str()).red(),
    }
}

/// Print a single lint issue.
fn print_issue(issue: &aprender::format::LintIssue, category: LintCategory) {
    let level_str = format_level(issue.level);
    println!("{} {}: {}", level_str, category.name(), issue.message);

    if let Some(ref suggestion) = issue.suggestion {
        println!("       {}", suggestion.dimmed());
    }
}

/// Print issues for a category.
fn print_category_issues(report: &LintReport, category: LintCategory) {
    let issues = report.issues_in_category(category);
    for issue in &issues {
        print_issue(issue, category);
    }
}

/// Print summary and final status.
fn print_summary(report: &LintReport) {
    let total = report.total_issues();
    let summary = format!(
        "Found {} issue(s): {} error(s), {} warning(s), {} info(s)",
        total, report.error_count, report.warn_count, report.info_count
    );

    if report.passed() {
        println!("{}", summary.green());
        println!("{}", "Lint passed (info only)".green().bold());
    } else {
        println!("{}", summary.yellow());
        println!("{}", "Lint failed (has warnings or errors)".red().bold());
    }
}

/// Display lint report
fn display_report(report: &LintReport) {
    if report.issues.is_empty() {
        println!("{}", "No issues found.".green().bold());
        println!();
        return;
    }

    // Print issues grouped by category
    print_category_issues(report, LintCategory::Metadata);
    print_category_issues(report, LintCategory::Naming);
    print_category_issues(report, LintCategory::Efficiency);

    println!();
    print_summary(report);
}

#[cfg(test)]
mod tests {
    use super::*;
    use aprender::format::{LintIssue, LintReport};
    use std::io::Write;
    use tempfile::NamedTempFile;

    // ========================================================================
    // Unit Tests for format_level
    // ========================================================================

    #[test]
    fn test_format_level_info() {
        let formatted = format_level(LintLevel::Info);
        assert!(formatted.to_string().contains("INFO"));
    }

    #[test]
    fn test_format_level_warn() {
        let formatted = format_level(LintLevel::Warn);
        assert!(formatted.to_string().contains("WARN"));
    }

    #[test]
    fn test_format_level_error() {
        let formatted = format_level(LintLevel::Error);
        assert!(formatted.to_string().contains("ERROR"));
    }

    // ========================================================================
    // Unit Tests for print_issue
    // ========================================================================

    #[test]
    fn test_print_issue_basic() {
        let issue = LintIssue::new(
            LintLevel::Warn,
            LintCategory::Metadata,
            "Test warning message",
        );
        // Just verify it doesn't panic
        print_issue(&issue, LintCategory::Metadata);
    }

    #[test]
    fn test_print_issue_with_suggestion() {
        let issue = LintIssue::new(
            LintLevel::Info,
            LintCategory::Naming,
            "Naming convention issue",
        )
        .with_suggestion("Use snake_case");
        // Just verify it doesn't panic
        print_issue(&issue, LintCategory::Naming);
    }

    // ========================================================================
    // Unit Tests for print_category_issues
    // ========================================================================

    #[test]
    fn test_print_category_issues_empty() {
        let report = LintReport::new();
        // Should not panic with empty report
        print_category_issues(&report, LintCategory::Metadata);
    }

    #[test]
    fn test_print_category_issues_with_issues() {
        let mut report = LintReport::new();
        report.add_issue(LintIssue::metadata_warn("Missing license"));
        report.add_issue(LintIssue::metadata_warn("Missing model_card"));
        // Should not panic
        print_category_issues(&report, LintCategory::Metadata);
    }

    // ========================================================================
    // Unit Tests for print_summary
    // ========================================================================

    #[test]
    fn test_print_summary_passed() {
        let report = LintReport::new();
        // Should not panic
        print_summary(&report);
    }

    #[test]
    fn test_print_summary_with_issues() {
        let mut report = LintReport::new();
        report.add_issue(LintIssue::metadata_warn("Test warning"));
        report.add_issue(LintIssue::new(
            LintLevel::Error,
            LintCategory::Metadata,
            "Test error",
        ));
        // Should not panic
        print_summary(&report);
    }

    #[test]
    fn test_print_summary_info_only() {
        let mut report = LintReport::new();
        report.add_issue(LintIssue::efficiency_info("Alignment suggestion"));
        // Should still pass (info only)
        assert!(report.passed());
        print_summary(&report);
    }

    // ========================================================================
    // Unit Tests for display_report
    // ========================================================================

    #[test]
    fn test_display_report_empty() {
        let report = LintReport::new();
        display_report(&report);
    }

    #[test]
    fn test_display_report_with_all_categories() {
        let mut report = LintReport::new();
        report.add_issue(LintIssue::metadata_warn("Missing license"));
        report.add_issue(LintIssue::naming_info("Use full names"));
        report.add_issue(LintIssue::efficiency_info("Consider alignment"));
        display_report(&report);
    }

    // ========================================================================
    // Integration Tests for run()
    // ========================================================================

    #[test]
    fn test_run_file_not_found() {
        let result = run(std::path::Path::new("/nonexistent/model.apr"));
        assert!(result.is_err());
        match result {
            Err(CliError::FileNotFound(path)) => {
                assert!(path.to_string_lossy().contains("nonexistent"));
            }
            _ => panic!("Expected FileNotFound error"),
        }
    }

    #[test]
    fn test_run_invalid_file() {
        // Create a temp file with invalid APR content
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not a valid APR file").expect("write to temp file");

        let result = run(file.path());
        // Should return error since it's not a valid APR file
        assert!(result.is_err());
    }

    // ========================================================================
    // LintReport behavior tests
    // ========================================================================

    #[test]
    fn test_lint_report_counts() {
        let mut report = LintReport::new();
        assert_eq!(report.info_count, 0);
        assert_eq!(report.warn_count, 0);
        assert_eq!(report.error_count, 0);

        report.add_issue(LintIssue::efficiency_info("Info 1"));
        report.add_issue(LintIssue::efficiency_info("Info 2"));
        report.add_issue(LintIssue::metadata_warn("Warn 1"));
        report.add_issue(LintIssue::new(
            LintLevel::Error,
            LintCategory::Naming,
            "Error 1",
        ));

        assert_eq!(report.info_count, 2);
        assert_eq!(report.warn_count, 1);
        assert_eq!(report.error_count, 1);
        assert_eq!(report.total_issues(), 4);
    }

    #[test]
    fn test_lint_report_passed() {
        let mut report = LintReport::new();
        assert!(report.passed());

        // Info only should still pass
        report.add_issue(LintIssue::efficiency_info("Just info"));
        assert!(report.passed());

        // Adding warning should fail
        report.add_issue(LintIssue::metadata_warn("Warning"));
        assert!(!report.passed());
    }

    #[test]
    fn test_lint_report_passed_strict() {
        let mut report = LintReport::new();
        assert!(report.passed_strict());

        // Even info should fail strict
        report.add_issue(LintIssue::efficiency_info("Just info"));
        assert!(!report.passed_strict());
    }

    #[test]
    fn test_lint_report_issues_at_level() {
        let mut report = LintReport::new();
        report.add_issue(LintIssue::efficiency_info("Info 1"));
        report.add_issue(LintIssue::metadata_warn("Warn 1"));
        report.add_issue(LintIssue::efficiency_info("Info 2"));

        let infos = report.issues_at_level(LintLevel::Info);
        assert_eq!(infos.len(), 2);

        let warns = report.issues_at_level(LintLevel::Warn);
        assert_eq!(warns.len(), 1);

        let errors = report.issues_at_level(LintLevel::Error);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_lint_report_issues_in_category() {
        let mut report = LintReport::new();
        report.add_issue(LintIssue::metadata_warn("Meta 1"));
        report.add_issue(LintIssue::naming_info("Name 1"));
        report.add_issue(LintIssue::metadata_warn("Meta 2"));

        let meta_issues = report.issues_in_category(LintCategory::Metadata);
        assert_eq!(meta_issues.len(), 2);

        let naming_issues = report.issues_in_category(LintCategory::Naming);
        assert_eq!(naming_issues.len(), 1);

        let efficiency_issues = report.issues_in_category(LintCategory::Efficiency);
        assert!(efficiency_issues.is_empty());
    }

    // ========================================================================
    // LintIssue tests
    // ========================================================================

    #[test]
    fn test_lint_issue_display() {
        let issue = LintIssue::new(LintLevel::Warn, LintCategory::Metadata, "Missing license");
        let display = format!("{}", issue);
        assert!(display.contains("WARN"));
        assert!(display.contains("Metadata"));
        assert!(display.contains("Missing license"));
    }

    #[test]
    fn test_lint_issue_display_with_suggestion() {
        let issue = LintIssue::new(LintLevel::Info, LintCategory::Naming, "Short name")
            .with_suggestion("Use longer name");
        let display = format!("{}", issue);
        assert!(display.contains("suggestion"));
        assert!(display.contains("Use longer name"));
    }

    #[test]
    fn test_lint_level_display() {
        assert_eq!(format!("{}", LintLevel::Info), "INFO");
        assert_eq!(format!("{}", LintLevel::Warn), "WARN");
        assert_eq!(format!("{}", LintLevel::Error), "ERROR");
    }

    #[test]
    fn test_lint_category_display() {
        assert_eq!(format!("{}", LintCategory::Metadata), "Metadata");
        assert_eq!(format!("{}", LintCategory::Naming), "Tensor Naming");
        assert_eq!(format!("{}", LintCategory::Efficiency), "Efficiency");
    }
}
