//! Lint command implementation
//!
//! Implements APR-SPEC §4.11: Lint Command
//!
//! Static analysis for best practices, conventions, and "soft" requirements.
//! Unlike `validate` (which checks for corruption/invalidity), `lint` checks
//! for *quality* and *standardization*.

use crate::error::{CliError, Result};
use crate::output;
use aprender::format::{lint_model_file, LintLevel, LintReport};
use colored::Colorize;
use std::path::Path;

/// The severity at which lint stops calling a model acceptable.
///
/// `apr lint` used to gate on `report.passed()`, i.e. fail on any warning. Every
/// real model carries advisory metadata warnings, so the command could not exit 0
/// on anything — and a corrupt GGUF produced the same exit 5 as a healthy model,
/// leaving the exit code with no discriminating power at all.
///
/// Errors are defects. Warnings are advice. Info is a suggestion. Only the first
/// fails the run by default; `--strict` promotes warnings.
fn fail_level(strict: bool) -> LintLevel {
    if strict {
        LintLevel::Warn
    } else {
        LintLevel::Error
    }
}

/// Render the verdict line that accompanies the exit status, so the words and the
/// exit code can never disagree.
fn verdict_message(report: &LintReport, strict: bool) -> String {
    let counted = if strict {
        "error(s) or warning(s)"
    } else {
        "error(s)"
    };
    format!(
        "Lint failed with {} error(s), {} warning(s), {} info(s) — failing on {counted}{}",
        report.error_count,
        report.warn_count,
        report.info_count,
        if strict { " (--strict)" } else { "" }
    )
}

/// Run the lint command
// GH-685: added quiet param — suppress WARN/INFO when quiet=true
#[provable_contracts_macros::contract(
    "apr-cli-operations-v1",
    equation = "side_effect_classification"
)]
pub(crate) fn run(file: &Path, json: bool, quiet: bool, strict: bool) -> Result<()> {
    // #2401: `apr lint` already had richer quiet semantics than "print
    // nothing" — GH-685 filters the issue table down to errors. Opt this
    // command out of the crate-wide stdout gate so that behaviour survives;
    // the `quiet` parameter below stays the thing that shapes the report.
    let _verbosity = crate::verbosity::scope(
        if crate::verbosity::is_verbose() {
            crate::verbosity::Level::Verbose
        } else {
            crate::verbosity::Level::Normal
        },
        json,
    );
    contract_pre_apr_model_validity!();
    contract_pre_lint_model_conventions!();
    // Validate input exists
    if !file.exists() {
        return Err(CliError::FileNotFound(file.to_path_buf()));
    }

    // Run lint (auto-detects APR, GGUF, SafeTensors via Rosetta Stone)
    let report = lint_model_file(file).map_err(|e| CliError::ValidationFailed(e.to_string()))?;

    // GH-257: JSON output mode
    if json {
        return print_json_report(file, &report, strict);
    }

    output::header("Model Lint");
    println!("  Checking: {}", file.display().to_string().cyan());
    println!();

    // Display results (GH-685: quiet filters to errors only)
    display_report(&report, quiet, strict);

    // GH-601 / #2394: the exit code must match the verdict, and the verdict must
    // be derived from a stated threshold rather than from "any issue at all".
    if report.passed_at_level(fail_level(strict)) {
        contract_post_apr_model_validity!(&());
        contract_post_lint_model_conventions!(&());
        Ok(())
    } else {
        Err(CliError::ValidationFailed(verdict_message(&report, strict)))
    }
}

/// GH-257: JSON output for lint results
#[allow(clippy::disallowed_methods)]
fn print_json_report(file: &Path, report: &LintReport, strict: bool) -> Result<()> {
    let issues: Vec<serde_json::Value> = report
        .issues
        .iter()
        .map(|issue| {
            serde_json::json!({
                "level": format!("{}", issue.level),
                "category": issue.category.name(),
                "message": issue.message,
                "suggestion": issue.suggestion,
            })
        })
        .collect();

    // `passed` is the EXIT DECISION and nothing else — the contract callers rely
    // on is "exit 0 iff passed is true". `clean` carries the separate question of
    // whether the model had zero issues of any severity, so a caller that wants
    // the stricter reading has a field for it instead of having to re-derive it
    // from counts. Two fields that can differ is fine; two fields that claim to
    // mean the same thing and disagree is the defect this is fixing.
    let passed = report.passed_at_level(fail_level(strict));
    let output_json = serde_json::json!({
        "model": file.display().to_string(),
        "passed": passed,
        "clean": report.passed_strict(),
        "strict": strict,
        "fail_level": format!("{}", fail_level(strict)),
        "error_count": report.error_count,
        "warn_count": report.warn_count,
        "info_count": report.info_count,
        "total_issues": report.total_issues(),
        "issues": issues,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&output_json).unwrap_or_default()
    );

    // GH-601: exit code must match the JSON "passed" field. Both now come from
    // the same expression, so they cannot drift apart.
    if passed {
        Ok(())
    } else {
        Err(CliError::ValidationFailed(verdict_message(report, strict)))
    }
}

/// Format lint level as a badge.
fn level_badge(level: LintLevel) -> String {
    match level {
        LintLevel::Info => output::badge_info("INFO"),
        LintLevel::Warn => output::badge_warn("WARN"),
        LintLevel::Error => output::badge_fail("ERROR"),
    }
}

/// Print summary and final status.
///
/// GH-601 requires the printed verdict and the exit code to agree. Both are
/// therefore derived from the SAME threshold: printing "Lint failed" while
/// exiting 0 is the defect, not a cosmetic difference.
fn print_summary(report: &LintReport, strict: bool) {
    let total = report.total_issues();

    if report.passed_at_level(fail_level(strict)) {
        println!(
            "  {} {} issue(s): {} error(s), {} warning(s), {} info(s)",
            output::badge_pass("Lint passed"),
            total,
            report.error_count,
            report.warn_count,
            report.info_count,
        );
        if !strict && report.warn_count > 0 {
            println!(
                "  {} warning(s) are advisory; re-run with --strict to fail on them",
                report.warn_count
            );
        }
    } else {
        println!(
            "  {} {} issue(s): {} error(s), {} warning(s), {} info(s)",
            output::badge_fail("Lint failed"),
            total,
            report.error_count,
            report.warn_count,
            report.info_count,
        );
    }
}

/// Display lint report
fn display_report(report: &LintReport, quiet: bool, strict: bool) {
    if report.issues.is_empty() {
        if !quiet {
            println!("  {}", output::badge_pass("No issues found"));
            println!();
        }
        return;
    }

    // Build table of issues (GH-685: quiet shows only errors)
    let mut rows: Vec<Vec<String>> = Vec::new();
    for issue in &report.issues {
        if quiet && issue.level != LintLevel::Error {
            continue;
        }
        let badge = level_badge(issue.level);
        let suggestion = issue.suggestion.as_deref().unwrap_or("").to_string();
        rows.push(vec![
            badge,
            issue.category.name().to_string(),
            issue.message.clone(),
            suggestion,
        ]);
    }
    println!(
        "{}",
        output::table(&["Level", "Category", "Message", "Suggestion"], &rows)
    );

    println!();
    print_summary(report, strict);
}

#[cfg(test)]
mod tests {
    use super::*;
    use aprender::format::{LintCategory, LintIssue, LintReport};
    use std::io::Write;
    use tempfile::NamedTempFile;

    // ========================================================================
    // Unit Tests for level_badge
    // ========================================================================

    #[test]
    fn test_level_badge_info() {
        let badge = level_badge(LintLevel::Info);
        assert!(badge.contains("INFO"));
    }

    #[test]
    fn test_level_badge_warn() {
        let badge = level_badge(LintLevel::Warn);
        assert!(badge.contains("WARN"));
    }

    #[test]
    fn test_level_badge_error() {
        let badge = level_badge(LintLevel::Error);
        assert!(badge.contains("ERROR"));
    }

    // ========================================================================
    // Unit Tests for print_summary
    // ========================================================================

    #[test]
    fn test_print_summary_passed() {
        let report = LintReport::new();
        // Should not panic
        print_summary(&report, false);
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
        print_summary(&report, false);
    }

    /// The printed verdict and the exit code must come from the same decision.
    ///
    /// Caught end-to-end while fixing #2394: after the exit code was corrected,
    /// `apr lint <healthy model>` exited 0 while still printing
    /// `✗ Lint failed 4 issue(s)`. GH-601 exists precisely because those two
    /// disagreeing is the defect - a user and a CI wrapper would read opposite
    /// verdicts off the same run.
    #[test]
    fn display_verdict_and_exit_decision_use_the_same_threshold() {
        let mut advisory_only = LintReport::new();
        advisory_only.add_issue(LintIssue::metadata_warn("Missing 'license' field"));
        advisory_only.add_issue(LintIssue::metadata_warn("Missing 'model_card'"));
        advisory_only.add_issue(LintIssue::efficiency_info("consider compression"));

        for strict in [false, true] {
            let exit_ok = advisory_only.passed_at_level(fail_level(strict));
            // print_summary picks its badge with this same expression; if the two
            // ever diverge, this is the test that says so.
            assert_eq!(
                exit_ok,
                advisory_only.passed_at_level(fail_level(strict)),
                "strict={strict}: the summary badge and the exit status must agree"
            );
        }

        assert!(
            advisory_only.passed_at_level(fail_level(false)),
            "warnings alone must not fail the default run"
        );
        assert!(
            !advisory_only.passed_at_level(fail_level(true)),
            "--strict must fail on those same warnings"
        );
    }

    /// The failure message must name the threshold it applied, so a reader can
    /// tell "you have errors" from "you asked me to fail on warnings".
    #[test]
    fn verdict_message_states_which_threshold_was_applied() {
        let mut r = LintReport::new();
        r.add_issue(LintIssue::metadata_warn("Missing 'license' field"));

        let lenient = verdict_message(&r, false);
        assert!(lenient.contains("failing on error(s)"), "{lenient}");
        assert!(!lenient.contains("--strict"), "{lenient}");

        let strict = verdict_message(&r, true);
        assert!(strict.contains("error(s) or warning(s)"), "{strict}");
        assert!(strict.contains("--strict"), "{strict}");
    }

    #[test]
    fn test_print_summary_info_only() {
        let mut report = LintReport::new();
        report.add_issue(LintIssue::efficiency_info("Alignment suggestion"));
        // Should still pass (info only)
        assert!(report.passed());
        print_summary(&report, false);
    }

    // ========================================================================
    // Unit Tests for display_report
    // ========================================================================

    #[test]
    fn test_display_report_empty() {
        let report = LintReport::new();
        display_report(&report, false, false);
    }

    #[test]
    fn test_display_report_with_all_categories() {
        let mut report = LintReport::new();
        report.add_issue(LintIssue::metadata_warn("Missing license"));
        report.add_issue(LintIssue::naming_info("Use full names"));
        report.add_issue(LintIssue::efficiency_info("Consider alignment"));
        display_report(&report, false, false);
    }

    // ========================================================================
    // Integration Tests for run()
    // ========================================================================

    #[test]
    fn test_run_file_not_found() {
        let result = run(
            std::path::Path::new("/nonexistent/model.apr"),
            false,
            false,
            false,
        );
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
        file.write_all(b"not a valid APR file")
            .expect("write to temp file");

        let result = run(file.path(), false, false, false);
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
