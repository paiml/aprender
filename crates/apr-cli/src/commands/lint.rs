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
