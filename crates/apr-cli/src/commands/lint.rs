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

/// Display lint report
fn display_report(report: &LintReport) {
    if report.issues.is_empty() {
        println!("{}", "No issues found.".green().bold());
        println!();
        return;
    }

    // Group by category
    for category in [
        LintCategory::Metadata,
        LintCategory::Naming,
        LintCategory::Efficiency,
    ] {
        let issues = report.issues_in_category(category);
        if !issues.is_empty() {
            // Print category issues
            for issue in &issues {
                let level_str = match issue.level {
                    LintLevel::Info => format!("[{}]", issue.level.as_str()).blue(),
                    LintLevel::Warn => format!("[{}]", issue.level.as_str()).yellow(),
                    LintLevel::Error => format!("[{}]", issue.level.as_str()).red(),
                };

                println!("{} {}: {}", level_str, category.name(), issue.message);

                if let Some(ref suggestion) = issue.suggestion {
                    println!("       {}", suggestion.dimmed());
                }
            }
        }
    }

    println!();

    // Summary
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
