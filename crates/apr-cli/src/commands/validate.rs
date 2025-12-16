//! Validate command implementation
//!
//! Toyota Way: Jidoka - Build quality in, stop on issues.
//! Validates model integrity using the 100-point QA checklist.

use crate::error::CliError;
use aprender::format::validation::{AprValidator, Category, CheckStatus, ValidationReport};
use colored::Colorize;
use std::fs;
use std::path::Path;

/// Run the validate command
pub(crate) fn run(
    path: &Path,
    quality: bool,
    strict: bool,
    min_score: Option<u8>,
) -> Result<(), CliError> {
    validate_path(path)?;
    println!("Validating {}...\n", path.display());

    // Read entire file
    let data = fs::read(path)?;

    // Run validation
    let mut validator = AprValidator::new();
    let report = validator.validate_bytes(&data);

    // Print detailed check results
    print_check_results(report);

    // Print summary
    print_summary(report, strict)?;

    if quality {
        print_quality_assessment(report);
    }

    // Check minimum score if specified
    if let Some(min) = min_score {
        if report.total_score < min {
            return Err(CliError::ValidationFailed(format!(
                "Score {}/100 below minimum {min}",
                report.total_score
            )));
        }
    }

    Ok(())
}

fn validate_path(path: &Path) -> Result<(), CliError> {
    if !path.exists() {
        return Err(CliError::FileNotFound(path.to_path_buf()));
    }
    if !path.is_file() {
        return Err(CliError::NotAFile(path.to_path_buf()));
    }
    Ok(())
}

fn print_check_results(report: &ValidationReport) {
    for check in &report.checks {
        let status_str = match &check.status {
            CheckStatus::Pass => "[PASS]".green().to_string(),
            CheckStatus::Fail(reason) => format!("{} {}", "[FAIL]".red(), reason),
            CheckStatus::Warn(reason) => format!("{} {}", "[WARN]".yellow(), reason),
            CheckStatus::Skip(reason) => format!("{} {}", "[SKIP]".cyan(), reason),
        };

        println!("  {:>3}. {:30} {}", check.id, check.name, status_str);
    }
}

fn print_summary(report: &ValidationReport, _strict: bool) -> Result<(), CliError> {
    println!();

    let failed_checks = report.failed_checks();

    if failed_checks.is_empty() {
        println!(
            "Result: {} ({}/100 points)",
            "VALID".green().bold(),
            report.total_score
        );
        Ok(())
    } else {
        println!(
            "Result: {} ({} checks failed)",
            "INVALID".red().bold(),
            failed_checks.len()
        );
        // Always fail when there are failed checks (corrupted files etc.)
        Err(CliError::ValidationFailed(format!(
            "{} validation checks failed",
            failed_checks.len()
        )))
    }
}

fn print_quality_assessment(report: &ValidationReport) {
    println!();
    println!("{}", "=== 100-Point Quality Assessment ===".cyan().bold());
    println!();

    // Print category scores
    print_category_score(
        report,
        Category::Structure,
        "A. Format & Structural Integrity",
    );
    print_category_score(report, Category::Physics, "B. Tensor Physics & Statistics");
    print_category_score(report, Category::Tooling, "C. Tooling & Operations");
    print_category_score(
        report,
        Category::Conversion,
        "D. Conversion & Interoperability",
    );

    println!();

    // Print total score with grade
    let grade = report.grade();
    let grade_color = match grade {
        "A+" | "A" => grade.green().bold(),
        "B+" | "B" => grade.green(),
        "C+" | "C" => grade.yellow(),
        "D" => grade.yellow().bold(),
        _ => grade.red().bold(),
    };

    println!("TOTAL: {}/100 (Grade: {})", report.total_score, grade_color);

    // Print failed checks summary
    let failed = report.failed_checks();
    if !failed.is_empty() {
        println!();
        println!("{}", "Failed Checks:".red().bold());
        for check in failed {
            if let CheckStatus::Fail(reason) = &check.status {
                println!("  - #{}: {} - {}", check.id, check.name, reason);
            }
        }
    }
}

fn print_category_score(report: &ValidationReport, category: Category, name: &str) {
    let score = report.category_scores.get(&category).copied().unwrap_or(0);
    let max = 25;

    let bar_filled = (score as usize * 20) / max as usize;
    let bar_empty = 20 - bar_filled;
    let bar = format!("[{}{}]", "█".repeat(bar_filled), "░".repeat(bar_empty));

    let color_bar = if score >= 20 {
        bar.green()
    } else if score >= 15 {
        bar.yellow()
    } else {
        bar.red()
    };

    println!("{name:40} {score:>2}/{max} {color_bar}");
}
