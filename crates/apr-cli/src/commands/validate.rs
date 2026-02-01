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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::io::Write;
    use tempfile::{tempdir, NamedTempFile};

    // ========================================================================
    // Path Validation Tests
    // ========================================================================

    #[test]
    fn test_validate_path_not_found() {
        let result = validate_path(Path::new("/nonexistent/model.apr"));
        assert!(result.is_err());
        match result {
            Err(CliError::FileNotFound(_)) => {}
            _ => panic!("Expected FileNotFound error"),
        }
    }

    #[test]
    fn test_validate_path_is_directory() {
        let dir = tempdir().expect("create temp dir");
        let result = validate_path(dir.path());
        assert!(result.is_err());
        match result {
            Err(CliError::NotAFile(_)) => {}
            _ => panic!("Expected NotAFile error"),
        }
    }

    #[test]
    fn test_validate_path_valid_file() {
        let file = NamedTempFile::new().expect("create temp file");
        let result = validate_path(file.path());
        assert!(result.is_ok());
    }

    // ========================================================================
    // Run Command Tests
    // ========================================================================

    #[test]
    fn test_run_file_not_found() {
        let result = run(Path::new("/nonexistent/model.apr"), false, false, None);
        assert!(result.is_err());
        match result {
            Err(CliError::FileNotFound(_)) => {}
            _ => panic!("Expected FileNotFound error"),
        }
    }

    #[test]
    fn test_run_is_directory() {
        let dir = tempdir().expect("create temp dir");
        let result = run(dir.path(), false, false, None);
        assert!(result.is_err());
        match result {
            Err(CliError::NotAFile(_)) => {}
            _ => panic!("Expected NotAFile error"),
        }
    }

    #[test]
    fn test_run_invalid_file() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not a valid APR file").expect("write");

        let result = run(file.path(), false, false, None);
        // Should fail validation because file is not valid APR
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_quality_flag() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"invalid data").expect("write");

        let result = run(file.path(), true, false, None);
        // Should fail but quality flag is handled
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_min_score() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"invalid data").expect("write");

        let result = run(file.path(), false, false, Some(100));
        // Should fail before min_score check because file is invalid
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_strict_flag() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"test data").expect("write");

        let result = run(file.path(), false, true, None);
        // Should fail with strict mode
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_all_flags() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"test data").expect("write");

        let result = run(file.path(), true, true, Some(50));
        // Should fail with all flags enabled
        assert!(result.is_err());
    }

    #[test]
    fn test_run_empty_file() {
        let file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        // Empty file - no write

        let result = run(file.path(), false, false, None);
        // Empty file should fail validation
        assert!(result.is_err());
    }

    // ========================================================================
    // Category Score Tests (using mocked reports via AprValidator)
    // ========================================================================

    #[test]
    fn test_category_score_display() {
        // Test that category score display doesn't panic
        let mut category_scores = HashMap::new();
        category_scores.insert(Category::Structure, 25);
        category_scores.insert(Category::Physics, 20);
        category_scores.insert(Category::Tooling, 15);
        category_scores.insert(Category::Conversion, 10);

        let report = ValidationReport {
            checks: Vec::new(),
            total_score: 70,
            category_scores,
        };

        // These functions should not panic
        print_category_score(&report, Category::Structure, "A. Format");
        print_category_score(&report, Category::Physics, "B. Physics");
        print_category_score(&report, Category::Tooling, "C. Tooling");
        print_category_score(&report, Category::Conversion, "D. Conversion");
    }

    #[test]
    fn test_category_score_missing() {
        let report = ValidationReport {
            checks: Vec::new(),
            total_score: 0,
            category_scores: HashMap::new(),
        };

        // Should handle missing category gracefully (default to 0)
        print_category_score(&report, Category::Structure, "A. Structure");
    }

    #[test]
    fn test_category_score_colors() {
        // Test all color thresholds
        let mut high_scores = HashMap::new();
        high_scores.insert(Category::Structure, 25); // Green

        let mut mid_scores = HashMap::new();
        mid_scores.insert(Category::Structure, 17); // Yellow

        let mut low_scores = HashMap::new();
        low_scores.insert(Category::Structure, 5); // Red

        let high_report = ValidationReport {
            checks: Vec::new(),
            total_score: 25,
            category_scores: high_scores,
        };

        let mid_report = ValidationReport {
            checks: Vec::new(),
            total_score: 17,
            category_scores: mid_scores,
        };

        let low_report = ValidationReport {
            checks: Vec::new(),
            total_score: 5,
            category_scores: low_scores,
        };

        // All should display without panic
        print_category_score(&high_report, Category::Structure, "High");
        print_category_score(&mid_report, Category::Structure, "Medium");
        print_category_score(&low_report, Category::Structure, "Low");
    }

    // ========================================================================
    // Print Summary Tests
    // ========================================================================

    #[test]
    fn test_print_summary_valid_report() {
        let report = ValidationReport {
            checks: Vec::new(), // No failed checks
            total_score: 100,
            category_scores: HashMap::new(),
        };

        let result = print_summary(&report, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_print_quality_assessment_empty() {
        let report = ValidationReport {
            checks: Vec::new(),
            total_score: 0,
            category_scores: HashMap::new(),
        };

        // Should not panic even with empty report
        print_quality_assessment(&report);
    }
}
