//! Validate command implementation
//!
//! Toyota Way: Jidoka - Build quality in, stop on issues.
//! Validates model integrity using the 100-point QA checklist.

use crate::error::CliError;
use crate::output;
use aprender::format::rosetta::{
    FormatType, RosettaStone, ValidationReport as RosettaValidationReport,
};
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
    json: bool,
) -> Result<(), CliError> {
    // BUG-VALIDATE-001 FIX: Validate min_score is in valid range [0, 100]
    if let Some(score) = min_score {
        if score > 100 {
            return Err(CliError::ValidationFailed(format!(
                "Invalid --min-score value: {}. Must be in range 0-100.",
                score
            )));
        }
    }

    validate_path(path)?;
    if !json {
        println!("Validating {}...\n", path.display());
    }

    // Detect format via magic bytes (Rosetta Stone dispatch)
    let format = FormatType::from_magic(path)
        .or_else(|_| FormatType::from_extension(path))
        .map_err(|e| CliError::InvalidFormat(format!("Cannot detect format: {e}")))?;

    match format {
        FormatType::Apr => run_apr_validation(path, quality, strict, min_score, json),
        FormatType::Gguf | FormatType::SafeTensors => {
            run_rosetta_validation(path, format, quality, json)
        }
    }
}

/// APR validation via 100-point QA checklist (existing path)
fn run_apr_validation(
    path: &Path,
    quality: bool,
    strict: bool,
    min_score: Option<u8>,
    json: bool,
) -> Result<(), CliError> {
    let data = fs::read(path)?;
    let mut validator = AprValidator::new();
    let report = validator.validate_bytes(&data);

    if json {
        return print_apr_validation_json(path, report, strict, min_score);
    }

    print_check_results(report);
    print_summary(report, strict)?;

    if quality {
        print_quality_assessment(report);
    }

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

/// GGUF/SafeTensors validation via RosettaStone (physics constraints)
fn run_rosetta_validation(
    path: &Path,
    format: FormatType,
    quality: bool,
    json: bool,
) -> Result<(), CliError> {
    let rosetta = RosettaStone::new();
    let report = rosetta
        .validate(path)
        .map_err(|e| CliError::ValidationFailed(format!("Validation failed: {e}")))?;

    if json {
        return print_rosetta_validation_json(path, &report);
    }

    output::header(&format!("Validate: {} (Rosetta Stone)", format));

    // Print per-tensor results as table
    let mut rows: Vec<Vec<String>> = Vec::new();
    for tv in &report.tensors {
        let badge = if tv.is_valid {
            output::badge_pass("PASS")
        } else {
            output::badge_fail("FAIL")
        };
        let failures_str = if tv.failures.is_empty() {
            String::new()
        } else {
            tv.failures.join("; ")
        };
        rows.push(vec![tv.name.clone(), badge, failures_str]);
    }
    if !rows.is_empty() {
        println!(
            "{}",
            output::table(&["Tensor", "Status", "Failures"], &rows)
        );
    }

    println!();
    println!("{}", report.summary());

    if quality {
        print_quality_constraints(&report);
    }

    if report.is_valid {
        Ok(())
    } else {
        Err(CliError::ValidationFailed(format!(
            "{} tensors failed validation",
            report.failed_tensor_count
        )))
    }
}

/// Print physics constraints and PMAT-235 contract gate breakdown.
fn print_quality_constraints(report: &RosettaValidationReport) {
    println!();
    println!(
        "{}",
        "=== Physics Constraints (APR-SPEC 10.9) ===".cyan().bold()
    );
    println!("  Total NaN:  {}", report.total_nan_count);
    println!("  Total Inf:  {}", report.total_inf_count);
    println!("  All-zeros:  {}", report.all_zero_tensors.len());
    println!("  Duration:   {} ms", report.duration_ms);

    let all_failures: Vec<(&str, &str)> = report
        .tensors
        .iter()
        .flat_map(|t| {
            t.failures
                .iter()
                .map(move |f| (t.name.as_str(), f.as_str()))
        })
        .collect();

    if all_failures.is_empty() {
        println!();
        println!(
            "  {} All tensors pass PMAT-235 contract gates",
            "[OK]".green()
        );
    } else {
        print_contract_violations(&all_failures);
    }
}

/// Print PMAT-235 contract violations grouped by rule ID.
fn print_contract_violations(failures: &[(&str, &str)]) {
    println!();
    println!("{}", "=== PMAT-235 Contract Violations ===".red().bold());
    let mut by_rule: std::collections::BTreeMap<&str, Vec<&str>> =
        std::collections::BTreeMap::new();
    for (tensor_name, failure) in failures {
        let rule_id = if failure.starts_with('[') {
            failure.find(']').map_or("UNKNOWN", |end| &failure[1..end])
        } else {
            "UNKNOWN"
        };
        by_rule.entry(rule_id).or_default().push(tensor_name);
    }
    for (rule, tensors) in &by_rule {
        println!("  {} {} tensor(s) failed", rule.red(), tensors.len());
        for name in tensors.iter().take(5) {
            println!("    - {}", name);
        }
        if tensors.len() > 5 {
            println!("    ... and {} more", tensors.len() - 5);
        }
    }
}

/// Print APR validation report as JSON (GH-240/GH-251: machine-parseable output).
// serde_json::json!() macro uses infallible unwrap internally
#[allow(clippy::disallowed_methods)]
fn print_apr_validation_json(
    path: &Path,
    report: &ValidationReport,
    _strict: bool,
    min_score: Option<u8>,
) -> Result<(), CliError> {
    let passed = report.failed_checks().is_empty()
        && min_score.map_or(true, |min| report.total_score >= min);
    // GH-251: Only include executed checks (PASS/FAIL) — SKIP/WARN are not actionable
    // and cause parity checker false positives
    let checks_json: Vec<serde_json::Value> = report
        .checks
        .iter()
        .filter(|c| matches!(&c.status, CheckStatus::Pass | CheckStatus::Fail(_)))
        .map(|c| {
            let (status, detail) = match &c.status {
                CheckStatus::Pass => ("PASS", String::new()),
                CheckStatus::Fail(r) => ("FAIL", r.clone()),
                CheckStatus::Warn(r) => ("WARN", r.clone()),
                CheckStatus::Skip(r) => ("SKIP", r.clone()),
            };
            serde_json::json!({
                "id": c.id,
                "name": c.name,
                "status": status,
                "detail": detail,
                "points": c.points,
            })
        })
        .collect();
    let output = serde_json::json!({
        "model": path.display().to_string(),
        "format": "apr",
        "total_score": report.total_score,
        "grade": report.grade(),
        "checks": checks_json,
        "total_checks": report.checks.len(),
        "failed": report.failed_checks().len(),
        "passed": passed,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&output).unwrap_or_default()
    );
    if !passed {
        return Err(CliError::ValidationFailed(format!(
            "Score {}/100",
            report.total_score
        )));
    }
    Ok(())
}

/// Print Rosetta validation report as JSON (GH-240/GH-251: machine-parseable output).
// serde_json::json!() macro uses infallible unwrap internally
#[allow(clippy::disallowed_methods)]
fn print_rosetta_validation_json(
    path: &Path,
    report: &RosettaValidationReport,
) -> Result<(), CliError> {
    // GH-251: Include individual tensor checks as a list (same schema as APR path)
    let checks_json: Vec<serde_json::Value> = report
        .tensors
        .iter()
        .map(|tv| {
            let status = if tv.is_valid { "PASS" } else { "FAIL" };
            let detail = if tv.failures.is_empty() {
                String::new()
            } else {
                tv.failures.join("; ")
            };
            serde_json::json!({
                "name": tv.name,
                "status": status,
                "detail": detail,
            })
        })
        .collect();

    let output = serde_json::json!({
        "model": path.display().to_string(),
        "format": "rosetta",
        "total_tensors": report.tensor_count,
        "failed_tensors": report.failed_tensor_count,
        "total_nan": report.total_nan_count,
        "total_inf": report.total_inf_count,
        "duration_ms": report.duration_ms,
        "checks": checks_json,
        "total_checks": report.tensor_count,
        "failed": report.failed_tensor_count,
        "passed": report.is_valid,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&output).unwrap_or_default()
    );
    if !report.is_valid {
        return Err(CliError::ValidationFailed(format!(
            "{} tensors failed validation",
            report.failed_tensor_count
        )));
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
    let mut rows: Vec<Vec<String>> = Vec::new();
    for check in &report.checks {
        let (badge, detail) = match &check.status {
            CheckStatus::Pass => (output::badge_pass("PASS"), String::new()),
            CheckStatus::Fail(reason) => (output::badge_fail("FAIL"), reason.clone()),
            CheckStatus::Warn(reason) => (output::badge_warn("WARN"), reason.clone()),
            CheckStatus::Skip(reason) => (output::badge_skip("SKIP"), reason.clone()),
        };
        rows.push(vec![
            format!("{}", check.id),
            check.name.to_string(),
            badge,
            detail,
        ]);
    }
    println!(
        "{}",
        output::table(&["#", "Check", "Status", "Detail"], &rows)
    );
}

fn print_summary(report: &ValidationReport, _strict: bool) -> Result<(), CliError> {
    println!();

    let failed_checks = report.failed_checks();

    if failed_checks.is_empty() {
        println!(
            "  {} {}/100 points",
            output::badge_pass("VALID"),
            report.total_score
        );
        Ok(())
    } else {
        println!(
            "  {} {} checks failed",
            output::badge_fail("INVALID"),
            failed_checks.len()
        );
        Err(CliError::ValidationFailed(format!(
            "{} validation checks failed",
            failed_checks.len()
        )))
    }
}

fn print_quality_assessment(report: &ValidationReport) {
    output::header("100-Point Quality Assessment");

    // Category score rows as table
    let categories = [
        (Category::Structure, "A. Format & Structural Integrity"),
        (Category::Physics, "B. Tensor Physics & Statistics"),
        (Category::Tooling, "C. Tooling & Operations"),
        (Category::Conversion, "D. Conversion & Interoperability"),
    ];

    let mut rows: Vec<Vec<String>> = Vec::new();
    for (cat, name) in &categories {
        let score = report.category_scores.get(cat).copied().unwrap_or(0);
        let max = 25;
        let bar = output::progress_bar(score as usize, max as usize, 20);
        rows.push(vec![(*name).to_string(), format!("{score}/{max}"), bar]);
    }
    println!(
        "{}",
        output::table(&["Category", "Score", "Progress"], &rows)
    );

    // Total score with grade
    let grade = report.grade();
    println!(
        "\n  TOTAL: {}/100  Grade: {}",
        format!("{}", report.total_score).white().bold(),
        output::grade_color(grade),
    );

    // Print failed checks summary
    let failed = report.failed_checks();
    if !failed.is_empty() {
        output::subheader("Failed Checks");
        for check in failed {
            if let CheckStatus::Fail(reason) = &check.status {
                println!(
                    "  {} #{}: {} - {}",
                    "✗".red().bold(),
                    check.id,
                    check.name,
                    reason.dimmed()
                );
            }
        }
    }
}

#[cfg(test)]
#[path = "validate_tests.rs"]
mod tests;
