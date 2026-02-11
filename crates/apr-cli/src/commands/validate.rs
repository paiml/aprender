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
    println!("Validating {}...\n", path.display());

    // Detect format via magic bytes (Rosetta Stone dispatch)
    let format = FormatType::from_magic(path)
        .or_else(|_| FormatType::from_extension(path))
        .map_err(|e| CliError::InvalidFormat(format!("Cannot detect format: {e}")))?;

    match format {
        FormatType::Apr => run_apr_validation(path, quality, strict, min_score),
        FormatType::Gguf | FormatType::SafeTensors => run_rosetta_validation(path, format, quality),
    }
}

/// APR validation via 100-point QA checklist (existing path)
fn run_apr_validation(
    path: &Path,
    quality: bool,
    strict: bool,
    min_score: Option<u8>,
) -> Result<(), CliError> {
    let data = fs::read(path)?;
    let mut validator = AprValidator::new();
    let report = validator.validate_bytes(&data);

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
fn run_rosetta_validation(path: &Path, format: FormatType, quality: bool) -> Result<(), CliError> {
    output::header(&format!("Validate: {} (Rosetta Stone)", format));

    let rosetta = RosettaStone::new();
    let report = rosetta
        .validate(path)
        .map_err(|e| CliError::ValidationFailed(format!("Validation failed: {e}")))?;

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
    fn test_quality_assessment_display() {
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

        // Should not panic
        print_quality_assessment(&report);
    }

    #[test]
    fn test_quality_assessment_missing_categories() {
        let report = ValidationReport {
            checks: Vec::new(),
            total_score: 0,
            category_scores: HashMap::new(),
        };

        // Should handle missing categories gracefully (default to 0)
        print_quality_assessment(&report);
    }

    #[test]
    fn test_quality_assessment_all_score_ranges() {
        // High scores
        let mut high_scores = HashMap::new();
        high_scores.insert(Category::Structure, 25);
        high_scores.insert(Category::Physics, 25);
        high_scores.insert(Category::Tooling, 25);
        high_scores.insert(Category::Conversion, 25);

        let high_report = ValidationReport {
            checks: Vec::new(),
            total_score: 100,
            category_scores: high_scores,
        };

        // Low scores
        let mut low_scores = HashMap::new();
        low_scores.insert(Category::Structure, 5);

        let low_report = ValidationReport {
            checks: Vec::new(),
            total_score: 5,
            category_scores: low_scores,
        };

        // All should display without panic
        print_quality_assessment(&high_report);
        print_quality_assessment(&low_report);
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

    // ========================================================================
    // Multi-Format Dispatch Tests (GGUF, SafeTensors)
    // ========================================================================

    #[test]
    fn test_run_gguf_format_dispatch() {
        use aprender::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};

        // Create valid GGUF file with non-zero tensor data
        let floats: Vec<f32> = (0..16).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let data: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        let tensor = GgufTensor {
            name: "model.weight".to_string(),
            shape: vec![4, 4],
            dtype: GgmlType::F32,
            data,
        };
        let metadata = vec![(
            "general.architecture".to_string(),
            GgufValue::String("test".to_string()),
        )];

        let mut gguf_bytes = Vec::new();
        export_tensors_to_gguf(&mut gguf_bytes, &[tensor], &metadata).expect("export GGUF");

        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(&gguf_bytes).expect("write GGUF");

        // Should dispatch to GGUF validation path (RosettaStone::validate)
        let result = run(file.path(), false, false, None);
        // GGUF validation should succeed (physics constraints pass)
        assert!(result.is_ok(), "GGUF format dispatch should work");
    }

    #[test]
    fn test_run_safetensors_format_dispatch() {
        // Create valid SafeTensors file manually
        let header_json = serde_json::json!({
            "test.weight": {
                "dtype": "F32",
                "shape": [2, 2],
                "data_offsets": [0, 16]
            }
        });
        let header_bytes = serde_json::to_vec(&header_json).expect("serialize header");
        let header_len = header_bytes.len() as u64;

        let mut st_bytes = Vec::new();
        st_bytes.extend_from_slice(&header_len.to_le_bytes());
        st_bytes.extend_from_slice(&header_bytes);
        // Add valid tensor data (4 floats = 16 bytes)
        let floats: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        for f in floats {
            st_bytes.extend_from_slice(&f.to_le_bytes());
        }

        let mut file = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
        file.write_all(&st_bytes).expect("write SafeTensors");

        // Should dispatch to SafeTensors validation path (RosettaStone::validate)
        let result = run(file.path(), false, false, None);
        // SafeTensors validation should succeed
        assert!(result.is_ok(), "SafeTensors format dispatch should work");
    }

    #[test]
    fn test_run_gguf_format_detection_by_magic() {
        use aprender::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};

        // Create GGUF with .bin extension (magic detection, not extension)
        // Use valid non-zero tensor data
        let floats: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let tensor_data: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();

        let tensor = GgufTensor {
            name: "test.weight".to_string(),
            shape: vec![2, 2],
            dtype: GgmlType::F32,
            data: tensor_data,
        };
        let metadata = vec![(
            "general.architecture".to_string(),
            GgufValue::String("test".to_string()),
        )];

        let mut gguf_bytes = Vec::new();
        export_tensors_to_gguf(&mut gguf_bytes, &[tensor], &metadata).expect("export GGUF");

        let mut file = NamedTempFile::with_suffix(".bin").expect("create temp file");
        file.write_all(&gguf_bytes).expect("write GGUF");

        // Should detect GGUF by magic bytes, not extension
        let result = run(file.path(), false, false, None);
        assert!(result.is_ok(), "Should detect GGUF by magic bytes");
    }

    #[test]
    fn test_run_gguf_with_physics_violations() {
        use aprender::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};

        // Create GGUF with NaN values (physics violation)
        let nan_f32 = f32::NAN.to_le_bytes();
        let mut tensor_data = Vec::new();
        for _ in 0..4 {
            tensor_data.extend_from_slice(&nan_f32);
        }

        let tensor = GgufTensor {
            name: "model.weight".to_string(),
            shape: vec![2, 2],
            dtype: GgmlType::F32,
            data: tensor_data,
        };
        let metadata = vec![(
            "general.architecture".to_string(),
            GgufValue::String("test".to_string()),
        )];

        let mut gguf_bytes = Vec::new();
        export_tensors_to_gguf(&mut gguf_bytes, &[tensor], &metadata).expect("export GGUF");

        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(&gguf_bytes).expect("write GGUF");

        // Should fail due to NaN physics violation
        let result = run(file.path(), false, false, None);
        assert!(result.is_err(), "Should fail with NaN tensors");
    }
}
