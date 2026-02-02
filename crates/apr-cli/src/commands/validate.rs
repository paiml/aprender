//! Validate command implementation
//!
//! Toyota Way: Jidoka - Build quality in, stop on issues.
//! Validates model integrity using the 100-point QA checklist.

use crate::error::CliError;
use aprender::format::rosetta::{FormatType, RosettaStone};
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
    println!(
        "Format: {} (using Rosetta Stone validation)\n",
        format.to_string().cyan()
    );

    let rosetta = RosettaStone::new();
    let report = rosetta
        .validate(path)
        .map_err(|e| CliError::ValidationFailed(format!("Validation failed: {e}")))?;

    // Print per-tensor results
    for tv in &report.tensors {
        let status = if tv.is_valid {
            "[PASS]".green().to_string()
        } else {
            "[FAIL]".red().to_string()
        };
        println!("  {} {}", status, tv.name);
        for failure in &tv.failures {
            println!("    - {}", failure.red());
        }
    }

    println!();
    println!("{}", report.summary());

    if quality {
        println!();
        println!(
            "{}",
            "=== Physics Constraints (APR-SPEC 10.9) ===".cyan().bold()
        );
        println!("  Total NaN:  {}", report.total_nan_count);
        println!("  Total Inf:  {}", report.total_inf_count);
        println!("  All-zeros:  {}", report.all_zero_tensors.len());
        println!("  Duration:   {} ms", report.duration_ms);
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
