//! Diff command implementation (Thin Shim)
//!
//! Compares two model files across APR, GGUF, and SafeTensors formats.
//! This is a thin CLI wrapper around the library's diff functions.
//!
//! Toyota Way: Kaizen - Continuous improvement through comparison.
//!
//! # TOOL-APR-002 Library Extraction
//!
//! All diff logic lives in `aprender::format::diff`. This CLI only handles:
//! - Path validation
//! - Option parsing
//! - Output formatting (text/JSON)

use crate::error::CliError;
use aprender::format::diff::{diff_models, DiffCategory, DiffOptions, DiffReport};
use colored::Colorize;
use serde::Serialize;
use std::path::Path;

// ============================================================================
// Serializable Types (for JSON output)
// ============================================================================

/// Diff entry for JSON output (mirrors library type)
#[derive(Serialize)]
struct DiffEntryJson {
    field: String,
    file1_value: String,
    file2_value: String,
    category: String,
}

/// Diff result for JSON output
#[derive(Serialize)]
struct DiffResultJson {
    file1: String,
    file2: String,
    format1: String,
    format2: String,
    identical: bool,
    difference_count: usize,
    differences: Vec<DiffEntryJson>,
}

impl From<&DiffReport> for DiffResultJson {
    fn from(report: &DiffReport) -> Self {
        Self {
            file1: report.path1.clone(),
            file2: report.path2.clone(),
            format1: report.format1.clone(),
            format2: report.format2.clone(),
            identical: report.is_identical(),
            difference_count: report.diff_count(),
            differences: report
                .differences
                .iter()
                .map(|d| DiffEntryJson {
                    field: d.field.clone(),
                    file1_value: d.value1.clone(),
                    file2_value: d.value2.clone(),
                    category: d.category.name().to_string(),
                })
                .collect(),
        }
    }
}

// ============================================================================
// Main Entry Point
// ============================================================================

/// Run the diff command
///
/// This is a thin shim that delegates to the library's diff functions.
/// All actual logic is in `aprender::format::diff`.
pub(crate) fn run(
    path1: &Path,
    path2: &Path,
    show_weights: bool,
    json_output: bool,
) -> Result<(), CliError> {
    // Validate paths exist
    validate_paths(path1, path2)?;

    // Build options
    let options = DiffOptions::new().with_tensors().with_metadata();

    // Call library function
    let report = diff_models(path1, path2, options)
        .map_err(|e| CliError::InvalidFormat(format!("Failed to diff models: {e}")))?;

    // Output results
    if json_output {
        output_json(&report);
    } else {
        output_text(&report, show_weights);
    }

    Ok(())
}

// ============================================================================
// Path Validation
// ============================================================================

fn validate_paths(path1: &Path, path2: &Path) -> Result<(), CliError> {
    for path in [path1, path2] {
        if !path.exists() {
            return Err(CliError::FileNotFound(path.to_path_buf()));
        }
        if !path.is_file() {
            return Err(CliError::NotAFile(path.to_path_buf()));
        }
    }
    Ok(())
}

// ============================================================================
// Output Functions
// ============================================================================

fn output_json(report: &DiffReport) {
    let json_result = DiffResultJson::from(report);
    if let Ok(json) = serde_json::to_string_pretty(&json_result) {
        println!("{json}");
    }
}

fn output_text(report: &DiffReport, show_weights: bool) {
    println!(
        "Comparing {} vs {}",
        report.path1.cyan(),
        report.path2.cyan()
    );
    println!();

    // Format info
    if !report.same_format() {
        println!(
            "{} Comparing different formats: {} vs {}",
            "NOTE:".yellow(),
            report.format1.white().bold(),
            report.format2.white().bold()
        );
        println!();
    } else {
        println!("Format: {}", report.format1.white().bold());
        println!();
    }

    if report.is_identical() {
        println!(
            "{}",
            "Models are IDENTICAL in structure and metadata"
                .green()
                .bold()
        );
    } else {
        let count = report.diff_count();
        println!("{} {} differences found:", "DIFF:".yellow().bold(), count);
        println!();

        // Group by category
        for category in [
            DiffCategory::Format,
            DiffCategory::Size,
            DiffCategory::Quantization,
            DiffCategory::Metadata,
            DiffCategory::Tensor,
        ] {
            let diffs = report.differences_by_category(category);
            if !diffs.is_empty() {
                println!("  {} ({}):", category.name().white().bold(), diffs.len());
                for diff in diffs {
                    println!(
                        "    {}: {} → {}",
                        diff.field.white(),
                        diff.value1.red(),
                        diff.value2.green()
                    );
                }
                println!();
            }
        }
    }

    if show_weights {
        println!();
        println!(
            "{} Weight comparison: Use --stats for tensor statistics",
            "TIP:".blue()
        );
    }
}

// ============================================================================
// Tests (Minimal - Most logic is tested in library)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::{tempdir, NamedTempFile};

    // ========================================================================
    // Path Validation Tests
    // ========================================================================

    #[test]
    fn test_validate_paths_first_not_found() {
        let file2 = NamedTempFile::new().expect("create file");
        let result = validate_paths(Path::new("/nonexistent/model1.apr"), file2.path());
        assert!(result.is_err());
        match result {
            Err(CliError::FileNotFound(_)) => {}
            _ => panic!("Expected FileNotFound error"),
        }
    }

    #[test]
    fn test_validate_paths_second_not_found() {
        let file1 = NamedTempFile::new().expect("create file");
        let result = validate_paths(file1.path(), Path::new("/nonexistent/model2.apr"));
        assert!(result.is_err());
        match result {
            Err(CliError::FileNotFound(_)) => {}
            _ => panic!("Expected FileNotFound error"),
        }
    }

    #[test]
    fn test_validate_paths_first_is_directory() {
        let dir = tempdir().expect("create dir");
        let file2 = NamedTempFile::new().expect("create file");
        let result = validate_paths(dir.path(), file2.path());
        assert!(result.is_err());
        match result {
            Err(CliError::NotAFile(_)) => {}
            _ => panic!("Expected NotAFile error"),
        }
    }

    #[test]
    fn test_validate_paths_valid() {
        let file1 = NamedTempFile::new().expect("create file");
        let file2 = NamedTempFile::new().expect("create file");
        let result = validate_paths(file1.path(), file2.path());
        assert!(result.is_ok());
    }

    // ========================================================================
    // Run Command Tests
    // ========================================================================

    #[test]
    fn test_run_file_not_found() {
        let file = NamedTempFile::new().expect("create file");
        let result = run(
            Path::new("/nonexistent/model.apr"),
            file.path(),
            false,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_invalid_files() {
        let mut file1 = NamedTempFile::with_suffix(".apr").expect("create file");
        let mut file2 = NamedTempFile::with_suffix(".apr").expect("create file");

        // Write minimal data (less than header size)
        file1.write_all(b"short").expect("write");
        file2.write_all(b"short").expect("write");

        let result = run(file1.path(), file2.path(), false, false);
        // Should fail because files are too small/invalid
        assert!(result.is_err());
    }

    // ========================================================================
    // JSON Output Tests
    // ========================================================================

    #[test]
    fn test_diff_entry_json_serialization() {
        let entry = DiffEntryJson {
            field: "version".to_string(),
            file1_value: "1.0".to_string(),
            file2_value: "2.0".to_string(),
            category: "format".to_string(),
        };
        let json = serde_json::to_string(&entry).expect("serialize");
        assert!(json.contains("version"));
        assert!(json.contains("1.0"));
        assert!(json.contains("2.0"));
    }

    #[test]
    fn test_diff_result_json_serialization() {
        let result = DiffResultJson {
            file1: "model1.apr".to_string(),
            file2: "model2.apr".to_string(),
            format1: "APR".to_string(),
            format2: "APR".to_string(),
            identical: false,
            difference_count: 1,
            differences: vec![DiffEntryJson {
                field: "payload_size".to_string(),
                file1_value: "1000".to_string(),
                file2_value: "2000".to_string(),
                category: "size".to_string(),
            }],
        };
        let json = serde_json::to_string(&result).expect("serialize");
        assert!(json.contains("model1.apr"));
        assert!(json.contains("model2.apr"));
        assert!(json.contains("payload_size"));
    }

    #[test]
    fn test_diff_result_json_identical() {
        let result = DiffResultJson {
            file1: "a.apr".to_string(),
            file2: "b.apr".to_string(),
            format1: "APR".to_string(),
            format2: "APR".to_string(),
            identical: true,
            difference_count: 0,
            differences: vec![],
        };
        let json = serde_json::to_string(&result).expect("serialize");
        assert!(json.contains("\"identical\":true"));
    }

    // ========================================================================
    // DiffReport Conversion Tests
    // ========================================================================

    #[test]
    fn test_diff_report_to_json() {
        use aprender::format::diff::{DiffCategory as LibDiffCategory, DiffEntry, DiffReport};

        let report = DiffReport {
            path1: "a.apr".to_string(),
            path2: "b.apr".to_string(),
            format1: "APR".to_string(),
            format2: "APR".to_string(),
            differences: vec![DiffEntry {
                field: "test".to_string(),
                value1: "x".to_string(),
                value2: "y".to_string(),
                category: LibDiffCategory::Metadata,
            }],
            inspection1: None,
            inspection2: None,
        };

        let json_result = DiffResultJson::from(&report);
        assert_eq!(json_result.file1, "a.apr");
        assert_eq!(json_result.file2, "b.apr");
        assert!(!json_result.identical);
        assert_eq!(json_result.difference_count, 1);
    }
}
