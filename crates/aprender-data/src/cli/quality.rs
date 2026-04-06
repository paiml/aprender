//! Quality checking CLI commands.

use std::path::{Path, PathBuf};

use clap::Subcommand;

use super::basic::load_dataset;
use crate::quality::{ColumnQuality, QualityChecker};

/// Quality checking commands.
#[derive(Subcommand)]
pub enum QualityCommands {
    /// Check data quality of a dataset
    Check {
        /// Path to dataset file
        path: PathBuf,
        /// Null ratio threshold (0.0 to 1.0)
        #[arg(long, default_value = "0.1")]
        null_threshold: f64,
        /// Duplicate ratio threshold (0.0 to 1.0)
        #[arg(long, default_value = "0.05")]
        duplicate_threshold: f64,
        /// Enable outlier detection
        #[arg(long, default_value = "true")]
        detect_outliers: bool,
        /// Output format (text, json)
        #[arg(short, long, default_value = "text")]
        format: String,
    },
    /// Generate a quality report
    Report {
        /// Path to dataset file
        path: PathBuf,
        /// Output file for the report (JSON format)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Calculate 100-point quality score with letter grade (GH-6)
    Score {
        /// Path to dataset file
        path: PathBuf,
        /// Quality profile to use (default, doctest-corpus, ml-training,
        /// time-series)
        #[arg(short, long, default_value = "default")]
        profile: String,
        /// Show improvement suggestions for failed checks
        #[arg(long)]
        suggest: bool,
        /// Output as JSON for CI/CD integration
        #[arg(long)]
        json: bool,
        /// Output badge URL for shields.io
        #[arg(long)]
        badge: bool,
    },
    /// List available quality profiles
    Profiles,
}

/// Check data quality of a dataset.
pub(crate) fn cmd_quality_check(
    path: &Path,
    null_threshold: f64,
    duplicate_threshold: f64,
    detect_outliers: bool,
    format: &str,
) -> crate::Result<()> {
    let dataset = load_dataset(path)?;

    // GH-38: --duplicate-threshold is not yet used by QualityChecker
    if (duplicate_threshold - 0.05_f64).abs() > f64::EPSILON {
        eprintln!(
            "Warning: --duplicate-threshold {duplicate_threshold} is not yet implemented. Using default behavior."
        );
    }
    let _ = duplicate_threshold;
    let mut checker = QualityChecker::new();

    if !detect_outliers {
        checker = checker.with_outlier_check(false);
    }

    let report = checker.check(&dataset)?;

    if format == "json" {
        let json = serde_json::json!({
            "path": path.display().to_string(),
            "rows": report.row_count,
            "columns": report.column_count,
            "has_issues": !report.issues.is_empty(),
            "score": report.score,
            "issues": report.issues.iter().map(|i| format!("{:?}", i)).collect::<Vec<_>>(),
            "column_qualities": report.columns.iter().map(|(name, c)| {
                serde_json::json!({
                    "column": name,
                    "null_ratio": c.null_ratio,
                    "unique_count": c.unique_count,
                    "is_constant": c.is_constant(),
                    "is_mostly_null": c.null_ratio > null_threshold,
                })
            }).collect::<Vec<_>>()
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).map_err(|e| crate::Error::Format(e.to_string()))?
        );
    } else {
        println!("Data Quality Report");
        println!("===================");
        println!("File: {}", path.display());
        println!("Rows: {}", report.row_count);
        println!("Columns: {}", report.column_count);
        println!();

        println!("Quality Score: {:.1}%", report.score);
        println!();

        if report.issues.is_empty() {
            println!("\u{2713} No quality issues found\n");
        } else {
            println!("Issues Found:");
            println!("-------------");
            for issue in &report.issues {
                println!("  - {:?}", issue);
            }
            println!();
        }

        println!(
            "{:<20} {:<12} {:<12} {:<10}",
            "COLUMN", "NULL %", "UNIQUE", "STATUS"
        );
        println!("{}", "-".repeat(60));

        for (name, col) in &report.columns {
            let status = if col.is_constant() {
                "CONSTANT"
            } else if col.null_ratio > null_threshold {
                "HIGH NULL"
            } else {
                "OK"
            };

            println!(
                "{:<20} {:<12.2} {:<12} {:<10}",
                name,
                col.null_ratio * 100.0,
                col.unique_count,
                status
            );
        }
    }

    Ok(())
}

/// Generate a quality report.
pub(crate) fn cmd_quality_report(path: &Path, output: Option<&Path>) -> crate::Result<()> {
    let dataset = load_dataset(path)?;
    let report = QualityChecker::new().check(&dataset)?;

    let json = serde_json::json!({
        "path": path.display().to_string(),
        "rows": report.row_count,
        "columns": report.column_count,
        "has_issues": !report.issues.is_empty(),
        "score": report.score,
        "issues": report.issues.iter().map(|i| format!("{:?}", i)).collect::<Vec<_>>(),
        "column_qualities": report.columns.iter().map(|(name, c)| {
            serde_json::json!({
                "column": name,
                "null_ratio": c.null_ratio,
                "unique_count": c.unique_count,
                "is_constant": c.is_constant(),
            })
        }).collect::<Vec<_>>()
    });

    let json_str =
        serde_json::to_string_pretty(&json).map_err(|e| crate::Error::Format(e.to_string()))?;

    if let Some(output_path) = output {
        std::fs::write(output_path, &json_str).map_err(|e| crate::Error::io(e, output_path))?;
        println!("Quality report written to: {}", output_path.display());
    } else {
        println!("{}", json_str);
    }

    Ok(())
}

/// Calculate 100-point quality score with letter grade (GH-6).
///
/// Implements the Doctest Corpus QA Checklist for Publication with
/// weighted scoring per Toyota Way Jidoka principles.
#[allow(clippy::too_many_lines)]
pub(crate) fn cmd_quality_score(
    path: &Path,
    profile_name: &str,
    suggest: bool,
    json_output: bool,
    badge_output: bool,
) -> crate::Result<()> {
    use crate::quality::{QualityProfile, QualityScore};

    // Load the quality profile
    let profile = QualityProfile::by_name(profile_name).ok_or_else(|| {
        crate::Error::Format(format!(
            "Unknown quality profile '{}'. Available: {:?}",
            profile_name,
            QualityProfile::available_profiles()
        ))
    })?;

    let dataset = load_dataset(path)?;
    let report = QualityChecker::new().check(&dataset)?;

    // Wire QualityReport to ChecklistItems per the 100-point checklist
    let checklist = build_checklist_from_report(&report, &profile);
    let score = QualityScore::from_checklist(checklist);

    // Output based on flags
    if badge_output {
        println!("{}", score.badge_url());
    } else if json_output {
        println!("{}", score.to_json());
    } else {
        print_text_report(&score, &profile, path, suggest);
    }

    // Exit with non-zero code if critical failures (for CI/CD)
    if score.has_critical_failures() {
        std::process::exit(1);
    }

    Ok(())
}

/// Print the text quality report (Andon-style visual management).
fn print_text_report(
    score: &crate::quality::QualityScore,
    profile: &crate::quality::QualityProfile,
    path: &Path,
    suggest: bool,
) {
    let grade_symbol = match score.grade {
        crate::quality::LetterGrade::A | crate::quality::LetterGrade::B => "\u{2713}",
        crate::quality::LetterGrade::C => "\u{25CB}",
        crate::quality::LetterGrade::D => "\u{25B3}",
        crate::quality::LetterGrade::F => "\u{2717}",
    };

    let separator = "\u{2550}".repeat(63);
    println!("{separator}");
    println!(
        "  Data Quality Score: {} {} ({:.1}%)  ",
        grade_symbol, score.grade, score.score
    );
    println!("  Profile: {}  ", profile.name);
    println!("  Decision: {}  ", score.grade.publication_decision());
    println!("{separator}");
    println!();
    println!("File: {}", path.display());
    println!(
        "Points: {:.1} / {:.1}",
        score.points_earned, score.max_points
    );
    println!();

    print_severity_breakdown(score);
    print_critical_failures(score, suggest);

    if suggest {
        print_other_issues(score);
    }
}

/// Print the severity breakdown section.
fn print_severity_breakdown(score: &crate::quality::QualityScore) {
    use crate::quality::Severity;

    println!("Severity Breakdown:");
    for severity in [
        Severity::Critical,
        Severity::High,
        Severity::Medium,
        Severity::Low,
    ] {
        if let Some(stats) = score.severity_breakdown.get(&severity) {
            let status = if stats.failed == 0 {
                "\u{2713}"
            } else {
                "\u{2717}"
            };
            println!(
                "  {} {:8}: {}/{} passed ({:.1}/{:.1} pts)",
                status,
                format!("{}", severity),
                stats.passed,
                stats.total,
                stats.points_earned,
                stats.max_points
            );
        }
    }
    println!();
}

/// Print critical failures section.
fn print_critical_failures(score: &crate::quality::QualityScore, suggest: bool) {
    let critical_failures = score.critical_failures();
    if !critical_failures.is_empty() {
        println!("CRITICAL FAILURES (blocks publication):");
        for item in critical_failures {
            println!("  \u{2717} #{}: {}", item.id, item.description);
            if suggest {
                if let Some(ref suggestion) = item.suggestion {
                    println!("    \u{2192} {}", suggestion);
                }
            }
        }
        println!();
    }
}

/// Print non-critical issues with suggestions.
fn print_other_issues(score: &crate::quality::QualityScore) {
    use crate::quality::Severity;

    let failed = score.failed_items();
    let non_critical: Vec<_> = failed
        .iter()
        .filter(|i| i.severity != Severity::Critical)
        .collect();

    if !non_critical.is_empty() {
        println!("Other Issues ({}):", non_critical.len());
        for item in non_critical {
            let sev = match item.severity {
                Severity::High => "[HIGH]",
                Severity::Medium => "[MED]",
                Severity::Low => "[LOW]",
                Severity::Critical => "[CRIT]",
            };
            println!("  {} #{}: {}", sev, item.id, item.description);
            if let Some(ref suggestion) = item.suggestion {
                println!("      \u{2192} {}", suggestion);
            }
        }
    }
}

/// List available quality profiles.
#[allow(clippy::unnecessary_wraps)]
pub(crate) fn cmd_quality_profiles() -> crate::Result<()> {
    use crate::quality::QualityProfile;

    println!("Available Quality Profiles");
    println!("\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}");
    println!();

    for name in QualityProfile::available_profiles() {
        if let Some(profile) = QualityProfile::by_name(name) {
            println!("  {} - {}", profile.name, profile.description);
            if !profile.expected_constant_columns.is_empty() {
                let cols: Vec<_> = profile.expected_constant_columns.iter().collect();
                println!("    Expected constants: {:?}", cols);
            }
            if !profile.nullable_columns.is_empty() {
                let cols: Vec<_> = profile.nullable_columns.iter().collect();
                println!("    Nullable columns: {:?}", cols);
            }
            println!("    Max null ratio: {:.0}%", profile.max_null_ratio * 100.0);
            println!(
                "    Max duplicate ratio: {:.0}%",
                profile.max_duplicate_ratio * 100.0
            );
            println!();
        }
    }

    println!("Usage: alimentar quality score <path> --profile <name>");
    Ok(())
}

/// Build checklist items from `QualityReport`.
///
/// Maps `QualityReport` findings to the 100-point checklist defined in GH-6.
/// This wires the existing quality checks to the weighted scoring system.
#[allow(clippy::too_many_lines)]
pub(crate) fn build_checklist_from_report(
    report: &crate::quality::QualityReport,
    profile: &crate::quality::QualityProfile,
) -> Vec<crate::quality::ChecklistItem> {
    use crate::quality::{ChecklistItem, Severity};

    let mut items = Vec::new();
    let mut id: u8 = 1;

    // === Critical Checks (2.0x weight) ===

    // Check 1: Dataset not empty
    let has_rows = report.row_count > 0;
    items.push(
        ChecklistItem::new(id, "Dataset contains rows", Severity::Critical, has_rows)
            .with_suggestion("Extract more doctests or check input source"),
    );
    id += 1;

    // Check 2: No empty schema
    let has_columns = report.column_count > 0;
    items.push(
        ChecklistItem::new(
            id,
            "Schema has columns defined",
            Severity::Critical,
            has_columns,
        )
        .with_suggestion("Verify parser is extracting fields correctly"),
    );
    id += 1;

    // Check 3: No unexpected constant columns (would break training)
    // Filter out columns that the profile expects to be constant (e.g., source,
    // version) Also allow nullable columns to be all-null (constant null is OK
    // for optional fields)
    let unexpected_constant_cols: Vec<String> = report
        .columns
        .iter()
        .filter(|(name, c): &(&String, &ColumnQuality)| {
            c.is_constant() && !profile.is_expected_constant(name) && !profile.is_nullable(name)
        })
        .map(|(n, _)| n.clone())
        .collect();
    let no_unexpected_constants = unexpected_constant_cols.is_empty();
    items.push(
        ChecklistItem::new(
            id,
            "No unexpected constant columns (zero variance)",
            Severity::Critical,
            no_unexpected_constants,
        )
        .with_suggestion(format!(
            "Remove or investigate constant columns: {:?}",
            unexpected_constant_cols
        )),
    );
    id += 1;

    // === High Priority Checks (1.5x weight) ===

    // Check 4: Duplicate ratio below threshold (default 5%)
    let duplicate_ratio = report
        .issues
        .iter()
        .find_map(|i| {
            if let crate::quality::QualityIssue::DuplicateRows {
                duplicate_ratio: dr,
                ..
            } = i
            {
                Some(*dr)
            } else {
                None
            }
        })
        .unwrap_or(0.0);
    let low_duplicates = duplicate_ratio <= 0.05;
    items.push(
        ChecklistItem::new(
            id,
            format!(
                "Duplicate ratio <= 5% (actual: {:.1}%)",
                duplicate_ratio * 100.0
            ),
            Severity::High,
            low_duplicates,
        )
        .with_suggestion("Run deduplication: alimentar dedupe <file>"),
    );
    id += 1;

    // Check 5: No columns with >50% nulls (except nullable columns per profile)
    let high_null_cols: Vec<String> = report
        .columns
        .iter()
        .filter(|(name, c): &(&String, &ColumnQuality)| {
            c.null_ratio > 0.5 && !profile.is_nullable(name)
        })
        .map(|(n, _)| n.clone())
        .collect();
    let no_high_null = high_null_cols.is_empty();
    items.push(
        ChecklistItem::new(
            id,
            "No columns with >50% null values",
            Severity::High,
            no_high_null,
        )
        .with_suggestion(format!(
            "Investigate high-null columns: {:?}",
            high_null_cols
        )),
    );
    id += 1;

    // Check 6: Minimum row count (at least 100 for meaningful training)
    let min_rows = report.row_count >= 100;
    items.push(
        ChecklistItem::new(
            id,
            format!("Minimum 100 rows (actual: {})", report.row_count),
            Severity::High,
            min_rows,
        )
        .with_suggestion("Extract more data or combine with other sources"),
    );
    id += 1;

    // === Medium Priority Checks (1.0x weight) ===

    // Check 7: Overall quality score from existing checker
    let good_score = report.score >= 70.0;
    items.push(
        ChecklistItem::new(
            id,
            format!("Quality score >= 70% (actual: {:.1}%)", report.score),
            Severity::Medium,
            good_score,
        )
        .with_suggestion("Address issues reported by quality check"),
    );
    id += 1;

    // Check 8: No columns with >10% nulls (stricter, except nullable columns per
    // profile)
    let moderate_null_cols: Vec<String> = report
        .columns
        .iter()
        .filter(|(name, c): &(&String, &ColumnQuality)| {
            c.null_ratio > 0.1 && c.null_ratio <= 0.5 && !profile.is_nullable(name)
        })
        .map(|(n, _)| n.clone())
        .collect();
    let low_null_ratio = moderate_null_cols.is_empty();
    items.push(
        ChecklistItem::new(
            id,
            "No columns with >10% null values",
            Severity::Medium,
            low_null_ratio,
        )
        .with_suggestion(format!("Consider imputation for: {:?}", moderate_null_cols)),
    );
    id += 1;

    // Check 9: Reasonable column count (not too few for ML)
    let enough_columns = report.column_count >= 2;
    items.push(
        ChecklistItem::new(
            id,
            format!("At least 2 columns (actual: {})", report.column_count),
            Severity::Medium,
            enough_columns,
        )
        .with_suggestion("Ensure input and target columns are present"),
    );
    id += 1;

    // Check 10: No outlier issues detected
    let outlier_issues: Vec<(String, f64)> = report
        .issues
        .iter()
        .filter_map(|i| {
            if let crate::quality::QualityIssue::OutliersDetected {
                column,
                outlier_ratio: or,
                ..
            } = i
            {
                Some((column.clone(), *or))
            } else {
                None
            }
        })
        .collect();
    let no_severe_outliers = outlier_issues.iter().all(|(_, r)| *r < 0.1);
    items.push(
        ChecklistItem::new(
            id,
            "No columns with >10% outliers",
            Severity::Medium,
            no_severe_outliers,
        )
        .with_suggestion("Review outlier columns for data quality issues"),
    );
    id += 1;

    // === Low Priority Checks (0.5x weight) ===

    // Check 11: No warnings at all
    let no_issues = report.issues.is_empty();
    items.push(
        ChecklistItem::new(id, "No quality warnings", Severity::Low, no_issues)
            .with_suggestion("Address all warnings for best results"),
    );
    id += 1;

    // Check 12: Good cardinality (unique values)
    let low_cardinality_cols: Vec<String> = report
        .columns
        .iter()
        .filter(|(_, c): &(&String, &ColumnQuality)| c.unique_count < 10 && !c.is_constant())
        .map(|(n, _)| n.clone())
        .collect();
    let good_cardinality = low_cardinality_cols.is_empty();
    items.push(
        ChecklistItem::new(
            id,
            "All columns have reasonable cardinality (>10 unique)",
            Severity::Low,
            good_cardinality,
        )
        .with_suggestion(format!(
            "Low cardinality columns: {:?}",
            low_cardinality_cols
        )),
    );
    let _ = id; // suppress warning

    items
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::uninlined_format_args,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::redundant_clone,
    clippy::cast_lossless,
    clippy::redundant_closure_for_method_calls,
    clippy::too_many_lines,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::needless_late_init,
    clippy::redundant_pattern_matching
)]
mod tests {
    use std::sync::Arc;

    use arrow::{
        array::{Int32Array, StringArray},
        datatypes::{DataType, Field, Schema},
    };

    use super::*;
    use crate::ArrowDataset;

    fn create_test_parquet(path: &PathBuf, rows: usize) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let ids: Vec<i32> = (0..rows as i32).collect();
        let names: Vec<String> = ids.iter().map(|i| format!("item_{}", i)).collect();

        let batch = arrow::array::RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(names)),
            ],
        )
        .ok()
        .unwrap_or_else(|| panic!("Should create batch"));

        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        dataset
            .to_parquet(path)
            .ok()
            .unwrap_or_else(|| panic!("Should write parquet"));
    }

    #[test]
    fn test_cmd_quality_check_text() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.parquet");
        create_test_parquet(&path, 100);

        let result = cmd_quality_check(&path, 0.1, 0.05, true, "text");
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_quality_check_json() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.parquet");
        create_test_parquet(&path, 100);

        let result = cmd_quality_check(&path, 0.1, 0.05, true, "json");
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_quality_check_no_outliers() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.parquet");
        create_test_parquet(&path, 100);

        let result = cmd_quality_check(&path, 0.1, 0.05, false, "text");
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_quality_report_basic() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.parquet");
        create_test_parquet(&path, 100);

        let result = cmd_quality_report(&path, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_quality_report_to_file() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let data_path = temp_dir.path().join("data.parquet");
        let output_path = temp_dir.path().join("quality.json");
        create_test_parquet(&data_path, 100);

        let result = cmd_quality_report(&data_path, Some(&output_path));
        assert!(result.is_ok());
        assert!(output_path.exists());

        // Verify JSON is valid
        let content = std::fs::read_to_string(&output_path)
            .ok()
            .unwrap_or_else(|| panic!("Should read file"));
        let parsed: serde_json::Value = serde_json::from_str(&content)
            .ok()
            .unwrap_or_else(|| panic!("Should parse JSON"));
        assert!(parsed.get("score").is_some());
        assert!(parsed.get("has_issues").is_some());
    }

    #[test]
    fn test_cmd_quality_check_with_constant_column() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.parquet");

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("constant", DataType::Int32, false),
        ]));

        let ids: Vec<i32> = (0..100).collect();
        let constants: Vec<i32> = vec![42; 100];

        let batch = arrow::array::RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(Int32Array::from(constants)),
            ],
        )
        .ok()
        .unwrap_or_else(|| panic!("Should create batch"));

        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        dataset
            .to_parquet(&path)
            .ok()
            .unwrap_or_else(|| panic!("Should write parquet"));

        let result = cmd_quality_check(&path, 0.1, 0.05, true, "text");
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_quality_report_default_output() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.parquet");
        create_test_parquet(&path, 50);

        let result = cmd_quality_report(&path, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_quality_report_with_output() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.parquet");
        let output = temp_dir.path().join("report.html");
        create_test_parquet(&path, 50);

        let result = cmd_quality_report(&path, Some(&output));
        assert!(result.is_ok());
        assert!(output.exists());
    }

    #[test]
    fn test_cmd_quality_score() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.parquet");
        create_test_parquet(&path, 100);

        let result = cmd_quality_score(&path, "default", false, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_quality_score_with_json() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.parquet");
        create_test_parquet(&path, 100);

        let result = cmd_quality_score(&path, "default", false, true, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_quality_score_with_badge() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.parquet");
        create_test_parquet(&path, 100);

        let result = cmd_quality_score(&path, "default", false, false, true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_quality_score_with_suggest() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.parquet");
        create_test_parquet(&path, 100);

        let result = cmd_quality_score(&path, "default", true, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_quality_score_with_doctest_profile() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.parquet");
        create_test_parquet(&path, 100);

        let result = cmd_quality_score(&path, "doctest-corpus", false, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_quality_profiles() {
        let result = cmd_quality_profiles();
        assert!(result.is_ok());
    }

    // === Additional quality CLI tests ===

    #[test]
    fn test_cmd_quality_check_with_high_null_threshold() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.parquet");
        create_test_parquet(&path, 100);

        // Use very high null threshold
        let result = cmd_quality_check(&path, 0.9, 0.9, true, "text");
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_quality_check_small_dataset() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("small.parquet");
        create_test_parquet(&path, 5);

        let result = cmd_quality_check(&path, 0.1, 0.05, true, "text");
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_quality_check_large_dataset() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("large.parquet");
        create_test_parquet(&path, 500);

        let result = cmd_quality_check(&path, 0.1, 0.05, false, "json");
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_quality_score_ml_training_profile() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("ml.parquet");
        create_test_parquet(&path, 150);

        let result = cmd_quality_score(&path, "ml-training", false, false, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_quality_score_invalid_profile() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.parquet");
        create_test_parquet(&path, 100);

        let result = cmd_quality_score(&path, "nonexistent-profile", false, false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_cmd_quality_score_all_output_modes() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.parquet");
        create_test_parquet(&path, 100);

        // Test text output
        let result = cmd_quality_score(&path, "default", false, false, false);
        assert!(result.is_ok());

        // Test JSON output
        let result = cmd_quality_score(&path, "default", false, true, false);
        assert!(result.is_ok());

        // Test badge output
        let result = cmd_quality_score(&path, "default", false, false, true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_quality_report_to_stdout() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.parquet");
        create_test_parquet(&path, 50);

        // Output to stdout (None)
        let result = cmd_quality_report(&path, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_build_checklist_from_report_empty_dataset() {
        // Create empty-ish dataset scenario
        use crate::quality::{QualityChecker, QualityProfile};

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));

        let batch =
            arrow::array::RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1]))])
                .unwrap();

        let dataset = ArrowDataset::from_batch(batch).unwrap();
        let report = QualityChecker::new().check(&dataset).unwrap();
        let profile = QualityProfile::by_name("default").unwrap();

        let checklist = build_checklist_from_report(&report, &profile);

        // Should have checklist items
        assert!(!checklist.is_empty());

        // Check that we have critical, high, medium, low items
        let has_critical = checklist
            .iter()
            .any(|i| i.severity == crate::quality::Severity::Critical);
        let has_high = checklist
            .iter()
            .any(|i| i.severity == crate::quality::Severity::High);
        let has_medium = checklist
            .iter()
            .any(|i| i.severity == crate::quality::Severity::Medium);
        let has_low = checklist
            .iter()
            .any(|i| i.severity == crate::quality::Severity::Low);

        assert!(has_critical);
        assert!(has_high);
        assert!(has_medium);
        assert!(has_low);
    }

    #[test]
    fn test_build_checklist_high_quality_dataset() {
        use crate::quality::{QualityChecker, QualityProfile};

        // Create a high-quality dataset
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let ids: Vec<i32> = (0..200).collect();
        let names: Vec<String> = ids.iter().map(|i| format!("name_{}", i)).collect();

        let batch = arrow::array::RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(names)),
            ],
        )
        .unwrap();

        let dataset = ArrowDataset::from_batch(batch).unwrap();
        let report = QualityChecker::new().check(&dataset).unwrap();
        let profile = QualityProfile::by_name("default").unwrap();

        let checklist = build_checklist_from_report(&report, &profile);

        // Count passed items
        let passed = checklist.iter().filter(|i| i.passed).count();
        // A good dataset should have most checks passing
        assert!(passed > checklist.len() / 2);
    }

    #[test]
    fn test_cmd_quality_check_with_issues() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("issues.parquet");

        // Create dataset with potential issues (constant column)
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("constant", DataType::Int32, false),
        ]));

        let ids: Vec<i32> = (0..50).collect();
        let constants: Vec<i32> = vec![42; 50];

        let batch = arrow::array::RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(Int32Array::from(constants)),
            ],
        )
        .unwrap();

        let dataset = ArrowDataset::from_batch(batch).unwrap();
        dataset.to_parquet(&path).unwrap();

        // Should handle issues gracefully
        let result = cmd_quality_check(&path, 0.1, 0.05, true, "text");
        assert!(result.is_ok());
    }
}
