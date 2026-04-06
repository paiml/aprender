//! Drift detection CLI commands.

use std::path::{Path, PathBuf};

use clap::Subcommand;

use super::basic::load_dataset;
use crate::{
    drift::{DriftDetector, DriftSeverity, DriftTest},
    sketch::{DataSketch, DistributedDriftDetector, SketchType},
    Dataset,
};

/// Drift detection commands.
#[derive(Subcommand)]
pub enum DriftCommands {
    /// Detect drift between reference and current datasets
    Detect {
        /// Reference (baseline) dataset
        #[arg(short, long)]
        reference: PathBuf,
        /// Current dataset to compare
        #[arg(short, long)]
        current: PathBuf,
        /// Statistical tests to use (ks, chi2, psi, js)
        #[arg(short, long, default_value = "ks,psi")]
        tests: String,
        /// Significance threshold (alpha)
        #[arg(short, long, default_value = "0.05")]
        alpha: f64,
        /// Output format (text, json)
        #[arg(short, long, default_value = "text")]
        format: String,
    },
    /// Generate a drift report summary
    Report {
        /// Reference (baseline) dataset
        #[arg(short, long)]
        reference: PathBuf,
        /// Current dataset to compare
        #[arg(short, long)]
        current: PathBuf,
        /// Output file for the report (JSON format)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Create a sketch from a dataset for distributed drift detection
    Sketch {
        /// Input dataset file
        input: PathBuf,
        /// Output sketch file
        #[arg(short, long)]
        output: PathBuf,
        /// Sketch algorithm type (tdigest, ddsketch)
        #[arg(short = 't', long, default_value = "tdigest")]
        sketch_type: String,
        /// Source identifier (e.g., node name)
        #[arg(short, long)]
        source: Option<String>,
        /// Output format (json, binary)
        #[arg(short, long, default_value = "json")]
        format: String,
    },
    /// Merge multiple sketches into one
    Merge {
        /// Sketch files to merge
        #[arg(required = true)]
        sketches: Vec<PathBuf>,
        /// Output merged sketch file
        #[arg(short, long)]
        output: PathBuf,
        /// Output format (json, binary)
        #[arg(short, long, default_value = "json")]
        format: String,
    },
    /// Compare two sketches for drift
    Compare {
        /// Reference (baseline) sketch
        #[arg(short, long)]
        reference: PathBuf,
        /// Current sketch to compare
        #[arg(short, long)]
        current: PathBuf,
        /// Drift detection threshold
        #[arg(short = 't', long, default_value = "0.1")]
        threshold: f64,
        /// Output format (text, json)
        #[arg(short, long, default_value = "text")]
        format: String,
    },
}

/// Parse drift test types from a comma-separated string.
pub(crate) fn parse_drift_tests(tests_str: &str) -> Vec<DriftTest> {
    tests_str
        .split(',')
        .filter_map(|t| match t.trim().to_lowercase().as_str() {
            "ks" => Some(DriftTest::KolmogorovSmirnov),
            "chi2" | "chisquared" => Some(DriftTest::ChiSquared),
            "psi" => Some(DriftTest::PSI),
            "js" | "jensenshannon" => Some(DriftTest::JensenShannon),
            _ => None,
        })
        .collect()
}

/// Get severity symbol for display.
pub(crate) fn severity_symbol(severity: DriftSeverity) -> &'static str {
    match severity {
        DriftSeverity::None => "\u{2713}",     // checkmark
        DriftSeverity::Low => "\u{25CB}",      // empty circle
        DriftSeverity::Medium => "\u{25CF}",   // filled circle
        DriftSeverity::High => "\u{25B2}",     // triangle
        DriftSeverity::Critical => "\u{2716}", // X mark
    }
}

/// Detect drift between reference and current datasets.
pub(crate) fn cmd_drift_detect(
    reference: &Path,
    current: &Path,
    tests_str: &str,
    alpha: f64,
    format: &str,
) -> crate::Result<()> {
    let ref_dataset = load_dataset(reference)?;
    let cur_dataset = load_dataset(current)?;

    let tests = parse_drift_tests(tests_str);
    if tests.is_empty() {
        return Err(crate::Error::invalid_config(
            "No valid tests specified. Use: ks, chi2, psi, js",
        ));
    }

    let mut detector = DriftDetector::new(ref_dataset).with_alpha(alpha);
    for test in tests {
        detector = detector.with_test(test);
    }

    let report = detector.detect(&cur_dataset)?;

    if format == "json" {
        // JSON output
        let json = serde_json::json!({
            "drift_detected": report.drift_detected,
            "columns": report.column_scores.values().map(|d| {
                serde_json::json!({
                    "column": d.column,
                    "test": format!("{:?}", d.test),
                    "statistic": d.statistic,
                    "p_value": d.p_value,
                    "drift_detected": d.drift_detected,
                    "severity": format!("{:?}", d.severity),
                })
            }).collect::<Vec<_>>()
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).map_err(|e| crate::Error::Format(e.to_string()))?
        );
    } else {
        // Text output
        println!("Drift Detection Report");
        println!("======================");
        println!("Reference: {}", reference.display());
        println!("Current:   {}", current.display());
        println!("Alpha:     {}", alpha);
        println!();

        if report.drift_detected {
            println!("\u{26A0}\u{FE0F}  DRIFT DETECTED\n");
        } else {
            println!("\u{2713} No significant drift detected\n");
        }

        println!(
            "{:<20} {:<15} {:<12} {:<12} {:<10} DRIFT",
            "COLUMN", "TEST", "STATISTIC", "P-VALUE", "SEVERITY"
        );
        println!("{}", "-".repeat(80));

        for drift in report.column_scores.values() {
            let drift_str = if drift.drift_detected { "YES" } else { "no" };
            let p_value_str = drift
                .p_value
                .map_or_else(|| "N/A".to_string(), |p| format!("{:.4}", p));
            println!(
                "{:<20} {:<15} {:<12.4} {:<12} {:<10} {} {}",
                drift.column,
                format!("{:?}", drift.test),
                drift.statistic,
                p_value_str,
                format!("{:?}", drift.severity),
                severity_symbol(drift.severity),
                drift_str
            );
        }

        println!();
        let drifted: Vec<_> = report
            .column_scores
            .values()
            .filter(|d| d.drift_detected)
            .map(|d| d.column.clone())
            .collect();
        if !drifted.is_empty() {
            println!("Columns with drift: {}", drifted.join(", "));
        }
    }

    Ok(())
}

/// Generate a drift report summary.
pub(crate) fn cmd_drift_report(
    reference: &Path,
    current: &Path,
    output: Option<&PathBuf>,
) -> crate::Result<()> {
    let ref_dataset = load_dataset(reference)?;
    let cur_dataset = load_dataset(current)?;

    let detector = DriftDetector::new(ref_dataset)
        .with_test(DriftTest::KolmogorovSmirnov)
        .with_test(DriftTest::PSI)
        .with_test(DriftTest::JensenShannon);

    let report = detector.detect(&cur_dataset)?;

    let drifted_count = report
        .column_scores
        .values()
        .filter(|d| d.drift_detected)
        .count();
    let json = serde_json::json!({
        "reference": reference.display().to_string(),
        "current": current.display().to_string(),
        "drift_detected": report.drift_detected,
        "summary": {
            "total_columns": report.column_scores.len(),
            "drifted_columns": drifted_count,
        },
        "columns": report.column_scores.values().map(|d| {
            serde_json::json!({
                "column": d.column,
                "test": format!("{:?}", d.test),
                "statistic": d.statistic,
                "p_value": d.p_value,
                "drift_detected": d.drift_detected,
                "severity": format!("{:?}", d.severity),
            })
        }).collect::<Vec<_>>()
    });

    let json_str =
        serde_json::to_string_pretty(&json).map_err(|e| crate::Error::Format(e.to_string()))?;

    if let Some(output_path) = output {
        std::fs::write(output_path, &json_str).map_err(|e| crate::Error::io(e, output_path))?;
        println!("Drift report written to: {}", output_path.display());
    } else {
        println!("{}", json_str);
    }

    Ok(())
}

/// Parse sketch type from string.
pub(crate) fn parse_sketch_type(s: &str) -> Option<SketchType> {
    match s.to_lowercase().as_str() {
        "tdigest" | "t-digest" => Some(SketchType::TDigest),
        "ddsketch" | "dd-sketch" => Some(SketchType::DDSketch),
        _ => None,
    }
}

/// Create a sketch from a dataset for distributed drift detection.
pub(crate) fn cmd_drift_sketch(
    input: &Path,
    output: &Path,
    sketch_type: &str,
    source: Option<&str>,
    format: &str,
) -> crate::Result<()> {
    let sketch_type = parse_sketch_type(sketch_type).ok_or_else(|| {
        crate::Error::invalid_config(format!(
            "Unknown sketch type: {}. Use 'tdigest' or 'ddsketch'",
            sketch_type
        ))
    })?;

    let dataset = load_dataset(input)?;
    let mut sketch = DataSketch::from_dataset(&dataset, sketch_type)?;

    if let Some(src) = source {
        sketch = sketch.with_source(src);
    }

    match format {
        "binary" | "bin" => {
            let bytes = sketch.to_bytes()?;
            std::fs::write(output, bytes).map_err(|e| crate::Error::io(e, output))?;
        }
        _ => {
            // Default to JSON
            let json = serde_json::to_string_pretty(&sketch)
                .map_err(|e| crate::Error::Format(e.to_string()))?;
            std::fs::write(output, json).map_err(|e| crate::Error::io(e, output))?;
        }
    }

    println!(
        "Created {} sketch from {} ({} rows) -> {}",
        sketch_type.name(),
        input.display(),
        dataset.len(),
        output.display()
    );

    Ok(())
}

/// Load a sketch from a file.
pub(crate) fn load_sketch(path: &Path) -> crate::Result<DataSketch> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext {
        "bin" | "binary" => {
            let bytes = std::fs::read(path).map_err(|e| crate::Error::io(e, path))?;
            DataSketch::from_bytes(&bytes)
        }
        _ => {
            // Default to JSON
            let json = std::fs::read_to_string(path).map_err(|e| crate::Error::io(e, path))?;
            serde_json::from_str(&json)
                .map_err(|e| crate::Error::Format(format!("Invalid sketch JSON: {}", e)))
        }
    }
}

/// Merge multiple sketches into one.
pub(crate) fn cmd_drift_merge(
    sketches: &[PathBuf],
    output: &Path,
    format: &str,
) -> crate::Result<()> {
    if sketches.is_empty() {
        return Err(crate::Error::invalid_config(
            "No sketches provided to merge",
        ));
    }

    let loaded: Vec<DataSketch> = sketches
        .iter()
        .map(|p| load_sketch(p))
        .collect::<Result<Vec<_>, _>>()?;

    let merged = DataSketch::merge(&loaded)?;

    match format {
        "binary" | "bin" => {
            let bytes = merged.to_bytes()?;
            std::fs::write(output, bytes).map_err(|e| crate::Error::io(e, output))?;
        }
        _ => {
            let json = serde_json::to_string_pretty(&merged)
                .map_err(|e| crate::Error::Format(e.to_string()))?;
            std::fs::write(output, json).map_err(|e| crate::Error::io(e, output))?;
        }
    }

    println!(
        "Merged {} sketches ({} total rows) -> {}",
        sketches.len(),
        merged.row_count,
        output.display()
    );

    Ok(())
}

/// Compare two sketches for drift.
pub(crate) fn cmd_drift_compare(
    reference: &Path,
    current: &Path,
    threshold: f64,
    format: &str,
) -> crate::Result<()> {
    let ref_sketch = load_sketch(reference)?;
    let cur_sketch = load_sketch(current)?;

    let detector = DistributedDriftDetector::new()
        .with_sketch_type(ref_sketch.sketch_type)
        .with_threshold(threshold);

    let results = detector.compare(&ref_sketch, &cur_sketch)?;

    let drift_detected = results.iter().any(|r| r.severity.is_drift());

    if format == "json" {
        let json = serde_json::json!({
            "reference": reference.display().to_string(),
            "current": current.display().to_string(),
            "drift_detected": drift_detected,
            "threshold": threshold,
            "columns": results.iter().map(|r| {
                serde_json::json!({
                    "column": r.column,
                    "statistic": r.statistic,
                    "severity": format!("{:?}", r.severity),
                    "drift_detected": r.severity.is_drift(),
                })
            }).collect::<Vec<_>>()
        });

        let json_str =
            serde_json::to_string_pretty(&json).map_err(|e| crate::Error::Format(e.to_string()))?;
        println!("{}", json_str);
    } else {
        // Text format
        println!("Sketch Drift Comparison");
        println!("=======================");
        println!("Reference: {}", reference.display());
        println!("Current:   {}", current.display());
        println!("Threshold: {}", threshold);
        println!();

        if results.is_empty() {
            println!("No numeric columns to compare.");
        } else {
            println!(
                "{:<20} {:>10} {:>10} DRIFT",
                "COLUMN", "STATISTIC", "SEVERITY"
            );
            println!("{}", "-".repeat(55));

            for result in &results {
                let drift_symbol = if result.severity.is_drift() {
                    severity_symbol(result.severity)
                } else {
                    "\u{2713}"
                };
                println!(
                    "{:<20} {:>10.4} {:>10} {}",
                    result.column,
                    result.statistic,
                    format!("{:?}", result.severity),
                    drift_symbol
                );
            }

            println!();
            if drift_detected {
                println!("\u{26A0} Drift detected in one or more columns");
            } else {
                println!("\u{2713} No significant drift detected");
            }
        }
    }

    Ok(())
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
        array::{Float64Array, Int32Array, StringArray},
        datatypes::{DataType, Field, Schema},
    };

    use super::*;
    use crate::ArrowDataset;

    fn create_test_parquet(path: &Path, rows: usize) {
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

    fn create_test_float_parquet(path: &Path, rows: usize) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Float64, false),
        ]));

        let ids: Vec<i32> = (0..rows as i32).collect();
        let values: Vec<f64> = ids.iter().map(|i| *i as f64 * 1.5).collect();

        let batch = arrow::array::RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(Float64Array::from(values)),
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
    fn test_parse_drift_tests() {
        let tests = parse_drift_tests("ks,psi");
        assert_eq!(tests.len(), 2);

        let tests = parse_drift_tests("ks, chi2, psi, js");
        assert_eq!(tests.len(), 4);

        let tests = parse_drift_tests("invalid");
        assert!(tests.is_empty());

        let tests = parse_drift_tests("KS,PSI");
        assert_eq!(tests.len(), 2);
    }

    #[test]
    fn test_severity_symbol() {
        assert_eq!(severity_symbol(DriftSeverity::None), "\u{2713}");
        assert_eq!(severity_symbol(DriftSeverity::Low), "\u{25CB}");
        assert_eq!(severity_symbol(DriftSeverity::Medium), "\u{25CF}");
        assert_eq!(severity_symbol(DriftSeverity::High), "\u{25B2}");
        assert_eq!(severity_symbol(DriftSeverity::Critical), "\u{2716}");
    }

    #[test]
    fn test_cmd_drift_detect_same_data() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.parquet");
        create_test_parquet(&path, 100);

        // Compare dataset with itself - should detect no drift
        let result = cmd_drift_detect(&path, &path, "ks,psi", 0.05, "text");
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_drift_detect_json_format() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.parquet");
        create_test_parquet(&path, 100);

        let result = cmd_drift_detect(&path, &path, "ks", 0.05, "json");
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_drift_detect_invalid_tests() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.parquet");
        create_test_parquet(&path, 100);

        let result = cmd_drift_detect(&path, &path, "invalid", 0.05, "text");
        assert!(result.is_err());
    }

    #[test]
    fn test_cmd_drift_report() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.parquet");
        create_test_parquet(&path, 100);

        // Report without output file (prints to stdout)
        let result = cmd_drift_report(&path, &path, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_drift_report_to_file() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let data_path = temp_dir.path().join("data.parquet");
        let output_path = temp_dir.path().join("report.json");
        create_test_parquet(&data_path, 100);

        let result = cmd_drift_report(&data_path, &data_path, Some(&output_path));
        assert!(result.is_ok());
        assert!(output_path.exists());

        // Verify JSON is valid
        let content = std::fs::read_to_string(&output_path)
            .ok()
            .unwrap_or_else(|| panic!("Should read file"));
        let parsed: serde_json::Value = serde_json::from_str(&content)
            .ok()
            .unwrap_or_else(|| panic!("Should parse JSON"));
        assert!(parsed.get("drift_detected").is_some());
    }

    #[test]
    fn test_parse_sketch_type() {
        assert!(matches!(
            parse_sketch_type("tdigest"),
            Some(SketchType::TDigest)
        ));
        assert!(matches!(
            parse_sketch_type("TDIGEST"),
            Some(SketchType::TDigest)
        ));
        assert!(matches!(
            parse_sketch_type("ddsketch"),
            Some(SketchType::DDSketch)
        ));
        assert!(matches!(
            parse_sketch_type("DDSKETCH"),
            Some(SketchType::DDSketch)
        ));
        assert!(parse_sketch_type("invalid").is_none());
    }

    #[test]
    fn test_cmd_drift_sketch_tdigest() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let data_path = temp_dir.path().join("data.parquet");
        let sketch_path = temp_dir.path().join("sketch.json");
        create_test_float_parquet(&data_path, 100);

        let result = cmd_drift_sketch(&data_path, &sketch_path, "tdigest", None, "json");
        assert!(result.is_ok());
        assert!(sketch_path.exists());

        // Verify sketch file is valid JSON
        let content = std::fs::read_to_string(&sketch_path)
            .ok()
            .unwrap_or_else(|| panic!("Should read file"));
        let parsed: serde_json::Value = serde_json::from_str(&content)
            .ok()
            .unwrap_or_else(|| panic!("Should parse JSON"));
        assert!(parsed.get("sketch_type").is_some());
        assert!(parsed.get("row_count").is_some());
    }

    #[test]
    fn test_cmd_drift_sketch_ddsketch_type() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let data_path = temp_dir.path().join("data.parquet");
        let sketch_path = temp_dir.path().join("sketch.json");
        create_test_float_parquet(&data_path, 100);

        let result = cmd_drift_sketch(&data_path, &sketch_path, "ddsketch", None, "json");
        assert!(result.is_ok());
        assert!(sketch_path.exists());
    }

    #[test]
    fn test_cmd_drift_sketch_with_source() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let data_path = temp_dir.path().join("data.parquet");
        let sketch_path = temp_dir.path().join("sketch.json");
        create_test_float_parquet(&data_path, 50);

        let result = cmd_drift_sketch(&data_path, &sketch_path, "tdigest", Some("node-1"), "json");
        assert!(result.is_ok());

        // Verify source is in the output
        let content = std::fs::read_to_string(&sketch_path)
            .ok()
            .unwrap_or_else(|| panic!("Should read file"));
        let parsed: serde_json::Value = serde_json::from_str(&content)
            .ok()
            .unwrap_or_else(|| panic!("Should parse JSON"));
        assert_eq!(
            parsed.get("source").and_then(|v| v.as_str()),
            Some("node-1")
        );
    }

    #[test]
    fn test_cmd_drift_sketch_binary_format() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let data_path = temp_dir.path().join("data.parquet");
        let sketch_path = temp_dir.path().join("sketch.bin");
        create_test_float_parquet(&data_path, 100);

        let result = cmd_drift_sketch(&data_path, &sketch_path, "tdigest", None, "binary");
        assert!(result.is_ok());
        assert!(sketch_path.exists());

        // Binary file should exist and be non-empty
        let metadata = std::fs::metadata(&sketch_path)
            .ok()
            .unwrap_or_else(|| panic!("Should get metadata"));
        assert!(metadata.len() > 0);
    }

    #[test]
    fn test_cmd_drift_sketch_invalid_type() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let data_path = temp_dir.path().join("data.parquet");
        let sketch_path = temp_dir.path().join("sketch.json");
        create_test_float_parquet(&data_path, 100);

        let result = cmd_drift_sketch(&data_path, &sketch_path, "invalid", None, "json");
        assert!(result.is_err());
    }

    #[test]
    fn test_cmd_drift_merge_tdigest() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));

        // Create two datasets
        let data1 = temp_dir.path().join("data1.parquet");
        let data2 = temp_dir.path().join("data2.parquet");
        let sketch1 = temp_dir.path().join("sketch1.json");
        let sketch2 = temp_dir.path().join("sketch2.json");
        let merged = temp_dir.path().join("merged.json");

        create_test_float_parquet(&data1, 50);
        create_test_float_parquet(&data2, 50);

        // Create sketches
        cmd_drift_sketch(&data1, &sketch1, "tdigest", Some("node-1"), "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create sketch1"));
        cmd_drift_sketch(&data2, &sketch2, "tdigest", Some("node-2"), "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create sketch2"));

        // Merge
        let sketches = vec![sketch1.clone(), sketch2.clone()];
        let result = cmd_drift_merge(&sketches, &merged, "json");
        assert!(result.is_ok());
        assert!(merged.exists());

        // Verify merged sketch
        let content = std::fs::read_to_string(&merged)
            .ok()
            .unwrap_or_else(|| panic!("Should read file"));
        let parsed: serde_json::Value = serde_json::from_str(&content)
            .ok()
            .unwrap_or_else(|| panic!("Should parse JSON"));
        assert_eq!(parsed.get("row_count").and_then(|v| v.as_u64()), Some(100));
    }

    #[test]
    fn test_cmd_drift_merge_single_sketch() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));

        let data = temp_dir.path().join("data.parquet");
        let sketch = temp_dir.path().join("sketch.json");
        let merged = temp_dir.path().join("merged.json");

        create_test_float_parquet(&data, 100);
        cmd_drift_sketch(&data, &sketch, "tdigest", None, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create sketch"));

        // Merge single sketch
        let sketches = vec![sketch.clone()];
        let result = cmd_drift_merge(&sketches, &merged, "json");
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_drift_merge_empty_fails() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let merged = temp_dir.path().join("merged.json");

        let sketches: Vec<PathBuf> = vec![];
        let result = cmd_drift_merge(&sketches, &merged, "json");
        assert!(result.is_err());
    }

    #[test]
    fn test_cmd_drift_compare_no_drift() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));

        let data = temp_dir.path().join("data.parquet");
        let sketch1 = temp_dir.path().join("sketch1.json");
        let sketch2 = temp_dir.path().join("sketch2.json");

        create_test_float_parquet(&data, 100);

        // Create two identical sketches
        cmd_drift_sketch(&data, &sketch1, "tdigest", None, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create sketch1"));
        cmd_drift_sketch(&data, &sketch2, "tdigest", None, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create sketch2"));

        let result = cmd_drift_compare(&sketch1, &sketch2, 0.1, "text");
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_drift_compare_json_format() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));

        let data = temp_dir.path().join("data.parquet");
        let sketch1 = temp_dir.path().join("sketch1.json");
        let sketch2 = temp_dir.path().join("sketch2.json");

        create_test_float_parquet(&data, 100);

        cmd_drift_sketch(&data, &sketch1, "tdigest", None, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create sketch1"));
        cmd_drift_sketch(&data, &sketch2, "tdigest", None, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create sketch2"));

        let result = cmd_drift_compare(&sketch1, &sketch2, 0.1, "json");
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_drift_merge_binary_format() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));

        let data = temp_dir.path().join("data.parquet");
        let sketch = temp_dir.path().join("sketch.bin");
        let merged = temp_dir.path().join("merged.bin");

        create_test_float_parquet(&data, 100);
        cmd_drift_sketch(&data, &sketch, "tdigest", None, "binary")
            .ok()
            .unwrap_or_else(|| panic!("Should create sketch"));

        let sketches = vec![sketch.clone()];
        let result = cmd_drift_merge(&sketches, &merged, "binary");
        assert!(result.is_ok());
        assert!(merged.exists());
    }

    #[test]
    fn test_load_sketch_invalid_json() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let sketch_path = temp_dir.path().join("invalid.json");

        std::fs::write(&sketch_path, "{ invalid json }")
            .ok()
            .unwrap_or_else(|| panic!("Should write file"));

        let result = load_sketch(&sketch_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_cmd_drift_detect_all_tests() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.parquet");
        create_test_parquet(&path, 100);

        let result = cmd_drift_detect(&path, &path, "ks,chi2,psi,js", 0.05, "text");
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_drift_detect_with_drift() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let ref_path = temp_dir.path().join("ref.parquet");
        let cur_path = temp_dir.path().join("cur.parquet");

        create_test_parquet(&ref_path, 100);

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let ids: Vec<i32> = (500..600).collect();
        let names: Vec<String> = ids.iter().map(|i| format!("different_{}", i)).collect();

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
            .to_parquet(&cur_path)
            .ok()
            .unwrap_or_else(|| panic!("Should write parquet"));

        let result = cmd_drift_detect(&ref_path, &cur_path, "ks,psi", 0.05, "text");
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_drift_compare_empty_results() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));

        // Create dataset with only string column (no numeric for sketch)
        let data_path = temp_dir.path().join("strings.parquet");
        let sketch1_path = temp_dir.path().join("sketch1.json");
        let sketch2_path = temp_dir.path().join("sketch2.json");

        let schema = Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, false)]));

        let names: Vec<String> = (0..50).map(|i| format!("name_{}", i)).collect();

        let batch =
            arrow::array::RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(names))])
                .ok()
                .unwrap_or_else(|| panic!("Should create batch"));

        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        dataset
            .to_parquet(&data_path)
            .ok()
            .unwrap_or_else(|| panic!("Should write parquet"));

        cmd_drift_sketch(&data_path, &sketch1_path, "tdigest", None, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create sketch1"));
        cmd_drift_sketch(&data_path, &sketch2_path, "tdigest", None, "json")
            .ok()
            .unwrap_or_else(|| panic!("Should create sketch2"));

        let result = cmd_drift_compare(&sketch1_path, &sketch2_path, 0.1, "text");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_drift_tests_single() {
        let tests = parse_drift_tests("ks");
        assert_eq!(tests.len(), 1);
        assert!(matches!(tests[0], DriftTest::KolmogorovSmirnov));
    }

    #[test]
    fn test_parse_drift_tests_multiple() {
        let tests = parse_drift_tests("ks,chi2,psi,js");
        assert_eq!(tests.len(), 4);
    }

    #[test]
    fn test_parse_drift_tests_with_spaces() {
        let tests = parse_drift_tests("ks, chi2 ,psi");
        assert_eq!(tests.len(), 3);
    }

    #[test]
    fn test_parse_drift_tests_unknown() {
        let tests = parse_drift_tests("unknown");
        assert!(tests.is_empty());
    }

    #[test]
    fn test_parse_sketch_type_tdigest() {
        assert!(matches!(
            parse_sketch_type("tdigest"),
            Some(SketchType::TDigest)
        ));
    }

    #[test]
    fn test_parse_sketch_type_ddsketch() {
        assert!(matches!(
            parse_sketch_type("ddsketch"),
            Some(SketchType::DDSketch)
        ));
    }

    #[test]
    fn test_parse_sketch_type_unknown() {
        assert!(parse_sketch_type("unknown").is_none());
    }

    #[test]
    fn test_cmd_drift_report_basic() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let ref_path = temp_dir.path().join("ref.parquet");
        let cur_path = temp_dir.path().join("cur.parquet");
        create_test_parquet(&ref_path, 100);
        create_test_parquet(&cur_path, 100);

        let result = cmd_drift_report(&ref_path, &cur_path, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_drift_report_with_output() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let ref_path = temp_dir.path().join("ref.parquet");
        let cur_path = temp_dir.path().join("cur.parquet");
        let output = temp_dir.path().join("report.html");
        create_test_parquet(&ref_path, 100);
        create_test_parquet(&cur_path, 100);

        let result = cmd_drift_report(&ref_path, &cur_path, Some(&output));
        assert!(result.is_ok());
        assert!(output.exists());
    }

    #[test]
    fn test_cmd_drift_sketch_ddsketch_json_output() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let data_path = temp_dir.path().join("data.parquet");
        let sketch_path = temp_dir.path().join("sketch.json");
        create_test_parquet(&data_path, 100);

        let result = cmd_drift_sketch(&data_path, &sketch_path, "ddsketch", None, "json");
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_drift_merge() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let data1 = temp_dir.path().join("data1.parquet");
        let data2 = temp_dir.path().join("data2.parquet");
        let sketch1 = temp_dir.path().join("sketch1.json");
        let sketch2 = temp_dir.path().join("sketch2.json");
        let merged = temp_dir.path().join("merged.json");

        // Create datasets with float column
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float64,
            false,
        )]));

        let batch1 = arrow::array::RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]))],
        )
        .unwrap();
        let dataset1 = ArrowDataset::from_batch(batch1).unwrap();
        dataset1.to_parquet(&data1).unwrap();

        let batch2 = arrow::array::RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![6.0, 7.0, 8.0, 9.0, 10.0]))],
        )
        .unwrap();
        let dataset2 = ArrowDataset::from_batch(batch2).unwrap();
        dataset2.to_parquet(&data2).unwrap();

        cmd_drift_sketch(&data1, &sketch1, "tdigest", None, "json").unwrap();
        cmd_drift_sketch(&data2, &sketch2, "tdigest", None, "json").unwrap();

        let sketches = vec![sketch1, sketch2];
        let result = cmd_drift_merge(&sketches, &merged, "json");
        assert!(result.is_ok());
    }
}
