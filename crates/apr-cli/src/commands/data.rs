//! `apr data` subcommands — thin CLI wrappers around alimentar.
//!
//! All data pipeline logic lives in alimentar. This module handles
//! argument parsing, output formatting, and exit codes only.

use std::path::Path;

use colored::Colorize;

use crate::{error::CliError, output};

type Result<T> = std::result::Result<T, CliError>;

// ── apr data audit ──────────────────────────────────────────────────────────

/// Collected results from the audit analysis phase, passed to the formatter.
struct AuditResult {
    total: usize,
    out_of_range: usize,
    num_classes: usize,
    duplicate_count: usize,
    imbalance_report: alimentar::imbalance::ImbalanceReport,
    text_stats: alimentar::quality::TextColumnStats,
    path: String,
}

/// Render the human-readable text report for `run_audit`.
fn print_audit_report(r: &AuditResult) {
    let total = r.total;
    let out_of_range = r.out_of_range;
    let num_classes = r.num_classes;
    let duplicate_count = r.duplicate_count;

    output::section(&format!("Data Audit: {}", r.path));
    println!();

    output::kv("Samples", total);
    output::kv("Valid JSON", format!("{total} (100.0%)  {}", "OK".green()));

    if out_of_range == 0 {
        output::kv(
            "Valid labels",
            format!("{total} (100.0%)  {}", "OK".green()),
        );
    } else {
        output::kv(
            "Valid labels",
            format!(
                "{} ({:.1}%)  {} ({out_of_range} out of range 0..{num_classes})",
                total - out_of_range,
                (total - out_of_range) as f64 / total as f64 * 100.0,
                "WARN".yellow(),
            ),
        );
    }

    println!();
    println!("{}", "Class Distribution:".white().bold());

    // Sort classes by count descending
    let mut classes: Vec<_> = r.imbalance_report.distribution.counts.iter().collect();
    classes.sort_by(|a, b| b.1.cmp(a.1));

    for (label, count) in &classes {
        let pct = **count as f64 / total as f64 * 100.0;
        let bar_len = (pct / 3.0) as usize;
        let bar: String = std::iter::repeat('█').take(bar_len).collect();
        println!("  {label:>20}  {count:>8}  {pct:5.1}%  {bar}");
    }

    let ratio = r.imbalance_report.metrics.imbalance_ratio;
    let severity_str = r.imbalance_report.metrics.severity.description();
    let severity_colored = if ratio > 5.0 {
        format!("{ratio:.1}:1  {} {severity_str}", "WARN".yellow())
    } else {
        format!("{ratio:.1}:1  {severity_str}")
    };
    output::kv("Imbalance ratio", severity_colored);

    println!();
    println!("{}", "Input Length:".white().bold());
    println!(
        "  Min: {} chars | Max: {} chars | Mean: {:.0} chars",
        r.text_stats.min_len, r.text_stats.max_len, r.text_stats.mean_len
    );
    println!(
        "  P50: {} | P95: {} | P99: {}",
        r.text_stats.p50_len, r.text_stats.p95_len, r.text_stats.p99_len
    );

    println!();
    let dup_status = if duplicate_count > 0 {
        format!(
            "{duplicate_count} ({:.1}%)  {}",
            duplicate_count as f64 / total as f64 * 100.0,
            "minor".yellow()
        )
    } else {
        format!("0 (0.0%)  {}", "OK".green())
    };
    output::kv("Duplicates", dup_status);

    let preamble_status = if r.text_stats.preamble_count > 0 {
        format!(
            "{} ({:.1}%)  {}",
            r.text_stats.preamble_count,
            r.text_stats.preamble_count as f64 / total as f64 * 100.0,
            "WARN".yellow()
        )
    } else {
        format!("0 (0.0%)  {}", "stripped".green())
    };
    output::kv("Preamble found", preamble_status);

    let empty_status = if r.text_stats.empty_count > 0 {
        format!(
            "{} ({:.1}%)  {}",
            r.text_stats.empty_count,
            r.text_stats.empty_count as f64 / total as f64 * 100.0,
            "WARN".yellow()
        )
    } else {
        format!("0 (0.0%)  {}", "OK".green())
    };
    output::kv("Empty inputs", empty_status);

    // Issues summary
    let mut issues: Vec<String> = Vec::new();
    if ratio > 5.0 {
        issues.push(format!(
            "Severe class imbalance ({ratio:.1}:1) -- use `apr data balance` to fix"
        ));
    }
    if duplicate_count > 0 {
        issues.push(format!(
            "{duplicate_count} duplicate inputs -- use `apr data dedup` to remove"
        ));
    }
    if out_of_range > 0 {
        issues.push(format!(
            "{out_of_range} labels outside 0..{num_classes} range"
        ));
    }
    if r.text_stats.preamble_count > 0 {
        issues.push(format!(
            "{} inputs with shell preamble -- strip before training",
            r.text_stats.preamble_count
        ));
    }

    if !issues.is_empty() {
        println!();
        println!("{}", "ISSUES:".yellow().bold());
        for issue in &issues {
            println!("  {} {issue}", "!".yellow());
        }
    }
}

/// Run data quality audit on a JSONL classification dataset.
pub(crate) fn run_audit(
    path: &Path,
    num_classes: usize,
    input_column: &str,
    label_column: &str,
    preamble_prefix: Option<&str>,
    json_output: bool,
) -> Result<()> {
    use alimentar::{
        imbalance::ImbalanceDetector,
        quality::{QualityChecker, TextColumnStats},
        ArrowDataset, Dataset,
    };

    if !path.exists() {
        return Err(CliError::FileNotFound(path.to_path_buf()));
    }

    // Load JSONL as Arrow dataset
    let dataset = ArrowDataset::from_json(path).map_err(|e| {
        CliError::ValidationFailed(format!("Failed to load JSONL: {e}"))
    })?;

    let total = dataset.len();
    if total == 0 {
        return Err(CliError::ValidationFailed("Dataset is empty".to_string()));
    }

    // Validate schema has required columns
    let schema = dataset.schema();
    if schema.column_with_name(input_column).is_none() {
        return Err(CliError::ValidationFailed(format!(
            "Required column '{input_column}' not found in schema"
        )));
    }
    if schema.column_with_name(label_column).is_none() {
        return Err(CliError::ValidationFailed(format!(
            "Required column '{label_column}' not found in schema"
        )));
    }

    // Quality check (duplicates, nulls, etc.)
    let checker = QualityChecker::new()
        .max_null_ratio(0.01)
        .max_duplicate_ratio(0.05);
    let quality_report = checker.check(&dataset).map_err(|e| {
        CliError::ValidationFailed(format!("Quality check failed: {e}"))
    })?;

    // Imbalance analysis
    let imbalance_report = ImbalanceDetector::new(label_column)
        .analyze(&dataset)
        .map_err(|e| {
            CliError::ValidationFailed(format!("Imbalance analysis failed: {e}"))
        })?;

    // Text column statistics
    let text_stats = TextColumnStats::from_dataset(
        &dataset,
        input_column,
        preamble_prefix,
    )
    .map_err(|e| {
        CliError::ValidationFailed(format!("Text stats failed: {e}"))
    })?;

    // Label range validation (use imbalance distribution counts)
    let mut out_of_range = 0usize;
    for (label_str, _) in &imbalance_report.distribution.counts {
        if let Ok(v) = label_str.parse::<i64>() {
            if v < 0 || v >= num_classes as i64 {
                out_of_range += imbalance_report.distribution.get_count(label_str);
            }
        }
    }

    // Duplicate count from quality report
    let duplicate_count: usize = quality_report
        .issues
        .iter()
        .filter_map(|issue| match issue {
            alimentar::quality::QualityIssue::DuplicateRows {
                duplicate_count, ..
            } => Some(*duplicate_count),
            _ => None,
        })
        .sum();

    if json_output {
        let report = serde_json::json!({
            "path": path.display().to_string(),
            "total_samples": total,
            "num_classes": num_classes,
            "out_of_range_labels": out_of_range,
            "class_distribution": imbalance_report.distribution.counts,
            "imbalance_ratio": imbalance_report.metrics.imbalance_ratio,
            "imbalance_severity": format!("{:?}", imbalance_report.metrics.severity),
            "duplicates": duplicate_count,
            "input_length": {
                "min": text_stats.min_len,
                "max": text_stats.max_len,
                "mean": text_stats.mean_len,
                "p50": text_stats.p50_len,
                "p95": text_stats.p95_len,
                "p99": text_stats.p99_len,
            },
            "empty_inputs": text_stats.empty_count,
            "preamble_found": text_stats.preamble_count,
        });
        println!("{}", serde_json::to_string_pretty(&report).unwrap_or_default());
        return Ok(());
    }

    print_audit_report(&AuditResult {
        total,
        out_of_range,
        num_classes,
        duplicate_count,
        imbalance_report,
        text_stats,
        path: path.display().to_string(),
    });

    Ok(())
}

// ── apr data split ──────────────────────────────────────────────────────────

/// Stratified train/val/test split using alimentar.
pub(crate) fn run_split(
    path: &Path,
    label_column: &str,
    train_ratio: f64,
    val_ratio: f64,
    test_ratio: f64,
    seed: u64,
    output_dir: &Path,
    json_output: bool,
) -> Result<()> {
    use alimentar::{split::DatasetSplit, ArrowDataset, Dataset};

    if !path.exists() {
        return Err(CliError::FileNotFound(path.to_path_buf()));
    }

    let dataset = ArrowDataset::from_json(path).map_err(|e| {
        CliError::ValidationFailed(format!("Failed to load JSONL: {e}"))
    })?;

    let total = dataset.len();

    // Perform stratified split
    let split = DatasetSplit::stratified(
        &dataset,
        label_column,
        train_ratio,
        test_ratio,
        Some(val_ratio),
        Some(seed),
    )
    .map_err(|e| CliError::ValidationFailed(format!("Split failed: {e}")))?;

    // Create output directory
    std::fs::create_dir_all(output_dir).map_err(|e| {
        CliError::ValidationFailed(format!(
            "Failed to create output dir {}: {e}",
            output_dir.display()
        ))
    })?;

    // Write splits as JSONL
    let train_path = output_dir.join("train.jsonl");
    let val_path = output_dir.join("val.jsonl");
    let test_path = output_dir.join("test.jsonl");

    split.train().to_json(&train_path).map_err(|e| {
        CliError::ValidationFailed(format!("Failed to write train.jsonl: {e}"))
    })?;
    split.test().to_json(&test_path).map_err(|e| {
        CliError::ValidationFailed(format!("Failed to write test.jsonl: {e}"))
    })?;
    if let Some(val) = split.validation() {
        val.to_json(&val_path).map_err(|e| {
            CliError::ValidationFailed(format!("Failed to write val.jsonl: {e}"))
        })?;
    }

    let train_len = split.train().len();
    let test_len = split.test().len();
    let val_len = split.validation().map_or(0, |v| v.len());

    if json_output {
        let report = serde_json::json!({
            "source": path.display().to_string(),
            "total": total,
            "seed": seed,
            "train": { "path": train_path.display().to_string(), "samples": train_len },
            "val": { "path": val_path.display().to_string(), "samples": val_len },
            "test": { "path": test_path.display().to_string(), "samples": test_len },
        });
        println!("{}", serde_json::to_string_pretty(&report).unwrap_or_default());
        return Ok(());
    }

    output::section("Stratified Split");
    println!();
    output::kv("Source", format!("{} ({total} samples)", path.display()));
    output::kv("Seed", seed);
    output::kv(
        "Ratios",
        format!("train={train_ratio}, val={val_ratio}, test={test_ratio}"),
    );
    println!();
    output::kv("Train", format!("{} ({train_len} samples)", train_path.display()));
    output::kv("Val", format!("{} ({val_len} samples)", val_path.display()));
    output::kv("Test", format!("{} ({test_len} samples)", test_path.display()));
    println!();
    println!("{} Splits written to {}", "OK".green(), output_dir.display());

    Ok(())
}

// ── apr data balance ────────────────────────────────────────────────────────

/// Resample a classification dataset to address class imbalance.
pub(crate) fn run_balance(
    path: &Path,
    label_column: &str,
    strategy: &str,
    num_classes: Option<usize>,
    seed: u64,
    output_path: Option<&Path>,
    json_output: bool,
) -> Result<()> {
    use alimentar::{
        imbalance::ImbalanceDetector, ArrowDataset, Dataset, ResampleStrategy,
    };

    if !path.exists() {
        return Err(CliError::FileNotFound(path.to_path_buf()));
    }

    let dataset = ArrowDataset::from_json(path).map_err(|e| {
        CliError::ValidationFailed(format!("Failed to load JSONL: {e}"))
    })?;

    let original_len = dataset.len();

    // sqrt-inverse mode: just compute and print weights, no resampling
    if strategy == "sqrt-inverse" {
        let report = ImbalanceDetector::new(label_column)
            .analyze(&dataset)
            .map_err(|e| CliError::ValidationFailed(format!("Imbalance analysis failed: {e}")))?;

        let k = num_classes.unwrap_or(report.distribution.num_classes);
        // Build ordered counts vector
        let mut ordered_counts = vec![0usize; k];
        for (label, count) in &report.distribution.counts {
            if let Ok(idx) = label.parse::<usize>() {
                if idx < k {
                    ordered_counts[idx] = *count;
                }
            }
        }

        let weights = alimentar::sqrt_inverse_weights(&ordered_counts);

        if json_output {
            let report = serde_json::json!({
                "strategy": "sqrt-inverse",
                "class_counts": ordered_counts,
                "weights": weights,
            });
            println!("{}", serde_json::to_string_pretty(&report).unwrap_or_default());
        } else {
            output::section("Sqrt-Inverse Class Weights");
            println!();
            for (i, w) in weights.iter().enumerate() {
                let count = ordered_counts.get(i).copied().unwrap_or(0);
                println!("  class {i}: count={count:>8}  weight={w:.4}");
            }
            let sum: f32 = weights.iter().sum();
            println!();
            output::kv("Weight sum", format!("{sum:.4} (should equal {k})"));
        }
        return Ok(());
    }

    let resample_strategy = match strategy {
        "oversample" => ResampleStrategy::Oversample,
        "undersample" => ResampleStrategy::Undersample,
        other => {
            return Err(CliError::ValidationFailed(format!(
                "Unknown strategy '{other}'. Use: oversample, undersample, sqrt-inverse"
            )));
        }
    };

    let resampled = alimentar::resample(&dataset, label_column, resample_strategy, seed)
        .map_err(|e| CliError::ValidationFailed(format!("Resampling failed: {e}")))?;

    let new_len = resampled.len();

    // Write output
    if let Some(out) = output_path {
        resampled.to_json(out).map_err(|e| {
            CliError::ValidationFailed(format!("Failed to write output: {e}"))
        })?;

        if json_output {
            let report = serde_json::json!({
                "strategy": strategy,
                "original_samples": original_len,
                "resampled_samples": new_len,
                "output": out.display().to_string(),
            });
            println!("{}", serde_json::to_string_pretty(&report).unwrap_or_default());
        } else {
            output::section("Class Rebalancing");
            println!();
            output::kv("Strategy", strategy);
            output::kv("Original", format!("{original_len} samples"));
            output::kv("Resampled", format!("{new_len} samples"));
            output::kv("Output", out.display());
            println!();
            println!("{} Resampled dataset written", "OK".green());
        }
    } else {
        return Err(CliError::ValidationFailed(
            "--output is required for oversample/undersample strategies".to_string(),
        ));
    }

    Ok(())
}
