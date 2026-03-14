//! `apr data` subcommands — thin CLI wrappers around alimentar.
//!
//! All data pipeline logic lives in alimentar. This module handles
//! argument parsing, output formatting, and exit codes only.

use std::path::Path;

use colored::Colorize;

use crate::{error::CliError, output};

type Result<T> = std::result::Result<T, CliError>;

// ── Local stubs for APIs not yet published in alimentar 0.2.6 ────────────────

/// Text column statistics — mirrors the alimentar::quality::TextColumnStats
/// that exists in local source but hasn't been published yet.
struct TextColumnStats {
    pub min_len: usize,
    pub max_len: usize,
    pub mean_len: f64,
    pub p50_len: usize,
    pub p95_len: usize,
    pub p99_len: usize,
    pub empty_count: usize,
    pub preamble_count: usize,
    #[allow(dead_code)]
    pub total: usize,
}

impl TextColumnStats {
    /// Compute text column statistics from a JSONL file by reading lines directly.
    ///
    /// Avoids depending on arrow array types (not a direct dep of apr-cli).
    /// Reads the JSONL file, extracts the text column, and computes stats.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn from_jsonl_path(
        path: &Path,
        column: &str,
        preamble_prefix: Option<&str>,
    ) -> std::result::Result<Self, String> {
        use std::io::{BufRead, BufReader};

        let file = std::fs::File::open(path)
            .map_err(|e| format!("Failed to open {}: {e}", path.display()))?;
        let reader = BufReader::new(file);

        let mut lengths: Vec<usize> = Vec::new();
        let mut empty_count = 0usize;
        let mut preamble_count = 0usize;

        for line in reader.lines() {
            let line = line.map_err(|e| format!("Read error: {e}"))?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let obj: serde_json::Value =
                serde_json::from_str(line).map_err(|e| format!("JSON parse error: {e}"))?;

            if let Some(val) = obj.get(column).and_then(|v| v.as_str()) {
                let len = val.len();
                lengths.push(len);

                if val.trim().is_empty() {
                    empty_count += 1;
                }
                if let Some(prefix) = preamble_prefix {
                    if val.starts_with(prefix) {
                        preamble_count += 1;
                    }
                }
            }
            // Skip nulls / missing column entries
        }

        if lengths.is_empty() {
            return Ok(Self {
                min_len: 0,
                max_len: 0,
                mean_len: 0.0,
                p50_len: 0,
                p95_len: 0,
                p99_len: 0,
                empty_count: 0,
                preamble_count: 0,
                total: 0,
            });
        }

        lengths.sort_unstable();
        let total = lengths.len();
        let min_len = lengths[0];
        let max_len = lengths[total - 1];
        let mean_len = lengths.iter().sum::<usize>() as f64 / total as f64;
        let p50_len = lengths[total / 2];
        let p95_len = lengths[(total as f64 * 0.95) as usize];
        let p99_len = lengths[(total as f64 * 0.99).min((total - 1) as f64) as usize];

        Ok(Self {
            min_len,
            max_len,
            mean_len,
            p50_len,
            p95_len,
            p99_len,
            empty_count,
            preamble_count,
            total,
        })
    }
}

/// Resampling strategy — mirrors alimentar::ResampleStrategy (not yet published).
#[derive(Debug, Clone, Copy)]
enum ResampleStrategy {
    Oversample,
    Undersample,
}

/// Compute sqrt-inverse class weights — mirrors alimentar::sqrt_inverse_weights.
fn sqrt_inverse_weights(counts: &[usize]) -> Vec<f32> {
    let total: usize = counts.iter().sum();
    if total == 0 || counts.is_empty() {
        return vec![];
    }
    let k = counts.len() as f32;
    counts
        .iter()
        .map(|&c| {
            if c == 0 {
                0.0
            } else {
                (total as f32 / (k * c as f32)).sqrt()
            }
        })
        .collect()
}

/// Select resampled indices using deterministic hashing for reproducibility.
fn select_resample_indices(
    label_indices: &std::collections::HashMap<String, Vec<usize>>,
    target_count: usize,
    seed: u64,
) -> Vec<usize> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut selected_indices: Vec<usize> = Vec::new();
    for (_label, indices) in label_indices {
        if indices.len() >= target_count {
            let mut shuffled = indices.clone();
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            let h = hasher.finish();
            shuffled.sort_by(|a, b| {
                let mut ha = DefaultHasher::new();
                (*a as u64 ^ h).hash(&mut ha);
                let mut hb = DefaultHasher::new();
                (*b as u64 ^ h).hash(&mut hb);
                ha.finish().cmp(&hb.finish())
            });
            selected_indices.extend_from_slice(&shuffled[..target_count]);
        } else {
            selected_indices.extend_from_slice(indices);
            let mut extra_needed = target_count - indices.len();
            let mut cycle_idx = 0;
            while extra_needed > 0 {
                selected_indices.push(indices[cycle_idx % indices.len()]);
                cycle_idx += 1;
                extra_needed -= 1;
            }
        }
    }
    selected_indices
}

/// Resample a JSONL file to balance classes — mirrors alimentar::resample (not yet published).
///
/// Operates on the JSONL file directly (read lines, resample, write to temp, reload)
/// to avoid depending on arrow array types which are not a direct dependency of apr-cli.
fn resample_jsonl(
    path: &Path,
    label_column: &str,
    strategy: ResampleStrategy,
    seed: u64,
) -> std::result::Result<alimentar::ArrowDataset, String> {
    use std::collections::HashMap;
    use std::io::{BufRead, BufReader, Write};

    // Read all JSONL lines and group by label
    let file =
        std::fs::File::open(path).map_err(|e| format!("Failed to open {}: {e}", path.display()))?;
    let reader = BufReader::new(file);

    let mut rows: Vec<String> = Vec::new();
    let mut label_indices: HashMap<String, Vec<usize>> = HashMap::new();

    for line in reader.lines() {
        let line = line.map_err(|e| format!("Read error: {e}"))?;
        let trimmed = line.trim().to_string();
        if trimmed.is_empty() {
            continue;
        }
        let obj: serde_json::Value =
            serde_json::from_str(&trimmed).map_err(|e| format!("JSON parse error: {e}"))?;

        let label = obj
            .get(label_column)
            .map(|v| match v {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Number(n) => n.to_string(),
                other => other.to_string(),
            })
            .unwrap_or_default();

        let idx = rows.len();
        label_indices.entry(label).or_default().push(idx);
        rows.push(trimmed);
    }

    if rows.is_empty() {
        return Err("Empty dataset".to_string());
    }

    let target_count = match strategy {
        ResampleStrategy::Oversample => label_indices.values().map(|v| v.len()).max().unwrap_or(0),
        ResampleStrategy::Undersample => label_indices.values().map(|v| v.len()).min().unwrap_or(0),
    };

    let mut selected_indices = select_resample_indices(&label_indices, target_count, seed);
    selected_indices.sort_unstable();

    let tmp_path = std::env::temp_dir().join("apr-resample-tmp.jsonl");
    {
        let mut out = std::fs::File::create(&tmp_path)
            .map_err(|e| format!("Failed to create temp file: {e}"))?;
        for &idx in &selected_indices {
            writeln!(out, "{}", rows[idx]).map_err(|e| format!("Write error: {e}"))?;
        }
    }

    let result = alimentar::ArrowDataset::from_json(&tmp_path)
        .map_err(|e| format!("Failed to reload resampled dataset: {e}"));

    let _ = std::fs::remove_file(&tmp_path);
    result
}

// ── apr data audit ──────────────────────────────────────────────────────────

/// Collected results from the audit analysis phase, passed to the formatter.
struct AuditResult {
    total: usize,
    out_of_range: usize,
    num_classes: usize,
    duplicate_count: usize,
    imbalance_report: alimentar::imbalance::ImbalanceReport,
    text_stats: TextColumnStats,
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
                total.saturating_sub(out_of_range),
                total.saturating_sub(out_of_range) as f64 / total as f64 * 100.0,
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
        let bar: String = "█".repeat(bar_len);
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

/// Validate dataset schema has required columns.
fn validate_audit_schema(
    dataset: &alimentar::ArrowDataset,
    input_column: &str,
    label_column: &str,
) -> Result<()> {
    use alimentar::Dataset;
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
    Ok(())
}

/// Count labels outside the valid range [0, num_classes).
fn count_out_of_range_labels(
    imbalance_report: &alimentar::imbalance::ImbalanceReport,
    num_classes: usize,
) -> usize {
    let mut out_of_range = 0usize;
    for label_str in imbalance_report.distribution.counts.keys() {
        if let Ok(v) = label_str.parse::<i64>() {
            if v < 0 || v >= num_classes as i64 {
                out_of_range += imbalance_report.distribution.get_count(label_str);
            }
        }
    }
    out_of_range
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
    use alimentar::{imbalance::ImbalanceDetector, quality::QualityChecker, ArrowDataset, Dataset};

    if !path.exists() {
        return Err(CliError::FileNotFound(path.to_path_buf()));
    }

    let dataset = ArrowDataset::from_json(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to load JSONL: {e}")))?;

    let total = dataset.len();
    if total == 0 {
        return Err(CliError::ValidationFailed("Dataset is empty".to_string()));
    }

    validate_audit_schema(&dataset, input_column, label_column)?;

    let checker = QualityChecker::new()
        .max_null_ratio(0.01)
        .max_duplicate_ratio(0.05);
    let quality_report = checker
        .check(&dataset)
        .map_err(|e| CliError::ValidationFailed(format!("Quality check failed: {e}")))?;

    let imbalance_report = ImbalanceDetector::new(label_column)
        .analyze(&dataset)
        .map_err(|e| CliError::ValidationFailed(format!("Imbalance analysis failed: {e}")))?;

    let text_stats = TextColumnStats::from_jsonl_path(path, input_column, preamble_prefix)
        .map_err(|e| CliError::ValidationFailed(format!("Text stats failed: {e}")))?;

    let out_of_range = count_out_of_range_labels(&imbalance_report, num_classes);

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
        #[allow(clippy::disallowed_methods)]
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
        println!(
            "{}",
            serde_json::to_string_pretty(&report).unwrap_or_default()
        );
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

    let dataset = ArrowDataset::from_json(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to load JSONL: {e}")))?;

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

    split
        .train()
        .to_json(&train_path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to write train.jsonl: {e}")))?;
    split
        .test()
        .to_json(&test_path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to write test.jsonl: {e}")))?;
    if let Some(val) = split.validation() {
        val.to_json(&val_path)
            .map_err(|e| CliError::ValidationFailed(format!("Failed to write val.jsonl: {e}")))?;
    }

    let train_len = split.train().len();
    let test_len = split.test().len();
    let val_len = split.validation().map_or(0, alimentar::Dataset::len);

    if json_output {
        #[allow(clippy::disallowed_methods)] // serde_json::json!() macro uses infallible unwrap
        let report = serde_json::json!({
            "source": path.display().to_string(),
            "total": total,
            "seed": seed,
            "train": { "path": train_path.display().to_string(), "samples": train_len },
            "val": { "path": val_path.display().to_string(), "samples": val_len },
            "test": { "path": test_path.display().to_string(), "samples": test_len },
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&report).unwrap_or_default()
        );
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
    output::kv(
        "Train",
        format!("{} ({train_len} samples)", train_path.display()),
    );
    output::kv("Val", format!("{} ({val_len} samples)", val_path.display()));
    output::kv(
        "Test",
        format!("{} ({test_len} samples)", test_path.display()),
    );
    println!();
    println!(
        "{} Splits written to {}",
        "OK".green(),
        output_dir.display()
    );

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
    use alimentar::{imbalance::ImbalanceDetector, ArrowDataset, Dataset};

    if !path.exists() {
        return Err(CliError::FileNotFound(path.to_path_buf()));
    }

    let dataset = ArrowDataset::from_json(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to load JSONL: {e}")))?;

    let original_len = dataset.len();

    // sqrt-inverse mode: just compute and print weights, no resampling
    if strategy == "sqrt-inverse" {
        return run_balance_sqrt_inverse(&dataset, label_column, num_classes, json_output);
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

    let resampled = resample_jsonl(path, label_column, resample_strategy, seed)
        .map_err(|e| CliError::ValidationFailed(format!("Resampling failed: {e}")))?;

    let new_len = resampled.len();

    let out = output_path.ok_or_else(|| {
        CliError::ValidationFailed(
            "--output is required for oversample/undersample strategies".to_string(),
        )
    })?;

    resampled
        .to_json(out)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to write output: {e}")))?;

    print_balance_result(strategy, original_len, new_len, out, json_output);
    Ok(())
}

/// Handle sqrt-inverse balance strategy: compute and display class weights.
fn run_balance_sqrt_inverse(
    dataset: &alimentar::ArrowDataset,
    label_column: &str,
    num_classes: Option<usize>,
    json_output: bool,
) -> Result<()> {
    use alimentar::{imbalance::ImbalanceDetector, Dataset};

    let report = ImbalanceDetector::new(label_column)
        .analyze(dataset)
        .map_err(|e| CliError::ValidationFailed(format!("Imbalance analysis failed: {e}")))?;

    let k = num_classes.unwrap_or(report.distribution.num_classes);
    let mut ordered_counts = vec![0usize; k];
    for (label, count) in &report.distribution.counts {
        if let Ok(idx) = label.parse::<usize>() {
            if idx < k {
                ordered_counts[idx] = *count;
            }
        }
    }

    let weights = sqrt_inverse_weights(&ordered_counts);

    if json_output {
        #[allow(clippy::disallowed_methods)]
        let report = serde_json::json!({
            "strategy": "sqrt-inverse",
            "class_counts": ordered_counts,
            "weights": weights,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&report).unwrap_or_default()
        );
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
    Ok(())
}

/// Print balance result output (JSON or human-readable).
#[allow(clippy::disallowed_methods)]
fn print_balance_result(
    strategy: &str,
    original_len: usize,
    new_len: usize,
    out: &Path,
    json_output: bool,
) {
    if json_output {
        let report = serde_json::json!({
            "strategy": strategy,
            "original_samples": original_len,
            "resampled_samples": new_len,
            "output": out.display().to_string(),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&report).unwrap_or_default()
        );
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
}

// ── apr data decontaminate ──────────────────────────────────────────────────

/// Check training data for benchmark contamination via n-gram overlap.
pub(crate) fn run_decontaminate(
    path: &Path,
    reference_paths: &[std::path::PathBuf],
    ngram_size: usize,
    threshold: f64,
    json_output: bool,
) -> Result<()> {
    use alimentar::quality::check_contamination;

    if !path.exists() {
        return Err(CliError::FileNotFound(path.to_path_buf()));
    }

    // Load training data (one text per line from JSONL)
    let training_text = std::fs::read_to_string(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to read training data: {e}")))?;
    let training_lines: Vec<&str> = training_text.lines().collect();

    // Load reference benchmark data
    let mut ref_texts = Vec::new();
    for ref_path in reference_paths {
        if !ref_path.exists() {
            return Err(CliError::FileNotFound(ref_path.clone()));
        }
        let text = std::fs::read_to_string(ref_path).map_err(|e| {
            CliError::ValidationFailed(format!(
                "Failed to read reference {}: {e}",
                ref_path.display()
            ))
        })?;
        for line in text.lines() {
            ref_texts.push(line.to_string());
        }
    }
    let ref_slices: Vec<&str> = ref_texts.iter().map(|s| s.as_str()).collect();

    let report = check_contamination(&training_lines, &ref_slices, ngram_size, threshold);

    if json_output {
        #[allow(clippy::disallowed_methods)]
        let json = serde_json::json!({
            "ngram_size": report.ngram_size,
            "threshold": report.threshold,
            "total_samples": report.total_samples,
            "contaminated_count": report.contaminated_count,
            "contamination_rate": report.contamination_rate,
            "gate": if report.contamination_rate < 0.01 { "PASS" } else { "FAIL" },
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
    } else {
        output::section("Decontamination Check");
        println!();
        output::kv("Training samples", format!("{}", report.total_samples));
        output::kv("Reference samples", format!("{}", ref_slices.len()));
        output::kv("N-gram size", format!("{}", report.ngram_size));
        output::kv("Threshold", format!("{:.2}", report.threshold));
        println!();
        output::kv("Contaminated", format!("{}", report.contaminated_count));
        output::kv("Rate", format!("{:.2}%", report.contamination_rate * 100.0));
        println!();
        if report.contamination_rate < 0.01 {
            println!("{} Contamination rate <1% (AC-016 gate)", "PASS".green());
        } else {
            println!(
                "{} Contamination rate {:.2}% exceeds 1% threshold",
                "FAIL".red(),
                report.contamination_rate * 100.0
            );
        }
    }

    if report.contamination_rate >= 0.01 {
        return Err(CliError::ValidationFailed(format!(
            "Contamination rate {:.2}% exceeds 1% gate (AC-016)",
            report.contamination_rate * 100.0
        )));
    }

    Ok(())
}
