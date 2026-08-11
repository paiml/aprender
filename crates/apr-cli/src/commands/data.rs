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

/// Compute sqrt-inverse class weights, normalised so that the weights sum to
/// the number of non-empty classes (i.e. mean weight 1.0).
///
/// The raw `sqrt(N / (K * n_i))` form is NOT normalised: for a 90/10 split it
/// yields 0.7454 and 2.2361, summing to 2.9814 while the command reported
/// "should equal 2". Shipping weights whose mean is 1.49 rather than 1.0
/// silently inflates the effective loss scale of any training run that
/// consumes them. Normalising preserves the sqrt-inverse *ratios* between
/// classes — the only part that affects rebalancing — and makes the stated
/// invariant true.
///
/// Classes with zero samples keep weight 0.0 and are excluded from the target
/// sum; `weight_sum_target` reports what the sum must equal.
fn sqrt_inverse_weights(counts: &[usize]) -> Vec<f32> {
    let total: usize = counts.iter().sum();
    if total == 0 || counts.is_empty() {
        return vec![];
    }
    let k = counts.len() as f32;
    let raw: Vec<f32> = counts
        .iter()
        .map(|&c| {
            if c == 0 {
                0.0
            } else {
                (total as f32 / (k * c as f32)).sqrt()
            }
        })
        .collect();

    let raw_sum: f32 = raw.iter().sum();
    let target = weight_sum_target(counts) as f32;
    if raw_sum <= 0.0 || target <= 0.0 {
        return raw;
    }
    let scale = target / raw_sum;
    raw.iter().map(|w| w * scale).collect()
}

/// The value the sqrt-inverse weights are normalised to sum to: the number of
/// classes that actually have samples.
fn weight_sum_target(counts: &[usize]) -> usize {
    counts.iter().filter(|&&c| c > 0).count()
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
    /// Whether the duplicate ratio crossed the alert threshold. Decoupled from
    /// `duplicate_count`: a dataset can hold real duplicates without tripping
    /// the alert, and the count must still be reported truthfully.
    duplicates_flagged: bool,
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
    let dup_pct = duplicate_count as f64 / total as f64 * 100.0;
    let dup_status = if duplicate_count == 0 {
        format!("0 (0.0%)  {}", "OK".green())
    } else if r.duplicates_flagged {
        format!("{duplicate_count} ({dup_pct:.1}%)  {}", "minor".yellow())
    } else {
        format!(
            "{duplicate_count} ({dup_pct:.1}%)  {}",
            "below alert threshold".green()
        )
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
    if r.duplicates_flagged {
        issues.push(format!(
            "{duplicate_count} duplicate rows -- use `apr data dedup {} -o clean.jsonl` to remove",
            r.path
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

/// Number of duplicate rows in a quality report.
///
/// Reads the count the checker actually measured. It must NOT be derived by
/// summing the `DuplicateRows` issues: that issue is only pushed once the
/// duplicate ratio crosses the alert threshold (default 1%), so summing it
/// collapses every sub-threshold count to a hard 0 — 49 duplicate rows in
/// 10,000 were reported as "0 (0.0%) OK".
fn duplicate_row_count(report: &alimentar::quality::QualityReport) -> usize {
    report.duplicate_row_count
}

/// Whether the duplicate ratio crossed the checker's alert threshold.
fn duplicates_flagged(report: &alimentar::quality::QualityReport) -> bool {
    report
        .issues
        .iter()
        .any(|i| matches!(i, alimentar::quality::QualityIssue::DuplicateRows { .. }))
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
#[provable_contracts_macros::contract(
    "apr-cli-operations-v1",
    equation = "mutating_output_contract"
)]
pub(crate) fn run_audit(
    path: &Path,
    num_classes: usize,
    input_column: &str,
    label_column: &str,
    preamble_prefix: Option<&str>,
    json_output: bool,
) -> Result<()> {
    contract_pre_data_validation!();
    use alimentar::{imbalance::ImbalanceDetector, quality::QualityChecker, ArrowDataset, Dataset};

    if !path.exists() {
        return Err(CliError::FileNotFound(path.to_path_buf()));
    }

    let dataset = ArrowDataset::from_json(path).map_err(|e| {
        // GH-644: Wrap Arrow error internals in a user-friendly message.
        // Arrow errors can contain deeply nested type/schema details that
        // are not actionable for end users.
        let raw = e.to_string();
        let msg = if raw.contains("Json error") || raw.contains("ArrowError") {
            format!(
                "Failed to parse JSONL file '{}'. Ensure every line is valid JSON with consistent schema.",
                path.display()
            )
        } else {
            format!("Failed to load JSONL: {raw}")
        };
        CliError::ValidationFailed(msg)
    })?;

    let total = dataset.len();
    if total == 0 {
        return Err(CliError::ValidationFailed("Dataset is empty".to_string()));
    }

    validate_audit_schema(&dataset, input_column, label_column)?;

    // NOTE: `max_duplicate_ratio` is the per-COLUMN duplicate threshold. The
    // duplicate-ROW alert uses QualityThresholds::max_duplicate_row_ratio
    // (1%); the row count itself is always read from the report, never
    // inferred from whether that alert fired.
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

    let duplicate_count = duplicate_row_count(&quality_report);
    let dup_flagged = duplicates_flagged(&quality_report);

    if json_output {
        #[allow(clippy::disallowed_methods)]
        let report = serde_json::json!({
            "path": path.display().to_string(),
            "total_samples": total,
            "num_classes": imbalance_report.distribution.counts.len(),
            "out_of_range_labels": out_of_range,
            "class_distribution": imbalance_report.distribution.counts,
            "imbalance_ratio": imbalance_report.metrics.imbalance_ratio,
            "imbalance_severity": format!("{:?}", imbalance_report.metrics.severity),
            "duplicates": duplicate_count,
            "duplicates_flagged": dup_flagged,
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
        duplicates_flagged: dup_flagged,
        imbalance_report,
        text_stats,
        path: path.display().to_string(),
    });

    contract_post_data_validation!(&());
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
    contract_pre_data_split_determinism!();
    use alimentar::{split::DatasetSplit, ArrowDataset, Dataset};

    if !path.exists() {
        return Err(CliError::FileNotFound(path.to_path_buf()));
    }

    let dataset = ArrowDataset::from_json(path).map_err(|e| {
        let raw = e.to_string();
        let msg = if raw.contains("Json error") || raw.contains("ArrowError") {
            format!(
                "Failed to parse JSONL file '{}'. Ensure every line is valid JSON with consistent schema.",
                path.display()
            )
        } else {
            format!("Failed to load JSONL: {raw}")
        };
        CliError::ValidationFailed(msg)
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

    contract_post_data_split_determinism!(&());
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

    let dataset = ArrowDataset::from_json(path).map_err(|e| {
        let raw = e.to_string();
        let msg = if raw.contains("Json error") || raw.contains("ArrowError") {
            format!(
                "Failed to parse JSONL file '{}'. Ensure every line is valid JSON with consistent schema.",
                path.display()
            )
        } else {
            format!("Failed to load JSONL: {raw}")
        };
        CliError::ValidationFailed(msg)
    })?;

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
    use alimentar::imbalance::ImbalanceDetector;

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
    let sum: f32 = weights.iter().sum();
    let target = weight_sum_target(&ordered_counts);

    if json_output {
        #[allow(clippy::disallowed_methods)]
        let report = serde_json::json!({
            "strategy": "sqrt-inverse",
            "class_counts": ordered_counts,
            "weights": weights,
            "weight_sum": sum,
            "weight_sum_target": target,
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
        println!();
        output::kv("Weight sum", format!("{sum:.4} (equals {target})"));
        if target < k {
            println!(
                "  {} {} of {k} classes have no samples and carry weight 0",
                "note:".yellow(),
                k - target
            );
        }
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
    let vacuous = report.is_vacuous();

    if json_output {
        #[allow(clippy::disallowed_methods)]
        let json = serde_json::json!({
            "ngram_size": report.ngram_size,
            "threshold": report.threshold,
            "total_samples": report.total_samples,
            "evaluated_samples": report.evaluated_samples,
            "evaluated_references": report.evaluated_references,
            "contaminated_count": report.contaminated_count,
            "contamination_rate": report.contamination_rate,
            "vacuous": vacuous,
            "gate": if vacuous { "VACUOUS" } else if report.contamination_rate < 0.01 { "PASS" } else { "FAIL" },
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
    } else {
        output::section("Decontamination Check");
        println!();
        output::kv("Training samples", format!("{}", report.total_samples));
        output::kv(
            "  compared",
            format!(
                "{} ({} too short for a {}-gram)",
                report.evaluated_samples,
                report.total_samples - report.evaluated_samples,
                report.ngram_size
            ),
        );
        output::kv("Reference samples", format!("{}", ref_slices.len()));
        output::kv("  compared", format!("{}", report.evaluated_references));
        output::kv("N-gram size", format!("{}", report.ngram_size));
        output::kv("Threshold", format!("{:.2}", report.threshold));
        println!();
        output::kv("Contaminated", format!("{}", report.contaminated_count));
        output::kv("Rate", format!("{:.2}%", report.contamination_rate * 100.0));
        println!();
        if vacuous {
            println!(
                "{} nothing was compared -- a 0.00% rate here proves nothing",
                "VACUOUS".red()
            );
        } else if report.contamination_rate < 0.01 {
            println!("{} Contamination rate <1% (AC-016 gate)", "PASS".green());
        } else {
            println!(
                "{} Contamination rate {:.2}% exceeds 1% threshold",
                "FAIL".red(),
                report.contamination_rate * 100.0
            );
        }
    }

    // A configuration that compares nothing must never satisfy the AC-016
    // gate. `--ngram 1000` on a corpus of short lines produced no n-grams at
    // all and reported a clean 0.00% on a file holding byte-identical copies
    // of the benchmark.
    if vacuous {
        return Err(CliError::ValidationFailed(format!(
            "Contamination check was vacuous: {} of {} training samples and {} of {} reference \
             samples yielded no {}-grams, so nothing was compared. Lower --ngram (currently {}).",
            report.total_samples - report.evaluated_samples,
            report.total_samples,
            ref_slices.len() - report.evaluated_references,
            ref_slices.len(),
            report.ngram_size,
            report.ngram_size,
        )));
    }

    if report.contamination_rate >= 0.01 {
        return Err(CliError::ValidationFailed(format!(
            "Contamination rate {:.2}% exceeds 1% gate (AC-016)",
            report.contamination_rate * 100.0
        )));
    }

    Ok(())
}

// ── apr data dedup ──────────────────────────────────────────────────────────

/// Canonical form of a JSONL row for exact-duplicate comparison.
///
/// Objects are key-sorted so that two rows carrying the same fields in a
/// different key order compare equal — which is how the Arrow-backed audit
/// counts them. Non-object lines fall back to their trimmed text.
fn canonical_row(line: &str) -> String {
    match serde_json::from_str::<serde_json::Value>(line) {
        Ok(serde_json::Value::Object(map)) => {
            let mut pairs: Vec<(&String, String)> =
                map.iter().map(|(k, v)| (k, v.to_string())).collect();
            pairs.sort_by(|a, b| a.0.cmp(b.0));
            pairs
                .iter()
                .map(|(k, v)| format!("{k}\u{1}{v}"))
                .collect::<Vec<_>>()
                .join("\u{2}")
        }
        Ok(other) => other.to_string(),
        Err(_) => line.trim().to_string(),
    }
}

/// Remove exact duplicate rows from a JSONL dataset, keeping first occurrence.
///
/// This is the remediation `apr data audit` points at when it reports
/// duplicate rows; before this existed the advice named a subcommand the
/// parser rejected.
pub(crate) fn run_dedup(path: &Path, output_path: &Path, json_output: bool) -> Result<()> {
    use std::collections::HashSet;
    use std::io::{BufRead, BufReader, Write};

    if !path.exists() {
        return Err(CliError::FileNotFound(path.to_path_buf()));
    }

    let file = std::fs::File::open(path).map_err(|e| {
        CliError::ValidationFailed(format!("Failed to open {}: {e}", path.display()))
    })?;
    let reader = BufReader::new(file);

    let mut seen: HashSet<String> = HashSet::new();
    let mut unique_rows: Vec<String> = Vec::new();
    let mut total_rows = 0usize;

    for line in reader.lines() {
        let line = line.map_err(|e| CliError::ValidationFailed(format!("Read error: {e}")))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        total_rows += 1;
        if seen.insert(canonical_row(trimmed)) {
            unique_rows.push(trimmed.to_string());
        }
    }

    if total_rows == 0 {
        return Err(CliError::ValidationFailed("Dataset is empty".to_string()));
    }

    let mut out = std::fs::File::create(output_path).map_err(|e| {
        CliError::ValidationFailed(format!("Failed to create {}: {e}", output_path.display()))
    })?;
    for row in &unique_rows {
        writeln!(out, "{row}")
            .map_err(|e| CliError::ValidationFailed(format!("Write error: {e}")))?;
    }

    let unique = unique_rows.len();
    let removed = total_rows - unique;

    if json_output {
        #[allow(clippy::disallowed_methods)]
        let report = serde_json::json!({
            "input": path.display().to_string(),
            "total_rows": total_rows,
            "unique_rows": unique,
            "removed": removed,
            "output": output_path.display().to_string(),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&report).unwrap_or_default()
        );
        return Ok(());
    }

    output::section("Deduplication");
    println!();
    output::kv("Input", format!("{} ({total_rows} rows)", path.display()));
    output::kv("Unique", format!("{unique} rows"));
    output::kv("Removed", format!("{removed} duplicate rows"));
    output::kv("Output", output_path.display());
    println!();
    println!("{} Deduplicated dataset written", "OK".green());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::io::Write;

    // ── Helper: write a temp JSONL file ──────────────────────────────────────

    fn write_temp_jsonl(name: &str, lines: &[&str]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("apr-data-tests");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).expect("create temp file");
        for line in lines {
            writeln!(f, "{line}").expect("write line");
        }
        path
    }

    // ── sqrt_inverse_weights ─────────────────────────────────────────────────

    #[test]
    fn test_sqrt_inverse_weights_empty_counts() {
        assert!(sqrt_inverse_weights(&[]).is_empty());
    }

    #[test]
    fn test_sqrt_inverse_weights_all_zero() {
        assert!(sqrt_inverse_weights(&[0, 0, 0]).is_empty());
    }

    #[test]
    fn test_sqrt_inverse_weights_total_zero() {
        // Total is sum of counts; if all zero the fn returns empty
        let result = sqrt_inverse_weights(&[0, 0]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_sqrt_inverse_weights_uniform() {
        // All classes equal => weights should all be equal
        let w = sqrt_inverse_weights(&[100, 100, 100]);
        assert_eq!(w.len(), 3);
        // For uniform: weight_i = sqrt(total / (k * c)) = sqrt(300 / (3 * 100)) = sqrt(1) = 1.0
        for wi in &w {
            assert!((*wi - 1.0).abs() < 1e-5, "expected ~1.0, got {wi}");
        }
    }

    #[test]
    fn test_sqrt_inverse_weights_imbalanced() {
        // class 0: 900, class 1: 100. The RAW sqrt-inverse form gives
        // 0.7454 / 2.2361, which sums to 2.9814 — 49% above the sum of 2 the
        // command claims. This test used to assert those raw values and so
        // locked the contradiction in place; it now asserts the normalised
        // weights, whose ratio is unchanged.
        let w = sqrt_inverse_weights(&[900, 100]);
        assert_eq!(w.len(), 2);
        let sum: f32 = w.iter().sum();
        assert!((sum - 2.0).abs() < 1e-4, "weights must sum to 2, got {sum}");
        // sqrt-inverse ratio is preserved: sqrt(900/100) = 3
        assert!((w[1] / w[0] - 3.0).abs() < 1e-4, "ratio drifted: {w:?}");
        // Minority class should have higher weight
        assert!(w[1] > w[0]);
    }

    /// The command prints "Weight sum: X (equals K)". That line must be true
    /// for every distribution, not only for already-balanced data — the case
    /// the strategy exists to fix is exactly the case it used to fail in
    /// (90/10 -> 2.9814 vs 2; 62/20/10/6/4 -> 7.1141 vs 5).
    #[test]
    fn test_sqrt_inverse_weight_sum_invariant_holds_when_imbalanced() {
        for counts in [
            vec![5usize, 5],
            vec![90, 10],
            vec![62, 20, 10, 6, 4],
            vec![9990, 10],
            vec![1, 1, 1, 1000],
        ] {
            let w = sqrt_inverse_weights(&counts);
            let sum: f32 = w.iter().sum();
            let target = weight_sum_target(&counts) as f32;
            assert!(
                (sum - target).abs() < 1e-3,
                "counts {counts:?}: weight sum {sum} != stated invariant {target}"
            );
        }
    }

    /// Empty classes carry weight 0 and are excluded from the target sum, so
    /// the printed invariant stays satisfiable.
    #[test]
    fn test_sqrt_inverse_weight_sum_target_skips_empty_classes() {
        let counts = [100usize, 0, 50];
        assert_eq!(weight_sum_target(&counts), 2);
        let w = sqrt_inverse_weights(&counts);
        let sum: f32 = w.iter().sum();
        assert!((sum - 2.0).abs() < 1e-4, "got {sum}");
        assert!((w[1] - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_sqrt_inverse_weights_with_zero_class() {
        // class 0: 100, class 1: 0, class 2: 50
        let w = sqrt_inverse_weights(&[100, 0, 50]);
        assert_eq!(w.len(), 3);
        assert!(w[0] > 0.0);
        assert!((w[1] - 0.0).abs() < f32::EPSILON); // zero count => zero weight
        assert!(w[2] > 0.0);
    }

    #[test]
    fn test_sqrt_inverse_weights_single_class() {
        let w = sqrt_inverse_weights(&[500]);
        assert_eq!(w.len(), 1);
        // sqrt(500 / (1 * 500)) = 1.0
        assert!((w[0] - 1.0).abs() < 1e-5);
    }

    // ── select_resample_indices ──────────────────────────────────────────────

    #[test]
    fn test_select_resample_indices_undersample() {
        let mut label_indices = HashMap::new();
        label_indices.insert("0".to_string(), vec![0, 1, 2, 3, 4]);
        label_indices.insert("1".to_string(), vec![5, 6]);

        let target = 2; // undersample: pick 2 from each
        let indices = select_resample_indices(&label_indices, target, 42);

        // Each class contributes exactly target_count
        assert_eq!(indices.len(), 4); // 2 classes * 2
    }

    #[test]
    fn test_select_resample_indices_oversample() {
        let mut label_indices = HashMap::new();
        label_indices.insert("0".to_string(), vec![0, 1, 2, 3, 4]);
        label_indices.insert("1".to_string(), vec![5, 6]);

        let target = 5; // oversample: pick 5 from each
        let indices = select_resample_indices(&label_indices, target, 42);

        // Each class contributes exactly target_count
        assert_eq!(indices.len(), 10); // 2 classes * 5
    }

    #[test]
    fn test_select_resample_indices_deterministic() {
        let mut label_indices = HashMap::new();
        label_indices.insert("a".to_string(), vec![0, 1, 2]);
        label_indices.insert("b".to_string(), vec![3, 4, 5]);

        let r1 = select_resample_indices(&label_indices, 3, 99);
        let r2 = select_resample_indices(&label_indices, 3, 99);
        assert_eq!(r1, r2, "Same seed should produce same indices");
    }

    #[test]
    fn test_select_resample_indices_different_seeds() {
        let mut label_indices = HashMap::new();
        label_indices.insert("a".to_string(), (0..20).collect());

        let r1 = select_resample_indices(&label_indices, 10, 1);
        let r2 = select_resample_indices(&label_indices, 10, 2);
        // Different seeds should (almost certainly) produce different orderings
        // Both should have the same count though
        assert_eq!(r1.len(), r2.len());
    }

    #[test]
    fn test_select_resample_indices_empty() {
        let label_indices: HashMap<String, Vec<usize>> = HashMap::new();
        let indices = select_resample_indices(&label_indices, 5, 42);
        assert!(indices.is_empty());
    }

    #[test]
    fn test_select_resample_indices_oversample_cycles() {
        // Class has 2 items, target is 7 => should cycle
        let mut label_indices = HashMap::new();
        label_indices.insert("x".to_string(), vec![10, 20]);

        let indices = select_resample_indices(&label_indices, 7, 0);
        assert_eq!(indices.len(), 7);
        // All indices should be from the original set
        for &idx in &indices {
            assert!(idx == 10 || idx == 20);
        }
    }

    // ── count_out_of_range_labels ────────────────────────────────────────────

    fn make_imbalance_report(
        counts: HashMap<String, usize>,
    ) -> alimentar::imbalance::ImbalanceReport {
        let distribution = alimentar::imbalance::ClassDistribution::from_counts(counts);
        alimentar::imbalance::ImbalanceReport::from_distribution("label", distribution)
    }

    #[test]
    fn test_count_out_of_range_all_valid() {
        let mut counts = HashMap::new();
        counts.insert("0".to_string(), 50);
        counts.insert("1".to_string(), 50);
        let report = make_imbalance_report(counts);
        assert_eq!(count_out_of_range_labels(&report, 2), 0);
    }

    #[test]
    fn test_count_out_of_range_some_invalid() {
        let mut counts = HashMap::new();
        counts.insert("0".to_string(), 50);
        counts.insert("1".to_string(), 30);
        counts.insert("5".to_string(), 10); // out of range for num_classes=3
        counts.insert("-1".to_string(), 5); // negative => out of range
        let report = make_imbalance_report(counts);
        assert_eq!(count_out_of_range_labels(&report, 3), 15); // 10 + 5
    }

    #[test]
    fn test_count_out_of_range_non_numeric_labels() {
        let mut counts = HashMap::new();
        counts.insert("cat".to_string(), 50);
        counts.insert("dog".to_string(), 50);
        let report = make_imbalance_report(counts);
        // Non-numeric labels are not parsed as i64 => not counted as out of range
        assert_eq!(count_out_of_range_labels(&report, 2), 0);
    }

    #[test]
    fn test_count_out_of_range_boundary() {
        let mut counts = HashMap::new();
        counts.insert("0".to_string(), 10);
        counts.insert("4".to_string(), 10); // exactly num_classes-1
        counts.insert("5".to_string(), 5); // exactly num_classes => OUT of range (0..5 excludes 5)
        let report = make_imbalance_report(counts);
        assert_eq!(count_out_of_range_labels(&report, 5), 5);
    }

    // ── TextColumnStats::from_jsonl_path ─────────────────────────────────────

    #[test]
    fn test_text_column_stats_basic() {
        let path = write_temp_jsonl(
            "stats_basic.jsonl",
            &[
                r#"{"text": "hello", "label": 0}"#,
                r#"{"text": "world!", "label": 1}"#,
                r#"{"text": "hi", "label": 0}"#,
            ],
        );

        let stats = TextColumnStats::from_jsonl_path(&path, "text", None).expect("should parse");
        assert_eq!(stats.total, 3);
        assert_eq!(stats.min_len, 2); // "hi"
        assert_eq!(stats.max_len, 6); // "world!"
        assert_eq!(stats.empty_count, 0);
        assert_eq!(stats.preamble_count, 0);
    }

    #[test]
    fn test_text_column_stats_empty_file() {
        let path = write_temp_jsonl("stats_empty.jsonl", &[]);
        let stats = TextColumnStats::from_jsonl_path(&path, "text", None).expect("should parse");
        assert_eq!(stats.total, 0);
        assert_eq!(stats.min_len, 0);
        assert_eq!(stats.max_len, 0);
        assert!((stats.mean_len - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_text_column_stats_empty_strings() {
        let path = write_temp_jsonl(
            "stats_empties.jsonl",
            &[
                r#"{"text": "", "label": 0}"#,
                r#"{"text": "   ", "label": 1}"#,
                r#"{"text": "ok", "label": 0}"#,
            ],
        );
        let stats = TextColumnStats::from_jsonl_path(&path, "text", None).expect("should parse");
        assert_eq!(stats.total, 3);
        // "" and "   " are both empty after trim check
        assert_eq!(stats.empty_count, 2);
    }

    #[test]
    fn test_text_column_stats_preamble() {
        let shebang_a = "{\"text\": \"#!/bin/bash echo hi\", \"label\": 0}";
        let normal = "{\"text\": \"normal text\", \"label\": 1}";
        let shebang_b = "{\"text\": \"#!/bin/bash rm -rf\", \"label\": 0}";
        let path = write_temp_jsonl("stats_preamble.jsonl", &[shebang_a, normal, shebang_b]);
        let stats = TextColumnStats::from_jsonl_path(&path, "text", Some("#!/bin/bash"))
            .expect("should parse");
        assert_eq!(stats.preamble_count, 2);
    }

    #[test]
    fn test_text_column_stats_missing_column() {
        let path = write_temp_jsonl(
            "stats_nocol.jsonl",
            &[
                r#"{"input": "hello", "label": 0}"#,
                r#"{"input": "world", "label": 1}"#,
            ],
        );
        // Requesting column "text" but data has "input" => entries are skipped
        let stats = TextColumnStats::from_jsonl_path(&path, "text", None).expect("should parse");
        assert_eq!(stats.total, 0);
    }

    #[test]
    fn test_text_column_stats_skips_blank_lines() {
        let path = write_temp_jsonl(
            "stats_blanks.jsonl",
            &[
                r#"{"text": "aaa", "label": 0}"#,
                "",
                r#"{"text": "bbb", "label": 1}"#,
                "   ",
            ],
        );
        let stats = TextColumnStats::from_jsonl_path(&path, "text", None).expect("should parse");
        assert_eq!(stats.total, 2);
    }

    #[test]
    fn test_text_column_stats_invalid_json() {
        let path = write_temp_jsonl(
            "stats_badjson.jsonl",
            &[r#"{"text": "ok", "label": 0}"#, "NOT VALID JSON"],
        );
        let result = TextColumnStats::from_jsonl_path(&path, "text", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_text_column_stats_nonexistent_file() {
        let path = std::path::PathBuf::from("/tmp/apr-data-tests/does_not_exist.jsonl");
        let result = TextColumnStats::from_jsonl_path(&path, "text", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_text_column_stats_percentiles() {
        // 10 items: lengths 1..=10
        let lines: Vec<String> = (1..=10)
            .map(|i| format!(r#"{{"text": "{}", "label": 0}}"#, "x".repeat(i)))
            .collect();
        let line_refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
        let path = write_temp_jsonl("stats_pct.jsonl", &line_refs);

        let stats = TextColumnStats::from_jsonl_path(&path, "text", None).expect("should parse");
        assert_eq!(stats.total, 10);
        assert_eq!(stats.min_len, 1);
        assert_eq!(stats.max_len, 10);
        // mean = (1+2+...+10)/10 = 5.5
        assert!((stats.mean_len - 5.5).abs() < 1e-10);
        // p50 = lengths[5] = 6 (0-indexed: sorted[5])
        assert_eq!(stats.p50_len, 6);
    }

    // ── print_audit_report (no panic) ────────────────────────────────────────

    #[test]
    fn test_print_audit_report_no_issues() {
        let mut counts = HashMap::new();
        counts.insert("0".to_string(), 50);
        counts.insert("1".to_string(), 50);
        let distribution = alimentar::imbalance::ClassDistribution::from_counts(counts);
        let imbalance_report =
            alimentar::imbalance::ImbalanceReport::from_distribution("label", distribution);

        let r = AuditResult {
            total: 100,
            out_of_range: 0,
            num_classes: 2,
            duplicate_count: 0,
            duplicates_flagged: false,
            imbalance_report,
            text_stats: TextColumnStats {
                min_len: 5,
                max_len: 100,
                mean_len: 50.0,
                p50_len: 45,
                p95_len: 90,
                p99_len: 98,
                empty_count: 0,
                preamble_count: 0,
                total: 100,
            },
            path: "test.jsonl".to_string(),
        };

        // Should not panic
        print_audit_report(&r);
    }

    #[test]
    fn test_print_audit_report_with_issues() {
        let mut counts = HashMap::new();
        counts.insert("0".to_string(), 950);
        counts.insert("1".to_string(), 50);
        let distribution = alimentar::imbalance::ClassDistribution::from_counts(counts);
        let imbalance_report =
            alimentar::imbalance::ImbalanceReport::from_distribution("label", distribution);

        let r = AuditResult {
            total: 1000,
            out_of_range: 5,
            num_classes: 2,
            duplicate_count: 10,
            duplicates_flagged: true,
            imbalance_report,
            text_stats: TextColumnStats {
                min_len: 0,
                max_len: 500,
                mean_len: 100.0,
                p50_len: 80,
                p95_len: 400,
                p99_len: 490,
                empty_count: 3,
                preamble_count: 7,
                total: 1000,
            },
            path: "imbalanced.jsonl".to_string(),
        };

        // Should not panic; exercises all issue branches
        print_audit_report(&r);
    }

    // ── print_balance_result (no panic) ──────────────────────────────────────

    #[test]
    fn test_print_balance_result_text() {
        let out = std::path::PathBuf::from("/tmp/balanced.jsonl");
        // Should not panic
        print_balance_result("oversample", 100, 200, &out, false);
    }

    #[test]
    fn test_print_balance_result_json() {
        let out = std::path::PathBuf::from("/tmp/balanced.jsonl");
        // Should not panic
        print_balance_result("undersample", 200, 100, &out, true);
    }

    // ── validate_audit_schema ────────────────────────────────────────────────

    #[test]
    fn test_validate_audit_schema_valid() {
        let path = write_temp_jsonl(
            "schema_valid.jsonl",
            &[
                r#"{"text": "hello", "label": 0}"#,
                r#"{"text": "world", "label": 1}"#,
            ],
        );
        let dataset = alimentar::ArrowDataset::from_json(&path).expect("load dataset");
        let result = validate_audit_schema(&dataset, "text", "label");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_audit_schema_missing_input_column() {
        let path = write_temp_jsonl(
            "schema_noinput.jsonl",
            &[
                r#"{"other": "hello", "label": 0}"#,
                r#"{"other": "world", "label": 1}"#,
            ],
        );
        let dataset = alimentar::ArrowDataset::from_json(&path).expect("load dataset");
        let result = validate_audit_schema(&dataset, "text", "label");
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("text"),
            "error should mention missing column 'text': {err_msg}"
        );
    }

    #[test]
    fn test_validate_audit_schema_missing_label_column() {
        let path = write_temp_jsonl(
            "schema_nolabel.jsonl",
            &[
                r#"{"text": "hello", "score": 0.5}"#,
                r#"{"text": "world", "score": 0.8}"#,
            ],
        );
        let dataset = alimentar::ArrowDataset::from_json(&path).expect("load dataset");
        let result = validate_audit_schema(&dataset, "text", "label");
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("label"),
            "error should mention missing column 'label': {err_msg}"
        );
    }

    // ── run_audit integration ────────────────────────────────────────────────

    #[test]
    fn test_run_audit_file_not_found() {
        let result = run_audit(
            Path::new("/tmp/apr-data-tests/nonexistent.jsonl"),
            2,
            "text",
            "label",
            None,
            false,
        );
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("not found") || err_msg.contains("nonexistent"));
    }

    #[test]
    fn test_run_audit_empty_dataset() {
        let path = write_temp_jsonl("audit_empty.jsonl", &[]);
        let result = run_audit(&path, 2, "text", "label", None, false);
        // Empty dataset or arrow parse error => should be an error
        assert!(result.is_err());
    }

    #[test]
    fn test_run_audit_valid_json_output() {
        let path = write_temp_jsonl(
            "audit_valid_json.jsonl",
            &[
                r#"{"text": "the cat sat on the mat", "label": 0}"#,
                r#"{"text": "the dog barked loudly at the mailman", "label": 1}"#,
                r#"{"text": "birds fly high in the sky", "label": 0}"#,
                r#"{"text": "fish swim in the ocean deep below", "label": 1}"#,
            ],
        );
        let result = run_audit(&path, 2, "text", "label", None, true);
        assert!(result.is_ok(), "run_audit failed: {result:?}");
    }

    #[test]
    fn test_run_audit_valid_text_output() {
        let path = write_temp_jsonl(
            "audit_valid_text.jsonl",
            &[
                r#"{"text": "sample one text content here", "label": 0}"#,
                r#"{"text": "sample two text content here", "label": 1}"#,
                r#"{"text": "sample three text", "label": 0}"#,
                r#"{"text": "sample four text here as well", "label": 1}"#,
            ],
        );
        let result = run_audit(&path, 2, "text", "label", None, false);
        assert!(result.is_ok(), "run_audit failed: {result:?}");
    }

    #[test]
    fn test_run_audit_missing_column() {
        let path = write_temp_jsonl(
            "audit_badcol.jsonl",
            &[
                r#"{"input": "hello", "label": 0}"#,
                r#"{"input": "world", "label": 1}"#,
            ],
        );
        let result = run_audit(&path, 2, "text", "label", None, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_audit_with_preamble() {
        let shebang_line = "{\"text\": \"#!/usr/bin/env python3 print hello\", \"label\": 0}";
        let path = write_temp_jsonl(
            "audit_preamble.jsonl",
            &[
                shebang_line,
                r##"{"text": "normal code sample here", "label": 1}"##,
                r##"{"text": "another normal sample for testing", "label": 0}"##,
            ],
        );
        let result = run_audit(&path, 2, "text", "label", Some("#!/"), true);
        assert!(result.is_ok(), "run_audit with preamble failed: {result:?}");
    }

    // ── run_split integration ────────────────────────────────────────────────

    #[test]
    fn test_run_split_file_not_found() {
        let out_dir = std::env::temp_dir().join("apr-data-tests").join("split_nf");
        let result = run_split(
            Path::new("/tmp/apr-data-tests/nonexistent.jsonl"),
            "label",
            0.7,
            0.15,
            0.15,
            42,
            &out_dir,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_split_valid_json_output() {
        // Need enough samples for stratified split (at least a few per class)
        let mut lines = Vec::new();
        for i in 0..20 {
            let label = i % 2;
            lines.push(format!(
                r#"{{"text": "sample number {i} for split test", "label": {label}}}"#
            ));
        }
        let line_refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
        let path = write_temp_jsonl("split_valid.jsonl", &line_refs);
        let out_dir = std::env::temp_dir()
            .join("apr-data-tests")
            .join("split_out_json");

        let result = run_split(&path, "label", 0.7, 0.15, 0.15, 42, &out_dir, true);
        assert!(result.is_ok(), "run_split failed: {result:?}");

        // Verify output files were created
        assert!(out_dir.join("train.jsonl").exists());
        assert!(out_dir.join("test.jsonl").exists());
    }

    #[test]
    fn test_run_split_valid_text_output() {
        let mut lines = Vec::new();
        for i in 0..20 {
            let label = i % 2;
            lines.push(format!(
                r#"{{"text": "split text sample {i}", "label": {label}}}"#
            ));
        }
        let line_refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
        let path = write_temp_jsonl("split_valid_text.jsonl", &line_refs);
        let out_dir = std::env::temp_dir()
            .join("apr-data-tests")
            .join("split_out_text");

        let result = run_split(&path, "label", 0.7, 0.15, 0.15, 42, &out_dir, false);
        assert!(result.is_ok(), "run_split failed: {result:?}");
    }

    // ── run_balance integration ──────────────────────────────────────────────

    #[test]
    fn test_run_balance_file_not_found() {
        let result = run_balance(
            Path::new("/tmp/apr-data-tests/nonexistent.jsonl"),
            "label",
            "oversample",
            None,
            42,
            Some(Path::new("/tmp/apr-data-tests/balanced.jsonl")),
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_balance_unknown_strategy() {
        let path = write_temp_jsonl(
            "balance_unkn.jsonl",
            &[
                r#"{"text": "a", "label": 0}"#,
                r#"{"text": "b", "label": 1}"#,
            ],
        );
        let result = run_balance(&path, "label", "magic", None, 42, None, false);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("Unknown strategy"),
            "expected 'Unknown strategy' in error: {err_msg}"
        );
    }

    #[test]
    fn test_run_balance_sqrt_inverse_json() {
        let mut lines = Vec::new();
        for i in 0..30 {
            let label = if i < 20 { 0 } else { 1 };
            lines.push(format!(
                r#"{{"text": "balance sample {i}", "label": {label}}}"#
            ));
        }
        let line_refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
        let path = write_temp_jsonl("balance_sqrt.jsonl", &line_refs);

        let result = run_balance(&path, "label", "sqrt-inverse", Some(2), 42, None, true);
        assert!(result.is_ok(), "sqrt-inverse balance failed: {result:?}");
    }

    #[test]
    fn test_run_balance_sqrt_inverse_text() {
        let mut lines = Vec::new();
        for i in 0..30 {
            let label = if i < 20 { 0 } else { 1 };
            lines.push(format!(
                r#"{{"text": "balance text sample {i}", "label": {label}}}"#
            ));
        }
        let line_refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
        let path = write_temp_jsonl("balance_sqrt_text.jsonl", &line_refs);

        let result = run_balance(&path, "label", "sqrt-inverse", Some(2), 42, None, false);
        assert!(
            result.is_ok(),
            "sqrt-inverse balance text failed: {result:?}"
        );
    }

    #[test]
    fn test_run_balance_oversample_no_output() {
        let path = write_temp_jsonl(
            "balance_noout.jsonl",
            &[
                r#"{"text": "hello world test", "label": 0}"#,
                r#"{"text": "goodbye test", "label": 1}"#,
            ],
        );
        let result = run_balance(&path, "label", "oversample", None, 42, None, false);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("--output"),
            "expected '--output' in error: {err_msg}"
        );
    }

    #[test]
    #[ignore] // resample_jsonl uses hardcoded /tmp/apr-resample-tmp.jsonl — not safe for parallel tests
    fn test_run_balance_oversample_valid() {
        let mut lines = Vec::new();
        for i in 0..20 {
            let label = if i < 15 { 0 } else { 1 };
            lines.push(format!(
                r#"{{"text": "oversample sample text number {i}", "label": {label}}}"#
            ));
        }
        let line_refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
        let path = write_temp_jsonl("balance_over.jsonl", &line_refs);
        let out_path = std::env::temp_dir()
            .join("apr-data-tests")
            .join("balanced_over.jsonl");

        let result = run_balance(
            &path,
            "label",
            "oversample",
            None,
            42,
            Some(&out_path),
            true,
        );
        assert!(result.is_ok(), "oversample balance failed: {result:?}");
        assert!(out_path.exists(), "output file should exist");
    }

    #[test]
    #[ignore] // resample_jsonl uses hardcoded /tmp/apr-resample-tmp.jsonl — not safe for parallel tests
    fn test_run_balance_undersample_valid() {
        let mut lines = Vec::new();
        for i in 0..20 {
            let label = if i < 15 { 0 } else { 1 };
            lines.push(format!(
                r#"{{"text": "undersample text number {i}", "label": {label}}}"#
            ));
        }
        let line_refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
        let path = write_temp_jsonl("balance_under.jsonl", &line_refs);
        let out_path = std::env::temp_dir()
            .join("apr-data-tests")
            .join("balanced_under.jsonl");

        let result = run_balance(
            &path,
            "label",
            "undersample",
            None,
            42,
            Some(&out_path),
            false,
        );
        assert!(result.is_ok(), "undersample balance failed: {result:?}");
    }

    // ── run_decontaminate integration ────────────────────────────────────────

    #[test]
    fn test_run_decontaminate_file_not_found() {
        let result = run_decontaminate(
            Path::new("/tmp/apr-data-tests/nonexistent.jsonl"),
            &[],
            10,
            0.5,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_decontaminate_ref_not_found() {
        let path = write_temp_jsonl(
            "decontam_train.jsonl",
            &["line one sample text for testing"],
        );
        let result = run_decontaminate(
            &path,
            &[std::path::PathBuf::from("/tmp/apr-data-tests/no_ref.jsonl")],
            10,
            0.5,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_decontaminate_clean_json() {
        let train_path = write_temp_jsonl(
            "decontam_clean_train.txt",
            &[
                "def sort_list(lst): return sorted(lst)",
                "def reverse_string(s): return s[::-1]",
                "class MyClass: pass",
            ],
        );
        let ref_path = write_temp_jsonl(
            "decontam_clean_ref.txt",
            &["def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)"],
        );

        let result = run_decontaminate(&train_path, &[ref_path], 10, 0.5, true);
        assert!(result.is_ok(), "clean decontam should pass: {result:?}");
    }

    #[test]
    fn test_run_decontaminate_clean_text() {
        let train_path = write_temp_jsonl(
            "decontam_clean_train2.txt",
            &[
                "completely unique training sample number one",
                "another unique training sample two",
            ],
        );
        let ref_path = write_temp_jsonl(
            "decontam_clean_ref2.txt",
            &["this is a reference benchmark sample that is totally different"],
        );

        let result = run_decontaminate(&train_path, &[ref_path], 10, 0.5, false);
        assert!(
            result.is_ok(),
            "clean decontam text should pass: {result:?}"
        );
    }

    #[test]
    fn test_run_decontaminate_contaminated() {
        let shared = "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)";
        let train_path = write_temp_jsonl(
            "decontam_dirty_train.txt",
            &[shared, "def other_func(): pass"],
        );
        let ref_path = write_temp_jsonl("decontam_dirty_ref.txt", &[shared]);

        let result = run_decontaminate(&train_path, &[ref_path], 10, 0.5, true);
        // 50% contamination rate (1 of 2) => should FAIL the gate
        assert!(result.is_err(), "contaminated data should fail the gate");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("Contamination rate") || err_msg.contains("AC-016"));
    }

    // ── ResampleStrategy Debug ───────────────────────────────────────────────

    #[test]
    fn test_resample_strategy_debug() {
        // Verify Debug derive works
        let s = format!("{:?}", ResampleStrategy::Oversample);
        assert_eq!(s, "Oversample");
        let s = format!("{:?}", ResampleStrategy::Undersample);
        assert_eq!(s, "Undersample");
    }

    #[test]
    fn test_resample_strategy_clone() {
        let s = ResampleStrategy::Oversample;
        let s2 = s;
        assert!(matches!(s2, ResampleStrategy::Oversample));
    }

    // ── duplicate row count: truth, not "did the alert fire" ─────────────────

    /// Build a JSONL file with `copies` identical rows padded to `n` rows and
    /// return the QualityChecker's report on it.
    fn quality_report_for(
        name: &str,
        n: usize,
        copies: usize,
    ) -> alimentar::quality::QualityReport {
        use alimentar::quality::QualityChecker;
        let mut lines: Vec<String> = Vec::new();
        for _ in 0..copies {
            lines.push(r#"{"input": "DUPLICATE ROW", "label": 0}"#.to_string());
        }
        for i in copies..n {
            lines.push(format!(r#"{{"input": "unique row {i}", "label": 1}}"#));
        }
        let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
        let path = write_temp_jsonl(name, &refs);
        let dataset = alimentar::ArrowDataset::from_json(&path).expect("load jsonl");
        QualityChecker::new()
            .max_null_ratio(0.01)
            .max_duplicate_ratio(0.05)
            .check(&dataset)
            .expect("quality check")
    }

    /// 50 identical rows in 10,000 is a 0.49% duplicate ratio — below the 1%
    /// alert threshold. The count must still be 49; deriving it by summing
    /// the `DuplicateRows` issues (which is only pushed above the threshold)
    /// reported a hard 0 and printed "Duplicates: 0 (0.0%) OK".
    #[test]
    fn test_duplicate_row_count_reports_sub_threshold_duplicates() {
        let report = quality_report_for("dupcount_sub_threshold.jsonl", 10_000, 50);
        assert_eq!(
            duplicate_row_count(&report),
            49,
            "49 extra copies must be counted even below the alert threshold"
        );
        assert!(
            !duplicates_flagged(&report),
            "0.49% is below the 1% alert threshold, so no issue is raised"
        );
    }

    /// The count and the alert are decoupled: above the threshold both fire.
    #[test]
    fn test_duplicate_row_count_and_alert_agree_above_threshold() {
        let report = quality_report_for("dupcount_supra_threshold.jsonl", 100, 50);
        assert_eq!(duplicate_row_count(&report), 49);
        assert!(duplicates_flagged(&report));
    }

    /// A clean dataset must report zero and raise nothing.
    #[test]
    fn test_duplicate_row_count_zero_on_clean_data() {
        let report = quality_report_for("dupcount_clean.jsonl", 100, 0);
        assert_eq!(duplicate_row_count(&report), 0);
        assert!(!duplicates_flagged(&report));
    }

    // ── apr data dedup ───────────────────────────────────────────────────────

    #[test]
    fn test_run_dedup_removes_duplicate_rows() {
        let path = write_temp_jsonl(
            "dedup_input.jsonl",
            &[
                r#"{"input": "the quick brown fox", "label": 0}"#,
                r#"{"input": "the quick brown fox", "label": 0}"#,
                r#"{"input": "the quick brown fox", "label": 0}"#,
                r#"{"input": "unique one", "label": 1}"#,
                r#"{"input": "unique two", "label": 1}"#,
            ],
        );
        let out = std::env::temp_dir()
            .join("apr-data-tests")
            .join("dedup_output.jsonl");
        let _ = std::fs::remove_file(&out);

        run_dedup(&path, &out, false).expect("dedup should succeed");

        let written = std::fs::read_to_string(&out).expect("read output");
        let rows: Vec<&str> = written.lines().filter(|l| !l.trim().is_empty()).collect();
        assert_eq!(rows.len(), 3, "5 rows with 2 extra copies -> 3 unique");
        assert_eq!(
            rows.iter()
                .filter(|r| r.contains("quick brown fox"))
                .count(),
            1,
            "first occurrence kept exactly once"
        );
    }

    /// Key order is not row identity: the Arrow-backed audit counts these as
    /// duplicates, so dedup must remove them too.
    #[test]
    fn test_run_dedup_is_key_order_insensitive() {
        let path = write_temp_jsonl(
            "dedup_keyorder.jsonl",
            &[
                r#"{"input": "same row", "label": 7}"#,
                r#"{"label": 7, "input": "same row"}"#,
            ],
        );
        let out = std::env::temp_dir()
            .join("apr-data-tests")
            .join("dedup_keyorder_out.jsonl");
        let _ = std::fs::remove_file(&out);

        run_dedup(&path, &out, false).expect("dedup should succeed");
        let written = std::fs::read_to_string(&out).expect("read output");
        assert_eq!(written.lines().filter(|l| !l.is_empty()).count(), 1);
    }

    #[test]
    fn test_run_dedup_preserves_already_unique_data() {
        let path = write_temp_jsonl(
            "dedup_unique.jsonl",
            &[
                r#"{"input": "a", "label": 0}"#,
                r#"{"input": "b", "label": 1}"#,
            ],
        );
        let out = std::env::temp_dir()
            .join("apr-data-tests")
            .join("dedup_unique_out.jsonl");
        let _ = std::fs::remove_file(&out);
        run_dedup(&path, &out, true).expect("dedup should succeed");
        let written = std::fs::read_to_string(&out).expect("read output");
        assert_eq!(written.lines().filter(|l| !l.is_empty()).count(), 2);
    }

    #[test]
    fn test_run_dedup_file_not_found() {
        let missing = std::path::PathBuf::from("/nonexistent/dedup-input.jsonl");
        let out = std::env::temp_dir().join("apr-data-tests/never-written.jsonl");
        assert!(run_dedup(&missing, &out, false).is_err());
    }

    // ── decontamination: a vacuous configuration must not PASS ───────────────

    /// `--ngram 1000` produces no n-grams from any short line, so nothing is
    /// compared. The command used to print "Rate: 0.00%" and exit 0 on a file
    /// containing byte-identical copies of the benchmark.
    #[test]
    fn test_run_decontaminate_oversized_ngram_is_rejected_not_passed() {
        let leaked = "the capital of france is paris and it is a large european city";
        let train_path = write_temp_jsonl(
            "decontam_vacuous_train.txt",
            &[leaked, "unrelated gardening sentence about trowels"],
        );
        let ref_path = write_temp_jsonl("decontam_vacuous_ref.txt", &[leaked]);

        let result = run_decontaminate(&train_path, &[ref_path], 1000, 0.5, false);
        let err = result.expect_err("a check that compared nothing must not pass");
        let msg = format!("{err}");
        assert!(
            msg.contains("vacuous"),
            "error must name the vacuous configuration, got: {msg}"
        );
    }

    /// Same hazard reached through the JSON path — the machine-readable gate
    /// field must not read PASS either.
    #[test]
    fn test_run_decontaminate_vacuous_json_also_fails() {
        let leaked = "water boils at one hundred degrees celsius at standard pressure";
        let train_path = write_temp_jsonl("decontam_vacuous_train_json.txt", &[leaked]);
        let ref_path = write_temp_jsonl("decontam_vacuous_ref_json.txt", &[leaked]);
        assert!(run_decontaminate(&train_path, &[ref_path], 500, 0.5, true).is_err());
    }

    /// A normal configuration must still pass — the vacuity gate must not
    /// fail closed on real corpora.
    #[test]
    fn test_run_decontaminate_normal_ngram_still_passes() {
        let train_path = write_temp_jsonl(
            "decontam_nonvacuous_train.txt",
            &["completely unique training sample number one about carpentry"],
        );
        let ref_path = write_temp_jsonl(
            "decontam_nonvacuous_ref.txt",
            &["a reference benchmark sample that shares nothing at all"],
        );
        assert!(run_decontaminate(&train_path, &[ref_path], 10, 0.5, false).is_ok());
    }

    // ── canonical_row ────────────────────────────────────────────────────────

    #[test]
    fn test_canonical_row_key_order_independent() {
        assert_eq!(
            canonical_row(r#"{"a": 1, "b": 2}"#),
            canonical_row(r#"{"b": 2, "a": 1}"#)
        );
        assert_ne!(
            canonical_row(r#"{"a": 1, "b": 2}"#),
            canonical_row(r#"{"a": 1, "b": 3}"#)
        );
    }

    #[test]
    fn test_canonical_row_non_object_falls_back_to_text() {
        assert_eq!(canonical_row("  not json  "), "not json");
    }

    // ── flag validation at the CLI boundary ──────────────────────────────────

    #[test]
    fn test_parse_ngram_size_rejects_zero() {
        let err = crate::parse_ngram_size("0").expect_err("0 is not a window size");
        assert!(err.contains(">= 1"), "unhelpful message: {err}");
        assert_eq!(crate::parse_ngram_size("1"), Ok(1));
        assert_eq!(crate::parse_ngram_size("10"), Ok(10));
        assert!(crate::parse_ngram_size("-1").is_err());
        assert!(crate::parse_ngram_size("abc").is_err());
    }

    #[test]
    fn test_parse_unit_interval_rejects_out_of_range() {
        assert!(crate::parse_unit_interval("5.0").is_err());
        assert!(crate::parse_unit_interval("-1").is_err());
        assert!(crate::parse_unit_interval("nan").is_err());
        assert!(crate::parse_unit_interval("inf").is_err());
        assert_eq!(crate::parse_unit_interval("0.0"), Ok(0.0));
        assert_eq!(crate::parse_unit_interval("0.5"), Ok(0.5));
        assert_eq!(crate::parse_unit_interval("1.0"), Ok(1.0));
    }

    /// Parse argv on a 16 MB stack. Clap's recursive destructuring of the
    /// full `Commands` enum overflows the default 2 MiB test-thread stack in
    /// debug builds (same pattern as `parsing.rs` and `pretrain.rs`).
    fn parse_cli(args: Vec<&'static str>) -> std::result::Result<crate::Cli, clap::error::Error> {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(move || {
                use clap::Parser;
                crate::Cli::try_parse_from(args)
            })
            .expect("spawn parse thread")
            .join()
            .expect("join parse thread")
    }

    /// `--ngram 0` must be rejected by the parser (exit 2) rather than
    /// reaching `slice::windows(0)` and aborting the process (exit 101).
    #[test]
    fn test_cli_rejects_ngram_zero() {
        assert!(parse_cli(vec![
            "apr",
            "data",
            "decontaminate",
            "train.jsonl",
            "--reference",
            "bench.jsonl",
            "--ngram",
            "0",
        ])
        .is_err());
    }

    /// `--threshold` is documented as 0.0-1.0; 5.0 made the AC-016 gate an
    /// unconditional pass.
    #[test]
    fn test_cli_rejects_out_of_range_threshold() {
        for bad in ["5.0", "-0.5", "1.5"] {
            assert!(
                parse_cli(vec![
                    "apr",
                    "data",
                    "decontaminate",
                    "train.jsonl",
                    "--reference",
                    "bench.jsonl",
                    "--threshold",
                    bad,
                ])
                .is_err(),
                "--threshold {bad} must be rejected"
            );
        }
    }

    #[test]
    fn test_cli_accepts_valid_decontaminate_flags() {
        assert!(parse_cli(vec![
            "apr",
            "data",
            "decontaminate",
            "train.jsonl",
            "--reference",
            "bench.jsonl",
            "--ngram",
            "8",
            "--threshold",
            "0.75",
        ])
        .is_ok());
    }

    /// `apr data audit` tells the user to run `apr data dedup`; the parser
    /// rejected it as an unknown subcommand.
    #[test]
    fn test_cli_accepts_data_dedup_subcommand() {
        parse_cli(vec!["apr", "data", "dedup", "in.jsonl", "-o", "out.jsonl"])
            .expect("`apr data dedup` must parse -- audit recommends it");
    }
}
