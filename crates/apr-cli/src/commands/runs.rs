//! `apr runs` — List, inspect, and compare training experiment runs (ALB-050/051)
//!
//! Reads from local SQLite experiment database (`.entrenar/experiments.db`)
//! or global registry (`~/.entrenar/experiments.db`).
//!
//! Features:
//! - Inline Unicode sparklines in table view (loss trajectory at a glance)
//! - Braille loss curves in detailed show view
//! - Side-by-side run comparison with config diff and overlaid metrics
//! - JSON output for LLM agent consumption

use crate::error::CliError;
use entrenar::storage::{ExperimentStorage, SqliteBackend};
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, CliError>;

// ─── Sparkline rendering ────────────────────────────────────────────────────

const SPARK_CHARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
const BRAILLE_BASE: u32 = 0x2800;

/// Render an inline sparkline from a slice of f64 values.
/// Returns a fixed-width string of Unicode block characters.
fn sparkline(data: &[f64], width: usize) -> String {
    if data.is_empty() {
        return "—".to_string();
    }
    let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min = data.iter().copied().fold(f64::INFINITY, f64::min);
    let range = (max - min).max(1e-10);

    // Sample data to fit width
    let n = data.len();
    let mut result = String::with_capacity(width * 4);
    for i in 0..width.min(n) {
        let idx = if n <= width {
            i
        } else {
            i * (n - 1) / (width - 1).max(1)
        };
        let val = data[idx.min(n - 1)];
        let normalized = ((val - min) / range).clamp(0.0, 1.0);
        let char_idx = ((normalized * 7.0).round() as usize).min(7);
        result.push(SPARK_CHARS[char_idx]);
    }
    result
}

/// Render a braille chart from a slice of f64 values.
/// Each character encodes a 2x4 dot matrix (2 columns, 4 rows per cell).
/// Returns multi-line string with y-axis labels.
/// Sample data points to fit braille grid, returning y-positions in dot coordinates.
fn sample_braille_data(
    data: &[f64],
    total_dots_x: usize,
    total_dots_y: usize,
    min: f64,
    range: f64,
) -> Vec<usize> {
    let n = data.len();
    let mut samples = Vec::with_capacity(total_dots_x);
    for i in 0..total_dots_x {
        let idx = if n <= total_dots_x {
            i.min(n - 1)
        } else {
            i * (n - 1) / (total_dots_x - 1).max(1)
        };
        let val = data[idx.min(n - 1)];
        let normalized = ((val - min) / range).clamp(0.0, 1.0);
        let dot_y = ((1.0 - normalized) * (total_dots_y - 1) as f64).round() as usize;
        samples.push(dot_y.min(total_dots_y - 1));
    }
    samples
}

/// Render one row of braille characters from sample data.
fn render_braille_row(
    samples: &[usize],
    row: usize,
    width: usize,
    dot_bits: &[[u8; 4]; 2],
) -> String {
    let row_start_y = row * 4;
    let mut row_chars = String::new();
    for col in 0..width {
        let mut pattern: u8 = 0;
        for dot_col in 0..2 {
            let x = col * 2 + dot_col;
            if x < samples.len() {
                let sample_y = samples[x];
                for dot_row in 0..4 {
                    let y = row_start_y + dot_row;
                    if sample_y == y {
                        pattern |= 1 << dot_bits[dot_col][dot_row];
                    }
                }
            }
        }
        let ch = char::from_u32(BRAILLE_BASE + pattern as u32).unwrap_or(' ');
        row_chars.push(ch);
    }
    row_chars
}

fn braille_chart(data: &[f64], width: usize, height: usize) -> String {
    if data.is_empty() {
        return String::new();
    }
    let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min = data.iter().copied().fold(f64::INFINITY, f64::min);
    let range = (max - min).max(1e-10);
    let rows = height;
    let total_dots_y = rows * 4;
    let total_dots_x = width * 2;

    let samples = sample_braille_data(data, total_dots_x, total_dots_y, min, range);

    let dot_bits: [[u8; 4]; 2] = [[0, 1, 2, 6], [3, 4, 5, 7]];

    let y_label_width = format!("{:.2}", max).len().max(format!("{:.2}", min).len());
    let mut lines = Vec::with_capacity(rows + 1);

    for row in 0..rows {
        let y_val = max - (row as f64 / (rows - 1).max(1) as f64) * range;
        let mut line = format!("{:>width$.2} │", y_val, width = y_label_width);
        line.push_str(&render_braille_row(&samples, row, width, &dot_bits));
        lines.push(line);
    }

    let axis_line = format!(
        "{:>width$} └{}",
        "",
        "─".repeat(width),
        width = y_label_width
    );
    lines.push(axis_line);

    let x_label = format!(
        "{:>width$}  0{:>pad$}{}",
        "",
        "",
        data.len(),
        width = y_label_width,
        pad = width.saturating_sub(format!("{}", data.len()).len()) - 1
    );
    lines.push(x_label);

    lines.join("\n")
}

/// Compute loss trend arrow from metric series.
fn loss_trend_arrow(data: &[f64]) -> &'static str {
    if data.len() < 3 {
        return "—";
    }
    let n = data.len();
    let half = n / 2;
    let first_half: f64 = data[..half].iter().sum::<f64>() / half as f64;
    let second_half: f64 = data[half..].iter().sum::<f64>() / (n - half) as f64;
    let change = (second_half - first_half) / first_half.abs().max(1e-10);
    if change < -0.02 {
        "\x1b[32m↓\x1b[0m" // green down = good
    } else if change > 0.02 {
        "\x1b[31m↑\x1b[0m" // red up = bad
    } else {
        "\x1b[33m→\x1b[0m" // yellow flat
    }
}

// ─── Status filtering ───────────────────────────────────────────────────────

/// The status names `apr runs ls --status` accepts, mapped to the stored
/// [`entrenar::storage::RunStatus`] variant they select.
///
/// The filter used to stringify the enum variant and compare it verbatim, so
/// the one non-trivial value `--help` advertises — `completed` — could never
/// match a row (the variant is `Success`), and an empty table with exit 0 was
/// indistinguishable from "you have no completed runs".
const STATUS_ALIASES: &[(&str, &str)] = &[
    ("pending", "Pending"),
    ("running", "Running"),
    ("active", "Running"),
    ("completed", "Success"),
    ("complete", "Success"),
    ("succeeded", "Success"),
    ("success", "Success"),
    ("failed", "Failed"),
    ("failure", "Failed"),
    ("cancelled", "Cancelled"),
    ("canceled", "Cancelled"),
];

/// Resolve a user-supplied `--status` value to the stored variant name.
///
/// `Ok(None)` means "no filtering" (`all`). An unrecognised value is a hard
/// error: silently returning zero rows for a typo is worse than refusing.
fn resolve_status_filter(status_filter: &str) -> Result<Option<&'static str>> {
    let needle = status_filter.trim().to_lowercase();
    if needle == "all" {
        return Ok(None);
    }
    if let Some((_, variant)) = STATUS_ALIASES.iter().find(|(alias, _)| *alias == needle) {
        return Ok(Some(variant));
    }
    Err(CliError::ValidationFailed(format!(
        "Unknown run status '{status_filter}'. Supported: all, pending, running, completed, failed, cancelled"
    )))
}

// ─── Commands ───────────────────────────────────────────────────────────────

/// List experiment runs from SQLite DB
#[provable_contracts_macros::contract(
    "apr-cli-command-safety-v1",
    equation = "read_only_no_side_effects"
)]
pub(crate) fn run_ls(
    dir: &Option<PathBuf>,
    global: bool,
    status_filter: &str,
    json: bool,
    limit: usize,
) -> Result<()> {
    // Reject an unknown status BEFORE touching the store, so a typo fails the
    // same way whether or not the user happens to have a database.
    let wanted_status = resolve_status_filter(status_filter)?;

    let store = open_store(dir, global)?;

    let experiments = store
        .list_experiments()
        .map_err(|e| CliError::ValidationFailed(format!("Failed to list experiments: {e}")))?;

    if experiments.is_empty() {
        if json {
            println!("[]");
        } else {
            println!("No experiments found.");
            println!();
            println!("Hint: Run `apr train apply --config <yaml>` to start a training run.");
            if !global {
                println!("      Use `apr runs ls --global` to check the global registry.");
            }
        }
        return Ok(());
    }

    // Collect all runs across all experiments
    let mut all_runs = Vec::new();
    for exp in &experiments {
        if let Ok(runs) = store.list_runs(&exp.id) {
            for run in runs {
                all_runs.push((exp.clone(), run));
            }
        }
    }

    // Filter by status
    if let Some(variant) = wanted_status {
        all_runs.retain(|(_, run)| format!("{:?}", run.status) == variant);
    }

    // Sort by start time descending (most recent first)
    all_runs.sort_by(|a, b| b.1.start_time.cmp(&a.1.start_time));

    // Limit
    all_runs.truncate(limit);

    if json {
        print_runs_json(&all_runs, &store);
    } else {
        print_runs_table(&all_runs, &store);
    }

    Ok(())
}

/// Show detailed run metrics with braille loss curve
pub(crate) fn run_show(
    run_id: &str,
    dir: &Option<PathBuf>,
    global: bool,
    json: bool,
) -> Result<()> {
    let store = open_store(dir, global)?;

    let run = store
        .get_run(run_id)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to get run: {e}")))?;

    let params = store.get_params(run_id).unwrap_or_default();

    // Get metrics
    let loss_metrics = store.get_metrics(run_id, "loss").unwrap_or_default();
    let lr_metrics = store
        .get_metrics(run_id, "learning_rate")
        .unwrap_or_default();
    let tps_metrics = store
        .get_metrics(run_id, "tokens_per_second")
        .unwrap_or_default();

    if json {
        print_show_json(&run, &params, &loss_metrics, &lr_metrics, &tps_metrics);
    } else {
        print_show_text(&run, &params, &loss_metrics, &lr_metrics, &tps_metrics);
    }

    Ok(())
}

/// Compare two runs side-by-side (ALB-051)
pub(crate) fn run_diff(
    run_id_a: &str,
    run_id_b: &str,
    dir: &Option<PathBuf>,
    global: bool,
    json: bool,
) -> Result<()> {
    let store = open_store(dir, global)?;

    let run_a = store
        .get_run(run_id_a)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to get run A: {e}")))?;
    let run_b = store
        .get_run(run_id_b)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to get run B: {e}")))?;

    let params_a = store.get_params(run_id_a).unwrap_or_default();
    let params_b = store.get_params(run_id_b).unwrap_or_default();

    let loss_a = store.get_metrics(run_id_a, "loss").unwrap_or_default();
    let loss_b = store.get_metrics(run_id_b, "loss").unwrap_or_default();
    let tps_a = store
        .get_metrics(run_id_a, "tokens_per_second")
        .unwrap_or_default();
    let tps_b = store
        .get_metrics(run_id_b, "tokens_per_second")
        .unwrap_or_default();
    let lr_a = store
        .get_metrics(run_id_a, "learning_rate")
        .unwrap_or_default();
    let lr_b = store
        .get_metrics(run_id_b, "learning_rate")
        .unwrap_or_default();

    if json {
        print_diff_json(
            &run_a, &run_b, &params_a, &params_b, &loss_a, &loss_b, &tps_a, &tps_b, &lr_a, &lr_b,
        );
    } else {
        print_diff_text(
            &run_a, &run_b, &params_a, &params_b, &loss_a, &loss_b, &tps_a, &tps_b, &lr_a, &lr_b,
        );
    }

    Ok(())
}

// ─── Store ──────────────────────────────────────────────────────────────────

fn open_store(dir: &Option<PathBuf>, global: bool) -> Result<SqliteBackend> {
    let db_path = if global {
        dirs::home_dir()
            .map(|h| h.join(".entrenar").join("experiments.db"))
            .ok_or_else(|| {
                CliError::ValidationFailed("Could not determine home directory".into())
            })?
    } else {
        let base = dir.as_deref().unwrap_or(Path::new("."));
        base.join(".entrenar").join("experiments.db")
    };

    if !db_path.exists() {
        return Err(CliError::ValidationFailed(format!(
            "No experiment database found at: {}\nHint: Training runs create this automatically.",
            db_path.display()
        )));
    }

    SqliteBackend::open(db_path.to_string_lossy().as_ref())
        .map_err(|e| CliError::ValidationFailed(format!("Failed to open experiment database: {e}")))
}

// ─── Table output (ls) ─────────────────────────────────────────────────────

fn print_runs_table(
    runs: &[(
        entrenar::storage::sqlite::Experiment,
        entrenar::storage::sqlite::Run,
    )],
    store: &SqliteBackend,
) {
    println!(
        "\x1b[1m{:<18} {:<18} {:>8} {:>10} {:>8} {:>8}  {}\x1b[0m",
        "EXPERIMENT", "RUN ID", "STATUS", "LOSS", "TOK/S", "TIME", "LOSS CURVE"
    );
    println!("{}", "─".repeat(100));

    for (exp, run) in runs {
        let status = format!("{:?}", run.status);
        let status_str = match run.status {
            entrenar::storage::RunStatus::Success => format!("\x1b[32m{:>8}\x1b[0m", status),
            entrenar::storage::RunStatus::Failed => format!("\x1b[31m{:>8}\x1b[0m", status),
            entrenar::storage::RunStatus::Running => format!("\x1b[33m{:>8}\x1b[0m", status),
            _ => format!("{:>8}", status),
        };

        // Get metrics
        let loss_data: Vec<f64> = store
            .get_metrics(&run.id, "loss")
            .ok()
            .map(|m| m.iter().map(|p| p.value).collect())
            .unwrap_or_default();

        let loss_str = loss_data
            .last()
            .map(|v| format!("{:.4}", v))
            .unwrap_or_else(|| "—".to_string());

        let tps_str = store
            .get_metrics(&run.id, "tokens_per_second")
            .ok()
            .and_then(|m| m.last().map(|p| format!("{:.0}", p.value)))
            .unwrap_or_else(|| "—".to_string());

        let duration_str = run
            .end_time
            .map(|end| {
                let secs = (end - run.start_time).num_seconds();
                if secs > 3600 {
                    format!("{}h{}m", secs / 3600, (secs % 3600) / 60)
                } else if secs > 60 {
                    format!("{}m{}s", secs / 60, secs % 60)
                } else {
                    format!("{}s", secs)
                }
            })
            .unwrap_or_else(|| "\x1b[33mrunning\x1b[0m".to_string());

        // Inline sparkline + trend
        let spark = if loss_data.len() >= 2 {
            format!(
                "{} {}",
                sparkline(&loss_data, 16),
                loss_trend_arrow(&loss_data)
            )
        } else {
            "—".to_string()
        };

        let exp_name = truncate_str(&exp.name, 16);
        let run_id_short = truncate_str(&run.id, 16);

        println!(
            "{:<18} {:<18} {} {:>10} {:>8} {:>8}  {}",
            exp_name, run_id_short, status_str, loss_str, tps_str, duration_str, spark
        );
    }

    println!();
    println!(
        "\x1b[2m{} run(s) | Use `apr runs show <ID>` for details | `apr runs diff <A> <B>` to compare\x1b[0m",
        runs.len()
    );
}

/// Truncate a string to `max_len` characters, adding `…` if truncated.
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() > max_len {
        format!("{}…", &s[..max_len - 1])
    } else {
        s.to_string()
    }
}

// ─── JSON output (ls) ──────────────────────────────────────────────────────

fn print_runs_json(
    runs: &[(
        entrenar::storage::sqlite::Experiment,
        entrenar::storage::sqlite::Run,
    )],
    store: &SqliteBackend,
) {
    let entries: Vec<serde_json::Value> = runs
        .iter()
        .map(|(exp, run)| {
            let loss_data: Vec<f64> = store
                .get_metrics(&run.id, "loss")
                .ok()
                .map(|m| m.iter().map(|p| p.value).collect())
                .unwrap_or_default();

            let final_tps = store
                .get_metrics(&run.id, "tokens_per_second")
                .ok()
                .and_then(|m| m.last().map(|p| p.value));

            let duration_secs = run.end_time.map(|end| (end - run.start_time).num_seconds());

            serde_json::json!({
                "experiment": exp.name,
                "run_id": run.id,
                "status": format!("{:?}", run.status),
                "start_time": run.start_time.to_rfc3339(),
                "end_time": run.end_time.map(|t| t.to_rfc3339()),
                "duration_seconds": duration_secs,
                "final_loss": loss_data.last().copied(),
                "min_loss": if loss_data.is_empty() { None } else {
                    Some(loss_data.iter().copied().fold(f64::INFINITY, f64::min))
                },
                "loss_history": loss_data,
                "tokens_per_second": final_tps,
                "steps": loss_data.len(),
            })
        })
        .collect();

    println!(
        "{}",
        serde_json::to_string_pretty(&entries).unwrap_or_else(|_| "[]".to_string())
    );
}

// ─── Show (detailed) ────────────────────────────────────────────────────────

fn print_show_json(
    run: &entrenar::storage::sqlite::Run,
    params: &std::collections::HashMap<String, entrenar::storage::ParameterValue>,
    loss_metrics: &[entrenar::storage::MetricPoint],
    lr_metrics: &[entrenar::storage::MetricPoint],
    tps_metrics: &[entrenar::storage::MetricPoint],
) {
    let mut metrics_map = serde_json::Map::new();
    if !loss_metrics.is_empty() {
        let loss_values: Vec<f64> = loss_metrics.iter().map(|p| p.value).collect();
        metrics_map.insert(
            "loss".into(),
            serde_json::json!({
                "count": loss_values.len(),
                "first": loss_values.first(),
                "last": loss_values.last(),
                "min": loss_values.iter().copied().fold(f64::INFINITY, f64::min),
                "max": loss_values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                "values": loss_values,
            }),
        );
    }
    if !tps_metrics.is_empty() {
        let tps_values: Vec<f64> = tps_metrics.iter().map(|p| p.value).collect();
        metrics_map.insert(
            "tokens_per_second".into(),
            serde_json::json!({
                "count": tps_values.len(),
                "last": tps_values.last(),
                "max": tps_values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                "mean": tps_values.iter().sum::<f64>() / tps_values.len() as f64,
                "values": tps_values,
            }),
        );
    }
    if !lr_metrics.is_empty() {
        let lr_values: Vec<f64> = lr_metrics.iter().map(|p| p.value).collect();
        metrics_map.insert(
            "learning_rate".into(),
            serde_json::json!({
                "count": lr_values.len(),
                "last": lr_values.last(),
                "values": lr_values,
            }),
        );
    }

    let params_map: std::collections::HashMap<_, _> = params
        .iter()
        .map(|(k, v)| (k.clone(), param_to_value(v)))
        .collect();

    let output = serde_json::json!({
        "run_id": run.id,
        "experiment_id": run.experiment_id,
        "status": format!("{:?}", run.status),
        "start_time": run.start_time.to_rfc3339(),
        "end_time": run.end_time.map(|t| t.to_rfc3339()),
        "duration_seconds": run.end_time.map(|end| (end - run.start_time).num_seconds()),
        "params": params_map,
        "metrics": serde_json::Value::Object(metrics_map),
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&output).unwrap_or_default()
    );
}

fn print_show_text(
    run: &entrenar::storage::sqlite::Run,
    params: &std::collections::HashMap<String, entrenar::storage::ParameterValue>,
    loss_metrics: &[entrenar::storage::MetricPoint],
    lr_metrics: &[entrenar::storage::MetricPoint],
    tps_metrics: &[entrenar::storage::MetricPoint],
) {
    // Header
    println!();
    println!("\x1b[1m  Run: {}\x1b[0m", run.id);
    println!("  Experiment: {}", run.experiment_id);

    let status_str = match run.status {
        entrenar::storage::RunStatus::Success => "\x1b[32mSuccess\x1b[0m",
        entrenar::storage::RunStatus::Failed => "\x1b[31mFailed\x1b[0m",
        entrenar::storage::RunStatus::Running => "\x1b[33mRunning\x1b[0m",
        _ => "Unknown",
    };
    println!("  Status:     {status_str}");
    println!(
        "  Started:    {}",
        run.start_time.format("%Y-%m-%d %H:%M:%S")
    );
    if let Some(end) = run.end_time {
        let duration = end - run.start_time;
        println!("  Ended:      {}", end.format("%Y-%m-%d %H:%M:%S"));
        println!(
            "  Duration:   {}",
            format_duration_long(duration.num_seconds())
        );
    }

    // Parameters
    if !params.is_empty() {
        println!();
        println!("\x1b[1m  Parameters:\x1b[0m");
        let mut sorted_params: Vec<_> = params.iter().collect();
        sorted_params.sort_by_key(|(k, _)| *k);
        for (k, v) in sorted_params {
            println!("    \x1b[36m{:<20}\x1b[0m {}", k, param_display(v));
        }
    }

    // Metrics summary
    println!();
    println!("\x1b[1m  Metrics:\x1b[0m");

    if !loss_metrics.is_empty() {
        let loss_values: Vec<f64> = loss_metrics.iter().map(|p| p.value).collect();
        let first = loss_values.first().copied().unwrap_or(0.0);
        let last = loss_values.last().copied().unwrap_or(0.0);
        let min = loss_values.iter().copied().fold(f64::INFINITY, f64::min);
        let trend = loss_trend_arrow(&loss_values);
        let change_pct = if first > 0.0 {
            ((last - first) / first) * 100.0
        } else {
            0.0
        };
        println!(
            "    Loss:        {:.4} → {:.4} {} ({:+.1}%, min: {:.4}, {} steps)",
            first,
            last,
            trend,
            change_pct,
            min,
            loss_values.len()
        );

        // Inline sparkline
        println!("    Sparkline:   {}", sparkline(&loss_values, 40));

        // Braille chart if enough data
        if loss_values.len() >= 3 {
            println!();
            println!("\x1b[1m  Loss Curve:\x1b[0m");
            let chart = braille_chart(&loss_values, 50, 6);
            for line in chart.lines() {
                println!("    {line}");
            }
        }
    } else {
        println!("    Loss:        —");
    }

    println!();
    if !tps_metrics.is_empty() {
        let tps_values: Vec<f64> = tps_metrics.iter().map(|p| p.value).collect();
        let last = tps_values.last().copied().unwrap_or(0.0);
        let max = tps_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mean = tps_values.iter().sum::<f64>() / tps_values.len() as f64;
        println!(
            "    Throughput:  {:.0} tok/s (peak: {:.0}, mean: {:.0})",
            last, max, mean
        );
        println!("    Sparkline:   {}", sparkline(&tps_values, 40));
    }

    if !lr_metrics.is_empty() {
        let lr_values: Vec<f64> = lr_metrics.iter().map(|p| p.value).collect();
        let last = lr_values.last().copied().unwrap_or(0.0);
        println!(
            "    LR:          {:.2e}  {}",
            last,
            sparkline(&lr_values, 20)
        );
    }

    println!();
}

// ─── Diff (comparison) ─────────────────────────────────────────────────────

/// Helper: extract values from metric points.
fn metric_values(points: &[entrenar::storage::MetricPoint]) -> Vec<f64> {
    points.iter().map(|p| p.value).collect()
}

/// Helper: compute mean of a non-empty slice.
fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        0.0
    } else {
        data.iter().sum::<f64>() / data.len() as f64
    }
}

/// Helper: min of a slice, or None if empty.
fn min_val(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        None
    } else {
        Some(data.iter().copied().fold(f64::INFINITY, f64::min))
    }
}

/// Helper: format an optional f64 as "X.XXXXXX" or "—".
fn fmt_loss(v: Option<f64>) -> String {
    v.map(|x| format!("{:.6}", x)).unwrap_or_else(|| "—".into())
}

/// Print a diff table row.
fn diff_row(label: &str, a: &str, b: &str) {
    println!("  {:<24} {:<20} {:<20}", label, a, b);
}

/// Compute config diff between two param sets.
fn config_diff(
    params_a: &std::collections::HashMap<String, entrenar::storage::ParameterValue>,
    params_b: &std::collections::HashMap<String, entrenar::storage::ParameterValue>,
) -> Vec<(String, String, String)> {
    let all_keys: std::collections::BTreeSet<_> = params_a.keys().chain(params_b.keys()).collect();
    let mut diffs = Vec::new();
    for key in &all_keys {
        let va = params_a.get(*key).map(param_display);
        let vb = params_b.get(*key).map(param_display);
        if va != vb {
            diffs.push((
                key.to_string(),
                va.unwrap_or_else(|| "—".into()),
                vb.unwrap_or_else(|| "—".into()),
            ));
        }
    }
    diffs
}

#[allow(clippy::too_many_arguments)]
fn print_diff_text(
    run_a: &entrenar::storage::sqlite::Run,
    run_b: &entrenar::storage::sqlite::Run,
    params_a: &std::collections::HashMap<String, entrenar::storage::ParameterValue>,
    params_b: &std::collections::HashMap<String, entrenar::storage::ParameterValue>,
    loss_a: &[entrenar::storage::MetricPoint],
    loss_b: &[entrenar::storage::MetricPoint],
    tps_a: &[entrenar::storage::MetricPoint],
    tps_b: &[entrenar::storage::MetricPoint],
    lr_a: &[entrenar::storage::MetricPoint],
    lr_b: &[entrenar::storage::MetricPoint],
) {
    let id_a = truncate_str(&run_a.id, 16);
    let id_b = truncate_str(&run_b.id, 16);

    println!();
    println!("\x1b[1m  Run Comparison\x1b[0m");
    println!("  ─────────────────────────────────────────────────────────────────");
    println!(
        "  {:<24} \x1b[36m{:<20}\x1b[0m \x1b[33m{:<20}\x1b[0m",
        "",
        format!("A: {id_a}"),
        format!("B: {id_b}")
    );
    println!("  ─────────────────────────────────────────────────────────────────");

    diff_row(
        "Status",
        &format!("{:?}", run_a.status),
        &format!("{:?}", run_b.status),
    );

    let dur_a = run_a
        .end_time
        .map(|e| format_duration_long((e - run_a.start_time).num_seconds()));
    let dur_b = run_b
        .end_time
        .map(|e| format_duration_long((e - run_b.start_time).num_seconds()));
    diff_row(
        "Duration",
        &dur_a.unwrap_or_else(|| "running".into()),
        &dur_b.unwrap_or_else(|| "running".into()),
    );

    let lv_a = metric_values(loss_a);
    let lv_b = metric_values(loss_b);
    diff_row(
        "Final loss",
        &fmt_loss(lv_a.last().copied()),
        &fmt_loss(lv_b.last().copied()),
    );
    diff_row(
        "Min loss",
        &fmt_loss(min_val(&lv_a)),
        &fmt_loss(min_val(&lv_b)),
    );
    diff_row("Steps", &lv_a.len().to_string(), &lv_b.len().to_string());

    // Verdict
    if let (Some(a), Some(b)) = (lv_a.last(), lv_b.last()) {
        let winner = if a < b {
            "\x1b[32mA wins\x1b[0m"
        } else if b < a {
            "\x1b[32mB wins\x1b[0m"
        } else {
            "tie"
        };
        let diff_pct = ((a - b) / b.abs().max(1e-10)) * 100.0;
        println!(
            "  {:<24} {} ({:+.2}% loss difference)",
            "\x1b[1mVerdict\x1b[0m", winner, diff_pct
        );
    }

    // Throughput
    let tv_a = metric_values(tps_a);
    let tv_b = metric_values(tps_b);
    if !tv_a.is_empty() || !tv_b.is_empty() {
        diff_row(
            "Mean tok/s",
            &format!("{:.0}", mean(&tv_a)),
            &format!("{:.0}", mean(&tv_b)),
        );
    }

    // Loss sparklines
    print_diff_sparklines("Loss Curves", &lv_a, &lv_b);

    // LR sparklines
    let lra = metric_values(lr_a);
    let lrb = metric_values(lr_b);
    if !lra.is_empty() || !lrb.is_empty() {
        print_diff_sparklines("LR Schedule", &lra, &lrb);
    }

    // Config diff
    let diffs = config_diff(params_a, params_b);
    if !diffs.is_empty() {
        println!();
        println!("\x1b[1m  Config Diff (changed params only):\x1b[0m");
        for (key, va, vb) in &diffs {
            println!(
                "    \x1b[36m{:<20}\x1b[0m \x1b[31m{}\x1b[0m → \x1b[32m{}\x1b[0m",
                key, va, vb
            );
        }
    } else {
        println!();
        println!("  \x1b[2mNo config differences.\x1b[0m");
    }
    println!();
}

fn print_diff_sparklines(label: &str, a: &[f64], b: &[f64]) {
    println!();
    println!("\x1b[1m  {label}:\x1b[0m");
    let sa = if a.len() >= 2 {
        sparkline(a, 40)
    } else {
        "—".to_string()
    };
    let sb = if b.len() >= 2 {
        sparkline(b, 40)
    } else {
        "—".to_string()
    };
    println!("    \x1b[36mA:\x1b[0m {sa}");
    println!("    \x1b[33mB:\x1b[0m {sb}");
}

/// Build JSON summary for one side of a diff.
fn run_summary_json(
    run: &entrenar::storage::sqlite::Run,
    loss_vals: &[f64],
    tps_vals: &[f64],
) -> serde_json::Value {
    serde_json::json!({
        "id": run.id,
        "status": format!("{:?}", run.status),
        "final_loss": loss_vals.last().copied(),
        "min_loss": min_val(loss_vals),
        "steps": loss_vals.len(),
        "mean_tps": mean(tps_vals),
        "loss_history": loss_vals,
    })
}

/// Build JSON config diff between two param sets.
fn config_diff_json(
    params_a: &std::collections::HashMap<String, entrenar::storage::ParameterValue>,
    params_b: &std::collections::HashMap<String, entrenar::storage::ParameterValue>,
) -> serde_json::Map<String, serde_json::Value> {
    let all_keys: std::collections::BTreeSet<_> = params_a.keys().chain(params_b.keys()).collect();
    let mut diff = serde_json::Map::new();
    for key in &all_keys {
        let va = params_a.get(*key).map(param_to_value);
        let vb = params_b.get(*key).map(param_to_value);
        if va != vb {
            diff.insert(key.to_string(), serde_json::json!({"a": va, "b": vb}));
        }
    }
    diff
}

#[allow(clippy::too_many_arguments)]
fn print_diff_json(
    run_a: &entrenar::storage::sqlite::Run,
    run_b: &entrenar::storage::sqlite::Run,
    params_a: &std::collections::HashMap<String, entrenar::storage::ParameterValue>,
    params_b: &std::collections::HashMap<String, entrenar::storage::ParameterValue>,
    loss_a: &[entrenar::storage::MetricPoint],
    loss_b: &[entrenar::storage::MetricPoint],
    tps_a: &[entrenar::storage::MetricPoint],
    tps_b: &[entrenar::storage::MetricPoint],
    _lr_a: &[entrenar::storage::MetricPoint],
    _lr_b: &[entrenar::storage::MetricPoint],
) {
    let lv_a = metric_values(loss_a);
    let lv_b = metric_values(loss_b);
    let tv_a = metric_values(tps_a);
    let tv_b = metric_values(tps_b);

    let final_a = lv_a.last().copied();
    let final_b = lv_b.last().copied();
    let winner = match (final_a, final_b) {
        (Some(a), Some(b)) if a < b => "a",
        (Some(a), Some(b)) if b < a => "b",
        (Some(_), Some(_)) => "tie",
        _ => "unknown",
    };

    let output = serde_json::json!({
        "run_a": run_summary_json(run_a, &lv_a, &tv_a),
        "run_b": run_summary_json(run_b, &lv_b, &tv_b),
        "verdict": {
            "winner": winner,
            "loss_diff_percent": match (final_a, final_b) {
                (Some(a), Some(b)) => Some(((a - b) / b.abs().max(1e-10)) * 100.0),
                _ => None,
            },
        },
        "config_diff": serde_json::Value::Object(config_diff_json(params_a, params_b)),
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&output).unwrap_or_default()
    );
}

// ─── Utilities ──────────────────────────────────────────────────────────────

fn format_duration_long(secs: i64) -> String {
    if secs > 86400 {
        format!(
            "{}d {}h {}m",
            secs / 86400,
            (secs % 86400) / 3600,
            (secs % 3600) / 60
        )
    } else if secs > 3600 {
        format!("{}h {}m {}s", secs / 3600, (secs % 3600) / 60, secs % 60)
    } else if secs > 60 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else {
        format!("{}s", secs)
    }
}

/// Convert ParameterValue to serde_json::Value (unwrapping the tagged enum).
fn param_to_value(pv: &entrenar::storage::ParameterValue) -> serde_json::Value {
    use entrenar::storage::ParameterValue;
    match pv {
        ParameterValue::String(s) => serde_json::Value::String(s.clone()),
        ParameterValue::Int(i) => serde_json::json!(*i),
        ParameterValue::Float(f) => serde_json::json!(*f),
        ParameterValue::Bool(b) => serde_json::json!(*b),
        ParameterValue::List(l) => serde_json::Value::Array(l.iter().map(param_to_value).collect()),
        ParameterValue::Dict(d) => {
            let map: serde_json::Map<_, _> = d
                .iter()
                .map(|(k, v)| (k.clone(), param_to_value(v)))
                .collect();
            serde_json::Value::Object(map)
        }
    }
}

/// Display a ParameterValue as a human-readable string.
fn param_display(pv: &entrenar::storage::ParameterValue) -> String {
    use entrenar::storage::ParameterValue;
    match pv {
        ParameterValue::String(s) => s.clone(),
        ParameterValue::Int(i) => i.to_string(),
        ParameterValue::Float(f) => format!("{f}"),
        ParameterValue::Bool(b) => b.to_string(),
        _ => pv.to_json(),
    }
}

// ═══════════════════════════════════════════════════════════════════
// PMAT-540 Phase 4: Inline tests for pure helper functions
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod runs_tests {
    use super::*;

    // ─── --status filtering (dogfood 0.63.0, issue #2374 finding 8) ─────────
    //
    // `apr runs ls --status completed` — the one non-trivial value --help
    // advertised — matched 0 of 2941 completed runs, because the filter
    // stringified the stored variant (`Success`) and compared it verbatim.
    // `--status bogusvalue` returned an empty table with exit 0, which a user
    // cannot distinguish from "nothing has finished".

    #[test]
    fn test_status_completed_selects_the_success_variant() {
        // The documented spelling must select the variant that is actually stored.
        assert_eq!(
            resolve_status_filter("completed").expect("'completed' is documented"),
            Some("Success")
        );
    }

    #[test]
    fn test_status_matches_the_debug_spelling_of_the_stored_variant() {
        // Guards the comparison itself: whatever resolve_status_filter returns
        // is compared against `format!("{:?}", run.status)`, so it must equal
        // that spelling exactly — not a lowercased one.
        let stored = format!("{:?}", entrenar::storage::RunStatus::Success);
        assert_eq!(
            resolve_status_filter("completed").expect("'completed' is documented"),
            Some(stored.as_str())
        );
        let running = format!("{:?}", entrenar::storage::RunStatus::Running);
        assert_eq!(
            resolve_status_filter("running").expect("'running' is documented"),
            Some(running.as_str())
        );
    }

    #[test]
    fn test_status_all_disables_filtering() {
        assert_eq!(
            resolve_status_filter("all").expect("'all' is the default"),
            None
        );
    }

    #[test]
    fn test_every_documented_status_resolves() {
        // --help promises these; none may silently return nothing.
        for name in [
            "all",
            "pending",
            "running",
            "completed",
            "failed",
            "cancelled",
        ] {
            assert!(
                resolve_status_filter(name).is_ok(),
                "--help advertises --status {name}, so it must resolve"
            );
        }
    }

    #[test]
    fn test_unknown_status_is_rejected_not_silently_empty() {
        let err = resolve_status_filter("bogusvalue")
            .expect_err("an unknown status must not quietly return zero rows");
        let msg = err.to_string();
        assert!(
            msg.contains("bogusvalue"),
            "error must echo the bad value: {msg}"
        );
        assert!(
            msg.contains("completed"),
            "error must list the supported values: {msg}"
        );
    }

    #[test]
    fn test_status_is_case_insensitive_and_trimmed() {
        assert_eq!(
            resolve_status_filter(" COMPLETED ").expect("case-insensitive"),
            Some("Success")
        );
    }

    #[test]
    fn test_sparkline_empty() {
        assert_eq!(sparkline(&[], 10), "—");
    }

    #[test]
    fn test_sparkline_single_value() {
        let result = sparkline(&[1.0], 5);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_sparkline_ascending() {
        let data: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let result = sparkline(&data, 8);
        assert_eq!(result.chars().count(), 8);
        // First char should be lowest, last should be highest
        let chars: Vec<char> = result.chars().collect();
        assert!(chars[0] <= chars[7]);
    }

    #[test]
    fn test_sparkline_constant() {
        let result = sparkline(&[5.0, 5.0, 5.0, 5.0], 4);
        // All same value → all same sparkline char
        let chars: Vec<char> = result.chars().collect();
        assert!(chars.iter().all(|c| *c == chars[0]));
    }

    #[test]
    fn test_loss_trend_arrow_decreasing() {
        let data = vec![1.0, 0.9, 0.8, 0.7, 0.6, 0.5];
        let arrow = loss_trend_arrow(&data);
        assert!(arrow.contains('↓'), "decreasing loss should show ↓");
    }

    #[test]
    fn test_loss_trend_arrow_increasing() {
        let data = vec![0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let arrow = loss_trend_arrow(&data);
        assert!(arrow.contains('↑'), "increasing loss should show ↑");
    }

    #[test]
    fn test_loss_trend_arrow_flat() {
        let data = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let arrow = loss_trend_arrow(&data);
        assert!(arrow.contains('→'), "flat loss should show →");
    }

    #[test]
    fn test_loss_trend_arrow_too_short() {
        assert_eq!(loss_trend_arrow(&[1.0, 2.0]), "—");
        assert_eq!(loss_trend_arrow(&[]), "—");
    }

    #[test]
    fn test_truncate_str_short() {
        assert_eq!(truncate_str("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_str_exact() {
        assert_eq!(truncate_str("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_str_long() {
        let result = truncate_str("hello world", 8);
        assert!(result.ends_with('…'));
        assert!(result.len() <= 10); // 7 chars + multi-byte …
    }

    #[test]
    fn test_mean_values() {
        assert!((mean(&[1.0, 2.0, 3.0]) - 2.0).abs() < 1e-10);
        assert!((mean(&[10.0]) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_mean_empty() {
        assert!((mean(&[]) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_min_val_some() {
        assert_eq!(min_val(&[3.0, 1.0, 2.0]), Some(1.0));
    }

    #[test]
    fn test_min_val_empty() {
        assert_eq!(min_val(&[]), None);
    }

    #[test]
    fn test_fmt_loss_some() {
        assert_eq!(fmt_loss(Some(0.123456)), "0.123456");
    }

    #[test]
    fn test_fmt_loss_none() {
        assert_eq!(fmt_loss(None), "—");
    }

    #[test]
    fn test_format_duration_long_seconds() {
        assert_eq!(format_duration_long(45), "45s");
    }

    #[test]
    fn test_format_duration_long_minutes() {
        assert_eq!(format_duration_long(125), "2m 5s");
    }

    #[test]
    fn test_format_duration_long_hours() {
        assert_eq!(format_duration_long(3725), "1h 2m 5s");
    }

    #[test]
    fn test_format_duration_long_days() {
        assert_eq!(format_duration_long(90061), "1d 1h 1m");
    }

    #[test]
    fn test_braille_chart_empty() {
        let result = braille_chart(&[], 10, 5);
        // Empty data produces empty or minimal output
        assert!(result.is_empty() || result.len() < 20);
    }

    #[test]
    fn test_braille_chart_single() {
        let result = braille_chart(&[1.0], 10, 3);
        assert!(!result.is_empty());
    }

    // ═══════════════════════════════════════════════════════════════════
    // PMAT-125 B4: additional coverage for braille / metric / param helpers
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_sparkline_truncates_to_width() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let result = sparkline(&data, 12);
        assert_eq!(result.chars().count(), 12);
    }

    #[test]
    fn test_sparkline_descending_endpoints() {
        let data: Vec<f64> = (0..8).rev().map(|i| i as f64).collect();
        let result = sparkline(&data, 8);
        let chars: Vec<char> = result.chars().collect();
        assert!(chars[0] >= chars[7]);
    }

    #[test]
    fn test_sample_braille_data_fewer_points_than_dots() {
        let data = vec![0.0, 1.0, 2.0];
        let samples = sample_braille_data(&data, 8, 8, 0.0, 2.0);
        assert_eq!(samples.len(), 8);
        assert!(samples.iter().all(|&y| y < 8));
        assert_eq!(samples[2], 0); // value 2.0 normalized=1.0 → top row
        assert_eq!(samples[0], 7); // value 0.0 normalized=0.0 → bottom row
    }

    #[test]
    fn test_sample_braille_data_more_points_than_dots() {
        let data: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let samples = sample_braille_data(&data, 10, 8, 0.0, 49.0);
        assert_eq!(samples.len(), 10);
        assert!(samples.iter().all(|&y| y < 8));
    }

    #[test]
    fn test_render_braille_row_produces_braille_chars() {
        let samples = vec![0usize, 1, 2, 3, 4, 5, 6, 7];
        let dot_bits: [[u8; 4]; 2] = [[0, 1, 2, 6], [3, 4, 5, 7]];
        let row = render_braille_row(&samples, 0, 4, &dot_bits);
        assert_eq!(row.chars().count(), 4);
        assert!(row.chars().all(|c| (0x2800..=0x28FF).contains(&(c as u32))));
    }

    #[test]
    fn test_render_braille_row_blank_when_no_dots_in_range() {
        let samples = vec![7usize; 8];
        let dot_bits: [[u8; 4]; 2] = [[0, 1, 2, 6], [3, 4, 5, 7]];
        let row = render_braille_row(&samples, 0, 4, &dot_bits);
        assert!(row.chars().all(|c| c == '\u{2800}'));
    }

    #[test]
    fn test_braille_chart_multiline_structure() {
        let data: Vec<f64> = (0..20).map(|i| (i as f64).sin()).collect();
        let chart = braille_chart(&data, 16, 4);
        let lines: Vec<&str> = chart.lines().collect();
        assert_eq!(lines.len(), 4 + 2);
        assert!(lines[0].contains('│'));
        assert!(lines[4].contains('└'));
        assert!(lines[5].contains("20"));
    }

    #[test]
    fn test_metric_values_extracts_values_in_order() {
        use entrenar::storage::MetricPoint;
        let points = vec![
            MetricPoint::new(0, 1.5),
            MetricPoint::new(1, 0.9),
            MetricPoint::new(2, 0.4),
        ];
        assert_eq!(metric_values(&points), vec![1.5, 0.9, 0.4]);
    }

    #[test]
    fn test_metric_values_empty() {
        assert!(metric_values(&[]).is_empty());
    }

    #[test]
    fn test_min_val_negative_and_mixed() {
        assert_eq!(min_val(&[2.0, -5.0, 0.0, 3.0]), Some(-5.0));
        assert_eq!(min_val(&[42.0]), Some(42.0));
    }

    #[test]
    fn test_format_duration_long_boundaries() {
        assert_eq!(format_duration_long(60), "60s");
        assert_eq!(format_duration_long(3600), "60m 0s");
        assert_eq!(format_duration_long(86400), "24h 0m 0s");
        assert_eq!(format_duration_long(0), "0s");
    }

    #[test]
    fn test_param_display_scalar_variants() {
        use entrenar::storage::ParameterValue;
        assert_eq!(
            param_display(&ParameterValue::String("adam".into())),
            "adam"
        );
        assert_eq!(param_display(&ParameterValue::Int(42)), "42");
        assert_eq!(param_display(&ParameterValue::Bool(true)), "true");
        assert_eq!(param_display(&ParameterValue::Float(0.001)), "0.001");
    }

    #[test]
    fn test_param_display_list_falls_back_to_json() {
        use entrenar::storage::ParameterValue;
        let pv = ParameterValue::List(vec![ParameterValue::Int(1), ParameterValue::Int(2)]);
        let s = param_display(&pv);
        assert!(s.contains("Int") || s.contains('1'));
    }

    #[test]
    fn test_param_to_value_scalars() {
        use entrenar::storage::ParameterValue;
        assert_eq!(
            param_to_value(&ParameterValue::String("x".into())),
            serde_json::json!("x")
        );
        assert_eq!(
            param_to_value(&ParameterValue::Int(7)),
            serde_json::json!(7)
        );
        assert_eq!(
            param_to_value(&ParameterValue::Bool(false)),
            serde_json::json!(false)
        );
        assert_eq!(
            param_to_value(&ParameterValue::Float(2.5)),
            serde_json::json!(2.5)
        );
    }

    #[test]
    fn test_param_to_value_nested_list_and_dict() {
        use entrenar::storage::ParameterValue;
        use std::collections::HashMap;
        let list = ParameterValue::List(vec![ParameterValue::Int(1), ParameterValue::Bool(true)]);
        assert_eq!(param_to_value(&list), serde_json::json!([1, true]));

        let mut d = HashMap::new();
        d.insert("lr".to_string(), ParameterValue::Float(0.01));
        let dict = ParameterValue::Dict(d);
        assert_eq!(param_to_value(&dict), serde_json::json!({"lr": 0.01}));
    }

    #[test]
    fn test_config_diff_detects_differences_and_skips_equal() {
        use entrenar::storage::ParameterValue;
        use std::collections::HashMap;
        let mut a = HashMap::new();
        a.insert("lr".to_string(), ParameterValue::Float(0.01));
        a.insert("opt".to_string(), ParameterValue::String("adam".into()));
        a.insert("only_a".to_string(), ParameterValue::Int(1));

        let mut b = HashMap::new();
        b.insert("lr".to_string(), ParameterValue::Float(0.02));
        b.insert("opt".to_string(), ParameterValue::String("adam".into()));
        b.insert("only_b".to_string(), ParameterValue::Int(2));

        let diffs = config_diff(&a, &b);
        let keys: Vec<&str> = diffs.iter().map(|(k, _, _)| k.as_str()).collect();
        assert!(!keys.contains(&"opt"));
        assert!(keys.contains(&"lr"));
        assert!(keys.contains(&"only_a"));
        assert!(keys.contains(&"only_b"));
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        assert_eq!(keys, sorted);
        let only_a = diffs.iter().find(|(k, _, _)| k == "only_a").unwrap();
        assert_eq!(only_a.2, "—");
        let only_b = diffs.iter().find(|(k, _, _)| k == "only_b").unwrap();
        assert_eq!(only_b.1, "—");
    }

    #[test]
    fn test_config_diff_identical_maps_empty() {
        use entrenar::storage::ParameterValue;
        use std::collections::HashMap;
        let mut a = HashMap::new();
        a.insert("x".to_string(), ParameterValue::Int(1));
        let b = a.clone();
        assert!(config_diff(&a, &b).is_empty());
    }
}
