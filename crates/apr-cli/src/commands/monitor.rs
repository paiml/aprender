//! Training monitor command (ALB-045, ALB-054, ALB-058)
//!
//! Attaches to a running training session and displays live metrics.
//! Supports TUI, JSON, and text output formats.
//!
//! # Usage
//!
//! ```bash
//! # Attach TUI to specific run
//! apr monitor /tmp/run-001
//!
//! # JSON output for LLM agents / CI
//! apr monitor --json /tmp/run-001
//!
//! # Discover active runs from global registry
//! apr monitor
//! ```

use crate::error::{CliError, Result};
use std::path::Path;

/// Run the training monitor.
///
/// When `experiment_dir` is Some, attaches to that specific run.
/// When None, discovers active runs from the global registry (ALB-054).
///
/// Output format:
/// - `tui` (default): Interactive terminal dashboard via presentar
/// - `json`: JSON lines per snapshot (ALB-053 parity with TUI)
/// - `text`: Human-readable log lines
pub(crate) fn run(
    experiment_dir: Option<&Path>,
    refresh_ms: u64,
    compact: bool,
    json: bool,
    format: &str,
) -> Result<()> {
    // Determine output format: --json flag overrides --format
    let output_format = if json { "json" } else { format };

    // If no directory specified, try to discover active runs
    let dir = match experiment_dir {
        Some(d) => d.to_path_buf(),
        None => {
            return discover_active_runs(output_format);
        }
    };

    if !dir.is_dir() {
        return Err(CliError::ValidationFailed(format!(
            "Experiment directory does not exist: {}",
            dir.display()
        )));
    }

    match output_format {
        "json" => run_headless(&dir, refresh_ms, entrenar::monitor::tui::OutputFormat::Json),
        "text" => run_headless(&dir, refresh_ms, entrenar::monitor::tui::OutputFormat::Text),
        _ => run_tui(&dir, refresh_ms, compact),
    }
}

/// Run interactive TUI monitor via presentar-terminal
fn run_tui(experiment_dir: &Path, refresh_ms: u64, compact: bool) -> Result<()> {
    let config = entrenar::monitor::tui::TuiMonitorConfig {
        refresh_ms,
        compact,
        exit_on_complete: true,
        ..Default::default()
    };

    let mut monitor = entrenar::monitor::tui::TuiMonitor::new(experiment_dir, config);

    monitor
        .run()
        .map_err(|e| CliError::ValidationFailed(format!("Monitor error: {e}")))
}

/// Run headless monitor (JSON or text output) — ALB-053/058
fn run_headless(
    experiment_dir: &Path,
    refresh_ms: u64,
    format: entrenar::monitor::tui::OutputFormat,
) -> Result<()> {
    let monitor = entrenar::monitor::tui::HeadlessMonitor::new(format, refresh_ms);
    monitor
        .run(experiment_dir)
        .map_err(|e| CliError::ValidationFailed(format!("Monitor error: {e}")))
}

/// Discover active training runs from the global registry (ALB-054)
fn discover_active_runs(format: &str) -> Result<()> {
    let db_path = dirs::home_dir()
        .map(|h| h.join(".entrenar").join("experiments.db"))
        .ok_or_else(|| CliError::ValidationFailed("Could not determine home directory".into()))?;

    if !db_path.exists() {
        return Err(CliError::ValidationFailed(
            "No global experiment registry found.\n\
             Hint: Start a training run first, or specify a directory:\n\
             \n\
             apr monitor <DIR>\n\
             apr train apply --config <yaml>"
                .into(),
        ));
    }

    let store = entrenar::storage::SqliteBackend::open(db_path.to_string_lossy().as_ref())
        .map_err(|e| {
            CliError::ValidationFailed(format!("Failed to open experiment database: {e}"))
        })?;

    let active_runs = scan_active_runs(&store)?;

    if active_runs.is_empty() {
        print_no_active_runs(format);
        return Ok(());
    }

    print_active_runs(&active_runs, format);
    Ok(())
}

/// Scan the global registry for runs with a live training_state.json.
fn scan_active_runs(
    store: &entrenar::storage::SqliteBackend,
) -> Result<Vec<(String, String, entrenar::storage::sqlite::Run)>> {
    let experiments = store
        .list_experiments()
        .map_err(|e| CliError::ValidationFailed(format!("Failed to list experiments: {e}")))?;

    let mut active = Vec::new();
    for exp in &experiments {
        let runs = match store.list_runs(&exp.id) {
            Ok(r) => r,
            Err(_) => continue,
        };
        for run in runs {
            let dir = run_output_dir(store, &run.id);
            if let Some(dir) = dir {
                active.push((exp.name.clone(), dir, run));
            }
        }
    }
    Ok(active)
}

/// Get the output directory for a run if it has a live training_state.json.
fn run_output_dir(store: &entrenar::storage::SqliteBackend, run_id: &str) -> Option<String> {
    let params = store.get_params(run_id).ok()?;
    let dir = match params.get("output_dir")? {
        entrenar::storage::ParameterValue::String(d) => d.clone(),
        _ => return None,
    };
    let state_path = std::path::PathBuf::from(&dir).join("training_state.json");
    state_path.exists().then_some(dir)
}

fn print_no_active_runs(format: &str) {
    if format == "json" {
        println!("[]");
    } else {
        println!("No active training runs found.");
        println!();
        println!("Hint: Start a training run, then attach:");
        println!("  apr train apply --config <yaml>");
        println!("  apr monitor <checkpoint-dir>");
    }
}

fn print_active_runs(
    active_runs: &[(String, String, entrenar::storage::sqlite::Run)],
    format: &str,
) {
    if format == "json" {
        let entries: Vec<serde_json::Value> = active_runs
            .iter()
            .map(|(name, dir, run)| {
                serde_json::json!({
                    "experiment": name,
                    "directory": dir,
                    "run_id": run.id,
                    "status": format!("{:?}", run.status),
                    "start_time": run.start_time.to_rfc3339(),
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&entries).unwrap_or_else(|_| "[]".to_string())
        );
    } else {
        println!("Active training runs:");
        println!();
        for (i, (name, dir, run)) in active_runs.iter().enumerate() {
            println!("  [{}] {} — {}", i + 1, name, dir);
            println!(
                "      Run: {} | Started: {}",
                run.id,
                run.start_time.format("%H:%M:%S")
            );
        }
        println!();
        println!("Attach to a run:");
        if let Some((_, dir, _)) = active_runs.first() {
            println!("  apr monitor {dir}");
        }
    }
}
