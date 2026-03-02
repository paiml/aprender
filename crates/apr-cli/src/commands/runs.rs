//! `apr runs` — List and inspect training experiment runs (ALB-050)
//!
//! Reads from local SQLite experiment database (`.entrenar/experiments.db`)
//! or global registry (`~/.entrenar/experiments.db`).

use crate::error::CliError;
use entrenar::storage::{ExperimentStorage, SqliteBackend};
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, CliError>;

/// List experiment runs from SQLite DB
pub(crate) fn run_ls(
    dir: &Option<PathBuf>,
    global: bool,
    status_filter: &str,
    json: bool,
    limit: usize,
) -> Result<()> {
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
    if status_filter != "all" {
        all_runs.retain(|(_, run)| {
            let run_status = format!("{:?}", run.status).to_lowercase();
            run_status == status_filter.to_lowercase()
        });
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

/// Show detailed run metrics
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
    let lr_metrics = store.get_metrics(run_id, "learning_rate").unwrap_or_default();
    let tps_metrics = store.get_metrics(run_id, "tokens_per_second").unwrap_or_default();

    if json {
        let mut metrics_map = serde_json::Map::new();
        if !loss_metrics.is_empty() {
            metrics_map.insert("loss".into(), serde_json::json!({
                "count": loss_metrics.len(),
                "first": loss_metrics.first().map(|p| p.value),
                "last": loss_metrics.last().map(|p| p.value),
                "min": loss_metrics.iter().map(|p| p.value).fold(f64::INFINITY, f64::min),
            }));
        }
        if !tps_metrics.is_empty() {
            metrics_map.insert("tokens_per_second".into(), serde_json::json!({
                "count": tps_metrics.len(),
                "last": tps_metrics.last().map(|p| p.value),
                "max": tps_metrics.iter().map(|p| p.value).fold(f64::NEG_INFINITY, f64::max),
            }));
        }

        let params_map: std::collections::HashMap<_, _> = params.iter()
            .map(|(k, v)| (k.clone(), v.to_json()))
            .collect();

        let output = serde_json::json!({
            "run_id": run.id,
            "experiment_id": run.experiment_id,
            "status": format!("{:?}", run.status),
            "start_time": run.start_time.to_rfc3339(),
            "end_time": run.end_time.map(|t| t.to_rfc3339()),
            "params": params_map,
            "metrics": serde_json::Value::Object(metrics_map),
        });
        println!("{}", serde_json::to_string_pretty(&output).unwrap_or_default());
    } else {
        println!("Run: {}", run.id);
        println!("  Experiment: {}", run.experiment_id);
        println!("  Status:     {:?}", run.status);
        println!("  Started:    {}", run.start_time.format("%Y-%m-%d %H:%M:%S"));
        if let Some(end) = run.end_time {
            let duration = end - run.start_time;
            println!("  Ended:      {}", end.format("%Y-%m-%d %H:%M:%S"));
            println!("  Duration:   {}s", duration.num_seconds());
        }
        println!();

        if !params.is_empty() {
            println!("  Parameters:");
            let mut sorted_params: Vec<_> = params.iter().collect();
            sorted_params.sort_by_key(|(k, _)| *k);
            for (k, v) in sorted_params {
                println!("    {:<20} {}", k, v.to_json());
            }
            println!();
        }

        if !loss_metrics.is_empty() {
            let first_loss = loss_metrics.first().map(|p| p.value).unwrap_or(0.0);
            let last_loss = loss_metrics.last().map(|p| p.value).unwrap_or(0.0);
            let min_loss = loss_metrics.iter().map(|p| p.value).fold(f64::INFINITY, f64::min);
            println!("  Loss: {:.6} → {:.6} (min: {:.6}, {} steps)",
                first_loss, last_loss, min_loss, loss_metrics.len());
        }

        if !tps_metrics.is_empty() {
            let last_tps = tps_metrics.last().map(|p| p.value).unwrap_or(0.0);
            let max_tps = tps_metrics.iter().map(|p| p.value).fold(f64::NEG_INFINITY, f64::max);
            println!("  Throughput: {:.0} tok/s (peak: {:.0})", last_tps, max_tps);
        }

        if !lr_metrics.is_empty() {
            let last_lr = lr_metrics.last().map(|p| p.value).unwrap_or(0.0);
            println!("  Learning rate: {:.2e}", last_lr);
        }
    }

    Ok(())
}

fn open_store(dir: &Option<PathBuf>, global: bool) -> Result<SqliteBackend> {
    let db_path = if global {
        dirs::home_dir()
            .map(|h| h.join(".entrenar").join("experiments.db"))
            .ok_or_else(|| CliError::ValidationFailed("Could not determine home directory".into()))?
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

fn print_runs_table(runs: &[(entrenar::storage::sqlite::Experiment, entrenar::storage::sqlite::Run)], store: &SqliteBackend) {
    // Header
    println!(
        "{:<20} {:<28} {:>10} {:>12} {:>12} {:>10}",
        "EXPERIMENT", "RUN ID", "STATUS", "LOSS", "TOK/S", "DURATION"
    );
    println!("{}", "─".repeat(96));

    for (exp, run) in runs {
        let status = format!("{:?}", run.status);
        let status_str = match run.status {
            entrenar::storage::RunStatus::Success => format!("\x1b[32m{:<10}\x1b[0m", status),
            entrenar::storage::RunStatus::Failed => format!("\x1b[31m{:<10}\x1b[0m", status),
            entrenar::storage::RunStatus::Running => format!("\x1b[33m{:<10}\x1b[0m", status),
            _ => format!("{:<10}", status),
        };

        // Get final loss and tok/s
        let loss_str = store
            .get_metrics(&run.id, "loss")
            .ok()
            .and_then(|m| m.last().map(|p| format!("{:.6}", p.value)))
            .unwrap_or_else(|| "—".to_string());

        let tps_str = store
            .get_metrics(&run.id, "tokens_per_second")
            .ok()
            .and_then(|m| m.last().map(|p| format!("{:.0}", p.value)))
            .unwrap_or_else(|| "—".to_string());

        let duration_str = run.end_time
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
            .unwrap_or_else(|| "running".to_string());

        let exp_name = if exp.name.len() > 18 {
            format!("{}…", &exp.name[..17])
        } else {
            exp.name.clone()
        };

        let run_id_short = if run.id.len() > 26 {
            format!("{}…", &run.id[..25])
        } else {
            run.id.clone()
        };

        println!(
            "{:<20} {:<28} {} {:>12} {:>12} {:>10}",
            exp_name, run_id_short, status_str, loss_str, tps_str, duration_str
        );
    }

    println!();
    println!("{} run(s)", runs.len());
}

fn print_runs_json(runs: &[(entrenar::storage::sqlite::Experiment, entrenar::storage::sqlite::Run)], store: &SqliteBackend) {
    let entries: Vec<serde_json::Value> = runs
        .iter()
        .map(|(exp, run)| {
            let final_loss = store
                .get_metrics(&run.id, "loss")
                .ok()
                .and_then(|m| m.last().map(|p| p.value));

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
                "final_loss": final_loss,
                "tokens_per_second": final_tps,
            })
        })
        .collect();

    println!("{}", serde_json::to_string_pretty(&entries).unwrap_or_else(|_| "[]".to_string()));
}
