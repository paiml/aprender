//! Experiment store CLI commands.
//!
//! Query experiments, runs, and metrics from the project-local SQLite store.

use crate::cli::LogLevel;
use crate::config::{ExperimentsArgs, ExperimentsCommand, OutputFormat};
use crate::storage::{ExperimentStorage, SqliteBackend};

pub fn run_experiments(args: ExperimentsArgs, log_level: LogLevel) -> Result<(), String> {
    let store = SqliteBackend::open_project(&args.project)
        .map_err(|e| format!("Failed to open experiment store: {e}"))?;

    match args.command {
        ExperimentsCommand::List => list_experiments(&store, &args.format, log_level),
        ExperimentsCommand::Show { id } => show_experiment(&store, &id, &args.format),
        ExperimentsCommand::Runs { experiment_id } => {
            list_runs(&store, &experiment_id, &args.format)
        }
        ExperimentsCommand::Metrics { run_id, key } => {
            show_metrics(&store, &run_id, &key, &args.format)
        }
        ExperimentsCommand::Delete { id } => delete_experiment(&store, &id, log_level),
    }
}

fn list_experiments(
    store: &SqliteBackend,
    format: &OutputFormat,
    _log_level: LogLevel,
) -> Result<(), String> {
    let experiments =
        store.list_experiments().map_err(|e| format!("Failed to list experiments: {e}"))?;

    if experiments.is_empty() {
        eprintln!("No experiments found in {}", store.path());
        return Ok(());
    }

    match format {
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&experiments)
                .map_err(|e| format!("JSON serialization failed: {e}"))?;
            println!("{json}");
        }
        _ => {
            println!("{:<20} {:<30} {:<24}", "ID", "NAME", "CREATED");
            println!("{}", "-".repeat(74));
            for exp in &experiments {
                println!(
                    "{:<20} {:<30} {:<24}",
                    truncate(&exp.id, 18),
                    truncate(&exp.name, 28),
                    exp.created_at.format("%Y-%m-%d %H:%M:%S"),
                );
            }
            println!("\n{} experiment(s)", experiments.len());
        }
    }

    Ok(())
}

fn show_experiment(store: &SqliteBackend, id: &str, format: &OutputFormat) -> Result<(), String> {
    let experiment =
        store.get_experiment(id).map_err(|e| format!("Failed to get experiment: {e}"))?;

    match format {
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&experiment)
                .map_err(|e| format!("JSON serialization failed: {e}"))?;
            println!("{json}");
        }
        _ => {
            println!("Experiment: {}", experiment.name);
            println!("  ID:      {}", experiment.id);
            println!("  Created: {}", experiment.created_at.format("%Y-%m-%d %H:%M:%S"));
            println!("  Updated: {}", experiment.updated_at.format("%Y-%m-%d %H:%M:%S"));
            if let Some(desc) = &experiment.description {
                println!("  Desc:    {desc}");
            }
            if let Some(config) = &experiment.config {
                println!("  Config:  {config}");
            }

            // Also show runs
            if let Ok(runs) = store.list_runs(id) {
                if !runs.is_empty() {
                    println!("\n  Runs ({}):", runs.len());
                    for run in &runs {
                        println!(
                            "    {:<18} {:?}  {}",
                            truncate(&run.id, 16),
                            run.status,
                            run.start_time.format("%Y-%m-%d %H:%M:%S"),
                        );
                    }
                }
            }
        }
    }

    Ok(())
}

fn list_runs(
    store: &SqliteBackend,
    experiment_id: &str,
    format: &OutputFormat,
) -> Result<(), String> {
    let runs = store.list_runs(experiment_id).map_err(|e| format!("Failed to list runs: {e}"))?;

    if runs.is_empty() {
        eprintln!("No runs found for experiment {experiment_id}");
        return Ok(());
    }

    match format {
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&runs)
                .map_err(|e| format!("JSON serialization failed: {e}"))?;
            println!("{json}");
        }
        _ => {
            println!("{:<20} {:<12} {:<24} {:<24}", "ID", "STATUS", "STARTED", "ENDED");
            println!("{}", "-".repeat(80));
            for run in &runs {
                let end = run
                    .end_time
                    .map_or_else(|| "-".to_string(), |t| t.format("%Y-%m-%d %H:%M:%S").to_string());
                println!(
                    "{:<20} {:<12} {:<24} {:<24}",
                    truncate(&run.id, 18),
                    format!("{:?}", run.status),
                    run.start_time.format("%Y-%m-%d %H:%M:%S"),
                    end,
                );
            }
            println!("\n{} run(s)", runs.len());
        }
    }

    Ok(())
}

fn show_metrics(
    store: &SqliteBackend,
    run_id: &str,
    key: &str,
    format: &OutputFormat,
) -> Result<(), String> {
    let metrics =
        store.get_metrics(run_id, key).map_err(|e| format!("Failed to get metrics: {e}"))?;

    if metrics.is_empty() {
        eprintln!("No metrics found for run {run_id}, key '{key}'");
        return Ok(());
    }

    match format {
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&metrics)
                .map_err(|e| format!("JSON serialization failed: {e}"))?;
            println!("{json}");
        }
        _ => {
            println!("Metrics: {key} (run {run_id})");
            println!("{:<8} {:<16} {:<24}", "STEP", "VALUE", "TIMESTAMP");
            println!("{}", "-".repeat(48));
            for point in &metrics {
                println!(
                    "{:<8} {:<16.6} {:<24}",
                    point.step,
                    point.value,
                    point.timestamp.format("%Y-%m-%d %H:%M:%S"),
                );
            }
            println!("\n{} point(s)", metrics.len());
        }
    }

    Ok(())
}

fn delete_experiment(store: &SqliteBackend, id: &str, _log_level: LogLevel) -> Result<(), String> {
    // Verify it exists first
    store.get_experiment(id).map_err(|e| format!("Failed to find experiment: {e}"))?;

    let conn = store.lock_conn().map_err(|e| format!("Lock error: {e}"))?;

    // Delete in dependency order: metrics → params → artifacts → span_ids → runs → experiment
    conn.execute(
        "DELETE FROM metrics WHERE run_id IN (SELECT id FROM runs WHERE experiment_id = ?1)",
        [id],
    )
    .map_err(|e| format!("Failed to delete metrics: {e}"))?;

    conn.execute(
        "DELETE FROM params WHERE run_id IN (SELECT id FROM runs WHERE experiment_id = ?1)",
        [id],
    )
    .map_err(|e| format!("Failed to delete params: {e}"))?;

    conn.execute(
        "DELETE FROM artifacts WHERE run_id IN (SELECT id FROM runs WHERE experiment_id = ?1)",
        [id],
    )
    .map_err(|e| format!("Failed to delete artifacts: {e}"))?;

    conn.execute(
        "DELETE FROM span_ids WHERE run_id IN (SELECT id FROM runs WHERE experiment_id = ?1)",
        [id],
    )
    .map_err(|e| format!("Failed to delete span IDs: {e}"))?;

    conn.execute("DELETE FROM runs WHERE experiment_id = ?1", [id])
        .map_err(|e| format!("Failed to delete runs: {e}"))?;

    conn.execute("DELETE FROM experiments WHERE id = ?1", [id])
        .map_err(|e| format!("Failed to delete experiment: {e}"))?;

    eprintln!("Deleted experiment {id}");
    Ok(())
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("{}...", &s[..max - 3])
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use crate::storage::ExperimentStorage;

    #[test]
    fn test_truncate_short() {
        assert_eq!(truncate("hi", 10), "hi");
    }

    #[test]
    fn test_truncate_exact() {
        assert_eq!(truncate("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_long() {
        assert_eq!(truncate("a very long name", 10), "a very ...");
    }

    #[test]
    fn test_truncate_boundary() {
        assert_eq!(truncate("abcde", 4), "a...");
    }

    #[test]
    fn test_list_experiments_empty() {
        let store = SqliteBackend::open_in_memory().unwrap();
        assert!(list_experiments(&store, &OutputFormat::Text, LogLevel::Normal).is_ok());
    }

    #[test]
    fn test_list_experiments_with_data() {
        let mut s = SqliteBackend::open_in_memory().unwrap();
        s.create_experiment("e1", None).unwrap();
        s.create_experiment("e2", None).unwrap();
        assert!(list_experiments(&s, &OutputFormat::Text, LogLevel::Normal).is_ok());
    }

    #[test]
    fn test_list_experiments_json() {
        let mut s = SqliteBackend::open_in_memory().unwrap();
        s.create_experiment("j", None).unwrap();
        assert!(list_experiments(&s, &OutputFormat::Json, LogLevel::Normal).is_ok());
    }

    #[test]
    fn test_show_experiment_text() {
        let mut s = SqliteBackend::open_in_memory().unwrap();
        let id = s.create_experiment("sh", None).unwrap();
        assert!(show_experiment(&s, &id, &OutputFormat::Text).is_ok());
    }

    #[test]
    fn test_show_experiment_json() {
        let mut s = SqliteBackend::open_in_memory().unwrap();
        let id = s.create_experiment("sj", None).unwrap();
        assert!(show_experiment(&s, &id, &OutputFormat::Json).is_ok());
    }

    #[test]
    fn test_show_experiment_not_found() {
        let s = SqliteBackend::open_in_memory().unwrap();
        assert!(show_experiment(&s, "x", &OutputFormat::Text).is_err());
    }

    #[test]
    fn test_show_experiment_with_runs() {
        let mut s = SqliteBackend::open_in_memory().unwrap();
        let eid = s.create_experiment("wr", None).unwrap();
        s.create_run(&eid).unwrap();
        assert!(show_experiment(&s, &eid, &OutputFormat::Text).is_ok());
    }

    #[test]
    fn test_list_runs_empty() {
        let mut s = SqliteBackend::open_in_memory().unwrap();
        let eid = s.create_experiment("nr", None).unwrap();
        assert!(list_runs(&s, &eid, &OutputFormat::Text).is_ok());
    }

    #[test]
    fn test_list_runs_with_data() {
        let mut s = SqliteBackend::open_in_memory().unwrap();
        let eid = s.create_experiment("hr", None).unwrap();
        let rid = s.create_run(&eid).unwrap();
        s.start_run(&rid).unwrap();
        assert!(list_runs(&s, &eid, &OutputFormat::Text).is_ok());
    }

    #[test]
    fn test_list_runs_json() {
        let mut s = SqliteBackend::open_in_memory().unwrap();
        let eid = s.create_experiment("rj", None).unwrap();
        s.create_run(&eid).unwrap();
        assert!(list_runs(&s, &eid, &OutputFormat::Json).is_ok());
    }

    #[test]
    fn test_list_runs_completed() {
        let mut s = SqliteBackend::open_in_memory().unwrap();
        let eid = s.create_experiment("c", None).unwrap();
        let rid = s.create_run(&eid).unwrap();
        s.start_run(&rid).unwrap();
        s.complete_run(&rid, crate::storage::RunStatus::Success).unwrap();
        assert!(list_runs(&s, &eid, &OutputFormat::Text).is_ok());
    }

    #[test]
    fn test_show_metrics_empty() {
        let mut s = SqliteBackend::open_in_memory().unwrap();
        let eid = s.create_experiment("me", None).unwrap();
        let rid = s.create_run(&eid).unwrap();
        s.start_run(&rid).unwrap();
        assert!(show_metrics(&s, &rid, "loss", &OutputFormat::Text).is_ok());
    }

    #[test]
    fn test_show_metrics_with_data() {
        let mut s = SqliteBackend::open_in_memory().unwrap();
        let eid = s.create_experiment("md", None).unwrap();
        let rid = s.create_run(&eid).unwrap();
        s.start_run(&rid).unwrap();
        s.log_metric(&rid, "loss", 0, 0.5).unwrap();
        s.log_metric(&rid, "loss", 1, 0.3).unwrap();
        assert!(show_metrics(&s, &rid, "loss", &OutputFormat::Text).is_ok());
    }

    #[test]
    fn test_show_metrics_json() {
        let mut s = SqliteBackend::open_in_memory().unwrap();
        let eid = s.create_experiment("mj", None).unwrap();
        let rid = s.create_run(&eid).unwrap();
        s.start_run(&rid).unwrap();
        s.log_metric(&rid, "a", 0, 0.9).unwrap();
        assert!(show_metrics(&s, &rid, "a", &OutputFormat::Json).is_ok());
    }

    #[test]
    fn test_delete_experiment_ok() {
        let mut s = SqliteBackend::open_in_memory().unwrap();
        let id = s.create_experiment("d", None).unwrap();
        assert!(delete_experiment(&s, &id, LogLevel::Normal).is_ok());
    }

    #[test]
    fn test_delete_experiment_not_found() {
        let s = SqliteBackend::open_in_memory().unwrap();
        assert!(delete_experiment(&s, "x", LogLevel::Normal).is_err());
    }

    #[test]
    fn test_delete_with_data() {
        let mut s = SqliteBackend::open_in_memory().unwrap();
        let eid = s.create_experiment("dd", None).unwrap();
        let rid = s.create_run(&eid).unwrap();
        s.start_run(&rid).unwrap();
        s.log_metric(&rid, "l", 0, 0.5).unwrap();
        assert!(delete_experiment(&s, &eid, LogLevel::Normal).is_ok());
    }

    #[test]
    fn test_run_experiments_list() {
        let d = std::env::temp_dir().join("ent_exp_l");
        let _ = std::fs::create_dir_all(&d);
        let a = ExperimentsArgs {
            command: ExperimentsCommand::List,
            project: d.clone(),
            format: OutputFormat::Text,
        };
        assert!(run_experiments(a, LogLevel::Normal).is_ok());
        let _ = std::fs::remove_dir_all(d.join(".entrenar"));
    }

    #[test]
    fn test_run_experiments_show_nf() {
        let d = std::env::temp_dir().join("ent_exp_s");
        let _ = std::fs::create_dir_all(&d);
        let a = ExperimentsArgs {
            command: ExperimentsCommand::Show { id: "x".into() },
            project: d.clone(),
            format: OutputFormat::Text,
        };
        assert!(run_experiments(a, LogLevel::Normal).is_err());
        let _ = std::fs::remove_dir_all(d.join(".entrenar"));
    }
}
