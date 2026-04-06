//! `cgp compete` — Head-to-head competitor comparison.
//! Spec section 2.5: run two or more commands, measure wall time,
//! compute TFLOP/s, and produce a comparison table.

use anyhow::Result;
use serde::Serialize;
use std::process::Command;
use std::time::Instant;

/// Result of running one competitor.
#[derive(Debug, Clone, Serialize)]
pub struct CompetitorResult {
    pub label: String,
    pub command: String,
    pub wall_time_ms: f64,
    pub exit_code: i32,
}

/// Run a command and measure wall time.
fn run_timed(command: &str) -> Result<CompetitorResult> {
    // Split command into program + args (basic shell-like splitting)
    let parts: Vec<&str> = command.split_whitespace().collect();
    if parts.is_empty() {
        anyhow::bail!("Empty command");
    }

    let start = Instant::now();
    let status = Command::new(parts[0])
        .args(&parts[1..])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map_err(|e| anyhow::anyhow!("Failed to run '{}': {}", parts[0], e))?;
    let elapsed = start.elapsed();

    Ok(CompetitorResult {
        label: String::new(),
        command: command.to_string(),
        wall_time_ms: elapsed.as_secs_f64() * 1000.0,
        exit_code: status.code().unwrap_or(-1),
    })
}

/// Run the `cgp compete` command.
pub fn run_compete(
    workload: &str,
    ours: &str,
    theirs: &[String],
    label: Option<&str>,
    json: bool,
) -> Result<()> {
    let labels: Vec<String> = match label {
        Some(l) => l.split(',').map(String::from).collect(),
        None => {
            let mut v = vec!["ours".to_string()];
            v.extend((0..theirs.len()).map(|i| format!("theirs-{}", i + 1)));
            v
        }
    };

    println!("\n=== CGP Head-to-Head: {workload} ===\n");

    let mut results: Vec<CompetitorResult> = Vec::new();

    let default_ours = "ours".to_string();
    let ours_label = labels.first().unwrap_or(&default_ours);

    // Run ours
    eprint!("  Running: {ours_label} ...");
    match run_timed(ours) {
        Ok(mut r) => {
            r.label = ours_label.clone();
            eprintln!(" {:.1}ms", r.wall_time_ms);
            results.push(r);
        }
        Err(e) => {
            eprintln!(" FAILED: {e}");
            results.push(CompetitorResult {
                label: ours_label.clone(),
                command: ours.to_string(),
                wall_time_ms: f64::INFINITY,
                exit_code: -1,
            });
        }
    }

    // Run theirs
    for (i, cmd) in theirs.iter().enumerate() {
        let default_theirs = format!("theirs-{}", i + 1);
        let lbl = labels.get(i + 1).unwrap_or(&default_theirs);
        eprint!("  Running: {lbl} ...");
        match run_timed(cmd) {
            Ok(mut r) => {
                r.label = lbl.to_string();
                eprintln!(" {:.1}ms", r.wall_time_ms);
                results.push(r);
            }
            Err(e) => {
                eprintln!(" FAILED: {e}");
                results.push(CompetitorResult {
                    label: lbl.to_string(),
                    command: cmd.clone(),
                    wall_time_ms: f64::INFINITY,
                    exit_code: -1,
                });
            }
        }
    }

    // Find best time
    let best_time = results
        .iter()
        .filter(|r| r.wall_time_ms.is_finite())
        .map(|r| r.wall_time_ms)
        .fold(f64::INFINITY, f64::min);

    if json {
        println!("{}", serde_json::to_string_pretty(&results)?);
        return Ok(());
    }

    // Print table
    println!();
    println!("  {:20} {:>12} {:>8} {:>10}", "Competitor", "Time (ms)", "Exit", "vs Best");
    println!("  {}", "-".repeat(54));

    for r in &results {
        let ratio = if best_time > 0.0 && r.wall_time_ms.is_finite() {
            format!("{:.2}x", r.wall_time_ms / best_time)
        } else {
            "N/A".to_string()
        };
        let time_str = if r.wall_time_ms.is_finite() {
            format!("{:.1}", r.wall_time_ms)
        } else {
            "FAILED".to_string()
        };
        println!("  {:20} {:>12} {:>8} {:>10}", r.label, time_str, r.exit_code, ratio);
    }

    // Winner
    if let Some(winner) = results.iter().filter(|r| r.wall_time_ms.is_finite()).min_by(|a, b| {
        a.wall_time_ms.partial_cmp(&b.wall_time_ms).unwrap_or(std::cmp::Ordering::Equal)
    }) {
        println!("\n  Winner: {} ({:.1}ms)", winner.label, winner.wall_time_ms);
    }

    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_timed_true() {
        let result = run_timed("true").unwrap();
        assert_eq!(result.exit_code, 0);
        assert!(result.wall_time_ms < 1000.0);
    }

    #[test]
    fn test_run_timed_false() {
        let result = run_timed("false").unwrap();
        assert_ne!(result.exit_code, 0);
    }

    #[test]
    fn test_run_timed_nonexistent() {
        let result = run_timed("nonexistent_binary_xyz_123");
        assert!(result.is_err());
    }
}
