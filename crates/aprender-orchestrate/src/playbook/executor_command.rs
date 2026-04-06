//! Command execution and public validation/status commands for the playbook executor.

use crate::playbook::cache;
use crate::playbook::dag;
use crate::playbook::parser;
use crate::playbook::types::*;
use anyhow::{Context, Result};
use std::path::Path;

/// Command execution error with exit code and stderr
#[derive(Debug, thiserror::Error)]
#[error("command failed (exit code: {exit_code:?}): {stderr}")]
pub(super) struct CommandError {
    pub(super) exit_code: Option<i32>,
    pub(super) stderr: String,
}

/// Execute a shell command via `sh -c`
pub(super) async fn execute_command(cmd: &str) -> Result<()> {
    let output = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .output()
        .await
        .context("failed to spawn command")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        return Err(CommandError {
            exit_code: output.status.code(),
            stderr: if stderr.is_empty() {
                format!("exit code {}", output.status.code().unwrap_or(-1))
            } else {
                stderr
            },
        }
        .into());
    }

    Ok(())
}

/// Validate a playbook without executing (for `batuta playbook validate`)
pub fn validate_only(playbook_path: &Path) -> Result<(Playbook, Vec<ValidationWarning>)> {
    let playbook = parser::parse_playbook_file(playbook_path)?;
    let warnings = parser::validate_playbook(&playbook)?;
    let _ = dag::build_dag(&playbook)?; // Validates DAG (no cycles, etc.)
    Ok((playbook, warnings))
}

/// Show playbook status (for `batuta playbook status`)
pub fn show_status(playbook_path: &Path) -> Result<()> {
    let playbook = parser::parse_playbook_file(playbook_path)?;
    let lock = cache::load_lock_file(playbook_path)?;

    println!("Playbook: {} ({})", playbook.name, playbook_path.display());
    println!("Version: {}", playbook.version);
    println!("Stages: {}", playbook.stages.len());

    if let Some(ref lock) = lock {
        println!("\nLock file: {} ({})", lock.generator, lock.generated_at);
        println!("{}", "-".repeat(60));
        for (name, _stage) in &playbook.stages {
            if let Some(stage_lock) = lock.stages.get(name) {
                let status_str = match stage_lock.status {
                    StageStatus::Completed => "COMPLETED",
                    StageStatus::Failed => "FAILED",
                    StageStatus::Cached => "CACHED",
                    StageStatus::Running => "RUNNING",
                    StageStatus::Pending => "PENDING",
                    StageStatus::Hashing => "HASHING",
                    StageStatus::Validating => "VALIDATING",
                };
                let duration = stage_lock
                    .duration_seconds
                    .map(|d| format!("{:.1}s", d))
                    .unwrap_or_else(|| "-".to_string());
                println!("  {:20} {:12} {}", name, status_str, duration);
            } else {
                println!("  {:20} {:12}", name, "NOT RUN");
            }
        }
    } else {
        println!("\nNo lock file found (pipeline has not been run)");
    }

    Ok(())
}
