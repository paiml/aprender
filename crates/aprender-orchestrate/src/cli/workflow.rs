//! Workflow status and reset command implementations
//!
//! This module contains workflow management CLI commands extracted from main.rs.

#![cfg(feature = "native")]

use crate::ansi_colors::Colorize;
use crate::types::{PhaseStatus, WorkflowPhase, WorkflowState};
use std::path::PathBuf;
use tracing::warn;

/// Get the workflow state file path
fn get_state_file_path() -> PathBuf {
    PathBuf::from(".batuta-state.json")
}

/// Display workflow progress
pub fn display_workflow_progress(state: &WorkflowState) {
    println!();
    println!("{}", "📊 Workflow Progress".bright_cyan().bold());
    println!("{}", "─".repeat(50).dimmed());

    for phase in WorkflowPhase::all() {
        let info = state.phases.get(&phase).expect("workflow phase missing from state");
        let status_icon = match info.status {
            PhaseStatus::Completed => "✓".bright_green(),
            PhaseStatus::InProgress => "⏳".bright_yellow(),
            PhaseStatus::Failed => "✗".bright_red(),
            PhaseStatus::NotStarted => "○".dimmed(),
        };

        let phase_name = format!("{}", phase);
        let status_text = format!("{}", info.status);

        let is_current = state.current_phase == Some(phase);
        if is_current {
            println!(
                "  {} {} [{}]",
                status_icon,
                phase_name.cyan().bold(),
                status_text.bright_yellow()
            );
        } else {
            println!("  {} {} [{}]", status_icon, phase_name.dimmed(), status_text.dimmed());
        }
    }

    let progress = state.progress_percentage();
    println!();
    println!("  Overall: {:.0}% complete", progress);
    println!("{}", "─".repeat(50).dimmed());
    println!();
}

/// Show workflow status
pub fn cmd_status() -> anyhow::Result<()> {
    println!("{}", "📊 Workflow Status".bright_cyan().bold());
    println!();

    let state_file = get_state_file_path();
    let state = WorkflowState::load(&state_file).unwrap_or_else(|e| {
        warn!("Failed to load workflow state, starting fresh: {e}");
        WorkflowState::new()
    });

    // Check if any work has been done
    let has_started = state.phases.values().any(|info| info.status != PhaseStatus::NotStarted);

    if !has_started {
        println!("{}", "No workflow started yet.".dimmed());
        println!();
        println!("{}", "💡 Get started:".bright_yellow().bold());
        println!(
            "  {} Run {} to analyze your project",
            "1.".bright_blue(),
            "batuta analyze".cyan()
        );
        println!(
            "  {} Run {} to initialize configuration",
            "2.".bright_blue(),
            "batuta init".cyan()
        );
        println!();
        return Ok(());
    }

    display_workflow_progress(&state);

    // Display detailed phase information
    println!("{}", "Phase Details:".bright_yellow().bold());
    println!("{}", "─".repeat(50).dimmed());

    for phase in WorkflowPhase::all() {
        let info = state.phases.get(&phase).expect("workflow phase missing from state");

        let status_icon = match info.status {
            PhaseStatus::Completed => "✓".bright_green(),
            PhaseStatus::InProgress => "⏳".bright_yellow(),
            PhaseStatus::Failed => "✗".bright_red(),
            PhaseStatus::NotStarted => "○".dimmed(),
        };

        println!();
        println!("{} {}", status_icon, format!("{}", phase).bold());

        if let Some(started) = info.started_at {
            println!("  Started: {}", started.format("%Y-%m-%d %H:%M:%S UTC").to_string().dimmed());
        }

        if let Some(completed) = info.completed_at {
            println!(
                "  Completed: {}",
                completed.format("%Y-%m-%d %H:%M:%S UTC").to_string().dimmed()
            );

            if let Some(started) = info.started_at {
                let duration = completed.signed_duration_since(started);
                println!("  Duration: {:.2}s", duration.num_milliseconds() as f64 / 1000.0);
            }
        }

        if let Some(error) = &info.error {
            println!("  {}: {}", "Error".red().bold(), error.red());
        }
    }

    println!();
    println!("{}", "─".repeat(50).dimmed());
    println!();

    // Show next recommended action
    if let Some(current) = state.current_phase {
        println!("{}", "💡 Next Step:".bright_green().bold());
        match current {
            WorkflowPhase::Analysis => {
                println!(
                    "  Run {} to analyze your project",
                    "batuta analyze --languages --tdg".cyan()
                );
            }
            WorkflowPhase::Transpilation => {
                println!("  Run {} to convert your code", "batuta transpile".cyan());
            }
            WorkflowPhase::Optimization => {
                println!("  Run {} to optimize performance", "batuta optimize".cyan());
            }
            WorkflowPhase::Validation => {
                println!("  Run {} to validate equivalence", "batuta validate".cyan());
            }
            WorkflowPhase::Deployment => {
                println!("  Run {} to build final binary", "batuta build --release".cyan());
            }
        }
        println!();
    }

    Ok(())
}

/// Reset workflow state
pub fn cmd_reset(skip_confirm: bool) -> anyhow::Result<()> {
    println!("{}", "🔄 Reset Workflow".bright_cyan().bold());
    println!();

    let state_file = get_state_file_path();

    if !state_file.exists() {
        println!("{}", "No workflow state found.".dimmed());
        return Ok(());
    }

    // Load current state to show what will be reset
    let state = WorkflowState::load(&state_file)?;
    let completed_count =
        state.phases.values().filter(|info| info.status == PhaseStatus::Completed).count();

    if completed_count > 0 {
        println!("{}", "⚠️  Warning:".yellow().bold());
        println!("  This will reset {} completed phase(s)", completed_count.to_string().yellow());
        println!();
    }

    // Confirm unless --yes flag provided
    if !skip_confirm {
        print!("Are you sure you want to reset the workflow? [y/N] ");
        use std::io::{self, Write};
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        let input = input.trim().to_lowercase();
        if input != "y" && input != "yes" {
            println!("{}", "Reset cancelled.".dimmed());
            return Ok(());
        }
    }

    // Delete state file
    std::fs::remove_file(&state_file)?;

    println!();
    println!("{}", "✅ Workflow state reset successfully!".bright_green().bold());
    println!();
    println!("{}", "💡 Next Step:".bright_yellow().bold());
    println!("  Run {} to start fresh", "batuta analyze --languages --tdg".cyan());
    println!();

    Ok(())
}
