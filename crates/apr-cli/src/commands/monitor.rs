//! Training monitor command
//!
//! Attaches to a running training session and displays live metrics via TUI.
//!
//! # Usage
//!
//! ```bash
//! # Shell 1: Start training
//! apr finetune --task classify --data corpus.jsonl -o /tmp/run-001
//!
//! # Shell 2: Attach live monitor
//! apr monitor /tmp/run-001
//! ```

use crate::error::{CliError, Result};
use std::path::Path;

/// Run the training monitor TUI.
///
/// Connects to a running (or completed) training session by reading
/// `training_state.json` from the experiment directory.
pub(crate) fn run(experiment_dir: &Path, refresh_ms: u64, compact: bool) -> Result<()> {
    if !experiment_dir.is_dir() {
        return Err(CliError::ValidationFailed(format!(
            "Experiment directory does not exist: {}",
            experiment_dir.display()
        )));
    }

    let config = entrenar::monitor::tui::TuiMonitorConfig {
        refresh_ms,
        compact,
        exit_on_complete: true,
        ..Default::default()
    };

    let mut monitor = entrenar::monitor::tui::TuiMonitor::new(experiment_dir, config);

    monitor.run().map_err(|e| {
        CliError::ValidationFailed(format!("Monitor error: {e}"))
    })
}
