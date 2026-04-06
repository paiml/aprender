//! Ruchy-specific types for session conversion.

use super::session::{CodeCell, EntrenarSession};
use std::collections::HashMap;

/// Simulated Ruchy session type for conversion.
///
/// In a real implementation, this would be `ruchy::Session`.
/// Here we define a compatible structure for testing.
#[derive(Debug, Clone)]
pub struct RuchySession {
    /// Session ID
    pub session_id: String,
    /// Session title
    pub title: String,
    /// Username
    pub username: Option<String>,
    /// Start time
    pub start_time: chrono::DateTime<chrono::Utc>,
    /// End time
    pub end_time: Option<chrono::DateTime<chrono::Utc>>,
    /// Kernel info
    pub kernel: Option<String>,
    /// Cells
    pub cells: Vec<RuchyCell>,
    /// Variables (serialized)
    pub variables: HashMap<String, String>,
    /// Training runs
    pub training_runs: Vec<TrainingRun>,
}

/// A cell in a Ruchy session.
#[derive(Debug, Clone)]
pub struct RuchyCell {
    /// Cell ID
    pub id: String,
    /// Cell type (code, markdown)
    pub cell_type: String,
    /// Source content
    pub source: String,
    /// Outputs
    pub outputs: Vec<String>,
    /// Execution count
    pub execution_count: Option<u32>,
    /// Timestamp
    pub executed_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// A training run within a session.
#[derive(Debug, Clone)]
pub struct TrainingRun {
    /// Run ID
    pub run_id: String,
    /// Model name
    pub model: String,
    /// Dataset
    pub dataset: Option<String>,
    /// Epochs
    pub epochs: u32,
    /// Loss values
    pub losses: Vec<f64>,
    /// Metrics
    pub metrics: HashMap<String, Vec<f64>>,
}

impl From<RuchySession> for EntrenarSession {
    fn from(ruchy: RuchySession) -> Self {
        let mut session = EntrenarSession::new(&ruchy.session_id, &ruchy.title);

        session.user = ruchy.username;
        session.created_at = ruchy.start_time;
        session.ended_at = ruchy.end_time;
        session.model_architecture = ruchy.kernel;

        // Convert cells
        for cell in ruchy.cells {
            if cell.cell_type == "code" {
                let code_cell = CodeCell {
                    execution_order: cell.execution_count.unwrap_or(0),
                    source: cell.source,
                    output: cell.outputs.first().cloned(),
                    timestamp: cell.executed_at.unwrap_or(ruchy.start_time),
                    duration_ms: None,
                };
                session.code_history.push(code_cell);
            }
        }

        // Convert training runs to metrics
        for run in ruchy.training_runs {
            for loss in run.losses {
                session.metrics.add_loss(loss);
            }
            for (name, values) in run.metrics {
                for value in values {
                    session.metrics.add_custom(&name, value);
                }
            }
            if session.dataset_id.is_none() {
                session.dataset_id = run.dataset;
            }
        }

        // Copy variables as config
        session.config = ruchy.variables;

        session
    }
}
