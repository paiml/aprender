//! Entrenar session representation and export functionality.

use super::metrics::SessionMetrics;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Entrenar session representation (converted from Ruchy).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntrenarSession {
    /// Session identifier
    pub id: String,
    /// Session name/title
    pub name: String,
    /// User who created the session
    pub user: Option<String>,
    /// Session creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Session end timestamp (None if still active)
    pub ended_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Model architecture used
    pub model_architecture: Option<String>,
    /// Dataset identifier
    pub dataset_id: Option<String>,
    /// Configuration parameters
    pub config: HashMap<String, String>,
    /// Training metrics
    pub metrics: SessionMetrics,
    /// Code cells/history from notebook
    pub code_history: Vec<CodeCell>,
    /// Session tags
    pub tags: Vec<String>,
    /// Notes/annotations
    pub notes: Option<String>,
}

/// A code cell from the session history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeCell {
    /// Cell execution order
    pub execution_order: u32,
    /// Source code
    pub source: String,
    /// Output (if captured)
    pub output: Option<String>,
    /// Execution timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Duration in milliseconds
    pub duration_ms: Option<u64>,
}

impl EntrenarSession {
    /// Create a new session.
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            user: None,
            created_at: chrono::Utc::now(),
            ended_at: None,
            model_architecture: None,
            dataset_id: None,
            config: HashMap::new(),
            metrics: SessionMetrics::new(),
            code_history: Vec::new(),
            tags: Vec::new(),
            notes: None,
        }
    }

    /// Set user.
    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    /// Set model architecture.
    pub fn with_architecture(mut self, arch: impl Into<String>) -> Self {
        self.model_architecture = Some(arch.into());
        self
    }

    /// Set dataset.
    pub fn with_dataset(mut self, dataset: impl Into<String>) -> Self {
        self.dataset_id = Some(dataset.into());
        self
    }

    /// Add configuration parameter.
    pub fn with_config(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.config.insert(key.into(), value.into());
        self
    }

    /// Add a tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Set notes.
    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = Some(notes.into());
        self
    }

    /// Add a code cell.
    pub fn add_code_cell(&mut self, cell: CodeCell) {
        self.code_history.push(cell);
    }

    /// Mark session as ended.
    pub fn end(&mut self) {
        self.ended_at = Some(chrono::Utc::now());
    }

    /// Calculate session duration.
    pub fn duration(&self) -> Option<chrono::Duration> {
        self.ended_at.map(|end| end - self.created_at)
    }

    /// Check if session has training data.
    pub fn has_training_data(&self) -> bool {
        !self.metrics.is_empty()
    }

    /// Export session to JSON for external tools (CLI, notebooks).
    /// (Issue #75: Session Export API for ruchy integration)
    ///
    /// # Errors
    /// Returns error if serialization fails.
    pub fn export_json(&self) -> Result<serde_json::Value, serde_json::Error> {
        let export = SessionExport {
            session_id: self.id.clone(),
            name: self.name.clone(),
            user: self.user.clone(),
            created_at: self.created_at.to_rfc3339(),
            ended_at: self.ended_at.map(|t| t.to_rfc3339()),
            duration_seconds: self.duration().map(|d| d.num_seconds()),
            model_architecture: self.model_architecture.clone(),
            dataset_id: self.dataset_id.clone(),
            config: self.config.clone(),
            metrics: MetricsExportSummary {
                total_steps: self.metrics.total_steps(),
                final_loss: self.metrics.final_loss(),
                best_loss: self.metrics.best_loss(),
                final_accuracy: self.metrics.final_accuracy(),
                best_accuracy: self.metrics.best_accuracy(),
                loss_history: self.metrics.loss_history.clone(),
                accuracy_history: self.metrics.accuracy_history.clone(),
                custom_metrics: self.metrics.custom.clone(),
            },
            code_cells_count: self.code_history.len(),
            tags: self.tags.clone(),
            notes: self.notes.clone(),
        };
        serde_json::to_value(export)
    }

    /// Export session to pretty-printed JSON string.
    ///
    /// # Errors
    /// Returns error if serialization fails.
    pub fn export_json_string(&self) -> Result<String, serde_json::Error> {
        let value = self.export_json()?;
        serde_json::to_string_pretty(&value)
    }
}

/// Session export structure for JSON serialization (Issue #75).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionExport {
    /// Session identifier
    pub session_id: String,
    /// Session name
    pub name: String,
    /// User who created the session
    pub user: Option<String>,
    /// Creation timestamp (RFC 3339)
    pub created_at: String,
    /// End timestamp (RFC 3339)
    pub ended_at: Option<String>,
    /// Duration in seconds
    pub duration_seconds: Option<i64>,
    /// Model architecture
    pub model_architecture: Option<String>,
    /// Dataset identifier
    pub dataset_id: Option<String>,
    /// Configuration parameters
    pub config: HashMap<String, String>,
    /// Training metrics summary
    pub metrics: MetricsExportSummary,
    /// Number of code cells
    pub code_cells_count: usize,
    /// Session tags
    pub tags: Vec<String>,
    /// Notes
    pub notes: Option<String>,
}

/// Metrics export summary for JSON serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsExportSummary {
    /// Total training steps
    pub total_steps: usize,
    /// Final loss value
    pub final_loss: Option<f64>,
    /// Best (minimum) loss value
    pub best_loss: Option<f64>,
    /// Final accuracy value
    pub final_accuracy: Option<f64>,
    /// Best (maximum) accuracy value
    pub best_accuracy: Option<f64>,
    /// Full loss history
    pub loss_history: Vec<f64>,
    /// Full accuracy history
    pub accuracy_history: Vec<f64>,
    /// Custom metrics
    pub custom_metrics: HashMap<String, Vec<f64>>,
}
