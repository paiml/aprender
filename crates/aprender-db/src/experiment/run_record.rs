//! Run Record - execution instance of an experiment

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Status of a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunStatus {
    /// Run is created but not yet started.
    Pending,
    /// Run is currently executing.
    Running,
    /// Run completed successfully.
    Success,
    /// Run failed with an error.
    Failed,
    /// Run was cancelled by user or system.
    Cancelled,
}

/// Run Record represents a single execution of an experiment.
///
/// Each experiment can have multiple runs. A run tracks the execution
/// lifecycle from start to completion.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RunRecord {
    run_id: String,
    experiment_id: String,
    status: RunStatus,
    started_at: Option<DateTime<Utc>>,
    ended_at: Option<DateTime<Utc>>,
    renacer_span_id: Option<String>,
}

impl RunRecord {
    /// Create a new run record in Pending status.
    ///
    /// # Arguments
    ///
    /// * `run_id` - Unique identifier for the run
    /// * `experiment_id` - ID of the parent experiment
    #[must_use]
    pub fn new(run_id: impl Into<String>, experiment_id: impl Into<String>) -> Self {
        Self {
            run_id: run_id.into(),
            experiment_id: experiment_id.into(),
            status: RunStatus::Pending,
            started_at: None,
            ended_at: None,
            renacer_span_id: None,
        }
    }

    /// Create a builder for constructing a run record with optional fields.
    #[must_use]
    pub fn builder(
        run_id: impl Into<String>,
        experiment_id: impl Into<String>,
    ) -> RunRecordBuilder {
        RunRecordBuilder::new(run_id, experiment_id)
    }

    /// Get the run ID.
    #[must_use]
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// Get the parent experiment ID.
    #[must_use]
    pub fn experiment_id(&self) -> &str {
        &self.experiment_id
    }

    /// Get the current run status.
    #[must_use]
    pub const fn status(&self) -> RunStatus {
        self.status
    }

    /// Get the start timestamp, if the run has started.
    #[must_use]
    pub const fn started_at(&self) -> Option<DateTime<Utc>> {
        self.started_at
    }

    /// Get the end timestamp, if the run has completed.
    #[must_use]
    pub const fn ended_at(&self) -> Option<DateTime<Utc>> {
        self.ended_at
    }

    /// Get the renacer span ID for distributed tracing, if set.
    #[must_use]
    pub fn renacer_span_id(&self) -> Option<&str> {
        self.renacer_span_id.as_deref()
    }

    /// Start the run, transitioning from Pending to Running.
    ///
    /// Sets the `started_at` timestamp to now.
    pub fn start(&mut self) {
        self.status = RunStatus::Running;
        self.started_at = Some(Utc::now());
    }

    /// Complete the run with the given final status.
    ///
    /// Sets the `ended_at` timestamp to now.
    ///
    /// # Arguments
    ///
    /// * `status` - Final status (Success, Failed, or Cancelled)
    pub fn complete(&mut self, status: RunStatus) {
        self.status = status;
        self.ended_at = Some(Utc::now());
    }
}

/// Builder for `RunRecord`.
#[derive(Debug)]
#[allow(clippy::struct_field_names)]
pub struct RunRecordBuilder {
    run_id: String,
    experiment_id: String,
    renacer_span_id: Option<String>,
}

impl RunRecordBuilder {
    /// Create a new builder with required fields.
    #[must_use]
    pub fn new(run_id: impl Into<String>, experiment_id: impl Into<String>) -> Self {
        Self { run_id: run_id.into(), experiment_id: experiment_id.into(), renacer_span_id: None }
    }

    /// Set the renacer span ID for distributed tracing.
    #[must_use]
    pub fn renacer_span_id(mut self, span_id: impl Into<String>) -> Self {
        self.renacer_span_id = Some(span_id.into());
        self
    }

    /// Build the `RunRecord`.
    #[must_use]
    pub fn build(self) -> RunRecord {
        RunRecord {
            run_id: self.run_id,
            experiment_id: self.experiment_id,
            status: RunStatus::Pending,
            started_at: None,
            ended_at: None,
            renacer_span_id: self.renacer_span_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_status_default() {
        let run = RunRecord::new("run-1", "exp-1");
        assert_eq!(run.status(), RunStatus::Pending);
    }

    #[test]
    fn test_run_lifecycle() {
        let mut run = RunRecord::new("run-1", "exp-1");
        run.start();
        assert_eq!(run.status(), RunStatus::Running);
        run.complete(RunStatus::Success);
        assert_eq!(run.status(), RunStatus::Success);
    }
}
