//! Experiment Tracking Schema (Phase 5: ENT-EPIC-001)
//!
//! This module provides the data structures for experiment tracking,
//! designed to integrate with entrenar's `ExperimentStorage`.
//!
//! ## Schema Overview
//!
//! ```text
//! ExperimentRecord (1) ──< RunRecord (N)
//!                              │
//!                              ├──< MetricRecord (N) [time-series]
//!                              └──< ArtifactRecord (N) [CAS]
//! ```
//!
//! ## Usage
//!
//! ```rust
//! use trueno_db::experiment::{ExperimentRecord, RunRecord, MetricRecord, RunStatus};
//!
//! // Create an experiment
//! let experiment = ExperimentRecord::new("exp-001", "My Experiment");
//!
//! // Start a run
//! let mut run = RunRecord::new("run-001", experiment.experiment_id());
//! run.start();
//!
//! // Log metrics
//! let metric = MetricRecord::new(run.run_id(), "loss", 0, 0.5);
//!
//! // Complete the run
//! run.complete(RunStatus::Success);
//! ```

mod artifact_record;
mod experiment_record;
mod metric_record;
mod run_record;
mod store;

pub use artifact_record::ArtifactRecord;
pub use experiment_record::{ExperimentRecord, ExperimentRecordBuilder};
pub use metric_record::{MetricRecord, MetricRecordBuilder};
pub use run_record::{RunRecord, RunRecordBuilder, RunStatus};
pub use store::ExperimentStore;
