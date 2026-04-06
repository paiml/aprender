//! SQLite Backend for Experiment Storage (MLOPS-001)
//!
//! Sovereign, local-first storage using SQLite with WAL mode.
//!
//! # Toyota Way: (Heijunka)
//!
//! SQLite provides consistent, predictable performance without external dependencies.
//!
//! # Example
//!
//! ```ignore
//! use entrenar::storage::{SqliteBackend, ExperimentStorage, RunStatus};
//!
//! let backend = SqliteBackend::open("./experiments.db")?;
//! let exp_id = backend.create_experiment("my-exp", None)?;
//! let run_id = backend.create_run(&exp_id)?;
//! backend.log_metric(&run_id, "loss", 0, 0.5)?;
//! ```
//!
//! # Module Organization
//!
//! - `backend` - Core SqliteBackend struct and basic CRUD operations
//! - `queries` - Search, list, and parameter filtering methods
//! - `metrics` - Metric logging and retrieval (implements ExperimentStorage trait)
//! - `artifacts` - Artifact storage and retrieval
//! - `types` - Type definitions (ParameterValue, Experiment, Run, ArtifactRef)

mod artifacts;
mod backend;
mod metrics;
mod queries;
mod types;

pub use backend::SqliteBackend;
pub use types::{ArtifactRef, Experiment, FilterOp, ParamFilter, ParameterValue, Run};
