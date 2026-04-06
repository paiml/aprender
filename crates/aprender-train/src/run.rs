//! Run Struct with Renacer Integration (ENT-002)
//!
//! Provides the `Run` struct which wraps experiment tracking with
//! distributed tracing via Renacer spans.
//!
//! # Example
//!
//! ```
//! use std::sync::{Arc, Mutex};
//! use entrenar::storage::{InMemoryStorage, ExperimentStorage};
//! use entrenar::run::{Run, TracingConfig};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut storage = InMemoryStorage::new();
//! let exp_id = storage.create_experiment("my-exp", None)?;
//! let storage = Arc::new(Mutex::new(storage));
//!
//! let config = TracingConfig::default();
//! let mut run = Run::new(&exp_id, storage.clone(), config)?;
//!
//! // Log metrics - auto-increments step
//! run.log_metric("loss", 0.5)?;
//! run.log_metric("loss", 0.4)?;
//!
//! // Or log with explicit step
//! run.log_metric_at("accuracy", 0, 0.85)?;
//!
//! // Complete the run
//! run.finish(entrenar::storage::RunStatus::Success)?;
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::storage::{ExperimentStorage, Result, RunStatus, StorageError};

/// Configuration for distributed tracing
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TracingConfig {
    /// Whether tracing is enabled (creates Renacer spans)
    pub tracing_enabled: bool,

    /// Whether to export traces via OTLP
    pub export_otlp: bool,

    /// Path for golden trace storage
    pub golden_trace_path: Option<PathBuf>,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self { tracing_enabled: true, export_otlp: false, golden_trace_path: None }
    }
}

impl TracingConfig {
    /// Create a disabled tracing configuration
    pub fn disabled() -> Self {
        Self { tracing_enabled: false, export_otlp: false, golden_trace_path: None }
    }

    /// Enable OTLP export
    pub fn with_otlp_export(mut self) -> Self {
        self.export_otlp = true;
        self
    }

    /// Set golden trace path
    pub fn with_golden_trace_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.golden_trace_path = Some(path.into());
        self
    }
}

/// A training run with integrated distributed tracing
///
/// Generic over the storage backend, allowing different backends
/// for production (TruenoBackend) and testing (InMemoryStorage).
pub struct Run<S: ExperimentStorage> {
    /// Run ID
    pub id: String,
    /// Parent experiment ID
    pub experiment_id: String,
    /// Storage backend (shared)
    pub(crate) storage: Arc<Mutex<S>>,
    /// Renacer span ID (if tracing enabled)
    span: Option<String>,
    /// Tracing configuration
    config: TracingConfig,
    /// Current step counters per metric key
    pub(crate) step_counters: HashMap<String, u64>,
    /// Whether the run has been finished
    finished: bool,
}

impl<S: ExperimentStorage> Run<S> {
    /// Acquire the storage mutex, mapping poison errors to `StorageError::Backend`.
    fn lock_storage(storage: &Arc<Mutex<S>>) -> Result<std::sync::MutexGuard<'_, S>> {
        storage.lock().map_err(|e| StorageError::Backend(format!("mutex poisoned: {e}")))
    }

    /// Acquire the storage mutex on `self`, mapping poison errors to `StorageError::Backend`.
    fn lock(&self) -> Result<std::sync::MutexGuard<'_, S>> {
        Self::lock_storage(&self.storage)
    }

    /// Create a new run with tracing
    ///
    /// Creates a run in the storage backend, starts it, and optionally
    /// creates a Renacer span for distributed tracing.
    ///
    /// # Arguments
    ///
    /// * `experiment_id` - ID of the parent experiment
    /// * `storage` - Shared storage backend
    /// * `config` - Tracing configuration
    pub fn new(experiment_id: &str, storage: Arc<Mutex<S>>, config: TracingConfig) -> Result<Self> {
        // Create run in storage
        let run_id = {
            let mut store = Self::lock_storage(&storage)?;
            let run_id = store.create_run(experiment_id)?;
            store.start_run(&run_id)?;
            run_id
        };

        // Create span if tracing enabled
        let span = if config.tracing_enabled {
            let span_id = Self::create_span(&run_id);
            Self::lock_storage(&storage)?.set_span_id(&run_id, &span_id)?;
            Some(span_id)
        } else {
            None
        };

        Ok(Self {
            id: run_id,
            experiment_id: experiment_id.to_string(),
            storage,
            span,
            config,
            step_counters: HashMap::new(),
            finished: false,
        })
    }

    /// Create a Renacer span for this run
    fn create_span(run_id: &str) -> String {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();

        format!("span-{}-{}", run_id, now.as_nanos())
    }

    /// Log a metric value, auto-incrementing the step
    ///
    /// Each metric key has its own step counter that starts at 0
    /// and increments with each call.
    ///
    /// # Arguments
    ///
    /// * `key` - Metric name (e.g., "loss", "accuracy")
    /// * `value` - Metric value
    pub fn log_metric(&mut self, key: &str, value: f64) -> Result<()> {
        let step = *self.step_counters.get(key).unwrap_or(&0);
        self.log_metric_at(key, step, value)?;
        self.step_counters.insert(key.to_string(), step + 1);
        Ok(())
    }

    /// Log a metric value at a specific step
    ///
    /// # Arguments
    ///
    /// * `key` - Metric name
    /// * `step` - Training step
    /// * `value` - Metric value
    pub fn log_metric_at(&mut self, key: &str, step: u64, value: f64) -> Result<()> {
        if self.finished {
            return Err(StorageError::InvalidState("Cannot log to finished run".to_string()));
        }

        self.lock()?.log_metric(&self.id, key, step, value)?;

        // Emit span event if tracing enabled
        if self.config.tracing_enabled {
            self.emit_metric_event(key, step, value);
        }

        Ok(())
    }

    /// Emit a metric event to the Renacer span
    fn emit_metric_event(&self, key: &str, step: u64, value: f64) {
        // In a full implementation, this would call renacer::record_event()
        if self.span.is_some() {
            let _ = (key, step, value);
        }
    }

    /// Finish the run with the given status
    ///
    /// Completes the run in storage and ends the Renacer span.
    /// Consumes the Run to prevent further operations.
    ///
    /// # Arguments
    ///
    /// * `status` - Final run status
    pub fn finish(mut self, status: RunStatus) -> Result<()> {
        if self.finished {
            return Ok(());
        }

        self.lock()?.complete_run(&self.id, status)?;

        self.finished = true;

        // End span if tracing enabled
        if self.config.tracing_enabled {
            self.end_span();
        }

        Ok(())
    }

    /// End the Renacer span
    fn end_span(&self) {
        // In a full implementation, this would call span.end()
        let _ = self.span.as_ref();
    }

    /// Get the Renacer span ID (if tracing is enabled)
    pub fn span_id(&self) -> Option<&str> {
        self.span.as_deref()
    }

    /// Get the run ID
    pub fn run_id(&self) -> &str {
        &self.id
    }

    /// Get the tracing configuration
    pub fn tracing_config(&self) -> &TracingConfig {
        &self.config
    }

    /// Check if the run has been finished
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Get current step for a metric key
    pub fn current_step(&self, key: &str) -> u64 {
        *self.step_counters.get(key).unwrap_or(&0)
    }
}

impl<S: ExperimentStorage> std::fmt::Debug for Run<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Run")
            .field("id", &self.id)
            .field("experiment_id", &self.experiment_id)
            .field("span", &self.span)
            .field("finished", &self.finished)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::InMemoryStorage;

    fn setup_storage() -> (Arc<Mutex<InMemoryStorage>>, String) {
        let mut storage = InMemoryStorage::new();
        let exp_id = storage.create_experiment("test-exp", None).expect("operation should succeed");
        (Arc::new(Mutex::new(storage)), exp_id)
    }

    #[test]
    fn test_tracing_config_default() {
        let config = TracingConfig::default();
        assert!(config.tracing_enabled);
        assert!(!config.export_otlp);
        assert!(config.golden_trace_path.is_none());
    }

    #[test]
    fn test_tracing_config_disabled() {
        let config = TracingConfig::disabled();
        assert!(!config.tracing_enabled);
    }

    #[test]
    fn test_tracing_config_builder() {
        let config =
            TracingConfig::default().with_otlp_export().with_golden_trace_path("/tmp/golden");

        assert!(config.tracing_enabled);
        assert!(config.export_otlp);
        assert_eq!(config.golden_trace_path, Some(PathBuf::from("/tmp/golden")));
    }

    #[test]
    fn test_run_new_creates_span() {
        let (storage, exp_id) = setup_storage();
        let config = TracingConfig::default();

        let run = Run::new(&exp_id, storage, config).expect("config should be valid");

        assert!(run.span_id().is_some());
        assert!(run.span_id().expect("operation should succeed").starts_with("span-"));
    }

    #[test]
    fn test_run_new_without_tracing() {
        let (storage, exp_id) = setup_storage();
        let config = TracingConfig::disabled();

        let run = Run::new(&exp_id, storage, config).expect("config should be valid");

        assert!(run.span_id().is_none());
    }

    #[test]
    fn test_run_log_metric_auto_increment() {
        let (storage, exp_id) = setup_storage();
        let config = TracingConfig::disabled();

        let mut run = Run::new(&exp_id, storage.clone(), config).expect("config should be valid");

        run.log_metric("loss", 0.5).expect("operation should succeed");
        run.log_metric("loss", 0.4).expect("operation should succeed");
        run.log_metric("loss", 0.3).expect("operation should succeed");

        assert_eq!(run.current_step("loss"), 3);

        let metrics = storage
            .lock()
            .expect("lock acquisition should succeed")
            .get_metrics(&run.id, "loss")
            .expect("lock acquisition should succeed");
        assert_eq!(metrics.len(), 3);
        assert_eq!(metrics[0].step, 0);
        assert_eq!(metrics[1].step, 1);
        assert_eq!(metrics[2].step, 2);
    }

    #[test]
    fn test_run_log_metric_at_explicit_step() {
        let (storage, exp_id) = setup_storage();
        let config = TracingConfig::disabled();

        let mut run = Run::new(&exp_id, storage.clone(), config).expect("config should be valid");

        run.log_metric_at("accuracy", 0, 0.7).expect("operation should succeed");
        run.log_metric_at("accuracy", 10, 0.8).expect("operation should succeed");
        run.log_metric_at("accuracy", 20, 0.9).expect("operation should succeed");

        let metrics = storage
            .lock()
            .expect("lock acquisition should succeed")
            .get_metrics(&run.id, "accuracy")
            .expect("lock acquisition should succeed");
        assert_eq!(metrics.len(), 3);
        assert_eq!(metrics[0].step, 0);
        assert_eq!(metrics[1].step, 10);
        assert_eq!(metrics[2].step, 20);
    }

    #[test]
    fn test_run_multiple_metrics() {
        let (storage, exp_id) = setup_storage();
        let config = TracingConfig::disabled();

        let mut run = Run::new(&exp_id, storage.clone(), config).expect("config should be valid");

        run.log_metric("loss", 0.5).expect("operation should succeed");
        run.log_metric("accuracy", 0.8).expect("operation should succeed");
        run.log_metric("loss", 0.4).expect("operation should succeed");

        assert_eq!(run.current_step("loss"), 2);
        assert_eq!(run.current_step("accuracy"), 1);
    }

    #[test]
    fn test_run_finish_success() {
        let (storage, exp_id) = setup_storage();
        let config = TracingConfig::disabled();

        let run = Run::new(&exp_id, storage.clone(), config).expect("config should be valid");
        let run_id = run.id.clone();

        run.finish(RunStatus::Success).expect("operation should succeed");

        let status = storage
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get_run_status(&run_id)
            .expect("operation should succeed");
        assert_eq!(status, RunStatus::Success);
    }

    #[test]
    fn test_run_finish_failed() {
        let (storage, exp_id) = setup_storage();
        let config = TracingConfig::disabled();

        let run = Run::new(&exp_id, storage.clone(), config).expect("config should be valid");
        let run_id = run.id.clone();

        run.finish(RunStatus::Failed).expect("operation should succeed");

        let status = storage
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get_run_status(&run_id)
            .expect("operation should succeed");
        assert_eq!(status, RunStatus::Failed);
    }

    #[test]
    fn test_run_stores_span_id() {
        let (storage, exp_id) = setup_storage();
        let config = TracingConfig::default();

        let run = Run::new(&exp_id, storage.clone(), config).expect("config should be valid");
        let span_id = run.span_id().expect("operation should succeed").to_string();

        let stored_span = storage
            .lock()
            .expect("lock acquisition should succeed")
            .get_span_id(&run.id)
            .expect("lock acquisition should succeed")
            .expect("lock acquisition should succeed");
        assert_eq!(stored_span, span_id);
    }

    #[test]
    fn test_run_accessors() {
        let (storage, exp_id) = setup_storage();
        let config = TracingConfig::default();

        let run = Run::new(&exp_id, storage, config).expect("config should be valid");

        assert!(!run.is_finished());
        assert!(run.run_id().starts_with("run-"));
        assert!(run.tracing_config().tracing_enabled);
    }

    #[test]
    fn test_run_debug() {
        let (storage, exp_id) = setup_storage();
        let config = TracingConfig::disabled();

        let run = Run::new(&exp_id, storage, config).expect("config should be valid");
        let debug_str = format!("{run:?}");

        assert!(debug_str.contains("Run"));
        assert!(debug_str.contains(&run.id));
    }
}
