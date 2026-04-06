//! Experiment Store - in-memory storage for experiment tracking data
//!
//! This module provides the storage layer for experiment tracking,
//! optimized for time-series metric queries.

use std::collections::HashMap;

use super::{ExperimentRecord, MetricRecord, RunRecord};

/// In-memory store for experiment tracking data.
///
/// ## Design
///
/// The store uses hash maps for O(1) lookups by ID, and stores metrics
/// in a vector that can be filtered and sorted for time-series queries.
///
/// ## Time-Series Optimization
///
/// The `get_metrics_for_run` function returns metrics ordered by step,
/// enabling efficient time-series visualization and analysis.
#[derive(Debug, Default)]
pub struct ExperimentStore {
    experiments: HashMap<String, ExperimentRecord>,
    runs: HashMap<String, RunRecord>,
    metrics: Vec<MetricRecord>,
}

impl ExperimentStore {
    /// Create a new empty experiment store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if the store is empty (no experiments, runs, or metrics).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.experiments.is_empty() && self.runs.is_empty() && self.metrics.is_empty()
    }

    /// Get the number of experiments in the store.
    #[must_use]
    pub fn experiment_count(&self) -> usize {
        self.experiments.len()
    }

    /// Get the number of runs in the store.
    #[must_use]
    pub fn run_count(&self) -> usize {
        self.runs.len()
    }

    /// Get the number of metrics in the store.
    #[must_use]
    pub fn metric_count(&self) -> usize {
        self.metrics.len()
    }

    /// Add an experiment to the store.
    pub fn add_experiment(&mut self, experiment: ExperimentRecord) {
        self.experiments.insert(experiment.experiment_id().to_string(), experiment);
    }

    /// Get an experiment by ID.
    #[must_use]
    pub fn get_experiment(&self, experiment_id: &str) -> Option<&ExperimentRecord> {
        self.experiments.get(experiment_id)
    }

    /// Add a run to the store.
    pub fn add_run(&mut self, run: RunRecord) {
        self.runs.insert(run.run_id().to_string(), run);
    }

    /// Get a run by ID.
    #[must_use]
    pub fn get_run(&self, run_id: &str) -> Option<&RunRecord> {
        self.runs.get(run_id)
    }

    /// Get all runs for an experiment.
    #[must_use]
    pub fn get_runs_for_experiment(&self, experiment_id: &str) -> Vec<&RunRecord> {
        self.runs.values().filter(|run| run.experiment_id() == experiment_id).collect()
    }

    /// Add a metric to the store.
    pub fn add_metric(&mut self, metric: MetricRecord) {
        self.metrics.push(metric);
    }

    /// Get metrics for a specific run and key, ordered by step.
    ///
    /// This is the primary query function for time-series metric data.
    ///
    /// ## Arguments
    ///
    /// * `run_id` - The ID of the run to query
    /// * `key` - The metric key/name to filter by
    ///
    /// ## Returns
    ///
    /// A vector of metrics matching the `run_id` and key, sorted by step
    /// in ascending order.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use trueno_db::experiment::{ExperimentStore, MetricRecord};
    ///
    /// let mut store = ExperimentStore::new();
    ///
    /// // Log some training metrics
    /// for step in 0..100 {
    ///     let loss = 1.0 / (step as f64 + 1.0);
    ///     store.add_metric(MetricRecord::new("run-001", "loss", step, loss));
    /// }
    ///
    /// // Query the loss curve
    /// let loss_metrics = store.get_metrics_for_run("run-001", "loss");
    /// assert_eq!(loss_metrics.len(), 100);
    /// ```
    #[must_use]
    pub fn get_metrics_for_run(&self, run_id: &str, key: &str) -> Vec<MetricRecord> {
        let mut metrics: Vec<MetricRecord> = self
            .metrics
            .iter()
            .filter(|m| m.run_id() == run_id && m.key() == key)
            .cloned()
            .collect();

        // Sort by step for time-series ordering
        metrics.sort_by_key(MetricRecord::step);

        metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_default() {
        let store = ExperimentStore::new();
        assert!(store.is_empty());
        assert_eq!(store.experiment_count(), 0);
        assert_eq!(store.run_count(), 0);
        assert_eq!(store.metric_count(), 0);
    }

    #[test]
    fn test_store_add_and_get() {
        let mut store = ExperimentStore::new();

        let experiment = ExperimentRecord::new("exp-1", "Test");
        store.add_experiment(experiment);

        let run = RunRecord::new("run-1", "exp-1");
        store.add_run(run);

        let metric = MetricRecord::new("run-1", "loss", 0, 0.5);
        store.add_metric(metric);

        assert!(!store.is_empty());
        assert!(store.get_experiment("exp-1").is_some());
        assert!(store.get_run("run-1").is_some());
    }

    #[test]
    fn test_get_metrics_for_run_ordering() {
        let mut store = ExperimentStore::new();

        // Add out of order
        store.add_metric(MetricRecord::new("run-1", "loss", 2, 0.2));
        store.add_metric(MetricRecord::new("run-1", "loss", 0, 0.0));
        store.add_metric(MetricRecord::new("run-1", "loss", 1, 0.1));

        let metrics = store.get_metrics_for_run("run-1", "loss");

        assert_eq!(metrics.len(), 3);
        assert_eq!(metrics[0].step(), 0);
        assert_eq!(metrics[1].step(), 1);
        assert_eq!(metrics[2].step(), 2);
    }
}
