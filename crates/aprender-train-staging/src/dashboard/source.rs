//! Dashboard data source trait and implementations.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::{MetricSnapshot, ResourceSnapshot};
use crate::run::Run;
use crate::storage::{ExperimentStorage, RunStatus};

/// Subscription callback type.
pub type SubscriptionCallback = Box<dyn Fn(&str, f64) + Send>;

/// Dashboard data source trait.
///
/// Implement this trait to provide data for real-time dashboards.
pub trait DashboardSource {
    /// Get the current run status.
    fn status(&self) -> RunStatus;

    /// Get recent metrics, limited to `limit` points per metric.
    fn recent_metrics(&self, limit: usize) -> HashMap<String, MetricSnapshot>;

    /// Subscribe to metric updates.
    ///
    /// Returns a receiver that will receive metric snapshots as they arrive.
    /// The callback is called with the metric key and latest value.
    fn subscribe(&self, callback: SubscriptionCallback);

    /// Get current resource usage.
    fn resource_usage(&self) -> ResourceSnapshot;
}

impl<S: ExperimentStorage> DashboardSource for Run<S> {
    fn status(&self) -> RunStatus {
        if self.is_finished() {
            // Query storage for actual status
            let storage = self.storage_ref();
            storage
                .lock()
                .ok()
                .and_then(|s| s.get_run_status(&self.id).ok())
                .unwrap_or(RunStatus::Success)
        } else {
            RunStatus::Running
        }
    }

    fn recent_metrics(&self, limit: usize) -> HashMap<String, MetricSnapshot> {
        let mut result = HashMap::new();

        // Get metrics from storage
        let storage = self.storage_ref();
        if let Ok(guard) = storage.lock() {
            // Get all metric keys we've logged
            for key in self.metric_keys() {
                if let Ok(points) = guard.get_metrics(&self.id, &key) {
                    // Take only the most recent `limit` points
                    let recent: Vec<_> = if points.len() > limit {
                        points[points.len() - limit..].to_vec()
                    } else {
                        points
                    };

                    let snapshot = MetricSnapshot::from_points(&key, &recent);
                    result.insert(key, snapshot);
                }
            }
        }

        result
    }

    fn subscribe(&self, _callback: SubscriptionCallback) {
        // Subscriptions are handled at a higher level
        // This is a placeholder for the trait requirement
        // In practice, use a channel-based approach or web sockets for WASM
    }

    fn resource_usage(&self) -> ResourceSnapshot {
        // Return simulated resource usage
        // In production, this would query actual system metrics
        ResourceSnapshot::new().with_cpu_util(0.0).with_gpu_util(0.0).with_memory(0, 0)
    }
}

impl<S: ExperimentStorage> Run<S> {
    /// Get a reference to the storage (for dashboard use).
    pub(crate) fn storage_ref(&self) -> &Arc<Mutex<S>> {
        &self.storage
    }

    /// Get all metric keys that have been logged.
    pub fn metric_keys(&self) -> Vec<String> {
        self.step_counters.keys().cloned().collect()
    }
}
