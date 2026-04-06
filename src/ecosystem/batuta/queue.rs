//! Queue state types and ETA adjustment.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Queue state information.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QueueState {
    /// Current queue depth (number of waiting jobs)
    pub queue_depth: u32,
    /// Average wait time in seconds
    pub avg_wait_seconds: u64,
    /// Number of available GPUs
    pub available_gpus: u32,
    /// Total GPUs in the pool
    pub total_gpus: u32,
    /// Estimated time until next available slot (seconds)
    pub eta_seconds: Option<u64>,
}

impl QueueState {
    /// Create a new queue state.
    pub fn new(queue_depth: u32, available_gpus: u32, total_gpus: u32) -> Self {
        Self { queue_depth, avg_wait_seconds: 0, available_gpus, total_gpus, eta_seconds: None }
    }

    /// Set average wait time.
    pub fn with_avg_wait(mut self, seconds: u64) -> Self {
        self.avg_wait_seconds = seconds;
        self
    }

    /// Set ETA to next available slot.
    pub fn with_eta(mut self, seconds: u64) -> Self {
        self.eta_seconds = Some(seconds);
        self
    }

    /// Check if GPUs are immediately available.
    pub fn is_available(&self) -> bool {
        self.available_gpus > 0
    }

    /// Calculate queue utilization (0.0 to 1.0).
    pub fn utilization(&self) -> f64 {
        if self.total_gpus == 0 {
            return 1.0;
        }
        1.0 - (f64::from(self.available_gpus) / f64::from(self.total_gpus))
    }
}

/// Adjust estimated completion time based on queue state.
///
/// Takes a base ETA and adjusts it based on:
/// - Current queue depth
/// - Average wait time
/// - Queue utilization
pub fn adjust_eta(base_eta_seconds: u64, queue_state: &QueueState) -> Duration {
    let mut adjusted = base_eta_seconds;

    // Add queue wait time if not immediately available
    if !queue_state.is_available() {
        // Estimate wait based on queue depth and average wait time
        let queue_wait = if queue_state.avg_wait_seconds > 0 {
            u64::from(queue_state.queue_depth) * queue_state.avg_wait_seconds
        } else {
            // Default: 5 minutes per queued job
            u64::from(queue_state.queue_depth) * 300
        };
        adjusted += queue_wait;
    }

    // Add ETA from queue state if available
    if let Some(eta) = queue_state.eta_seconds {
        adjusted = adjusted.max(eta);
    }

    // Apply utilization multiplier (high utilization = longer times)
    let utilization = queue_state.utilization();
    if utilization > 0.8 {
        let multiplier = 1.0 + (utilization - 0.8) * 2.0; // Up to 40% increase at 100% util
        adjusted = (adjusted as f64 * multiplier) as u64;
    }

    Duration::from_secs(adjusted)
}
