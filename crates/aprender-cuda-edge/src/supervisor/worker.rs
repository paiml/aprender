//! GPU worker process types and spawning.
//!
//! Each GPU test runs in an isolated worker process to prevent one test's
//! GPU crash from corrupting another test's state.

use serde::{Deserialize, Serialize};

/// Opaque identifier for a worker task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkerTaskId(pub u64);

impl std::fmt::Display for WorkerTaskId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "worker-{}", self.0)
    }
}

/// Status of a worker's heartbeat.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HeartbeatStatus {
    /// Worker is alive and responding.
    Alive,
    /// Worker missed one or more heartbeats.
    MissedBeats(u32),
    /// Worker process has exited.
    Dead,
}

impl HeartbeatStatus {
    /// Returns true if the worker is healthy (alive).
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        matches!(self, Self::Alive)
    }
}

/// Configuration for spawning a GPU worker process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerConfig {
    /// Task identifier.
    pub task_id: WorkerTaskId,
    /// GPU device index to bind the worker to.
    pub device_index: u32,
    /// Maximum heartbeat misses before declaring worker dead.
    pub max_missed_heartbeats: u32,
    /// Timeout in milliseconds for each heartbeat interval.
    pub heartbeat_interval_ms: u64,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            task_id: WorkerTaskId(0),
            device_index: 0,
            max_missed_heartbeats: 3,
            heartbeat_interval_ms: 1000,
        }
    }
}

/// Spawn a worker process (requires CUDA runtime).
///
/// # Errors
///
/// Returns an error if the GPU device is not available.
#[cfg(feature = "cuda")]
pub fn spawn_worker(_config: &WorkerConfig) -> crate::error::Result<()> {
    // Real implementation would re-exec with WORKER_FLAG env var
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_task_id_display() {
        let id = WorkerTaskId(42);
        assert_eq!(id.to_string(), "worker-42");
    }

    #[test]
    fn heartbeat_alive_is_healthy() {
        assert!(HeartbeatStatus::Alive.is_healthy());
    }

    #[test]
    fn heartbeat_missed_is_not_healthy() {
        assert!(!HeartbeatStatus::MissedBeats(1).is_healthy());
        assert!(!HeartbeatStatus::MissedBeats(5).is_healthy());
    }

    #[test]
    fn heartbeat_dead_is_not_healthy() {
        assert!(!HeartbeatStatus::Dead.is_healthy());
    }

    #[test]
    fn worker_config_default() {
        let cfg = WorkerConfig::default();
        assert_eq!(cfg.device_index, 0);
        assert_eq!(cfg.max_missed_heartbeats, 3);
        assert_eq!(cfg.heartbeat_interval_ms, 1000);
    }
}
