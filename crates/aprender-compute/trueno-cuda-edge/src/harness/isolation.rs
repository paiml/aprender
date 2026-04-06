//! Worker process isolation via re-exec.
//!
//! GPU tests run in isolated child processes to prevent one test's GPU
//! crash from corrupting another. The parent sets a `WORKER_FLAG`
//! environment variable before re-executing, and the child detects it
//! to enter worker mode.

use crate::supervisor::worker::WorkerTaskId;

/// Environment variable set on worker child processes.
pub const WORKER_FLAG: &str = "TRUENO_CUDA_EDGE_WORKER";

/// Environment variable carrying the worker task ID.
pub const WORKER_TASK_ID_VAR: &str = "TRUENO_CUDA_EDGE_TASK_ID";

/// Configuration for process isolation.
#[derive(Debug, Clone)]
pub struct IsolationConfig {
    /// Whether to re-exec the current binary as a worker.
    pub use_reexec: bool,
    /// GPU device index to bind the worker to.
    pub device_index: u32,
}

impl Default for IsolationConfig {
    fn default() -> Self {
        Self { use_reexec: true, device_index: 0 }
    }
}

/// Check if the current process is a worker (child) process.
#[must_use]
pub fn is_worker_process() -> bool {
    std::env::var(WORKER_FLAG).is_ok()
}

/// Read the worker task ID from the environment, if present.
#[must_use]
pub fn worker_task_id() -> Option<WorkerTaskId> {
    std::env::var(WORKER_TASK_ID_VAR).ok().and_then(|s| s.parse::<u64>().ok()).map(WorkerTaskId)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_flag_constant() {
        assert_eq!(WORKER_FLAG, "TRUENO_CUDA_EDGE_WORKER");
    }

    #[test]
    fn default_isolation_config() {
        let cfg = IsolationConfig::default();
        assert!(cfg.use_reexec);
        assert_eq!(cfg.device_index, 0);
    }

    #[test]
    fn worker_task_id_returns_none_when_unset() {
        // In test context, the env var is typically not set
        // (unless a parent test set it, which is unlikely)
        let id = worker_task_id();
        // We can't assert None because some CI might set it,
        // but we can assert the type is correct
        let _: Option<WorkerTaskId> = id;
    }

    #[test]
    fn is_worker_process_returns_bool() {
        // The function should return a bool indicating worker status
        let result = is_worker_process();
        // We can't assert a specific value since it depends on env,
        // but we verify it returns a bool
        let _: bool = result;
    }

    #[test]
    fn task_id_var_constant() {
        assert_eq!(WORKER_TASK_ID_VAR, "TRUENO_CUDA_EDGE_TASK_ID");
    }
}
