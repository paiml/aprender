//! Test harness utilities.
//!
//! Provides GPU detection, process isolation, and thermal monitoring
//! for the edge-case test framework.
//!
//! # Process Isolation
//!
//! GPU tests run in isolated child processes to prevent one test's GPU
//! crash from corrupting another. The parent sets `TRUENO_CUDA_EDGE_WORKER`
//! before re-executing, and the child detects it to enter worker mode.
//!
//! # Example
//!
//! ```
//! use trueno_cuda_edge::harness::{
//!     is_worker_process, gpu_available, ThermalConfig, evaluate_thermal
//! };
//! use trueno_cuda_edge::supervisor::HealthAction;
//!
//! // Check if running as worker
//! if is_worker_process() {
//!     // Child worker mode
//! }
//!
//! // Thermal evaluation
//! let config = ThermalConfig::default();
//! assert_eq!(evaluate_thermal(&config, 70), HealthAction::Healthy);
//! assert_eq!(evaluate_thermal(&config, 95), HealthAction::Shutdown);
//! ```

pub mod gpu_detect;
pub mod isolation;
pub mod thermal;

pub use gpu_detect::{detect_gpus, gpu_available, GpuCapability};
pub use isolation::{
    is_worker_process, worker_task_id, IsolationConfig, WORKER_FLAG, WORKER_TASK_ID_VAR,
};
pub use thermal::{evaluate_thermal, ThermalConfig};
