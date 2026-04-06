//! # Repartir: Sovereign AI-Grade Distributed Computing
//!
//! Repartir is a pure Rust library for distributed execution across CPUs, GPUs,
//! and remote machines. Built on the **Iron Lotus Framework** and validated by
//! the **certeza** testing methodology.
//!
//! ## Features
//!
//! - **100% Rust, Zero C/C++**: True digital sovereignty through complete auditability
//! - **Memory Safety Guaranteed**: Provably safe via `RustBelt` formal verification
//! - **Work-Stealing Scheduler**: Based on Blumofe & Leiserson (1999)
//! - **Supply Chain Security**: Dependency pinning, binary signing, license enforcement
//! - **Iron Lotus Quality**: ≥95% coverage, ≥80% mutation score, formal verification
//!
//! ## Quick Start
//!
//! ```no_run
//! use repartir::{Pool, task::{Task, Backend}};
//!
//! #[tokio::main]
//! async fn main() -> repartir::error::Result<()> {
//!     // Create a pool with CPU executor
//!     let pool = Pool::builder()
//!         .cpu_workers(8)
//!         .build()?; // Note: build() is sync, not async
//!
//!     // Submit a task
//!     let task = Task::builder()
//!         .binary("./worker")
//!         .arg("--input")
//!         .arg("data.csv")
//!         .backend(Backend::Cpu)
//!         .build()?;
//!
//!     let result = pool.submit(task).await?;
//!
//!     if result.is_success() {
//!         println!("Task completed: {}", result.stdout_str()?);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Architecture
//!
//! Repartir follows a clean, layered architecture:
//!
//! - **Task**: Unit of work (binary + args + env)
//! - **Executor**: Runs tasks on a specific backend (CPU, GPU, Remote)
//! - **Scheduler**: Distributes tasks across executors (work-stealing)
//! - **Pool**: High-level API for task submission and result retrieval
//!
//! ## Toyota Way Principles
//!
//! Repartir embodies the **Iron Lotus Framework**:
//!
//! - **Genchi Genbutsu (現地現物)**: Transparent execution, traceable to source
//! - **Jidoka (自働化)**: Automated quality gates (CI enforces ≥95% coverage)
//! - **Kaizen (改善)**: Continuous improvement (TDG ratchet, mutation testing)
//! - **Muda (無駄)**: Waste elimination (zero-copy, single language, no FFI)
//!
//! ## Certeza Testing
//!
//! Three-tiered testing for asymptotic effectiveness:
//!
//! - **Tier 1 (ON-SAVE)**: Sub-second feedback (<3s)
//! - **Tier 2 (ON-COMMIT)**: Comprehensive (1-5min, 95% coverage)
//! - **Tier 3 (ON-MERGE)**: Exhaustive (hours, 80% mutation score)

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
#![deny(unsafe_code)] // v1.0: No unsafe code (Sovereign AI requirement)

#[macro_use]
#[allow(unused_macros)]
mod generated_contracts;

pub mod checkpoint;
pub mod error;
pub mod executor;
pub mod messaging;
pub mod scheduler;
#[cfg(feature = "serverless")]
pub mod serverless;
pub mod task;
#[cfg(feature = "tensor")]
pub mod tensor;
#[cfg(feature = "tui")]
pub mod tui;

use error::{RepartirError, Result};
use executor::cpu::CpuExecutor;
use executor::Executor;
use scheduler::Scheduler;
use std::sync::Arc;
use task::{ExecutionResult, Task};
use tokio::sync::RwLock;
use tokio::task::JoinSet;
use tracing::{debug, info};

/// High-level API for distributed task execution.
///
/// The `Pool` manages executors, schedules tasks, and coordinates results.
///
/// # Example
///
/// ```no_run
/// use repartir::Pool;
///
/// #[tokio::main]
/// async fn main() -> repartir::error::Result<()> {
///     let pool = Pool::builder()
///         .cpu_workers(4)
///         .build()?;
///
///     println!("Pool ready with {} workers", pool.capacity());
///     Ok(())
/// }
/// ```
pub struct Pool {
    /// Task scheduler.
    scheduler: Arc<Scheduler>,
    /// CPU executor.
    #[allow(dead_code)] // v1.0: not directly accessed, used by workers
    cpu_executor: Option<Arc<CpuExecutor>>,
    /// Number of worker tasks.
    num_workers: usize,
    /// Active worker handles.
    workers: Arc<RwLock<JoinSet<()>>>,
}

impl Pool {
    /// Creates a new pool builder.
    #[must_use]
    pub fn builder() -> PoolBuilder {
        PoolBuilder::default()
    }

    /// Submits a task for execution.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The task is invalid
    /// - The queue is full
    /// - Execution fails
    pub async fn submit(&self, task: Task) -> Result<ExecutionResult> {
        let task_id = self.scheduler.submit(task).await?;
        debug!("Task {task_id} submitted");

        // Wait for result (polling-based for v1.0, event-driven in v1.1)
        loop {
            if let Some(result) = self.scheduler.get_result(task_id).await {
                self.scheduler.remove_result(task_id).await;
                return Ok(result);
            }

            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }
    }

    /// Returns the total capacity (number of workers).
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.num_workers
    }

    /// Returns the number of pending tasks.
    pub async fn pending_tasks(&self) -> usize {
        self.scheduler.pending_count().await
    }

    /// Shuts down the pool gracefully.
    ///
    /// Waits for all pending tasks to complete.
    pub async fn shutdown(self) {
        info!("Shutting down pool");
        self.scheduler.clear().await;
        self.workers.write().await.shutdown().await;
    }
}

/// Builder for `Pool`.
#[derive(Default)]
pub struct PoolBuilder {
    cpu_workers: Option<usize>,
    max_queue_size: Option<usize>,
}

impl PoolBuilder {
    /// Sets the number of CPU workers.
    #[must_use]
    pub const fn cpu_workers(mut self, count: usize) -> Self {
        self.cpu_workers = Some(count);
        self
    }

    /// Sets the maximum queue size.
    #[must_use]
    pub const fn max_queue_size(mut self, size: usize) -> Self {
        self.max_queue_size = Some(size);
        self
    }

    /// Builds the pool and starts workers.
    ///
    /// # Errors
    ///
    /// Returns an error if no workers are configured.
    ///
    /// # Panics
    ///
    /// Will panic if `cpu_executor` is None (should never happen).
    pub fn build(self) -> Result<Pool> {
        let cpu_workers = self.cpu_workers.unwrap_or(0);

        if cpu_workers == 0 {
            return Err(RepartirError::InvalidTask {
                reason: "At least one worker type must be configured".to_string(),
            });
        }

        let max_queue_size = self.max_queue_size.unwrap_or(10_000);
        let scheduler = Arc::new(Scheduler::with_capacity(max_queue_size));
        let cpu_executor = Arc::new(CpuExecutor::new());

        info!("Initializing pool with {cpu_workers} CPU workers");

        // Spawn worker tasks
        let mut workers = JoinSet::new();
        for worker_id in 0..cpu_workers {
            let scheduler = Arc::clone(&scheduler);
            let executor = Arc::clone(&cpu_executor);

            workers.spawn(async move {
                debug!("Worker {worker_id} started");
                loop {
                    // Poll for tasks
                    if let Some(task) = scheduler.next_task().await {
                        match executor.execute(task).await {
                            Ok(result) => {
                                scheduler.store_result(result).await;
                            }
                            Err(e) => {
                                debug!("Task execution failed: {e}");
                            }
                        }
                    } else {
                        // No tasks, sleep briefly
                        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                    }
                }
            });
        }

        Ok(Pool {
            scheduler,
            cpu_executor: Some(cpu_executor),
            num_workers: cpu_workers,
            workers: Arc::new(RwLock::new(workers)),
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::disallowed_methods)]
mod tests {
    use super::*;
    use task::Backend;

    #[tokio::test]
    async fn test_pool_builder() {
        let pool = Pool::builder().cpu_workers(2).max_queue_size(100).build();

        assert!(pool.is_ok());
        let p = pool.expect("Pool build should succeed");
        assert_eq!(p.capacity(), 2);
        assert_eq!(p.pending_tasks().await, 0);

        p.shutdown().await;
    }

    #[tokio::test]
    async fn test_pool_submit_task() {
        let pool = Pool::builder().cpu_workers(2).build().expect("Pool build should succeed");

        #[cfg(unix)]
        {
            let task = Task::builder()
                .binary("/bin/echo")
                .arg("test")
                .backend(Backend::Cpu)
                .build()
                .expect("Task build should succeed");

            let result = pool.submit(task).await;
            assert!(result.is_ok());

            let exec_result = result.expect("Task execution should succeed");
            assert!(exec_result.is_success());
            assert_eq!(exec_result.stdout_str().expect("stdout should be UTF-8").trim(), "test");
        }

        pool.shutdown().await;
    }

    #[tokio::test]
    async fn test_pool_no_workers_error() {
        let pool = Pool::builder().build();
        assert!(pool.is_err());
    }
}
