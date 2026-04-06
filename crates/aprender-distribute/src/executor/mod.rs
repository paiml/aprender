//! Execution backends for repartir.
//!
//! This module defines the `Executor` trait and provides implementations
//! for different backends (CPU, GPU, Remote).

pub mod cpu;

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "microvm")]
pub mod microvm;

#[cfg(feature = "remote")]
pub mod remote;

#[cfg(feature = "remote-tls")]
pub mod tls;

#[cfg(feature = "simd")]
pub mod simd;

use crate::error::Result;
use crate::task::{ExecutionResult, Task};
use std::future::Future;
use std::pin::Pin;

/// Type alias for boxed async futures.
///
/// Used to enable dynamic dispatch for async trait methods.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Trait for executing tasks on different backends.
///
/// Executors are responsible for running tasks and returning results.
/// Each backend (CPU, GPU, Remote) implements this trait.
///
/// # Safety
///
/// Implementations must ensure that:
/// - Tasks are executed in isolation (no cross-task state leakage)
/// - Resources are properly cleaned up (no leaks)
/// - Failures are reported, not panicked
pub trait Executor: Send + Sync {
    /// Executes a task and returns the result.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The binary cannot be found
    /// - Execution fails
    /// - Timeout is exceeded
    fn execute(&self, task: Task) -> BoxFuture<'_, Result<ExecutionResult>>;

    /// Returns the number of available workers/slots.
    ///
    /// For CPU: number of threads in pool
    /// For GPU: number of GPU devices
    /// For Remote: number of connected workers
    fn capacity(&self) -> usize;

    /// Returns a human-readable name for this executor.
    fn name(&self) -> &'static str;
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::no_effect_underscore_binding,
    clippy::doc_markdown
)]
mod tests {
    use super::*;
    use crate::executor::cpu::CpuExecutor;

    /// Tests that the Executor trait is object-safe (can be used with dyn).
    #[test]
    fn test_executor_trait_object_safe() {
        let executor = CpuExecutor::new();
        let _dyn_executor: &dyn Executor = &executor;
        assert!(executor.capacity() > 0);
    }

    /// Tests that BoxFuture type alias works correctly.
    #[test]
    fn test_box_future_type_alias() {
        fn create_box_future<'a>() -> BoxFuture<'a, i32> {
            Box::pin(async { 42 })
        }

        let future = create_box_future();
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let result = runtime.block_on(future);
        assert_eq!(result, 42);
    }

    /// Tests executor name method.
    #[test]
    fn test_executor_name() {
        let executor = CpuExecutor::new();
        assert_eq!(executor.name(), "CPU");
    }
}
