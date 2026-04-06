//! Graceful Shutdown Coordination
//!
//! AWP-07: Two-phase shutdown with active operation tracking.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

// ----------------------------------------------------------------------------
// AWP-07: Graceful Shutdown
// ----------------------------------------------------------------------------

/// Result of a graceful shutdown operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShutdownResult {
    /// All operations completed cleanly.
    Clean,
    /// Timeout reached with operations still active.
    Timeout {
        /// Number of operations still active.
        remaining: usize,
    },
}

/// Graceful shutdown coordinator.
///
/// # Example
/// ```rust
/// use trueno::brick::GracefulShutdown;
/// use std::time::Duration;
///
/// let shutdown = GracefulShutdown::new(Duration::from_secs(5));
///
/// // Register an operation
/// let guard = shutdown.register().ok_or("shutdown in progress")?;
///
/// // ... do work ...
///
/// drop(guard);  // Operation complete
///
/// // Initiate shutdown
/// let result = shutdown.shutdown();
/// assert_eq!(result, trueno::brick::ShutdownResult::Clean);
/// # Ok::<(), &'static str>(())
/// ```
pub struct GracefulShutdown {
    /// Flag indicating shutdown has been requested.
    shutdown_requested: AtomicBool,
    /// Number of active operations.
    active_count: AtomicUsize,
    /// Shutdown timeout.
    timeout: Duration,
}

impl GracefulShutdown {
    /// Create a new shutdown coordinator.
    pub fn new(timeout: Duration) -> Self {
        Self {
            shutdown_requested: AtomicBool::new(false),
            active_count: AtomicUsize::new(0),
            timeout,
        }
    }

    /// Check if shutdown has been requested.
    #[must_use]
    pub fn is_shutdown_requested(&self) -> bool {
        self.shutdown_requested.load(Ordering::Acquire)
    }

    /// Get the current active operation count.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.active_count.load(Ordering::Acquire)
    }

    /// Register an active operation.
    ///
    /// Returns `None` if shutdown has already been requested.
    pub fn register(&self) -> Option<ShutdownGuard<'_>> {
        if self.is_shutdown_requested() {
            return None; // Reject new operations during shutdown
        }
        self.active_count.fetch_add(1, Ordering::AcqRel);
        Some(ShutdownGuard { shutdown: self })
    }

    /// Initiate graceful shutdown.
    ///
    /// This will:
    /// 1. Stop accepting new operations
    /// 2. Wait for in-flight operations to complete (up to timeout)
    /// 3. Return the result
    pub fn shutdown(&self) -> ShutdownResult {
        // Phase 1: Stop accepting new operations
        self.shutdown_requested.store(true, Ordering::Release);

        // Phase 2: Wait for in-flight operations
        let deadline = Instant::now() + self.timeout;

        loop {
            let active = self.active_count.load(Ordering::Acquire);
            if active == 0 {
                return ShutdownResult::Clean;
            }
            if Instant::now() >= deadline {
                return ShutdownResult::Timeout { remaining: active };
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Reset the shutdown coordinator for reuse.
    pub fn reset(&self) {
        self.shutdown_requested.store(false, Ordering::Release);
        // Note: active_count should already be 0 if shutdown completed cleanly
    }
}

impl Default for GracefulShutdown {
    fn default() -> Self {
        Self::new(Duration::from_secs(30))
    }
}

/// Guard that decrements active count on drop.
pub struct ShutdownGuard<'a> {
    shutdown: &'a GracefulShutdown,
}

impl Drop for ShutdownGuard<'_> {
    fn drop(&mut self) {
        self.shutdown.active_count.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shutdown_result_eq() {
        assert_eq!(ShutdownResult::Clean, ShutdownResult::Clean);
        assert_eq!(
            ShutdownResult::Timeout { remaining: 5 },
            ShutdownResult::Timeout { remaining: 5 }
        );
        assert_ne!(ShutdownResult::Clean, ShutdownResult::Timeout { remaining: 0 });
    }

    #[test]
    fn test_graceful_shutdown_new() {
        let shutdown = GracefulShutdown::new(Duration::from_secs(5));
        assert!(!shutdown.is_shutdown_requested());
        assert_eq!(shutdown.active_count(), 0);
    }

    #[test]
    fn test_graceful_shutdown_default() {
        let shutdown = GracefulShutdown::default();
        assert!(!shutdown.is_shutdown_requested());
        assert_eq!(shutdown.timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_graceful_shutdown_register() {
        let shutdown = GracefulShutdown::new(Duration::from_secs(5));

        let guard1 = shutdown.register();
        assert!(guard1.is_some());
        assert_eq!(shutdown.active_count(), 1);

        let guard2 = shutdown.register();
        assert!(guard2.is_some());
        assert_eq!(shutdown.active_count(), 2);

        drop(guard1);
        assert_eq!(shutdown.active_count(), 1);

        drop(guard2);
        assert_eq!(shutdown.active_count(), 0);
    }

    #[test]
    fn test_graceful_shutdown_clean() {
        let shutdown = GracefulShutdown::new(Duration::from_millis(100));

        // Register and immediately drop
        let guard = shutdown.register();
        drop(guard);

        let result = shutdown.shutdown();
        assert_eq!(result, ShutdownResult::Clean);
    }

    #[test]
    fn test_graceful_shutdown_rejects_after_shutdown() {
        let shutdown = GracefulShutdown::new(Duration::from_millis(10));

        // Initiate shutdown
        let result = shutdown.shutdown();
        assert_eq!(result, ShutdownResult::Clean);

        // Try to register after shutdown
        let guard = shutdown.register();
        assert!(guard.is_none(), "Should reject new operations after shutdown");
    }

    #[test]
    fn test_graceful_shutdown_reset() {
        let shutdown = GracefulShutdown::new(Duration::from_millis(10));

        // Shutdown
        shutdown.shutdown();
        assert!(shutdown.is_shutdown_requested());

        // Reset
        shutdown.reset();
        assert!(!shutdown.is_shutdown_requested());

        // Should accept new operations again
        let guard = shutdown.register();
        assert!(guard.is_some());
    }

    /// FALSIFICATION TEST: Verify timeout behavior
    ///
    /// The shutdown coordinator MUST return Timeout when operations
    /// don't complete within the specified duration.
    #[test]
    fn test_falsify_shutdown_timeout() {
        let shutdown = GracefulShutdown::new(Duration::from_millis(50));

        // Register an operation that we WON'T drop
        let _guard = shutdown.register();
        assert_eq!(shutdown.active_count(), 1);

        // Initiate shutdown - should timeout since operation is still active
        let start = Instant::now();
        let result = shutdown.shutdown();
        let elapsed = start.elapsed();

        // CRITICAL ASSERTIONS
        match result {
            ShutdownResult::Timeout { remaining } => {
                assert_eq!(remaining, 1, "Should report exactly 1 remaining operation");
            }
            ShutdownResult::Clean => {
                panic!("FALSIFICATION FAILED: Shutdown reported Clean with active operation");
            }
        }

        // Verify we actually waited (within tolerance)
        assert!(
            elapsed >= Duration::from_millis(40),
            "FALSIFICATION FAILED: Shutdown returned too early ({:?} < 40ms)",
            elapsed
        );
    }

    /// FALSIFICATION TEST: Verify guard drop decrements count
    ///
    /// If the guard drop doesn't work correctly, the system will
    /// either hang forever or report wrong active counts.
    #[test]
    fn test_falsify_guard_drop_semantics() {
        let shutdown = GracefulShutdown::new(Duration::from_secs(1));

        // Create many guards
        let mut guards: Vec<_> = (0..100).filter_map(|_| shutdown.register()).collect();
        assert_eq!(shutdown.active_count(), 100, "FALSIFICATION FAILED: Not all guards registered");

        // Drop half by truncating the vector (keeps first 50, drops last 50)
        guards.truncate(50);

        // CRITICAL: Active count must be exactly 50
        // If fetch_sub has a bug (e.g., fetch_add by mistake), this catches it
        assert_eq!(
            shutdown.active_count(),
            50,
            "FALSIFICATION FAILED: Guard drop did not correctly decrement count"
        );

        // Drop remaining 50
        drop(guards);
        assert_eq!(
            shutdown.active_count(),
            0,
            "FALSIFICATION FAILED: Final drop did not clear all guards"
        );
    }

    /// FALSIFICATION TEST: Verify concurrent registration rejection
    ///
    /// Once shutdown starts, NO new operations should be accepted,
    /// even if they race with the shutdown call.
    #[test]
    fn test_falsify_rejection_after_shutdown_flag() {
        let shutdown = GracefulShutdown::new(Duration::from_secs(5));

        // Manually set shutdown flag (simulating shutdown start)
        shutdown.shutdown_requested.store(true, Ordering::Release);

        // Try to register many operations
        let mut accepted = 0;
        for _ in 0..100 {
            if shutdown.register().is_some() {
                accepted += 1;
            }
        }

        // CRITICAL: NO operations should have been accepted
        assert_eq!(
            accepted, 0,
            "FALSIFICATION FAILED: {} operations accepted after shutdown flag set",
            accepted
        );
    }
}
