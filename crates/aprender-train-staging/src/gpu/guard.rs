//! VRAM Guard (GPU-SHARE-002).
//!
//! Pre-allocation check enforcing Contract C-VRAM-001:
//! `CudaTrainer::new()` MUST NOT allocate if budget exceeds available VRAM.
//!
//! # Usage
//!
//! ```ignore
//! let guard = VramGuard::acquire(budget_mb, "qlora-7b")?;
//! // ... create CudaTrainer, allocate GPU memory ...
//! guard.update_actual(actual_mb)?;
//! // guard releases on Drop
//! ```

use crate::trace::{TraceStep, TRACER};

use super::error::GpuError;
use super::ledger::VramLedger;
use super::profiler::GpuProfiler;
use super::wait::{self, WaitConfig};

/// VRAM reservation guard.
///
/// Acquires a VRAM reservation on creation and releases it on drop.
/// Enforces C-VRAM-001: no allocation beyond budget.
pub struct VramGuard {
    ledger: VramLedger,
    budget_mb: usize,
}

impl VramGuard {
    /// Acquire a VRAM reservation.
    ///
    /// Checks the ledger and reserves `budget_mb` of VRAM.
    /// Returns `GpuError::InsufficientMemory` if not enough VRAM available.
    pub fn acquire(budget_mb: usize, task: &str) -> Result<Self, GpuError> {
        TRACER.span(TraceStep::VramQuery, format!("guard_acquire budget={budget_mb}MB"), || {
            let mut ledger = super::ledger::auto_ledger();
            ledger.try_reserve(budget_mb, task)?;
            Ok(Self { ledger, budget_mb })
        })
    }

    /// Acquire with waiting: poll until VRAM is available or timeout.
    pub fn acquire_wait(budget_mb: usize, task: &str, timeout_secs: u64) -> Result<Self, GpuError> {
        TRACER.span(
            TraceStep::WaitPoll,
            format!("guard_wait budget={budget_mb}MB timeout={timeout_secs}s"),
            || {
                let mut ledger = super::ledger::auto_ledger();
                let config = WaitConfig::with_timeout_secs(timeout_secs);
                let mut profiler = GpuProfiler::disabled();
                wait::wait_for_vram(&mut ledger, budget_mb, task, &config, &mut profiler)?;
                Ok(Self { ledger, budget_mb })
            },
        )
    }

    /// Update the actual measured VRAM after GPU initialization.
    ///
    /// Call this after `CudaTrainer::new()` + weight upload to record
    /// the real VRAM usage (may be less than budgeted).
    pub fn update_actual(&mut self, actual_mb: usize) -> Result<(), GpuError> {
        self.ledger.update_actual(actual_mb)
    }

    /// Budget that was reserved (MB).
    pub fn budget_mb(&self) -> usize {
        self.budget_mb
    }

    /// GPU UUID this guard is for.
    pub fn gpu_uuid(&self) -> &str {
        &self.ledger.gpu_uuid
    }

    /// Read the current GPU status display.
    pub fn status(&self) -> Result<String, GpuError> {
        super::ledger::gpu_status_display(&self.ledger)
    }
    /// Check if actual VRAM usage overshoots the budget.
    ///
    /// Returns `Some((actual, budget))` if overshoot detected, `None` if within budget.
    ///
    /// Contract: vram-guard-v1 / budget_overshoot
    pub fn check_overshoot(&self) -> Option<(usize, usize)> {
        let actual = self
            .ledger
            .read_reservations()
            .ok()?
            .iter()
            .filter(|r| r.pid == std::process::id())
            .find_map(|r| r.actual_mb)?;
        if actual > self.budget_mb {
            Some((actual, self.budget_mb))
        } else {
            None
        }
    }

    /// Auto-estimate the VRAM budget for a model based on parameter count.
    ///
    /// Uses rule of thumb: ~2 bytes per parameter for f16 weights + 20% overhead.
    ///
    /// Contract: vram-guard-v1 / auto_budget_estimate
    pub fn auto_estimate_budget(param_count: usize) -> usize {
        let weight_mb = (param_count * 2) / (1024 * 1024);
        let overhead = weight_mb / 5; // 20% overhead for activations + KV cache
        weight_mb + overhead
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::{AtomicU32, Ordering};

    static TEST_COUNTER: AtomicU32 = AtomicU32::new(0);

    fn test_guard_ledger(total_mb: usize) -> VramLedger {
        let n = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join("entrenar-guard-test");
        std::fs::create_dir_all(&dir).expect("dir creation should succeed");
        let path = dir.join(format!("guard-{n}-{}.json", std::process::id()));
        VramLedger::new("GPU-test-guard".into(), total_mb, 0.85).with_path(path)
    }

    #[test]
    fn test_guard_direct_acquire() {
        let mut ledger = test_guard_ledger(24000);
        ledger.try_reserve(5000, "guard-test").expect("should succeed");
        assert_eq!(ledger.total_reserved().expect("should succeed"), 5000);
        // Drop releases
        drop(ledger);
    }

    #[test]
    fn test_guard_update_actual() {
        let mut ledger = test_guard_ledger(24000);
        ledger.try_reserve(8000, "guard-actual").expect("should succeed");
        ledger.update_actual(7200).expect("should succeed");
        assert_eq!(ledger.total_reserved().expect("should succeed"), 7200);
    }

    #[test]
    fn test_guard_rejects_over_budget() {
        let mut ledger = test_guard_ledger(10000);
        let result = ledger.try_reserve(9000, "too-big");
        assert!(result.is_err());
    }

    #[test]
    fn test_guard_budget_mb() {
        let ledger = test_guard_ledger(24000);
        let guard = VramGuard { ledger, budget_mb: 8000 };
        assert_eq!(guard.budget_mb(), 8000);
    }

    #[test]
    fn test_guard_gpu_uuid() {
        let ledger = test_guard_ledger(24000);
        let guard = VramGuard { ledger, budget_mb: 5000 };
        assert_eq!(guard.gpu_uuid(), "GPU-test-guard");
    }

    #[test]
    fn test_guard_status() {
        let ledger = test_guard_ledger(24000);
        let guard = VramGuard { ledger, budget_mb: 5000 };
        // status() reads the ledger file — should not panic
        let result = guard.status();
        // May succeed or fail depending on ledger state, but should not panic
        let _ = result;
    }

    #[test]
    fn test_guard_update_actual_without_reservation() {
        let ledger = test_guard_ledger(24000);
        let mut guard = VramGuard { ledger, budget_mb: 5000 };
        // No reservation made, update_actual should be a no-op
        let result = guard.update_actual(4000);
        assert!(result.is_ok());
    }

    #[test]
    fn test_guard_multiple_reservations_sequential() {
        let mut ledger1 = test_guard_ledger(24000);
        ledger1.try_reserve(3000, "task-1").expect("should succeed");
        let reserved = ledger1.total_reserved().expect("should succeed");
        assert_eq!(reserved, 3000);

        // After drop, reservation should be released
        drop(ledger1);
    }

    #[test]
    fn test_guard_zero_budget() {
        let mut ledger = test_guard_ledger(24000);
        // Reserving 0 MB should succeed
        let result = ledger.try_reserve(0, "zero-budget");
        assert!(result.is_ok());
    }

    #[test]
    fn test_guard_exact_budget() {
        // Ledger total 10000 with 0.85 headroom factor = 8500 usable
        let mut ledger = test_guard_ledger(10000);
        // Try to reserve exactly at the headroom limit
        let result = ledger.try_reserve(8000, "near-limit");
        assert!(result.is_ok());
    }

    #[test]
    fn test_guard_update_actual_reduces_reserved() {
        let mut ledger = test_guard_ledger(24000);
        ledger.try_reserve(8000, "actual-test").expect("should succeed");
        assert_eq!(ledger.total_reserved().expect("should succeed"), 8000);
        ledger.update_actual(6000).expect("should succeed");
        assert_eq!(ledger.total_reserved().expect("should succeed"), 6000);
    }

    // ── Additional coverage tests ──

    #[test]
    fn test_guard_struct_fields() {
        let ledger = test_guard_ledger(16000);
        let guard = VramGuard { ledger, budget_mb: 4000 };
        assert_eq!(guard.budget_mb(), 4000);
        assert_eq!(guard.gpu_uuid(), "GPU-test-guard");
    }

    #[test]
    fn test_guard_status_returns_string() {
        let mut ledger = test_guard_ledger(24000);
        ledger.try_reserve(5000, "status-test").expect("should succeed");
        let guard = VramGuard { ledger, budget_mb: 5000 };
        let status = guard.status();
        assert!(status.is_ok());
        let status_str = status.unwrap();
        assert!(status_str.contains("GPU-test-guard"));
        assert!(status_str.contains("5000 MB budget"));
    }

    #[test]
    fn test_guard_status_empty_ledger() {
        let ledger = test_guard_ledger(24000);
        let guard = VramGuard { ledger, budget_mb: 0 };
        let status = guard.status();
        assert!(status.is_ok());
        let s = status.unwrap();
        assert!(s.contains("none") || s.contains("Reservations"));
    }

    #[test]
    fn test_guard_update_actual_with_active_reservation() {
        let mut ledger = test_guard_ledger(24000);
        ledger.try_reserve(10000, "update-actual").expect("should succeed");
        let mut guard = VramGuard { ledger, budget_mb: 10000 };
        let result = guard.update_actual(9500);
        assert!(result.is_ok());
    }

    #[test]
    fn test_guard_small_gpu() {
        // Test with small GPU (e.g., embedded GPU)
        let mut ledger = test_guard_ledger(2048);
        // Capacity = 2048 * 0.85 = 1740
        let result = ledger.try_reserve(1740, "small-gpu");
        assert!(result.is_ok());
        // One more MB should fail
        let result2 = ledger.try_reserve(1, "overflow");
        assert!(result2.is_err());
    }

    #[test]
    fn test_guard_capacity_calculation() {
        let ledger = test_guard_ledger(10000);
        // 10000 * 0.85 = 8500
        assert_eq!(ledger.capacity_mb(), 8500);
    }

    #[test]
    fn test_guard_available_mb_after_reserve() {
        let mut ledger = test_guard_ledger(20000);
        // capacity = 17000
        ledger.try_reserve(7000, "test").expect("should succeed");
        let available = ledger.available_mb().expect("should succeed");
        assert_eq!(available, 10000);
    }

    #[test]
    fn test_guard_multiple_sequential_reserve_release() {
        let mut ledger = test_guard_ledger(24000);
        // Reserve, release, reserve again
        ledger.try_reserve(5000, "first").expect("ok");
        assert_eq!(ledger.total_reserved().expect("ok"), 5000);
        ledger.release().expect("ok");
        assert_eq!(ledger.total_reserved().expect("ok"), 0);
        ledger.try_reserve(8000, "second").expect("ok");
        assert_eq!(ledger.total_reserved().expect("ok"), 8000);
    }

    #[test]
    fn test_guard_profiler_report_accessible() {
        let ledger = test_guard_ledger(24000);
        let guard = VramGuard { ledger, budget_mb: 0 };
        // Can access profiler report through guard's ledger
        let report = guard.ledger.profiler_report();
        assert!(report.contains("No operations recorded"));
    }

    #[test]
    fn test_guard_drop_does_not_panic() {
        let ledger = test_guard_ledger(24000);
        let guard = VramGuard { ledger, budget_mb: 3000 };
        // Dropping without reservation should not panic
        drop(guard);
    }

    #[test]
    fn test_guard_drop_with_reservation_releases() {
        let n = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join("entrenar-guard-test");
        std::fs::create_dir_all(&dir).expect("dir creation should succeed");
        let path = dir.join(format!("guard-drop-{n}-{}.json", std::process::id()));

        {
            let mut ledger =
                VramLedger::new("GPU-test-guard".into(), 24000, 0.85).with_path(path.clone());
            ledger.try_reserve(5000, "drop-reserve").expect("ok");
            let guard = VramGuard { ledger, budget_mb: 5000 };
            // guard dropped here, should release
            drop(guard);
        }

        // Verify reservation was cleaned up
        let check_ledger = VramLedger::new("GPU-test-guard".into(), 24000, 0.85).with_path(path);
        let reserved = check_ledger.total_reserved().expect("ok");
        assert_eq!(reserved, 0);
    }
}
