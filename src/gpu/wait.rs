//! Wait-for-VRAM polling queue (GPU-SHARE-003).
//!
//! Polls the VRAM ledger until sufficient budget is available or timeout.
//! Uses exponential backoff: 30s base, 300s max.
//!
//! # Contract
//!
//! - `wait_for_vram()` returns within `timeout + max_interval` (worst case)
//! - Each poll iteration prunes dead PIDs + expired leases
//! - CPU usage < 1% during wait (sleeping between polls)

use std::time::{Duration, Instant};

use super::error::GpuError;
use super::ledger::VramLedger;
use super::profiler::GpuProfiler;

/// Configuration for wait-for-VRAM polling.
pub struct WaitConfig {
    /// Maximum time to wait before giving up.
    pub timeout: Duration,
    /// Base poll interval (doubles each attempt, capped at max_interval).
    pub base_interval: Duration,
    /// Maximum poll interval.
    pub max_interval: Duration,
}

impl Default for WaitConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(3600),     // 1 hour
            base_interval: Duration::from_secs(30), // 30 seconds
            max_interval: Duration::from_secs(300), // 5 minutes
        }
    }
}

impl WaitConfig {
    /// Create config with custom timeout in seconds.
    pub fn with_timeout_secs(secs: u64) -> Self {
        Self { timeout: Duration::from_secs(secs), ..Default::default() }
    }

    /// Compute the sleep interval for a given attempt number.
    fn interval_for_attempt(&self, attempt: u32) -> Duration {
        let multiplier = 2u64.saturating_pow(attempt);
        let interval_secs = self.base_interval.as_secs().saturating_mul(multiplier);
        Duration::from_secs(interval_secs.min(self.max_interval.as_secs()))
    }
}

/// Poll the ledger until VRAM budget is available.
///
/// Returns Ok(()) when reservation is acquired,
/// or Err(Timeout) when timeout is exceeded.
pub fn wait_for_vram(
    ledger: &mut VramLedger,
    budget_mb: usize,
    task: &str,
    config: &WaitConfig,
    profiler: &mut GpuProfiler,
) -> Result<u64, GpuError> {
    let start = Instant::now();
    let mut attempt: u32 = 0;

    loop {
        // Check timeout BEFORE trying (not after sleep)
        if start.elapsed() > config.timeout {
            return Err(GpuError::Timeout { budget_mb, timeout_secs: config.timeout.as_secs() });
        }

        // Phase: wait_poll
        profiler.begin(GpuProfiler::WAIT_POLL);
        let result = ledger.try_reserve(budget_mb, task);
        profiler.end(GpuProfiler::WAIT_POLL);

        match result {
            Ok(reservation_id) => {
                profiler.finish_op();
                return Ok(reservation_id);
            }
            Err(GpuError::InsufficientMemory { available_mb, reserved_mb, .. }) => {
                let elapsed = start.elapsed();
                let remaining = config.timeout.saturating_sub(elapsed);

                eprintln!(
                    "[GPU] Waiting for {} MB VRAM ({} MB available, {} MB reserved) \
                     [{:.0}s elapsed, {:.0}s remaining]",
                    budget_mb,
                    available_mb,
                    reserved_mb,
                    elapsed.as_secs_f64(),
                    remaining.as_secs_f64(),
                );

                let interval = config.interval_for_attempt(attempt);
                // Don't sleep past the timeout
                let sleep_time = interval.min(remaining);
                std::thread::sleep(sleep_time);
                attempt = attempt.saturating_add(1);
            }
            Err(e) => return Err(e),
        }
    }
}

/// Compute the maximum wait duration bound for a given config.
///
/// Returns the configured timeout plus one max_interval (worst-case overshoot
/// from the last sleep before timeout check).
///
/// Contract: gpu-wait-queue-v1 / timeout_bound
pub fn timeout_bound(config: &WaitConfig) -> Duration {
    config.timeout + config.max_interval
}

/// Identify expired reservations eligible for reclamation (fairness via lease expiry).
///
/// Returns the reservation IDs that have exceeded their 24h lease and should
/// be pruned to prevent starvation of waiting processes.
///
/// Contract: gpu-wait-queue-v1 / fairness_via_expiry
pub fn fairness_via_expiry(ledger: &mut VramLedger) -> Vec<u32> {
    ledger
        .read_reservations()
        .unwrap_or_default()
        .iter()
        .filter(|r| r.is_expired())
        .map(|r| r.pid)
        .collect()
}

/// Produce a structured progress report for the current wait state.
///
/// Contract: gpu-wait-queue-v1 / progress_report
pub struct WaitProgress {
    /// Current attempt number
    pub attempt: u32,
    /// Time elapsed since wait started
    pub elapsed: Duration,
    /// Time remaining before timeout
    pub remaining: Duration,
    /// VRAM budget requested (MB)
    pub budget_mb: usize,
    /// VRAM currently available (MB)
    pub available_mb: usize,
    /// VRAM currently reserved by other processes (MB)
    pub reserved_mb: usize,
}

/// Build a progress report snapshot for the current wait state.
///
/// Contract: gpu-wait-queue-v1 / progress_report
pub fn progress_report(
    config: &WaitConfig,
    start: Instant,
    attempt: u32,
    budget_mb: usize,
    available_mb: usize,
    reserved_mb: usize,
) -> WaitProgress {
    let elapsed = start.elapsed();
    let remaining = config.timeout.saturating_sub(elapsed);
    WaitProgress { attempt, elapsed, remaining, budget_mb, available_mb, reserved_mb }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU32, Ordering};

    static TEST_COUNTER: AtomicU32 = AtomicU32::new(0);

    fn test_ledger_path() -> PathBuf {
        let n = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join("entrenar-wait-test");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(format!("wait-ledger-{n}-{}.json", std::process::id()))
    }

    fn cleanup(path: &std::path::Path) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(path.with_extension("tmp"));
    }

    #[test]
    fn test_immediate_success() {
        let path = test_ledger_path();
        let mut ledger = VramLedger::new("GPU-test".into(), 24000, 0.85).with_path(path.clone());
        let mut profiler = GpuProfiler::disabled();
        let config = WaitConfig::with_timeout_secs(5);

        let id = wait_for_vram(&mut ledger, 5000, "test", &config, &mut profiler).unwrap();
        assert!(id != 0);

        cleanup(&path);
    }

    #[test]
    fn test_timeout_when_full() {
        let path = test_ledger_path();
        let mut ledger = VramLedger::new("GPU-test".into(), 10000, 0.85).with_path(path.clone());

        // Fill the ledger
        ledger.try_reserve(8000, "blocker").unwrap();

        // Try to wait for more than available — should timeout quickly
        let mut profiler = GpuProfiler::disabled();
        let config = WaitConfig {
            timeout: Duration::from_millis(100),
            base_interval: Duration::from_millis(50),
            max_interval: Duration::from_millis(100),
        };

        let result = wait_for_vram(&mut ledger, 5000, "waiter", &config, &mut profiler);
        assert!(result.is_err());
        match result.unwrap_err() {
            GpuError::Timeout { budget_mb, .. } => assert_eq!(budget_mb, 5000),
            other => panic!("expected Timeout, got {other}"),
        }

        cleanup(&path);
    }

    #[test]
    fn test_interval_exponential_backoff() {
        let config = WaitConfig {
            base_interval: Duration::from_secs(30),
            max_interval: Duration::from_secs(300),
            ..Default::default()
        };

        assert_eq!(config.interval_for_attempt(0), Duration::from_secs(30));
        assert_eq!(config.interval_for_attempt(1), Duration::from_secs(60));
        assert_eq!(config.interval_for_attempt(2), Duration::from_secs(120));
        assert_eq!(config.interval_for_attempt(3), Duration::from_secs(240));
        assert_eq!(config.interval_for_attempt(4), Duration::from_secs(300)); // capped
        assert_eq!(config.interval_for_attempt(10), Duration::from_secs(300)); // still capped
    }

    #[test]
    fn test_expired_lease_unblocks_waiter() {
        let path = test_ledger_path();
        let mut blocker = VramLedger::new("GPU-test".into(), 10000, 0.85)
            .with_path(path.clone())
            .with_lease_hours(0); // Immediate expiry

        // Reserve the full capacity
        blocker.try_reserve(8000, "expiring").unwrap();
        // Forget the blocker so it doesn't release on Drop
        blocker.our_reservation_id = None;

        // Sleep to let the lease expire
        std::thread::sleep(Duration::from_millis(10));

        // Waiter should succeed because the lease expired
        let mut waiter = VramLedger::new("GPU-test".into(), 10000, 0.85).with_path(path.clone());
        let mut profiler = GpuProfiler::disabled();
        let config = WaitConfig::with_timeout_secs(5);

        let id = wait_for_vram(&mut waiter, 5000, "waiter", &config, &mut profiler).unwrap();
        assert!(id != 0);

        cleanup(&path);
    }
}
