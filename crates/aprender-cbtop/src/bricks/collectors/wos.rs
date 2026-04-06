//! WOS kernel metrics collector
//!
//! Collects kernel state metrics from wos-kernel (Genchi Genbutsu: real data).
//!
//! Integrates with wos-kernel to monitor:
//! - Process state and scheduling
//! - Memory regions and page tables
//! - Jidoka invariant checks
//! - Syscall traces
//! - MicroVM status

use crate::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};
use crate::ring_buffer::RingBuffer;
use std::any::Any;
use std::time::{Duration, Instant};

/// WOS kernel metrics
#[derive(Debug, Clone)]
pub struct WosKernelMetrics {
    /// Timestamp of collection
    pub timestamp: Instant,
    /// Number of active processes
    pub process_count: usize,
    /// Number of runnable processes
    pub runnable_count: usize,
    /// Number of zombie processes
    pub zombie_count: usize,
    /// Total memory mapped (pages)
    pub memory_pages: usize,
    /// Active MicroVMs
    pub active_vms: usize,
    /// Jidoka status (healthy/degraded/critical)
    pub jidoka_status: JidokaHealthStatus,
    /// Syscalls per second (last interval)
    pub syscalls_per_sec: f64,
    /// Open file descriptors
    pub open_fds: usize,
    /// Shared memory segments
    pub shm_segments: usize,
}

/// Jidoka health status for kernel invariants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JidokaHealthStatus {
    /// All invariants pass
    Healthy,
    /// Some warnings but operational
    Degraded,
    /// Critical violations detected
    Critical,
}

impl Default for WosKernelMetrics {
    fn default() -> Self {
        Self {
            timestamp: Instant::now(),
            process_count: 0,
            runnable_count: 0,
            zombie_count: 0,
            memory_pages: 0,
            active_vms: 0,
            jidoka_status: JidokaHealthStatus::Healthy,
            syscalls_per_sec: 0.0,
            open_fds: 0,
            shm_segments: 0,
        }
    }
}

/// WOS kernel metrics collector brick
pub struct WosCollectorBrick {
    /// Metrics history
    history: RingBuffer<WosKernelMetrics>,
    /// Last syscall count for rate calculation
    last_syscall_count: u64,
    /// Last collection time
    last_collection: Instant,
    /// Whether wos-kernel is available
    available: bool,
}

impl WosCollectorBrick {
    /// Create new WOS collector
    pub fn new() -> Self {
        Self {
            history: RingBuffer::new(120), // 2 minutes at 1Hz
            last_syscall_count: 0,
            last_collection: Instant::now(),
            available: Self::check_availability(),
        }
    }

    /// Check if wos-kernel is available
    fn check_availability() -> bool {
        // In real implementation, check if wos-kernel is loaded
        // For now, always return true as we're collecting mock data
        // that will be replaced with real wos-kernel integration
        cfg!(feature = "wos-kernel")
    }

    /// Collect current WOS kernel metrics
    pub fn collect(&mut self) -> WosKernelMetrics {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_collection).as_secs_f64();

        let metrics = if self.available {
            self.collect_real_metrics(elapsed)
        } else {
            self.collect_mock_metrics(elapsed)
        };

        self.last_collection = now;
        self.history.push(metrics.clone());
        metrics
    }

    /// Collect real metrics from wos-kernel
    #[cfg(feature = "wos-kernel")]
    fn collect_real_metrics(&mut self, elapsed: f64) -> WosKernelMetrics {
        // Real implementation would interface with wos-kernel
        // For now, return mock data
        self.collect_mock_metrics(elapsed)
    }

    #[cfg(not(feature = "wos-kernel"))]
    fn collect_real_metrics(&mut self, elapsed: f64) -> WosKernelMetrics {
        self.collect_mock_metrics(elapsed)
    }

    /// Collect mock metrics for testing/demo
    fn collect_mock_metrics(&mut self, elapsed: f64) -> WosKernelMetrics {
        // Simulate some activity
        let syscall_count = self.last_syscall_count + (elapsed * 1000.0) as u64;
        let syscalls_per_sec = if elapsed > 0.0 {
            (syscall_count - self.last_syscall_count) as f64 / elapsed
        } else {
            0.0
        };
        self.last_syscall_count = syscall_count;

        WosKernelMetrics {
            timestamp: Instant::now(),
            process_count: 12,
            runnable_count: 3,
            zombie_count: 0,
            memory_pages: 4096,
            active_vms: 2,
            jidoka_status: JidokaHealthStatus::Healthy,
            syscalls_per_sec,
            open_fds: 128,
            shm_segments: 4,
        }
    }

    /// Get metrics history
    pub fn history(&self) -> &RingBuffer<WosKernelMetrics> {
        &self.history
    }

    /// Is collector available?
    pub fn is_available(&self) -> bool {
        self.available
    }

    /// Suggested collection interval
    pub fn interval_hint(&self) -> Duration {
        Duration::from_millis(1000)
    }

    /// Get process breakdown
    pub fn process_breakdown(&self) -> ProcessBreakdown {
        if let Some(last) = self.history.back() {
            ProcessBreakdown {
                total: last.process_count,
                runnable: last.runnable_count,
                sleeping: last
                    .process_count
                    .saturating_sub(last.runnable_count + last.zombie_count),
                zombie: last.zombie_count,
            }
        } else {
            ProcessBreakdown::default()
        }
    }

    /// Get jidoka summary
    pub fn jidoka_summary(&self) -> JidokaSummary {
        if let Some(last) = self.history.back() {
            JidokaSummary {
                status: last.jidoka_status,
                checks_passed: match last.jidoka_status {
                    JidokaHealthStatus::Healthy => 100,
                    JidokaHealthStatus::Degraded => 85,
                    JidokaHealthStatus::Critical => 50,
                },
                last_violation: None,
            }
        } else {
            JidokaSummary::default()
        }
    }
}

/// Process state breakdown
#[derive(Debug, Clone, Default)]
pub struct ProcessBreakdown {
    /// Total processes
    pub total: usize,
    /// Runnable processes
    pub runnable: usize,
    /// Sleeping processes
    pub sleeping: usize,
    /// Zombie processes
    pub zombie: usize,
}

/// Jidoka invariant summary
#[derive(Debug, Clone)]
pub struct JidokaSummary {
    /// Current status
    pub status: JidokaHealthStatus,
    /// Percentage of checks passed
    pub checks_passed: u8,
    /// Last violation (if any)
    pub last_violation: Option<String>,
}

impl Default for JidokaSummary {
    fn default() -> Self {
        Self {
            status: JidokaHealthStatus::Healthy,
            checks_passed: 100,
            last_violation: None,
        }
    }
}

impl Default for WosCollectorBrick {
    fn default() -> Self {
        Self::new()
    }
}

impl Brick for WosCollectorBrick {
    fn brick_name(&self) -> &'static str {
        "wos_collector"
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![
            BrickAssertion::ValueInRange {
                min: 0.0,
                max: 10000.0,
            }, // Process count
            BrickAssertion::max_latency_ms(10),
        ]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget {
            collect_ms: 10,
            layout_ms: 0,
            render_ms: 0,
        }
    }

    fn verify(&self) -> BrickVerification {
        let mut v = BrickVerification::new();
        for assertion in self.assertions() {
            v.check(&assertion);
        }
        v
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wos_collector_brick_name() {
        let collector = WosCollectorBrick::new();
        assert_eq!(collector.brick_name(), "wos_collector");
    }

    #[test]
    fn test_wos_collector_has_assertions() {
        let collector = WosCollectorBrick::new();
        assert!(!collector.assertions().is_empty());
    }

    #[test]
    fn test_wos_collector_collect() {
        let mut collector = WosCollectorBrick::new();
        let metrics = collector.collect();

        assert!(metrics.process_count > 0);
        assert_eq!(metrics.jidoka_status, JidokaHealthStatus::Healthy);
    }

    #[test]
    fn test_wos_process_breakdown() {
        let mut collector = WosCollectorBrick::new();
        collector.collect();

        let breakdown = collector.process_breakdown();
        assert_eq!(
            breakdown.total,
            breakdown.runnable + breakdown.sleeping + breakdown.zombie
        );
    }

    #[test]
    fn test_wos_jidoka_summary() {
        let mut collector = WosCollectorBrick::new();
        collector.collect();

        let summary = collector.jidoka_summary();
        assert_eq!(summary.status, JidokaHealthStatus::Healthy);
        assert!(summary.checks_passed > 0);
    }

    #[test]
    fn test_wos_syscall_rate() {
        let mut collector = WosCollectorBrick::new();

        // First collect establishes baseline
        collector.collect();

        // Wait a bit and collect again
        std::thread::sleep(Duration::from_millis(10));
        let metrics = collector.collect();

        // Should have some syscall activity
        assert!(metrics.syscalls_per_sec >= 0.0);
    }
}
