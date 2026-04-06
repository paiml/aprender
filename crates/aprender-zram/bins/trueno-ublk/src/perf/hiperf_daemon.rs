//! High-Performance Daemon Integration (PERF-001)
//!
//! Integrates polling mode, batch coalescing, CPU affinity, and NUMA awareness
//! into the ublk daemon I/O path for maximum throughput.
//!
//! Target: 800K+ IOPS (currently 225K baseline)
//!
//! Status: Polling mode, affinity, and NUMA are integrated in daemon.

use super::{
    affinity::CpuAffinity,
    polling::{PollResult, PollingConfig, PollingLoop, PollingStats},
    PerfConfig, TenXContext,
};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// High-performance I/O context for the daemon
///
/// Combines all PERF-001 optimizations into a unified interface.
/// Note: NUMA allocation is handled directly in daemon.rs during buffer allocation.
/// Note: Batch coalescing is handled by BatchedPageStore for compression.
/// Note: 10X optimizations (PERF-005 through PERF-012) handled via TenXContext.
pub struct HiPerfContext {
    /// Polling loop for spin-wait on completions
    polling: Option<PollingLoop>,
    /// CPU affinity configuration
    affinity: Option<CpuAffinity>,
    /// Configuration
    config: PerfConfig,
    /// Statistics
    stats: HiPerfStats,
    /// 10X optimization context (PERF-005 through PERF-012)
    tenx: Option<TenXContext>,
}

/// Statistics for high-performance I/O
#[derive(Debug, Default)]
pub struct HiPerfStats {
    /// Total I/O operations processed
    pub total_ios: AtomicU64,
    /// I/O operations processed via polling (no interrupt)
    pub polled_ios: AtomicU64,
    /// I/O operations processed via interrupt (yielded)
    pub interrupted_ios: AtomicU64,
    /// Batches submitted
    pub batches_submitted: AtomicU64,
    /// Total pages in batches
    pub batched_pages: AtomicU64,
    /// Sequential I/O runs detected
    pub sequential_runs: AtomicU64,
    /// CPU affinity applied
    pub affinity_applied: AtomicBool,
    /// NUMA node used
    pub numa_node: AtomicU64,
}

impl HiPerfStats {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a polled I/O completion
    pub fn record_polled(&self, count: u32) {
        self.total_ios.fetch_add(count as u64, Ordering::Relaxed);
        self.polled_ios.fetch_add(count as u64, Ordering::Relaxed);
    }

    /// Record an interrupted I/O completion (after yield)
    pub fn record_interrupted(&self, count: u32) {
        self.total_ios.fetch_add(count as u64, Ordering::Relaxed);
        self.interrupted_ios.fetch_add(count as u64, Ordering::Relaxed);
    }

    /// Record a batch submission
    pub fn record_batch(&self, pages: usize, is_sequential: bool) {
        self.batches_submitted.fetch_add(1, Ordering::Relaxed);
        self.batched_pages.fetch_add(pages as u64, Ordering::Relaxed);
        if is_sequential {
            self.sequential_runs.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get polling efficiency (polled / total)
    pub fn polling_efficiency(&self) -> f64 {
        let total = self.total_ios.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        let polled = self.polled_ios.load(Ordering::Relaxed);
        polled as f64 / total as f64
    }

    /// Get average batch size
    pub fn avg_batch_size(&self) -> f64 {
        let batches = self.batches_submitted.load(Ordering::Relaxed);
        if batches == 0 {
            return 0.0;
        }
        let pages = self.batched_pages.load(Ordering::Relaxed);
        pages as f64 / batches as f64
    }

    /// Snapshot current stats
    pub fn snapshot(&self) -> HiPerfStatsSnapshot {
        HiPerfStatsSnapshot {
            total_ios: self.total_ios.load(Ordering::Relaxed),
            polled_ios: self.polled_ios.load(Ordering::Relaxed),
            interrupted_ios: self.interrupted_ios.load(Ordering::Relaxed),
            batches_submitted: self.batches_submitted.load(Ordering::Relaxed),
            batched_pages: self.batched_pages.load(Ordering::Relaxed),
            sequential_runs: self.sequential_runs.load(Ordering::Relaxed),
            affinity_applied: self.affinity_applied.load(Ordering::Relaxed),
            numa_node: self.numa_node.load(Ordering::Relaxed) as i32,
        }
    }
}

/// Immutable snapshot of HiPerfStats
#[derive(Debug, Clone)]
pub struct HiPerfStatsSnapshot {
    pub total_ios: u64,
    pub polled_ios: u64,
    pub interrupted_ios: u64,
    pub batches_submitted: u64,
    pub batched_pages: u64,
    pub sequential_runs: u64,
    pub affinity_applied: bool,
    pub numa_node: i32,
}

impl HiPerfStatsSnapshot {
    pub fn polling_efficiency(&self) -> f64 {
        if self.total_ios == 0 {
            return 0.0;
        }
        self.polled_ios as f64 / self.total_ios as f64
    }

    pub fn avg_batch_size(&self) -> f64 {
        if self.batches_submitted == 0 {
            return 0.0;
        }
        self.batched_pages as f64 / self.batches_submitted as f64
    }
}

impl HiPerfContext {
    /// Create a new high-performance context with given configuration
    ///
    /// Note: NUMA allocation is handled directly in daemon.rs during buffer allocation.
    /// Note: Batch coalescing is handled by BatchedPageStore for compression.
    pub fn new(config: PerfConfig) -> Self {
        let polling = if config.polling_enabled {
            Some(PollingLoop::new(config.polling.clone()))
        } else {
            None
        };

        let affinity = if !config.cpu_cores.is_empty() {
            Some(CpuAffinity::new(config.cpu_cores.clone()))
        } else {
            None
        };

        // Initialize 10X context if any optimizations are enabled
        let tenx = TenXContext::new(config.tenx.clone()).ok();

        Self { polling, affinity, config, stats: HiPerfStats::new(), tenx }
    }

    /// Create context from CLI arguments
    pub fn from_cli(
        polling: bool,
        poll_spin_cycles: u32,
        poll_adaptive: bool,
        batch_pages: usize,
        batch_timeout_us: u64,
        cpu_affinity: Option<String>,
        numa_node: i32,
    ) -> Self {
        let cpu_cores = cpu_affinity
            .map(|s| s.split(',').filter_map(|c| c.trim().parse().ok()).collect())
            .unwrap_or_default();

        let config = PerfConfig {
            polling_enabled: polling,
            polling: PollingConfig {
                enabled: polling,
                spin_cycles: poll_spin_cycles,
                adaptive: poll_adaptive,
                ..Default::default()
            },
            batch_size: batch_pages,
            batch_timeout_us,
            cpu_cores,
            numa_node,
            tenx: crate::perf::TenXConfig::default(),
        };

        Self::new(config)
    }

    /// Apply high-perf preset (moderate optimization)
    pub fn high_perf() -> Self {
        Self::new(PerfConfig::high_performance())
    }

    /// Apply max-perf preset (aggressive optimization)
    pub fn max_perf() -> Self {
        Self::new(PerfConfig::maximum())
    }

    /// Initialize the context (apply affinity, NUMA binding)
    pub fn init(&self) -> Result<(), HiPerfError> {
        // Apply CPU affinity if configured
        if let Some(ref affinity) = self.affinity {
            affinity.pin_current_thread()?;
            self.stats.affinity_applied.store(true, Ordering::Relaxed);
        }

        // Record NUMA node
        if self.config.numa_node >= 0 {
            self.stats.numa_node.store(self.config.numa_node as u64, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Check if polling mode is enabled
    pub fn is_polling_enabled(&self) -> bool {
        self.polling.is_some()
    }

    /// Check if batching is enabled (based on config, actual batching done by BatchedPageStore)
    pub fn is_batching_enabled(&self) -> bool {
        self.config.batch_size > 1
    }

    /// Poll for I/O completions (spin-wait)
    ///
    /// Takes `has_completions` (whether completions are ready) and `count`.
    /// Returns the poll result indicating what action to take.
    pub fn poll_once(&mut self, has_completions: bool, completion_count: u32) -> PollResult {
        if let Some(ref mut polling) = self.polling {
            let result = polling.poll_once(has_completions, completion_count);
            match result {
                PollResult::Ready(n) => {
                    self.stats.record_polled(n);
                }
                PollResult::SwitchToInterrupt => {
                    // Will use blocking wait - stats recorded when completions arrive
                }
                PollResult::Empty => {
                    // Continue polling
                }
            }
            result
        } else {
            // No polling, treat as ready if there are completions
            if has_completions {
                PollResult::Ready(completion_count)
            } else {
                PollResult::SwitchToInterrupt
            }
        }
    }

    /// Record completions from interrupt/blocking mode
    pub fn record_interrupt_completions(&self, count: u32) {
        self.stats.record_interrupted(count);
    }

    /// Allocate a buffer (NUMA binding handled in daemon.rs)
    pub fn alloc_buffer(&self, size: usize) -> Vec<u8> {
        vec![0u8; size]
    }

    /// Record a batch submission for statistics
    pub fn record_batch(&self, pages: usize, is_sequential: bool) {
        self.stats.record_batch(pages, is_sequential);
    }

    /// Get current statistics
    pub fn stats(&self) -> &HiPerfStats {
        &self.stats
    }

    /// Get polling statistics (if enabled)
    pub fn polling_stats(&self) -> Option<PollingStats> {
        self.polling.as_ref().map(|p| p.stats().clone())
    }

    /// Get configuration
    pub fn config(&self) -> &PerfConfig {
        &self.config
    }

    /// Get 10X optimization context (if available)
    pub fn tenx(&self) -> Option<&TenXContext> {
        self.tenx.as_ref()
    }

    /// Get current batch size (from 10X adaptive batching or default)
    pub fn adaptive_batch_size(&self) -> usize {
        self.tenx.as_ref().map(|t| t.current_batch_size()).unwrap_or(self.config.batch_size)
    }

    /// Record I/O latency for adaptive batching
    pub fn record_io_latency(&self, latency_us: u64) {
        if let Some(tenx) = &self.tenx {
            tenx.record_latency(latency_us);
        }
    }

    /// Resume polling after interrupt mode
    pub fn resume_polling(&mut self) {
        if let Some(ref mut polling) = self.polling {
            polling.resume_polling();
        }
    }

    /// Check if currently in interrupt mode
    pub fn in_interrupt_mode(&self) -> bool {
        self.polling.as_ref().map(|p| p.in_interrupt_mode()).unwrap_or(true)
    }
}

/// Errors from high-performance context
#[derive(Debug)]
pub enum HiPerfError {
    Affinity(super::affinity::AffinityError),
    Numa(super::numa::NumaError),
}

impl From<super::affinity::AffinityError> for HiPerfError {
    fn from(e: super::affinity::AffinityError) -> Self {
        HiPerfError::Affinity(e)
    }
}

impl From<super::numa::NumaError> for HiPerfError {
    fn from(e: super::numa::NumaError) -> Self {
        HiPerfError::Numa(e)
    }
}

impl std::fmt::Display for HiPerfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HiPerfError::Affinity(e) => write!(f, "Affinity error: {}", e),
            HiPerfError::Numa(e) => write!(f, "NUMA error: {}", e),
        }
    }
}

impl std::error::Error for HiPerfError {}

// ============================================================================
// Tests (TDD - written first)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // HiPerfContext Creation Tests
    // ========================================================================

    #[test]
    fn test_hiperf_context_new_default() {
        let config = PerfConfig::default();
        let ctx = HiPerfContext::new(config);

        // Default config has polling disabled but batching enabled (batch_size=64)
        assert!(!ctx.is_polling_enabled());
        assert!(ctx.is_batching_enabled());
        assert_eq!(ctx.config().batch_size, 64);
    }

    #[test]
    fn test_hiperf_context_with_polling() {
        let config = PerfConfig {
            polling_enabled: true,
            polling: PollingConfig::default(),
            ..Default::default()
        };
        let ctx = HiPerfContext::new(config);

        assert!(ctx.is_polling_enabled());
    }

    #[test]
    fn test_hiperf_context_with_batching() {
        let config = PerfConfig { batch_size: 64, ..Default::default() };
        let ctx = HiPerfContext::new(config);

        assert!(ctx.is_batching_enabled());
    }

    #[test]
    fn test_hiperf_context_high_perf_preset() {
        let ctx = HiPerfContext::high_perf();

        assert!(ctx.is_polling_enabled());
        assert!(ctx.is_batching_enabled());
        assert_eq!(ctx.config().batch_size, 128);
    }

    #[test]
    fn test_hiperf_context_max_perf_preset() {
        let ctx = HiPerfContext::max_perf();

        assert!(ctx.is_polling_enabled());
        assert!(ctx.is_batching_enabled());
        assert_eq!(ctx.config().batch_size, 256);
        assert_eq!(ctx.config().polling.spin_cycles, 100_000);
    }

    #[test]
    fn test_hiperf_context_from_cli_full() {
        let ctx = HiPerfContext::from_cli(
            true,                      // polling
            50000,                     // spin_cycles
            true,                      // adaptive
            128,                       // batch_pages
            75,                        // batch_timeout_us
            Some("0,1,2".to_string()), // cpu_affinity
            0,                         // numa_node
        );

        assert!(ctx.is_polling_enabled());
        assert!(ctx.is_batching_enabled());
        assert_eq!(ctx.config().cpu_cores, vec![0, 1, 2]);
        assert_eq!(ctx.config().numa_node, 0);
    }

    #[test]
    fn test_hiperf_context_from_cli_minimal() {
        let ctx = HiPerfContext::from_cli(false, 10000, false, 1, 100, None, -1);

        assert!(!ctx.is_polling_enabled());
        assert!(!ctx.is_batching_enabled());
    }

    // ========================================================================
    // Polling Integration Tests
    // ========================================================================

    #[test]
    fn test_hiperf_poll_completion_ready() {
        let config = PerfConfig {
            polling_enabled: true,
            polling: PollingConfig::default(),
            ..Default::default()
        };
        let mut ctx = HiPerfContext::new(config);

        // Simulate completions ready
        let result = ctx.poll_once(true, 5);

        assert!(result.is_ready());
        assert_eq!(result.count(), 5);
        assert_eq!(ctx.stats().polled_ios.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn test_hiperf_poll_completion_disabled() {
        let config = PerfConfig::default();
        let mut ctx = HiPerfContext::new(config);

        // With polling disabled, should go to interrupt mode when no completions
        let result = ctx.poll_once(false, 0);
        assert_eq!(result, PollResult::SwitchToInterrupt);

        // Should return ready when completions exist
        let result = ctx.poll_once(true, 3);
        assert!(result.is_ready());
        assert_eq!(result.count(), 3);
    }

    #[test]
    fn test_hiperf_poll_empty() {
        let config = PerfConfig {
            polling_enabled: true,
            polling: PollingConfig {
                enabled: true,
                spin_cycles: 10,
                idle_threshold: 1000,
                adaptive: false,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut ctx = HiPerfContext::new(config);

        // Simulate no completions - should keep polling (Empty)
        let result = ctx.poll_once(false, 0);
        assert_eq!(result, PollResult::Empty);
    }

    // ========================================================================
    // Batch Statistics Tests (batching now handled by BatchedPageStore)
    // ========================================================================

    #[test]
    fn test_hiperf_record_batch() {
        let ctx = HiPerfContext::high_perf();

        // Record batches via the new API
        ctx.record_batch(64, true);
        ctx.record_batch(32, false);

        let stats = ctx.stats().snapshot();
        assert_eq!(stats.batches_submitted, 2);
        assert_eq!(stats.batched_pages, 96);
        assert_eq!(stats.sequential_runs, 1);
    }

    #[test]
    fn test_hiperf_batching_enabled_check() {
        // batch_size > 1 means batching is enabled (done by BatchedPageStore)
        let config = PerfConfig { batch_size: 64, ..Default::default() };
        let ctx = HiPerfContext::new(config);
        assert!(ctx.is_batching_enabled());

        // batch_size <= 1 means batching disabled
        let config2 = PerfConfig { batch_size: 1, ..Default::default() };
        let ctx2 = HiPerfContext::new(config2);
        assert!(!ctx2.is_batching_enabled());
    }

    // ========================================================================
    // NUMA Integration Tests
    // ========================================================================

    #[test]
    fn test_hiperf_alloc_buffer_no_numa() {
        let config = PerfConfig::default();
        let ctx = HiPerfContext::new(config);

        let buf = ctx.alloc_buffer(4096);
        assert_eq!(buf.len(), 4096);
    }

    #[test]
    fn test_hiperf_alloc_buffer_with_numa() {
        let config = PerfConfig { numa_node: 0, ..Default::default() };
        let ctx = HiPerfContext::new(config);

        let buf = ctx.alloc_buffer(4096);
        assert_eq!(buf.len(), 4096);
    }

    // ========================================================================
    // Statistics Tests
    // ========================================================================

    #[test]
    fn test_hiperf_stats_new() {
        let stats = HiPerfStats::new();

        assert_eq!(stats.total_ios.load(Ordering::Relaxed), 0);
        assert_eq!(stats.polled_ios.load(Ordering::Relaxed), 0);
        assert_eq!(stats.interrupted_ios.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_hiperf_stats_record_polled() {
        let stats = HiPerfStats::new();

        stats.record_polled(5);
        stats.record_polled(3);

        assert_eq!(stats.total_ios.load(Ordering::Relaxed), 8);
        assert_eq!(stats.polled_ios.load(Ordering::Relaxed), 8);
        assert_eq!(stats.interrupted_ios.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_hiperf_stats_record_interrupted() {
        let stats = HiPerfStats::new();

        stats.record_interrupted(4);

        assert_eq!(stats.total_ios.load(Ordering::Relaxed), 4);
        assert_eq!(stats.polled_ios.load(Ordering::Relaxed), 0);
        assert_eq!(stats.interrupted_ios.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn test_hiperf_stats_record_batch() {
        let stats = HiPerfStats::new();

        stats.record_batch(64, true);
        stats.record_batch(32, false);

        assert_eq!(stats.batches_submitted.load(Ordering::Relaxed), 2);
        assert_eq!(stats.batched_pages.load(Ordering::Relaxed), 96);
        assert_eq!(stats.sequential_runs.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_hiperf_stats_polling_efficiency() {
        let stats = HiPerfStats::new();

        // No I/Os yet
        assert_eq!(stats.polling_efficiency(), 0.0);

        // 6 polled, 2 interrupted = 75% efficiency
        stats.record_polled(6);
        stats.record_interrupted(2);

        assert!((stats.polling_efficiency() - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_hiperf_stats_avg_batch_size() {
        let stats = HiPerfStats::new();

        // No batches yet
        assert_eq!(stats.avg_batch_size(), 0.0);

        // 2 batches, 150 total pages = 75 avg
        stats.record_batch(100, false);
        stats.record_batch(50, false);

        assert!((stats.avg_batch_size() - 75.0).abs() < 0.001);
    }

    #[test]
    fn test_hiperf_stats_snapshot() {
        let stats = HiPerfStats::new();

        stats.record_polled(1);
        stats.record_batch(64, true);
        stats.affinity_applied.store(true, Ordering::Relaxed);
        stats.numa_node.store(1, Ordering::Relaxed);

        let snapshot = stats.snapshot();

        assert_eq!(snapshot.total_ios, 1);
        assert_eq!(snapshot.polled_ios, 1);
        assert_eq!(snapshot.batches_submitted, 1);
        assert_eq!(snapshot.batched_pages, 64);
        assert_eq!(snapshot.sequential_runs, 1);
        assert!(snapshot.affinity_applied);
        assert_eq!(snapshot.numa_node, 1);
    }

    #[test]
    fn test_hiperf_stats_snapshot_derived_metrics() {
        let stats = HiPerfStats::new();

        stats.record_polled(4);
        stats.record_interrupted(2);
        stats.record_batch(100, false);
        stats.record_batch(50, false);

        let snapshot = stats.snapshot();

        assert!((snapshot.polling_efficiency() - 0.666).abs() < 0.01);
        assert!((snapshot.avg_batch_size() - 75.0).abs() < 0.001);
    }

    // ========================================================================
    // Error Handling Tests
    // ========================================================================

    #[test]
    fn test_hiperf_error_display_affinity() {
        let err = HiPerfError::Affinity(super::super::affinity::AffinityError::InvalidCore(999));
        let msg = format!("{}", err);
        assert!(msg.contains("Affinity"));
    }

    #[test]
    fn test_hiperf_error_display_numa() {
        let err = HiPerfError::Numa(super::super::numa::NumaError::InvalidNode(999));
        let msg = format!("{}", err);
        assert!(msg.contains("NUMA"));
    }

    // ========================================================================
    // Integration Scenario Tests
    // ========================================================================

    #[test]
    fn test_hiperf_full_polling_flow() {
        let mut ctx = HiPerfContext::high_perf();

        // Simulate high-throughput I/O with polling
        for _ in 0..256 {
            let result = ctx.poll_once(true, 1);
            assert!(result.is_ready());
        }

        let stats = ctx.stats().snapshot();
        assert_eq!(stats.polled_ios, 256);
        assert_eq!(stats.total_ios, 256);
        assert!((stats.polling_efficiency() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_hiperf_mixed_polling_and_stats() {
        let mut ctx = HiPerfContext::max_perf();

        // Simulate I/O completions via polling
        for _ in 0..10 {
            let result = ctx.poll_once(true, 1);
            assert!(result.is_ready());
        }

        // Record batch submissions (done externally by BatchedPageStore)
        ctx.record_batch(128, true);
        ctx.record_batch(128, false);

        let stats = ctx.stats().snapshot();
        assert_eq!(stats.polled_ios, 10);
        assert_eq!(stats.batched_pages, 256);
        assert_eq!(stats.batches_submitted, 2);
        assert_eq!(stats.sequential_runs, 1);
    }

    #[test]
    fn test_hiperf_config_passthrough() {
        let config = PerfConfig {
            batch_size: 128,
            batch_timeout_us: 50,
            numa_node: 0,
            cpu_cores: vec![0, 1],
            polling_enabled: true,
            polling: PollingConfig::aggressive(),
            tenx: crate::perf::TenXConfig::default(),
        };
        let ctx = HiPerfContext::new(config);

        // Verify config is accessible for external components
        assert_eq!(ctx.config().batch_size, 128);
        assert_eq!(ctx.config().batch_timeout_us, 50);
        assert_eq!(ctx.config().numa_node, 0);
        assert_eq!(ctx.config().cpu_cores, vec![0, 1]);
        assert!(ctx.config().polling_enabled);
    }

    #[test]
    fn test_hiperf_interrupt_mode_tracking() {
        let ctx = HiPerfContext::high_perf();

        // Record some completions from interrupt mode
        ctx.record_interrupt_completions(5);

        let stats = ctx.stats().snapshot();
        assert_eq!(stats.interrupted_ios, 5);
        assert_eq!(stats.total_ios, 5);
    }

    #[test]
    fn test_hiperf_resume_polling() {
        let mut ctx = HiPerfContext::high_perf();

        // Force into interrupt mode by many empty polls
        for _ in 0..10000 {
            ctx.poll_once(false, 0);
        }

        // Resume polling
        ctx.resume_polling();

        // Should be able to poll again
        let result = ctx.poll_once(true, 1);
        assert!(result.is_ready());
    }
}
