//! Polling mode for io_uring completions
//!
//! Eliminates interrupt overhead by spin-waiting on completion queue.
//! Target: Reduce latency from ~10µs (interrupt) to ~1µs (poll)
//!
//! Status: PollingConfig and PollResult are integrated in daemon.
//!
//! ## io_uring Polling Modes
//!
//! 1. **IORING_SETUP_SQPOLL**: Kernel thread polls submission queue
//!    - Kernel 5.11+ required
//!    - Reduces syscall overhead
//!    - Requires CAP_SYS_NICE for dedicated thread
//!
//! 2. **Userspace polling**: We spin-wait on completion queue
//!    - Works on any kernel with io_uring
//!    - Higher CPU usage but lower latency
//!    - Adaptive backoff prevents 100% CPU burn
//!
//! ## Usage
//!
//! ```ignore
//! let config = PollingConfig::default();
//! let mut poller = PollingLoop::new(config);
//!
//! loop {
//!     let completions = poller.poll(&mut ring);
//!     for cqe in completions {
//!         // Process completion
//!     }
//! }
//! ```

use std::time::{Duration, Instant};

/// Polling configuration
#[derive(Debug, Clone)]
pub struct PollingConfig {
    /// Enable polling mode
    pub enabled: bool,

    /// Number of spin cycles before yielding
    pub spin_cycles: u32,

    /// Nanoseconds to yield after spin_cycles
    pub yield_ns: u64,

    /// Maximum time to wait for completion (microseconds)
    pub max_wait_us: u64,

    /// Enable adaptive polling (reduce CPU when idle)
    pub adaptive: bool,

    /// Idle threshold - switch to interrupt mode after N empty polls
    pub idle_threshold: u32,
}

impl Default for PollingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            spin_cycles: 10_000,
            yield_ns: 100,
            max_wait_us: 1000,
            adaptive: true,
            idle_threshold: 1000,
        }
    }
}

impl PollingConfig {
    /// Aggressive polling for maximum performance
    pub fn aggressive() -> Self {
        Self {
            enabled: true,
            spin_cycles: 50_000,
            yield_ns: 50,
            max_wait_us: 500,
            adaptive: true,
            idle_threshold: 5000,
        }
    }

    /// Maximum polling - burns CPU but lowest latency
    pub fn maximum() -> Self {
        Self {
            enabled: true,
            spin_cycles: 100_000,
            yield_ns: 10,
            max_wait_us: 100,
            adaptive: false,
            idle_threshold: u32::MAX,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.spin_cycles == 0 && self.enabled {
            return Err("spin_cycles must be > 0 when polling enabled");
        }
        if self.max_wait_us == 0 {
            return Err("max_wait_us must be > 0");
        }
        Ok(())
    }
}

/// Statistics for polling loop
#[derive(Debug, Clone, Default)]
pub struct PollingStats {
    /// Total poll iterations
    pub poll_count: u64,

    /// Iterations that found completions
    pub successful_polls: u64,

    /// Iterations that found nothing (empty)
    pub empty_polls: u64,

    /// Total completions processed
    pub completions: u64,

    /// Times we yielded CPU
    pub yields: u64,

    /// Times we switched to interrupt mode (adaptive)
    pub interrupt_fallbacks: u64,

    /// Minimum completions per poll (when non-empty)
    pub min_batch: u32,

    /// Maximum completions per poll
    pub max_batch: u32,

    /// Total spin cycles
    pub total_spins: u64,
}

impl PollingStats {
    /// Calculate hit rate (successful / total polls)
    pub fn hit_rate(&self) -> f64 {
        if self.poll_count == 0 {
            return 0.0;
        }
        self.successful_polls as f64 / self.poll_count as f64
    }

    /// Average batch size when successful
    pub fn avg_batch_size(&self) -> f64 {
        if self.successful_polls == 0 {
            return 0.0;
        }
        self.completions as f64 / self.successful_polls as f64
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Polling loop for io_uring completions
pub struct PollingLoop {
    config: PollingConfig,
    stats: PollingStats,
    consecutive_empty: u32,
    last_completion: Instant,
    in_interrupt_mode: bool,
}

impl PollingLoop {
    /// Create new polling loop
    pub fn new(config: PollingConfig) -> Self {
        Self {
            config,
            stats: PollingStats::default(),
            consecutive_empty: 0,
            last_completion: Instant::now(),
            in_interrupt_mode: false,
        }
    }

    /// Get current configuration
    pub fn config(&self) -> &PollingConfig {
        &self.config
    }

    /// Get statistics
    pub fn stats(&self) -> &PollingStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats.reset();
    }

    /// Check if currently in interrupt fallback mode
    pub fn in_interrupt_mode(&self) -> bool {
        self.in_interrupt_mode
    }

    /// Spin-wait for completions with adaptive backoff
    ///
    /// Returns number of completions ready (caller should drain CQ)
    #[inline]
    pub fn poll_once(&mut self, has_completions: bool, completion_count: u32) -> PollResult {
        self.stats.poll_count += 1;

        if has_completions {
            self.stats.successful_polls += 1;
            self.stats.completions += completion_count as u64;
            self.consecutive_empty = 0;
            self.last_completion = Instant::now();
            self.in_interrupt_mode = false;

            // Update batch stats
            if self.stats.min_batch == 0 || completion_count < self.stats.min_batch {
                self.stats.min_batch = completion_count;
            }
            if completion_count > self.stats.max_batch {
                self.stats.max_batch = completion_count;
            }

            return PollResult::Ready(completion_count);
        }

        // Empty poll
        self.stats.empty_polls += 1;
        self.consecutive_empty += 1;

        // Adaptive: switch to interrupt mode if idle too long
        if self.config.adaptive && self.consecutive_empty >= self.config.idle_threshold {
            self.stats.interrupt_fallbacks += 1;
            self.in_interrupt_mode = true;
            return PollResult::SwitchToInterrupt;
        }

        // Spin for a while
        let mut spins = 0u32;
        while spins < self.config.spin_cycles {
            std::hint::spin_loop();
            spins += 1;
        }
        self.stats.total_spins += spins as u64;

        // Yield briefly
        if self.config.yield_ns > 0 {
            self.stats.yields += 1;
            std::thread::sleep(Duration::from_nanos(self.config.yield_ns));
        }

        PollResult::Empty
    }

    /// Force switch back to polling mode (from interrupt mode)
    pub fn resume_polling(&mut self) {
        self.in_interrupt_mode = false;
        self.consecutive_empty = 0;
    }
}

/// Result of a poll operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PollResult {
    /// Completions are ready
    Ready(u32),
    /// No completions, continue polling
    Empty,
    /// Switch to interrupt mode (adaptive)
    SwitchToInterrupt,
}

impl PollResult {
    /// Check if completions are ready
    pub fn is_ready(&self) -> bool {
        matches!(self, PollResult::Ready(_))
    }

    /// Get completion count (0 if not ready)
    pub fn count(&self) -> u32 {
        match self {
            PollResult::Ready(n) => *n,
            _ => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================================
    // PollingConfig Tests
    // ============================================================================

    #[test]
    fn test_polling_config_default() {
        let config = PollingConfig::default();
        assert!(config.enabled);
        assert_eq!(config.spin_cycles, 10_000);
        assert_eq!(config.yield_ns, 100);
        assert_eq!(config.max_wait_us, 1000);
        assert!(config.adaptive);
        assert_eq!(config.idle_threshold, 1000);
    }

    #[test]
    fn test_polling_config_aggressive() {
        let config = PollingConfig::aggressive();
        assert!(config.enabled);
        assert_eq!(config.spin_cycles, 50_000);
        assert_eq!(config.yield_ns, 50);
        assert!(config.adaptive);
    }

    #[test]
    fn test_polling_config_maximum() {
        let config = PollingConfig::maximum();
        assert!(config.enabled);
        assert_eq!(config.spin_cycles, 100_000);
        assert_eq!(config.yield_ns, 10);
        assert!(!config.adaptive); // No adaptive in maximum mode
    }

    #[test]
    fn test_polling_config_validate_success() {
        let config = PollingConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_polling_config_validate_zero_spin_cycles() {
        let mut config = PollingConfig::default();
        config.spin_cycles = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_polling_config_validate_zero_max_wait() {
        let mut config = PollingConfig::default();
        config.max_wait_us = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_polling_config_validate_disabled_zero_spin_ok() {
        let mut config = PollingConfig::default();
        config.enabled = false;
        config.spin_cycles = 0;
        // Zero spin cycles OK when disabled
        assert!(config.validate().is_ok());
    }

    // ============================================================================
    // PollingStats Tests
    // ============================================================================

    #[test]
    fn test_polling_stats_default() {
        let stats = PollingStats::default();
        assert_eq!(stats.poll_count, 0);
        assert_eq!(stats.successful_polls, 0);
        assert_eq!(stats.empty_polls, 0);
        assert_eq!(stats.completions, 0);
    }

    #[test]
    fn test_polling_stats_hit_rate_zero() {
        let stats = PollingStats::default();
        assert_eq!(stats.hit_rate(), 0.0);
    }

    #[test]
    fn test_polling_stats_hit_rate() {
        let stats = PollingStats { poll_count: 100, successful_polls: 75, ..Default::default() };
        assert!((stats.hit_rate() - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_polling_stats_avg_batch_size_zero() {
        let stats = PollingStats::default();
        assert_eq!(stats.avg_batch_size(), 0.0);
    }

    #[test]
    fn test_polling_stats_avg_batch_size() {
        let stats = PollingStats { successful_polls: 10, completions: 50, ..Default::default() };
        assert!((stats.avg_batch_size() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_polling_stats_reset() {
        let mut stats = PollingStats {
            poll_count: 100,
            successful_polls: 75,
            completions: 500,
            ..Default::default()
        };
        stats.reset();
        assert_eq!(stats.poll_count, 0);
        assert_eq!(stats.successful_polls, 0);
        assert_eq!(stats.completions, 0);
    }

    // ============================================================================
    // PollResult Tests
    // ============================================================================

    #[test]
    fn test_poll_result_ready() {
        let result = PollResult::Ready(10);
        assert!(result.is_ready());
        assert_eq!(result.count(), 10);
    }

    #[test]
    fn test_poll_result_empty() {
        let result = PollResult::Empty;
        assert!(!result.is_ready());
        assert_eq!(result.count(), 0);
    }

    #[test]
    fn test_poll_result_switch_to_interrupt() {
        let result = PollResult::SwitchToInterrupt;
        assert!(!result.is_ready());
        assert_eq!(result.count(), 0);
    }

    // ============================================================================
    // PollingLoop Tests
    // ============================================================================

    #[test]
    fn test_polling_loop_new() {
        let config = PollingConfig::default();
        let poller = PollingLoop::new(config.clone());
        assert_eq!(poller.config().spin_cycles, config.spin_cycles);
        assert!(!poller.in_interrupt_mode());
    }

    #[test]
    fn test_polling_loop_successful_poll() {
        let config = PollingConfig::default();
        let mut poller = PollingLoop::new(config);

        let result = poller.poll_once(true, 5);
        assert_eq!(result, PollResult::Ready(5));
        assert_eq!(poller.stats().poll_count, 1);
        assert_eq!(poller.stats().successful_polls, 1);
        assert_eq!(poller.stats().completions, 5);
        assert_eq!(poller.stats().empty_polls, 0);
    }

    #[test]
    fn test_polling_loop_empty_poll() {
        let mut config = PollingConfig::default();
        config.spin_cycles = 10; // Low for fast test
        config.yield_ns = 0; // No sleep for fast test

        let mut poller = PollingLoop::new(config);

        let result = poller.poll_once(false, 0);
        assert_eq!(result, PollResult::Empty);
        assert_eq!(poller.stats().poll_count, 1);
        assert_eq!(poller.stats().successful_polls, 0);
        assert_eq!(poller.stats().empty_polls, 1);
    }

    #[test]
    fn test_polling_loop_adaptive_switch_to_interrupt() {
        let mut config = PollingConfig::default();
        config.adaptive = true;
        config.idle_threshold = 3;
        config.spin_cycles = 1;
        config.yield_ns = 0;

        let mut poller = PollingLoop::new(config);

        // Simulate idle polls
        for _ in 0..2 {
            let result = poller.poll_once(false, 0);
            assert_eq!(result, PollResult::Empty);
            assert!(!poller.in_interrupt_mode());
        }

        // Third empty poll should trigger switch
        let result = poller.poll_once(false, 0);
        assert_eq!(result, PollResult::SwitchToInterrupt);
        assert!(poller.in_interrupt_mode());
        assert_eq!(poller.stats().interrupt_fallbacks, 1);
    }

    #[test]
    fn test_polling_loop_resume_polling() {
        let mut config = PollingConfig::default();
        config.adaptive = true;
        config.idle_threshold = 1;
        config.spin_cycles = 1;
        config.yield_ns = 0;

        let mut poller = PollingLoop::new(config);

        // Trigger switch to interrupt mode
        poller.poll_once(false, 0);
        assert!(poller.in_interrupt_mode());

        // Resume polling
        poller.resume_polling();
        assert!(!poller.in_interrupt_mode());
    }

    #[test]
    fn test_polling_loop_batch_stats() {
        let config = PollingConfig::default();
        let mut poller = PollingLoop::new(config);

        // First completion batch
        poller.poll_once(true, 5);
        assert_eq!(poller.stats().min_batch, 5);
        assert_eq!(poller.stats().max_batch, 5);

        // Larger batch
        poller.poll_once(true, 10);
        assert_eq!(poller.stats().min_batch, 5);
        assert_eq!(poller.stats().max_batch, 10);

        // Smaller batch
        poller.poll_once(true, 2);
        assert_eq!(poller.stats().min_batch, 2);
        assert_eq!(poller.stats().max_batch, 10);
    }

    #[test]
    fn test_polling_loop_reset_stats() {
        let config = PollingConfig::default();
        let mut poller = PollingLoop::new(config);

        poller.poll_once(true, 5);
        assert!(poller.stats().poll_count > 0);

        poller.reset_stats();
        assert_eq!(poller.stats().poll_count, 0);
        assert_eq!(poller.stats().completions, 0);
    }

    #[test]
    fn test_polling_loop_completion_resets_empty_count() {
        let mut config = PollingConfig::default();
        config.adaptive = true;
        config.idle_threshold = 10;
        config.spin_cycles = 1;
        config.yield_ns = 0;

        let mut poller = PollingLoop::new(config);

        // Build up empty polls
        for _ in 0..5 {
            poller.poll_once(false, 0);
        }
        assert_eq!(poller.stats().empty_polls, 5);

        // Successful poll should reset consecutive empty counter
        poller.poll_once(true, 1);
        assert!(!poller.in_interrupt_mode());

        // More empty polls - should start from 0
        for _ in 0..5 {
            poller.poll_once(false, 0);
        }
        // Still shouldn't switch (need 10, only did 5 after reset)
        assert!(!poller.in_interrupt_mode());
    }

    #[test]
    fn test_polling_loop_yields_tracked() {
        let mut config = PollingConfig::default();
        config.spin_cycles = 1;
        config.yield_ns = 1; // Very short yield

        let mut poller = PollingLoop::new(config);

        poller.poll_once(false, 0);
        assert!(poller.stats().yields > 0);
    }

    #[test]
    fn test_polling_loop_spins_tracked() {
        let mut config = PollingConfig::default();
        config.spin_cycles = 100;
        config.yield_ns = 0;
        config.idle_threshold = u32::MAX;

        let mut poller = PollingLoop::new(config);

        poller.poll_once(false, 0);
        assert_eq!(poller.stats().total_spins, 100);

        poller.poll_once(false, 0);
        assert_eq!(poller.stats().total_spins, 200);
    }

    #[test]
    fn test_polling_loop_maximum_config_no_adaptive() {
        let config = PollingConfig::maximum();
        let mut poller = PollingLoop::new(config);

        // Even many empty polls shouldn't trigger interrupt mode
        for _ in 0..100 {
            let result = poller.poll_once(false, 0);
            assert_ne!(result, PollResult::SwitchToInterrupt);
        }
        assert!(!poller.in_interrupt_mode());
    }

    // ============================================================================
    // Integration-style Tests
    // ============================================================================

    #[test]
    fn test_polling_simulation_mixed_workload() {
        let mut config = PollingConfig::default();
        config.spin_cycles = 1;
        config.yield_ns = 0;
        config.idle_threshold = 100;

        let mut poller = PollingLoop::new(config);

        // Simulate mixed workload: some completions, some empty
        // Pattern: 6 true (completions), 4 false (empty)
        let pattern = [true, true, false, true, false, false, true, true, true, false];
        let counts = [5, 3, 0, 7, 0, 0, 2, 4, 1, 0];

        for (has_completion, count) in pattern.iter().zip(counts.iter()) {
            poller.poll_once(*has_completion, *count);
        }

        assert_eq!(poller.stats().poll_count, 10);
        assert_eq!(poller.stats().successful_polls, 6); // 6 true values in pattern
        assert_eq!(poller.stats().empty_polls, 4); // 4 false values in pattern
        assert_eq!(poller.stats().completions, 5 + 3 + 7 + 2 + 4 + 1); // 22 total
        assert_eq!(poller.stats().min_batch, 1);
        assert_eq!(poller.stats().max_batch, 7);

        let hit_rate = poller.stats().hit_rate();
        assert!((hit_rate - 0.6).abs() < 0.001); // 6/10 = 0.6
    }

    #[test]
    fn test_polling_config_clone() {
        let config = PollingConfig::aggressive();
        let cloned = config.clone();
        assert_eq!(config.spin_cycles, cloned.spin_cycles);
        assert_eq!(config.adaptive, cloned.adaptive);
    }

    #[test]
    fn test_polling_stats_clone() {
        let stats = PollingStats { poll_count: 100, completions: 500, ..Default::default() };
        let cloned = stats.clone();
        assert_eq!(stats.poll_count, cloned.poll_count);
        assert_eq!(stats.completions, cloned.completions);
    }
}
