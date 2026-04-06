//! PERF-001: Performance Optimization Module
//!
//! High-performance ublk optimizations targeting 800K+ IOPS:
//! - Polling mode: spin-wait on io_uring completions
//! - Batch coalescing: group 64-256 pages per submission
//! - CPU affinity: pin worker threads to dedicated cores
//! - NUMA awareness: allocate memory on local NUMA node
//!
//! Target: 800K+ IOPS (vs current 225K)
//! Reference: Ming Lei LPC 2022 - ublk can achieve 1.2M IOPS
//!
//! ## Status
//! - polling: INTEGRATED - PollingConfig, PollResult used in daemon
//! - hiperf_daemon: INTEGRATED - HiPerfContext used in daemon
//! - affinity: INTEGRATED - CPU pinning in multi_queue.rs
//! - batch: INTEGRATED - BatchedPageStore handles compression batching
//! - numa: INTEGRATED - NUMA binding in daemon.rs buffer allocation
//!
//! ## 10X Performance Stack (trueno-ublk-spec v3.0.0)
//!
//! The `tenx` submodule implements the full 10X optimization stack:
//! - PERF-005: io_uring Registered Buffers (Target: 1.75x)
//! - PERF-006: True Zero-Copy (Target: 2.5x)
//! - PERF-007: SQPOLL Mode (Target: 4x)
//! - PERF-008: Fixed File Descriptors (Target: 4.5x)
//! - PERF-009: Huge Pages 2MB (Target: 6x)
//! - PERF-010: NUMA-Aware Allocation (Target: 7x)
//! - PERF-011: Lock-Free Multi-Queue (Target: 9x)
//! - PERF-012: Adaptive Batch Sizing (Target: 10x)
//!
//! Note: Many items in submodules are used only by tests (TDD infrastructure).
//! This is intentional per "extreme TDD" development methodology.

// TDD infrastructure - many items only used in tests
#![allow(dead_code)]

pub mod affinity;
pub mod batch;
pub mod hiperf_daemon;
pub mod numa;
pub mod polling;
pub mod tenx;

// Active exports - used in daemon
pub use hiperf_daemon::HiPerfContext;
pub use polling::{PollResult, PollingConfig};

// 10X optimization stack - integrated into PerfConfig
pub use tenx::{TenXConfig, TenXContext};

/// Performance configuration for ublk daemon
#[derive(Debug, Clone)]
pub struct PerfConfig {
    /// Enable polling mode (spin-wait instead of interrupt-driven)
    pub polling_enabled: bool,

    /// Polling configuration
    pub polling: PollingConfig,

    /// Batch size for page coalescing
    pub batch_size: usize,

    /// Batch timeout in microseconds
    pub batch_timeout_us: u64,

    /// CPU cores to pin workers to (empty = no pinning)
    pub cpu_cores: Vec<usize>,

    /// NUMA node for memory allocation (-1 = auto)
    pub numa_node: i32,

    /// 10X optimization stack (PERF-005 through PERF-012)
    pub tenx: TenXConfig,
}

impl Default for PerfConfig {
    fn default() -> Self {
        Self {
            polling_enabled: false,
            polling: PollingConfig::default(),
            batch_size: 64,
            batch_timeout_us: 100,
            cpu_cores: Vec::new(),
            numa_node: -1,
            tenx: TenXConfig::conservative(),
        }
    }
}

impl PerfConfig {
    /// High-performance configuration for maximum IOPS
    pub fn high_performance() -> Self {
        Self {
            polling_enabled: true,
            polling: PollingConfig::aggressive(),
            batch_size: 128,
            batch_timeout_us: 50,
            cpu_cores: Vec::new(), // Will be auto-detected
            numa_node: -1,         // Auto-detect
            tenx: TenXConfig::default(),
        }
    }

    /// Maximum performance - use with caution (high CPU usage)
    pub fn maximum() -> Self {
        Self {
            polling_enabled: true,
            polling: PollingConfig::maximum(),
            batch_size: 256,
            batch_timeout_us: 25,
            cpu_cores: Vec::new(),
            numa_node: -1,
            tenx: TenXConfig::aggressive(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perf_config_default() {
        let config = PerfConfig::default();
        assert!(!config.polling_enabled);
        assert_eq!(config.batch_size, 64);
        assert_eq!(config.batch_timeout_us, 100);
        assert!(config.cpu_cores.is_empty());
        assert_eq!(config.numa_node, -1);
    }

    #[test]
    fn test_perf_config_high_performance() {
        let config = PerfConfig::high_performance();
        assert!(config.polling_enabled);
        assert_eq!(config.batch_size, 128);
        assert_eq!(config.batch_timeout_us, 50);
    }

    #[test]
    fn test_perf_config_maximum() {
        let config = PerfConfig::maximum();
        assert!(config.polling_enabled);
        assert_eq!(config.batch_size, 256);
        assert_eq!(config.batch_timeout_us, 25);
    }

    #[test]
    fn test_perf_config_clone() {
        let config = PerfConfig::high_performance();
        let cloned = config.clone();
        assert_eq!(config.polling_enabled, cloned.polling_enabled);
        assert_eq!(config.batch_size, cloned.batch_size);
    }

    #[test]
    fn test_tenx_context_from_perf_config() {
        let config = PerfConfig::maximum();
        let ctx = TenXContext::new(config.tenx).expect("TenXContext creation should succeed");
        // Verify context was created with expected settings
        assert!(ctx.config().registered_buffers.enabled);
        assert!(ctx.config().adaptive_batch_enabled);
        // Check batch size is working
        let batch_size = ctx.current_batch_size();
        assert!(batch_size > 0);
    }

    #[test]
    fn test_tenx_context_conservative() {
        let config = PerfConfig::default();
        let ctx = TenXContext::new(config.tenx).expect("TenXContext creation should succeed");
        // Conservative config disables most features
        assert!(!ctx.config().zero_copy.enabled);
        assert!(!ctx.config().sqpoll.enabled);
        assert!(ctx.sqpoll().is_none());
    }
}
