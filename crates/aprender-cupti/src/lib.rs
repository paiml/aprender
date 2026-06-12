//! trueno-cupti: CUDA Profiling for ComputeBrick Analysis
//!
//! Rust bindings for NVIDIA CUPTI (CUDA Profiling Tools Interface).
//! Enables detailed GPU profiling for cbtop visualization.
//!
//! # Features
//!
//! - **Activity Tracing**: Kernel execution, memory copies, synchronization
//! - **Metrics Collection**: SM utilization, warp occupancy, memory throughput
//! - **Callback API**: Real-time notifications of CUDA operations
//! - **PC Sampling**: Instruction-level performance analysis
//!
//! # Example
//!
//! ```no_run,ignore
//! use trueno_cupti::{Profiler, ActivityKind};
//!
//! let profiler = Profiler::new()?;
//! profiler.enable(ActivityKind::Kernel)?;
//! profiler.enable(ActivityKind::MemoryCopy)?;
//!
//! // Run CUDA workload...
//!
//! let records = profiler.flush()?;
//! for record in records {
//!     println!("{}: {} ns", record.name, record.duration_ns);
//! }
//! ```
//!
//! # Hardware Requirements
//!
//! - NVIDIA GPU (Compute Capability 3.0+)
//! - CUDA Toolkit 11.0+ with CUPTI
//! - Linux (primary), Windows (experimental)

// `.unwrap()` is banned in production via workspace `.clippy.toml`, but is
// idiomatic in `#[cfg(test)]` assertions. Allow it only under `cfg(test)`.
#![cfg_attr(test, allow(clippy::disallowed_methods))]

pub mod activity;
pub mod callback;
pub mod error;
pub mod metrics;
mod sys;

pub use activity::{ActivityKind, ActivityRecord, KernelRecord, MemcpyRecord};
pub use callback::{CallbackDomain, CallbackId};
pub use error::{CuptiError, CuptiResult};
pub use metrics::{MetricId, MetricValue, SmMetrics, WarpMetrics};

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// CUPTI Profiler - main entry point for GPU profiling.
///
/// Thread-safe profiler that collects CUDA activity and metrics.
pub struct Profiler {
    /// Whether profiling is currently active
    active: Arc<AtomicBool>,
    /// Enabled activity kinds
    enabled_activities: Vec<ActivityKind>,
    /// Collected records (flushed on request)
    records: Vec<ActivityRecord>,
    /// Whether CUPTI is available on this system
    available: bool,
}

impl Profiler {
    /// Create a new profiler instance.
    ///
    /// Returns error if CUPTI is not available.
    pub fn new() -> CuptiResult<Self> {
        // Check if CUPTI is available
        let available = Self::check_cupti_available();

        Ok(Self {
            active: Arc::new(AtomicBool::new(false)),
            enabled_activities: Vec::new(),
            records: Vec::new(),
            available,
        })
    }

    /// Check if CUPTI is available on this system.
    pub fn is_available(&self) -> bool {
        self.available
    }

    /// Enable profiling for a specific activity kind.
    pub fn enable(&mut self, kind: ActivityKind) -> CuptiResult<()> {
        if !self.available {
            return Err(CuptiError::NotAvailable);
        }
        if !self.enabled_activities.contains(&kind) {
            self.enabled_activities.push(kind);
        }
        Ok(())
    }

    /// Disable profiling for a specific activity kind.
    pub fn disable(&mut self, kind: ActivityKind) -> CuptiResult<()> {
        self.enabled_activities.retain(|k| *k != kind);
        Ok(())
    }

    /// Start profiling.
    pub fn start(&self) -> CuptiResult<()> {
        if !self.available {
            return Err(CuptiError::NotAvailable);
        }
        self.active.store(true, Ordering::SeqCst);
        // In real implementation: cuptiActivityEnable for each kind
        Ok(())
    }

    /// Stop profiling.
    pub fn stop(&self) -> CuptiResult<()> {
        self.active.store(false, Ordering::SeqCst);
        // In real implementation: cuptiActivityDisable for each kind
        Ok(())
    }

    /// Check if profiling is active.
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::SeqCst)
    }

    /// Flush collected records and return them.
    pub fn flush(&mut self) -> CuptiResult<Vec<ActivityRecord>> {
        // In real implementation: cuptiActivityFlushAll
        Ok(std::mem::take(&mut self.records))
    }

    /// Get enabled activity kinds.
    pub fn enabled_activities(&self) -> &[ActivityKind] {
        &self.enabled_activities
    }

    /// Check if CUPTI library is available.
    fn check_cupti_available() -> bool {
        // Try to find libcupti.so
        #[cfg(target_os = "linux")]
        {
            // Check common CUPTI locations
            let paths = [
                "/usr/local/cuda/lib64/libcupti.so",
                "/usr/lib/x86_64-linux-gnu/libcupti.so",
                "/opt/cuda/lib64/libcupti.so",
            ];
            for path in &paths {
                if std::path::Path::new(path).exists() {
                    return true;
                }
            }
            // Also check LD_LIBRARY_PATH
            if let Ok(ld_path) = std::env::var("LD_LIBRARY_PATH") {
                for dir in ld_path.split(':') {
                    let cupti_path = std::path::Path::new(dir).join("libcupti.so");
                    if cupti_path.exists() {
                        return true;
                    }
                }
            }
        }
        false
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new().unwrap_or(Self {
            active: Arc::new(AtomicBool::new(false)),
            enabled_activities: Vec::new(),
            records: Vec::new(),
            available: false,
        })
    }
}

/// Builder for configuring profiler options.
#[derive(Debug, Default)]
pub struct ProfilerBuilder {
    activities: Vec<ActivityKind>,
    buffer_size: Option<usize>,
    flush_period_ms: Option<u64>,
}

impl ProfilerBuilder {
    /// Create a new profiler builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable an activity kind.
    #[must_use]
    pub fn activity(mut self, kind: ActivityKind) -> Self {
        self.activities.push(kind);
        self
    }

    /// Enable kernel profiling.
    #[must_use]
    pub fn kernels(self) -> Self {
        self.activity(ActivityKind::Kernel)
    }

    /// Enable memory copy profiling.
    #[must_use]
    pub fn memcpy(self) -> Self {
        self.activity(ActivityKind::MemoryCopy)
    }

    /// Enable synchronization profiling.
    #[must_use]
    pub fn sync(self) -> Self {
        self.activity(ActivityKind::Synchronization)
    }

    /// Set activity buffer size.
    #[must_use]
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = Some(size);
        self
    }

    /// Set automatic flush period in milliseconds.
    #[must_use]
    pub fn flush_period_ms(mut self, ms: u64) -> Self {
        self.flush_period_ms = Some(ms);
        self
    }

    /// Build the profiler.
    pub fn build(self) -> CuptiResult<Profiler> {
        let mut profiler = Profiler::new()?;
        for activity in self.activities {
            profiler.enable(activity)?;
        }
        Ok(profiler)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_creation() {
        let profiler = Profiler::new();
        assert!(profiler.is_ok());
    }

    #[test]
    fn test_profiler_default() {
        let profiler = Profiler::default();
        assert!(!profiler.is_active());
    }

    #[test]
    fn test_profiler_builder() {
        let builder = ProfilerBuilder::new()
            .kernels()
            .memcpy()
            .buffer_size(1024 * 1024);

        let profiler = builder.build();
        // May fail if CUPTI not available, that's OK
        if let Ok(p) = profiler {
            assert!(p.enabled_activities().contains(&ActivityKind::Kernel));
            assert!(p.enabled_activities().contains(&ActivityKind::MemoryCopy));
        }
    }

    #[test]
    fn test_activity_enable_disable() {
        let mut profiler = Profiler::default();
        if profiler.is_available() {
            assert!(profiler.enable(ActivityKind::Kernel).is_ok());
            assert!(profiler
                .enabled_activities()
                .contains(&ActivityKind::Kernel));
            assert!(profiler.disable(ActivityKind::Kernel).is_ok());
            assert!(!profiler
                .enabled_activities()
                .contains(&ActivityKind::Kernel));
        }
    }

    #[test]
    fn test_profiler_start_stop() {
        let profiler = Profiler::default();
        if profiler.is_available() {
            assert!(profiler.start().is_ok());
            assert!(profiler.is_active());
            assert!(profiler.stop().is_ok());
            assert!(!profiler.is_active());
        }
    }
}
