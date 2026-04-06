//! Types for the tracing escalation framework.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Reason for escalation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EscalationReason {
    /// CV (Coefficient of Variation) exceeded threshold
    CvExceeded,
    /// Efficiency below threshold
    EfficiencyLow,
    /// Both CV and efficiency triggered
    Both,
    /// Memory cliff detected (sudden performance drop)
    MemoryCliff,
    /// GPU transfer overhead exceeded threshold
    GpuTransferOverhead,
    /// Manual escalation requested
    Manual,
}

impl EscalationReason {
    /// Get a human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            EscalationReason::CvExceeded => "CV exceeded threshold (unstable performance)",
            EscalationReason::EfficiencyLow => "Efficiency below threshold",
            EscalationReason::Both => "Both CV exceeded and efficiency low",
            EscalationReason::MemoryCliff => "Memory cliff detected (sudden drop)",
            EscalationReason::GpuTransferOverhead => "GPU transfer overhead exceeded threshold",
            EscalationReason::Manual => "Manual escalation requested",
        }
    }

    /// Get the OTLP span attribute value
    pub fn otlp_value(&self) -> &'static str {
        match self {
            EscalationReason::CvExceeded => "cv_exceeded",
            EscalationReason::EfficiencyLow => "efficiency_low",
            EscalationReason::Both => "both",
            EscalationReason::MemoryCliff => "memory_cliff",
            EscalationReason::GpuTransferOverhead => "gpu_transfer_overhead",
            EscalationReason::Manual => "manual",
        }
    }
}

/// Escalation thresholds configuration
#[derive(Debug, Clone)]
pub struct EscalationThresholds {
    /// CV threshold (default: 15%)
    pub cv_threshold: f64,
    /// Efficiency threshold (default: 25%)
    pub efficiency_threshold: f64,
    /// GPU transfer overhead threshold (default: 50%)
    pub gpu_transfer_threshold: f64,
    /// Memory cliff threshold (percentage drop, default: 30%)
    pub memory_cliff_threshold: f64,
    /// Rate limit: max traces per interval
    pub rate_limit: u32,
    /// Rate limit interval
    pub rate_interval: Duration,
}

impl Default for EscalationThresholds {
    fn default() -> Self {
        Self {
            cv_threshold: 15.0,
            efficiency_threshold: 25.0,
            gpu_transfer_threshold: 50.0,
            memory_cliff_threshold: 30.0,
            rate_limit: 100,
            rate_interval: Duration::from_secs(60),
        }
    }
}

impl EscalationThresholds {
    /// Create new thresholds
    pub fn new() -> Self {
        Self::default()
    }

    /// Set CV threshold
    pub fn with_cv(mut self, threshold: f64) -> Self {
        self.cv_threshold = threshold;
        self
    }

    /// Set efficiency threshold
    pub fn with_efficiency(mut self, threshold: f64) -> Self {
        self.efficiency_threshold = threshold;
        self
    }

    /// Set GPU transfer threshold
    pub fn with_gpu_transfer(mut self, threshold: f64) -> Self {
        self.gpu_transfer_threshold = threshold;
        self
    }

    /// Set rate limit
    pub fn with_rate_limit(mut self, limit: u32) -> Self {
        self.rate_limit = limit;
        self
    }

    /// Set rate interval
    pub fn with_rate_interval(mut self, interval: Duration) -> Self {
        self.rate_interval = interval;
        self
    }
}

/// Syscall breakdown categories per section 35.2
#[derive(Debug, Clone, Default)]
pub struct SyscallBreakdown {
    /// mmap, munmap, mprotect, brk - Memory allocation overhead
    pub mmap_us: u64,
    /// futex - Thread contention
    pub futex_us: u64,
    /// ioctl - CUDA driver overhead
    pub ioctl_us: u64,
    /// read, pread64, readv - I/O read bottleneck
    pub read_us: u64,
    /// write, pwrite64, writev - I/O write bottleneck
    pub write_us: u64,
    /// Other syscalls not categorized
    pub other_us: u64,
    /// Total duration
    pub total_us: u64,
}

impl SyscallBreakdown {
    /// Create a new empty breakdown
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate compute time (total - all syscall overhead)
    pub fn compute_us(&self) -> u64 {
        let syscall_total = self.mmap_us
            + self.futex_us
            + self.ioctl_us
            + self.read_us
            + self.write_us
            + self.other_us;
        self.total_us.saturating_sub(syscall_total)
    }

    /// Calculate syscall overhead percentage
    pub fn syscall_overhead_percent(&self) -> f64 {
        if self.total_us == 0 {
            return 0.0;
        }
        let syscall_total = self.mmap_us
            + self.futex_us
            + self.ioctl_us
            + self.read_us
            + self.write_us
            + self.other_us;
        (syscall_total as f64 / self.total_us as f64) * 100.0
    }

    /// Get the dominant syscall category
    pub fn dominant_syscall(&self) -> &'static str {
        let categories = [
            (self.mmap_us, "mmap"),
            (self.futex_us, "futex"),
            (self.ioctl_us, "ioctl"),
            (self.read_us, "read"),
            (self.write_us, "write"),
            (self.other_us, "other"),
        ];

        // Return "none" if all categories are zero
        if categories.iter().all(|(time, _)| *time == 0) {
            return "none";
        }

        categories
            .iter()
            .max_by_key(|(time, _)| time)
            .map(|(_, name)| *name)
            .unwrap_or("none")
    }

    /// Add syscall time to appropriate category
    ///
    /// Uses saturating arithmetic to prevent integer overflow on extreme values.
    pub fn add_syscall(&mut self, syscall: &str, duration_us: u64) {
        match syscall {
            "mmap" | "munmap" | "mprotect" | "brk" => {
                self.mmap_us = self.mmap_us.saturating_add(duration_us)
            }
            "futex" => self.futex_us = self.futex_us.saturating_add(duration_us),
            "ioctl" => self.ioctl_us = self.ioctl_us.saturating_add(duration_us),
            "read" | "pread64" | "readv" => self.read_us = self.read_us.saturating_add(duration_us),
            "write" | "pwrite64" | "writev" => {
                self.write_us = self.write_us.saturating_add(duration_us)
            }
            _ => self.other_us = self.other_us.saturating_add(duration_us),
        }
    }

    /// Get breakdown as a map for OTLP attributes
    pub fn as_otlp_attributes(&self) -> HashMap<String, u64> {
        let mut attrs = HashMap::new();
        attrs.insert("syscall.mmap_us".to_string(), self.mmap_us);
        attrs.insert("syscall.futex_us".to_string(), self.futex_us);
        attrs.insert("syscall.ioctl_us".to_string(), self.ioctl_us);
        attrs.insert("syscall.read_us".to_string(), self.read_us);
        attrs.insert("syscall.write_us".to_string(), self.write_us);
        attrs.insert("syscall.other_us".to_string(), self.other_us);
        attrs.insert("syscall.compute_us".to_string(), self.compute_us());
        attrs.insert("syscall.total_us".to_string(), self.total_us);
        attrs
    }
}

/// Result of a trace operation
#[derive(Debug, Clone)]
pub struct TraceResult {
    /// Brick name that was traced
    pub brick_name: String,
    /// Budget in microseconds
    pub budget_us: u64,
    /// Actual duration in microseconds
    pub actual_us: u64,
    /// Reason for escalation
    pub reason: EscalationReason,
    /// Syscall breakdown
    pub syscall_breakdown: SyscallBreakdown,
    /// Timestamp of trace
    pub timestamp: Instant,
}

impl TraceResult {
    /// Check if over budget
    pub fn over_budget(&self) -> bool {
        self.actual_us > self.budget_us
    }

    /// Calculate efficiency (budget / actual * 100)
    pub fn efficiency(&self) -> f64 {
        if self.actual_us == 0 {
            return 100.0;
        }
        (self.budget_us as f64 / self.actual_us as f64) * 100.0
    }

    /// Get OTLP span attributes
    pub fn as_otlp_attributes(&self) -> HashMap<String, String> {
        let mut attrs = HashMap::new();
        attrs.insert("brick.name".to_string(), self.brick_name.clone());
        attrs.insert("brick.budget_us".to_string(), self.budget_us.to_string());
        attrs.insert("brick.actual_us".to_string(), self.actual_us.to_string());
        attrs.insert(
            "brick.efficiency".to_string(),
            format!("{:.1}", self.efficiency()),
        );
        attrs.insert(
            "brick.over_budget".to_string(),
            self.over_budget().to_string(),
        );
        attrs.insert(
            "escalation.reason".to_string(),
            self.reason.otlp_value().to_string(),
        );
        attrs.insert(
            "syscall.overhead_percent".to_string(),
            format!("{:.1}", self.syscall_breakdown.syscall_overhead_percent()),
        );
        attrs.insert(
            "syscall.dominant".to_string(),
            self.syscall_breakdown.dominant_syscall().to_string(),
        );
        attrs
    }
}
