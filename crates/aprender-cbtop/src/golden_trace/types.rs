//! Core types for golden traces: errors, syscall breakdowns, and metrics.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Golden trace error
#[derive(Debug, Clone, PartialEq)]
pub enum GoldenTraceError {
    /// No golden trace exists
    NoBaseline,
    /// IO error
    IoError(String),
    /// Parse error
    ParseError(String),
    /// Invalid trace data
    InvalidTrace(String),
    /// Version mismatch
    VersionMismatch { expected: String, actual: String },
}

impl std::fmt::Display for GoldenTraceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoBaseline => write!(f, "No golden baseline exists"),
            Self::IoError(msg) => write!(f, "IO error: {}", msg),
            Self::ParseError(msg) => write!(f, "Parse error: {}", msg),
            Self::InvalidTrace(msg) => write!(f, "Invalid trace: {}", msg),
            Self::VersionMismatch { expected, actual } => {
                write!(f, "Version mismatch: expected {}, got {}", expected, actual)
            }
        }
    }
}

impl std::error::Error for GoldenTraceError {}

/// Result type for golden trace operations
pub type GoldenTraceResult<T> = Result<T, GoldenTraceError>;

/// Syscall breakdown for trace comparison
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SyscallBreakdown {
    /// Read syscall count
    pub read_count: u64,
    /// Write syscall count
    pub write_count: u64,
    /// Mmap syscall count
    pub mmap_count: u64,
    /// Futex syscall count
    pub futex_count: u64,
    /// Other syscall count
    pub other_count: u64,
}

impl SyscallBreakdown {
    /// Create new breakdown
    pub fn new() -> Self {
        Self::default()
    }

    /// Total syscall count
    pub fn total(&self) -> u64 {
        self.read_count + self.write_count + self.mmap_count + self.futex_count + self.other_count
    }

    /// Calculate percentage difference from baseline
    pub fn percentage_diff(&self, baseline: &SyscallBreakdown) -> SyscallBreakdownDelta {
        SyscallBreakdownDelta {
            read_delta: Self::calc_delta(self.read_count, baseline.read_count),
            write_delta: Self::calc_delta(self.write_count, baseline.write_count),
            mmap_delta: Self::calc_delta(self.mmap_count, baseline.mmap_count),
            futex_delta: Self::calc_delta(self.futex_count, baseline.futex_count),
            other_delta: Self::calc_delta(self.other_count, baseline.other_count),
            total_delta: Self::calc_delta(self.total(), baseline.total()),
        }
    }

    fn calc_delta(current: u64, baseline: u64) -> f64 {
        if baseline == 0 {
            if current == 0 {
                0.0
            } else {
                100.0 // New syscalls appeared
            }
        } else {
            ((current as f64 - baseline as f64) / baseline as f64) * 100.0
        }
    }
}

/// Syscall breakdown delta (percentage changes)
#[derive(Debug, Clone)]
pub struct SyscallBreakdownDelta {
    /// Read syscall delta %
    pub read_delta: f64,
    /// Write syscall delta %
    pub write_delta: f64,
    /// Mmap syscall delta %
    pub mmap_delta: f64,
    /// Futex syscall delta %
    pub futex_delta: f64,
    /// Other syscall delta %
    pub other_delta: f64,
    /// Total syscall delta %
    pub total_delta: f64,
}

impl SyscallBreakdownDelta {
    /// Get maximum absolute delta
    pub fn max_delta(&self) -> f64 {
        self.read_delta
            .abs()
            .max(self.write_delta.abs())
            .max(self.mmap_delta.abs())
            .max(self.futex_delta.abs())
            .max(self.other_delta.abs())
    }

    /// Check if any delta exceeds threshold
    pub fn exceeds_threshold(&self, threshold_percent: f64) -> bool {
        self.read_delta.abs() > threshold_percent
            || self.write_delta.abs() > threshold_percent
            || self.mmap_delta.abs() > threshold_percent
            || self.futex_delta.abs() > threshold_percent
            || self.other_delta.abs() > threshold_percent
    }
}

/// Performance metrics for trace comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceMetrics {
    /// Total execution time in microseconds
    pub total_time_us: f64,
    /// P50 latency in microseconds
    pub p50_latency_us: f64,
    /// P99 latency in microseconds
    pub p99_latency_us: f64,
    /// Throughput (ops/sec)
    pub throughput: f64,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: u64,
    /// Syscall breakdown
    pub syscalls: SyscallBreakdown,
    /// Custom metrics
    #[serde(default)]
    pub custom: HashMap<String, f64>,
}

impl Default for TraceMetrics {
    fn default() -> Self {
        Self {
            total_time_us: 0.0,
            p50_latency_us: 0.0,
            p99_latency_us: 0.0,
            throughput: 0.0,
            peak_memory_bytes: 0,
            syscalls: SyscallBreakdown::default(),
            custom: HashMap::new(),
        }
    }
}

impl TraceMetrics {
    /// Create new metrics
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set total time
    pub fn total_time_us(mut self, us: f64) -> Self {
        self.total_time_us = us;
        self
    }

    /// Builder: set P50 latency
    pub fn p50_latency_us(mut self, us: f64) -> Self {
        self.p50_latency_us = us;
        self
    }

    /// Builder: set P99 latency
    pub fn p99_latency_us(mut self, us: f64) -> Self {
        self.p99_latency_us = us;
        self
    }

    /// Builder: set throughput
    pub fn throughput(mut self, ops_per_sec: f64) -> Self {
        self.throughput = ops_per_sec;
        self
    }

    /// Builder: set peak memory
    pub fn peak_memory_bytes(mut self, bytes: u64) -> Self {
        self.peak_memory_bytes = bytes;
        self
    }

    /// Builder: set syscall breakdown
    pub fn syscalls(mut self, syscalls: SyscallBreakdown) -> Self {
        self.syscalls = syscalls;
        self
    }

    /// Builder: add custom metric
    pub fn with_custom(mut self, key: &str, value: f64) -> Self {
        self.custom.insert(key.to_string(), value);
        self
    }

    /// Check if metrics are valid
    pub fn is_valid(&self) -> bool {
        self.total_time_us >= 0.0
            && self.p50_latency_us >= 0.0
            && self.p99_latency_us >= 0.0
            && self.throughput >= 0.0
            && !self.total_time_us.is_nan()
            && !self.p50_latency_us.is_nan()
            && !self.p99_latency_us.is_nan()
            && !self.throughput.is_nan()
    }
}
