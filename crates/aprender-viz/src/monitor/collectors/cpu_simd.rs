//! SIMD-accelerated CPU metrics collector.
//!
//! This module provides a high-performance CPU collector using SIMD operations
//! for parsing `/proc/stat` and computing metrics.
//!
//! ## Performance Targets (Falsifiable)
//!
//! - 4-core system: < 50μs
//! - 16-core system: < 80μs
//! - 64-core system: < 150μs
//! - 128-core system: < 250μs
//!
//! ## Design
//!
//! Uses Structure-of-Arrays (SoA) layout for SIMD-friendly access:
//! - All user times in one contiguous array
//! - All system times in another
//! - Enables parallel delta and percentage calculations

use crate::monitor::error::{MonitorError, Result};
use crate::monitor::simd::ring_buffer::SimdRingBuffer;
use crate::monitor::simd::soa::CpuMetricsSoA;
use crate::monitor::simd::{kernels, SimdStats};
use crate::monitor::types::{Collector, MetricValue, Metrics};
use std::time::Duration;

/// SIMD-accelerated CPU collector.
///
/// Uses SoA layout and SIMD operations for high-performance metric collection.
#[derive(Debug)]
pub struct SimdCpuCollector {
    /// Current metrics in SoA layout.
    current: CpuMetricsSoA,
    /// Previous metrics for delta calculation.
    previous: CpuMetricsSoA,
    /// History of total CPU usage (normalized 0.0-1.0).
    history: SimdRingBuffer,
    /// Per-core usage history.
    core_history: Vec<SimdRingBuffer>,
    /// Number of CPU cores.
    core_count: usize,
    /// Load average values.
    load_average: LoadAverage,
    /// Pre-allocated read buffer (8KB, aligned).
    #[cfg(target_os = "linux")]
    read_buffer: Vec<u8>,
    /// Whether we have previous data for delta calculation.
    has_previous: bool,
}

/// Load average values.
#[derive(Debug, Clone, Copy, Default)]
pub struct LoadAverage {
    /// 1-minute load average.
    pub one: f64,
    /// 5-minute load average.
    pub five: f64,
    /// 15-minute load average.
    pub fifteen: f64,
}

impl SimdCpuCollector {
    /// Creates a new SIMD CPU collector.
    #[must_use]
    pub fn new() -> Self {
        let core_count = Self::detect_core_count();

        // Pre-allocate per-core history buffers
        let mut core_history = Vec::with_capacity(core_count);
        for _ in 0..core_count {
            core_history.push(SimdRingBuffer::new(300));
        }

        Self {
            current: CpuMetricsSoA::new(core_count),
            previous: CpuMetricsSoA::new(core_count),
            history: SimdRingBuffer::new(300),
            core_history,
            core_count,
            load_average: LoadAverage::default(),
            #[cfg(target_os = "linux")]
            read_buffer: vec![0u8; 8192],
            has_previous: false,
        }
    }

    /// Detects the number of CPU cores.
    fn detect_core_count() -> usize {
        #[cfg(target_os = "linux")]
        {
            // Fast detection via /sys
            std::fs::read_to_string("/sys/devices/system/cpu/present")
                .ok()
                .and_then(|s| {
                    // Format: "0-N" or "0"
                    let trimmed = s.trim();
                    if let Some(pos) = trimmed.rfind('-') {
                        trimmed[pos + 1..].parse::<usize>().ok().map(|n| n + 1)
                    } else {
                        Some(1)
                    }
                })
                .unwrap_or(1)
        }
        #[cfg(not(target_os = "linux"))]
        {
            1
        }
    }

    /// Parses /proc/stat using optimized reading.
    #[cfg(target_os = "linux")]
    fn parse_proc_stat(&mut self) -> Result<()> {
        use std::fs::File;
        use std::io::Read;

        let mut file = File::open("/proc/stat").map_err(|e| MonitorError::CollectionFailed {
            collector: "cpu_simd",
            message: format!("Failed to open /proc/stat: {e}"),
        })?;

        let bytes_read =
            file.read(&mut self.read_buffer).map_err(|e| MonitorError::CollectionFailed {
                collector: "cpu_simd",
                message: format!("Failed to read /proc/stat: {e}"),
            })?;

        // Copy to local buffer to avoid borrow conflict (self.read_buffer vs &mut self)
        #[allow(clippy::unnecessary_to_owned)]
        let buffer = self.read_buffer[..bytes_read].to_vec();
        self.parse_stat_buffer(&buffer)
    }

    /// Parses the stat buffer content.
    #[cfg(target_os = "linux")]
    fn parse_stat_buffer(&mut self, buffer: &[u8]) -> Result<()> {
        // Find line boundaries
        let newlines = kernels::simd_find_newlines(buffer);

        let mut line_start = 0;
        let mut core_idx = 0;

        for &newline_pos in &newlines {
            let line = &buffer[line_start..newline_pos];

            if line.starts_with(b"cpu ") {
                // Total CPU line - skip the "cpu " prefix (4 bytes)
                let values = kernels::simd_parse_integers(&line[4..]);
                if values.len() >= 8 {
                    // Store in "total" position (we'll compute it separately)
                    self.current.set_core(
                        self.core_count, // Use an extra slot for total
                        values[0],
                        values[1],
                        values[2],
                        values[3],
                        values.get(4).copied().unwrap_or(0),
                        values.get(5).copied().unwrap_or(0),
                        values.get(6).copied().unwrap_or(0),
                        values.get(7).copied().unwrap_or(0),
                    );
                }
            } else if line.starts_with(b"cpu") && line.len() > 3 {
                // Per-core CPU line - find where numbers start
                if let Some(space_pos) = line.iter().position(|&b| b == b' ') {
                    let values = kernels::simd_parse_integers(&line[space_pos + 1..]);
                    if values.len() >= 8 && core_idx < self.core_count {
                        self.current.set_core(
                            core_idx,
                            values[0],
                            values[1],
                            values[2],
                            values[3],
                            values.get(4).copied().unwrap_or(0),
                            values.get(5).copied().unwrap_or(0),
                            values.get(6).copied().unwrap_or(0),
                            values.get(7).copied().unwrap_or(0),
                        );
                        core_idx += 1;
                    }
                }
            }

            line_start = newline_pos + 1;
        }

        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    fn parse_proc_stat(&mut self) -> Result<()> {
        // Non-Linux platforms: generate dummy data
        for i in 0..self.core_count {
            self.current.set_core(i, 100, 10, 50, 800, 20, 5, 5, 10);
        }
        Ok(())
    }

    /// Computes usage percentages from current and previous metrics.
    fn compute_usage(&mut self) {
        if !self.has_previous {
            return;
        }

        // Use SIMD delta and percentage calculations
        self.current.compute_usage_from_delta(&self.previous);

        // Update history buffers
        let avg = self.current.avg_usage();
        self.history.push(avg / 100.0); // Normalized for graphs

        for i in 0..self.core_count.min(self.core_history.len()) {
            let usage = self.current.usage(i);
            self.core_history[i].push(usage / 100.0);
        }
    }

    /// Reads load average.
    fn read_load_average(&mut self) {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/loadavg") {
                // Load averages are floating point in /proc/loadavg
                let parts: Vec<&str> = content.split_whitespace().collect();
                if parts.len() >= 3 {
                    self.load_average = LoadAverage {
                        one: parts[0].parse().unwrap_or(0.0),
                        five: parts[1].parse().unwrap_or(0.0),
                        fifteen: parts[2].parse().unwrap_or(0.0),
                    };
                }
            }
        }
    }

    /// Returns the CPU usage history.
    #[must_use]
    pub fn history(&self) -> &SimdRingBuffer {
        &self.history
    }

    /// Returns the per-core usage history.
    #[must_use]
    pub fn core_history(&self, core: usize) -> Option<&SimdRingBuffer> {
        self.core_history.get(core)
    }

    /// Returns the number of CPU cores.
    #[must_use]
    pub fn core_count(&self) -> usize {
        self.core_count
    }

    /// Returns the latest load average.
    #[must_use]
    pub fn load_average(&self) -> LoadAverage {
        self.load_average
    }

    /// Returns statistics for total CPU usage.
    #[must_use]
    pub fn usage_stats(&self) -> &SimdStats {
        self.history.statistics()
    }

    /// Returns current per-core usage percentages.
    #[must_use]
    pub fn per_core_usage(&self) -> &[f64] {
        &self.current.usage_pct[..self.core_count]
    }
}

impl Default for SimdCpuCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl Collector for SimdCpuCollector {
    fn id(&self) -> &'static str {
        "cpu_simd"
    }

    fn collect(&mut self) -> Result<Metrics> {
        // Parse /proc/stat
        self.parse_proc_stat()?;

        // Compute usage from deltas
        self.compute_usage();

        // Read load average
        self.read_load_average();

        // Build metrics map
        let mut metrics = Metrics::new();

        // Total CPU usage (average across cores)
        let total_usage = self.current.avg_usage();
        metrics.insert("cpu.total", total_usage);

        // Per-core usage
        for i in 0..self.core_count {
            metrics.insert(format!("cpu.core.{i}"), self.current.usage(i));
        }

        // Core count
        metrics.insert("cpu.cores", MetricValue::Counter(self.core_count as u64));

        // Load average
        metrics.insert("cpu.load.1", self.load_average.one);
        metrics.insert("cpu.load.5", self.load_average.five);
        metrics.insert("cpu.load.15", self.load_average.fifteen);

        // Swap buffers for next delta calculation
        std::mem::swap(&mut self.current, &mut self.previous);
        self.has_previous = true;

        Ok(metrics)
    }

    fn is_available(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            std::path::Path::new("/proc/stat").exists()
        }
        #[cfg(not(target_os = "linux"))]
        {
            false
        }
    }

    fn interval_hint(&self) -> Duration {
        Duration::from_millis(1000)
    }

    fn display_name(&self) -> &'static str {
        "CPU (SIMD)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_cpu_collector_new() {
        let collector = SimdCpuCollector::new();
        assert!(collector.core_count >= 1);
    }

    #[test]
    fn test_simd_cpu_collector_id() {
        let collector = SimdCpuCollector::new();
        assert_eq!(collector.id(), "cpu_simd");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_simd_cpu_collector_available() {
        let collector = SimdCpuCollector::new();
        assert!(collector.is_available());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_simd_cpu_collector_collect() {
        let mut collector = SimdCpuCollector::new();

        // First collection establishes baseline
        let result1 = collector.collect();
        assert!(result1.is_ok());

        // Second collection computes deltas
        std::thread::sleep(Duration::from_millis(100));
        let result2 = collector.collect();
        assert!(result2.is_ok());

        let metrics = result2.expect("collect should succeed");
        assert!(metrics.get_gauge("cpu.total").is_some());
    }

    #[test]
    fn test_load_average_default() {
        let la = LoadAverage::default();
        assert_eq!(la.one, 0.0);
        assert_eq!(la.five, 0.0);
        assert_eq!(la.fifteen, 0.0);
    }

    #[test]
    fn test_simd_cpu_history() {
        let mut collector = SimdCpuCollector::new();

        // Push some test data
        collector.history.push(0.5);
        collector.history.push(0.6);
        collector.history.push(0.7);

        assert_eq!(collector.history().len(), 3);
        assert_eq!(collector.history().latest(), Some(0.7));
    }
}
