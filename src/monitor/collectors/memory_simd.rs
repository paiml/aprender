//! SIMD-accelerated Memory metrics collector.
//!
//! This module provides a high-performance memory collector using SIMD operations
//! for parsing `/proc/meminfo` and computing metrics.
//!
//! ## Performance Targets (Falsifiable)
//!
//! - Standard meminfo: < 30μs
//! - Extended meminfo (all fields): < 50μs
//!
//! ## Design
//!
//! Uses SIMD multi-pattern key search to parse all meminfo fields in a single pass.
//! Key fields are identified using byte pattern matching for optimal performance.

use crate::monitor::error::{MonitorError, Result};
use crate::monitor::simd::ring_buffer::SimdRingBuffer;
use crate::monitor::simd::soa::MemoryMetricsSoA;
use crate::monitor::simd::{kernels, SimdStats};
use crate::monitor::types::{Collector, MetricValue, Metrics};
use std::time::Duration;

/// SIMD-accelerated memory collector.
///
/// Uses optimized parsing and SIMD operations for high-performance metric collection.
#[derive(Debug)]
pub struct SimdMemoryCollector {
    /// Current memory metrics.
    metrics: MemoryMetricsSoA,
    /// History of memory usage (normalized 0.0-1.0).
    history: SimdRingBuffer,
    /// Swap usage history.
    swap_history: SimdRingBuffer,
    /// Pre-allocated read buffer (4KB, aligned).
    #[cfg(target_os = "linux")]
    read_buffer: Vec<u8>,
}

/// Memory field keys for fast lookup.
/// Using byte patterns for SIMD-friendly comparison.
#[cfg(target_os = "linux")]
mod meminfo_keys {
    pub(super) const MEM_TOTAL: &[u8] = b"MemTotal:";
    pub(super) const MEM_FREE: &[u8] = b"MemFree:";
    pub(super) const MEM_AVAILABLE: &[u8] = b"MemAvailable:";
    pub(super) const BUFFERS: &[u8] = b"Buffers:";
    pub(super) const CACHED: &[u8] = b"Cached:";
    pub(super) const SWAP_TOTAL: &[u8] = b"SwapTotal:";
    pub(super) const SWAP_FREE: &[u8] = b"SwapFree:";
    pub(super) const DIRTY: &[u8] = b"Dirty:";
    pub(super) const SLAB: &[u8] = b"Slab:";
}

impl SimdMemoryCollector {
    /// Creates a new SIMD memory collector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metrics: MemoryMetricsSoA::new(),
            history: SimdRingBuffer::new(300),
            swap_history: SimdRingBuffer::new(300),
            #[cfg(target_os = "linux")]
            read_buffer: vec![0u8; 4096],
        }
    }

    /// Parses /proc/meminfo using SIMD-optimized reading.
    #[cfg(target_os = "linux")]
    fn parse_meminfo(&mut self) -> Result<()> {
        use std::fs::File;
        use std::io::Read;

        let mut file = File::open("/proc/meminfo").map_err(|e| MonitorError::CollectionFailed {
            collector: "memory_simd",
            message: format!("Failed to open /proc/meminfo: {e}"),
        })?;

        let bytes_read =
            file.read(&mut self.read_buffer).map_err(|e| MonitorError::CollectionFailed {
                collector: "memory_simd",
                message: format!("Failed to read /proc/meminfo: {e}"),
            })?;

        // Copy to local buffer to avoid borrow conflict (self.read_buffer vs &mut self)
        #[allow(clippy::unnecessary_to_owned)]
        let buffer = self.read_buffer[..bytes_read].to_vec();
        self.parse_meminfo_buffer(&buffer)
    }

    /// Parses meminfo buffer using SIMD-assisted line finding.
    #[cfg(target_os = "linux")]
    fn parse_meminfo_buffer(&mut self, buffer: &[u8]) -> Result<()> {
        use meminfo_keys::{
            BUFFERS, CACHED, DIRTY, MEM_AVAILABLE, MEM_FREE, MEM_TOTAL, SLAB, SWAP_FREE, SWAP_TOTAL,
        };

        // Find line boundaries using SIMD
        let newlines = kernels::simd_find_newlines(buffer);

        let mut line_start = 0;

        for &newline_pos in &newlines {
            let line = &buffer[line_start..newline_pos];

            // Multi-pattern matching for memory fields
            // Each check uses prefix comparison for efficiency
            if line.starts_with(MEM_TOTAL) {
                self.metrics.total = parse_kb_value(line) * 1024;
            } else if line.starts_with(MEM_FREE) {
                self.metrics.free = parse_kb_value(line) * 1024;
            } else if line.starts_with(MEM_AVAILABLE) {
                self.metrics.available = parse_kb_value(line) * 1024;
            } else if line.starts_with(BUFFERS) {
                self.metrics.buffers = parse_kb_value(line) * 1024;
            } else if line.starts_with(CACHED) && !line.starts_with(b"Cached:") {
                // Skip SwapCached by checking exact prefix
            } else if line.starts_with(b"Cached:") {
                self.metrics.cached = parse_kb_value(line) * 1024;
            } else if line.starts_with(SWAP_TOTAL) {
                self.metrics.swap_total = parse_kb_value(line) * 1024;
            } else if line.starts_with(SWAP_FREE) {
                self.metrics.swap_free = parse_kb_value(line) * 1024;
            } else if line.starts_with(DIRTY) {
                self.metrics.dirty = parse_kb_value(line) * 1024;
            } else if line.starts_with(SLAB) {
                self.metrics.slab = parse_kb_value(line) * 1024;
            }

            line_start = newline_pos + 1;
        }

        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    fn parse_meminfo(&mut self) -> Result<()> {
        // Non-Linux platforms: generate dummy data
        self.metrics.total = 16 * 1024 * 1024 * 1024;
        self.metrics.free = 4 * 1024 * 1024 * 1024;
        self.metrics.available = 8 * 1024 * 1024 * 1024;
        self.metrics.buffers = 512 * 1024 * 1024;
        self.metrics.cached = 2 * 1024 * 1024 * 1024;
        self.metrics.swap_total = 8 * 1024 * 1024 * 1024;
        self.metrics.swap_free = 7 * 1024 * 1024 * 1024;
        Ok(())
    }

    /// Returns the memory usage history.
    #[must_use]
    pub fn history(&self) -> &SimdRingBuffer {
        &self.history
    }

    /// Returns the swap usage history.
    #[must_use]
    pub fn swap_history(&self) -> &SimdRingBuffer {
        &self.swap_history
    }

    /// Returns statistics for memory usage.
    #[must_use]
    pub fn usage_stats(&self) -> &SimdStats {
        self.history.statistics()
    }

    /// Returns current memory metrics.
    #[must_use]
    pub fn metrics(&self) -> &MemoryMetricsSoA {
        &self.metrics
    }
}

/// Parses KB value from a meminfo line.
///
/// Format: "Key:     12345 kB"
#[inline]
fn parse_kb_value(line: &[u8]) -> u64 {
    // Find the colon and skip whitespace
    if let Some(colon_pos) = line.iter().position(|&b| b == b':') {
        let value_part = &line[colon_pos + 1..];

        // Use SIMD integer parsing
        let integers = kernels::simd_parse_integers(value_part);
        integers.first().copied().unwrap_or(0)
    } else {
        0
    }
}

impl Default for SimdMemoryCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl Collector for SimdMemoryCollector {
    fn id(&self) -> &'static str {
        "memory_simd"
    }

    fn collect(&mut self) -> Result<Metrics> {
        // Parse /proc/meminfo
        self.parse_meminfo()?;

        // Build metrics map
        let mut metrics = Metrics::new();

        metrics.insert("memory.total", MetricValue::Counter(self.metrics.total));
        metrics.insert("memory.free", MetricValue::Counter(self.metrics.free));
        metrics.insert("memory.available", MetricValue::Counter(self.metrics.available));
        metrics.insert("memory.used", MetricValue::Counter(self.metrics.used()));
        metrics.insert("memory.buffers", MetricValue::Counter(self.metrics.buffers));
        metrics.insert("memory.cached", MetricValue::Counter(self.metrics.cached));
        metrics.insert("memory.swap.total", MetricValue::Counter(self.metrics.swap_total));
        metrics.insert("memory.swap.free", MetricValue::Counter(self.metrics.swap_free));
        metrics.insert(
            "memory.swap.used",
            MetricValue::Counter(self.metrics.swap_total.saturating_sub(self.metrics.swap_free)),
        );
        metrics.insert("memory.dirty", MetricValue::Counter(self.metrics.dirty));
        metrics.insert("memory.slab", MetricValue::Counter(self.metrics.slab));

        // Calculate percentages using the SoA methods
        let used_percent = self.metrics.usage_pct();
        metrics.insert("memory.used.percent", used_percent);

        let swap_percent = self.metrics.swap_usage_pct();
        if self.metrics.swap_total > 0 {
            metrics.insert("memory.swap.percent", swap_percent);
        }

        // Update history buffers
        self.history.push(used_percent / 100.0);
        self.swap_history.push(swap_percent / 100.0);

        Ok(metrics)
    }

    fn is_available(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            std::path::Path::new("/proc/meminfo").exists()
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
        "Memory (SIMD)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_memory_collector_new() {
        let collector = SimdMemoryCollector::new();
        assert!(collector.history.is_empty());
    }

    #[test]
    fn test_simd_memory_collector_id() {
        let collector = SimdMemoryCollector::new();
        assert_eq!(collector.id(), "memory_simd");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_simd_memory_collector_available() {
        let collector = SimdMemoryCollector::new();
        assert!(collector.is_available());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_simd_memory_collector_collect() {
        let mut collector = SimdMemoryCollector::new();
        let result = collector.collect();
        assert!(result.is_ok());

        let metrics = result.expect("collect should succeed");
        assert!(metrics.get_counter("memory.total").is_some());
        assert!(metrics.get_gauge("memory.used.percent").is_some());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_simd_memory_parse_buffer() {
        let mut collector = SimdMemoryCollector::new();
        let test_data = b"MemTotal:       16000000 kB\nMemFree:         4000000 kB\nMemAvailable:    8000000 kB\nBuffers:          500000 kB\nCached:          2000000 kB\nSwapTotal:       8000000 kB\nSwapFree:        7000000 kB\nDirty:              1000 kB\nSlab:             500000 kB\n";

        let result = collector.parse_meminfo_buffer(test_data);
        assert!(result.is_ok());

        assert_eq!(collector.metrics.total, 16_000_000 * 1024);
        assert_eq!(collector.metrics.free, 4_000_000 * 1024);
        assert_eq!(collector.metrics.available, 8_000_000 * 1024);
        assert_eq!(collector.metrics.buffers, 500_000 * 1024);
        assert_eq!(collector.metrics.cached, 2_000_000 * 1024);
        assert_eq!(collector.metrics.swap_total, 8_000_000 * 1024);
        assert_eq!(collector.metrics.swap_free, 7_000_000 * 1024);
    }

    #[test]
    fn test_parse_kb_value() {
        let line = b"MemTotal:       16000000 kB";
        assert_eq!(parse_kb_value(line), 16000000);

        let line2 = b"Cached:          2000000 kB";
        assert_eq!(parse_kb_value(line2), 2000000);
    }

    #[test]
    fn test_simd_memory_history() {
        let mut collector = SimdMemoryCollector::new();

        // Push some test data
        collector.history.push(0.5);
        collector.history.push(0.6);
        collector.history.push(0.7);

        assert_eq!(collector.history().len(), 3);
        assert_eq!(collector.history().latest(), Some(0.7));
    }

    #[test]
    fn test_memory_metrics_soa() {
        let mut metrics = MemoryMetricsSoA::new();
        metrics.total = 16_000_000_000;
        metrics.available = 8_000_000_000;

        assert_eq!(metrics.used(), 8_000_000_000);
        assert!((metrics.usage_pct() - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_simd_memory_display_name() {
        let collector = SimdMemoryCollector::new();
        assert_eq!(collector.display_name(), "Memory (SIMD)");
    }

    #[test]
    fn test_simd_memory_interval() {
        let collector = SimdMemoryCollector::new();
        assert_eq!(collector.interval_hint(), Duration::from_millis(1000));
    }
}
