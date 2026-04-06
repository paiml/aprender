//! SIMD-accelerated Disk I/O metrics collector.
//!
//! This module provides a high-performance disk collector using SIMD operations
//! for parsing `/proc/diskstats` and computing I/O throughput metrics.
//!
//! ## Performance Targets (Falsifiable)
//!
//! - Standard disks (≤8): < 50μs
//! - Many disks (≤32): < 100μs
//!
//! ## Design
//!
//! Uses Structure-of-Arrays (SoA) layout for SIMD-friendly access to
//! per-disk statistics. Delta calculations and rate computations
//! use vectorized operations.

use crate::monitor::error::Result;
use crate::monitor::simd::ring_buffer::SimdRingBuffer;
use crate::monitor::simd::soa::DiskMetricsSoA;
use crate::monitor::simd::{kernels, SimdStats};
use crate::monitor::types::{Collector, MetricValue, Metrics};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// Reads a file with a timeout to prevent hanging on blocked devices.
fn read_file_with_timeout(path: &str, timeout: Duration) -> Option<String> {
    let path = path.to_string();
    let (tx, rx) = mpsc::channel();

    std::thread::spawn(move || {
        let result = std::fs::read_to_string(&path);
        let _ = tx.send(result);
    });

    match rx.recv_timeout(timeout) {
        Ok(Ok(content)) => Some(content),
        _ => None,
    }
}

/// SIMD-accelerated disk collector.
///
/// Uses SoA layout and SIMD operations for high-performance metric collection.
#[derive(Debug)]
pub struct SimdDiskCollector {
    /// Current metrics in SoA layout.
    current: DiskMetricsSoA,
    /// Previous metrics for delta calculation.
    previous: DiskMetricsSoA,
    /// Previous collection time.
    prev_time: Option<Instant>,
    /// Calculated read rates per disk (bytes/sec).
    read_rates: Vec<f64>,
    /// Calculated write rates per disk (bytes/sec).
    write_rates: Vec<f64>,
    /// Read IOPS per disk.
    read_iops: Vec<f64>,
    /// Write IOPS per disk.
    write_iops: Vec<f64>,
    /// IO utilization per disk.
    io_utilization: Vec<f64>,
    /// Read throughput history (normalized 0-1).
    read_history: SimdRingBuffer,
    /// Write throughput history (normalized 0-1).
    write_history: SimdRingBuffer,
    /// Sector size (typically 512 bytes).
    sector_size: u64,
    /// Maximum throughput for normalization.
    max_throughput: f64,
    /// Pre-allocated read buffer (16KB) for future SIMD parsing.
    #[cfg(target_os = "linux")]
    #[allow(dead_code)]
    read_buffer: Vec<u8>,
    /// Whether we have previous data for delta calculation.
    has_previous: bool,
}

impl SimdDiskCollector {
    /// Creates a new SIMD disk collector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            current: DiskMetricsSoA::new(32),
            previous: DiskMetricsSoA::new(32),
            prev_time: None,
            read_rates: Vec::with_capacity(32),
            write_rates: Vec::with_capacity(32),
            read_iops: Vec::with_capacity(32),
            write_iops: Vec::with_capacity(32),
            io_utilization: Vec::with_capacity(32),
            read_history: SimdRingBuffer::new(300),
            write_history: SimdRingBuffer::new(300),
            sector_size: 512,
            max_throughput: 1_000_000_000.0, // 1 GB/s default max
            #[cfg(target_os = "linux")]
            read_buffer: vec![0u8; 16384],
            has_previous: false,
        }
    }

    /// Parses /proc/diskstats using SIMD-optimized reading.
    #[cfg(target_os = "linux")]
    fn parse_diskstats(&mut self) -> Result<()> {
        // Use timeout to prevent hanging on blocked/hung disk devices
        let content = match read_file_with_timeout("/proc/diskstats", Duration::from_secs(2)) {
            Some(c) => c,
            None => return Ok(()), // Graceful degradation on timeout
        };

        // Reset current metrics for fresh parse
        self.current = DiskMetricsSoA::new(32);

        self.parse_diskstats_buffer(content.as_bytes())
    }

    /// Parses diskstats buffer using SIMD-assisted line finding.
    #[cfg(target_os = "linux")]
    fn parse_diskstats_buffer(&mut self, buffer: &[u8]) -> Result<()> {
        // Find line boundaries using SIMD
        let newlines = kernels::simd_find_newlines(buffer);

        let mut line_start = 0;

        for &newline_pos in &newlines {
            let line = &buffer[line_start..newline_pos];

            // Parse the line using SIMD integer parsing
            let values = kernels::simd_parse_integers(line);

            // diskstats format: major minor name reads_completed reads_merged sectors_read ...
            if values.len() >= 14 {
                // Extract device name (field 2)
                let fields: Vec<&str> =
                    std::str::from_utf8(line).unwrap_or("").split_whitespace().collect();

                if fields.len() >= 3 {
                    let name = fields[2];

                    // Skip virtual/pseudo block devices
                    if name.starts_with("loop")
                        || name.starts_with("ram")
                        || name.starts_with("zram")
                        || name.starts_with("dm-")
                    {
                        line_start = newline_pos + 1;
                        continue;
                    }

                    // Check if partition (ends with digit, except nvme with 'p')
                    let is_partition = name.chars().last().is_some_and(|c| c.is_ascii_digit())
                        && !name.contains("nvme")
                        || (name.contains("nvme") && name.contains('p'));

                    // Skip partitions unless they have activity
                    if is_partition && values[0] == 0 && values[4] == 0 {
                        line_start = newline_pos + 1;
                        continue;
                    }

                    // Set disk metrics in SoA structure
                    // values[0..1] = major, minor
                    // values[2] = reads_completed (field 3 in file, but 0-indexed after major/minor)
                    // Looking at simd_parse_integers, it parses all integers including major/minor
                    // So: [major, minor, reads, reads_merged, sectors_read, read_time, writes, ...]
                    self.current.set_disk(
                        name, values[2],  // reads_completed
                        values[4],  // sectors_read
                        values[5],  // read_time_ms
                        values[6],  // writes_completed
                        values[8],  // sectors_written
                        values[9],  // write_time_ms
                        values[10], // io_in_progress
                        values[11], // io_time_ms
                    );
                }
            }

            line_start = newline_pos + 1;
        }

        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    fn parse_diskstats(&mut self) -> Result<()> {
        // Non-Linux platforms: generate dummy data
        self.current = DiskMetricsSoA::new(32);
        self.current.set_disk("sda", 1000, 2000, 100, 500, 1000, 50, 0, 150);
        Ok(())
    }

    /// Computes rates from current and previous metrics.
    fn compute_rates(&mut self, elapsed_secs: f64) {
        if !self.has_previous || elapsed_secs <= 0.0 {
            return;
        }

        let count = self.current.disk_count;
        let prev_count = self.previous.disk_count;
        let min_count = count.min(prev_count);

        // Calculate sector deltas using SIMD
        let read_sector_delta = kernels::simd_delta(
            &self.current.sectors_read[..min_count],
            &self.previous.sectors_read[..min_count],
        );

        let write_sector_delta = kernels::simd_delta(
            &self.current.sectors_written[..min_count],
            &self.previous.sectors_written[..min_count],
        );

        // Calculate operation deltas
        let read_ops_delta = kernels::simd_delta(
            &self.current.reads_completed[..min_count],
            &self.previous.reads_completed[..min_count],
        );

        let write_ops_delta = kernels::simd_delta(
            &self.current.writes_completed[..min_count],
            &self.previous.writes_completed[..min_count],
        );

        // Calculate IO time delta
        let io_time_delta = kernels::simd_delta(
            &self.current.io_time_ms[..min_count],
            &self.previous.io_time_ms[..min_count],
        );

        // Compute rates
        self.read_rates.clear();
        self.write_rates.clear();
        self.read_iops.clear();
        self.write_iops.clear();
        self.io_utilization.clear();

        for i in 0..read_sector_delta.len() {
            let read_bytes = read_sector_delta[i] * self.sector_size;
            let write_bytes = write_sector_delta[i] * self.sector_size;

            self.read_rates.push(read_bytes as f64 / elapsed_secs);
            self.write_rates.push(write_bytes as f64 / elapsed_secs);
            self.read_iops.push(read_ops_delta[i] as f64 / elapsed_secs);
            self.write_iops.push(write_ops_delta[i] as f64 / elapsed_secs);

            // IO utilization: time spent doing IO / total time
            let util = (io_time_delta[i] as f64 / (elapsed_secs * 1000.0) * 100.0).min(100.0);
            self.io_utilization.push(util);
        }
    }

    /// Updates history buffers.
    fn update_history(&mut self) {
        // Calculate total throughput using SIMD sum
        let total_read = kernels::simd_sum(&self.read_rates);
        let total_write = kernels::simd_sum(&self.write_rates);

        // Normalize to 0-1 range
        let read_norm = (total_read / self.max_throughput).min(1.0);
        let write_norm = (total_write / self.max_throughput).min(1.0);

        self.read_history.push(read_norm);
        self.write_history.push(write_norm);
    }

    /// Returns the disk names.
    #[must_use]
    pub fn disk_names(&self) -> &[String] {
        &self.current.names
    }

    /// Returns read throughput history (normalized 0-1).
    #[must_use]
    pub fn read_history(&self) -> &SimdRingBuffer {
        &self.read_history
    }

    /// Returns write throughput history (normalized 0-1).
    #[must_use]
    pub fn write_history(&self) -> &SimdRingBuffer {
        &self.write_history
    }

    /// Returns read rate statistics.
    #[must_use]
    pub fn read_stats(&self) -> &SimdStats {
        self.read_history.statistics()
    }

    /// Returns write rate statistics.
    #[must_use]
    pub fn write_stats(&self) -> &SimdStats {
        self.write_history.statistics()
    }

    /// Returns read rate for a specific disk.
    #[must_use]
    pub fn read_rate(&self, idx: usize) -> f64 {
        self.read_rates.get(idx).copied().unwrap_or(0.0)
    }

    /// Returns write rate for a specific disk.
    #[must_use]
    pub fn write_rate(&self, idx: usize) -> f64 {
        self.write_rates.get(idx).copied().unwrap_or(0.0)
    }

    /// Returns read IOPS for a specific disk.
    #[must_use]
    pub fn read_iops(&self, idx: usize) -> f64 {
        self.read_iops.get(idx).copied().unwrap_or(0.0)
    }

    /// Returns write IOPS for a specific disk.
    #[must_use]
    pub fn write_iops(&self, idx: usize) -> f64 {
        self.write_iops.get(idx).copied().unwrap_or(0.0)
    }

    /// Returns IO utilization for a specific disk.
    #[must_use]
    pub fn io_utilization(&self, idx: usize) -> f64 {
        self.io_utilization.get(idx).copied().unwrap_or(0.0)
    }

    /// Returns total bytes read using SIMD sum.
    #[must_use]
    pub fn total_bytes_read(&self) -> u64 {
        self.current.total_bytes_read()
    }

    /// Returns total bytes written using SIMD sum.
    #[must_use]
    pub fn total_bytes_written(&self) -> u64 {
        self.current.total_bytes_written()
    }
}

impl Default for SimdDiskCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl Collector for SimdDiskCollector {
    fn id(&self) -> &'static str {
        "disk_simd"
    }

    fn collect(&mut self) -> Result<Metrics> {
        let now = Instant::now();

        // Parse /proc/diskstats
        self.parse_diskstats()?;

        // Calculate rates if we have previous data
        if let Some(prev_time) = self.prev_time {
            let elapsed = now.duration_since(prev_time);
            let elapsed_secs = elapsed.as_secs_f64();

            if elapsed_secs > 0.0 {
                self.compute_rates(elapsed_secs);
                self.update_history();
            }
        }

        // Swap buffers for next delta calculation
        std::mem::swap(&mut self.current, &mut self.previous);
        self.current = DiskMetricsSoA::new(32);
        self.prev_time = Some(now);
        self.has_previous = true;

        // Build metrics
        let mut metrics = Metrics::new();

        // Total read/write throughput using SIMD sum
        let total_read = kernels::simd_sum(&self.read_rates);
        let total_write = kernels::simd_sum(&self.write_rates);

        metrics.insert("disk.read_bytes_per_sec", MetricValue::Gauge(total_read));
        metrics.insert("disk.write_bytes_per_sec", MetricValue::Gauge(total_write));

        // Total IOPS using SIMD sum
        let total_read_iops = kernels::simd_sum(&self.read_iops);
        let total_write_iops = kernels::simd_sum(&self.write_iops);

        metrics.insert("disk.read_iops", MetricValue::Gauge(total_read_iops));
        metrics.insert("disk.write_iops", MetricValue::Gauge(total_write_iops));

        // Disk count
        metrics.insert("disk.disk_count", MetricValue::Counter(self.previous.disk_count as u64));

        Ok(metrics)
    }

    fn is_available(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            std::path::Path::new("/proc/diskstats").exists()
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
        "Disk (SIMD)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_disk_collector_new() {
        let collector = SimdDiskCollector::new();
        assert!(collector.read_history.is_empty());
        assert_eq!(collector.sector_size, 512);
    }

    #[test]
    fn test_simd_disk_collector_id() {
        let collector = SimdDiskCollector::new();
        assert_eq!(collector.id(), "disk_simd");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_simd_disk_collector_available() {
        let collector = SimdDiskCollector::new();
        assert!(collector.is_available());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_simd_disk_collector_collect() {
        let mut collector = SimdDiskCollector::new();

        // First collection establishes baseline
        let result1 = collector.collect();
        assert!(result1.is_ok());

        // Wait and collect again for rate calculation
        std::thread::sleep(Duration::from_millis(100));
        let result2 = collector.collect();
        assert!(result2.is_ok());

        let metrics = result2.expect("collect should succeed");
        assert!(metrics.get_gauge("disk.read_bytes_per_sec").is_some());
        assert!(metrics.get_gauge("disk.write_bytes_per_sec").is_some());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_simd_disk_parse_buffer() {
        let mut collector = SimdDiskCollector::new();
        // Full diskstats format with 14 fields: major minor name reads reads_merged sectors_read read_time writes writes_merged sectors_written write_time io_in_progress io_time weighted_io_time
        let test_data = b"   8       0 sda 1000 100 2000 500 800 50 1000 250 0 750 750 500\n   8       1 sda1 100 10 200 50 80 5 100 25 0 75 75 50\n   7       0 loop0 0 0 0 0 0 0 0 0 0 0 0 0\n";

        let result = collector.parse_diskstats_buffer(test_data);
        assert!(result.is_ok());

        // Should have parsed sda (sda1 has activity so included too, loop0 skipped)
        assert!(
            collector.current.disk_count >= 1,
            "Expected at least 1 disk, got {}",
            collector.current.disk_count
        );
    }

    #[test]
    fn test_simd_disk_display_name() {
        let collector = SimdDiskCollector::new();
        assert_eq!(collector.display_name(), "Disk (SIMD)");
    }

    #[test]
    fn test_simd_disk_interval() {
        let collector = SimdDiskCollector::new();
        assert_eq!(collector.interval_hint(), Duration::from_millis(1000));
    }

    #[test]
    fn test_simd_disk_rates() {
        let mut collector = SimdDiskCollector::new();

        // Set up test data
        collector.read_rates = vec![100_000.0, 200_000.0];
        collector.write_rates = vec![50_000.0, 100_000.0];
        collector.read_iops = vec![100.0, 200.0];
        collector.write_iops = vec![50.0, 100.0];
        collector.io_utilization = vec![25.0, 50.0];

        assert!((collector.read_rate(0) - 100_000.0).abs() < 0.01);
        assert!((collector.write_rate(0) - 50_000.0).abs() < 0.01);
        assert!((collector.read_iops(0) - 100.0).abs() < 0.01);
        assert!((collector.write_iops(0) - 50.0).abs() < 0.01);
        assert!((collector.io_utilization(0) - 25.0).abs() < 0.01);
        assert!((collector.read_rate(2) - 0.0).abs() < 0.01); // Out of bounds
    }

    #[test]
    fn test_disk_metrics_soa() {
        let mut disk = DiskMetricsSoA::new(4);
        disk.set_disk("sda", 100, 1000, 50, 200, 2000, 100, 0, 150);

        assert_eq!(disk.disk_count, 1);
        assert_eq!(disk.total_bytes_read(), 512_000);
        assert_eq!(disk.total_bytes_written(), 1_024_000);
    }

    #[test]
    fn test_simd_disk_history() {
        let mut collector = SimdDiskCollector::new();

        // Push some test data
        collector.read_history.push(0.3);
        collector.read_history.push(0.4);
        collector.read_history.push(0.5);

        assert_eq!(collector.read_history().len(), 3);
        assert_eq!(collector.read_history().latest(), Some(0.5));
    }
}
