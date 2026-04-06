//! Disk I/O Analyzer
//!
//! Parses `/proc/diskstats` to provide detailed disk I/O metrics including
//! read/write throughput, IOPS, and latency statistics.

#![allow(clippy::uninlined_format_args)]

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use super::{Analyzer, AnalyzerError};

/// I/O statistics for a single disk
#[derive(Debug, Clone, Default)]
pub struct DiskIoStats {
    /// Device name (e.g., "sda", "nvme0n1")
    pub device: String,
    /// Major device number
    pub major: u32,
    /// Minor device number
    pub minor: u32,
    /// Reads completed successfully
    pub reads_completed: u64,
    /// Reads merged
    pub reads_merged: u64,
    /// Sectors read
    pub sectors_read: u64,
    /// Time spent reading (ms)
    pub time_reading_ms: u64,
    /// Writes completed successfully
    pub writes_completed: u64,
    /// Writes merged
    pub writes_merged: u64,
    /// Sectors written
    pub sectors_written: u64,
    /// Time spent writing (ms)
    pub time_writing_ms: u64,
    /// I/Os currently in progress
    pub io_in_progress: u64,
    /// Time spent doing I/Os (ms)
    pub time_io_ms: u64,
    /// Weighted time spent doing I/Os (ms)
    pub weighted_time_io_ms: u64,
    /// Discards completed (kernel 4.18+)
    pub discards_completed: Option<u64>,
    /// Sectors discarded
    pub sectors_discarded: Option<u64>,
    /// Flush requests completed (kernel 5.5+)
    pub flush_requests: Option<u64>,
}

impl DiskIoStats {
    /// Calculate read bytes (assuming 512-byte sectors)
    pub fn read_bytes(&self) -> u64 {
        self.sectors_read * 512
    }

    /// Calculate written bytes (assuming 512-byte sectors)
    pub fn write_bytes(&self) -> u64 {
        self.sectors_written * 512
    }

    /// Is this a partition (vs whole disk)?
    pub fn is_partition(&self) -> bool {
        // Partitions typically have non-zero minor numbers
        // and names like sda1, nvme0n1p1
        self.device
            .chars()
            .last()
            .is_some_and(|c| c.is_ascii_digit())
            && !self.device.starts_with("nvme")
            || self.device.contains('p') && self.device.starts_with("nvme")
    }
}

/// Calculated I/O rates (per second)
#[derive(Debug, Clone, Default)]
pub struct DiskIoRates {
    /// Device name
    pub device: String,
    /// Read bytes per second
    pub read_bytes_per_sec: f64,
    /// Write bytes per second
    pub write_bytes_per_sec: f64,
    /// Read operations per second (IOPS)
    pub reads_per_sec: f64,
    /// Write operations per second (IOPS)
    pub writes_per_sec: f64,
    /// Average read latency (ms)
    pub avg_read_latency_ms: f64,
    /// Average write latency (ms)
    pub avg_write_latency_ms: f64,
    /// I/O utilization percentage (0-100)
    pub utilization_percent: f64,
}

impl DiskIoRates {
    /// Format read rate for display
    pub fn read_rate_display(&self) -> String {
        format_bytes_rate(self.read_bytes_per_sec)
    }

    /// Format write rate for display
    pub fn write_rate_display(&self) -> String {
        format_bytes_rate(self.write_bytes_per_sec)
    }

    /// Total IOPS
    pub fn total_iops(&self) -> f64 {
        self.reads_per_sec + self.writes_per_sec
    }
}

/// Disk I/O data
#[derive(Debug, Clone, Default)]
pub struct DiskIoData {
    /// Raw stats per device
    pub stats: HashMap<String, DiskIoStats>,
    /// Calculated rates per device
    pub rates: HashMap<String, DiskIoRates>,
    /// Total read bytes per second
    pub total_read_bytes_per_sec: f64,
    /// Total write bytes per second
    pub total_write_bytes_per_sec: f64,
}

impl DiskIoData {
    /// Get stats for physical disks only (no partitions)
    pub fn physical_disks(&self) -> impl Iterator<Item = (&String, &DiskIoStats)> {
        self.stats.iter().filter(|(_, s)| !s.is_partition())
    }

    /// Get rates for physical disks only
    pub fn physical_disk_rates(&self) -> impl Iterator<Item = (&String, &DiskIoRates)> {
        self.rates
            .iter()
            .filter(|(name, _)| self.stats.get(*name).is_some_and(|s| !s.is_partition()))
    }
}

/// Analyzer for disk I/O statistics
pub struct DiskIoAnalyzer {
    data: DiskIoData,
    interval: Duration,
    /// Previous stats for rate calculation
    prev_stats: HashMap<String, DiskIoStats>,
    /// Time of previous collection
    prev_time: Option<Instant>,
}

impl Default for DiskIoAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl DiskIoAnalyzer {
    /// Create a new disk I/O analyzer
    pub fn new() -> Self {
        Self {
            data: DiskIoData::default(),
            interval: Duration::from_secs(1),
            prev_stats: HashMap::new(),
            prev_time: None,
        }
    }

    /// Get the current data
    pub fn data(&self) -> &DiskIoData {
        &self.data
    }

    /// Parse /proc/diskstats
    fn parse_diskstats(&self) -> Result<HashMap<String, DiskIoStats>, AnalyzerError> {
        let contents = fs::read_to_string("/proc/diskstats").map_err(|e| {
            AnalyzerError::IoError(format!("Failed to read /proc/diskstats: {}", e))
        })?;

        let mut stats = HashMap::new();

        for line in contents.lines() {
            if let Some(disk_stats) = self.parse_diskstats_line(line) {
                stats.insert(disk_stats.device.clone(), disk_stats);
            }
        }

        Ok(stats)
    }

    /// Parse a single line from /proc/diskstats
    fn parse_diskstats_line(&self, line: &str) -> Option<DiskIoStats> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 14 {
            return None;
        }

        let major: u32 = parts[0].parse().ok()?;
        let minor: u32 = parts[1].parse().ok()?;
        let device = parts[2].to_string();

        // Skip loop devices and ram disks
        if device.starts_with("loop") || device.starts_with("ram") {
            return None;
        }

        let mut stats = DiskIoStats {
            device,
            major,
            minor,
            reads_completed: parts[3].parse().ok()?,
            reads_merged: parts[4].parse().ok()?,
            sectors_read: parts[5].parse().ok()?,
            time_reading_ms: parts[6].parse().ok()?,
            writes_completed: parts[7].parse().ok()?,
            writes_merged: parts[8].parse().ok()?,
            sectors_written: parts[9].parse().ok()?,
            time_writing_ms: parts[10].parse().ok()?,
            io_in_progress: parts[11].parse().ok()?,
            time_io_ms: parts[12].parse().ok()?,
            weighted_time_io_ms: parts[13].parse().ok()?,
            ..Default::default()
        };

        // Parse extended fields if available (kernel 4.18+)
        if parts.len() >= 18 {
            stats.discards_completed = parts[14].parse().ok();
            stats.sectors_discarded = parts[16].parse().ok();
        }

        // Parse flush requests if available (kernel 5.5+)
        if parts.len() >= 20 {
            stats.flush_requests = parts[18].parse().ok();
        }

        Some(stats)
    }

    /// Calculate rates from previous and current stats
    fn calculate_rates(
        &self,
        current: &HashMap<String, DiskIoStats>,
        elapsed_secs: f64,
    ) -> HashMap<String, DiskIoRates> {
        let mut rates = HashMap::new();

        for (device, curr) in current {
            if let Some(prev) = self.prev_stats.get(device) {
                let read_bytes_delta = (curr.sectors_read.saturating_sub(prev.sectors_read)) * 512;
                let write_bytes_delta =
                    (curr.sectors_written.saturating_sub(prev.sectors_written)) * 512;
                let reads_delta = curr.reads_completed.saturating_sub(prev.reads_completed);
                let writes_delta = curr.writes_completed.saturating_sub(prev.writes_completed);
                let time_reading_delta = curr.time_reading_ms.saturating_sub(prev.time_reading_ms);
                let time_writing_delta = curr.time_writing_ms.saturating_sub(prev.time_writing_ms);
                let time_io_delta = curr.time_io_ms.saturating_sub(prev.time_io_ms);

                let avg_read_latency = if reads_delta > 0 {
                    time_reading_delta as f64 / reads_delta as f64
                } else {
                    0.0
                };

                let avg_write_latency = if writes_delta > 0 {
                    time_writing_delta as f64 / writes_delta as f64
                } else {
                    0.0
                };

                // Utilization: time spent doing I/O as percentage of elapsed time
                let utilization = (time_io_delta as f64 / (elapsed_secs * 1000.0) * 100.0)
                    .min(100.0)
                    .max(0.0);

                rates.insert(
                    device.clone(),
                    DiskIoRates {
                        device: device.clone(),
                        read_bytes_per_sec: read_bytes_delta as f64 / elapsed_secs,
                        write_bytes_per_sec: write_bytes_delta as f64 / elapsed_secs,
                        reads_per_sec: reads_delta as f64 / elapsed_secs,
                        writes_per_sec: writes_delta as f64 / elapsed_secs,
                        avg_read_latency_ms: avg_read_latency,
                        avg_write_latency_ms: avg_write_latency,
                        utilization_percent: utilization,
                    },
                );
            }
        }

        rates
    }
}

impl Analyzer for DiskIoAnalyzer {
    fn name(&self) -> &'static str {
        "disk_io"
    }

    fn collect(&mut self) -> Result<(), AnalyzerError> {
        let current_stats = self.parse_diskstats()?;
        let now = Instant::now();

        let rates = if let Some(prev_time) = self.prev_time {
            let elapsed = now.duration_since(prev_time).as_secs_f64();
            if elapsed > 0.0 {
                self.calculate_rates(&current_stats, elapsed)
            } else {
                HashMap::new()
            }
        } else {
            HashMap::new()
        };

        // Calculate totals
        let total_read: f64 = rates.values().map(|r| r.read_bytes_per_sec).sum();
        let total_write: f64 = rates.values().map(|r| r.write_bytes_per_sec).sum();

        self.data = DiskIoData {
            stats: current_stats.clone(),
            rates,
            total_read_bytes_per_sec: total_read,
            total_write_bytes_per_sec: total_write,
        };

        self.prev_stats = current_stats;
        self.prev_time = Some(now);

        Ok(())
    }

    fn interval(&self) -> Duration {
        self.interval
    }

    fn available(&self) -> bool {
        Path::new("/proc/diskstats").exists()
    }
}

/// Format bytes per second for display
fn format_bytes_rate(bytes_per_sec: f64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;

    if bytes_per_sec >= GB {
        format!("{:.1}G/s", bytes_per_sec / GB)
    } else if bytes_per_sec >= MB {
        format!("{:.1}M/s", bytes_per_sec / MB)
    } else if bytes_per_sec >= KB {
        format!("{:.1}K/s", bytes_per_sec / KB)
    } else {
        format!("{:.0}B/s", bytes_per_sec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes_rate() {
        assert_eq!(format_bytes_rate(500.0), "500B/s");
        assert_eq!(format_bytes_rate(1536.0), "1.5K/s");
        assert_eq!(format_bytes_rate(1_500_000.0), "1.4M/s");
        assert_eq!(format_bytes_rate(1_500_000_000.0), "1.4G/s");
    }

    #[test]
    fn test_disk_io_stats_bytes() {
        let stats = DiskIoStats {
            sectors_read: 1000,
            sectors_written: 2000,
            ..Default::default()
        };
        assert_eq!(stats.read_bytes(), 512_000);
        assert_eq!(stats.write_bytes(), 1_024_000);
    }

    #[test]
    fn test_disk_io_stats_is_partition() {
        let whole_disk = DiskIoStats {
            device: "sda".to_string(),
            ..Default::default()
        };
        assert!(!whole_disk.is_partition());

        let partition = DiskIoStats {
            device: "sda1".to_string(),
            ..Default::default()
        };
        assert!(partition.is_partition());

        let nvme_disk = DiskIoStats {
            device: "nvme0n1".to_string(),
            ..Default::default()
        };
        assert!(!nvme_disk.is_partition());

        let nvme_partition = DiskIoStats {
            device: "nvme0n1p1".to_string(),
            ..Default::default()
        };
        assert!(nvme_partition.is_partition());
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = DiskIoAnalyzer::new();
        assert_eq!(analyzer.name(), "disk_io");
    }

    #[test]
    fn test_analyzer_available() {
        let analyzer = DiskIoAnalyzer::new();
        #[cfg(target_os = "linux")]
        assert!(analyzer.available());
    }

    #[test]
    fn test_analyzer_collect() {
        let mut analyzer = DiskIoAnalyzer::new();
        // First collection establishes baseline
        let result1 = analyzer.collect();
        assert!(result1.is_ok());

        // Second collection calculates rates
        std::thread::sleep(Duration::from_millis(100));
        let result2 = analyzer.collect();
        assert!(result2.is_ok());

        let data = analyzer.data();
        // Should have some disks on Linux
        #[cfg(target_os = "linux")]
        assert!(!data.stats.is_empty());
    }

    #[test]
    fn test_disk_io_rates_display() {
        let rates = DiskIoRates {
            device: "sda".to_string(),
            read_bytes_per_sec: 1_500_000.0,
            write_bytes_per_sec: 500_000.0,
            reads_per_sec: 100.0,
            writes_per_sec: 50.0,
            ..Default::default()
        };
        assert_eq!(rates.read_rate_display(), "1.4M/s");
        assert_eq!(rates.write_rate_display(), "488.3K/s");
        assert!((rates.total_iops() - 150.0).abs() < 0.01);
    }

    #[test]
    fn test_disk_io_stats_default() {
        let stats = DiskIoStats::default();
        assert!(stats.device.is_empty());
        assert_eq!(stats.major, 0);
        assert_eq!(stats.minor, 0);
        assert_eq!(stats.reads_completed, 0);
        assert_eq!(stats.sectors_read, 0);
    }

    #[test]
    fn test_disk_io_rates_default() {
        let rates = DiskIoRates::default();
        assert!(rates.device.is_empty());
        assert_eq!(rates.read_bytes_per_sec, 0.0);
        assert_eq!(rates.write_bytes_per_sec, 0.0);
    }

    #[test]
    fn test_disk_io_data_default() {
        let data = DiskIoData::default();
        assert!(data.stats.is_empty());
        assert!(data.rates.is_empty());
        assert_eq!(data.total_read_bytes_per_sec, 0.0);
        assert_eq!(data.total_write_bytes_per_sec, 0.0);
    }

    #[test]
    fn test_disk_io_data_physical_disks() {
        let mut data = DiskIoData::default();
        data.stats.insert(
            "sda".to_string(),
            DiskIoStats {
                device: "sda".to_string(),
                ..Default::default()
            },
        );
        data.stats.insert(
            "sda1".to_string(),
            DiskIoStats {
                device: "sda1".to_string(),
                ..Default::default()
            },
        );

        let physical: Vec<_> = data.physical_disks().collect();
        assert_eq!(physical.len(), 1);
        assert_eq!(physical[0].0, "sda");
    }

    #[test]
    fn test_disk_io_data_physical_disk_rates() {
        let mut data = DiskIoData::default();
        data.stats.insert(
            "sda".to_string(),
            DiskIoStats {
                device: "sda".to_string(),
                ..Default::default()
            },
        );
        data.stats.insert(
            "sda1".to_string(),
            DiskIoStats {
                device: "sda1".to_string(),
                ..Default::default()
            },
        );
        data.rates.insert(
            "sda".to_string(),
            DiskIoRates {
                device: "sda".to_string(),
                read_bytes_per_sec: 1000.0,
                ..Default::default()
            },
        );
        data.rates.insert(
            "sda1".to_string(),
            DiskIoRates {
                device: "sda1".to_string(),
                read_bytes_per_sec: 500.0,
                ..Default::default()
            },
        );

        let physical: Vec<_> = data.physical_disk_rates().collect();
        assert_eq!(physical.len(), 1);
        assert_eq!(physical[0].0, "sda");
    }

    #[test]
    fn test_disk_io_stats_nvme_variants() {
        // nvme0n1 is a whole disk
        let nvme = DiskIoStats {
            device: "nvme0n1".to_string(),
            ..Default::default()
        };
        assert!(!nvme.is_partition());

        // nvme0n1p2 is a partition
        let nvme_part = DiskIoStats {
            device: "nvme0n1p2".to_string(),
            ..Default::default()
        };
        assert!(nvme_part.is_partition());
    }

    #[test]
    fn test_disk_io_stats_sd_variants() {
        // sdb is a whole disk
        let sd = DiskIoStats {
            device: "sdb".to_string(),
            ..Default::default()
        };
        assert!(!sd.is_partition());

        // sdb3 is a partition
        let sd_part = DiskIoStats {
            device: "sdb3".to_string(),
            ..Default::default()
        };
        assert!(sd_part.is_partition());
    }

    #[test]
    fn test_disk_io_analyzer_default() {
        let analyzer = DiskIoAnalyzer::default();
        assert_eq!(analyzer.name(), "disk_io");
    }

    #[test]
    fn test_disk_io_analyzer_interval() {
        let analyzer = DiskIoAnalyzer::new();
        assert_eq!(analyzer.interval(), Duration::from_secs(1));
    }

    #[test]
    fn test_disk_io_analyzer_data() {
        let analyzer = DiskIoAnalyzer::new();
        let data = analyzer.data();
        assert!(data.stats.is_empty());
    }

    #[test]
    fn test_format_bytes_rate_edge_cases() {
        assert_eq!(format_bytes_rate(0.0), "0B/s");
        assert_eq!(format_bytes_rate(1023.0), "1023B/s");
        assert_eq!(format_bytes_rate(1024.0), "1.0K/s");
        assert_eq!(format_bytes_rate(1048576.0), "1.0M/s");
        assert_eq!(format_bytes_rate(1073741824.0), "1.0G/s");
    }

    #[test]
    fn test_disk_io_rates_latency_values() {
        let rates = DiskIoRates {
            device: "sda".to_string(),
            avg_read_latency_ms: 1.5,
            avg_write_latency_ms: 2.5,
            utilization_percent: 75.0,
            ..Default::default()
        };
        assert!((rates.avg_read_latency_ms - 1.5).abs() < 0.01);
        assert!((rates.avg_write_latency_ms - 2.5).abs() < 0.01);
        assert!((rates.utilization_percent - 75.0).abs() < 0.01);
    }

    #[test]
    fn test_disk_io_stats_extended_fields() {
        let stats = DiskIoStats {
            device: "sda".to_string(),
            discards_completed: Some(100),
            sectors_discarded: Some(50000),
            flush_requests: Some(10),
            ..Default::default()
        };
        assert_eq!(stats.discards_completed, Some(100));
        assert_eq!(stats.sectors_discarded, Some(50000));
        assert_eq!(stats.flush_requests, Some(10));
    }

    #[test]
    fn test_disk_io_stats_clone() {
        let stats = DiskIoStats {
            device: "sda".to_string(),
            sectors_read: 1000,
            sectors_written: 2000,
            ..Default::default()
        };
        let cloned = stats.clone();
        assert_eq!(cloned.device, "sda");
        assert_eq!(cloned.sectors_read, 1000);
    }

    #[test]
    fn test_disk_io_rates_clone() {
        let rates = DiskIoRates {
            device: "sda".to_string(),
            read_bytes_per_sec: 1000.0,
            ..Default::default()
        };
        let cloned = rates.clone();
        assert_eq!(cloned.device, "sda");
        assert_eq!(cloned.read_bytes_per_sec, 1000.0);
    }

    #[test]
    fn test_disk_io_data_clone() {
        let mut data = DiskIoData::default();
        data.total_read_bytes_per_sec = 5000.0;
        let cloned = data.clone();
        assert_eq!(cloned.total_read_bytes_per_sec, 5000.0);
    }
}
