//! Disk metrics collector.
//!
//! Parses `/proc/diskstats` and `/sys/block/` on Linux to collect disk I/O metrics.
//!
//! ## Falsification Criteria
//!
//! - #39: Disk IO matches `iostat` within ±5%
//! - #49: Disk mount points match `df` output

use crate::monitor::error::Result;
use crate::monitor::ring_buffer::RingBuffer;
use crate::monitor::subprocess::run_with_timeout;
use crate::monitor::types::{Collector, MetricValue, Metrics};
use std::collections::HashMap;
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// Reads a file with a timeout to prevent hanging on blocked devices.
///
/// This is critical for /proc files that can block indefinitely if a disk device
/// is hung or unresponsive (e.g., disconnected NFS, failed hardware).
fn read_file_with_timeout(path: &str, timeout: Duration) -> Option<String> {
    let path = path.to_string();
    let (tx, rx) = mpsc::channel();

    std::thread::spawn(move || {
        let result = std::fs::read_to_string(&path);
        // Ignore send errors - receiver may have timed out and dropped
        let _ = tx.send(result);
    });

    match rx.recv_timeout(timeout) {
        Ok(Ok(content)) => Some(content),
        Ok(Err(_)) => None, // File read error
        Err(_) => None,     // Timeout
    }
}

/// Statistics for a single disk device.
#[derive(Debug, Clone, Default)]
pub struct DiskStats {
    /// Device name (e.g., "sda", "nvme0n1").
    pub name: String,
    /// Reads completed.
    pub reads_completed: u64,
    /// Reads merged.
    pub reads_merged: u64,
    /// Sectors read.
    pub sectors_read: u64,
    /// Time spent reading (ms).
    pub read_time_ms: u64,
    /// Writes completed.
    pub writes_completed: u64,
    /// Writes merged.
    pub writes_merged: u64,
    /// Sectors written.
    pub sectors_written: u64,
    /// Time spent writing (ms).
    pub write_time_ms: u64,
    /// IO currently in progress.
    pub io_in_progress: u64,
    /// Time spent doing IO (ms).
    pub io_time_ms: u64,
    /// Weighted time spent doing IO (ms).
    pub weighted_io_time_ms: u64,
}

/// Calculated disk I/O rates.
#[derive(Debug, Clone, Default)]
pub struct DiskIoRates {
    /// Device name.
    pub name: String,
    /// Read bytes per second.
    pub read_bytes_per_sec: f64,
    /// Write bytes per second.
    pub write_bytes_per_sec: f64,
    /// Read operations per second.
    pub read_iops: f64,
    /// Write operations per second.
    pub write_iops: f64,
    /// IO utilization percentage.
    pub io_utilization: f64,
}

/// Information about a mounted filesystem.
#[derive(Debug, Clone)]
pub struct MountInfo {
    /// Device path (e.g., "/dev/sda1").
    pub device: String,
    /// Mount point (e.g., "/home").
    pub mount_point: String,
    /// Filesystem type (e.g., "ext4").
    pub fs_type: String,
    /// Total size in bytes.
    pub total_bytes: u64,
    /// Used bytes.
    pub used_bytes: u64,
    /// Available bytes.
    pub available_bytes: u64,
}

impl MountInfo {
    /// Returns the usage percentage.
    #[must_use]
    pub fn usage_percent(&self) -> f64 {
        batuta_common::math::usage_percent(self.used_bytes, self.total_bytes)
    }
}

/// Collector for disk metrics.
#[derive(Debug)]
pub struct DiskCollector {
    /// Previous stats for delta calculation.
    prev_stats: HashMap<String, DiskStats>,
    /// Previous collection time.
    prev_time: Option<Instant>,
    /// Calculated IO rates.
    rates: HashMap<String, DiskIoRates>,
    /// History of total read throughput.
    read_history: RingBuffer<f64>,
    /// History of total write throughput.
    write_history: RingBuffer<f64>,
    /// Mounted filesystems.
    mounts: Vec<MountInfo>,
    /// Sector size (typically 512 bytes).
    sector_size: u64,
}

impl DiskCollector {
    /// Creates a new disk collector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            prev_stats: HashMap::new(),
            prev_time: None,
            rates: HashMap::new(),
            read_history: RingBuffer::new(300),
            write_history: RingBuffer::new(300),
            mounts: Vec::new(),
            sector_size: 512,
        }
    }

    /// Reads disk statistics from /proc/diskstats.
    #[cfg(target_os = "linux")]
    fn read_diskstats(&self) -> Result<HashMap<String, DiskStats>> {
        // Use timeout to prevent hanging on blocked/hung disk devices.
        // /proc/diskstats can block indefinitely if a disk device is unresponsive.
        let content = match read_file_with_timeout("/proc/diskstats", Duration::from_secs(2)) {
            Some(c) => c,
            None => {
                // Timeout or read error - return empty stats for graceful degradation
                return Ok(HashMap::new());
            }
        };

        let mut stats = HashMap::new();

        for line in content.lines() {
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() < 14 {
                continue;
            }

            let name = fields[2].to_string();

            // Skip virtual/pseudo block devices
            if name.starts_with("loop")
                || name.starts_with("ram")
                || name.starts_with("zram")
                || name.starts_with("dm-")
            {
                continue;
            }

            // Skip partitions (only collect whole disks and nvme namespaces)
            // Disks: sda, nvme0n1, vda, etc.
            // Partitions: sda1, nvme0n1p1, vda1, etc.
            let is_partition = name.chars().last().is_some_and(|c| c.is_ascii_digit())
                && !name.contains("nvme")
                || (name.contains("nvme") && name.contains('p'));

            // Include both disks and partitions for mount point matching
            let disk_stats = DiskStats {
                name: name.clone(),
                reads_completed: fields[3].parse().unwrap_or(0),
                reads_merged: fields[4].parse().unwrap_or(0),
                sectors_read: fields[5].parse().unwrap_or(0),
                read_time_ms: fields[6].parse().unwrap_or(0),
                writes_completed: fields[7].parse().unwrap_or(0),
                writes_merged: fields[8].parse().unwrap_or(0),
                sectors_written: fields[9].parse().unwrap_or(0),
                write_time_ms: fields[10].parse().unwrap_or(0),
                io_in_progress: fields[11].parse().unwrap_or(0),
                io_time_ms: fields[12].parse().unwrap_or(0),
                weighted_io_time_ms: fields[13].parse().unwrap_or(0),
            };

            // Only include if it has any IO activity (filters out loop devices, etc.)
            if !is_partition || disk_stats.reads_completed > 0 || disk_stats.writes_completed > 0 {
                stats.insert(name, disk_stats);
            }
        }

        Ok(stats)
    }

    #[cfg(target_os = "macos")]
    fn read_diskstats(&self) -> Result<HashMap<String, DiskStats>> {
        // Use iostat -d to get disk I/O stats on macOS
        // Wrap in timeout to prevent hangs (iostat can block on slow I/O)
        let result = run_with_timeout("iostat", &["-d", "-c", "1"], Duration::from_secs(5));

        let content = match result.stdout_string() {
            Some(s) => s,
            None => {
                // Timeout or error - return empty stats rather than hanging
                return Ok(HashMap::new());
            }
        };
        let mut stats = HashMap::new();

        // Parse iostat -d output:
        //           disk0
        // KB/t  tps  MB/s
        // 45.23   12  0.53

        let lines: Vec<&str> = content.lines().collect();
        if lines.len() < 3 {
            return Ok(stats);
        }

        // First line has disk names
        let disk_names: Vec<&str> = lines[0].split_whitespace().collect();

        // Third line has stats (KB/t, tps, MB/s per disk)
        let values: Vec<&str> = lines[2].split_whitespace().collect();

        // Each disk has 3 values: KB/t, tps, MB/s
        for (i, name) in disk_names.iter().enumerate() {
            let base_idx = i * 3;
            if base_idx + 2 >= values.len() {
                continue;
            }

            let _kb_per_transfer: f64 = values[base_idx].parse().unwrap_or(0.0);
            let transfers_per_sec: f64 = values[base_idx + 1].parse().unwrap_or(0.0);
            let mb_per_sec: f64 = values[base_idx + 2].parse().unwrap_or(0.0);

            // Convert to cumulative-style stats for rate calculation
            // Since iostat gives us rates directly, we'll store current timestamp-based values
            let bytes_per_sec = mb_per_sec * 1024.0 * 1024.0;
            let sectors = (bytes_per_sec / self.sector_size as f64) as u64;

            stats.insert(
                name.to_string(),
                DiskStats {
                    name: name.to_string(),
                    reads_completed: (transfers_per_sec / 2.0) as u64, // Approximate split
                    reads_merged: 0,
                    sectors_read: sectors / 2, // Approximate split
                    read_time_ms: 0,
                    writes_completed: (transfers_per_sec / 2.0) as u64,
                    writes_merged: 0,
                    sectors_written: sectors / 2,
                    write_time_ms: 0,
                    io_in_progress: 0,
                    io_time_ms: 0,
                    weighted_io_time_ms: 0,
                },
            );
        }

        Ok(stats)
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    fn read_diskstats(&self) -> Result<HashMap<String, DiskStats>> {
        Ok(HashMap::new())
    }

    /// Reads mount information from /proc/mounts and statvfs.
    #[cfg(target_os = "linux")]
    fn read_mounts(&self) -> Result<Vec<MountInfo>> {
        // Use timeout for /proc/mounts as well (can also hang on problem devices)
        let content = match read_file_with_timeout("/proc/mounts", Duration::from_secs(2)) {
            Some(c) => c,
            None => {
                // Timeout or read error - return empty mounts for graceful degradation
                return Ok(Vec::new());
            }
        };

        let mut mounts = Vec::new();

        for line in content.lines() {
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() < 3 {
                continue;
            }

            let device = fields[0];
            let mount_point = fields[1];
            let fs_type = fields[2];

            // Skip virtual filesystems (device doesn't start with /)
            if !device.starts_with('/') {
                continue;
            }

            // Skip special mount points
            if mount_point.starts_with("/sys")
                || mount_point.starts_with("/proc")
                || mount_point.starts_with("/dev")
                || mount_point.starts_with("/run")
                || mount_point.starts_with("/snap")
            {
                continue;
            }

            // Skip network/remote filesystems that can hang df
            // These include: NFS, CIFS/SMB, autofs, fuse-based network mounts
            if fs_type == "nfs"
                || fs_type == "nfs4"
                || fs_type == "cifs"
                || fs_type == "smbfs"
                || fs_type == "autofs"
                || fs_type == "fuse.sshfs"
                || fs_type == "fuse.rclone"
                || fs_type == "fuse.gvfsd-fuse"
                || fs_type == "9p"
            {
                continue;
            }

            // Get filesystem stats using statvfs
            if let Some(stats) = Self::statvfs(mount_point) {
                mounts.push(MountInfo {
                    device: device.to_string(),
                    mount_point: mount_point.to_string(),
                    fs_type: fs_type.to_string(),
                    total_bytes: stats.0,
                    used_bytes: stats.1,
                    available_bytes: stats.2,
                });
            }
        }

        Ok(mounts)
    }

    #[cfg(target_os = "macos")]
    fn read_mounts(&self) -> Result<Vec<MountInfo>> {
        // Use df to get mount information on macOS
        // Wrap in timeout to prevent hangs (df can block on NFS/network mounts)
        let result = run_with_timeout("df", &["-k"], Duration::from_secs(5));

        let content = match result.stdout_string() {
            Some(s) => s,
            None => {
                // Timeout or error - return empty mounts rather than hanging
                return Ok(Vec::new());
            }
        };
        let mut mounts = Vec::new();

        // Parse df -k output:
        // Filesystem    1024-blocks      Used Available Capacity  Mounted on
        // /dev/disk1s1   976490576 123456789 876543210    13%    /

        for line in content.lines().skip(1) {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 6 {
                continue;
            }

            let device = parts[0];

            // Skip virtual filesystems
            if !device.starts_with("/dev") {
                continue;
            }

            // Mount point is the last element (may contain spaces, but we'll take the last)
            let mount_point = parts[parts.len() - 1];

            // Skip system volumes
            if mount_point.starts_with("/System")
                || mount_point.starts_with("/private/var/vm")
                || mount_point.contains("/Preboot")
                || mount_point.contains("/Recovery")
                || mount_point.contains("/Update")
            {
                continue;
            }

            let total_kb: u64 = parts[1].parse().unwrap_or(0);
            let used_kb: u64 = parts[2].parse().unwrap_or(0);
            let avail_kb: u64 = parts[3].parse().unwrap_or(0);

            // Determine filesystem type (macOS typically uses APFS)
            let fs_type =
                if device.contains("disk") { "apfs".to_string() } else { "unknown".to_string() };

            mounts.push(MountInfo {
                device: device.to_string(),
                mount_point: mount_point.to_string(),
                fs_type,
                total_bytes: total_kb * 1024,
                used_bytes: used_kb * 1024,
                available_bytes: avail_kb * 1024,
            });
        }

        Ok(mounts)
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    fn read_mounts(&self) -> Result<Vec<MountInfo>> {
        Ok(Vec::new())
    }

    /// Gets filesystem statistics by parsing df output.
    ///
    /// This is a simpler approach that doesn't require libc.
    #[cfg(target_os = "linux")]
    fn statvfs(path: &str) -> Option<(u64, u64, u64)> {
        // Try reading from /proc/mounts and statfs via df
        // For now, use a simple approximation from /proc/diskstats
        // Full implementation would use nix::sys::statvfs or libc

        // Read from df if available (simpler than FFI)
        // Wrap in timeout to prevent hangs on NFS/network mounts
        let result = run_with_timeout(
            "df",
            &["--output=size,used,avail", "-B1", path],
            Duration::from_secs(2),
        );

        let stdout = result.stdout_string()?;

        if !result.is_success() {
            return None;
        }

        let lines: Vec<&str> = stdout.lines().collect();

        if lines.len() < 2 {
            return None;
        }

        let values: Vec<u64> = lines[1].split_whitespace().filter_map(|s| s.parse().ok()).collect();

        if values.len() < 3 {
            return None;
        }

        Some((values[0], values[1], values[2]))
    }

    #[cfg(target_os = "macos")]
    fn statvfs(path: &str) -> Option<(u64, u64, u64)> {
        // Use df on macOS as well
        // Wrap in timeout to prevent hangs on NFS/network mounts
        let result = run_with_timeout("df", &["-k", path], Duration::from_secs(5));

        let stdout = result.stdout_string()?;

        if !result.is_success() {
            return None;
        }

        let lines: Vec<&str> = stdout.lines().collect();

        if lines.len() < 2 {
            return None;
        }

        let values: Vec<&str> = lines[1].split_whitespace().collect();
        if values.len() < 4 {
            return None;
        }

        let total: u64 = values[1].parse().ok()?;
        let used: u64 = values[2].parse().ok()?;
        let avail: u64 = values[3].parse().ok()?;

        // Convert from KB to bytes
        Some((total * 1024, used * 1024, avail * 1024))
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    fn statvfs(_path: &str) -> Option<(u64, u64, u64)> {
        None
    }

    /// Calculates IO rates from delta between samples.
    fn calculate_rates(
        &self,
        current: &HashMap<String, DiskStats>,
        elapsed_secs: f64,
    ) -> HashMap<String, DiskIoRates> {
        let mut rates = HashMap::new();

        for (name, curr) in current {
            if let Some(prev) = self.prev_stats.get(name) {
                let read_sectors = curr.sectors_read.saturating_sub(prev.sectors_read);
                let write_sectors = curr.sectors_written.saturating_sub(prev.sectors_written);
                let read_ops = curr.reads_completed.saturating_sub(prev.reads_completed);
                let write_ops = curr.writes_completed.saturating_sub(prev.writes_completed);
                let io_time = curr.io_time_ms.saturating_sub(prev.io_time_ms);

                let io_rates = DiskIoRates {
                    name: name.clone(),
                    read_bytes_per_sec: (read_sectors * self.sector_size) as f64 / elapsed_secs,
                    write_bytes_per_sec: (write_sectors * self.sector_size) as f64 / elapsed_secs,
                    read_iops: read_ops as f64 / elapsed_secs,
                    write_iops: write_ops as f64 / elapsed_secs,
                    // IO utilization: time spent doing IO / total time
                    io_utilization: (io_time as f64 / (elapsed_secs * 1000.0) * 100.0).min(100.0),
                };

                rates.insert(name.clone(), io_rates);
            }
        }

        rates
    }

    /// Returns the calculated IO rates.
    #[must_use]
    pub fn rates(&self) -> &HashMap<String, DiskIoRates> {
        &self.rates
    }

    /// Returns mounted filesystems.
    #[must_use]
    pub fn mounts(&self) -> &[MountInfo] {
        &self.mounts
    }

    /// Returns total read throughput history (normalized 0-1).
    #[must_use]
    pub fn read_history(&self) -> &RingBuffer<f64> {
        &self.read_history
    }

    /// Returns total write throughput history (normalized 0-1).
    #[must_use]
    pub fn write_history(&self) -> &RingBuffer<f64> {
        &self.write_history
    }
}

impl Default for DiskCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl Collector for DiskCollector {
    fn id(&self) -> &'static str {
        "disk"
    }

    fn collect(&mut self) -> Result<Metrics> {
        let now = Instant::now();
        let current_stats = self.read_diskstats()?;

        // Calculate rates if we have previous data
        if let Some(prev_time) = self.prev_time {
            let elapsed = now.duration_since(prev_time);
            let elapsed_secs = elapsed.as_secs_f64();

            if elapsed_secs > 0.0 {
                self.rates = self.calculate_rates(&current_stats, elapsed_secs);

                // Calculate total throughput for history
                let total_read: f64 = self.rates.values().map(|r| r.read_bytes_per_sec).sum();
                let total_write: f64 = self.rates.values().map(|r| r.write_bytes_per_sec).sum();

                // Normalize to 0-1 range (assuming 1 GB/s max for visualization)
                let max_throughput = 1_000_000_000.0_f64;
                self.read_history.push((total_read / max_throughput).min(1.0));
                self.write_history.push((total_write / max_throughput).min(1.0));
            }
        }

        // Update previous state
        self.prev_stats = current_stats;
        self.prev_time = Some(now);

        // Read mount information
        self.mounts = self.read_mounts().unwrap_or_default();

        // Build metrics
        let mut metrics = Metrics::new();

        // Total read/write throughput
        let total_read: f64 = self.rates.values().map(|r| r.read_bytes_per_sec).sum();
        let total_write: f64 = self.rates.values().map(|r| r.write_bytes_per_sec).sum();

        metrics.insert("disk.read_bytes_per_sec", MetricValue::Gauge(total_read));
        metrics.insert("disk.write_bytes_per_sec", MetricValue::Gauge(total_write));

        // Total IOPS
        let total_read_iops: f64 = self.rates.values().map(|r| r.read_iops).sum();
        let total_write_iops: f64 = self.rates.values().map(|r| r.write_iops).sum();

        metrics.insert("disk.read_iops", MetricValue::Gauge(total_read_iops));
        metrics.insert("disk.write_iops", MetricValue::Gauge(total_write_iops));

        // Mount count
        metrics.insert("disk.mount_count", MetricValue::Counter(self.mounts.len() as u64));

        Ok(metrics)
    }

    fn is_available(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            std::path::Path::new("/proc/diskstats").exists()
        }
        #[cfg(target_os = "macos")]
        {
            true // macOS uses iostat and df which are always available
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            false
        }
    }

    fn interval_hint(&self) -> Duration {
        Duration::from_millis(1000)
    }

    fn display_name(&self) -> &'static str {
        "Disk"
    }
}

// ============================================================================
// Tests (TDD - Written First)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Unit Tests
    // ========================================================================

    #[test]
    fn test_disk_collector_new() {
        let collector = DiskCollector::new();
        assert!(collector.prev_stats.is_empty());
        assert!(collector.prev_time.is_none());
        assert_eq!(collector.sector_size, 512);
    }

    #[test]
    fn test_disk_collector_default() {
        let collector = DiskCollector::default();
        assert!(collector.prev_stats.is_empty());
    }

    #[test]
    fn test_mount_info_usage_percent() {
        let mount = MountInfo {
            device: "/dev/sda1".to_string(),
            mount_point: "/".to_string(),
            fs_type: "ext4".to_string(),
            total_bytes: 100_000_000_000,
            used_bytes: 50_000_000_000,
            available_bytes: 50_000_000_000,
        };

        let usage = mount.usage_percent();
        assert!((usage - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_mount_info_usage_percent_zero_total() {
        let mount = MountInfo {
            device: "/dev/sda1".to_string(),
            mount_point: "/".to_string(),
            fs_type: "ext4".to_string(),
            total_bytes: 0,
            used_bytes: 0,
            available_bytes: 0,
        };

        assert_eq!(mount.usage_percent(), 0.0);
    }

    #[test]
    fn test_disk_stats_default() {
        let stats = DiskStats::default();
        assert!(stats.name.is_empty());
        assert_eq!(stats.reads_completed, 0);
        assert_eq!(stats.writes_completed, 0);
    }

    #[test]
    fn test_disk_io_rates_default() {
        let rates = DiskIoRates::default();
        assert!(rates.name.is_empty());
        assert_eq!(rates.read_bytes_per_sec, 0.0);
        assert_eq!(rates.write_bytes_per_sec, 0.0);
    }

    #[test]
    fn test_disk_collector_interval() {
        let collector = DiskCollector::new();
        assert_eq!(collector.interval_hint(), Duration::from_millis(1000));
    }

    #[test]
    fn test_disk_collector_display_name() {
        let collector = DiskCollector::new();
        assert_eq!(collector.display_name(), "Disk");
    }

    #[test]
    fn test_disk_collector_id() {
        let collector = DiskCollector::new();
        assert_eq!(collector.id(), "disk");
    }

    #[test]
    fn test_calculate_rates() {
        let mut collector = DiskCollector::new();

        // Set up previous stats
        collector.prev_stats.insert(
            "sda".to_string(),
            DiskStats {
                name: "sda".to_string(),
                sectors_read: 1000,
                sectors_written: 500,
                reads_completed: 100,
                writes_completed: 50,
                io_time_ms: 1000,
                ..Default::default()
            },
        );

        // Current stats (after 1 second)
        let mut current = HashMap::new();
        current.insert(
            "sda".to_string(),
            DiskStats {
                name: "sda".to_string(),
                sectors_read: 2000,    // 1000 sectors read
                sectors_written: 1000, // 500 sectors written
                reads_completed: 200,  // 100 reads
                writes_completed: 100, // 50 writes
                io_time_ms: 1500,      // 500ms IO time
                ..Default::default()
            },
        );

        let rates = collector.calculate_rates(&current, 1.0);

        let sda_rates = rates.get("sda").expect("Should have sda rates");

        // 1000 sectors * 512 bytes / 1 second = 512000 bytes/sec
        assert!((sda_rates.read_bytes_per_sec - 512000.0).abs() < 1.0);
        assert!((sda_rates.write_bytes_per_sec - 256000.0).abs() < 1.0);
        assert!((sda_rates.read_iops - 100.0).abs() < 0.01);
        assert!((sda_rates.write_iops - 50.0).abs() < 0.01);
        // 500ms IO time / 1000ms total = 50%
        assert!((sda_rates.io_utilization - 50.0).abs() < 0.01);
    }

    // ========================================================================
    // Linux-specific Tests
    // ========================================================================

    #[cfg(target_os = "linux")]
    #[test]
    fn test_disk_collector_is_available() {
        let collector = DiskCollector::new();
        assert!(collector.is_available());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_disk_collector_collect() {
        let mut collector = DiskCollector::new();

        // First collection (no rates yet)
        let result = collector.collect();
        assert!(result.is_ok());

        // Wait a bit and collect again
        std::thread::sleep(Duration::from_millis(100));

        let result = collector.collect();
        assert!(result.is_ok());

        let metrics = result.expect("collect should succeed");
        assert!(metrics.get_gauge("disk.read_bytes_per_sec").is_some());
        assert!(metrics.get_gauge("disk.write_bytes_per_sec").is_some());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_disk_collector_mounts() {
        let mut collector = DiskCollector::new();
        let _ = collector.collect();

        let mounts = collector.mounts();
        // Should have at least root mount on Linux
        assert!(!mounts.is_empty(), "Should find at least one mount point");

        // Root should be present
        let has_root = mounts.iter().any(|m| m.mount_point == "/");
        assert!(has_root, "Root mount should be present");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_disk_collector_history() {
        let mut collector = DiskCollector::new();

        // Collect twice to generate history
        let _ = collector.collect();
        std::thread::sleep(Duration::from_millis(50));
        let _ = collector.collect();

        // History should have at least one entry
        assert!(!collector.read_history().is_empty());
        assert!(!collector.write_history().is_empty());
    }
}
