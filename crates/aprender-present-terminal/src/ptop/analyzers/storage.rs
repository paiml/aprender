//! Storage Analyzer
//!
//! Provides filesystem and mount point information by parsing `/proc/mounts`
//! and using statvfs for capacity statistics.

#![allow(clippy::uninlined_format_args)]

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Duration;

use super::{Analyzer, AnalyzerError};

/// Information about a mounted filesystem
#[derive(Debug, Clone, Default)]
pub struct MountInfo {
    /// Device name (e.g., "/dev/sda1", "tmpfs")
    pub device: String,
    /// Mount point path
    pub mount_point: String,
    /// Filesystem type (e.g., "ext4", "btrfs", "tmpfs")
    pub fs_type: String,
    /// Mount options
    pub options: Vec<String>,
    /// Total size in bytes
    pub total: u64,
    /// Used bytes
    pub used: u64,
    /// Available bytes
    pub available: u64,
    /// Inodes total
    pub inodes_total: u64,
    /// Inodes used
    pub inodes_used: u64,
    /// Inodes free
    pub inodes_free: u64,
}

impl MountInfo {
    /// Usage percentage
    pub fn usage_percent(&self) -> f64 {
        if self.total > 0 {
            self.used as f64 / self.total as f64 * 100.0
        } else {
            0.0
        }
    }

    /// Inode usage percentage
    pub fn inode_usage_percent(&self) -> f64 {
        if self.inodes_total > 0 {
            self.inodes_used as f64 / self.inodes_total as f64 * 100.0
        } else {
            0.0
        }
    }

    /// Is this a real filesystem (not virtual)?
    pub fn is_real_fs(&self) -> bool {
        // Exclude virtual filesystems
        !matches!(
            self.fs_type.as_str(),
            "proc"
                | "sysfs"
                | "devtmpfs"
                | "devpts"
                | "securityfs"
                | "cgroup"
                | "cgroup2"
                | "pstore"
                | "debugfs"
                | "tracefs"
                | "hugetlbfs"
                | "mqueue"
                | "fusectl"
                | "configfs"
                | "bpf"
                | "efivarfs"
                | "autofs"
                | "rpc_pipefs"
                | "overlay"
        ) && !self.mount_point.starts_with("/sys")
            && !self.mount_point.starts_with("/proc")
            && !self.mount_point.starts_with("/run/user")
            && !self.mount_point.starts_with("/snap")
    }

    /// Is this a network filesystem?
    pub fn is_network_fs(&self) -> bool {
        matches!(
            self.fs_type.as_str(),
            "nfs" | "nfs4" | "cifs" | "smb" | "smbfs" | "sshfs" | "fuse.sshfs"
        )
    }

    /// Is tmpfs?
    pub fn is_tmpfs(&self) -> bool {
        self.fs_type == "tmpfs"
    }

    /// Format total size for display
    pub fn total_display(&self) -> String {
        format_size(self.total)
    }

    /// Format used size for display
    pub fn used_display(&self) -> String {
        format_size(self.used)
    }

    /// Format available size for display
    pub fn available_display(&self) -> String {
        format_size(self.available)
    }
}

/// Storage data
#[derive(Debug, Clone, Default)]
pub struct StorageData {
    /// All mounts
    pub mounts: Vec<MountInfo>,
    /// Mounts by mount point
    pub by_mount_point: HashMap<String, MountInfo>,
    /// Total storage capacity (real filesystems only)
    pub total_capacity: u64,
    /// Total used (real filesystems only)
    pub total_used: u64,
}

impl StorageData {
    /// Get real filesystems only
    pub fn real_filesystems(&self) -> impl Iterator<Item = &MountInfo> {
        self.mounts.iter().filter(|m| m.is_real_fs())
    }

    /// Get filesystem by mount point
    pub fn get_mount(&self, path: &str) -> Option<&MountInfo> {
        self.by_mount_point.get(path)
    }

    /// Overall usage percentage
    pub fn overall_usage_percent(&self) -> f64 {
        if self.total_capacity > 0 {
            self.total_used as f64 / self.total_capacity as f64 * 100.0
        } else {
            0.0
        }
    }
}

/// Analyzer for storage/filesystem information
pub struct StorageAnalyzer {
    data: StorageData,
    interval: Duration,
}

impl Default for StorageAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl StorageAnalyzer {
    /// Create a new storage analyzer
    pub fn new() -> Self {
        Self {
            data: StorageData::default(),
            interval: Duration::from_secs(30), // Filesystems don't change often
        }
    }

    /// Get the current data
    pub fn data(&self) -> &StorageData {
        &self.data
    }

    /// Parse /proc/mounts
    fn parse_mounts(&self) -> Result<Vec<MountInfo>, AnalyzerError> {
        let contents = fs::read_to_string("/proc/mounts")
            .map_err(|e| AnalyzerError::IoError(format!("Failed to read /proc/mounts: {}", e)))?;

        let mut mounts = Vec::new();

        for line in contents.lines() {
            if let Some(mut mount) = self.parse_mounts_line(line) {
                // Get capacity info via statvfs
                self.get_fs_stats(&mut mount);
                mounts.push(mount);
            }
        }

        Ok(mounts)
    }

    /// Parse a single line from /proc/mounts
    fn parse_mounts_line(&self, line: &str) -> Option<MountInfo> {
        // Format: device mount_point fs_type options dump pass
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 4 {
            return None;
        }

        let device = parts[0].to_string();
        let mount_point = unescape_mount_point(parts[1]);
        let fs_type = parts[2].to_string();
        let options: Vec<String> = parts[3].split(',').map(String::from).collect();

        Some(MountInfo {
            device,
            mount_point,
            fs_type,
            options,
            ..Default::default()
        })
    }

    /// Get filesystem statistics by parsing /proc/self/mountinfo and df output
    fn get_fs_stats(&self, mount: &mut MountInfo) {
        // Parse /proc/[pid]/statfs for this mount point by reading from df output
        // This avoids unsafe code while still getting capacity info
        if let Ok(output) = std::process::Command::new("df")
            .arg("-B1") // bytes
            .arg("--output=size,used,avail")
            .arg(&mount.mount_point)
            .output()
        {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                // Skip header line
                if let Some(line) = stdout.lines().nth(1) {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 3 {
                        mount.total = parts[0].parse().unwrap_or(0);
                        mount.used = parts[1].parse().unwrap_or(0);
                        mount.available = parts[2].parse().unwrap_or(0);
                    }
                }
            }
        }

        // Get inode info via df -i
        if let Ok(output) = std::process::Command::new("df")
            .arg("-i")
            .arg("--output=itotal,iused,iavail")
            .arg(&mount.mount_point)
            .output()
        {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if let Some(line) = stdout.lines().nth(1) {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 3 {
                        mount.inodes_total = parts[0].parse().unwrap_or(0);
                        mount.inodes_used = parts[1].parse().unwrap_or(0);
                        mount.inodes_free = parts[2].parse().unwrap_or(0);
                    }
                }
            }
        }
    }
}

impl Analyzer for StorageAnalyzer {
    fn name(&self) -> &'static str {
        "storage"
    }

    fn collect(&mut self) -> Result<(), AnalyzerError> {
        let mounts = self.parse_mounts()?;

        let mut by_mount_point = HashMap::new();
        for mount in &mounts {
            by_mount_point.insert(mount.mount_point.clone(), mount.clone());
        }

        // Calculate totals for real filesystems
        let (total_capacity, total_used) = mounts
            .iter()
            .filter(|m| m.is_real_fs() && !m.is_tmpfs())
            .fold((0u64, 0u64), |(cap, used), m| {
                (cap + m.total, used + m.used)
            });

        self.data = StorageData {
            mounts,
            by_mount_point,
            total_capacity,
            total_used,
        };

        Ok(())
    }

    fn interval(&self) -> Duration {
        self.interval
    }

    fn available(&self) -> bool {
        Path::new("/proc/mounts").exists()
    }
}

/// Unescape special characters in mount point paths
fn unescape_mount_point(s: &str) -> String {
    // /proc/mounts escapes special chars like space (\040)
    s.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
}

/// Format size for display
fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    const TB: u64 = GB * 1024;

    if bytes >= TB {
        format!("{:.1}T", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.1}G", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1}M", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1}K", bytes as f64 / KB as f64)
    } else {
        format!("{}B", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mount_info_usage() {
        let mount = MountInfo {
            device: "/dev/sda1".to_string(),
            mount_point: "/".to_string(),
            fs_type: "ext4".to_string(),
            total: 100 * 1024 * 1024 * 1024, // 100GB
            used: 40 * 1024 * 1024 * 1024,   // 40GB
            available: 55 * 1024 * 1024 * 1024,
            inodes_total: 1_000_000,
            inodes_used: 250_000,
            inodes_free: 750_000,
            ..Default::default()
        };

        assert!((mount.usage_percent() - 40.0).abs() < 0.1);
        assert!((mount.inode_usage_percent() - 25.0).abs() < 0.1);
    }

    #[test]
    fn test_mount_info_usage_zero_total() {
        let mount = MountInfo {
            total: 0,
            used: 0,
            inodes_total: 0,
            inodes_used: 0,
            ..Default::default()
        };
        assert!((mount.usage_percent() - 0.0).abs() < f64::EPSILON);
        assert!((mount.inode_usage_percent() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mount_info_is_real_fs() {
        let ext4 = MountInfo {
            fs_type: "ext4".to_string(),
            mount_point: "/".to_string(),
            ..Default::default()
        };
        assert!(ext4.is_real_fs());

        let proc = MountInfo {
            fs_type: "proc".to_string(),
            mount_point: "/proc".to_string(),
            ..Default::default()
        };
        assert!(!proc.is_real_fs());

        let sysfs = MountInfo {
            fs_type: "sysfs".to_string(),
            mount_point: "/sys".to_string(),
            ..Default::default()
        };
        assert!(!sysfs.is_real_fs());
    }

    #[test]
    fn test_mount_info_is_real_fs_all_virtual_types() {
        // Test all virtual fs types in the match
        let virtual_types = [
            "proc",
            "sysfs",
            "devtmpfs",
            "devpts",
            "securityfs",
            "cgroup",
            "cgroup2",
            "pstore",
            "debugfs",
            "tracefs",
            "hugetlbfs",
            "mqueue",
            "fusectl",
            "configfs",
            "bpf",
            "efivarfs",
            "autofs",
            "rpc_pipefs",
            "overlay",
        ];
        for fs in virtual_types {
            let mount = MountInfo {
                fs_type: fs.to_string(),
                mount_point: "/some/path".to_string(),
                ..Default::default()
            };
            assert!(!mount.is_real_fs(), "{} should be virtual", fs);
        }
    }

    #[test]
    fn test_mount_info_is_real_fs_mount_point_exclusions() {
        // Test mount point exclusions
        let ext4_sys = MountInfo {
            fs_type: "ext4".to_string(),
            mount_point: "/sys/fs/cgroup".to_string(),
            ..Default::default()
        };
        assert!(!ext4_sys.is_real_fs());

        let ext4_proc = MountInfo {
            fs_type: "ext4".to_string(),
            mount_point: "/proc/sys".to_string(),
            ..Default::default()
        };
        assert!(!ext4_proc.is_real_fs());

        let ext4_run_user = MountInfo {
            fs_type: "ext4".to_string(),
            mount_point: "/run/user/1000".to_string(),
            ..Default::default()
        };
        assert!(!ext4_run_user.is_real_fs());

        let ext4_snap = MountInfo {
            fs_type: "ext4".to_string(),
            mount_point: "/snap/core/12345".to_string(),
            ..Default::default()
        };
        assert!(!ext4_snap.is_real_fs());
    }

    #[test]
    fn test_mount_info_is_network_fs() {
        let nfs = MountInfo {
            fs_type: "nfs4".to_string(),
            ..Default::default()
        };
        assert!(nfs.is_network_fs());

        let ext4 = MountInfo {
            fs_type: "ext4".to_string(),
            ..Default::default()
        };
        assert!(!ext4.is_network_fs());
    }

    #[test]
    fn test_mount_info_is_network_fs_all_types() {
        let network_types = ["nfs", "nfs4", "cifs", "smb", "smbfs", "sshfs", "fuse.sshfs"];
        for fs in network_types {
            let mount = MountInfo {
                fs_type: fs.to_string(),
                ..Default::default()
            };
            assert!(mount.is_network_fs(), "{} should be network fs", fs);
        }
    }

    #[test]
    fn test_mount_info_is_tmpfs() {
        let tmpfs = MountInfo {
            fs_type: "tmpfs".to_string(),
            ..Default::default()
        };
        assert!(tmpfs.is_tmpfs());

        let ext4 = MountInfo {
            fs_type: "ext4".to_string(),
            ..Default::default()
        };
        assert!(!ext4.is_tmpfs());
    }

    #[test]
    fn test_mount_info_display_methods() {
        let mount = MountInfo {
            total: 1099511627776,    // 1TB
            used: 536870912000,      // ~500GB
            available: 562949953536, // ~524GB
            ..Default::default()
        };
        assert_eq!(mount.total_display(), "1.0T");
        assert_eq!(mount.used_display(), "500.0G");
        assert!(mount.available_display().contains("G"));
    }

    #[test]
    fn test_unescape_mount_point() {
        assert_eq!(unescape_mount_point("/mnt/My\\040Drive"), "/mnt/My Drive");
        assert_eq!(unescape_mount_point("/normal/path"), "/normal/path");
    }

    #[test]
    fn test_unescape_mount_point_all_escapes() {
        // Test tab escape
        assert_eq!(unescape_mount_point("/mnt/tab\\011here"), "/mnt/tab\there");
        // Test newline escape
        assert_eq!(unescape_mount_point("/mnt/new\\012line"), "/mnt/new\nline");
        // Test backslash escape
        assert_eq!(
            unescape_mount_point("/mnt/back\\134slash"),
            "/mnt/back\\slash"
        );
        // Test multiple escapes
        assert_eq!(
            unescape_mount_point("/mnt/My\\040Drive\\040With\\040Spaces"),
            "/mnt/My Drive With Spaces"
        );
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(512), "512B");
        assert_eq!(format_size(1024), "1.0K");
        assert_eq!(format_size(1073741824), "1.0G");
        assert_eq!(format_size(1099511627776), "1.0T");
    }

    #[test]
    fn test_format_size_all_ranges() {
        // Bytes
        assert_eq!(format_size(0), "0B");
        assert_eq!(format_size(1023), "1023B");
        // KB
        assert_eq!(format_size(1024), "1.0K");
        assert_eq!(format_size(1024 * 500), "500.0K");
        assert_eq!(format_size(1024 * 1024 - 1), "1024.0K");
        // MB
        assert_eq!(format_size(1024 * 1024), "1.0M");
        assert_eq!(format_size(1024 * 1024 * 500), "500.0M");
        // GB
        assert_eq!(format_size(1024 * 1024 * 1024), "1.0G");
        // TB
        assert_eq!(format_size(1024u64 * 1024 * 1024 * 1024), "1.0T");
        assert_eq!(format_size(1024u64 * 1024 * 1024 * 1024 * 5), "5.0T");
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = StorageAnalyzer::new();
        assert_eq!(analyzer.name(), "storage");
    }

    #[test]
    fn test_analyzer_default() {
        let analyzer = StorageAnalyzer::default();
        assert_eq!(analyzer.name(), "storage");
        assert_eq!(analyzer.interval(), Duration::from_secs(30));
    }

    #[test]
    fn test_analyzer_available() {
        let analyzer = StorageAnalyzer::new();
        #[cfg(target_os = "linux")]
        assert!(analyzer.available());
    }

    #[test]
    fn test_analyzer_interval() {
        let analyzer = StorageAnalyzer::new();
        assert_eq!(analyzer.interval(), Duration::from_secs(30));
    }

    #[test]
    fn test_analyzer_data() {
        let analyzer = StorageAnalyzer::new();
        let data = analyzer.data();
        // Initial data should be empty
        assert!(data.mounts.is_empty());
        assert!(data.by_mount_point.is_empty());
        assert_eq!(data.total_capacity, 0);
        assert_eq!(data.total_used, 0);
    }

    #[test]
    fn test_analyzer_collect() {
        let mut analyzer = StorageAnalyzer::new();
        let result = analyzer.collect();
        assert!(result.is_ok());

        let data = analyzer.data();
        // Should have at least root filesystem
        #[cfg(target_os = "linux")]
        assert!(!data.mounts.is_empty());
    }

    #[test]
    fn test_analyzer_parse_mounts_line() {
        let analyzer = StorageAnalyzer::new();

        // Valid line
        let line = "/dev/sda1 / ext4 rw,relatime,errors=remount-ro 0 1";
        let mount = analyzer.parse_mounts_line(line);
        assert!(mount.is_some());
        let mount = mount.unwrap();
        assert_eq!(mount.device, "/dev/sda1");
        assert_eq!(mount.mount_point, "/");
        assert_eq!(mount.fs_type, "ext4");
        assert!(mount.options.contains(&"rw".to_string()));
        assert!(mount.options.contains(&"relatime".to_string()));

        // Invalid line (too few parts)
        let invalid_line = "/dev/sda1 /";
        let mount = analyzer.parse_mounts_line(invalid_line);
        assert!(mount.is_none());

        // Empty line
        let empty_line = "";
        let mount = analyzer.parse_mounts_line(empty_line);
        assert!(mount.is_none());
    }

    #[test]
    fn test_analyzer_parse_mounts_line_with_escaped_path() {
        let analyzer = StorageAnalyzer::new();
        let line = "/dev/sda1 /mnt/My\\040Drive ext4 rw 0 1";
        let mount = analyzer.parse_mounts_line(line);
        assert!(mount.is_some());
        let mount = mount.unwrap();
        assert_eq!(mount.mount_point, "/mnt/My Drive");
    }

    #[test]
    fn test_storage_data_real_filesystems() {
        let data = StorageData {
            mounts: vec![
                MountInfo {
                    fs_type: "ext4".to_string(),
                    mount_point: "/".to_string(),
                    ..Default::default()
                },
                MountInfo {
                    fs_type: "proc".to_string(),
                    mount_point: "/proc".to_string(),
                    ..Default::default()
                },
                MountInfo {
                    fs_type: "btrfs".to_string(),
                    mount_point: "/home".to_string(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        let real: Vec<_> = data.real_filesystems().collect();
        assert_eq!(real.len(), 2);
        assert_eq!(real[0].mount_point, "/");
        assert_eq!(real[1].mount_point, "/home");
    }

    #[test]
    fn test_storage_data_get_mount() {
        let mut by_mount_point = HashMap::new();
        by_mount_point.insert(
            "/".to_string(),
            MountInfo {
                device: "/dev/sda1".to_string(),
                mount_point: "/".to_string(),
                ..Default::default()
            },
        );
        by_mount_point.insert(
            "/home".to_string(),
            MountInfo {
                device: "/dev/sda2".to_string(),
                mount_point: "/home".to_string(),
                ..Default::default()
            },
        );

        let data = StorageData {
            mounts: vec![],
            by_mount_point,
            total_capacity: 0,
            total_used: 0,
        };

        assert!(data.get_mount("/").is_some());
        assert_eq!(data.get_mount("/").unwrap().device, "/dev/sda1");
        assert!(data.get_mount("/home").is_some());
        assert!(data.get_mount("/nonexistent").is_none());
    }

    #[test]
    fn test_storage_data_overall_usage_percent() {
        let data = StorageData {
            total_capacity: 1000,
            total_used: 400,
            ..Default::default()
        };
        assert!((data.overall_usage_percent() - 40.0).abs() < 0.1);

        let empty_data = StorageData {
            total_capacity: 0,
            total_used: 0,
            ..Default::default()
        };
        assert!((empty_data.overall_usage_percent() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_storage_data_default() {
        let data = StorageData::default();
        assert!(data.mounts.is_empty());
        assert!(data.by_mount_point.is_empty());
        assert_eq!(data.total_capacity, 0);
        assert_eq!(data.total_used, 0);
    }

    #[test]
    fn test_mount_info_default() {
        let mount = MountInfo::default();
        assert!(mount.device.is_empty());
        assert!(mount.mount_point.is_empty());
        assert!(mount.fs_type.is_empty());
        assert!(mount.options.is_empty());
        assert_eq!(mount.total, 0);
        assert_eq!(mount.used, 0);
        assert_eq!(mount.available, 0);
        assert_eq!(mount.inodes_total, 0);
        assert_eq!(mount.inodes_used, 0);
        assert_eq!(mount.inodes_free, 0);
    }

    #[test]
    fn test_mount_info_clone() {
        let mount = MountInfo {
            device: "/dev/sda1".to_string(),
            mount_point: "/".to_string(),
            fs_type: "ext4".to_string(),
            options: vec!["rw".to_string()],
            total: 100,
            used: 50,
            available: 50,
            inodes_total: 1000,
            inodes_used: 500,
            inodes_free: 500,
        };
        let cloned = mount.clone();
        assert_eq!(cloned.device, mount.device);
        assert_eq!(cloned.mount_point, mount.mount_point);
        assert_eq!(cloned.total, mount.total);
    }

    #[test]
    fn test_mount_info_debug() {
        let mount = MountInfo {
            device: "/dev/sda1".to_string(),
            ..Default::default()
        };
        let debug = format!("{:?}", mount);
        assert!(debug.contains("MountInfo"));
        assert!(debug.contains("/dev/sda1"));
    }

    #[test]
    fn test_storage_data_clone() {
        let data = StorageData {
            mounts: vec![MountInfo::default()],
            by_mount_point: HashMap::new(),
            total_capacity: 100,
            total_used: 50,
        };
        let cloned = data.clone();
        assert_eq!(cloned.mounts.len(), 1);
        assert_eq!(cloned.total_capacity, 100);
    }

    #[test]
    fn test_storage_data_debug() {
        let data = StorageData::default();
        let debug = format!("{:?}", data);
        assert!(debug.contains("StorageData"));
    }

    #[test]
    fn test_analyzer_collect_updates_by_mount_point() {
        let mut analyzer = StorageAnalyzer::new();
        let _ = analyzer.collect();
        let data = analyzer.data();

        // Each mount should be in by_mount_point
        for mount in &data.mounts {
            assert!(data.by_mount_point.contains_key(&mount.mount_point));
        }
    }

    #[test]
    fn test_analyzer_collect_calculates_totals() {
        let mut analyzer = StorageAnalyzer::new();
        let _ = analyzer.collect();
        let data = analyzer.data();

        // total_capacity should be sum of real fs totals
        let expected_total: u64 = data
            .mounts
            .iter()
            .filter(|m| m.is_real_fs() && !m.is_tmpfs())
            .map(|m| m.total)
            .sum();
        assert_eq!(data.total_capacity, expected_total);
    }
}
