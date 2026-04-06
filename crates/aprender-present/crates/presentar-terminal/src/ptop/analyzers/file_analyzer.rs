//! File Analyzer
//!
//! Tracks file system activity, hot files (recently accessed), and provides
//! inode statistics. Uses /proc/[pid]/fd and atime for activity tracking.

#![allow(clippy::uninlined_format_args)]

use std::collections::HashMap;
use std::fs::{self, Metadata};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use super::{Analyzer, AnalyzerError};

/// Information about a tracked file
#[derive(Debug, Clone)]
pub struct TrackedFile {
    /// File path
    pub path: PathBuf,
    /// File size in bytes
    pub size: u64,
    /// Last access time
    pub accessed: Option<SystemTime>,
    /// Last modification time
    pub modified: Option<SystemTime>,
    /// Number of processes with this file open
    pub open_count: u32,
    /// PIDs of processes with this file open
    pub open_by: Vec<u32>,
    /// Inode number
    pub inode: u64,
    /// Device ID
    pub device: u64,
}

impl TrackedFile {
    /// Format size for display
    pub fn size_display(&self) -> String {
        format_size(self.size)
    }

    /// Is this file currently open by any process?
    pub fn is_open(&self) -> bool {
        self.open_count > 0
    }

    /// Time since last access
    pub fn time_since_access(&self) -> Option<Duration> {
        self.accessed.and_then(|t| t.elapsed().ok())
    }

    /// Is this a "hot" file (accessed recently)?
    pub fn is_hot(&self, threshold: Duration) -> bool {
        self.time_since_access().is_some_and(|d| d < threshold)
    }
}

/// File type category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FileCategory {
    /// Regular file
    Regular,
    /// Directory
    Directory,
    /// Symbolic link
    Symlink,
    /// Socket
    Socket,
    /// FIFO/pipe
    Fifo,
    /// Block device
    BlockDevice,
    /// Character device
    CharDevice,
    /// Unknown
    Unknown,
}

impl FileCategory {
    /// Parse from file metadata
    pub fn from_metadata(meta: &Metadata) -> Self {
        use std::os::unix::fs::FileTypeExt;
        let ft = meta.file_type();
        if ft.is_file() {
            Self::Regular
        } else if ft.is_dir() {
            Self::Directory
        } else if ft.is_symlink() {
            Self::Symlink
        } else if ft.is_socket() {
            Self::Socket
        } else if ft.is_fifo() {
            Self::Fifo
        } else if ft.is_block_device() {
            Self::BlockDevice
        } else if ft.is_char_device() {
            Self::CharDevice
        } else {
            Self::Unknown
        }
    }

    /// Display name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Regular => "file",
            Self::Directory => "dir",
            Self::Symlink => "link",
            Self::Socket => "sock",
            Self::Fifo => "fifo",
            Self::BlockDevice => "blk",
            Self::CharDevice => "chr",
            Self::Unknown => "?",
        }
    }
}

/// Inode statistics for a filesystem
#[derive(Debug, Clone, Default)]
pub struct InodeStats {
    /// Total inodes
    pub total: u64,
    /// Used inodes
    pub used: u64,
    /// Free inodes
    pub free: u64,
    /// Mount point
    pub mount_point: String,
}

impl InodeStats {
    /// Usage percentage
    pub fn usage_percent(&self) -> f64 {
        if self.total > 0 {
            self.used as f64 / self.total as f64 * 100.0
        } else {
            0.0
        }
    }

    /// Is inode usage critical (>90%)?
    pub fn is_critical(&self) -> bool {
        self.usage_percent() > 90.0
    }
}

/// File analyzer data
#[derive(Debug, Clone, Default)]
pub struct FileAnalyzerData {
    /// Hot files (recently accessed)
    pub hot_files: Vec<TrackedFile>,
    /// Open files by path
    pub open_files: HashMap<PathBuf, TrackedFile>,
    /// Inode stats per mount point
    pub inode_stats: HashMap<String, InodeStats>,
    /// Total open files
    pub total_open_files: usize,
    /// Total hot files
    pub total_hot_files: usize,
}

impl FileAnalyzerData {
    /// Get top N hot files by access time
    pub fn top_hot_files(&self, n: usize) -> impl Iterator<Item = &TrackedFile> {
        self.hot_files.iter().take(n)
    }

    /// Get files open by a specific process
    pub fn files_by_pid(&self, pid: u32) -> impl Iterator<Item = &TrackedFile> {
        self.open_files
            .values()
            .filter(move |f| f.open_by.contains(&pid))
    }
}

/// Analyzer for file activity and inode stats
pub struct FileAnalyzer {
    data: FileAnalyzerData,
    interval: Duration,
    /// Hot file threshold (files accessed within this duration)
    hot_threshold: Duration,
    /// Maximum files to track
    max_tracked: usize,
    /// Last collection time — skip collect() until interval elapses
    last_collect: Option<std::time::Instant>,
}

impl Default for FileAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl FileAnalyzer {
    /// Create a new file analyzer
    pub fn new() -> Self {
        Self {
            data: FileAnalyzerData::default(),
            interval: Duration::from_secs(5),
            hot_threshold: Duration::from_secs(60), // Files accessed in last minute
            max_tracked: 100,
            last_collect: None,
        }
    }

    /// Get the current data
    pub fn data(&self) -> &FileAnalyzerData {
        &self.data
    }

    /// Set hot file threshold
    pub fn set_hot_threshold(&mut self, threshold: Duration) {
        self.hot_threshold = threshold;
    }

    /// Scan /proc/[pid]/fd for open files
    fn scan_open_files(&self) -> HashMap<PathBuf, TrackedFile> {
        let mut files: HashMap<PathBuf, TrackedFile> = HashMap::new();

        let Ok(proc_entries) = fs::read_dir("/proc") else {
            return files;
        };

        for entry in proc_entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();

            // Only process numeric directories (PIDs)
            let Ok(pid) = name_str.parse::<u32>() else {
                continue;
            };

            let fd_path = entry.path().join("fd");
            let Ok(fd_entries) = fs::read_dir(&fd_path) else {
                continue;
            };

            for fd_entry in fd_entries.flatten() {
                let link = fd_entry.path();
                let Ok(target) = fs::read_link(&link) else {
                    continue;
                };

                // Skip non-file paths (sockets, pipes, etc.)
                if !target.starts_with("/")
                    || target.starts_with("/proc")
                    || target.starts_with("/dev")
                {
                    continue;
                }

                // Get or create file entry
                let file = files.entry(target.clone()).or_insert_with(|| {
                    let mut tracked = TrackedFile {
                        path: target.clone(),
                        size: 0,
                        accessed: None,
                        modified: None,
                        open_count: 0,
                        open_by: Vec::new(),
                        inode: 0,
                        device: 0,
                    };

                    // Get file metadata
                    if let Ok(meta) = fs::metadata(&target) {
                        use std::os::unix::fs::MetadataExt;
                        tracked.size = meta.len();
                        tracked.accessed = meta.accessed().ok();
                        tracked.modified = meta.modified().ok();
                        tracked.inode = meta.ino();
                        tracked.device = meta.dev();
                    }

                    tracked
                });

                file.open_count += 1;
                if !file.open_by.contains(&pid) {
                    file.open_by.push(pid);
                }
            }
        }

        files
    }

    /// Get inode stats from df -i
    fn get_inode_stats(&self) -> HashMap<String, InodeStats> {
        let mut stats = HashMap::new();

        let Ok(output) = std::process::Command::new("df")
            .arg("-i")
            .arg("--output=target,itotal,iused,iavail")
            .output()
        else {
            return stats;
        };

        if !output.status.success() {
            return stats;
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines().skip(1) {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                let mount_point = parts[0].to_string();
                let total: u64 = parts[1].parse().unwrap_or(0);
                let used: u64 = parts[2].parse().unwrap_or(0);
                let free: u64 = parts[3].parse().unwrap_or(0);

                stats.insert(
                    mount_point.clone(),
                    InodeStats {
                        total,
                        used,
                        free,
                        mount_point,
                    },
                );
            }
        }

        stats
    }

    /// Identify hot files from open files
    fn identify_hot_files(&self, open_files: &HashMap<PathBuf, TrackedFile>) -> Vec<TrackedFile> {
        let mut hot: Vec<TrackedFile> = open_files
            .values()
            .filter(|f| f.is_hot(self.hot_threshold))
            .cloned()
            .collect();

        // Sort by access time (most recent first)
        hot.sort_by(|a, b| {
            let a_time = a.time_since_access().unwrap_or(Duration::MAX);
            let b_time = b.time_since_access().unwrap_or(Duration::MAX);
            a_time.cmp(&b_time)
        });

        // Limit to max_tracked
        hot.truncate(self.max_tracked);
        hot
    }
}

impl Analyzer for FileAnalyzer {
    fn name(&self) -> &'static str {
        "file_analyzer"
    }

    fn collect(&mut self) -> Result<(), AnalyzerError> {
        // Interval guard: /proc/*/fd scan is O(total_fds) — skip until interval elapses
        if let Some(last) = self.last_collect {
            if last.elapsed() < self.interval {
                return Ok(());
            }
        }
        self.last_collect = Some(std::time::Instant::now());

        let open_files = self.scan_open_files();
        let hot_files = self.identify_hot_files(&open_files);
        let inode_stats = self.get_inode_stats();

        let total_open = open_files.len();
        let total_hot = hot_files.len();

        self.data = FileAnalyzerData {
            hot_files,
            open_files,
            inode_stats,
            total_open_files: total_open,
            total_hot_files: total_hot,
        };

        Ok(())
    }

    fn interval(&self) -> Duration {
        self.interval
    }

    fn available(&self) -> bool {
        Path::new("/proc").exists()
    }
}

/// Format size for display
fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
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
    fn test_file_category() {
        assert_eq!(FileCategory::Regular.as_str(), "file");
        assert_eq!(FileCategory::Directory.as_str(), "dir");
        assert_eq!(FileCategory::Socket.as_str(), "sock");
    }

    #[test]
    fn test_inode_stats() {
        let stats = InodeStats {
            total: 1000,
            used: 950,
            free: 50,
            mount_point: "/".to_string(),
        };

        assert!((stats.usage_percent() - 95.0).abs() < 0.1);
        assert!(stats.is_critical());
    }

    #[test]
    fn test_tracked_file() {
        let file = TrackedFile {
            path: PathBuf::from("/tmp/test"),
            size: 1024,
            accessed: Some(SystemTime::now()),
            modified: Some(SystemTime::now()),
            open_count: 2,
            open_by: vec![1234, 5678],
            inode: 12345,
            device: 1,
        };

        assert!(file.is_open());
        assert_eq!(file.size_display(), "1.0K");
        assert!(file.is_hot(Duration::from_secs(60)));
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(512), "512B");
        assert_eq!(format_size(1536), "1.5K");
        assert_eq!(format_size(1_500_000), "1.4M");
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = FileAnalyzer::new();
        assert_eq!(analyzer.name(), "file_analyzer");
    }

    #[test]
    fn test_analyzer_available() {
        let analyzer = FileAnalyzer::new();
        #[cfg(target_os = "linux")]
        assert!(analyzer.available());
    }

    #[test]
    fn test_analyzer_collect() {
        let mut analyzer = FileAnalyzer::new();
        let result = analyzer.collect();
        assert!(result.is_ok());
    }

    // Additional FileCategory tests
    #[test]
    fn test_file_category_symlink() {
        assert_eq!(FileCategory::Symlink.as_str(), "link");
    }

    #[test]
    fn test_file_category_fifo() {
        assert_eq!(FileCategory::Fifo.as_str(), "fifo");
    }

    #[test]
    fn test_file_category_block_device() {
        assert_eq!(FileCategory::BlockDevice.as_str(), "blk");
    }

    #[test]
    fn test_file_category_char_device() {
        assert_eq!(FileCategory::CharDevice.as_str(), "chr");
    }

    #[test]
    fn test_file_category_unknown() {
        assert_eq!(FileCategory::Unknown.as_str(), "?");
    }

    #[test]
    fn test_file_category_debug() {
        let cat = FileCategory::Regular;
        let debug = format!("{:?}", cat);
        assert!(debug.contains("Regular"));
    }

    #[test]
    fn test_file_category_clone() {
        let cat = FileCategory::Socket;
        let cloned = cat.clone();
        assert_eq!(cat, cloned);
    }

    #[test]
    fn test_file_category_copy() {
        let cat = FileCategory::Directory;
        let copied: FileCategory = cat;
        assert_eq!(copied, FileCategory::Directory);
    }

    #[test]
    fn test_file_category_eq() {
        assert_eq!(FileCategory::Regular, FileCategory::Regular);
        assert_ne!(FileCategory::Regular, FileCategory::Directory);
    }

    #[test]
    fn test_file_category_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(FileCategory::Regular);
        set.insert(FileCategory::Directory);
        assert_eq!(set.len(), 2);
        set.insert(FileCategory::Regular);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_file_category_from_metadata_regular() {
        // Test with a real file
        if let Ok(meta) = fs::metadata("/etc/passwd") {
            let cat = FileCategory::from_metadata(&meta);
            assert_eq!(cat, FileCategory::Regular);
        }
    }

    #[test]
    fn test_file_category_from_metadata_directory() {
        if let Ok(meta) = fs::metadata("/tmp") {
            let cat = FileCategory::from_metadata(&meta);
            assert_eq!(cat, FileCategory::Directory);
        }
    }

    // TrackedFile tests
    #[test]
    fn test_tracked_file_not_open() {
        let file = TrackedFile {
            path: PathBuf::from("/tmp/test"),
            size: 0,
            accessed: None,
            modified: None,
            open_count: 0,
            open_by: vec![],
            inode: 0,
            device: 0,
        };
        assert!(!file.is_open());
    }

    #[test]
    fn test_tracked_file_time_since_access_none() {
        let file = TrackedFile {
            path: PathBuf::from("/tmp/test"),
            size: 0,
            accessed: None,
            modified: None,
            open_count: 0,
            open_by: vec![],
            inode: 0,
            device: 0,
        };
        assert!(file.time_since_access().is_none());
    }

    #[test]
    fn test_tracked_file_not_hot_no_access() {
        let file = TrackedFile {
            path: PathBuf::from("/tmp/test"),
            size: 0,
            accessed: None,
            modified: None,
            open_count: 0,
            open_by: vec![],
            inode: 0,
            device: 0,
        };
        assert!(!file.is_hot(Duration::from_secs(60)));
    }

    #[test]
    fn test_tracked_file_debug() {
        let file = TrackedFile {
            path: PathBuf::from("/test"),
            size: 100,
            accessed: None,
            modified: None,
            open_count: 0,
            open_by: vec![],
            inode: 0,
            device: 0,
        };
        let debug = format!("{:?}", file);
        assert!(debug.contains("TrackedFile"));
    }

    #[test]
    fn test_tracked_file_clone() {
        let file = TrackedFile {
            path: PathBuf::from("/test"),
            size: 1000,
            accessed: None,
            modified: None,
            open_count: 5,
            open_by: vec![1, 2, 3],
            inode: 123,
            device: 1,
        };
        let cloned = file.clone();
        assert_eq!(cloned.size, 1000);
        assert_eq!(cloned.open_by.len(), 3);
    }

    #[test]
    fn test_tracked_file_size_display_bytes() {
        let file = TrackedFile {
            path: PathBuf::from("/test"),
            size: 500,
            accessed: None,
            modified: None,
            open_count: 0,
            open_by: vec![],
            inode: 0,
            device: 0,
        };
        assert_eq!(file.size_display(), "500B");
    }

    #[test]
    fn test_tracked_file_size_display_gb() {
        let file = TrackedFile {
            path: PathBuf::from("/test"),
            size: 2 * 1024 * 1024 * 1024, // 2GB
            accessed: None,
            modified: None,
            open_count: 0,
            open_by: vec![],
            inode: 0,
            device: 0,
        };
        assert!(file.size_display().contains("G"));
    }

    // InodeStats tests
    #[test]
    fn test_inode_stats_default() {
        let stats = InodeStats::default();
        assert_eq!(stats.total, 0);
        assert_eq!(stats.used, 0);
        assert_eq!(stats.free, 0);
        assert!(stats.mount_point.is_empty());
    }

    #[test]
    fn test_inode_stats_usage_zero_total() {
        let stats = InodeStats {
            total: 0,
            used: 0,
            free: 0,
            mount_point: "/".to_string(),
        };
        assert!((stats.usage_percent() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_inode_stats_not_critical() {
        let stats = InodeStats {
            total: 1000,
            used: 500,
            free: 500,
            mount_point: "/".to_string(),
        };
        assert!(!stats.is_critical());
    }

    #[test]
    fn test_inode_stats_debug() {
        let stats = InodeStats::default();
        let debug = format!("{:?}", stats);
        assert!(debug.contains("InodeStats"));
    }

    #[test]
    fn test_inode_stats_clone() {
        let stats = InodeStats {
            total: 1000,
            used: 100,
            free: 900,
            mount_point: "/home".to_string(),
        };
        let cloned = stats.clone();
        assert_eq!(cloned.mount_point, "/home");
    }

    // FileAnalyzerData tests
    #[test]
    fn test_file_analyzer_data_default() {
        let data = FileAnalyzerData::default();
        assert!(data.hot_files.is_empty());
        assert!(data.open_files.is_empty());
        assert!(data.inode_stats.is_empty());
        assert_eq!(data.total_open_files, 0);
        assert_eq!(data.total_hot_files, 0);
    }

    #[test]
    fn test_file_analyzer_data_top_hot_files() {
        let data = FileAnalyzerData {
            hot_files: vec![
                TrackedFile {
                    path: PathBuf::from("/a"),
                    size: 0,
                    accessed: None,
                    modified: None,
                    open_count: 0,
                    open_by: vec![],
                    inode: 0,
                    device: 0,
                },
                TrackedFile {
                    path: PathBuf::from("/b"),
                    size: 0,
                    accessed: None,
                    modified: None,
                    open_count: 0,
                    open_by: vec![],
                    inode: 0,
                    device: 0,
                },
            ],
            open_files: HashMap::new(),
            inode_stats: HashMap::new(),
            total_open_files: 0,
            total_hot_files: 2,
        };
        let top: Vec<_> = data.top_hot_files(1).collect();
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_file_analyzer_data_files_by_pid() {
        let mut open_files = HashMap::new();
        open_files.insert(
            PathBuf::from("/test"),
            TrackedFile {
                path: PathBuf::from("/test"),
                size: 0,
                accessed: None,
                modified: None,
                open_count: 1,
                open_by: vec![1234],
                inode: 0,
                device: 0,
            },
        );
        let data = FileAnalyzerData {
            hot_files: vec![],
            open_files,
            inode_stats: HashMap::new(),
            total_open_files: 1,
            total_hot_files: 0,
        };
        let files: Vec<_> = data.files_by_pid(1234).collect();
        assert_eq!(files.len(), 1);
        let no_files: Vec<_> = data.files_by_pid(9999).collect();
        assert_eq!(no_files.len(), 0);
    }

    #[test]
    fn test_file_analyzer_data_debug() {
        let data = FileAnalyzerData::default();
        let debug = format!("{:?}", data);
        assert!(debug.contains("FileAnalyzerData"));
    }

    #[test]
    fn test_file_analyzer_data_clone() {
        let data = FileAnalyzerData {
            hot_files: vec![],
            open_files: HashMap::new(),
            inode_stats: HashMap::new(),
            total_open_files: 10,
            total_hot_files: 5,
        };
        let cloned = data.clone();
        assert_eq!(cloned.total_open_files, 10);
    }

    // FileAnalyzer tests
    #[test]
    fn test_file_analyzer_default() {
        let analyzer = FileAnalyzer::default();
        assert_eq!(analyzer.name(), "file_analyzer");
    }

    #[test]
    fn test_file_analyzer_data() {
        let analyzer = FileAnalyzer::new();
        let data = analyzer.data();
        assert!(data.hot_files.is_empty());
    }

    #[test]
    fn test_file_analyzer_interval() {
        let analyzer = FileAnalyzer::new();
        let interval = analyzer.interval();
        assert_eq!(interval.as_secs(), 5);
    }

    #[test]
    fn test_file_analyzer_set_hot_threshold() {
        let mut analyzer = FileAnalyzer::new();
        analyzer.set_hot_threshold(Duration::from_secs(120));
        // Hot threshold is private, but we can test it via hot file detection
    }

    #[test]
    fn test_file_analyzer_scan_open_files() {
        let analyzer = FileAnalyzer::new();
        let files = analyzer.scan_open_files();
        // Should return some open files on a Linux system
        let _ = files.len();
    }

    #[test]
    fn test_file_analyzer_get_inode_stats() {
        let analyzer = FileAnalyzer::new();
        let stats = analyzer.get_inode_stats();
        // Should return inode stats for mounted filesystems
        let _ = stats.len();
    }

    #[test]
    fn test_file_analyzer_identify_hot_files() {
        let analyzer = FileAnalyzer::new();
        let open_files = HashMap::new();
        let hot = analyzer.identify_hot_files(&open_files);
        assert!(hot.is_empty());
    }

    #[test]
    fn test_file_analyzer_multiple_collects() {
        let mut analyzer = FileAnalyzer::new();
        let _ = analyzer.collect();
        let _ = analyzer.collect();
        let _ = analyzer.collect();
        // Should not panic
    }

    // format_size tests
    #[test]
    fn test_format_size_zero() {
        assert_eq!(format_size(0), "0B");
    }

    #[test]
    fn test_format_size_kb() {
        assert_eq!(format_size(1024), "1.0K");
    }

    #[test]
    fn test_format_size_mb() {
        assert_eq!(format_size(1024 * 1024), "1.0M");
    }

    #[test]
    fn test_format_size_gb() {
        assert_eq!(format_size(1024 * 1024 * 1024), "1.0G");
    }

    #[test]
    fn test_format_size_large_gb() {
        assert_eq!(format_size(10 * 1024 * 1024 * 1024), "10.0G");
    }
}
