//! Treemap Analyzer
//!
//! Scans filesystem to build treemap data showing directory sizes.
//! Caches results to avoid re-scanning every frame.

#![allow(clippy::uninlined_format_args)]

use std::fs::{self};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use super::{Analyzer, AnalyzerError};

/// A node in the treemap (file or directory)
#[derive(Debug, Clone)]
pub struct TreemapNode {
    /// Node name (file or directory name)
    pub name: String,
    /// Full path
    pub path: PathBuf,
    /// Size in bytes
    pub size: u64,
    /// Whether this is a directory
    pub is_dir: bool,
    /// Number of files (if directory)
    pub file_count: u32,
    /// Number of subdirectories (if directory)
    pub dir_count: u32,
    /// Depth from root
    pub depth: u32,
    /// Children (if directory and expanded)
    pub children: Vec<Self>,
}

impl TreemapNode {
    /// Create a new node for a file
    pub fn file(name: String, path: PathBuf, size: u64, depth: u32) -> Self {
        Self {
            name,
            path,
            size,
            is_dir: false,
            file_count: 1,
            dir_count: 0,
            depth,
            children: Vec::new(),
        }
    }

    /// Create a new node for a directory
    pub fn directory(name: String, path: PathBuf, depth: u32) -> Self {
        Self {
            name,
            path,
            size: 0,
            is_dir: true,
            file_count: 0,
            dir_count: 0,
            depth,
            children: Vec::new(),
        }
    }

    /// Format size for display
    pub fn display_size(&self) -> String {
        format_size(self.size)
    }

    /// Get percentage of parent
    pub fn percent_of(&self, total: u64) -> f32 {
        if total > 0 {
            (self.size as f64 / total as f64 * 100.0) as f32
        } else {
            0.0
        }
    }
}

/// Treemap data
#[derive(Debug, Clone, Default)]
pub struct TreemapData {
    /// Root path being scanned
    pub root_path: PathBuf,
    /// Root node
    pub root: Option<TreemapNode>,
    /// Flattened list of top-level children (for display)
    pub top_items: Vec<TreemapNode>,
    /// Total size
    pub total_size: u64,
    /// Total file count
    pub total_files: u32,
    /// Total directory count
    pub total_dirs: u32,
    /// Scan depth
    pub depth: u32,
    /// Last scan time
    pub last_scan: Option<Instant>,
    /// Scan duration
    pub scan_duration: Duration,
}

impl TreemapData {
    /// Check if data is stale (older than `cache_ttl`)
    pub fn is_stale(&self, cache_ttl: Duration) -> bool {
        match self.last_scan {
            Some(last) => last.elapsed() > cache_ttl,
            None => true,
        }
    }
}

/// Configuration for treemap scanning
#[derive(Debug, Clone)]
pub struct TreemapConfig {
    /// Root path to scan
    pub root_path: PathBuf,
    /// Maximum depth to scan
    pub max_depth: u32,
    /// Maximum number of items to track per directory
    pub max_items_per_dir: usize,
    /// Skip hidden files/directories
    pub skip_hidden: bool,
    /// Cache TTL (how long before re-scanning)
    pub cache_ttl: Duration,
}

impl Default for TreemapConfig {
    fn default() -> Self {
        Self {
            root_path: PathBuf::from("/home"),
            max_depth: 2,
            max_items_per_dir: 100,
            skip_hidden: true,
            cache_ttl: Duration::from_secs(60),
        }
    }
}

/// Analyzer for filesystem treemap
pub struct TreemapAnalyzer {
    data: TreemapData,
    config: TreemapConfig,
    interval: Duration,
}

impl Default for TreemapAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl TreemapAnalyzer {
    /// Create a new treemap analyzer with default config
    pub fn new() -> Self {
        Self::with_config(TreemapConfig::default())
    }

    /// Create with custom config
    pub fn with_config(config: TreemapConfig) -> Self {
        Self {
            data: TreemapData {
                root_path: config.root_path.clone(),
                ..Default::default()
            },
            config,
            interval: Duration::from_secs(60), // Re-scan every minute
        }
    }

    /// Get the current treemap data
    pub fn data(&self) -> &TreemapData {
        &self.data
    }

    /// Set the root path to scan
    pub fn set_root_path(&mut self, path: PathBuf) {
        if self.config.root_path != path {
            self.config.root_path = path.clone();
            self.data.root_path = path;
            self.data.last_scan = None; // Force re-scan
        }
    }

    /// Set max depth
    pub fn set_max_depth(&mut self, depth: u32) {
        self.config.max_depth = depth;
    }

    /// Scan a directory recursively
    fn scan_directory(&self, path: &Path, depth: u32) -> Option<TreemapNode> {
        if depth > self.config.max_depth {
            return None;
        }

        let name = path.file_name().map_or_else(
            || path.to_string_lossy().to_string(),
            |s| s.to_string_lossy().to_string(),
        );

        // Skip hidden files if configured
        if self.config.skip_hidden && name.starts_with('.') {
            return None;
        }

        let metadata = match fs::metadata(path) {
            Ok(m) => m,
            Err(_) => return None,
        };

        if metadata.is_file() {
            return Some(TreemapNode::file(
                name,
                path.to_path_buf(),
                metadata.len(),
                depth,
            ));
        }

        if !metadata.is_dir() {
            return None;
        }

        let mut node = TreemapNode::directory(name, path.to_path_buf(), depth);
        let mut children = Vec::new();

        // Read directory entries
        let entries = match fs::read_dir(path) {
            Ok(entries) => entries,
            Err(_) => {
                // Can't read directory, return empty
                return Some(node);
            }
        };

        for entry in entries.take(self.config.max_items_per_dir * 10) {
            let Ok(entry) = entry else { continue };
            let child_path = entry.path();

            if let Some(child) = self.scan_directory(&child_path, depth + 1) {
                node.size += child.size;
                node.file_count += child.file_count;
                if child.is_dir {
                    node.dir_count += 1;
                    node.dir_count += child.dir_count;
                }
                children.push(child);
            }
        }

        // Sort children by size (largest first)
        children.sort_by(|a, b| b.size.cmp(&a.size));

        // Keep only top items
        children.truncate(self.config.max_items_per_dir);

        node.children = children;
        Some(node)
    }
}

impl Analyzer for TreemapAnalyzer {
    fn name(&self) -> &'static str {
        "treemap"
    }

    fn collect(&mut self) -> Result<(), AnalyzerError> {
        // Check cache
        if !self.data.is_stale(self.config.cache_ttl) {
            return Ok(());
        }

        let start = Instant::now();
        let root_path = self.config.root_path.clone();

        if !root_path.exists() {
            return Err(AnalyzerError::IoError(format!(
                "Path does not exist: {}",
                root_path.display()
            )));
        }

        // Scan the directory tree
        let root = self.scan_directory(&root_path, 0);

        let (total_size, total_files, total_dirs, top_items) = if let Some(ref node) = root {
            let top = node.children.clone();
            (node.size, node.file_count, node.dir_count, top)
        } else {
            (0, 0, 0, Vec::new())
        };

        self.data = TreemapData {
            root_path,
            root,
            top_items,
            total_size,
            total_files,
            total_dirs,
            depth: self.config.max_depth,
            last_scan: Some(Instant::now()),
            scan_duration: start.elapsed(),
        };

        Ok(())
    }

    fn interval(&self) -> Duration {
        self.interval
    }

    fn available(&self) -> bool {
        self.config.root_path.exists()
    }
}

/// Format bytes for human-readable display
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
    use std::env;

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(512), "512B");
        assert_eq!(format_size(1024), "1.0K");
        assert_eq!(format_size(1536), "1.5K");
        assert_eq!(format_size(1048576), "1.0M");
        assert_eq!(format_size(1073741824), "1.0G");
        assert_eq!(format_size(1099511627776), "1.0T");
    }

    #[test]
    fn test_treemap_node_file() {
        let node = TreemapNode::file(
            "test.txt".to_string(),
            PathBuf::from("/tmp/test.txt"),
            1024,
            0,
        );

        assert_eq!(node.name, "test.txt");
        assert_eq!(node.size, 1024);
        assert!(!node.is_dir);
        assert_eq!(node.file_count, 1);
        assert_eq!(node.display_size(), "1.0K");
    }

    #[test]
    fn test_treemap_node_directory() {
        let node = TreemapNode::directory("dir".to_string(), PathBuf::from("/tmp/dir"), 0);

        assert_eq!(node.name, "dir");
        assert_eq!(node.size, 0);
        assert!(node.is_dir);
        assert_eq!(node.file_count, 0);
    }

    #[test]
    fn test_treemap_node_percent() {
        let node = TreemapNode::file("test".to_string(), PathBuf::from("/test"), 250, 0);

        assert!((node.percent_of(1000) - 25.0).abs() < 0.01);
        assert!((node.percent_of(0) - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_treemap_data_stale() {
        let mut data = TreemapData::default();

        // No scan yet, should be stale
        assert!(data.is_stale(Duration::from_secs(60)));

        // Set last scan to now
        data.last_scan = Some(Instant::now());
        assert!(!data.is_stale(Duration::from_secs(60)));

        // Very short TTL, should be stale
        assert!(data.is_stale(Duration::from_nanos(1)));
    }

    #[test]
    fn test_treemap_config_default() {
        let config = TreemapConfig::default();

        assert_eq!(config.root_path, PathBuf::from("/home"));
        assert_eq!(config.max_depth, 2);
        assert!(config.skip_hidden);
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = TreemapAnalyzer::new();
        // Just verify it doesn't panic
        let _ = analyzer.available();
    }

    #[test]
    fn test_analyzer_scan_tmp() {
        // Use temp directory for test
        let temp_dir = env::temp_dir();
        let config = TreemapConfig {
            root_path: temp_dir.clone(),
            max_depth: 1,
            max_items_per_dir: 10,
            skip_hidden: true,
            cache_ttl: Duration::from_secs(60),
        };

        let mut analyzer = TreemapAnalyzer::with_config(config);

        // Should be able to scan temp directory
        if temp_dir.exists() {
            let result = analyzer.collect();
            assert!(result.is_ok());

            let data = analyzer.data();
            assert!(data.last_scan.is_some());
            // Temp directory should have some content
        }
    }

    #[test]
    fn test_set_root_path() {
        let mut analyzer = TreemapAnalyzer::new();
        let new_path = PathBuf::from("/tmp");

        analyzer.set_root_path(new_path.clone());
        assert_eq!(analyzer.data().root_path, new_path);
    }

    #[test]
    fn test_analyzer_name() {
        let analyzer = TreemapAnalyzer::new();
        assert_eq!(analyzer.name(), "treemap");
    }

    #[test]
    fn test_analyzer_interval() {
        let analyzer = TreemapAnalyzer::new();
        assert_eq!(analyzer.interval(), Duration::from_secs(60));
    }

    #[test]
    fn test_treemap_data_default() {
        let data = TreemapData::default();
        assert!(data.root.is_none());
        assert!(data.top_items.is_empty());
        assert_eq!(data.total_size, 0);
        assert_eq!(data.total_files, 0);
        assert_eq!(data.total_dirs, 0);
        assert!(data.last_scan.is_none());
    }

    #[test]
    fn test_treemap_data_clone() {
        let mut data = TreemapData::default();
        data.total_size = 1000;
        data.total_files = 10;

        let cloned = data.clone();
        assert_eq!(cloned.total_size, 1000);
        assert_eq!(cloned.total_files, 10);
    }

    #[test]
    fn test_treemap_config_clone() {
        let config = TreemapConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.max_depth, config.max_depth);
        assert_eq!(cloned.skip_hidden, config.skip_hidden);
    }

    #[test]
    fn test_treemap_node_clone() {
        let node = TreemapNode::file(
            "test.txt".to_string(),
            PathBuf::from("/tmp/test.txt"),
            1024,
            0,
        );
        let cloned = node.clone();
        assert_eq!(cloned.name, node.name);
        assert_eq!(cloned.size, node.size);
    }

    #[test]
    fn test_format_size_kb() {
        assert_eq!(format_size(2048), "2.0K");
        assert_eq!(format_size(3072), "3.0K");
    }

    #[test]
    fn test_format_size_mb() {
        assert_eq!(format_size(5 * 1024 * 1024), "5.0M");
        assert_eq!(format_size(10 * 1024 * 1024), "10.0M");
    }

    #[test]
    fn test_format_size_gb() {
        assert_eq!(format_size(2 * 1024 * 1024 * 1024), "2.0G");
    }

    #[test]
    fn test_set_max_depth() {
        let mut analyzer = TreemapAnalyzer::new();
        analyzer.set_max_depth(5);
        // Can't easily verify, but should not panic
    }

    #[test]
    fn test_treemap_node_debug() {
        let node = TreemapNode::file("test".to_string(), PathBuf::from("/test"), 100, 0);
        let debug = format!("{:?}", node);
        assert!(debug.contains("TreemapNode"));
    }

    #[test]
    fn test_treemap_data_debug() {
        let data = TreemapData::default();
        let debug = format!("{:?}", data);
        assert!(debug.contains("TreemapData"));
    }

    #[test]
    fn test_treemap_config_debug() {
        let config = TreemapConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("TreemapConfig"));
    }

    #[test]
    fn test_set_root_path_same_path() {
        let mut analyzer = TreemapAnalyzer::new();
        let path = analyzer.data().root_path.clone();

        // Set same path - should not force re-scan
        analyzer.set_root_path(path.clone());
        // Should not panic
    }

    #[test]
    fn test_analyzer_default() {
        let analyzer = TreemapAnalyzer::default();
        assert_eq!(analyzer.name(), "treemap");
    }

    #[test]
    fn test_collect_nonexistent_path() {
        let config = TreemapConfig {
            root_path: PathBuf::from("/nonexistent/path/that/does/not/exist"),
            ..Default::default()
        };
        let mut analyzer = TreemapAnalyzer::with_config(config);
        let result = analyzer.collect();
        assert!(result.is_err());
    }

    #[test]
    fn test_available_nonexistent() {
        let config = TreemapConfig {
            root_path: PathBuf::from("/nonexistent/path"),
            ..Default::default()
        };
        let analyzer = TreemapAnalyzer::with_config(config);
        assert!(!analyzer.available());
    }

    #[test]
    fn test_treemap_node_children() {
        let mut parent = TreemapNode::directory("parent".to_string(), PathBuf::from("/parent"), 0);
        let child = TreemapNode::file(
            "child.txt".to_string(),
            PathBuf::from("/parent/child.txt"),
            100,
            1,
        );
        parent.children.push(child);

        assert_eq!(parent.children.len(), 1);
        assert_eq!(parent.children[0].name, "child.txt");
    }

    #[test]
    fn test_cache_not_stale() {
        let mut data = TreemapData::default();
        data.last_scan = Some(Instant::now());

        // Very long TTL, should not be stale
        assert!(!data.is_stale(Duration::from_secs(3600)));
    }
}
