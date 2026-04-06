//! Swap Analyzer
//!
//! Parses `/proc/swaps` and `/proc/meminfo` for swap usage statistics.
//! Includes thrashing detection and ZRAM support (PMAT-GAP-027, PMAT-GAP-028).

#![allow(clippy::uninlined_format_args)]

use std::fs;
use std::path::Path;
use std::time::Duration;

use super::{Analyzer, AnalyzerError};
use presentar_core::Color;

// =============================================================================
// THRASHING SEVERITY (PMAT-GAP-027) - ttop parity
// =============================================================================

/// Swap thrashing severity level (ttop parity).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThrashingSeverity {
    /// No thrashing detected
    #[default]
    None,
    /// Mild thrashing - system slightly impacted
    Mild,
    /// Moderate thrashing - noticeable performance impact
    Moderate,
    /// Severe thrashing - system heavily impacted
    Severe,
}

impl ThrashingSeverity {
    /// Get a human-readable status string.
    #[must_use]
    pub fn status(&self) -> &'static str {
        match self {
            Self::None => "OK",
            Self::Mild => "Mild",
            Self::Moderate => "Warning",
            Self::Severe => "Critical",
        }
    }

    /// Get color for UI rendering.
    #[must_use]
    pub fn color(&self) -> Color {
        match self {
            Self::None => Color::new(0.3, 0.9, 0.5, 1.0), // Green
            Self::Mild => Color::new(1.0, 0.9, 0.3, 1.0), // Yellow
            Self::Moderate => Color::new(1.0, 0.6, 0.2, 1.0), // Orange
            Self::Severe => Color::new(1.0, 0.3, 0.3, 1.0), // Red
        }
    }

    /// Get icon for display.
    #[must_use]
    pub fn icon(&self) -> &'static str {
        match self {
            Self::None => "○",
            Self::Mild => "◔",
            Self::Moderate => "◐",
            Self::Severe => "●",
        }
    }

    /// Check if thrashing is occurring.
    #[must_use]
    pub fn is_thrashing(&self) -> bool {
        !matches!(self, Self::None)
    }
}

// =============================================================================
// ZRAM STATISTICS (PMAT-GAP-028)
// =============================================================================

/// ZRAM compression statistics.
#[derive(Debug, Clone, Default)]
pub struct ZramStats {
    /// Original (uncompressed) data size in bytes.
    pub orig_data_size: u64,
    /// Compressed data size in bytes.
    pub compr_data_size: u64,
    /// Total memory used including metadata.
    pub mem_used_total: u64,
    /// Compression algorithm (e.g., "lz4", "zstd").
    pub algorithm: String,
}

impl ZramStats {
    /// Compression ratio (original / compressed).
    #[must_use]
    pub fn compression_ratio(&self) -> f64 {
        if self.compr_data_size > 0 {
            self.orig_data_size as f64 / self.compr_data_size as f64
        } else {
            0.0
        }
    }

    /// Check if ZRAM is active.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.orig_data_size > 0
    }

    /// Format for display (e.g., "2.5x lz4").
    #[must_use]
    pub fn format_display(&self) -> String {
        format!("{:.1}x {}", self.compression_ratio(), self.algorithm)
    }
}

/// Information about a single swap device
#[derive(Debug, Clone, Default)]
pub struct SwapDevice {
    /// Device path (e.g., "/dev/sda2", "/swapfile")
    pub filename: String,
    /// Swap type (partition, file)
    pub swap_type: SwapType,
    /// Total size in bytes
    pub size: u64,
    /// Used size in bytes
    pub used: u64,
    /// Priority (-1 to 32767)
    pub priority: i32,
}

impl SwapDevice {
    /// Available (free) bytes
    pub fn available(&self) -> u64 {
        self.size.saturating_sub(self.used)
    }

    /// Usage percentage
    pub fn usage_percent(&self) -> f64 {
        if self.size > 0 {
            self.used as f64 / self.size as f64 * 100.0
        } else {
            0.0
        }
    }

    /// Format size for display
    pub fn size_display(&self) -> String {
        format_size(self.size)
    }

    /// Format used for display
    pub fn used_display(&self) -> String {
        format_size(self.used)
    }
}

/// Type of swap device
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SwapType {
    /// Swap partition
    #[default]
    Partition,
    /// Swap file
    File,
    /// Unknown type
    Unknown,
}

impl SwapType {
    /// Parse from /proc/swaps type field
    pub fn from_str(s: &str) -> Self {
        match s {
            "partition" => Self::Partition,
            "file" => Self::File,
            _ => Self::Unknown,
        }
    }

    /// Display name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Partition => "partition",
            Self::File => "file",
            Self::Unknown => "unknown",
        }
    }
}

/// Swap statistics data
#[derive(Debug, Clone, Default)]
pub struct SwapData {
    /// Swap devices
    pub devices: Vec<SwapDevice>,
    /// Total swap size in bytes
    pub total: u64,
    /// Total used swap in bytes
    pub used: u64,
    /// Total free swap in bytes
    pub free: u64,
    /// Swap cached in bytes (from /proc/meminfo)
    pub cached: u64,
    /// Swap in rate (pages/sec) - requires delta
    pub swap_in_rate: f64,
    /// Swap out rate (pages/sec) - requires delta
    pub swap_out_rate: f64,
}

impl SwapData {
    /// Overall usage percentage
    pub fn usage_percent(&self) -> f64 {
        if self.total > 0 {
            self.used as f64 / self.total as f64 * 100.0
        } else {
            0.0
        }
    }

    /// Check if swap is under pressure (high usage)
    pub fn is_under_pressure(&self) -> bool {
        self.usage_percent() > 80.0
    }

    /// Check if swap is critical (very high usage)
    pub fn is_critical(&self) -> bool {
        self.usage_percent() > 95.0
    }

    /// Check if swap is thrashing (high I/O rate indicates thrashing) - CB-MEM-004
    /// Returns (is_thrashing, severity) where severity is 0.0-1.0
    pub fn is_thrashing(&self) -> (bool, f64) {
        // Thrashing thresholds (pages per second):
        // - Low: >10 pages/sec combined
        // - Medium: >100 pages/sec combined
        // - High: >1000 pages/sec combined
        let combined_rate = self.swap_in_rate + self.swap_out_rate;
        if combined_rate > 1000.0 {
            (true, 1.0) // Critical thrashing
        } else if combined_rate > 100.0 {
            (true, 0.7) // Moderate thrashing
        } else if combined_rate > 10.0 {
            (true, 0.4) // Light thrashing
        } else {
            (false, 0.0)
        }
    }

    /// Get thrashing severity indicator for display
    /// Returns symbol and description
    pub fn thrashing_indicator(&self) -> (&'static str, &'static str) {
        let (is_thrashing, severity) = self.is_thrashing();
        if !is_thrashing {
            ("○", "idle")
        } else if severity >= 1.0 {
            ("●", "critical")
        } else if severity >= 0.7 {
            ("◐", "thrashing")
        } else {
            ("◔", "swapping")
        }
    }

    /// Number of swap devices
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Detect thrashing severity (PMAT-GAP-027) - ttop parity.
    ///
    /// Multi-signal detection based on swap I/O rates:
    /// - Severe: >1000 pages/sec combined
    /// - Moderate: >500 pages/sec combined
    /// - Mild: >100 pages/sec combined
    /// - None: ≤100 pages/sec combined
    #[must_use]
    pub fn detect_thrashing(&self) -> ThrashingSeverity {
        let combined_rate = self.swap_in_rate + self.swap_out_rate;

        if combined_rate > 1000.0 {
            ThrashingSeverity::Severe
        } else if combined_rate > 500.0 {
            ThrashingSeverity::Moderate
        } else if combined_rate > 100.0 {
            ThrashingSeverity::Mild
        } else {
            ThrashingSeverity::None
        }
    }

    /// Get thrashing severity with PSI integration (enhanced detection).
    ///
    /// Uses both swap I/O rates and memory PSI pressure for more accurate detection.
    #[must_use]
    pub fn detect_thrashing_with_psi(&self, psi_some_avg10: f64) -> ThrashingSeverity {
        let combined_rate = self.swap_in_rate + self.swap_out_rate;

        // Multi-signal detection per ttop specification
        if psi_some_avg10 > 50.0 || (combined_rate > 1000.0) {
            ThrashingSeverity::Severe
        } else if psi_some_avg10 > 25.0 || (combined_rate > 500.0) {
            ThrashingSeverity::Moderate
        } else if psi_some_avg10 > 10.0 || (combined_rate > 100.0) {
            ThrashingSeverity::Mild
        } else {
            ThrashingSeverity::None
        }
    }
}

/// Analyzer for swap statistics
pub struct SwapAnalyzer {
    data: SwapData,
    interval: Duration,
    /// Previous swap in/out pages for rate calculation
    prev_swap_in: u64,
    prev_swap_out: u64,
    /// ZRAM statistics (PMAT-GAP-028)
    zram: Option<ZramStats>,
}

impl Default for SwapAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl SwapAnalyzer {
    /// Create a new swap analyzer
    pub fn new() -> Self {
        Self {
            data: SwapData::default(),
            interval: Duration::from_secs(2),
            prev_swap_in: 0,
            prev_swap_out: 0,
            zram: None,
        }
    }

    /// Get the current data
    pub fn data(&self) -> &SwapData {
        &self.data
    }

    /// Get ZRAM statistics (PMAT-GAP-028).
    #[must_use]
    pub fn zram(&self) -> Option<&ZramStats> {
        self.zram.as_ref()
    }

    /// Check if ZRAM is available.
    #[must_use]
    pub fn has_zram(&self) -> bool {
        self.zram.as_ref().is_some_and(|z| z.is_active())
    }

    /// Get ZRAM compression ratio.
    #[must_use]
    pub fn zram_compression_ratio(&self) -> f64 {
        self.zram.as_ref().map_or(0.0, |z| z.compression_ratio())
    }

    /// Detect thrashing severity (convenience wrapper).
    #[must_use]
    pub fn detect_thrashing(&self) -> ThrashingSeverity {
        self.data.detect_thrashing()
    }

    /// Parse /proc/swaps
    fn parse_swaps(&self) -> Result<Vec<SwapDevice>, AnalyzerError> {
        let contents = fs::read_to_string("/proc/swaps")
            .map_err(|e| AnalyzerError::IoError(format!("Failed to read /proc/swaps: {}", e)))?;

        let mut devices = Vec::new();

        for line in contents.lines().skip(1) {
            // Skip header
            if let Some(device) = self.parse_swaps_line(line) {
                devices.push(device);
            }
        }

        Ok(devices)
    }

    /// Parse a single line from /proc/swaps
    fn parse_swaps_line(&self, line: &str) -> Option<SwapDevice> {
        // Format: Filename  Type  Size  Used  Priority
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 5 {
            return None;
        }

        let filename = parts[0].to_string();
        let swap_type = SwapType::from_str(parts[1]);
        // Sizes are in KB in /proc/swaps
        let size: u64 = parts[2].parse::<u64>().ok()? * 1024;
        let used: u64 = parts[3].parse::<u64>().ok()? * 1024;
        let priority: i32 = parts[4].parse().ok()?;

        Some(SwapDevice {
            filename,
            swap_type,
            size,
            used,
            priority,
        })
    }

    /// Parse swap info from /proc/meminfo
    fn parse_meminfo_swap(&self) -> Result<(u64, u64, u64, u64), AnalyzerError> {
        let contents = fs::read_to_string("/proc/meminfo")
            .map_err(|e| AnalyzerError::IoError(format!("Failed to read /proc/meminfo: {}", e)))?;

        let mut swap_total = 0u64;
        let mut swap_free = 0u64;
        let mut swap_cached = 0u64;

        for line in contents.lines() {
            if line.starts_with("SwapTotal:") {
                swap_total = parse_meminfo_value(line);
            } else if line.starts_with("SwapFree:") {
                swap_free = parse_meminfo_value(line);
            } else if line.starts_with("SwapCached:") {
                swap_cached = parse_meminfo_value(line);
            }
        }

        let swap_used = swap_total.saturating_sub(swap_free);
        Ok((swap_total, swap_used, swap_free, swap_cached))
    }

    /// Parse swap in/out from /proc/vmstat
    fn parse_vmstat_swap(&self) -> (u64, u64) {
        let contents = match fs::read_to_string("/proc/vmstat") {
            Ok(c) => c,
            Err(_) => return (0, 0),
        };

        let mut pswpin = 0u64;
        let mut pswpout = 0u64;

        for line in contents.lines() {
            if line.starts_with("pswpin ") {
                pswpin = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0);
            } else if line.starts_with("pswpout ") {
                pswpout = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0);
            }
        }

        (pswpin, pswpout)
    }

    /// Read ZRAM statistics from /sys/block/zram* (PMAT-GAP-028).
    fn read_zram_stats(&self) -> Option<ZramStats> {
        // Try zram0 first (most common), then iterate
        for i in 0..4 {
            let base = format!("/sys/block/zram{}", i);
            if !Path::new(&base).exists() {
                continue;
            }

            // Read mm_stat for compression statistics
            let mm_stat_path = format!("{}/mm_stat", base);
            if let Ok(contents) = fs::read_to_string(&mm_stat_path) {
                // mm_stat format: orig_data_size compr_data_size mem_used_total ...
                let parts: Vec<&str> = contents.split_whitespace().collect();
                if parts.len() >= 3 {
                    let orig_data_size = parts[0].parse().unwrap_or(0);
                    let compr_data_size = parts[1].parse().unwrap_or(0);
                    let mem_used_total = parts[2].parse().unwrap_or(0);

                    // Read compression algorithm
                    let algo_path = format!("{}/comp_algorithm", base);
                    let algorithm = fs::read_to_string(algo_path)
                        .ok()
                        .and_then(|s| {
                            // Format: "lzo lzo-rle [lz4] zstd" - bracketed is active
                            s.split_whitespace()
                                .find(|a| a.starts_with('[') && a.ends_with(']'))
                                .map(|a| a.trim_matches(|c| c == '[' || c == ']').to_string())
                        })
                        .unwrap_or_else(|| "lz4".to_string());

                    if orig_data_size > 0 {
                        return Some(ZramStats {
                            orig_data_size,
                            compr_data_size,
                            mem_used_total,
                            algorithm,
                        });
                    }
                }
            }
        }
        None
    }
}

impl Analyzer for SwapAnalyzer {
    fn name(&self) -> &'static str {
        "swap"
    }

    fn collect(&mut self) -> Result<(), AnalyzerError> {
        let devices = self.parse_swaps()?;
        let (total, used, free, cached) = self.parse_meminfo_swap()?;
        let (swap_in, swap_out) = self.parse_vmstat_swap();

        // Calculate rates (pages per second, assuming 1 second interval)
        let swap_in_rate = (swap_in.saturating_sub(self.prev_swap_in)) as f64;
        let swap_out_rate = (swap_out.saturating_sub(self.prev_swap_out)) as f64;

        self.prev_swap_in = swap_in;
        self.prev_swap_out = swap_out;

        self.data = SwapData {
            devices,
            total,
            used,
            free,
            cached,
            swap_in_rate,
            swap_out_rate,
        };

        // Collect ZRAM stats (PMAT-GAP-028)
        self.zram = self.read_zram_stats();

        Ok(())
    }

    fn interval(&self) -> Duration {
        self.interval
    }

    fn available(&self) -> bool {
        Path::new("/proc/swaps").exists()
    }
}

/// Parse a value from /proc/meminfo (returns bytes)
fn parse_meminfo_value(line: &str) -> u64 {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() >= 2 {
        // Value is in kB, convert to bytes
        parts[1].parse::<u64>().unwrap_or(0) * 1024
    } else {
        0
    }
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
    fn test_swap_type_parsing() {
        assert_eq!(SwapType::from_str("partition"), SwapType::Partition);
        assert_eq!(SwapType::from_str("file"), SwapType::File);
        assert_eq!(SwapType::from_str("unknown"), SwapType::Unknown);
    }

    #[test]
    fn test_swap_device_calculations() {
        let device = SwapDevice {
            filename: "/dev/sda2".to_string(),
            swap_type: SwapType::Partition,
            size: 8 * 1024 * 1024 * 1024, // 8GB
            used: 2 * 1024 * 1024 * 1024, // 2GB
            priority: -1,
        };

        assert_eq!(device.available(), 6 * 1024 * 1024 * 1024);
        assert!((device.usage_percent() - 25.0).abs() < 0.1);
        assert_eq!(device.size_display(), "8.0G");
        assert_eq!(device.used_display(), "2.0G");
    }

    #[test]
    fn test_swap_data_pressure() {
        let mut data = SwapData::default();
        data.total = 100;
        data.used = 50;
        assert!(!data.is_under_pressure());
        assert!(!data.is_critical());

        data.used = 85;
        assert!(data.is_under_pressure());
        assert!(!data.is_critical());

        data.used = 96;
        assert!(data.is_under_pressure());
        assert!(data.is_critical());
    }

    #[test]
    fn test_swap_data_thrashing() {
        let mut data = SwapData::default();

        // No thrashing when rates are 0
        data.swap_in_rate = 0.0;
        data.swap_out_rate = 0.0;
        let (is_thrashing, severity) = data.is_thrashing();
        assert!(!is_thrashing);
        assert_eq!(severity, 0.0);

        // Light thrashing (>10 pages/sec)
        data.swap_in_rate = 8.0;
        data.swap_out_rate = 5.0;
        let (is_thrashing, severity) = data.is_thrashing();
        assert!(is_thrashing);
        assert!((severity - 0.4).abs() < 0.01);

        // Moderate thrashing (>100 pages/sec)
        data.swap_in_rate = 60.0;
        data.swap_out_rate = 50.0;
        let (is_thrashing, severity) = data.is_thrashing();
        assert!(is_thrashing);
        assert!((severity - 0.7).abs() < 0.01);

        // Critical thrashing (>1000 pages/sec)
        data.swap_in_rate = 600.0;
        data.swap_out_rate = 500.0;
        let (is_thrashing, severity) = data.is_thrashing();
        assert!(is_thrashing);
        assert!((severity - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_swap_thrashing_indicator() {
        let mut data = SwapData::default();

        // Idle
        data.swap_in_rate = 0.0;
        data.swap_out_rate = 0.0;
        let (symbol, desc) = data.thrashing_indicator();
        assert_eq!(symbol, "○");
        assert_eq!(desc, "idle");

        // Swapping
        data.swap_in_rate = 15.0;
        data.swap_out_rate = 0.0;
        let (symbol, desc) = data.thrashing_indicator();
        assert_eq!(symbol, "◔");
        assert_eq!(desc, "swapping");

        // Thrashing
        data.swap_in_rate = 150.0;
        data.swap_out_rate = 0.0;
        let (symbol, desc) = data.thrashing_indicator();
        assert_eq!(symbol, "◐");
        assert_eq!(desc, "thrashing");

        // Critical
        data.swap_in_rate = 1500.0;
        data.swap_out_rate = 0.0;
        let (symbol, desc) = data.thrashing_indicator();
        assert_eq!(symbol, "●");
        assert_eq!(desc, "critical");
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(512), "512B");
        assert_eq!(format_size(1024), "1.0K");
        assert_eq!(format_size(1536), "1.5K");
        assert_eq!(format_size(1048576), "1.0M");
        assert_eq!(format_size(1073741824), "1.0G");
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = SwapAnalyzer::new();
        assert_eq!(analyzer.name(), "swap");
    }

    #[test]
    fn test_analyzer_available() {
        let analyzer = SwapAnalyzer::new();
        #[cfg(target_os = "linux")]
        assert!(analyzer.available());
    }

    #[test]
    fn test_analyzer_collect() {
        let mut analyzer = SwapAnalyzer::new();
        let result = analyzer.collect();
        assert!(result.is_ok());
    }

    #[test]
    fn test_swap_type_as_str() {
        assert_eq!(SwapType::Partition.as_str(), "partition");
        assert_eq!(SwapType::File.as_str(), "file");
        assert_eq!(SwapType::Unknown.as_str(), "unknown");
    }

    #[test]
    fn test_swap_type_default() {
        let default = SwapType::default();
        assert_eq!(default, SwapType::Partition);
    }

    #[test]
    fn test_swap_device_default() {
        let device = SwapDevice::default();
        assert!(device.filename.is_empty());
        assert_eq!(device.swap_type, SwapType::Partition);
        assert_eq!(device.size, 0);
        assert_eq!(device.used, 0);
        assert_eq!(device.priority, 0);
    }

    #[test]
    fn test_swap_device_usage_percent_zero_size() {
        let device = SwapDevice {
            filename: "/dev/sda2".to_string(),
            swap_type: SwapType::Partition,
            size: 0,
            used: 0,
            priority: 0,
        };
        assert_eq!(device.usage_percent(), 0.0);
    }

    #[test]
    fn test_swap_data_usage_percent() {
        let mut data = SwapData::default();
        data.total = 1000;
        data.used = 250;
        assert!((data.usage_percent() - 25.0).abs() < 0.01);
    }

    #[test]
    fn test_swap_data_usage_percent_zero_total() {
        let data = SwapData::default();
        assert_eq!(data.usage_percent(), 0.0);
    }

    #[test]
    fn test_swap_data_device_count() {
        let mut data = SwapData::default();
        assert_eq!(data.device_count(), 0);

        data.devices.push(SwapDevice::default());
        data.devices.push(SwapDevice::default());
        assert_eq!(data.device_count(), 2);
    }

    #[test]
    fn test_swap_data_default() {
        let data = SwapData::default();
        assert!(data.devices.is_empty());
        assert_eq!(data.total, 0);
        assert_eq!(data.used, 0);
        assert_eq!(data.free, 0);
        assert_eq!(data.cached, 0);
        assert_eq!(data.swap_in_rate, 0.0);
        assert_eq!(data.swap_out_rate, 0.0);
    }

    #[test]
    fn test_format_size_tb() {
        // 1.5TB
        let tb_bytes = 1_649_267_441_664u64;
        let result = format_size(tb_bytes);
        assert!(result.contains("T"));
    }

    #[test]
    fn test_analyzer_default() {
        let analyzer = SwapAnalyzer::default();
        assert_eq!(analyzer.name(), "swap");
    }

    #[test]
    fn test_analyzer_interval() {
        let analyzer = SwapAnalyzer::new();
        assert_eq!(analyzer.interval(), Duration::from_secs(2));
    }

    #[test]
    fn test_analyzer_data() {
        let analyzer = SwapAnalyzer::new();
        let data = analyzer.data();
        assert!(data.devices.is_empty());
    }

    #[test]
    fn test_swap_device_clone() {
        let device = SwapDevice {
            filename: "/dev/sda2".to_string(),
            swap_type: SwapType::Partition,
            size: 1000,
            used: 500,
            priority: -1,
        };
        let cloned = device.clone();
        assert_eq!(cloned.filename, "/dev/sda2");
        assert_eq!(cloned.priority, -1);
    }

    #[test]
    fn test_swap_data_clone() {
        let mut data = SwapData::default();
        data.total = 1000;
        data.used = 500;
        let cloned = data.clone();
        assert_eq!(cloned.total, 1000);
        assert_eq!(cloned.used, 500);
    }

    #[test]
    fn test_swap_type_clone_copy() {
        let t = SwapType::File;
        let copied = t;
        assert_eq!(copied, SwapType::File);
    }

    #[test]
    fn test_swap_device_available_saturating() {
        // Test when used > size (shouldn't happen, but handle gracefully)
        let device = SwapDevice {
            filename: "/dev/sda2".to_string(),
            swap_type: SwapType::Partition,
            size: 100,
            used: 200, // More than size
            priority: 0,
        };
        assert_eq!(device.available(), 0); // Should saturate to 0
    }

    #[test]
    fn test_swap_device_debug() {
        let device = SwapDevice {
            filename: "/dev/sda2".to_string(),
            swap_type: SwapType::File,
            size: 1024,
            used: 512,
            priority: 5,
        };
        let debug = format!("{device:?}");
        assert!(debug.contains("sda2"));
        assert!(debug.contains("File"));
    }

    #[test]
    fn test_swap_data_debug() {
        let data = SwapData::default();
        let debug = format!("{data:?}");
        assert!(debug.contains("SwapData"));
    }

    #[test]
    fn test_swap_type_eq() {
        assert_eq!(SwapType::Partition, SwapType::Partition);
        assert_ne!(SwapType::Partition, SwapType::File);
        assert_ne!(SwapType::File, SwapType::Unknown);
    }

    #[test]
    fn test_parse_meminfo_value_empty() {
        // Test with line that doesn't have enough parts
        let result = parse_meminfo_value("SwapTotal:");
        assert_eq!(result, 0);
    }

    #[test]
    fn test_parse_meminfo_value_invalid() {
        let result = parse_meminfo_value("SwapTotal: invalid");
        assert_eq!(result, 0);
    }

    #[test]
    fn test_parse_meminfo_value_valid() {
        let result = parse_meminfo_value("SwapTotal: 1024 kB");
        assert_eq!(result, 1024 * 1024); // Converted to bytes
    }

    #[test]
    fn test_format_size_edge_cases() {
        assert_eq!(format_size(0), "0B");
        assert_eq!(format_size(1), "1B");
        assert_eq!(format_size(1023), "1023B");
    }

    // =========================================================================
    // ThrashingSeverity tests (PMAT-GAP-027)
    // =========================================================================

    #[test]
    fn test_thrashing_severity_default() {
        let severity = ThrashingSeverity::default();
        assert_eq!(severity, ThrashingSeverity::None);
    }

    #[test]
    fn test_thrashing_severity_status() {
        assert_eq!(ThrashingSeverity::None.status(), "OK");
        assert_eq!(ThrashingSeverity::Mild.status(), "Mild");
        assert_eq!(ThrashingSeverity::Moderate.status(), "Warning");
        assert_eq!(ThrashingSeverity::Severe.status(), "Critical");
    }

    #[test]
    fn test_thrashing_severity_icon() {
        assert_eq!(ThrashingSeverity::None.icon(), "○");
        assert_eq!(ThrashingSeverity::Mild.icon(), "◔");
        assert_eq!(ThrashingSeverity::Moderate.icon(), "◐");
        assert_eq!(ThrashingSeverity::Severe.icon(), "●");
    }

    #[test]
    fn test_thrashing_severity_color() {
        let none_color = ThrashingSeverity::None.color();
        assert!(none_color.g > 0.8); // Green

        let severe_color = ThrashingSeverity::Severe.color();
        assert!(severe_color.r > 0.9); // Red
    }

    #[test]
    fn test_thrashing_severity_is_thrashing() {
        assert!(!ThrashingSeverity::None.is_thrashing());
        assert!(ThrashingSeverity::Mild.is_thrashing());
        assert!(ThrashingSeverity::Moderate.is_thrashing());
        assert!(ThrashingSeverity::Severe.is_thrashing());
    }

    #[test]
    fn test_detect_thrashing_none() {
        let mut data = SwapData::default();
        data.swap_in_rate = 50.0;
        data.swap_out_rate = 40.0; // 90 total < 100
        assert_eq!(data.detect_thrashing(), ThrashingSeverity::None);
    }

    #[test]
    fn test_detect_thrashing_mild() {
        let mut data = SwapData::default();
        data.swap_in_rate = 200.0;
        data.swap_out_rate = 100.0; // 300 total
        assert_eq!(data.detect_thrashing(), ThrashingSeverity::Mild);
    }

    #[test]
    fn test_detect_thrashing_moderate() {
        let mut data = SwapData::default();
        data.swap_in_rate = 400.0;
        data.swap_out_rate = 200.0; // 600 total
        assert_eq!(data.detect_thrashing(), ThrashingSeverity::Moderate);
    }

    #[test]
    fn test_detect_thrashing_severe() {
        let mut data = SwapData::default();
        data.swap_in_rate = 800.0;
        data.swap_out_rate = 500.0; // 1300 total
        assert_eq!(data.detect_thrashing(), ThrashingSeverity::Severe);
    }

    #[test]
    fn test_detect_thrashing_with_psi_none() {
        let data = SwapData::default();
        assert_eq!(data.detect_thrashing_with_psi(5.0), ThrashingSeverity::None);
    }

    #[test]
    fn test_detect_thrashing_with_psi_mild() {
        let data = SwapData::default();
        assert_eq!(
            data.detect_thrashing_with_psi(15.0),
            ThrashingSeverity::Mild
        );
    }

    #[test]
    fn test_detect_thrashing_with_psi_severe() {
        let data = SwapData::default();
        assert_eq!(
            data.detect_thrashing_with_psi(60.0),
            ThrashingSeverity::Severe
        );
    }

    // =========================================================================
    // ZramStats tests (PMAT-GAP-028)
    // =========================================================================

    #[test]
    fn test_zram_stats_default() {
        let stats = ZramStats::default();
        assert_eq!(stats.orig_data_size, 0);
        assert_eq!(stats.compr_data_size, 0);
        assert_eq!(stats.mem_used_total, 0);
        assert!(stats.algorithm.is_empty());
    }

    #[test]
    fn test_zram_stats_compression_ratio() {
        let stats = ZramStats {
            orig_data_size: 2_000_000_000,
            compr_data_size: 1_000_000_000,
            mem_used_total: 1_100_000_000,
            algorithm: "lz4".to_string(),
        };
        assert!((stats.compression_ratio() - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_zram_stats_compression_ratio_zero() {
        let stats = ZramStats::default();
        assert_eq!(stats.compression_ratio(), 0.0);
    }

    #[test]
    fn test_zram_stats_is_active() {
        let inactive = ZramStats::default();
        assert!(!inactive.is_active());

        let active = ZramStats {
            orig_data_size: 1000,
            compr_data_size: 500,
            mem_used_total: 600,
            algorithm: "zstd".to_string(),
        };
        assert!(active.is_active());
    }

    #[test]
    fn test_zram_stats_format_display() {
        let stats = ZramStats {
            orig_data_size: 2_000_000_000,
            compr_data_size: 1_000_000_000,
            mem_used_total: 1_100_000_000,
            algorithm: "lz4".to_string(),
        };
        let display = stats.format_display();
        assert!(display.contains("2.0x"));
        assert!(display.contains("lz4"));
    }

    #[test]
    fn test_zram_stats_debug() {
        let stats = ZramStats {
            orig_data_size: 1000,
            compr_data_size: 500,
            mem_used_total: 600,
            algorithm: "zstd".to_string(),
        };
        let debug = format!("{:?}", stats);
        assert!(debug.contains("ZramStats"));
        assert!(debug.contains("zstd"));
    }

    #[test]
    fn test_zram_stats_clone() {
        let stats = ZramStats {
            orig_data_size: 1000,
            compr_data_size: 500,
            mem_used_total: 600,
            algorithm: "lzo".to_string(),
        };
        let cloned = stats.clone();
        assert_eq!(cloned.orig_data_size, 1000);
        assert_eq!(cloned.algorithm, "lzo");
    }

    #[test]
    fn test_analyzer_has_zram() {
        let analyzer = SwapAnalyzer::new();
        // New analyzer has no ZRAM data
        assert!(!analyzer.has_zram());
    }

    #[test]
    fn test_analyzer_zram_compression_ratio() {
        let analyzer = SwapAnalyzer::new();
        assert_eq!(analyzer.zram_compression_ratio(), 0.0);
    }

    #[test]
    fn test_analyzer_detect_thrashing() {
        let analyzer = SwapAnalyzer::new();
        assert_eq!(analyzer.detect_thrashing(), ThrashingSeverity::None);
    }

    #[test]
    fn test_thrashing_severity_clone_copy() {
        let severity = ThrashingSeverity::Moderate;
        let copied = severity;
        assert_eq!(copied, ThrashingSeverity::Moderate);
    }

    #[test]
    fn test_thrashing_severity_debug() {
        let severity = ThrashingSeverity::Severe;
        let debug = format!("{:?}", severity);
        assert!(debug.contains("Severe"));
    }
}
