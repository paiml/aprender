//! Memory panel rendering and utilities.
//!
//! Provides memory panel title building, memory stats calculation,
//! and helper functions for rendering memory metrics.

use super::super::helpers::format_bytes;
use presentar_core::Color;

// =============================================================================
// MEMORY TITLE BUILDING
// =============================================================================

/// Build memory panel title string.
///
/// Format: "Memory │ 8.5G / 32G (26%) │ Swap: 0.1G / 8G"
#[must_use]
pub fn build_memory_title(
    used_bytes: u64,
    total_bytes: u64,
    swap_used: u64,
    swap_total: u64,
) -> String {
    let used_pct = if total_bytes > 0 {
        (used_bytes as f64 / total_bytes as f64) * 100.0
    } else {
        0.0
    };

    let used_str = format_bytes(used_bytes);
    let total_str = format_bytes(total_bytes);

    if swap_total > 0 {
        let swap_used_str = format_bytes(swap_used);
        let swap_total_str = format_bytes(swap_total);
        format!(
            "Memory │ {} / {} ({:.0}%) │ Swap: {} / {}",
            used_str, total_str, used_pct, swap_used_str, swap_total_str
        )
    } else {
        format!("Memory │ {} / {} ({:.0}%)", used_str, total_str, used_pct)
    }
}

/// Build compact memory title for narrow panels.
///
/// Format: "Memory │ 8.5G / 32G"
#[must_use]
pub fn build_memory_title_compact(used_bytes: u64, total_bytes: u64) -> String {
    format!(
        "Memory │ {} / {}",
        format_bytes(used_bytes),
        format_bytes(total_bytes)
    )
}

// =============================================================================
// MEMORY STATS
// =============================================================================

/// Memory statistics for display.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryStats {
    /// Used memory in GB
    pub used_gb: f64,
    /// Cached memory in GB
    pub cached_gb: f64,
    /// Free memory in GB
    pub free_gb: f64,
    /// Total memory in GB
    pub total_gb: f64,
}

impl MemoryStats {
    /// Create from raw byte values.
    #[must_use]
    pub fn new(used: u64, cached: u64, available: u64, total: u64) -> Self {
        const GB: f64 = 1024.0 * 1024.0 * 1024.0;
        Self {
            used_gb: used as f64 / GB,
            cached_gb: cached as f64 / GB,
            free_gb: available as f64 / GB,
            total_gb: total as f64 / GB,
        }
    }

    /// Calculate used percentage.
    #[must_use]
    pub fn used_percent(&self) -> f64 {
        if self.total_gb > 0.0 {
            (self.used_gb / self.total_gb) * 100.0
        } else {
            0.0
        }
    }

    /// Calculate cached percentage.
    #[must_use]
    pub fn cached_percent(&self) -> f64 {
        if self.total_gb > 0.0 {
            (self.cached_gb / self.total_gb) * 100.0
        } else {
            0.0
        }
    }

    /// Calculate free percentage.
    #[must_use]
    pub fn free_percent(&self) -> f64 {
        if self.total_gb > 0.0 {
            (self.free_gb / self.total_gb) * 100.0
        } else {
            0.0
        }
    }
}

// =============================================================================
// MEMORY COLORS
// =============================================================================

/// Get color for memory usage percentage.
#[must_use]
pub fn memory_usage_color(percent: f64) -> Color {
    if percent > 90.0 {
        Color::new(1.0, 0.3, 0.3, 1.0) // Critical red
    } else if percent > 75.0 {
        Color::new(1.0, 0.6, 0.2, 1.0) // Warning orange
    } else if percent > 50.0 {
        Color::new(1.0, 0.8, 0.2, 1.0) // Yellow
    } else {
        Color::new(0.3, 0.9, 0.3, 1.0) // Green
    }
}

/// Get color for swap usage percentage.
#[must_use]
pub fn swap_usage_color(percent: f64) -> Color {
    if percent > 80.0 {
        Color::new(1.0, 0.3, 0.3, 1.0) // Critical
    } else if percent > 50.0 {
        Color::new(1.0, 0.5, 0.3, 1.0) // Warning
    } else if percent > 25.0 {
        Color::new(1.0, 0.8, 0.3, 1.0) // Caution
    } else {
        Color::new(0.4, 0.8, 0.4, 1.0) // Normal
    }
}

// =============================================================================
// ZRAM STATS
// =============================================================================

/// ZRAM compression statistics.
#[derive(Debug, Clone, Default)]
pub struct ZramStats {
    /// Original (uncompressed) data size in bytes
    pub orig_data_size: u64,
    /// Compressed data size in bytes
    pub compr_data_size: u64,
    /// Compression algorithm (lzo, lz4, zstd, etc.)
    pub algorithm: String,
}

impl ZramStats {
    /// Get compression ratio (original / compressed).
    #[must_use]
    pub fn ratio(&self) -> f64 {
        if self.compr_data_size == 0 {
            1.0
        } else {
            self.orig_data_size as f64 / self.compr_data_size as f64
        }
    }

    /// Check if ZRAM is active.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.orig_data_size > 0
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // build_memory_title tests
    // =========================================================================

    #[test]
    fn test_build_memory_title_with_swap() {
        let title = build_memory_title(
            8 * 1024 * 1024 * 1024,  // 8GB used
            32 * 1024 * 1024 * 1024, // 32GB total
            1024 * 1024 * 1024,      // 1GB swap used
            8 * 1024 * 1024 * 1024,  // 8GB swap total
        );
        assert!(title.contains("Memory"));
        assert!(title.contains("8.0G"));
        assert!(title.contains("32.0G"));
        assert!(title.contains("25%"));
        assert!(title.contains("Swap"));
    }

    #[test]
    fn test_build_memory_title_no_swap() {
        let title = build_memory_title(4 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024, 0, 0);
        assert!(!title.contains("Swap"));
        assert!(title.contains("25%"));
    }

    #[test]
    fn test_build_memory_title_zero() {
        let title = build_memory_title(0, 0, 0, 0);
        assert!(title.contains("0%"));
    }

    #[test]
    fn test_build_memory_title_full() {
        let total = 16 * 1024 * 1024 * 1024_u64;
        let title = build_memory_title(total, total, 0, 0);
        assert!(title.contains("100%"));
    }

    // =========================================================================
    // build_memory_title_compact tests
    // =========================================================================

    #[test]
    fn test_build_memory_title_compact() {
        let title = build_memory_title_compact(8 * 1024 * 1024 * 1024, 32 * 1024 * 1024 * 1024);
        assert!(title.contains("Memory"));
        assert!(title.contains("8.0G"));
        assert!(title.contains("32.0G"));
        assert!(!title.contains("%")); // Compact doesn't show percentage
    }

    // =========================================================================
    // MemoryStats tests
    // =========================================================================

    #[test]
    fn test_memory_stats_new() {
        let gb = 1024 * 1024 * 1024_u64;
        let stats = MemoryStats::new(8 * gb, 4 * gb, 20 * gb, 32 * gb);
        assert!((stats.used_gb - 8.0).abs() < 0.01);
        assert!((stats.cached_gb - 4.0).abs() < 0.01);
        assert!((stats.free_gb - 20.0).abs() < 0.01);
        assert!((stats.total_gb - 32.0).abs() < 0.01);
    }

    #[test]
    fn test_memory_stats_used_percent() {
        let gb = 1024 * 1024 * 1024_u64;
        let stats = MemoryStats::new(8 * gb, 0, 24 * gb, 32 * gb);
        assert!((stats.used_percent() - 25.0).abs() < 0.1);
    }

    #[test]
    fn test_memory_stats_cached_percent() {
        let gb = 1024 * 1024 * 1024_u64;
        let stats = MemoryStats::new(0, 16 * gb, 16 * gb, 32 * gb);
        assert!((stats.cached_percent() - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_memory_stats_free_percent() {
        let gb = 1024 * 1024 * 1024_u64;
        let stats = MemoryStats::new(8 * gb, 8 * gb, 16 * gb, 32 * gb);
        assert!((stats.free_percent() - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_memory_stats_zero_total() {
        let stats = MemoryStats::new(0, 0, 0, 0);
        assert_eq!(stats.used_percent(), 0.0);
        assert_eq!(stats.cached_percent(), 0.0);
        assert_eq!(stats.free_percent(), 0.0);
    }

    #[test]
    fn test_memory_stats_derive_debug() {
        let stats = MemoryStats::new(1024, 512, 512, 2048);
        let debug = format!("{:?}", stats);
        assert!(debug.contains("MemoryStats"));
    }

    #[test]
    fn test_memory_stats_derive_clone() {
        let stats = MemoryStats::new(1024, 512, 512, 2048);
        let cloned = stats.clone();
        assert_eq!(stats, cloned);
    }

    // =========================================================================
    // memory_usage_color tests
    // =========================================================================

    #[test]
    fn test_memory_usage_color_low() {
        let color = memory_usage_color(25.0);
        assert!(color.g > 0.8, "Low usage should be green");
    }

    #[test]
    fn test_memory_usage_color_medium() {
        let color = memory_usage_color(60.0);
        assert!(color.r > 0.9, "Medium usage should be yellow");
        assert!(color.g > 0.7);
    }

    #[test]
    fn test_memory_usage_color_high() {
        let color = memory_usage_color(80.0);
        assert!(color.r > 0.9, "High usage should be orange");
        assert!(color.g > 0.5 && color.g < 0.7);
    }

    #[test]
    fn test_memory_usage_color_critical() {
        let color = memory_usage_color(95.0);
        assert!(color.r > 0.9, "Critical should be red");
        assert!(color.g < 0.5);
    }

    // =========================================================================
    // swap_usage_color tests
    // =========================================================================

    #[test]
    fn test_swap_usage_color_low() {
        let color = swap_usage_color(10.0);
        assert!(color.g > 0.7, "Low swap should be green-ish");
    }

    #[test]
    fn test_swap_usage_color_medium() {
        let color = swap_usage_color(40.0);
        assert!(color.r > 0.9);
        assert!(color.g > 0.7);
    }

    #[test]
    fn test_swap_usage_color_high() {
        let color = swap_usage_color(60.0);
        assert!(color.r > 0.9);
    }

    #[test]
    fn test_swap_usage_color_critical() {
        let color = swap_usage_color(90.0);
        assert!(color.r > 0.9);
        assert!(color.g < 0.5);
    }

    // =========================================================================
    // ZramStats tests
    // =========================================================================

    #[test]
    fn test_zram_stats_default() {
        let stats = ZramStats::default();
        assert_eq!(stats.orig_data_size, 0);
        assert_eq!(stats.compr_data_size, 0);
        assert!(stats.algorithm.is_empty());
    }

    #[test]
    fn test_zram_stats_ratio_2x() {
        let stats = ZramStats {
            orig_data_size: 1000,
            compr_data_size: 500,
            algorithm: "lz4".to_string(),
        };
        assert!((stats.ratio() - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_zram_stats_ratio_no_compression() {
        let stats = ZramStats {
            orig_data_size: 1000,
            compr_data_size: 1000,
            algorithm: "none".to_string(),
        };
        assert!((stats.ratio() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_zram_stats_ratio_zero_compressed() {
        let stats = ZramStats {
            orig_data_size: 1000,
            compr_data_size: 0,
            algorithm: "lz4".to_string(),
        };
        assert!(
            (stats.ratio() - 1.0).abs() < 0.01,
            "Zero compressed should return 1.0"
        );
    }

    #[test]
    fn test_zram_stats_is_active() {
        let active = ZramStats {
            orig_data_size: 1000,
            compr_data_size: 500,
            algorithm: "lz4".to_string(),
        };
        assert!(active.is_active());

        let inactive = ZramStats::default();
        assert!(!inactive.is_active());
    }

    #[test]
    fn test_zram_stats_derive_debug() {
        let stats = ZramStats {
            orig_data_size: 1000,
            compr_data_size: 500,
            algorithm: "zstd".to_string(),
        };
        let debug = format!("{:?}", stats);
        assert!(debug.contains("ZramStats"));
        assert!(debug.contains("zstd"));
    }

    #[test]
    fn test_zram_stats_derive_clone() {
        let stats = ZramStats {
            orig_data_size: 1000,
            compr_data_size: 500,
            algorithm: "lz4".to_string(),
        };
        let cloned = stats.clone();
        assert_eq!(cloned.orig_data_size, stats.orig_data_size);
        assert_eq!(cloned.algorithm, stats.algorithm);
    }
}
