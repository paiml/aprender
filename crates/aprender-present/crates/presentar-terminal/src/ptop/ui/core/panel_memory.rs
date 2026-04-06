//! Memory Panel Rendering Components
//!
//! Extracted from render.rs to reduce cyclomatic complexity.
//! Contains helper functions for memory panel statistics, colors, and formatting.

#![allow(dead_code)]

use presentar_core::Color;

/// Memory statistics for rendering (GB values).
#[derive(Debug, Clone, Copy)]
pub(crate) struct MemoryStats {
    /// Used memory in GB
    pub used_gb: f64,
    /// Cached memory in GB
    pub cached_gb: f64,
    /// Free/available memory in GB
    pub free_gb: f64,
    /// Total memory in GB
    pub total_gb: f64,
}

impl MemoryStats {
    /// Create memory stats from raw byte values.
    #[must_use]
    pub fn from_bytes(used: u64, cached: u64, available: u64, total: u64) -> Self {
        let gb = |b: u64| b as f64 / 1024.0 / 1024.0 / 1024.0;
        Self {
            used_gb: gb(used),
            cached_gb: gb(cached),
            free_gb: gb(available),
            total_gb: gb(total),
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

/// Get color for swap usage percentage.
///
/// - Red: swap > 50% (critical)
/// - Yellow: swap > 10% (warning)
/// - Green: swap <= 10% (normal)
#[must_use]
pub(crate) fn swap_color(pct: f64) -> Color {
    if pct > 50.0 {
        Color::new(1.0, 0.3, 0.3, 1.0) // Red
    } else if pct > 10.0 {
        Color::new(1.0, 0.8, 0.2, 1.0) // Yellow
    } else {
        Color::new(0.3, 0.9, 0.3, 1.0) // Green
    }
}

/// Get swap thrashing indicator and color based on severity.
///
/// Returns (indicator_char, color) tuple.
#[must_use]
pub(crate) fn thrashing_indicator(severity: f64) -> (&'static str, Color) {
    if severity >= 1.0 {
        ("●", Color::new(1.0, 0.3, 0.3, 1.0)) // Red - critical
    } else if severity >= 0.7 {
        ("◐", Color::new(1.0, 0.6, 0.2, 1.0)) // Orange - thrashing
    } else if severity >= 0.4 {
        ("◔", Color::new(1.0, 0.8, 0.2, 1.0)) // Yellow - swapping
    } else {
        ("○", Color::new(0.3, 0.9, 0.3, 1.0)) // Green - normal
    }
}

/// Standard dim color for labels.
pub(crate) const DIM_COLOR: Color = Color {
    r: 0.3,
    g: 0.3,
    b: 0.3,
    a: 1.0,
};

/// Cyan color for cached memory.
pub(crate) const CACHED_COLOR: Color = Color {
    r: 0.3,
    g: 0.8,
    b: 0.9,
    a: 1.0,
};

/// Blue color for free memory.
pub(crate) const FREE_COLOR: Color = Color {
    r: 0.4,
    g: 0.4,
    b: 0.9,
    a: 1.0,
};

/// Magenta color for ZRAM.
pub(crate) const ZRAM_COLOR: Color = Color {
    r: 0.8,
    g: 0.4,
    b: 1.0,
    a: 1.0,
};

/// Green color for compression ratios.
pub(crate) const RATIO_COLOR: Color = Color {
    r: 0.3,
    g: 0.9,
    b: 0.3,
    a: 1.0,
};

/// Build memory bar (filled + empty blocks).
#[must_use]
pub(crate) fn build_memory_bar(pct: f64, bar_width: usize) -> String {
    let filled = ((pct / 100.0) * bar_width as f64) as usize;
    "█".repeat(filled.min(bar_width)) + &"░".repeat(bar_width.saturating_sub(filled))
}

/// Format memory value as human-readable string.
#[must_use]
#[allow(dead_code)]
pub(crate) fn format_memory_value(gb: f64) -> String {
    if gb >= 1024.0 {
        format!("{:.1}T", gb / 1024.0)
    } else if gb >= 1.0 {
        format!("{gb:.1}G")
    } else {
        format!("{:.0}M", gb * 1024.0)
    }
}

/// Format memory row line for display.
#[must_use]
#[allow(dead_code)]
pub(crate) fn format_memory_row(label: &str, value_gb: f64, pct: f64, bar_width: usize) -> String {
    let bar = build_memory_bar(pct, bar_width);
    format!("{label:>6} {:>5.1}G {bar} {:>5.1}%", value_gb, pct)
}

/// Calculate bar segment widths for stacked memory bar.
#[must_use]
#[allow(dead_code)]
pub(crate) fn calculate_bar_segments(
    used_pct: f64,
    cached_pct: f64,
    total_width: usize,
) -> (usize, usize, usize) {
    let used_chars = ((used_pct / 100.0) * total_width as f64) as usize;
    let cached_chars = ((cached_pct / 100.0) * total_width as f64) as usize;
    let free_chars = total_width.saturating_sub(used_chars + cached_chars);
    (used_chars, cached_chars, free_chars)
}

// =============================================================================
// PSI Memory Pressure Helpers
// =============================================================================

/// Get PSI memory pressure indicator and color.
///
/// Returns (symbol, color) based on memory pressure levels.
#[must_use]
pub(crate) fn psi_memory_indicator(some_pct: f64, full_pct: f64) -> (&'static str, Color) {
    if some_pct > 20.0 || full_pct > 5.0 {
        ("●", Color::new(1.0, 0.3, 0.3, 1.0)) // Red - critical
    } else if some_pct > 10.0 || full_pct > 1.0 {
        ("◐", Color::new(1.0, 0.8, 0.2, 1.0)) // Yellow - warning
    } else {
        ("○", Color::new(0.3, 0.9, 0.3, 1.0)) // Green - healthy
    }
}

/// Format PSI memory pressure line.
#[must_use]
pub(crate) fn format_psi_line(some_pct: f64, full_pct: f64) -> String {
    let (symbol, _) = psi_memory_indicator(some_pct, full_pct);
    format!(
        "   PSI {symbol} {:>5.1}% some {:>5.1}% full",
        some_pct, full_pct
    )
}

// =============================================================================
// ZRAM Helpers
// =============================================================================

/// ZRAM statistics for rendering.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ZramDisplay {
    /// Original data size in GB
    pub orig_gb: f64,
    /// Compressed data size in GB
    pub compr_gb: f64,
    /// Compression ratio
    pub ratio: f64,
}

impl ZramDisplay {
    /// Create ZRAM display from byte values.
    #[must_use]
    pub fn from_bytes(orig_size: u64, compr_size: u64) -> Self {
        let gb = |b: u64| b as f64 / 1024.0 / 1024.0 / 1024.0;
        let orig_gb = gb(orig_size);
        let compr_gb = gb(compr_size);
        let ratio = if compr_gb > 0.0 {
            orig_gb / compr_gb
        } else {
            0.0
        };
        Self {
            orig_gb,
            compr_gb,
            ratio,
        }
    }

    /// Format size as human-readable string.
    #[must_use]
    pub fn format_size(gb: f64) -> String {
        if gb >= 1024.0 {
            format!("{:.1}T", gb / 1024.0)
        } else {
            format!("{gb:.1}G")
        }
    }

    /// Format ZRAM info for title suffix.
    #[must_use]
    pub fn title_suffix(&self) -> String {
        format!(" │ ZRAM:{:.1}x", self.ratio)
    }
}

/// Format ZRAM row line for display.
///
/// Format: "  ZRAM 2.5G→1.0G 2.5x lz4"
#[must_use]
pub(crate) fn format_zram_row(orig_gb: f64, compr_gb: f64, ratio: f64, algo: &str) -> String {
    let orig_str = ZramDisplay::format_size(orig_gb);
    let compr_str = ZramDisplay::format_size(compr_gb);
    format!("  ZRAM {orig_str}→{compr_str} {ratio:.1}x {algo}")
}

// =============================================================================
// Swap Thrashing Helpers
// =============================================================================

/// Format swap thrashing info for display.
#[must_use]
pub(crate) fn format_thrashing_info(severity: f64, swap_in: f64, swap_out: f64) -> String {
    let (indicator, _) = thrashing_indicator(severity);
    format!(" {indicator} I:{swap_in:.0}/O:{swap_out:.0}")
}

/// Check if swap activity is noteworthy.
#[must_use]
pub(crate) fn has_swap_activity(is_thrashing: bool, swap_in_rate: f64, swap_out_rate: f64) -> bool {
    is_thrashing || swap_in_rate > 0.0 || swap_out_rate > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    // F-MEM-001: MemoryStats converts bytes to GB
    #[test]
    fn test_memory_stats_conversion() {
        let stats = MemoryStats::from_bytes(
            1024 * 1024 * 1024,     // 1 GB used
            512 * 1024 * 1024,      // 0.5 GB cached
            2 * 1024 * 1024 * 1024, // 2 GB available
            4 * 1024 * 1024 * 1024, // 4 GB total
        );
        assert!((stats.used_gb - 1.0).abs() < 0.01);
        assert!((stats.cached_gb - 0.5).abs() < 0.01);
        assert!((stats.free_gb - 2.0).abs() < 0.01);
        assert!((stats.total_gb - 4.0).abs() < 0.01);
    }

    // F-MEM-002: MemoryStats calculates used percent
    #[test]
    fn test_memory_stats_used_percent() {
        let stats = MemoryStats::from_bytes(
            1024 * 1024 * 1024,
            0,
            3 * 1024 * 1024 * 1024,
            4 * 1024 * 1024 * 1024,
        );
        assert!((stats.used_percent() - 25.0).abs() < 0.1);
    }

    // F-MEM-003: MemoryStats handles zero total
    #[test]
    fn test_memory_stats_zero_total() {
        let stats = MemoryStats::from_bytes(0, 0, 0, 0);
        assert_eq!(stats.used_percent(), 0.0);
        assert_eq!(stats.cached_percent(), 0.0);
        assert_eq!(stats.free_percent(), 0.0);
    }

    // F-MEM-004: Swap color red when critical
    #[test]
    fn test_swap_color_red() {
        let color = swap_color(75.0);
        assert!(color.r > 0.8 && color.g < 0.5, "Should be red above 50%");
    }

    // F-MEM-005: Swap color yellow when warning
    #[test]
    fn test_swap_color_yellow() {
        let color = swap_color(30.0);
        assert!(color.r > 0.8 && color.g > 0.5, "Should be yellow 10-50%");
    }

    // F-MEM-006: Swap color green when normal
    #[test]
    fn test_swap_color_green() {
        let color = swap_color(5.0);
        assert!(color.g > 0.8 && color.r < 0.5, "Should be green below 10%");
    }

    // F-MEM-007: Thrashing indicator critical
    #[test]
    fn test_thrashing_critical() {
        let (indicator, color) = thrashing_indicator(1.0);
        assert_eq!(indicator, "●");
        assert!(color.r > 0.8);
    }

    // F-MEM-008: Thrashing indicator thrashing
    #[test]
    fn test_thrashing_thrashing() {
        let (indicator, _) = thrashing_indicator(0.8);
        assert_eq!(indicator, "◐");
    }

    // F-MEM-009: Thrashing indicator swapping
    #[test]
    fn test_thrashing_swapping() {
        let (indicator, _) = thrashing_indicator(0.5);
        assert_eq!(indicator, "◔");
    }

    // F-MEM-010: Thrashing indicator normal
    #[test]
    fn test_thrashing_normal() {
        let (indicator, color) = thrashing_indicator(0.2);
        assert_eq!(indicator, "○");
        assert!(color.g > 0.8);
    }

    // F-MEM-011: DIM_COLOR is gray
    #[test]
    fn test_dim_color_gray() {
        assert!(DIM_COLOR.r == DIM_COLOR.g && DIM_COLOR.g == DIM_COLOR.b);
        assert!(DIM_COLOR.r < 0.5);
    }

    // F-MEM-012: CACHED_COLOR is cyan
    #[test]
    fn test_cached_color_cyan() {
        assert!(CACHED_COLOR.g > CACHED_COLOR.r);
        assert!(CACHED_COLOR.b > CACHED_COLOR.r);
    }

    // F-MEM-013: FREE_COLOR is blue
    #[test]
    fn test_free_color_blue() {
        assert!(FREE_COLOR.b > FREE_COLOR.r);
        assert!(FREE_COLOR.b > FREE_COLOR.g);
    }

    // F-MEM-014: ZRAM_COLOR is magenta
    #[test]
    fn test_zram_color_magenta() {
        assert!(ZRAM_COLOR.r > 0.5 && ZRAM_COLOR.b > 0.8);
    }

    // F-MEM-015: RATIO_COLOR is green
    #[test]
    fn test_ratio_color_green() {
        assert!(RATIO_COLOR.g > 0.8);
    }

    // F-MEM-016: Memory bar full
    #[test]
    fn test_memory_bar_full() {
        let bar = build_memory_bar(100.0, 10);
        assert_eq!(bar.chars().filter(|&c| c == '█').count(), 10);
    }

    // F-MEM-017: Memory bar empty
    #[test]
    fn test_memory_bar_empty() {
        let bar = build_memory_bar(0.0, 10);
        assert_eq!(bar.chars().filter(|&c| c == '░').count(), 10);
    }

    // F-MEM-018: Memory bar half
    #[test]
    fn test_memory_bar_half() {
        let bar = build_memory_bar(50.0, 10);
        assert_eq!(bar.chars().filter(|&c| c == '█').count(), 5);
        assert_eq!(bar.chars().filter(|&c| c == '░').count(), 5);
    }

    // F-MEM-019: Format memory value GB
    #[test]
    fn test_format_memory_gb() {
        let formatted = format_memory_value(16.5);
        assert!(formatted.contains("16.5G"));
    }

    // F-MEM-020: Format memory value TB
    #[test]
    fn test_format_memory_tb() {
        let formatted = format_memory_value(2048.0);
        assert!(formatted.contains("T"));
    }

    // F-MEM-021: Format memory value MB
    #[test]
    fn test_format_memory_mb() {
        let formatted = format_memory_value(0.5);
        assert!(formatted.contains("M"));
    }

    // F-MEM-022: Format memory row
    #[test]
    fn test_format_memory_row() {
        let row = format_memory_row("Used", 8.5, 50.0, 10);
        assert!(row.contains("Used"));
        assert!(row.contains("8.5G"));
        assert!(row.contains("50.0%"));
    }

    // F-MEM-023: Calculate bar segments
    #[test]
    fn test_bar_segments() {
        let (used, cached, free) = calculate_bar_segments(50.0, 25.0, 20);
        assert_eq!(used, 10);
        assert_eq!(cached, 5);
        assert_eq!(free, 5);
    }

    // F-MEM-024: Swap color boundary at 50%
    #[test]
    fn test_swap_boundary_50() {
        let below = swap_color(49.0);
        let above = swap_color(51.0);
        assert!(above.r > below.r || below.g > above.g);
    }

    // F-MEM-025: Swap color boundary at 10%
    #[test]
    fn test_swap_boundary_10() {
        let below = swap_color(9.0);
        let above = swap_color(11.0);
        assert!(above.r > below.r);
    }

    // F-MEM-026: Thrashing boundary at 0.7
    #[test]
    fn test_thrashing_boundary_07() {
        let (below, _) = thrashing_indicator(0.69);
        let (above, _) = thrashing_indicator(0.71);
        assert_ne!(below, above);
    }

    // F-MEM-027: Thrashing boundary at 0.4
    #[test]
    fn test_thrashing_boundary_04() {
        let (below, _) = thrashing_indicator(0.39);
        let (above, _) = thrashing_indicator(0.41);
        assert_ne!(below, above);
    }

    // F-MEM-028: Memory bar handles overflow
    #[test]
    fn test_memory_bar_overflow() {
        let bar = build_memory_bar(150.0, 10);
        assert_eq!(bar.chars().count(), 10);
    }

    // F-MEM-029: MemoryStats cached percent
    #[test]
    fn test_memory_stats_cached_percent() {
        let stats = MemoryStats::from_bytes(
            0,
            2 * 1024 * 1024 * 1024, // 2 GB cached
            0,
            8 * 1024 * 1024 * 1024, // 8 GB total
        );
        assert!((stats.cached_percent() - 25.0).abs() < 0.1);
    }

    // F-MEM-030: MemoryStats free percent
    #[test]
    fn test_memory_stats_free_percent() {
        let stats = MemoryStats::from_bytes(
            0,
            0,
            4 * 1024 * 1024 * 1024,  // 4 GB free
            16 * 1024 * 1024 * 1024, // 16 GB total
        );
        assert!((stats.free_percent() - 25.0).abs() < 0.1);
    }

    // F-MEM-031: Bar segments sum to width
    #[test]
    fn test_bar_segments_sum() {
        let (used, cached, free) = calculate_bar_segments(33.0, 33.0, 30);
        assert_eq!(used + cached + free, 30);
    }

    // F-MEM-032: Bar segments zero percentages
    #[test]
    fn test_bar_segments_zero() {
        let (used, cached, free) = calculate_bar_segments(0.0, 0.0, 20);
        assert_eq!(used, 0);
        assert_eq!(cached, 0);
        assert_eq!(free, 20);
    }

    // F-MEM-033: Format memory value zero
    #[test]
    fn test_format_memory_zero() {
        let formatted = format_memory_value(0.0);
        assert!(formatted.contains("0"));
    }

    // F-MEM-034: Memory row formatting alignment
    #[test]
    fn test_memory_row_alignment() {
        let row1 = format_memory_row("Used", 1.0, 10.0, 10);
        let row2 = format_memory_row("Free", 10.0, 90.0, 10);
        // Both should have consistent width (label is right-aligned to 6 chars)
        assert!(row1.contains("  Used"));
        assert!(row2.contains("  Free"));
    }

    // F-MEM-035: Thrashing severity zero
    #[test]
    fn test_thrashing_zero() {
        let (indicator, color) = thrashing_indicator(0.0);
        assert_eq!(indicator, "○");
        assert!(color.g > 0.8);
    }

    // =========================================================================
    // PSI Memory Pressure Tests (F-MEM-036 to F-MEM-042)
    // =========================================================================

    // F-MEM-036: PSI critical pressure
    #[test]
    fn test_psi_critical() {
        let (symbol, color) = psi_memory_indicator(25.0, 0.0);
        assert_eq!(symbol, "●");
        assert!(color.r > 0.8);
    }

    // F-MEM-037: PSI critical full pressure
    #[test]
    fn test_psi_critical_full() {
        let (symbol, _) = psi_memory_indicator(0.0, 10.0);
        assert_eq!(symbol, "●");
    }

    // F-MEM-038: PSI warning pressure
    #[test]
    fn test_psi_warning() {
        let (symbol, color) = psi_memory_indicator(15.0, 0.0);
        assert_eq!(symbol, "◐");
        assert!(color.r > 0.8 && color.g > 0.5);
    }

    // F-MEM-039: PSI healthy
    #[test]
    fn test_psi_healthy() {
        let (symbol, color) = psi_memory_indicator(5.0, 0.0);
        assert_eq!(symbol, "○");
        assert!(color.g > 0.8);
    }

    // F-MEM-040: Format PSI line
    #[test]
    fn test_format_psi_line() {
        let line = format_psi_line(15.5, 2.5);
        assert!(line.contains("PSI"));
        assert!(line.contains("15.5%"));
        assert!(line.contains("2.5%"));
    }

    // F-MEM-041: PSI boundary at 20%
    #[test]
    fn test_psi_boundary_20() {
        let (below, _) = psi_memory_indicator(19.0, 0.0);
        let (above, _) = psi_memory_indicator(21.0, 0.0);
        assert_ne!(below, above);
    }

    // F-MEM-042: PSI boundary at 10%
    #[test]
    fn test_psi_boundary_10() {
        let (below, _) = psi_memory_indicator(9.0, 0.0);
        let (above, _) = psi_memory_indicator(11.0, 0.0);
        assert_ne!(below, above);
    }

    // =========================================================================
    // ZRAM Display Tests (F-MEM-043 to F-MEM-050)
    // =========================================================================

    // F-MEM-043: ZramDisplay from bytes
    #[test]
    fn test_zram_from_bytes() {
        let zram = ZramDisplay::from_bytes(
            2 * 1024 * 1024 * 1024, // 2 GB original
            1024 * 1024 * 1024,     // 1 GB compressed
        );
        assert!((zram.orig_gb - 2.0).abs() < 0.01);
        assert!((zram.compr_gb - 1.0).abs() < 0.01);
        assert!((zram.ratio - 2.0).abs() < 0.01);
    }

    // F-MEM-044: ZramDisplay format size GB
    #[test]
    fn test_zram_format_gb() {
        let size = ZramDisplay::format_size(2.5);
        assert_eq!(size, "2.5G");
    }

    // F-MEM-045: ZramDisplay format size TB
    #[test]
    fn test_zram_format_tb() {
        let size = ZramDisplay::format_size(2048.0);
        assert!(size.contains("T"));
    }

    // F-MEM-046: ZramDisplay title suffix
    #[test]
    fn test_zram_title_suffix() {
        let zram = ZramDisplay {
            orig_gb: 4.0,
            compr_gb: 2.0,
            ratio: 2.0,
        };
        let suffix = zram.title_suffix();
        assert!(suffix.contains("ZRAM:2.0x"));
    }

    // F-MEM-047: Format ZRAM row
    #[test]
    fn test_format_zram_row() {
        let row = format_zram_row(2.5, 1.0, 2.5, "lz4");
        assert!(row.contains("ZRAM"));
        assert!(row.contains("2.5G"));
        assert!(row.contains("1.0G"));
        assert!(row.contains("2.5x"));
        assert!(row.contains("lz4"));
    }

    // F-MEM-048: ZRAM zero compressed
    #[test]
    fn test_zram_zero_compressed() {
        let zram = ZramDisplay::from_bytes(1024, 0);
        assert_eq!(zram.ratio, 0.0);
    }

    // F-MEM-049: ZRAM arrow in row
    #[test]
    fn test_zram_row_arrow() {
        let row = format_zram_row(2.0, 1.0, 2.0, "zstd");
        assert!(row.contains("→"));
    }

    // F-MEM-050: ZRAM format size boundary
    #[test]
    fn test_zram_size_boundary() {
        let below = ZramDisplay::format_size(1023.9);
        let above = ZramDisplay::format_size(1024.1);
        assert!(below.contains("G"));
        assert!(above.contains("T"));
    }

    // =========================================================================
    // Swap Thrashing Helper Tests (F-MEM-051 to F-MEM-055)
    // =========================================================================

    // F-MEM-051: Format thrashing info
    #[test]
    fn test_format_thrashing_info() {
        let info = format_thrashing_info(0.8, 100.0, 50.0);
        assert!(info.contains("I:100"));
        assert!(info.contains("O:50"));
    }

    // F-MEM-052: Has swap activity thrashing
    #[test]
    fn test_has_swap_activity_thrashing() {
        assert!(has_swap_activity(true, 0.0, 0.0));
    }

    // F-MEM-053: Has swap activity in rate
    #[test]
    fn test_has_swap_activity_in() {
        assert!(has_swap_activity(false, 10.0, 0.0));
    }

    // F-MEM-054: Has swap activity out rate
    #[test]
    fn test_has_swap_activity_out() {
        assert!(has_swap_activity(false, 0.0, 10.0));
    }

    // F-MEM-055: No swap activity
    #[test]
    fn test_no_swap_activity() {
        assert!(!has_swap_activity(false, 0.0, 0.0));
    }
}
