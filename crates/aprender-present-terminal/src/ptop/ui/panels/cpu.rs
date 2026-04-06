//! CPU panel rendering and utilities.
//!
//! Provides CPU panel title building, meter layout calculation,
//! and helper functions for rendering CPU metrics.

use super::super::helpers::format_uptime;

// =============================================================================
// CPU TITLE BUILDING
// =============================================================================

/// Build full CPU panel title string.
///
/// Format: "CPU 45% │ 8 cores │ 3.6GHz⚡ │ up 5d 3h │ LAV 2.1"
#[must_use]
pub fn build_cpu_title(
    cpu_pct: f64,
    core_count: usize,
    freq_ghz: f64,
    is_boosting: bool,
    uptime_secs: u64,
    load_one: f64,
    deterministic: bool,
) -> String {
    let boost_icon = if is_boosting { "⚡" } else { "" };
    if deterministic {
        format!(
            "CPU {cpu_pct:.0}% │ {core_count} cores │ {freq_ghz:.1}GHz │ up {} │",
            format_uptime(uptime_secs)
        )
    } else {
        format!(
            "CPU {cpu_pct:.0}% │ {core_count} cores │ {freq_ghz:.1}GHz{boost_icon} │ up {} │ LAV {load_one:.1}",
            format_uptime(uptime_secs)
        )
    }
}

/// Build compact CPU title for narrow panels (prioritizes frequency).
///
/// Format: "CPU 45% │ 8c │ 3.6GHz⚡" (~22 chars)
#[must_use]
pub fn build_cpu_title_compact(
    cpu_pct: f64,
    core_count: usize,
    freq_ghz: f64,
    is_boosting: bool,
) -> String {
    let boost_icon = if is_boosting { "⚡" } else { "" };
    format!("CPU {cpu_pct:.0}% │ {core_count}c │ {freq_ghz:.1}GHz{boost_icon}")
}

// =============================================================================
// CPU METER LAYOUT
// =============================================================================

/// CPU meter layout parameters for multi-column core display.
#[derive(Debug, Clone, PartialEq)]
pub struct CpuMeterLayout {
    /// Length of meter bars (6 normal, 8 exploded)
    pub bar_len: usize,
    /// Total width of one meter column (bar_len + 9)
    pub meter_bar_width: f32,
    /// Number of cores per column
    pub cores_per_col: usize,
    /// Number of meter columns needed
    pub num_meter_cols: usize,
}

impl CpuMeterLayout {
    /// Calculate layout for given core count and available height.
    ///
    /// # Arguments
    /// * `core_count` - Number of CPU cores
    /// * `core_area_height` - Available height in rows
    /// * `is_exploded` - Whether in exploded/zoomed view
    #[must_use]
    pub fn calculate(core_count: usize, core_area_height: f32, is_exploded: bool) -> Self {
        let bar_len: usize = if is_exploded { 8 } else { 6 };
        let meter_bar_width = (bar_len + 9) as f32;

        let max_cores_per_col = if is_exploded {
            (core_area_height as usize).min(12)
        } else {
            core_area_height as usize
        };
        let cores_per_col = max_cores_per_col.max(1);
        let num_meter_cols = core_count.div_ceil(cores_per_col);

        Self {
            bar_len,
            meter_bar_width,
            cores_per_col,
            num_meter_cols,
        }
    }
}

// =============================================================================
// LOAD AVERAGE COLORS
// =============================================================================

use presentar_core::Color;

/// Get color for load average based on normalized value.
///
/// # Arguments
/// * `load_normalized` - Load average divided by core count
#[must_use]
pub fn load_color(load_normalized: f64) -> Color {
    if load_normalized > 1.0 {
        Color::new(1.0, 0.3, 0.3, 1.0) // Red - overloaded
    } else if load_normalized > 0.7 {
        Color::new(1.0, 0.8, 0.2, 1.0) // Yellow - high
    } else {
        Color::new(0.3, 0.9, 0.3, 1.0) // Green - normal
    }
}

/// Get trend arrow for load average comparison.
///
/// # Returns
/// - "↑" if increasing significantly
/// - "↓" if decreasing significantly
/// - "" if stable
#[must_use]
pub fn load_trend_arrow(current: f64, previous: f64) -> &'static str {
    let delta = current - previous;
    if delta >= 0.5 {
        "↑"
    } else if delta <= -0.5 {
        "↓"
    } else {
        ""
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // build_cpu_title tests
    // =========================================================================

    #[test]
    fn test_build_cpu_title_deterministic() {
        let title = build_cpu_title(45.0, 8, 3.6, false, 3600, 2.1, true);
        assert!(title.contains("CPU 45%"));
        assert!(title.contains("8 cores"));
        assert!(title.contains("3.6GHz"));
        assert!(title.contains("up 1h"));
        assert!(!title.contains("LAV"), "Deterministic should not show LAV");
    }

    #[test]
    fn test_build_cpu_title_normal() {
        let title = build_cpu_title(75.0, 16, 4.8, true, 86400, 5.5, false);
        assert!(title.contains("CPU 75%"));
        assert!(title.contains("16 cores"));
        assert!(title.contains("4.8GHz"));
        assert!(title.contains("⚡"), "Boosting should show lightning");
        assert!(title.contains("LAV 5.5"));
    }

    #[test]
    fn test_build_cpu_title_no_boost() {
        let title = build_cpu_title(50.0, 4, 2.4, false, 0, 1.0, false);
        assert!(
            !title.contains("⚡"),
            "Not boosting should not show lightning"
        );
    }

    #[test]
    fn test_build_cpu_title_zero_values() {
        let title = build_cpu_title(0.0, 1, 0.0, false, 0, 0.0, true);
        assert!(title.contains("CPU 0%"));
        assert!(title.contains("1 cores"));
        assert!(title.contains("0.0GHz"));
    }

    #[test]
    fn test_build_cpu_title_high_cpu() {
        let title = build_cpu_title(100.0, 64, 5.5, true, 604800, 32.0, false);
        assert!(title.contains("CPU 100%"));
        assert!(title.contains("64 cores"));
        assert!(title.contains("7d")); // week uptime
    }

    // =========================================================================
    // build_cpu_title_compact tests
    // =========================================================================

    #[test]
    fn test_build_cpu_title_compact_basic() {
        let title = build_cpu_title_compact(45.0, 8, 3.6, false);
        assert!(title.contains("CPU 45%"));
        assert!(title.contains("8c"));
        assert!(title.contains("3.6GHz"));
    }

    #[test]
    fn test_build_cpu_title_compact_boosting() {
        let title = build_cpu_title_compact(99.0, 48, 4.8, true);
        assert!(title.contains("⚡"));
        assert!(title.contains("48c"));
    }

    #[test]
    fn test_build_cpu_title_compact_length() {
        let title = build_cpu_title_compact(45.0, 8, 3.6, false);
        assert!(title.chars().count() < 30, "Compact title should be short");
    }

    #[test]
    fn test_build_cpu_title_compact_no_uptime() {
        let title = build_cpu_title_compact(45.0, 8, 3.6, false);
        assert!(!title.contains("up"), "Compact should not have uptime");
        assert!(!title.contains("LAV"), "Compact should not have LAV");
    }

    // =========================================================================
    // CpuMeterLayout tests
    // =========================================================================

    #[test]
    fn test_cpu_meter_layout_normal() {
        let layout = CpuMeterLayout::calculate(8, 10.0, false);
        assert_eq!(layout.bar_len, 6, "Normal mode bar length");
        assert_eq!(layout.meter_bar_width, 15.0); // 6 + 9
    }

    #[test]
    fn test_cpu_meter_layout_exploded() {
        let layout = CpuMeterLayout::calculate(8, 10.0, true);
        assert_eq!(layout.bar_len, 8, "Exploded mode bar length");
        assert_eq!(layout.meter_bar_width, 17.0); // 8 + 9
    }

    #[test]
    fn test_cpu_meter_layout_single_column() {
        let layout = CpuMeterLayout::calculate(4, 10.0, false);
        assert_eq!(layout.cores_per_col, 10);
        assert_eq!(layout.num_meter_cols, 1, "4 cores in 10 rows = 1 column");
    }

    #[test]
    fn test_cpu_meter_layout_multi_column() {
        let layout = CpuMeterLayout::calculate(16, 4.0, false);
        assert_eq!(layout.cores_per_col, 4);
        assert_eq!(layout.num_meter_cols, 4, "16 cores / 4 per col = 4 columns");
    }

    #[test]
    fn test_cpu_meter_layout_exploded_max_12_per_col() {
        let layout = CpuMeterLayout::calculate(48, 20.0, true);
        assert_eq!(layout.cores_per_col, 12, "Exploded caps at 12 per column");
        assert_eq!(layout.num_meter_cols, 4);
    }

    #[test]
    fn test_cpu_meter_layout_tiny_height() {
        let layout = CpuMeterLayout::calculate(8, 0.5, false);
        assert_eq!(layout.cores_per_col, 1, "Minimum 1 core per column");
        assert_eq!(layout.num_meter_cols, 8);
    }

    #[test]
    fn test_cpu_meter_layout_single_core() {
        let layout = CpuMeterLayout::calculate(1, 10.0, false);
        assert_eq!(layout.num_meter_cols, 1);
        assert_eq!(layout.cores_per_col, 10);
    }

    #[test]
    fn test_cpu_meter_layout_many_cores() {
        let layout = CpuMeterLayout::calculate(128, 8.0, false);
        assert_eq!(layout.cores_per_col, 8);
        assert_eq!(layout.num_meter_cols, 16);
    }

    #[test]
    fn test_cpu_meter_layout_derive_debug() {
        let layout = CpuMeterLayout::calculate(8, 10.0, false);
        let debug = format!("{:?}", layout);
        assert!(debug.contains("CpuMeterLayout"));
    }

    #[test]
    fn test_cpu_meter_layout_derive_clone() {
        let layout = CpuMeterLayout::calculate(8, 10.0, false);
        let cloned = layout.clone();
        assert_eq!(layout, cloned);
    }

    // =========================================================================
    // load_color tests
    // =========================================================================

    #[test]
    fn test_load_color_low() {
        let color = load_color(0.3);
        assert!(color.g > 0.8, "Low load should be green");
        assert!(color.r < 0.5);
    }

    #[test]
    fn test_load_color_medium() {
        let color = load_color(0.8);
        assert!(color.r > 0.9, "Medium load should be yellow");
        assert!(color.g > 0.7);
    }

    #[test]
    fn test_load_color_high() {
        let color = load_color(1.5);
        assert!(color.r > 0.9, "High load should be red");
        assert!(color.g < 0.5);
    }

    #[test]
    fn test_load_color_boundary_07() {
        let below = load_color(0.69);
        let above = load_color(0.71);
        assert!(
            below.g > above.g,
            "0.7 is boundary between green and yellow"
        );
    }

    #[test]
    fn test_load_color_boundary_10() {
        let below = load_color(0.99);
        let above = load_color(1.01);
        assert!(below.g > above.g, "1.0 is boundary between yellow and red");
    }

    // =========================================================================
    // load_trend_arrow tests
    // =========================================================================

    #[test]
    fn test_load_trend_arrow_increasing() {
        assert_eq!(load_trend_arrow(2.0, 1.0), "↑");
        assert_eq!(load_trend_arrow(5.0, 4.0), "↑");
    }

    #[test]
    fn test_load_trend_arrow_decreasing() {
        assert_eq!(load_trend_arrow(1.0, 2.0), "↓");
        assert_eq!(load_trend_arrow(3.0, 4.0), "↓");
    }

    #[test]
    fn test_load_trend_arrow_stable() {
        assert_eq!(load_trend_arrow(1.0, 1.0), "");
        assert_eq!(load_trend_arrow(1.2, 1.0), "");
        assert_eq!(load_trend_arrow(0.8, 1.0), "");
    }

    #[test]
    fn test_load_trend_arrow_threshold() {
        // Exactly at threshold
        assert_eq!(load_trend_arrow(1.5, 1.0), "↑"); // delta = 0.5
        assert_eq!(load_trend_arrow(1.0, 1.5), "↓"); // delta = -0.5
    }

    #[test]
    fn test_load_trend_arrow_just_below_threshold() {
        assert_eq!(load_trend_arrow(1.49, 1.0), ""); // delta = 0.49
        assert_eq!(load_trend_arrow(1.0, 1.49), ""); // delta = -0.49
    }
}
