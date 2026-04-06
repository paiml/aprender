//! CPU Panel Rendering Components
//!
//! Extracted from render.rs to reduce cyclomatic complexity.
//! Contains helper functions for CPU panel title, load gauge, and consumers display.

#![allow(dead_code)]

use presentar_core::Color;

use crate::ptop::ui::core::format::format_uptime;

/// Build CPU panel title with full information.
///
/// # Arguments
/// * `cpu_pct` - Current CPU usage percentage
/// * `core_count` - Number of CPU cores
/// * `freq_ghz` - Current frequency in GHz
/// * `is_boosting` - Whether CPU is in boost mode (>3GHz)
/// * `uptime` - System uptime in seconds
/// * `load_one` - 1-minute load average
/// * `deterministic` - Whether in deterministic mode (for testing)
#[must_use]
pub(crate) fn build_cpu_title(
    cpu_pct: f64,
    core_count: usize,
    freq_ghz: f64,
    is_boosting: bool,
    uptime: u64,
    load_one: f64,
    deterministic: bool,
) -> String {
    let boost_icon = if is_boosting { "⚡" } else { "" };
    if deterministic {
        format!(
            "CPU {cpu_pct:.0}% │ {core_count} cores │ {freq_ghz:.1}GHz │ up {} │",
            format_uptime(uptime)
        )
    } else {
        // Prioritize: CPU% > cores > freq > uptime > LAV
        // Compact format: "CPU 14% │ 48 cores │ 4.8GHz⚡ │ up 3d 3h │ LAV 30.28"
        format!(
            "CPU {cpu_pct:.0}% │ {core_count} cores │ {freq_ghz:.1}GHz{boost_icon} │ up {} │ LAV {load_one:.1}",
            format_uptime(uptime)
        )
    }
}

/// Build a compact CPU title for narrow panels (prioritizes frequency).
#[must_use]
pub(crate) fn build_cpu_title_compact(
    cpu_pct: f64,
    core_count: usize,
    freq_ghz: f64,
    is_boosting: bool,
) -> String {
    let boost_icon = if is_boosting { "⚡" } else { "" };
    // Compact: "CPU 14% │ 48c │ 4.8GHz⚡" (~22 chars)
    format!("CPU {cpu_pct:.0}% │ {core_count}c │ {freq_ghz:.1}GHz{boost_icon}")
}

/// Calculate CPU meter layout parameters.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CpuMeterLayout {
    /// Length of the meter bar in characters
    pub bar_len: usize,
    /// Width of each meter column in characters
    pub meter_bar_width: f32,
    /// Number of cores per column
    pub cores_per_col: usize,
    /// Number of meter columns
    pub num_meter_cols: usize,
}

impl CpuMeterLayout {
    /// Calculate layout parameters for CPU meters.
    ///
    /// # Arguments
    /// * `core_count` - Total number of CPU cores
    /// * `core_area_height` - Available height for core meters
    /// * `is_exploded` - Whether in exploded view (wider layout)
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

/// Determine color for load average based on normalized value.
///
/// - Red: load > core count (overloaded)
/// - Yellow: load > 70% of core count
/// - Green: otherwise
#[must_use]
pub(crate) fn load_color(load_normalized: f64) -> Color {
    if load_normalized > 1.0 {
        Color::new(1.0, 0.3, 0.3, 1.0) // Red - overloaded
    } else if load_normalized > 0.7 {
        Color::new(1.0, 0.8, 0.2, 1.0) // Yellow - high
    } else {
        Color::new(0.3, 0.9, 0.3, 1.0) // Green - normal
    }
}

/// Get trend arrow for comparing two load values.
#[must_use]
pub(crate) fn load_trend_arrow(newer: f64, older: f64) -> &'static str {
    if newer > older {
        "↑"
    } else if newer < older {
        "↓"
    } else {
        "→"
    }
}

/// Build load bar string (filled/empty blocks).
///
/// # Arguments
/// * `load_pct` - Load percentage (0.0-1.0)
/// * `bar_width` - Width of the bar in characters
#[must_use]
pub(crate) fn build_load_bar(load_pct: f64, bar_width: usize) -> String {
    let filled = (load_pct * bar_width as f64) as usize;
    "█".repeat(filled.min(bar_width)) + &"░".repeat(bar_width - filled.min(bar_width))
}

/// Format load average line with adaptive width.
///
/// # Arguments
/// * `load_one` - 1-minute load average
/// * `load_five` - 5-minute load average
/// * `load_fifteen` - 15-minute load average
/// * `freq_ghz` - Current frequency in GHz
/// * `available_width` - Available width in characters
/// * `deterministic` - Whether in deterministic mode
#[must_use]
pub(crate) fn format_load_line(
    load_one: f64,
    load_five: f64,
    load_fifteen: f64,
    freq_ghz: f64,
    available_width: usize,
    deterministic: bool,
    core_count: usize,
) -> String {
    let load_normalized = load_one / core_count as f64;
    let load_pct = (load_normalized / 2.0).min(1.0);

    let trend_1_5 = load_trend_arrow(load_one, load_five);
    let trend_5_15 = load_trend_arrow(load_five, load_fifteen);

    if deterministic {
        let bar = build_load_bar(load_pct, 10);
        format!(
            "Load {bar} {load_one:.2}{trend_1_5} {load_five:.2}{trend_5_15} {load_fifteen:.2} │ Fre"
        )
    } else if available_width >= 45 && freq_ghz > 0.0 {
        // Full format with frequency
        let bar = build_load_bar(load_pct, 10);
        format!(
            "Load {bar} {load_one:.2}{trend_1_5} {load_five:.2}{trend_5_15} {load_fifteen:.2}→ │ {freq_ghz:.1}GHz"
        )
    } else if available_width >= 35 {
        // Medium format without frequency
        let bar = build_load_bar(load_pct, 10);
        format!("Load {bar} {load_one:.2}{trend_1_5} {load_five:.2}{trend_5_15} {load_fifteen:.2}→")
    } else {
        // Compact format for narrow panels
        let bar = build_load_bar(load_pct, 4);
        format!("Load {bar} {load_one:.1}{trend_1_5} {load_five:.1}{trend_5_15} {load_fifteen:.1}→")
    }
}

/// Determine color for CPU consumer based on usage.
#[must_use]
pub(crate) fn consumer_cpu_color(cpu: f64) -> Color {
    if cpu > 50.0 {
        Color::new(1.0, 0.3, 0.3, 1.0) // Red - high usage
    } else if cpu > 20.0 {
        Color::new(1.0, 0.8, 0.2, 1.0) // Yellow - moderate
    } else {
        Color::new(0.3, 0.9, 0.3, 1.0) // Green - low
    }
}

/// Standard dim color for labels.
pub(crate) const DIM_LABEL_COLOR: Color = Color {
    r: 0.4,
    g: 0.4,
    b: 0.4,
    a: 1.0,
};

/// Bright white for process names.
pub(crate) const PROCESS_NAME_COLOR: Color = Color {
    r: 0.9,
    g: 0.9,
    b: 0.9,
    a: 1.0,
};

// =============================================================================
// TOP CPU CONSUMERS (extracted to reduce draw_cpu_panel complexity)
// =============================================================================

/// A top CPU consumer for display.
#[derive(Debug, Clone)]
pub struct TopConsumer {
    /// Process name (truncated to 12 chars)
    pub name: String,
    /// CPU usage percentage
    pub cpu_percent: f64,
}

impl TopConsumer {
    /// Create a new consumer from name and CPU percentage.
    #[must_use]
    pub fn new(name: impl Into<String>, cpu_percent: f64) -> Self {
        let full_name = name.into();
        let truncated: String = full_name.chars().take(12).collect();
        Self {
            name: truncated,
            cpu_percent,
        }
    }

    /// Get the color for this consumer's CPU display.
    #[must_use]
    pub fn color(&self) -> Color {
        consumer_cpu_color(self.cpu_percent)
    }

    /// Format the CPU percentage string (e.g., "42%").
    #[must_use]
    pub fn cpu_string(&self) -> String {
        format!("{:.0}%", self.cpu_percent)
    }
}

/// Format a row of top consumers for display.
///
/// Returns formatted segments: [(text, color), ...]
/// Format: "Top 42% firefox │ 15% code │ 8% chrome"
#[must_use]
pub fn format_top_consumers_row(consumers: &[TopConsumer]) -> Vec<(String, Color)> {
    let mut segments = Vec::new();

    if consumers.is_empty() {
        return segments;
    }

    segments.push(("Top ".to_string(), DIM_LABEL_COLOR));

    for (i, consumer) in consumers.iter().enumerate() {
        if i > 0 {
            segments.push((" │ ".to_string(), DIM_LABEL_COLOR));
        }

        // CPU percentage in color
        segments.push((consumer.cpu_string(), consumer.color()));

        // Process name in white
        segments.push((format!(" {}", consumer.name), PROCESS_NAME_COLOR));
    }

    segments
}

/// Calculate total width of formatted consumer row.
#[must_use]
pub fn consumer_row_width(consumers: &[TopConsumer]) -> usize {
    if consumers.is_empty() {
        return 0;
    }
    let segments = format_top_consumers_row(consumers);
    segments.iter().map(|(s, _)| s.chars().count()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    // F-CPU-001: Title contains CPU percentage
    #[test]
    fn test_title_contains_cpu_pct() {
        let title = build_cpu_title(45.0, 8, 3.5, false, 3600, 1.5, false);
        assert!(title.contains("45%"), "Title should contain CPU percentage");
    }

    // F-CPU-002: Title contains core count
    #[test]
    fn test_title_contains_core_count() {
        let title = build_cpu_title(45.0, 8, 3.5, false, 3600, 1.5, false);
        assert!(title.contains("8 cores"), "Title should contain core count");
    }

    // F-CPU-003: Title contains frequency
    #[test]
    fn test_title_contains_frequency() {
        let title = build_cpu_title(45.0, 8, 3.5, false, 3600, 1.5, false);
        assert!(title.contains("3.5GHz"), "Title should contain frequency");
    }

    // F-CPU-004: Title shows boost icon when boosting
    #[test]
    fn test_title_shows_boost_icon() {
        let title = build_cpu_title(45.0, 8, 4.5, true, 3600, 1.5, false);
        assert!(title.contains("⚡"), "Title should show boost icon");
    }

    // F-CPU-005: Title hides boost icon when not boosting
    #[test]
    fn test_title_hides_boost_icon() {
        let title = build_cpu_title(45.0, 8, 2.5, false, 3600, 1.5, false);
        assert!(!title.contains("⚡"), "Title should not show boost icon");
    }

    // F-CPU-006: Title contains load average in non-deterministic mode
    #[test]
    fn test_title_contains_lav() {
        let title = build_cpu_title(45.0, 8, 3.5, false, 3600, 2.5, false);
        assert!(
            title.contains("LAV 2.5"),
            "Title should contain load average"
        );
    }

    // F-CPU-007: Deterministic title omits LAV
    #[test]
    fn test_deterministic_title_omits_lav() {
        let title = build_cpu_title(45.0, 8, 3.5, false, 3600, 2.5, true);
        assert!(
            !title.contains("LAV"),
            "Deterministic title should not contain LAV"
        );
    }

    // F-CPU-008: Compact title is shorter
    #[test]
    fn test_compact_title_shorter() {
        let full = build_cpu_title(45.0, 8, 3.5, false, 3600, 1.5, false);
        let compact = build_cpu_title_compact(45.0, 8, 3.5, false);
        assert!(
            compact.len() < full.len(),
            "Compact title should be shorter"
        );
    }

    // F-CPU-009: Compact title uses abbreviated core count
    #[test]
    fn test_compact_title_abbreviated_cores() {
        let compact = build_cpu_title_compact(45.0, 8, 3.5, false);
        assert!(
            compact.contains("8c"),
            "Compact title should use abbreviated core count"
        );
    }

    // F-CPU-010: Meter layout for 8 cores standard
    #[test]
    fn test_meter_layout_8_cores() {
        let layout = CpuMeterLayout::calculate(8, 10.0, false);
        assert!(layout.cores_per_col <= 10);
        assert!(layout.num_meter_cols >= 1);
    }

    // F-CPU-011: Meter layout for 48 cores exploded
    #[test]
    fn test_meter_layout_48_cores_exploded() {
        let layout = CpuMeterLayout::calculate(48, 20.0, true);
        assert_eq!(layout.bar_len, 8, "Exploded mode should have longer bars");
        assert!(layout.cores_per_col <= 12, "Exploded caps at 12 per col");
    }

    // F-CPU-012: Meter layout standard bar length
    #[test]
    fn test_meter_layout_standard_bar() {
        let layout = CpuMeterLayout::calculate(8, 10.0, false);
        assert_eq!(layout.bar_len, 6, "Standard mode should have 6-char bars");
    }

    // F-CPU-013: Load color red when overloaded
    #[test]
    fn test_load_color_red_overloaded() {
        let color = load_color(1.5);
        assert!(
            color.r > 0.8 && color.g < 0.5,
            "Should be red when overloaded"
        );
    }

    // F-CPU-014: Load color yellow when high
    #[test]
    fn test_load_color_yellow_high() {
        let color = load_color(0.85);
        assert!(color.r > 0.8 && color.g > 0.5, "Should be yellow when high");
    }

    // F-CPU-015: Load color green when normal
    #[test]
    fn test_load_color_green_normal() {
        let color = load_color(0.5);
        assert!(
            color.g > 0.8 && color.r < 0.5,
            "Should be green when normal"
        );
    }

    // F-CPU-016: Load trend up arrow
    #[test]
    fn test_load_trend_up() {
        assert_eq!(load_trend_arrow(2.0, 1.0), "↑");
    }

    // F-CPU-017: Load trend down arrow
    #[test]
    fn test_load_trend_down() {
        assert_eq!(load_trend_arrow(1.0, 2.0), "↓");
    }

    // F-CPU-018: Load trend same arrow
    #[test]
    fn test_load_trend_same() {
        assert_eq!(load_trend_arrow(1.5, 1.5), "→");
    }

    // F-CPU-019: Load bar fully filled
    #[test]
    fn test_load_bar_full() {
        let bar = build_load_bar(1.0, 10);
        assert_eq!(bar.chars().filter(|&c| c == '█').count(), 10);
    }

    // F-CPU-020: Load bar empty
    #[test]
    fn test_load_bar_empty() {
        let bar = build_load_bar(0.0, 10);
        assert_eq!(bar.chars().filter(|&c| c == '░').count(), 10);
    }

    // F-CPU-021: Load bar half filled
    #[test]
    fn test_load_bar_half() {
        let bar = build_load_bar(0.5, 10);
        assert_eq!(bar.chars().filter(|&c| c == '█').count(), 5);
        assert_eq!(bar.chars().filter(|&c| c == '░').count(), 5);
    }

    // F-CPU-022: Load line format full width
    #[test]
    fn test_load_line_full_width() {
        let line = format_load_line(2.0, 1.5, 1.0, 4.5, 50, false, 8);
        assert!(line.contains("GHz"), "Full width should contain frequency");
    }

    // F-CPU-023: Load line format medium width
    #[test]
    fn test_load_line_medium_width() {
        let line = format_load_line(2.0, 1.5, 1.0, 4.5, 40, false, 8);
        assert!(!line.contains("GHz"), "Medium width should omit frequency");
    }

    // F-CPU-024: Load line format compact
    #[test]
    fn test_load_line_compact() {
        let line = format_load_line(2.0, 1.5, 1.0, 4.5, 25, false, 8);
        assert!(line.len() < 40, "Compact format should be short");
    }

    // F-CPU-025: Consumer color red for high usage
    #[test]
    fn test_consumer_color_high() {
        let color = consumer_cpu_color(75.0);
        assert!(color.r > 0.8, "High usage should be red");
    }

    // F-CPU-026: Consumer color yellow for moderate
    #[test]
    fn test_consumer_color_moderate() {
        let color = consumer_cpu_color(35.0);
        assert!(color.r > 0.8 && color.g > 0.5, "Moderate should be yellow");
    }

    // F-CPU-027: Consumer color green for low
    #[test]
    fn test_consumer_color_low() {
        let color = consumer_cpu_color(10.0);
        assert!(color.g > 0.8, "Low usage should be green");
    }

    // F-CPU-028: Dim label color is gray
    #[test]
    fn test_dim_label_color() {
        assert!(DIM_LABEL_COLOR.r < 0.5 && DIM_LABEL_COLOR.g < 0.5 && DIM_LABEL_COLOR.b < 0.5);
    }

    // F-CPU-029: Process name color is bright
    #[test]
    fn test_process_name_color_bright() {
        assert!(
            PROCESS_NAME_COLOR.r > 0.8 && PROCESS_NAME_COLOR.g > 0.8 && PROCESS_NAME_COLOR.b > 0.8
        );
    }

    // F-CPU-030: Title with zero CPU percentage
    #[test]
    fn test_title_zero_cpu() {
        let title = build_cpu_title(0.0, 8, 3.5, false, 0, 0.0, false);
        assert!(title.contains("0%"));
    }

    // F-CPU-031: Title with 100% CPU
    #[test]
    fn test_title_full_cpu() {
        let title = build_cpu_title(100.0, 8, 3.5, false, 3600, 8.0, false);
        assert!(title.contains("100%"));
    }

    // F-CPU-032: Compact title with many cores
    #[test]
    fn test_compact_title_many_cores() {
        let compact = build_cpu_title_compact(50.0, 128, 3.5, false);
        assert!(compact.contains("128c"));
    }

    // F-CPU-033: Meter layout single core
    #[test]
    fn test_meter_layout_single_core() {
        let layout = CpuMeterLayout::calculate(1, 10.0, false);
        assert_eq!(layout.num_meter_cols, 1);
        assert_eq!(layout.cores_per_col, 10);
    }

    // F-CPU-034: Meter layout zero height handled
    #[test]
    fn test_meter_layout_zero_height() {
        let layout = CpuMeterLayout::calculate(8, 0.0, false);
        assert_eq!(layout.cores_per_col, 1, "Should default to 1 core per col");
    }

    // F-CPU-035: Load color boundary at 0.7
    #[test]
    fn test_load_color_boundary_07() {
        let below = load_color(0.69);
        let above = load_color(0.71);
        assert!(below.g > above.g, "Below 0.7 should be greener");
    }

    // F-CPU-036: Load color boundary at 1.0
    #[test]
    fn test_load_color_boundary_10() {
        let below = load_color(0.5);
        let above = load_color(1.5);
        // Above 1.0 is red (r=1.0), below is green (r=0.3)
        assert!(above.r > below.r, "Above 1.0 should be redder");
    }

    // F-CPU-037: Load bar handles overflow
    #[test]
    fn test_load_bar_overflow() {
        let bar = build_load_bar(1.5, 10);
        // Bar should have exactly bar_width unicode characters (filled blocks)
        assert_eq!(bar.chars().count(), 10, "Bar should not overflow");
    }

    // F-CPU-038: Format load line deterministic mode
    #[test]
    fn test_format_load_deterministic() {
        let line = format_load_line(0.0, 0.0, 0.0, 0.0, 50, true, 8);
        assert!(
            line.contains("Fre"),
            "Deterministic mode has special format"
        );
    }

    // F-CPU-039: Consumer threshold at 50%
    #[test]
    fn test_consumer_threshold_50() {
        let below = consumer_cpu_color(49.9);
        let above = consumer_cpu_color(50.1);
        assert!(above.g < below.g, "Above 50% should be redder");
    }

    // F-CPU-040: Consumer threshold at 20%
    #[test]
    fn test_consumer_threshold_20() {
        let below = consumer_cpu_color(19.9);
        let above = consumer_cpu_color(20.1);
        assert!(above.r > below.r, "Above 20% should be more yellow");
    }

    // =========================================================================
    // TopConsumer tests
    // =========================================================================

    // F-CONSUMER-001: TopConsumer creation
    #[test]
    fn f_consumer_001_new() {
        let c = TopConsumer::new("firefox", 42.5);
        assert_eq!(c.name, "firefox");
        assert!((c.cpu_percent - 42.5).abs() < 0.01);
    }

    // F-CONSUMER-002: TopConsumer truncates long names
    #[test]
    fn f_consumer_002_truncate_long_name() {
        let c = TopConsumer::new("very_long_process_name_here", 10.0);
        assert_eq!(c.name.chars().count(), 12);
        assert_eq!(c.name, "very_long_pr");
    }

    // F-CONSUMER-003: TopConsumer cpu_string format
    #[test]
    fn f_consumer_003_cpu_string() {
        let c = TopConsumer::new("test", 42.7);
        assert_eq!(c.cpu_string(), "43%");
    }

    // F-CONSUMER-004: TopConsumer color high CPU
    #[test]
    fn f_consumer_004_color_high() {
        let c = TopConsumer::new("test", 75.0);
        let color = c.color();
        assert!(color.r > 0.8, "High CPU should be red");
    }

    // F-CONSUMER-005: TopConsumer color low CPU
    #[test]
    fn f_consumer_005_color_low() {
        let c = TopConsumer::new("test", 5.0);
        let color = c.color();
        assert!(color.g > 0.8, "Low CPU should be green");
    }

    // F-CONSUMER-006: TopConsumer Debug trait
    #[test]
    fn f_consumer_006_debug() {
        let c = TopConsumer::new("firefox", 42.0);
        let debug = format!("{:?}", c);
        assert!(debug.contains("TopConsumer"));
        assert!(debug.contains("firefox"));
    }

    // F-CONSUMER-007: TopConsumer Clone trait
    #[test]
    fn f_consumer_007_clone() {
        let c = TopConsumer::new("test", 50.0);
        let cloned = c.clone();
        assert_eq!(c.name, cloned.name);
        assert!((c.cpu_percent - cloned.cpu_percent).abs() < 0.01);
    }

    // F-CONSUMER-008: format_top_consumers_row empty
    #[test]
    fn f_consumer_008_format_empty() {
        let segments = format_top_consumers_row(&[]);
        assert!(segments.is_empty());
    }

    // F-CONSUMER-009: format_top_consumers_row single
    #[test]
    fn f_consumer_009_format_single() {
        let consumers = vec![TopConsumer::new("firefox", 42.0)];
        let segments = format_top_consumers_row(&consumers);
        assert!(!segments.is_empty());
        let text: String = segments.iter().map(|(s, _)| s.as_str()).collect();
        assert!(text.contains("Top"));
        assert!(text.contains("firefox"));
    }

    // F-CONSUMER-010: format_top_consumers_row multiple
    #[test]
    fn f_consumer_010_format_multiple() {
        let consumers = vec![
            TopConsumer::new("firefox", 42.0),
            TopConsumer::new("code", 15.0),
        ];
        let segments = format_top_consumers_row(&consumers);
        let text: String = segments.iter().map(|(s, _)| s.as_str()).collect();
        assert!(text.contains("│"), "Multiple should have separator");
        assert!(text.contains("firefox"));
        assert!(text.contains("code"));
    }

    // F-CONSUMER-011: consumer_row_width empty
    #[test]
    fn f_consumer_011_width_empty() {
        assert_eq!(consumer_row_width(&[]), 0);
    }

    // F-CONSUMER-012: consumer_row_width single
    #[test]
    fn f_consumer_012_width_single() {
        let consumers = vec![TopConsumer::new("test", 42.0)];
        let width = consumer_row_width(&consumers);
        assert!(width > 0);
        assert!(width < 30, "Single consumer width should be reasonable");
    }

    // F-CONSUMER-013: consumer_row_width multiple
    #[test]
    fn f_consumer_013_width_multiple() {
        let single = vec![TopConsumer::new("test", 42.0)];
        let double = vec![
            TopConsumer::new("test", 42.0),
            TopConsumer::new("more", 10.0),
        ];
        assert!(consumer_row_width(&double) > consumer_row_width(&single));
    }

    // F-CONSUMER-014: TopConsumer zero CPU
    #[test]
    fn f_consumer_014_zero_cpu() {
        let c = TopConsumer::new("idle", 0.0);
        assert_eq!(c.cpu_string(), "0%");
        assert!(c.color().g > 0.8, "Zero CPU should be green");
    }

    // F-CONSUMER-015: TopConsumer 100% CPU
    #[test]
    fn f_consumer_015_full_cpu() {
        let c = TopConsumer::new("busy", 100.0);
        assert_eq!(c.cpu_string(), "100%");
        assert!(c.color().r > 0.8, "100% CPU should be red");
    }
}
