//! Formatting utilities for ptop UI.
//!
//! Functions for formatting bytes, rates, uptime, percentages, etc.

#![allow(dead_code)]

use presentar_core::Color;

use super::constants::{CYAN, GREEN, RED, WHITE, YELLOW};

/// Format bytes with appropriate unit (B, KB, MB, GB, TB).
///
/// # Examples
///
/// ```ignore
/// use presentar_terminal::ptop::ui::core::format::format_bytes;
///
/// assert_eq!(format_bytes(0), "0 B");
/// assert_eq!(format_bytes(1024), "1.0 KB");
/// assert_eq!(format_bytes(1048576), "1.0 MB");
/// ```
#[must_use]
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    const TB: u64 = GB * 1024;

    if bytes >= TB {
        format!("{:.1} TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// Format bytes per second with appropriate unit.
///
/// # Examples
///
/// ```ignore
/// use presentar_terminal::ptop::ui::core::format::format_bytes_rate;
///
/// assert_eq!(format_bytes_rate(0.0), "0 B/s");
/// assert_eq!(format_bytes_rate(1024.0), "1.0 KB/s");
/// ```
#[must_use]
pub fn format_bytes_rate(bytes_per_sec: f64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;

    if bytes_per_sec >= GB {
        format!("{:.1} GB/s", bytes_per_sec / GB)
    } else if bytes_per_sec >= MB {
        format!("{:.1} MB/s", bytes_per_sec / MB)
    } else if bytes_per_sec >= KB {
        format!("{:.1} KB/s", bytes_per_sec / KB)
    } else {
        format!("{:.0} B/s", bytes_per_sec)
    }
}

/// Format uptime in human-readable form.
///
/// # Examples
///
/// ```ignore
/// use presentar_terminal::ptop::ui::core::format::format_uptime;
///
/// assert_eq!(format_uptime(3600), "1h 0m");
/// assert_eq!(format_uptime(90000), "1d 1h");
/// ```
#[must_use]
pub fn format_uptime(secs: u64) -> String {
    let days = secs / 86400;
    let hours = (secs % 86400) / 3600;
    let mins = (secs % 3600) / 60;

    if days > 0 {
        // At day scale, minutes are too granular
        format!("{days}d {hours}h")
    } else if hours > 0 {
        format!("{hours}h {mins}m")
    } else {
        format!("{mins}m")
    }
}

/// Format duration in milliseconds or seconds.
#[must_use]
pub fn format_duration_ms(ms: f64) -> String {
    if ms >= 1000.0 {
        format!("{:.1}s", ms / 1000.0)
    } else {
        format!("{:.0}ms", ms)
    }
}

/// Format percentage with optional decimal places.
#[must_use]
pub fn format_percent(value: f64, decimals: usize) -> String {
    match decimals {
        0 => format!("{:.0}%", value),
        1 => format!("{:.1}%", value),
        _ => format!("{:.2}%", value),
    }
}

/// Format frequency in GHz or MHz.
#[must_use]
pub fn format_frequency(mhz: f64) -> String {
    if mhz >= 1000.0 {
        format!("{:.2}GHz", mhz / 1000.0)
    } else {
        format!("{:.0}MHz", mhz)
    }
}

/// Format temperature with degree symbol.
#[must_use]
pub fn format_temp(celsius: f64) -> String {
    format!("{:.0}°C", celsius)
}

/// Format count with K/M suffix for large numbers.
#[must_use]
pub fn format_count(count: u64) -> String {
    if count >= 1_000_000 {
        format!("{:.1}M", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.1}K", count as f64 / 1_000.0)
    } else {
        format!("{count}")
    }
}

/// Get color based on percentage (0-100).
///
/// Returns green for low values, yellow for medium, red for high.
#[must_use]
pub fn percent_color(percent: f64) -> Color {
    if percent >= 90.0 {
        RED
    } else if percent >= 70.0 {
        // Yellow-orange gradient
        let t = ((percent - 70.0) / 20.0) as f32;
        Color {
            r: 1.0,
            g: 0.8 - (t * 0.5),
            b: 0.0,
            a: 1.0,
        }
    } else if percent >= 50.0 {
        YELLOW
    } else if percent >= 30.0 {
        // Green-yellow gradient
        let t = ((percent - 30.0) / 20.0) as f32;
        Color {
            r: t,
            g: 0.8,
            b: 0.0,
            a: 1.0,
        }
    } else {
        GREEN
    }
}

/// Get color based on temperature (Celsius).
///
/// Returns cyan for cool, green for normal, yellow for warm, red for hot.
#[must_use]
pub fn temp_color(temp: f64) -> Color {
    if temp >= 85.0 {
        RED
    } else if temp >= 70.0 {
        let t = ((temp - 70.0) / 15.0) as f32;
        Color {
            r: 1.0,
            g: 0.8 - (t * 0.6),
            b: 0.0,
            a: 1.0,
        }
    } else if temp >= 50.0 {
        YELLOW
    } else if temp >= 35.0 {
        GREEN
    } else {
        CYAN
    }
}

/// Get color for swap usage percentage.
#[must_use]
pub fn swap_color(pct: f64) -> Color {
    if pct > 50.0 {
        RED
    } else if pct > 20.0 {
        YELLOW
    } else {
        WHITE
    }
}

/// Get color for memory type (used, cached, buffers, etc.).
#[must_use]
pub fn memory_segment_color(segment: &str) -> Color {
    match segment {
        "used" => Color {
            r: 0.3,
            g: 0.6,
            b: 0.9,
            a: 1.0,
        },
        "cached" => Color {
            r: 0.5,
            g: 0.8,
            b: 0.5,
            a: 1.0,
        },
        "buffers" => Color {
            r: 0.8,
            g: 0.8,
            b: 0.3,
            a: 1.0,
        },
        "available" | "free" => Color {
            r: 0.3,
            g: 0.3,
            b: 0.3,
            a: 1.0,
        },
        _ => WHITE,
    }
}

/// Truncate string with ellipsis if too long.
#[must_use]
pub fn truncate_with_ellipsis(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else if max_len <= 3 {
        s.chars().take(max_len).collect()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

/// Pad string to fixed width (left-aligned).
#[must_use]
pub fn pad_left(s: &str, width: usize) -> String {
    if s.len() >= width {
        s[..width].to_string()
    } else {
        format!("{s:<width$}")
    }
}

/// Pad string to fixed width (right-aligned).
#[must_use]
pub fn pad_right(s: &str, width: usize) -> String {
    if s.len() >= width {
        s[..width].to_string()
    } else {
        format!("{s:>width$}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // F-FORMAT-001: format_bytes with 0
    #[test]
    fn test_format_bytes_zero() {
        assert_eq!(format_bytes(0), "0 B");
    }

    // F-FORMAT-002: format_bytes with kilobytes
    #[test]
    fn test_format_bytes_kb() {
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(2048), "2.0 KB");
    }

    // F-FORMAT-003: format_bytes with megabytes
    #[test]
    fn test_format_bytes_mb() {
        assert_eq!(format_bytes(1048576), "1.0 MB");
        assert_eq!(format_bytes(10485760), "10.0 MB");
    }

    // F-FORMAT-004: format_bytes with gigabytes
    #[test]
    fn test_format_bytes_gb() {
        assert_eq!(format_bytes(1073741824), "1.0 GB");
    }

    // F-FORMAT-005: format_bytes with terabytes
    #[test]
    fn test_format_bytes_tb() {
        assert_eq!(format_bytes(1099511627776), "1.0 TB");
    }

    // F-FORMAT-006: format_bytes_rate zero
    #[test]
    fn test_format_bytes_rate_zero() {
        assert_eq!(format_bytes_rate(0.0), "0 B/s");
    }

    // F-FORMAT-007: format_bytes_rate KB/s
    #[test]
    fn test_format_bytes_rate_kb() {
        assert_eq!(format_bytes_rate(1024.0), "1.0 KB/s");
    }

    // F-FORMAT-008: format_bytes_rate MB/s
    #[test]
    fn test_format_bytes_rate_mb() {
        assert_eq!(format_bytes_rate(1048576.0), "1.0 MB/s");
    }

    // F-FORMAT-009: format_uptime minutes only
    #[test]
    fn test_format_uptime_minutes() {
        assert_eq!(format_uptime(60), "1m");
        assert_eq!(format_uptime(300), "5m");
    }

    // F-FORMAT-010: format_uptime hours
    #[test]
    fn test_format_uptime_hours() {
        assert_eq!(format_uptime(3600), "1h 0m");
        assert_eq!(format_uptime(3660), "1h 1m");
    }

    // F-FORMAT-011: format_uptime days (minutes omitted at day scale)
    #[test]
    fn test_format_uptime_days() {
        assert_eq!(format_uptime(86400), "1d 0h");
        assert_eq!(format_uptime(90061), "1d 1h");
    }

    // F-FORMAT-012: format_percent no decimals
    #[test]
    fn test_format_percent_no_decimals() {
        assert_eq!(format_percent(50.0, 0), "50%");
    }

    // F-FORMAT-013: format_percent one decimal
    #[test]
    fn test_format_percent_one_decimal() {
        assert_eq!(format_percent(50.5, 1), "50.5%");
    }

    // F-FORMAT-014: format_frequency GHz
    #[test]
    fn test_format_frequency_ghz() {
        assert_eq!(format_frequency(3500.0), "3.50GHz");
    }

    // F-FORMAT-015: format_frequency MHz
    #[test]
    fn test_format_frequency_mhz() {
        assert_eq!(format_frequency(800.0), "800MHz");
    }

    // F-FORMAT-016: format_temp
    #[test]
    fn test_format_temp() {
        assert_eq!(format_temp(45.7), "46°C");
    }

    // F-FORMAT-017: format_count small
    #[test]
    fn test_format_count_small() {
        assert_eq!(format_count(100), "100");
    }

    // F-FORMAT-018: format_count thousands
    #[test]
    fn test_format_count_thousands() {
        assert_eq!(format_count(1500), "1.5K");
    }

    // F-FORMAT-019: format_count millions
    #[test]
    fn test_format_count_millions() {
        assert_eq!(format_count(1500000), "1.5M");
    }

    // F-FORMAT-020: percent_color low (green)
    #[test]
    fn test_percent_color_low() {
        let color = percent_color(10.0);
        assert!(color.g > color.r, "Low percent should be green");
    }

    // F-FORMAT-021: percent_color medium (yellow)
    #[test]
    fn test_percent_color_medium() {
        let color = percent_color(60.0);
        assert!(
            color.r > 0.5 && color.g > 0.5,
            "Medium percent should be yellow"
        );
    }

    // F-FORMAT-022: percent_color high (red)
    #[test]
    fn test_percent_color_high() {
        let color = percent_color(95.0);
        assert!(color.r > color.g, "High percent should be red");
    }

    // F-FORMAT-023: temp_color cool (cyan)
    #[test]
    fn test_temp_color_cool() {
        let color = temp_color(25.0);
        assert!(color.b > 0.5 && color.g > 0.5, "Cool temp should be cyan");
    }

    // F-FORMAT-024: temp_color normal (green)
    #[test]
    fn test_temp_color_normal() {
        let color = temp_color(45.0);
        assert!(color.g > color.r, "Normal temp should be green");
    }

    // F-FORMAT-025: temp_color hot (red)
    #[test]
    fn test_temp_color_hot() {
        let color = temp_color(90.0);
        assert!(color.r > color.g, "Hot temp should be red");
    }

    // F-FORMAT-026: swap_color low
    #[test]
    fn test_swap_color_low() {
        let color = swap_color(10.0);
        assert!((color.r - 1.0).abs() < 0.01, "Low swap should be white");
    }

    // F-FORMAT-027: swap_color medium
    #[test]
    fn test_swap_color_medium() {
        let color = swap_color(30.0);
        assert!(color.g > 0.5, "Medium swap should be yellow");
    }

    // F-FORMAT-028: swap_color high
    #[test]
    fn test_swap_color_high() {
        let color = swap_color(60.0);
        assert!(color.r > color.g, "High swap should be red");
    }

    // F-FORMAT-029: truncate_with_ellipsis short string
    #[test]
    fn test_truncate_short() {
        assert_eq!(truncate_with_ellipsis("hello", 10), "hello");
    }

    // F-FORMAT-030: truncate_with_ellipsis long string
    #[test]
    fn test_truncate_long() {
        assert_eq!(truncate_with_ellipsis("hello world", 8), "hello...");
    }

    // F-FORMAT-031: truncate_with_ellipsis exact length
    #[test]
    fn test_truncate_exact() {
        assert_eq!(truncate_with_ellipsis("hello", 5), "hello");
    }

    // F-FORMAT-032: truncate_with_ellipsis very short max
    #[test]
    fn test_truncate_very_short_max() {
        assert_eq!(truncate_with_ellipsis("hello", 2), "he");
    }

    // F-FORMAT-033: pad_left shorter string
    #[test]
    fn test_pad_left_shorter() {
        assert_eq!(pad_left("hi", 5), "hi   ");
    }

    // F-FORMAT-034: pad_left longer string
    #[test]
    fn test_pad_left_longer() {
        assert_eq!(pad_left("hello world", 5), "hello");
    }

    // F-FORMAT-035: pad_right shorter string
    #[test]
    fn test_pad_right_shorter() {
        assert_eq!(pad_right("hi", 5), "   hi");
    }

    // F-FORMAT-036: pad_right longer string
    #[test]
    fn test_pad_right_longer() {
        assert_eq!(pad_right("hello world", 5), "hello");
    }

    // F-FORMAT-037: memory_segment_color used
    #[test]
    fn test_memory_segment_color_used() {
        let color = memory_segment_color("used");
        assert!(color.b > color.r, "Used should be blue-ish");
    }

    // F-FORMAT-038: memory_segment_color cached
    #[test]
    fn test_memory_segment_color_cached() {
        let color = memory_segment_color("cached");
        assert!(
            color.g > color.r && color.g > color.b,
            "Cached should be green-ish"
        );
    }

    // F-FORMAT-039: memory_segment_color unknown
    #[test]
    fn test_memory_segment_color_unknown() {
        let color = memory_segment_color("unknown");
        assert!((color.r - 1.0).abs() < 0.01, "Unknown should be white");
    }

    // F-FORMAT-040: format_duration_ms milliseconds
    #[test]
    fn test_format_duration_ms() {
        assert_eq!(format_duration_ms(500.0), "500ms");
    }

    // F-FORMAT-041: format_duration_ms seconds
    #[test]
    fn test_format_duration_seconds() {
        assert_eq!(format_duration_ms(1500.0), "1.5s");
    }

    // F-FORMAT-042: percent_color boundary 90%
    #[test]
    fn test_percent_color_boundary_90() {
        let color = percent_color(90.0);
        assert!(color.r > 0.9, "90% should be red");
    }

    // F-FORMAT-043: percent_color boundary 70%
    #[test]
    fn test_percent_color_boundary_70() {
        let color = percent_color(70.0);
        assert!(
            color.r > 0.5 && color.g > 0.3,
            "70% should be orange-yellow"
        );
    }

    // F-FORMAT-044: percent_color boundary 50%
    #[test]
    fn test_percent_color_boundary_50() {
        let color = percent_color(50.0);
        assert!(color.r > 0.5 && color.g > 0.5, "50% should be yellow");
    }

    // F-FORMAT-045: percent_color boundary 30%
    #[test]
    fn test_percent_color_boundary_30() {
        let color = percent_color(30.0);
        assert!(color.g > 0.5, "30% should be greenish");
    }

    // F-FORMAT-046: temp_color boundary 85°C
    #[test]
    fn test_temp_color_boundary_85() {
        let color = temp_color(85.0);
        assert!(color.r > 0.9, "85°C should be red");
    }

    // F-FORMAT-047: temp_color boundary 70°C
    #[test]
    fn test_temp_color_boundary_70() {
        let color = temp_color(70.0);
        assert!(color.r > 0.5, "70°C should be warm");
    }

    // F-FORMAT-048: temp_color boundary 50°C
    #[test]
    fn test_temp_color_boundary_50() {
        let color = temp_color(50.0);
        assert!(color.g > 0.5, "50°C should be yellow");
    }

    // F-FORMAT-049: temp_color boundary 35°C
    #[test]
    fn test_temp_color_boundary_35() {
        let color = temp_color(35.0);
        assert!(color.g > color.r, "35°C should be green");
    }

    // F-FORMAT-050: format_bytes edge case 1023 bytes
    #[test]
    fn test_format_bytes_edge_1023() {
        assert_eq!(format_bytes(1023), "1023 B");
    }
}
