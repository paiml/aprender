//! TUI widget utilities - Color functions and rendering helpers.
//!
//! Provides safe percentage calculations, color coding, and format helpers
//! for the job flow TUI.

use ratatui::style::Color;

// =============================================================================
// Safety Utilities
// =============================================================================

/// Calculate percentage safely, preventing NaN/Inf and clamping to 0-100.
#[inline]
#[must_use]
pub fn safe_pct(used: f64, total: f64) -> f64 {
    if total <= 0.0 || used.is_nan() || total.is_nan() {
        return 0.0;
    }
    let pct = (used / total) * 100.0;
    if pct.is_nan() || pct.is_infinite() {
        0.0
    } else {
        pct.clamp(0.0, 100.0)
    }
}

/// Calculate bar width safely, preventing overflow.
#[inline]
#[must_use]
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn safe_bar(pct: f64, max_width: usize) -> usize {
    if pct.is_nan() || pct.is_infinite() || pct < 0.0 || max_width == 0 {
        return 0;
    }
    let width = (pct / 100.0 * max_width as f64).round() as usize;
    width.min(max_width)
}

// =============================================================================
// Color Functions
// =============================================================================

/// Color based on utilization percentage.
/// - Green: < 50%
/// - Yellow: 50-80%
/// - Red: > 80%
#[inline]
#[must_use]
pub fn utilization_color(pct: f64) -> Color {
    if pct.is_nan() || pct < 50.0 {
        Color::Green
    } else if pct < 80.0 {
        Color::Yellow
    } else {
        Color::Red
    }
}

/// Color based on latency.
/// - Green: < 10ms
/// - Yellow: 10-50ms
/// - Red: > 50ms
#[inline]
#[must_use]
pub const fn latency_color(ms: u32) -> Color {
    if ms < 10 {
        Color::Green
    } else if ms < 50 {
        Color::Yellow
    } else {
        Color::Red
    }
}

/// Color based on temperature.
/// - Green: < 60C
/// - Yellow: 60-80C
/// - Red: > 80C
#[inline]
#[must_use]
pub fn temp_color(temp_c: f64) -> Color {
    if temp_c.is_nan() || temp_c < 60.0 {
        Color::Green
    } else if temp_c < 80.0 {
        Color::Yellow
    } else {
        Color::Red
    }
}

/// Color for node state.
#[inline]
#[must_use]
pub const fn state_color(online: bool) -> Color {
    if online {
        Color::Green
    } else {
        Color::Red
    }
}

/// Color for success/failure.
#[inline]
#[must_use]
pub const fn success_color(success: bool) -> Color {
    if success {
        Color::Green
    } else {
        Color::Red
    }
}

// =============================================================================
// Format Helpers
// =============================================================================

/// Format duration as human-readable string.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn format_duration(duration: std::time::Duration) -> String {
    let millis = duration.as_millis();
    if millis < 1000 {
        format!("{millis}ms")
    } else if millis < 60_000 {
        format!("{:.1}s", millis as f64 / 1000.0)
    } else {
        let secs = duration.as_secs();
        format!("{}m{}s", secs / 60, secs % 60)
    }
}

/// Format bytes as human-readable string.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.1}GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1}MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1}KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes}B")
    }
}

/// Create a text-based progress bar.
#[must_use]
pub fn progress_bar(pct: f64, width: usize) -> String {
    let filled = safe_bar(pct, width);
    let empty = width.saturating_sub(filled);
    format!("{}{}", "█".repeat(filled), "░".repeat(empty))
}

/// Truncate string to max length with ellipsis.
#[must_use]
pub fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else if max_len <= 3 {
        "...".chars().take(max_len).collect()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::uninlined_format_args,
    clippy::panic
)]
mod tests {
    use super::*;

    // =========================================================================
    // Safe Percentage Tests
    // =========================================================================

    #[test]
    fn test_safe_pct_normal() {
        assert!((safe_pct(50.0, 100.0) - 50.0).abs() < 0.001);
        assert!((safe_pct(0.0, 100.0) - 0.0).abs() < 0.001);
        assert!((safe_pct(100.0, 100.0) - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_safe_pct_edge_cases() {
        assert_eq!(safe_pct(50.0, 0.0), 0.0);
        assert_eq!(safe_pct(f64::NAN, 100.0), 0.0);
        assert_eq!(safe_pct(50.0, f64::NAN), 0.0);
        assert_eq!(safe_pct(f64::INFINITY, 100.0), 0.0);
    }

    #[test]
    fn test_safe_pct_clamps() {
        assert_eq!(safe_pct(150.0, 100.0), 100.0);
        assert_eq!(safe_pct(-50.0, 100.0), 0.0);
    }

    // =========================================================================
    // Safe Bar Tests
    // =========================================================================

    #[test]
    fn test_safe_bar_normal() {
        assert_eq!(safe_bar(50.0, 10), 5);
        assert_eq!(safe_bar(100.0, 10), 10);
        assert_eq!(safe_bar(0.0, 10), 0);
    }

    #[test]
    fn test_safe_bar_edge_cases() {
        assert_eq!(safe_bar(f64::NAN, 10), 0);
        assert_eq!(safe_bar(f64::INFINITY, 10), 0);
        assert_eq!(safe_bar(-50.0, 10), 0);
        assert_eq!(safe_bar(50.0, 0), 0);
    }

    // =========================================================================
    // Color Tests
    // =========================================================================

    #[test]
    fn test_utilization_color_green() {
        assert_eq!(utilization_color(0.0), Color::Green);
        assert_eq!(utilization_color(49.9), Color::Green);
    }

    #[test]
    fn test_utilization_color_yellow() {
        assert_eq!(utilization_color(50.0), Color::Yellow);
        assert_eq!(utilization_color(79.9), Color::Yellow);
    }

    #[test]
    fn test_utilization_color_red() {
        assert_eq!(utilization_color(80.0), Color::Red);
        assert_eq!(utilization_color(100.0), Color::Red);
    }

    #[test]
    fn test_utilization_color_nan() {
        assert_eq!(utilization_color(f64::NAN), Color::Green);
    }

    #[test]
    fn test_latency_color() {
        assert_eq!(latency_color(5), Color::Green);
        assert_eq!(latency_color(25), Color::Yellow);
        assert_eq!(latency_color(100), Color::Red);
    }

    #[test]
    fn test_temp_color() {
        assert_eq!(temp_color(50.0), Color::Green);
        assert_eq!(temp_color(70.0), Color::Yellow);
        assert_eq!(temp_color(85.0), Color::Red);
    }

    #[test]
    fn test_state_color() {
        assert_eq!(state_color(true), Color::Green);
        assert_eq!(state_color(false), Color::Red);
    }

    #[test]
    fn test_success_color() {
        assert_eq!(success_color(true), Color::Green);
        assert_eq!(success_color(false), Color::Red);
    }

    // =========================================================================
    // Format Tests
    // =========================================================================

    #[test]
    fn test_format_duration_ms() {
        assert_eq!(format_duration(std::time::Duration::from_millis(45)), "45ms");
        assert_eq!(format_duration(std::time::Duration::from_millis(999)), "999ms");
    }

    #[test]
    fn test_format_duration_seconds() {
        assert_eq!(format_duration(std::time::Duration::from_millis(1500)), "1.5s");
        assert_eq!(format_duration(std::time::Duration::from_secs(30)), "30.0s");
    }

    #[test]
    fn test_format_duration_minutes() {
        assert_eq!(format_duration(std::time::Duration::from_secs(90)), "1m30s");
        assert_eq!(format_duration(std::time::Duration::from_secs(3600)), "60m0s");
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500B");
        assert_eq!(format_bytes(1500), "1.5KB");
        assert_eq!(format_bytes(1_500_000), "1.4MB");
        assert_eq!(format_bytes(1_500_000_000), "1.4GB");
    }

    #[test]
    fn test_progress_bar() {
        assert_eq!(progress_bar(50.0, 10), "█████░░░░░");
        assert_eq!(progress_bar(100.0, 10), "██████████");
        assert_eq!(progress_bar(0.0, 10), "░░░░░░░░░░");
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("this is a long string", 10), "this is...");
        assert_eq!(truncate("abc", 3), "abc");
        assert_eq!(truncate("abcd", 3), "...");
    }

    // =========================================================================
    // Property-Based Tests
    // =========================================================================

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_safe_pct_bounded(used in -1000.0f64..1000.0, total in 0.1f64..1000.0) {
            let pct = safe_pct(used, total);
            prop_assert!((0.0..=100.0).contains(&pct));
        }

        #[test]
        fn prop_safe_bar_bounded(pct in 0.0f64..100.0, width in 1usize..100) {
            let bar = safe_bar(pct, width);
            prop_assert!(bar <= width);
        }

        #[test]
        fn prop_progress_bar_length(pct in 0.0f64..100.0, width in 1usize..50) {
            let bar = progress_bar(pct, width);
            // Each character is a multi-byte UTF-8 character
            prop_assert_eq!(bar.chars().count(), width);
        }

        #[test]
        fn prop_truncate_length(s in "[a-z]{0,100}", max_len in 1usize..50) {
            let result = truncate(&s, max_len);
            prop_assert!(result.len() <= max_len + 2); // +2 for possible multi-byte chars
        }
    }
}
