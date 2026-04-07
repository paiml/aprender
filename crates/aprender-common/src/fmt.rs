//! Formatting utilities for bytes, percentages, durations, and numbers.
//!
//! Consolidates byte-formatting implementations previously duplicated across
//! pmat, trueno-viz, aprender, and trueno.

// Byte formatting intentionally casts u64 -> f64 for display purposes.
// Precision loss beyond 2^52 is acceptable for human-readable output.
#![allow(clippy::cast_precision_loss)]

// =============================================================================
// BYTE FORMATTING
// =============================================================================

/// Format bytes using SI units (powers of 1000).
///
/// Uses adaptive precision: 3+ digits show no decimal, 2 digits show 1 decimal,
/// 1 digit shows 2 decimals.
///
/// # Examples
/// ```
/// use batuta_common::fmt::format_bytes_si;
/// assert_eq!(format_bytes_si(0), "0B");
/// assert_eq!(format_bytes_si(500), "500B");
/// assert_eq!(format_bytes_si(1500), "1.50K");
/// assert_eq!(format_bytes_si(1_500_000), "1.50M");
/// assert_eq!(format_bytes_si(1_500_000_000), "1.50G");
/// ```
#[must_use]
pub fn format_bytes_si(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "K", "M", "G", "T", "P", "E"];
    const THRESHOLD: f64 = 1000.0;

    if bytes == 0 {
        return "0B".to_string();
    }

    let mut value = bytes as f64;
    let mut unit_idx = 0;

    while value >= THRESHOLD && unit_idx < UNITS.len() - 1 {
        value /= THRESHOLD;
        unit_idx += 1;
    }

    if unit_idx == 0 {
        format!("{bytes}B")
    } else if value >= 100.0 {
        format!("{value:.0}{}", UNITS[unit_idx])
    } else if value >= 10.0 {
        format!("{value:.1}{}", UNITS[unit_idx])
    } else {
        format!("{value:.2}{}", UNITS[unit_idx])
    }
}

/// Format bytes using IEC units (powers of 1024).
///
/// # Examples
/// ```
/// use batuta_common::fmt::format_bytes_iec;
/// assert_eq!(format_bytes_iec(0), "0B");
/// assert_eq!(format_bytes_iec(1024), "1.00KiB");
/// assert_eq!(format_bytes_iec(1536), "1.50KiB");
/// ```
#[must_use]
pub fn format_bytes_iec(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"];
    const THRESHOLD: f64 = 1024.0;

    if bytes == 0 {
        return "0B".to_string();
    }

    let mut value = bytes as f64;
    let mut unit_idx = 0;

    while value >= THRESHOLD && unit_idx < UNITS.len() - 1 {
        value /= THRESHOLD;
        unit_idx += 1;
    }

    if unit_idx == 0 {
        format!("{bytes}B")
    } else {
        format!("{value:.2}{}", UNITS[unit_idx])
    }
}

/// Format bytes in human-readable form (IEC-style with short units).
///
/// Uses "KB", "MB", "GB" labels (1024-based) with 1 decimal place.
/// This is the most common format for CLI output.
///
/// # Examples
/// ```
/// use batuta_common::fmt::format_bytes;
/// assert_eq!(format_bytes(0), "0 B");
/// assert_eq!(format_bytes(1023), "1023 B");
/// assert_eq!(format_bytes(1024), "1.0 KB");
/// assert_eq!(format_bytes(1_048_576), "1.0 MB");
/// ```
#[must_use]
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{bytes} {}", UNITS[unit_index])
    } else {
        format!("{size:.1} {}", UNITS[unit_index])
    }
}

/// Format bytes compactly with short IEC-based units and no spaces.
///
/// Uses 1024-based thresholds with single-letter units (K, M, G, T)
/// and 1 decimal place. Values below 1024 are shown as fractional K.
/// Intended for tight UI columns like disk panels.
///
/// # Examples
/// ```
/// use batuta_common::fmt::format_bytes_compact;
/// assert_eq!(format_bytes_compact(1_073_741_824), "1.0G");
/// assert_eq!(format_bytes_compact(1_048_576), "1.0M");
/// assert_eq!(format_bytes_compact(1024), "1.0K");
/// assert_eq!(format_bytes_compact(512), "0.5K");
/// ```
#[must_use]
pub fn format_bytes_compact(bytes: u64) -> String {
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
    } else {
        format!("{:.1}K", bytes as f64 / KB as f64)
    }
}

/// Format bytes with variable precision and full unit labels.
///
/// Uses 1024-based thresholds. TB and GB use 2 decimal places for
/// precision; MB and KB use 1 decimal place. Values below 1024 show
/// the raw count with " B" suffix. Intended for transfer totals
/// and `PCIe` bandwidth displays.
///
/// # Examples
/// ```
/// use batuta_common::fmt::format_bytes_full;
/// assert_eq!(format_bytes_full(1_099_511_627_776), "1.00 TB");
/// assert_eq!(format_bytes_full(1_073_741_824), "1.00 GB");
/// assert_eq!(format_bytes_full(1_048_576), "1.0 MB");
/// assert_eq!(format_bytes_full(1024), "1.0 KB");
/// assert_eq!(format_bytes_full(500), "500 B");
/// ```
#[must_use]
pub fn format_bytes_full(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    const TB: u64 = GB * 1024;

    if bytes >= TB {
        format!("{:.2} TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// Format bytes per second as a rate string.
///
/// # Examples
/// ```
/// use batuta_common::fmt::format_bytes_rate;
/// assert_eq!(format_bytes_rate(1500.0), "1.50K/s");
/// ```
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn format_bytes_rate(bytes_per_sec: f64) -> String {
    format!("{}/s", format_bytes_si(bytes_per_sec as u64))
}

// =============================================================================
// PERCENTAGE FORMATTING
// =============================================================================

/// Format a percentage value (0.0 to 100.0) with 1 decimal place.
///
/// # Examples
/// ```
/// use batuta_common::fmt::format_percent;
/// assert_eq!(format_percent(45.3), "45.3%");
/// assert_eq!(format_percent(100.0), "100.0%");
/// ```
#[must_use]
pub fn format_percent(value: f64) -> String {
    format!("{value:.1}%")
}

/// Format a percentage clamped to 0-100 range.
///
/// # Examples
/// ```
/// use batuta_common::fmt::format_percent_clamped;
/// assert_eq!(format_percent_clamped(150.0), "100.0%");
/// assert_eq!(format_percent_clamped(-10.0), "0.0%");
/// ```
#[must_use]
pub fn format_percent_clamped(value: f64) -> String {
    format_percent(value.clamp(0.0, 100.0))
}

/// Format a percentage with fixed decimal places.
///
/// # Examples
/// ```
/// use batuta_common::fmt::format_percent_fixed;
/// assert_eq!(format_percent_fixed(45.333, 2), "45.33%");
/// ```
#[must_use]
pub fn format_percent_fixed(value: f64, decimals: usize) -> String {
    format!("{value:.decimals$}%")
}

/// Compute and format a percentage from a fraction.
///
/// # Examples
/// ```
/// use batuta_common::fmt::usage_percent;
/// assert_eq!(usage_percent(50, 200), "25.0%");
/// assert_eq!(usage_percent(0, 0), "0.0%");
/// ```
#[must_use]
pub fn usage_percent(used: u64, total: u64) -> String {
    if total == 0 {
        return "0.0%".to_string();
    }
    format_percent(used as f64 / total as f64 * 100.0)
}

// =============================================================================
// DURATION FORMATTING
// =============================================================================

/// Format a duration in seconds to human-readable form.
///
/// # Examples
/// ```
/// use batuta_common::fmt::format_duration;
/// assert_eq!(format_duration(45), "45s");
/// assert_eq!(format_duration(125), "2m 5s");
/// assert_eq!(format_duration(3725), "1h 2m");
/// assert_eq!(format_duration(90061), "1d 1h");
/// ```
#[must_use]
pub fn format_duration(seconds: u64) -> String {
    const MINUTE: u64 = 60;
    const HOUR: u64 = 60 * MINUTE;
    const DAY: u64 = 24 * HOUR;

    if seconds < MINUTE {
        format!("{seconds}s")
    } else if seconds < HOUR {
        let mins = seconds / MINUTE;
        let secs = seconds % MINUTE;
        if secs == 0 {
            format!("{mins}m")
        } else {
            format!("{mins}m {secs}s")
        }
    } else if seconds < DAY {
        let hours = seconds / HOUR;
        let mins = (seconds % HOUR) / MINUTE;
        if mins == 0 {
            format!("{hours}h")
        } else {
            format!("{hours}h {mins}m")
        }
    } else {
        let days = seconds / DAY;
        let hours = (seconds % DAY) / HOUR;
        if hours == 0 {
            format!("{days}d")
        } else {
            format!("{days}d {hours}h")
        }
    }
}

/// Format a duration compactly (always fixed-width, 6 chars).
///
/// # Examples
/// ```
/// use batuta_common::fmt::format_duration_compact;
/// assert_eq!(format_duration_compact(45), "   45s");
/// assert_eq!(format_duration_compact(3725), " 1h02m");
/// ```
#[must_use]
pub fn format_duration_compact(seconds: u64) -> String {
    const MINUTE: u64 = 60;
    const HOUR: u64 = 60 * MINUTE;
    const DAY: u64 = 24 * HOUR;

    if seconds < MINUTE {
        format!("{seconds:>5}s")
    } else if seconds < HOUR {
        let mins = seconds / MINUTE;
        let secs = seconds % MINUTE;
        format!("{mins:>2}m{secs:02}s")
    } else if seconds < DAY {
        let hours = seconds / HOUR;
        let mins = (seconds % HOUR) / MINUTE;
        format!("{hours:>2}h{mins:02}m")
    } else {
        let days = seconds / DAY;
        let hours = (seconds % DAY) / HOUR;
        format!("{days:>2}d{hours:02}h")
    }
}

// =============================================================================
// NUMBER FORMATTING
// =============================================================================

/// Format a number with thousands separators.
///
/// # Examples
/// ```
/// use batuta_common::fmt::format_number;
/// assert_eq!(format_number(1234567), "1,234,567");
/// assert_eq!(format_number(999), "999");
/// assert_eq!(format_number(0), "0");
/// ```
#[must_use]
pub fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    let chars: Vec<char> = s.chars().collect();

    for (i, c) in chars.iter().enumerate() {
        if i > 0 && (chars.len() - i).is_multiple_of(3) {
            result.push(',');
        }
        result.push(*c);
    }

    result
}

/// Format frequency in MHz to human-readable GHz/MHz.
///
/// # Examples
/// ```
/// use batuta_common::fmt::format_freq_mhz;
/// assert_eq!(format_freq_mhz(3200), "3.2GHz");
/// assert_eq!(format_freq_mhz(800), "800MHz");
/// ```
#[must_use]
pub fn format_freq_mhz(mhz: u32) -> String {
    if mhz >= 1000 {
        format!("{:.1}GHz", f64::from(mhz) / 1000.0)
    } else {
        format!("{mhz}MHz")
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes_si_zero() {
        assert_eq!(format_bytes_si(0), "0B");
    }

    #[test]
    fn test_format_bytes_si_small() {
        assert_eq!(format_bytes_si(1), "1B");
        assert_eq!(format_bytes_si(999), "999B");
    }

    #[test]
    fn test_format_bytes_si_kilobytes() {
        assert_eq!(format_bytes_si(1000), "1.00K");
        assert_eq!(format_bytes_si(1500), "1.50K");
        assert_eq!(format_bytes_si(10_000), "10.0K");
        assert_eq!(format_bytes_si(100_000), "100K");
    }

    #[test]
    fn test_format_bytes_si_megabytes() {
        assert_eq!(format_bytes_si(1_000_000), "1.00M");
        assert_eq!(format_bytes_si(1_500_000), "1.50M");
    }

    #[test]
    fn test_format_bytes_si_gigabytes() {
        assert_eq!(format_bytes_si(1_000_000_000), "1.00G");
        assert_eq!(format_bytes_si(1_500_000_000), "1.50G");
    }

    #[test]
    fn test_format_bytes_si_terabytes() {
        assert_eq!(format_bytes_si(1_000_000_000_000), "1.00T");
    }

    #[test]
    fn test_format_bytes_iec_zero() {
        assert_eq!(format_bytes_iec(0), "0B");
    }

    #[test]
    fn test_format_bytes_iec_kib() {
        assert_eq!(format_bytes_iec(1024), "1.00KiB");
        assert_eq!(format_bytes_iec(1536), "1.50KiB");
    }

    #[test]
    fn test_format_bytes_iec_mib() {
        assert_eq!(format_bytes_iec(1_048_576), "1.00MiB");
    }

    #[test]
    fn test_format_bytes_human() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(1023), "1023 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1_048_576), "1.0 MB");
        assert_eq!(format_bytes(1_073_741_824), "1.0 GB");
    }

    #[test]
    fn test_format_bytes_compact() {
        assert_eq!(format_bytes_compact(0), "0.0K");
        assert_eq!(format_bytes_compact(512), "0.5K");
        assert_eq!(format_bytes_compact(1024), "1.0K");
        assert_eq!(format_bytes_compact(1_048_576), "1.0M");
        assert_eq!(format_bytes_compact(1_073_741_824), "1.0G");
        assert_eq!(format_bytes_compact(1_099_511_627_776), "1.0T");
    }

    #[test]
    fn test_format_bytes_full() {
        assert_eq!(format_bytes_full(500), "500 B");
        assert_eq!(format_bytes_full(1024), "1.0 KB");
        assert_eq!(format_bytes_full(1_048_576), "1.0 MB");
        assert_eq!(format_bytes_full(1_073_741_824), "1.00 GB");
        assert_eq!(format_bytes_full(1_099_511_627_776), "1.00 TB");
    }

    #[test]
    fn test_format_bytes_rate() {
        assert_eq!(format_bytes_rate(1500.0), "1.50K/s");
        assert_eq!(format_bytes_rate(0.0), "0B/s");
    }

    #[test]
    fn test_format_percent() {
        assert_eq!(format_percent(45.3), "45.3%");
        assert_eq!(format_percent(0.0), "0.0%");
        assert_eq!(format_percent(100.0), "100.0%");
    }

    #[test]
    fn test_format_percent_clamped() {
        assert_eq!(format_percent_clamped(150.0), "100.0%");
        assert_eq!(format_percent_clamped(-10.0), "0.0%");
        assert_eq!(format_percent_clamped(50.0), "50.0%");
    }

    #[test]
    fn test_format_percent_fixed() {
        assert_eq!(format_percent_fixed(45.333, 2), "45.33%");
        assert_eq!(format_percent_fixed(5.0, 0), "5%");
    }

    #[test]
    fn test_usage_percent() {
        assert_eq!(usage_percent(50, 200), "25.0%");
        assert_eq!(usage_percent(100, 100), "100.0%");
        assert_eq!(usage_percent(0, 0), "0.0%");
        assert_eq!(usage_percent(0, 100), "0.0%");
    }

    #[test]
    fn test_format_duration_seconds() {
        assert_eq!(format_duration(0), "0s");
        assert_eq!(format_duration(45), "45s");
        assert_eq!(format_duration(59), "59s");
    }

    #[test]
    fn test_format_duration_minutes() {
        assert_eq!(format_duration(60), "1m");
        assert_eq!(format_duration(125), "2m 5s");
        assert_eq!(format_duration(3599), "59m 59s");
    }

    #[test]
    fn test_format_duration_hours() {
        assert_eq!(format_duration(3600), "1h");
        assert_eq!(format_duration(3725), "1h 2m");
        assert_eq!(format_duration(86399), "23h 59m");
    }

    #[test]
    fn test_format_duration_days() {
        assert_eq!(format_duration(86400), "1d");
        assert_eq!(format_duration(90061), "1d 1h");
    }

    #[test]
    fn test_format_duration_compact() {
        assert_eq!(format_duration_compact(45), "   45s");
        assert_eq!(format_duration_compact(125), " 2m05s");
        assert_eq!(format_duration_compact(3725), " 1h02m");
        assert_eq!(format_duration_compact(90061), " 1d01h");
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(0), "0");
        assert_eq!(format_number(999), "999");
        assert_eq!(format_number(1000), "1,000");
        assert_eq!(format_number(1_234_567), "1,234,567");
    }

    #[test]
    fn test_format_freq_mhz() {
        assert_eq!(format_freq_mhz(800), "800MHz");
        assert_eq!(format_freq_mhz(3200), "3.2GHz");
        assert_eq!(format_freq_mhz(1000), "1.0GHz");
    }
}
