//! UI helper functions for ptop.
//!
//! Formatting and utility functions used across panels.

// =============================================================================
// FORMATTING FUNCTIONS
// =============================================================================

/// Format bytes to human-readable string (e.g., "1.5G", "128M", "512K").
///
/// Uses binary units (1024-based):
/// - T = 1024^4 bytes
/// - G = 1024^3 bytes
/// - M = 1024^2 bytes
/// - K = 1024 bytes
///
/// # Examples
/// ```ignore
/// assert_eq!(format_bytes(1536), "1.5K");
/// assert_eq!(format_bytes(1073741824), "1.0G");
/// ```
#[must_use]
pub fn format_bytes(bytes: u64) -> String {
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
        format!("{bytes}B")
    }
}

/// Format bytes per second rate (e.g., "1.5G", "128M", "512K").
///
/// Similar to `format_bytes` but for transfer rates.
/// Uses one decimal place for GB/MB, no decimals for KB/B.
#[must_use]
pub fn format_bytes_rate(bytes_per_sec: f64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;

    if bytes_per_sec >= GB {
        format!("{:.1}G", bytes_per_sec / GB)
    } else if bytes_per_sec >= MB {
        format!("{:.1}M", bytes_per_sec / MB)
    } else if bytes_per_sec >= KB {
        format!("{:.0}K", bytes_per_sec / KB)
    } else {
        format!("{bytes_per_sec:.0}B")
    }
}

/// Format uptime seconds to human-readable string (e.g., "5d 3h", "2h 15m").
///
/// # Format
/// - Days + hours if >= 1 day
/// - Hours + minutes if >= 1 hour
/// - Just minutes otherwise
#[must_use]
pub fn format_uptime(secs: u64) -> String {
    let days = secs / 86400;
    let hours = (secs % 86400) / 3600;
    let mins = (secs % 3600) / 60;

    if days > 0 {
        format!("{days}d {hours}h")
    } else if hours > 0 {
        format!("{hours}h {mins}m")
    } else {
        format!("{mins}m")
    }
}

// =============================================================================
// SYMBOLS
// =============================================================================

/// Get symbol for PSI (Pressure Stall Information) percentage.
///
/// # Symbols
/// - "▲▲" - Critical (>50%)
/// - "▲" - High (>20%)
/// - "▼" - Medium (>5%)
/// - "◐" - Low (>1%)
/// - "—" - None (≤1%)
#[must_use]
pub fn pressure_symbol(pct: f64) -> &'static str {
    if pct > 50.0 {
        "▲▲"
    } else if pct > 20.0 {
        "▲"
    } else if pct > 5.0 {
        "▼"
    } else if pct > 1.0 {
        "◐"
    } else {
        "—"
    }
}

/// Get service name from well-known port number.
///
/// # Returns
/// - Service name for known ports (SSH, HTTP, etc.)
/// - "App" for application ports (9000-9999)
/// - Empty string for unknown ports
#[must_use]
pub fn port_to_service(port: u16) -> &'static str {
    match port {
        22 => "SSH",
        80 => "HTTP",
        443 => "HTTPS",
        53 => "DNS",
        25 => "SMTP",
        21 => "FTP",
        3306 => "MySQL",
        5432 => "Pgsql",
        6379 => "Redis",
        27017 => "Mongo",
        8080 => "HTTP",
        8443 => "HTTPS",
        9000..=9999 => "App",
        _ => "",
    }
}

// =============================================================================
// STANDARD COLORS (used across helper functions)
// =============================================================================

use presentar_core::Color;

/// Standard dim color for labels.
pub const DIM_COLOR: Color = Color {
    r: 0.3,
    g: 0.3,
    b: 0.3,
    a: 1.0,
};

/// Cyan color for cached memory.
pub const CACHED_COLOR: Color = Color {
    r: 0.3,
    g: 0.8,
    b: 0.9,
    a: 1.0,
};

/// Blue color for free memory.
pub const FREE_COLOR: Color = Color {
    r: 0.4,
    g: 0.4,
    b: 0.9,
    a: 1.0,
};

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // format_bytes tests
    // =========================================================================

    #[test]
    fn test_format_bytes_zero() {
        assert_eq!(format_bytes(0), "0B");
    }

    #[test]
    fn test_format_bytes_single_byte() {
        assert_eq!(format_bytes(1), "1B");
    }

    #[test]
    fn test_format_bytes_under_1k() {
        assert_eq!(format_bytes(512), "512B");
        assert_eq!(format_bytes(1023), "1023B");
    }

    #[test]
    fn test_format_bytes_exactly_1k() {
        assert_eq!(format_bytes(1024), "1.0K");
    }

    #[test]
    fn test_format_bytes_kilobytes() {
        assert_eq!(format_bytes(1536), "1.5K");
        assert_eq!(format_bytes(10 * 1024), "10.0K");
        assert_eq!(format_bytes(512 * 1024), "512.0K");
    }

    #[test]
    fn test_format_bytes_exactly_1m() {
        assert_eq!(format_bytes(1024 * 1024), "1.0M");
    }

    #[test]
    fn test_format_bytes_megabytes() {
        assert_eq!(format_bytes(1024 * 1024 * 2), "2.0M");
        assert_eq!(format_bytes(1024 * 1024 + 512 * 1024), "1.5M");
        assert_eq!(format_bytes(100 * 1024 * 1024), "100.0M");
    }

    #[test]
    fn test_format_bytes_exactly_1g() {
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.0G");
    }

    #[test]
    fn test_format_bytes_gigabytes() {
        let gb = 1024 * 1024 * 1024;
        assert_eq!(format_bytes(gb + gb / 2), "1.5G");
        assert_eq!(format_bytes(8 * gb), "8.0G");
        assert_eq!(format_bytes(16 * gb), "16.0G");
    }

    #[test]
    fn test_format_bytes_exactly_1t() {
        let tb: u64 = 1024 * 1024 * 1024 * 1024;
        assert_eq!(format_bytes(tb), "1.0T");
    }

    #[test]
    fn test_format_bytes_terabytes() {
        let tb: u64 = 1024 * 1024 * 1024 * 1024;
        assert_eq!(format_bytes(2 * tb), "2.0T");
        assert_eq!(format_bytes(tb + tb / 2), "1.5T");
    }

    #[test]
    fn test_format_bytes_large_terabytes() {
        let tb: u64 = 1024 * 1024 * 1024 * 1024;
        assert_eq!(format_bytes(100 * tb), "100.0T");
    }

    #[test]
    fn test_format_bytes_typical_ram_sizes() {
        let gb = 1024 * 1024 * 1024;
        assert_eq!(format_bytes(4 * gb), "4.0G");
        assert_eq!(format_bytes(8 * gb), "8.0G");
        assert_eq!(format_bytes(16 * gb), "16.0G");
        assert_eq!(format_bytes(32 * gb), "32.0G");
        assert_eq!(format_bytes(64 * gb), "64.0G");
    }

    #[test]
    fn test_format_bytes_typical_disk_sizes() {
        let gb: u64 = 1024 * 1024 * 1024;
        let tb: u64 = gb * 1024;
        assert_eq!(format_bytes(256 * gb), "256.0G");
        assert_eq!(format_bytes(512 * gb), "512.0G");
        assert_eq!(format_bytes(tb), "1.0T");
        assert_eq!(format_bytes(2 * tb), "2.0T");
    }

    // =========================================================================
    // format_bytes_rate tests
    // =========================================================================

    #[test]
    fn test_format_bytes_rate_zero() {
        assert_eq!(format_bytes_rate(0.0), "0B");
    }

    #[test]
    fn test_format_bytes_rate_under_1k() {
        assert_eq!(format_bytes_rate(512.0), "512B");
        assert_eq!(format_bytes_rate(1023.5), "1024B");
    }

    #[test]
    fn test_format_bytes_rate_kilobytes() {
        assert_eq!(format_bytes_rate(1024.0), "1K");
        assert_eq!(format_bytes_rate(2048.0), "2K");
        assert_eq!(format_bytes_rate(100.0 * 1024.0), "100K");
    }

    #[test]
    fn test_format_bytes_rate_megabytes() {
        let mb = 1024.0 * 1024.0;
        assert_eq!(format_bytes_rate(mb), "1.0M");
        assert_eq!(format_bytes_rate(1.5 * mb), "1.5M");
        assert_eq!(format_bytes_rate(100.0 * mb), "100.0M");
    }

    #[test]
    fn test_format_bytes_rate_gigabytes() {
        let gb = 1024.0 * 1024.0 * 1024.0;
        assert_eq!(format_bytes_rate(gb), "1.0G");
        assert_eq!(format_bytes_rate(1.5 * gb), "1.5G");
        assert_eq!(format_bytes_rate(10.0 * gb), "10.0G");
    }

    #[test]
    fn test_format_bytes_rate_typical_network_speeds() {
        let mb = 1024.0 * 1024.0;
        // 100 Mbps = ~12.5 MB/s
        assert_eq!(format_bytes_rate(12.5 * mb), "12.5M");
        // 1 Gbps = ~125 MB/s
        assert_eq!(format_bytes_rate(125.0 * mb), "125.0M");
    }

    // =========================================================================
    // format_uptime tests
    // =========================================================================

    #[test]
    fn test_format_uptime_zero() {
        assert_eq!(format_uptime(0), "0m");
    }

    #[test]
    fn test_format_uptime_seconds_only() {
        assert_eq!(format_uptime(30), "0m");
        assert_eq!(format_uptime(59), "0m");
    }

    #[test]
    fn test_format_uptime_minutes() {
        assert_eq!(format_uptime(60), "1m");
        assert_eq!(format_uptime(120), "2m");
        assert_eq!(format_uptime(45 * 60), "45m");
        assert_eq!(format_uptime(59 * 60 + 59), "59m");
    }

    #[test]
    fn test_format_uptime_hours() {
        assert_eq!(format_uptime(3600), "1h 0m");
        assert_eq!(format_uptime(3600 + 30 * 60), "1h 30m");
        assert_eq!(format_uptime(5 * 3600 + 15 * 60), "5h 15m");
        assert_eq!(format_uptime(23 * 3600 + 59 * 60), "23h 59m");
    }

    #[test]
    fn test_format_uptime_days() {
        assert_eq!(format_uptime(86400), "1d 0h");
        assert_eq!(format_uptime(86400 + 3600), "1d 1h");
        assert_eq!(format_uptime(5 * 86400 + 3 * 3600), "5d 3h");
        assert_eq!(format_uptime(30 * 86400 + 12 * 3600), "30d 12h");
    }

    #[test]
    fn test_format_uptime_typical_values() {
        // Fresh boot
        assert_eq!(format_uptime(300), "5m");
        // Few hours
        assert_eq!(format_uptime(7200), "2h 0m");
        // A day
        assert_eq!(format_uptime(86400), "1d 0h");
        // A week
        assert_eq!(format_uptime(7 * 86400), "7d 0h");
        // A month
        assert_eq!(format_uptime(30 * 86400), "30d 0h");
    }

    #[test]
    fn test_format_uptime_ignores_extra_seconds() {
        // Seconds are dropped, only full minutes count
        assert_eq!(format_uptime(61), "1m");
        assert_eq!(format_uptime(119), "1m");
    }

    // =========================================================================
    // pressure_symbol tests
    // =========================================================================

    #[test]
    fn test_pressure_symbol_none() {
        assert_eq!(pressure_symbol(0.0), "—");
        assert_eq!(pressure_symbol(0.5), "—");
        assert_eq!(pressure_symbol(1.0), "—");
    }

    #[test]
    fn test_pressure_symbol_low() {
        assert_eq!(pressure_symbol(1.5), "◐");
        assert_eq!(pressure_symbol(3.0), "◐");
        assert_eq!(pressure_symbol(5.0), "◐");
    }

    #[test]
    fn test_pressure_symbol_medium() {
        assert_eq!(pressure_symbol(5.1), "▼");
        assert_eq!(pressure_symbol(10.0), "▼");
        assert_eq!(pressure_symbol(20.0), "▼");
    }

    #[test]
    fn test_pressure_symbol_high() {
        assert_eq!(pressure_symbol(20.1), "▲");
        assert_eq!(pressure_symbol(30.0), "▲");
        assert_eq!(pressure_symbol(50.0), "▲");
    }

    #[test]
    fn test_pressure_symbol_critical() {
        assert_eq!(pressure_symbol(50.1), "▲▲");
        assert_eq!(pressure_symbol(75.0), "▲▲");
        assert_eq!(pressure_symbol(100.0), "▲▲");
    }

    #[test]
    fn test_pressure_symbol_boundary_values() {
        // Exact boundaries
        assert_eq!(pressure_symbol(1.0), "—"); // <= 1.0
        assert_eq!(pressure_symbol(5.0), "◐"); // > 1.0, <= 5.0
        assert_eq!(pressure_symbol(20.0), "▼"); // > 5.0, <= 20.0
        assert_eq!(pressure_symbol(50.0), "▲"); // > 20.0, <= 50.0
    }

    // =========================================================================
    // port_to_service tests
    // =========================================================================

    #[test]
    fn test_port_to_service_ssh() {
        assert_eq!(port_to_service(22), "SSH");
    }

    #[test]
    fn test_port_to_service_http() {
        assert_eq!(port_to_service(80), "HTTP");
        assert_eq!(port_to_service(8080), "HTTP");
    }

    #[test]
    fn test_port_to_service_https() {
        assert_eq!(port_to_service(443), "HTTPS");
        assert_eq!(port_to_service(8443), "HTTPS");
    }

    #[test]
    fn test_port_to_service_dns() {
        assert_eq!(port_to_service(53), "DNS");
    }

    #[test]
    fn test_port_to_service_smtp() {
        assert_eq!(port_to_service(25), "SMTP");
    }

    #[test]
    fn test_port_to_service_ftp() {
        assert_eq!(port_to_service(21), "FTP");
    }

    #[test]
    fn test_port_to_service_databases() {
        assert_eq!(port_to_service(3306), "MySQL");
        assert_eq!(port_to_service(5432), "Pgsql");
        assert_eq!(port_to_service(6379), "Redis");
        assert_eq!(port_to_service(27017), "Mongo");
    }

    #[test]
    fn test_port_to_service_app_range() {
        assert_eq!(port_to_service(9000), "App");
        assert_eq!(port_to_service(9001), "App");
        assert_eq!(port_to_service(9500), "App");
        assert_eq!(port_to_service(9999), "App");
    }

    #[test]
    fn test_port_to_service_unknown() {
        assert_eq!(port_to_service(0), "");
        assert_eq!(port_to_service(1), "");
        assert_eq!(port_to_service(12345), "");
        assert_eq!(port_to_service(65535), "");
    }

    #[test]
    fn test_port_to_service_just_outside_app_range() {
        assert_eq!(port_to_service(8999), "");
        assert_eq!(port_to_service(10000), "");
    }

    // =========================================================================
    // Color constant tests
    // =========================================================================

    #[test]
    fn test_dim_color_is_gray() {
        assert_eq!(DIM_COLOR.r, 0.3);
        assert_eq!(DIM_COLOR.g, 0.3);
        assert_eq!(DIM_COLOR.b, 0.3);
        assert_eq!(DIM_COLOR.a, 1.0);
    }

    #[test]
    fn test_cached_color_is_cyan() {
        assert!(
            CACHED_COLOR.b > CACHED_COLOR.r,
            "Cached should be cyan (more blue than red)"
        );
        assert!(CACHED_COLOR.g > 0.7, "Cached should have high green");
    }

    #[test]
    fn test_free_color_is_blue() {
        assert!(FREE_COLOR.b > FREE_COLOR.r, "Free should be blue");
        assert!(
            FREE_COLOR.b > FREE_COLOR.g,
            "Free should be more blue than green"
        );
    }

    #[test]
    fn test_all_helper_colors_have_full_alpha() {
        assert_eq!(DIM_COLOR.a, 1.0);
        assert_eq!(CACHED_COLOR.a, 1.0);
        assert_eq!(FREE_COLOR.a, 1.0);
    }

    // =========================================================================
    // Edge case tests
    // =========================================================================

    #[test]
    fn test_format_bytes_max_u64() {
        // Should not panic and should return terabytes
        let result = format_bytes(u64::MAX);
        assert!(result.contains("T"), "Max u64 should be in terabytes");
    }

    #[test]
    fn test_format_bytes_rate_negative() {
        // Negative rates shouldn't happen but should be handled
        let result = format_bytes_rate(-100.0);
        assert!(result.contains("B"), "Negative should still format");
    }

    #[test]
    fn test_format_bytes_rate_infinity() {
        let result = format_bytes_rate(f64::INFINITY);
        assert!(
            result.contains("inf") || result.contains("G"),
            "Infinity should be handled"
        );
    }

    #[test]
    fn test_format_bytes_rate_nan() {
        let result = format_bytes_rate(f64::NAN);
        assert!(
            result.contains("NaN") || result.contains("nan"),
            "NaN should be handled"
        );
    }

    #[test]
    fn test_pressure_symbol_negative() {
        // Negative should be treated as "none"
        assert_eq!(pressure_symbol(-10.0), "—");
    }

    #[test]
    fn test_pressure_symbol_very_high() {
        // Very high values should still be critical
        assert_eq!(pressure_symbol(1000.0), "▲▲");
    }
}
