//! Disk panel rendering and utilities.
//!
//! Provides disk panel title building, I/O rate formatting,
//! and helper functions for rendering disk metrics.

use super::super::helpers::format_bytes;
use presentar_core::Color;

// =============================================================================
// DISK TITLE BUILDING
// =============================================================================

/// Build disk panel title string.
///
/// Format: "Disk │ 256G / 1T (25%) │ R: 50MB/s W: 10MB/s"
#[must_use]
pub fn build_disk_title(
    used_bytes: u64,
    total_bytes: u64,
    read_rate: u64,
    write_rate: u64,
) -> String {
    let used_pct = if total_bytes > 0 {
        (used_bytes as f64 / total_bytes as f64) * 100.0
    } else {
        0.0
    };

    let used_str = format_bytes(used_bytes);
    let total_str = format_bytes(total_bytes);

    if read_rate > 0 || write_rate > 0 {
        let read_str = format_rate(read_rate);
        let write_str = format_rate(write_rate);
        format!(
            "Disk │ {} / {} ({:.0}%) │ R:{} W:{}",
            used_str, total_str, used_pct, read_str, write_str
        )
    } else {
        format!("Disk │ {} / {} ({:.0}%)", used_str, total_str, used_pct)
    }
}

/// Build compact disk title for narrow panels.
///
/// Format: "Disk │ 256G / 1T"
#[must_use]
pub fn build_disk_title_compact(used_bytes: u64, total_bytes: u64) -> String {
    format!(
        "Disk │ {} / {}",
        format_bytes(used_bytes),
        format_bytes(total_bytes)
    )
}

// =============================================================================
// RATE FORMATTING
// =============================================================================

/// Format I/O rate as human-readable string.
///
/// # Examples
/// - 0 -> "0B/s"
/// - 1024 -> "1.0K/s"
/// - 1048576 -> "1.0M/s"
#[must_use]
pub fn format_rate(bytes_per_sec: u64) -> String {
    if bytes_per_sec == 0 {
        return "0B/s".to_string();
    }

    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;

    let bytes = bytes_per_sec as f64;

    if bytes >= GB {
        format!("{:.1}G/s", bytes / GB)
    } else if bytes >= MB {
        format!("{:.1}M/s", bytes / MB)
    } else if bytes >= KB {
        format!("{:.1}K/s", bytes / KB)
    } else {
        format!("{}B/s", bytes_per_sec)
    }
}

// =============================================================================
// DISK USAGE COLORS
// =============================================================================

/// Get color for disk usage percentage.
#[must_use]
pub fn disk_usage_color(percent: f64) -> Color {
    if percent > 95.0 {
        Color::new(1.0, 0.3, 0.3, 1.0) // Critical red
    } else if percent > 85.0 {
        Color::new(1.0, 0.5, 0.2, 1.0) // Warning orange
    } else if percent > 70.0 {
        Color::new(1.0, 0.8, 0.2, 1.0) // Yellow
    } else {
        Color::new(0.3, 0.7, 1.0, 1.0) // Blue (disk color)
    }
}

/// Get color for I/O rate activity.
#[must_use]
pub fn io_activity_color(bytes_per_sec: u64) -> Color {
    const MB: u64 = 1024 * 1024;

    if bytes_per_sec > 100 * MB {
        Color::new(1.0, 0.4, 0.4, 1.0) // Heavy I/O - red
    } else if bytes_per_sec > 10 * MB {
        Color::new(1.0, 0.8, 0.3, 1.0) // Moderate I/O - yellow
    } else if bytes_per_sec > MB {
        Color::new(0.4, 0.8, 1.0, 1.0) // Light I/O - blue
    } else {
        Color::new(0.5, 0.5, 0.5, 1.0) // Idle - gray
    }
}

// =============================================================================
// DISK BAR SEGMENTS
// =============================================================================

/// Disk usage bar segment.
#[derive(Debug, Clone, PartialEq)]
pub struct DiskBarSegment {
    /// Segment label (e.g., "used", "free")
    pub label: String,
    /// Percentage of total (0.0-100.0)
    pub percent: f64,
    /// Segment color
    pub color: Color,
}

impl DiskBarSegment {
    /// Create a new disk bar segment.
    #[must_use]
    pub fn new(label: impl Into<String>, percent: f64, color: Color) -> Self {
        Self {
            label: label.into(),
            percent: percent.clamp(0.0, 100.0),
            color,
        }
    }

    /// Calculate width in characters for a given total bar width.
    #[must_use]
    pub fn char_width(&self, total_width: usize) -> usize {
        ((self.percent / 100.0) * total_width as f64).round() as usize
    }
}

/// Create standard disk usage segments.
#[must_use]
pub fn create_disk_segments(used_pct: f64, free_pct: f64) -> Vec<DiskBarSegment> {
    vec![
        DiskBarSegment::new(
            "used",
            used_pct,
            Color::new(0.4, 0.6, 1.0, 1.0), // Blue
        ),
        DiskBarSegment::new(
            "free",
            free_pct,
            Color::new(0.3, 0.3, 0.3, 1.0), // Dark gray
        ),
    ]
}

// =============================================================================
// MOUNT POINT UTILITIES
// =============================================================================

/// Try to shorten /home/user/path to ~/path format.
fn try_home_shortening(path: &str, max_width: usize) -> Option<String> {
    let short = path.replacen("/home/", "~/", 1);
    let rest = short.strip_prefix("~/")?;
    let idx = rest.find('/')?;
    let shortened = format!("~{}", &rest[idx..]);
    if shortened.chars().count() <= max_width {
        Some(shortened)
    } else {
        None
    }
}

/// Try to shorten path by showing last components with ellipsis.
fn try_component_shortening(parts: &[&str], max_width: usize) -> Option<String> {
    // Try last 2 components
    if parts.len() >= 2 {
        let last_two = format!(".../{}/{}", parts[parts.len() - 2], parts[parts.len() - 1]);
        if last_two.chars().count() <= max_width {
            return Some(last_two);
        }
    }

    // Just last component
    let last = format!(".../{}", parts.last().unwrap_or(&""));
    if last.chars().count() <= max_width {
        Some(last)
    } else {
        None
    }
}

/// Shorten mount point path for display.
///
/// "/home/user/data" -> "/home/user/data" (if fits)
/// "/home/user/data" -> "~/data" (if user path)
/// "/very/long/path/to/mount" -> ".../to/mount"
#[must_use]
pub fn shorten_mount_point(path: &str, max_width: usize) -> String {
    if path.chars().count() <= max_width {
        return path.to_string();
    }

    // Try to shorten /home/user to ~
    if path.starts_with("/home/") {
        if let Some(shortened) = try_home_shortening(path, max_width) {
            return shortened;
        }
    }

    // Fallback: show last few path components
    let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    if parts.len() <= 2 {
        return truncate_path(path, max_width);
    }

    try_component_shortening(&parts, max_width).unwrap_or_else(|| truncate_path(path, max_width))
}

/// Simple path truncation with ellipsis.
fn truncate_path(path: &str, max_width: usize) -> String {
    if max_width < 4 {
        return path.chars().take(max_width).collect();
    }

    let char_count = path.chars().count();
    if char_count <= max_width {
        path.to_string()
    } else {
        let truncated: String = path.chars().take(max_width - 3).collect();
        format!("{}...", truncated)
    }
}

// =============================================================================
// FILESYSTEM TYPE
// =============================================================================

/// Get display name for filesystem type.
#[must_use]
pub fn fs_type_display(fs_type: &str) -> &str {
    match fs_type {
        "ext4" => "ext4",
        "ext3" => "ext3",
        "ext2" => "ext2",
        "btrfs" => "btrfs",
        "xfs" => "xfs",
        "zfs" => "zfs",
        "ntfs" => "NTFS",
        "vfat" | "fat32" => "FAT32",
        "exfat" => "exFAT",
        "tmpfs" => "tmpfs",
        "devtmpfs" => "devfs",
        "overlay" => "overlay",
        "squashfs" => "squash",
        "nfs" | "nfs4" => "NFS",
        "cifs" | "smb" => "SMB",
        _ => fs_type,
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // build_disk_title tests
    // =========================================================================

    #[test]
    fn test_build_disk_title_with_io() {
        let title = build_disk_title(
            256 * 1024 * 1024 * 1024,  // 256G used
            1024 * 1024 * 1024 * 1024, // 1T total
            50 * 1024 * 1024,          // 50MB/s read
            10 * 1024 * 1024,          // 10MB/s write
        );
        assert!(title.contains("Disk"));
        assert!(title.contains("256.0G"));
        assert!(title.contains("1024.0G") || title.contains("1.0T"));
        assert!(title.contains("25%"));
        assert!(title.contains("R:"));
        assert!(title.contains("W:"));
    }

    #[test]
    fn test_build_disk_title_no_io() {
        let title = build_disk_title(100 * 1024 * 1024 * 1024, 500 * 1024 * 1024 * 1024, 0, 0);
        assert!(!title.contains("R:"));
        assert!(!title.contains("W:"));
    }

    #[test]
    fn test_build_disk_title_zero() {
        let title = build_disk_title(0, 0, 0, 0);
        assert!(title.contains("0%"));
    }

    #[test]
    fn test_build_disk_title_full() {
        let total = 1024 * 1024 * 1024 * 1024_u64;
        let title = build_disk_title(total, total, 0, 0);
        assert!(title.contains("100%"));
    }

    // =========================================================================
    // build_disk_title_compact tests
    // =========================================================================

    #[test]
    fn test_build_disk_title_compact() {
        let title = build_disk_title_compact(256 * 1024 * 1024 * 1024, 1024 * 1024 * 1024 * 1024);
        assert!(title.contains("Disk"));
        assert!(!title.contains("%")); // Compact doesn't show percentage
        assert!(!title.contains("R:")); // No I/O rates
    }

    // =========================================================================
    // format_rate tests
    // =========================================================================

    #[test]
    fn test_format_rate_zero() {
        assert_eq!(format_rate(0), "0B/s");
    }

    #[test]
    fn test_format_rate_bytes() {
        assert_eq!(format_rate(500), "500B/s");
    }

    #[test]
    fn test_format_rate_kb() {
        assert_eq!(format_rate(1024), "1.0K/s");
        assert_eq!(format_rate(2048), "2.0K/s");
    }

    #[test]
    fn test_format_rate_mb() {
        assert_eq!(format_rate(1024 * 1024), "1.0M/s");
        assert_eq!(format_rate(50 * 1024 * 1024), "50.0M/s");
    }

    #[test]
    fn test_format_rate_gb() {
        assert_eq!(format_rate(1024 * 1024 * 1024), "1.0G/s");
    }

    // =========================================================================
    // disk_usage_color tests
    // =========================================================================

    #[test]
    fn test_disk_usage_color_low() {
        let color = disk_usage_color(50.0);
        assert!(color.b > 0.9, "Low usage should be blue");
    }

    #[test]
    fn test_disk_usage_color_medium() {
        let color = disk_usage_color(75.0);
        assert!(color.r > 0.9, "Medium usage should be yellow");
        assert!(color.g > 0.7);
    }

    #[test]
    fn test_disk_usage_color_high() {
        let color = disk_usage_color(90.0);
        assert!(color.r > 0.9, "High usage should be orange");
    }

    #[test]
    fn test_disk_usage_color_critical() {
        let color = disk_usage_color(98.0);
        assert!(color.r > 0.9, "Critical should be red");
        assert!(color.g < 0.5);
    }

    // =========================================================================
    // io_activity_color tests
    // =========================================================================

    #[test]
    fn test_io_activity_color_idle() {
        let color = io_activity_color(0);
        assert!((color.r - color.g).abs() < 0.1, "Idle should be gray");
    }

    #[test]
    fn test_io_activity_color_light() {
        let color = io_activity_color(5 * 1024 * 1024);
        assert!(color.b > 0.9, "Light I/O should be blue");
    }

    #[test]
    fn test_io_activity_color_moderate() {
        let color = io_activity_color(50 * 1024 * 1024);
        assert!(
            color.r > 0.9 && color.g > 0.7,
            "Moderate I/O should be yellow"
        );
    }

    #[test]
    fn test_io_activity_color_heavy() {
        let color = io_activity_color(200 * 1024 * 1024);
        assert!(color.r > 0.9 && color.g < 0.5, "Heavy I/O should be red");
    }

    // =========================================================================
    // DiskBarSegment tests
    // =========================================================================

    #[test]
    fn test_disk_bar_segment_new() {
        let seg = DiskBarSegment::new("used", 50.0, Color::new(1.0, 0.0, 0.0, 1.0));
        assert_eq!(seg.label, "used");
        assert_eq!(seg.percent, 50.0);
    }

    #[test]
    fn test_disk_bar_segment_clamp() {
        let seg = DiskBarSegment::new("test", 150.0, Color::new(1.0, 0.0, 0.0, 1.0));
        assert_eq!(seg.percent, 100.0); // Clamped

        let seg2 = DiskBarSegment::new("test", -10.0, Color::new(1.0, 0.0, 0.0, 1.0));
        assert_eq!(seg2.percent, 0.0); // Clamped
    }

    #[test]
    fn test_disk_bar_segment_char_width() {
        let seg = DiskBarSegment::new("used", 50.0, Color::new(1.0, 0.0, 0.0, 1.0));
        assert_eq!(seg.char_width(20), 10);
        assert_eq!(seg.char_width(100), 50);
    }

    #[test]
    fn test_disk_bar_segment_derive_debug() {
        let seg = DiskBarSegment::new("test", 50.0, Color::new(1.0, 0.0, 0.0, 1.0));
        let debug = format!("{:?}", seg);
        assert!(debug.contains("DiskBarSegment"));
    }

    #[test]
    fn test_disk_bar_segment_derive_clone() {
        let seg = DiskBarSegment::new("test", 50.0, Color::new(1.0, 0.0, 0.0, 1.0));
        let cloned = seg.clone();
        assert_eq!(seg, cloned);
    }

    // =========================================================================
    // create_disk_segments tests
    // =========================================================================

    #[test]
    fn test_create_disk_segments() {
        let segments = create_disk_segments(75.0, 25.0);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].label, "used");
        assert_eq!(segments[0].percent, 75.0);
        assert_eq!(segments[1].label, "free");
        assert_eq!(segments[1].percent, 25.0);
    }

    // =========================================================================
    // shorten_mount_point tests
    // =========================================================================

    #[test]
    fn test_shorten_mount_point_fits() {
        let result = shorten_mount_point("/home", 20);
        assert_eq!(result, "/home");
    }

    #[test]
    fn test_shorten_mount_point_long() {
        let result = shorten_mount_point("/very/long/path/to/mount", 15);
        assert!(result.chars().count() <= 15);
        assert!(result.contains("..."));
    }

    #[test]
    fn test_shorten_mount_point_home() {
        let result = shorten_mount_point("/home/user/data/files", 15);
        // Should try to shorten /home/user
        assert!(result.chars().count() <= 15);
    }

    #[test]
    fn test_shorten_mount_point_root() {
        let result = shorten_mount_point("/", 10);
        assert_eq!(result, "/");
    }

    // =========================================================================
    // fs_type_display tests
    // =========================================================================

    #[test]
    fn test_fs_type_display_ext4() {
        assert_eq!(fs_type_display("ext4"), "ext4");
    }

    #[test]
    fn test_fs_type_display_ntfs() {
        assert_eq!(fs_type_display("ntfs"), "NTFS");
    }

    #[test]
    fn test_fs_type_display_vfat() {
        assert_eq!(fs_type_display("vfat"), "FAT32");
    }

    #[test]
    fn test_fs_type_display_nfs() {
        assert_eq!(fs_type_display("nfs"), "NFS");
        assert_eq!(fs_type_display("nfs4"), "NFS");
    }

    #[test]
    fn test_fs_type_display_cifs() {
        assert_eq!(fs_type_display("cifs"), "SMB");
        assert_eq!(fs_type_display("smb"), "SMB");
    }

    #[test]
    fn test_fs_type_display_unknown() {
        assert_eq!(fs_type_display("myfs"), "myfs");
    }

    #[test]
    fn test_fs_type_display_tmpfs() {
        assert_eq!(fs_type_display("tmpfs"), "tmpfs");
        assert_eq!(fs_type_display("devtmpfs"), "devfs");
    }
}
