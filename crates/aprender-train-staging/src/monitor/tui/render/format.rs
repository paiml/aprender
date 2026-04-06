//! Formatting utilities for duration, bytes, and learning rate display.

use std::time::Duration;

pub fn format_duration(d: Duration) -> String {
    let secs = d.as_secs();
    format!("{:02}:{:02}:{:02}", secs / 3600, (secs % 3600) / 60, secs % 60)
}

#[allow(clippy::cast_precision_loss)]
pub fn format_bytes(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        // Both conversions are u64-to-f64 which may lose precision for very
        // large values, but formatting to one decimal place makes this benign.
        let gb_f = bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        format!("{gb_f:.1}G")
    } else if bytes >= 1024 * 1024 {
        let mb = bytes / (1024 * 1024);
        format!("{mb}M")
    } else if bytes >= 1024 {
        let kb = bytes / 1024;
        format!("{kb}K")
    } else {
        format!("{bytes}B")
    }
}

pub fn format_lr(lr: f32) -> String {
    if !lr.is_finite() {
        return "???".to_string();
    }
    let lr = lr.max(0.0);
    if lr >= 0.01 {
        format!("{lr:.4}")
    } else if lr >= 0.001 {
        format!("{lr:.5}")
    } else {
        format!("{lr:.6}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(Duration::from_secs(3661)), "01:01:01");
        assert_eq!(format_duration(Duration::from_secs(0)), "00:00:00");
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.0G");
        assert_eq!(format_bytes(512 * 1024 * 1024), "512M");
        assert_eq!(format_bytes(1024), "1K");
    }

    #[test]
    fn test_format_lr() {
        assert_eq!(format_lr(0.01), "0.0100");
        assert_eq!(format_lr(0.001), "0.00100");
        assert_eq!(format_lr(0.0001), "0.000100");
    }
}
