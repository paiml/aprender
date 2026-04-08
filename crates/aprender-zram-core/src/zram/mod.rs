//! Zram device management.
//!
//! This module provides a shim-style abstraction over Linux zram devices,
//! allowing the logic to be tested without mocking filesystem operations.

mod device;
mod ops;

pub use device::{ZramConfig, ZramDevice, ZramStatus};
pub use ops::{SysfsOps, ZramOps};

use crate::{Error, Result};
use std::path::Path;

/// Parse size string (e.g., "4G", "512M", "ram/2") to bytes.
pub fn parse_size(size: &str) -> Result<u64> {
    let size = size.trim().to_uppercase();

    // Handle "ram/N" format
    if size.starts_with("RAM/") {
        let divisor: u64 = size[4..]
            .parse()
            .map_err(|_| Error::InvalidInput(format!("invalid RAM divisor: {}", &size[4..])))?;
        let total_ram = get_total_ram()?;
        return Ok(total_ram / divisor);
    }

    // Handle suffix format (K, M, G, T)
    let (num_str, multiplier) = if size.ends_with('K') {
        (&size[..size.len() - 1], 1024u64)
    } else if size.ends_with('M') {
        (&size[..size.len() - 1], 1024u64 * 1024)
    } else if size.ends_with('G') {
        (&size[..size.len() - 1], 1024u64 * 1024 * 1024)
    } else if size.ends_with('T') {
        (&size[..size.len() - 1], 1024u64 * 1024 * 1024 * 1024)
    } else {
        (size.as_str(), 1u64)
    };

    let num: u64 = num_str
        .parse()
        .map_err(|_| Error::InvalidInput(format!("invalid size number: {num_str}")))?;
    Ok(num * multiplier)
}

/// Format bytes as human-readable string.
pub fn format_size(bytes: u64) -> String {
    const GB: u64 = 1024 * 1024 * 1024;
    const MB: u64 = 1024 * 1024;
    const KB: u64 = 1024;

    if bytes >= GB {
        format!("{:.1}G", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1}M", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1}K", bytes as f64 / KB as f64)
    } else if bytes > 0 {
        format!("{bytes}B")
    } else {
        "0".to_string()
    }
}

/// Get total system RAM in bytes.
fn get_total_ram() -> Result<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo")
        .map_err(|e| Error::IoError(format!("failed to read /proc/meminfo: {e}")))?;

    for line in meminfo.lines() {
        if line.starts_with("MemTotal:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let kb: u64 = parts[1]
                    .parse()
                    .map_err(|_| Error::InvalidInput("invalid MemTotal value".to_string()))?;
                return Ok(kb * 1024);
            }
        }
    }
    Err(Error::InvalidInput(
        "could not determine total RAM".to_string(),
    ))
}

/// Find all existing zram devices.
pub fn find_devices() -> Vec<u32> {
    let mut devices = Vec::new();
    for i in 0..16 {
        let path = format!("/sys/block/zram{i}");
        if Path::new(&path).exists() {
            devices.push(i);
        }
    }
    devices
}

/// Check if zram module is loaded.
pub fn is_zram_available() -> bool {
    Path::new("/sys/class/zram-control").exists() || Path::new("/sys/block/zram0").exists()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_size_bytes() {
        assert_eq!(parse_size("1024").unwrap(), 1024);
        assert_eq!(parse_size("0").unwrap(), 0);
    }

    #[test]
    fn test_parse_size_kb() {
        assert_eq!(parse_size("4K").unwrap(), 4 * 1024);
        assert_eq!(parse_size("4k").unwrap(), 4 * 1024);
    }

    #[test]
    fn test_parse_size_mb() {
        assert_eq!(parse_size("512M").unwrap(), 512 * 1024 * 1024);
        assert_eq!(parse_size("1m").unwrap(), 1024 * 1024);
    }

    #[test]
    fn test_parse_size_gb() {
        assert_eq!(parse_size("4G").unwrap(), 4 * 1024 * 1024 * 1024);
        assert_eq!(parse_size("1g").unwrap(), 1024 * 1024 * 1024);
    }

    #[test]
    fn test_parse_size_tb() {
        assert_eq!(parse_size("1T").unwrap(), 1024u64 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_parse_size_invalid() {
        assert!(parse_size("invalid").is_err());
        assert!(parse_size("4X").is_err());
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(0), "0");
        assert_eq!(format_size(512), "512B");
        assert_eq!(format_size(4 * 1024), "4.0K");
        assert_eq!(format_size(512 * 1024 * 1024), "512.0M");
        assert_eq!(format_size(4 * 1024 * 1024 * 1024), "4.0G");
    }

    #[test]
    fn test_format_size_fractional() {
        assert_eq!(format_size(1536 * 1024 * 1024), "1.5G");
        assert_eq!(format_size(256 * 1024), "256.0K");
    }

    #[test]
    fn test_get_total_ram() {
        // Should work on any Linux system
        let ram = get_total_ram();
        if let Ok(ram) = ram {
            assert!(ram > 0);
        }
    }

    #[test]
    fn test_parse_size_ram_divisor() {
        // Only works if /proc/meminfo exists
        if get_total_ram().is_ok() {
            let half = parse_size("ram/2").unwrap();
            let quarter = parse_size("ram/4").unwrap();
            assert!(half > quarter);
            assert_eq!(half, quarter * 2);
        }
    }

    #[test]
    fn test_parse_size_ram_invalid_divisor() {
        let result = parse_size("ram/invalid");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("invalid RAM divisor"));
    }

    #[test]
    fn test_find_devices() {
        let devices = find_devices();
        // Should return a vector (may be empty if no zram)
        // On systems with zram, device 0 should be present
        if is_zram_available() && Path::new("/sys/block/zram0").exists() {
            assert!(devices.contains(&0));
        }
    }

    #[test]
    fn test_is_zram_available() {
        // Just verify it doesn't panic
        let available = is_zram_available();
        let _ = available;
    }

    #[test]
    fn test_format_size_edge_cases() {
        // Test boundaries
        assert_eq!(format_size(1023), "1023B");
        assert_eq!(format_size(1024), "1.0K");
        assert_eq!(format_size(1024 * 1024 - 1), "1024.0K");
        assert_eq!(format_size(1024 * 1024), "1.0M");
        assert_eq!(format_size(1024 * 1024 * 1024 - 1), "1024.0M");
        assert_eq!(format_size(1024 * 1024 * 1024), "1.0G");
    }
}
