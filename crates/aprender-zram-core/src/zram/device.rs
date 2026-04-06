//! Zram device abstraction.

use crate::{Error, Result};
use std::fmt;

/// Zram device configuration.
#[derive(Debug, Clone)]
pub struct ZramConfig {
    /// Device number (0-15).
    pub device: u32,
    /// Size in bytes.
    pub size: u64,
    /// Compression algorithm.
    pub algorithm: String,
    /// Number of compression streams (0 = auto).
    pub streams: u32,
}

impl Default for ZramConfig {
    fn default() -> Self {
        Self { device: 0, size: 0, algorithm: "lz4".to_string(), streams: 0 }
    }
}

/// Status of a zram device.
#[derive(Debug, Clone, Default)]
pub struct ZramStatus {
    /// Device number.
    pub device: u32,
    /// Configured disk size in bytes.
    pub disksize: u64,
    /// Original data size (uncompressed) in bytes.
    pub orig_data_size: u64,
    /// Compressed data size in bytes.
    pub compr_data_size: u64,
    /// Total memory used in bytes.
    pub mem_used_total: u64,
    /// Compression algorithm.
    pub algorithm: String,
}

impl ZramStatus {
    /// Calculate compression ratio.
    #[must_use]
    pub fn compression_ratio(&self) -> f64 {
        if self.compr_data_size > 0 {
            self.orig_data_size as f64 / self.compr_data_size as f64
        } else {
            0.0
        }
    }

    /// Calculate space savings percentage.
    #[must_use]
    pub fn space_savings(&self) -> f64 {
        if self.orig_data_size > 0 {
            (1.0 - (self.compr_data_size as f64 / self.orig_data_size as f64)) * 100.0
        } else {
            0.0
        }
    }
}

impl fmt::Display for ZramStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "zram{}: {}B disksize, {}B data, {}B compressed ({:.2}x), {} algorithm",
            self.device,
            self.disksize,
            self.orig_data_size,
            self.compr_data_size,
            self.compression_ratio(),
            self.algorithm
        )
    }
}

/// Zram device handle.
#[derive(Debug)]
pub struct ZramDevice {
    /// Device number.
    pub device: u32,
    /// Sysfs path.
    sys_path: String,
    /// Device path.
    dev_path: String,
}

impl ZramDevice {
    /// Create a handle for an existing device.
    #[must_use]
    pub fn new(device: u32) -> Self {
        Self {
            device,
            sys_path: format!("/sys/block/zram{device}"),
            dev_path: format!("/dev/zram{device}"),
        }
    }

    /// Get the sysfs path.
    #[must_use]
    pub fn sys_path(&self) -> &str {
        &self.sys_path
    }

    /// Get the device path.
    #[must_use]
    pub fn dev_path(&self) -> &str {
        &self.dev_path
    }

    /// Check if device exists.
    #[must_use]
    pub fn exists(&self) -> bool {
        std::path::Path::new(&self.sys_path).exists()
    }

    /// Read a sysfs attribute as u64.
    pub fn read_attr_u64(&self, attr: &str) -> Result<u64> {
        let path = format!("{}/{attr}", self.sys_path);
        let content = std::fs::read_to_string(&path)
            .map_err(|e| Error::IoError(format!("failed to read {path}: {e}")))?;
        content.trim().parse().map_err(|_| Error::InvalidInput(format!("invalid value in {path}")))
    }

    /// Read a sysfs attribute as string.
    pub fn read_attr_str(&self, attr: &str) -> Result<String> {
        let path = format!("{}/{attr}", self.sys_path);
        let content = std::fs::read_to_string(&path)
            .map_err(|e| Error::IoError(format!("failed to read {path}: {e}")))?;
        Ok(content.trim().to_string())
    }

    /// Read the active compression algorithm.
    /// Parses the comp_algorithm sysfs file which contains all available algorithms
    /// with the selected one in brackets, e.g., "lzo lzo-rle [lz4] lz4hc 842 zstd"
    pub fn read_active_algorithm(&self) -> Result<String> {
        let raw = self.read_attr_str("comp_algorithm")?;
        // Extract the algorithm name from brackets [algo]
        if let Some(start) = raw.find('[') {
            if let Some(end) = raw.find(']') {
                if start < end {
                    return Ok(raw[start + 1..end].to_string());
                }
            }
        }
        // Fallback: return first word if no brackets found
        Ok(raw.split_whitespace().next().unwrap_or("unknown").to_string())
    }

    /// Write a sysfs attribute.
    pub fn write_attr(&self, attr: &str, value: &str) -> Result<()> {
        let path = format!("{}/{attr}", self.sys_path);
        std::fs::write(&path, value)
            .map_err(|e| Error::IoError(format!("failed to write {path}: {e}")))
    }

    /// Get device status.
    pub fn status(&self) -> Result<ZramStatus> {
        if !self.exists() {
            return Err(Error::InvalidInput(format!("zram{} does not exist", self.device)));
        }

        Ok(ZramStatus {
            device: self.device,
            disksize: self.read_attr_u64("disksize").unwrap_or(0),
            orig_data_size: self.read_attr_u64("orig_data_size").unwrap_or(0),
            compr_data_size: self.read_attr_u64("compr_data_size").unwrap_or(0),
            mem_used_total: self.read_attr_u64("mem_used_total").unwrap_or(0),
            algorithm: self.read_active_algorithm().unwrap_or_else(|_| "unknown".to_string()),
        })
    }

    /// Configure and activate the device.
    pub fn configure(&self, config: &ZramConfig) -> Result<()> {
        if !self.exists() {
            return Err(Error::InvalidInput(format!("zram{} does not exist", self.device)));
        }

        // Set algorithm first (must be done before disksize)
        if !config.algorithm.is_empty() {
            self.write_attr("comp_algorithm", &config.algorithm)?;
        }

        // Set streams if specified
        if config.streams > 0 {
            self.write_attr("max_comp_streams", &config.streams.to_string())?;
        }

        // Set disksize (this activates the device)
        self.write_attr("disksize", &config.size.to_string())
    }

    /// Reset the device (clears all data).
    pub fn reset(&self) -> Result<()> {
        self.write_attr("reset", "1")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zram_config_default() {
        let config = ZramConfig::default();
        assert_eq!(config.device, 0);
        assert_eq!(config.size, 0);
        assert_eq!(config.algorithm, "lz4");
        assert_eq!(config.streams, 0);
    }

    #[test]
    fn test_zram_config_clone() {
        let config =
            ZramConfig { device: 5, size: 1024, algorithm: "zstd".to_string(), streams: 4 };
        let cloned = config.clone();
        assert_eq!(cloned.device, 5);
        assert_eq!(cloned.size, 1024);
        assert_eq!(cloned.algorithm, "zstd");
        assert_eq!(cloned.streams, 4);
    }

    #[test]
    fn test_zram_config_debug() {
        let config = ZramConfig::default();
        let debug = format!("{config:?}");
        assert!(debug.contains("ZramConfig"));
        assert!(debug.contains("lz4"));
    }

    #[test]
    fn test_zram_status_ratio() {
        let status = ZramStatus {
            device: 0,
            disksize: 1024 * 1024,
            orig_data_size: 1000,
            compr_data_size: 500,
            mem_used_total: 600,
            algorithm: "lz4".to_string(),
        };
        assert!((status.compression_ratio() - 2.0).abs() < 0.001);
        assert!((status.space_savings() - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_zram_status_ratio_zero() {
        let status = ZramStatus::default();
        assert!(status.compression_ratio().abs() < f64::EPSILON);
        assert!(status.space_savings().abs() < f64::EPSILON);
    }

    #[test]
    fn test_zram_status_clone() {
        let status = ZramStatus {
            device: 1,
            disksize: 2048,
            orig_data_size: 1000,
            compr_data_size: 500,
            mem_used_total: 600,
            algorithm: "zstd".to_string(),
        };
        let cloned = status.clone();
        assert_eq!(cloned.device, 1);
        assert_eq!(cloned.disksize, 2048);
    }

    #[test]
    fn test_zram_status_debug() {
        let status = ZramStatus::default();
        let debug = format!("{status:?}");
        assert!(debug.contains("ZramStatus"));
    }

    #[test]
    fn test_zram_device_paths() {
        let dev = ZramDevice::new(5);
        assert_eq!(dev.device, 5);
        assert_eq!(dev.sys_path(), "/sys/block/zram5");
        assert_eq!(dev.dev_path(), "/dev/zram5");
    }

    #[test]
    fn test_zram_device_debug() {
        let dev = ZramDevice::new(0);
        let debug = format!("{dev:?}");
        assert!(debug.contains("ZramDevice"));
        assert!(debug.contains("zram0"));
    }

    // Use u32::MAX to ensure device doesn't exist (system won't have billions of zram devices)
    const NONEXISTENT_DEVICE: u32 = u32::MAX;

    #[test]
    fn test_zram_device_nonexistent() {
        let dev = ZramDevice::new(NONEXISTENT_DEVICE);
        assert!(!dev.exists());
    }

    #[test]
    fn test_zram_device_read_attr_nonexistent() {
        let dev = ZramDevice::new(NONEXISTENT_DEVICE);
        let result = dev.read_attr_u64("disksize");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("I/O error"));
    }

    #[test]
    fn test_zram_device_read_attr_str_nonexistent() {
        let dev = ZramDevice::new(NONEXISTENT_DEVICE);
        let result = dev.read_attr_str("comp_algorithm");
        assert!(result.is_err());
    }

    #[test]
    fn test_zram_device_write_attr_nonexistent() {
        let dev = ZramDevice::new(NONEXISTENT_DEVICE);
        let result = dev.write_attr("disksize", "1024");
        assert!(result.is_err());
    }

    #[test]
    fn test_zram_device_status_nonexistent() {
        let dev = ZramDevice::new(NONEXISTENT_DEVICE);
        let result = dev.status();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("does not exist"));
    }

    #[test]
    fn test_zram_device_configure_nonexistent() {
        let dev = ZramDevice::new(NONEXISTENT_DEVICE);
        let config = ZramConfig::default();
        let result = dev.configure(&config);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("does not exist"));
    }

    #[test]
    fn test_zram_device_reset_nonexistent() {
        let dev = ZramDevice::new(NONEXISTENT_DEVICE);
        let result = dev.reset();
        assert!(result.is_err());
    }

    #[test]
    fn test_zram_status_display() {
        let status = ZramStatus {
            device: 0,
            disksize: 1048576,
            orig_data_size: 1000,
            compr_data_size: 500,
            mem_used_total: 600,
            algorithm: "lz4".to_string(),
        };
        let s = format!("{status}");
        assert!(s.contains("zram0"));
        assert!(s.contains("lz4"));
        assert!(s.contains("2.00x"));
    }

    #[test]
    fn test_zram_status_display_zero_compression() {
        let status = ZramStatus {
            device: 1,
            disksize: 1024,
            orig_data_size: 0,
            compr_data_size: 0,
            mem_used_total: 0,
            algorithm: "zstd".to_string(),
        };
        let s = format!("{status}");
        assert!(s.contains("zram1"));
        assert!(s.contains("0.00x"));
    }

    #[test]
    fn test_zram_status_high_ratio() {
        let status = ZramStatus {
            device: 0,
            disksize: 1024 * 1024,
            orig_data_size: 10000,
            compr_data_size: 100,
            mem_used_total: 150,
            algorithm: "lz4".to_string(),
        };
        assert!((status.compression_ratio() - 100.0).abs() < 0.001);
        assert!((status.space_savings() - 99.0).abs() < 0.001);
    }
}
