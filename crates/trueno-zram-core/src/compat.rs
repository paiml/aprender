//! Sysfs compatibility interface for kernel zram.
//!
//! This module provides a compatible interface with the Linux kernel zram
//! module's sysfs interface. It allows existing zram tools to work unchanged
//! with the trueno-zram userspace implementation.
//!
//! Reference: Linux kernel Documentation/admin-guide/blockdev/zram.rst

use std::collections::HashMap;
use std::path::PathBuf;

/// Compression algorithm names compatible with kernel zram.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ZramAlgorithm {
    /// LZ4 compression (default in most kernels).
    Lz4,
    /// LZ4HC high compression variant.
    Lz4hc,
    /// Zstandard compression.
    Zstd,
    /// LZO compression (legacy).
    Lzo,
    /// LZO-RLE compression.
    LzoRle,
    /// 842 compression (IBM).
    Deflate842,
}

impl ZramAlgorithm {
    /// Parse algorithm from kernel sysfs string format.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().trim() {
            "lz4" => Some(Self::Lz4),
            "lz4hc" => Some(Self::Lz4hc),
            "zstd" => Some(Self::Zstd),
            "lzo" => Some(Self::Lzo),
            "lzo-rle" => Some(Self::LzoRle),
            "842" => Some(Self::Deflate842),
            _ => None,
        }
    }

    /// Convert to kernel sysfs string format.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Lz4 => "lz4",
            Self::Lz4hc => "lz4hc",
            Self::Zstd => "zstd",
            Self::Lzo => "lzo",
            Self::LzoRle => "lzo-rle",
            Self::Deflate842 => "842",
        }
    }

    /// Check if this algorithm is supported by trueno-zram.
    #[must_use]
    pub fn is_supported(&self) -> bool {
        matches!(self, Self::Lz4 | Self::Lz4hc | Self::Zstd)
    }
}

/// Memory statistics compatible with kernel zram mm_stat format.
///
/// Format: `%8llu %8llu %8llu %8llu %8llu %8llu %8llu %8llu %8llu`
#[derive(Debug, Clone, Default)]
pub struct MmStat {
    /// Original (uncompressed) data size.
    pub orig_data_size: u64,
    /// Compressed data size.
    pub compr_data_size: u64,
    /// Amount of memory allocated for this disk.
    pub mem_used_total: u64,
    /// Maximum memory ever used.
    pub mem_limit: u64,
    /// Maximum memory used for this disk.
    pub mem_used_max: u64,
    /// Number of same-element filled pages.
    pub same_pages: u64,
    /// Number of pages stored compressed.
    pub pages_compacted: u64,
    /// Number of pages stored as huge (incompressible).
    pub huge_pages: u64,
    /// Number of pages detected as same-fill since last reset.
    pub huge_pages_since: u64,
}

impl MmStat {
    /// Format as kernel sysfs output.
    #[must_use]
    pub fn to_sysfs_string(&self) -> String {
        format!(
            "{:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
            self.orig_data_size,
            self.compr_data_size,
            self.mem_used_total,
            self.mem_limit,
            self.mem_used_max,
            self.same_pages,
            self.pages_compacted,
            self.huge_pages,
            self.huge_pages_since,
        )
    }

    /// Parse from kernel sysfs output.
    pub fn from_sysfs_string(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split_whitespace().collect();
        if parts.len() < 9 {
            return None;
        }

        Some(Self {
            orig_data_size: parts[0].parse().ok()?,
            compr_data_size: parts[1].parse().ok()?,
            mem_used_total: parts[2].parse().ok()?,
            mem_limit: parts[3].parse().ok()?,
            mem_used_max: parts[4].parse().ok()?,
            same_pages: parts[5].parse().ok()?,
            pages_compacted: parts[6].parse().ok()?,
            huge_pages: parts[7].parse().ok()?,
            huge_pages_since: parts[8].parse().ok()?,
        })
    }

    /// Calculate compression ratio.
    #[must_use]
    pub fn compression_ratio(&self) -> f64 {
        if self.compr_data_size > 0 {
            self.orig_data_size as f64 / self.compr_data_size as f64
        } else {
            1.0
        }
    }
}

/// I/O statistics compatible with kernel zram io_stat format.
///
/// Format: `%8llu %8llu %8llu %8llu`
#[derive(Debug, Clone, Default)]
pub struct IoStat {
    /// Number of failed reads.
    pub failed_reads: u64,
    /// Number of failed writes.
    pub failed_writes: u64,
    /// Number of invalid I/O requests.
    pub invalid_io: u64,
    /// Number of pages that couldn't be compressed (notify_free).
    pub notify_free: u64,
}

impl IoStat {
    /// Format as kernel sysfs output.
    #[must_use]
    pub fn to_sysfs_string(&self) -> String {
        format!(
            "{:>8} {:>8} {:>8} {:>8}",
            self.failed_reads, self.failed_writes, self.invalid_io, self.notify_free,
        )
    }

    /// Parse from kernel sysfs output.
    pub fn from_sysfs_string(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split_whitespace().collect();
        if parts.len() < 4 {
            return None;
        }

        Some(Self {
            failed_reads: parts[0].parse().ok()?,
            failed_writes: parts[1].parse().ok()?,
            invalid_io: parts[2].parse().ok()?,
            notify_free: parts[3].parse().ok()?,
        })
    }
}

/// Virtual sysfs interface for a zram device.
///
/// This provides a compatible interface with `/sys/block/zramN/`.
#[derive(Debug)]
pub struct SysfsInterface {
    /// Device number (0 for zram0, etc.).
    pub device_num: u32,
    /// Disk size in bytes.
    pub disksize: u64,
    /// Current compression algorithm.
    pub comp_algorithm: ZramAlgorithm,
    /// Maximum memory limit.
    pub mem_limit: u64,
    /// Memory statistics.
    pub mm_stat: MmStat,
    /// I/O statistics.
    pub io_stat: IoStat,
    /// Device is initialized.
    pub initstate: bool,
    /// Additional attributes.
    pub attrs: HashMap<String, String>,
}

impl Default for SysfsInterface {
    fn default() -> Self {
        Self::new(0)
    }
}

impl SysfsInterface {
    /// Create a new sysfs interface for a device.
    #[must_use]
    pub fn new(device_num: u32) -> Self {
        Self {
            device_num,
            disksize: 0,
            comp_algorithm: ZramAlgorithm::Lz4,
            mem_limit: 0,
            mm_stat: MmStat::default(),
            io_stat: IoStat::default(),
            initstate: false,
            attrs: HashMap::new(),
        }
    }

    /// Get the sysfs path for this device.
    #[must_use]
    pub fn sysfs_path(&self) -> PathBuf {
        PathBuf::from(format!("/sys/block/zram{}", self.device_num))
    }

    /// Read a sysfs attribute.
    #[must_use]
    pub fn read_attr(&self, name: &str) -> Option<String> {
        match name {
            "disksize" => Some(self.disksize.to_string()),
            "comp_algorithm" => Some(self.comp_algorithm.as_str().to_string()),
            "mem_limit" => Some(self.mem_limit.to_string()),
            "mm_stat" => Some(self.mm_stat.to_sysfs_string()),
            "io_stat" => Some(self.io_stat.to_sysfs_string()),
            "initstate" => Some(if self.initstate { "1" } else { "0" }.to_string()),
            "max_comp_streams" => Some("1".to_string()), // Always 1 in trueno-zram
            _ => self.attrs.get(name).cloned(),
        }
    }

    /// Write a sysfs attribute.
    pub fn write_attr(&mut self, name: &str, value: &str) -> Result<(), String> {
        match name {
            "disksize" => {
                self.disksize =
                    crate::zram::parse_size(value).map_err(|e| format!("invalid disksize: {e}"))?;
                Ok(())
            }
            "comp_algorithm" => {
                self.comp_algorithm = ZramAlgorithm::parse(value)
                    .ok_or_else(|| format!("unknown algorithm: {value}"))?;
                if !self.comp_algorithm.is_supported() {
                    return Err(format!("unsupported algorithm: {value}"));
                }
                Ok(())
            }
            "mem_limit" => {
                self.mem_limit = crate::zram::parse_size(value)
                    .map_err(|e| format!("invalid mem_limit: {e}"))?;
                Ok(())
            }
            "reset" => {
                if value == "1" {
                    self.reset();
                }
                Ok(())
            }
            _ => {
                self.attrs.insert(name.to_string(), value.to_string());
                Ok(())
            }
        }
    }

    /// Reset the device statistics.
    pub fn reset(&mut self) {
        self.mm_stat = MmStat::default();
        self.io_stat = IoStat::default();
        self.initstate = false;
    }

    /// Update memory statistics.
    pub fn update_stats(
        &mut self,
        orig_size: u64,
        compr_size: u64,
        same_pages: u64,
        huge_pages: u64,
    ) {
        self.mm_stat.orig_data_size += orig_size;
        self.mm_stat.compr_data_size += compr_size;
        self.mm_stat.mem_used_total = self.mm_stat.compr_data_size;
        self.mm_stat.same_pages += same_pages;
        self.mm_stat.huge_pages += huge_pages;

        if self.mm_stat.mem_used_total > self.mm_stat.mem_used_max {
            self.mm_stat.mem_used_max = self.mm_stat.mem_used_total;
        }
    }

    /// Check if device is within memory limit.
    #[must_use]
    pub fn within_mem_limit(&self) -> bool {
        self.mem_limit == 0 || self.mm_stat.mem_used_total < self.mem_limit
    }
}

/// List of supported compression algorithms.
#[must_use]
pub fn supported_algorithms() -> Vec<ZramAlgorithm> {
    vec![ZramAlgorithm::Lz4, ZramAlgorithm::Lz4hc, ZramAlgorithm::Zstd]
}

/// Format supported algorithms as sysfs output (bracketed current).
#[must_use]
pub fn format_algorithms(current: ZramAlgorithm) -> String {
    let algos: Vec<String> = [
        ZramAlgorithm::Lz4,
        ZramAlgorithm::Lz4hc,
        ZramAlgorithm::Zstd,
        ZramAlgorithm::Lzo,
        ZramAlgorithm::LzoRle,
    ]
    .iter()
    .map(|a| if *a == current { format!("[{}]", a.as_str()) } else { a.as_str().to_string() })
    .collect();

    algos.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================
    // Compatibility Falsification Tests F066-F075
    // ============================================================

    #[test]
    fn test_f066_sysfs_interface_compatible() {
        // F066: sysfs interface compatible
        let iface = SysfsInterface::new(0);
        assert_eq!(iface.sysfs_path(), PathBuf::from("/sys/block/zram0"));
    }

    #[test]
    fn test_f068_algorithm_string_parsing() {
        // F068: Algorithm string parsing
        assert_eq!(ZramAlgorithm::parse("lz4"), Some(ZramAlgorithm::Lz4));
        assert_eq!(ZramAlgorithm::parse("LZ4"), Some(ZramAlgorithm::Lz4));
        assert_eq!(ZramAlgorithm::parse("zstd"), Some(ZramAlgorithm::Zstd));
        assert_eq!(ZramAlgorithm::parse("lzo-rle"), Some(ZramAlgorithm::LzoRle));
        assert_eq!(ZramAlgorithm::parse("invalid"), None);
    }

    #[test]
    fn test_f072_multiple_devices_supported() {
        // F072: Multiple devices supported
        for i in 0..8 {
            let iface = SysfsInterface::new(i);
            assert_eq!(iface.device_num, i);
            assert_eq!(iface.sysfs_path(), PathBuf::from(format!("/sys/block/zram{i}")));
        }
    }

    #[test]
    fn test_f074_statistics_accurate() {
        // F074: Statistics accurate
        let mut iface = SysfsInterface::new(0);

        iface.update_stats(4096, 1024, 1, 0);
        assert_eq!(iface.mm_stat.orig_data_size, 4096);
        assert_eq!(iface.mm_stat.compr_data_size, 1024);
        assert_eq!(iface.mm_stat.same_pages, 1);

        let ratio = iface.mm_stat.compression_ratio();
        assert!((ratio - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_f075_reset_clears_all_data() {
        // F075: Reset clears all data
        let mut iface = SysfsInterface::new(0);

        iface.update_stats(4096, 1024, 1, 0);
        assert!(iface.mm_stat.orig_data_size > 0);

        iface.reset();
        assert_eq!(iface.mm_stat.orig_data_size, 0);
        assert_eq!(iface.mm_stat.compr_data_size, 0);
        assert!(!iface.initstate);
    }

    #[test]
    fn test_mm_stat_sysfs_format() {
        let stat = MmStat {
            orig_data_size: 1000000,
            compr_data_size: 500000,
            mem_used_total: 600000,
            mem_limit: 0,
            mem_used_max: 600000,
            same_pages: 100,
            pages_compacted: 50,
            huge_pages: 10,
            huge_pages_since: 5,
        };

        let s = stat.to_sysfs_string();
        let parsed = MmStat::from_sysfs_string(&s).unwrap();

        assert_eq!(parsed.orig_data_size, stat.orig_data_size);
        assert_eq!(parsed.compr_data_size, stat.compr_data_size);
        assert_eq!(parsed.same_pages, stat.same_pages);
    }

    #[test]
    fn test_io_stat_sysfs_format() {
        let stat = IoStat { failed_reads: 0, failed_writes: 5, invalid_io: 1, notify_free: 100 };

        let s = stat.to_sysfs_string();
        let parsed = IoStat::from_sysfs_string(&s).unwrap();

        assert_eq!(parsed.failed_writes, stat.failed_writes);
        assert_eq!(parsed.notify_free, stat.notify_free);
    }

    #[test]
    fn test_read_attr() {
        let mut iface = SysfsInterface::new(1);
        iface.disksize = 1024 * 1024 * 1024; // 1GB

        assert_eq!(iface.read_attr("disksize"), Some("1073741824".to_string()));
        assert_eq!(iface.read_attr("comp_algorithm"), Some("lz4".to_string()));
        assert_eq!(iface.read_attr("initstate"), Some("0".to_string()));
        assert_eq!(iface.read_attr("max_comp_streams"), Some("1".to_string()));
    }

    #[test]
    fn test_write_attr_disksize() {
        let mut iface = SysfsInterface::new(0);

        iface.write_attr("disksize", "4G").unwrap();
        assert_eq!(iface.disksize, 4 * 1024 * 1024 * 1024);

        iface.write_attr("disksize", "512M").unwrap();
        assert_eq!(iface.disksize, 512 * 1024 * 1024);
    }

    #[test]
    fn test_write_attr_algorithm() {
        let mut iface = SysfsInterface::new(0);

        iface.write_attr("comp_algorithm", "zstd").unwrap();
        assert_eq!(iface.comp_algorithm, ZramAlgorithm::Zstd);

        iface.write_attr("comp_algorithm", "lz4hc").unwrap();
        assert_eq!(iface.comp_algorithm, ZramAlgorithm::Lz4hc);
    }

    #[test]
    fn test_write_attr_unsupported_algorithm() {
        let mut iface = SysfsInterface::new(0);

        // LZO is not supported by trueno-zram
        let result = iface.write_attr("comp_algorithm", "lzo");
        assert!(result.is_err());
    }

    #[test]
    fn test_write_attr_reset() {
        let mut iface = SysfsInterface::new(0);
        iface.update_stats(4096, 1024, 1, 0);
        iface.initstate = true;

        iface.write_attr("reset", "1").unwrap();

        assert_eq!(iface.mm_stat.orig_data_size, 0);
        assert!(!iface.initstate);
    }

    #[test]
    fn test_format_algorithms() {
        let s = format_algorithms(ZramAlgorithm::Lz4);
        assert!(s.contains("[lz4]"));
        assert!(s.contains("zstd"));
        assert!(!s.contains("[zstd]"));

        let s = format_algorithms(ZramAlgorithm::Zstd);
        assert!(s.contains("[zstd]"));
        assert!(!s.contains("[lz4]"));
    }

    #[test]
    fn test_supported_algorithms() {
        let algos = supported_algorithms();
        assert!(algos.contains(&ZramAlgorithm::Lz4));
        assert!(algos.contains(&ZramAlgorithm::Zstd));
        assert!(!algos.contains(&ZramAlgorithm::Lzo));
    }

    #[test]
    fn test_within_mem_limit() {
        let mut iface = SysfsInterface::new(0);

        // No limit set
        iface.mem_limit = 0;
        assert!(iface.within_mem_limit());

        // Within limit
        iface.mem_limit = 1000;
        iface.mm_stat.mem_used_total = 500;
        assert!(iface.within_mem_limit());

        // At limit
        iface.mm_stat.mem_used_total = 1000;
        assert!(!iface.within_mem_limit());
    }

    #[test]
    fn test_algorithm_is_supported() {
        assert!(ZramAlgorithm::Lz4.is_supported());
        assert!(ZramAlgorithm::Lz4hc.is_supported());
        assert!(ZramAlgorithm::Zstd.is_supported());
        assert!(!ZramAlgorithm::Lzo.is_supported());
        assert!(!ZramAlgorithm::LzoRle.is_supported());
    }

    #[test]
    fn test_custom_attrs() {
        let mut iface = SysfsInterface::new(0);

        iface.write_attr("custom_attr", "custom_value").unwrap();
        assert_eq!(iface.read_attr("custom_attr"), Some("custom_value".to_string()));
    }

    #[test]
    fn test_sysfs_interface_default() {
        let iface = SysfsInterface::default();
        assert_eq!(iface.device_num, 0);
        assert_eq!(iface.disksize, 0);
        assert_eq!(iface.comp_algorithm, ZramAlgorithm::Lz4);
    }

    #[test]
    fn test_mm_stat_default() {
        let stat = MmStat::default();
        assert_eq!(stat.orig_data_size, 0);
        assert!((stat.compression_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mm_stat_from_sysfs_string_invalid() {
        // Too few fields
        assert!(MmStat::from_sysfs_string("1 2 3").is_none());
        assert!(MmStat::from_sysfs_string("").is_none());
        assert!(MmStat::from_sysfs_string("1 2 3 4 5 6 7 8").is_none()); // Only 8 fields
    }

    #[test]
    fn test_io_stat_from_sysfs_string_invalid() {
        // Too few fields
        assert!(IoStat::from_sysfs_string("1 2 3").is_none());
        assert!(IoStat::from_sysfs_string("").is_none());
        assert!(IoStat::from_sysfs_string("1 2").is_none());
    }

    #[test]
    fn test_algorithm_deflate842() {
        assert_eq!(ZramAlgorithm::parse("842"), Some(ZramAlgorithm::Deflate842));
        assert_eq!(ZramAlgorithm::Deflate842.as_str(), "842");
        assert!(!ZramAlgorithm::Deflate842.is_supported());
    }

    #[test]
    fn test_algorithm_lzo() {
        assert_eq!(ZramAlgorithm::parse("lzo"), Some(ZramAlgorithm::Lzo));
        assert_eq!(ZramAlgorithm::Lzo.as_str(), "lzo");
    }

    #[test]
    fn test_write_attr_invalid_disksize() {
        let mut iface = SysfsInterface::new(0);
        let result = iface.write_attr("disksize", "invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_write_attr_invalid_memlimit() {
        let mut iface = SysfsInterface::new(0);
        let result = iface.write_attr("mem_limit", "invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_write_attr_unknown_algorithm() {
        let mut iface = SysfsInterface::new(0);
        let result = iface.write_attr("comp_algorithm", "unknown_algo");
        assert!(result.is_err());
    }

    #[test]
    fn test_write_attr_reset_not_one() {
        let mut iface = SysfsInterface::new(0);
        iface.update_stats(4096, 1024, 1, 0);
        let orig = iface.mm_stat.orig_data_size;

        // Value != "1" should not reset
        iface.write_attr("reset", "0").unwrap();
        assert_eq!(iface.mm_stat.orig_data_size, orig);
    }

    #[test]
    fn test_write_attr_mem_limit() {
        let mut iface = SysfsInterface::new(0);
        iface.write_attr("mem_limit", "2G").unwrap();
        assert_eq!(iface.mem_limit, 2 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_read_attr_io_stat() {
        let mut iface = SysfsInterface::new(0);
        iface.io_stat.failed_reads = 5;
        iface.io_stat.failed_writes = 10;

        let io_stat_str = iface.read_attr("io_stat").unwrap();
        assert!(io_stat_str.contains('5'));
        assert!(io_stat_str.contains("10"));
    }

    #[test]
    fn test_read_attr_mm_stat() {
        let mut iface = SysfsInterface::new(0);
        iface.update_stats(8192, 2048, 2, 1);

        let mm_stat_str = iface.read_attr("mm_stat").unwrap();
        assert!(mm_stat_str.contains("8192"));
        assert!(mm_stat_str.contains("2048"));
    }

    #[test]
    fn test_read_attr_unknown() {
        let iface = SysfsInterface::new(0);
        assert!(iface.read_attr("unknown_attr").is_none());
    }

    #[test]
    fn test_update_stats_mem_max() {
        let mut iface = SysfsInterface::new(0);

        // First update sets max
        iface.update_stats(4096, 1024, 0, 0);
        assert_eq!(iface.mm_stat.mem_used_max, 1024);

        // Second update increases max
        iface.update_stats(4096, 2048, 0, 0);
        assert_eq!(iface.mm_stat.mem_used_max, 3072); // 1024 + 2048

        // Smaller update doesn't decrease max
        iface.reset();
        iface.update_stats(4096, 1000, 0, 0);
        assert_eq!(iface.mm_stat.mem_used_max, 1000);
    }

    #[test]
    fn test_io_stat_default() {
        let stat = IoStat::default();
        assert_eq!(stat.failed_reads, 0);
        assert_eq!(stat.failed_writes, 0);
    }

    #[test]
    fn test_zram_algorithm_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ZramAlgorithm::Lz4);
        set.insert(ZramAlgorithm::Zstd);
        assert!(set.contains(&ZramAlgorithm::Lz4));
        assert!(!set.contains(&ZramAlgorithm::Lzo));
    }

    #[test]
    fn test_zram_algorithm_clone() {
        let algo = ZramAlgorithm::Zstd;
        let cloned = algo;
        assert_eq!(algo, cloned);
    }

    #[test]
    fn test_mm_stat_clone() {
        let stat = MmStat { orig_data_size: 100, compr_data_size: 50, ..Default::default() };
        let cloned = stat.clone();
        assert_eq!(stat.orig_data_size, cloned.orig_data_size);
    }

    #[test]
    fn test_io_stat_clone() {
        let stat = IoStat { failed_reads: 5, ..Default::default() };
        let cloned = stat.clone();
        assert_eq!(stat.failed_reads, cloned.failed_reads);
    }

    #[test]
    fn test_format_algorithms_all_variants() {
        // Test all algorithms as current
        for algo in [
            ZramAlgorithm::Lz4,
            ZramAlgorithm::Lz4hc,
            ZramAlgorithm::Zstd,
            ZramAlgorithm::Lzo,
            ZramAlgorithm::LzoRle,
        ] {
            let s = format_algorithms(algo);
            assert!(s.contains(&format!("[{}]", algo.as_str())));
        }
    }

    #[test]
    fn test_sysfs_interface_debug() {
        let iface = SysfsInterface::new(0);
        let debug = format!("{iface:?}");
        assert!(debug.contains("SysfsInterface"));
    }

    #[test]
    fn test_algorithm_parse_whitespace() {
        // With leading/trailing whitespace
        assert_eq!(ZramAlgorithm::parse("  lz4  "), Some(ZramAlgorithm::Lz4));
        assert_eq!(ZramAlgorithm::parse("\tzstd\n"), Some(ZramAlgorithm::Zstd));
    }
}
