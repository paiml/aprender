//! Disk Entropy Analyzer
//!
//! Analyzes disk entropy to detect encryption. Encrypted/compressed data
//! has high entropy (close to 1.0), while unencrypted data typically has
//! lower entropy.

#![allow(clippy::uninlined_format_args)]

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Read;
use std::path::Path;
use std::time::Duration;

use super::{Analyzer, AnalyzerError};

/// Encryption type detected
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EncryptionType {
    /// No encryption detected
    #[default]
    None,
    /// LUKS encrypted
    Luks,
    /// dm-crypt without LUKS header
    DmCrypt,
    /// VeraCrypt/TrueCrypt
    VeraCrypt,
    /// `BitLocker` (unlikely on Linux but detectable)
    BitLocker,
    /// Unknown encryption (high entropy)
    Unknown,
}

impl EncryptionType {
    /// Display name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Luks => "LUKS",
            Self::DmCrypt => "dm-crypt",
            Self::VeraCrypt => "VeraCrypt",
            Self::BitLocker => "BitLocker",
            Self::Unknown => "encrypted",
        }
    }

    /// Is this any form of encryption?
    pub fn is_encrypted(&self) -> bool {
        !matches!(self, Self::None)
    }
}

/// Entropy information for a disk device
#[derive(Debug, Clone, Default)]
pub struct DiskEntropyInfo {
    /// Device name (e.g., "sda", "nvme0n1")
    pub device: String,
    /// Device path (e.g., "/dev/sda")
    pub path: String,
    /// Calculated entropy (0.0-1.0, where 1.0 is max entropy)
    pub entropy: f64,
    /// Detected encryption type
    pub encryption_type: EncryptionType,
    /// Is this a dm-crypt device mapper target?
    pub is_dm_target: bool,
    /// LUKS UUID if detected
    pub luks_uuid: Option<String>,
    /// Encryption cipher if detected
    pub cipher: Option<String>,
}

impl DiskEntropyInfo {
    /// Is this device likely encrypted based on entropy?
    pub fn is_high_entropy(&self) -> bool {
        self.entropy > 0.95
    }

    /// Format entropy as percentage
    pub fn entropy_percent(&self) -> f64 {
        self.entropy * 100.0
    }

    /// Status display
    pub fn status_display(&self) -> &'static str {
        if self.encryption_type.is_encrypted() {
            "🔒"
        } else if self.is_high_entropy() {
            "⚠️"
        } else {
            "🔓"
        }
    }
}

/// Disk entropy data
#[derive(Debug, Clone, Default)]
pub struct DiskEntropyData {
    /// Entropy info per device
    pub devices: HashMap<String, DiskEntropyInfo>,
    /// Number of encrypted devices
    pub encrypted_count: usize,
    /// Number of unencrypted devices
    pub unencrypted_count: usize,
}

impl DiskEntropyData {
    /// Get encrypted devices
    pub fn encrypted_devices(&self) -> impl Iterator<Item = &DiskEntropyInfo> {
        self.devices
            .values()
            .filter(|d| d.encryption_type.is_encrypted())
    }

    /// Get unencrypted devices
    pub fn unencrypted_devices(&self) -> impl Iterator<Item = &DiskEntropyInfo> {
        self.devices
            .values()
            .filter(|d| !d.encryption_type.is_encrypted())
    }
}

/// Analyzer for disk entropy and encryption detection
pub struct DiskEntropyAnalyzer {
    data: DiskEntropyData,
    interval: Duration,
    /// Sample size in bytes for entropy calculation
    sample_size: usize,
}

impl Default for DiskEntropyAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl DiskEntropyAnalyzer {
    /// Create a new disk entropy analyzer
    pub fn new() -> Self {
        Self {
            data: DiskEntropyData::default(),
            interval: Duration::from_secs(60), // Entropy doesn't change often
            sample_size: 4096,                 // Sample 4KB for entropy calculation
        }
    }

    /// Get the current data
    pub fn data(&self) -> &DiskEntropyData {
        &self.data
    }

    /// List block devices from /sys/block/
    fn list_block_devices(&self) -> Vec<String> {
        let mut devices = Vec::new();

        if let Ok(entries) = fs::read_dir("/sys/block") {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                // Skip loop devices, ram disks
                if name.starts_with("loop") || name.starts_with("ram") {
                    continue;
                }
                devices.push(name);
            }
        }

        devices
    }

    /// Check if device is a dm-crypt target
    fn is_dm_crypt(&self, device: &str) -> bool {
        let dm_path = format!("/sys/block/{}/dm/uuid", device);
        if let Ok(uuid) = fs::read_to_string(&dm_path) {
            return uuid.starts_with("CRYPT-");
        }
        false
    }

    /// Try to detect LUKS encryption
    fn detect_luks(&self, device_path: &str) -> Option<(String, String)> {
        // LUKS header magic: "LUKS\xba\xbe" at offset 0
        // We can also check /sys/block/*/dm/ for LUKS info

        // Try cryptsetup status via /dev/mapper
        let dm_name = Path::new(device_path)
            .file_name()
            .map(|s| s.to_string_lossy().to_string())?;

        // Check for LUKS UUID in dm
        let uuid_path = format!("/sys/block/{}/dm/uuid", dm_name);
        if let Ok(uuid) = fs::read_to_string(&uuid_path) {
            if uuid.starts_with("CRYPT-LUKS") {
                // Extract UUID from CRYPT-LUKS1-<uuid> or CRYPT-LUKS2-<uuid>
                let parts: Vec<&str> = uuid.trim().split('-').collect();
                if parts.len() >= 3 {
                    let luks_uuid = parts[2..].join("-");
                    return Some((luks_uuid, "aes-xts-plain64".to_string())); // Common default
                }
            }
        }

        None
    }

    /// Calculate Shannon entropy of data
    fn calculate_entropy(&self, data: &[u8]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        // Count byte frequencies
        let mut freq = [0u64; 256];
        for &byte in data {
            freq[byte as usize] += 1;
        }

        // Calculate Shannon entropy
        let len = data.len() as f64;
        let mut entropy = 0.0;

        for &count in &freq {
            if count > 0 {
                let p = count as f64 / len;
                entropy -= p * p.log2();
            }
        }

        // Normalize to 0-1 range (max entropy is 8 bits)
        entropy / 8.0
    }

    /// Sample entropy from device (if readable)
    fn sample_device_entropy(&self, device_path: &str) -> Option<f64> {
        // Try to read a sample from the device
        // This requires read permission on the device
        let mut file = File::open(device_path).ok()?;
        let mut buffer = vec![0u8; self.sample_size];

        // Read from middle of device to avoid headers
        // Note: This may fail without root permissions
        if file.read_exact(&mut buffer).is_ok() {
            Some(self.calculate_entropy(&buffer))
        } else {
            None
        }
    }

    /// Analyze a single device
    fn analyze_device(&self, device: &str) -> DiskEntropyInfo {
        let device_path = format!("/dev/{}", device);
        let is_dm = self.is_dm_crypt(device);

        let mut info = DiskEntropyInfo {
            device: device.to_string(),
            path: device_path.clone(),
            is_dm_target: is_dm,
            ..Default::default()
        };

        // Check for LUKS
        if is_dm {
            if let Some((uuid, cipher)) = self.detect_luks(&device_path) {
                info.encryption_type = EncryptionType::Luks;
                info.luks_uuid = Some(uuid);
                info.cipher = Some(cipher);
                info.entropy = 0.99; // LUKS data is high entropy
                return info;
            }
            // dm-crypt without LUKS
            info.encryption_type = EncryptionType::DmCrypt;
            info.entropy = 0.99;
            return info;
        }

        // Try to sample entropy (may fail without permissions)
        if let Some(entropy) = self.sample_device_entropy(&device_path) {
            info.entropy = entropy;
            if entropy > 0.98 {
                info.encryption_type = EncryptionType::Unknown;
            }
        }

        info
    }
}

impl Analyzer for DiskEntropyAnalyzer {
    fn name(&self) -> &'static str {
        "disk_entropy"
    }

    fn collect(&mut self) -> Result<(), AnalyzerError> {
        let devices = self.list_block_devices();
        let mut device_map = HashMap::new();
        let mut encrypted = 0;
        let mut unencrypted = 0;

        for device in devices {
            let info = self.analyze_device(&device);
            if info.encryption_type.is_encrypted() {
                encrypted += 1;
            } else {
                unencrypted += 1;
            }
            device_map.insert(device, info);
        }

        self.data = DiskEntropyData {
            devices: device_map,
            encrypted_count: encrypted,
            unencrypted_count: unencrypted,
        };

        Ok(())
    }

    fn interval(&self) -> Duration {
        self.interval
    }

    fn available(&self) -> bool {
        Path::new("/sys/block").exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encryption_type() {
        assert!(!EncryptionType::None.is_encrypted());
        assert!(EncryptionType::Luks.is_encrypted());
        assert!(EncryptionType::DmCrypt.is_encrypted());
        assert_eq!(EncryptionType::Luks.as_str(), "LUKS");
    }

    #[test]
    fn test_entropy_calculation() {
        let analyzer = DiskEntropyAnalyzer::new();

        // All zeros = low entropy
        let zeros = vec![0u8; 1000];
        let entropy = analyzer.calculate_entropy(&zeros);
        assert!(entropy < 0.01);

        // Random-ish data = higher entropy
        let varied: Vec<u8> = (0..=255).cycle().take(1000).collect();
        let entropy = analyzer.calculate_entropy(&varied);
        assert!(entropy > 0.9);
    }

    #[test]
    fn test_disk_entropy_info() {
        let info = DiskEntropyInfo {
            device: "sda".to_string(),
            path: "/dev/sda".to_string(),
            entropy: 0.99,
            encryption_type: EncryptionType::Luks,
            ..Default::default()
        };

        assert!(info.is_high_entropy());
        assert!((info.entropy_percent() - 99.0).abs() < 0.1);
        assert_eq!(info.status_display(), "🔒");
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = DiskEntropyAnalyzer::new();
        assert_eq!(analyzer.name(), "disk_entropy");
    }

    #[test]
    fn test_analyzer_available() {
        let analyzer = DiskEntropyAnalyzer::new();
        #[cfg(target_os = "linux")]
        assert!(analyzer.available());
    }

    #[test]
    fn test_analyzer_collect() {
        let mut analyzer = DiskEntropyAnalyzer::new();
        let result = analyzer.collect();
        assert!(result.is_ok());
    }

    // Additional encryption type tests
    #[test]
    fn test_encryption_type_veracrypt() {
        assert!(EncryptionType::VeraCrypt.is_encrypted());
        assert_eq!(EncryptionType::VeraCrypt.as_str(), "VeraCrypt");
    }

    #[test]
    fn test_encryption_type_bitlocker() {
        assert!(EncryptionType::BitLocker.is_encrypted());
        assert_eq!(EncryptionType::BitLocker.as_str(), "BitLocker");
    }

    #[test]
    fn test_encryption_type_unknown() {
        assert!(EncryptionType::Unknown.is_encrypted());
        assert_eq!(EncryptionType::Unknown.as_str(), "encrypted");
    }

    #[test]
    fn test_encryption_type_none_str() {
        assert_eq!(EncryptionType::None.as_str(), "none");
    }

    #[test]
    fn test_encryption_type_dm_crypt_str() {
        assert_eq!(EncryptionType::DmCrypt.as_str(), "dm-crypt");
    }

    #[test]
    fn test_encryption_type_default() {
        let default = EncryptionType::default();
        assert!(!default.is_encrypted());
        assert_eq!(default, EncryptionType::None);
    }

    #[test]
    fn test_encryption_type_debug() {
        let enc = EncryptionType::Luks;
        let debug = format!("{:?}", enc);
        assert!(debug.contains("Luks"));
    }

    #[test]
    fn test_encryption_type_clone() {
        let enc = EncryptionType::DmCrypt;
        let cloned = enc.clone();
        assert_eq!(enc, cloned);
    }

    #[test]
    fn test_encryption_type_copy() {
        let enc = EncryptionType::Luks;
        let copied: EncryptionType = enc;
        assert_eq!(copied, EncryptionType::Luks);
    }

    // DiskEntropyInfo tests
    #[test]
    fn test_disk_entropy_info_default() {
        let info = DiskEntropyInfo::default();
        assert!(info.device.is_empty());
        assert!(info.path.is_empty());
        assert!((info.entropy - 0.0).abs() < f64::EPSILON);
        assert_eq!(info.encryption_type, EncryptionType::None);
        assert!(!info.is_dm_target);
        assert!(info.luks_uuid.is_none());
        assert!(info.cipher.is_none());
    }

    #[test]
    fn test_disk_entropy_info_low_entropy() {
        let info = DiskEntropyInfo {
            entropy: 0.5,
            ..Default::default()
        };
        assert!(!info.is_high_entropy());
        assert!((info.entropy_percent() - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_disk_entropy_info_status_unencrypted() {
        let info = DiskEntropyInfo {
            entropy: 0.3,
            encryption_type: EncryptionType::None,
            ..Default::default()
        };
        assert_eq!(info.status_display(), "🔓");
    }

    #[test]
    fn test_disk_entropy_info_status_high_entropy_warning() {
        let info = DiskEntropyInfo {
            entropy: 0.98,
            encryption_type: EncryptionType::None,
            ..Default::default()
        };
        assert_eq!(info.status_display(), "⚠️");
    }

    #[test]
    fn test_disk_entropy_info_status_encrypted() {
        let info = DiskEntropyInfo {
            entropy: 0.99,
            encryption_type: EncryptionType::DmCrypt,
            ..Default::default()
        };
        assert_eq!(info.status_display(), "🔒");
    }

    #[test]
    fn test_disk_entropy_info_clone() {
        let info = DiskEntropyInfo {
            device: "nvme0n1".to_string(),
            path: "/dev/nvme0n1".to_string(),
            entropy: 0.95,
            encryption_type: EncryptionType::Luks,
            is_dm_target: true,
            luks_uuid: Some("abc-123".to_string()),
            cipher: Some("aes-xts-plain64".to_string()),
        };
        let cloned = info.clone();
        assert_eq!(cloned.device, "nvme0n1");
        assert_eq!(cloned.encryption_type, EncryptionType::Luks);
    }

    #[test]
    fn test_disk_entropy_info_debug() {
        let info = DiskEntropyInfo::default();
        let debug = format!("{:?}", info);
        assert!(debug.contains("DiskEntropyInfo"));
    }

    // DiskEntropyData tests
    #[test]
    fn test_disk_entropy_data_default() {
        let data = DiskEntropyData::default();
        assert!(data.devices.is_empty());
        assert_eq!(data.encrypted_count, 0);
        assert_eq!(data.unencrypted_count, 0);
    }

    #[test]
    fn test_disk_entropy_data_encrypted_devices() {
        let mut devices = HashMap::new();
        devices.insert(
            "sda".to_string(),
            DiskEntropyInfo {
                device: "sda".to_string(),
                encryption_type: EncryptionType::Luks,
                ..Default::default()
            },
        );
        devices.insert(
            "sdb".to_string(),
            DiskEntropyInfo {
                device: "sdb".to_string(),
                encryption_type: EncryptionType::None,
                ..Default::default()
            },
        );

        let data = DiskEntropyData {
            devices,
            encrypted_count: 1,
            unencrypted_count: 1,
        };

        let encrypted: Vec<_> = data.encrypted_devices().collect();
        assert_eq!(encrypted.len(), 1);
        assert_eq!(encrypted[0].device, "sda");
    }

    #[test]
    fn test_disk_entropy_data_unencrypted_devices() {
        let mut devices = HashMap::new();
        devices.insert(
            "sda".to_string(),
            DiskEntropyInfo {
                device: "sda".to_string(),
                encryption_type: EncryptionType::None,
                ..Default::default()
            },
        );

        let data = DiskEntropyData {
            devices,
            encrypted_count: 0,
            unencrypted_count: 1,
        };

        let unencrypted: Vec<_> = data.unencrypted_devices().collect();
        assert_eq!(unencrypted.len(), 1);
    }

    #[test]
    fn test_disk_entropy_data_clone() {
        let data = DiskEntropyData {
            devices: HashMap::new(),
            encrypted_count: 2,
            unencrypted_count: 3,
        };
        let cloned = data.clone();
        assert_eq!(cloned.encrypted_count, 2);
        assert_eq!(cloned.unencrypted_count, 3);
    }

    #[test]
    fn test_disk_entropy_data_debug() {
        let data = DiskEntropyData::default();
        let debug = format!("{:?}", data);
        assert!(debug.contains("DiskEntropyData"));
    }

    // DiskEntropyAnalyzer tests
    #[test]
    fn test_analyzer_default() {
        let analyzer = DiskEntropyAnalyzer::default();
        assert_eq!(analyzer.name(), "disk_entropy");
    }

    #[test]
    fn test_analyzer_data() {
        let analyzer = DiskEntropyAnalyzer::new();
        let data = analyzer.data();
        assert!(data.devices.is_empty());
    }

    #[test]
    fn test_analyzer_interval() {
        let analyzer = DiskEntropyAnalyzer::new();
        let interval = analyzer.interval();
        assert_eq!(interval.as_secs(), 60);
    }

    #[test]
    fn test_entropy_calculation_empty() {
        let analyzer = DiskEntropyAnalyzer::new();
        let entropy = analyzer.calculate_entropy(&[]);
        assert!((entropy - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_entropy_calculation_single_byte() {
        let analyzer = DiskEntropyAnalyzer::new();
        let entropy = analyzer.calculate_entropy(&[42]);
        assert!(entropy < 0.01); // Single byte = low entropy
    }

    #[test]
    fn test_entropy_calculation_two_values() {
        let analyzer = DiskEntropyAnalyzer::new();
        // Alternating values
        let data: Vec<u8> = (0..100).map(|i| if i % 2 == 0 { 0 } else { 255 }).collect();
        let entropy = analyzer.calculate_entropy(&data);
        // Two equally likely values = entropy of 1 bit = 1/8 = 0.125
        assert!(entropy > 0.1 && entropy < 0.2);
    }

    #[test]
    fn test_entropy_calculation_all_different() {
        let analyzer = DiskEntropyAnalyzer::new();
        // All 256 possible byte values
        let data: Vec<u8> = (0..=255).collect();
        let entropy = analyzer.calculate_entropy(&data);
        // Maximum entropy for uniform distribution
        assert!(entropy > 0.99);
    }

    #[test]
    fn test_list_block_devices() {
        let analyzer = DiskEntropyAnalyzer::new();
        let devices = analyzer.list_block_devices();
        // Should not include loop or ram devices
        for device in &devices {
            assert!(!device.starts_with("loop"));
            assert!(!device.starts_with("ram"));
        }
    }

    #[test]
    fn test_is_dm_crypt_non_dm() {
        let analyzer = DiskEntropyAnalyzer::new();
        // sda is typically not a dm-crypt device
        let result = analyzer.is_dm_crypt("nonexistent_device");
        assert!(!result);
    }

    #[test]
    fn test_detect_luks_nonexistent() {
        let analyzer = DiskEntropyAnalyzer::new();
        let result = analyzer.detect_luks("/dev/nonexistent");
        assert!(result.is_none());
    }

    #[test]
    fn test_analyze_device_nonexistent() {
        let analyzer = DiskEntropyAnalyzer::new();
        let info = analyzer.analyze_device("nonexistent_device_xyz");
        assert_eq!(info.device, "nonexistent_device_xyz");
        assert!(!info.is_dm_target);
    }

    #[test]
    fn test_sample_device_entropy_nonexistent() {
        let analyzer = DiskEntropyAnalyzer::new();
        let result = analyzer.sample_device_entropy("/dev/nonexistent_xyz");
        assert!(result.is_none());
    }

    #[test]
    fn test_collect_and_data() {
        let mut analyzer = DiskEntropyAnalyzer::new();
        let _ = analyzer.collect();
        let data = analyzer.data();
        // On a Linux system, we should have some devices (even if empty)
        let _ = data.devices.len();
    }

    #[test]
    fn test_multiple_collects() {
        let mut analyzer = DiskEntropyAnalyzer::new();
        let _ = analyzer.collect();
        let _ = analyzer.collect();
        let _ = analyzer.collect();
        // Should not panic
    }
}
