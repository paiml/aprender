//! Configuration parsing for zram devices.

use serde::Deserialize;
use std::path::Path;

/// Configuration for a zram device.
#[derive(Debug, Clone, Deserialize)]
pub struct ZramConfig {
    /// Device index (0 = /dev/zram0).
    #[serde(default)]
    pub device: u32,

    /// Size in bytes (or with suffix: K, M, G).
    pub size: String,

    /// Compression algorithm.
    #[serde(default = "default_algorithm")]
    pub algorithm: String,

    /// Number of compression streams.
    #[serde(default = "default_streams")]
    pub streams: u32,

    /// Swap priority.
    #[serde(default = "default_priority")]
    pub priority: i32,
}

fn default_algorithm() -> String {
    "lz4".to_string()
}

fn default_streams() -> u32 {
    0 // 0 means auto-detect based on CPU count
}

fn default_priority() -> i32 {
    100
}

/// Root configuration structure.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// List of zram devices to configure.
    #[serde(default)]
    pub devices: Vec<ZramConfig>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            devices: vec![ZramConfig {
                device: 0,
                size: "ram/2".to_string(),
                algorithm: default_algorithm(),
                streams: default_streams(),
                priority: default_priority(),
            }],
        }
    }
}

/// Load configuration from standard locations.
pub fn load_config() -> Result<Config, Box<dyn std::error::Error>> {
    let config_paths = [
        "/etc/trueno-zram.conf",
        "/etc/trueno-zram.conf.d/",
        "/usr/lib/trueno-zram.conf.d/",
    ];

    for path_str in &config_paths {
        let path = Path::new(path_str);
        if path.exists() && path.is_file() {
            let content = std::fs::read_to_string(path)?;
            return Ok(toml::from_str(&content)?);
        }
        // Handle .d directories later
    }

    // Return default configuration if no config found
    Ok(Config::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.devices.len(), 1);
        assert_eq!(config.devices[0].algorithm, "lz4");
        assert_eq!(config.devices[0].streams, 0); // Auto-detect
        assert_eq!(config.devices[0].priority, 100);
        assert_eq!(config.devices[0].size, "ram/2");
    }

    #[test]
    fn test_parse_config() {
        let toml = r#"
            [[devices]]
            device = 0
            size = "4G"
            algorithm = "zstd"
            streams = 4
            priority = 100
        "#;

        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.devices.len(), 1);
        assert_eq!(config.devices[0].size, "4G");
        assert_eq!(config.devices[0].algorithm, "zstd");
    }

    #[test]
    fn test_parse_config_multiple_devices() {
        let toml = r#"
            [[devices]]
            device = 0
            size = "2G"

            [[devices]]
            device = 1
            size = "2G"
            algorithm = "zstd"
        "#;

        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.devices.len(), 2);
        assert_eq!(config.devices[0].device, 0);
        assert_eq!(config.devices[1].device, 1);
        assert_eq!(config.devices[0].algorithm, "lz4"); // Default
        assert_eq!(config.devices[1].algorithm, "zstd");
    }

    #[test]
    fn test_parse_config_defaults() {
        let toml = r#"
            [[devices]]
            size = "1G"
        "#;

        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.devices.len(), 1);
        assert_eq!(config.devices[0].device, 0); // Default
        assert_eq!(config.devices[0].algorithm, "lz4"); // Default
        assert_eq!(config.devices[0].streams, 0); // Default (auto)
        assert_eq!(config.devices[0].priority, 100); // Default
    }

    #[test]
    fn test_parse_empty_devices() {
        let toml = r"
            devices = []
        ";

        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.devices.len(), 0);
    }

    // ...

    #[test]
    fn test_zram_config_debug() {
        let config = ZramConfig {
            device: 0,
            size: "1G".to_string(),
            algorithm: "lz4".to_string(),
            streams: 2,
            priority: 100,
        };
        let debug_str = format!("{config:?}");
        assert!(debug_str.contains("ZramConfig"));
        assert!(debug_str.contains("1G"));
    }

    #[test]
    fn test_config_debug() {
        let config = Config::default();
        let debug_str = format!("{config:?}");
        assert!(debug_str.contains("Config"));
        assert!(debug_str.contains("devices"));
    }
}
