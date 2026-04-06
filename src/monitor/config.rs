//! Configuration system for the TUI monitor.
//!
//! Supports YAML configuration with precedence: CLI > ENV > file > defaults.

use crate::monitor::error::{MonitorError, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;

/// Global configuration settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalConfig {
    /// Refresh interval in milliseconds.
    #[serde(default = "default_update_ms")]
    pub update_ms: u64,

    /// Number of data points to retain in history.
    #[serde(default = "default_history_size")]
    pub history_size: usize,

    /// Temperature scale (celsius, fahrenheit, kelvin).
    #[serde(default = "default_temp_scale")]
    pub temp_scale: String,

    /// Enable vim-style navigation keys (hjkl).
    #[serde(default = "default_vim_keys")]
    pub vim_keys: bool,

    /// Enable mouse support.
    #[serde(default = "default_mouse")]
    pub mouse: bool,
}

fn default_update_ms() -> u64 {
    1000
}
fn default_history_size() -> usize {
    300
}
fn default_temp_scale() -> String {
    "celsius".to_string()
}
fn default_vim_keys() -> bool {
    true
}
fn default_mouse() -> bool {
    true
}

impl Default for GlobalConfig {
    fn default() -> Self {
        Self {
            update_ms: default_update_ms(),
            history_size: default_history_size(),
            temp_scale: default_temp_scale(),
            vim_keys: default_vim_keys(),
            mouse: default_mouse(),
        }
    }
}

/// Main configuration structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Configuration version.
    #[serde(default = "default_version")]
    pub version: u32,

    /// Global settings.
    #[serde(default)]
    pub global: GlobalConfig,

    /// Theme name or inline theme.
    #[serde(default = "default_theme")]
    pub theme: String,
}

fn default_version() -> u32 {
    1
}
fn default_theme() -> String {
    "default".to_string()
}

impl Default for Config {
    fn default() -> Self {
        Self { version: default_version(), global: GlobalConfig::default(), theme: default_theme() }
    }
}

impl Config {
    /// Creates a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Loads configuration from a YAML file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();

        let content = std::fs::read_to_string(path)
            .map_err(|_| MonitorError::ConfigNotFound(path.display().to_string()))?;

        Self::parse(&content)
    }

    /// Parses configuration from a YAML string.
    ///
    /// # Errors
    ///
    /// Returns an error with line number if parsing fails.
    pub fn parse(yaml: &str) -> Result<Self> {
        serde_yaml_ng::from_str(yaml).map_err(|e| {
            let line = e.location().map_or(0, |l| l.line());
            MonitorError::ConfigParse { line, message: e.to_string() }
        })
    }

    /// Returns the update interval as a Duration.
    #[must_use]
    pub fn update_interval(&self) -> Duration {
        Duration::from_millis(self.global.update_ms)
    }

    /// Loads configuration with fallback to defaults.
    #[must_use]
    pub fn load_or_default(path: impl AsRef<Path>) -> Self {
        Self::load(path).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = Config::new();

        assert_eq!(config.version, 1);
        assert_eq!(config.global.update_ms, 1000);
        assert_eq!(config.global.history_size, 300);
        assert!(config.global.vim_keys);
    }

    #[test]
    fn test_config_parse_minimal() {
        let yaml = "version: 1";
        let config = Config::parse(yaml).expect("parsing should succeed");

        assert_eq!(config.version, 1);
    }

    #[test]
    fn test_config_parse_full() {
        let yaml = r"
version: 1
global:
  update_ms: 500
  history_size: 100
  vim_keys: false
theme: dracula
";

        let config = Config::parse(yaml).expect("parsing should succeed");

        assert_eq!(config.global.update_ms, 500);
        assert_eq!(config.global.history_size, 100);
        assert!(!config.global.vim_keys);
        assert_eq!(config.theme, "dracula");
    }

    #[test]
    fn test_config_parse_error_includes_line() {
        let yaml = r"
version: 1
global:
  update_ms: not_a_number
";

        let result = Config::parse(yaml);
        assert!(result.is_err());

        let err = result.unwrap_err();
        let display = err.to_string();
        assert!(display.contains('4'), "Error should include line number");
    }

    #[test]
    fn test_config_update_interval() {
        let mut config = Config::new();
        config.global.update_ms = 500;

        assert_eq!(config.update_interval(), Duration::from_millis(500));
    }

    #[test]
    fn test_config_load_or_default() {
        let config = Config::load_or_default("/nonexistent/path");
        assert_eq!(config.version, 1);
    }
}
