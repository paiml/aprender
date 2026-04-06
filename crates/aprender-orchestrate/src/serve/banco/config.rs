//! Banco configuration persistence.
//!
//! Loads from `~/.banco/config.toml` at startup. Falls back to defaults
//! if the file doesn't exist or is malformed (warn, don't fail).

use crate::serve::backends::PrivacyTier;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Top-level Banco configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BancoConfig {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub inference: InferenceConfig,
    #[serde(default)]
    pub budget: BudgetConfig,
}

/// Server binding and privacy settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default)]
    pub privacy_tier: PrivacyTierConfig,
}

/// Inference parameter defaults.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
}

/// Cost circuit breaker budget settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetConfig {
    #[serde(default = "default_daily_limit")]
    pub daily_limit_usd: f64,
    #[serde(default = "default_max_request")]
    pub max_request_usd: f64,
}

/// Privacy tier as a string for TOML ergonomics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PrivacyTierConfig {
    Sovereign,
    Private,
    #[default]
    Standard,
}

impl From<PrivacyTierConfig> for PrivacyTier {
    fn from(c: PrivacyTierConfig) -> Self {
        match c {
            PrivacyTierConfig::Sovereign => Self::Sovereign,
            PrivacyTierConfig::Private => Self::Private,
            PrivacyTierConfig::Standard => Self::Standard,
        }
    }
}

fn default_host() -> String {
    "127.0.0.1".to_string()
}
fn default_port() -> u16 {
    8090
}
fn default_temperature() -> f32 {
    0.7
}
fn default_top_p() -> f32 {
    1.0
}
fn default_max_tokens() -> u32 {
    256
}
fn default_daily_limit() -> f64 {
    10.0
}
fn default_max_request() -> f64 {
    1.0
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            privacy_tier: PrivacyTierConfig::default(),
        }
    }
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            temperature: default_temperature(),
            top_p: default_top_p(),
            max_tokens: default_max_tokens(),
        }
    }
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self { daily_limit_usd: default_daily_limit(), max_request_usd: default_max_request() }
    }
}

impl BancoConfig {
    /// Default config directory: `~/.banco/`
    #[must_use]
    pub fn config_dir() -> Option<PathBuf> {
        dirs::home_dir().map(|h| h.join(".banco"))
    }

    /// Default config file: `~/.banco/config.toml`
    #[must_use]
    pub fn config_path() -> Option<PathBuf> {
        Self::config_dir().map(|d| d.join("config.toml"))
    }

    /// Load from `~/.banco/config.toml`. Returns defaults if file missing or malformed.
    #[must_use]
    pub fn load() -> Self {
        let Some(path) = Self::config_path() else {
            return Self::default();
        };
        match std::fs::read_to_string(&path) {
            Ok(content) => match toml::from_str(&content) {
                Ok(config) => config,
                Err(e) => {
                    eprintln!("[banco] Warning: failed to parse {}: {e}", path.display());
                    Self::default()
                }
            },
            Err(_) => Self::default(),
        }
    }

    /// Save current config to `~/.banco/config.toml`.
    pub fn save(&self) -> anyhow::Result<()> {
        let Some(dir) = Self::config_dir() else {
            anyhow::bail!("Cannot determine home directory");
        };
        std::fs::create_dir_all(&dir)?;
        let content = toml::to_string_pretty(self)?;
        let path = dir.join("config.toml");
        std::fs::write(&path, content)?;
        Ok(())
    }
}
