//! Profile configuration types and serialization.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{ProfileError, ProfileResult};

/// Serializable profile configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProfileConfig {
    /// Profile name
    pub name: String,
    /// Profile description
    #[serde(default)]
    pub description: String,
    /// Profile version
    #[serde(default = "default_version")]
    pub version: String,
    /// Refresh rate in milliseconds
    #[serde(default = "default_refresh_ms")]
    pub refresh_ms: u64,
    /// GPU device index
    #[serde(default)]
    pub device_index: u32,
    /// Compute backend
    #[serde(default)]
    pub backend: BackendConfig,
    /// Load intensity (0.0 - 1.0)
    #[serde(default = "default_load_intensity")]
    pub load_intensity: f64,
    /// Workload type
    #[serde(default)]
    pub workload: WorkloadConfig,
    /// Problem size in elements
    #[serde(default = "default_problem_size")]
    pub problem_size: usize,
    /// Thread count for SIMD
    #[serde(default = "default_threads")]
    pub threads: usize,
    /// Enable deterministic mode
    #[serde(default)]
    pub deterministic: bool,
    /// Custom metadata
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

pub(super) fn default_version() -> String {
    "1.0".to_string()
}

pub(super) fn default_refresh_ms() -> u64 {
    100
}

pub(super) fn default_load_intensity() -> f64 {
    0.0
}

pub(super) fn default_problem_size() -> usize {
    1_048_576
}

pub(super) fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Backend configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum BackendConfig {
    Simd,
    Wgpu,
    Cuda,
    #[default]
    All,
}

/// Workload configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum WorkloadConfig {
    #[default]
    Gemm,
    Conv2d,
    Attention,
    Bandwidth,
    Elementwise,
    Reduction,
    All,
}

impl Default for ProfileConfig {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            description: String::new(),
            version: default_version(),
            refresh_ms: default_refresh_ms(),
            device_index: 0,
            backend: BackendConfig::default(),
            load_intensity: default_load_intensity(),
            workload: WorkloadConfig::default(),
            problem_size: default_problem_size(),
            threads: default_threads(),
            deterministic: false,
            metadata: HashMap::new(),
        }
    }
}

impl ProfileConfig {
    /// Create a new profile with the given name
    pub fn new(name: &str) -> ProfileResult<Self> {
        validate_profile_name(name)?;
        let mut config = Self::default();
        config.name = name.to_string();
        Ok(config)
    }

    /// Create with name and description
    pub fn with_description(name: &str, description: &str) -> ProfileResult<Self> {
        let mut config = Self::new(name)?;
        config.description = description.to_string();
        Ok(config)
    }

    /// Set backend
    pub fn backend(mut self, backend: BackendConfig) -> Self {
        self.backend = backend;
        self
    }

    /// Set workload
    pub fn workload(mut self, workload: WorkloadConfig) -> Self {
        self.workload = workload;
        self
    }

    /// Set problem size
    pub fn problem_size(mut self, size: usize) -> Self {
        self.problem_size = size;
        self
    }

    /// Set load intensity
    pub fn load_intensity(mut self, intensity: f64) -> Self {
        self.load_intensity = intensity.clamp(0.0, 1.0);
        self
    }

    /// Set threads
    pub fn threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Serialize to TOML string
    pub fn to_toml(&self) -> ProfileResult<String> {
        toml::to_string_pretty(self).map_err(|e| ProfileError::ParseError(e.to_string()))
    }

    /// Parse from TOML string
    pub fn from_toml(toml_str: &str) -> ProfileResult<Self> {
        toml::from_str(toml_str).map_err(|e| ProfileError::ParseError(e.to_string()))
    }
}

/// Validate profile name
pub(super) fn validate_profile_name(name: &str) -> ProfileResult<()> {
    if name.is_empty() {
        return Err(ProfileError::InvalidName(
            "name cannot be empty".to_string(),
        ));
    }

    if name.len() > 64 {
        return Err(ProfileError::InvalidName(
            "name cannot exceed 64 characters".to_string(),
        ));
    }

    // Only allow alphanumeric, underscore, hyphen
    if !name
        .chars()
        .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
    {
        return Err(ProfileError::InvalidName(
            "name can only contain alphanumeric, underscore, or hyphen".to_string(),
        ));
    }

    // Cannot start with hyphen or number
    if let Some(first) = name.chars().next() {
        if first == '-' || first.is_numeric() {
            return Err(ProfileError::InvalidName(
                "name cannot start with hyphen or number".to_string(),
            ));
        }
    }

    Ok(())
}
