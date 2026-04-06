//! Profile Persistence and Rotation (PMAT-028)
//!
//! Configuration profile management with save/load/switch/export capabilities
//! for different workload scenarios.
//!
//! # Features
//!
//! - Named profiles for different workloads (ml_training, inference, stress_test)
//! - Profile save/load/list/export operations
//! - CLI overlay merging (CLI > profile > default)
//! - TOML-based serialization
//!
//! # Falsification Criteria (F1201-F1210)
//!
//! See `tests/profile_persistence_f1201.rs` for falsification tests.

mod config;
mod overlay;

pub use config::{BackendConfig, ProfileConfig, WorkloadConfig};
pub use overlay::{templates, ProfileOverlay};

use config::validate_profile_name;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Profile persistence error
#[derive(Debug, Clone, PartialEq)]
pub enum ProfileError {
    /// Profile not found
    NotFound(String),
    /// Invalid profile name (contains invalid characters)
    InvalidName(String),
    /// IO error during read/write
    IoError(String),
    /// TOML parse error
    ParseError(String),
    /// Profile directory creation failed
    DirectoryError(String),
}

impl std::fmt::Display for ProfileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(name) => write!(f, "Profile not found: {}", name),
            Self::InvalidName(msg) => write!(f, "Invalid profile name: {}", msg),
            Self::IoError(msg) => write!(f, "IO error: {}", msg),
            Self::ParseError(msg) => write!(f, "Parse error: {}", msg),
            Self::DirectoryError(msg) => write!(f, "Directory error: {}", msg),
        }
    }
}

impl std::error::Error for ProfileError {}

/// Result type for profile operations
pub type ProfileResult<T> = Result<T, ProfileError>;

/// Profile manager for save/load/list operations
#[derive(Debug, Clone)]
pub struct ProfileManager {
    /// Profile directory
    profile_dir: PathBuf,
    /// Cached profiles
    cache: HashMap<String, ProfileConfig>,
    /// Default profile name
    default_profile: Option<String>,
}

impl ProfileManager {
    /// Create a new profile manager with the given directory
    pub fn new(profile_dir: PathBuf) -> Self {
        Self {
            profile_dir,
            cache: HashMap::new(),
            default_profile: None,
        }
    }

    /// Create with default profile directory (~/.config/cbtop/profiles)
    pub fn with_default_dir() -> Self {
        let profile_dir = dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("cbtop")
            .join("profiles");
        Self::new(profile_dir)
    }

    /// Get profile directory
    pub fn profile_dir(&self) -> &Path {
        &self.profile_dir
    }

    /// Set default profile name
    pub fn set_default(&mut self, name: &str) {
        self.default_profile = Some(name.to_string());
    }

    /// Get default profile name
    pub fn default_profile(&self) -> Option<&str> {
        self.default_profile.as_deref()
    }

    /// Ensure profile directory exists
    pub fn ensure_directory(&self) -> ProfileResult<()> {
        if !self.profile_dir.exists() {
            std::fs::create_dir_all(&self.profile_dir)
                .map_err(|e| ProfileError::DirectoryError(e.to_string()))?;
        }
        Ok(())
    }

    /// Save a profile to disk
    pub fn save_profile(&mut self, profile: &ProfileConfig) -> ProfileResult<PathBuf> {
        self.ensure_directory()?;

        let filename = format!("{}.toml", profile.name);
        let path = self.profile_dir.join(&filename);

        let toml_content = profile.to_toml()?;
        std::fs::write(&path, toml_content).map_err(|e| ProfileError::IoError(e.to_string()))?;

        // Update cache
        self.cache.insert(profile.name.clone(), profile.clone());

        Ok(path)
    }

    /// Load a profile by name
    pub fn load_profile(&mut self, name: &str) -> ProfileResult<ProfileConfig> {
        validate_profile_name(name)?;

        // Check cache first
        if let Some(profile) = self.cache.get(name) {
            return Ok(profile.clone());
        }

        let filename = format!("{}.toml", name);
        let path = self.profile_dir.join(&filename);

        if !path.exists() {
            return Err(ProfileError::NotFound(name.to_string()));
        }

        let content =
            std::fs::read_to_string(&path).map_err(|e| ProfileError::IoError(e.to_string()))?;

        let profile = ProfileConfig::from_toml(&content)?;

        // Update cache
        self.cache.insert(name.to_string(), profile.clone());

        Ok(profile)
    }

    /// Load default profile or return default config
    pub fn load_default(&mut self) -> ProfileConfig {
        if let Some(name) = &self.default_profile.clone() {
            if let Ok(profile) = self.load_profile(name) {
                return profile;
            }
        }
        ProfileConfig::default()
    }

    /// List all available profiles
    pub fn list_profiles(&self) -> ProfileResult<Vec<String>> {
        if !self.profile_dir.exists() {
            return Ok(vec![]);
        }

        let entries = std::fs::read_dir(&self.profile_dir)
            .map_err(|e| ProfileError::IoError(e.to_string()))?;

        let mut profiles = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "toml") {
                if let Some(stem) = path.file_stem() {
                    if let Some(name) = stem.to_str() {
                        profiles.push(name.to_string());
                    }
                }
            }
        }

        profiles.sort();
        Ok(profiles)
    }

    /// Delete a profile by name
    pub fn delete_profile(&mut self, name: &str) -> ProfileResult<()> {
        validate_profile_name(name)?;

        let filename = format!("{}.toml", name);
        let path = self.profile_dir.join(&filename);

        if !path.exists() {
            return Err(ProfileError::NotFound(name.to_string()));
        }

        std::fs::remove_file(&path).map_err(|e| ProfileError::IoError(e.to_string()))?;

        // Remove from cache
        self.cache.remove(name);

        Ok(())
    }

    /// Export a profile to a specific path
    pub fn export_profile(&self, name: &str, export_path: &Path) -> ProfileResult<()> {
        let profile = if let Some(cached) = self.cache.get(name) {
            cached.clone()
        } else {
            let filename = format!("{}.toml", name);
            let path = self.profile_dir.join(&filename);

            if !path.exists() {
                return Err(ProfileError::NotFound(name.to_string()));
            }

            let content =
                std::fs::read_to_string(&path).map_err(|e| ProfileError::IoError(e.to_string()))?;

            ProfileConfig::from_toml(&content)?
        };

        let toml_content = profile.to_toml()?;
        std::fs::write(export_path, toml_content).map_err(|e| ProfileError::IoError(e.to_string()))
    }

    /// Import a profile from a specific path
    pub fn import_profile(&mut self, import_path: &Path) -> ProfileResult<ProfileConfig> {
        if !import_path.exists() {
            return Err(ProfileError::NotFound(import_path.display().to_string()));
        }

        let content = std::fs::read_to_string(import_path)
            .map_err(|e| ProfileError::IoError(e.to_string()))?;

        let profile = ProfileConfig::from_toml(&content)?;

        // Save to local profile directory
        self.save_profile(&profile)?;

        Ok(profile)
    }

    /// Check if a profile exists
    pub fn profile_exists(&self, name: &str) -> bool {
        if self.cache.contains_key(name) {
            return true;
        }

        let filename = format!("{}.toml", name);
        let path = self.profile_dir.join(&filename);
        path.exists()
    }

    /// Get profile count
    pub fn profile_count(&self) -> ProfileResult<usize> {
        self.list_profiles().map(|p| p.len())
    }

    /// Clear cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

#[cfg(test)]
mod tests;
