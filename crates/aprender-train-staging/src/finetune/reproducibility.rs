//! Reproducibility configuration and experiment locking
//!
//! Ensures scientific reproducibility of fine-tuning experiments.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Reproducibility configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproducibilityConfig {
    /// Random seed for all operations
    pub seed: u64,
    /// Use deterministic algorithms (may be slower)
    pub deterministic_algorithms: bool,
    /// Disable cuDNN benchmark mode
    pub cudnn_benchmark: bool,
    /// Enable cuDNN deterministic mode
    pub cudnn_deterministic: bool,
}

impl Default for ReproducibilityConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            deterministic_algorithms: true,
            cudnn_benchmark: false,
            cudnn_deterministic: true,
        }
    }
}

impl ReproducibilityConfig {
    /// Create config with specific seed
    #[must_use]
    pub const fn with_seed(seed: u64) -> Self {
        Self {
            seed,
            deterministic_algorithms: true,
            cudnn_benchmark: false,
            cudnn_deterministic: true,
        }
    }

    /// Disable deterministic mode (faster but not reproducible)
    #[must_use]
    pub const fn non_deterministic(mut self) -> Self {
        self.deterministic_algorithms = false;
        self.cudnn_benchmark = true;
        self.cudnn_deterministic = false;
        self
    }

    /// Apply reproducibility settings to environment
    #[allow(clippy::disallowed_methods)] // Intentional: CUDA env vars for reproducibility
    pub fn apply(&self) {
        // Set environment variables for PyTorch/CUDA if used
        std::env::set_var("PYTHONHASHSEED", self.seed.to_string());
        std::env::set_var("CUBLAS_WORKSPACE_CONFIG", ":4096:8");

        if self.cudnn_deterministic {
            std::env::set_var("CUDNN_DETERMINISTIC", "1");
        }

        if !self.cudnn_benchmark {
            std::env::set_var("CUDNN_BENCHMARK", "0");
        }
    }
}

/// Experiment lockfile for reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentLock {
    /// Experiment ID
    pub experiment_id: String,
    /// Timestamp (ISO 8601)
    pub timestamp: String,
    /// Git commit hash
    pub git_commit: Option<String>,
    /// Rust version
    pub rust_version: String,
    /// CUDA version (if available)
    pub cuda_version: Option<String>,
    /// cuDNN version (if available)
    pub cudnn_version: Option<String>,
    /// Dependencies with versions
    pub dependencies: Vec<DependencyVersion>,
    /// Reproducibility config
    pub reproducibility: ReproducibilityConfig,
    /// Config checksum
    pub config_checksum: String,
    /// Model checksum
    pub model_checksum: Option<String>,
    /// Dataset checksum
    pub dataset_checksum: Option<String>,
}

/// Dependency version info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyVersion {
    /// Crate/package name
    pub name: String,
    /// Version string
    pub version: String,
}

impl ExperimentLock {
    /// Create new experiment lock
    #[must_use]
    pub fn new(experiment_id: impl Into<String>) -> Self {
        Self {
            experiment_id: experiment_id.into(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            git_commit: Self::get_git_commit(),
            rust_version: Self::get_rust_version(),
            cuda_version: Self::get_cuda_version(),
            cudnn_version: None,
            dependencies: Self::get_dependencies(),
            reproducibility: ReproducibilityConfig::default(),
            config_checksum: String::new(),
            model_checksum: None,
            dataset_checksum: None,
        }
    }

    /// Set reproducibility config
    #[must_use]
    pub fn with_reproducibility(mut self, config: ReproducibilityConfig) -> Self {
        self.reproducibility = config;
        self
    }

    /// Set config checksum
    #[must_use]
    pub fn with_config_checksum(mut self, checksum: impl Into<String>) -> Self {
        self.config_checksum = checksum.into();
        self
    }

    /// Set model checksum
    #[must_use]
    pub fn with_model_checksum(mut self, checksum: impl Into<String>) -> Self {
        self.model_checksum = Some(checksum.into());
        self
    }

    /// Set dataset checksum
    #[must_use]
    pub fn with_dataset_checksum(mut self, checksum: impl Into<String>) -> Self {
        self.dataset_checksum = Some(checksum.into());
        self
    }

    /// Save lockfile to path
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be written.
    pub fn save(&self, path: &Path) -> Result<(), std::io::Error> {
        let yaml = serde_yaml::to_string(self).map_err(std::io::Error::other)?;
        std::fs::write(path, yaml)
    }

    /// Load lockfile from path
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be read or parsed.
    pub fn load(path: &Path) -> Result<Self, std::io::Error> {
        let content = std::fs::read_to_string(path)?;
        serde_yaml::from_str(&content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Get current git commit hash
    fn get_git_commit() -> Option<String> {
        std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    }

    /// Get Rust version
    fn get_rust_version() -> String {
        std::process::Command::new("rustc").arg("--version").output().ok().map_or_else(
            || "unknown".into(),
            |o| String::from_utf8_lossy(&o.stdout).trim().to_string(),
        )
    }

    /// Get CUDA version
    fn get_cuda_version() -> Option<String> {
        std::process::Command::new("nvcc")
            .arg("--version")
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| {
                let stdout = String::from_utf8_lossy(&o.stdout);
                stdout.lines().find(|l| l.contains("release")).map(|l| l.trim().to_string())
            })
    }

    /// Get key dependencies from Cargo.lock
    fn get_dependencies() -> Vec<DependencyVersion> {
        // Read from Cargo.lock if available
        let cargo_lock = Path::new("Cargo.lock");
        if !cargo_lock.exists() {
            return Vec::new();
        }

        // Parse relevant dependencies
        let key_deps = ["entrenar", "trueno", "serde", "ndarray"];
        let mut deps = Vec::new();

        if let Ok(content) = std::fs::read_to_string(cargo_lock) {
            let mut current_name = String::new();
            for line in content.lines() {
                if line.starts_with("name = ") {
                    current_name = line
                        .strip_prefix("name = \"")
                        .and_then(|s| s.strip_suffix('"'))
                        .unwrap_or("")
                        .to_string();
                } else if line.starts_with("version = ")
                    && !current_name.is_empty()
                    && key_deps.contains(&current_name.as_str())
                {
                    let version = line
                        .strip_prefix("version = \"")
                        .and_then(|s| s.strip_suffix('"'))
                        .unwrap_or("")
                        .to_string();
                    deps.push(DependencyVersion { name: current_name.clone(), version });
                }
            }
        }

        deps
    }

    /// Verify current environment matches lockfile
    #[must_use]
    pub fn verify(&self) -> VerificationResult {
        let mut result = VerificationResult::default();

        // Check git commit
        if let Some(ref expected) = self.git_commit {
            if let Some(current) = Self::get_git_commit() {
                if &current != expected {
                    result.git_mismatch = Some((expected.clone(), current));
                }
            }
        }

        // Check Rust version
        let current_rust = Self::get_rust_version();
        if current_rust != self.rust_version {
            result.rust_mismatch = Some((self.rust_version.clone(), current_rust));
        }

        // Check CUDA version
        if let Some(ref expected) = self.cuda_version {
            if let Some(current) = Self::get_cuda_version() {
                if &current != expected {
                    result.cuda_mismatch = Some((expected.clone(), current));
                }
            }
        }

        result
    }
}

/// Verification result
#[derive(Debug, Clone, Default)]
pub struct VerificationResult {
    /// Git commit mismatch (expected, actual)
    pub git_mismatch: Option<(String, String)>,
    /// Rust version mismatch (expected, actual)
    pub rust_mismatch: Option<(String, String)>,
    /// CUDA version mismatch (expected, actual)
    pub cuda_mismatch: Option<(String, String)>,
}

impl VerificationResult {
    /// Check if verification passed
    #[must_use]
    pub fn passed(&self) -> bool {
        self.git_mismatch.is_none() && self.rust_mismatch.is_none() && self.cuda_mismatch.is_none()
    }

    /// Get list of warnings
    #[must_use]
    pub fn warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        if let Some((expected, actual)) = &self.git_mismatch {
            warnings.push(format!(
                "Git commit mismatch: expected {}, got {}",
                &expected[..8.min(expected.len())],
                &actual[..8.min(actual.len())]
            ));
        }

        if let Some((expected, actual)) = &self.rust_mismatch {
            warnings.push(format!("Rust version mismatch: expected {expected}, got {actual}"));
        }

        if let Some((expected, actual)) = &self.cuda_mismatch {
            warnings.push(format!("CUDA version mismatch: expected {expected}, got {actual}"));
        }

        warnings
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reproducibility_config_default() {
        let config = ReproducibilityConfig::default();
        assert_eq!(config.seed, 42);
        assert!(config.deterministic_algorithms);
        assert!(!config.cudnn_benchmark);
        assert!(config.cudnn_deterministic);
    }

    #[test]
    fn test_reproducibility_config_with_seed() {
        let config = ReproducibilityConfig::with_seed(123);
        assert_eq!(config.seed, 123);
        assert!(config.deterministic_algorithms);
    }

    #[test]
    fn test_reproducibility_config_non_deterministic() {
        let config = ReproducibilityConfig::default().non_deterministic();
        assert!(!config.deterministic_algorithms);
        assert!(config.cudnn_benchmark);
        assert!(!config.cudnn_deterministic);
    }

    #[test]
    fn test_experiment_lock_new() {
        let lock = ExperimentLock::new("test-001");
        assert_eq!(lock.experiment_id, "test-001");
        assert!(!lock.timestamp.is_empty());
        assert!(!lock.rust_version.is_empty());
    }

    #[test]
    fn test_experiment_lock_with_checksums() {
        let lock = ExperimentLock::new("test")
            .with_config_checksum("abc123")
            .with_model_checksum("def456")
            .with_dataset_checksum("ghi789");

        assert_eq!(lock.config_checksum, "abc123");
        assert_eq!(lock.model_checksum, Some("def456".into()));
        assert_eq!(lock.dataset_checksum, Some("ghi789".into()));
    }

    #[test]
    fn test_experiment_lock_serialization() {
        let lock =
            ExperimentLock::new("test").with_reproducibility(ReproducibilityConfig::with_seed(100));

        let yaml = serde_yaml::to_string(&lock).expect("operation should succeed");
        assert!(yaml.contains("experiment_id: test"));
        assert!(yaml.contains("seed: 100"));

        let restored: ExperimentLock =
            serde_yaml::from_str(&yaml).expect("lock acquisition should succeed");
        assert_eq!(restored.experiment_id, "test");
        assert_eq!(restored.reproducibility.seed, 100);
    }

    #[test]
    fn test_verification_result_passed() {
        let result = VerificationResult::default();
        assert!(result.passed());
        assert!(result.warnings().is_empty());
    }

    #[test]
    fn test_verification_result_with_mismatches() {
        let result = VerificationResult {
            git_mismatch: Some(("abc123".into(), "def456".into())),
            rust_mismatch: None,
            cuda_mismatch: None,
        };

        assert!(!result.passed());
        assert_eq!(result.warnings().len(), 1);
        assert!(result.warnings()[0].contains("Git commit"));
    }

    #[test]
    fn test_dependency_version() {
        let dep = DependencyVersion { name: "entrenar".into(), version: "0.5.6".into() };

        let json = serde_json::to_string(&dep).expect("JSON serialization should succeed");
        assert!(json.contains("entrenar"));
        assert!(json.contains("0.5.6"));
    }

    #[test]
    fn test_reproducibility_config_apply() {
        let config = ReproducibilityConfig::with_seed(12345);
        config.apply();

        // Verify environment variables were set
        assert_eq!(std::env::var("PYTHONHASHSEED").expect("operation should succeed"), "12345");
        assert_eq!(
            std::env::var("CUBLAS_WORKSPACE_CONFIG").expect("config should be valid"),
            ":4096:8"
        );
    }

    #[test]
    fn test_experiment_lock_save_load() {
        let lock = ExperimentLock::new("save-load-test")
            .with_reproducibility(ReproducibilityConfig::with_seed(999))
            .with_config_checksum("sha256:test");

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_lock.yaml");

        // Save
        lock.save(&path).expect("save should succeed");

        // Load
        let loaded = ExperimentLock::load(&path).expect("lock acquisition should succeed");
        assert_eq!(loaded.experiment_id, "save-load-test");
        assert_eq!(loaded.reproducibility.seed, 999);
        assert_eq!(loaded.config_checksum, "sha256:test");

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_experiment_lock_verify() {
        let lock = ExperimentLock::new("verify-test");
        let result = lock.verify();
        // At minimum, the result should be valid
        let _ = result.passed();
        let _ = result.warnings();
    }

    #[test]
    fn test_verification_result_multiple_warnings() {
        let result = VerificationResult {
            git_mismatch: Some(("abc12345".into(), "def67890".into())),
            rust_mismatch: Some(("1.70.0".into(), "1.75.0".into())),
            cuda_mismatch: Some(("12.0".into(), "12.1".into())),
        };

        assert!(!result.passed());
        let warnings = result.warnings();
        assert_eq!(warnings.len(), 3);
        assert!(warnings.iter().any(|w| w.contains("Git")));
        assert!(warnings.iter().any(|w| w.contains("Rust")));
        assert!(warnings.iter().any(|w| w.contains("CUDA")));
    }

    #[test]
    fn test_experiment_lock_with_all_checksums() {
        let lock = ExperimentLock::new("checksum-test")
            .with_config_checksum("sha256:config")
            .with_model_checksum("sha256:model")
            .with_dataset_checksum("sha256:dataset");

        assert_eq!(lock.config_checksum, "sha256:config");
        assert_eq!(lock.model_checksum, Some("sha256:model".into()));
        assert_eq!(lock.dataset_checksum, Some("sha256:dataset".into()));
    }

    #[test]
    fn test_experiment_lock_yaml_format() {
        let lock =
            ExperimentLock::new("yaml-test").with_reproducibility(ReproducibilityConfig::default());

        let yaml = serde_yaml::to_string(&lock).expect("operation should succeed");
        assert!(yaml.contains("experiment_id"));
        assert!(yaml.contains("timestamp"));
        assert!(yaml.contains("reproducibility"));
        assert!(yaml.contains("seed: 42"));
    }
}
