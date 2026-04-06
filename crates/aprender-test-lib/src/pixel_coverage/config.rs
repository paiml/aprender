//! Configuration Schema for Pixel Coverage (PIXEL-001 v2.1 Phase 9)
//!
//! Provides probar.toml compatible configuration for pixel coverage settings.
//! Supports JSON/YAML deserialization via serde.

use serde::{Deserialize, Serialize};

/// Root configuration for pixel coverage in probar.toml
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PixelCoverageConfig {
    /// Enable pixel-perfect verification
    pub enabled: bool,
    /// Primary methodology: "falsification" or "simple"
    pub methodology: String,
    /// Coverage thresholds
    pub thresholds: ThresholdConfig,
    /// Verification metric settings
    pub verification: VerificationConfig,
    /// Output settings
    pub output: OutputConfig,
    /// Performance settings
    pub performance: PerformanceConfig,
}

impl Default for PixelCoverageConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            methodology: "falsification".to_string(),
            thresholds: ThresholdConfig::default(),
            verification: VerificationConfig::default(),
            output: OutputConfig::default(),
            performance: PerformanceConfig::default(),
        }
    }
}

/// Threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ThresholdConfig {
    /// Minimum coverage percentage (0-100)
    pub min_coverage: f32,
    /// Maximum allowed gap size (percent of screen)
    pub max_gap_size: f32,
    /// Falsifiability gateway threshold (0-25)
    pub falsifiability_threshold: f32,
}

impl Default for ThresholdConfig {
    fn default() -> Self {
        Self {
            min_coverage: 85.0,
            max_gap_size: 5.0,
            falsifiability_threshold: 15.0,
        }
    }
}

/// Verification metric configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VerificationConfig {
    /// Structural Similarity Index threshold (0.0 - 1.0)
    pub ssim_threshold: f32,
    /// CIEDE2000 Color Difference threshold (JND)
    pub delta_e_threshold: f32,
    /// Perceptual Hash distance threshold
    pub phash_distance: u32,
    /// PSNR threshold (dB)
    pub psnr_threshold: f32,
}

impl Default for VerificationConfig {
    fn default() -> Self {
        Self {
            ssim_threshold: 0.99,
            delta_e_threshold: 1.0,
            phash_distance: 5,
            psnr_threshold: 40.0,
        }
    }
}

/// Output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OutputConfig {
    /// Generate PNG heatmap
    pub heatmap: bool,
    /// Enable rich terminal output
    pub terminal_gui: bool,
    /// Color palette: "viridis", "magma", "heat"
    pub palette: String,
    /// Show gap highlighting
    pub highlight_gaps: bool,
    /// Show legend
    pub show_legend: bool,
    /// Show confidence intervals
    pub show_confidence: bool,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            heatmap: true,
            terminal_gui: true,
            palette: "viridis".to_string(),
            highlight_gaps: true,
            show_legend: true,
            show_confidence: true,
        }
    }
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PerformanceConfig {
    /// Enable parallel processing (via Rayon)
    pub parallel: bool,
    /// Number of threads (0 = auto-detect)
    pub threads: usize,
    /// Enable downscaling for rapid L1 checks
    pub enable_downscaling: bool,
    /// Downscale factor for rapid checks (e.g., 2 = 50% resolution)
    pub downscale_factor: u32,
    /// Cache perceptual hashes
    pub cache_hashes: bool,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            parallel: true,
            threads: 0, // Auto-detect
            enable_downscaling: true,
            downscale_factor: 2,
            cache_hashes: true,
        }
    }
}

impl PixelCoverageConfig {
    /// Load configuration from JSON string
    ///
    /// # Errors
    /// Returns error if JSON parsing fails
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Load configuration from YAML string
    ///
    /// # Errors
    /// Returns error if YAML parsing fails
    pub fn from_yaml(yaml: &str) -> Result<Self, serde_yaml_ng::Error> {
        serde_yaml_ng::from_str(yaml)
    }

    /// Serialize to JSON
    ///
    /// # Errors
    /// Returns error if serialization fails
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Serialize to YAML
    ///
    /// # Errors
    /// Returns error if serialization fails
    pub fn to_yaml(&self) -> Result<String, serde_yaml_ng::Error> {
        serde_yaml_ng::to_string(self)
    }

    /// Validate configuration values
    #[must_use]
    pub fn validate(&self) -> Vec<ConfigValidationError> {
        let mut errors = Vec::new();

        // Threshold validation
        if !(0.0..=100.0).contains(&self.thresholds.min_coverage) {
            errors.push(ConfigValidationError {
                field: "thresholds.min_coverage".to_string(),
                message: "Must be between 0 and 100".to_string(),
            });
        }

        if !(0.0..=100.0).contains(&self.thresholds.max_gap_size) {
            errors.push(ConfigValidationError {
                field: "thresholds.max_gap_size".to_string(),
                message: "Must be between 0 and 100".to_string(),
            });
        }

        if !(0.0..=25.0).contains(&self.thresholds.falsifiability_threshold) {
            errors.push(ConfigValidationError {
                field: "thresholds.falsifiability_threshold".to_string(),
                message: "Must be between 0 and 25".to_string(),
            });
        }

        // Verification validation
        if !(0.0..=1.0).contains(&self.verification.ssim_threshold) {
            errors.push(ConfigValidationError {
                field: "verification.ssim_threshold".to_string(),
                message: "Must be between 0 and 1".to_string(),
            });
        }

        if self.verification.delta_e_threshold < 0.0 {
            errors.push(ConfigValidationError {
                field: "verification.delta_e_threshold".to_string(),
                message: "Must be non-negative".to_string(),
            });
        }

        // Output validation
        let valid_palettes = ["viridis", "magma", "heat"];
        if !valid_palettes.contains(&self.output.palette.as_str()) {
            errors.push(ConfigValidationError {
                field: "output.palette".to_string(),
                message: format!("Must be one of: {}", valid_palettes.join(", ")),
            });
        }

        // Performance validation
        if self.performance.downscale_factor == 0 {
            errors.push(ConfigValidationError {
                field: "performance.downscale_factor".to_string(),
                message: "Must be at least 1".to_string(),
            });
        }

        errors
    }

    /// Check if configuration is valid
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.validate().is_empty()
    }

    /// Normalize thresholds to 0.0-1.0 range
    #[must_use]
    pub fn normalized_min_coverage(&self) -> f32 {
        self.thresholds.min_coverage / 100.0
    }

    /// Normalize max gap size to 0.0-1.0 range
    #[must_use]
    pub fn normalized_max_gap(&self) -> f32 {
        self.thresholds.max_gap_size / 100.0
    }
}

/// Configuration validation error
#[derive(Debug, Clone)]
pub struct ConfigValidationError {
    /// Field that failed validation
    pub field: String,
    /// Error message
    pub message: String,
}

impl std::fmt::Display for ConfigValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    // =========================================================================
    // Config Tests (H0-CONFIG-XX)
    // =========================================================================

    #[test]
    fn h0_config_01_default() {
        let config = PixelCoverageConfig::default();
        assert!(config.enabled);
        assert_eq!(config.methodology, "falsification");
        assert!((config.thresholds.min_coverage - 85.0).abs() < f32::EPSILON);
    }

    #[test]
    fn h0_config_02_json_roundtrip() {
        let config = PixelCoverageConfig::default();
        let json = config.to_json().unwrap();
        let parsed = PixelCoverageConfig::from_json(&json).unwrap();
        assert!((parsed.thresholds.min_coverage - 85.0).abs() < f32::EPSILON);
    }

    #[test]
    fn h0_config_03_yaml_roundtrip() {
        let config = PixelCoverageConfig::default();
        let yaml = config.to_yaml().unwrap();
        let parsed = PixelCoverageConfig::from_yaml(&yaml).unwrap();
        assert!((parsed.thresholds.min_coverage - 85.0).abs() < f32::EPSILON);
    }

    #[test]
    fn h0_config_04_validation_pass() {
        let config = PixelCoverageConfig::default();
        assert!(config.is_valid());
        assert!(config.validate().is_empty());
    }

    #[test]
    fn h0_config_05_validation_fail() {
        let mut config = PixelCoverageConfig::default();
        config.thresholds.min_coverage = 150.0; // Invalid
        assert!(!config.is_valid());
        assert!(!config.validate().is_empty());
    }

    #[test]
    fn h0_config_06_validation_ssim() {
        let mut config = PixelCoverageConfig::default();
        config.verification.ssim_threshold = 1.5; // Invalid
        let errors = config.validate();
        assert!(errors.iter().any(|e| e.field.contains("ssim")));
    }

    #[test]
    fn h0_config_07_validation_palette() {
        let mut config = PixelCoverageConfig::default();
        config.output.palette = "invalid".to_string();
        let errors = config.validate();
        assert!(errors.iter().any(|e| e.field.contains("palette")));
    }

    #[test]
    fn h0_config_08_normalized() {
        let config = PixelCoverageConfig::default();
        assert!((config.normalized_min_coverage() - 0.85).abs() < f32::EPSILON);
        assert!((config.normalized_max_gap() - 0.05).abs() < f32::EPSILON);
    }

    #[test]
    fn h0_config_09_threshold_defaults() {
        let threshold = ThresholdConfig::default();
        assert!((threshold.falsifiability_threshold - 15.0).abs() < f32::EPSILON);
    }

    #[test]
    fn h0_config_10_performance_defaults() {
        let perf = PerformanceConfig::default();
        assert!(perf.parallel);
        assert_eq!(perf.threads, 0);
        assert!(perf.enable_downscaling);
    }

    #[test]
    fn h0_config_11_output_defaults() {
        let output = OutputConfig::default();
        assert!(output.heatmap);
        assert!(output.terminal_gui);
        assert_eq!(output.palette, "viridis");
    }

    #[test]
    fn h0_config_12_verification_defaults() {
        let verify = VerificationConfig::default();
        assert!((verify.ssim_threshold - 0.99).abs() < f32::EPSILON);
        assert_eq!(verify.phash_distance, 5);
    }

    #[test]
    fn h0_config_13_downscale_validation() {
        let mut config = PixelCoverageConfig::default();
        config.performance.downscale_factor = 0; // Invalid
        let errors = config.validate();
        assert!(errors.iter().any(|e| e.field.contains("downscale")));
    }

    #[test]
    fn h0_config_14_error_display() {
        let error = ConfigValidationError {
            field: "test.field".to_string(),
            message: "test message".to_string(),
        };
        let display = format!("{}", error);
        assert!(display.contains("test.field"));
        assert!(display.contains("test message"));
    }
}
