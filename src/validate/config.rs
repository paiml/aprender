//! Configuration for validation operations
//!
//! Per specification Section 3.3 and Section 8

use std::path::PathBuf;

/// Output format for validation results
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ValidationOutputFormat {
    /// Human-readable text format (default)
    #[default]
    Text,
    /// JSON format for machine parsing
    Json,
    /// `JUnit` XML format for CI systems
    JUnit,
}

impl std::str::FromStr for ValidationOutputFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "text" => Ok(Self::Text),
            "json" => Ok(Self::Json),
            "junit" => Ok(Self::JUnit),
            other => Err(format!("Unknown output format: {other}")),
        }
    }
}

/// Configuration for validation operations
///
/// Uses builder pattern for fluent configuration.
#[derive(Debug, Clone)]
pub struct ValidateConfig {
    /// Path to golden trace baseline directory
    pub baseline_dir: Option<PathBuf>,
    /// Directory to generate new baseline
    pub generate_dir: Option<PathBuf>,
    /// Timing tolerance percentage (default: 10.0)
    pub tolerance_percent: f32,
    /// Strict validation mode (zero tolerance)
    pub strict_mode: bool,
    /// Ignore timing, compare behavior only
    pub ignore_timing: bool,
    /// Stop on first regression
    pub fail_fast: bool,
    /// Output format for results
    pub output_format: ValidationOutputFormat,
    /// APR model file for model-aware validation
    pub apr_model: Option<PathBuf>,
    /// Enable canary output validation
    pub canary_validation: bool,
    /// Trace model layers for debugging
    pub trace_layers: bool,
}

impl Default for ValidateConfig {
    fn default() -> Self {
        Self {
            baseline_dir: None,
            generate_dir: None,
            tolerance_percent: 10.0,
            strict_mode: false,
            ignore_timing: false,
            fail_fast: false,
            output_format: ValidationOutputFormat::Text,
            apr_model: None,
            canary_validation: false,
            trace_layers: false,
        }
    }
}

impl ValidateConfig {
    /// Create a new config with specified tolerance
    pub fn with_tolerance(tolerance: f32) -> Self {
        Self { tolerance_percent: tolerance, ..Self::default() }
    }

    /// Set the tolerance percentage
    pub fn set_tolerance(mut self, tolerance: f32) -> Self {
        self.tolerance_percent = tolerance;
        self
    }

    /// Enable or disable strict mode
    ///
    /// When strict mode is enabled, tolerance is set to 0.0
    pub fn with_strict_mode(mut self, strict: bool) -> Self {
        self.strict_mode = strict;
        if strict {
            self.tolerance_percent = 0.0;
        }
        self
    }

    /// Enable or disable timing comparison
    pub fn with_ignore_timing(mut self, ignore: bool) -> Self {
        self.ignore_timing = ignore;
        self
    }

    /// Enable or disable fail-fast mode
    pub fn with_fail_fast(mut self, fail_fast: bool) -> Self {
        self.fail_fast = fail_fast;
        self
    }

    /// Set the output format
    pub fn with_output_format(mut self, format: ValidationOutputFormat) -> Self {
        self.output_format = format;
        self
    }

    /// Set the baseline directory
    pub fn with_baseline(mut self, path: PathBuf) -> Self {
        self.baseline_dir = Some(path);
        self
    }

    /// Set the generate directory
    pub fn with_generate(mut self, path: PathBuf) -> Self {
        self.generate_dir = Some(path);
        self
    }

    /// Set the APR model path
    pub fn with_apr_model(mut self, path: PathBuf) -> Self {
        self.apr_model = Some(path);
        self
    }

    /// Enable canary validation
    pub fn with_canary_validation(mut self, enabled: bool) -> Self {
        self.canary_validation = enabled;
        self
    }

    /// Enable layer tracing
    pub fn with_trace_layers(mut self, enabled: bool) -> Self {
        contract_pre_error_handling!();
        self.trace_layers = enabled;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ValidateConfig::default();
        assert!((config.tolerance_percent - 10.0).abs() < f32::EPSILON);
        assert!(!config.strict_mode);
        assert!(!config.ignore_timing);
        assert!(!config.fail_fast);
        assert_eq!(config.output_format, ValidationOutputFormat::Text);
        assert!(config.baseline_dir.is_none());
        assert!(config.generate_dir.is_none());
        assert!(config.apr_model.is_none());
        assert!(!config.canary_validation);
        assert!(!config.trace_layers);
    }

    #[test]
    fn test_with_tolerance_constructor() {
        let config = ValidateConfig::with_tolerance(25.0);
        assert!((config.tolerance_percent - 25.0).abs() < f32::EPSILON);
        assert!(!config.strict_mode); // Other defaults preserved
    }

    #[test]
    fn test_set_tolerance_builder() {
        let config = ValidateConfig::default().set_tolerance(15.0);
        assert!((config.tolerance_percent - 15.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_with_strict_mode_sets_zero_tolerance() {
        let config = ValidateConfig::default().with_strict_mode(true);
        assert!(config.strict_mode);
        assert!((config.tolerance_percent - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_with_strict_mode_false_preserves_tolerance() {
        let config = ValidateConfig::default().set_tolerance(20.0).with_strict_mode(false);
        assert!(!config.strict_mode);
        assert!((config.tolerance_percent - 20.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_with_ignore_timing() {
        let config = ValidateConfig::default().with_ignore_timing(true);
        assert!(config.ignore_timing);
    }

    #[test]
    fn test_with_fail_fast() {
        let config = ValidateConfig::default().with_fail_fast(true);
        assert!(config.fail_fast);
    }

    #[test]
    fn test_with_output_format() {
        let config = ValidateConfig::default().with_output_format(ValidationOutputFormat::Json);
        assert_eq!(config.output_format, ValidationOutputFormat::Json);

        let config = ValidateConfig::default().with_output_format(ValidationOutputFormat::JUnit);
        assert_eq!(config.output_format, ValidationOutputFormat::JUnit);
    }

    #[test]
    fn test_with_baseline() {
        let path = PathBuf::from("/tmp/baseline");
        let config = ValidateConfig::default().with_baseline(path.clone());
        assert_eq!(config.baseline_dir, Some(path));
    }

    #[test]
    fn test_with_generate() {
        let path = PathBuf::from("/tmp/generate");
        let config = ValidateConfig::default().with_generate(path.clone());
        assert_eq!(config.generate_dir, Some(path));
    }

    #[test]
    fn test_with_apr_model() {
        let path = PathBuf::from("/tmp/model.apr");
        let config = ValidateConfig::default().with_apr_model(path.clone());
        assert_eq!(config.apr_model, Some(path));
    }

    #[test]
    fn test_with_canary_validation() {
        let config = ValidateConfig::default().with_canary_validation(true);
        assert!(config.canary_validation);
    }

    #[test]
    fn test_with_trace_layers() {
        let config = ValidateConfig::default().with_trace_layers(true);
        assert!(config.trace_layers);
    }

    #[test]
    fn test_builder_chaining() {
        let config = ValidateConfig::default()
            .set_tolerance(5.0)
            .with_ignore_timing(true)
            .with_fail_fast(true)
            .with_output_format(ValidationOutputFormat::Json)
            .with_baseline(PathBuf::from("/baseline"))
            .with_apr_model(PathBuf::from("/model.apr"))
            .with_canary_validation(true)
            .with_trace_layers(true);

        assert!((config.tolerance_percent - 5.0).abs() < f32::EPSILON);
        assert!(config.ignore_timing);
        assert!(config.fail_fast);
        assert_eq!(config.output_format, ValidationOutputFormat::Json);
        assert_eq!(config.baseline_dir, Some(PathBuf::from("/baseline")));
        assert_eq!(config.apr_model, Some(PathBuf::from("/model.apr")));
        assert!(config.canary_validation);
        assert!(config.trace_layers);
    }

    #[test]
    fn test_output_format_parse() {
        assert!(matches!(
            "text".parse::<ValidationOutputFormat>(),
            Ok(ValidationOutputFormat::Text)
        ));
        assert!(matches!(
            "JSON".parse::<ValidationOutputFormat>(),
            Ok(ValidationOutputFormat::Json)
        ));
        assert!(matches!(
            "junit".parse::<ValidationOutputFormat>(),
            Ok(ValidationOutputFormat::JUnit)
        ));
        assert!("invalid".parse::<ValidationOutputFormat>().is_err());
    }

    #[test]
    fn test_output_format_default() {
        let format = ValidationOutputFormat::default();
        assert_eq!(format, ValidationOutputFormat::Text);
    }

    #[test]
    fn test_output_format_debug() {
        // Test that Debug is implemented
        let format = ValidationOutputFormat::Json;
        let debug_str = format!("{:?}", format);
        assert!(debug_str.contains("Json"));
    }

    #[test]
    fn test_output_format_clone() {
        let format = ValidationOutputFormat::JUnit;
        let cloned = format;
        assert_eq!(cloned, ValidationOutputFormat::JUnit);
    }
}
