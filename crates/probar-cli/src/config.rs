//! CLI configuration

use serde::{Deserialize, Serialize};

/// CLI verbosity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Verbosity {
    /// Quiet - minimal output
    Quiet,
    /// Normal - default output
    #[default]
    Normal,
    /// Verbose - extra output
    Verbose,
    /// Debug - maximum output
    Debug,
}

impl Verbosity {
    /// Check if quiet mode
    #[must_use]
    pub const fn is_quiet(self) -> bool {
        matches!(self, Self::Quiet)
    }

    /// Check if verbose or higher
    #[must_use]
    pub const fn is_verbose(self) -> bool {
        matches!(self, Self::Verbose | Self::Debug)
    }

    /// Check if debug mode
    #[must_use]
    pub const fn is_debug(self) -> bool {
        matches!(self, Self::Debug)
    }
}

/// Color output choice
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ColorChoice {
    /// Always use colors
    Always,
    /// Use colors when output is a terminal
    #[default]
    Auto,
    /// Never use colors
    Never,
}

impl ColorChoice {
    /// Should use colors based on output detection
    #[must_use]
    pub fn should_color(self) -> bool {
        match self {
            Self::Always => true,
            Self::Never => false,
            Self::Auto => atty_is_terminal(),
        }
    }
}

/// Check if stdout is a terminal
fn atty_is_terminal() -> bool {
    // Use std library detection
    std::io::IsTerminal::is_terminal(&std::io::stdout())
}

/// CLI configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliConfig {
    /// Verbosity level
    pub verbosity: Verbosity,
    /// Color output choice
    pub color: ColorChoice,
    /// Number of parallel jobs (0 = auto-detect)
    pub parallel_jobs: usize,
    /// Fail fast on first error
    pub fail_fast: bool,
    /// Watch mode enabled
    pub watch: bool,
    /// Coverage enabled
    pub coverage: bool,
    /// Output directory for reports
    pub output_dir: String,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            verbosity: Verbosity::Normal,
            color: ColorChoice::Auto,
            parallel_jobs: 0, // Auto-detect
            fail_fast: false,
            watch: false,
            coverage: false,
            output_dir: "target/probar".to_string(),
        }
    }
}

impl CliConfig {
    /// Create new default configuration
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set verbosity
    #[must_use]
    pub const fn with_verbosity(mut self, verbosity: Verbosity) -> Self {
        self.verbosity = verbosity;
        self
    }

    /// Set color choice
    #[must_use]
    pub const fn with_color(mut self, color: ColorChoice) -> Self {
        self.color = color;
        self
    }

    /// Set parallel jobs
    #[must_use]
    pub const fn with_parallel_jobs(mut self, jobs: usize) -> Self {
        self.parallel_jobs = jobs;
        self
    }

    /// Set fail fast
    #[must_use]
    pub const fn with_fail_fast(mut self, fail_fast: bool) -> Self {
        self.fail_fast = fail_fast;
        self
    }

    /// Set watch mode
    #[must_use]
    pub const fn with_watch(mut self, watch: bool) -> Self {
        self.watch = watch;
        self
    }

    /// Set coverage
    #[must_use]
    pub const fn with_coverage(mut self, coverage: bool) -> Self {
        self.coverage = coverage;
        self
    }

    /// Set output directory
    #[must_use]
    pub fn with_output_dir(mut self, dir: impl Into<String>) -> Self {
        self.output_dir = dir.into();
        self
    }

    /// Get effective number of parallel jobs
    #[must_use]
    #[allow(clippy::redundant_closure_for_method_calls)] // Cannot use NonZero::get directly due to MSRV 1.75 (stable in 1.79)
    pub fn effective_jobs(&self) -> usize {
        if self.parallel_jobs == 0 {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        } else {
            self.parallel_jobs
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    mod verbosity_tests {
        use super::*;

        #[test]
        fn test_default_verbosity() {
            let v = Verbosity::default();
            assert_eq!(v, Verbosity::Normal);
        }

        #[test]
        fn test_is_quiet() {
            assert!(Verbosity::Quiet.is_quiet());
            assert!(!Verbosity::Normal.is_quiet());
            assert!(!Verbosity::Verbose.is_quiet());
            assert!(!Verbosity::Debug.is_quiet());
        }

        #[test]
        fn test_is_verbose() {
            assert!(!Verbosity::Quiet.is_verbose());
            assert!(!Verbosity::Normal.is_verbose());
            assert!(Verbosity::Verbose.is_verbose());
            assert!(Verbosity::Debug.is_verbose());
        }

        #[test]
        fn test_is_debug() {
            assert!(!Verbosity::Quiet.is_debug());
            assert!(!Verbosity::Normal.is_debug());
            assert!(!Verbosity::Verbose.is_debug());
            assert!(Verbosity::Debug.is_debug());
        }

        #[test]
        fn test_clone() {
            let v = Verbosity::Debug;
            let cloned = v;
            assert_eq!(v, cloned);
        }

        #[test]
        fn test_debug_trait() {
            let debug = format!("{:?}", Verbosity::Verbose);
            assert!(debug.contains("Verbose"));
        }

        #[test]
        fn test_serialize() {
            let json = serde_json::to_string(&Verbosity::Debug).unwrap();
            assert!(json.contains("Debug"));
        }

        #[test]
        fn test_deserialize() {
            let v: Verbosity = serde_json::from_str("\"Quiet\"").unwrap();
            assert_eq!(v, Verbosity::Quiet);
        }
    }

    mod color_choice_tests {
        use super::*;

        #[test]
        fn test_default_color() {
            let c = ColorChoice::default();
            assert_eq!(c, ColorChoice::Auto);
        }

        #[test]
        fn test_should_color_always() {
            assert!(ColorChoice::Always.should_color());
        }

        #[test]
        fn test_should_color_never() {
            assert!(!ColorChoice::Never.should_color());
        }

        #[test]
        fn test_should_color_auto() {
            // Auto depends on terminal detection, just ensure it doesn't panic
            let _ = ColorChoice::Auto.should_color();
        }

        #[test]
        fn test_clone() {
            let c = ColorChoice::Always;
            let cloned = c;
            assert_eq!(c, cloned);
        }

        #[test]
        fn test_debug_trait() {
            let debug = format!("{:?}", ColorChoice::Never);
            assert!(debug.contains("Never"));
        }

        #[test]
        fn test_serialize() {
            let json = serde_json::to_string(&ColorChoice::Always).unwrap();
            assert!(json.contains("Always"));
        }

        #[test]
        fn test_deserialize() {
            let c: ColorChoice = serde_json::from_str("\"Never\"").unwrap();
            assert_eq!(c, ColorChoice::Never);
        }
    }

    mod cli_config_tests {
        use super::*;

        #[test]
        fn test_default_config() {
            let config = CliConfig::default();
            assert_eq!(config.verbosity, Verbosity::Normal);
            assert_eq!(config.color, ColorChoice::Auto);
            assert_eq!(config.parallel_jobs, 0);
            assert!(!config.fail_fast);
            assert!(!config.watch);
            assert!(!config.coverage);
        }

        #[test]
        fn test_with_verbosity() {
            let config = CliConfig::new().with_verbosity(Verbosity::Debug);
            assert_eq!(config.verbosity, Verbosity::Debug);
        }

        #[test]
        fn test_with_color() {
            let config = CliConfig::new().with_color(ColorChoice::Never);
            assert_eq!(config.color, ColorChoice::Never);
        }

        #[test]
        fn test_with_parallel_jobs() {
            let config = CliConfig::new().with_parallel_jobs(4);
            assert_eq!(config.parallel_jobs, 4);
        }

        #[test]
        fn test_with_fail_fast() {
            let config = CliConfig::new().with_fail_fast(true);
            assert!(config.fail_fast);
        }

        #[test]
        fn test_with_watch() {
            let config = CliConfig::new().with_watch(true);
            assert!(config.watch);
        }

        #[test]
        fn test_with_coverage() {
            let config = CliConfig::new().with_coverage(true);
            assert!(config.coverage);
        }

        #[test]
        fn test_with_output_dir() {
            let config = CliConfig::new().with_output_dir("custom/output");
            assert_eq!(config.output_dir, "custom/output");
        }

        #[test]
        fn test_effective_jobs_specified() {
            let config = CliConfig::new().with_parallel_jobs(8);
            assert_eq!(config.effective_jobs(), 8);
        }

        #[test]
        fn test_effective_jobs_auto() {
            let config = CliConfig::new().with_parallel_jobs(0);
            assert!(config.effective_jobs() >= 1);
        }

        #[test]
        fn test_chained_builders() {
            let config = CliConfig::new()
                .with_verbosity(Verbosity::Verbose)
                .with_color(ColorChoice::Always)
                .with_parallel_jobs(2)
                .with_fail_fast(true)
                .with_coverage(true);

            assert_eq!(config.verbosity, Verbosity::Verbose);
            assert_eq!(config.color, ColorChoice::Always);
            assert_eq!(config.parallel_jobs, 2);
            assert!(config.fail_fast);
            assert!(config.coverage);
        }

        #[test]
        fn test_clone() {
            let config = CliConfig::new()
                .with_verbosity(Verbosity::Debug)
                .with_fail_fast(true);
            let cloned = config.clone();
            assert_eq!(config.verbosity, cloned.verbosity);
            assert_eq!(config.fail_fast, cloned.fail_fast);
        }

        #[test]
        fn test_debug_trait() {
            let config = CliConfig::default();
            let debug = format!("{config:?}");
            assert!(debug.contains("CliConfig"));
        }

        #[test]
        fn test_serialize() {
            let config = CliConfig::new().with_fail_fast(true);
            let json = serde_json::to_string(&config).unwrap();
            assert!(json.contains("fail_fast"));
            assert!(json.contains("true"));
        }

        #[test]
        fn test_deserialize() {
            let json = r#"{"verbosity":"Debug","color":"Always","parallel_jobs":4,"fail_fast":true,"watch":false,"coverage":true,"output_dir":"test"}"#;
            let config: CliConfig = serde_json::from_str(json).unwrap();
            assert_eq!(config.verbosity, Verbosity::Debug);
            assert_eq!(config.color, ColorChoice::Always);
            assert_eq!(config.parallel_jobs, 4);
            assert!(config.fail_fast);
            assert!(config.coverage);
        }

        #[test]
        fn test_output_dir_default() {
            let config = CliConfig::default();
            assert_eq!(config.output_dir, "target/probar");
        }
    }
}
