//! CLI framework and styling (Standardized Work principle).
//!
//! Provides consistent CLI patterns across all entrenar tools.

use clap::Parser;
use std::str::FromStr;

/// Common CLI configuration shared across all entrenar tools.
#[derive(Debug, Clone)]
pub struct Cli {
    /// Output format
    pub format: OutputFormat,
    /// Verbosity level (0 = quiet, 1 = normal, 2 = verbose)
    pub verbosity: u8,
    /// Whether colors are enabled
    pub color: bool,
}

impl Default for Cli {
    fn default() -> Self {
        Self { format: OutputFormat::Table, verbosity: 1, color: true }
    }
}

impl Cli {
    /// Create a new CLI config with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set output format.
    pub fn with_format(mut self, format: OutputFormat) -> Self {
        self.format = format;
        self
    }

    /// Set verbosity level.
    pub fn with_verbosity(mut self, level: u8) -> Self {
        self.verbosity = level;
        self
    }

    /// Enable or disable colors.
    pub fn with_color(mut self, enabled: bool) -> Self {
        self.color = enabled;
        self
    }

    /// Check if quiet mode is enabled.
    pub fn is_quiet(&self) -> bool {
        self.verbosity == 0
    }

    /// Check if verbose mode is enabled.
    pub fn is_verbose(&self) -> bool {
        self.verbosity >= 2
    }
}

/// Output format for CLI commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputFormat {
    /// Human-readable table format (default).
    #[default]
    Table,
    /// JSON format for machine parsing.
    Json,
    /// Compact single-line format.
    Compact,
}

impl FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "table" | "text" => Ok(Self::Table),
            "json" => Ok(Self::Json),
            "compact" | "line" => Ok(Self::Compact),
            _ => Err(format!("Unknown output format '{s}'. Valid options: table, json, compact")),
        }
    }
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Table => write!(f, "table"),
            Self::Json => write!(f, "json"),
            Self::Compact => write!(f, "compact"),
        }
    }
}

/// Common CLI arguments that can be mixed into any command.
#[derive(Parser, Debug, Clone)]
pub struct CommonArgs {
    /// Output format: table, json, or compact
    #[arg(short, long, default_value = "table")]
    pub format: String,

    /// Suppress non-essential output
    #[arg(short, long)]
    pub quiet: bool,

    /// Enable verbose output
    #[arg(short, long)]
    pub verbose: bool,

    /// Disable colored output
    #[arg(long)]
    pub no_color: bool,
}

impl CommonArgs {
    /// Convert to Cli config.
    pub fn to_cli(&self) -> Cli {
        let verbosity = if self.quiet {
            0
        } else if self.verbose {
            2
        } else {
            1
        };

        Cli { format: self.format.parse().unwrap_or_default(), verbosity, color: !self.no_color }
    }
}

/// Terminal styling helpers.
pub mod styles {
    /// ANSI color codes for consistent styling.
    pub struct Colors;

    impl Colors {
        pub const RESET: &'static str = "\x1b[0m";
        pub const BOLD: &'static str = "\x1b[1m";
        pub const DIM: &'static str = "\x1b[2m";
        pub const RED: &'static str = "\x1b[31m";
        pub const GREEN: &'static str = "\x1b[32m";
        pub const YELLOW: &'static str = "\x1b[33m";
        pub const BLUE: &'static str = "\x1b[34m";
        pub const CYAN: &'static str = "\x1b[36m";
    }

    /// Format a success message.
    pub fn success(msg: &str) -> String {
        format!("{}✓{} {}", Colors::GREEN, Colors::RESET, msg)
    }

    /// Format an error message.
    pub fn error(msg: &str) -> String {
        format!("{}✗{} {}", Colors::RED, Colors::RESET, msg)
    }

    /// Format a warning message.
    pub fn warning(msg: &str) -> String {
        format!("{}⚠{} {}", Colors::YELLOW, Colors::RESET, msg)
    }

    /// Format an info message.
    pub fn info(msg: &str) -> String {
        format!("{}ℹ{} {}", Colors::BLUE, Colors::RESET, msg)
    }

    /// Format a header/title.
    pub fn header(msg: &str) -> String {
        format!("{}{}{}", Colors::BOLD, msg, Colors::RESET)
    }

    /// Format a dim/secondary message.
    pub fn dim(msg: &str) -> String {
        format!("{}{}{}", Colors::DIM, msg, Colors::RESET)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_format_roundtrip() {
        for format in [OutputFormat::Table, OutputFormat::Json, OutputFormat::Compact] {
            let s = format.to_string();
            let parsed: OutputFormat = s.parse().expect("parsing should succeed");
            assert_eq!(format, parsed);
        }
    }

    #[test]
    fn test_output_format_case_insensitive() {
        assert_eq!(
            "JSON".parse::<OutputFormat>().expect("parsing should succeed"),
            OutputFormat::Json
        );
        assert_eq!(
            "Table".parse::<OutputFormat>().expect("parsing should succeed"),
            OutputFormat::Table
        );
        assert_eq!(
            "COMPACT".parse::<OutputFormat>().expect("parsing should succeed"),
            OutputFormat::Compact
        );
    }

    #[test]
    fn test_cli_verbosity_levels() {
        let quiet = Cli::new().with_verbosity(0);
        assert!(quiet.is_quiet());
        assert!(!quiet.is_verbose());

        let normal = Cli::new().with_verbosity(1);
        assert!(!normal.is_quiet());
        assert!(!normal.is_verbose());

        let verbose = Cli::new().with_verbosity(2);
        assert!(!verbose.is_quiet());
        assert!(verbose.is_verbose());
    }

    #[test]
    fn test_styles_include_ansi_codes() {
        let success = styles::success("done");
        assert!(success.contains('\x1b'));
        assert!(success.contains("done"));
        assert!(success.contains('✓'));
    }

    #[test]
    fn test_common_args_to_cli() {
        let args =
            CommonArgs { format: "json".to_string(), quiet: false, verbose: true, no_color: true };
        let cli = args.to_cli();

        assert_eq!(cli.format, OutputFormat::Json);
        assert_eq!(cli.verbosity, 2);
        assert!(!cli.color);
    }

    #[test]
    fn test_output_format_aliases() {
        // "text" is alias for Table
        assert_eq!(
            "text".parse::<OutputFormat>().expect("parsing should succeed"),
            OutputFormat::Table
        );
        // "line" is alias for Compact
        assert_eq!(
            "line".parse::<OutputFormat>().expect("parsing should succeed"),
            OutputFormat::Compact
        );
    }

    #[test]
    fn test_output_format_invalid() {
        let result = "invalid_format".parse::<OutputFormat>();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Unknown output format"));
        assert!(err.contains("invalid_format"));
    }

    #[test]
    fn test_cli_default_values() {
        let cli = Cli::default();
        assert_eq!(cli.format, OutputFormat::Table);
        assert_eq!(cli.verbosity, 1);
        assert!(cli.color);
        assert!(!cli.is_quiet());
        assert!(!cli.is_verbose());
    }

    #[test]
    fn test_cli_builder_pattern() {
        let cli = Cli::new().with_format(OutputFormat::Json).with_verbosity(2).with_color(false);

        assert_eq!(cli.format, OutputFormat::Json);
        assert_eq!(cli.verbosity, 2);
        assert!(!cli.color);
    }

    #[test]
    fn test_styles_all_variants() {
        let success = styles::success("ok");
        assert!(success.contains('✓'));
        assert!(success.contains(styles::Colors::GREEN));

        let error = styles::error("fail");
        assert!(error.contains('✗'));
        assert!(error.contains(styles::Colors::RED));

        let warning = styles::warning("warn");
        assert!(warning.contains('⚠'));
        assert!(warning.contains(styles::Colors::YELLOW));

        let info = styles::info("note");
        assert!(info.contains('ℹ'));
        assert!(info.contains(styles::Colors::BLUE));

        let header = styles::header("title");
        assert!(header.contains("title"));
        assert!(header.contains(styles::Colors::BOLD));

        let dim = styles::dim("secondary");
        assert!(dim.contains("secondary"));
        assert!(dim.contains(styles::Colors::DIM));
    }

    #[test]
    fn test_common_args_quiet_mode() {
        let args = CommonArgs {
            format: "table".to_string(),
            quiet: true,
            verbose: false,
            no_color: false,
        };
        let cli = args.to_cli();

        assert_eq!(cli.verbosity, 0);
        assert!(cli.is_quiet());
    }

    #[test]
    fn test_common_args_default_format_fallback() {
        let args = CommonArgs {
            format: "invalid".to_string(),
            quiet: false,
            verbose: false,
            no_color: false,
        };
        let cli = args.to_cli();

        // Should fallback to default (Table) on parse error
        assert_eq!(cli.format, OutputFormat::Table);
    }

    #[test]
    fn test_verbosity_level_3_is_verbose() {
        let cli = Cli::new().with_verbosity(3);
        assert!(cli.is_verbose());
        assert!(!cli.is_quiet());
    }
}
