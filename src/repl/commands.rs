//! REPL Command Parser (ALIM-REPL-003)
//!
//! Parses user input into structured commands.
//! Command grammar mirrors batch CLI for Standard Work (Hyojun) compliance.

use crate::{Error, Result};

/// REPL commands matching batch CLI grammar (Standard Work)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplCommand {
    // ─────────────────────────────────────────────────────────────────────────────
    // Data Loading Commands
    // ─────────────────────────────────────────────────────────────────────────────
    /// Load a dataset from file
    Load {
        /// Path to the dataset file
        path: String,
    },

    /// List loaded datasets
    Datasets,

    /// Switch active dataset
    Use {
        /// Name of the dataset to switch to
        name: String,
    },

    // ─────────────────────────────────────────────────────────────────────────────
    // Data Inspection Commands
    // ─────────────────────────────────────────────────────────────────────────────
    /// Display dataset metadata
    Info,

    /// Show first n rows
    Head {
        /// Number of rows to display
        n: usize,
    },

    /// Display column schema
    Schema,

    // ─────────────────────────────────────────────────────────────────────────────
    // Quality Commands (Andon)
    // ─────────────────────────────────────────────────────────────────────────────
    /// Run quality checks
    QualityCheck,

    /// Compute 100-point quality score
    QualityScore {
        /// Show improvement suggestions
        suggest: bool,
        /// Output as JSON
        json: bool,
        /// Output as badge URL
        badge: bool,
    },

    // ─────────────────────────────────────────────────────────────────────────────
    // Analysis Commands
    // ─────────────────────────────────────────────────────────────────────────────
    /// Detect drift against reference
    DriftDetect {
        /// Path to reference dataset
        reference: String,
    },

    /// Convert/export to format
    Convert {
        /// Target format (csv, parquet, json)
        format: String,
    },

    // ─────────────────────────────────────────────────────────────────────────────
    // Pipeline Commands (Batuta Integration - ALIM-REPL-005)
    // ─────────────────────────────────────────────────────────────────────────────
    /// Export data for pipeline integration
    Export {
        /// What to export (quality)
        what: String,
        /// Output as JSON
        json: bool,
    },

    /// Validate against schema spec
    Validate {
        /// Path to schema specification
        schema: String,
    },

    // ─────────────────────────────────────────────────────────────────────────────
    // Session Commands
    // ─────────────────────────────────────────────────────────────────────────────
    /// Show/export command history (ALIM-REPL-006)
    History {
        /// Export as reproducible script
        export: bool,
    },

    /// Show help
    Help {
        /// Help topic (quality, drift, export)
        topic: Option<String>,
    },

    /// Exit REPL
    Quit,
}

/// Parser for REPL commands (Poka-Yoke input validation)
pub struct CommandParser;

impl CommandParser {
    /// Parse a command string into a ReplCommand
    ///
    /// # Errors
    ///
    /// Returns an error if the command is invalid or unknown.
    pub fn parse(input: &str) -> Result<ReplCommand> {
        let input = input.trim();

        if input.is_empty() {
            return Err(Error::parse("Empty command"));
        }

        let parts: Vec<&str> = input.split_whitespace().collect();
        let cmd = parts[0].to_lowercase();
        let args = &parts[1..];

        match cmd.as_str() {
            // Data Loading
            "load" => Self::parse_load(args),
            "datasets" => Ok(ReplCommand::Datasets),
            "use" => Self::parse_use(args),

            // Data Inspection
            "info" => Ok(ReplCommand::Info),
            "head" => Self::parse_head(args),
            "schema" => Ok(ReplCommand::Schema),

            // Quality
            "quality" => Self::parse_quality(args),

            // Analysis
            "drift" => Self::parse_drift(args),
            "convert" => Self::parse_convert(args),

            // Pipeline
            "export" => Self::parse_export(args),
            "validate" => Self::parse_validate(args),

            // Session
            "history" => Ok(Self::parse_history(args)),
            "help" | "?" => Ok(Self::parse_help(args)),
            "quit" | "exit" | "q" => Ok(ReplCommand::Quit),

            _ => Err(Error::parse(format!("Unknown command: '{}'", cmd))),
        }
    }

    fn parse_load(args: &[&str]) -> Result<ReplCommand> {
        if args.is_empty() {
            return Err(Error::parse("load requires a file path"));
        }
        Ok(ReplCommand::Load {
            path: args[0].to_string(),
        })
    }

    fn parse_use(args: &[&str]) -> Result<ReplCommand> {
        if args.is_empty() {
            return Err(Error::parse("use requires a dataset name"));
        }
        Ok(ReplCommand::Use {
            name: args[0].to_string(),
        })
    }

    fn parse_head(args: &[&str]) -> Result<ReplCommand> {
        let n = if args.is_empty() {
            10 // Default per spec
        } else {
            args[0]
                .parse()
                .map_err(|_| Error::parse(format!("Invalid number: '{}'", args[0])))?
        };
        Ok(ReplCommand::Head { n })
    }

    fn parse_quality(args: &[&str]) -> Result<ReplCommand> {
        if args.is_empty() {
            return Err(Error::parse("quality requires subcommand: check, score"));
        }

        match args[0].to_lowercase().as_str() {
            "check" => Ok(ReplCommand::QualityCheck),
            "score" => {
                let suggest = args.contains(&"--suggest");
                let json = args.contains(&"--json");
                let badge = args.contains(&"--badge");
                Ok(ReplCommand::QualityScore {
                    suggest,
                    json,
                    badge,
                })
            }
            _ => Err(Error::parse(format!(
                "Unknown quality subcommand: '{}'. Use: check, score",
                args[0]
            ))),
        }
    }

    fn parse_drift(args: &[&str]) -> Result<ReplCommand> {
        if args.is_empty() {
            return Err(Error::parse("drift requires subcommand: detect"));
        }

        match args[0].to_lowercase().as_str() {
            "detect" => {
                if args.len() < 2 {
                    return Err(Error::parse("drift detect requires a reference file"));
                }
                Ok(ReplCommand::DriftDetect {
                    reference: args[1].to_string(),
                })
            }
            _ => Err(Error::parse(format!(
                "Unknown drift subcommand: '{}'. Use: detect",
                args[0]
            ))),
        }
    }

    fn parse_convert(args: &[&str]) -> Result<ReplCommand> {
        if args.is_empty() {
            return Err(Error::parse(
                "convert requires a format: csv, parquet, json",
            ));
        }
        let format = args[0].to_lowercase();
        match format.as_str() {
            "csv" | "parquet" | "json" => Ok(ReplCommand::Convert { format }),
            _ => Err(Error::parse(format!(
                "Unknown format: '{}'. Use: csv, parquet, json",
                args[0]
            ))),
        }
    }

    fn parse_export(args: &[&str]) -> Result<ReplCommand> {
        if args.is_empty() {
            return Err(Error::parse("export requires what to export: quality"));
        }

        let what = args[0].to_lowercase();
        let json = args.iter().any(|f| *f == "--json" || *f == "-j");

        Ok(ReplCommand::Export { what, json })
    }

    fn parse_validate(args: &[&str]) -> Result<ReplCommand> {
        // Look for --schema flag
        let schema_idx = args.iter().position(|f| *f == "--schema" || *f == "-s");

        let schema = if let Some(idx) = schema_idx {
            if idx + 1 < args.len() {
                args[idx + 1].to_string()
            } else {
                return Err(Error::parse("--schema requires a file path"));
            }
        } else {
            return Err(Error::parse("validate requires --schema <file>"));
        };

        Ok(ReplCommand::Validate { schema })
    }

    fn parse_history(args: &[&str]) -> ReplCommand {
        let export = args.iter().any(|f| *f == "--export" || *f == "-e");
        ReplCommand::History { export }
    }

    fn parse_help(args: &[&str]) -> ReplCommand {
        let topic = args.first().map(|s| (*s).to_string());
        ReplCommand::Help { topic }
    }

    /// Get all valid command names for autocomplete
    #[must_use]
    pub fn command_names() -> Vec<&'static str> {
        vec![
            "load", "datasets", "use", "info", "head", "schema", "quality", "drift", "convert",
            "export", "validate", "history", "help", "quit", "exit",
        ]
    }

    /// Get subcommands for a given command
    #[must_use]
    pub fn subcommands(command: &str) -> Vec<&'static str> {
        match command {
            "quality" => vec!["check", "score"],
            "drift" => vec!["detect"],
            _ => vec![],
        }
    }

    /// Get flags for a given command
    #[must_use]
    pub fn flags(command: &str, subcommand: Option<&str>) -> Vec<&'static str> {
        match (command, subcommand) {
            ("quality", Some("score")) => vec!["--suggest", "--json", "--badge"],
            ("export", _) => vec!["--json"],
            ("validate", _) => vec!["--schema"],
            ("history", _) => vec!["--export"],
            _ => vec![],
        }
    }
}
