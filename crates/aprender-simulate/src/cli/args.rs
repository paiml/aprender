//! CLI argument parsing.
//!
//! This module provides the argument parser for the simular CLI.
//! Extracted to enable comprehensive testing of argument parsing logic.

use std::path::PathBuf;

/// CLI arguments container.
#[derive(Debug, Clone, PartialEq)]
pub struct Args {
    /// The command to execute.
    pub command: Command,
}

/// Available CLI commands.
#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    /// Run an experiment
    Run {
        /// Path to the experiment YAML file.
        experiment_path: PathBuf,
        /// Optional seed override.
        seed_override: Option<u64>,
        /// Enable verbose output.
        verbose: bool,
    },
    /// Render simulation to SVG + keyframes
    Render {
        /// Simulation domain (orbit, `monte_carlo`, optimization).
        domain: String,
        /// Output format: svg-frames or svg-keyframes.
        format: RenderFormat,
        /// Output directory.
        output: PathBuf,
        /// Frames per second.
        fps: u32,
        /// Simulation duration in seconds.
        duration: f64,
        /// Random seed for deterministic output.
        seed: u64,
    },
    /// Validate experiment YAML against EDD v2 schema
    Validate {
        /// Path to the experiment YAML file.
        experiment_path: PathBuf,
    },
    /// Verify reproducibility of an experiment
    Verify {
        /// Path to the experiment YAML file.
        experiment_path: PathBuf,
        /// Number of verification runs.
        runs: usize,
    },
    /// Check EMC compliance
    EmcCheck {
        /// Path to the experiment YAML file.
        experiment_path: PathBuf,
    },
    /// Validate an EMC YAML file against EDD v2 EMC schema
    EmcValidate {
        /// Path to the EMC file.
        emc_path: PathBuf,
    },
    /// List available EMCs in the library
    ListEmc,
    /// Show help
    Help,
    /// Show version
    Version,
    /// Parse error (missing args, unknown command)
    Error(String),
}

/// SVG render output format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderFormat {
    /// One SVG file per frame.
    SvgFrames,
    /// One template SVG + keyframes JSON.
    SvgKeyframes,
}

impl Args {
    /// Parse command-line arguments from an iterator.
    ///
    /// This method is testable as it accepts any iterator of strings,
    /// not just `std::env::args()`.
    #[must_use]
    pub fn parse_from<I, S>(args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let args: Vec<String> = args.into_iter().map(|s| s.as_ref().to_string()).collect();
        Self::parse_from_vec(&args)
    }

    /// Parse command-line arguments from the environment.
    #[must_use]
    pub fn parse() -> Self {
        Self::parse_from(std::env::args())
    }

    /// Internal parsing from a vector of strings.
    fn parse_from_vec(args: &[String]) -> Self {
        if args.len() < 2 {
            return Self {
                command: Command::Help,
            };
        }

        let command = match args[1].as_str() {
            "run" => Self::parse_run_command(args),
            "render" => Self::parse_render_command(args),
            "validate" => Self::parse_validate_command(args),
            "verify" => Self::parse_verify_command(args),
            "emc-check" => Self::parse_emc_check_command(args),
            "emc-validate" => Self::parse_emc_validate_command(args),
            "list-emc" => Command::ListEmc,
            "-h" | "--help" | "help" => Command::Help,
            "-V" | "--version" | "version" => Command::Version,
            unknown => Command::Error(format!("Unknown command: {unknown}")),
        };

        Self { command }
    }

    /// Check if a positional arg is a help flag.
    fn is_help_flag(arg: &str) -> bool {
        arg == "--help" || arg == "-h"
    }

    /// Parse the 'run' command arguments.
    fn parse_run_command(args: &[String]) -> Command {
        if args.len() < 3 || Self::is_help_flag(&args[2]) {
            return Command::Help;
        }

        let mut seed_override = None;
        let mut verbose = false;

        let mut i = 3;
        while i < args.len() {
            match args[i].as_str() {
                "--seed" => {
                    if i + 1 < args.len() {
                        if let Ok(seed) = args[i + 1].parse() {
                            seed_override = Some(seed);
                        }
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                "-v" | "--verbose" => {
                    verbose = true;
                    i += 1;
                }
                _ => i += 1,
            }
        }

        Command::Run {
            experiment_path: PathBuf::from(&args[2]),
            seed_override,
            verbose,
        }
    }

    /// Parse the 'validate' command arguments.
    fn parse_validate_command(args: &[String]) -> Command {
        if args.len() < 3 || Self::is_help_flag(&args[2]) {
            return Command::Help;
        }

        Command::Validate {
            experiment_path: PathBuf::from(&args[2]),
        }
    }

    /// Parse the 'verify' command arguments.
    fn parse_verify_command(args: &[String]) -> Command {
        if args.len() < 3 || Self::is_help_flag(&args[2]) {
            return Command::Help;
        }

        let mut runs = 3;
        if args.len() > 3 && args[3] == "--runs" && args.len() > 4 {
            if let Ok(n) = args[4].parse() {
                runs = n;
            }
        }

        Command::Verify {
            experiment_path: PathBuf::from(&args[2]),
            runs,
        }
    }

    /// Parse the 'emc-check' command arguments.
    fn parse_emc_check_command(args: &[String]) -> Command {
        if args.len() < 3 || Self::is_help_flag(&args[2]) {
            return Command::Help;
        }

        Command::EmcCheck {
            experiment_path: PathBuf::from(&args[2]),
        }
    }

    /// Parse the 'emc-validate' command arguments.
    fn parse_emc_validate_command(args: &[String]) -> Command {
        if args.len() < 3 || Self::is_help_flag(&args[2]) {
            return Command::Help;
        }

        Command::EmcValidate {
            emc_path: PathBuf::from(&args[2]),
        }
    }

    /// Collect all `--key value` pairs from args starting at position `start`.
    fn collect_flags(args: &[String], start: usize) -> std::collections::HashMap<String, String> {
        let mut flags = std::collections::HashMap::new();
        let mut i = start;
        while i < args.len() {
            if args[i].starts_with("--") && i + 1 < args.len() {
                flags.insert(args[i].clone(), args[i + 1].clone());
                i += 2;
            } else {
                i += 1;
            }
        }
        flags
    }

    /// Parse the 'render' command arguments.
    fn parse_render_command(args: &[String]) -> Command {
        if args.len() >= 3 && Self::is_help_flag(&args[2]) {
            return Command::Help;
        }
        let flags = Self::collect_flags(args, 2);

        let domain = flags
            .get("--domain")
            .cloned()
            .unwrap_or_else(|| "orbit".to_string());
        let format = match flags.get("--format").map(String::as_str) {
            Some("svg-frames") => RenderFormat::SvgFrames,
            _ => RenderFormat::SvgKeyframes,
        };
        let output = flags
            .get("--output")
            .map_or_else(|| PathBuf::from("."), PathBuf::from);
        let fps = flags
            .get("--fps")
            .and_then(|v| v.parse().ok())
            .unwrap_or(60);
        let duration = flags
            .get("--duration")
            .and_then(|v| v.parse().ok())
            .unwrap_or(10.0);
        let seed = flags
            .get("--seed")
            .and_then(|v| v.parse().ok())
            .unwrap_or(42);

        Command::Render {
            domain,
            format,
            output,
            fps,
            duration,
            seed,
        }
    }
}
