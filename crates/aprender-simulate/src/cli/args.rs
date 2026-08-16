//! CLI argument parsing.
//!
//! The accepted grammar is **declarative**: it is the `Cli` and `Commands`
//! types below, parsed by clap derive. Nothing in this module inspects `argv`
//! by hand.
//!
//! ## Why declarative
//!
//! This module previously hand-rolled a `match args[1]` parser, and every one of
//! its failure modes was silent — the command exited 0 and did the wrong thing:
//!
//! - `--seed notanumber` used `.parse().ok().unwrap_or(default)`, so a typo
//!   became the DEFAULT seed rather than an error. A simulation is only
//!   reproducible if the seed it reports is the seed you asked for.
//! - `--seed` with no value was discarded by the `else { i += 1 }` arm.
//! - An unknown flag fell through a `_ => i += 1` catch-all and vanished.
//! - `verify --runs N` was only honoured when `--runs` sat at exactly `argv[3]`.
//!
//! clap rejects all four. The grammar is data, not control flow, so it cannot
//! drift out of sync with itself the way the hand-rolled arms did. Enforced
//! repo-wide by `scripts/check_no_hand_rolled_parsers.sh`.

use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

/// CLI arguments container.
#[derive(Debug, Clone, PartialEq, Parser)]
#[command(
    name = "simular",
    version,
    about = "Unified Simulation Engine for the Sovereign AI Stack",
    // `help` and `version` are real subcommands below, so that `simular help`
    // keeps printing simular's own help text (see `output::print_help`) rather
    // than clap's auto-generated one. The `-h`/`--help` and `-V`/`--version`
    // FLAGS are still clap's.
    disable_help_subcommand = true
)]
pub struct Cli {
    /// The command to execute. `None` means no subcommand was given at all.
    #[command(subcommand)]
    pub command: Option<Commands>,
}

impl Cli {
    /// The selected subcommand, defaulting to [`Commands::Help`].
    ///
    /// An empty `argv` showed the help text before the clap conversion; that
    /// behaviour (help on stdout, exit 0) is preserved here rather than in
    /// clap's `arg_required_else_help`, which would exit 2 instead.
    #[must_use]
    pub fn into_command(self) -> Commands {
        self.command.unwrap_or(Commands::Help)
    }
}

/// Available CLI commands.
#[derive(Debug, Clone, PartialEq, Subcommand)]
pub enum Commands {
    /// Run an experiment
    Run {
        /// Path to the experiment YAML file.
        experiment_path: PathBuf,
        /// Optional seed override.
        #[arg(long = "seed", value_name = "N")]
        seed_override: Option<u64>,
        /// Enable verbose output.
        #[arg(short = 'v', long)]
        verbose: bool,
    },
    /// Render simulation to SVG + keyframes
    Render {
        /// Simulation domain (orbit, `bouncing_balls`).
        #[arg(long, default_value = "orbit")]
        domain: String,
        /// Output format: svg-frames or svg-keyframes.
        #[arg(long, value_enum, default_value = "svg-keyframes")]
        format: RenderFormat,
        /// Output directory.
        #[arg(long, default_value = ".")]
        output: PathBuf,
        /// Frames per second.
        #[arg(long, default_value_t = 60)]
        fps: u32,
        /// Simulation duration in seconds.
        #[arg(long, default_value_t = 10.0)]
        duration: f64,
        /// Random seed for deterministic output.
        #[arg(long, default_value_t = 42)]
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
        #[arg(long, default_value_t = 3, value_name = "N")]
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
}

/// SVG render output format.
#[derive(Debug, Clone, PartialEq, Eq, ValueEnum)]
pub enum RenderFormat {
    /// One SVG file per frame.
    SvgFrames,
    /// One template SVG + keyframes JSON.
    SvgKeyframes,
}
