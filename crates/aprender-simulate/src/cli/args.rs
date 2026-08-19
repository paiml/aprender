//! CLI argument parsing.
//!
//! The grammar is **declarative**: one `#[derive(Parser)]` / `#[derive(Subcommand)]`
//! declaration is the single source of truth for parsing, `--help`, error
//! messages and shell completion, and `apr sim` embeds this same [`Command`]
//! rather than re-declaring it.
//!
//! It replaces a hand-rolled `match args[1].as_str()` walk over a
//! `Vec<String>`. That parser did not merely duplicate what clap does -- it
//! **silently discarded input**, which is disqualifying in a tool whose entire
//! claim is deterministic, reproducible simulation:
//!
//! * `--seed notanumber` parsed to `None` and the run proceeded on the default
//!   seed. A typo in the one flag that pins reproducibility was unobservable.
//! * `--seed` with no value was dropped the same silent way, as were bad
//!   `--runs`, `--fps` and `--duration` values.
//! * Unknown flags fell through `_ => i += 1`, so `--verbse` did nothing and
//!   said nothing.
//! * `verify --runs N` was only honoured when `--runs` sat at exactly argv[3];
//!   `verify exp.yaml -v --runs 5` silently used 3 runs.
//! * `render --format bogus` fell through to `SvgKeyframes` rather than failing.
//! * A `Command::Error(String)` variant turned a parse failure into a *value*,
//!   deferring the failure to whoever remembered to match on it.
//!
//! Every one of those is now a hard parse error, which is what `try_parse_from`
//! in the tests asserts.

use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

/// CLI arguments container.
#[derive(Parser, Debug, Clone, PartialEq)]
#[command(name = "simular")]
#[command(about = "Unified Simulation Engine for the Sovereign AI Stack", long_about = None)]
#[command(version)]
pub struct Args {
    /// The command to execute.
    #[command(subcommand)]
    pub command: Command,
}

/// Available CLI commands.
#[derive(Subcommand, Debug, Clone, PartialEq)]
pub enum Command {
    /// Run an experiment
    Run {
        /// Path to the experiment YAML file.
        experiment_path: PathBuf,
        /// Override the experiment seed.
        #[arg(long = "seed")]
        seed_override: Option<u64>,
        /// Enable verbose output.
        #[arg(short, long)]
        verbose: bool,
    },
    /// Render simulation to SVG + keyframes
    Render {
        /// Simulation domain (orbit, `monte_carlo`, optimization).
        #[arg(long, default_value = "orbit")]
        domain: String,
        /// Output format.
        #[arg(long, value_enum, default_value_t = RenderFormat::SvgKeyframes)]
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
        #[arg(long, default_value_t = 3)]
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
}

/// SVG render output format.
#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderFormat {
    /// One SVG file per frame.
    SvgFrames,
    /// One template SVG + keyframes JSON.
    SvgKeyframes,
}
