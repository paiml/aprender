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

/// The version line of `simular --version` and `simular version`.
///
/// The semver and the first 9 hex of the
/// commit it was built from (G0.1, #4476) — the semver is a workspace version shared
/// by every worktree, so without the sha a stale simular reads as HEAD.
pub const VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), " (", env!("APR_GIT_SHA"), ")");

/// CLI arguments container.
#[derive(Debug, Clone, PartialEq, Parser)]
#[command(
    name = "aprender-simulate",
    version = VERSION,
    about = "Unified Simulation Engine for the Sovereign AI Stack",
    // `help` and `version` are real subcommands below, so that `aprender-simulate help`
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
#[allow(clippy::large_enum_variant)] // PrFlow carries the measured-input overrides; parsed once at startup
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
        #[arg(long, default_value = "orbit", value_parser = ["orbit", "bouncing_balls"])]
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
    /// Run the aprender PR-flow scenario (FLOW-005) for a named design
    PrFlow {
        /// Design: D, A, G or E.
        #[arg(long, default_value = "D", value_parser = ["D", "A", "G", "E"])]
        design: String,
        /// Print the cycle-record "sim" block as one JSON line.
        #[arg(long)]
        json: bool,
        /// Override the PR-check runner pool (measured input).
        #[arg(long)]
        r_pr: Option<usize>,
        /// Override the merge-queue runner pool (measured input).
        #[arg(long)]
        r_mq: Option<usize>,
        /// Override the WIP cap (measured input).
        #[arg(long)]
        cap: Option<usize>,
        /// Override concurrent merge-queue groups (measured input).
        #[arg(long)]
        builds: Option<usize>,
        /// Override the merge-group flake rate (measured input).
        #[arg(long)]
        f: Option<f64>,
        /// Override mean concurrent authors (may be fractional).
        #[arg(long)]
        n: Option<f64>,
        /// Measured PR-check runner-minutes per run.
        #[arg(long)]
        pr_min: Option<f64>,
        /// Measured merge-queue runner-minutes per run.
        #[arg(long)]
        mq_min: Option<f64>,
        /// Measured share of PR runs that go red.
        #[arg(long)]
        p_red: Option<f64>,
        /// Measured queue wait per PR run, minutes.
        #[arg(long)]
        pr_setup: Option<f64>,
        /// Review wait per PR after green checks, minutes (default 0).
        #[arg(long)]
        t_rev: Option<f64>,
        /// Run the review alongside the checks instead of after them.
        #[arg(long)]
        rev_concurrent: bool,
        /// Parallel reviewers (0 = unlimited).
        #[arg(long)]
        rev_lanes: Option<usize>,
        /// Merged PRs in the last 24 h, from git (for drift_pct_24h).
        #[arg(long)]
        observed_24h: Option<f64>,
        /// Comma-separated run ids the measured inputs came from.
        #[arg(long)]
        src_runs: Option<String>,
        /// Arbiter params file for one repo (ARB-SOT-001 §3); prints {actual, one_change, optimum}.
        #[arg(long)]
        params: Option<PathBuf>,
        /// Seeds 1..=N for `--params`.
        #[arg(long, default_value_t = 5)]
        seeds: u64,
        /// Simulated hours per seed for `--params`.
        #[arg(long, default_value_t = 400.0)]
        hours: f64,
        /// Warm-up hours discarded for `--params`.
        #[arg(long, default_value_t = 20.0)]
        warmup: f64,
        /// Commit the sim block is stamped with.
        #[arg(long, default_value = "unknown")]
        sha: String,
    },
    /// Run a FLOW-006 repo case (conflict step, optional merge queue); --json prints the sim block
    RepoFlow {
        /// Case name, for example rmedia-design or infra-rule.
        #[arg(long)]
        case: String,
        /// JSON file of measured inputs from the ledger; overrides the case.
        #[arg(long)]
        inputs: Option<std::path::PathBuf>,
        /// Print the sim block as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Run a FLOW-006 course-factory case; --json prints the sim block
    CourseFlow {
        /// Case: cal-today, design, nine-builders, coursera-24h or render-300.
        #[arg(long)]
        case: String,
        /// JSON file of measured inputs from the ledger; overrides the case.
        #[arg(long)]
        inputs: Option<std::path::PathBuf>,
        /// Print the sim block as JSON.
        #[arg(long)]
        json: bool,
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
