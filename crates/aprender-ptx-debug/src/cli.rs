//! Declarative CLI definition for the `aprender-ptx-debug` binary.
//!
//! The parser lives in the library rather than in `src/bin/main.rs` so that
//! integration tests can exercise it directly, matching the house pattern used
//! by the other CLI crates in this workspace.
//!
//! Hand-rolled `match args[1]` dispatch is banned here: unknown flags fall
//! through catch-all arms, a valued flag given without a value gets discarded,
//! and an unparseable value degrades into a default instead of an error. clap
//! derive makes each of those a hard parse failure.

use clap::error::ErrorKind;
use clap::{Args, CommandFactory, Parser, Subcommand};

/// Trailing help text, preserved verbatim from the original usage banner.
const AFTER_HELP: &str = "EXIT CODES:
    0 - Analysis passed (score >= 90)
    1 - Analysis passed with warnings (score 70-89)
    2 - Analysis failed (score < 70)
    3 - Critical bugs detected
    10 - Parse error
    11 - I/O error

EXAMPLES:
    aprender-ptx-debug analyze kernel.ptx --falsify
    aprender-ptx-debug analyze kernel.ptx --min-score 90 --html report.html
    aprender-ptx-debug gen-fkr kernel.ptx -o tests/kernel_fkr.rs";

/// Top-level command line for `aprender-ptx-debug`.
#[derive(Debug, Parser)]
#[command(
    name = "aprender-ptx-debug",
    about = "Pure Rust PTX debugging and static analysis tool",
    version,
    subcommand_required = true,
    arg_required_else_help = true,
    after_help = AFTER_HELP
)]
pub struct Cli {
    /// Subcommand to execute.
    #[command(subcommand)]
    pub command: Command,
}

/// Available subcommands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Analyze PTX file for bugs and issues
    Analyze(AnalyzeArgs),

    /// Generate FKR tests for jugar-probar
    #[command(name = "gen-fkr")]
    GenFkr(GenFkrArgs),

    /// Show version information
    Version,
}

/// Arguments for the `analyze` subcommand.
#[derive(Debug, Args)]
pub struct AnalyzeArgs {
    /// PTX file to analyze
    #[arg(value_name = "FILE")]
    pub file: String,

    /// Run full 100-point falsification framework.
    ///
    /// The framework is always evaluated by `analyze`, so this flag is accepted
    /// for backwards compatibility and does not currently change the output.
    #[arg(long)]
    pub falsify: bool,

    /// Fail if score < N
    #[arg(long = "min-score", value_name = "N", default_value_t = 70.0)]
    pub min_score: f64,

    /// Write HTML report to file
    #[arg(long, value_name = "FILE")]
    pub html: Option<String>,

    /// Output JSON format
    #[arg(long)]
    pub json: bool,
}

/// Arguments for the `gen-fkr` subcommand.
#[derive(Debug, Args)]
pub struct GenFkrArgs {
    /// PTX file to generate tests from
    #[arg(value_name = "FILE")]
    pub file: String,

    /// Output file (default: stdout)
    #[arg(short = 'o', value_name = "FILE")]
    pub output: Option<String>,
}

/// Render the version string used by both `--version` and the `version`
/// subcommand, so the two surfaces cannot drift apart.
#[must_use]
pub fn version_string() -> String {
    Cli::command().render_version()
}

/// Map a clap parse failure onto the process exit code.
///
/// `--help` and `--version` are reported by clap as errors but are successful
/// invocations. Every other parse failure exits 1, preserving the exit status
/// the hand-rolled parser used for an unknown command, a missing argument, or a
/// bad option value.
#[must_use]
pub fn exit_code_for_parse_error(err: &clap::Error) -> i32 {
    match err.kind() {
        ErrorKind::DisplayHelp | ErrorKind::DisplayVersion => 0,
        _ => 1,
    }
}
