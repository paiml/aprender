//! apr-cli library
//!
//! This library is the foundation for the apr CLI binary.
//! Exports CLI structures for testing and reuse.

use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};

mod commands;
pub mod error;
mod output;
pub mod pipe;

pub use error::CliError;

// Public re-exports for integration tests
pub mod qa_types {
    pub use crate::commands::qa::{GateResult, QaReport, SystemInfo};
}

// Public re-exports for downstream crates (whisper-apr proxies these)
pub mod model_pull {
    pub use crate::commands::pull::{list, run};
}

#[cfg(feature = "inference")]
pub mod federation;

// Commands are crate-private, used internally by execute_command
use commands::{
    bench, canary, canary::CanaryCommands, cbtop, chat, compare_hf, convert, debug, diff, distill,
    eval, explain, export, finetune, flow, hex, import, inspect, lint, merge, oracle, probar,
    profile, prune, ptx_explain, publish, pull, qa, qualify, quantize, rosetta,
    rosetta::RosettaCommands, run, serve, showcase, tensors, trace, tree, tui, tune, validate,
};

/// apr - APR Model Operations Tool
///
/// Inspect, debug, and manage .apr model files.
/// Toyota Way: Genchi Genbutsu - Go and see the actual data.
#[derive(Parser, Debug)]
#[command(name = "apr")]
#[command(author, version = concat!(env!("CARGO_PKG_VERSION"), " (", env!("APR_GIT_SHA"), ")"), about, long_about = None)]
#[command(propagate_version = true)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Box<Commands>,

    /// Output as JSON
    #[arg(long, global = true)]
    pub json: bool,

    /// Verbose output
    #[arg(short, long, global = true)]
    pub verbose: bool,

    /// Quiet mode (errors only)
    #[arg(short, long, global = true)]
    pub quiet: bool,

    /// Disable network access (Sovereign AI compliance, Section 9)
    #[arg(long, global = true)]
    pub offline: bool,

    /// Skip tensor contract validation (PMAT-237: use with diagnostic tooling)
    #[arg(long, global = true)]
    pub skip_contract: bool,
}

include!("commands_enum.rs");
include!("model_ops_commands.rs");
include!("extended_commands.rs");
include!("tool_commands.rs");
include!("validate.rs");
include!("dispatch_run.rs");
include!("dispatch.rs");
include!("dispatch_analysis.rs");
include!("lib_07.rs");
