//! apr-cli library
//!
//! This library is the foundation for the apr CLI binary.
//! Exports CLI structures for testing and reuse.

// APR-MONO: Clippy pedantic allows for monorepo transition.
// unwrap() eliminated (524 → expect()). Style lints from 20 merged crates
// are suppressed at crate level. Will be incrementally addressed.
#![allow(clippy::all, clippy::pedantic, clippy::disallowed_methods)]
#![allow(
    unreachable_code,
    unused_variables,
    unused_imports,
    dead_code,
    unused_assignments
)]

use clap::{Parser, Subcommand, ValueEnum};
use std::path::{Path, PathBuf};

// Contract assertions from YAML (pv codegen)
#[macro_use]
#[allow(unused_macros, clippy::duplicated_attributes)]
mod generated_contracts;

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
    bench, canary, canary::CanaryCommands, cbtop, chat, compare_hf, compile, convert, data, debug,
    diagnose, diff, distill, eval, explain, export, flow, hex, import, inspect, lint, mcp, merge,
    oracle, pipeline, probar, profile, prune, publish, pull, qa, qualify, quantize, rosetta,
    rosetta::RosettaCommands, run, serve, showcase, tensors, tokenize, trace, tree, tui, validate,
    validate_manifest,
};
#[cfg(feature = "training")]
use commands::{finetune, gpu, train, tune};

#[cfg(feature = "training")]
pub use commands::pretrain::PretrainMode;

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
include!("data_commands.rs");
#[cfg(feature = "training")]
include!("train_commands.rs");
include!("serve_commands.rs");
include!("tokenize_commands.rs");
include!("pipeline_commands.rs");
include!("validate.rs");
include!("dispatch_run.rs");
include!("dispatch.rs");
include!("dispatch_analysis.rs");
include!("lib_07.rs");

/// Full CLI entry point for `cargo install aprender`.
///
/// This function encapsulates the complete `apr` binary logic so that
/// the `aprender` facade crate can produce the same binary via
/// `cargo install aprender` (in addition to `cargo install apr-cli`).
pub fn cli_main() -> std::process::ExitCode {
    // GH-667: Reset SIGPIPE to default so piping to head/less doesn't panic.
    #[cfg(unix)]
    #[allow(unsafe_code)]
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }

    // GH-646: Clear FPCR.FZ16 on aarch64 so f16 subnormals work.
    #[cfg(target_arch = "aarch64")]
    #[allow(unsafe_code)]
    unsafe {
        let fpcr: u64;
        core::arch::asm!("mrs {}, fpcr", out(reg) fpcr);
        if fpcr & (1 << 19) != 0 {
            let new_fpcr = fpcr & !(1 << 19);
            core::arch::asm!("msr fpcr, {}", in(reg) new_fpcr);
        }
    }

    // GH-662: Respect NO_COLOR env var and non-TTY output.
    let no_color = std::env::var("NO_COLOR").is_ok();
    let is_tty = std::io::IsTerminal::is_terminal(&std::io::stdout());
    if no_color || !is_tty {
        colored::control::set_override(false);
    }

    // GATE-GPUTRAIN-006 / INV-GPUTRAIN-007: `apr --version --json` must
    // report structured cuda wiring state. clap's `--version` exits
    // before our global `--json` flag is observed, so we peek raw args
    // first. Only the combined `--version --json` form intercepts —
    // plain `--version` falls through to clap unchanged.
    if commands::version::should_emit_version_json(std::env::args().skip(1)) {
        let report = commands::version::collect_version_report();
        match commands::version::emit_version_json(&report) {
            Ok(()) => return std::process::ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                return std::process::ExitCode::FAILURE;
            }
        }
    }

    let cli = Cli::parse();
    match execute_command(&cli) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            e.exit_code()
        }
    }
}
