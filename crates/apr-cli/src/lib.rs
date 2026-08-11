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

// #2401: `--quiet` / `--verbose` control. MUST come before `mod commands;` —
// it shadows `println!`/`print!` for every module declared after it, which is
// how the two global flags reach ~9 000 call sites without any command having
// to remember to forward a parameter.
#[macro_use]
#[allow(unused_macros)]
pub mod verbosity;

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

// HELIX-IDEA-009: API key auth re-exported so integration tests in
// `tests/falsify_auth_*.rs` can construct gates and middleware without
// reaching into the private `commands` tree.
pub mod serve_auth {
    #[cfg(feature = "inference")]
    pub use crate::commands::serve::auth::layer;
    pub use crate::commands::serve::auth::{apply, AuthGate};
}

// PMAT-923: e2e seam so `tests/ollama_api_serve_compat.rs` can build the REAL
// `apr serve` APR-CPU router (the one mounted for a `.apr` model) and prove the
// Ollama `/api/chat` + `/api/generate` routes are wired — not realizar's
// `create_router`.
#[cfg(feature = "inference")]
pub mod serve_test_support {
    pub use crate::commands::serve::handlers::build_demo_apr_cpu_router_for_test;
    // PMAT-928: streaming-capable demo router whose NDJSON path is driven by a
    // scripted token sequence, so the streaming falsifier observes real
    // multi-chunk NDJSON without loading a model.
    pub use crate::commands::serve::handlers::build_demo_streaming_apr_cpu_router_for_test;
}

#[cfg(feature = "inference")]
pub mod federation;

// Commands are crate-private, used internally by execute_command
use commands::{
    bench, canary, canary::CanaryCommands, cbtop, chat, compare_hf, compile, convert, data, debug,
    diagnose, diff, distill, eval, explain, export, flow, hex, import, inspect, lint, mcp, merge,
    oracle, pipeline, probar, profile, prune, publish, pull, qa, qualify, quantize, rosetta,
    rosetta::RosettaCommands, run, serve, showcase, stamp, tensors, tokenize, trace, tree, tui,
    validate, validate_manifest,
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

    /// Verbose output: dispatch resolution, plus per-command detail where there is more to show
    #[arg(short, long, global = true)]
    pub verbose: bool,

    /// Quiet mode: suppress stdout (errors still go to stderr; --json still prints)
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

    // FALSIFY-GPUTRAIN-007 / INV-GPUTRAIN-007 — `apr --version --json` MUST
    // emit a machine-parseable object containing at least the three keys
    // { cuda_feature, cuda_runtime_available, visible_devices }. Intercept
    // this flag combo BEFORE clap's default `--version` handler exits the
    // process with a plain string. Bound by `gputrain_007.rs` — see
    // `AC_GPUTRAIN_007_REQUIRED_VERSION_JSON_KEYS`.
    let raw: Vec<String> = std::env::args().collect();
    if raw.iter().any(|a| a == "--version") && raw.iter().any(|a| a == "--json") {
        emit_version_json();
        return std::process::ExitCode::SUCCESS;
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

/// FALSIFY-GPUTRAIN-007 — emit `apr --version --json` output with the
/// 3-key schema required by `AC_GPUTRAIN_007_REQUIRED_VERSION_JSON_KEYS`.
///
/// Schema:
/// ```json
/// {
///   "name":    "apr",
///   "version": "<semver>",
///   "git_sha": "<commit>",
///   "cuda_feature":           <bool>,   // was the binary built --features cuda?
///   "cuda_runtime_available": <bool>,   // does cudaRuntimeGetVersion succeed?
///   "visible_devices":        ["0", "1", ...]  // nvidia-smi -L indices, empty if no runtime
/// }
/// ```
///
/// Consumers (`entrenar::train::gputrain_007::verdict_from_version_json_keys`
/// and `verdict_from_version_json_fields`) parse this and assert schema
/// completeness + field invariants (`visible_devices.len() <= 16`, no
/// `cuda_feature && !cuda_runtime_available` inconsistency).
pub fn emit_version_json() {
    let cuda_feature = cfg!(feature = "cuda");

    // cuda_runtime_available: try nvidia-smi -L. Present-and-exits-0 ⇒ true.
    // This matches how gputrain_003 queries nvidia-smi — keep the probe
    // consistent with the residency check.
    let cuda_runtime_available = std::process::Command::new("nvidia-smi")
        .arg("-L")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    // visible_devices: if the runtime is available, parse nvidia-smi -L
    // output (one GPU per line, "GPU 0: ...", "GPU 1: ..."). Emit the
    // index strings to match INV-GPUTRAIN-001 grammar (:0..:15).
    let visible_devices: Vec<String> = if cuda_runtime_available {
        std::process::Command::new("nvidia-smi")
            .arg("-L")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| {
                s.lines()
                    .filter_map(|line| {
                        // Expect "GPU <idx>: <name> (UUID: <uuid>)"
                        line.strip_prefix("GPU ").and_then(|rest| {
                            rest.split_once(':').map(|(idx, _)| idx.trim().to_string())
                        })
                    })
                    .collect()
            })
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    let body = serde_json::json!({
        "name": "apr",
        "version": env!("CARGO_PKG_VERSION"),
        "git_sha": env!("APR_GIT_SHA"),
        "cuda_feature": cuda_feature,
        "cuda_runtime_available": cuda_runtime_available,
        "visible_devices": visible_devices,
    });

    // Emit pretty-printed JSON on stdout so it's grep-friendly and
    // round-trippable through `| jq .cuda_feature`.
    println!(
        "{}",
        serde_json::to_string_pretty(&body).expect("build version json")
    );
}
