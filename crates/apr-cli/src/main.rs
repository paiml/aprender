//! apr - APR Model Operations CLI
//!
//! Entry point shim. See lib.rs for implementation.

use apr_cli::{execute_command, Cli};
use clap::Parser;
use colored::control;
use std::process::ExitCode;

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

/// GH-667: Reset SIGPIPE to SIG_DFL so piping to head/less exits cleanly.
#[cfg(unix)]
#[allow(unsafe_code)]
fn reset_sigpipe() {
    // SAFETY: signal(SIGPIPE, SIG_DFL) is async-signal-safe per POSIX.
    // Called once at program start before any threads are spawned.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
}

#[cfg(not(unix))]
fn reset_sigpipe() {}

fn main() -> ExitCode {
    // GH-667: Reset SIGPIPE to default so piping to head/less doesn't panic.
    // Rust sets SIG_IGN for SIGPIPE, causing panics on write to closed pipe.
    reset_sigpipe();

    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();
    // GH-662: Respect NO_COLOR env var and non-TTY output.
    // The `colored` crate's auto-detect doesn't reliably work in all environments.
    let no_color = std::env::var("NO_COLOR").is_ok();
    let is_tty = std::io::IsTerminal::is_terminal(&std::io::stdout());
    if no_color || !is_tty {
        control::set_override(false);
    }
    let cli = Cli::parse();
    match execute_command(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            e.exit_code()
        }
    }
}
