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

/// GH-646: Clear FPCR.FZ16 on aarch64 so f16 subnormal scales aren't flushed to zero.
/// The ARM `fcvt s, h` instruction (used by the `half` crate) respects FPCR.FZ16.
/// When set (e.g., by NVIDIA's CUDA runtime on Jetson), subnormal f16 scale values
/// in Q6_K super-blocks are flushed to 0.0, causing `d * scale * q = 0` for all elements.
/// This produces all-zero tensors that pass on x86_64 but fail on aarch64.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
fn clear_fz16() {
    // SAFETY: Reading/writing FPCR is safe — it only affects floating-point behavior
    // for the current thread. Called once at program start before any FP operations.
    unsafe {
        let fpcr: u64;
        core::arch::asm!("mrs {}, fpcr", out(reg) fpcr);
        if fpcr & (1 << 19) != 0 {
            // FZ16 is set — clear it
            let new_fpcr = fpcr & !(1 << 19);
            core::arch::asm!("msr fpcr, {}", in(reg) new_fpcr);
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn clear_fz16() {}

fn main() -> ExitCode {
    // GH-667: Reset SIGPIPE to default so piping to head/less doesn't panic.
    reset_sigpipe();
    // GH-646: Clear FPCR.FZ16 on ARM so f16 subnormals dequantize correctly.
    clear_fz16();

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
