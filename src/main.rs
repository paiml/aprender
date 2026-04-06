//! Renacer CLI entry point - thin wrapper around library
//!
//! Renacer uses ptrace, which is Linux-only.

#[cfg(not(target_os = "linux"))]
compile_error!("renacer requires Linux (ptrace syscall tracing)");

#[cfg(target_os = "linux")]
fn main() {
    match renacer::cli::run() {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }
}
