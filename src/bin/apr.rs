//! `apr` — the Aprender CLI.
//!
//! `cargo install aprender` installs this binary.
//! All logic lives in the `apr-cli` workspace crate (internal library).

// PMAT-124: enforce `// SAFETY:` on every unsafe block in the `apr` binary.
#![deny(clippy::undocumented_unsafe_blocks)]

fn main() -> std::process::ExitCode {
    apr_cli::cli_main()
}
