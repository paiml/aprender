//! `apr` — the Aprender CLI.
//!
//! `cargo install aprender` installs this binary.
//! All logic lives in the `apr-cli` workspace crate (internal library).

fn main() -> std::process::ExitCode {
    apr_cli::cli_main()
}
