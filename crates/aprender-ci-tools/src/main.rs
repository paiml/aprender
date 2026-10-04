//! `aprender-ci-tools`: one binary, one subcommand per ported Python helper.

use aprender_ci_tools::{coverage_report_scope, package_include_diff, publishable_crates};
use clap::{Parser, Subcommand};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Command, ExitCode};

#[derive(Parser)]
#[command(
    name = "aprender-ci-tools",
    version = concat!(env!("CARGO_PKG_VERSION"), " (", env!("APR_GIT_SHA"), ")"),
    about = "CI and build helpers ported from scripts/**/*.py"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// `<name>\t<dir>` for every publishable package (was scripts/lib/publishable_crates.py).
    /// Reads `cargo metadata --no-deps --format-version 1` on stdin.
    PublishableCrates,
    /// Include rows whose target the package listing lacks (was
    /// scripts/lib/package_include_diff.py).
    PackageIncludeDiff {
        /// One packaged path per line (`cargo package --list`).
        listing: PathBuf,
        /// `<target>\t<including file>` per line.
        includes: PathBuf,
    },
    /// The `-p <crate>` list scoping `cargo llvm-cov report` (was
    /// scripts/coverage_report_scope.py).
    CoverageReportScope {
        /// A workspace member to leave out; repeatable. Must name a member.
        #[arg(long, value_name = "NAME", allow_hyphen_values = true)]
        exclude: Vec<String>,
    },
}

fn cargo_metadata() -> Result<String, String> {
    let out = Command::new("cargo")
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .output()
        .map_err(|e| format!("cannot run cargo metadata: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "cargo metadata failed ({}): {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    String::from_utf8(out.stdout).map_err(|e| format!("cargo metadata is not UTF-8: {e}"))
}

fn read(path: &PathBuf) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))
}

fn run(cmd: Cmd) -> Result<String, String> {
    match cmd {
        Cmd::PublishableCrates => {
            let mut meta = String::new();
            std::io::stdin()
                .read_to_string(&mut meta)
                .map_err(|e| format!("stdin: {e}"))?;
            publishable_crates::run(&meta)
        }
        Cmd::PackageIncludeDiff { listing, includes } => {
            let listing = read(&listing)?;
            let includes = read(&includes)?;
            Ok(package_include_diff::diff(&listing, &includes))
        }
        Cmd::CoverageReportScope { exclude } => {
            coverage_report_scope::scope(&cargo_metadata()?, &exclude)
        }
    }
}

fn main() -> ExitCode {
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        // --help/--version exit 0; a usage error exits 1, as the originals' did.
        Err(e) => {
            let code = u8::from(e.use_stderr());
            let _ = e.print();
            return ExitCode::from(code);
        }
    };
    match run(cli.cmd) {
        Ok(out) => {
            let mut stdout = std::io::stdout().lock();
            if let Err(e) = stdout
                .write_all(out.as_bytes())
                .and_then(|()| stdout.flush())
            {
                eprintln!("aprender-ci-tools: stdout: {e}");
                return ExitCode::FAILURE;
            }
            ExitCode::SUCCESS
        }
        Err(msg) => {
            eprintln!("{msg}");
            ExitCode::FAILURE
        }
    }
}
