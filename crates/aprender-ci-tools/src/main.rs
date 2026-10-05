//! `aprender-ci-tools`: one binary, one subcommand per ported Python helper.

use aprender_ci_tools::{
    coverage_report_scope, dag_status, git_patch_id, package_include_diff, privscan,
    publishable_crates, tarball_build_errors, tarball_shrink_report, tarball_workspace,
};
use clap::{ArgGroup, Parser, Subcommand};
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
    /// scripts/coverage_report_scope.py, deleted once callers moved here).
    CoverageReportScope {
        /// A workspace member to leave out; repeatable. Must name a member.
        #[arg(long, value_name = "NAME", allow_hyphen_values = true)]
        exclude: Vec<String>,
    },
    /// What the published tarballs do NOT test: dropped integration targets and run-time
    /// `*_or_skip(` sites (was scripts/lib/tarball_shrink_report.py). Exit 2 on a missing input.
    TarballShrinkReport {
        /// `cargo package` output.
        package_log: PathBuf,
        /// The tarball workspace; its crates are under `pkgs/`.
        ws_dir: PathBuf,
    },
    /// Turn DIR/pkgs/<name>-<ver>/ (unpacked .crate files) into one workspace and print
    /// `<name>\t<version>\t<dir>` per crate (was scripts/lib/tarball_workspace.py).
    #[command(group(ArgGroup::new("mode").required(true).args(["dir", "name", "target_dir"])))]
    TarballWorkspace {
        /// Write DIR/Cargo.toml over every crate under DIR/pkgs.
        dir: Option<String>,
        /// Print the [package] name of the unpacked crate in this directory.
        #[arg(long, value_name = "DIR")]
        name: Option<String>,
        /// Print `target_directory` of the `cargo metadata` JSON on stdin.
        #[arg(long)]
        target_dir: bool,
    },
    /// Attribute a tarball-workspace build's errors to the crates that own them (was
    /// scripts/lib/tarball_build_errors.py). Exit 0 no error, 1 a crate is RED, 2 bad input,
    /// 3 only unowned errors, 4 the build host failed.
    TarballBuildErrors {
        /// The `cargo build --message-format short` log. Exactly one, else exit 2 as the
        /// original, so the count is checked by the port and not by clap (which exits 1).
        #[arg(num_args = 0.., allow_hyphen_values = true, trailing_var_arg = true)]
        log: Vec<PathBuf>,
    },
    /// Each DAG row's status, derived from its receipt, and the D7 lines for a typed
    /// `status:` that disagrees (was scripts/lib/dag_status.py). Reads the rows as a JSON
    /// array of `[id, row]` pairs on stdin.
    DagStatus {
        /// The repository root the `docs/audits/` receipts live under.
        #[arg(long, value_name = "DIR")]
        root: String,
    },
    /// `<patch-id> <commit-oid>` per patch in the diff on stdin, byte-exact with
    /// `git patch-id` (was scripts/lib/git_patch_id.py). Exit 2 on an unknown mode.
    GitPatchId {
        /// --stable, --verbatim or --unstable (the default), as git spells them.
        #[arg(allow_hyphen_values = true, value_name = "MODE")]
        mode: Option<String>,
    },
    /// Flag lines that would leak private detail: `FILE:LINE:pattern-N` per hit, then
    /// `privscan: N hit(s)` (replaces the untracked per-worktree `privscan.sh`). Exit 0 no
    /// hit, 1 any hit, 2 a path that cannot be read or no path.
    Privscan {
        /// Files, or directories walked in name order.
        #[arg(num_args = 0..)]
        paths: Vec<PathBuf>,
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

/// `(already printed, exit code, reason)`.
type Refusal = (String, u8, String);

fn nothing_printed(reason: String) -> Refusal {
    (String::new(), 1, reason)
}

fn read(path: &PathBuf) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))
}

/// The output, or a refusal.
fn run(cmd: Cmd) -> Result<String, Refusal> {
    match cmd {
        Cmd::GitPatchId { mode } => {
            let mode = git_patch_id::Mode::parse(mode.as_deref()).ok_or_else(|| {
                (
                    String::new(),
                    2,
                    "usage: git_patch_id.py --stable|--verbatim|--unstable".to_owned(),
                )
            })?;
            let mut diff = Vec::new();
            std::io::stdin()
                .read_to_end(&mut diff)
                .map_err(|e| nothing_printed(format!("stdin: {e}")))?;
            Ok(git_patch_id::run(&diff, mode))
        }
        Cmd::PublishableCrates => {
            let mut meta = String::new();
            std::io::stdin()
                .read_to_string(&mut meta)
                .map_err(|e| nothing_printed(format!("stdin: {e}")))?;
            publishable_crates::run(&meta).map_err(|(printed, reason)| (printed, 1, reason))
        }
        Cmd::PackageIncludeDiff { listing, includes } => {
            let listing = read(&listing).map_err(nothing_printed)?;
            let includes = read(&includes).map_err(nothing_printed)?;
            Ok(package_include_diff::diff(&listing, &includes))
        }
        Cmd::CoverageReportScope { exclude } => {
            let meta = cargo_metadata().map_err(nothing_printed)?;
            coverage_report_scope::scope(&meta, &exclude).map_err(nothing_printed)
        }
        Cmd::TarballShrinkReport {
            package_log,
            ws_dir,
        } => tarball_shrink_report::report(&package_log, &ws_dir)
            .map_err(|(code, reason)| (String::new(), code, reason)),
        Cmd::TarballWorkspace {
            target_dir: true, ..
        } => {
            let mut meta = Vec::new();
            std::io::stdin()
                .read_to_end(&mut meta)
                .map_err(|e| nothing_printed(format!("stdin: {e}")))?;
            tarball_workspace::target_dir(&meta).map_err(nothing_printed)
        }
        Cmd::TarballWorkspace {
            name: Some(dir), ..
        } => tarball_workspace::package_name(&dir).map_err(nothing_printed),
        Cmd::TarballWorkspace { dir, .. } => {
            // The required `mode` group leaves DIR as the only other way in.
            let dir = dir.ok_or_else(|| nothing_printed("tarball-workspace: no DIR".to_owned()))?;
            tarball_workspace::write_workspace(&dir).map_err(nothing_printed)
        }
        Cmd::TarballBuildErrors { log } => {
            let o = tarball_build_errors::run(&log);
            if o.code == 0 {
                Ok(o.stdout)
            } else {
                Err((o.stdout, o.code, o.stderr))
            }
        }
        Cmd::DagStatus { root } => {
            let mut rows = String::new();
            std::io::stdin()
                .read_to_string(&mut rows)
                .map_err(|e| nothing_printed(format!("stdin: {e}")))?;
            dag_status::run(&root, &rows).map_err(nothing_printed)
        }
        Cmd::Privscan { paths } => {
            if paths.is_empty() {
                return Err((
                    String::new(),
                    2,
                    "usage: aprender-ci-tools privscan PATH...".to_owned(),
                ));
            }
            match privscan::run(&paths) {
                Err(reason) => Err((String::new(), 2, reason)),
                Ok((out, 0)) => Ok(out),
                Ok((out, hits)) => Err((out, 1, format!("privscan: exit 1 on {hits} hit(s)"))),
            }
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
    let (out, refusal) = match run(cli.cmd) {
        Ok(out) => (out, None),
        Err((printed, code, reason)) => (printed, Some((code, reason))),
    };
    let mut stdout = std::io::stdout().lock();
    if let Err(e) = stdout
        .write_all(out.as_bytes())
        .and_then(|()| stdout.flush())
    {
        eprintln!("aprender-ci-tools: stdout: {e}");
        return ExitCode::FAILURE;
    }
    match refusal {
        None => ExitCode::SUCCESS,
        Some((code, reason)) => {
            eprintln!("{reason}");
            ExitCode::from(code)
        }
    }
}
