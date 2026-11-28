//! `aprender-ci-tools`: one binary, one subcommand per ported Python helper.

use aprender_ci_tools::{
    coverage_report_scope, dag_status, git_patch_id, ladder_cells_produce, llama_fit_verdict,
    lockfile_registry_packages, nextest_fail_fast, package_include_diff, publishable_crates,
    spec_conformance_andon, spec_conformance_run, tarball_build_errors, tarball_shrink_report,
    tarball_workspace,
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
    /// The §12 expiry DAG: the DAG lines, D0-D4, D6 (an expired LAB row is a LAB line, not a
    /// D6) and D5 against `--derived` (was `check_dag` in scripts/lib/spec_conformance.py).
    /// Exit 0 clean, 1 any VIOLATION, 2 an unreadable input.
    SpecConformanceDag {
        /// The master spec (the §12 table is read from it).
        #[arg(long, value_name = "MD")]
        spec: PathBuf,
        /// The spec path the report names it by (the D5 message and the derived file's `source`).
        /// Default: `--spec` as given; CI passes the path relative to the repo root.
        #[arg(long, value_name = "REL")]
        source: Option<String>,
        /// The committed derived-expiry JSON to compare with (D5); absent means no D5 check.
        #[arg(long, value_name = "JSON")]
        derived: Option<PathBuf>,
        /// The date D6 compares against, `YYYY-MM-DD`. Default: `SPEC_CONFORMANCE_TODAY`, else today.
        #[arg(long, value_name = "YYYY-MM-DD")]
        today: Option<String>,
    },
    /// The whole spec-conformance scan of one repository root: the C rules (§6 and C5), the L
    /// rules (the ledger), then the D rules (the §12 DAG), as TSV in the Python's order (was
    /// scripts/lib/spec_conformance.py's `check` and `main`). Exit 0 clean, 1 any VIOLATION
    /// (with `--write`: 1 only if the derivation was not written), 2 an unreadable input.
    SpecConformance {
        /// The repository root.
        root: PathBuf,
        /// The master spec. Default: the one `docs/specifications/PP-LLAMA-001-MASTER*.md`.
        #[arg(long, value_name = "MD")]
        spec: Option<PathBuf>,
        /// The ledger. Default: `evidence/parity/LEDGER.md` under the root.
        #[arg(long, value_name = "MD")]
        ledger: Option<PathBuf>,
        /// The derived-expiry JSON to compare with (D5), or to write with `--write`. Default:
        /// `evidence/parity/derived_expiries.json` under the root.
        #[arg(long, value_name = "JSON", conflicts_with = "no_out")]
        out: Option<PathBuf>,
        /// No derived-expiry file: skip D5 and do not write.
        #[arg(long)]
        no_out: bool,
        /// Write the derivation to `--out` instead of comparing it (no D5).
        #[arg(long)]
        write: bool,
    },
    /// `<patch-id> <commit-oid>` per patch in the diff on stdin, byte-exact with
    /// `git patch-id` (was scripts/lib/git_patch_id.py). Exit 2 on an unknown mode.
    GitPatchId {
        /// --stable, --verbatim or --unstable (the default), as git spells them.
        #[arg(allow_hyphen_values = true, value_name = "MODE")]
        mode: Option<String>,
    },
    /// Name of every crates.io-sourced package in a Cargo.lock, one per line (was
    /// scripts/lib/lockfile_registry_packages.py). No argument exits 1, as the original did.
    LockfileRegistryPackages {
        /// The Cargo.lock to read.
        lock: Option<PathBuf>,
    },
    /// llama.cpp's fit verdict for one model-ladder cell, one JSON line (was
    /// scripts/lib/llama_fit_verdict.py). `--help` is an argument here, as it was there; use
    /// `aprender-ci-tools help llama-fit-verdict`.
    #[command(disable_help_flag = true)]
    LlamaFitVerdict {
        /// tool_found(0/1) pin rc free_mib version_file stdout_file model_path. `-` or an
        /// empty path reads as empty. Any other count exits 1, as the original's unpacking did.
        #[arg(num_args = 0.., allow_hyphen_values = true, trailing_var_arg = true)]
        args: Vec<String>,
    },
    /// Measure the model-ladder cells: `gpumem`, `enrich` and `measure` (was
    /// scripts/lib/model_ladder_cells_produce.py, kept while scripts/model_ladder.sh calls it).
    /// Exit 2 on a usage error or an `--only` that names no model.
    LadderCellsProduce {
        #[command(subcommand)]
        cmd: Box<ladder_cells_produce::Sub>,
    },
    /// `VERDICT <tag> <path>: <msg>` for `[profile.ci].fail-fast` in a nextest config (was
    /// the inline Python judge in scripts/check_nextest_ci_profile_no_fail_fast.sh). Exit 0
    /// ok, 1 FAIL, 2 ENV (cannot judge). The runner name is $NEXTEST_GUARD_RUNNER.
    NextestFailFast {
        /// `library` (a full TOML parser) or `fallback` (the purpose-built line reader).
        #[arg(long, value_parser = ["library", "fallback"], default_value = "library")]
        reader: String,
        /// The nextest config, e.g. .config/nextest.toml.
        toml: PathBuf,
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

/// tarball_workspace.py exits 2 on the errors it catches.
fn cannot_write(reason: String) -> Refusal {
    (String::new(), 2, reason)
}

fn read(path: &PathBuf) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))
}

/// The output, or a refusal.
/// The `spec-conformance` verb: the TSV on stdout, and the exit code the driver chose.
fn spec_conformance_cmd(opts: spec_conformance_run::Options) -> Result<String, Refusal> {
    match spec_conformance_run::run(&opts) {
        Ok((out, 0)) => Ok(out),
        Ok((out, code)) => Err((out, code, String::new())),
        Err(reason) => Err((String::new(), 2, reason)),
    }
}

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
        Cmd::LockfileRegistryPackages { lock } => {
            let lock = lock.ok_or_else(|| {
                nothing_printed("usage: lockfile-registry-packages <Cargo.lock>".to_owned())
            })?;
            let bytes = read(&lock).map_err(nothing_printed)?;
            Ok(lockfile_registry_packages::run(&bytes))
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
                .map_err(|e| cannot_write(format!("stdin: {e}")))?;
            tarball_workspace::target_dir(&meta).map_err(cannot_write)
        }
        Cmd::TarballWorkspace {
            name: Some(dir), ..
        } => tarball_workspace::package_name(&dir)
            .map_err(|(code, reason)| (String::new(), code, reason)),
        Cmd::TarballWorkspace { dir, .. } => {
            // The required `mode` group leaves DIR as the only other way in.
            let dir = dir.ok_or_else(|| cannot_write("tarball-workspace: no DIR".to_owned()))?;
            tarball_workspace::write_workspace(&dir)
                .map_err(|(code, reason)| (String::new(), code, reason))
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
        Cmd::LlamaFitVerdict { args } => llama_fit_verdict::run(&args).map_err(nothing_printed),
        Cmd::LadderCellsProduce { cmd } => {
            ladder_cells_produce::run(&cmd).map_err(|(code, reason)| (String::new(), code, reason))
        }
        Cmd::SpecConformanceDag {
            spec,
            source,
            derived,
            today,
        } => match spec_conformance_andon::run_cli(&spec, source, derived.as_deref(), today) {
            Ok((out, 0)) => Ok(out),
            Ok((out, code)) => Err((out, code, String::new())),
            Err(reason) => Err((String::new(), 2, reason)),
        },
        Cmd::SpecConformance {
            root,
            spec,
            ledger,
            out,
            no_out,
            write,
        } => spec_conformance_cmd(spec_conformance_run::Options {
            root,
            spec,
            ledger,
            out,
            no_out,
            write,
        }),
        Cmd::NextestFailFast { reader, toml } => {
            // clap's value_parser admits only the two names parse() knows.
            let reader = nextest_fail_fast::Reader::parse(&reader)
                .ok_or_else(|| nothing_printed(format!("unknown --reader {reader}")))?;
            let runner = std::env::var_os("NEXTEST_GUARD_RUNNER").map_or_else(
                || "unknown".to_owned(),
                |r| r.to_string_lossy().into_owned(),
            );
            match nextest_fail_fast::run(&toml, reader, &runner) {
                (0, out) => Ok(out),
                // The verdict line is the whole report, as in the original: no stderr.
                (code, out) => Err((out, code, String::new())),
            }
        }
    }
}

fn main() -> ExitCode {
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        // --help/--version exit 0. A usage error exits as its original did: 2 for the two
        // tarball tools (their argv-length checks return 2) and for ladder-cells-produce
        // (argparse), 1 for the rest.
        Err(e) => {
            let usage = match std::env::args().nth(1).as_deref() {
                Some("tarball-shrink-report" | "tarball-workspace" | "ladder-cells-produce") => 2,
                _ => 1,
            };
            let code = if e.use_stderr() { usage } else { 0 };
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
            if !reason.is_empty() {
                eprintln!("{reason}");
            }
            ExitCode::from(code)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAIN: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/spec_dag/pp_main_efa749e.md"
    );
    const LAB: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/spec_dag/pp_lab_9a10119.md"
    );

    /// The exit code `run` would give `spec-conformance-dag` on `spec` at `today`.
    fn exit_of(spec: &str) -> u8 {
        let cmd = Cmd::SpecConformanceDag {
            spec: PathBuf::from(spec),
            source: None,
            derived: None,
            today: Some("2026-11-28".to_owned()),
        };
        match run(cmd) {
            Ok(_) => 0,
            Err((_, code, _)) => code,
        }
    }

    #[test]
    fn spec_conformance_dag_exits_1_on_the_main_fixture_and_0_on_the_lab_fixture() {
        assert_eq!(exit_of(MAIN), 1);
        assert_eq!(exit_of(LAB), 0);
    }
}
