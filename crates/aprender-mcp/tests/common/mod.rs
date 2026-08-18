//! Declared resolution of the `apr` binary for the falsifiers that drive the
//! real CLI (`falsify_mcp_dogfood_001`, `falsify_mcp_stdio_protocol`).
//!
//! # Why this module exists
//!
//! `aprender-mcp` is a **lib-only** package: it declares no `[[bin]]`, so cargo
//! never sets `CARGO_BIN_EXE_apr` for these test targets. Both files used to
//! call `assert_cmd::cargo::cargo_bin("apr")`, which on assert_cmd 2.2 reads
//! `CARGO_BIN_EXE_apr` and, finding it unset, falls back to guessing
//! `<dir of current_exe>/../apr` — the target directory of *whoever happened to
//! build last*. Neither half is a declared dependency:
//!
//! * The env-var half can never fire here. Only the package that *builds* a
//!   binary gets `CARGO_BIN_EXE_<name>`, and this package builds none.
//! * The guess half depends on another package having already built `apr` into
//!   that exact directory. When it has, the test silently runs whatever commit's
//!   binary is lying there; when it has not, `cargo_bin` **panics** with
//!   "`CARGO_BIN_EXE_apr` is unset". Measured on a fresh worktree: all six
//!   falsifiers in these two files failed that way, before a single assertion ran.
//!
//! The panic also made `falsify_mcp_dogfood_001`'s
//! `if candidate.is_file() { .. } else { build_apr_binary() }` unreachable —
//! `cargo_bin` returns a path only when the file already exists, so the
//! build-on-demand arm was dead code that could never repair the missing binary.
//!
//! # What replaces it
//!
//! Ask cargo to build the binary we name, then take the path **cargo reports**
//! for it. Same doctrine as `scripts/apr_bin.sh` ("Ask cargo; never guess"),
//! for the same reason: every strategy that *searches* for an `apr` eventually
//! finds the wrong one. `--message-format=json` emits a `compiler-artifact`
//! record whose `executable` field is the authoritative path, so this is
//! immune to `CARGO_TARGET_DIR`, to `.cargo/config.toml` target-dir redirects
//! (gitignored here, so main and a worktree build to different places), and to
//! cargo's `build-dir` split — the three things the directory guess gets wrong.
//!
//! `cargo build` is a cheap no-op when the binary is already current, so the
//! build is unconditional: short-circuiting on "a file exists there" is exactly
//! the stale-artifact hole documented above.
//!
//! # No `$APR_BIN` escape hatch, deliberately
//!
//! `aprender_mcp::apr_bin` honours `$APR_BIN` at *runtime*, and the spawned
//! `apr mcp` child inherits this process's environment. Reading `$APR_BIN` here
//! would therefore also redirect the server's own subprocess resolution, past
//! the mock shim the dogfood falsifier installs on `PATH` — the override would
//! silently change what is under test rather than just where it lives.

use std::path::PathBuf;
use std::process::{Command, Stdio};

/// The workspace package that owns the `apr` binary (root `Cargo.toml`,
/// `[[bin]] name = "apr"`). Pinned by version because crates.io ships older
/// `aprender` releases that can land in the dependency graph and make a bare
/// `-p aprender` spec ambiguous. `aprender-mcp` and the root package both take
/// `version.workspace = true`, so `CARGO_PKG_VERSION` here is the right one.
fn apr_package_spec() -> String {
    format!("aprender@{}", env!("CARGO_PKG_VERSION"))
}

/// Build `apr` and return the path cargo reports for it.
///
/// Panics with the cargo failure surfaced on stderr if the build fails — a
/// broken `apr` is a real failure these falsifiers must report, not skip.
pub fn apr_binary() -> PathBuf {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let pkg_spec = apr_package_spec();

    // `json-render-diagnostics` keeps the machine-readable artifact records on
    // stdout while compiler errors stay human-readable on the inherited stderr,
    // so a build failure here is as legible as a normal `cargo build`.
    let output = Command::new(&cargo)
        .args([
            "build",
            "--bin",
            "apr",
            "-p",
            &pkg_spec,
            "--message-format=json-render-diagnostics",
        ])
        .stderr(Stdio::inherit())
        .output()
        .unwrap_or_else(|e| panic!("invoke `{cargo} build --bin apr -p {pkg_spec}`: {e}"));
    assert!(
        output.status.success(),
        "`cargo build --bin apr -p {pkg_spec}` failed with {:?}",
        output.status
    );

    let stdout = String::from_utf8(output.stdout).expect("cargo --message-format=json emits UTF-8");
    let mut executables: Vec<PathBuf> = stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|msg| msg["reason"] == "compiler-artifact" && msg["target"]["name"] == "apr")
        .filter_map(|msg| msg["executable"].as_str().map(PathBuf::from))
        .collect();
    executables.sort();
    executables.dedup();

    // Exactly one, or we do not know which `apr` we are testing. `--bin apr -p
    // <one package>` compiles a single bin target, so two distinct paths means
    // the graph grew a second `apr` and a "pick the last one" rule would decide
    // it by luck.
    assert_eq!(
        executables.len(),
        1,
        "expected exactly one `apr` executable from `cargo build --bin apr -p {pkg_spec}`, \
         cargo reported {executables:?}"
    );
    let path = executables.remove(0);
    assert!(
        path.is_file(),
        "cargo reported `apr` at {} but nothing is there",
        path.display()
    );
    path
}
