//! Falsification tests for CRUX-A-20 — Offline mode, zero network calls.
//!
//! Contract: `contracts/crux-A-20-v1.yaml`.
//!
//! Scope honesty (READ THIS):
//!
//! The canonical FALSIFY-CRUX-A-20-{001,002,003,004} gates in the contract
//! assert behavior verified via `strace -e trace=connect` — "zero outbound
//! TCP connect() syscalls under offline mode". That strace-level
//! verification is a separate follow-up harness (requires strace, a primed
//! cache, and a live comparison against the online path).
//!
//! What this harness discharges at `PARTIAL_ALGORITHM_LEVEL` are the
//! offline sub-claims that MUST hold before any strace verification can
//! succeed:
//!   - ALGO-001: `--offline` flag is accepted by the CLI parser and
//!     echoed as `Offline: true` in `--dry-run` output (no network I/O).
//!   - ALGO-002: `APR_OFFLINE=1` env var triggers the same offline signal
//!     in `--dry-run` output without `--offline` on the argv.
//!   - ALGO-003: `HF_HUB_OFFLINE=1` env var (HuggingFace compat) triggers
//!     the same offline signal.
//!   - ALGO-004: `--offline` flag, `APR_OFFLINE=1`, and `HF_HUB_OFFLINE=1`
//!     all produce byte-identical `Offline:` lines (observational
//!     equivalence).
//!   - ALGO-005: Without any offline signal, default is `Offline: false`.
//!
//! The full-network discharge of FALSIFY-001/002/003/004/005 is tracked as
//! follow-up and will live in a separate (strace-gated) harness.

#![allow(clippy::unwrap_used)]

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .ancestors()
        .nth(2)
        .expect("CARGO_MANIFEST_DIR has repo root 2 ancestors up")
        .to_path_buf()
}

fn run_apr_pull_with_env(
    args: &[&str],
    extra_env: &[(&str, &str)],
    clear_offline_env: bool,
) -> (std::process::Output, String, String) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_apr"));
    cmd.current_dir(repo_root()).arg("pull").args(args);
    if clear_offline_env {
        cmd.env_remove("APR_OFFLINE").env_remove("HF_HUB_OFFLINE");
    }
    for (k, v) in extra_env {
        cmd.env(k, *v);
    }
    let output = cmd
        .output()
        .unwrap_or_else(|e| panic!("CRUX-A-20: failed to spawn apr: {e}"));
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    (output, stdout, stderr)
}

fn offline_line(stdout: &str) -> String {
    stdout
        .lines()
        .find(|l| l.contains("Offline:"))
        .unwrap_or_else(|| {
            panic!("CRUX-A-20: stdout must contain an 'Offline:' line, got:\n{stdout}")
        })
        .to_string()
}

// ---------------------------------------------------------------------------
// ALGO-001: `apr pull <short> --dry-run --offline` is accepted and echoed.
// ---------------------------------------------------------------------------

#[test]
fn falsify_crux_a_20_algo_001_dry_run_offline_flag_accepted() {
    let (output, stdout, stderr) = run_apr_pull_with_env(
        &["llama3", "--dry-run", "--offline"],
        &[],
        true, // clear APR_OFFLINE/HF_HUB_OFFLINE
    );
    assert!(
        output.status.success(),
        "CRUX-A-20 ALGO-001: --offline --dry-run must exit 0\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    let line = offline_line(&stdout);
    assert!(
        line.contains("true"),
        "CRUX-A-20 ALGO-001: --offline line must contain 'true', got: {line:?}"
    );
}

#[test]
fn falsify_crux_a_20_algo_005_default_offline_is_false() {
    let (output, stdout, _stderr) = run_apr_pull_with_env(
        &["llama3", "--dry-run"],
        &[],
        true, // clear both env vars
    );
    assert!(output.status.success());
    let line = offline_line(&stdout);
    assert!(
        line.contains("false"),
        "CRUX-A-20 ALGO-005: default offline line must contain 'false', got: {line:?}"
    );
}

// ---------------------------------------------------------------------------
// ALGO-002: APR_OFFLINE=1 alone triggers offline signal.
// ---------------------------------------------------------------------------

#[test]
fn falsify_crux_a_20_algo_002_apr_offline_env_triggers_offline() {
    let (output, stdout, _stderr) = run_apr_pull_with_env(
        &["llama3", "--dry-run"],
        &[("APR_OFFLINE", "1")],
        true, // clear both, then set APR_OFFLINE=1
    );
    assert!(output.status.success());
    let line = offline_line(&stdout);
    assert!(
        line.contains("true"),
        "CRUX-A-20 ALGO-002: APR_OFFLINE=1 line must contain 'true', got: {line:?}"
    );
}

#[test]
fn falsify_crux_a_20_algo_002_apr_offline_env_zero_is_false() {
    let (output, stdout, _stderr) =
        run_apr_pull_with_env(&["llama3", "--dry-run"], &[("APR_OFFLINE", "0")], true);
    assert!(output.status.success());
    let line = offline_line(&stdout);
    assert!(
        line.contains("false"),
        "CRUX-A-20 ALGO-002: APR_OFFLINE=0 must NOT trigger offline, got: {line:?}"
    );
}

// ---------------------------------------------------------------------------
// ALGO-003: HF_HUB_OFFLINE=1 (HF compat) triggers offline signal.
// ---------------------------------------------------------------------------

#[test]
fn falsify_crux_a_20_algo_003_hf_hub_offline_env_triggers_offline() {
    let (output, stdout, _stderr) =
        run_apr_pull_with_env(&["llama3", "--dry-run"], &[("HF_HUB_OFFLINE", "1")], true);
    assert!(output.status.success());
    let line = offline_line(&stdout);
    assert!(
        line.contains("true"),
        "CRUX-A-20 ALGO-003: HF_HUB_OFFLINE=1 line must contain 'true', got: {line:?}"
    );
}

// ---------------------------------------------------------------------------
// ALGO-004: --offline, APR_OFFLINE=1, HF_HUB_OFFLINE=1 are equivalent.
// ---------------------------------------------------------------------------

#[test]
fn falsify_crux_a_20_algo_004_all_three_offline_signals_equivalent() {
    let (_, flag_stdout, _) =
        run_apr_pull_with_env(&["llama3", "--dry-run", "--offline"], &[], true);
    let (_, apr_env_stdout, _) =
        run_apr_pull_with_env(&["llama3", "--dry-run"], &[("APR_OFFLINE", "1")], true);
    let (_, hf_env_stdout, _) =
        run_apr_pull_with_env(&["llama3", "--dry-run"], &[("HF_HUB_OFFLINE", "1")], true);
    let flag_line = offline_line(&flag_stdout);
    let apr_env_line = offline_line(&apr_env_stdout);
    let hf_env_line = offline_line(&hf_env_stdout);
    assert_eq!(
        flag_line, apr_env_line,
        "CRUX-A-20 ALGO-004: --offline must equal APR_OFFLINE=1 (observationally)"
    );
    assert_eq!(
        apr_env_line, hf_env_line,
        "CRUX-A-20 ALGO-004: APR_OFFLINE=1 must equal HF_HUB_OFFLINE=1 (observationally)"
    );
}

// ---------------------------------------------------------------------------
// Determinism: two back-to-back --offline invocations emit byte-identical
// Offline lines. Mirrors A-01/A-03 determinism gates.
// ---------------------------------------------------------------------------

#[test]
fn falsify_crux_a_20_algo_001_dry_run_offline_is_deterministic() {
    let args: &[&str] = &["llama3", "--dry-run", "--offline"];
    let (_, a, _) = run_apr_pull_with_env(args, &[], true);
    let (_, b, _) = run_apr_pull_with_env(args, &[], true);
    assert_eq!(
        offline_line(&a),
        offline_line(&b),
        "CRUX-A-20: back-to-back --dry-run --offline must be byte-identical"
    );
}
