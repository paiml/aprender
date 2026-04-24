//! Falsification tests for CRUX-A-03 — Pin to revision/branch/SHA.
//!
//! Contract: `contracts/crux-A-03-v1.yaml`.
//!
//! Scope honesty (READ THIS):
//!
//! The canonical FALSIFY-CRUX-A-03-{001,002,003} gates in the contract
//! assert behavior that requires a live HuggingFace Hub call
//! (`GET /api/models/<repo>/revision/<REV>`) and a real file download.
//! Those gates remain network-dependent and are NOT discharged here.
//!
//! What this harness discharges at `PARTIAL_ALGORITHM_LEVEL` are three
//! offline sub-claims that MUST hold before any network resolution can be
//! correct:
//!   - ALGO-001: `--revision <REV>` is accepted by the CLI parser and
//!     echoed in the `--dry-run` output (no network I/O).
//!   - ALGO-002: the default revision is "main" when `--revision` is
//!     omitted (mirrors huggingface_hub default).
//!   - ALGO-003: malformed revision specs (empty, whitespace, URL) are
//!     rejected locally with a non-zero exit before any network call.
//!
//! The full-network discharge of FALSIFY-001/002/003 is tracked as a
//! follow-up and will live in a separate (network-gated) harness.

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

fn run_apr_pull(args: &[&str]) -> (std::process::Output, String, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_apr"))
        .current_dir(repo_root())
        .arg("pull")
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("CRUX-A-03: failed to spawn apr: {e}"));
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    (output, stdout, stderr)
}

// ---------------------------------------------------------------------------
// ALGO-001: `apr pull <short> --dry-run --revision <REV>` accepts and echoes
// the revision. Exercises the CLI parser wiring + the run_dry_run hook.
// ---------------------------------------------------------------------------

#[test]
fn falsify_crux_a_03_algo_001_dry_run_echoes_revision_main() {
    let (output, stdout, stderr) = run_apr_pull(&["llama3", "--dry-run", "--revision", "main"]);
    assert!(
        output.status.success(),
        "CRUX-A-03 ALGO-001: --revision main --dry-run must exit 0\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.contains("Revision:") && stdout.contains("main"),
        "CRUX-A-03 ALGO-001: dry-run stdout must echo 'Revision: main', got:\n{stdout}"
    );
}

#[test]
fn falsify_crux_a_03_algo_001_dry_run_echoes_full_sha() {
    let sha = "0123456789abcdef0123456789abcdef01234567";
    let (output, stdout, _stderr) = run_apr_pull(&["llama3", "--dry-run", "--revision", sha]);
    assert!(
        output.status.success(),
        "CRUX-A-03 ALGO-001: full 40-hex SHA must be accepted locally"
    );
    assert!(
        stdout.contains(sha) && stdout.contains("FullSha"),
        "CRUX-A-03 ALGO-001: dry-run must echo the full SHA and classify it as FullSha, got:\n{stdout}"
    );
}

#[test]
fn falsify_crux_a_03_algo_001_dry_run_echoes_tag() {
    let (output, stdout, _stderr) = run_apr_pull(&["llama3", "--dry-run", "--revision", "v1.0"]);
    assert!(output.status.success());
    assert!(
        stdout.contains("v1.0") && stdout.contains("RefName"),
        "CRUX-A-03 ALGO-001: tag 'v1.0' must be classified as RefName, got:\n{stdout}"
    );
}

// ---------------------------------------------------------------------------
// ALGO-002: default revision is "main" when `--revision` is omitted.
// ---------------------------------------------------------------------------

#[test]
fn falsify_crux_a_03_algo_002_default_revision_is_main() {
    let (output, stdout, stderr) = run_apr_pull(&["llama3", "--dry-run"]);
    assert!(
        output.status.success(),
        "CRUX-A-03 ALGO-002: --dry-run without --revision must exit 0\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    let rev_line = stdout
        .lines()
        .find(|l| l.contains("Revision:"))
        .unwrap_or_else(|| {
            panic!("CRUX-A-03 ALGO-002: stdout must contain a 'Revision:' line, got:\n{stdout}")
        });
    assert!(
        rev_line.contains("main"),
        "CRUX-A-03 ALGO-002: default revision must be 'main', got: {rev_line:?}"
    );
}

#[test]
fn falsify_crux_a_03_algo_002_default_matches_explicit_main() {
    let (_, default_stdout, _) = run_apr_pull(&["llama3", "--dry-run"]);
    let (_, explicit_stdout, _) = run_apr_pull(&["llama3", "--dry-run", "--revision", "main"]);
    let extract = |s: &str| -> String {
        s.lines()
            .find(|l| l.contains("Revision:"))
            .unwrap_or("")
            .to_string()
    };
    assert_eq!(
        extract(&default_stdout),
        extract(&explicit_stdout),
        "CRUX-A-03 ALGO-002: omitting --revision must be byte-equivalent to --revision main"
    );
}

// ---------------------------------------------------------------------------
// ALGO-003: malformed revision specs fail locally (pre-network) with
// non-zero exit. This is the offline precondition of FALSIFY-CRUX-A-03-003
// "unknown revision exits non-zero" — we prove the malformed-format subset
// here, leaving the "valid-format but unknown on remote" case network-gated.
// ---------------------------------------------------------------------------

#[test]
fn falsify_crux_a_03_algo_003_empty_revision_rejected() {
    let (output, _stdout, stderr) = run_apr_pull(&["llama3", "--dry-run", "--revision", ""]);
    assert!(
        !output.status.success(),
        "CRUX-A-03 ALGO-003: empty --revision must exit non-zero"
    );
    assert!(
        stderr.to_lowercase().contains("revision") || stderr.to_lowercase().contains("empty"),
        "CRUX-A-03 ALGO-003: stderr must explain the rejection, got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_a_03_algo_003_url_in_revision_rejected() {
    let (output, _stdout, stderr) =
        run_apr_pull(&["llama3", "--dry-run", "--revision", "https://example.com/x"]);
    assert!(
        !output.status.success(),
        "CRUX-A-03 ALGO-003: URL-shaped --revision must exit non-zero"
    );
    assert!(
        stderr.to_lowercase().contains("revision"),
        "CRUX-A-03 ALGO-003: stderr must explain the rejection, got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_a_03_algo_003_whitespace_rejected() {
    let (output, _stdout, _stderr) =
        run_apr_pull(&["llama3", "--dry-run", "--revision", "has space"]);
    assert!(
        !output.status.success(),
        "CRUX-A-03 ALGO-003: whitespace in --revision must exit non-zero"
    );
}

// ---------------------------------------------------------------------------
// Determinism: two back-to-back --dry-run invocations with the same
// revision emit byte-identical Revision lines. Mirrors the A-01 pattern.
// ---------------------------------------------------------------------------

#[test]
fn falsify_crux_a_03_algo_001_dry_run_is_deterministic() {
    let args: &[&str] = &[
        "llama3",
        "--dry-run",
        "--revision",
        "0123456789abcdef0123456789abcdef01234567",
    ];
    let (_, a, _) = run_apr_pull(args);
    let (_, b, _) = run_apr_pull(args);
    let rev_line = |s: &str| -> String {
        s.lines()
            .find(|l| l.contains("Revision:"))
            .unwrap_or("")
            .to_string()
    };
    assert_eq!(
        rev_line(&a),
        rev_line(&b),
        "CRUX-A-03: back-to-back --dry-run --revision <sha> must be byte-identical"
    );
}
