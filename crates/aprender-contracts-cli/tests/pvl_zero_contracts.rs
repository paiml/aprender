//! PVL-1 (PMAT-1099) — `pv` refuses to report over ZERO contracts.
//!
//! GROUND TRUTH (PV-LEAN-AUDIT-001, 2026-09-10, in-tree pv 0.66.0):
//!
//! ```text
//! pv lint /nonexistent-path     → rc=0  "Result: PASS"  (9 gates ✓ over 0 contracts)
//! pv lint <empty dir>           → rc=0  "Result: PASS"
//! pv proof-status <empty dir>   → rc=0  "Proof Status (0 contracts)"
//! ```
//!
//! A gate that measures nothing and reports PASS is this fleet's signature
//! defect ("0 violations over 0 files"). This file is the falsifier: every
//! reporting subcommand that walks a contract directory — `lint`,
//! `proof-status`, `coverage`, `graph`, `lean-status`, `verify-pipeline` —
//! must refuse an empty corpus with
//! exit 2 and one stderr line, `error: 0 contracts under <path>`.
//!
//! Exit 2, not 1: exit 1 means "the corpus was measured and failed"; exit 2
//! means "nothing was measured" — the invocation itself is wrong, which is
//! also what clap's usage errors return. A caller that treats both as failure
//! is unchanged; a caller that wants to tell them apart now can.
//!
//! DISCRIMINATION: the `control_*` case proves the guard is not "everything
//! exits 2": a directory holding one real contract is reported at exit 0 by
//! every command, and each report visibly counts that ONE contract — so a
//! build that panics, or that walks nothing and reports over zero, fails the
//! control too (a `!= 2` control was measured vacuous by the plan grill).

use std::path::{Path, PathBuf};
use std::process::Command;

/// The `pv` built from THIS tree. `CARGO_BIN_EXE_pv` is set by cargo for the
/// crate's own `[[bin]]`, so this cannot resolve a stale `pv` off `$PATH`
/// (#2552: PATH pv was 0.49.0 while in-tree was 0.63.0).
fn pv_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pv"))
}

struct Run {
    code: i32,
    stdout: String,
    stderr: String,
}

/// Run `pv` from a scratch cwd. `pv lint` writes `.pv/lint-previous.json`
/// into the CURRENT directory (measured, PV-LEAN-AUDIT-001), so the cwd is
/// never the tree under test and never the directory being judged.
fn pv(args: &[&str]) -> Run {
    let scratch = tempfile::tempdir().expect("scratch cwd is creatable");
    let out = Command::new(pv_bin())
        .current_dir(scratch.path())
        .args(args)
        .output()
        .expect("failed to spawn pv");
    Run {
        code: out.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

const REFUSAL: &str = "error: 0 contracts under ";
const REFUSAL_EXIT: i32 = 2;

fn assert_refused(run: &Run, path: &Path, what: &str) {
    assert_eq!(
        run.code, REFUSAL_EXIT,
        "{what}: expected exit {REFUSAL_EXIT} for an empty corpus, got {}\n--- stdout\n{}\n--- stderr\n{}",
        run.code, run.stdout, run.stderr
    );
    let expected = format!("{REFUSAL}{}", path.display());
    assert!(
        run.stderr.contains(&expected),
        "{what}: stderr lacks `{expected}`\n--- stderr\n{}",
        run.stderr
    );
    assert!(
        !run.stdout.contains("Result: PASS"),
        "{what}: a PASS verdict was printed over 0 contracts\n--- stdout\n{}",
        run.stdout
    );
}

fn assert_reported(run: &Run, what: &str, evidence_of_one: &str) {
    assert_eq!(
        run.code, 0,
        "{what}: expected exit 0 on a one-contract corpus, got {}\n--- stdout\n{}\n--- stderr\n{}",
        run.code, run.stdout, run.stderr
    );
    assert!(
        run.stdout.contains(evidence_of_one),
        "{what}: the report does not show the one contract (`{evidence_of_one}` absent)\n--- stdout\n{}",
        run.stdout
    );
    assert!(
        !run.stderr.contains(REFUSAL),
        "{what}: a refusal was printed for a non-empty corpus\n--- stderr\n{}",
        run.stderr
    );
}

fn empty_dir() -> tempfile::TempDir {
    tempfile::tempdir().expect("empty contract dir is creatable")
}

/// A path that does not exist and is never created.
fn nonexistent() -> PathBuf {
    std::env::temp_dir().join(format!("pvl-1-nonexistent-{}", std::process::id()))
}

fn s(p: &Path) -> &str {
    p.to_str().expect("utf-8 path")
}

#[test]
fn lint_nonexistent_path_is_refused() {
    let p = nonexistent();
    assert_refused(&pv(&["lint", s(&p)]), &p, "pv lint <nonexistent>");
}

#[test]
fn lint_empty_dir_is_refused() {
    let d = empty_dir();
    assert_refused(&pv(&["lint", s(d.path())]), d.path(), "pv lint <empty dir>");
}

#[test]
fn proof_status_empty_dir_is_refused() {
    let d = empty_dir();
    assert_refused(
        &pv(&["proof-status", s(d.path())]),
        d.path(),
        "pv proof-status <empty dir>",
    );
}

#[test]
fn proof_status_nonexistent_path_is_refused() {
    let p = nonexistent();
    assert_refused(
        &pv(&["proof-status", s(&p)]),
        &p,
        "pv proof-status <nonexistent>",
    );
}

#[test]
fn coverage_empty_dir_is_refused() {
    let d = empty_dir();
    assert_refused(
        &pv(&["coverage", s(d.path())]),
        d.path(),
        "pv coverage <empty dir>",
    );
}

#[test]
fn coverage_nonexistent_path_is_refused() {
    let p = nonexistent();
    assert_refused(&pv(&["coverage", s(&p)]), &p, "pv coverage <nonexistent>");
}

#[test]
fn graph_empty_dir_is_refused() {
    let d = empty_dir();
    assert_refused(
        &pv(&["graph", s(d.path())]),
        d.path(),
        "pv graph <empty dir>",
    );
}

#[test]
fn graph_nonexistent_path_is_refused() {
    let p = nonexistent();
    assert_refused(&pv(&["graph", s(&p)]), &p, "pv graph <nonexistent>");
}

/// Sidecars are not contracts: a directory holding only `binding.yaml` and a
/// dot-prefixed file is an empty corpus (`is_contract_yaml` skips both — the
/// ONE file rule `pv lint` walks with), so it is refused too.
#[test]
fn sidecar_only_dir_is_refused() {
    let d = empty_dir();
    std::fs::write(d.path().join("binding.yaml"), "crates: []\nbindings: []\n")
        .expect("fixture is writable");
    std::fs::write(d.path().join(".draft.yaml"), "steps: []\n").expect("fixture is writable");
    assert_refused(
        &pv(&["proof-status", s(d.path())]),
        d.path(),
        "pv proof-status <sidecars only>",
    );
    assert_refused(
        &pv(&["graph", s(d.path())]),
        d.path(),
        "pv graph <sidecars only>",
    );
}

/// Control: one real contract is reported at exit 0 by every command, and each
/// report shows the count of ONE. The contract is one the repo gate already
/// lints green (`pv lint contracts/` in CI), so exit 0 is the honest answer.
#[test]
fn control_one_contract_is_reported_not_refused() {
    let d = empty_dir();
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts/softmax-kernel-v1.yaml");
    std::fs::copy(&src, d.path().join("softmax-kernel-v1.yaml")).expect("contract copies");
    let dir = s(d.path());
    assert_reported(
        &pv(&["lint", "--format", "json", dir]),
        "pv lint <one contract>",
        "\"contracts\": 1",
    );
    assert_reported(
        &pv(&["proof-status", dir]),
        "pv proof-status <one contract>",
        "Proof Status (1 contracts)",
    );
    assert_reported(
        &pv(&["coverage", dir]),
        "pv coverage <one contract>",
        "Contracts:            1",
    );
    assert_reported(&pv(&["graph", dir]), "pv graph <one contract>", "Nodes: 1");
    assert_reported(
        &pv(&["lean-status", dir]),
        "pv lean-status <one contract>",
        "Softmax kernel",
    );
    assert_reported(
        &pv(&["verify-pipeline", dir]),
        "pv verify-pipeline <one contract>",
        "Contracts: 1",
    );
}

/// A single contract FILE is a one-contract corpus for every command (before
/// PVL-1, `lint`, `coverage` and `graph` walked a file path with `read_dir`,
/// found nothing, and reported PASS over 0 contracts — measured 2026-09-10).
#[test]
fn control_one_contract_file_is_reported_not_refused() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts/softmax-kernel-v1.yaml");
    let file = s(&src);
    assert_reported(
        &pv(&["lint", "--format", "json", file]),
        "pv lint <file>",
        "\"contracts\": 1",
    );
    assert_reported(
        &pv(&["proof-status", file]),
        "pv proof-status <file>",
        "Proof Status (1 contracts)",
    );
    assert_reported(
        &pv(&["coverage", file]),
        "pv coverage <file>",
        "Contracts:            1",
    );
    assert_reported(&pv(&["graph", file]), "pv graph <file>", "Nodes: 1");
}

#[test]
fn lean_status_empty_dir_is_refused() {
    let d = empty_dir();
    assert_refused(
        &pv(&["lean-status", s(d.path())]),
        d.path(),
        "pv lean-status <empty dir>",
    );
}

#[test]
fn verify_pipeline_empty_dir_is_refused() {
    let d = empty_dir();
    assert_refused(
        &pv(&["verify-pipeline", s(d.path())]),
        d.path(),
        "pv verify-pipeline <empty dir>",
    );
}

#[test]
fn verify_pipeline_nonexistent_path_is_refused() {
    let p = nonexistent();
    assert_refused(
        &pv(&["verify-pipeline", s(&p)]),
        &p,
        "pv verify-pipeline <nonexistent>",
    );
}

/// `--kind` that filters a real corpus down to nothing is an empty corpus too.
#[test]
fn proof_status_kind_filter_to_zero_is_refused() {
    let d = empty_dir();
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts/pv-cli-surface-v1.yaml");
    std::fs::copy(&src, d.path().join("pv-cli-surface-v1.yaml")).expect("contract copies");
    let run = pv(&["proof-status", "--kind", "kernel", s(d.path())]);
    assert_refused(
        &run,
        d.path(),
        "pv proof-status --kind kernel <patterns only>",
    );
    assert!(
        run.stderr.contains("(after --kind kernel)"),
        "the refusal must name the filter that emptied the corpus\n--- stderr\n{}",
        run.stderr
    );
}

// ---------------------------------------------------------------------------
// Review quorum on PR #3093 (3/3 FAIL; agy lanes ae646ee3, a8b61a5b, 7ba5404a):
// `--watch` and `--diff` returned from `lint::run` BEFORE the guard, so
// `pv lint --watch <empty>` printed a report over 0 contracts every 5 s and
// `pv lint --diff HEAD <empty>` said "Nothing to lint" at exit 0.
// ---------------------------------------------------------------------------

use std::process::Stdio;
use std::time::{Duration, Instant};

/// Like `pv`, but with a deadline: watch mode never returns on its own, so a
/// build that loops instead of refusing is killed and reported as `code -1`.
fn pv_deadline(args: &[&str], deadline: Duration) -> Run {
    let scratch = tempfile::tempdir().expect("scratch cwd is creatable");
    let mut child = Command::new(pv_bin())
        .current_dir(scratch.path())
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn pv");
    let started = Instant::now();
    let mut killed = false;
    while child.try_wait().expect("try_wait").is_none() {
        if started.elapsed() > deadline {
            child.kill().expect("kill pv");
            killed = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    let out = child.wait_with_output().expect("collect pv output");
    Run {
        code: if killed {
            -1
        } else {
            out.status.code().unwrap_or(-1)
        },
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

#[test]
fn lint_watch_empty_dir_is_refused() {
    let dir = tempfile::tempdir().expect("empty dir");
    let path = dir.path().to_str().expect("utf-8 path");
    let run = pv_deadline(&["lint", "--watch", path], Duration::from_secs(20));
    assert_refused(&run, dir.path(), "lint --watch <empty dir>");
}

#[test]
fn lint_diff_empty_dir_is_refused() {
    // `--diff` asks git for the contracts changed since <base>; the empty dir
    // must sit inside a repository with a HEAD or the diff arm is never taken.
    let repo = tempfile::tempdir().expect("repo dir");
    let git = |args: &[&str]| {
        let out = Command::new("git")
            .current_dir(repo.path())
            .args(args)
            .output()
            .expect("git is runnable");
        assert!(
            out.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    };
    git(&["init", "-q"]);
    git(&[
        "-c",
        "user.name=pvl",
        "-c",
        "user.email=pvl@test",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        "init",
    ]);
    let empty = repo.path().join("empty");
    std::fs::create_dir(&empty).expect("empty dir");
    let run = pv(&[
        "lint",
        "--diff",
        "HEAD",
        empty.to_str().expect("utf-8 path"),
    ]);
    assert_refused(&run, &empty, "lint --diff HEAD <empty dir>");
}

// ---------------------------------------------------------------------------
// Second review quorum on PR #3093 (3/3 FAIL; agy lanes 39742bf6, 7a9f134e,
// d12f6807): the walker silently DROPPED unparsable files, so a directory of
// only broken YAML was "0 contracts" (exit 2) to proof-status and `lint --diff`
// while `lint` measured it as `contracts: 1, errors: 1` (exit 1). One
// definition now: a broken contract file was measured, and failed.
// ---------------------------------------------------------------------------

const BROKEN_CONTRACT: &str = "contract: pvl-broken\nmetadata: [not a map\n";

fn assert_measured_failed(run: &Run, what: &str, file: &str) {
    assert_eq!(
        run.code, 1,
        "{what}: a broken contract file is measured and FAILED (exit 1), got {}\n--- stdout\n{}\n--- stderr\n{}",
        run.code, run.stdout, run.stderr
    );
    assert!(
        !run.stderr.contains(REFUSAL),
        "{what}: a broken file was reported as `0 contracts`\n--- stderr\n{}",
        run.stderr
    );
    assert!(
        run.stderr.contains(file),
        "{what}: the broken file `{file}` is not named\n--- stderr\n{}",
        run.stderr
    );
    assert!(
        !run.stdout.contains("Result: PASS") && !run.stdout.contains("(0 contracts)"),
        "{what}: a report was printed over a broken corpus\n--- stdout\n{}",
        run.stdout
    );
}

#[test]
fn unparsable_only_dir_is_measured_not_refused() {
    let d = empty_dir();
    std::fs::write(d.path().join("broken.yaml"), BROKEN_CONTRACT).expect("fixture is writable");
    let dir = s(d.path());
    for cmd in [
        "proof-status",
        "coverage",
        "graph",
        "lean-status",
        "verify-pipeline",
    ] {
        assert_measured_failed(
            &pv(&[cmd, dir]),
            &format!("pv {cmd} <broken only>"),
            "broken.yaml",
        );
    }
    // `lint` measures it through its validate gate: exit 1, never PASS, never a refusal.
    let run = pv(&["lint", dir]);
    assert_eq!(
        run.code, 1,
        "pv lint <broken only>: exit 1 expected, got {}\n{}",
        run.code, run.stderr
    );
    assert!(!run.stdout.contains("Result: PASS") && !run.stderr.contains(REFUSAL));
}

#[test]
fn mixed_dir_one_broken_file_fails_and_names_it() {
    // One real contract beside one broken file: the broken one is NOT skipped
    // (the old walker dropped it silently and reported over the rest).
    let d = empty_dir();
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts/softmax-kernel-v1.yaml");
    std::fs::copy(&src, d.path().join("softmax-kernel-v1.yaml")).expect("contract copies");
    std::fs::write(d.path().join("broken.yaml"), BROKEN_CONTRACT).expect("fixture is writable");
    let run = pv(&["proof-status", s(d.path())]);
    assert_measured_failed(
        &run,
        "pv proof-status <one good + one broken>",
        "broken.yaml",
    );
    assert!(
        run.stderr.contains("1 of 2 contract files"),
        "the count of measured files is not reported\n--- stderr\n{}",
        run.stderr
    );
}
