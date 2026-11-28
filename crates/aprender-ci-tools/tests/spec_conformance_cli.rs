//! The `spec-conformance <root>` subcommand: the Rust home of `scripts/lib/spec_conformance.py`'s
//! driver (PMAT-7534, slice 2d). It runs the C rules (§6 and C5), the L rules (the ledger), then
//! the D rules (the §12 DAG), and prints the Python's TSV lines in the Python's order.
//!
//! The one intended difference from the Python: a row past its expiry whose expiry cell starts
//! with `LAB` is a `LAB` line, not a D6 VIOLATION (C324: LAB cannot block).
//!
//! Each test builds a synthetic repository root under the OS temp dir, runs the built binary,
//! and reads its stdout and exit code. Exit 0 = no VIOLATION, 1 = at least one VIOLATION.

use std::path::{Path, PathBuf};
use std::process::Command;

const TODAY: &str = "2026-11-28";

const LEDGER: &str = "# Ledger (fixture)\n\n\
| # | started_utc | host | class | model · quant | workload | commit | interleaved | receipts | conformance |\n\
|---|---|---|---|---|---|---|---|---|---|\n\
| 1 | t | lambda | cuda | q · q4 | W1 | `aaaa` | false | e | RECORDED |\n";

/// A master with one §6 row (OPEN, so no case names are checked) and one §12 row whose expiry
/// cell is `expires`.
fn master(expires: &str) -> String {
    format!(
        "# PP-LLAMA-001 (fixture)\n\n\
## §6 Invariants\n\n\
| id | rule | must-fire | must-not-fire | status | producer · selftest |\n\
|---|---|---|---|---|---|\n\
| PP-1 | a rule | mutate | hold | OPEN | `scripts/x.sh` · `pg:alpha` / `pg:beta` |\n\n\
## §12 Owed work\n\n\
| row | what is owed | owner | blocked_by | expires |\n\
|---|---|---|---|---|\n\
| r1 | a | o | — | {expires} |\n\n\
## §13 End\n"
    )
}

/// A fresh root named after the test, holding a master with the given expiry cell and a ledger.
fn fixture(name: &str, expires: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "spec-conformance-cli-{}-{name}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    write(
        &dir.join("docs/specifications/PP-LLAMA-001-MASTER.md"),
        &master(expires),
    );
    write(&dir.join("evidence/parity/LEDGER.md"), LEDGER);
    dir
}

fn write(path: &Path, text: &str) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create fixture dir");
    }
    std::fs::write(path, text).expect("write fixture file");
}

/// Runs `spec-conformance <root> <args>` with `today` pinned, and returns (exit code, stdout).
fn run(root: &Path, args: &[&str]) -> (Option<i32>, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_aprender-ci-tools"))
        .arg("spec-conformance")
        .arg(root)
        .args(args)
        .env("SPEC_CONFORMANCE_TODAY", TODAY)
        .output()
        .expect("spawn aprender-ci-tools");
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
    )
}

fn lines(stdout: &str) -> Vec<&str> {
    stdout.lines().collect()
}

/// Must-fire: an expired §12 row that is not discharged is D6, and the run exits 1.
#[test]
fn an_expired_open_dag_row_is_d6_and_exit_1() {
    let root = fixture("expired_open", "Expires **2026-09-10**");
    let (code, stdout) = run(&root, &["--no-out"]);
    let _ = std::fs::remove_dir_all(&root);
    assert_eq!(code, Some(1), "stdout:\n{stdout}");
    assert!(
        lines(&stdout)
            .iter()
            .any(|l| l.starts_with("VIOLATION\tD6\tr1\t")),
        "no D6 for r1 in:\n{stdout}"
    );
}

/// Must-not-fire: an expired LAB row is a LAB line, never a VIOLATION, and the run exits 0.
#[test]
fn an_expired_lab_dag_row_is_a_lab_line_and_exit_0() {
    let root = fixture(
        "expired_lab",
        "**LAB** (operator ruling), Expires **2026-09-10**",
    );
    let (code, stdout) = run(&root, &["--no-out"]);
    let _ = std::fs::remove_dir_all(&root);
    assert_eq!(code, Some(0), "stdout:\n{stdout}");
    assert!(
        lines(&stdout).iter().any(|l| l.starts_with("LAB\tr1\t")),
        "no LAB line for r1 in:\n{stdout}"
    );
    assert!(
        !stdout.lines().any(|l| l.starts_with("VIOLATION")),
        "a LAB row produced a VIOLATION:\n{stdout}"
    );
}

/// The Python's order: SPEC first, then the §6 rows, then the ledger, then the DAG, and PARSED
/// (the §6 row count) last.
#[test]
fn records_come_in_the_python_order_and_parsed_is_last() {
    let root = fixture("order", "Expires **2026-12-01**");
    let (code, stdout) = run(&root, &["--no-out"]);
    let _ = std::fs::remove_dir_all(&root);
    assert_eq!(code, Some(0), "stdout:\n{stdout}");
    let kinds: Vec<&str> = lines(&stdout)
        .iter()
        .map(|l| l.split('\t').next().unwrap_or(""))
        .collect();
    assert_eq!(kinds.first(), Some(&"SPEC"), "kinds: {kinds:?}");
    assert_eq!(kinds.last(), Some(&"PARSED"), "kinds: {kinds:?}");
    let at = |k: &str| kinds.iter().position(|x| *x == k).unwrap_or(usize::MAX);
    assert!(at("ROW") < at("LEDGER"), "kinds: {kinds:?}");
    assert!(at("LEDGER") < at("DAG"), "kinds: {kinds:?}");
    assert!(
        stdout.lines().any(|l| l == "PARSED\t1"),
        "stdout:\n{stdout}"
    );
}

/// C0: no master under docs/specifications is a violation, and PARSED is 0.
#[test]
fn a_missing_master_is_c0_and_parsed_0_and_exit_1() {
    let root = std::env::temp_dir().join(format!(
        "spec-conformance-cli-{}-no_master",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("create empty root");
    let (code, stdout) = run(&root, &["--no-out"]);
    let _ = std::fs::remove_dir_all(&root);
    assert_eq!(code, Some(1), "stdout:\n{stdout}");
    assert!(
        stdout.lines().any(|l| l.starts_with("VIOLATION\tC0\t")),
        "no C0 in:\n{stdout}"
    );
    assert!(
        stdout.lines().any(|l| l == "PARSED\t0"),
        "stdout:\n{stdout}"
    );
}

/// `--write --out <path>` writes the §12 derivation there and exits 0, and a later plain run
/// that compares against it is clean (no D5).
#[test]
fn write_then_compare_is_clean() {
    let root = fixture("write", "Expires **2026-12-01**");
    let out = root.join("evidence/parity/derived_expiries.json");
    let out_arg = out.to_string_lossy().into_owned();
    let (wcode, _) = run(&root, &["--write", "--out", &out_arg]);
    assert_eq!(wcode, Some(0), "--write did not exit 0");
    assert!(out.exists(), "--write did not create {}", out.display());
    let (code, stdout) = run(&root, &["--out", &out_arg]);
    let _ = std::fs::remove_dir_all(&root);
    assert_eq!(code, Some(0), "stdout:\n{stdout}");
    assert!(
        !stdout.lines().any(|l| l.starts_with("VIOLATION\tD5")),
        "a fresh derivation drifted:\n{stdout}"
    );
}
