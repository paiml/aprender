//! `presentar gate` must be able to FAIL.
//!
//! `run_gates` is the only subcommand with an exit-code contract: it calls
//! `std::process::exit(1)` when the manifest's computed grade falls below
//! `--min-grade`. A gate that returns success for every input is theater, so
//! this test pins BOTH directions — a threadbare manifest must be rejected and
//! a rich one must be accepted. Asserting only the passing case would not
//! exclude "the gate always exits 0".

use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// Write `yaml` into the per-target tmpdir under `name` and return its path.
fn manifest(name: &str, yaml: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("gate_can_fail");
    fs::create_dir_all(&dir).expect("create tmpdir");
    let path = dir.join(name);
    fs::write(&path, yaml).expect("write manifest");
    path
}

/// Run `presentar gate <path>` at the default `--min-grade B`.
fn gate(path: &PathBuf) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_presentar"))
        .args(["gate", path.to_str().expect("utf-8 path")])
        .output()
        .expect("run presentar")
}

/// No description, no data sources, no widgets: scores well under grade B.
const THREADBARE: &str = r#"
presentar: "0.1"
name: threadbare
version: "1.0.0"
layout:
  type: dashboard
  columns: 12
  sections:
    - id: only-section
"#;

/// Description, five sections, twenty typed widgets, three refreshing data
/// sources: scores in the A band.
fn rich() -> String {
    let mut yaml = String::from(
        r#"
presentar: "0.1"
name: rich
version: "1.0.0"
description: A fully specified dashboard used to prove the gate can pass.
data:
  a:
    source: "file://a.csv"
    format: csv
    refresh: 60s
  b:
    source: "file://b.csv"
    format: csv
  c:
    source: "file://c.csv"
    format: csv
layout:
  type: dashboard
  columns: 12
  sections:
"#,
    );
    for section in 0..5 {
        yaml.push_str(&format!("    - id: section-{section}\n      widgets:\n"));
        for widget in 0..4 {
            yaml.push_str(&format!(
                "        - type: text\n          id: w-{section}-{widget}\n"
            ));
        }
    }
    yaml
}

#[test]
fn gate_rejects_a_threadbare_manifest() {
    let out = gate(&manifest("threadbare.yaml", THREADBARE));
    let stderr = String::from_utf8_lossy(&out.stderr);

    assert!(
        !out.status.success(),
        "presentar gate exited 0 on a manifest with no description, no data \
         sources and no widgets — the gate cannot fail. stdout:\n{}\nstderr:\n{stderr}",
        String::from_utf8_lossy(&out.stdout)
    );
    assert!(
        stderr.contains("GATE FAILED"),
        "expected a GATE FAILED diagnostic on stderr, got:\n{stderr}"
    );
}

#[test]
fn gate_accepts_a_rich_manifest() {
    let out = gate(&manifest("rich.yaml", &rich()));
    let stdout = String::from_utf8_lossy(&out.stdout);

    assert!(
        out.status.success(),
        "presentar gate rejected a fully specified manifest — the gate cannot \
         pass. stdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout.contains("GATE PASSED"),
        "expected a GATE PASSED line on stdout, got:\n{stdout}"
    );
}
