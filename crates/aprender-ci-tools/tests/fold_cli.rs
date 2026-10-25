#![allow(clippy::disallowed_methods)] // unwrap in tests, as the crate's other test files
//! The `fold` binary's contract: aggregate on stdout and exit 0, or exit 2 (not measured / bad usage).
use std::process::{Command, Output};

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_aprender-rollup"))
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn fold_prints_the_aggregate_and_exits_0() {
    let d = std::env::temp_dir().join(format!("rollup-fold-cli-{}", std::process::id()));
    std::fs::create_dir_all(d.join("e")).unwrap();
    std::fs::write(
        d.join("b.yaml"),
        "- id: PMAT-1\n  t: a\n- id: PMAT-3\n  t: c\n",
    )
    .unwrap();
    std::fs::write(d.join("e/PMAT-2.yaml"), "- id: PMAT-2\n  t: b\n").unwrap();
    let o = run(&[
        "fold",
        d.join("b.yaml").to_str().unwrap(),
        d.join("e").to_str().unwrap(),
    ]);
    assert_eq!(o.status.code(), Some(0));
    assert_eq!(
        String::from_utf8_lossy(&o.stdout),
        "- id: PMAT-1\n  t: a\n- id: PMAT-2\n  t: b\n- id: PMAT-3\n  t: c\n"
    );
    assert!(o.stderr.is_empty());
}

#[test]
fn a_missing_base_exits_2_not_measured() {
    let o = run(&["fold", "/nonexistent/b.yaml", "/nonexistent/e"]);
    assert_eq!(o.status.code(), Some(2));
    assert!(o.stdout.is_empty());
    assert!(String::from_utf8_lossy(&o.stderr).contains("not measured"));
}

#[test]
fn bad_usage_exits_2_with_the_usage_line() {
    let o = run(&["fold", "only-one"]);
    assert_eq!(o.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&o.stderr).contains("fold <base.yaml> <entries-dir>"));
}
