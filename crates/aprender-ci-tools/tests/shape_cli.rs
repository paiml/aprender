#![allow(clippy::disallowed_methods)] // unwrap in tests, as the crate's other test files
//! The `shape` binary's exit contract: 0 ok / 1 refused (reason on stdout) / 2 not measured or bad usage.
use std::process::{Command, Output};

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_aprender-rollup"))
        .args(args)
        .output()
        .unwrap()
}

fn frag(tag: &str, name: &str, text: &str) -> String {
    let d = std::env::temp_dir().join(format!("rollup-shape-cli-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let p = d.join(name);
    std::fs::write(&p, text).unwrap();
    p.to_str().unwrap().to_string()
}

#[test]
fn a_plain_fragment_exits_0_silently() {
    let f = frag("ok", "PMAT-1.yaml", "- id: PMAT-1\n  title: b\n");
    let o = run(&["shape", &f]);
    assert_eq!(o.status.code(), Some(0));
    assert!(o.stdout.is_empty() && o.stderr.is_empty());
}

#[test]
fn a_refused_fragment_exits_1_with_the_reason_on_stdout() {
    let f = frag("bad", "PMAT-2.txt", "id: PMAT-2\n");
    let o = run(&["shape", &f]);
    assert_eq!(o.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&o.stdout).contains("is not named <id>.yaml"));
}

#[test]
fn a_missing_file_exits_2_not_measured() {
    let o = run(&["shape", "/nonexistent/PMAT-3.yaml"]);
    assert_eq!(o.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&o.stderr).contains("not measured"));
}

#[test]
fn bad_usage_exits_2_with_the_usage_line() {
    let o = run(&["shape"]);
    assert_eq!(o.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&o.stderr).contains("usage: aprender-rollup shape"));
}

#[test]
fn an_unknown_subcommand_exits_2_and_the_usage_names_shape() {
    let o = run(&["bogus"]);
    assert_eq!(o.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&o.stderr).contains("shape <fragment>"));
}
