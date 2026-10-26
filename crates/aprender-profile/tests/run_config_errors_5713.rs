//! `run` rejects bad configuration before it traces anything, with a message that
//! names the offending flag (#5713). No ptrace is needed: every case fails early.
#![allow(deprecated)] // suppress assert_cmd::Command::cargo_bin deprecation in tests

use predicates::prelude::*;

fn profile(args: &[&str]) -> assert_cmd::assert::Assert {
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.args(args).args(["--", "/bin/true"]);
    cmd.assert()
}

#[test]
fn ml_clusters_below_two_is_rejected_with_the_value() {
    profile(&["--ml-clusters", "1"])
        .code(1)
        .stderr(predicate::str::contains("Invalid value for --ml-clusters: 1 (must be >= 2)"));
}

#[test]
fn missing_transpiler_map_names_the_path() {
    profile(&["--transpiler-map", "/nonexistent/map.json"])
        .code(1)
        .stderr(predicate::str::contains("Source map file not found: /nonexistent/map.json"));
}

#[test]
fn malformed_transpiler_map_is_an_error_not_an_empty_map() {
    let dir = tempfile::tempdir().expect("tempdir");
    let map = dir.path().join("bad.json");
    std::fs::write(&map, "{bad").expect("write");
    profile(&["--transpiler-map", map.to_str().expect("utf8")])
        .code(1)
        .stderr(predicate::str::contains("Invalid source map JSON"));
}

#[test]
fn unknown_chaos_preset_lists_the_valid_ones() {
    profile(&["--chaos", "nosuchpreset"])
        .code(1)
        .stderr(predicate::str::contains("Chaos config error"))
        .stderr(predicate::str::contains("gentle"))
        .stderr(predicate::str::contains("aggressive"));
}

#[test]
fn unparsable_chaos_memory_limit_is_rejected() {
    profile(&["--chaos-memory-limit", "zzz"])
        .code(1)
        .stderr(predicate::str::contains("Chaos config error"));
}
