//! #6353 (PMAT-5902 S1) — the `aprender-roadmap` stub, run as the built binary.

use std::process::Command;

fn aprender_roadmap(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_aprender-roadmap"))
        .args(args)
        .output()
        .expect("spawn aprender-roadmap")
}

#[test]
fn help_exits_zero_and_names_the_tool() {
    let o = aprender_roadmap(&["--help"]);
    assert_eq!(o.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&o.stdout).contains("usage: aprender-roadmap"));
}

#[test]
fn version_is_name_semver_sha() {
    let o = aprender_roadmap(&["--version"]);
    assert_eq!(o.status.code(), Some(0));
    let s = String::from_utf8_lossy(&o.stdout);
    assert!(
        s.starts_with(&format!("aprender-roadmap {} (", env!("CARGO_PKG_VERSION"))),
        "{s}"
    );
}

#[test]
fn a_subcommand_is_refused_with_exit_two_never_a_silent_pass() {
    let o = aprender_roadmap(&["aggregate", "--check"]);
    assert_eq!(o.status.code(), Some(2));
    assert!(o.stdout.is_empty());
    assert!(String::from_utf8_lossy(&o.stderr).contains("not ported yet"));
}
