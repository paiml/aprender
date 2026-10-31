//! #6353 (PMAT-5902 S1) — the `pv-roadmap` stub, run as the built binary.

use std::process::Command;

fn pv_roadmap(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_pv-roadmap"))
        .args(args)
        .output()
        .expect("spawn pv-roadmap")
}

#[test]
fn help_exits_zero_and_names_the_tool() {
    let o = pv_roadmap(&["--help"]);
    assert_eq!(o.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&o.stdout).contains("usage: pv-roadmap"));
}

#[test]
fn version_is_name_semver_sha() {
    let o = pv_roadmap(&["--version"]);
    assert_eq!(o.status.code(), Some(0));
    let s = String::from_utf8_lossy(&o.stdout);
    assert!(
        s.starts_with(&format!("pv-roadmap {} (", env!("CARGO_PKG_VERSION"))),
        "{s}"
    );
}

#[test]
fn a_subcommand_is_refused_with_exit_two_never_a_silent_pass() {
    let o = pv_roadmap(&["aggregate", "--check"]);
    assert_eq!(o.status.code(), Some(2));
    assert!(o.stdout.is_empty());
    assert!(String::from_utf8_lossy(&o.stderr).contains("not ported yet"));
}
