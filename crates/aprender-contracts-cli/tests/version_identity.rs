//! Falsifier for #2559: `pv --version` must say WHICH `pv` this is.
//!
//! Four things claim the name `pv` on a developer box:
//!   1. `pv(1)`, the pipe viewer shipped by every distro (package `pv`)
//!   2. the `pv` crate on crates.io (a pipe viewer, first published 2019)
//!   3. `aprender-contracts-cli` — this binary
//!   4. the aprender facade (removed by #2553)
//!
//! Before this test existed the binary printed exactly `pv 0.63.0`: a bare name
//! and a semver, indistinguishable at a glance from the pipe viewer. The name is
//! settled (operator ruling 2026-08-21) — so the version line is the mitigation
//! the project relies on, and it has to actually carry identity.
//!
//! Every assertion below is written to EXCLUDE the pre-fix output. Running this
//! file against a binary that prints `pv <semver>` and nothing else must go RED.

use std::process::Command;

const PV: &str = env!("CARGO_BIN_EXE_pv");
const VERSION: &str = env!("CARGO_PKG_VERSION");

fn run(flag: &str) -> (String, String, i32) {
    let out = Command::new(PV)
        .arg(flag)
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn {PV} {flag}: {e}"));
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        out.status.code().unwrap_or(-1),
    )
}

/// The ambiguous form this ticket exists to kill: a line that is EXACTLY the
/// binary name followed by a semver, which is byte-for-byte what `pv(1)` and the
/// crates.io `pv` also print.
fn is_bare_name_and_semver(line: &str) -> bool {
    let mut fields = line.split_whitespace();
    let (Some(name), Some(ver), None) = (fields.next(), fields.next(), fields.next()) else {
        return false;
    };
    name == "pv" && ver.split('.').count() == 3
}

#[test]
fn version_flag_exits_zero_on_stdout() {
    let (stdout, _stderr, rc) = run("--version");
    assert_eq!(rc, 0, "pv --version must exit 0, got {rc}");
    assert!(
        !stdout.trim().is_empty(),
        "pv --version must print to stdout"
    );
}

/// RED before the fix: `pv 0.63.0` IS the bare form.
#[test]
fn long_version_is_not_the_bare_name_and_semver() {
    let (stdout, _, _) = run("--version");
    let first = stdout.lines().next().unwrap_or_default();
    assert!(
        !is_bare_name_and_semver(first),
        "pv --version first line is `{first}` — that is indistinguishable from \
         pv(1) the pipe viewer. It must name this tool."
    );
}

/// RED before the fix: the string `provable-contracts` never appeared.
#[test]
fn long_version_names_what_this_tool_is() {
    let (stdout, _, _) = run("--version");
    let lower = stdout.to_lowercase();
    assert!(
        lower.contains("provable-contracts"),
        "pv --version must say this is the provable-contracts verifier; got:\n{stdout}"
    );
    assert!(
        lower.contains("aprender"),
        "pv --version must name the project it belongs to; got:\n{stdout}"
    );
}

/// RED before the fix: nothing identified the shipping crate, so a user who
/// wanted to uninstall or report a bug had no name to use.
#[test]
fn long_version_names_the_shipping_crate() {
    let (stdout, _, _) = run("--version");
    assert!(
        stdout.contains("aprender-contracts-cli"),
        "pv --version must name the crate that ships it; got:\n{stdout}"
    );
}

/// RED before the fix. This is the assertion that speaks to the actual user
/// confusion: someone with BOTH installed needs the line to rule the other out.
#[test]
fn long_version_disclaims_the_pipe_viewer() {
    let (stdout, _, _) = run("--version");
    let lower = stdout.to_lowercase();
    assert!(
        lower.contains("pipe viewer"),
        "pv --version must explicitly disclaim pv(1), the pipe viewer, because \
         a user with both installed cannot otherwise tell them apart; got:\n{stdout}"
    );
}

/// `-V` is the glance form. It must still be unambiguous, and it must stay ONE
/// line — a version line nobody reads is no better than an ambiguous one.
#[test]
fn short_version_is_one_unambiguous_line() {
    let (stdout, _, rc) = run("-V");
    assert_eq!(rc, 0, "pv -V must exit 0");
    let lines: Vec<&str> = stdout.trim_end().lines().collect();
    assert_eq!(
        lines.len(),
        1,
        "pv -V must stay a single line, got {}:\n{stdout}",
        lines.len()
    );
    assert!(
        !is_bare_name_and_semver(lines[0]),
        "pv -V is `{}` — bare name plus semver, same as pv(1)",
        lines[0]
    );
    assert!(
        lines[0].to_lowercase().contains("provable-contracts"),
        "pv -V must name the tool; got `{}`",
        lines[0]
    );
}

/// Regression guard, not a falsifier: `scripts/pv_bin.sh` proves a resolved
/// binary is the HEAD build by comparing the semver from `pv --version` against
/// the version the tree declares. Adding identity text must not break that
/// parse, so the semver stays the SECOND whitespace field of the FIRST line.
#[test]
fn semver_stays_the_second_field_of_the_first_line() {
    let (stdout, _, _) = run("--version");
    let first = stdout.lines().next().unwrap_or_default();
    let mut fields = first.split_whitespace();
    assert_eq!(
        fields.next(),
        Some("pv"),
        "first field must be the tool name"
    );
    assert_eq!(
        fields.next(),
        Some(VERSION),
        "second field of `pv --version` must be the bare declared semver \
         (scripts/pv_bin.sh parses it positionally); got line `{first}`"
    );
}

/// Both flags must agree on the version and on the identity, so that whichever
/// one a user or script reaches for gives the same answer.
#[test]
fn short_and_long_version_agree_on_the_first_line() {
    let (long, _, _) = run("--version");
    let (short, _, _) = run("-V");
    assert_eq!(
        long.lines().next(),
        short.lines().next(),
        "pv -V and pv --version must share a first line"
    );
}
