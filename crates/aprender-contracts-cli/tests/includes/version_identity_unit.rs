//! Unit half of the #2559 falsifier — see `tests/version_identity.rs` for the
//! integration half that spawns the real binary.
//!
//! WHY BOTH. CI's `workspace-test` job runs `--lib` across the workspace plus ONE
//! explicit line naming individual `--test` targets (`.github/workflows/ci.yml`).
//! A new `tests/*.rs` file is DARK until it is added to that line, and only one
//! PR at a time can edit it without conflicting. These unit tests live in the
//! library, so they run under `--lib` unconditionally and the guarantee does not
//! depend on winning that race.
//!
//! clap renders `--version` from `Command::render_long_version()` and `-V` from
//! `render_version()`, so asserting on those is asserting on the same strings the
//! binary prints — `version_identity.rs` confirms that end to end.

use clap::CommandFactory;

use super::Cli;

fn short() -> String {
    Cli::command().render_version()
}

fn long() -> String {
    Cli::command().render_long_version()
}

/// The ambiguous form: exactly `pv <semver>`, byte-identical to what `pv(1)` the
/// pipe viewer and the crates.io `pv` crate print.
fn is_bare_name_and_semver(line: &str) -> bool {
    let mut fields = line.split_whitespace();
    let (Some(name), Some(ver), None) = (fields.next(), fields.next(), fields.next()) else {
        return false;
    };
    name == "pv" && ver.split('.').count() == 3
}

#[test]
fn long_version_is_not_bare_name_and_semver() {
    let v = long();
    let first = v.lines().next().unwrap_or_default();
    assert!(
        !is_bare_name_and_semver(first),
        "`{first}` is indistinguishable from pv(1), the pipe viewer"
    );
}

#[test]
fn long_version_identifies_this_tool() {
    let v = long();
    let lower = v.to_lowercase();
    assert!(
        lower.contains("provable-contracts"),
        "pv --version must say what this tool is; got:\n{v}"
    );
    assert!(
        lower.contains("aprender"),
        "pv --version must name the project; got:\n{v}"
    );
    assert!(
        v.contains("aprender-contracts-cli"),
        "pv --version must name the shipping crate; got:\n{v}"
    );
    assert!(
        lower.contains("pipe viewer"),
        "pv --version must disclaim pv(1), the pipe viewer; got:\n{v}"
    );
}

#[test]
fn short_version_is_one_unambiguous_line() {
    let v = short();
    let lines: Vec<&str> = v.trim_end().lines().collect();
    assert_eq!(lines.len(), 1, "pv -V must stay one line; got:\n{v}");
    assert!(
        !is_bare_name_and_semver(lines[0]),
        "pv -V is bare: `{}`",
        lines[0]
    );
    assert!(
        lines[0].to_lowercase().contains("provable-contracts"),
        "pv -V must name the tool; got `{}`",
        lines[0]
    );
}

/// `scripts/pv_bin.sh` proves a resolved binary is the HEAD build by reading the
/// semver positionally out of `pv --version`. Identity text must not displace it.
#[test]
fn semver_stays_the_second_field_of_the_first_line() {
    let v = long();
    let first = v.lines().next().unwrap_or_default();
    let mut fields = first.split_whitespace();
    assert_eq!(fields.next(), Some("pv"));
    assert_eq!(
        fields.next(),
        Some(env!("CARGO_PKG_VERSION")),
        "scripts/pv_bin.sh reads field 2 of line 1; got `{first}`"
    );
}

#[test]
fn short_and_long_agree_on_the_first_line() {
    assert_eq!(long().lines().next(), short().lines().next());
}
