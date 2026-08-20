//! FALSIFY-DATA-REACH-001: every alimentar command must be reachable via `apr`.
//!
//! APR-MONO consolidated alimentar in-tree as `crates/aprender-data`, but the
//! capability stayed reachable only through the standalone `alimentar` binary:
//! `apr data` shipped 5 commands against alimentar's 20, so 18 had no route
//! through `apr` at all. Retiring the binary before exposing them would have
//! removed the capability rather than relocating it -- which is the ordering
//! error this test exists to make impossible to repeat.
//!
//! Asserts on `--help` OUTPUT of the built binary rather than on the Rust enum,
//! because the enum being correct is not the claim. The claim is that a user
//! typing `apr data x <cmd>` reaches the implementation.

use std::process::Command;

fn apr() -> Command {
    Command::new(env!("CARGO_BIN_EXE_apr"))
}

/// Commands alimentar's own CLI declares. If alimentar gains one and `apr` does
/// not, this list is the thing that has to change -- and changing it is the
/// prompt to check reach.
const ALIMENTAR_COMMANDS: &[&str] = &[
    "convert",
    "info",
    "head",
    "schema",
    "mix",
    "fim",
    "dedup",
    "filter-text",
    "view",
    "import",
    "registry",
    "drift",
    "quality",
    "fed",
    "repl",
];

#[test]
fn every_alimentar_command_is_reachable_through_apr_data() {
    let out = apr()
        .args(["data", "x", "--help"])
        .output()
        .expect("apr data x --help");
    assert!(out.status.success(), "apr data x --help must succeed");
    let help = String::from_utf8_lossy(&out.stdout);

    // Control first: an empty or error help body would make every assertion
    // below pass vacuously.
    assert!(
        help.contains("Commands:"),
        "help body has no command list, so the assertions below prove nothing: {help}"
    );

    let missing: Vec<&str> = ALIMENTAR_COMMANDS
        .iter()
        .copied()
        .filter(|c| !help.contains(c))
        .collect();
    assert!(
        missing.is_empty(),
        "these alimentar commands are not reachable via `apr data x`: {missing:?}"
    );
}

#[test]
fn apr_data_x_actually_executes_rather_than_only_listing() {
    // Reach is not the same as dispatch. A subcommand can appear in --help and
    // still be wired to nothing, which is what makes a help-only assertion weak.
    let dir = std::env::temp_dir().join("apr_data_reach_test");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let f = dir.join("t.jsonl");
    std::fs::write(
        &f,
        "{\"input\":\"a\",\"label\":0}\n{\"input\":\"b\",\"label\":1}\n",
    )
    .expect("fixture");

    let out = apr()
        .args(["data", "x", "info"])
        .arg(&f)
        .output()
        .expect("apr data x info");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "apr data x info failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout.contains("Rows: 2"),
        "expected alimentar's own info output with the real row count, got: {stdout}"
    );

    // And it must FAIL on a missing file rather than reporting success -- an
    // exit code of 0 on nonexistent input is how a passthrough silently becomes
    // a no-op.
    let bad = apr()
        .args(["data", "x", "info", "/nonexistent-apr-data-reach.jsonl"])
        .output()
        .expect("run");
    assert!(
        !bad.status.success(),
        "a missing input must fail; a passthrough that always exits 0 is not wired to anything"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// A global flag before the subcommand must not shift the dispatch.
#[test]
fn a_global_flag_before_the_subcommand_does_not_break_dispatch() {
    let out = apr()
        .args(["--json", "data", "x", "--help"])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "`apr --json data x --help` failed; argv rewriting is index-sensitive again"
    );
}
