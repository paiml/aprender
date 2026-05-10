//! FALSIFY-APR-TOK-PAR-004 — `apr tokenize encode-corpus --help` MUST
//! advertise the `--num-workers` flag introduced in issue #1547.
//!
//! In-process clap rendering via `Cli::command()` overflows the default
//! test-thread stack on this binary's command surface (a known clap-derive
//! cost; see `commands/tokenize.rs::tests` comment). This integration test
//! invokes the compiled `apr` binary directly — the same surface the
//! operator hits — and greps `--help` output for the flag.
//!
//! Pinning this test means a future refactor that drops the flag (e.g.
//! when migrating subcommand definitions) tripwires loudly instead of
//! silently re-introducing the 47-hour single-thread regression.
//!
//! Contract: contracts/apr-tokenize-parallel-bpe-v1.yaml v1.1.0.

#![cfg(feature = "training")]

use std::process::Command;

fn apr_binary() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_apr"));
    cmd.env("NO_COLOR", "1");
    cmd
}

#[test]
fn falsify_apr_tok_par_004_help_advertises_num_workers_flag() {
    let out = apr_binary()
        .args(["tokenize", "encode-corpus", "--help"])
        .output()
        .expect("run apr tokenize encode-corpus --help");
    assert!(
        out.status.success(),
        "apr tokenize encode-corpus --help must exit 0; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("--num-workers"),
        "FALSIFY-APR-TOK-PAR-004: --help must advertise --num-workers \
         (issue #1547); got:\n{stdout}"
    );
}

#[test]
fn falsify_apr_tok_par_004_help_documents_default_resolution() {
    // Anti-regression: the flag's help text should mention the default
    // resolution mechanism so an operator skimming `--help` sees that
    // omitting the flag yields full-CPU parallelism (not single-thread).
    let out = apr_binary()
        .args(["tokenize", "encode-corpus", "--help"])
        .output()
        .expect("run apr tokenize encode-corpus --help");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("available_parallelism") || stdout.contains("logical CPU"),
        "--help should explain the default num_workers value; got:\n{stdout}"
    );
}
