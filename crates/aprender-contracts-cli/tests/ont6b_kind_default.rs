//! ONT-6b (infra#751) — a kind-less contract that fails a kernel-only rule says so.
//!
//! ONT-001 v4.9 §5 ONT-6b RED, verbatim: "`pv validate` on a contract with no `metadata.kind`
//! and none of `equations` / `proof_obligations` / `falsification_tests` / `kani_harnesses` →
//! exit 1, and the first `[ERROR]` line the default caused (`SCHEMA-003` or `PROVABILITY-001`,
//! whichever fires first) carries `no metadata.kind, judged kernel by default`, once; the same
//! bytes with `metadata.kind: kernel` → the same errors and no such text (it was judged by the
//! kind it declares); a kind-less contract carrying all four kernel blocks → exit 0 and no
//! mention of kind; with `metadata.kind: pattern` the kernel rules do not apply and the text
//! never appears."
//!
//! WHY THIS IS A CASE TABLE AND NOT ONE ASSERTION. The defect was not a missing message, it was
//! an ABSENT DISTINCTION: measured 2026-09-19 on pv 0.68.1, `kindless-failing.yaml` and
//! `declared-kernel-failing.yaml` — which differ by one line — produced BYTE-IDENTICAL output.
//! A single "the text appears" assertion is satisfied by printing it unconditionally, which
//! re-creates the same blindness pointing the other way. Each arm below is therefore a fixture
//! that the other arms' failure modes would break:
//!
//!   kindless-failing        the text, on the FIRST error, exactly once   (drop it      -> RED)
//!   declared-kernel-failing the same errors, and NO text                 (print always -> RED)
//!   kindless-passing        exit 0, and NO text                          (print on ok  -> RED)
//!   pattern                 kernel rules never run, and NO text          (printer-side -> RED)
//!   once                    one occurrence, not one per kernel rule      (decorate all -> RED)
//!
//! The pair (kindless-failing, declared-kernel-failing) is the discrimination proof: the two
//! files differ by `kind: kernel` and nothing else, so any implementation that cannot tell a
//! DECLARED kernel from a DEFAULTED one fails one of them whatever it prints.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The sentence under test. Written out here rather than imported from
/// `aprender_contracts::schema::validator` on purpose: this test is the CLI's
/// contract with its reader, and importing the constant would make a rename
/// that silently changes user-visible output pass.
const EXPLANATION: &str = "no metadata.kind, judged kernel by default";

fn pv_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pv"))
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/ont/kind-default")
        .join(name)
}

struct Run {
    code: i32,
    out: String,
}

fn validate(name: &str) -> Run {
    let path = fixture(name);
    assert!(path.is_file(), "fixture {name} is tracked and readable");
    let out = Command::new(pv_bin())
        .arg("validate")
        .arg(&path)
        .output()
        .expect("failed to spawn pv");
    let mut merged = String::from_utf8_lossy(&out.stdout).into_owned();
    merged.push_str(&String::from_utf8_lossy(&out.stderr));
    Run {
        code: out.status.code().unwrap_or(-1),
        out: merged,
    }
}

/// The first `[ERROR]` line, which is where the explanation must land.
fn first_error(out: &str) -> &str {
    out.lines()
        .find(|l| l.contains("[ERROR]"))
        .unwrap_or("<no [ERROR] line>")
}

#[test]
fn kindless_failing_names_the_default_on_its_first_error() {
    let r = validate("kindless-failing.yaml");
    assert_eq!(
        r.code, 1,
        "a kind-less contract with no kernel blocks is rejected\n{}",
        r.out
    );
    let first = first_error(&r.out);
    assert!(
        first.contains(EXPLANATION),
        "the FIRST error must carry the explanation; it read:\n  {first}\nfull output:\n{}",
        r.out
    );
}

#[test]
fn the_explanation_is_printed_once_not_once_per_kernel_rule() {
    let r = validate("kindless-failing.yaml");
    let hits = r.out.matches(EXPLANATION).count();
    assert_eq!(
        hits, 1,
        "the default is explained once, not on every kernel-only error (found {hits})\n{}",
        r.out
    );
    // Anti-vacuity: the file really does produce several kernel-only errors, so
    // "once" is a constraint here and not an accident of there being one error.
    let errors = r.out.matches("[ERROR]").count();
    assert!(
        errors > 1,
        "fixture must produce MORE than one error for the `once` assertion to mean anything (found {errors})\n{}",
        r.out
    );
}

#[test]
fn declared_kernel_is_judged_by_the_kind_it_declares_and_says_nothing_about_a_default() {
    let r = validate("declared-kernel-failing.yaml");
    assert_eq!(
        r.code, 1,
        "a declared kernel with no kernel blocks is still rejected\n{}",
        r.out
    );
    assert!(
        !r.out.contains("judged kernel by default"),
        "a DECLARED kernel had no default applied and must not claim one\n{}",
        r.out
    );
}

/// The discrimination proof, stated as its own case: the two fixtures differ by
/// one line, so their outputs must differ in exactly the explanation and be
/// otherwise identical. Before ONT-6b they were byte-identical.
#[test]
fn the_two_failing_fixtures_differ_only_by_the_explanation() {
    let kindless = validate("kindless-failing.yaml");
    let declared = validate("declared-kernel-failing.yaml");
    assert_eq!(
        kindless.code, declared.code,
        "same rules fire on both; only the explanation may differ"
    );
    let stripped = kindless.out.replace(&format!(" ({EXPLANATION})"), "");
    assert_eq!(
        stripped, declared.out,
        "removing the explanation from the kind-less output must reproduce the declared-kernel \
         output exactly — anything else means the default changed which RULES ran, not just what \
         was said about them\nkind-less:\n{}\ndeclared:\n{}",
        kindless.out, declared.out
    );
}

#[test]
fn kindless_but_complete_passes_and_never_mentions_kind() {
    let r = validate("kindless-passing.yaml");
    assert_eq!(
        r.code, 0,
        "a kind-less contract carrying all four kernel blocks is valid — 666 of this corpus's \
         contracts are this shape\n{}",
        r.out
    );
    assert!(
        !r.out.contains("judged kernel by default"),
        "nothing failed, so there is no verdict for the default to explain\n{}",
        r.out
    );
}

#[test]
fn a_declared_non_kernel_kind_never_reaches_the_explanation() {
    let r = validate("pattern.yaml");
    assert_eq!(
        r.code, 0,
        "the kernel rules do not apply to `kind: pattern`\n{}",
        r.out
    );
    assert!(
        !r.out.contains("judged kernel by default"),
        "a pattern contract was never judged as a kernel\n{}",
        r.out
    );
}
