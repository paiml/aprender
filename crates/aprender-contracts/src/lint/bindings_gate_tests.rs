//! ONT-3a `bindings` gate. Each test copies `tests/fixtures/ont/code-bound` (a one-member workspace whose registry
//! binds five present items and one ghost, `kern::nn::functional::no_such_function`) and varies one input.

use std::path::{Path, PathBuf};

use super::*;
use crate::lint::{run_named_gate, NamedGateOutcome, NAMED_GATES};

const GHOST: &str = "kern::nn::functional::no_such_function";

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/ont/code-bound")
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        let dst = to.join(e.file_name());
        if e.file_type().unwrap().is_dir() {
            copy_dir(&e.path(), &dst);
        } else {
            std::fs::copy(e.path(), dst).unwrap();
        }
    }
}

/// A private copy of the fixture workspace; returns (guard, contract dir).
fn workspace() -> (tempfile::TempDir, PathBuf) {
    let tmp = tempfile::tempdir().unwrap();
    copy_dir(&fixture(), tmp.path());
    let contracts = tmp.path().join("contracts");
    (tmp, contracts)
}

fn allow(contracts: &Path, json: &str) {
    std::fs::write(contracts.join(ALLOWLIST), json).unwrap();
}

fn entry(symbol: &str) -> String {
    format!(r##"{{"symbol": "{symbol}", "reason": "ghost", "ticket": "#1"}}"##)
}

fn ran(outcome: RatchetOutcome) -> (GateResult, Vec<LintFinding>) {
    match outcome {
        RatchetOutcome::Ran { result, findings } => (*result, findings),
        RatchetOutcome::Declined(why) => panic!("declined: {why}"),
    }
}

fn counters(r: &GateResult) -> BindingsCounters {
    match &r.extra {
        Some(GateExtra::Bindings(c)) => (**c).clone(),
        other => panic!("no bindings counters: {other:?}"),
    }
}

fn rules(findings: &[LintFinding]) -> Vec<&str> {
    findings.iter().map(|f| f.rule_id.as_str()).collect()
}

#[test]
fn an_unallowlisted_ghost_is_rejected_by_name() {
    let (_g, c) = workspace();
    let (r, f) = ran(run_bindings_gate(&c));
    assert_eq!(r.verdict, Verdict::Fail);
    assert_eq!(rules(&f), ["PV-ONT-028"]);
    assert!(f[0].message.contains(GHOST), "{}", f[0].message);
    let k = counters(&r);
    assert_eq!(
        (k.checked, k.resolved, k.ghosts, k.allowlisted),
        (6, 5, 1, 0)
    );
    assert_eq!(k.pc_resolver, "fired");
}

#[test]
fn an_allowlisted_ghost_passes_and_is_counted() {
    let (_g, c) = workspace();
    allow(&c, &format!(r#"{{"entries": [{}]}}"#, entry(GHOST)));
    let (r, f) = ran(run_bindings_gate(&c));
    assert!(f.is_empty(), "{f:?}");
    assert_eq!(r.verdict, Verdict::Pass);
    assert_eq!(counters(&r).allowlisted, 1);
}

#[test]
fn an_allowlist_entry_for_a_resolving_symbol_is_stale() {
    let (_g, c) = workspace();
    allow(
        &c,
        &format!(
            r#"{{"entries": [{}, {}]}}"#,
            entry(GHOST),
            entry("kern::helper")
        ),
    );
    let (r, f) = ran(run_bindings_gate(&c));
    assert_eq!(r.verdict, Verdict::Fail);
    assert_eq!(rules(&f), ["PV-ONT-029"]);
    assert!(f[0].message.contains("kern::helper"));
}

#[test]
fn an_allowlist_entry_no_registry_binds_is_stale() {
    let (_g, c) = workspace();
    allow(
        &c,
        &format!(
            r#"{{"entries": [{}, {}]}}"#,
            entry(GHOST),
            entry("kern::never_bound")
        ),
    );
    let (_, f) = ran(run_bindings_gate(&c));
    assert_eq!(rules(&f), ["PV-ONT-029"]);
}

#[test]
fn a_malformed_or_duplicated_allowlist_is_rejected() {
    for bad in [
        "not json".to_string(),
        r#"{"entries": [{"symbol": "x"}]}"#.to_string(),
        format!(r#"{{"entries": [{}], "extra": 1}}"#, entry(GHOST)),
        format!(r#"{{"entries": [{}, {}]}}"#, entry(GHOST), entry(GHOST)),
        r##"{"entries": [{"symbol": "kern::nn::functional::no_such_function", "reason": " ", "ticket": "#1"}]}"##
            .to_string(),
    ] {
        let (_g, c) = workspace();
        allow(&c, &bad);
        let (r, f) = ran(run_bindings_gate(&c));
        assert_eq!(r.verdict, Verdict::Fail, "{bad}");
        assert!(rules(&f).contains(&"PV-ONT-030"), "{bad}: {f:?}");
    }
}

#[test]
fn a_not_implemented_binding_claims_no_code() {
    let (_g, c) = workspace();
    let reg = c.join("binding.yaml");
    let text = std::fs::read_to_string(&reg).unwrap();
    let marked = text.replace(
        "  function: no_such_function\n  status: implemented",
        "  function: no_such_function\n  status: not_implemented",
    );
    assert_ne!(text, marked, "the fixture's ghost row moved");
    std::fs::write(&reg, marked).unwrap();
    let (r, f) = ran(run_bindings_gate(&c));
    assert!(f.is_empty(), "{f:?}");
    assert_eq!(counters(&r).checked, 5);
}

#[test]
fn a_corpus_outside_a_workspace_declines() {
    let (g, c) = workspace();
    std::fs::remove_file(g.path().join("Cargo.toml")).unwrap();
    match run_bindings_gate(&c) {
        RatchetOutcome::Declined(why) => assert!(why.contains("not at a workspace root"), "{why}"),
        RatchetOutcome::Ran { .. } => panic!("measured a corpus with no workspace"),
    }
}

#[test]
fn a_corpus_with_no_implemented_binding_declines() {
    let (_g, c) = workspace();
    let reg = c.join("binding.yaml");
    let text = std::fs::read_to_string(&reg).unwrap();
    std::fs::write(&reg, text.replace("status: implemented", "status: pending")).unwrap();
    assert!(matches!(run_bindings_gate(&c), RatchetOutcome::Declined(_)));
}

#[test]
fn the_gate_runs_alone_under_its_name() {
    assert!(NAMED_GATES.contains(&GATE));
    let (_g, c) = workspace();
    match run_named_gate(&c, GATE) {
        NamedGateOutcome::Ratchet(RatchetOutcome::Ran { result, .. }) => {
            assert_eq!(result.name, GATE);
        }
        _ => panic!("`bindings` did not run as a ratchet gate"),
    }
}

/// The enforcing run: CI's `--lib` job is where this gate binds the repo, since no workflow calls `make contracts`.
/// A new ghost fails here by name, and a fixed ghost fails here until its allowlist entry is removed.
#[test]
fn the_repo_corpus_has_no_unallowlisted_ghost_and_no_stale_entry() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts");
    let (r, f) = ran(run_bindings_gate(&repo));
    assert!(
        f.is_empty(),
        "{:#?}",
        f.iter().map(|x| &x.message).collect::<Vec<_>>()
    );
    let k = counters(&r);
    assert_eq!((k.ghosts, k.stale_allowlist), (0, 0));
    assert!(
        k.resolved > 200 && k.crates_scanned > 1,
        "the walk measured little: {k:?}"
    );
}
