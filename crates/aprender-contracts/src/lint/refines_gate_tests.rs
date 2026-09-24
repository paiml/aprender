//! The gate over the committed `tests/fixtures/ont/refines-*` corpora, copied into a tempdir. The witnesses here are
//! written by hand from the fixture's own clauses, so each case differs from a passing one by exactly what it names;
//! the CLI suite (`ont4e_refines_gate`) runs the same fixtures through the real `pv-sat`.

use std::path::{Path, PathBuf};

use super::*;
use crate::ontology::liskov::{Kind, Obligation, PairWitness, Step};

fn fixture(name: &str) -> tempfile::TempDir {
    let src: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/ont")
        .join(name);
    let dir = tempfile::tempdir().expect("tempdir");
    for entry in std::fs::read_dir(src).expect("fixture dir") {
        let entry = entry.expect("entry");
        std::fs::copy(entry.path(), dir.path().join(entry.file_name())).expect("copy");
    }
    dir
}

fn ran(outcome: RefinesOutcome) -> (GateResult, Vec<LintFinding>, RefinesCounters) {
    match outcome {
        RefinesOutcome::Ran { result, findings } => {
            let c = match &result.extra {
                Some(GateExtra::Refines(c)) => (**c).clone(),
                other => panic!("no refines counters: {other:?}"),
            };
            (*result, findings, c)
        }
        other => panic!("expected a verdict, got {}", why(&other)),
    }
}

fn chain(clause: &str) -> Vec<Step> {
    vec![Step {
        clause: clause.into(),
        from: clause.into(),
    }]
}

/// The honest witness for `refines-ok`: every obligation is a one-step chain from the same-atom clause.
fn write_ok_witness(dir: &Path, pc_reasoner: &str) {
    let (_, edges) = match typed_graph(dir) {
        TypedGraph::Read { ids, edges } => (ids, edges),
        _ => panic!("fixture did not read"),
    };
    let pairs = liskov_corpus(&corpus_documents(dir), &edges).pairs;
    let sha = liskov_sha256(&pairs);
    let w = LiskovWitness {
        liskov_sha256: sha.clone(),
        pairs_checked: 1,
        reasoner_git_sha: None,
        pc_reasoner: pc_reasoner.into(),
        pairs: vec![PairWitness {
            a: "a".into(),
            b: "b".into(),
            obligations: vec![
                Obligation {
                    kind: Kind::Pre,
                    chain: Some(chain("PRE-1")),
                    counter_model: None,
                },
                Obligation {
                    kind: Kind::Post,
                    chain: Some(chain("POST-1")),
                    counter_model: None,
                },
                Obligation {
                    kind: Kind::Inv,
                    chain: Some(chain("INV-1")),
                    counter_model: None,
                },
            ],
        }],
    };
    let path = liskov_witness_path(dir, &sha);
    std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
    std::fs::write(path, serde_json::to_string_pretty(&w).expect("ser")).expect("write");
}

#[test]
fn a_liskov_pair_with_an_honest_witness_passes() {
    let dir = fixture("refines-ok");
    write_ok_witness(dir.path(), FIRED);
    let (r, findings, c) = ran(run_refines_gate(dir.path()));
    assert!(findings.is_empty(), "{findings:?}");
    assert_eq!(r.verdict, Verdict::Pass);
    assert_eq!(c.liskov_pairs_checked, 1);
    assert_eq!(c.pc_checker, FIRED);
    assert_eq!((c.requires_n, c.ensures_n, c.invariants_n), (2, 3, 2));
}

#[test]
fn a_witness_whose_reasoner_control_did_not_fire_is_not_a_verdict() {
    let dir = fixture("refines-ok");
    write_ok_witness(dir.path(), "not-fired");
    let outcome = run_refines_gate(dir.path());
    assert!(
        matches!(
            outcome,
            RefinesOutcome::PositiveControlFailed {
                which: "pc_reasoner",
                ..
            }
        ),
        "{}",
        why(&outcome)
    );
    assert_eq!(
        decline_reason(&outcome),
        Some(Reason::PositiveControlFailed)
    );
}

#[test]
fn no_witness_is_stale_and_names_make_contracts() {
    let dir = fixture("refines-ok");
    let outcome = run_refines_gate(dir.path());
    assert!(matches!(outcome, RefinesOutcome::WitnessStale { .. }));
    assert_eq!(decline_reason(&outcome), Some(Reason::WitnessStale));
    assert!(
        why(&outcome).contains("make contracts"),
        "{}",
        why(&outcome)
    );
}

#[test]
fn a_legacy_pair_is_pass_with_nothing_checked_and_needs_no_witness() {
    let (r, findings, c) = ran(run_refines_gate(fixture("refines-legacy").path()));
    assert!(findings.is_empty(), "{findings:?}");
    assert_eq!(r.verdict, Verdict::Pass);
    assert_eq!((c.liskov_pairs_checked, c.liskov_pairs_legacy), (0, 1));
    assert!(c.witness.is_none());
}

#[test]
fn a_prose_clause_is_unknown_prose_naming_it() {
    let (r, findings, c) = ran(run_refines_gate(fixture("refines-prose").path()));
    assert!(findings.is_empty(), "{findings:?}");
    assert_eq!(r.verdict, Verdict::Unknown(Reason::Prose));
    assert_eq!(c.prose_clauses, ["a.requires PRE-1"]);
    assert_eq!(
        explain(&r, &findings),
        ["refines: Unknown{Prose} — prose clause(s), not checked: a.requires PRE-1"]
    );
}

#[test]
fn prose_above_the_baseline_is_rejected_and_at_it_is_not() {
    let dir = fixture("refines-prose");
    let baseline = dir.path().join("lint-baseline.json");
    std::fs::write(&baseline, r#"{"ont": {"liskov_prose": 0}}"#).expect("write");
    let (r, findings, c) = ran(run_refines_gate(dir.path()));
    assert_eq!(r.verdict, Verdict::Fail);
    assert_eq!(c.liskov_prose_baseline, Some(0));
    assert_eq!(findings[0].rule_id, "PV-ONT-027", "{findings:?}");

    std::fs::write(&baseline, r#"{"ont": {"liskov_prose": 1}}"#).expect("write");
    let (r, findings, _) = ran(run_refines_gate(dir.path()));
    assert!(findings.is_empty(), "{findings:?}");
    assert_eq!(r.verdict, Verdict::Unknown(Reason::Prose));
}

#[test]
fn a_malformed_clause_is_rejected_with_its_file() {
    let dir = fixture("refines-legacy");
    let b = dir.path().join("b.yaml");
    let text = std::fs::read_to_string(&b).expect("b.yaml");
    std::fs::write(
        &b,
        format!("{text}requires:\n  - id: PRE-1\n    statement: s\n    formal_status: parsed\n"),
    )
    .expect("write");
    let (r, findings, _) = ran(run_refines_gate(dir.path()));
    assert_eq!(r.verdict, Verdict::Fail);
    assert_eq!(findings[0].rule_id, "PV-ONT-026", "{findings:?}");
    assert!(findings[0].message.contains("PRE-1"), "{findings:?}");
}

#[test]
fn no_sigma_declines_no_checkable() {
    let dir = tempfile::tempdir().expect("tempdir");
    let outcome = run_refines_gate(dir.path());
    assert!(matches!(outcome, RefinesOutcome::NoSigma));
    assert_eq!(decline_reason(&outcome), Some(Reason::NoCheckable));
}
