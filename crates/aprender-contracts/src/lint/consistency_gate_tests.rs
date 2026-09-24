//! The gate over real corpora: `tests/fixtures/ont/relations-ok` copied into a tempdir, where `a contradicts d` and
//! both are live — inconsistent as it stands, consistent once that line goes. Witnesses are built here from the
//! gate's own reading of the graph, so each case differs from a passing one by exactly the defect it names.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use super::*;
use crate::ontology::witness::{Core, Model, Step, TypedEdge};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/ont/relations-ok")
}

/// A copy of the fixture; `consistent` drops `a contradicts d`.
fn corpus(consistent: bool) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    for entry in std::fs::read_dir(fixture()).expect("fixture dir") {
        let entry = entry.expect("entry");
        let mut text = std::fs::read_to_string(entry.path()).expect("fixture file");
        if consistent && entry.file_name() == "a.yaml" {
            assert!(
                text.contains("  contradicts: [d]\n"),
                "the fixture changed under this test"
            );
            text = text.replace("  contradicts: [d]\n", "");
        }
        std::fs::write(dir.path().join(entry.file_name()), text).expect("copy");
    }
    dir
}

fn graph(dir: &Path) -> (BTreeSet<String>, BTreeSet<TypedEdge>) {
    match typed_graph(dir) {
        TypedGraph::Read { ids, edges } => (ids, edges),
        other => panic!(
            "fixture did not read: {}",
            matches!(other, TypedGraph::NoSigma)
        ),
    }
}

/// Write a witness for the corpus as it stands, then let `edit` spoil it.
fn witness(dir: &Path, result: WitnessResult, edit: impl FnOnce(&mut Witness)) {
    let (ids, edges) = graph(dir);
    let cs = ClauseSet::from_graph(&ids, &edges);
    let mut w = Witness {
        census_id_set_sha256: census_id_set_sha256(&ids),
        relations_sha256: relations_sha256(&edges),
        reasoner_git_sha: None,
        checkable_n: cs.checkable_n(),
        result,
        pc_reasoner: FIRED.into(),
        cpu_ms: 0,
    };
    let path = witness_path(dir, &w.relations_sha256);
    edit(&mut w);
    std::fs::create_dir_all(path.parent().expect("witness dir")).expect("mkdir");
    std::fs::write(path, serde_json::to_string_pretty(&w).expect("json")).expect("write");
}

fn model(false_vars: &[&str]) -> WitnessResult {
    WitnessResult::Model(Model {
        false_vars: false_vars.iter().map(|v| (*v).to_string()).collect(),
    })
}

/// The true core of the fixture: `a` and `d` are both asserted and contradict.
fn a_d_core() -> WitnessResult {
    WitnessResult::UnsatCore(Core {
        steps: vec![Step::Unit("a".into()), Step::Unit("d".into())],
        conflict: ("a".into(), "d".into()),
    })
}

fn ran(dir: &Path) -> (Box<GateResult>, Vec<LintFinding>) {
    match run_consistency_gate(dir) {
        ConsistencyOutcome::Ran { result, findings } => (result, findings),
        other => panic!("expected a verdict, got {other:?}"),
    }
}

fn rules(findings: &[LintFinding]) -> Vec<&str> {
    findings.iter().map(|f| f.rule_id.as_str()).collect()
}

#[test]
fn a_consistent_corpus_with_a_checked_model_passes() {
    let dir = corpus(true);
    witness(dir.path(), model(&[]), |_| {});
    let (result, findings) = ran(dir.path());
    assert!(findings.is_empty(), "{findings:?}");
    assert!(result.passed);
    let Some(GateExtra::Consistency {
        checkable_n,
        units,
        pc_checker,
        core,
        witness,
        ..
    }) = &result.extra
    else {
        panic!("no consistency extra")
    };
    // a⇒b, a⇒c; units a, b, d (c is superseded by b).
    assert_eq!((*checkable_n, *units), (2, 3));
    assert_eq!(pc_checker, FIRED);
    assert!(core.is_empty());
    assert_eq!(
        (
            witness.kind.as_str(),
            witness.pc_reasoner.as_str(),
            witness.stale
        ),
        ("model", FIRED, false)
    );
}

#[test]
fn an_inconsistent_corpus_fails_with_its_core_named() {
    let dir = corpus(false);
    witness(dir.path(), a_d_core(), |_| {});
    let (result, findings) = ran(dir.path());
    assert!(!result.passed);
    assert_eq!(rules(&findings), ["PV-ONT-022"]);
    assert!(
        findings[0].message.contains("`a` contradicts `d`"),
        "{}",
        findings[0].message
    );
    let Some(GateExtra::Consistency { core, .. }) = &result.extra else {
        panic!("no consistency extra")
    };
    assert_eq!(core, &["a", "d"]);
}

#[test]
fn a_model_for_an_inconsistent_corpus_is_a_lie() {
    // The reasoner claims SAT where there is none: all-true breaks `a ⊥ d`.
    let dir = corpus(false);
    witness(dir.path(), model(&[]), |_| {});
    let (_, findings) = ran(dir.path());
    assert_eq!(rules(&findings), ["PV-ONT-023"]);
    assert!(findings[0].message.contains("never edit a witness by hand"));
}

#[test]
fn a_tampered_witness_does_not_check() {
    let dir = corpus(true);
    witness(dir.path(), model(&["b"]), |_| {});
    assert_eq!(rules(&ran(dir.path()).1), ["PV-ONT-023"]);

    let dir = corpus(true);
    witness(dir.path(), model(&[]), |w| w.checkable_n += 1);
    assert_eq!(rules(&ran(dir.path()).1), ["PV-ONT-023"]);

    // A core over a clause the graph does not hold: `d` is live, but nothing contradicts it once the line is gone.
    let dir = corpus(true);
    witness(dir.path(), a_d_core(), |_| {});
    assert_eq!(rules(&ran(dir.path()).1), ["PV-ONT-023"]);
}

#[test]
fn a_missing_or_foreign_witness_is_stale_not_a_pass() {
    let dir = corpus(true);
    let out = run_consistency_gate(dir.path());
    assert!(
        matches!(out, ConsistencyOutcome::WitnessStale { .. }),
        "{out:?}"
    );
    assert_eq!(decline_reason(&out), Some(Reason::WitnessStale));
    assert!(why(&out).contains("make contracts"), "{}", why(&out));

    let dir = corpus(true);
    witness(dir.path(), model(&[]), |w| {
        w.census_id_set_sha256 = "0".repeat(64)
    });
    let out = run_consistency_gate(dir.path());
    assert!(
        matches!(out, ConsistencyOutcome::WitnessStale { .. }),
        "{out:?}"
    );

    // A witness for the inconsistent graph does not speak for the consistent one: different relations, different name.
    let dir = corpus(false);
    witness(dir.path(), a_d_core(), |_| {});
    std::fs::write(
        dir.path().join("a.yaml"),
        std::fs::read_to_string(dir.path().join("a.yaml"))
            .expect("a")
            .replace("  contradicts: [d]\n", ""),
    )
    .expect("edit");
    assert!(matches!(
        run_consistency_gate(dir.path()),
        ConsistencyOutcome::WitnessStale { .. }
    ));

    let dir = corpus(true);
    let (_, edges) = graph(dir.path());
    let path = witness_path(dir.path(), &relations_sha256(&edges));
    std::fs::create_dir_all(path.parent().expect("dir")).expect("mkdir");
    std::fs::write(&path, "{\"not\": \"a witness\"}").expect("write");
    assert!(matches!(
        run_consistency_gate(dir.path()),
        ConsistencyOutcome::WitnessStale { .. }
    ));
}

#[test]
fn a_reasoner_whose_control_did_not_fire_gets_no_verdict() {
    let dir = corpus(true);
    witness(dir.path(), model(&[]), |w| w.pc_reasoner = "skipped".into());
    let out = run_consistency_gate(dir.path());
    assert!(
        matches!(
            out,
            ConsistencyOutcome::PositiveControlFailed {
                which: "pc_reasoner",
                ..
            }
        ),
        "{out:?}"
    );
    assert_eq!(decline_reason(&out), Some(Reason::PositiveControlFailed));
}

#[test]
fn zero_clauses_and_no_sigma_decline() {
    let dir = corpus(true);
    for name in ["a.yaml", "b.yaml"] {
        let p = dir.path().join(name);
        let text = std::fs::read_to_string(&p).expect("read");
        let cut = text.find("relations:").expect("relations block");
        std::fs::write(&p, &text[..cut]).expect("write");
    }
    let out = run_consistency_gate(dir.path());
    assert!(
        matches!(
            out,
            ConsistencyOutcome::NoCheckable {
                contracts_checked: 4
            }
        ),
        "{out:?}"
    );
    assert_eq!(decline_reason(&out), Some(Reason::NoCheckable));

    let dir = corpus(true);
    std::fs::remove_file(dir.path().join("ontology.yaml")).expect("rm");
    let out = run_consistency_gate(dir.path());
    assert!(matches!(out, ConsistencyOutcome::NoSigma), "{out:?}");
    assert_eq!(decline_reason(&out), Some(Reason::NoCheckable));
}

#[test]
fn a_malformed_sigma_is_an_error_not_a_decline() {
    let dir = corpus(true);
    std::fs::write(dir.path().join("ontology.yaml"), "roles: [unclosed\n").expect("write");
    let out = run_consistency_gate(dir.path());
    assert!(matches!(out, ConsistencyOutcome::Malformed(_)), "{out:?}");
    assert_eq!(decline_reason(&out), None);
}
