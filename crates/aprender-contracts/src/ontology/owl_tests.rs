//! ONT-2c: the OWL writer's case table. The expected `.ofn` for the fixture Σ is committed beside it
//! (`tests/fixtures/ont/owl/expected.ofn`). The oracle (`tests/oracle/`) re-parses that file with horned-owl
//! and requires the same axiom set, so the writer is pinned from two sides.
//!
//! These tests READ THE TREE (`contracts/ontology.yaml`, `tests/fixtures/ont/owl/`), so
//! scripts/check_tree_reader_tests.sh must find them. Its oracle needs `cfg(test)` in the SAME file as the
//! tree path, hence the redundant inner attribute below. The parent's `#[cfg(test)] #[path]` alone was not seen.
#![cfg(test)]

use std::path::PathBuf;

use super::*;
use crate::ontology::sigma::Sigma;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture_sigma() -> Sigma {
    let text = std::fs::read_to_string(repo().join("tests/fixtures/ont/owl/ontology.yaml"))
        .expect("fixture Σ");
    let s = Sigma::from_yaml(&text).expect("fixture Σ parses");
    s.check_integrity().expect("fixture Σ is well-formed");
    s
}

#[test]
fn ont2c_fixture_writes_exactly_the_committed_axiom_set() {
    let got = to_ofn(&export(&fixture_sigma()).expect("export"));
    let want = std::fs::read_to_string(repo().join("tests/fixtures/ont/owl/expected.ofn"))
        .expect("expected.ofn");
    assert_eq!(
        got, want,
        "the writer drifted from the committed fixture axiom set"
    );
}

#[test]
fn ont2c_two_writes_are_byte_identical() {
    let a = to_ofn(&export(&fixture_sigma()).expect("export"));
    let b = to_ofn(&export(&fixture_sigma()).expect("export"));
    assert_eq!(a, b);
}

#[test]
fn ont2c_acyclic_yields_no_axiom_and_is_accounted() {
    let e = export(&fixture_sigma()).expect("export");
    let ofn = to_ofn(&e);
    for forbidden in [
        "TransitiveObjectProperty",
        "IrreflexiveObjectProperty",
        "AsymmetricObjectProperty",
    ] {
        assert!(
            !ofn.contains(forbidden),
            "`acyclic` must yield no axiom, found {forbidden}"
        );
    }
    // `refines` is still a property with its domain and range; only the flag is unexpressed.
    assert!(e
        .axioms
        .contains(&Axiom::DeclareObjectProperty("refines".into())));
    assert!(e.not_expressed.contains_key("acyclic"));
}

#[test]
fn ont2c_symmetric_is_written() {
    let e = export(&fixture_sigma()).expect("export");
    assert!(e
        .axioms
        .contains(&Axiom::SymmetricObjectProperty("pairs_with".into())));
    assert!(!e
        .axioms
        .contains(&Axiom::SymmetricObjectProperty("binds".into())));
}

#[test]
fn ont2c_undeclared_unexpressed_key_is_refused() {
    let mut s = fixture_sigma();
    s.not_expressible.retain(|n| n.key != "symbols");
    match export(&s) {
        Err(OwlError::Unexpressed { key, .. }) => assert_eq!(key, "symbols"),
        other => panic!("a populated key OWL does not express must be declared: {other:?}"),
    }
}

#[test]
fn ont2c_role_over_an_undeclared_concept_is_refused() {
    let mut s = fixture_sigma();
    s.concepts.remove("Test");
    assert!(matches!(
        export(&s),
        Err(OwlError::UndeclaredConcept { .. })
    ));
}

#[test]
fn ont2c_real_sigma_exports_and_classifies_clean() {
    let text = std::fs::read_to_string(repo().join("contracts/ontology.yaml")).expect("Σ");
    let s = Sigma::from_yaml(&text).expect("Σ parses");
    let e = export(&s).expect("the repository's Σ must be writable as OWL");
    let r = tbox(&e);
    assert!(r.advisory, "classification is advisory, always");
    assert!(r.precondition.holds, "{:?}", r.precondition.refused);
    assert!(r.consistent);
    assert!(
        r.unintended_subsumptions.is_empty(),
        "{:?}",
        r.unintended_subsumptions
    );
}

#[test]
fn ont2c_positive_control_a_planted_subsumption_is_unintended() {
    let mut e = export(&fixture_sigma()).expect("export");
    e.axioms
        .insert(Axiom::SubClassOf("Contract".into(), "Code".into()));
    e.axioms
        .insert(Axiom::SubClassOf("Code".into(), "Test".into()));
    let r = tbox(&e);
    assert!(r.precondition.holds);
    // told edges plus their transitive consequence, none of them intended
    assert_eq!(
        r.unintended_subsumptions,
        vec![
            ("Code".to_string(), "Test".to_string()),
            ("Contract".to_string(), "Code".to_string()),
            ("Contract".to_string(), "Test".to_string()),
        ]
    );
}

#[test]
fn ont2c_an_intended_subsumption_is_not_unintended() {
    let mut e = export(&fixture_sigma()).expect("export");
    e.axioms
        .insert(Axiom::SubClassOf("Contract".into(), "Code".into()));
    e.intended_subsumptions
        .insert(("Contract".into(), "Code".into()));
    assert!(tbox(&e).unintended_subsumptions.is_empty());
}

#[test]
fn ont2c_precondition_refuses_an_undeclared_class_and_claims_no_consistency() {
    let mut e = export(&fixture_sigma()).expect("export");
    e.axioms
        .insert(Axiom::SubClassOf("Contract".into(), "Nothing".into()));
    let r = tbox(&e);
    assert!(!r.precondition.holds);
    assert!(
        !r.consistent,
        "without the precondition the method makes no consistency claim"
    );
    assert!(r.precondition.refused[0].contains("Nothing"));
}

#[test]
fn ont2c_tracked_ofn_and_report_are_fresh() {
    // R-18: the tracked artifacts are what the writer produces from the tracked Σ.
    let text = std::fs::read_to_string(repo().join("contracts/ontology.yaml")).expect("Σ");
    let e = export(&Sigma::from_yaml(&text).expect("Σ")).expect("export");
    let ofn = std::fs::read_to_string(repo().join("contracts/ontology.ofn"))
        .expect("contracts/ontology.ofn is tracked");
    assert_eq!(
        ofn,
        to_ofn(&e),
        "contracts/ontology.ofn is stale: run `pv ontology export --owl --write`"
    );
    let rep = std::fs::read_to_string(repo().join("contracts/tbox-report.json"))
        .expect("contracts/tbox-report.json is tracked");
    assert_eq!(
        rep,
        report_json(&tbox(&e)),
        "contracts/tbox-report.json is stale: run `pv ontology tbox --write`"
    );
}
