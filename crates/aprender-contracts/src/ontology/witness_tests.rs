//! The encoding, the digests and the checker. The checker is the part the gate trusts, so every way a certificate
//! can be wrong has a case here, and each names the error it must draw — a checker that refuses everything fails
//! the two acceptance cases, and one that accepts everything fails the rest.

use super::*;

fn s(x: &str) -> String {
    x.to_string()
}

fn edge(from: &str, role: &str, to: &str) -> TypedEdge {
    TypedEdge {
        from: s(from),
        role: s(role),
        to: s(to),
    }
}

/// The plant: only `A` asserted, `A⇒B, B⇒C`, `A ⊥ C`.
fn plant() -> ClauseSet {
    ClauseSet {
        units: [s("A")].into(),
        implies: [(s("A"), s("B")), (s("B"), s("C"))].into(),
        conflicts: [(s("A"), s("C"))].into(),
    }
}

fn core(steps: Vec<Step>, a: &str, b: &str) -> WitnessResult {
    WitnessResult::UnsatCore(Core {
        steps,
        conflict: (s(a), s(b)),
    })
}

fn good_core() -> WitnessResult {
    core(
        vec![
            Step::Unit(s("A")),
            Step::Implies(s("A"), s("B")),
            Step::Implies(s("B"), s("C")),
        ],
        "A",
        "C",
    )
}

#[test]
fn the_encoding_asserts_live_contracts_and_types_each_role() {
    let ids: BTreeSet<String> = ["a", "b", "c", "d", "e"].map(s).into();
    let edges: BTreeSet<TypedEdge> = [
        edge("a", "refines", "b"),
        edge("a", "depends_on", "c"),
        edge("a", "contradicts", "d"),
        edge("d", "contradicts", "a"),
        edge("b", "supersedes", "c"),
    ]
    .into();
    let cs = ClauseSet::from_graph(&ids, &edges);
    assert_eq!(
        cs.units,
        ["a", "b", "d", "e"].map(s).into(),
        "c is superseded, so not asserted"
    );
    assert_eq!(cs.implies, [(s("a"), s("b")), (s("a"), s("c"))].into());
    assert_eq!(
        cs.conflicts,
        [(s("a"), s("d"))].into(),
        "a symmetric pair is one clause"
    );
    assert_eq!(cs.checkable_n(), 3);
    assert_eq!(unencoded_edges(&edges), 0);
    assert_eq!(unencoded_edges(&[edge("a", "binds", "b")].into()), 1);
}

#[test]
fn the_digests_are_order_free_and_sensitive_to_every_edge() {
    let ids: BTreeSet<String> = ["a", "b"].map(s).into();
    let e1: BTreeSet<TypedEdge> = [edge("a", "depends_on", "b")].into();
    let e2: BTreeSet<TypedEdge> = [edge("a", "refines", "b")].into();
    assert_eq!(census_id_set_sha256(&ids).len(), 64);
    assert_ne!(
        relations_sha256(&e1),
        relations_sha256(&e2),
        "the role is hashed"
    );
    assert_ne!(
        census_id_set_sha256(&ids),
        census_id_set_sha256(&["a", "b", "c"].map(s).into())
    );
    // Known answer, from `printf 'a\nb\n' | sha256sum`: sorted ids, one per line, each newline-terminated.
    assert_eq!(
        census_id_set_sha256(&ids),
        "911169ddaaf146aff539f58c26c489af3b892dff0fe283c1c264c65ae5aa59a2"
    );
}

#[test]
fn a_real_derivation_checks_and_names_its_core() {
    assert_eq!(
        check(&plant(), &good_core()),
        Ok(Checked::Unsat {
            core: ["A", "B", "C"].map(s).into()
        })
    );
}

#[test]
fn a_premise_used_before_it_is_derived_is_refused() {
    let r = core(
        vec![
            Step::Unit(s("A")),
            Step::Implies(s("B"), s("C")),
            Step::Implies(s("A"), s("B")),
        ],
        "A",
        "C",
    );
    assert_eq!(
        check(&plant(), &r),
        Err(CheckError::PremiseNotDerived {
            premise: s("B"),
            step: 1
        })
    );
}

#[test]
fn a_clause_the_graph_does_not_hold_is_refused() {
    let invented_unit = core(vec![Step::Unit(s("C")), Step::Unit(s("A"))], "A", "C");
    assert_eq!(
        check(&plant(), &invented_unit),
        Err(CheckError::NotAUnit(s("C")))
    );
    let invented_edge = core(
        vec![Step::Unit(s("A")), Step::Implies(s("A"), s("C"))],
        "A",
        "C",
    );
    assert_eq!(
        check(&plant(), &invented_edge),
        Err(CheckError::NoSuchImplication(s("A"), s("C")))
    );
    let invented_conflict = core(
        vec![Step::Unit(s("A")), Step::Implies(s("A"), s("B"))],
        "A",
        "B",
    );
    assert_eq!(
        check(&plant(), &invented_conflict),
        Err(CheckError::NoSuchConflict(s("A"), s("B")))
    );
}

#[test]
fn a_conflict_whose_side_was_never_derived_is_refused() {
    let r = core(
        vec![Step::Unit(s("A")), Step::Implies(s("A"), s("B"))],
        "A",
        "C",
    );
    assert_eq!(
        check(&plant(), &r),
        Err(CheckError::ConflictSideNotDerived(s("C")))
    );
    assert_eq!(
        check(&plant(), &core(vec![], "A", "C")),
        Err(CheckError::EmptyCore)
    );
}

#[test]
fn a_model_is_checked_against_every_clause() {
    let mut sat = plant();
    sat.conflicts.clear();
    let model = |f: &[&str]| {
        WitnessResult::Model(Model {
            false_vars: f.iter().map(|x| s(x)).collect(),
        })
    };
    assert_eq!(check(&sat, &model(&[])), Ok(Checked::Sat));
    assert_eq!(
        check(&sat, &model(&["A"])),
        Err(CheckError::ModelFalsifiesUnit(s("A")))
    );
    assert_eq!(
        check(&sat, &model(&["C"])),
        Err(CheckError::ModelFalsifiesImplication(s("B"), s("C")))
    );
    // The plant has no model: all-true breaks the conflict, and anything less breaks a unit or an implication.
    assert_eq!(
        check(&plant(), &model(&[])),
        Err(CheckError::ModelFalsifiesConflict(s("A"), s("C")))
    );
}

#[test]
fn pc_checker_fires_on_the_shipped_fixture() {
    assert_eq!(pc_checker(), Ok(FIRED));
    // And the fixture is corrupt for the reason it says, not because it fails to parse.
    let fx: CertificateFixture =
        serde_json::from_str(CORRUPT_CORE_FIXTURE).expect("fixture parses");
    assert_eq!(fx.clauses, plant());
    assert!(matches!(
        check(&fx.clauses, &fx.result),
        Err(CheckError::PremiseNotDerived { .. })
    ));
}

#[test]
fn the_witness_serializes_in_the_spec_shape() {
    let w = Witness {
        census_id_set_sha256: "0".repeat(64),
        relations_sha256: "1".repeat(64),
        reasoner_git_sha: None,
        checkable_n: 3,
        result: good_core(),
        pc_reasoner: FIRED.into(),
        cpu_ms: 0,
    };
    let v = serde_json::to_value(&w).expect("serializes");
    assert_eq!(v["result"]["kind"], "unsat_core");
    assert_eq!(v["result"]["payload"]["steps"][0]["unit"], "A");
    assert_eq!(v["result"]["payload"]["steps"][1]["implies"][1], "B");
    assert_eq!(v["result"]["payload"]["conflict"][1], "C");
    let back: Witness = serde_json::from_value(v).expect("round-trips");
    assert_eq!(back, w);
    let m = serde_json::to_value(WitnessResult::Model(Model { false_vars: vec![] }))
        .expect("serializes");
    assert_eq!(
        m,
        serde_json::json!({"kind": "model", "payload": {"false": []}})
    );
}
