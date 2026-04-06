use super::*;

#[test]
fn obligation_type_display() {
    assert_eq!(ObligationType::Invariant.to_string(), "invariant");
    assert_eq!(ObligationType::Equivalence.to_string(), "equivalence");
    assert_eq!(ObligationType::Bound.to_string(), "bound");
    assert_eq!(ObligationType::Monotonicity.to_string(), "monotonicity");
    assert_eq!(ObligationType::Idempotency.to_string(), "idempotency");
    assert_eq!(ObligationType::Linearity.to_string(), "linearity");
    assert_eq!(ObligationType::Symmetry.to_string(), "symmetry");
    assert_eq!(ObligationType::Associativity.to_string(), "associativity");
    assert_eq!(ObligationType::Conservation.to_string(), "conservation");
    assert_eq!(ObligationType::Ordering.to_string(), "ordering");
    assert_eq!(ObligationType::Completeness.to_string(), "completeness");
    assert_eq!(ObligationType::Soundness.to_string(), "soundness");
    assert_eq!(ObligationType::Involution.to_string(), "involution");
    assert_eq!(ObligationType::Determinism.to_string(), "determinism");
    assert_eq!(ObligationType::Roundtrip.to_string(), "roundtrip");
    assert_eq!(ObligationType::StateMachine.to_string(), "state_machine");
    assert_eq!(ObligationType::Classification.to_string(), "classification");
    assert_eq!(ObligationType::Independence.to_string(), "independence");
    assert_eq!(ObligationType::Termination.to_string(), "termination");
    // Eiffel DbC types
    assert_eq!(ObligationType::Precondition.to_string(), "precondition");
    assert_eq!(ObligationType::Postcondition.to_string(), "postcondition");
    assert_eq!(ObligationType::Frame.to_string(), "frame");
    assert_eq!(ObligationType::LoopInvariant.to_string(), "loop_invariant");
    assert_eq!(ObligationType::LoopVariant.to_string(), "loop_variant");
    assert_eq!(ObligationType::OldState.to_string(), "old_state");
    assert_eq!(ObligationType::Subcontract.to_string(), "subcontract");
}

#[test]
fn lean_status_display() {
    assert_eq!(LeanStatus::Proved.to_string(), "proved");
    assert_eq!(LeanStatus::Sorry.to_string(), "sorry");
    assert_eq!(LeanStatus::Wip.to_string(), "wip");
    assert_eq!(LeanStatus::NotApplicable.to_string(), "not-applicable");
}

#[test]
fn lean_status_default_is_sorry() {
    assert_eq!(LeanStatus::default(), LeanStatus::Sorry);
}

#[test]
fn kani_strategy_display() {
    assert_eq!(KaniStrategy::Exhaustive.to_string(), "exhaustive");
    assert_eq!(KaniStrategy::StubFloat.to_string(), "stub_float");
    assert_eq!(KaniStrategy::Compositional.to_string(), "compositional");
    assert_eq!(KaniStrategy::BoundedInt.to_string(), "bounded_int");
}

#[test]
fn contract_kind_display() {
    assert_eq!(ContractKind::Kernel.to_string(), "kernel");
    assert_eq!(ContractKind::Registry.to_string(), "registry");
    assert_eq!(ContractKind::ModelFamily.to_string(), "model-family");
    assert_eq!(ContractKind::Pattern.to_string(), "pattern");
    assert_eq!(ContractKind::Schema.to_string(), "schema");
}

#[test]
fn contract_kind_default_is_kernel() {
    assert_eq!(ContractKind::default(), ContractKind::Kernel);
}

#[test]
fn kernel_contract_requires_proofs() {
    let mut c = Contract::default();
    c.metadata.kind = ContractKind::Kernel;
    assert!(c.requires_proofs());
    assert_eq!(c.kind(), ContractKind::Kernel);
    assert!(!c.is_registry());
    // Provability violations: no proof_obligations, no falsification_tests,
    // no kani_harnesses.
    assert!(!c.provability_violations().is_empty());
}

#[test]
fn non_kernel_kinds_exempt_from_provability() {
    for kind in [
        ContractKind::Registry,
        ContractKind::ModelFamily,
        ContractKind::Pattern,
        ContractKind::Schema,
    ] {
        let mut c = Contract::default();
        c.metadata.kind = kind;
        assert!(
            !c.requires_proofs(),
            "kind {kind:?} should not require proofs",
        );
        assert!(
            c.provability_violations().is_empty(),
            "kind {kind:?} should have no provability violations",
        );
    }
}

#[test]
fn legacy_registry_flag_maps_to_registry_kind() {
    let mut c = Contract::default();
    c.metadata.registry = true;
    // kind remains Kernel (default), but is_registry() and kind() both
    // reflect the legacy flag for back-compat.
    assert!(c.is_registry());
    assert_eq!(c.kind(), ContractKind::Registry);
    assert!(!c.requires_proofs());
    assert!(c.provability_violations().is_empty());
}

#[test]
fn explicit_kind_overrides_default() {
    let mut c = Contract::default();
    c.metadata.kind = ContractKind::ModelFamily;
    assert_eq!(c.kind(), ContractKind::ModelFamily);
    // ModelFamily is not a registry
    assert!(!c.is_registry());
}
