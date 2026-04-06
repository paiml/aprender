// ============================================================================
// F5: PTX Compilation Poison Trap -- Kernel Verification
// ============================================================================

use trueno_cuda_edge::ptx_poison::{default_mutators, PtxMutator, PtxVerifier, MINIMAL_VALID_PTX};

/// Test PTX structural verification.
#[test]
fn ptx_structural_verification() {
    let verifier = PtxVerifier::new();

    // Valid PTX passes
    let result = verifier.verify(MINIMAL_VALID_PTX);
    assert!(result.is_ok());

    // Empty PTX fails
    let result = verifier.verify("");
    assert!(result.is_err());

    // Missing .version fails
    let no_version = ".target sm_80\n.address_size 64\n.entry k() { ret; }";
    let errors = verifier.check_all(no_version);
    assert!(!errors.is_empty());
}

/// Test mutation operators for kernel testing.
#[test]
fn mutation_operators() {
    let mutators = default_mutators();
    assert_eq!(mutators.len(), 8);

    // Arithmetic mutations
    assert!(mutators.contains(&PtxMutator::FlipAddSub));
    assert!(mutators.contains(&PtxMutator::FlipMulDiv));

    // Control flow mutations
    assert!(mutators.contains(&PtxMutator::InvertPredicate));
    assert!(mutators.contains(&PtxMutator::RemoveBarrier));

    // Precision mutations
    assert!(mutators.contains(&PtxMutator::WidenPrecision));
}

/// Test mutation application to PTX source.
#[test]
fn mutation_application() {
    // FlipAddSub: add -> sub
    let ptx = "add.f32 %f1, %f2, %f3;";
    let mutated = PtxMutator::FlipAddSub.apply(ptx);
    assert!(mutated.is_some());
    assert!(mutated.unwrap().contains("sub.f32"));

    // FlipMulDiv: mul -> div
    let ptx = "mul.f32 %f1, %f2, %f3;";
    let mutated = PtxMutator::FlipMulDiv.apply(ptx);
    assert!(mutated.is_some());
    assert!(mutated.unwrap().contains("div.f32"));

    // InvertPredicate: setp.lt -> setp.ge
    let ptx = "setp.lt.f32 %p1, %f1, %f2;";
    let mutated = PtxMutator::InvertPredicate.apply(ptx);
    assert!(mutated.is_some());
    assert!(mutated.unwrap().contains("setp.ge"));
}

/// Test PTX verification catches common errors.
#[test]
fn ptx_common_errors() {
    let verifier = PtxVerifier::new();

    // Missing .target
    let no_target = ".version 7.0\n.address_size 64\n.entry k() { ret; }";
    let errors = verifier.check_all(no_target);
    assert!(!errors.is_empty());

    // Missing .address_size
    let no_addr = ".version 7.0\n.target sm_80\n.entry k() { ret; }";
    let errors = verifier.check_all(no_addr);
    assert!(!errors.is_empty());

    // Missing entry point
    let no_entry = ".version 7.0\n.target sm_80\n.address_size 64\n";
    let errors = verifier.check_all(no_entry);
    assert!(!errors.is_empty());
}
