//! Integration tests for the PTX Compilation Poison Trap (TCE-PTX).
//!
//! Falsification tests from Section 8.2, F5 (claims 41-50).

#![allow(clippy::unwrap_used, clippy::similar_names, clippy::disallowed_methods)]

use trueno_cuda_edge::ptx_poison::{
    default_mutators, MutantResult, PoisonTrapReport, PtxMutator, PtxVerificationError,
    PtxVerifier, MINIMAL_VALID_PTX,
};

/// Claim 41: 8 mutation operators are defined
#[test]
fn claim_41_eight_mutators_defined() {
    let mutators = default_mutators();
    assert_eq!(mutators.len(), 8, "Must have exactly 8 mutation operators");
}

/// Claim 42: `FlipAddSub` replaces add with sub
#[test]
fn claim_42_flip_add_sub() {
    let src = "add.f32 %r1, %r2, %r3;";
    let mutated = PtxMutator::FlipAddSub.apply(src).unwrap();
    assert!(mutated.contains("sub.f32"), "Must replace add with sub");
    assert!(!mutated.contains("add.f32"), "Original add must be gone");
}

/// Claim 43: PTX verifier rejects empty source
#[test]
fn claim_43_verifier_rejects_empty() {
    let verifier = PtxVerifier::new();
    let result = verifier.verify("");
    assert!(result.is_err(), "Empty PTX must be rejected");

    let errors = verifier.check_all("");
    assert!(errors.contains(&PtxVerificationError::EmptySource));
}

/// Claim 44: PTX verifier requires .version directive
#[test]
fn claim_44_verifier_requires_version() {
    let src = ".target sm_80\n.address_size 64\n.entry k() {\nret;\n}\n";
    let verifier = PtxVerifier::new();
    let errors = verifier.check_all(src);
    assert!(
        errors.contains(&PtxVerificationError::MissingVersion),
        "Missing .version must be detected"
    );
}

/// Claim 45: Mutation score excludes compile errors
#[test]
fn claim_45_mutation_score_excludes_compile_errors() {
    let report = PoisonTrapReport {
        results: vec![
            (PtxMutator::FlipAddSub, MutantResult::Killed),
            (PtxMutator::FlipMulDiv, MutantResult::CompileError),
            (PtxMutator::RemoveBarrier, MutantResult::Survived),
        ],
        sources_processed: 1,
    };

    // 2 applicable (excluding compile error), 1 killed → 50%
    let score = report.mutation_score();
    assert!((score - 0.5).abs() < f64::EPSILON, "Mutation score must exclude compile errors");
}

/// Claim 46: `VerifiedPtx` cannot be constructed externally
/// This is a compile-time guarantee - the struct has a private field
#[test]
fn claim_46_verified_ptx_only_via_verifier() {
    let ptx_verifier = PtxVerifier::new();
    let ptx_verified = ptx_verifier.verify(MINIMAL_VALID_PTX).unwrap();
    assert!(ptx_verified.source().contains(".version"), "VerifiedPtx must expose verified source");
}

/// Claim 47: Timeout counts as killed
#[test]
fn claim_47_timeout_counts_as_killed() {
    assert!(MutantResult::Timeout.is_killed(), "Timeout must count as killed");
    assert!(MutantResult::Killed.is_killed(), "Killed must count as killed");
    assert!(!MutantResult::Survived.is_killed(), "Survived must not count as killed");
    assert!(!MutantResult::CompileError.is_killed(), "CompileError must not count as killed");
}

/// Claim 48: Mutation not found returns None
#[test]
fn claim_48_mutation_not_found_returns_none() {
    let src = "mul.f32 %r1, %r2, %r3;"; // No add instruction
    assert!(
        PtxMutator::FlipAddSub.apply(src).is_none(),
        "FlipAddSub on source without add must return None"
    );
}

/// Claim 49: All mutators have descriptions
#[test]
fn claim_49_all_mutators_have_descriptions() {
    for mutator in default_mutators() {
        let desc = mutator.description();
        assert!(!desc.is_empty(), "Mutator {mutator:?} must have a description");
    }
}

/// Claim 50: Minimal valid PTX passes verification
#[test]
fn claim_50_minimal_valid_ptx_passes() {
    let verifier = PtxVerifier::new();
    let result = verifier.verify(MINIMAL_VALID_PTX);
    assert!(result.is_ok(), "Minimal valid PTX must pass verification");
}

/// Test `FlipMulDiv` mutation
#[test]
fn flip_mul_div_mutation() {
    let src = "mul.f32 %r1, %r2, %r3;";
    let mutated = PtxMutator::FlipMulDiv.apply(src).unwrap();
    assert!(mutated.contains("div.f32"));
}

/// Test `WidenPrecision` mutation
#[test]
fn widen_precision_mutation() {
    let src = "add.f32 %r1, %r2, %r3;";
    let mutated = PtxMutator::WidenPrecision.apply(src).unwrap();
    assert!(mutated.contains(".f64"));
}

/// Test `SwapMemorySpace` mutation
#[test]
fn swap_memory_space_mutation() {
    let src = "ld.shared.f32 %r1, [%r2];";
    let mutated = PtxMutator::SwapMemorySpace.apply(src).unwrap();
    assert!(mutated.contains(".global"));
}

/// Test `RemoveBarrier` mutation
#[test]
fn remove_barrier_mutation() {
    let src = "add.f32 %r1, %r2, %r3;\nbar.sync 0;\nmul.f32 %r4, %r5, %r6;";
    let mutated = PtxMutator::RemoveBarrier.apply(src).unwrap();
    assert!(!mutated.contains("bar.sync"));
}

/// Test `InvertPredicate` mutation variants
#[test]
fn invert_predicate_mutations() {
    // lt → ge
    let ptx_lt = "setp.lt.f32 %p1, %r1, %r2;";
    let mut_lt = PtxMutator::InvertPredicate.apply(ptx_lt).unwrap();
    assert!(mut_lt.contains("setp.ge"));

    // gt → le
    let ptx_gt = "setp.gt.f32 %p1, %r1, %r2;";
    let mut_gt = PtxMutator::InvertPredicate.apply(ptx_gt).unwrap();
    assert!(mut_gt.contains("setp.le"));

    // eq → ne
    let ptx_eq = "setp.eq.s32 %p1, %r1, %r2;";
    let mut_eq = PtxMutator::InvertPredicate.apply(ptx_eq).unwrap();
    assert!(mut_eq.contains("setp.ne"));
}

/// Test mutation score with all killed
#[test]
fn mutation_score_all_killed() {
    let report = PoisonTrapReport {
        results: vec![
            (PtxMutator::FlipAddSub, MutantResult::Killed),
            (PtxMutator::FlipMulDiv, MutantResult::Killed),
            (PtxMutator::RemoveBarrier, MutantResult::Timeout),
        ],
        sources_processed: 1,
    };
    assert!((report.mutation_score() - 1.0).abs() < f64::EPSILON);
    assert!(report.all_killed());
}

/// Test survivors list
#[test]
fn survivors_list() {
    let report = PoisonTrapReport {
        results: vec![
            (PtxMutator::FlipAddSub, MutantResult::Killed),
            (PtxMutator::FlipMulDiv, MutantResult::Survived),
            (PtxMutator::RemoveBarrier, MutantResult::Survived),
        ],
        sources_processed: 1,
    };
    let survivors = report.survivors();
    assert_eq!(survivors.len(), 2);
}

/// Test verifier unbalanced braces detection
#[test]
fn verifier_unbalanced_braces() {
    let src = ".version 7.0\n.target sm_80\n.address_size 64\n.entry k() {\nret;\n";
    let verifier = PtxVerifier::new();
    let errors = verifier.check_all(src);
    assert!(errors.iter().any(|e| matches!(e, PtxVerificationError::UnbalancedBraces { .. })));
}
