//! Credential tests (plan 04-17 G2, TRN-07, D-11).
//!
//! Every test name starts `credential_`.
//!
//! # The one thing these tests are FOR
//!
//! 04-16 refused to mint a verified state from artifact bytes because five of
//! `HeadFittedEvidence`'s seven fields are unrecoverable and `PassedEvidence` is private
//! (E0451). The credential exists so the lock doors stop demanding that state. If it ever
//! grows a way to supply, default or reconstruct any of that evidence, the refusal 04-16
//! made was for nothing — so `credential_module_fabricates_no_evidence` scans for exactly
//! that, and `credential_validate_evidence_is_still_the_only_passed_evidence_producer`
//! holds the other end.

use super::super::test_fixtures as fx;
use super::*;
use crate::train::setfit::evaluate::{evaluate_validation, ValidationMetricKind};
use crate::train::setfit::lock::{
    create_selection_lock, CanonicalTestAccess, CanonicalTestGrant, LockError, SelectionCandidate,
    SelectionRule,
};
use crate::train::setfit::tune::TuningProbes;
use crate::train::setfit::verify::SerdeJsonCodec;
use crate::train::setfit::Prepared;
use aprender_contrastive_data::prepared::{Canonical, PreparedDataset};

/// This module's subject, for the source assertions.
const CREDENTIAL_SOURCE: &str = include_str!("credential.rs");

/// A verified run through the shipped codec.
fn verified_run() -> SetFitRun<ArtifactReloadedAndVerified> {
    fx::verified_run(fx::calibrated_variant())
}

// ===========================================================================================
// The claim: the three doors no longer need the train-time run
// ===========================================================================================

/// Lock, mint and grant, driven through a value known ONLY to be a [`SetFitCredential`].
///
/// # This function is the evidence, not the test that calls it
///
/// Its body cannot name `SetFitRun`, `ArtifactReloadedAndVerified`, `HeadFittedEvidence` or
/// `PassedEvidence` — the bound does not provide them. So the fact that it COMPILES is the
/// proof that all three doors are now reachable from the three values a fresh process can
/// hold, which is the whole of 04-17 G2. The test below then runs it with the train-time run
/// to show the behaviour is unchanged.
///
/// When 04-16 lands `reload_verified_run_from_apr`, its credential is passed to this same
/// function and nothing here changes.
fn drive_every_door<'a, C: SetFitCredential>(
    model: &C,
    candidates: Vec<SelectionCandidate>,
    dataset: &'a PreparedDataset<Canonical>,
) -> Result<CanonicalTestGrant<'a>, LockError> {
    let lock =
        create_selection_lock(model, candidates, SelectionRule::MaxMetricLowestIndexTieBreak)?;
    let token = lock.mint_test_token(model)?;
    CanonicalTestAccess::grant(token, model, dataset)
}

/// The whole lock chain runs over a bare credential, and admits the right rows.
#[test]
fn credential_drives_the_lock_chain_end_to_end() {
    let run = verified_run();
    let dataset = fx::fixture_dataset();
    let evaluation = evaluate_validation(&run, &dataset, ValidationMetricKind::MacroF1)
        .expect("the fixture dataset is the one the run was prepared from");
    let candidates = vec![SelectionCandidate::from_evaluation("config-a", evaluation)];

    let expected_rows = dataset.test().rows().len();
    assert!(expected_rows > 0, "non-vacuity: the grant must admit real rows");

    let grant = drive_every_door(&run, candidates, &dataset)
        .expect("the credential is the model the lock chose, over the corpus it was chosen on");

    assert_eq!(
        grant.artifact_hash(),
        run.artifact_hash(),
        "the grant must belong to the artifact the credential named",
    );
    assert_eq!(grant.test().rows().len(), expected_rows);
}

/// A credential that is NOT the locked model is still refused — the retype kept the check.
///
/// Non-vacuity for the test above: if the doors had stopped comparing anything, both would
/// pass. Here the lock is taken over one run and the token is minted against a DIFFERENT one,
/// and the refusal must still be `StaleLock` naming both hashes.
#[test]
fn credential_that_is_not_the_locked_model_is_still_refused() {
    let locked = verified_run();
    let dataset = fx::fixture_dataset();
    let evaluation = evaluate_validation(&locked, &dataset, ValidationMetricKind::MacroF1)
        .expect("the fixture dataset is the run's own");
    let candidates = vec![SelectionCandidate::from_evaluation("config-a", evaluation)];
    let lock =
        create_selection_lock(&locked, candidates, SelectionRule::MaxMetricLowestIndexTieBreak)
            .expect("the locked run is its own candidate");

    let other = SetFitRun::<Prepared>::tune_encoder_with_probes(
        fx::prepared_run(fx::calibrated_variant(), None),
        TuningProbes::REVERSE_INTRA_BATCH_PULL,
    )
    .expect("the perturbed pipeline must still reach EncoderTuned")
    .fit_head()
    .expect("and HeadFitted")
    .verify_artifact(&SerdeJsonCodec::new())
    .expect("and must verify");
    assert_ne!(
        locked.artifact_hash(),
        other.artifact_hash(),
        "the two runs must really differ, or the refusal below would hold vacuously",
    );

    let error = lock.mint_test_token(&other).expect_err("a different artifact must not mint");
    let LockError::StaleLock { locked: recorded, observed } = error else {
        panic!("expected a stale lock, got {error:?}");
    };
    assert_eq!(recorded, locked.artifact_hash());
    assert_eq!(observed, other.artifact_hash());
}

// ===========================================================================================
// The delegation is an identity, so no train-time behaviour changed
// ===========================================================================================

/// Each credential method returns EXACTLY what the run's own accessor returns.
///
/// Called through the trait explicitly (`SetFitCredential::artifact_hash(&run)`) because a
/// bare `run.artifact_hash()` resolves to the INHERENT method — which would make this test
/// compare a value with itself and pass for any trait impl whatsoever.
#[test]
fn credential_reads_the_same_three_values_as_the_run_accessors() {
    let run = verified_run();

    assert_eq!(SetFitCredential::artifact_hash(&run), run.artifact_hash());
    assert_eq!(SetFitCredential::selection_semantic_hash(&run), run.selection_semantic_hash(),);
    assert_eq!(SetFitCredential::selection_ledger_hash(&run), run.selection().ledger_hash());

    // Non-vacuity: the three are genuinely different values, so an impl that returned the
    // same string three times could not pass the assertions above.
    assert_ne!(SetFitCredential::artifact_hash(&run), run.selection_semantic_hash());
    assert_eq!(SetFitCredential::artifact_hash(&run).len(), 64, "SHA-256 as hex");
}

// ===========================================================================================
// No evidence is fabricated, and the seal holds
// ===========================================================================================

/// The credential module contains no evidence type, no `Default` and no placeholder.
///
/// This is the assertion 04-16's refusal is worth something because of. A credential that
/// could produce a `HeadFittedEvidence` or a `PassedEvidence` would be the second minting
/// path with weaker evidence (T-04-46) wearing a new name.
#[test]
fn credential_module_fabricates_no_evidence() {
    const FORBIDDEN: [&str; 9] = [
        "HeadFittedEvidence",
        "PassedEvidence",
        "UpdateEvidence",
        "EvidenceSummary",
        "validate_evidence",
        "Default",
        "default()",
        "unimplemented!",
        "todo!",
    ];

    // Scanned over CODE LINES ONLY. The module header quotes four of these names while
    // explaining why the evidence is deliberately NOT here, and that explanation is worth
    // more than the names' absence — so the scan distinguishes prose from code rather than
    // maintaining a table of permitted occurrence counts, which would go stale on the first
    // reword and say nothing about behaviour either way.
    let code: Vec<&str> =
        CREDENTIAL_SOURCE.lines().filter(|line| !line.trim_start().starts_with("//")).collect();
    assert!(code.len() > 10, "non-vacuity: the filter must not have eaten the whole module");

    for line in &code {
        for forbidden in FORBIDDEN {
            assert!(
                !line.contains(forbidden),
                "`{forbidden}` appears on a CODE line of credential.rs: `{line}`. Plan 04-16 \
                 refused to mint a verified state because five of `HeadFittedEvidence`'s \
                 seven fields are unrecoverable; a credential that could supply, default or \
                 reconstruct any of them makes that refusal pointless",
            );
        }
    }

    // And the prose really is there, so a future edit that deletes the explanation instead
    // of the code is also caught.
    for named in ["HeadFittedEvidence", "PassedEvidence", "validate_evidence"] {
        assert!(
            CREDENTIAL_SOURCE.contains(named),
            "the module header must keep explaining why `{named}` is not carried here",
        );
    }
}

/// The trait exposes exactly the three values the doors read.
#[test]
fn credential_exposes_exactly_three_methods() {
    let start = CREDENTIAL_SOURCE
        .find("pub trait SetFitCredential: sealed::Sealed {")
        .expect("the trait must exist");
    let rest = &CREDENTIAL_SOURCE[start..];
    let end = rest.find("\n}").expect("the trait must close at column zero");
    let block = &rest[..end];

    assert_eq!(
        block.matches("\n    fn ").count(),
        3,
        "the credential is three values; a fourth is a fact no door reads and nothing \
         checks: {block}",
    );
    for method in [
        "fn artifact_hash(&self)",
        "fn selection_semantic_hash(&self)",
        "fn selection_ledger_hash(&self)",
    ] {
        assert!(block.contains(method), "`{method}` must be declared as `&self`: {block}");
    }
    assert!(
        !block.contains("&mut self") && !block.contains("(self)"),
        "a credential must be readable without being consumed or mutated: {block}",
    );
}

/// The seal is a SUPERTRAIT on a private module, and the compiler case exists.
///
/// The compile-time half lives in `tests/ui/setfit_external_credential_impl.rs`, because a
/// seal is a claim about programs that must NOT compile and no runtime assertion can make
/// it. This test holds the two structural halves that case depends on — the supertrait and
/// the private module — so a change that quietly unsealed the trait fails here as well as
/// there.
#[test]
fn credential_seal_is_a_private_supertrait() {
    assert!(
        CREDENTIAL_SOURCE.contains("pub trait SetFitCredential: sealed::Sealed {"),
        "the trait must keep its sealing supertrait",
    );
    assert!(
        CREDENTIAL_SOURCE.contains("mod sealed {") && !CREDENTIAL_SOURCE.contains("pub mod sealed"),
        "the sealing module must stay private, or the supertrait is nameable and the seal \
         is decoration",
    );
    // TWO types satisfy the seal, and the count is what makes a third deliberate. 04-17
    // shipped the first and left this number at one; 04-16 added the second, which is the
    // fresh-process credential the whole trait was introduced for. Both are named below, so
    // a change that swaps one for another fails here rather than passing on the count.
    assert_eq!(
        CREDENTIAL_SOURCE.matches("impl sealed::Sealed for ").count(),
        2,
        "exactly two types may satisfy the seal today; a third is a deliberate addition",
    );
    assert!(
        CREDENTIAL_SOURCE
            .contains("impl sealed::Sealed for SetFitRun<ArtifactReloadedAndVerified> {}"),
        "the train-time run must still be a credential",
    );
    assert!(
        CREDENTIAL_SOURCE.contains("impl sealed::Sealed for ReloadedSetFitCredential {}"),
        "and so must the fresh-process credential 04-16 mints from artifact bytes",
    );
}

/// `tune::validate_evidence` is still the ONLY producer of `PassedEvidence`.
///
/// Scanned over the whole `setfit` module rather than over `credential.rs`, because the
/// property is about the crate: 04-17 must not have opened a second door anywhere. The
/// producer is identified by its return type, so a helper that merely takes one does not
/// count.
#[test]
fn credential_validate_evidence_is_still_the_only_passed_evidence_producer() {
    const SOURCES: [(&str, &str); 6] = [
        ("credential.rs", include_str!("credential.rs")),
        ("lock.rs", include_str!("lock.rs")),
        ("mod.rs", include_str!("mod.rs")),
        ("verify.rs", include_str!("verify.rs")),
        ("evaluate.rs", include_str!("evaluate.rs")),
        ("bundle.rs", include_str!("bundle.rs")),
    ];
    for (name, src) in SOURCES {
        assert_eq!(
            construction_sites(src, "PassedEvidence"),
            0,
            "`{name}` constructs a `PassedEvidence`; only `tune::validate_evidence` may",
        );
    }

    // POSITIVE CONTROL, and it is the point: a counter that could only ever return zero
    // would satisfy the loop above on a tree where the type had been deleted outright.
    // `tune.rs` must still hold exactly one site, and it must still be that function.
    let tune = include_str!("tune.rs");
    assert_eq!(
        construction_sites(tune, "PassedEvidence"),
        1,
        "`tune.rs` must hold exactly one construction site — `validate_evidence`",
    );
    assert!(tune.contains("fn validate_evidence"), "and it must still be that function");
}

/// Struct-literal construction sites of `name` in `src`.
///
/// Counts `name {` while excluding the three places that spelling appears without building
/// anything: a return type (`-> &tune::PassedEvidence {` opens a FUNCTION body), the
/// declaration, and an `impl` header. Getting this wrong in either direction matters — a
/// count that includes return types reports two sites in `mod.rs` and the guard becomes
/// noise; one that excludes too much reports zero everywhere and the guard becomes theatre.
fn construction_sites(src: &str, name: &str) -> usize {
    let needle = format!("{name} {{");
    src.lines()
        .filter(|line| {
            let trimmed = line.trim_start();
            !trimmed.starts_with("//")
                && !line.contains("fn ")
                && !line.contains("struct ")
                && !line.contains("impl ")
        })
        .filter(|line| line.contains(&needle))
        .count()
}
