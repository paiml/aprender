//! Fresh-process reload tests (plan 04-16, D-11, D-16, APR-04, TRN-07).
//!
//! Every test name starts `apr_reload_`.
//!
//! # What "fresh process" means here, and how these tests avoid faking it
//!
//! A test that reloaded an artifact and then handed the door `run.selection()` and
//! `run.dataset()` would prove almost nothing: the gate would pass because the caller passed
//! back the very objects the artifact was written from. So the honest-path tests below
//! rebuild the inputs INDEPENDENTLY — the dataset through Phase 2's ingest ladder, the
//! selection through a persisted `SelectionManifest` and `Selection::replay`, which is the
//! route `apr eval` takes a day later. The gate then passes by FINGERPRINT, which is the
//! property being claimed, rather than by identity, which is not.

use aprender::setfit::{artifact_sha256_hex, SetFitArtifactError};
use aprender_contrastive_data::ledger::AccessLedger;
use aprender_contrastive_data::manifest::SelectionManifest;
use aprender_contrastive_data::select::{FewShotSelector, SelectionConfig};

use super::super::apr_codec::fixture::{apr_capable_run, artifact_bytes_of};
use super::super::apr_codec::{AprCodec, APR_FORMAT_ID};
use super::super::credential::SetFitCredential;
use super::super::evaluate::{evaluate_validation, ValidationMetricKind};
use super::super::lock::{
    create_selection_lock, CanonicalTestAccess, CanonicalTestGrant, LockError, SelectionCandidate,
    SelectionRule,
};
use super::super::test_fixtures as fx;
use super::super::verify::CodecError;
use super::super::{ArtifactReloadedAndVerified, SetFitRun};
use super::*;

/// This module's subject, for the source assertions.
const RELOAD_SOURCE: &str = include_str!("apr_reload.rs");

// ===========================================================================================
// Fixtures: the artifact, and the inputs a LATER process would hold
// ===========================================================================================

/// The APR-capable run carried through the trusted policy, so its recorded artifact hash is
/// the sha256 of [`apr_artifact_bytes`]' output.
///
/// Needed because a `SelectionCandidate` reads its artifact hash out of a
/// `ValidationEvaluation`, and the evaluator takes the train-time run — so a candidate set
/// containing THIS artifact can only be built from the run that produced it.
fn apr_verified_run() -> SetFitRun<ArtifactReloadedAndVerified> {
    apr_capable_run()
        .verify_artifact(&AprCodec::new())
        .expect("the APR round trip closes for the APR-capable fixture")
}

/// The artifact under test, as bytes.
fn apr_artifact_bytes() -> Vec<u8> {
    artifact_bytes_of(&apr_capable_run())
}

/// The dataset and selection a SEPARATE process would hold, rebuilt from scratch.
///
/// The selection travels through a real persisted manifest — `from_selection` ->
/// `to_file_bytes` -> `from_bytes` -> `replay` — because a later process has a FILE, not a
/// `Selection`. `replay` is also what makes the ledger hash comparable at all, so routing
/// through it is load-bearing rather than decorative.
fn fresh_process_inputs() -> (PreparedDataset<Canonical>, Selection) {
    let variant = fx::calibrated_variant();

    // The writing process.
    let mut ledger = AccessLedger::new();
    let dataset = fx::synthetic_dataset(&mut ledger);
    let selection = FewShotSelector::select(
        &dataset,
        &SelectionConfig { root_seed: variant.root_seed, shots_per_class: variant.shots_per_class },
        &mut ledger,
    )
    .expect("the synthetic corpus must support the calibrated selection");
    let manifest = SelectionManifest::from_selection(&selection, &ledger)
        .expect("the live ledger is the one the selection was taken under");
    let bytes = manifest.to_file_bytes().expect("the manifest must serialize");

    // A LATER process: fresh ledger, corpus rebuilt, manifest read back from its file form.
    let mut fresh_ledger = AccessLedger::new();
    let fresh_dataset = fx::synthetic_dataset(&mut fresh_ledger);
    let restored = SelectionManifest::from_bytes(&bytes).expect("the manifest must parse back");
    let replayed = Selection::replay(&restored, &fresh_dataset, &mut fresh_ledger)
        .expect("the manifest must replay against the rebuilt corpus");
    (fresh_dataset, replayed)
}

/// A selection over the SAME corpus at the SAME shot count, from a different seed.
///
/// The right probe for the FINEST gate: corpus, class balance and shot count are held
/// constant, so only the draw differs.
fn differently_seeded_selection() -> Selection {
    let variant = fx::calibrated_variant();
    fx::fixture_selection(variant.root_seed.wrapping_add(0x5eed), variant.shots_per_class)
}

/// A selection over the SAME corpus, at the SAME seed, under a POLLUTED access ledger.
///
/// The right probe for the middle gate. The extra record is an ordinary ledger append —
/// exactly what a process that touched the corpus for another purpose before selecting
/// would produce — so the rows drawn are identical and only the audit trail differs.
fn selection_under_a_polluted_ledger() -> Selection {
    let variant = fx::calibrated_variant();
    let mut ledger = AccessLedger::new();
    let dataset = fx::synthetic_dataset(&mut ledger);
    ledger.record("train", "canonical", "unrelated_inspection", &dataset.fingerprint().hex());
    FewShotSelector::select(
        &dataset,
        &SelectionConfig { root_seed: variant.root_seed, shots_per_class: variant.shots_per_class },
        &mut ledger,
    )
    .expect("the synthetic corpus must support the calibrated selection")
}

// ===========================================================================================
// The claim: a fresh process reaches the lock chain
// ===========================================================================================

/// Artifact bytes plus independently-rebuilt inputs mint a credential over the same hashes.
#[test]
fn apr_reload_mints_a_credential_whose_hash_is_the_artifacts_own() {
    let run = apr_capable_run();
    let bytes = apr_artifact_bytes();
    let (dataset, selection) = fresh_process_inputs();

    let credential = reload_verified_run_from_apr(&bytes, &dataset, &selection)
        .expect("the artifact and its own inputs must reload");

    assert_eq!(
        credential.artifact_hash(),
        artifact_sha256_hex(&bytes),
        "the credential's identity must be the sha256 of the bytes it was minted from",
    );
    assert_eq!(credential.artifact_hash().len(), 64, "SHA-256 as lowercase hex");

    // The inputs really were rebuilt rather than borrowed: different objects, same
    // fingerprints. Without this the assertion above would hold for a door that compared
    // nothing at all.
    assert_eq!(credential.selection_semantic_hash(), hex::encode(run.selection().semantic_hash()));
    assert_eq!(credential.selection_ledger_hash(), run.selection().ledger_hash());
    assert_eq!(
        dataset.validation_witness().dataset_fingerprint_hex(),
        run.dataset().validation_witness().dataset_fingerprint_hex(),
    );
}

/// Lock, mint and grant driven through a value known ONLY to be a [`SetFitCredential`].
///
/// The body cannot name `SetFitRun`, `ArtifactReloadedAndVerified`, `HeadFittedEvidence` or
/// `PassedEvidence` — the bound does not provide them — so the fact that it compiles with a
/// fresh-process credential is the whole of this plan's claim. Same shape, and the same
/// reason, as `credential_tests::drive_every_door`.
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

/// TRN-07's positive tier, from artifact bytes, in a process that trained nothing.
#[test]
fn apr_reload_credential_reaches_lock_token_and_grant() {
    let bytes = apr_artifact_bytes();
    let (dataset, selection) = fresh_process_inputs();
    let credential = reload_verified_run_from_apr(&bytes, &dataset, &selection)
        .expect("the artifact and its own inputs must reload");

    let trained = apr_verified_run();
    assert_eq!(
        credential.artifact_hash(),
        trained.artifact_hash(),
        "the reload and the trusted policy must agree on the artifact's identity",
    );
    let evaluation = evaluate_validation(&trained, &dataset, ValidationMetricKind::MacroF1)
        .expect("the rebuilt corpus is the one the run was prepared from");
    let candidates = vec![SelectionCandidate::from_evaluation("config-apr", evaluation)];

    let expected_rows = dataset.test().rows().len();
    assert!(expected_rows > 0, "non-vacuity: the grant must admit real rows");

    let grant = drive_every_door(&credential, candidates, &dataset)
        .expect("the credential is the model the lock chose, over the corpus it was chosen on");

    assert_eq!(grant.artifact_hash(), credential.artifact_hash());
    assert_eq!(grant.test().rows().len(), expected_rows);
}

/// A credential whose artifact is not in the candidate set is refused.
///
/// Non-vacuity for the test above: if the doors had stopped comparing anything, both would
/// pass. The candidate here is a genuine evaluation of a DIFFERENT artifact — the phase-3
/// `SerdeJsonCodec` run — so the refusal is about identity and nothing else.
#[test]
fn apr_reload_credential_is_refused_by_a_lock_it_is_not_a_candidate_of() {
    let bytes = apr_artifact_bytes();
    let (dataset, selection) = fresh_process_inputs();
    let credential =
        reload_verified_run_from_apr(&bytes, &dataset, &selection).expect("the reload must mint");

    let other = fx::verified_run(fx::calibrated_variant());
    assert_ne!(
        other.artifact_hash(),
        credential.artifact_hash(),
        "the two artifacts must really differ, or the refusal below holds vacuously",
    );
    let evaluation = evaluate_validation(&other, &dataset, ValidationMetricKind::MacroF1)
        .expect("the fixture dataset is that run's own");
    let candidates = vec![SelectionCandidate::from_evaluation("config-other", evaluation)];

    let error =
        create_selection_lock(&credential, candidates, SelectionRule::MaxMetricLowestIndexTieBreak)
            .expect_err("a lock whose candidates exclude the credential must not be created");
    let LockError::ChosenModelNotACandidate { artifact_hash, candidates } = error else {
        panic!("expected a candidate-membership refusal, got {error:?}");
    };
    assert_eq!(artifact_hash, credential.artifact_hash());
    assert_eq!(candidates, 1);
}

// ===========================================================================================
// The provenance identity gate: three refusals, each naming both values
// ===========================================================================================

/// A different corpus is refused — the COARSEST gate — naming both fingerprints.
#[test]
fn apr_reload_refuses_a_dataset_with_a_different_fingerprint() {
    let bytes = apr_artifact_bytes();
    let (honest_dataset, selection) = fresh_process_inputs();
    let altered = fx::dataset_with_altered_test_row();
    assert_ne!(
        altered.validation_witness().dataset_fingerprint_hex(),
        honest_dataset.validation_witness().dataset_fingerprint_hex(),
        "non-vacuity: the two corpora must really differ",
    );

    let error = reload_verified_run_from_apr(&bytes, &altered, &selection)
        .expect_err("a corpus the artifact was not trained on must not mint");

    let SetFitTrainError::AprReload(AprReloadError::DatasetFingerprintMismatch {
        recorded,
        supplied,
    }) = error
    else {
        panic!("expected a dataset-fingerprint refusal, got {error:?}");
    };
    assert_eq!(recorded, honest_dataset.validation_witness().dataset_fingerprint_hex());
    assert_eq!(supplied, altered.validation_witness().dataset_fingerprint_hex());
    assert_ne!(recorded, supplied, "a refusal naming one value twice would say nothing");
}

/// A selection taken under a different ACCESS LEDGER is refused, distinctly.
///
/// The rows are identical — `assert_eq!` on the ordered ids proves it — so this is the case
/// the semantic-hash comparison would have reported as "different rows were selected". That
/// is a wrong diagnosis rather than a coarse one, and avoiding it is why the ledger gate
/// runs first.
#[test]
fn apr_reload_refuses_a_selection_with_a_different_ledger_hash() {
    let bytes = apr_artifact_bytes();
    let (dataset, honest) = fresh_process_inputs();
    let polluted = selection_under_a_polluted_ledger();

    assert_eq!(
        polluted.ordered_ids(),
        honest.ordered_ids(),
        "the two selections must draw the SAME rows, or this is not the ledger case",
    );
    assert_ne!(
        polluted.ledger_hash(),
        honest.ledger_hash(),
        "non-vacuity: the ledgers must really differ",
    );

    let error = reload_verified_run_from_apr(&bytes, &dataset, &polluted)
        .expect_err("a selection taken under a different ledger must not mint");

    let SetFitTrainError::AprReload(AprReloadError::SelectionLedgerHashMismatch {
        recorded,
        supplied,
    }) = error
    else {
        panic!("expected a LEDGER-hash refusal, got {error:?}");
    };
    assert_eq!(recorded, hex::encode(honest.ledger_hash()));
    assert_eq!(supplied, hex::encode(polluted.ledger_hash()));
    assert_ne!(recorded, supplied);
}

/// A selection that drew DIFFERENT ROWS is refused — the FINEST gate — naming both hashes.
#[test]
fn apr_reload_refuses_a_selection_with_a_different_semantic_hash() {
    let bytes = apr_artifact_bytes();
    let (dataset, honest) = fresh_process_inputs();
    let other = differently_seeded_selection();

    assert_eq!(
        other.ledger_hash(),
        honest.ledger_hash(),
        "the two selections must share a ledger, or the coarser gate would answer first",
    );
    assert_ne!(
        other.semantic_hash(),
        honest.semantic_hash(),
        "non-vacuity: the two draws must really differ",
    );

    let error = reload_verified_run_from_apr(&bytes, &dataset, &other)
        .expect_err("a selection the artifact was not trained under must not mint");

    let SetFitTrainError::AprReload(AprReloadError::SelectionSemanticHashMismatch {
        recorded,
        supplied,
    }) = error
    else {
        panic!("expected a semantic-hash refusal, got {error:?}");
    };
    assert_eq!(recorded, hex::encode(honest.semantic_hash()), "the artifact's own record");
    assert_eq!(supplied, hex::encode(other.semantic_hash()), "and the value supplied");
    assert_ne!(recorded, supplied);
}

/// The three refusals are three DISTINCT variants, and each rendering names both values.
///
/// Three tests each matching a different variant would still be consistent with two of the
/// names being aliases of one shape. This also holds the rendering, because a typed refusal
/// whose `Display` drops one of the two values is diagnosable only with a debugger.
#[test]
fn apr_reload_the_three_refusals_are_distinct_and_each_names_both_values() {
    let corpus = AprReloadError::DatasetFingerprintMismatch {
        recorded: "aa".to_string(),
        supplied: "bb".to_string(),
    };
    let ledger = AprReloadError::SelectionLedgerHashMismatch {
        recorded: "aa".to_string(),
        supplied: "bb".to_string(),
    };
    let semantic = AprReloadError::SelectionSemanticHashMismatch {
        recorded: "aa".to_string(),
        supplied: "bb".to_string(),
    };

    assert_ne!(corpus, ledger);
    assert_ne!(ledger, semantic);
    assert_ne!(corpus, semantic);

    for error in [&corpus, &ledger, &semantic] {
        let rendered = error.to_string();
        assert!(rendered.contains("aa"), "the recorded value must be named: {rendered}");
        assert!(rendered.contains("bb"), "the supplied value must be named: {rendered}");
    }
    assert_ne!(corpus.to_string(), ledger.to_string());
    assert_ne!(ledger.to_string(), semantic.to_string());
}

// ===========================================================================================
// The structural facts the gate ORDER rests on — measured, not assumed
// ===========================================================================================

/// The three recorded identifiers NEST, coarse to fine, and the order follows from that.
///
/// The plan that specified this door asserted the ledger hash "distinguishes a different
/// selection over the same dataset with the same shots — which the semantic hash alone does
/// not". The opposite is true, and the three measurements below are why the gate is ordered
/// corpus -> ledger -> rows rather than the reverse. Every claim here is a property of
/// `aprender-contrastive-data`, so if any of them changes this test is where the ordering
/// argument is re-derived.
#[test]
fn apr_reload_the_three_recorded_identifiers_are_ordered_coarse_to_fine() {
    let (_, honest) = fresh_process_inputs();

    // (1) A different LEDGER implies a different SEMANTIC hash, because `Selection`'s
    //     semantic hash is a digest of the whole payload and the payload embeds both
    //     `access_ledger` and `ledger_hash` (`select.rs::assemble`). So a semantic-first
    //     gate makes the ledger arm unreachable.
    let polluted = selection_under_a_polluted_ledger();
    assert_ne!(polluted.ledger_hash(), honest.ledger_hash());
    assert_ne!(
        polluted.semantic_hash(),
        honest.semantic_hash(),
        "the semantic hash must move with the ledger, or the ordering argument is unfounded",
    );

    // (2) A different SEED does NOT change the ledger: its records are the dataset
    //     fingerprint and the purposes, none of which depend on the draw. So the semantic
    //     gate is genuinely reachable behind the ledger gate.
    let other_seed = differently_seeded_selection();
    assert_eq!(
        other_seed.ledger_hash(),
        honest.ledger_hash(),
        "the ledger must be blind to the seed, or the finest gate is unreachable",
    );
    assert_ne!(other_seed.semantic_hash(), honest.semantic_hash());

    // (3) A different CORPUS changes the ledger too, because the records carry the dataset
    //     fingerprint — which is why the corpus is checked first and not second.
    let mut ledger = AccessLedger::new();
    let altered = fx::dataset_with_altered_test_row();
    let variant = fx::calibrated_variant();
    let over_altered = FewShotSelector::select(
        &altered,
        &SelectionConfig { root_seed: variant.root_seed, shots_per_class: variant.shots_per_class },
        &mut ledger,
    )
    .expect("the altered corpus still supports the selection");
    assert_ne!(
        over_altered.ledger_hash(),
        honest.ledger_hash(),
        "a corpus change must reach the ledger, or the corpus gate could be second",
    );
}

// ===========================================================================================
// The loader runs FIRST, and its refusals arrive whole
// ===========================================================================================

/// Corrupted bytes are refused by the production loader, typed, before any gate runs.
///
/// The dataset and selection handed in are the HONEST ones, so the artifact is the only
/// thing wrong — which makes a `Codec(Artifact { .. })` here evidence that the ladder ran
/// before the provenance comparison rather than after it.
#[test]
fn apr_reload_corrupt_bytes_are_refused_by_the_production_loader() {
    let mut bytes = apr_artifact_bytes();
    let (dataset, selection) = fresh_process_inputs();
    let midpoint = bytes.len() / 2;
    bytes[midpoint] ^= 0xff;

    let error = reload_verified_run_from_apr(&bytes, &dataset, &selection)
        .expect_err("a corrupted artifact must not reload");

    let SetFitTrainError::Codec(CodecError::Artifact { format_id, source }) = error else {
        panic!("expected the core loader's typed refusal, carried whole, got {error:?}");
    };
    assert_eq!(format_id, APR_FORMAT_ID);
    assert!(
        !matches!(source, SetFitArtifactError::ProbeComputation { .. }),
        "a flipped byte belongs to an integrity rung, not to probe computation: {source:?}",
    );
}

/// A MISMATCHED artifact is refused by the loader, not by the gate, when both are wrong.
///
/// Order evidence from the other side: empty bytes with an honest selection produce the
/// loader's refusal, so nothing about the caller's inputs is examined for an artifact that
/// cannot be verified.
#[test]
fn apr_reload_empty_bytes_produce_no_credential() {
    let (dataset, selection) = fresh_process_inputs();
    let error = reload_verified_run_from_apr(&[], &dataset, &selection)
        .expect_err("an empty slice is not an artifact");
    assert!(
        matches!(error, SetFitTrainError::Codec(CodecError::Artifact { .. })),
        "expected the loader's typed refusal, got {error:?}",
    );
}

// ===========================================================================================
// Source assertions: what this module must NOT contain
// ===========================================================================================

/// The reload module fabricates no evidence and mints no lifecycle state.
///
/// 04-16 refused to mint `SetFitRun<ArtifactReloadedAndVerified>` because five of
/// `HeadFittedEvidence`'s seven fields are unrecoverable and `PassedEvidence`'s fields are
/// private (E0451). If this module ever grows a way to supply, default or reconstruct any of
/// them, that refusal was for nothing — so the scan lives here as well as in
/// `credential_tests.rs`.
///
/// Scanned over CODE LINES ONLY: the module header explains at length why the evidence is
/// deliberately absent, and that explanation is worth more than the names' absence.
#[test]
fn apr_reload_module_fabricates_no_evidence_and_mints_no_state() {
    const FORBIDDEN: [&str; 7] = [
        "HeadFittedEvidence",
        "PassedEvidence",
        "UpdateEvidence",
        "validate_evidence",
        "verify_artifact",
        "unimplemented!",
        "todo!",
    ];

    let code: Vec<&str> =
        RELOAD_SOURCE.lines().filter(|line| !line.trim_start().starts_with("//")).collect();
    assert!(code.len() > 20, "non-vacuity: the filter must not have eaten the module");

    for line in &code {
        for forbidden in FORBIDDEN {
            assert!(
                !line.contains(forbidden),
                "`{forbidden}` appears on a CODE line of apr_reload.rs: `{line}`. This door \
                 mints a credential from the production loader plus a provenance gate; it \
                 neither reconstructs train-time evidence nor re-enters the train-time \
                 minting policy",
            );
        }
        assert!(
            !line.contains("SetFitRun {") && !line.contains("ArtifactVerifiedEvidence {"),
            "apr_reload.rs constructs a lifecycle value: `{line}`",
        );
    }

    // And the explanation really is there, so deleting the prose is caught as well as
    // adding the code.
    for named in ["HeadFittedEvidence", "PassedEvidence", "E0451"] {
        assert!(
            RELOAD_SOURCE.contains(named),
            "the module header must keep explaining why `{named}` is not reconstructed here",
        );
    }
}

/// The production loader is called, and it is called BEFORE the provenance gate.
///
/// A rung order visible in the source rather than only intended. Byte offsets are the whole
/// assertion: a door that gated first and loaded second would let a caller learn whether
/// their selection matches an artifact that cannot even be parsed.
#[test]
fn apr_reload_calls_the_production_loader_before_the_provenance_gate() {
    let body = fx::source_block_after(RELOAD_SOURCE, "pub fn reload_verified_run_from_apr(");

    let load_at =
        body.find("load_setfit_apr(bytes)").expect("the production loader must be called");
    let build_at = body
        .find("Ok(ReloadedSetFitCredential {")
        .expect("the credential must be built in this function");

    // Coarse to fine, and all three before the credential exists.
    let gates = [
        "DatasetFingerprintMismatch",
        "SelectionLedgerHashMismatch",
        "SelectionSemanticHashMismatch",
    ];
    let mut previous = load_at;
    for gate in gates {
        let at = body.find(gate).unwrap_or_else(|| panic!("`{gate}` must be raised here"));
        assert!(
            previous < at,
            "the gates must appear coarse to fine, after the loader: `{gate}` is out of order",
        );
        assert!(at < build_at, "`{gate}` must be checked before the credential exists");
        previous = at;
    }
}

/// The ledger comparison ships with its REASON, and not either false one.
///
/// The first draft declined the check by claiming a reloading process is entitled to an
/// audit trail of its own — false for a manifest-loaded selection. The plan that replaced it justified
/// the check by a distinction the semantic hash supposedly could not make — also false, and
/// measured so by
/// `apr_reload_the_three_recorded_identifiers_are_ordered_coarse_to_fine`. Neither may ship.
#[test]
fn apr_reload_the_ledger_rationale_is_the_true_one() {
    assert!(
        !RELOAD_SOURCE.contains("own ledger state"),
        "the first draft's incorrect rationale must not ship",
    );
    assert!(
        RELOAD_SOURCE.contains("selection_ledger_hash"),
        "the ledger hash must be compared by name",
    );
    assert!(
        RELOAD_SOURCE.contains("Selection::replay"),
        "and the comment must cite the manifest replay path that makes it comparable",
    );
    assert!(
        RELOAD_SOURCE.contains("SHA-256` of the WHOLE `SelectionPayload"),
        "and it must state the nesting the gate order depends on",
    );
}

/// The door hands back a credential, not a lifecycle state, and borrows its inputs.
#[test]
fn apr_reload_signature_returns_a_credential_and_borrows_its_inputs() {
    let signature =
        fx::source_signature_after(RELOAD_SOURCE, "pub fn reload_verified_run_from_apr(");

    assert!(signature.contains("bytes: &[u8]"), "the artifact arrives as a slice: {signature}");
    assert!(
        signature.contains("dataset: &PreparedDataset<Canonical>"),
        "the dataset is BORROWED — `grant` needs it next: {signature}",
    );
    assert!(signature.contains("selection: &Selection"), "and so is the selection: {signature}");
    assert!(
        signature.contains("Result<ReloadedSetFitCredential, SetFitTrainError>"),
        "the door returns a credential, never a lifecycle state: {signature}",
    );
}

// ===========================================================================================
// The credential itself
// ===========================================================================================

/// `Debug` prints the three digests and NOT the model.
///
/// Exact string equality, not `contains`: a derive that printed the hashes and then the
/// whole rebuilt encoder would satisfy a containment check. 04-17 measured the same hazard
/// on a 1.74 MiB buffer; a tensor set is worse.
#[test]
fn apr_reload_credential_debug_does_not_print_the_model() {
    let bytes = apr_artifact_bytes();
    let (dataset, selection) = fresh_process_inputs();
    let credential =
        reload_verified_run_from_apr(&bytes, &dataset, &selection).expect("the reload must mint");

    let rendered = format!("{credential:?}");
    assert_eq!(
        rendered,
        format!(
            "ReloadedSetFitCredential {{ artifact_hash: \"{}\", selection_semantic_hash: \
             \"{}\", selection_ledger_hash: \"{}\" }}",
            credential.artifact_hash(),
            credential.selection_semantic_hash(),
            hex::encode(credential.selection_ledger_hash()),
        ),
        "the credential's Debug must be the three digests and nothing else",
    );
}

/// The credential carries the verified model through, so `apr eval` need not load twice.
#[test]
fn apr_reload_credential_carries_the_verified_model() {
    let run = apr_capable_run();
    let bytes = apr_artifact_bytes();
    let (dataset, selection) = fresh_process_inputs();
    let credential =
        reload_verified_run_from_apr(&bytes, &dataset, &selection).expect("the reload must mint");

    assert_eq!(credential.model().ordered_labels(), run.evidence().ordered_labels());
    assert_eq!(credential.model().artifact_sha256(), credential.artifact_hash());

    let model = credential.into_model();
    let embedded = model
        .embed(&["the quick brown fox".to_string()])
        .expect("the verified model embeds through the path rung 8 replayed");
    assert_eq!(embedded.len(), 1);
}

/// The trait accessors return exactly what the inherent ones do.
///
/// Called through the trait explicitly, because a bare method call resolves to the INHERENT
/// one — which would compare a value with itself and pass for any impl whatsoever.
#[test]
fn apr_reload_credential_trait_and_inherent_accessors_agree() {
    let bytes = apr_artifact_bytes();
    let (dataset, selection) = fresh_process_inputs();
    let credential =
        reload_verified_run_from_apr(&bytes, &dataset, &selection).expect("the reload must mint");

    assert_eq!(
        SetFitCredential::artifact_hash(&credential),
        ReloadedSetFitCredential::artifact_hash(&credential),
    );
    assert_eq!(
        SetFitCredential::selection_semantic_hash(&credential),
        ReloadedSetFitCredential::selection_semantic_hash(&credential),
    );
    assert_eq!(
        SetFitCredential::selection_ledger_hash(&credential),
        ReloadedSetFitCredential::selection_ledger_hash(&credential),
    );

    // Non-vacuity: the three are genuinely different values, so an impl returning one
    // string three times could not satisfy the assertions above.
    assert_ne!(
        SetFitCredential::artifact_hash(&credential),
        SetFitCredential::selection_semantic_hash(&credential),
    );
}
