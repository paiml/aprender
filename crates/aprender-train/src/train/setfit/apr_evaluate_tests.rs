//! Fresh-process evaluation tests (plan 04-07, TRN-07).
//!
//! Every test name starts `apr_evaluate_`.
//!
//! The fixture estate is 04-16's: the ONLY run in this crate that can produce real
//! `setfit-apr-v1` bytes, plus inputs an independent process would rebuild from Phase 2's
//! ingest ladder and a persisted selection manifest. It is REUSED, never copied — a second
//! APR-capable fixture would be free to drift in exactly the dimensions the load ladder's
//! probes are sensitive to.

use aprender_contrastive_data::ledger::AccessLedger;
use aprender_contrastive_data::manifest::SelectionManifest;
use aprender_contrastive_data::select::{FewShotSelector, Selection, SelectionConfig};

use super::super::apr_codec::fixture::{apr_capable_run, artifact_bytes_of};
use super::super::apr_reload::reload_verified_run_from_apr;
use super::super::evaluate::evaluate_validation;
use super::super::lock::{
    create_selection_lock, CanonicalTestAccess, LockError, SelectionCandidate, SelectionLock,
    SelectionRule,
};
use super::super::test_fixtures as fx;
use super::*;

/// This module's subject, for the source assertions.
const APR_EVALUATE_SOURCE: &str = include_str!("apr_evaluate.rs");

/// The artifact under test, as bytes.
fn apr_artifact_bytes() -> Vec<u8> {
    artifact_bytes_of(&apr_capable_run())
}

/// The dataset and selection a SEPARATE process would hold, rebuilt from scratch.
///
/// Identical in shape to `apr_reload_tests::fresh_process_inputs` and for the same reason: a
/// test that handed the door back the objects the artifact was written from would prove the
/// gate passes by IDENTITY, which is not the property claimed. Routing the selection through
/// a persisted manifest and `Selection::replay` is what makes the ledger hash comparable.
fn fresh_process_inputs() -> (PreparedDataset<Canonical>, Selection) {
    let variant = fx::calibrated_variant();

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

    let mut fresh_ledger = AccessLedger::new();
    let fresh_dataset = fx::synthetic_dataset(&mut fresh_ledger);
    let restored = SelectionManifest::from_bytes(&bytes).expect("the manifest must parse back");
    let replayed = Selection::replay(&restored, &fresh_dataset, &mut fresh_ledger)
        .expect("the manifest must replay against the rebuilt corpus");
    (fresh_dataset, replayed)
}

/// A credential minted the way `apr eval` mints one, plus the dataset it was gated against.
fn fresh_credential() -> (ReloadedSetFitCredential, PreparedDataset<Canonical>) {
    let bytes = apr_artifact_bytes();
    let (dataset, selection) = fresh_process_inputs();
    let credential = reload_verified_run_from_apr(&bytes, &dataset, &selection)
        .expect("the artifact and the independently-rebuilt inputs must agree");
    (credential, dataset)
}

// ===========================================================================================
// The claim: a process that trained nothing can measure, and therefore can be a candidate
// ===========================================================================================

/// The gap 04-16 recorded open is closed: a `.apr` file becomes a `SelectionCandidate`.
#[test]
fn apr_evaluate_lets_a_fresh_process_build_the_candidate_set_it_locks_over() {
    let (credential, dataset) = fresh_credential();

    let evaluation =
        evaluate_validation_from_artifact(&credential, &dataset, ValidationMetricKind::Accuracy)
            .expect("a reloaded artifact must be measurable on the corpus it was gated against");

    // The measurement is REAL: a value in range, over a non-empty split, bound to this
    // artifact. A test that only asserted `is_ok` would pass for an evaluator that returned a
    // constant.
    assert!(
        (0.0..=1.0).contains(&evaluation.value()),
        "accuracy must be a fraction, got {}",
        evaluation.value()
    );
    assert!(evaluation.n_rows() > 0, "the split measured must be non-empty");
    assert_eq!(
        evaluation.artifact_hash(),
        credential.artifact_hash(),
        "the evaluation must be bound to the artifact it was computed with, read off the \
         credential rather than supplied",
    );

    // And it is what a candidate is made of — the whole point of the door.
    let candidate = SelectionCandidate::from_evaluation("cfg-hash-for-this-test", evaluation);
    let lock = create_selection_lock(
        &credential,
        vec![candidate],
        SelectionRule::MaxMetricLowestIndexTieBreak,
    )
    .expect("the credential's own artifact is in the candidate set");
    assert_eq!(
        lock.chosen_artifact_hash(),
        credential.artifact_hash(),
        "the lock must record THIS artifact as the chosen one",
    );
    assert_eq!(lock.lock_hash().len(), 64, "a lock hash is a SHA-256");
}

/// The value is DETERMINISTIC: two evaluations of one artifact agree down to the bits.
#[test]
fn apr_evaluate_is_bitwise_deterministic_for_one_artifact_and_corpus() {
    let (credential, dataset) = fresh_credential();
    let first =
        evaluate_validation_from_artifact(&credential, &dataset, ValidationMetricKind::MacroF1)
            .expect("measurable");
    let second =
        evaluate_validation_from_artifact(&credential, &dataset, ValidationMetricKind::MacroF1)
            .expect("measurable");
    assert_eq!(
        first.value_bits(),
        second.value_bits(),
        "two runs must agree BY BITS; a decimal comparison would accept two values differing \
         in the last place, and the selection lock hashes the bits",
    );
    assert_eq!(first.metric_kind(), ValidationMetricKind::MacroF1);
}

/// Both metric kinds are reachable and the closed enum is honoured.
#[test]
fn apr_evaluate_computes_both_contracted_metric_kinds() {
    let (credential, dataset) = fresh_credential();
    for kind in [ValidationMetricKind::Accuracy, ValidationMetricKind::MacroF1] {
        let evaluation =
            evaluate_validation_from_artifact(&credential, &dataset, kind).expect("measurable");
        assert_eq!(evaluation.metric_kind(), kind);
        assert!((0.0..=1.0).contains(&evaluation.value()));
    }
}

/// The committed facts are read off the CORPUS's witness, not off the artifact's copy.
#[test]
fn apr_evaluate_commits_both_fingerprints_from_the_supplied_datasets_witness() {
    let (credential, dataset) = fresh_credential();
    let evaluation =
        evaluate_validation_from_artifact(&credential, &dataset, ValidationMetricKind::Accuracy)
            .expect("measurable");

    let witness = dataset.validation_witness();
    assert_eq!(evaluation.dataset_fingerprint(), witness.dataset_fingerprint_hex());
    assert_eq!(evaluation.validation_split_fingerprint(), witness.fingerprint_hex());
    assert_ne!(
        evaluation.dataset_fingerprint(),
        evaluation.validation_split_fingerprint(),
        "the two fingerprints digest different things; equal values would mean one path was \
         read twice",
    );
}

// ===========================================================================================
// The refusals
// ===========================================================================================

/// A different corpus is refused, naming both fingerprints.
#[test]
fn apr_evaluate_refuses_a_dataset_that_is_not_the_artifacts_corpus() {
    let (credential, own) = fresh_credential();

    // ONE TEST ROW ALTERED, and nothing else. That is the sharpest available probe: the
    // VALIDATION split is byte-identical, so an evaluator that only compared the split's
    // fingerprint would happily measure this corpus and report a number about a dataset the
    // artifact was never trained on.
    let other = fx::dataset_with_altered_test_row();
    assert_ne!(
        other.validation_witness().dataset_fingerprint_hex(),
        own.validation_witness().dataset_fingerprint_hex(),
        "non-vacuity: the probe corpus must actually differ, or this test proves nothing",
    );
    assert_eq!(
        other.validation_witness().fingerprint_hex(),
        own.validation_witness().fingerprint_hex(),
        "the probe must hold the VALIDATION split constant, so only the corpus check can fire",
    );

    let error =
        evaluate_validation_from_artifact(&credential, &other, ValidationMetricKind::Accuracy)
            .expect_err("a metric on another corpus is not evidence about this artifact");
    let rendered = error.to_string();
    assert!(
        matches!(
            error,
            SetFitTrainError::AprEvaluate(AprEvaluateError::DatasetFingerprintMismatch { .. })
        ),
        "the corpus disagreement must be its own typed refusal; got: {rendered}",
    );
    assert!(
        rendered.contains(&other.validation_witness().dataset_fingerprint_hex()),
        "the refusal must name the supplied fingerprint; got: {rendered}",
    );
}

/// The refusals are distinct values, and each rendering names both compared values.
#[test]
fn apr_evaluate_refusals_are_distinct_and_name_what_they_compared() {
    let a = AprEvaluateError::DatasetFingerprintMismatch {
        recorded: "aaaa".to_string(),
        supplied: "bbbb".to_string(),
    };
    let b = AprEvaluateError::ValidationSplitFingerprintMismatch {
        recorded: "cccc".to_string(),
        supplied: "dddd".to_string(),
    };
    let c = AprEvaluateError::LabelMapMismatch {
        artifact: vec!["x".to_string()],
        dataset: vec!["y".to_string()],
    };
    let d = AprEvaluateError::ProvenanceUnreadable { field: "dataset_fingerprint" };
    let e = AprEvaluateError::ValidationSplitEmpty;

    let rendered: Vec<String> = [&a, &b, &c, &d, &e].iter().map(|x| x.to_string()).collect();
    for (i, left) in rendered.iter().enumerate() {
        for (j, right) in rendered.iter().enumerate() {
            if i != j {
                assert_ne!(left, right, "two refusals must not read the same");
            }
        }
    }
    assert!(rendered[0].contains("aaaa") && rendered[0].contains("bbbb"));
    assert!(rendered[1].contains("cccc") && rendered[1].contains("dddd"));
    assert!(rendered[2].contains('x') && rendered[2].contains('y'));
    assert!(rendered[3].contains("dataset_fingerprint"));
}

// ===========================================================================================
// The structural claims
// ===========================================================================================

/// This module computes NO metric and constructs NO evaluation of its own.
#[test]
fn apr_evaluate_reduction_is_the_trainers_own() {
    // CODE LINES only: the module header explains the sharing at length using the very names
    // scanned for, and a guard that fails on its own documentation is the F-05 defect.
    let code: String = APR_EVALUATE_SOURCE
        .lines()
        .filter(|line| {
            let t = line.trim_start();
            !t.starts_with("//")
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        code.contains("fn evaluate_validation_from_artifact"),
        "non-vacuity: the comment filter must not have eaten the module",
    );

    assert!(
        code.contains("evaluate::evaluation_from_predictions("),
        "the bounds check, the metric dispatch and the record construction must be the \
         trainer's shared tail, called — not restated",
    );
    for banned in [
        "fn accuracy(",
        "fn macro_f1(",
        "ValidationEvaluation {",
        "mean_in_index_order",
        "true_positive",
    ] {
        assert!(
            !code.contains(banned),
            "`{banned}` here would be a SECOND reduction implementation; OPS-03 says one per \
             operation, and two float pipelines that must agree are two that eventually will \
             not",
        );
    }
}

/// The prediction path is core's ONE classify door, not a second encode.
#[test]
fn apr_evaluate_predicts_through_cores_single_classification_path() {
    let code: String = APR_EVALUATE_SOURCE
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        code.contains("model.classify(&request)"),
        "prediction must go through `VerifiedSetFitModel::classify` — the path the artifact's \
         own probe replay is evidence about",
    );
    assert!(
        code.contains("MAX_BATCH_TEXTS"),
        "the split must be chunked by core's own bound; a single oversized call would be \
         REFUSED rather than measured",
    );
    for banned in ["encode_eval_rows", "predict_indices", "encode_batch_traced"] {
        assert!(
            !code.contains(banned),
            "`{banned}` would be a second encode path, measuring something the artifact carries \
             no probes for",
        );
    }
}

/// No lifecycle state is minted and no evidence is fabricated here.
#[test]
fn apr_evaluate_mints_no_state_and_fabricates_no_evidence() {
    let code: String = APR_EVALUATE_SOURCE
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    for banned in [
        "HeadFittedEvidence",
        "PassedEvidence",
        "UpdateEvidence",
        "validate_evidence",
        "verify_artifact",
        "SetFitRun {",
        "unimplemented!",
        "todo!",
    ] {
        assert!(!code.contains(banned), "this module must not construct or fabricate `{banned}`",);
    }
}

// ===========================================================================================
// TRN-07: the durable selection lock, across two invocations, mediated by a FILE
// ===========================================================================================

/// Build the lock a `--split validation --lock-out` invocation would commit.
fn committed_lock(
    credential: &ReloadedSetFitCredential,
    dataset: &PreparedDataset<Canonical>,
) -> SelectionLock {
    let evaluation =
        evaluate_validation_from_artifact(credential, dataset, ValidationMetricKind::Accuracy)
            .expect("the reloaded artifact is measurable on its own corpus");
    let candidate = SelectionCandidate::from_evaluation("cfg-hash", evaluation);
    create_selection_lock(credential, vec![candidate], SelectionRule::MaxMetricLowestIndexTieBreak)
        .expect("the creating model is in its own candidate set")
}

/// TRN-07's POSITIVE tier: a user reaches lock -> token -> grant across two invocations.
///
/// # This is the IN-PROCESS, FILE-MEDIATED proof
///
/// The two halves below share no `SelectionLock` value: the first serializes to disk and drops
/// everything, the second reads the FILE and reconstructs through
/// `SelectionLock::from_canonical_bytes`. That is the property review finding B4 is about — the
/// selection must be COMMITTED before test access is taken, and a lock that only ever existed
/// in one call stack proves nothing about that ordering.
///
/// It is deliberately NOT labelled a cross-process proof. Two `std::process::Command`
/// invocations of the real `apr` binary are 04-15's deliverable
/// (`crates/apr-cli/tests/setfit_cli_lifecycle.rs`); this test runs in one process and its
/// claim is exactly the FILE boundary, not the process boundary.
///
/// It lives in `aprender-train` rather than beside `apr eval` because this is the only crate in
/// which a `setfit-apr-v1` artifact can be produced WITHOUT the 86.7 MB production checkout —
/// measured, not assumed: the phase-3 slice cannot compute two of the six probes, and core's
/// APR-capable view fixture is `#[cfg(test)] pub(crate)`, so `apr-cli`'s own suites have no
/// in-repository artifact to build a lock from.
///
/// **Before Phase 5's 05-03 calibration edit (commit `a63bb130b`), no user-reachable path
/// produced a `setfit-apr-v1` at all.** That is closed for the production encoder, and the
/// SPAWNED cross-process form of the claim below — a lock written by one `apr eval` process
/// and consumed by another — is proven end-to-end by 05-07's
/// `setfit_cli_production_chain_completes_after_the_calibration_edit`. This test's own scope is
/// unchanged: one process, the FILE boundary, no production checkout required.
#[test]
fn apr_evaluate_the_lock_travels_between_two_invocations_as_a_file() {
    let temp = tempfile::TempDir::new().expect("tempdir");
    let lock_path = temp.path().join("selection-lock.json");

    // ---- INVOCATION ONE: measure, commit, write. ------------------------------------
    let recorded_lock_hash = {
        let (credential, dataset) = fresh_credential();
        let lock = committed_lock(&credential, &dataset);
        std::fs::write(&lock_path, lock.to_canonical_bytes()).expect("the lock file is writable");
        lock.lock_hash().to_string()
        // `credential`, `dataset` and `lock` all drop here. Nothing but the FILE survives.
    };

    // ---- INVOCATION TWO: read the file, mint, grant, read the test rows. ------------
    let (credential, dataset) = fresh_credential();
    let bytes = std::fs::read(&lock_path).expect("the lock file is readable");
    let lock = SelectionLock::from_canonical_bytes(&bytes)
        .expect("a lock written by the previous invocation must reconstruct");
    assert_eq!(
        lock.lock_hash(),
        recorded_lock_hash,
        "the reconstructed lock must be the record the first invocation committed, not a fresh \
         opinion about a payload silently repaired on the way in",
    );

    let token = lock.mint_test_token(&credential).expect("the lock names this artifact as chosen");
    let grant = CanonicalTestAccess::grant(token, &credential, &dataset)
        .expect("the corpus is the one the lock was taken over");

    assert!(
        !grant.test().rows().is_empty(),
        "the grant must admit the canonical test rows; an empty split would make every claim \
         above vacuous",
    );
    assert_eq!(
        grant.artifact_hash(),
        credential.artifact_hash(),
        "the grant belongs to THIS artifact",
    );
    assert_eq!(
        grant.lock_hash(),
        recorded_lock_hash,
        "and it traces back to the lock the FIRST invocation wrote",
    );
}

/// A lock naming a DIFFERENT artifact is refused as stale, naming both hashes.
#[test]
fn apr_evaluate_a_lock_naming_a_different_artifact_is_refused_as_stale() {
    // A genuinely different artifact: the same fixture run closed with phase 3's debug codec,
    // whose bytes — and therefore whose digest — are not the APR artifact's.
    let other_run = fx::verified_run(fx::calibrated_variant());
    let other_dataset = fx::fixture_dataset();
    let other_evaluation =
        evaluate_validation(&other_run, &other_dataset, ValidationMetricKind::Accuracy)
            .expect("the fixture run is measurable on its own dataset");
    let other_lock = create_selection_lock(
        &other_run,
        vec![SelectionCandidate::from_evaluation("cfg-hash", other_evaluation)],
        SelectionRule::MaxMetricLowestIndexTieBreak,
    )
    .expect("the creating model is in its own candidate set");

    let (credential, _dataset) = fresh_credential();
    assert_ne!(
        other_lock.chosen_artifact_hash(),
        credential.artifact_hash(),
        "non-vacuity: the two artifacts must actually differ, or this test proves nothing",
    );

    let error = other_lock
        .mint_test_token(&credential)
        .expect_err("a lock that names another artifact must not admit this one");
    assert!(
        matches!(error, LockError::StaleLock { .. }),
        "the refusal must be typed StaleLock; got: {error:?}",
    );
    let rendered = error.to_string();
    assert!(
        rendered.contains(other_lock.chosen_artifact_hash())
            && rendered.contains(credential.artifact_hash()),
        "the refusal must name BOTH the locked hash and the observed one; got: {rendered}",
    );
}

/// A grant taken over a DIFFERENT corpus is refused, naming both fingerprints.
#[test]
fn apr_evaluate_a_grant_over_a_different_corpus_is_refused() {
    let (credential, dataset) = fresh_credential();
    let lock = committed_lock(&credential, &dataset);
    let token = lock.mint_test_token(&credential).expect("the lock names this artifact");

    // ONE TEST ROW ALTERED. The MODEL half of the substitution is held constant — the token is
    // valid and the artifact is the locked one — so only the DATA half can fire.
    let other = fx::dataset_with_altered_test_row();
    assert_ne!(
        other.validation_witness().dataset_fingerprint_hex(),
        dataset.validation_witness().dataset_fingerprint_hex(),
        "non-vacuity: the probe corpus must actually differ",
    );

    let error = CanonicalTestAccess::grant(token, &credential, &other)
        .expect_err("a valid token must not admit test rows from another corpus");
    assert!(
        matches!(error, LockError::TokenDatasetMismatch { .. }),
        "the refusal must be typed TokenDatasetMismatch; got: {error:?}",
    );
}
