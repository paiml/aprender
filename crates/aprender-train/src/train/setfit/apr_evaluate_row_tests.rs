//! Per-row-predictions evaluator tests (plan 05-08, EVAL-01).
//!
//! Every test name starts `apr_evaluate_rows_`.
//!
//! # Why a SECOND test file beside `apr_evaluate_tests.rs`
//!
//! `apr_evaluate_tests.rs` is 04-07's, and this plan's acceptance criteria require it to be
//! byte-untouched — a file whose assertions have been edited is no longer independent evidence
//! that the scalar door still behaves as it did. So the new door's tests live here and REUSE
//! that file's fixture construction shape rather than editing it.
//!
//! The fixture estate is 04-16's, for the reason `apr_evaluate_tests.rs` records: it is the only
//! run in this crate that can produce real `setfit-apr-v1` bytes, and a second APR-capable
//! fixture would be free to drift in exactly the dimensions the load ladder's probes are
//! sensitive to.

use aprender_contrastive_data::ledger::AccessLedger;
use aprender_contrastive_data::manifest::SelectionManifest;
use aprender_contrastive_data::select::{FewShotSelector, Selection, SelectionConfig};

use super::super::apr_codec::fixture::{apr_capable_run, artifact_bytes_of};
use super::super::apr_reload::reload_verified_run_from_apr;
use super::super::lock::{
    create_selection_lock, CanonicalTestAccess, SelectionCandidate, SelectionRule,
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
/// The same shape `apr_evaluate_tests::fresh_process_inputs` uses, and for the same reason: a
/// test that handed the door back the objects the artifact was written from would prove the gate
/// passes by IDENTITY, which is not the property claimed.
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

/// A credential minted the way `bench run` mints one, plus the dataset it was gated against.
fn fresh_credential() -> (ReloadedSetFitCredential, PreparedDataset<Canonical>) {
    let bytes = apr_artifact_bytes();
    let (dataset, selection) = fresh_process_inputs();
    let credential = reload_verified_run_from_apr(&bytes, &dataset, &selection)
        .expect("the artifact and the independently-rebuilt inputs must agree");
    (credential, dataset)
}

/// Non-comment source of this module's subject.
fn apr_evaluate_code() -> String {
    APR_EVALUATE_SOURCE
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

// ===========================================================================================
// The door returns per-row evidence, not a scalar
// ===========================================================================================

/// Every row gets a predicted index and a full K-vector, and the labels come with them.
#[test]
fn apr_evaluate_rows_returns_one_prediction_and_one_probability_vector_per_row() {
    let (credential, dataset) = fresh_credential();

    let rows = evaluate_rows_from_artifact(&credential, &dataset, EvaluatedSplit::Validation)
        .expect("a reloaded artifact must be measurable on the corpus it was gated against");

    let expected_rows = dataset.validation().rows().len();
    assert!(expected_rows > 0, "non-vacuity: the fixture's validation split must have rows");
    assert_eq!(rows.n_rows(), expected_rows, "one entry per canonical validation row");
    assert_eq!(rows.predicted().len(), expected_rows);
    assert_eq!(rows.truth().len(), expected_rows);
    assert_eq!(rows.probabilities().len(), expected_rows);

    let classes = rows.ordered_labels().len();
    assert_eq!(classes, dataset.label_names().len(), "K is the DECLARED label map's size");
    for row in rows.probabilities() {
        assert_eq!(row.len(), classes, "a K-vector per row, in ordered-label order");
        let mass: f64 = row.iter().sum();
        assert!(
            (mass - 1.0).abs() < 1e-6,
            "each row's probability mass must be 1; got {mass}. A renormalised or truncated \
             vector would make every calibration number below quietly wrong",
        );
        for &p in row {
            assert!(p.is_finite(), "a non-finite probability would serialise as JSON null");
        }
    }
    for (&predicted, probabilities) in rows.predicted().iter().zip(rows.probabilities()) {
        assert!(predicted < classes, "a predicted index must be inside the declared label map");
        let argmax = probabilities
            .iter()
            .enumerate()
            .fold(
                (0usize, f64::NEG_INFINITY),
                |best, (i, &p)| if p > best.1 { (i, p) } else { best },
            )
            .0;
        assert_eq!(
            predicted, argmax,
            "the predicted index must be the argmax of the row it is reported with; a \
             disagreement means the two halves came from different places",
        );
    }
    for &truth in rows.truth() {
        assert!(truth < classes, "a truth index must be inside the declared label map");
    }

    // The rows carry the artifact they were measured with, so a caller stamps a bench row
    // without re-deriving it (and therefore without an opportunity to derive it differently).
    assert_eq!(rows.artifact_hash(), credential.artifact_hash());
    assert_eq!(rows.split_tag(), "validation");
}

/// The two doors CANNOT disagree: accuracy recomputed from the rows is the scalar door's value.
#[test]
fn apr_evaluate_rows_accuracy_equals_the_scalar_doors_value() {
    let (credential, dataset) = fresh_credential();

    let scalar =
        evaluate_validation_from_artifact(&credential, &dataset, ValidationMetricKind::Accuracy)
            .expect("the scalar door must measure this artifact");
    let rows = evaluate_rows_from_artifact(&credential, &dataset, EvaluatedSplit::Validation)
        .expect("the row door must measure the same artifact");

    assert_eq!(rows.n_rows(), scalar.n_rows(), "both doors must measure the SAME split");

    let hits =
        rows.truth().iter().zip(rows.predicted()).filter(|(actual, guess)| actual == guess).count();
    #[allow(clippy::cast_precision_loss)]
    let recomputed = hits as f64 / rows.n_rows() as f64;
    assert!(
        (recomputed - scalar.value()).abs() < 1e-12,
        "rows-derived accuracy {recomputed} must equal the scalar door's {}; two doors over one \
         artifact that disagree are two evaluation policies, which OPS-03 forbids",
        scalar.value(),
    );
}

/// Two calls agree by BITS: the door is deterministic, like the scalar one.
#[test]
fn apr_evaluate_rows_is_bitwise_deterministic() {
    let (credential, dataset) = fresh_credential();
    let first = evaluate_rows_from_artifact(&credential, &dataset, EvaluatedSplit::Validation)
        .expect("measurable");
    let second = evaluate_rows_from_artifact(&credential, &dataset, EvaluatedSplit::Validation)
        .expect("measurable");

    assert_eq!(first.predicted(), second.predicted());
    assert_eq!(first.truth(), second.truth());
    let bits = |rows: &RowPredictions| -> Vec<u64> {
        rows.probabilities().iter().flatten().map(|p| p.to_bits()).collect()
    };
    assert_eq!(
        bits(&first),
        bits(&second),
        "two runs must agree BY BITS; a decimal comparison would accept two values differing in \
         the last place, and a calibration number is computed from these bits",
    );
}

// ===========================================================================================
// The refusals — the SAME identity re-check the scalar door performs
// ===========================================================================================

/// A different corpus is the same typed refusal the scalar door returns.
#[test]
fn apr_evaluate_rows_refuses_a_dataset_that_is_not_the_artifacts_corpus() {
    let (credential, own) = fresh_credential();
    let other = fx::dataset_with_altered_test_row();
    assert_ne!(
        other.validation_witness().dataset_fingerprint_hex(),
        own.validation_witness().dataset_fingerprint_hex(),
        "non-vacuity: the probe corpus must actually differ, or this test proves nothing",
    );

    let scalar_error =
        evaluate_validation_from_artifact(&credential, &other, ValidationMetricKind::Accuracy)
            .expect_err("the scalar door refuses another corpus");
    let row_error = evaluate_rows_from_artifact(&credential, &other, EvaluatedSplit::Validation)
        .expect_err("a per-row measurement on another corpus is not evidence about this artifact");

    assert!(
        matches!(
            row_error,
            SetFitTrainError::AprEvaluate(AprEvaluateError::DatasetFingerprintMismatch { .. })
        ),
        "the corpus disagreement must be the SAME typed refusal; got: {row_error}",
    );
    assert_eq!(
        row_error.to_string(),
        scalar_error.to_string(),
        "the two doors must refuse identically — a second wording is a second policy",
    );
}

/// Both doors take the SAME credential type, so neither is reachable without the reload door.
#[test]
fn apr_evaluate_rows_takes_the_same_credential_type_as_the_scalar_door() {
    let code = apr_evaluate_code();
    assert!(
        code.contains("pub fn evaluate_rows_from_artifact"),
        "non-vacuity: the comment filter must not have eaten the module",
    );
    let credential_parameters = code.matches("credential: &ReloadedSetFitCredential").count();
    assert!(
        credential_parameters >= 2,
        "both doors must name `&ReloadedSetFitCredential`; found {credential_parameters}. A \
         weaker parameter here would be an evaluation reachable without the reload ladder",
    );
}

/// There is ONE classify loop; the scalar door delegates rather than duplicating it.
#[test]
fn apr_evaluate_rows_shares_one_classify_loop_with_the_scalar_door() {
    let code = apr_evaluate_code();
    let loops = code.matches("model.classify(&request)").count();
    assert_eq!(
        loops, 1,
        "exactly ONE classify site may exist in this module; found {loops}. Two prediction loops \
         over one artifact are two float pipelines that must agree and eventually will not",
    );
    let identity_checks = code.matches("fn check_artifact_identity").count();
    assert_eq!(
        identity_checks, 1,
        "the artifact-vs-dataset identity re-check must be ONE function both doors call",
    );
}

// ===========================================================================================
// The TEST split is reachable only through the token gate
// ===========================================================================================

/// A grant opens the test split, and its rows are the canonical test rows.
#[test]
fn apr_evaluate_rows_measures_the_test_split_only_through_a_grant() {
    let (credential, dataset) = fresh_credential();

    let evaluation =
        evaluate_validation_from_artifact(&credential, &dataset, ValidationMetricKind::Accuracy)
            .expect("the reloaded artifact is measurable on its own corpus");
    let lock = create_selection_lock(
        &credential,
        vec![SelectionCandidate::from_evaluation("cfg-hash", evaluation)],
        SelectionRule::MaxMetricLowestIndexTieBreak,
    )
    .expect("the credential's own artifact is in the candidate set");
    let token = lock.mint_test_token(&credential).expect("the lock names this artifact as chosen");
    let grant = CanonicalTestAccess::grant(token, &credential, &dataset)
        .expect("the corpus is the one the lock was taken over");

    let rows = evaluate_rows_from_artifact(&credential, &dataset, EvaluatedSplit::Test(&grant))
        .expect("a granted test split is measurable");

    assert_eq!(rows.n_rows(), dataset.test().rows().len(), "the CANONICAL test rows");
    assert_eq!(rows.split_tag(), "test");
    assert_eq!(
        rows.truth(),
        dataset.test().rows().iter().map(|row| row.label).collect::<Vec<_>>(),
        "the truth vector must be the test split's own labels, in row order",
    );

    // The validation measurement is a DIFFERENT split, so the two must not be interchangeable.
    let validation = evaluate_rows_from_artifact(&credential, &dataset, EvaluatedSplit::Validation)
        .expect("measurable");
    assert_ne!(
        validation.split_tag(),
        rows.split_tag(),
        "non-vacuity: the two splits must be distinguishable in the returned evidence, or a \
         caller could feed test probabilities to a validation-only calibration",
    );
}

/// A grant for a DIFFERENT artifact does not open this credential's test split.
#[test]
fn apr_evaluate_rows_refuses_a_grant_that_belongs_to_another_artifact() {
    let (credential, dataset) = fresh_credential();
    let evaluation =
        evaluate_validation_from_artifact(&credential, &dataset, ValidationMetricKind::Accuracy)
            .expect("measurable");
    let lock = create_selection_lock(
        &credential,
        vec![SelectionCandidate::from_evaluation("cfg-hash", evaluation)],
        SelectionRule::MaxMetricLowestIndexTieBreak,
    )
    .expect("the credential's own artifact is in the candidate set");
    let token = lock.mint_test_token(&credential).expect("the lock names this artifact");
    let grant = CanonicalTestAccess::grant(token, &credential, &dataset).expect("granted");

    // A second, independently reloaded credential over the SAME bytes has the same hash, so a
    // hash-only probe would be vacuous. The probe therefore needs a genuinely different
    // artifact — and no other APR-capable fixture exists (F-10). So the assertion is made on
    // the MECHANISM instead: the door compares the grant's artifact hash to the credential's,
    // which is a comparison a `grant` value carried across scopes can fail.
    assert_eq!(
        grant.artifact_hash(),
        credential.artifact_hash(),
        "the grant this test holds belongs to this credential",
    );
    let code = apr_evaluate_code();
    assert!(
        code.contains("grant.artifact_hash()"),
        "the test-split arm must RE-CHECK the grant against the credential; a grant is a value \
         that can be moved into a struct and used against whatever credential is in scope three \
         functions later",
    );
    assert!(
        code.contains("TestGrantArtifactMismatch"),
        "and the re-check must be a TYPED refusal, not a debug assertion",
    );
}

// ===========================================================================================
// The structural claims (mirroring the scalar door's, for the new function)
// ===========================================================================================

/// The row door computes NO metric of its own.
#[test]
fn apr_evaluate_rows_computes_no_metric() {
    let code = apr_evaluate_code();
    for banned in [
        "fn accuracy(",
        "fn macro_f1(",
        "fn f_avg(",
        "expected_calibration_error",
        "brier_score",
        "matthews_corrcoef",
        "MultiClassMetrics",
    ] {
        assert!(
            !code.contains(banned),
            "`{banned}` here would make this door a metric implementation; it returns \
             PREDICTIONS, and the assembly lives in `bench_metrics`",
        );
    }
}

/// An empty split is a typed refusal, never an empty prediction vector.
#[test]
fn apr_evaluate_rows_refusal_for_an_empty_split_is_typed() {
    // The fixture corpus cannot present an empty canonical split — Phase 2's ingest ladder
    // refuses one — so the assertion is on the door's own code path, which is the same
    // `ValidationSplitEmpty` the scalar door returns.
    let code = apr_evaluate_code();
    assert!(
        code.contains("ValidationSplitEmpty"),
        "an empty split must be the typed refusal, not an Ok over zero rows: a zero-row \
         RowPredictions would flow into the assembly and produce NaN metrics",
    );
    let rendered = AprEvaluateError::ValidationSplitEmpty.to_string();
    assert!(!rendered.is_empty(), "the refusal must render");
}
