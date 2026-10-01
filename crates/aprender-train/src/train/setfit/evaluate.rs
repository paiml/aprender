//! Canonical-validation evaluation: a metric is COMPUTED by trusted code, never asserted.
//!
//! Contract: `setfit-train-lifecycle-v1`, equation `validation_evaluation_provenance`.
//! Requirement: TRN-07.
//!
//! # The one sentence this module exists for
//!
//! An evaluation is a computation performed by trusted code from a verified artifact and a
//! typed canonical split — not a number a caller says it obtained.
//!
//! # What the review found, and what changed
//!
//! The shape this module replaces was `ValidationMetric::new(&Split<Validation>, name, value)`.
//! It looked safe: you had to hold a `Split<Validation>` to build one, and Phase 2's typestate
//! makes that split unreachable on a compatibility profile. But possessing a split does not
//! prove a number came OUT of it, and it does not prove the number was computed with the
//! artifact whose access the metric goes on to unlock. A caller holding the canonical dataset
//! could hand over any `f64` at all — including one measured on the TEST split, which is
//! precisely the substitution TRN-07 exists to block.
//!
//! So there is no constructor here that accepts a metric value. [`evaluate_validation`] takes
//! the verified run and the canonical dataset, predicts every validation row with THAT model,
//! computes the metric itself, and commits the artifact hash it read off the run together with
//! the validation-split fingerprint it read off the witness. A caller can choose WHICH metric
//! ([`ValidationMetricKind`] is a closed enum, so not even the metric's NAME is free-form) and
//! nothing else.
//!
//! # Why the fingerprint pair is two fields and not one
//!
//! `ValidationWitness::fingerprint_hex()` digests the VALIDATION SPLIT ALONE and
//! `dataset_fingerprint_hex()` digests the whole dataset; Phase 2 made them deliberately
//! distinct (`prepared.rs`). Committing both is what lets a downstream reader distinguish "a
//! different dataset" from "the same dataset whose validation rows changed", and it is what
//! the selection lock's candidate-consistency checks compare candidates on.
//!
//! # Reproducibility
//!
//! Every reduction goes through [`super::reduce`], in index order, accumulating in f64 — so the
//! value is bitwise stable and two evaluations of the same run are equal down to the bit
//! pattern. The encode goes through the ONE encode-once path
//! ([`super::head_input::encode_eval_rows`]) the trainer itself uses; a second encode path here
//! would measure a model the trainer never ran.

use serde::{Deserialize, Serialize};

use aprender_contrastive_data::prepared::{Canonical, PreparedDataset};

use super::head_input;
use super::reduce;
use super::{ArtifactReloadedAndVerified, SetFitRun, SetFitTrainError};

/// The canonical evaluation wire schema version.
const EVALUATION_SCHEMA_VERSION: u32 = 1;

/// The closed set of validation metrics.
///
/// A CLOSED ENUM rather than a metric name. A free-form `String` lets two evaluations that
/// measured different quantities present the same label — and the selection lock's
/// candidate-consistency check, which refuses candidates whose metric kinds disagree, would then
/// be comparing spellings instead of quantities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationMetricKind {
    /// Fraction of validation rows whose predicted class equals the recorded class.
    Accuracy,
    /// Unweighted mean of the per-class F1 scores over the DECLARED label map.
    MacroF1,
}

impl ValidationMetricKind {
    /// The metric's stable tag, as it appears in canonical bytes and in error messages.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Accuracy => "accuracy",
            Self::MacroF1 => "macro_f1",
        }
    }
}

impl core::fmt::Display for ValidationMetricKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.tag())
    }
}

/// One metric, computed by trusted code, bound to the artifact and the split it came from.
///
/// # There is no public constructor, and no API here accepts a metric value
///
/// Every field is private and the only production path to a value of this type is
/// [`evaluate_validation`]. That is the whole point: a `ValidationEvaluation` is EVIDENCE that
/// a computation happened, so a door that let a caller supply the number would make the type a
/// container for a claim rather than a record of a measurement.
///
/// # Serialize, but deliberately not Deserialize
///
/// The evaluation travels into the selection lock's canonical bytes, so it must serialize. It
/// must NOT deserialize into a usable value: a lock read back from bytes would otherwise yield
/// evaluations that could be fed straight back into minting, which is the same caller-asserted
/// number wearing a file's clothes. The private [`ValidationEvaluationWire`] is where a
/// persisted form is parsed, exactly as plan 03-03 does for the configuration, and there is no
/// conversion from the wire type back into this one.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(into = "ValidationEvaluationWire")]
pub struct ValidationEvaluation {
    metric_kind: ValidationMetricKind,
    value: f64,
    artifact_hash: String,
    validation_split_fingerprint: String,
    dataset_fingerprint: String,
    n_rows: usize,
}

impl ValidationEvaluation {
    /// Which quantity was measured.
    #[must_use]
    pub const fn metric_kind(&self) -> ValidationMetricKind {
        self.metric_kind
    }

    /// The measured value.
    #[must_use]
    pub const fn value(&self) -> f64 {
        self.value
    }

    /// The measured value's IEEE-754 bit pattern.
    ///
    /// Exposed because the lock hashes the bits rather than a rendering of them, and because a
    /// determinism assertion written against the decimal form would accept two values that
    /// differ in the last place.
    #[must_use]
    pub fn value_bits(&self) -> u64 {
        self.value.to_bits()
    }

    /// The artifact hash READ OFF the verified run this metric was computed with.
    #[must_use]
    pub fn artifact_hash(&self) -> &str {
        &self.artifact_hash
    }

    /// The fingerprint of the validation split alone.
    #[must_use]
    pub fn validation_split_fingerprint(&self) -> &str {
        &self.validation_split_fingerprint
    }

    /// The fingerprint of the whole dataset the split belongs to.
    #[must_use]
    pub fn dataset_fingerprint(&self) -> &str {
        &self.dataset_fingerprint
    }

    /// How many validation rows the metric was computed over.
    #[must_use]
    pub const fn n_rows(&self) -> usize {
        self.n_rows
    }
}

/// The canonical wire form. A struct, not a map, so field order is fixed by the type.
///
/// `pub(super)` so the selection lock can embed it in ITS canonical bytes without either module
/// growing a second serialization of the same facts.
///
/// # The value travels as BITS
///
/// `value_bits` rather than `value` is not an optimization. A JSON float is a RENDERING, so
/// hashing one binds the digest to a formatting library's shortest-representation choice; and
/// `serde_json` writes every non-finite f64 as `null`, which would make a NaN and an infinity
/// indistinguishable inside a hash whose whole job is to distinguish things. The bits are the
/// number.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ValidationEvaluationWire {
    pub(super) schema_version: u32,
    pub(super) metric_kind: ValidationMetricKind,
    pub(super) value_bits: u64,
    pub(super) n_rows: u64,
    pub(super) artifact_hash: String,
    pub(super) validation_split_fingerprint: String,
    pub(super) dataset_fingerprint: String,
}

impl From<ValidationEvaluation> for ValidationEvaluationWire {
    fn from(value: ValidationEvaluation) -> Self {
        Self {
            schema_version: EVALUATION_SCHEMA_VERSION,
            metric_kind: value.metric_kind,
            value_bits: value.value.to_bits(),
            n_rows: value.n_rows as u64,
            artifact_hash: value.artifact_hash,
            validation_split_fingerprint: value.validation_split_fingerprint,
            dataset_fingerprint: value.dataset_fingerprint,
        }
    }
}

impl ValidationEvaluation {
    /// Rebuild an evaluation from the canonical wire form a lock committed.
    ///
    /// # This is not a second construction path for a caller's number
    ///
    /// The module doc says this type must not `Deserialize`, and it still does not: there is no
    /// `impl Deserialize`, and the only way to reach this function is to hold a
    /// [`ValidationEvaluationWire`], which is `pub(super)` and therefore unnameable outside this
    /// module tree. What it exists for is the ONE place a persisted evaluation legitimately
    /// comes back: `SelectionLock::from_canonical_bytes`, reading a lock a PRIOR process wrote.
    /// A lock file whose evaluations could not be reconstructed is a file nothing can use, which
    /// is what review finding B4 was about.
    ///
    /// # It takes BITS, not a float
    ///
    /// `evaluate_source_exposes_no_public_api_taking_a_float_parameter` requires that
    /// `evaluation_for_tests` is the only float-taking door in this module, and this one takes
    /// the wire struct — so the guard is unaffected rather than excused. The bits form is also
    /// what makes the reconstruction EXACT: a decimal round trip would bind the recovered value
    /// to a formatting library's shortest-representation choice, and the lock hashes the bits.
    ///
    /// The wire's `schema_version` is deliberately not re-checked here. The caller checks the
    /// LOCK's version, and a drifted evaluation version changes the bytes the reconstructed lock
    /// re-serializes to — which its canonical-form check refuses. Checking it twice, in two
    /// places, is how the two answers eventually disagree.
    pub(super) fn from_wire(wire: ValidationEvaluationWire) -> Self {
        Self {
            metric_kind: wire.metric_kind,
            value: f64::from_bits(wire.value_bits),
            artifact_hash: wire.artifact_hash,
            validation_split_fingerprint: wire.validation_split_fingerprint,
            dataset_fingerprint: wire.dataset_fingerprint,
            n_rows: wire.n_rows as usize,
        }
    }
}

/// Build an evaluation directly, from parts — TEST ONLY.
///
/// `#[cfg(test)]` and `pub(super)`, and nothing weaker. The selection lock's rule,
/// candidate-consistency and hash-binding tests need evaluations at CONTROLLED values,
/// fingerprints and metric kinds, and producing those from real runs is impossible by
/// construction — the evaluator computes the value, which is the entire point of this module.
/// A shipped constructor of this shape would be the caller-asserted number the plan removed, so
/// it is compiled only for test targets, exactly as `test_fixtures` is.
///
/// `evaluate_source_exposes_no_public_api_taking_a_float_parameter` asserts this door exists,
/// is the only float-taking one, and is gated — so the guard NAMES the exception rather than
/// silently failing to match it.
#[cfg(test)]
pub(super) fn evaluation_for_tests(
    metric_kind: ValidationMetricKind,
    value: f64,
    artifact_hash: &str,
    validation_split_fingerprint: &str,
    dataset_fingerprint: &str,
    n_rows: usize,
) -> ValidationEvaluation {
    ValidationEvaluation {
        metric_kind,
        value,
        artifact_hash: artifact_hash.to_string(),
        validation_split_fingerprint: validation_split_fingerprint.to_string(),
        dataset_fingerprint: dataset_fingerprint.to_string(),
        n_rows,
    }
}

/// Compute `metric` on the canonical validation split, with the VERIFIED model.
///
/// # The two arguments are both load-bearing
///
/// The run carries its own dataset, so `dataset` is not how the evaluator finds the rows — it is
/// the caller's DECLARATION of which canonical dataset this evaluation is about, and the first
/// thing the evaluator does is refuse a declaration that disagrees with the run. An evaluation
/// computed against a different dataset is not evidence about this run, and silently preferring
/// one of the two would hide exactly that.
///
/// # Errors
///
/// [`SetFitTrainError::ValidationDatasetMismatch`] when the supplied dataset is not the one the
/// run was prepared from, naming both fingerprint pairs;
/// [`SetFitTrainError::ValidationSplitEmpty`] for a split with no rows;
/// [`SetFitTrainError::SelectionLabelOutOfRange`] for a row or prediction outside the declared
/// label map; plus anything the shared encode-once path or the head reports.
pub fn evaluate_validation(
    run: &SetFitRun<ArtifactReloadedAndVerified>,
    dataset: &PreparedDataset<Canonical>,
    metric: ValidationMetricKind,
) -> Result<ValidationEvaluation, SetFitTrainError> {
    // (1) The supplied dataset must BE the run's dataset. Checked on both fingerprints before
    //     a single row is encoded, so a mismatch costs a comparison rather than a model pass.
    let expected = run.dataset().validation_witness();
    let observed = dataset.validation_witness();
    let validation_split_fingerprint = observed.fingerprint_hex();
    let dataset_fingerprint = observed.dataset_fingerprint_hex();
    // Each `*_hex()` allocates, so bind both sides once and compare the bindings — mirroring
    // what is already done for `observed`.
    let expected_validation_split_fingerprint = expected.fingerprint_hex();
    let expected_dataset_fingerprint = expected.dataset_fingerprint_hex();
    if expected_validation_split_fingerprint != validation_split_fingerprint
        || expected_dataset_fingerprint != dataset_fingerprint
    {
        return Err(SetFitTrainError::ValidationDatasetMismatch {
            expected_validation_split_fingerprint,
            observed_validation_split_fingerprint: validation_split_fingerprint,
            expected_dataset_fingerprint,
            observed_dataset_fingerprint: dataset_fingerprint,
        });
    }

    // (2) The rows come from `validation()` — a `&Split<Validation>` by type. The compatibility
    //     profile has no such method (Ph2 D-19), so a compatibility-selected evaluation is not
    //     merely rejected here, it is non-constructible at the call site.
    let split = dataset.validation();
    let rows = split.rows();
    if rows.is_empty() {
        return Err(SetFitTrainError::ValidationSplitEmpty);
    }
    let classes = dataset.label_names().len();

    // (3) Encode with THIS model, through the ONE encode-once path, and predict with THIS head.
    let texts: Vec<(&str, &str)> =
        rows.iter().map(|row| (row.id.as_str(), row.input.as_str())).collect();
    let embeddings =
        head_input::encode_eval_rows(run.encoder(), &texts, run.config().requested().batch_size())?;
    let predicted =
        run.evidence().head().predict_indices(&embeddings).map_err(SetFitTrainError::HeadFit)?;

    // (4) Both index vectors are checked against the DECLARED label map, and the metric is
    //     computed, by the ONE shared tail below.
    let truth: Vec<usize> = rows.iter().map(|row| row.label).collect();
    evaluation_from_predictions(
        metric,
        &truth,
        &predicted,
        classes,
        // READ OFF the run. There is no parameter here a caller could have supplied.
        run.artifact_hash(),
        validation_split_fingerprint,
        dataset_fingerprint,
    )
}

/// The shared tail: bounds-check two index vectors, compute the metric, commit the facts.
///
/// # Why this is a function and not two copies of eight lines
///
/// [`evaluate_validation`] and
/// [`super::apr_evaluate::evaluate_validation_from_artifact`] differ ONLY in how they obtain
/// `predicted` — the trainer has a live run's encoder and head, a fresh process has a reloaded
/// artifact and core's one classify path. Everything downstream of the predictions is the same
/// decision, so it is the same code: the bounds check, the metric dispatch, the empty-class
/// convention that lives inside [`macro_f1`], and the construction of the evidence record.
///
/// # It takes NO float, deliberately
///
/// The module's one sentence is that an evaluation is a computation performed by trusted code,
/// never a number a caller says it obtained. A `value: f64` parameter here would be that number
/// with one more step, and it would be reachable from a sibling module rather than only from
/// `#[cfg(test)]`. The caller hands over the PREDICTIONS; the value is computed here.
/// `evaluate_source_exposes_no_public_api_taking_a_float_parameter` names this door and asserts
/// its parameter list is float-free.
///
/// # Errors
///
/// [`SetFitTrainError::SelectionLabelOutOfRange`] for a truth or predicted index outside the
/// declared label map.
pub(super) fn evaluation_from_predictions(
    metric: ValidationMetricKind,
    truth: &[usize],
    predicted: &[usize],
    classes: usize,
    artifact_hash: String,
    validation_split_fingerprint: String,
    dataset_fingerprint: String,
) -> Result<ValidationEvaluation, SetFitTrainError> {
    if truth.len() != predicted.len() {
        // Not reachable from either caller — both derive one prediction per row and check the
        // arity — and refused rather than `debug_assert`ed because a silent mis-pairing here
        // turns every number below into a confidently wrong measurement rather than an error.
        return Err(SetFitTrainError::SelectionLabelOutOfRange {
            label: predicted.len(),
            classes: truth.len(),
        });
    }
    for &label in truth.iter().chain(predicted.iter()) {
        if label >= classes {
            return Err(SetFitTrainError::SelectionLabelOutOfRange { label, classes });
        }
    }

    let value = match metric {
        ValidationMetricKind::Accuracy => accuracy(truth, predicted),
        ValidationMetricKind::MacroF1 => macro_f1(truth, predicted, classes),
    };

    Ok(ValidationEvaluation {
        metric_kind: metric,
        value,
        artifact_hash,
        validation_split_fingerprint,
        dataset_fingerprint,
        n_rows: truth.len(),
    })
}

/// Fraction of rows whose prediction equals the recorded class.
///
/// The indicator vector is averaged through [`reduce::mean_in_index_order`] rather than counted
/// in a `usize`, so the ONE reduction door D-13 names is the door this metric uses too — and the
/// empty-slice convention lives in that door alone rather than being restated here.
fn accuracy(truth: &[usize], predicted: &[usize]) -> f64 {
    debug_assert_eq!(truth.len(), predicted.len(), "one prediction per row");
    let indicators: Vec<f32> = truth
        .iter()
        .zip(predicted.iter())
        .map(|(actual, guess)| if actual == guess { 1.0 } else { 0.0 })
        .collect();
    reduce::mean_in_index_order(&indicators)
}

/// Unweighted mean of the per-class F1 over the DECLARED label map.
///
/// # The empty-class convention, and why it is 0.0
///
/// For a class with no actual and no predicted positives, precision and recall are both `0/0`.
/// The convention here is `F1 = 0.0` — the same one `sklearn`'s `zero_division` default
/// applies — and it is chosen for two reasons that the alternatives fail:
///
/// * A NaN would be a total loss of comparability. The selection rule orders candidates by
///   this value, and every comparison against a NaN is false, so one degenerate class would
///   make the ordering meaningless while every candidate still looked like it had a score.
/// * EXCLUDING the class from the mean would let a candidate raise its macro-F1 by declaring a
///   class it never predicts, which is a reward for a degenerate label map.
///
/// Changing this convention is a contract edit, not a code edit: it is stated in
/// `validation_evaluation_provenance` precisely so a later change is visible in a `pv diff`.
fn macro_f1(truth: &[usize], predicted: &[usize], classes: usize) -> f64 {
    debug_assert_eq!(truth.len(), predicted.len(), "one prediction per row");
    // One vector of one struct rather than three parallel vectors: "the three counts have the
    // same length" becomes structural instead of a maintained invariant, and the averaging pass
    // below reads each class's counts together rather than re-indexing three times.
    let mut counts = vec![ClassCounts::default(); classes];
    for (&actual, &guess) in truth.iter().zip(predicted.iter()) {
        // Bounds were checked by the caller; a `get_mut` here would need a failure arm that
        // the type of this function cannot express.
        if actual == guess {
            if let Some(slot) = counts.get_mut(actual) {
                slot.true_positive += 1;
            }
        } else {
            if let Some(slot) = counts.get_mut(guess) {
                slot.false_positive += 1;
            }
            if let Some(slot) = counts.get_mut(actual) {
                slot.false_negative += 1;
            }
        }
    }

    let per_class: Vec<f64> = counts
        .iter()
        .map(|count| {
            let denominator = 2 * count.true_positive + count.false_positive + count.false_negative;
            if denominator == 0 {
                return 0.0;
            }
            #[allow(clippy::cast_precision_loss)]
            let ratio = (2 * count.true_positive) as f64 / denominator as f64;
            ratio
        })
        .collect();
    reduce::mean_f64_in_index_order(&per_class)
}

/// The three per-class tallies macro-F1 needs, kept together so they cannot fall out of step.
#[derive(Clone, Copy, Default)]
struct ClassCounts {
    true_positive: u64,
    false_positive: u64,
    false_negative: u64,
}

#[cfg(test)]
#[path = "evaluate_tests.rs"]
mod evaluate_tests;
