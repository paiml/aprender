//! The fresh-process validation evaluator — the candidate a `.apr` file can be.
//!
//! Contract: `setfit-train-lifecycle-v1`, equation `validation_evaluation_provenance`.
//! Requirement: TRN-07.
//!
//! # The gap this closes, stated exactly
//!
//! 04-16 landed [`reload_verified_run_from_apr`](super::apr_reload::reload_verified_run_from_apr)
//! and recorded, in its own summary, the edge it deliberately left open:
//!
//! > A `SelectionCandidate` reads its artifact hash out of a `ValidationEvaluation`, and
//! > `evaluate_validation` takes a train-time `SetFitRun<ArtifactReloadedAndVerified>`. So a
//! > process that only has `.apr` files cannot yet build the candidate set it locks over — it
//! > can only consume a lock somebody else wrote. […] that is a second evaluation policy
//! > question and belongs to whichever plan owns `apr eval`'s candidate story.
//!
//! This module is that answer. Without it `apr eval --split validation --lock-out` cannot
//! exist at all: `ValidationEvaluation` has no public constructor by design, and its only
//! producer takes a type a fresh process cannot mint.
//!
//! # It is NOT a second evaluation policy, and here is the precise sense in which not
//!
//! An evaluation is two things: a PREDICTION over the canonical validation rows, and a
//! REDUCTION of those predictions to a number.
//!
//! * The reduction is literally the same code, and it is reached through the same door. Both
//!   evaluators hand their two index vectors to `evaluate::evaluation_from_predictions`, which
//!   owns the bounds check, the metric dispatch, macro-F1's empty-class convention, the
//!   index-order accumulation and the construction of the evidence record. That function was
//!   EXTRACTED from `evaluate_validation`'s tail by this plan rather than written beside it,
//!   so there is one implementation and both callers are it.
//!   `apr_evaluate_reduction_is_the_trainers_own` asserts this module computes no metric and
//!   constructs no `ValidationEvaluation`.
//! * The prediction goes through `VerifiedSetFitModel::classify` — `aprender-core`'s ONE
//!   classification path, the one `apr predict` and `POST /v1/classify` use. The trainer's
//!   `evaluate_validation` instead reaches `head_input::encode_eval_rows` plus the run's own
//!   head, because a train-time run HAS those objects and never serialized them.
//!
//! Choosing core's path here is not a convenience. A fresh process's whole claim is about the
//! artifact, and the artifact's conformance evidence — the six-probe replay every rung-8 load
//! performs — is evidence about core's encode path specifically. Reaching around it to a
//! second encode would measure something the artifact carries no probes for.
//!
//! # One caveat, recorded rather than hidden
//!
//! Two evaluations of the SAME model, one taken through the trainer's path and one through
//! this one, are not guaranteed bit-identical: they are two float pipelines. Nothing shipped
//! mixes them — a lock's candidate set is built by ONE caller and `apr eval` uses this door
//! for every candidate — but the possibility is real, so it is written here rather than left
//! for someone to discover. `ValidationEvaluation`'s wire form cannot record which path
//! produced it without changing the selection lock's canonical bytes, which would invalidate
//! every lock in existence; that trade was not worth making for a mixture no code performs.
//!
//! # What this module does NOT do
//!
//! It mints no lifecycle state, fabricates no evidence and constructs no `SetFitRun`. It takes
//! a credential that already passed the full ladder and the three-identifier provenance gate,
//! re-checks the two dataset fingerprints against the artifact's own record so the function is
//! total on its own arguments, and returns a measurement.

use aprender::setfit::{ClassifyRequestDocument, VerifiedSetFitModel, MAX_BATCH_TEXTS};
use aprender_contrastive_data::prepared::{Canonical, PreparedDataset};
use aprender_contrastive_data::schema::LabeledExample;

use super::apr_reload::ReloadedSetFitCredential;
use super::evaluate::{ValidationEvaluation, ValidationMetricKind};
use super::lock::CanonicalTestGrant;
use super::SetFitTrainError;

/// A fresh-process evaluation failure.
///
/// A module-owned vocabulary, on the precedent [`super::apr_reload::AprReloadError`] set: the
/// module that owns an operation owns its failure names, and the trainer error wraps it whole
/// so a caller can tell a label-map disagreement from a corpus disagreement without matching
/// on text.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum AprEvaluateError {
    /// The dataset supplied is not the corpus the artifact records.
    DatasetFingerprintMismatch {
        /// The fingerprint the artifact's provenance records.
        recorded: String,
        /// The fingerprint the supplied dataset computes.
        supplied: String,
    },
    /// The validation SPLIT differs, even though the corpus matches.
    ValidationSplitFingerprintMismatch {
        /// The fingerprint the artifact's provenance records.
        recorded: String,
        /// The fingerprint the supplied dataset's validation witness computes.
        supplied: String,
    },
    /// The artifact's provenance is not readable as the two fingerprints.
    ProvenanceUnreadable {
        /// Which field could not be read.
        field: &'static str,
    },
    /// The artifact's ordered labels are not the dataset's declared label map.
    LabelMapMismatch {
        /// The labels the artifact's head indexes by, in row order.
        artifact: Vec<String>,
        /// The labels the dataset declares, in index order.
        dataset: Vec<String>,
    },
    /// The validation split has no rows.
    ValidationSplitEmpty,
    /// The canonical TEST split has no rows.
    ///
    /// Its own variant rather than a reuse of [`Self::ValidationSplitEmpty`]: the two name
    /// different splits, and a report reading "the validation split is empty" about a test
    /// measurement sends the reader to the wrong file.
    TestSplitEmpty,
    /// The grant offered for a test-split measurement belongs to another artifact.
    ///
    /// [`CanonicalTestGrant`] checked this when it was issued. A grant is a VALUE, though: it
    /// can be moved into a struct, cloned and used against whatever credential is in scope
    /// three functions later, so the door that reads its rows re-checks rather than assuming.
    TestGrantArtifactMismatch {
        /// The artifact the grant admits.
        grant: String,
        /// The artifact the credential carries.
        credential: String,
    },
    /// Core's one classification path refused a batch.
    ClassifyFailed {
        /// The typed error's rendering.
        reason: String,
    },
    /// A predicted label is not in the artifact's own ordered set.
    ///
    /// Unreachable while `classify` reports the head's own labels, and kept because the
    /// alternative to a typed refusal is an `expect` in the one place a silent mis-mapping
    /// would turn every metric below into a confidently wrong number.
    UnknownPredictedLabel {
        /// The label the classifier returned.
        label: String,
    },
}

impl core::fmt::Display for AprEvaluateError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DatasetFingerprintMismatch { recorded, supplied } => write!(
                f,
                "this artifact was trained on the dataset fingerprinted `{recorded}`, and the \
                 dataset supplied fingerprints to `{supplied}`. A validation metric measured on \
                 a different corpus is not evidence about this artifact, and a selection lock \
                 taken over it would record a decision nobody made",
            ),
            Self::ValidationSplitFingerprintMismatch { recorded, supplied } => write!(
                f,
                "the artifact records the validation split `{recorded}` and the dataset supplied \
                 carries `{supplied}`. The corpus matches, so this is the SAME dataset whose \
                 validation rows changed — which the two fingerprints exist to distinguish",
            ),
            Self::ProvenanceUnreadable { field } => write!(
                f,
                "the artifact's provenance record does not carry a readable `{field}`, so there \
                 is nothing to check the supplied dataset against; the evaluation is refused \
                 rather than performed against an unverified corpus",
            ),
            Self::LabelMapMismatch { artifact, dataset } => write!(
                f,
                "the artifact's head indexes by {artifact:?} and the dataset declares \
                 {dataset:?}. Index i of one is not index i of the other, so every prediction \
                 would be compared against a different class's truth and the metric would be a \
                 confidently wrong number rather than an error",
            ),
            Self::ValidationSplitEmpty => {
                write!(f, "the canonical validation split has no rows to measure")
            }
            Self::TestSplitEmpty => {
                write!(f, "the canonical test split has no rows to measure")
            }
            Self::TestGrantArtifactMismatch { grant, credential } => write!(
                f,
                "the grant offered admits the artifact `{grant}` and the credential carries \
                 `{credential}`. Test rows released against the wrong artifact are the exact \
                 substitution the lock chain exists to block",
            ),
            Self::ClassifyFailed { reason } => {
                write!(f, "the verified model refused a validation batch: {reason}")
            }
            Self::UnknownPredictedLabel { label } => write!(
                f,
                "the classifier returned the label `{label}`, which is not in the artifact's own \
                 ordered label set",
            ),
        }
    }
}

impl std::error::Error for AprEvaluateError {}

impl From<AprEvaluateError> for SetFitTrainError {
    fn from(inner: AprEvaluateError) -> Self {
        Self::AprEvaluate(inner)
    }
}

/// Measure `metric` on the canonical validation split, with a RELOADED artifact.
///
/// The counterpart of [`super::evaluate::evaluate_validation`] for a process that trained
/// nothing. Same metric implementations, same committed facts, same refusal to accept a
/// caller-supplied number; a different prediction path, for the reason the module header gives.
///
/// # Errors
///
/// [`AprEvaluateError`], wrapped as [`SetFitTrainError::AprEvaluate`]: both fingerprint
/// disagreements naming the two values, an unreadable provenance record naming the field, a
/// label-map disagreement naming both maps, an empty split, and anything core's classify path
/// reports.
pub fn evaluate_validation_from_artifact(
    credential: &ReloadedSetFitCredential,
    dataset: &PreparedDataset<Canonical>,
    metric: ValidationMetricKind,
) -> Result<ValidationEvaluation, SetFitTrainError> {
    // (1) and (2) — the corpus and the label map — are `check_artifact_identity`, which is
    //     the SAME function the per-row door calls. See its header for why re-checking here
    //     is not redundancy.
    let identity = check_artifact_identity(credential, dataset)?;

    // (3) THE ROWS. `dataset.validation()` is a `&Split<Validation>` BY TYPE; the
    //     compatibility profile has no such method (Ph2 D-19), so a compatibility-selected
    //     evaluation is non-constructible at this call site rather than merely rejected.
    let split = dataset.validation();
    let rows = split.rows();
    if rows.is_empty() {
        return Err(AprEvaluateError::ValidationSplitEmpty.into());
    }

    // (4) PREDICT, through the ONE prediction loop this module owns.
    let predictions = predict_rows(credential.model(), &identity.artifact_labels, rows)?;

    // (5) THE SHARED TAIL. The bounds check, the metric dispatch and the construction of the
    //     evidence record are the trainer's own `evaluation_from_predictions` — the same
    //     function `evaluate_validation` calls. This module computes no metric and constructs
    //     no `ValidationEvaluation`; it hands over PREDICTIONS, never a number.
    let truth: Vec<usize> = rows.iter().map(|row| row.label).collect();
    super::evaluate::evaluation_from_predictions(
        metric,
        &truth,
        &predictions.predicted,
        identity.artifact_labels.len(),
        // READ OFF the credential, which read it off the loader's digest of the bytes. There
        // is no parameter here a caller could have supplied.
        credential.artifact_hash().to_string(),
        identity.validation_split_fingerprint,
        identity.dataset_fingerprint,
    )
}

// ===========================================================================================
// The per-row door (05-08, EVAL-01)
// ===========================================================================================

/// Which split a per-row measurement is taken over.
///
/// # The test arm CARRIES the grant, and that is the whole design
///
/// `Validation` is a unit variant because the canonical validation split is readable by anyone
/// holding the dataset. The test split is not: Phase 3's lock chain says test rows are released
/// only through a [`CanonicalTestGrant`], which is minted from a committed selection lock. So
/// the test arm takes a grant BY VALUE-REFERENCE rather than a `split: &str` or a
/// `include_test: bool`. A caller who has not been through the lock chain cannot NAME this
/// variant's contents, which makes the gate a compile-time fact instead of a runtime check
/// somebody can forget to write.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum EvaluatedSplit<'grant, 'data> {
    /// The canonical validation split.
    Validation,
    /// The canonical test split, released by a grant.
    Test(&'grant CanonicalTestGrant<'data>),
}

impl EvaluatedSplit<'_, '_> {
    /// The split's stable tag, as it appears in a bench row and in error messages.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Validation => "validation",
            Self::Test(_) => "test",
        }
    }
}

/// Per-row predictions over one evaluated split: the evidence a benchmark row is assembled from.
///
/// # Why this exists beside the scalar door rather than instead of it
///
/// [`ValidationEvaluation`] is ONE number, because that is all a selection lock needs to order
/// candidates. `F_avg`, MCC, a confusion matrix and the calibration diagnostics cannot be
/// recovered from a scalar: they need the per-row predicted class and, for calibration, the full
/// K-vector. Widening `ValidationEvaluation` to carry them was rejected — it travels inside the
/// selection lock's canonical bytes, so adding fields would invalidate every lock in existence
/// for the benefit of a consumer that does not take locks.
///
/// # Every field is private and there is no public constructor
///
/// The same discipline `ValidationEvaluation` is built on: a value of this type is EVIDENCE that
/// a measurement happened through the credentialed door, so a constructor taking prediction
/// vectors would make it a container for a claim.
#[derive(Debug, Clone, PartialEq)]
pub struct RowPredictions {
    predicted: Vec<usize>,
    probabilities: Vec<Vec<f64>>,
    truth: Vec<usize>,
    ordered_labels: Vec<String>,
    artifact_hash: String,
    split_tag: &'static str,
}

impl RowPredictions {
    /// The predicted class index per row, in split-row order.
    #[must_use]
    pub fn predicted(&self) -> &[usize] {
        &self.predicted
    }

    /// The full K-vector of class probabilities per row, in `ordered_labels` order.
    #[must_use]
    pub fn probabilities(&self) -> &[Vec<f64>] {
        &self.probabilities
    }

    /// The recorded class index per row, in split-row order.
    #[must_use]
    pub fn truth(&self) -> &[usize] {
        &self.truth
    }

    /// The labels the head indexes by, in row order.
    #[must_use]
    pub fn ordered_labels(&self) -> &[String] {
        &self.ordered_labels
    }

    /// How many rows were measured. Never zero: an empty split is a typed refusal.
    #[must_use]
    pub fn n_rows(&self) -> usize {
        self.truth.len()
    }

    /// The artifact these predictions were produced by, READ OFF the credential.
    ///
    /// Carried so a caller stamps a bench row without re-deriving it — and therefore without an
    /// opportunity to derive it differently.
    #[must_use]
    pub fn artifact_hash(&self) -> &str {
        &self.artifact_hash
    }

    /// Which split was measured: `"validation"` or `"test"`.
    ///
    /// Recorded rather than inferred, because D-07 makes calibration a validation-only
    /// diagnostic and the assembly must be able to see which vector it was handed.
    #[must_use]
    pub const fn split_tag(&self) -> &'static str {
        self.split_tag
    }
}

/// Build per-row predictions for the 9B LoRA BASELINE, from its reloaded pipeline's output.
///
/// # This is NOT a hole in the SetFit witness, and the difference is structural
///
/// [`RowPredictions`] has no public constructor because a SetFit value of it is EVIDENCE that
/// a measurement went through the credentialed door: [`evaluate_rows_from_artifact`] takes a
/// [`ReloadedSetFitCredential`], which only the load ladder can mint. That guarantee is about
/// `setfit-apr-v1` ARTIFACTS, and the LoRA baseline has none — its artifacts are a base
/// transformer plus an adapter, and no `setfit-apr-v1` load ladder can ever be run over them.
///
/// So the benchmark's two methods need two entrances to ONE metric assembly. The alternative
/// designs were both worse:
///
/// - A second assembly for LoRA would be a second definition of `F_avg`, MCC, the confusion
///   matrix and the calibration diagnostics — the exact drift `bench_metrics` exists to
///   prevent, and it would be invisible because both would look right.
/// - Widening `ReloadedSetFitCredential` to admit a LoRA pipeline would make the SetFit
///   witness a container for a claim, which is what its private constructor forbids.
///
/// What this door does NOT do is let a caller assert a SetFit measurement: `split_tag` is
/// checked against the two the assembly accepts, the shapes are checked against each other and
/// against `ordered_labels`, and `artifact_hash` is recorded rather than trusted. It cannot
/// produce a row that claims to have come from a verified `setfit-apr-v1`, because the
/// artifact hash it stamps is the ADAPTER's and the row that carries it declares
/// `method: lora` — which [`super::bench_row::BenchRow::from_bytes`] independently requires to
/// agree with a `MethodEvidence::Lora` block.
///
/// # Errors
///
/// [`AprEvaluateError::ValidationSplitEmpty`] or [`AprEvaluateError::TestSplitEmpty`] for an
/// empty split; [`SetFitTrainError::SelectionLabelOutOfRange`] for a truth index outside the
/// label map; [`AprEvaluateError::LabelMapMismatch`] when a probability row is not
/// `ordered_labels.len()` long or the two vectors disagree in length.
pub fn row_predictions_from_lora(
    probabilities: &[Vec<f64>],
    truth: &[usize],
    ordered_labels: &[String],
    artifact_hash: &str,
    split_tag: &'static str,
) -> Result<RowPredictions, SetFitTrainError> {
    let classes = ordered_labels.len();
    if probabilities.len() != truth.len() {
        return Err(AprEvaluateError::LabelMapMismatch {
            artifact: vec![format!("{} probability rows", probabilities.len())],
            dataset: vec![format!("{} truth rows", truth.len())],
        }
        .into());
    }
    if probabilities.is_empty() {
        return Err(match split_tag {
            "test" => AprEvaluateError::TestSplitEmpty,
            _ => AprEvaluateError::ValidationSplitEmpty,
        }
        .into());
    }
    for row in probabilities {
        if row.len() != classes {
            return Err(AprEvaluateError::LabelMapMismatch {
                artifact: vec![format!("a {}-wide probability row", row.len())],
                dataset: ordered_labels.to_vec(),
            }
            .into());
        }
    }
    // Bounds-checked HERE, for the reason `evaluate_rows_from_artifact` states: an
    // out-of-range index reaching the assembly would index a metric vector's wrong slot and
    // produce a confidently wrong number rather than an error.
    for &label in truth {
        if label >= classes {
            return Err(SetFitTrainError::SelectionLabelOutOfRange { label, classes });
        }
    }

    // The predicted class is `argmax` with the LOWEST index winning a tie — the same
    // deterministic rule the SetFit head's own reduction uses, so a tie does not resolve
    // differently depending on which method produced the vector.
    let predicted: Vec<usize> = probabilities
        .iter()
        .map(|row| {
            let mut best = 0_usize;
            for index in 1..row.len() {
                if row[index] > row[best] {
                    best = index;
                }
            }
            best
        })
        .collect();

    Ok(RowPredictions {
        predicted,
        probabilities: probabilities.to_vec(),
        truth: truth.to_vec(),
        ordered_labels: ordered_labels.to_vec(),
        artifact_hash: artifact_hash.to_string(),
        split_tag,
    })
}

/// Build per-row predictions directly, from parts — TEST ONLY.
///
/// `#[cfg(test)]` and `pub(super)`, and nothing weaker — the precedent
/// [`super::evaluate::evaluation_for_tests`] set, for the same reason. The metric assembly in
/// [`super::bench_metrics`] has to be falsified against HAND-COMPUTED cases: six rows whose
/// confusion matrix a reader can check, and four validation rows whose top confidences sit at
/// bin centres. Producing those from a real artifact is impossible by construction — the door
/// computes the predictions, which is the entire point of this module — and a shipped
/// constructor of this shape would be exactly the caller-asserted evidence `RowPredictions`
/// exists to refuse.
#[cfg(test)]
pub(super) fn row_predictions_for_tests(
    predicted: Vec<usize>,
    probabilities: Vec<Vec<f64>>,
    truth: Vec<usize>,
    ordered_labels: Vec<String>,
    artifact_hash: &str,
    split_tag: &'static str,
) -> RowPredictions {
    RowPredictions {
        predicted,
        probabilities,
        truth,
        ordered_labels,
        artifact_hash: artifact_hash.to_string(),
        split_tag,
    }
}

/// Measure per-row predictions on one canonical split, with a RELOADED artifact.
///
/// The per-row sibling of [`evaluate_validation_from_artifact`]. Same credential type, so it is
/// unreachable without the reload door; same artifact-vs-dataset identity re-check, returning
/// the same typed errors; the same single prediction loop. It differs only in RETURN SHAPE — it
/// hands back the two index vectors and the probability matrix instead of reducing them.
///
/// This is the ONE evaluation door a benchmark cell calls (OPS-03). Classifying in the adapter
/// would be a second prediction path over an artifact whose conformance evidence is about this
/// one.
///
/// # Errors
///
/// Everything [`evaluate_validation_from_artifact`] returns, plus
/// [`AprEvaluateError::TestSplitEmpty`] and [`AprEvaluateError::TestGrantArtifactMismatch`] on
/// the test arm.
pub fn evaluate_rows_from_artifact(
    credential: &ReloadedSetFitCredential,
    dataset: &PreparedDataset<Canonical>,
    split: EvaluatedSplit<'_, '_>,
) -> Result<RowPredictions, SetFitTrainError> {
    let identity = check_artifact_identity(credential, dataset)?;

    // THE ROWS. The validation arm reads them off the dataset by type; the test arm reads them
    // off the GRANT, which is the only thing entitled to hand them out.
    let rows: &[LabeledExample] = match split {
        EvaluatedSplit::Validation => {
            let rows = dataset.validation().rows();
            if rows.is_empty() {
                return Err(AprEvaluateError::ValidationSplitEmpty.into());
            }
            rows
        }
        EvaluatedSplit::Test(grant) => {
            if grant.artifact_hash() != credential.artifact_hash() {
                return Err(AprEvaluateError::TestGrantArtifactMismatch {
                    grant: grant.artifact_hash().to_string(),
                    credential: credential.artifact_hash().to_string(),
                }
                .into());
            }
            let rows = grant.test().rows();
            if rows.is_empty() {
                return Err(AprEvaluateError::TestSplitEmpty.into());
            }
            rows
        }
    };

    let predictions = predict_rows(credential.model(), &identity.artifact_labels, rows)?;

    // The truth vector is bounds-checked HERE rather than downstream. The scalar door gets that
    // check for free from `evaluation_from_predictions`; this door returns before reaching it,
    // and an out-of-range index reaching the assembly would index a metric vector's wrong slot
    // and produce a confidently wrong number rather than an error.
    let classes = identity.artifact_labels.len();
    let truth: Vec<usize> = rows.iter().map(|row| row.label).collect();
    for &label in &truth {
        if label >= classes {
            return Err(SetFitTrainError::SelectionLabelOutOfRange { label, classes });
        }
    }

    Ok(RowPredictions {
        predicted: predictions.predicted,
        probabilities: predictions.probabilities,
        truth,
        ordered_labels: identity.artifact_labels,
        artifact_hash: credential.artifact_hash().to_string(),
        split_tag: split.tag(),
    })
}

// ===========================================================================================
// The two shared halves — ONE implementation, two return shapes
// ===========================================================================================

/// What the identity re-check establishes, so neither door recomputes it.
struct ArtifactIdentity {
    artifact_labels: Vec<String>,
    dataset_fingerprint: String,
    validation_split_fingerprint: String,
}

/// THE CORPUS AND THE LABEL MAP, from the ARTIFACT'S OWN RECORD.
///
/// The reload door checked the corpus against the dataset it was handed; checking it again here
/// is not redundancy — both evaluation doors take their own `dataset` argument, and a version
/// that trusted the caller to pass the same one would be a door whose guarantee depends on a
/// convention.
///
/// The label map is read off the REBUILT HEAD, which is the list a classification will actually
/// index into — not the document's copy of it, which could have drifted.
///
/// # Errors
///
/// [`AprEvaluateError::ProvenanceUnreadable`], [`AprEvaluateError::DatasetFingerprintMismatch`],
/// [`AprEvaluateError::ValidationSplitFingerprintMismatch`] or
/// [`AprEvaluateError::LabelMapMismatch`].
fn check_artifact_identity(
    credential: &ReloadedSetFitCredential,
    dataset: &PreparedDataset<Canonical>,
) -> Result<ArtifactIdentity, SetFitTrainError> {
    let model = credential.model();
    let provenance = model.doc_view().provenance.clone();
    let recorded_dataset = read_provenance_hex(&provenance, "dataset_fingerprint")?;
    let recorded_validation = read_provenance_hex(&provenance, "validation_split_fingerprint")?;

    let witness = dataset.validation_witness();
    let dataset_fingerprint = witness.dataset_fingerprint_hex();
    let validation_split_fingerprint = witness.fingerprint_hex();

    if recorded_dataset != dataset_fingerprint {
        return Err(AprEvaluateError::DatasetFingerprintMismatch {
            recorded: recorded_dataset,
            supplied: dataset_fingerprint,
        }
        .into());
    }
    if recorded_validation != validation_split_fingerprint {
        return Err(AprEvaluateError::ValidationSplitFingerprintMismatch {
            recorded: recorded_validation,
            supplied: validation_split_fingerprint,
        }
        .into());
    }

    let artifact_labels: Vec<String> = model.ordered_labels().to_vec();
    let dataset_labels: Vec<String> = dataset.label_names().to_vec();
    if artifact_labels != dataset_labels {
        return Err(AprEvaluateError::LabelMapMismatch {
            artifact: artifact_labels,
            dataset: dataset_labels,
        }
        .into());
    }

    Ok(ArtifactIdentity { artifact_labels, dataset_fingerprint, validation_split_fingerprint })
}

/// One prediction loop's output: the argmax index and the full K-vector, per row.
struct RowClassifications {
    predicted: Vec<usize>,
    probabilities: Vec<Vec<f64>>,
}

/// PREDICT, through core's ONE classify path, in batches core's own bound allows.
///
/// Chunked rather than one call: `MAX_BATCH_TEXTS` is enforced INSIDE classify, so a split
/// larger than it would be refused rather than measured, and silently dropping the tail would be
/// worse than either.
///
/// # This is the module's ONLY classify site
///
/// Both doors reach it. `apr_evaluate_rows_shares_one_classify_loop_with_the_scalar_door`
/// asserts the count is exactly one, because two prediction loops over one artifact are two
/// float pipelines that must agree and eventually will not.
///
/// # Errors
///
/// [`AprEvaluateError::ClassifyFailed`] for anything core reports, for a response arity that
/// does not match the request, and for a probability vector whose length is not the label map's;
/// [`AprEvaluateError::UnknownPredictedLabel`] for a label outside the artifact's ordered set.
fn predict_rows(
    model: &VerifiedSetFitModel,
    artifact_labels: &[String],
    rows: &[LabeledExample],
) -> Result<RowClassifications, SetFitTrainError> {
    let classes = artifact_labels.len();
    let mut predicted: Vec<usize> = Vec::with_capacity(rows.len());
    let mut probabilities: Vec<Vec<f64>> = Vec::with_capacity(rows.len());

    for chunk in rows.chunks(MAX_BATCH_TEXTS) {
        let request = ClassifyRequestDocument::new(chunk.iter().map(|row| row.input.clone()));
        let response = model.classify(&request).map_err(|error| {
            SetFitTrainError::from(AprEvaluateError::ClassifyFailed { reason: error.to_string() })
        })?;
        for result in response.results() {
            let index = artifact_labels
                .iter()
                .position(|label| label == result.label())
                .ok_or_else(|| {
                    SetFitTrainError::from(AprEvaluateError::UnknownPredictedLabel {
                        label: result.label().to_string(),
                    })
                })?;
            // A K-vector of the wrong arity would make the calibration functions read a row of
            // one length as a row of another, silently mixing classes across row boundaries.
            // Core's constructor already ties logits to probabilities; nothing ties either to
            // the HEAD's label count, so that is checked here.
            let row = result.probabilities();
            if row.len() != classes {
                return Err(AprEvaluateError::ClassifyFailed {
                    reason: format!(
                        "the classifier returned a {}-wide probability vector for a {classes}-label \
                         head",
                        row.len(),
                    ),
                }
                .into());
            }
            predicted.push(index);
            probabilities.push(row.to_vec());
        }
    }

    // The response is in request order and one row per text (core's envelope constructor
    // enforces a uniform arity), but a length disagreement here would silently mis-pair every
    // prediction with a truth, so it is checked rather than assumed.
    if predicted.len() != rows.len() {
        return Err(AprEvaluateError::ClassifyFailed {
            reason: format!(
                "the classifier returned {} results for {} rows",
                predicted.len(),
                rows.len()
            ),
        }
        .into());
    }

    Ok(RowClassifications { predicted, probabilities })
}

/// Read one lowercase-hex provenance field, or refuse by name.
fn read_provenance_hex(
    provenance: &serde_json::Value,
    field: &'static str,
) -> Result<String, SetFitTrainError> {
    provenance
        .get(field)
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| AprEvaluateError::ProvenanceUnreadable { field }.into())
}

#[cfg(test)]
#[path = "apr_evaluate_tests.rs"]
mod apr_evaluate_tests;

// 05-08's tests live in their OWN sibling rather than in `apr_evaluate_tests.rs`, whose
// assertions are 04-07's independent evidence that the scalar door still behaves as it did.
#[cfg(test)]
#[path = "apr_evaluate_row_tests.rs"]
mod apr_evaluate_row_tests;
