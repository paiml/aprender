//! The two-stage SetFit trainer (D-05), and its lifecycle as a phantom typestate (D-06).
//!
//! Contract: `setfit-train-lifecycle-v1` (authored in plan 03-06).
//! Requirements: TRN-01 (the trainer's home and lifecycle), TRN-02 (configuration).
//!
//! # The lifecycle is a TYPE PARAMETER, never a runtime field
//!
//! `SetFitRun<Prepared>`, `<EncoderTuned>`, `<HeadFitted>` and
//! `<ArtifactReloadedAndVerified>` are four distinct types, and each transition consumes
//! `self` and returns the next one. An illegal ordering — fitting the head before the
//! encoder is tuned, exporting before the artifact round-trips — is therefore not
//! *rejected*, it is **inexpressible**. That is the same argument Phase 2 used for
//! `Split<Train>` and `PreparedDataset<Canonical>`, and it is what makes plan 03-09's
//! trybuild non-constructibility proof possible at all: proving a runtime field is always
//! in the right state needs whole-program reasoning; proving a type does not exist is a
//! compiler error.
//!
//! # Why the state trait is SEALED
//!
//! [`LifecycleState`] has a private supertrait, so no out-of-crate type can implement it.
//! Without the seal a caller could declare its own marker, implement the trait, and mint a
//! `SetFitRun` in a state this crate never defined — which is exactly the spoofing route
//! (T-3-09) the typestate exists to close.
//!
//! # Interface-first: four markers, one impl
//!
//! All four markers are declared HERE, in the plan that opens the module, because
//! everything wave 2 builds programs against them. Only [`Prepared`] implements
//! [`LifecycleState`] in this plan: a state's `Evidence` associated type is the evidence
//! its transition produces, and those evidence types land with their transitions
//! (03-05 for `EncoderTuned`, 03-07 for `HeadFitted`, 03-08 for
//! `ArtifactReloadedAndVerified`). A marker without an impl is a declared contract; a
//! marker with an impl and a placeholder evidence type would be a lie that compiles.

/// The `setfit-apr-v1` adapter behind the sealed codec seam (04-05, APR-03).
pub mod apr_codec;
/// The fresh-process validation evaluator: a reloaded artifact -> a candidate (04-07, TRN-07).
///
/// Closes the edge 04-16 recorded open: `SelectionCandidate::from_evaluation` needs a
/// `ValidationEvaluation`, whose only producer took a train-time run, so a process holding
/// only `.apr` files could consume a lock but never write one.
pub mod apr_evaluate;
/// The fresh-process door: artifact bytes -> a sealed credential (04-16, D-11, D-16).
pub mod apr_reload;
pub mod baseline;
/// The claims gate: fail-closed verification + closed-form aggregation (05-10, EVAL-04).
///
/// Sited beside `bench_row` and NOT in the CLI adapter for the reason `bench_metrics` records:
/// "which cells make a run complete" and "which arithmetic turns rows into published numbers"
/// ARE the claim, and an adapter that decided either would be a second definition of the thing
/// the contract exists to fix. The CLI's `bench report` is a filesystem-and-rendering shim over
/// `verify_run` + `aggregate`.
pub mod bench_gate;
/// The EVAL-01 metric assembly: per-row predictions -> a row's `QualityBlock` (05-08).
///
/// Sited beside `bench_row` and NOT in the CLI adapter for the same reason: the reduction
/// from predictions to a published number is the claim, and an adapter that computed it
/// would be a second implementation of a metric set that already has one (OPS-03).
pub mod bench_metrics;
/// The benchmark row and the hashed run manifest (05-05, EVAL-03 / EVAL-04).
///
/// Sited in the library rather than in `apr-cli` because the row schema, the digest
/// discipline and the 80-cell expectation set are the claim itself; an adapter is a
/// filesystem shim over them.
pub mod bench_row;
/// The complete deterministic state of a finished run, and its canonical wire form (03-08).
pub mod bundle;
pub mod config;
/// The sealed credential the lock doors are typed against (04-17 G2, D-11).
pub mod credential;
pub mod epoch;
/// Canonical-validation evaluation: the trusted evaluator and its bound metric (03-09).
pub mod evaluate;
pub mod evidence;
/// Stage two's encode-once input (D-08).
///
/// `pub(crate)`: `HeadDataset` is an intermediate, and a public one would be a second way to
/// reach the head's fitting input — one that does not travel through the typestate.
pub(crate) mod head_input;
/// The selection lock and the canonical-test token it mints (D-14, 03-09).
pub mod lock;
pub mod reduce;
pub mod thresholds;
pub mod tune;
/// The sealed codec seam and the trusted verify-by-reload policy (03-08).
pub mod verify;

/// The deterministic, network-free, synthetic-text fixture every Phase 3 trainer test uses.
///
/// `#[cfg(test)]` and nothing weaker: 03-10's acceptance criteria reject a `#[doc(hidden)]`
/// test-support door on the shipped surface.
#[cfg(test)]
pub(crate) mod test_fixtures;

/// The in-band pair-weighted fitter TRN-05's structural claim is measured against.
///
/// `#[cfg(test)]`: a pair-weighted head fitter must not be reachable from a shipped build by
/// any door, including a `#[doc(hidden)]` one.
#[cfg(test)]
mod negative;

use core::fmt;
use core::marker::PhantomData;

use aprender::classification::{HeadFitError, HeadFitReport, MultinomialLogisticRegression};
use aprender::setfit::SetFitMiniLm;
use aprender_contrastive_data::pairs::resolve_budget;
use aprender_contrastive_data::prepared::{Canonical, PreparedDataset};
use aprender_contrastive_data::select::Selection;
use aprender_contrastive_data::ContrastiveDataError;
use sha2::{Digest, Sha256};

use crate::train::device::{Device, DeviceError};
use bundle::BundleError;
use config::{ResolvedSetFitConfig, SetFitConfigError, SetFitTrainConfig};
use verify::CodecError;

/// The seal. Private module, public-in-private trait — the standard Rust sealing idiom.
mod sealed {
    /// Implemented only by this crate's four lifecycle markers.
    pub trait Sealed {}
}

/// One state of the SetFit training lifecycle.
///
/// Sealed: the four markers below are the complete, closed set.
pub trait LifecycleState: sealed::Sealed {
    /// The evidence this state carries.
    ///
    /// An ASSOCIATED TYPE rather than a field on `SetFitRun`, so a state that has no
    /// evidence has no evidence *field* — not an `Option` that every accessor has to
    /// `expect` on. Phase 2's `DatasetProfile::Splits` established this exact shape and
    /// the reason travels with it: an absent field beats `Option` + `expect`, and it is
    /// what makes the non-constructibility proof provable.
    type Evidence: fmt::Debug;

    /// The state's name, for error messages and provenance records.
    const STATE: &'static str;
}

/// Inputs validated, device probed, nothing trained yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Prepared;

/// The contrastive stage has run AND its SetFit-identity evidence passed (D-11).
///
/// Declared here, implemented in plan 03-05 together with the evidence type it carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncoderTuned;

/// The multiclass head has been fitted on each unique selected row exactly once (D-08).
///
/// Declared here, implemented in plan 03-07.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeadFitted;

/// The artifact has been closed, reloaded from bytes and re-verified (D-07).
///
/// Declared here, implemented in plan 03-08.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArtifactReloadedAndVerified;

impl sealed::Sealed for Prepared {}
impl sealed::Sealed for EncoderTuned {}
impl sealed::Sealed for HeadFitted {}
impl sealed::Sealed for ArtifactReloadedAndVerified {}

/// The evidence a state that has produced none carries.
///
/// A NAMED unit rather than a bare `()`. It reads better at the use site, and it is what
/// makes B-3's guard non-vacuous: with `type Evidence = ();` written out, a scan for an
/// anonymous tuple evidence type matches the deliberate empty case and can therefore never
/// distinguish it from an accidental `(A, B)`. Naming the empty case leaves the scan free to
/// mean exactly one thing.
pub type NoEvidence = ();

impl LifecycleState for Prepared {
    type Evidence = NoEvidence;
    const STATE: &'static str = "prepared";
}

/// Everything stage two produced, and the complete chain it stands on.
///
/// # A NAMED struct, not a `(PassedEvidence, HeadFitReport)` tuple (B-3)
///
/// `SetFitRun`'s field list is fixed, so `Evidence` is the ONLY home stage-two state has. A
/// two-tuple has no slot for the fitted head, the ordered labels, the effective lambda, the
/// encode ledger or the encode-call count — every one of which this plan and 03-08 require an
/// accessor for. Anything dropped here becomes something a downstream plan RECOMPUTES, which
/// is exactly the false green the in-band evidence discipline exists to remove.
///
/// Every field is private and every accessor hands out a shared borrow. There is no `Option`
/// anywhere: the absent-field pattern says a state that has not produced a thing has no field
/// for it, rather than a `None` every reader has to `expect` on.
#[derive(Debug)]
pub struct HeadFittedEvidence {
    passed: tune::PassedEvidence,
    head: MultinomialLogisticRegression,
    report: HeadFitReport,
    effective_lambda: f64,
    ordered_labels: Vec<String>,
    encode_ledger: Vec<String>,
    encode_call_count: usize,
}

impl HeadFittedEvidence {
    /// The complete stage-one chain this head was fitted on top of.
    #[must_use]
    pub fn passed(&self) -> &tune::PassedEvidence {
        &self.passed
    }

    /// The fitted head. A SHARED borrow: predict through it, never re-fit it.
    #[must_use]
    pub fn head(&self) -> &MultinomialLogisticRegression {
        &self.head
    }

    /// The optimizer's deterministic record of the fit.
    #[must_use]
    pub fn report(&self) -> &HeadFitReport {
        &self.report
    }

    /// The L2 coefficient the fit actually minimized under.
    ///
    /// Recorded rather than left to be recomputed. Without it nothing downstream can state
    /// which objective produced these weights, and 03-08's bundle would have to re-derive it
    /// — a second derivation of the one number TRN-05 is about.
    #[must_use]
    pub fn effective_lambda(&self) -> f64 {
        self.effective_lambda
    }

    /// The declared label map the head's weight rows are indexed by.
    #[must_use]
    pub fn ordered_labels(&self) -> &[String] {
        &self.ordered_labels
    }

    /// The ordered identifiers actually handed to the encoder (D-08's exactly-once proof).
    #[must_use]
    pub fn encode_ledger(&self) -> &[String] {
        &self.encode_ledger
    }

    /// How many times the encoder was invoked while building the head's input.
    #[must_use]
    pub fn encode_call_count(&self) -> usize {
        self.encode_call_count
    }

    /// Split the evidence into the RECORDED facts and the LIVE head.
    ///
    /// `pub(crate)`, and the reason it exists is the verify transition: that
    /// transition must drop the live head before it reloads anything, while still
    /// moving every recorded measurement into the final state. Keeping the struct
    /// whole would have kept the head alive across the persistence boundary, which
    /// is precisely the thing the boundary is supposed to have.
    ///
    /// The distinction is not cosmetic. The head is MODEL STATE — the object whose
    /// replacement by a reloaded one is the whole claim. Everything in
    /// [`HeadFittedParts`] is a MEASUREMENT of a run that already happened, and a
    /// measurement is not made truer or falser by being carried forward.
    pub(crate) fn into_parts(self) -> (HeadFittedParts, MultinomialLogisticRegression) {
        (
            HeadFittedParts {
                passed: self.passed,
                report: self.report,
                effective_lambda: self.effective_lambda,
                ordered_labels: self.ordered_labels,
                encode_ledger: self.encode_ledger,
                encode_call_count: self.encode_call_count,
            },
            self.head,
        )
    }
}

/// Everything [`HeadFittedEvidence`] recorded, minus the live head.
pub(crate) struct HeadFittedParts {
    pub(crate) passed: tune::PassedEvidence,
    pub(crate) report: HeadFitReport,
    pub(crate) effective_lambda: f64,
    pub(crate) ordered_labels: Vec<String>,
    pub(crate) encode_ledger: Vec<String>,
    pub(crate) encode_call_count: usize,
}

// ===========================================================================================
// The final state's evidence (plan 03-08)
// ===========================================================================================

/// What one model answered on the probe rows.
///
/// Held for BOTH sides of the verification and compared element by element. The
/// ids come from the encode ledger the encode-once path wrote as it went, so a
/// windowing defect that reordered rows shows up here rather than being papered
/// over by a re-derived id list.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifyProbe {
    pub(crate) ids: Vec<String>,
    pub(crate) embeddings: Vec<Vec<f32>>,
    pub(crate) probabilities: Vec<Vec<f64>>,
    pub(crate) labels: Vec<String>,
}

impl VerifyProbe {
    /// The probed row identifiers, in encode order.
    #[must_use]
    pub fn ids(&self) -> &[String] {
        &self.ids
    }

    /// One unit-norm embedding row per probed row.
    #[must_use]
    pub fn embeddings(&self) -> &[Vec<f32>] {
        &self.embeddings
    }

    /// One probability vector per probed row, in declared-label order.
    #[must_use]
    pub fn probabilities(&self) -> &[Vec<f64>] {
        &self.probabilities
    }

    /// One predicted label per probed row.
    #[must_use]
    pub fn labels(&self) -> &[String] {
        &self.labels
    }
}

/// The measured outcome of a verify-by-reload.
///
/// The measured maxima are recorded even on success. A verification that reports
/// only "passed" cannot tell a run that matched exactly from one that matched
/// within a tolerance it happened to be given.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifyReport {
    pub(crate) artifact_bytes: usize,
    pub(crate) probe_rows: usize,
    pub(crate) embedding_dim: usize,
    pub(crate) class_count: usize,
    pub(crate) max_embedding_abs_diff: f32,
    pub(crate) max_probability_abs_diff: f64,
    pub(crate) tolerance_embedding_abs: f32,
    pub(crate) tolerance_probability_abs: f64,
    pub(crate) round_trip_closed: bool,
}

impl VerifyReport {
    /// Length of the serialized artifact.
    #[must_use]
    pub fn artifact_bytes(&self) -> usize {
        self.artifact_bytes
    }

    /// Rows the verification compared.
    #[must_use]
    pub fn probe_rows(&self) -> usize {
        self.probe_rows
    }

    /// Width of an embedding row.
    #[must_use]
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Declared classes.
    #[must_use]
    pub fn class_count(&self) -> usize {
        self.class_count
    }

    /// Largest absolute embedding difference observed.
    #[must_use]
    pub fn max_embedding_abs_diff(&self) -> f32 {
        self.max_embedding_abs_diff
    }

    /// Largest absolute probability difference observed.
    #[must_use]
    pub fn max_probability_abs_diff(&self) -> f64 {
        self.max_probability_abs_diff
    }

    /// The embedding tolerance the comparison ran at.
    #[must_use]
    pub fn tolerance_embedding_abs(&self) -> f32 {
        self.tolerance_embedding_abs
    }

    /// The probability tolerance the comparison ran at.
    #[must_use]
    pub fn tolerance_probability_abs(&self) -> f64 {
        self.tolerance_probability_abs
    }

    /// Whether re-serializing the reloaded bundle reproduced the hashed bytes.
    #[must_use]
    pub fn round_trip_closed(&self) -> bool {
        self.round_trip_closed
    }
}

/// The verified artifact's bytes, with a `Debug` that prints their LENGTH and never them.
///
/// # Why the bytes are not a bare `Vec<u8>` field
///
/// [`ArtifactVerifiedEvidence`] is `Debug`, and it must be — `LifecycleState::Evidence`
/// requires it, so `{:?}` on any `SetFitRun` reaches this value. A bare `Vec<u8>` under
/// `#[derive(Debug)]` renders every byte: on the calibrated fixture that is 1,824,298
/// bytes rendered as `[123, 34, 115, ...]` — several megabytes of text — in whatever log
/// line formatted the run, and on a full pin it is two orders of magnitude worse than
/// that. Retaining the bytes (04-17 G1) must not turn a debug
/// print into a denial of service, so the newtype makes the short rendering the only one
/// available rather than a discipline every caller has to remember.
///
/// It carries no accessor. The bytes leave through exactly one door —
/// [`SetFitRun::<ArtifactReloadedAndVerified>::into_artifact_bytes`] — which consumes the
/// run.
pub(crate) struct RetainedArtifactBytes(pub(crate) Vec<u8>);

impl fmt::Debug for RetainedArtifactBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RetainedArtifactBytes").field("len", &self.0.len()).finish()
    }
}

/// The final state's evidence: the whole recorded chain, plus the artifact's identity.
///
/// The `head` here is the REBUILT one. The head that was fitted is dropped before
/// the reload, so a value of this type cannot hand out the pre-close model.
#[derive(Debug)]
pub struct ArtifactVerifiedEvidence {
    pub(crate) passed: tune::PassedEvidence,
    pub(crate) head: MultinomialLogisticRegression,
    pub(crate) report: HeadFitReport,
    pub(crate) effective_lambda: f64,
    pub(crate) ordered_labels: Vec<String>,
    pub(crate) encode_ledger: Vec<String>,
    pub(crate) encode_call_count: usize,
    pub(crate) artifact_hash: [u8; 32],
    pub(crate) format_id: String,
    pub(crate) verify: VerifyReport,
    pub(crate) probe: VerifyProbe,
    /// The bytes `artifact_hash` was taken over. No accessor; see
    /// [`SetFitRun::<ArtifactReloadedAndVerified>::into_artifact_bytes`].
    pub(crate) artifact_bytes: RetainedArtifactBytes,
}

impl ArtifactVerifiedEvidence {
    /// The complete stage-one chain.
    #[must_use]
    pub fn passed(&self) -> &tune::PassedEvidence {
        &self.passed
    }

    /// The head REBUILT from the artifact's bytes. A shared borrow.
    #[must_use]
    pub fn head(&self) -> &MultinomialLogisticRegression {
        &self.head
    }

    /// The optimizer's deterministic record of the original fit.
    #[must_use]
    pub fn report(&self) -> &HeadFitReport {
        &self.report
    }

    /// The L2 coefficient the original fit minimized under.
    #[must_use]
    pub fn effective_lambda(&self) -> f64 {
        self.effective_lambda
    }

    /// The declared label map the head's weight rows are indexed by.
    #[must_use]
    pub fn ordered_labels(&self) -> &[String] {
        &self.ordered_labels
    }

    /// The ordered identifiers handed to the encoder while the head's input was built.
    #[must_use]
    pub fn encode_ledger(&self) -> &[String] {
        &self.encode_ledger
    }

    /// How many times the encoder was invoked while building the head's input.
    #[must_use]
    pub fn encode_call_count(&self) -> usize {
        self.encode_call_count
    }

    /// The measured outcome of the verification.
    #[must_use]
    pub fn verify_report(&self) -> &VerifyReport {
        &self.verify
    }
}

impl LifecycleState for ArtifactReloadedAndVerified {
    /// PROOF that a real close-reload-rebuild-verify round trip happened.
    ///
    /// The absent-field pattern again: the artifact hash, the format id and the
    /// verify report exist only in this state, so "verified implies an artifact
    /// hash exists" is a fact about the type rather than about a caller's habits.
    type Evidence = ArtifactVerifiedEvidence;
    const STATE: &'static str = "artifact_reloaded_and_verified";
}

impl LifecycleState for EncoderTuned {
    /// PROOF that the gate passed, not a report about it.
    ///
    /// `PassedEvidence` is constructible only by `tune::validate_evidence`, so a value of
    /// this type cannot exist unless every gated parameter cleared its contracted epsilon.
    /// The evidence FIELD exists only in this state — there is no `Option` for a caller to
    /// `expect` on, which is what makes "EncoderTuned implies passed evidence" provable
    /// rather than conventional.
    type Evidence = tune::PassedEvidence;
    const STATE: &'static str = "encoder_tuned";
}

impl LifecycleState for HeadFitted {
    type Evidence = HeadFittedEvidence;
    const STATE: &'static str = "head_fitted";
}

/// A SetFit training run in lifecycle state `S`.
///
/// # Every field is private and there is no public constructor
///
/// The only door is [`SetFitRun::<Prepared>::prepare`]. Later plans add transitions, each
/// consuming `self`.
///
/// # The run holds its own DATASET, not only the selection
///
/// `SelectedExample` carries `{id, label, exact_hash, normalized_hash}` and **no text**
/// (contrastive-data `select.rs`); the text lives in `Split<Train>::rows()`. A run holding
/// only the `Selection` therefore could not obtain a single string to encode — neither the
/// tuning loop (03-05) nor the encode-once head input (03-07) would have an input. The
/// `dataset` field is what makes those plans writable, and `prepare` proves the two agree
/// before the run exists.
#[derive(Debug)]
pub struct SetFitRun<S: LifecycleState> {
    encoder: SetFitMiniLm,
    dataset: PreparedDataset<Canonical>,
    selection: Selection,
    config: ResolvedSetFitConfig,
    evidence: S::Evidence,
    _state: PhantomData<S>,
}

impl<S: LifecycleState> SetFitRun<S> {
    /// The resolved configuration.
    #[must_use]
    pub fn config(&self) -> &ResolvedSetFitConfig {
        &self.config
    }

    /// The typed selection this run trains on.
    #[must_use]
    pub fn selection(&self) -> &Selection {
        &self.selection
    }

    /// The prepared dataset — the run's only source of text.
    #[must_use]
    pub fn dataset(&self) -> &PreparedDataset<Canonical> {
        &self.dataset
    }

    /// The encoder, read-only.
    #[must_use]
    pub fn encoder(&self) -> &SetFitMiniLm {
        &self.encoder
    }

    /// This state's evidence.
    #[must_use]
    pub fn evidence(&self) -> &S::Evidence {
        &self.evidence
    }

    /// The lifecycle state's name.
    #[must_use]
    pub fn state_name(&self) -> &'static str {
        S::STATE
    }
}

impl SetFitRun<Prepared> {
    /// The ONLY door into the lifecycle.
    ///
    /// Three checks, in the order that names the thing the reader has to change:
    ///
    /// 1. **Device probe.** The requested spec is resolved; an explicit CUDA request on a
    ///    host without CUDA fails closed in `resolve_device` rather than falling back
    ///    silently. Phase 3 is CPU-only by decision, so a resolved non-CPU device is
    ///    [`SetFitTrainError::UnsupportedDeviceForPhase3`] — including `auto` on a CUDA
    ///    host, which is deliberate: a phase that only supports CPU should make the
    ///    operator say `cpu`, not quietly disagree with the machine.
    /// 2. **Pair budget against selection capacity.** Delegated wholesale to Phase 2's
    ///    `resolve_budget`, which owns the binds-not-clamps rule and every degenerate
    ///    layout. Reimplementing capacity arithmetic here would create a second answer.
    /// 3. **Every selected id resolves to a row of `dataset.train()`, with matching
    ///    content.** A selection whose ids are not in this dataset is a typed error at the
    ///    door, not a panic three plans later. The exact hash is compared as well as the
    ///    id: an id that resolves to a row whose bytes have since changed is a DIFFERENT
    ///    example wearing the same name, and it would silently invalidate every provenance
    ///    claim the run goes on to make.
    ///
    /// # Errors
    ///
    /// [`SetFitTrainError`] — see each variant.
    pub fn prepare(
        encoder: SetFitMiniLm,
        dataset: PreparedDataset<Canonical>,
        selection: Selection,
        config: SetFitTrainConfig,
    ) -> Result<Self, SetFitTrainError> {
        let config = config.resolve()?;
        if config.device() != Device::Cpu {
            return Err(SetFitTrainError::UnsupportedDeviceForPhase3 {
                resolved: config.device().tag(),
            });
        }

        let class_sizes = dense_class_sizes(&dataset, &selection)?;
        let (budget, _default_was_clamped) =
            resolve_budget(config.requested().pair_config(), &class_sizes)?;
        debug_assert!(budget > 0, "resolve_budget never returns a zero budget");

        let train = dataset.train();
        for example in selection.examples() {
            let Some(observed) = train.exact_hash_of(&example.id) else {
                return Err(SetFitTrainError::SelectionRowMissing { id: example.id.clone() });
            };
            if observed != &example.exact_hash {
                return Err(SetFitTrainError::SelectionRowContentMismatch {
                    id: example.id.clone(),
                });
            }
        }

        Ok(Self { encoder, dataset, selection, config, evidence: (), _state: PhantomData })
    }

    /// Consume the run and hand back the four inputs stage one operates on.
    ///
    /// `pub(crate)`: the lifecycle's public transitions consume `self` and return the next
    /// state, and this is how they get at the parts. It is deliberately NOT public — a public
    /// destructor would let a caller take the encoder out of a `Prepared` run, tune it by
    /// hand, and put nothing back, which is the ungated tuning path the typestate exists to
    /// forbid.
    pub(crate) fn into_parts(
        self,
    ) -> (SetFitMiniLm, PreparedDataset<Canonical>, Selection, ResolvedSetFitConfig) {
        (self.encoder, self.dataset, self.selection, self.config)
    }

    /// Run the contrastive stage and, ONLY if its evidence passes, mint `EncoderTuned`.
    ///
    /// The gate runs INSIDE the transition. There is no ordering in which a caller obtains a
    /// tuned run and then decides whether to check it, because the checked value is the only
    /// thing this function can return.
    ///
    /// # The regime is recorded by the RUN, not chosen at judgement time
    ///
    /// `calibration_regime_id` is stamped when the evidence table is built, from the encoder,
    /// selection and configuration the run actually used — its own architecture, its own seed,
    /// its own cell. A run cannot present a regime it did not execute in, and the gate refuses
    /// any (architecture, seed, cell) the thresholds were not measured at.
    ///
    /// # Errors
    ///
    /// [`SetFitTrainError::NoTrainableParameters`] for an all-frozen run,
    /// [`SetFitTrainError::NoTestifyingParameters`] when nothing trainable can testify,
    /// [`SetFitTrainError::UncalibratedRegime`] outside the calibrated set, and
    /// [`SetFitTrainError::EvidenceRejected`] — carrying the complete table — when the
    /// measured movement misses the contracted thresholds.
    pub fn tune_encoder(self) -> Result<SetFitRun<EncoderTuned>, SetFitTrainError> {
        self.tune_encoder_with_probes(tune::TuningProbes::NONE)
    }

    /// The same transition at a probe set only tests can build.
    ///
    /// Module-private, and `TuningProbes`'s non-default constructors are
    /// `#[cfg(test)]`, so a shipped build can reach this only with
    /// [`tune::TuningProbes::NONE`] and it is therefore exactly `tune_encoder`.
    /// It exists because 03-08's reproducibility accessors claim to report what a
    /// run EXECUTED, and the only honest falsification of that claim is a run
    /// whose execution differs while its configuration does not.
    fn tune_encoder_with_probes(
        self,
        probes: tune::TuningProbes,
    ) -> Result<SetFitRun<EncoderTuned>, SetFitTrainError> {
        let (encoder, dataset, selection, config) = self.into_parts();
        let regime = calibration_regime_id(&encoder, &selection, &config);
        #[cfg(test)]
        let out = tune::run_tuning_with_probes(encoder, &dataset, &selection, &config, probes)?;
        #[cfg(not(test))]
        let out = {
            debug_assert_eq!(probes, tune::TuningProbes::NONE);
            tune::run_tuning(encoder, &dataset, &selection, &config)?
        };

        let table = evidence::UpdateEvidence::from_tune_output(&out, &regime)
            .map_err(|e| SetFitTrainError::Evidence { reason: e.to_string() })?;
        let passed = tune::validate_evidence(
            &table,
            &thresholds::Thresholds::frozen(),
            out.trainable_count,
            out.frozen_count,
        )?;

        Ok(SetFitRun {
            encoder: out.encoder,
            dataset,
            selection,
            config,
            evidence: passed,
            _state: PhantomData,
        })
    }
}

impl SetFitRun<EncoderTuned> {
    /// Fit the multiclass head on each unique selected row, exactly once (D-08, TRN-05).
    ///
    /// # Pair multiplicity is INEXPRESSIBLE, not rejected
    ///
    /// This function takes NO parameters beyond `self`. Everything it fits on comes from the
    /// run it consumes: the dataset, the selection and the resolved configuration. There is
    /// no argument a caller could pass that says "weight this row twice", and no field on
    /// `SetFitRun` that could carry one — the pair stream is stage one's input and it does
    /// not survive into this state at all. A runtime check against multiplicity would need a
    /// multiplicity to check; the point is that there is nowhere for one to live.
    ///
    /// The adversarial half of that claim is `negative.rs`, which builds the pair-weighted
    /// fitter from the only surface that can still express it — raw embedding rows plus the
    /// public head — and shows it moves the coefficients at an IDENTICAL lambda.
    ///
    /// # The effective lambda tracks UNIQUE ROWS
    ///
    /// `SklearnEquivalentC { c }` resolves through `head_input::resolve_lambda` against the
    /// encode-once row count, so the reference default `C = 1.0` over a 24-row selection is
    /// `lambda = 1/48` whatever the pair budget is. The resolved value is handed to the head
    /// as `Regularization::Lambda`, so the head does not re-resolve it against its own row
    /// count and there is exactly one place the choice of `n` is made.
    ///
    /// # Errors
    ///
    /// Anything the encode-once input rejects (see `head_input::head_dataset`), plus
    /// [`SetFitTrainError::HeadFit`] carrying the head's own typed failure — a head that
    /// cannot converge is an error, never a warning and never a silently accepted fit.
    pub fn fit_head(self) -> Result<SetFitRun<HeadFitted>, SetFitTrainError> {
        self.fit_head_with_iteration_budget(head_input::HEAD_MAX_ITER)
    }

    /// The same body at a caller-chosen L-BFGS budget.
    ///
    /// Module-private, and it stays that way: the non-convergence path has to be reachable
    /// to be proven typed, and the honest way to reach it is a tiny iteration budget, but
    /// shipping that as a public knob would put a door on the surface whose only use is to
    /// make the head fail. `tests` is a descendant module, so it already sees this.
    fn fit_head_with_iteration_budget(
        self,
        max_iter: usize,
    ) -> Result<SetFitRun<HeadFitted>, SetFitTrainError> {
        let Self { mut encoder, dataset, selection, config, evidence: passed, _state } = self;
        let head_input::FittedHead { head, report, lambda, input } =
            head_input::fit_on_selection(&mut encoder, &dataset, &selection, &config, max_iter)?;
        let (ordered_labels, encode_ledger, encode_call_count) = input.into_evidence_parts();
        let evidence = HeadFittedEvidence {
            passed,
            head,
            report,
            effective_lambda: lambda,
            ordered_labels,
            encode_ledger,
            encode_call_count,
        };
        Ok(SetFitRun { encoder, dataset, selection, config, evidence, _state: PhantomData })
    }
}

impl SetFitRun<HeadFitted> {
    /// Close the artifact, reload it FROM BYTES, rebuild, re-predict and compare (D-07).
    ///
    /// # What the codec can and cannot do
    ///
    /// `codec` is a [`verify::SetFitCodec`]: a format id and a bytes <-> bundle
    /// pair. It cannot hash, cannot compare, cannot set a tolerance and cannot
    /// construct a lifecycle state. The hash, the drop, the rebuild, the
    /// comparison and the minting below are trusted crate-internal code an
    /// implementor has no way to reach.
    ///
    /// # The sequence, and why each step is where it is
    ///
    /// 1. Probe the LIVE model on the selected rows, through the same encode-once
    ///    path stage two used.
    /// 2. Assemble the bundle, serialize it, and hash exactly those bytes.
    /// 3. DROP the live encoder and head. This is structural: steps 1-3 happen
    ///    inside `verify::run_verify_policy`, which takes both by value, so after
    ///    it returns there is no binding to the pre-close model to compare
    ///    against even by accident.
    /// 4. Deserialize, then RE-SERIALIZE the reloaded bundle and require the
    ///    result to be byte-equal to what was hashed. This is what makes the
    ///    reloaded value provably a function of the bytes rather than of anything
    ///    the codec had lying around, and it costs one extra serialize on a path
    ///    that already serializes once.
    /// 5. Rebuild the model from the reloaded bundle alone, re-encode, re-predict,
    ///    and compare embeddings, probabilities and labels at the trusted
    ///    tolerance — exact for this codec.
    ///
    /// # Errors
    ///
    /// [`SetFitTrainError::Codec`] or [`SetFitTrainError::Bundle`] for a payload
    /// the codec or the bundle layer refuses — corrupt, truncated, oversized or
    /// from another schema, each typed and named;
    /// [`SetFitTrainError::ReloadNotFromBytes`] when the round trip is not closed;
    /// [`SetFitTrainError::ReloadDiverged`] naming the field, row, index, expected
    /// and observed value of the first difference.
    pub fn verify_artifact<C: verify::SetFitCodec>(
        self,
        codec: &C,
    ) -> Result<SetFitRun<ArtifactReloadedAndVerified>, SetFitTrainError> {
        let Self { encoder, dataset, selection, config, evidence, _state } = self;
        let (parts, head) = evidence.into_parts();

        let outcome = verify::run_verify_policy(
            codec,
            encoder,
            head,
            &parts.ordered_labels,
            &dataset,
            &selection,
            &config,
            parts.passed.summary(),
            verify::Tolerance::EXACT,
        )?;

        let evidence = verify::verified_evidence(
            parts,
            outcome.artifact_hash,
            codec.format_id(),
            outcome.report,
            outcome.probe,
            outcome.head,
            outcome.bytes,
        );
        Ok(SetFitRun {
            encoder: outcome.encoder,
            dataset,
            selection,
            config,
            evidence,
            _state: PhantomData,
        })
    }
}

// ===========================================================================================
// The reproducibility surface (plan 03-08, TRN-06)
// ===========================================================================================

/// SHA-256 over an ordered list of strings, LENGTH-PREFIXED so no regrouping of the
/// entries can collide with a different list.
///
/// The prefix replaces a NUL TERMINATOR, which did not give the separation it claimed:
/// a Rust `String` may contain a NUL byte, so `["a\0b"]` and `["a", "b"]` both hashed
/// the byte stream `a 00 b 00` and were indistinguishable. An eight-byte little-endian
/// length in front of each entry is unambiguous for every input, including entries that
/// contain the separator, and it costs one `update` either way.
fn digest_of_ordered(entries: &[String]) -> String {
    let mut hasher = Sha256::new();
    for entry in entries {
        hasher.update((entry.len() as u64).to_le_bytes());
        hasher.update(entry.as_bytes());
    }
    hex::encode(hasher.finalize())
}

/// The out-of-crate reproducibility surface.
///
/// # Every one of these is READ-ONLY and RECORDED
///
/// None returns `&mut`, none consumes `self` into another state, and none
/// recomputes a value from configuration. That last one is the substantive
/// property: an accessor that rebuilt a digest from the config it was given would
/// describe what the run was CONFIGURED to do, and two runs that share the same
/// wrong order would then reproduce perfectly.
impl SetFitRun<ArtifactReloadedAndVerified> {
    /// The selection's semantic hash — the ordered ids, label map and content hashes.
    #[must_use]
    pub fn selection_semantic_hash(&self) -> String {
        hex::encode(self.selection.semantic_hash())
    }

    /// The digest the tuning loop ABSORBED as it consumed pairs.
    ///
    /// Returns the digest the tuning loop absorbed at the draw; recomputing it
    /// from configuration would make two runs that share the same wrong order
    /// reproduce perfectly.
    #[must_use]
    pub fn pair_order_digest(&self) -> &str {
        &self.evidence.passed.table().consumed_pair_digest
    }

    /// The batch boundaries the loop actually opened, in the order it opened them.
    ///
    /// Returns the `(epoch, start_ordinal, len)` triples the loop recorded at
    /// batch open; recomputing them from configuration would make two runs that
    /// share the same wrong order reproduce perfectly.
    #[must_use]
    pub fn batch_boundaries(&self) -> &[(u32, u64, u32)] {
        &self.evidence.passed.table().batch_boundary_list
    }

    /// The digest the loop ABSORBED as it opened each batch.
    ///
    /// [`Self::batch_boundaries`] hands back the readable triples; this is the digest the
    /// loop accumulated at batch open, and it commits one field the list does not carry —
    /// the `global_step` in force when the batch opened. Two runs with identical boundary
    /// LISTS and a different step alignment therefore differ here and agree there.
    ///
    /// Added by plan 03-10 T2 rather than digesting the list in the caller: the
    /// cross-process gate's whole claim is that it compares RECORDED execution, and a
    /// digest computed from the accessor's output would be a digest of a description.
    #[must_use]
    pub fn batch_boundary_digest(&self) -> &str {
        &self.evidence.passed.table().batch_boundary_digest
    }

    /// Optimizer steps the tuning loop took.
    #[must_use]
    pub fn step_count(&self) -> u64 {
        self.evidence.passed.table().step_count
    }

    /// SHA-256 of the loss trace's little-endian `f32` bits, in step order.
    #[must_use]
    pub fn loss_trace_hash(&self) -> &str {
        &self.evidence.passed.table().loss_trace_hash
    }

    /// SHA-256 of the canonical evidence table, binding the summary to it.
    #[must_use]
    pub fn evidence_table_hash(&self) -> &str {
        &self.evidence.passed.summary().table_hash
    }

    /// SHA-256 of the ordered trainable parameter names.
    #[must_use]
    pub fn parameter_registry_hash(&self) -> &str {
        &self.evidence.passed.table().parameter_registry_hash
    }

    /// SHA-256 over the RECORDED encode ledger, entries NUL-terminated in order.
    ///
    /// The ledger itself is the list of ids the encode-once path wrote as each
    /// window was handed to the encoder; this hashes that recording, and does not
    /// re-derive the list from the selection.
    #[must_use]
    pub fn encode_ledger_hash(&self) -> String {
        digest_of_ordered(&self.evidence.encode_ledger)
    }

    /// SHA-256 of the artifact's canonical bytes.
    #[must_use]
    pub fn artifact_hash(&self) -> String {
        hex::encode(self.evidence.artifact_hash)
    }

    /// What the RELOADED model answered on the probe rows.
    #[must_use]
    pub fn probe_predictions(&self) -> &VerifyProbe {
        &self.evidence.probe
    }

    /// The identifier of the codec that wrote the artifact.
    #[must_use]
    pub fn artifact_format_id(&self) -> &str {
        &self.evidence.format_id
    }
}

/// The bytes door (04-17 G1, APR-04 / OPS-01 / OPS-02).
///
/// # Why this is a SEPARATE block from the accessors above
///
/// The same reason `create_selection_lock` lives in `lock.rs`: the block above is THE
/// reproducibility surface, and `verify_reproducibility_accessors_are_read_only_and_complete`
/// binds a SHARED reference and calls every member through it — which is what proves each one
/// is read-only. [`Self::into_artifact_bytes`] CONSUMES the run and therefore cannot be called
/// through `&T` at all. Putting it in that block would have forced the guard to stop making its
/// claim. It is counted by its own assertion in `verify_tests.rs` instead, so neither surface
/// can grow unobserved.
impl SetFitRun<ArtifactReloadedAndVerified> {
    /// Take the artifact's bytes — the exact ones that were hashed, reloaded and closed.
    ///
    /// # This is the ONLY public door to them, and it CONSUMES the run
    ///
    /// Before 04-17 there was no door at all. `verify::run_verify_policy` dropped the buffer
    /// and kept `bytes.len()`, and `VerifyReport::artifact_bytes()` — a `usize` — was the trap:
    /// it compiles and reads as if the payload were in hand. The measured consequence was that
    /// `apr setfit train` could report an artifact's SHA-256 and could not write the file that
    /// digest is of (04-06 THE FINDING, 04-12 OPS-01-F1, both proved with rustc).
    ///
    /// It consumes `self` by the phase's explicit decision. A borrowing accessor would let a
    /// caller hold the bytes AND go on using the run, which means the ~90 MB buffer and the
    /// whole live model stay resident together for as long as the caller likes. Consuming makes
    /// the handover a transfer: the run's encoder, dataset and evidence are dropped as this
    /// returns, so the peak is one buffer, not a buffer plus a model. A caller that needs both
    /// the file and the lock/token chain must therefore create the lock BEFORE writing the
    /// file — which is the correct order anyway, since a lock records a selection decision and
    /// the file is its subject.
    ///
    /// # What "the exact ones" means, and how it is checked
    ///
    /// The `Vec` returned here is the one [`verify::run_verify_policy`] serialized, hashed with
    /// SHA-256 and required to be reproduced by re-serializing the reloaded bundle. It is moved,
    /// never rebuilt. `verify_into_artifact_bytes_are_the_hashed_bytes` re-hashes the return
    /// value and requires the digest to equal [`Self::artifact_hash`] — a test that only checked
    /// for a non-empty `Vec` would pass for a re-serialization, which is precisely the substitute
    /// this door exists to make unnecessary.
    #[must_use]
    pub fn into_artifact_bytes(self) -> Vec<u8> {
        self.evidence.artifact_bytes.0
    }
}

/// The short source-revision tag of the pinned MiniLM slice the epsilons were measured on.
///
/// The first eight hex digits of `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, the upstream
/// all-MiniLM-L6-v2 revision the fixture slice was carved from.
///
/// # It is a REVISION TAG, not a weights hash, and it is ASSERTED rather than observed
///
/// `SetFitMiniLm` publishes its dimensions ([`SetFitMiniLm::architecture_fingerprint`]) and
/// nothing about where its weights came from, so this constant is appended by this crate
/// rather than read off the model. Two encoders with identical dimensions and different weight
/// values are therefore indistinguishable to the gate. Closing that needs the encoder to carry
/// a content hash of its weights, which is a larger change than this constant; it is recorded
/// here as a KNOWN LIMIT so a later reader does not mistake the tag for a proof of identity.
const SLICE_SOURCE_REVISION: &str = "1110a243";

/// The regime a run executes in, derived from the encoder, selection and configuration it
/// actually used.
///
/// Derived rather than supplied: a caller-provided regime string would let a run claim to be
/// something it is not, and the gate's fail-closed check would then be checking a label
/// instead of the run.
///
/// # The id states the run's OWN coordinates
///
/// Architecture from the encoder, seed from the configuration, cell from the selection AND the
/// configuration — every component measured from the thing that produced it. It is never the
/// enumerated calibrated string: that string names the seeds and cells the CALIBRATION swept,
/// and a run that stamped it would be claiming to have executed six cells it never ran. The
/// gate then decides membership component-wise ([`thresholds::Thresholds::is_calibrated`]),
/// which is what makes the recorded id and the calibrated set two halves of one check.
fn calibration_regime_id(
    encoder: &SetFitMiniLm,
    selection: &Selection,
    config: &ResolvedSetFitConfig,
) -> String {
    let architecture = format!("{}@{SLICE_SOURCE_REVISION}", encoder.architecture_fingerprint());
    thresholds::RegimeCoordinates::render_run(
        &architecture,
        config.requested().root_seed(),
        &cell_label(selection, config),
    )
}

/// The cell label `s{shots}e{epochs}b{batch}` this run executed in.
///
/// The shot count comes from the SELECTION, not from a configuration knob: the selection is
/// the object that drew the rows, and a label derived from an intent could disagree with the
/// run it labels.
fn cell_label(selection: &Selection, config: &ResolvedSetFitConfig) -> String {
    format!(
        "{}e{}b{}",
        shots_component(selection.class_sizes()),
        config.requested().epochs(),
        config.requested().batch_size(),
    )
}

/// The `s{n}` component of a cell label, from the SELECTED rows per class.
///
/// # A non-uniform selection is labelled as such, never as one of its classes
///
/// `FewShotSelector` draws the same number of rows for every class, so every calibrated cell
/// takes the uniform branch. A selection whose classes disagree is not an n-shot cell in any
/// honest sense, and naming one class's count would mint the label of a cell the run did not
/// execute — the very substitution this function exists to remove. Such a run is labelled
/// `smixed{min}-{max}`, and a selection that drew nothing is `sempty`; neither is a member of
/// any calibrated cell set, so both fail closed rather than borrowing another cell's epsilons.
///
/// Takes the sparse `(label, size)` list rather than the `Selection` so the two branches
/// `FewShotSelector` cannot currently produce are still directly testable.
fn shots_component(class_sizes: &[(usize, u64)]) -> String {
    let mut sizes = class_sizes.iter().map(|&(_, size)| size);
    let Some(first) = sizes.next() else {
        return "sempty".to_string();
    };
    let (min, max) = sizes.fold((first, first), |(lo, hi), size| (lo.min(size), hi.max(size)));
    if min == max {
        format!("s{min}")
    } else {
        format!("smixed{min}-{max}")
    }
}

/// Build the dense, label-indexed class-size vector Phase 2's capacity functions take.
///
/// `Selection::class_sizes()` is a sparse `(label, size)` list ascending by label; the
/// capacity functions index by position. The label map's length is the authority on `K`,
/// so a class that was declared but drew no rows contributes a zero rather than shifting
/// every later class's index by one.
fn dense_class_sizes(
    dataset: &PreparedDataset<Canonical>,
    selection: &Selection,
) -> Result<Vec<u64>, SetFitTrainError> {
    let classes = dataset.label_names().len();
    let mut sizes = vec![0_u64; classes];
    for &(label, size) in selection.class_sizes() {
        let slot = sizes
            .get_mut(label)
            .ok_or(SetFitTrainError::SelectionLabelOutOfRange { label, classes })?;
        *slot = size;
    }
    Ok(sizes)
}

/// Failure modes of the SetFit training lifecycle.
///
/// Later plans extend this enum with their own transitions' failures; it is
/// `#[non_exhaustive]` so doing so is not a breaking change.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum SetFitTrainError {
    /// A configuration knob was rejected.
    Config(SetFitConfigError),
    /// The device spec was rejected or the probe failed closed.
    Device(DeviceError),
    /// The device resolved to something Phase 3 does not support.
    UnsupportedDeviceForPhase3 {
        /// The resolved device's tag.
        resolved: String,
    },
    /// Phase 2's pair capacity / budget ladder rejected the request.
    Capacity(ContrastiveDataError),
    /// A selected row's label is outside the dataset's declared label map.
    SelectionLabelOutOfRange {
        /// The offending label.
        label: usize,
        /// The number of declared classes.
        classes: usize,
    },
    /// A selected id does not name any row of `dataset.train()`.
    SelectionRowMissing {
        /// The offending row identifier.
        id: String,
    },
    /// A selected id names a row whose exact content hash disagrees with the selection's.
    SelectionRowContentMismatch {
        /// The offending row identifier.
        id: String,
    },
    /// The multiclass head refused the fit, with its own typed reason.
    ///
    /// The inner [`HeadFitError`] is preserved rather than rendered to a string, because
    /// TRN-04's whole point is that a caller can distinguish "your data was bad" from "the
    /// optimizer ran out of budget" from "the arithmetic went non-finite" without matching on
    /// message text.
    HeadFit(HeadFitError),
    /// The head's encode did not run isolated from training mode and the autograd graph.
    ///
    /// Unreachable while `head_dataset` sets eval mode, wraps the encode in `no_grad` and
    /// detaches every result — which is exactly why it is a CHECK rather than a comment. The
    /// three mechanisms are observed while they run and refused if any is absent, so a future
    /// encoder change that starts recording under `no_grad` fails closed instead of silently
    /// making the head's input irreproducible (T-3-24).
    HeadEncodeNotIsolated {
        /// The encoder reported training mode inside an encode window.
        training_observed: bool,
        /// A stored embedding tensor still required gradients after `detach`.
        requires_grad_observed: bool,
        /// Operations the encode appended to the autograd tape.
        tape_growth: usize,
    },
    /// The head's encode window size was zero.
    ///
    /// `ResolvedSetFitConfig` cannot carry a zero batch size, so this is unreachable from the
    /// shipped transition. It exists because `chunks(0)` PANICS: a typed refusal is what keeps
    /// the one internal caller that could ever get it wrong from taking the process down.
    HeadEncodeBatchSizeZero,
    /// The `max_length` knob does not equal the encoder's pinned sequence length.
    ///
    /// Distinct from [`config::SetFitConfigError::MaxLengthNotSupported`], which rejects the
    /// same disagreement at CONSTRUCTION. This variant is what the tuning loop raises when it
    /// CONSUMES the knob: the knob was validated in 03-03 and never read, which is how a
    /// validated-but-ignored setting silently becomes decoration.
    MaxLengthNotConsumable {
        /// The requested length.
        requested: u32,
        /// The encoder's pinned length.
        pinned: u32,
    },
    /// The encoder rejected a tokenize, encode, freeze or forward-ordinal call.
    ///
    /// Carries a RENDERED string rather than the typed `SetFitError`, following 03-02's
    /// precedent: this enum derives `PartialEq` and the encoder's error type is free to grow
    /// float payloads, which would make that derive a liability at a distance.
    Encoder {
        /// The encoder's rendered diagnostic.
        reason: String,
    },
    /// The trainable parameter registry changed order or membership mid-run.
    ///
    /// `AdamW` indexes its first and second moment buffers POSITIONALLY, so a reordered
    /// registry pairs each moment with a different parameter — an update that is silently
    /// wrong rather than loudly broken (T-3-54).
    ParameterRegistryMoved,
    /// The evidence layer could not build or render the table.
    Evidence {
        /// The evidence layer's rendered diagnostic.
        reason: String,
    },
    /// The run recorded a calibration regime these thresholds were not measured in.
    ///
    /// FAIL CLOSED, and checked FIRST. An epsilon measured on one architecture is not
    /// evidence about another, so the gate refuses to judge rather than judging leniently.
    UncalibratedRegime {
        /// The regime the run recorded.
        observed: String,
        /// The regimes the frozen thresholds were measured in.
        calibrated: Vec<String>,
    },
    /// The trainable parameter set is empty, so no evidence of movement can exist.
    ///
    /// SAFE-03 automatic under D-09: an all-frozen run cannot pass by being un-checkable.
    NoTrainableParameters {
        /// The observed count. Zero, and reported so the message is self-contained.
        trainable_count: usize,
    },
    /// Every trainable parameter belongs to a class that cannot serve as evidence.
    ///
    /// Distinct from [`Self::NoTrainableParameters`]: the set is NOT empty, but every member
    /// is analytically gradient-free, so nothing in it can testify that tuning occurred.
    /// Without this variant, freezing everything except the attention key biases would leave
    /// the gate with nothing to check and produce a vacuous pass.
    NoTestifyingParameters {
        /// Trainable parameters after `apply_freeze`.
        trainable_count: usize,
        /// How many of them are ungated.
        ungated_count: usize,
    },
    /// The codec refused to encode or decode the artifact.
    Codec(CodecError),
    /// The fresh-process reload door refused the artifact/inputs pair.
    ///
    /// A SEPARATE variant rather than a set of arms on this enum, on the precedent
    /// [`Self::Codec`] and [`Self::Bundle`] set: the module that owns an operation owns its
    /// failure vocabulary, and the trainer error wraps it whole so a caller can still tell
    /// a provenance disagreement from a container CRC failure without matching on text.
    AprReload(apr_reload::AprReloadError),
    /// The fresh-process validation evaluator refused the artifact/dataset pair.
    ///
    /// Same reasoning as [`Self::AprReload`]: the module that owns the operation owns its
    /// failure vocabulary, so a label-map disagreement stays distinguishable from a corpus
    /// disagreement without matching on rendered text.
    AprEvaluate(apr_evaluate::AprEvaluateError),
    /// The bundle layer refused the payload — a contracted limit, a parse failure,
    /// an unknown schema version, or a shape the declared tensor does not have.
    Bundle(BundleError),
    /// Re-serializing the reloaded bundle did NOT reproduce the bytes that were hashed.
    ///
    /// This is the check that makes the reloaded value provably a FUNCTION OF THE
    /// BYTES. A codec that ignored its input and returned an object it had lying
    /// around would otherwise satisfy every comparison downstream, because the
    /// object it returned was never required to have come from anywhere.
    ///
    /// A faithful codec satisfies it by construction — it is the
    /// serialize/deserialize round-trip identity the bundle's own tests prove. An
    /// implementor that fails it is either not round-tripping or not canonical;
    /// both are contract-visible incompatibilities rather than bugs to work around.
    ReloadNotFromBytes {
        /// Length of the bytes that were hashed.
        hashed_len: usize,
        /// Length of the bytes the reloaded bundle re-serialized to.
        reserialized_len: usize,
        /// Offset of the first differing byte, or `None` when one stream is a
        /// PREFIX of the other — in which case the two lengths are the difference.
        first_diff_offset: Option<usize>,
    },
    /// The reloaded model did not reproduce the pre-close answer.
    ReloadDiverged {
        /// What diverged: `embedding`, `probability`, `label`, `probe_id`,
        /// `probe_row_count`, `embedding_width` or `class_count`.
        field: &'static str,
        /// The probe row it diverged on.
        row: usize,
        /// The element index within that row.
        index: usize,
        /// What the model answered before the artifact was closed.
        expected: String,
        /// What the model rebuilt from the artifact answered.
        observed: String,
        /// The tolerance the comparison ran at.
        tolerance: f64,
    },
    /// The dataset handed to the validation evaluator is not the one the run was prepared from.
    ///
    /// BOTH fingerprint pairs are named. Phase 2 made the validation-split fingerprint and the
    /// dataset fingerprint deliberately distinct values, and which of the two disagrees is the
    /// difference between "a different corpus" and "the same corpus whose validation rows
    /// changed" — a reader who is told only that something disagreed cannot tell those apart.
    ValidationDatasetMismatch {
        /// The run's own validation-split fingerprint.
        expected_validation_split_fingerprint: String,
        /// The supplied dataset's validation-split fingerprint.
        observed_validation_split_fingerprint: String,
        /// The run's own dataset fingerprint.
        expected_dataset_fingerprint: String,
        /// The supplied dataset's dataset fingerprint.
        observed_dataset_fingerprint: String,
    },
    /// The canonical validation split has no rows.
    ///
    /// Unreachable through the canonical ingest ladder, which refuses a split whose observed
    /// class counts do not match a declaration — and a declaration of all zeroes is refused in
    /// its own right. It is a CHECK rather than a comment because the alternative is a division
    /// by zero: an accuracy over no rows is `0/0`, and a NaN metric would propagate silently
    /// through the selection rule instead of failing here.
    ValidationSplitEmpty,
    /// A training step produced a non-finite loss.
    ///
    /// The `loss_trace_hash` equation of `contracts/setfit-train-lifecycle-v1.yaml` carries the
    /// precondition "every loss value is finite; a NaN or infinite step is a typed failure
    /// BEFORE hashing". This is that failure. It had no implementation until REVIEW CR-03:
    /// `run_batch` pushed the value unchecked, and `serde_json` renders every non-finite `f64`
    /// as `null`, so `+inf`, `-inf` and `NaN` all collapsed to the SAME canonical bytes — a
    /// digest that cannot distinguish three different divergences, over a bundle that then
    /// fails its own reload because `null` is not an `f64`.
    ///
    /// Carries `value_bits` rather than the `f64`, following [`lock::LockError::NonFiniteMetric`]:
    /// a `NaN` does not equal itself, so an error carrying one could not be compared in a test.
    NonFiniteLoss {
        /// The global step that produced it.
        step: u64,
        /// `f64::to_bits` of the offending value — distinguishes `+inf` from `-inf` from each
        /// `NaN` payload, which the decimal rendering does not.
        value_bits: u64,
    },
    /// The evidence failed the gate. Carries the COMPLETE auditable record.
    ///
    /// `Box`ed because this variant is far larger than every other, and an enum is as big as
    /// its widest arm on every success path too.
    EvidenceRejected {
        /// The bound summary.
        summary: Box<evidence::EvidenceSummary>,
        /// The complete per-parameter table — one row per trainable parameter.
        table: Box<evidence::UpdateEvidence>,
        /// The parameter furthest from passing, with its class, delta and contracted epsilon.
        worst: tune::FailedParameter,
    },
}

impl fmt::Display for SetFitTrainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Evidence { reason } => {
                write!(f, "setfit evidence could not be built: {reason}")
            }
            Self::UncalibratedRegime { observed, calibrated } => write!(
                f,
                "this run recorded calibration regime `{observed}`, which the frozen \
                 thresholds were NOT measured in (calibrated: {}). The gate fails closed \
                 rather than applying an epsilon measured on a different architecture; \
                 extending the calibrated set requires a calibration run on this encoder AND \
                 a deliberate edit to contracts/setfit-train-lifecycle-v1.yaml (D-10(c))",
                calibrated.join(", "),
            ),
            Self::NoTrainableParameters { trainable_count } => write!(
                f,
                "the trainable parameter set is empty (trainable_count {trainable_count}), so \
                 no evidence that the encoder moved can exist; a run in this state cannot \
                 identify as SetFit (contract setfit-train-lifecycle-v1, requirement SAFE-03)",
            ),
            Self::NoTestifyingParameters { trainable_count, ungated_count } => write!(
                f,
                "all {ungated_count} of the {trainable_count} trainable parameters belong to \
                 a class that cannot serve as encoder-update evidence (their gradients are \
                 analytically zero), so the gate has nothing to check and refuses to pass the \
                 run (contract setfit-train-lifecycle-v1, requirement SAFE-03)",
            ),
            Self::NonFiniteLoss { step, value_bits } => write!(
                f,
                "step {step} produced a non-finite loss ({}; bits {value_bits:#018x}). The \
                 loss_trace_hash precondition requires a typed failure BEFORE hashing: \
                 serde_json renders every non-finite f64 as `null`, so continuing would fold \
                 +inf, -inf and NaN into one indistinguishable digest and write a bundle that \
                 cannot be reloaded (contract setfit-train-lifecycle-v1, requirement TRN-06)",
                f64::from_bits(*value_bits),
            ),
            Self::EvidenceRejected { worst, summary, .. } => write!(
                f,
                "setfit evidence REJECTED: parameter `{}` (class {}) moved by relative delta \
                 {:e}, which does not exceed the contracted epsilon {:e} for that class; \
                 {} trainable parameters were measured under regime `{}` (contract \
                 setfit-train-lifecycle-v1, requirement TRN-03)",
                worst.name,
                worst.class,
                worst.relative_delta,
                worst.eps,
                summary.trainable_count,
                summary.calibration_regime_id,
            ),
            Self::Config(inner) => write!(f, "setfit configuration rejected: {inner}"),
            Self::Device(inner) => write!(f, "setfit device rejected: {inner}"),
            Self::UnsupportedDeviceForPhase3 { resolved } => write!(
                f,
                "device resolved to `{resolved}`, but the Phase 3 SetFit trainer is \
                 CPU-only; pass `--device cpu` explicitly \
                 (contract setfit-train-lifecycle-v1, requirement TRN-02)",
            ),
            Self::Capacity(inner) => write!(f, "pair budget rejected: {inner}"),
            Self::SelectionLabelOutOfRange { label, classes } => write!(
                f,
                "selection names class {label} but the dataset declares {classes} classes \
                 (contract setfit-train-lifecycle-v1, requirement TRN-01)",
            ),
            Self::SelectionRowMissing { id } => write!(
                f,
                "selected id `{id}` is not a row of this dataset's train split \
                 (contract setfit-train-lifecycle-v1, requirement TRN-01)",
            ),
            Self::SelectionRowContentMismatch { id } => write!(
                f,
                "selected id `{id}` resolves to a row whose exact content hash disagrees \
                 with the selection's — the same name, different bytes \
                 (contract setfit-train-lifecycle-v1, requirement TRN-01)",
            ),
            Self::HeadFit(inner) => write!(
                f,
                "the multiclass head refused the fit: {inner}; a head that does not converge \
                 is an error rather than a warning, so no coefficients were produced \
                 (contract setfit-train-lifecycle-v1, requirement TRN-04)",
            ),
            Self::HeadEncodeNotIsolated {
                training_observed,
                requires_grad_observed,
                tape_growth,
            } => write!(
                f,
                "the head's encode did not run isolated (training mode observed: \
                 {training_observed}; embeddings still requiring grad: \
                 {requires_grad_observed}; autograd tape grew by {tape_growth} operations), so \
                 its embeddings are not reproducible and the head fitted on them could not be \
                 replayed \
                 (contract setfit-train-lifecycle-v1, requirement TRN-05)",
            ),
            Self::HeadEncodeBatchSizeZero => write!(
                f,
                "the head's encode window size is zero, so the selection would be chunked \
                 into nothing; a validated configuration cannot produce this \
                 (contract setfit-train-lifecycle-v1, requirement TRN-05)",
            ),
            Self::MaxLengthNotConsumable { requested, pinned } => write!(
                f,
                "the tuning loop consumes max_length {pinned}, but this run requested \
                 {requested}; the encoder's sequence length is an equality constraint, not a \
                 runtime setting \
                 (contract setfit-train-lifecycle-v1, requirement TRN-02)",
            ),
            Self::Encoder { reason } => write!(
                f,
                "the encoder rejected a tuning-loop call: {reason} \
                 (contract setfit-train-lifecycle-v1, requirement TRN-03)",
            ),
            Self::ParameterRegistryMoved => write!(
                f,
                "the trainable parameter registry changed between the pre-loop snapshot and \
                 an optimizer step; AdamW's moment state is positional, so the update would \
                 have paired moments with the wrong parameters \
                 (contract setfit-train-lifecycle-v1, requirement TRN-03)",
            ),
            Self::Codec(inner) => write!(
                f,
                "the artifact codec refused the payload: {inner} \
                 (contract setfit-train-lifecycle-v1, requirement TRN-01)",
            ),
            Self::Bundle(inner) => write!(
                f,
                "the artifact bundle refused the payload: {inner} \
                 (contract setfit-train-lifecycle-v1, requirement TRN-01)",
            ),
            Self::AprReload(inner) => write!(
                f,
                "the artifact could not be reloaded against the supplied inputs: {inner}",
            ),
            Self::AprEvaluate(inner) => write!(
                f,
                "the reloaded artifact could not be evaluated on the supplied dataset: {inner}",
            ),
            Self::ReloadNotFromBytes { hashed_len, reserialized_len, first_diff_offset } => write!(
                f,
                "the reloaded bundle did not re-serialize to the bytes that were hashed \
                 ({hashed_len} bytes hashed, {reserialized_len} re-serialized, first \
                 difference at {}); the value the codec returned is therefore not a function \
                 of its input, so no persistence boundary was crossed and the verification \
                 cannot mean what it claims \
                 (contract setfit-train-lifecycle-v1, equation reload_verify_roundtrip)",
                first_diff_offset
                    .map_or_else(|| "a length difference".to_string(), |at| at.to_string()),
            ),
            Self::ValidationDatasetMismatch {
                expected_validation_split_fingerprint,
                observed_validation_split_fingerprint,
                expected_dataset_fingerprint,
                observed_dataset_fingerprint,
            } => write!(
                f,
                "the dataset handed to the validation evaluator is not the one this run was \
                 prepared from: validation-split fingerprint `{expected_validation_split_fingerprint}` \
                 was expected and `{observed_validation_split_fingerprint}` was supplied; dataset \
                 fingerprint `{expected_dataset_fingerprint}` was expected and \
                 `{observed_dataset_fingerprint}` was supplied. A metric computed against a \
                 different dataset is not evidence about this run \
                 (contract setfit-train-lifecycle-v1, equation validation_evaluation_provenance)",
            ),
            Self::ValidationSplitEmpty => write!(
                f,
                "the canonical validation split has no rows, so no metric computed on it could \
                 be evidence; a validated canonical dataset cannot produce this \
                 (contract setfit-train-lifecycle-v1, equation validation_evaluation_provenance)",
            ),
            Self::ReloadDiverged { field, row, index, expected, observed, tolerance } => write!(
                f,
                "the model rebuilt from the artifact disagreed with the model that wrote it: \
                 {field} at row {row} index {index} was `{expected}` before the close and \
                 `{observed}` after, which exceeds the tolerance {tolerance:e} \
                 (contract setfit-train-lifecycle-v1, equation reload_verify_roundtrip)",
            ),
        }
    }
}

impl std::error::Error for SetFitTrainError {}

impl From<SetFitConfigError> for SetFitTrainError {
    fn from(inner: SetFitConfigError) -> Self {
        Self::Config(inner)
    }
}

impl From<DeviceError> for SetFitTrainError {
    fn from(inner: DeviceError) -> Self {
        Self::Device(inner)
    }
}

impl From<ContrastiveDataError> for SetFitTrainError {
    fn from(inner: ContrastiveDataError) -> Self {
        Self::Capacity(inner)
    }
}

impl From<apr_reload::AprReloadError> for SetFitTrainError {
    fn from(inner: apr_reload::AprReloadError) -> Self {
        Self::AprReload(inner)
    }
}

#[cfg(test)]
mod tests {
    use super::test_fixtures as fx;
    use super::thresholds::Thresholds;
    use super::*;

    /// The regime id a fixture cell's run would record, built through the shipped derivation.
    fn regime_of(variant: fx::CalibrationVariant) -> String {
        let run = fx::prepared_run(variant, None);
        calibration_regime_id(run.encoder(), run.selection(), run.config())
    }

    /// The recorded id states the RUN's coordinates — its seed, its cell, no placeholders.
    ///
    /// The exact strings, not a `contains`: the whole defect this test exists for was an id
    /// that looked plausible (it named the right architecture) while its seed and cell
    /// components described something other than the run carrying it.
    #[test]
    fn regime_id_states_the_runs_own_seed_and_cell() {
        assert_eq!(
            regime_of(fx::calibrated_variant()),
            "minilm-slice-h64-l2-a2-i256-v97@1110a243|seeds=1|cells=s8e1b4",
        );
        assert_eq!(
            regime_of(fx::default_variant()),
            format!(
                "minilm-slice-h64-l2-a2-i256-v97@1110a243|seeds={}|cells=s8e1b4",
                fx::FIXTURE_SEED,
            ),
        );
        assert_eq!(
            regime_of(fx::uncalibrated_cell_variant()),
            "minilm-slice-h64-l2-a2-i256-v97@1110a243|seeds=1|cells=s8e1b3",
        );

        // No run may record the enumerated CALIBRATED string, and none may record a
        // placeholder. Both were observed: the first was returned for every run on the
        // fixture architecture, the second for every run off it.
        for variant in
            [fx::calibrated_variant(), fx::default_variant(), fx::uncalibrated_cell_variant()]
        {
            let id = regime_of(variant);
            assert!(!id.contains('?'), "a recorded id must never be a placeholder: `{id}`");
            assert!(
                !id.contains("seeds=1,42,7"),
                "a single run must never claim the calibration's whole seed sweep: `{id}`",
            );
        }
    }

    /// And the gate agrees with those coordinates, one cell at a time.
    #[test]
    fn regime_id_is_calibrated_only_at_a_measured_seed_and_cell() {
        let frozen = Thresholds::frozen();
        assert!(
            frozen.is_calibrated(&regime_of(fx::calibrated_variant())),
            "seed 1 in cell s8e1b4 was measured",
        );
        assert!(
            !frozen.is_calibrated(&regime_of(fx::default_variant())),
            "FIXTURE_SEED was never swept",
        );
        assert!(
            !frozen.is_calibrated(&regime_of(fx::uncalibrated_cell_variant())),
            "cell s8e1b3 was never measured",
        );
    }

    /// A compact description of a `tune_encoder` outcome, for the negatives' failure messages.
    ///
    /// `{other:?}` on the `Ok` arm renders the entire `SetFitRun` — encoder dims, every
    /// synthetic row, the whole evidence table — which is ~100 KB of scrollback in place of
    /// the one fact that matters: which regime the gate accepted.
    fn outcome(result: &Result<SetFitRun<EncoderTuned>, SetFitTrainError>) -> String {
        match result {
            Ok(run) => format!(
                "Ok — the gate ACCEPTED the run and recorded regime `{}`",
                run.evidence().summary().calibration_regime_id,
            ),
            Err(err) => format!("Err({err})"),
        }
    }

    /// CONTROL — a run at a calibrated (architecture, seed, cell) still passes the WHOLE gate.
    ///
    /// Without this the two negatives below would be satisfied by a regime check that refuses
    /// everything, which is the failure mode a fail-closed gate is one keystroke away from.
    ///
    /// It asserts the PLUMBING, not the id's spelling: that the id `tune_encoder` stamps into
    /// the evidence summary is the one the derivation produced for this run, and that the gate
    /// accepts it. The spelling is pinned by `regime_id_states_the_runs_own_seed_and_cell`
    /// above, which is where a wrong id belongs — keeping it out of here is what lets this
    /// test be green both before and after the derivation was fixed, and therefore evidence
    /// that the new check is not simply refusing everything.
    #[test]
    fn regime_gate_a_calibrated_run_passes() {
        let result = fx::prepared_run(fx::calibrated_variant(), None).tune_encoder();
        let run = result.expect("a run at a measured seed and cell must pass the regime check");
        let recorded = &run.evidence().summary().calibration_regime_id;
        assert!(
            Thresholds::frozen().is_calibrated(recorded),
            "the passing run's OWN recorded id must be a calibrated one, got `{recorded}`",
        );
        assert_eq!(
            recorded,
            &regime_of(fx::calibrated_variant()),
            "the id the run RECORDS must be the id the derivation produces for it; a divergence \
             here would mean the gate judged something other than what it stamped",
        );
    }

    /// NEGATIVE (seed) — a run at an unswept seed is refused, on a calibrated cell.
    ///
    /// The epsilons were measured over seeds {1, 7, 42}. `FIXTURE_SEED` is not one of them, so
    /// applying those numbers to this run would be applying a measurement to a run it was
    /// never taken on.
    #[test]
    fn regime_gate_an_unswept_seed_is_refused() {
        let result = fx::prepared_run(fx::default_variant(), None).tune_encoder();
        match &result {
            Err(SetFitTrainError::UncalibratedRegime { observed, calibrated }) => {
                assert_eq!(
                    observed,
                    &format!(
                        "minilm-slice-h64-l2-a2-i256-v97@1110a243|seeds={}|cells=s8e1b4",
                        fx::FIXTURE_SEED,
                    ),
                    "the refusal must name the run's REAL coordinates",
                );
                assert_eq!(
                    calibrated.len(),
                    2,
                    "two calibrated entries since plan 05-03 — the Phase 3 fixture slice and the \
                     Phase 5 production encoder. This run matches NEITHER, and the count is \
                     asserted so that a THIRD entry appearing cannot pass unnoticed.",
                );
                assert!(
                    calibrated[0].contains("seeds=1,42,7"),
                    "the diagnosis must show the seeds that WERE measured: {calibrated:?}",
                );
            }
            _ => panic!(
                "a run at seed {} — which the calibration never swept — must fail closed, got \
                 {}. If the gate accepted it, every epsilon in the contract is being applied to \
                 a run it was not measured on.",
                fx::FIXTURE_SEED,
                outcome(&result),
            ),
        }
    }

    /// NEGATIVE (cell) — a run in an unmeasured cell is refused, at a calibrated seed.
    ///
    /// Same architecture, same seed as the control, one knob apart: batch 3 instead of 4. The
    /// cell is the only thing that changed, so it is the only thing the refusal can be about.
    #[test]
    fn regime_gate_an_unmeasured_cell_is_refused() {
        let result = fx::prepared_run(fx::uncalibrated_cell_variant(), None).tune_encoder();
        match &result {
            Err(SetFitTrainError::UncalibratedRegime { observed, calibrated }) => {
                assert_eq!(
                    observed, "minilm-slice-h64-l2-a2-i256-v97@1110a243|seeds=1|cells=s8e1b3",
                    "the refusal must name the run's REAL cell",
                );
                assert!(
                    calibrated[0].contains("cells=s16e2b8,s8e1b4"),
                    "the diagnosis must show the cells that WERE measured: {calibrated:?}",
                );
            }
            _ => panic!(
                "a run in cell s8e1b3 — which the calibration never measured — must fail \
                 closed, got {}",
                outcome(&result),
            ),
        }
    }

    /// A non-uniform selection is labelled as one, not as an n-shot cell it is not.
    ///
    /// `FewShotSelector` cannot currently emit a non-uniform draw, which is exactly why the
    /// branch needs a test: an unexercised default is how a future selector would silently
    /// inherit a calibrated cell's epsilons.
    #[test]
    fn regime_shots_component_never_names_one_class_of_a_non_uniform_selection() {
        // The uniform case, cross-checked against a real selection.
        let uniform = fx::fixture_selection(fx::FIXTURE_SEED, 8);
        assert_eq!(shots_component(uniform.class_sizes()), "s8");
        assert_eq!(shots_component(&[(0, 8), (1, 8), (2, 8)]), "s8");

        // Non-uniform: the label names the SPREAD, never `s8` (the majority) or `s5`.
        assert_eq!(shots_component(&[(0, 8), (1, 5), (2, 8)]), "smixed5-8");
        assert_eq!(shots_component(&[]), "sempty");

        // And neither label is a member of any calibrated cell set.
        let frozen = Thresholds::frozen();
        for cell in ["smixed5-8e1b4", "semptye1b4"] {
            assert!(
                !frozen.is_calibrated(&thresholds::RegimeCoordinates::render_run(
                    "minilm-slice-h64-l2-a2-i256-v97@1110a243",
                    1,
                    cell,
                )),
                "`{cell}` must not borrow a measured cell's epsilons",
            );
        }
    }

    // =======================================================================================
    // Stage two — plan 03-07, TRN-05
    // =======================================================================================

    use aprender::optim::ConvergenceStatus;

    use config::HeadRegularization;

    /// The reference default the fixture configures the head with.
    const REFERENCE_C: f64 = 1.0;
    /// 3 classes x 8 shots.
    const SELECTED_ROWS: usize = 24;

    /// A complete calibrated pipeline: prepare -> tune_encoder -> fit_head.
    fn head_fitted_run(variant: fx::CalibrationVariant) -> SetFitRun<HeadFitted> {
        fx::head_fitted_run(variant)
    }

    /// Embedding rows for arbitrary probe texts, through the run's OWN encoder.
    fn probe_rows(run: &SetFitRun<HeadFitted>, texts: &[&str]) -> Vec<Vec<f32>> {
        let embedded = run.encoder().encode_texts(texts).expect("probe texts encode");
        let hidden = embedded.shape()[1];
        embedded.data().chunks(hidden).map(<[f32]>::to_vec).collect()
    }

    /// The pipeline reaches `HeadFitted`, and the head answers with a real distribution.
    #[test]
    fn fit_head_completes_the_pipeline_and_probabilities_sum_to_one() {
        let run = head_fitted_run(fx::calibrated_variant());
        assert_eq!(run.state_name(), "head_fitted");

        let texts: Vec<&str> =
            run.dataset().test().rows().iter().map(|r| r.input.as_str()).collect();
        assert!(!texts.is_empty(), "the fixture's test split must have rows to probe with");
        let rows = probe_rows(&run, &texts);
        let probs =
            run.evidence().head().predict_proba(&rows).expect("the fitted head must predict");

        assert_eq!(probs.len(), rows.len());
        for (row, p) in probs.iter().enumerate() {
            assert_eq!(p.len(), 3, "row {row}: one probability per declared class");
            assert!(p.iter().all(|v| v.is_finite()), "row {row}: {p:?} is not finite");
            let total: f64 = p.iter().sum();
            assert!((total - 1.0).abs() < 1e-6, "row {row}: probabilities sum to {total}");
        }
        assert_eq!(run.evidence().report().status, ConvergenceStatus::Converged);
    }

    /// Two identical pipelines agree BITWISE on the stored `f32` head.
    #[test]
    fn fit_head_two_identical_pipelines_agree_bitwise() {
        let first = head_fitted_run(fx::calibrated_variant());
        let second = head_fitted_run(fx::calibrated_variant());

        assert!(!first.evidence().head().weights().is_empty(), "a fitted head has weights");
        assert_eq!(
            first.evidence().head().weights(),
            second.evidence().head().weights(),
            "the whole path — pair stream, tuning, encode-once, L-BFGS from a zero start — \
             contains no randomness, so two identical runs must store identical f32 weights",
        );
        assert_eq!(first.evidence().head().intercepts(), second.evidence().head().intercepts());
        assert_eq!(first.evidence().encode_ledger(), second.evidence().encode_ledger());
        assert_eq!(first.evidence().effective_lambda(), second.evidence().effective_lambda());
    }

    /// The effective lambda resolves against UNIQUE ROWS: `C = 1` over 24 rows is exactly 1/48.
    #[test]
    fn fit_head_lambda_resolves_against_unique_rows_to_one_over_forty_eight() {
        let reg = HeadRegularization::SklearnEquivalentC { c: REFERENCE_C };
        assert_eq!(
            head_input::resolve_lambda(&reg, SELECTED_ROWS),
            1.0 / 48.0,
            "lambda = 1/(2*C*n) with n = 24 unique rows",
        );

        let run = head_fitted_run(fx::calibrated_variant());
        // `n` is PINNED to the selection's own length, not to a literal.
        assert_eq!(run.selection().len(), SELECTED_ROWS);
        assert_eq!(run.evidence().encode_ledger().len(), run.selection().len());
        assert_eq!(
            run.evidence().effective_lambda(),
            head_input::resolve_lambda(&reg, run.selection().len()),
            "the fit's lambda must be the one the unique-row count resolves to",
        );
        assert_eq!(run.evidence().effective_lambda(), 1.0 / 48.0);

        // The pair budget resolves to a DIFFERENT number, which is what makes the assertion
        // above discriminating rather than a coincidence of the fixture.
        let budget = run
            .config()
            .requested()
            .pair_config()
            .budget
            .expect("the fixture pins an explicit budget") as usize;
        assert_ne!(budget, SELECTED_ROWS);
        assert_ne!(head_input::resolve_lambda(&reg, budget), 1.0 / 48.0);
    }

    /// The pair budget does not reach the head's objective.
    ///
    /// # Why this is asserted against `fit_on_selection` and not against `fit_head`
    ///
    /// The plan asked for "varying the pair budget leaves the fitted weights bitwise
    /// unchanged" end to end. That statement is FALSE end to end, and asserting it would have
    /// been asserting something untrue: the budget is stage ONE's input, so two budgets take
    /// different numbers of optimizer steps and hand stage two two different encoders. The
    /// head's weights are then legitimately different, for a reason that has nothing to do
    /// with TRN-05.
    ///
    /// What IS true, and what TRN-05 actually claims, is that stage two never reads the
    /// budget: given the SAME encoder, two configurations differing only in pair budget
    /// produce the same lambda and the same coefficients, bitwise. That is asserted here
    /// against the very function `fit_head` runs. The end-to-end half — the recorded lambda
    /// is budget-independent even when the encoder is not — is asserted below it.
    #[test]
    fn fit_head_pair_budget_does_not_reach_the_head_objective() {
        let dataset = fx::fixture_dataset();
        let selection = fx::fixture_selection(fx::FIXTURE_SEED, 8);
        let mut encoder = fx::slice_encoder(fx::FIXTURE_SEED);

        let base = fx::calibrated_variant();
        let resolved = |budget: u64| {
            fx::config_for(fx::CalibrationVariant { budget, ..base }, None)
                .resolve()
                .expect("the fixture configuration resolves on a cpu host")
        };
        let (small, large) = (resolved(12), resolved(20));
        assert_ne!(
            small.requested().pair_config().budget,
            large.requested().pair_config().budget,
            "the two configurations must actually differ in the budget",
        );

        let fit = |config: &config::ResolvedSetFitConfig, encoder: &mut _| {
            head_input::fit_on_selection(
                encoder,
                &dataset,
                &selection,
                config,
                head_input::HEAD_MAX_ITER,
            )
            .expect("stage two must fit at either budget")
        };
        let a = fit(&small, &mut encoder);
        let b = fit(&large, &mut encoder);

        assert_eq!(a.lambda, 1.0 / 48.0);
        assert_eq!(a.lambda, b.lambda, "the budget must not move the head's L2 coefficient");
        assert_eq!(
            a.head.weights(),
            b.head.weights(),
            "the budget must not move a single coefficient of the head",
        );
        assert_eq!(a.head.intercepts(), b.head.intercepts());
        assert_eq!(a.input.encode_ledger(), b.input.encode_ledger());

        // End to end: two complete pipelines at different budgets record the SAME lambda,
        // even though their encoders — and therefore their coefficients — differ.
        let e2e = |budget: u64| {
            fx::prepared_run(fx::CalibrationVariant { budget, ..base }, None)
                .tune_encoder()
                .expect("both budgets clear the evidence gate at a calibrated seed and cell")
                .fit_head()
                .expect("both budgets fit a head")
        };
        let (run_small, run_large) = (e2e(12), e2e(20));
        assert_eq!(
            run_small.evidence().effective_lambda(),
            run_large.evidence().effective_lambda(),
            "the recorded objective must be budget-independent",
        );
        assert_eq!(run_small.evidence().effective_lambda(), 1.0 / 48.0);
        assert_eq!(
            run_small.evidence().encode_ledger(),
            run_large.evidence().encode_ledger(),
            "the head's input is the same 24 unique rows at either budget",
        );
    }

    /// A head that cannot converge is a TYPED error, never a silently accepted fit.
    #[test]
    fn fit_head_a_head_that_cannot_converge_surfaces_the_typed_error() {
        let tuned = fx::prepared_run(fx::calibrated_variant(), None)
            .tune_encoder()
            .expect("the calibrated cell passes the evidence gate");
        match tuned.fit_head_with_iteration_budget(1) {
            Err(SetFitTrainError::HeadFit(HeadFitError::NotConverged {
                iterations,
                gradient_norm,
                tol,
            })) => {
                assert_eq!(iterations, 1, "the budget was one iteration");
                assert!(
                    gradient_norm > tol,
                    "a non-convergence must report a gradient norm ({gradient_norm:e}) above \
                     the tolerance ({tol:e})",
                );
            }
            other => panic!(
                "a one-iteration budget must surface HeadFitError::NotConverged inside \
                 SetFitTrainError; a warning-and-return would hand back coefficients from a \
                 fit that never finished. Got {other:?}",
            ),
        }
    }

    /// The evidence carries the exactly-once proof and the whole stage-one chain.
    #[test]
    fn fit_head_evidence_carries_the_ledger_the_labels_and_the_passed_chain() {
        let run = head_fitted_run(fx::calibrated_variant());
        let evidence = run.evidence();

        let mut ledger: Vec<&str> = evidence.encode_ledger().iter().map(String::as_str).collect();
        assert_eq!(ledger.len(), SELECTED_ROWS);
        ledger.sort_unstable();
        let mut selected = run.selection().ordered_ids();
        selected.sort_unstable();
        assert_eq!(ledger, selected, "the ledger must survive the transition intact");

        let batch = run.config().requested().batch_size() as usize;
        assert_eq!(evidence.encode_call_count(), SELECTED_ROWS.div_ceil(batch));
        assert_eq!(evidence.ordered_labels(), run.dataset().label_names());
        assert_eq!(evidence.ordered_labels(), evidence.head().labels());
        assert!(
            Thresholds::frozen().is_calibrated(&evidence.passed().summary().calibration_regime_id),
            "the complete stage-one chain must survive into HeadFitted, not just a verdict",
        );
    }

    /// The setfit module directory.
    fn setfit_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/train/setfit")
    }

    /// This module's own source, with its test module removed.
    ///
    /// A source guard that reads the file it lives in finds every needle IN ITSELF. Scanning
    /// the whole of `mod.rs` for `pub fn fit_head(self)` would be satisfied by the assertion
    /// that spells it and would stay green if the transition were deleted; the first draft of
    /// this test did exactly that, and its `pub struct HeadFittedEvidence` count came back as
    /// 2. The cut is what turns the scan back into evidence about the shipped code.
    fn shipped_mod_source() -> String {
        let text = std::fs::read_to_string(setfit_dir().join("mod.rs")).expect("mod.rs readable");
        let header = format!("\nmod {} {{", "tests");
        let cut = text.find(&header).expect("mod.rs ends with its test module");
        text[..cut].to_string()
    }

    /// The same source with every comment removed.
    ///
    /// A structural guard has to scan the DECLARATIONS, not the prose about them: this
    /// module's doc comments legitimately quote the shapes the guard forbids, and a scan that
    /// counts them is red for writing the explanation. `negative_leaky.rs` established the
    /// same `split("//")` discipline in Phase 2.
    fn shipped_mod_code() -> String {
        shipped_mod_source()
            .lines()
            .map(|line| line.split("//").next().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// The structural claims, read off the SHIPPED source rather than trusted.
    #[test]
    fn fit_head_signature_and_evidence_shape_are_pinned() {
        let shipped = shipped_mod_code();

        assert_eq!(
            shipped.matches("pub fn fit_head(self) -> Result<SetFitRun<HeadFitted>").count(),
            1,
            "fit_head must exist and take ZERO non-self parameters; a pair-shaped argument \
             would make multiplicity expressible again",
        );
        assert_eq!(shipped.matches("pub struct HeadFittedEvidence").count(), 1);

        // B-3. The needle is the ASSOCIATED TYPE declaration, `type Evidence = (`, not the
        // looser `Evidence = (` the plan wrote: the loose form also matches the deliberate
        // `pub type NoEvidence = ();` — a substring straddling a word boundary — and so can
        // never distinguish the empty case from an accidental `(A, B)`.
        let tuple_evidence = format!("{}{}", "type Evidence = ", "(");
        assert_eq!(
            shipped.matches(&tuple_evidence).count(),
            0,
            "no lifecycle state may declare an anonymous tuple as its evidence (B-3)",
        );
        // The guard is two-sided: it must FIRE on the shape it forbids.
        assert_eq!(
            format!("    {tuple_evidence});").matches(&tuple_evidence).count(),
            1,
            "the needle must match the forbidden declaration, or the count above is 0 for \
             the wrong reason",
        );
        let optional_evidence = format!("{}{}", "Option<HeadFittedEvidence", ">");
        assert!(!shipped.contains(&optional_evidence), "the evidence is never optional");

        // Exactly one lambda resolution across the whole module directory, shared with the
        // adversary. Assembled at runtime so this file's own copy is not one of them.
        let resolver = format!("{}{}", "fn resolve_", "lambda");
        let mut sites = 0;
        for entry in std::fs::read_dir(setfit_dir()).expect("the setfit module is readable") {
            let path = entry.expect("a readable directory entry").path();
            if path.extension().is_some_and(|e| e == "rs") {
                sites += std::fs::read_to_string(&path)
                    .expect("every module file is readable")
                    .matches(&resolver)
                    .count();
            }
        }
        assert_eq!(
            sites, 1,
            "a second lambda resolution is how the adversarial control silently stops \
             isolating multiplicity",
        );

        // The six fields the plan names, plus the effective lambda 03-08's bundle needs.
        let source = shipped_mod_source();
        let declaration = source
            .split_once("pub struct HeadFittedEvidence {")
            .expect("the evidence struct is declared")
            .1
            .split_once("\n}")
            .expect("the declaration is closed")
            .0;
        for field in [
            "passed:",
            "head:",
            "report:",
            "effective_lambda:",
            "ordered_labels:",
            "encode_ledger:",
            "encode_call_count:",
        ] {
            assert!(declaration.contains(field), "HeadFittedEvidence must carry `{field}`");
        }
        assert!(!declaration.contains("Option<"), "no field of the evidence may be optional");
        assert!(!declaration.contains("pub "), "every field of the evidence is private");
    }
}
