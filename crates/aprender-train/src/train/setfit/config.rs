//! The SetFit trainer's twelve configuration knobs, in two types.
//!
//! Contract: `setfit-train-lifecycle-v1` (authored in plan 03-06). Requirement: TRN-02.
//!
//! # Two types, because "what was asked for" and "what was resolved" are different facts
//!
//! [`SetFitTrainConfig`] is the REQUESTED form. It is serializable, it holds the device as
//! a string-shaped [`DeviceRequest`] rather than a probed `Device`, and it is what plan
//! 03-08's artifact bundle embeds and hashes. [`ResolvedSetFitConfig`] adds the fact only
//! the host can supply — the probed `Device` — and is produced by
//! `SetFitRun::<Prepared>::prepare`. It is `Serialize` and deliberately NOT `Deserialize`:
//! a resolved runtime device arriving from a file would be an assertion about a machine
//! nobody checked (T-3-50).
//!
//! # There is exactly ONE validation implementation, and serde cannot route around it
//!
//! Deriving `Deserialize` field-by-field on [`SetFitTrainConfig`] would create a SECOND
//! construction path. `deny_unknown_fields` does not help: it rejects unknown KEYS, not
//! invalid VALUES, so a payload carrying `encoder_lr: -1.0` would deserialize cleanly past
//! the whole knob table (T-3-36). Instead a private wire struct derives `Deserialize`, and
//! [`SetFitTrainConfig`] carries `#[serde(try_from = "SetFitTrainConfigWire")]` whose
//! `TryFrom` impl calls the same public [`SetFitTrainConfig::new`] every other caller uses.
//!
//! Note that `#[derive(Deserialize)]` is still literally present on the type: serde cannot
//! generate a `try_from` impl without it. What matters is that the derive is REDIRECTED —
//! there is no field-by-field path — and that is what the four invalid-payload tests below
//! actually falsify.
//!
//! # The parser / probe split
//!
//! Copied in shape from `train/device.rs`: every knob has ONE pure validator that touches
//! no environment. The device knob is the exception that proves it — `resolve_device` does
//! grammar AND probe in one call, so construction runs it and accepts BOTH `Ok` and
//! `CudaNotAvailable` (grammar was fine; availability is a `prepare()`-time fact), while
//! rejecting `InvalidSpec`. The outcome is therefore host-independent even though the call
//! is not, and no device grammar is re-implemented here.

use core::fmt;

use serde::{Deserialize, Serialize};

use crate::train::device::{resolve_device, Device, DeviceError};
use aprender::setfit::{FreezeGroup, MAX_SEQUENCE_LENGTH};
use aprender_contrastive_data::pairs::{PairConfig, PairStrategy, SingletonPolicy};

// ===========================================================================================
// Reference recipe constants
// ===========================================================================================

/// Reference encoder learning rate (SetFit recipe).
pub const REFERENCE_ENCODER_LR: f64 = 2e-5;
/// Reference epoch count.
pub const REFERENCE_EPOCHS: u32 = 1;
/// Reference pair batch size.
pub const REFERENCE_BATCH_SIZE: u32 = 16;
/// Reference warmup fraction.
pub const REFERENCE_WARMUP_RATIO: f64 = 0.1;
/// Reference gradient-clipping max norm.
pub const REFERENCE_GRAD_CLIP_MAX_NORM: f32 = 1.0;
/// Reference head regularization: sklearn `LogisticRegression`'s default `C`.
pub const REFERENCE_SKLEARN_C: f64 = 1.0;

/// AdamW's beta1. FIXED, not a knob — a reference-recipe value.
pub const ADAMW_BETA1: f32 = 0.9;
/// AdamW's beta2. FIXED, not a knob — a reference-recipe value.
pub const ADAMW_BETA2: f32 = 0.999;
/// AdamW's epsilon. FIXED, not a knob — a reference-recipe value.
pub const ADAMW_EPSILON: f32 = 1e-8;
/// AdamW's weight decay. FIXED, not a knob — a reference-recipe value.
///
/// SetFit's reference training uses `sentence-transformers`' default AdamW configuration,
/// whose weight decay is 0.0. Exposing it as a knob would invite a silent deviation from
/// the recipe that no gate in this milestone would catch.
pub const ADAMW_WEIGHT_DECAY: f32 = 0.0;

// ===========================================================================================
// Knob value types
// ===========================================================================================

/// The learning-rate schedule. v1 ships exactly one, and the enum exists so a future
/// second schedule is a versioned change rather than a silent one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum LrSchedule {
    /// Linear warmup to `encoder_lr`, then linear decay to zero — the reference recipe.
    #[default]
    WarmupLinearDecay,
}

/// How the multiclass head's L2 penalty is expressed.
///
/// The native form is `lambda`; `SklearnEquivalentC` is the reference-comparison form and
/// resolves at fit time through the contracted relation `lambda = 1 / (2 * C * n)` with
/// `n` the number of UNIQUE SELECTED ROWS. The relation itself — its half-constant, its
/// sum-vs-mean convention and its intercept exclusion — is the plan 03-04 / 03-06 contract
/// equation. This type carries the REQUEST; it never performs the arithmetic, so there is
/// no second place for the factor of two to go wrong.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum HeadRegularization {
    /// Native: the penalty coefficient on `||W||^2`, intercept unpenalized. Zero means no
    /// penalty, which is legal and is why the bound is non-negative rather than positive.
    Lambda(f64),
    /// sklearn's inverse regularization strength.
    SklearnEquivalentC {
        /// sklearn's `C`. Strictly positive and finite.
        c: f64,
    },
}

/// A device string as REQUESTED, before availability is known.
///
/// A newtype rather than a bare `String` so a probed [`Device`] and a requested spec cannot
/// be swapped at a call site: they are different facts carrying different trust. Its field
/// is private and the only constructor validates the grammar, so every value of this type
/// has already passed `resolve_device`'s parser.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceRequest(String);

impl DeviceRequest {
    /// The requested spec, verbatim.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for DeviceRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

// ===========================================================================================
// The request
// ===========================================================================================

/// The twelve knobs as a caller supplies them, before validation.
///
/// Plain data with public fields: it is the ARGUMENT to the validating constructor, never a
/// substitute for it. Passing one of these around proves nothing; only a
/// [`SetFitTrainConfig`] does.
#[derive(Debug, Clone, PartialEq)]
pub struct SetFitTrainRequest {
    /// Knob 1 — the encoder learning rate. Finite and strictly positive.
    pub encoder_lr: f64,
    /// Knob 2 — contrastive epochs. Non-zero.
    pub epochs: u32,
    /// Knob 3 — pair batch size. Non-zero.
    pub batch_size: u32,
    /// Knob 4 — warmup fraction of total steps, in `[0.0, 1.0]`.
    pub warmup_ratio: f64,
    /// Knob 5 — gradient-clipping max norm. Finite and strictly positive.
    pub grad_clip_max_norm: f32,
    /// Knob 6 — requested max sequence length. Only [`MAX_SEQUENCE_LENGTH`].
    pub max_length: u32,
    /// Knob 7 — the pair-stream configuration, delegated wholesale to Phase 2.
    pub pair_config: PairConfig,
    /// Knob 8 — the freeze policy. Empty is D-20's all-trainable default.
    pub freeze_policy: Vec<FreezeGroup>,
    /// Knob 9 — the head's regularization request.
    pub head_regularization: HeadRegularization,
    /// Knob 10 — the root seed. No default; every `u64` is legal.
    pub root_seed: u64,
    /// Knob 11 — the requested device spec.
    pub device: String,
    /// Knob 12 — the learning-rate schedule.
    pub lr_schedule: LrSchedule,
}

/// The twelve requested knobs, validated.
///
/// Every field is private and the only door is [`Self::new`]; accessors are read-only, so a
/// value of this type is a validated value for its whole life.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(into = "SetFitTrainConfigWire", try_from = "SetFitTrainConfigWire")]
pub struct SetFitTrainConfig {
    encoder_lr: f64,
    epochs: u32,
    batch_size: u32,
    warmup_ratio: f64,
    grad_clip_max_norm: f32,
    max_length: u32,
    pair_config: PairConfig,
    freeze_policy: Vec<FreezeGroup>,
    head_regularization: HeadRegularization,
    root_seed: u64,
    device: DeviceRequest,
    lr_schedule: LrSchedule,
}

/// The requested configuration plus the fact only the host can supply.
///
/// `Serialize` so 03-08's bundle can record what was ASKED FOR and what was RESOLVED side
/// by side. NOT `Deserialize`, deliberately — see the module doc.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(into = "ResolvedSetFitConfigWire")]
pub struct ResolvedSetFitConfig {
    requested: SetFitTrainConfig,
    device: Device,
}

impl ResolvedSetFitConfig {
    /// The requested form this was resolved from.
    #[must_use]
    pub fn requested(&self) -> &SetFitTrainConfig {
        &self.requested
    }

    /// The probed device.
    #[must_use]
    pub fn device(&self) -> Device {
        self.device
    }
}

// ===========================================================================================
// Errors
// ===========================================================================================

/// Failure modes of the twelve-knob table.
///
/// Every variant naming a knob rejection carries named fields for the knob and the observed
/// value: an error that says only "invalid configuration" cannot be acted on.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum SetFitConfigError {
    /// A knob that must be finite and strictly positive was not.
    NotFinitePositive {
        /// The knob's name, exactly as spelled in the wire form.
        knob: &'static str,
        /// The value observed.
        observed: f64,
    },
    /// A knob that must be finite and non-negative was not.
    NotFiniteNonNegative {
        /// The knob's name, exactly as spelled in the wire form.
        knob: &'static str,
        /// The value observed.
        observed: f64,
    },
    /// A count knob that must be non-zero was zero.
    MustBeNonZero {
        /// The knob's name, exactly as spelled in the wire form.
        knob: &'static str,
    },
    /// A ratio knob fell outside its closed range, or was not a number.
    RatioOutOfRange {
        /// The knob's name, exactly as spelled in the wire form.
        knob: &'static str,
        /// The value observed.
        observed: f64,
        /// Inclusive lower bound.
        min: f64,
        /// Inclusive upper bound.
        max: f64,
    },
    /// `max_length` may only be the tokenizer's pinned bound.
    MaxLengthNotSupported {
        /// What the caller asked for.
        requested: u32,
        /// [`MAX_SEQUENCE_LENGTH`], the only supported value.
        pinned: u32,
    },
    /// A pair policy this protocol version does not ship.
    UnsupportedPairPolicy {
        /// The knob's name.
        knob: &'static str,
        /// The observed policy, rendered.
        observed: String,
        /// The only supported value.
        supported: &'static str,
    },
    /// The device spec did not parse.
    Device(DeviceError),
}

impl fmt::Display for SetFitConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFinitePositive { knob, observed } => write!(
                f,
                "`{knob}` must be finite and strictly positive, observed {observed} \
                 (contract setfit-train-lifecycle-v1, requirement TRN-02)",
            ),
            Self::NotFiniteNonNegative { knob, observed } => write!(
                f,
                "`{knob}` must be finite and non-negative, observed {observed} \
                 (contract setfit-train-lifecycle-v1, requirement TRN-02)",
            ),
            Self::MustBeNonZero { knob } => write!(
                f,
                "`{knob}` must be greater than zero, observed 0 \
                 (contract setfit-train-lifecycle-v1, requirement TRN-02)",
            ),
            Self::RatioOutOfRange { knob, observed, min, max } => write!(
                f,
                "`{knob}` must lie in [{min}, {max}], observed {observed} \
                 (contract setfit-train-lifecycle-v1, requirement TRN-02)",
            ),
            Self::MaxLengthNotSupported { requested, pinned } => write!(
                f,
                "`max_length` {requested} is not supported: the pinned tokenizer truncates \
                 at {pinned} and takes no max-length parameter \
                 (contract setfit-train-lifecycle-v1, requirement TRN-02)",
            ),
            Self::UnsupportedPairPolicy { knob, observed, supported } => write!(
                f,
                "`{knob}` is {observed}, but protocol v1 ships only {supported} \
                 (contract setfit-train-lifecycle-v1, requirement TRN-02)",
            ),
            Self::Device(inner) => write!(f, "`device` rejected: {inner}"),
        }
    }
}

impl std::error::Error for SetFitConfigError {}

impl From<DeviceError> for SetFitConfigError {
    fn from(inner: DeviceError) -> Self {
        Self::Device(inner)
    }
}

// ===========================================================================================
// Per-knob validators — one pure function per knob, no environment access
// ===========================================================================================

fn validate_finite_positive_f64(knob: &'static str, value: f64) -> Result<(), SetFitConfigError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(SetFitConfigError::NotFinitePositive { knob, observed: value })
    }
}

fn validate_finite_non_negative_f64(
    knob: &'static str,
    value: f64,
) -> Result<(), SetFitConfigError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(SetFitConfigError::NotFiniteNonNegative { knob, observed: value })
    }
}

fn validate_non_zero_u32(knob: &'static str, value: u32) -> Result<(), SetFitConfigError> {
    if value == 0 {
        Err(SetFitConfigError::MustBeNonZero { knob })
    } else {
        Ok(())
    }
}

fn validate_closed_ratio(knob: &'static str, value: f64) -> Result<(), SetFitConfigError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(SetFitConfigError::RatioOutOfRange { knob, observed: value, min: 0.0, max: 1.0 })
    }
}

fn validate_max_length(requested: u32) -> Result<(), SetFitConfigError> {
    let pinned = pinned_max_length();
    if requested == pinned {
        Ok(())
    } else {
        Err(SetFitConfigError::MaxLengthNotSupported { requested, pinned })
    }
}

fn validate_head_regularization(value: HeadRegularization) -> Result<(), SetFitConfigError> {
    match value {
        // Zero lambda is legal: it means "no penalty", which is a coherent request.
        HeadRegularization::Lambda(lambda) => {
            validate_finite_non_negative_f64("head_regularization.lambda", lambda)
        }
        // C is an INVERSE strength, so zero is a division by zero rather than "no penalty".
        HeadRegularization::SklearnEquivalentC { c } => {
            validate_finite_positive_f64("head_regularization.c", c)
        }
    }
}

/// The pair POLICY knobs, compared rather than matched.
///
/// `PairStrategy` and `SingletonPolicy` are `#[non_exhaustive]`, so an out-of-crate
/// exhaustive `match` is impossible — and a wildcard arm would silently accept a future
/// variant this config's wire form cannot represent, which is exactly the data loss that
/// would make a serialized provenance record wrong. Comparing against the v1 values is
/// total, needs no wildcard, and fails closed.
fn validate_pair_policies(pair_config: &PairConfig) -> Result<(), SetFitConfigError> {
    if pair_config.strategy != PairStrategy::Oversampling {
        return Err(SetFitConfigError::UnsupportedPairPolicy {
            knob: "pair_config.strategy",
            observed: format!("{:?}", pair_config.strategy),
            supported: "Oversampling",
        });
    }
    if pair_config.singleton_policy != SingletonPolicy::NegativesOnly {
        return Err(SetFitConfigError::UnsupportedPairPolicy {
            knob: "pair_config.singleton_policy",
            observed: format!("{:?}", pair_config.singleton_policy),
            supported: "NegativesOnly",
        });
    }
    Ok(())
}

/// Grammar-only device validation, with the probe deliberately discarded.
///
/// `resolve_device` is the SINGLE definition of the device grammar (contract
/// `gpu-training-backend-v1` INV-GPUTRAIN-001); re-implementing it here would create a
/// second answer that could drift. `CudaNotAvailable` means the grammar was fine and only
/// the host disagreed — a `prepare()`-time fact — so it is accepted here and re-raised
/// there. `InvalidSpec` is host-independent and is rejected now.
fn validate_device_grammar(spec: &str) -> Result<DeviceRequest, SetFitConfigError> {
    match resolve_device(spec) {
        Ok(_) | Err(DeviceError::CudaNotAvailable { .. }) => Ok(DeviceRequest(spec.to_string())),
        Err(invalid @ DeviceError::InvalidSpec(_)) => Err(SetFitConfigError::Device(invalid)),
    }
}

/// [`MAX_SEQUENCE_LENGTH`] as a `u32`.
///
/// The pinned bound is 256 and the cast is a constant fold, but it is written as a function
/// so the `usize -> u32` narrowing has exactly one site.
#[must_use]
pub fn pinned_max_length() -> u32 {
    u32::try_from(MAX_SEQUENCE_LENGTH).unwrap_or(u32::MAX)
}

// ===========================================================================================
// The single validating constructor
// ===========================================================================================

impl SetFitTrainConfig {
    /// Validate a request. THE only way to obtain a [`SetFitTrainConfig`].
    ///
    /// Every knob is checked before any training state exists, and the deserialization path
    /// routes through this same function — see the module doc.
    ///
    /// # Errors
    ///
    /// [`SetFitConfigError`], naming the knob and the observed value.
    pub fn new(request: SetFitTrainRequest) -> Result<Self, SetFitConfigError> {
        validate_finite_positive_f64("encoder_lr", request.encoder_lr)?;
        validate_non_zero_u32("epochs", request.epochs)?;
        validate_non_zero_u32("batch_size", request.batch_size)?;
        validate_closed_ratio("warmup_ratio", request.warmup_ratio)?;
        validate_finite_positive_f64("grad_clip_max_norm", f64::from(request.grad_clip_max_norm))?;
        validate_max_length(request.max_length)?;
        validate_pair_policies(&request.pair_config)?;
        validate_head_regularization(request.head_regularization)?;
        // Knob 10, root_seed: EVERY u64 is legal. The rejected form is ABSENCE, and absence
        // is unrepresentable here because the field is not an Option — the wire struct is
        // where that rejection lives, and `falsify_config_deserialize_rejects_missing_root_seed`
        // is what proves it.
        let device = validate_device_grammar(&request.device)?;
        // Knob 12, lr_schedule: the type has exactly one variant, so an unsupported schedule
        // is unrepresentable in Rust; the wire form is where a bad value can arrive, and
        // serde rejects an unknown variant there.

        // Knob 8's structural half is the wire enum's shape (serde rejects a malformed
        // group). What is done HERE is canonicalization — sort and dedup, the same
        // normalization `SetFitMiniLm::apply_freeze` performs — so two policies that mean
        // the same thing serialize to the same bytes and therefore hash the same, which
        // 03-08's bundle depends on. Zero-match enforcement stays with `apply_freeze`,
        // which is the only place that knows the encoder's parameter names; 03-05 MUST call
        // it BEFORE the initial parameter snapshot and BEFORE the pre-tuning baseline
        // encode, so a zero-match policy cannot slip past into captured evidence.
        let mut freeze_policy = request.freeze_policy;
        freeze_policy.sort_unstable();
        freeze_policy.dedup();

        // Knob 7's OTHER canonicalization, and it closes a round-trip hole rather than a
        // hashing one. `PairConfigWire` does not carry `pair_config.root_seed` — the wire
        // form has exactly one seed field, and `TryFrom` rebuilds the `PairConfig` from the
        // top-level `root_seed`. So a value constructed with a pair seed that DISAGREED with
        // `root_seed` (both fields are public on `PairConfig`) would serialize, deserialize,
        // and come back with a DIFFERENT pair-sampling stream — silently, and with the
        // provenance record still claiming to describe the original run. Normalizing here
        // makes construction agree with deserialization, so serialize -> deserialize is an
        // identity for every value of this type. Knob 10 is THE root seed; every RNG domain
        // derives from it, and there is no coherent request for a second one.
        let mut pair_config = request.pair_config;
        pair_config.root_seed = request.root_seed;

        Ok(Self {
            encoder_lr: request.encoder_lr,
            epochs: request.epochs,
            batch_size: request.batch_size,
            warmup_ratio: request.warmup_ratio,
            grad_clip_max_norm: request.grad_clip_max_norm,
            max_length: request.max_length,
            pair_config,
            freeze_policy,
            head_regularization: request.head_regularization,
            root_seed: request.root_seed,
            device,
            lr_schedule: request.lr_schedule,
        })
    }

    /// The SetFit reference recipe's defaults.
    ///
    /// `root_seed` has no default by design: a seed nobody chose is a reproducibility claim
    /// nobody made.
    ///
    /// # Panics
    ///
    /// Never in practice — the reference values are constants that satisfy the table, and a
    /// panic here would mean a constant was edited to an invalid value, which is a
    /// programming error rather than a caller error.
    #[must_use]
    pub fn reference_defaults(root_seed: u64) -> Self {
        Self::new(SetFitTrainRequest {
            encoder_lr: REFERENCE_ENCODER_LR,
            epochs: REFERENCE_EPOCHS,
            batch_size: REFERENCE_BATCH_SIZE,
            warmup_ratio: REFERENCE_WARMUP_RATIO,
            grad_clip_max_norm: REFERENCE_GRAD_CLIP_MAX_NORM,
            max_length: pinned_max_length(),
            pair_config: PairConfig::new(root_seed),
            freeze_policy: Vec::new(),
            head_regularization: HeadRegularization::SklearnEquivalentC { c: REFERENCE_SKLEARN_C },
            root_seed,
            device: "cpu".to_string(),
            lr_schedule: LrSchedule::WarmupLinearDecay,
        })
        .expect("the reference recipe constants satisfy the knob table by construction")
    }

    /// Encoder learning rate.
    #[must_use]
    pub fn encoder_lr(&self) -> f64 {
        self.encoder_lr
    }

    /// Number of contrastive epochs.
    #[must_use]
    pub fn epochs(&self) -> u32 {
        self.epochs
    }

    /// Pair batch size.
    #[must_use]
    pub fn batch_size(&self) -> u32 {
        self.batch_size
    }

    /// Warmup fraction of total steps.
    #[must_use]
    pub fn warmup_ratio(&self) -> f64 {
        self.warmup_ratio
    }

    /// Gradient-clipping max norm.
    #[must_use]
    pub fn grad_clip_max_norm(&self) -> f32 {
        self.grad_clip_max_norm
    }

    /// Requested maximum sequence length.
    #[must_use]
    pub fn max_length(&self) -> u32 {
        self.max_length
    }

    /// The pair-stream configuration, delegated wholesale to Phase 2.
    #[must_use]
    pub fn pair_config(&self) -> &PairConfig {
        &self.pair_config
    }

    /// The freeze policy, canonicalized. Empty is D-20's all-trainable default.
    #[must_use]
    pub fn freeze_policy(&self) -> &[FreezeGroup] {
        &self.freeze_policy
    }

    /// The head's regularization request.
    #[must_use]
    pub fn head_regularization(&self) -> HeadRegularization {
        self.head_regularization
    }

    /// The root seed every RNG domain derives from.
    #[must_use]
    pub fn root_seed(&self) -> u64 {
        self.root_seed
    }

    /// The REQUESTED device spec, unprobed.
    #[must_use]
    pub fn device(&self) -> &DeviceRequest {
        &self.device
    }

    /// The learning-rate schedule.
    #[must_use]
    pub fn lr_schedule(&self) -> LrSchedule {
        self.lr_schedule
    }

    /// Read the twelve knobs back out as a REQUEST, so overrides can be merged and revalidated.
    ///
    /// # What this is for
    ///
    /// D-07's `apr setfit train` is config-file-first: a file supplies the twelve knobs and
    /// `--seed` / `--device` may override two of them. The override is applied to the REQUEST
    /// this returns, and the result goes back through [`Self::new`] — so the merged
    /// configuration is validated AS A WHOLE, by the same single implementation that validated
    /// the file.
    ///
    /// ```ignore
    /// let mut request = config.to_request();
    /// request.root_seed = cli_seed;          // the override
    /// let merged = SetFitTrainConfig::new(request)?;  // revalidated, as a whole
    /// ```
    ///
    /// # The two alternatives, and why both are wrong
    ///
    /// Overriding a field on an already-validated [`SetFitTrainConfig`] would let an invalid
    /// merge through: the value would never meet the knob table, and the type's promise — that
    /// every value of it has been validated for its whole life — would be false. That is why
    /// there is no mutable-receiver method here and why the accessors are read-only.
    ///
    /// Deserializing into `SetFitTrainConfigWire` and editing THAT is not possible from another
    /// crate: the wire type is private, deliberately (see the module doc), and `apr-cli`
    /// physically cannot name it. Plan 04-06 was originally written against that private type;
    /// this door is what closes that finding without widening the wire form's visibility.
    ///
    /// # There is deliberately no `with_seed` / `with_device`
    ///
    /// Two doors to the same merge is how the two diverge. `to_request` plus the existing
    /// [`Self::new`] is the whole surface, and
    /// `falsify_config_to_request_is_the_whole_merge_surface` is what keeps it that way.
    ///
    /// # Round-trip note
    ///
    /// Feeding the result straight back to [`Self::new`] reproduces an EQUAL config: `new`'s
    /// two canonicalizations (the freeze policy is already sorted and deduped, and
    /// `pair_config.root_seed` already equals `root_seed`) are idempotent on a config that has
    /// been through them once.
    #[must_use]
    pub fn to_request(&self) -> SetFitTrainRequest {
        SetFitTrainRequest {
            encoder_lr: self.encoder_lr,
            epochs: self.epochs,
            batch_size: self.batch_size,
            warmup_ratio: self.warmup_ratio,
            grad_clip_max_norm: self.grad_clip_max_norm,
            max_length: self.max_length,
            pair_config: self.pair_config,
            freeze_policy: self.freeze_policy.clone(),
            head_regularization: self.head_regularization,
            root_seed: self.root_seed,
            device: self.device.as_str().to_string(),
            lr_schedule: self.lr_schedule,
        }
    }

    /// Probe the host and pair the requested form with the resolved device.
    ///
    /// # Errors
    ///
    /// [`SetFitConfigError::Device`] wrapping whatever `resolve_device` fails closed with —
    /// in practice `CudaNotAvailable`, since the grammar was already checked at
    /// construction.
    pub fn resolve(self) -> Result<ResolvedSetFitConfig, SetFitConfigError> {
        let device = resolve_device(self.device.as_str())?;
        Ok(ResolvedSetFitConfig { requested: self, device })
    }
}

// ===========================================================================================
// The wire forms — the ONLY place serde touches these types
// ===========================================================================================

/// The freeze policy's wire mirror.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case", tag = "group")]
enum FreezeGroupWire {
    Embeddings,
    LayerAttention { layer: usize },
    LayerFfn { layer: usize },
    LayerNorm { layer: usize },
}

impl From<FreezeGroup> for FreezeGroupWire {
    fn from(value: FreezeGroup) -> Self {
        match value {
            FreezeGroup::Embeddings => Self::Embeddings,
            FreezeGroup::LayerAttention(layer) => Self::LayerAttention { layer },
            FreezeGroup::LayerFfn(layer) => Self::LayerFfn { layer },
            FreezeGroup::LayerNorm(layer) => Self::LayerNorm { layer },
        }
    }
}

impl From<FreezeGroupWire> for FreezeGroup {
    fn from(value: FreezeGroupWire) -> Self {
        match value {
            FreezeGroupWire::Embeddings => Self::Embeddings,
            FreezeGroupWire::LayerAttention { layer } => Self::LayerAttention(layer),
            FreezeGroupWire::LayerFfn { layer } => Self::LayerFfn(layer),
            FreezeGroupWire::LayerNorm { layer } => Self::LayerNorm(layer),
        }
    }
}

/// The head-regularization wire mirror.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case", tag = "kind")]
enum HeadRegularizationWire {
    Lambda { lambda: f64 },
    SklearnEquivalentC { c: f64 },
}

impl From<HeadRegularization> for HeadRegularizationWire {
    fn from(value: HeadRegularization) -> Self {
        match value {
            HeadRegularization::Lambda(lambda) => Self::Lambda { lambda },
            HeadRegularization::SklearnEquivalentC { c } => Self::SklearnEquivalentC { c },
        }
    }
}

impl From<HeadRegularizationWire> for HeadRegularization {
    fn from(value: HeadRegularizationWire) -> Self {
        match value {
            HeadRegularizationWire::Lambda { lambda } => Self::Lambda(lambda),
            HeadRegularizationWire::SklearnEquivalentC { c } => Self::SklearnEquivalentC { c },
        }
    }
}

/// The schedule's wire mirror. One variant; serde rejects any other spelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum LrScheduleWire {
    WarmupLinearDecay,
}

impl From<LrSchedule> for LrScheduleWire {
    fn from(value: LrSchedule) -> Self {
        match value {
            LrSchedule::WarmupLinearDecay => Self::WarmupLinearDecay,
        }
    }
}

impl From<LrScheduleWire> for LrSchedule {
    fn from(value: LrScheduleWire) -> Self {
        match value {
            LrScheduleWire::WarmupLinearDecay => Self::WarmupLinearDecay,
        }
    }
}

/// The pair knobs' wire mirror.
///
/// Strategy and singleton policy are recorded as STRINGS rather than omitted, so the
/// serialized provenance says which policy version produced the run instead of leaving a
/// reader to assume the default. `validate_pair_policies` is what guarantees the mapping
/// back is total.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PairConfigWire {
    strategy: PairStrategyWire,
    singleton_policy: SingletonPolicyWire,
    budget: Option<u64>,
    hard_cap: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum PairStrategyWire {
    Oversampling,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum SingletonPolicyWire {
    NegativesOnly,
}

/// The requested config's wire form.
///
/// `deny_unknown_fields` rejects unknown KEYS. It says nothing about invalid VALUES — that
/// is what [`SetFitTrainConfig::new`] is for, and routing through it is the whole point of
/// this type. Field order here IS the serialized order (`serde_json` emits declaration
/// order), so the round trip is byte-stable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SetFitTrainConfigWire {
    encoder_lr: f64,
    epochs: u32,
    batch_size: u32,
    warmup_ratio: f64,
    grad_clip_max_norm: f32,
    max_length: u32,
    pair_config: PairConfigWire,
    freeze_policy: Vec<FreezeGroupWire>,
    head_regularization: HeadRegularizationWire,
    /// NOT an `Option`, so an absent seed is `missing field \`root_seed\`` rather than a
    /// silent default. A defaulted seed would make every downstream reproducibility claim
    /// about a number nobody chose.
    root_seed: u64,
    device: String,
    lr_schedule: LrScheduleWire,
}

impl From<SetFitTrainConfig> for SetFitTrainConfigWire {
    fn from(value: SetFitTrainConfig) -> Self {
        Self {
            encoder_lr: value.encoder_lr,
            epochs: value.epochs,
            batch_size: value.batch_size,
            warmup_ratio: value.warmup_ratio,
            grad_clip_max_norm: value.grad_clip_max_norm,
            max_length: value.max_length,
            pair_config: PairConfigWire {
                strategy: PairStrategyWire::Oversampling,
                singleton_policy: SingletonPolicyWire::NegativesOnly,
                budget: value.pair_config.budget,
                hard_cap: value.pair_config.hard_cap,
            },
            freeze_policy: value.freeze_policy.into_iter().map(FreezeGroupWire::from).collect(),
            head_regularization: value.head_regularization.into(),
            root_seed: value.root_seed,
            device: value.device.0,
            lr_schedule: value.lr_schedule.into(),
        }
    }
}

impl TryFrom<SetFitTrainConfigWire> for SetFitTrainConfig {
    type Error = SetFitConfigError;

    /// The deserialization door, and it opens onto [`SetFitTrainConfig::new`].
    fn try_from(wire: SetFitTrainConfigWire) -> Result<Self, Self::Error> {
        let PairConfigWire { strategy, singleton_policy, budget, hard_cap } = wire.pair_config;
        let pair_config = PairConfig {
            root_seed: wire.root_seed,
            strategy: match strategy {
                PairStrategyWire::Oversampling => PairStrategy::Oversampling,
            },
            singleton_policy: match singleton_policy {
                SingletonPolicyWire::NegativesOnly => SingletonPolicy::NegativesOnly,
            },
            budget,
            hard_cap,
        };
        Self::new(SetFitTrainRequest {
            encoder_lr: wire.encoder_lr,
            epochs: wire.epochs,
            batch_size: wire.batch_size,
            warmup_ratio: wire.warmup_ratio,
            grad_clip_max_norm: wire.grad_clip_max_norm,
            max_length: wire.max_length,
            pair_config,
            freeze_policy: wire.freeze_policy.into_iter().map(FreezeGroup::from).collect(),
            head_regularization: wire.head_regularization.into(),
            root_seed: wire.root_seed,
            device: wire.device,
            lr_schedule: wire.lr_schedule.into(),
        })
    }
}

/// The resolved config's wire form. Serialize-only by construction: nothing implements
/// `Deserialize` for it, so there is no path from bytes to a resolved device.
#[derive(Debug, Clone, PartialEq, Serialize)]
struct ResolvedSetFitConfigWire {
    requested: SetFitTrainConfigWire,
    resolved_device: String,
}

impl From<ResolvedSetFitConfig> for ResolvedSetFitConfigWire {
    fn from(value: ResolvedSetFitConfig) -> Self {
        Self { resolved_device: value.device.tag(), requested: value.requested.into() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aprender_contrastive_data::pairs::resolve_budget;
    use aprender_contrastive_data::ContrastiveDataError;

    /// This module's own source, for the non-existence assertions on the merge surface.
    const CONFIG_SOURCE: &str = include_str!("config.rs");

    /// Assemble a search needle from fragments.
    ///
    /// The scans below read the file they live in, so a whole literal would appear IN
    /// `CONFIG_SOURCE`: every count would be one too high and every non-existence assertion
    /// would fail against its own text. Assembling at runtime keeps the searched-for string out
    /// of the file — INCLUDING out of this doc comment, which is where the first draft still
    /// spelled the merge door's signature whole and made the count 2 against a source with one
    /// definition in it. A scan of its own file is only as good as the discipline that no prose
    /// nearby spells the needle.
    fn needle(parts: &[&str]) -> String {
        parts.concat()
    }

    /// A request that passes every rung, so each FALSIFY row varies exactly ONE knob.
    fn valid_request() -> SetFitTrainRequest {
        SetFitTrainRequest {
            encoder_lr: REFERENCE_ENCODER_LR,
            epochs: REFERENCE_EPOCHS,
            batch_size: REFERENCE_BATCH_SIZE,
            warmup_ratio: REFERENCE_WARMUP_RATIO,
            grad_clip_max_norm: REFERENCE_GRAD_CLIP_MAX_NORM,
            max_length: pinned_max_length(),
            pair_config: PairConfig::new(42),
            freeze_policy: Vec::new(),
            head_regularization: HeadRegularization::SklearnEquivalentC { c: 1.0 },
            root_seed: 42,
            device: "cpu".to_string(),
            lr_schedule: LrSchedule::WarmupLinearDecay,
        }
    }

    fn reject(request: SetFitTrainRequest) -> SetFitConfigError {
        SetFitTrainConfig::new(request).expect_err("this request must be rejected")
    }

    // ─── knob 1: encoder_lr ────────────────────────────────────────────────

    #[test]
    fn falsify_config_encoder_lr_rejects_zero() {
        let mut request = valid_request();
        request.encoder_lr = 0.0;
        assert!(matches!(
            reject(request),
            SetFitConfigError::NotFinitePositive { knob: "encoder_lr", .. }
        ));
    }

    #[test]
    fn falsify_config_encoder_lr_rejects_negative() {
        let mut request = valid_request();
        request.encoder_lr = -1e-5;
        let error = reject(request);
        assert!(error.to_string().contains("encoder_lr"), "{error}");
        assert!(error.to_string().contains("-0.00001"), "{error}");
    }

    #[test]
    fn falsify_config_encoder_lr_rejects_nan_and_infinities() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut request = valid_request();
            request.encoder_lr = bad;
            let error = reject(request);
            assert!(
                error.to_string().contains("encoder_lr"),
                "the error for {bad} must name the knob: {error}",
            );
        }
    }

    // ─── knob 2: epochs ────────────────────────────────────────────────────

    #[test]
    fn falsify_config_epochs_rejects_zero() {
        let mut request = valid_request();
        request.epochs = 0;
        assert_eq!(reject(request), SetFitConfigError::MustBeNonZero { knob: "epochs" });
    }

    // ─── knob 3: batch_size ────────────────────────────────────────────────

    #[test]
    fn falsify_config_batch_size_rejects_zero() {
        let mut request = valid_request();
        request.batch_size = 0;
        assert_eq!(reject(request), SetFitConfigError::MustBeNonZero { knob: "batch_size" });
    }

    // ─── knob 4: warmup_ratio ──────────────────────────────────────────────

    #[test]
    fn falsify_config_warmup_ratio_rejects_out_of_range() {
        for bad in [-0.1_f64, 1.5, f64::NAN] {
            let mut request = valid_request();
            request.warmup_ratio = bad;
            let error = reject(request);
            assert!(
                matches!(error, SetFitConfigError::RatioOutOfRange { knob: "warmup_ratio", .. }),
                "{bad} must be rejected, got {error}",
            );
        }
    }

    #[test]
    fn falsify_config_warmup_ratio_accepts_both_closed_endpoints() {
        for good in [0.0_f64, 1.0] {
            let mut request = valid_request();
            request.warmup_ratio = good;
            let config = SetFitTrainConfig::new(request).expect("endpoints are in range");
            assert!((config.warmup_ratio() - good).abs() < f64::EPSILON);
        }
    }

    // ─── knob 5: grad_clip_max_norm ────────────────────────────────────────

    #[test]
    fn falsify_config_grad_clip_rejects_zero_negative_and_nan() {
        for bad in [0.0_f32, -1.0, f32::NAN] {
            let mut request = valid_request();
            request.grad_clip_max_norm = bad;
            let error = reject(request);
            assert!(
                error.to_string().contains("grad_clip_max_norm"),
                "{bad} must be rejected naming the knob, got {error}",
            );
        }
    }

    // ─── knob 6: max_length ────────────────────────────────────────────────

    #[test]
    fn falsify_config_max_length_rejects_anything_but_the_pinned_bound() {
        let mut request = valid_request();
        request.max_length = 512;
        let error = reject(request);
        assert_eq!(error, SetFitConfigError::MaxLengthNotSupported { requested: 512, pinned: 256 },);
        let rendered = error.to_string();
        assert!(rendered.contains("512"), "{rendered}");
        assert!(rendered.contains("256"), "{rendered}");
    }

    #[test]
    fn falsify_config_max_length_accepts_only_256() {
        assert_eq!(pinned_max_length(), 256);
        for bad in [0_u32, 1, 128, 255, 257, 1024] {
            let mut request = valid_request();
            request.max_length = bad;
            assert!(
                matches!(reject(request), SetFitConfigError::MaxLengthNotSupported { .. }),
                "max_length {bad} must be rejected",
            );
        }
    }

    // ─── knob 7: pair_config ───────────────────────────────────────────────

    /// The pair budget is DELEGATED, and this proves the delegation rather than a copy.
    ///
    /// The capacity rungs need class sizes, which do not exist until `prepare()`. So the
    /// config carries the `PairConfig` unchanged and Phase 2's own `resolve_budget` is the
    /// single implementation; here we hand it a config built through the validating
    /// constructor and confirm its rejections arrive verbatim.
    #[test]
    fn falsify_config_pair_config_zero_budget_surfaces_from_phase_2_unchanged() {
        let mut request = valid_request();
        request.pair_config.budget = Some(0);
        let config = SetFitTrainConfig::new(request).expect("a zero budget is a capacity fact");
        let error = resolve_budget(config.pair_config(), &[8, 8, 8])
            .expect_err("a zero budget must be rejected");
        assert!(matches!(error, ContrastiveDataError::ZeroBudget), "{error:?}");
    }

    #[test]
    fn falsify_config_pair_config_budget_exceeding_hard_cap_binds_rather_than_clamps() {
        let mut request = valid_request();
        request.pair_config.budget = Some(1_000);
        request.pair_config.hard_cap = Some(10);
        let config = SetFitTrainConfig::new(request).expect("a cap breach is a capacity fact");
        let error = resolve_budget(config.pair_config(), &[8, 8, 8])
            .expect_err("an over-cap budget must be rejected, never clamped");
        assert!(
            matches!(
                error,
                ContrastiveDataError::BudgetExceedsHardCap { budget: 1_000, hard_cap: 10 }
            ),
            "{error:?}",
        );
    }

    #[test]
    fn falsify_config_pair_config_rejects_a_policy_v1_does_not_ship() {
        // Both policy enums ship exactly one variant in v1, so this rung cannot be reached
        // through the public enums today. What IS assertable, and what the rung exists for,
        // is that the accepted values are the v1 ones and that they are compared rather
        // than wildcarded.
        let request = valid_request();
        assert_eq!(request.pair_config.strategy, PairStrategy::Oversampling);
        assert_eq!(request.pair_config.singleton_policy, SingletonPolicy::NegativesOnly);
        assert!(validate_pair_policies(&request.pair_config).is_ok());
    }

    // ─── knob 8: freeze_policy ─────────────────────────────────────────────

    #[test]
    fn falsify_config_freeze_policy_is_canonicalized_at_construction() {
        let mut request = valid_request();
        request.freeze_policy = vec![
            FreezeGroup::LayerFfn(3),
            FreezeGroup::Embeddings,
            FreezeGroup::LayerFfn(3),
            FreezeGroup::LayerAttention(1),
        ];
        let config = SetFitTrainConfig::new(request).expect("a well-formed policy is accepted");
        assert_eq!(
            config.freeze_policy(),
            &[FreezeGroup::Embeddings, FreezeGroup::LayerAttention(1), FreezeGroup::LayerFfn(3)],
        );
    }

    #[test]
    fn falsify_config_freeze_policy_empty_is_the_all_trainable_default() {
        let config = SetFitTrainConfig::reference_defaults(13);
        assert!(config.freeze_policy().is_empty());
    }

    // ─── knob 9: head_regularization ───────────────────────────────────────

    #[test]
    fn falsify_config_head_regularization_rejects_negative_or_nonfinite_lambda() {
        for bad in [-1.0_f64, f64::NAN, f64::INFINITY] {
            let mut request = valid_request();
            request.head_regularization = HeadRegularization::Lambda(bad);
            let error = reject(request);
            assert!(
                error.to_string().contains("head_regularization.lambda"),
                "{bad} must be rejected naming the knob, got {error}",
            );
        }
    }

    #[test]
    fn falsify_config_head_regularization_accepts_zero_lambda_as_no_penalty() {
        let mut request = valid_request();
        request.head_regularization = HeadRegularization::Lambda(0.0);
        let config = SetFitTrainConfig::new(request).expect("zero lambda means no penalty");
        assert_eq!(config.head_regularization(), HeadRegularization::Lambda(0.0));
    }

    #[test]
    fn falsify_config_head_regularization_rejects_non_positive_or_nonfinite_c() {
        for bad in [0.0_f64, -1.0, f64::NAN, f64::INFINITY] {
            let mut request = valid_request();
            request.head_regularization = HeadRegularization::SklearnEquivalentC { c: bad };
            let error = reject(request);
            assert!(
                error.to_string().contains("head_regularization.c"),
                "C={bad} must be rejected naming the knob, got {error}",
            );
        }
    }

    // ─── knob 10: root_seed ────────────────────────────────────────────────

    #[test]
    fn falsify_config_root_seed_accepts_every_u64() {
        for seed in [0_u64, 1, 13, 42, u64::MAX / 2, u64::MAX] {
            let mut request = valid_request();
            request.root_seed = seed;
            request.pair_config = PairConfig::new(seed);
            let config = SetFitTrainConfig::new(request)
                .unwrap_or_else(|error| panic!("seed {seed} must be accepted: {error}"));
            assert_eq!(config.root_seed(), seed);
        }
    }

    // ─── knob 11: device ───────────────────────────────────────────────────

    #[test]
    fn falsify_config_device_rejects_the_grammar_violation_at_construction() {
        for bad in ["cuda:01", "gpu", "CUDA", "", " cpu", "cuda:16"] {
            let mut request = valid_request();
            request.device = bad.to_string();
            let error = reject(request);
            assert!(
                matches!(error, SetFitConfigError::Device(DeviceError::InvalidSpec(_))),
                "device `{bad}` must be rejected by the grammar, got {error}",
            );
        }
    }

    #[test]
    fn falsify_config_device_accepts_the_grammar_and_defers_availability() {
        // `cuda` is GRAMMATICALLY valid; whether this host has it is a `prepare()`-time
        // fact, so construction must accept it and `resolve()` must fail closed.
        let mut request = valid_request();
        request.device = "cuda".to_string();
        let config = SetFitTrainConfig::new(request).expect("`cuda` is valid grammar");
        assert_eq!(config.device().as_str(), "cuda");
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn falsify_config_device_explicit_cuda_fails_closed_at_resolve() {
        let mut request = valid_request();
        request.device = "cuda".to_string();
        let config = SetFitTrainConfig::new(request).expect("`cuda` is valid grammar");
        let error = config.resolve().expect_err("a build without CUDA must fail closed");
        assert!(
            matches!(error, SetFitConfigError::Device(DeviceError::CudaNotAvailable { .. })),
            "{error}",
        );
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn falsify_config_device_auto_resolves_to_cpu_without_cuda() {
        let mut request = valid_request();
        request.device = "auto".to_string();
        let resolved = SetFitTrainConfig::new(request)
            .expect("`auto` is valid grammar")
            .resolve()
            .expect("`auto` falls back to CPU");
        assert_eq!(resolved.device(), Device::Cpu);
    }

    // ─── knob 12: lr_schedule ──────────────────────────────────────────────

    #[test]
    fn falsify_config_lr_schedule_is_the_fixed_reference_marker() {
        let config = SetFitTrainConfig::reference_defaults(13);
        assert_eq!(config.lr_schedule(), LrSchedule::WarmupLinearDecay);
        assert_eq!(LrSchedule::default(), LrSchedule::WarmupLinearDecay);
    }

    // ─── reference defaults ────────────────────────────────────────────────

    #[test]
    fn falsify_config_reference_defaults_are_the_reference_values() {
        let config = SetFitTrainConfig::reference_defaults(13);
        assert!((config.encoder_lr() - 2e-5).abs() < f64::EPSILON, "{}", config.encoder_lr());
        assert_eq!(config.epochs(), 1);
        assert_eq!(config.batch_size(), 16);
        assert!((config.warmup_ratio() - 0.1).abs() < f64::EPSILON);
        assert!((config.grad_clip_max_norm() - 1.0).abs() < f32::EPSILON);
        assert_eq!(config.max_length(), 256);
        assert_eq!(config.root_seed(), 13);
        assert_eq!(config.device().as_str(), "cpu");
    }

    #[test]
    fn falsify_config_adamw_hyperparameters_are_fixed_not_knobs() {
        // Pinned as VALUES so an edit to the reference recipe is visible in a diff rather
        // than only in a training curve.
        assert!((ADAMW_BETA1 - 0.9).abs() < f32::EPSILON);
        assert!((ADAMW_BETA2 - 0.999).abs() < f32::EPSILON);
        assert!((ADAMW_EPSILON - 1e-8).abs() < f32::EPSILON);
        assert!((ADAMW_WEIGHT_DECAY - 0.0).abs() < f32::EPSILON);
    }

    // ─── deserialization: the second construction path that must not exist ─

    fn valid_payload() -> String {
        serde_json::to_string(&SetFitTrainConfig::reference_defaults(13))
            .expect("the reference config serializes")
    }

    /// String surgery that PROVES it applied.
    ///
    /// This helper exists because the first draft of these tests did not have it and was
    /// silently vacuous: `serde_json` renders `2e-5` as `0.00002`, so a `.replace()` keyed
    /// on the literal `2e-5` matched nothing, and the test then asserted that an
    /// UNMODIFIED, perfectly valid payload failed to deserialize. It went red for the right
    /// reason only because a substitution-applied assertion happened to be there. Every
    /// substitution below now goes through this function so none of them can rot into a
    /// test of the valid payload.
    fn substitute(payload: &str, from: &str, to: &str) -> String {
        let out = payload.replace(from, to);
        assert_ne!(
            out, payload,
            "the substitution `{from}` -> `{to}` did not apply; the test would be vacuous. \
             Payload was: {payload}",
        );
        out
    }

    #[test]
    fn falsify_config_deserialize_rejects_negative_encoder_lr() {
        let payload = substitute(&valid_payload(), "\"encoder_lr\":0.00002", "\"encoder_lr\":-1.0");
        let error = serde_json::from_str::<SetFitTrainConfig>(&payload)
            .expect_err("a negative encoder_lr must not deserialize");
        assert!(error.to_string().contains("encoder_lr"), "{error}");
    }

    #[test]
    fn falsify_config_deserialize_rejects_missing_root_seed() {
        let payload = substitute(&valid_payload(), "\"root_seed\":13,", "");
        assert!(!payload.contains("root_seed"), "the key must be gone: {payload}");
        let error = serde_json::from_str::<SetFitTrainConfig>(&payload)
            .expect_err("an absent root_seed must not deserialize");
        assert!(error.to_string().contains("root_seed"), "{error}");
    }

    #[test]
    fn falsify_config_deserialize_rejects_warmup_ratio_above_one() {
        let payload = substitute(&valid_payload(), "\"warmup_ratio\":0.1", "\"warmup_ratio\":1.5");
        let error = serde_json::from_str::<SetFitTrainConfig>(&payload)
            .expect_err("an out-of-range warmup_ratio must not deserialize");
        assert!(error.to_string().contains("warmup_ratio"), "{error}");
    }

    #[test]
    fn falsify_config_deserialize_rejects_max_length_512() {
        let payload = substitute(&valid_payload(), "\"max_length\":256", "\"max_length\":512");
        let error = serde_json::from_str::<SetFitTrainConfig>(&payload)
            .expect_err("an unsupported max_length must not deserialize");
        let rendered = error.to_string();
        assert!(rendered.contains("512"), "{rendered}");
        assert!(rendered.contains("256"), "{rendered}");
    }

    #[test]
    fn falsify_config_deserialize_rejects_an_unknown_key() {
        let payload =
            substitute(&valid_payload(), "{\"encoder_lr\"", "{\"secret_knob\":1,\"encoder_lr\"");
        let error = serde_json::from_str::<SetFitTrainConfig>(&payload)
            .expect_err("deny_unknown_fields must reject an unknown key");
        assert!(error.to_string().contains("secret_knob"), "{error}");
    }

    #[test]
    fn falsify_config_deserialize_rejects_an_unknown_lr_schedule() {
        let payload =
            substitute(&valid_payload(), "\"warmup_linear_decay\"", "\"warmup_cosine_decay\"");
        assert!(
            serde_json::from_str::<SetFitTrainConfig>(&payload).is_err(),
            "an unshipped schedule must not deserialize",
        );
    }

    #[test]
    fn falsify_config_deserialize_rejects_a_malformed_freeze_group() {
        let base = SetFitTrainConfig::reference_defaults(13);
        let mut request = SetFitTrainRequest {
            encoder_lr: base.encoder_lr(),
            epochs: base.epochs(),
            batch_size: base.batch_size(),
            warmup_ratio: base.warmup_ratio(),
            grad_clip_max_norm: base.grad_clip_max_norm(),
            max_length: base.max_length(),
            pair_config: *base.pair_config(),
            freeze_policy: Vec::new(),
            head_regularization: base.head_regularization(),
            root_seed: base.root_seed(),
            device: base.device().as_str().to_string(),
            lr_schedule: base.lr_schedule(),
        };
        request.freeze_policy = vec![FreezeGroup::LayerFfn(2)];
        let payload = serde_json::to_string(
            &SetFitTrainConfig::new(request).expect("a well-formed policy serializes"),
        )
        .expect("serializes");

        // A group missing its `layer` field, and a group nobody ships.
        let missing_layer = substitute(&payload, ",\"layer\":2", "");
        assert!(
            serde_json::from_str::<SetFitTrainConfig>(&missing_layer).is_err(),
            "a freeze group missing its layer must not deserialize",
        );
        let unknown_group = substitute(&payload, "\"layer_ffn\"", "\"layer_bogus\"");
        assert!(
            serde_json::from_str::<SetFitTrainConfig>(&unknown_group).is_err(),
            "an unknown freeze group must not deserialize",
        );
    }

    #[test]
    fn falsify_config_deserialize_round_trips_byte_identically() {
        let original = SetFitTrainConfig::reference_defaults(29);
        let first = serde_json::to_string(&original).expect("serializes");
        let parsed: SetFitTrainConfig =
            serde_json::from_str(&first).expect("a valid payload deserializes");
        let second = serde_json::to_string(&parsed).expect("re-serializes");
        assert_eq!(first, second, "serialize -> deserialize -> serialize must be stable");
        assert_eq!(original, parsed);
    }

    #[test]
    fn falsify_config_resolved_form_serializes_both_what_was_asked_and_what_was_resolved() {
        let resolved = SetFitTrainConfig::reference_defaults(31)
            .resolve()
            .expect("cpu resolves on every host");
        let json = serde_json::to_string(&resolved).expect("serializes");
        assert!(json.contains("\"requested\""), "{json}");
        assert!(json.contains("\"resolved_device\":\"cpu\""), "{json}");
    }

    // ─── the override/merge door (04-14, D-07) ─────────────────────────────

    /// A request whose knobs all DIFFER from the reference defaults.
    ///
    /// A round trip over `reference_defaults` would pass even if a field were dropped and
    /// re-defaulted, because the dropped value and the default are the same number. Every knob
    /// that CAN vary is varied here; `max_length` and `lr_schedule` each have exactly one legal
    /// value, so they are pinned rather than varied and the twelve-knob assertion says so.
    fn distinctive_request() -> SetFitTrainRequest {
        SetFitTrainRequest {
            encoder_lr: 3e-5,
            epochs: 4,
            batch_size: 8,
            warmup_ratio: 0.25,
            grad_clip_max_norm: 0.5,
            max_length: pinned_max_length(),
            pair_config: PairConfig { budget: Some(64), ..PairConfig::new(7) },
            freeze_policy: vec![FreezeGroup::Embeddings, FreezeGroup::LayerFfn(2)],
            head_regularization: HeadRegularization::Lambda(0.75),
            root_seed: 7,
            device: "cpu".to_string(),
            lr_schedule: LrSchedule::WarmupLinearDecay,
        }
    }

    #[test]
    fn falsify_config_to_request_round_trips_all_twelve_knobs_identically() {
        let original =
            SetFitTrainConfig::new(distinctive_request()).expect("the fixture request is valid");
        let request = original.to_request();

        // All TWELVE, knob by knob. A subset would let a forgotten field drift silently, and
        // the equality assertion at the bottom alone would not localize which one.
        assert!((request.encoder_lr - original.encoder_lr()).abs() < f64::EPSILON); // 1
        assert_eq!(request.epochs, original.epochs()); // 2
        assert_eq!(request.batch_size, original.batch_size()); // 3
        assert!((request.warmup_ratio - original.warmup_ratio()).abs() < f64::EPSILON); // 4
        assert!((request.grad_clip_max_norm - original.grad_clip_max_norm()).abs() < f32::EPSILON); // 5
        assert_eq!(request.max_length, original.max_length()); // 6
        assert_eq!(&request.pair_config, original.pair_config()); // 7
        assert_eq!(request.freeze_policy.as_slice(), original.freeze_policy()); // 8
        assert_eq!(request.head_regularization, original.head_regularization()); // 9
        assert_eq!(request.root_seed, original.root_seed()); // 10
        assert_eq!(request.device, original.device().as_str()); // 11
        assert_eq!(request.lr_schedule, original.lr_schedule()); // 12

        // And the whole thing revalidates to the config it was read off.
        let rebuilt = SetFitTrainConfig::new(request)
            .expect("a request read off a validated config is itself valid");
        assert_eq!(rebuilt, original);
    }

    #[test]
    fn falsify_config_to_request_seed_override_moves_the_seed_and_nothing_else() {
        let original =
            SetFitTrainConfig::new(distinctive_request()).expect("the fixture request is valid");
        let mut request = original.to_request();
        request.root_seed = 99;
        let merged =
            SetFitTrainConfig::new(request).expect("a seed override is still a valid request");

        assert_eq!(merged.root_seed(), 99);
        assert_ne!(merged.root_seed(), original.root_seed());

        // The other ten knobs, unchanged.
        assert!((merged.encoder_lr() - original.encoder_lr()).abs() < f64::EPSILON);
        assert_eq!(merged.epochs(), original.epochs());
        assert_eq!(merged.batch_size(), original.batch_size());
        assert!((merged.warmup_ratio() - original.warmup_ratio()).abs() < f64::EPSILON);
        assert!((merged.grad_clip_max_norm() - original.grad_clip_max_norm()).abs() < f32::EPSILON);
        assert_eq!(merged.max_length(), original.max_length());
        assert_eq!(merged.freeze_policy(), original.freeze_policy());
        assert_eq!(merged.head_regularization(), original.head_regularization());
        assert_eq!(merged.device().as_str(), original.device().as_str());
        assert_eq!(merged.lr_schedule(), original.lr_schedule());

        // Knob 7 is the one that MOVES WITH the seed, and it must: `new` normalizes
        // `pair_config.root_seed` to the top-level seed (see the knob-7 comment there), which
        // is what makes a `--seed` override actually reseed the pair stream instead of leaving
        // a stale second seed behind that the provenance record would still describe. Its
        // policy half is unchanged, so the override is a reseed and not a reconfiguration.
        assert_eq!(merged.pair_config().root_seed, 99);
        assert_eq!(merged.pair_config().budget, original.pair_config().budget);
        assert_eq!(merged.pair_config().hard_cap, original.pair_config().hard_cap);
        assert_eq!(merged.pair_config().strategy, original.pair_config().strategy);
        assert_eq!(merged.pair_config().singleton_policy, original.pair_config().singleton_policy,);
    }

    #[test]
    fn falsify_config_to_request_device_override_is_validated_by_new_not_by_assignment() {
        let original = SetFitTrainConfig::reference_defaults(13);
        let mut request = original.to_request();
        request.device = "gpu".to_string();

        let error = SetFitTrainConfig::new(request)
            .expect_err("`gpu` is not in the device grammar, whichever door it arrives through");
        assert!(
            matches!(error, SetFitConfigError::Device(DeviceError::InvalidSpec(_))),
            "the merge must be validated AS A WHOLE, got {error}",
        );
        assert!(error.to_string().contains("device"), "{error}");

        // The config the request was read off is untouched: an override that fails cannot have
        // half-applied itself, because `to_request` hands back a REQUEST and the only path to a
        // config is `new`.
        assert_eq!(original.device().as_str(), "cpu");
    }

    #[test]
    fn falsify_config_to_request_max_length_override_is_still_refused_by_the_pinned_bound() {
        let original = SetFitTrainConfig::reference_defaults(13);
        let mut request = original.to_request();
        request.max_length = 512;

        let error = SetFitTrainConfig::new(request)
            .expect_err("TRN-02's pinned bound survives the override path");
        assert_eq!(error, SetFitConfigError::MaxLengthNotSupported { requested: 512, pinned: 256 });
        assert_eq!(original.max_length(), 256, "the original is untouched");
    }

    #[test]
    fn falsify_config_to_request_is_the_whole_merge_surface() {
        assert_eq!(
            CONFIG_SOURCE.matches(&needle(&["pub fn ", "to_request"])).count(),
            1,
            "exactly one merge door",
        );

        // No method takes a mutable receiver. Two doors to the same merge is how the two
        // diverge, and a setter on an ALREADY-VALIDATED config would skip `new` entirely --
        // which is the invalid-merge hole this door exists to close.
        assert!(
            !CONFIG_SOURCE.contains(&needle(&["&mut ", "self"])),
            "no method on a validated config may take a mutable receiver",
        );
        for banned in [
            needle(&["pub fn ", "with_seed"]),
            needle(&["pub fn ", "with_device"]),
            needle(&["pub fn ", "set_"]),
        ] {
            assert!(
                !CONFIG_SOURCE.contains(&banned),
                "`{banned}` would be a second merge path that `new` does not police",
            );
        }
    }
}
