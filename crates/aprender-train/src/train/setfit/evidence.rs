//! The canonical SetFit-identity evidence table and its binding hash (D-09, D-10, D-12).
//!
//! Contract: `setfit-train-lifecycle-v1` (authored in plan 03-06). Requirement: TRN-03.
//!
//! # The relative-delta formula, and why it has a floored, support-restricted denominator
//!
//! ```text
//! relative_delta(p) = ||dTheta_p||_2 / max(denom(p), s_class(p))
//!
//! denom(p) = ||theta_init_p restricted to the SUPPORT of dTheta_p||_2   for the SPARSE class
//!          = ||theta_init_p||_2                                          otherwise
//! ```
//!
//! Two properties follow, and both are load-bearing.
//!
//! **It is FINITE at zero initialization.** The naive `||dTheta|| / ||theta_init||` is `0/0`
//! or `x/0` for a bias initialized to exactly zero. `NaN > eps` is FALSE, so an un-floored
//! form would have silently REJECTED every run containing a zero-init parameter — a gate that
//! fails closed for the wrong reason is worse than one that does not exist, because the
//! diagnosis points at the model instead of at the metric. The positive contracted floor
//! `s_class` removes the case entirely: when the initial norm is below unit scale the ratio
//! degrades gracefully into an ABSOLUTE movement measure, which is the right question to ask
//! about a parameter that started at zero.
//!
//! **For the sparse class it is INVARIANT to vocabulary size.** A whole-table denominator
//! would make the embedding class's ratio a function of how many rows the table has: a
//! few-shot batch touches a handful of rows regardless, so `||dTheta||` is fixed while
//! `||theta_init||` grows as `sqrt(V)`, and a ratio calibrated on a 97-row fixture would be
//! roughly `sqrt(30522/97) ~ 17.7` times too large for the production encoder. Restricting
//! the denominator to the rows the delta actually touched cancels the vocabulary factor,
//! which is what makes a fixture-calibrated epsilon transferable at all. Every row records
//! its observed `delta_support_fraction` so the claim stays checkable rather than asserted:
//! see `evidence_sparse_denominator_is_restricted_to_the_support`.
//!
//! A parameter that did not move AT ALL is never SetFit, whatever its ratio says, so the
//! strict predicate `||dTheta||_2 > 0` is recorded ALONGSIDE the ratio rather than folded
//! into it.
//!
//! # The summary's hash BINDS it to the table
//!
//! [`EvidenceSummary::table_hash`] is the SHA-256 of the full canonical table bytes. Editing
//! a row and leaving the summary alone produces a summary whose hash does not match its
//! table, and `evidence_hash_binds_the_summary_to_the_table` proves the detection. That is
//! tamper-EVIDENCE and linkage. It is not a signature and it does not make the pair
//! unforgeable: anyone who can edit the table can recompute the hash. The distinction is
//! recorded because the earlier wording of D-12 overclaimed it.
//!
//! # No wall-clock field, anywhere
//!
//! Every field of every serialized struct here is a hash, a count or a measured norm. A
//! timing field would make two identical runs serialize differently and would take TRN-06's
//! bitwise claim with it.
//!
//! # The DERIVATION surface, and what its verdict does NOT establish (D-18, plan 05-14)
//!
//! The calibration matrices below derive an epsilon BASIS — a per-class window
//! `[lower, upper]` a frozen epsilon would have to sit inside. Until plan 05-14 that window
//! was only REPORTED: plan 05-01's four-cell combine exited `rc=0` while printing `EMPTY` for
//! five of six classes, so a green derivation run was not evidence that an epsilon basis
//! existed at all. `epsilon_basis` is now the single site that applies the window rule, and it
//! returns a typed refusal when any class the run's regime GATES has no legal window. The
//! required set is READ from that regime's frozen table (`Thresholds::table_for`); where no
//! table resolves — the production regime's state today — EVERY class is required, because a
//! class the contract has never recorded must not be exempted by omission.
//!
//! The RESIDUAL, stated rather than left to be discovered:
//!
//! - This makes an empty basis unable to coexist with a green run. It does **not** make a
//!   non-empty basis SUFFICIENT. A window can be legal and still too thin to be worth
//!   freezing — `eps/noise` is the column that speaks to that, and it is a judgement input,
//!   not a gate.
//! - Coverage can still be PROVISIONAL. A partial measurement stays legal deliberately (05-01's
//!   per-pass resumable workflow depends on it) and is carried as a typed value rather than
//!   banner prose, so a consumer cannot read a provisional basis as a freezable one. Collapse
//!   fails and provisionality does not because, within a fixed rule, `lower` is monotone
//!   non-decreasing and `upper` monotone non-increasing in the measured cells: a collapse seen
//!   on a subset can never be cleared by measuring more, while a provisional basis genuinely
//!   can still change.
//! - The verdict is RULE-AGNOSTIC. It enforces that the emitted window is a non-empty interval
//!   for every gated class; it does not choose the arithmetic that produces the bounds. That
//!   arithmetic is `WINDOW_SAFETY_FACTOR` and the two lines beside it, which plan
//!   05-03 revisits against 05-01's evidence with these tests still holding afterwards.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::tune::{ParamRecord, TuneOutput};

/// The schema version of the evidence wire form.
pub(crate) const EVIDENCE_SCHEMA_VERSION: u32 = 1;

/// The contract this evidence discharges.
pub(crate) const EVIDENCE_CONTRACT_VERSION: &str = "setfit-train-lifecycle-v1";

// ===========================================================================================
// Parameter classification
// ===========================================================================================

/// The parameter classes epsilon is frozen per, in plan 03-06.
///
/// Five rather than three, because weight and bias initializations differ by orders of
/// magnitude within the same block and a single epsilon over both would be set by whichever
/// is noisier. `Ord` so the class map iterates deterministically.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParameterClass {
    /// `embeddings.*_embeddings.weight` — the SPARSE class.
    Embedding,
    /// `*.LayerNorm.weight`.
    LayerNormWeight,
    /// `*.LayerNorm.bias`.
    LayerNormBias,
    /// `*.attention.self.*.weight`, `*.dense.weight`.
    ProjectionWeight,
    /// `*.attention.self.query.bias`, `*.attention.self.value.bias`, `*.dense.bias`.
    ProjectionBias,
    /// `*.attention.self.key.bias` — the ANALYTICALLY GRADIENT-FREE class.
    ///
    /// Split out of [`Self::ProjectionBias`] on a mechanism, not on a failing margin. Softmax
    /// is invariant to a constant shift of its inputs, and adding the key bias `b_k` to every
    /// key contributes `q_i . b_k` to the pre-softmax logit of EVERY key `j` for a given query
    /// `i` — the same amount for all `j`. The shift cancels, so `dL/db_k = 0` in exact
    /// arithmetic. The query bias does not have this property (`(q + b_q) . k_j` varies with
    /// `j`), and neither does the value bias (it adds a constant to the attention output,
    /// which the downstream layers see), which is why this class is the key bias ALONE and not
    /// "attention biases".
    ///
    /// Measured on the fixture slice: `grad_norm_max` 2.290e-10 for
    /// `encoder.layer.1.attention.self.key.bias` against 8.007e-3 for
    /// `encoder.layer.0.attention.self.query.weight` in the same block — 3.5e7x smaller, which
    /// is f32 cancellation residue rather than a small gradient.
    AttentionKeyBias,
}

impl ParameterClass {
    /// Every class, in `Ord` order — the iteration order every report uses.
    pub(crate) const ALL: [Self; 6] = [
        Self::Embedding,
        Self::LayerNormWeight,
        Self::LayerNormBias,
        Self::ProjectionWeight,
        Self::ProjectionBias,
        Self::AttentionKeyBias,
    ];

    /// Whether the denominator is restricted to the delta's support.
    ///
    /// Only the embedding tables. Every other parameter here is dense: a batch that touches
    /// the block at all touches every element of it, so the support IS the whole tensor and
    /// restricting would be a no-op dressed up as a policy.
    pub(crate) fn is_sparse(self) -> bool {
        matches!(self, Self::Embedding)
    }

    /// The contracted positive scale floor `s_class`.
    ///
    /// Unit for every class in v1, and stated as a per-class table anyway because plan 03-06
    /// freezes a per-class EPSILON and the two want to be read side by side. The value is 1.0
    /// rather than something smaller because it only ENGAGES when the (possibly
    /// support-restricted) initial norm is below unit scale — that is, for a parameter that
    /// started at or near zero — and unit scale is where the ratio's meaning changes from
    /// "fraction of the initial magnitude" to "absolute movement". Making the transition
    /// happen at 1.0 puts it somewhere a reader can name.
    pub(crate) fn scale_floor(self) -> f64 {
        match self {
            Self::Embedding
            | Self::LayerNormWeight
            | Self::LayerNormBias
            | Self::ProjectionWeight
            | Self::ProjectionBias
            | Self::AttentionKeyBias => 1.0,
        }
    }

    /// The stable string this class serializes as.
    pub(crate) fn tag(self) -> &'static str {
        match self {
            Self::Embedding => "embedding",
            Self::LayerNormWeight => "layer_norm_weight",
            Self::LayerNormBias => "layer_norm_bias",
            Self::ProjectionWeight => "projection_weight",
            Self::ProjectionBias => "projection_bias",
            Self::AttentionKeyBias => "attention_key_bias",
        }
    }
}

impl fmt::Display for ParameterClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.tag())
    }
}

/// Failure modes of the evidence layer.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub(crate) enum EvidenceError {
    /// A parameter name matched no class.
    ///
    /// FAIL CLOSED. A default bucket would absorb a renamed parameter into whichever class
    /// happened to be the fallback, and the epsilon frozen for that class would then be
    /// applied to something it was never measured on — silently, since nothing about the
    /// output would change shape.
    UnclassifiedParameter {
        /// The offending dotted name.
        name: String,
    },
    /// The canonical bytes could not be produced.
    Serialization {
        /// The renderer's diagnostic.
        reason: String,
    },
    /// A measurement in the table is not finite, so the canonical bytes would carry a `null`.
    ///
    /// FAIL CLOSED, for two independent reasons (REVIEW CR-03): the digest over those bytes is
    /// not injective — `+inf`, `-inf` and every `NaN` all render as the same `null` — and a
    /// bundle sealed with it cannot be reloaded, because `null` does not deserialize to `f64`.
    /// Neither failure is visible in the digest's shape, which is why this is refused at
    /// production rather than diagnosed later.
    NonFiniteMeasurement {
        /// Dotted path to the offending field, e.g. `rows.encoder.layer.0.attn.q.init_norm`.
        field: String,
    },
}

/// The dotted path of the first `null` in a JSON tree, in document order.
///
/// `None` means every leaf is a real value. Used only as a finiteness check — see
/// [`UpdateEvidence::to_canonical_bytes`] for why the check is a null-scan and not a list of
/// float field names.
fn first_null_path(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::Null => Some(String::new()),
        serde_json::Value::Object(map) => map
            .iter()
            .find_map(|(key, child)| first_null_path(child).map(|rest| join_path(key, &rest))),
        serde_json::Value::Array(items) => items.iter().enumerate().find_map(|(index, child)| {
            first_null_path(child).map(|rest| join_path(&index.to_string(), &rest))
        }),
        _ => None,
    }
}

/// Join one path segment onto a (possibly empty) remainder.
fn join_path(head: &str, rest: &str) -> String {
    if rest.is_empty() {
        head.to_string()
    } else {
        format!("{head}.{rest}")
    }
}

impl fmt::Display for EvidenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnclassifiedParameter { name } => write!(
                f,
                "parameter `{name}` matches no parameter class; the encoder's dotted naming \
                 has drifted from the class mapping and a per-class epsilon cannot be applied \
                 to it (contract setfit-train-lifecycle-v1, requirement TRN-03)",
            ),
            Self::Serialization { reason } => write!(
                f,
                "the evidence table could not be rendered canonically: {reason} \
                 (contract setfit-train-lifecycle-v1, requirement TRN-03)",
            ),
            Self::NonFiniteMeasurement { field } => write!(
                f,
                "the evidence field `{field}` is not finite, so the canonical bytes would carry \
                 a `null` there. Refused at production: serde_json renders +inf, -inf and every \
                 NaN identically, so the table hash could not tell three different divergences \
                 apart, and the bundle sealed with it would fail its own reload (contract \
                 setfit-train-lifecycle-v1, requirement TRN-03)",
            ),
        }
    }
}

impl std::error::Error for EvidenceError {}

/// Classify a parameter by its HF dotted name. Pure, total on its accepted domain, fail-closed
/// everywhere else.
///
/// # Errors
///
/// [`EvidenceError::UnclassifiedParameter`] for any name outside the mapping.
pub(crate) fn classify_parameter(name: &str) -> Result<ParameterClass, EvidenceError> {
    let unclassified = || EvidenceError::UnclassifiedParameter { name: name.to_string() };
    let leaf = name.rsplit('.').next().unwrap_or("");

    // The embedding TABLES only. `embeddings.LayerNorm.weight` lives under the same prefix
    // and is deliberately NOT sparse, which is why this test is on the suffix rather than on
    // the `embeddings.` prefix alone.
    if name.starts_with("embeddings.") && name.ends_with("_embeddings.weight") {
        return Ok(ParameterClass::Embedding);
    }
    if name.contains(".LayerNorm.") {
        return match leaf {
            "weight" => Ok(ParameterClass::LayerNormWeight),
            "bias" => Ok(ParameterClass::LayerNormBias),
            _ => Err(unclassified()),
        };
    }
    // BEFORE the general projection branch: the key BIAS is gradient-free, the key WEIGHT is
    // not (only a constant shift of the logits cancels, and `W_k x` is not constant in x).
    if name.contains(".attention.self.key.") {
        return match leaf {
            "weight" => Ok(ParameterClass::ProjectionWeight),
            "bias" => Ok(ParameterClass::AttentionKeyBias),
            _ => Err(unclassified()),
        };
    }
    if name.contains(".dense.") || name.contains(".attention.self.") {
        return match leaf {
            "weight" => Ok(ParameterClass::ProjectionWeight),
            "bias" => Ok(ParameterClass::ProjectionBias),
            _ => Err(unclassified()),
        };
    }
    Err(unclassified())
}

// ===========================================================================================
// The relative delta
// ===========================================================================================

/// The denominator actually used, and the floor that was applied to it.
///
/// Returned as a pair rather than folded into the ratio so the evidence row can record BOTH
/// and a reader can tell a small ratio caused by a large denominator from one caused by a
/// small delta.
pub(crate) fn denominator_of(record: &ParamRecord, class: ParameterClass) -> (f64, f64) {
    let raw = if class.is_sparse() { record.init_norm_on_support } else { record.init_norm };
    (raw, class.scale_floor())
}

/// `||dTheta|| / max(denom, s_class)` — THE formula, defined exactly once.
pub(crate) fn relative_delta(record: &ParamRecord, class: ParameterClass) -> f64 {
    let (raw, floor) = denominator_of(record, class);
    record.delta_norm / raw.max(floor)
}

/// The strict predicate: did this parameter move AT ALL?
///
/// Separate from the ratio on purpose. A parameter whose delta is exactly zero has a ratio of
/// exactly zero too, but the two facts answer different questions and 03-06 gates on both.
pub(crate) fn moved(record: &ParamRecord) -> bool {
    record.delta_norm > 0.0
}

// ===========================================================================================
// The canonical table
// ===========================================================================================

/// One parameter's row. Fixed field order; the wire form is the canonical form.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceRow {
    /// The HF dotted name.
    pub(crate) name: String,
    /// The class the per-class epsilon is frozen for.
    pub(crate) class: ParameterClass,
    /// Elements in the tensor.
    pub(crate) element_count: u64,
    /// `||theta_init||_2` over the whole tensor.
    pub(crate) init_norm: f64,
    /// `||theta_final - theta_init||_2`.
    pub(crate) delta_norm: f64,
    /// Elements whose delta is not exactly zero.
    pub(crate) delta_support_count: u64,
    /// `delta_support_count / element_count`.
    pub(crate) delta_support_fraction: f64,
    /// The denominator BEFORE the floor was applied.
    pub(crate) denom_used: f64,
    /// The contracted floor for this class.
    pub(crate) scale_floor_used: f64,
    /// `delta_norm / max(denom_used, scale_floor_used)`.
    pub(crate) relative_delta: f64,
    /// The strict `||dTheta|| > 0` predicate.
    pub(crate) moved: bool,
    /// Largest per-step PRE-clip gradient norm.
    pub(crate) grad_norm_max: f64,
    /// Index-order mean of the per-step PRE-clip gradient norms.
    pub(crate) grad_norm_mean: f64,
    /// Steps at which this parameter had a gradient.
    pub(crate) steps_observed: u64,
}

/// The full evidence table: per-parameter rows plus the run-level recorded facts.
///
/// `rows` is a `BTreeMap`, so name ordering is discharged STRUCTURALLY rather than by a sort
/// somebody has to remember to call. `batch_boundary_list` is a `Vec` and its order IS the
/// event order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpdateEvidence {
    /// Wire schema version.
    pub(crate) schema_version: u32,
    /// Per-parameter rows, in name order.
    pub(crate) rows: BTreeMap<String, EvidenceRow>,
    /// Hex SHA-256 of the loss trace's LE `f32` bits, in step order.
    pub(crate) loss_trace_hash: String,
    /// Hex SHA-256 of the pairs actually consumed, absorbed at the draw.
    pub(crate) consumed_pair_digest: String,
    /// Hex SHA-256 of the batch boundaries actually opened.
    pub(crate) batch_boundary_digest: String,
    /// `(epoch, start_ordinal, len)` per batch, in the order the batches opened.
    ///
    /// The RECORDED source for plan 03-08's `batch_boundaries()` accessor. Without it that
    /// accessor would have to recompute the boundaries from configuration, which is the
    /// false-green the in-band digests exist to remove.
    pub(crate) batch_boundary_list: Vec<(u32, u64, u32)>,
    /// Hex SHA-256 of the ordered trainable parameter names.
    pub(crate) parameter_registry_hash: String,
    /// Optimizer steps taken.
    pub(crate) step_count: u64,
    /// Mean of the first `k` losses.
    pub(crate) first_k_mean: f64,
    /// Mean of the last `k` losses.
    pub(crate) last_k_mean: f64,
    /// The endpoint window.
    pub(crate) k: usize,
    /// Smallest relative delta among the sparse class.
    pub(crate) embedding_delta_min: f64,
    /// Median relative delta among the sparse class.
    pub(crate) embedding_delta_median: f64,
    /// Largest relative delta among the sparse class.
    pub(crate) embedding_delta_max: f64,
    /// Largest PRE-clip global gradient norm across steps.
    pub(crate) pre_clip_norm_max: f64,
    /// The regime this evidence was measured under.
    pub(crate) calibration_regime_id: String,
}

impl UpdateEvidence {
    /// Build the table from a recorded [`TuneOutput`].
    ///
    /// Every digest is MOVED from the recorded output; nothing here recomputes one.
    ///
    /// # Errors
    ///
    /// [`EvidenceError::UnclassifiedParameter`] if any parameter name matches no class.
    pub(crate) fn from_tune_output(
        out: &TuneOutput,
        calibration_regime_id: &str,
    ) -> Result<Self, EvidenceError> {
        let mut rows = BTreeMap::new();
        for (name, record) in &out.per_name {
            rows.insert(name.clone(), row_for(name, record)?);
        }
        let embedding_deltas: Vec<f64> =
            rows.values().filter(|r| r.class.is_sparse()).map(|r| r.relative_delta).collect();
        Ok(Self {
            schema_version: EVIDENCE_SCHEMA_VERSION,
            rows,
            loss_trace_hash: hex::encode(out.loss_trace_hash),
            consumed_pair_digest: hex::encode(out.consumed_pair_digest),
            batch_boundary_digest: hex::encode(out.batch_boundary_digest),
            batch_boundary_list: out.batch_boundaries.clone(),
            parameter_registry_hash: hex::encode(out.parameter_registry_hash),
            step_count: out.step_count,
            first_k_mean: out.first_k_mean,
            last_k_mean: out.last_k_mean,
            k: out.endpoint_k,
            embedding_delta_min: min_of(&embedding_deltas),
            embedding_delta_median: median_of(&embedding_deltas),
            embedding_delta_max: max_of(&embedding_deltas),
            pre_clip_norm_max: max_of(
                &out.pre_clip_norms.iter().map(|v| f64::from(*v)).collect::<Vec<f64>>(),
            ),
            calibration_regime_id: calibration_regime_id.to_string(),
        })
    }

    /// The canonical bytes.
    ///
    /// `serde_json` over a fixed-field-order struct with `BTreeMap` rows. Two runs producing
    /// the same measurements produce the same bytes.
    ///
    /// # Errors
    ///
    /// [`EvidenceError::Serialization`].
    pub(crate) fn to_canonical_bytes(&self) -> Result<Vec<u8>, EvidenceError> {
        let bytes = serde_json::to_vec(self)
            .map_err(|e| EvidenceError::Serialization { reason: e.to_string() })?;

        // REVIEW CR-03. `serde_json` renders EVERY non-finite `f64` as `null`, silently: `+inf`,
        // `-inf` and every `NaN` payload produce byte-identical output. So `table_hash` was not
        // injective over exactly the values a diverged run produces, and the bundle sealed with
        // that digest could not be reloaded — `null` is not an `f64`. A digest is not a
        // finiteness check; this is the check.
        //
        // The scan runs on the bytes PARSED BACK, never on a `Value` used to produce them:
        // routing emission through `serde_json::Value` could reorder keys relative to struct
        // field order and would invalidate every digest already recorded. Parsing cannot.
        //
        // Scanning for `null` rather than enumerating the float fields is deliberate. Neither
        // `EvidenceRow` nor `UpdateEvidence` has an `Option` field, so a `null` can ONLY be a
        // non-finite float — which makes this exact today and still exact after someone adds an
        // `f64`. An enumerated field list is the guard that silently stops covering the newest
        // field, which is how these checks rot.
        if let Some(path) = first_null_path(
            &serde_json::from_slice::<serde_json::Value>(&bytes)
                .map_err(|e| EvidenceError::Serialization { reason: e.to_string() })?,
        ) {
            return Err(EvidenceError::NonFiniteMeasurement { field: path });
        }

        Ok(bytes)
    }

    /// SHA-256 of the canonical bytes.
    ///
    /// # Errors
    ///
    /// [`EvidenceError::Serialization`].
    pub(crate) fn table_hash(&self) -> Result<[u8; 32], EvidenceError> {
        let mut hasher = Sha256::new();
        hasher.update(self.to_canonical_bytes()?);
        Ok(hasher.finalize().into())
    }

    /// Rows of one class, in name order.
    pub(crate) fn rows_of_class(&self, class: ParameterClass) -> Vec<&EvidenceRow> {
        self.rows.values().filter(|r| r.class == class).collect()
    }
}

/// Build one row. The single place a `ParamRecord` becomes evidence.
fn row_for(name: &str, record: &ParamRecord) -> Result<EvidenceRow, EvidenceError> {
    let class = classify_parameter(name)?;
    let (denom_used, scale_floor_used) = denominator_of(record, class);
    #[allow(clippy::cast_precision_loss)]
    let fraction = if record.element_count == 0 {
        0.0
    } else {
        record.delta_support_count as f64 / record.element_count as f64
    };
    Ok(EvidenceRow {
        name: name.to_string(),
        class,
        element_count: record.element_count,
        init_norm: record.init_norm,
        delta_norm: record.delta_norm,
        delta_support_count: record.delta_support_count,
        delta_support_fraction: fraction,
        denom_used,
        scale_floor_used,
        relative_delta: relative_delta(record, class),
        moved: moved(record),
        grad_norm_max: record.grad_norm_max,
        grad_norm_mean: record.grad_norm_mean,
        steps_observed: record.steps_observed,
    })
}

// ===========================================================================================
// The summary (D-12)
// ===========================================================================================

/// The verdict. `Unjudged` is the only arm plan 03-05 can produce.
///
/// `Pass`/`Fail` arrive in plan 03-06, together with the frozen per-class epsilon they
/// compare against. Shipping them now would invite a comparison against a threshold that does
/// not exist yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Verdict {
    /// No threshold has been frozen, so no verdict is available.
    ///
    /// Still reachable: `EvidenceSummary::of` builds an UNJUDGED summary, and only the gate
    /// promotes it. A summary that never reached the gate must not claim a verdict.
    Unjudged,
    /// Every gated parameter cleared its contracted epsilon.
    Pass,
    /// At least one gated parameter missed it, or the run-level floor was not met.
    Fail,
}

/// Min / median / worst relative delta for one class.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClassStats {
    /// Smallest relative delta in the class.
    pub(crate) min: f64,
    /// Median relative delta in the class.
    pub(crate) median: f64,
    /// Largest relative delta in the class. "Worst" in the D-12 sense: furthest from frozen.
    pub(crate) worst: f64,
    /// Rows in the class.
    pub(crate) count: usize,
    /// Whether EVERY row of the class satisfies the strict `||dTheta|| > 0` predicate.
    pub(crate) all_moved: bool,
}

/// The D-12 summary, BOUND to its table by [`Self::table_hash`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceSummary {
    /// Wire schema version.
    pub(crate) schema_version: u32,
    /// The verdict. `Unjudged` in this plan.
    pub(crate) verdict: Verdict,
    /// Trainable parameters after `apply_freeze`.
    pub(crate) trainable_count: usize,
    /// Frozen parameters after `apply_freeze`.
    pub(crate) frozen_count: usize,
    /// Per-class statistics, in class order.
    pub(crate) per_class: BTreeMap<String, ClassStats>,
    /// The name of the row with the SMALLEST relative delta, over EVERY row in the table.
    ///
    /// # It is NOT "the parameter the gate would reject first"
    ///
    /// The table records ungated classes too — `attention_key_bias` above all, whose gradient
    /// is analytically zero, so its movement is f32 cancellation residue and is very nearly
    /// always the smallest number here. The gate never judges it. The parameter a rejection
    /// actually blames is `SetFitTrainError::EvidenceRejected::worst`, which is chosen among
    /// the GATED rows and by margin against that class's own epsilon.
    ///
    /// This field stays table-wide on purpose: `EvidenceSummary::of` is built without a
    /// `Thresholds`, and giving `ParameterClass` its own `gated` predicate would put the
    /// gated/ungated decision in two places — the failure the frozen table exists to prevent.
    pub(crate) worst_param_name: String,
    /// The frozen epsilon, when one exists. `None` while unjudged.
    pub(crate) epsilon_used: Option<f64>,
    /// The regime this evidence was measured under.
    pub(crate) calibration_regime_id: String,
    /// The contract this evidence discharges.
    pub(crate) contract_version: String,
    /// Hex SHA-256 of the full canonical table, BINDING this summary to it.
    pub(crate) table_hash: String,
}

impl EvidenceSummary {
    /// Summarize a table.
    ///
    /// # Errors
    ///
    /// [`EvidenceError::Serialization`] from the binding hash.
    pub(crate) fn of(
        evidence: &UpdateEvidence,
        trainable_count: usize,
        frozen_count: usize,
    ) -> Result<Self, EvidenceError> {
        let mut per_class = BTreeMap::new();
        for class in ParameterClass::ALL {
            let rows = evidence.rows_of_class(class);
            if rows.is_empty() {
                continue;
            }
            let values: Vec<f64> = rows.iter().map(|r| r.relative_delta).collect();
            per_class.insert(
                class.tag().to_string(),
                ClassStats {
                    min: min_of(&values),
                    median: median_of(&values),
                    worst: max_of(&values),
                    count: rows.len(),
                    all_moved: rows.iter().all(|r| r.moved),
                },
            );
        }
        let worst_param_name = evidence
            .rows
            .values()
            .min_by(|a, b| {
                a.relative_delta
                    .partial_cmp(&b.relative_delta)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.name.cmp(&b.name))
            })
            .map_or_else(String::new, |r| r.name.clone());

        Ok(Self {
            schema_version: EVIDENCE_SCHEMA_VERSION,
            verdict: Verdict::Unjudged,
            trainable_count,
            frozen_count,
            per_class,
            worst_param_name,
            epsilon_used: None,
            calibration_regime_id: evidence.calibration_regime_id.clone(),
            contract_version: EVIDENCE_CONTRACT_VERSION.to_string(),
            table_hash: hex::encode(evidence.table_hash()?),
        })
    }
}

// ===========================================================================================
// Fixed-order statistics
// ===========================================================================================

/// Smallest value, or `0.0` for an empty slice.
fn min_of(values: &[f64]) -> f64 {
    // An empty slice must not report +inf, which would serialize as `null`.
    if values.is_empty() {
        return 0.0;
    }
    values.iter().copied().fold(f64::INFINITY, f64::min)
}

/// Largest value, or `0.0` for an empty slice.
fn max_of(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

/// Median, in a FIXED order: sort ascending, then the middle, averaging the two middles for an
/// even count. `0.0` for an empty slice.
///
/// `sort_by` with a total comparator rather than `partial_cmp().unwrap()`: every value here is
/// finite by construction, and a comparator that panics on the one input that violates that
/// assumption turns a measurement into a crash.
///
/// # The even-count average stays in f64
///
/// An earlier form routed the two middle values through `reduce::sum_in_index_order`, which
/// takes `&[f32]` — so it NARROWED two `f64` relative deltas to `f32` before averaging them.
/// That is the exact defect `reduce`'s own module doc exists to prevent: a relative delta
/// below `f32::MIN_POSITIVE` (reachable for a near-frozen parameter, since the denominator is
/// floored at 1.0) flushes to zero, and the median that the run-level embedding floor is
/// compared against would report no movement for a parameter that moved. The two middles are
/// added in `f64` here, in index order, which is what the reduction discipline asks for at
/// this width.
fn median_of(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use aprender::setfit::FreezeGroup;

    use crate::train::setfit::test_fixtures as fx;
    use crate::train::setfit::thresholds::{RegimeThresholds, Thresholds};

    /// The frozen table for the regime a run ACTUALLY executed in.
    ///
    /// Since plan 05-03 TWO regimes are calibrated, so a regime-less threshold read has no
    /// answer: the fixture slice's 97-row-vocabulary epsilons and the production encoder's
    /// 30522-row ones are different measurements, and answering with "the first" is exactly the
    /// non-transfer D-10(c) forbids. Plan 05-02 made that unexpressible by deleting the
    /// regime-less accessors; this helper is how the fixture-scale assertions below say WHICH
    /// regime they mean — by naming the id the run itself recorded, never a list position.
    fn table_of_run<'a>(frozen: &'a Thresholds, regime_id: &str) -> &'a RegimeThresholds {
        frozen.table_for(regime_id).unwrap_or_else(|| {
            panic!(
                "the run's recorded regime `{regime_id}` resolves to no calibrated table; this \
                 assertion is about a run that must be INSIDE a calibrated regime",
            )
        })
    }
    use crate::train::setfit::tune::{run_tuning, validate_evidence};
    use crate::train::setfit::{EncoderTuned, SetFitRun, SetFitTrainError};

    // Imports the PRODUCTION calibration harness (plan 05-01) needs and the fixture matrix
    // does not: the production checkout's public constructor, and the Phase 2 ingest ladder
    // used to build a 64-shot-capable corpus (the committed fixture pool tops out at 16).
    use aprender::setfit::SetFitMiniLm;
    use aprender_contrastive_data::ledger::AccessLedger;
    use aprender_contrastive_data::pairs::PairConfig;
    use aprender_contrastive_data::prepared::{Canonical, CanonicalDeclarations, PreparedDataset};
    use aprender_contrastive_data::schema::LabeledExample;
    use aprender_contrastive_data::select::{FewShotSelector, SelectionConfig};
    use aprender_contrastive_data::split::SplitDeclaration;

    use crate::train::setfit::config::{SetFitTrainConfig, SetFitTrainRequest};

    /// Every name the fixture encoder emits, with its expected class. A CASE TABLE, not a
    /// spot check: a mapping tested on three names is a mapping that has not been tested.
    const CASE_TABLE: [(&str, ParameterClass); 18] = [
        ("embeddings.word_embeddings.weight", ParameterClass::Embedding),
        ("embeddings.position_embeddings.weight", ParameterClass::Embedding),
        ("embeddings.token_type_embeddings.weight", ParameterClass::Embedding),
        ("embeddings.LayerNorm.weight", ParameterClass::LayerNormWeight),
        ("embeddings.LayerNorm.bias", ParameterClass::LayerNormBias),
        ("encoder.layer.0.attention.self.query.weight", ParameterClass::ProjectionWeight),
        ("encoder.layer.0.attention.self.key.bias", ParameterClass::AttentionKeyBias),
        ("encoder.layer.1.attention.self.key.bias", ParameterClass::AttentionKeyBias),
        // The BOUNDARY of the split, tested from both sides: the key WEIGHT is an ordinary
        // projection weight, and the query/value biases are ordinary projection biases. Only
        // the key BIAS is gradient-free.
        ("encoder.layer.0.attention.self.key.weight", ParameterClass::ProjectionWeight),
        ("encoder.layer.0.attention.self.query.bias", ParameterClass::ProjectionBias),
        ("encoder.layer.0.attention.self.value.bias", ParameterClass::ProjectionBias),
        ("encoder.layer.0.attention.self.value.weight", ParameterClass::ProjectionWeight),
        ("encoder.layer.0.attention.output.dense.weight", ParameterClass::ProjectionWeight),
        ("encoder.layer.0.attention.output.LayerNorm.bias", ParameterClass::LayerNormBias),
        ("encoder.layer.1.intermediate.dense.weight", ParameterClass::ProjectionWeight),
        ("encoder.layer.1.intermediate.dense.bias", ParameterClass::ProjectionBias),
        ("encoder.layer.1.output.dense.weight", ParameterClass::ProjectionWeight),
        ("encoder.layer.1.output.LayerNorm.weight", ParameterClass::LayerNormWeight),
    ];

    fn record(init_norm: f64, support_norm: f64, delta_norm: f64, support: u64) -> ParamRecord {
        ParamRecord {
            element_count: 100,
            init_norm,
            init_norm_on_support: support_norm,
            delta_norm,
            delta_support_count: support,
            grad_norm_max: 0.0,
            grad_norm_mean: 0.0,
            steps_observed: 0,
        }
    }

    #[test]
    fn evidence_classify_parameter_case_table() {
        for (name, expected) in CASE_TABLE {
            assert_eq!(classify_parameter(name), Ok(expected), "case table row `{name}`",);
        }
    }

    /// Every name the fixture ACTUALLY emits is covered by the case table AND classifiable.
    ///
    /// Without this the case table is a list of names somebody typed, which can drift from the
    /// encoder's real naming without anything turning red.
    #[test]
    fn evidence_case_table_covers_every_name_the_fixture_emits() {
        let mut encoder = fx::slice_encoder(fx::FIXTURE_SEED);
        let names: Vec<String> =
            encoder.trainable_parameters_mut().into_iter().map(|(n, _)| n).collect();
        assert_eq!(names.len(), 37);
        for name in &names {
            classify_parameter(name)
                .unwrap_or_else(|e| panic!("the fixture emits an unclassifiable name: {e}"));
        }
        // And every SHAPE of name in the table really occurs: no table row is fiction.
        for (name, _) in CASE_TABLE {
            assert!(
                names.iter().any(|n| n == name),
                "case-table row `{name}` is not a name the fixture emits",
            );
        }
    }

    /// An unmatched name is a typed error, NOT a default bucket.
    #[test]
    fn evidence_unclassified_parameter_fails_closed() {
        for name in [
            "",
            "classifier.weight",
            "encoder.layer.0.attention.self.query.gamma",
            "embeddings.word_embeddings.bias",
            "pooler.dense",
        ] {
            match classify_parameter(name) {
                Err(EvidenceError::UnclassifiedParameter { name: reported }) => {
                    assert_eq!(reported, name);
                }
                other => panic!("`{name}` must be unclassified, got {other:?}"),
            }
        }
        // Non-vacuity: a neighbouring valid name IS classified, so the rejections above are
        // not "everything is rejected".
        assert!(classify_parameter("pooler.dense.weight").is_ok());
    }

    /// THE zero-init proof: a parameter whose initial tensor is exactly zero has a FINITE,
    /// non-NaN relative delta.
    #[test]
    // The 0.0/0.0 below is the SUBJECT of this test, not an accident: it demonstrates the
    // NaN the un-floored ratio produces and that `NaN > eps` is false. Computing it any
    // other way would stop demonstrating the artifact.
    // Likewise the negated comparison: `!(NaN > eps)` IS the incomparability being
    // demonstrated. `partial_cmp` would state it in a form that no longer shows the artifact
    // the un-floored ratio produces.
    #[allow(clippy::zero_divided_by_zero, clippy::neg_cmp_op_on_partial_ord)]
    fn evidence_relative_delta_is_finite_at_zero_initialization() {
        let zero_init = record(0.0, 0.0, 0.25, 40);
        for class in ParameterClass::ALL {
            let value = relative_delta(&zero_init, class);
            assert!(
                value.is_finite(),
                "{class}: relative delta must be finite at zero init, got {value}",
            );
            assert!(!value.is_nan(), "{class}: and not NaN");
            assert_eq!(
                value,
                0.25 / class.scale_floor(),
                "{class}: the floor must be what divides",
            );
        }

        // The un-floored form has TWO failure modes at zero init, and they fail in OPPOSITE
        // directions. Both are demonstrated here rather than asserted from memory, because
        // the plan's prose named only one of them and named it as the only one.
        //
        // (1) delta > 0, init == 0  ->  +inf, which is GREATER than every threshold. A
        //     parameter that crawled 1e-45 away from zero would pass the gate outright.
        let moved_from_zero = zero_init.delta_norm / zero_init.init_norm;
        assert!(moved_from_zero.is_infinite());
        assert!(
            moved_from_zero > 1e-3,
            "the un-floored form ACCEPTS any movement from a zero init, however small",
        );
        // (2) delta == 0, init == 0  ->  NaN, and `NaN > eps` is FALSE. A parameter that did
        //     not move is rejected — correctly, but for a reason the diagnosis cannot state,
        //     and by the same expression that wrongly accepted case (1).
        let never_moved = 0.0_f64 / 0.0_f64;
        assert!(never_moved.is_nan());
        assert!(
            !(never_moved > 1e-3),
            "NaN > eps is FALSE, so the un-floored form's rejection is a NaN artifact",
        );

        // The floored form gives a finite, ORDERED answer in both cases, and the strict
        // predicate — not the ratio — is what distinguishes them.
        let floored_moved = relative_delta(&zero_init, ParameterClass::LayerNormBias);
        let floored_still =
            relative_delta(&record(0.0, 0.0, 0.0, 0), ParameterClass::LayerNormBias);
        assert!(floored_moved.is_finite() && floored_still.is_finite());
        assert!(floored_moved > floored_still);
        assert!(moved(&zero_init));
        assert!(!moved(&record(0.0, 0.0, 0.0, 0)));
    }

    /// The companion: a zeroed AND unchanged parameter fails the strict predicate.
    #[test]
    fn evidence_strict_predicate_rejects_a_parameter_that_did_not_move() {
        let frozen = record(0.0, 0.0, 0.0, 0);
        assert!(!moved(&frozen), "a zero delta is not movement");
        assert!(
            relative_delta(&frozen, ParameterClass::LayerNormBias).is_finite(),
            "and its ratio is still finite, which is exactly why the two predicates differ",
        );
        assert_eq!(relative_delta(&frozen, ParameterClass::LayerNormBias), 0.0);

        let moved_a_little = record(0.0, 0.0, f64::MIN_POSITIVE, 1);
        assert!(moved(&moved_a_little));
    }

    /// The sparse denominator uses ONLY the rows the delta touched.
    #[test]
    fn evidence_sparse_denominator_is_restricted_to_the_support() {
        // A 2-of-N support: the whole-table norm is 100.0, the support-restricted norm 3.0.
        let sparse = record(100.0, 3.0, 6.0, 2);
        let (denom, floor) = denominator_of(&sparse, ParameterClass::Embedding);
        assert_eq!(denom, 3.0, "the sparse denominator must be support-restricted");
        assert_eq!(floor, 1.0);
        assert_eq!(relative_delta(&sparse, ParameterClass::Embedding), 2.0);

        // Every dense class uses the whole tensor.
        for class in ParameterClass::ALL.into_iter().filter(|c| !c.is_sparse()) {
            let (denom, _) = denominator_of(&sparse, class);
            assert_eq!(denom, 100.0, "{class} must use the whole initial tensor");
        }
    }

    /// Vocabulary-size invariance, which is the transfer argument in one assertion.
    ///
    /// Two embedding tables with the SAME touched rows and the SAME delta, differing only in
    /// how many untouched rows they carry, must produce the SAME relative delta.
    #[test]
    fn evidence_sparse_relative_delta_is_invariant_to_vocabulary_size() {
        let small_vocab = record(10.0, 3.0, 6.0, 2);
        let mut large_vocab = record(500.0, 3.0, 6.0, 2);
        large_vocab.element_count = 1_000_000;

        assert_eq!(
            relative_delta(&small_vocab, ParameterClass::Embedding),
            relative_delta(&large_vocab, ParameterClass::Embedding),
            "the sparse ratio must not depend on the untouched rows",
        );
        // Control: the whole-table form DOES depend on them, so the invariance above is a
        // property of the support restriction and not of the numbers chosen.
        assert_ne!(
            small_vocab.delta_norm / small_vocab.init_norm,
            large_vocab.delta_norm / large_vocab.init_norm,
        );
    }

    /// Canonical bytes are bitwise stable across two identical runs.
    #[test]
    fn evidence_canonical_bytes_are_stable_across_two_runs() {
        let a = evidence_for(fx::default_variant(), None);
        let b = evidence_for(fx::default_variant(), None);
        assert_eq!(
            a.to_canonical_bytes().expect("bytes"),
            b.to_canonical_bytes().expect("bytes"),
            "two identical runs must serialize identically",
        );
        assert_eq!(a.table_hash().expect("hash"), b.table_hash().expect("hash"));
        assert_eq!(a.rows.len(), 37, "non-vacuity: the table has rows");
    }

    /// A single-field mutation changes `table_hash`, and the summary detects the divergence.
    #[test]
    fn evidence_hash_binds_the_summary_to_the_table() {
        let evidence = evidence_for(fx::default_variant(), None);
        let summary = EvidenceSummary::of(&evidence, 37, 0).expect("summary");
        assert_eq!(summary.table_hash, hex::encode(evidence.table_hash().expect("h")));

        let mut tampered = evidence.clone();
        let key = tampered.rows.keys().next().cloned().expect("the table has rows");
        let row = tampered.rows.get_mut(&key).expect("row");
        row.relative_delta += 1.0;

        assert_ne!(
            evidence.table_hash().expect("h"),
            tampered.table_hash().expect("h"),
            "a single-field mutation must change the table hash",
        );
        assert_ne!(
            summary.table_hash,
            hex::encode(tampered.table_hash().expect("h")),
            "so the untouched summary no longer matches the edited table",
        );
    }

    /// `deny_unknown_fields` rejects an extended payload — verified with a SUBSTITUTION that
    /// is asserted to have applied, so this cannot become a test of the valid payload.
    #[test]
    fn evidence_deny_unknown_fields_rejects_an_extended_payload() {
        let evidence = evidence_for(fx::default_variant(), None);
        let json = String::from_utf8(evidence.to_canonical_bytes().expect("bytes")).expect("utf8");

        serde_json::from_str::<UpdateEvidence>(&json).expect("the valid payload must round-trip");

        let extended =
            json.replacen("{\"schema_version\":", "{\"injected\":1,\"schema_version\":", 1);
        assert_ne!(extended, json, "the substitution must have applied");
        assert!(
            serde_json::from_str::<UpdateEvidence>(&extended).is_err(),
            "an unknown field must be rejected",
        );
    }

    /// No wall-clock field survives into the wire form.
    #[test]
    fn evidence_wire_form_carries_no_wall_clock_field() {
        let evidence = evidence_for(fx::default_variant(), None);
        let json = String::from_utf8(evidence.to_canonical_bytes().expect("bytes")).expect("utf8");
        for banned in ["elapsed", "seconds", "nanos", "millis", "timestamp", "_at\""] {
            assert!(!json.contains(banned), "the evidence wire form must not contain `{banned}`",);
        }
        let summary = EvidenceSummary::of(&evidence, 37, 0).expect("summary");
        let summary_json = serde_json::to_string(&summary).expect("summary json");
        for banned in ["elapsed", "seconds", "nanos", "millis", "timestamp"] {
            assert!(!summary_json.contains(banned), "summary contains `{banned}`");
        }
    }

    /// A non-finite measurement is REFUSED at canonical-bytes time (REVIEW CR-03).
    ///
    /// The premise, measured directly rather than assumed: `serde_json` renders `+inf`, `-inf`
    /// and every `NaN` as `null` — three different divergences, one byte string — and
    /// `from_str::<f64>("null")` is `Err("invalid type: null, expected f64")`. So before this
    /// guard existed, `table_hash` was NOT injective over exactly the values a diverged run
    /// produces, and the bundle sealed with that digest could not be reloaded.
    ///
    /// A case table over all three values and over both structs, because a guard tested on one
    /// non-finite value in one field is a guard that has not been tested.
    #[test]
    fn evidence_refuses_a_non_finite_measurement_in_either_struct() {
        // The premise itself, so this test does not rest on a claim made in prose.
        for probe in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            assert_eq!(
                serde_json::to_string(&probe).expect("serde renders any f64"),
                "null",
                "the whole defect rests on this rendering; if it ever changes, revisit the guard",
            );
        }

        for poison in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            // (a) a row-level field, nested under the rows map.
            let mut evidence = evidence_for(fx::default_variant(), None);
            let row_name = evidence.rows.keys().next().expect("the fixture table has rows").clone();
            evidence.rows.get_mut(&row_name).expect("row present").relative_delta = poison;
            match evidence.to_canonical_bytes() {
                Err(EvidenceError::NonFiniteMeasurement { field }) => {
                    assert!(
                        field.contains(&row_name) && field.ends_with("relative_delta"),
                        "the path must name the offending field, got `{field}`",
                    );
                }
                other => panic!("expected NonFiniteMeasurement for {poison:?}, got {other:?}"),
            }

            // (b) a top-level field of UpdateEvidence itself.
            let mut evidence = evidence_for(fx::default_variant(), None);
            evidence.pre_clip_norm_max = poison;
            match evidence.to_canonical_bytes() {
                Err(EvidenceError::NonFiniteMeasurement { field }) => {
                    assert_eq!(field, "pre_clip_norm_max");
                }
                other => panic!("expected NonFiniteMeasurement for {poison:?}, got {other:?}"),
            }

            // (c) and table_hash, the actual consumer, must fail too rather than hash a null.
            let mut evidence = evidence_for(fx::default_variant(), None);
            evidence.pre_clip_norm_max = poison;
            assert!(
                evidence.table_hash().is_err(),
                "table_hash must not produce a digest over a null",
            );
        }
    }

    /// The control: an untouched fixture table serializes and hashes.
    ///
    /// Without this, the test above could pass because `to_canonical_bytes` rejects everything.
    #[test]
    fn evidence_control_finite_table_still_serializes_and_hashes() {
        let evidence = evidence_for(fx::default_variant(), None);
        let bytes = evidence.to_canonical_bytes().expect("a finite table must serialize");
        assert!(!bytes.is_empty(), "the canonical bytes are non-empty");
        evidence.table_hash().expect("a finite table must hash");
    }

    /// The digests are MOVED from the recorded output, not recomputed.
    #[test]
    fn evidence_digests_come_from_the_recorded_output() {
        let out = tune_output(fx::default_variant(), None);
        let evidence = UpdateEvidence::from_tune_output(&out, "test").expect("evidence");
        assert_eq!(evidence.consumed_pair_digest, hex::encode(out.consumed_pair_digest));
        assert_eq!(evidence.batch_boundary_digest, hex::encode(out.batch_boundary_digest));
        assert_eq!(evidence.loss_trace_hash, hex::encode(out.loss_trace_hash));
        assert_eq!(evidence.parameter_registry_hash, hex::encode(out.parameter_registry_hash));
        assert_eq!(evidence.batch_boundary_list, out.batch_boundaries);
        assert_eq!(evidence.step_count, out.step_count);
        assert!(
            !evidence.batch_boundary_list.is_empty(),
            "03-08's batch_boundaries() accessor needs a recorded source",
        );
    }

    /// The split is a MECHANISM, and this is the measurement that pins it.
    ///
    /// `AttentionKeyBias` exists because softmax is invariant to a constant shift of its
    /// inputs, so the key bias contributes the same amount to every key's pre-softmax logit
    /// for a given query and `dL/db_k = 0` in exact arithmetic. If that reasoning is right,
    /// the key bias's gradient must be orders of magnitude below its OWN BLOCK's query bias,
    /// which has no such invariance since `(q + b_q) . k_j` varies with `j`.
    ///
    /// This runs in every `cargo test`, not behind `--ignored`, because it is the load-bearing
    /// justification for a class boundary that removes parameters from the gate. If a future
    /// encoder, kernel or reduction order makes the key bias gradient real, the split loses
    /// its basis and this test says so instead of the boundary quietly becoming folklore.
    #[test]
    fn evidence_attention_key_bias_is_gradient_free_relative_to_its_own_block() {
        let evidence = evidence_for(fx::default_variant(), None);

        let key_rows = evidence.rows_of_class(ParameterClass::AttentionKeyBias);
        let bias_rows = evidence.rows_of_class(ParameterClass::ProjectionBias);
        assert!(!key_rows.is_empty(), "the fixture must emit key biases");
        assert!(!bias_rows.is_empty(), "and ordinary projection biases to compare against");

        let key_worst = max_of(&key_rows.iter().map(|r| r.grad_norm_max).collect::<Vec<f64>>());
        let bias_best = min_of(&bias_rows.iter().map(|r| r.grad_norm_max).collect::<Vec<f64>>());

        // Non-vacuity FIRST: an all-zero comparison would satisfy any ratio.
        assert!(bias_best > 0.0, "the comparison class must have real gradients");
        assert!(key_worst > 0.0, "and the key bias must have a measurable residue, not a hole");

        assert!(
            key_worst * 1e4 < bias_best,
            "the key bias gradient ({key_worst:e}) is not >=1e4x below the smallest ordinary \
             projection-bias gradient ({bias_best:e}); the shift-invariance argument the \
             AttentionKeyBias split rests on no longer holds and the class must be re-derived",
        );

        // The OTHER half of the boundary: only the key BIAS is split out. The key WEIGHT and
        // the query/value biases stay ordinary, so the split cannot silently widen into
        // "attention parameters are exempt".
        assert_eq!(
            classify_parameter("encoder.layer.0.attention.self.key.weight"),
            Ok(ParameterClass::ProjectionWeight),
        );
        assert_eq!(
            classify_parameter("encoder.layer.0.attention.self.query.bias"),
            Ok(ParameterClass::ProjectionBias),
        );
        assert_eq!(
            classify_parameter("encoder.layer.0.attention.self.value.bias"),
            Ok(ParameterClass::ProjectionBias),
        );
    }

    /// The summary is `Unjudged` and carries no epsilon in this plan.
    #[test]
    fn evidence_summary_is_unjudged_with_no_epsilon() {
        let evidence = evidence_for(fx::default_variant(), None);
        let summary = EvidenceSummary::of(&evidence, 37, 0).expect("summary");
        assert_eq!(summary.verdict, Verdict::Unjudged);
        assert_eq!(summary.epsilon_used, None);
        assert_eq!(summary.contract_version, EVIDENCE_CONTRACT_VERSION);
        assert_eq!(
            summary.per_class.len(),
            ParameterClass::ALL.len(),
            "every class the mapping can emit must be populated by the fixture; a class with \
             no rows would have its epsilon frozen against nothing",
        );
        assert_eq!(ParameterClass::ALL.len(), 6, "five classes plus the gradient-free split");
        assert!(!summary.worst_param_name.is_empty());
        for (class, stats) in &summary.per_class {
            assert!(stats.count > 0, "{class}");
            assert!(stats.min <= stats.median, "{class}");
            assert!(stats.median <= stats.worst, "{class}");
        }
    }

    /// Fixed-order statistics behave on the edge cases the table can present.
    #[test]
    fn evidence_statistics_handle_empty_and_even_inputs() {
        assert_eq!(min_of(&[]), 0.0);
        assert_eq!(max_of(&[]), 0.0);
        assert_eq!(median_of(&[]), 0.0);
        assert_eq!(min_of(&[3.0, 1.0, 2.0]), 1.0);
        assert_eq!(max_of(&[3.0, 1.0, 2.0]), 3.0);
        assert_eq!(median_of(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median_of(&[4.0, 1.0, 3.0, 2.0]), 2.5);
        assert!(min_of(&[]).is_finite(), "an empty min must serialize as a number");
    }

    // -----------------------------------------------------------------------------------
    // The armed gate (plan 03-06) — negative / control / mirror
    //
    // Ph1 D-24 / Ph2 D-25 discipline: each negative is REJECTED and its message NAMES the
    // offender, a CONTROL with the same setup minus the poison PASSES, and a MIRROR shows the
    // untouched path is unchanged by the gate's presence. All of them run in every `cargo
    // test` — a gate that is only ever exercised on the honest path is not evidence.
    // -----------------------------------------------------------------------------------

    /// The reference-defaults control run, built once and reused.
    ///
    /// A full `tune_encoder` pass per test would repeat the fixture load and the tuning loop
    /// several times over for no additional evidence.
    ///
    /// `calibrated_variant` rather than `default_variant`: everything below reaches a
    /// THRESHOLD comparison, and the regime check runs first, so these tests need a run at a
    /// seed and cell the epsilons were actually measured at. `default_variant` sits on
    /// `FIXTURE_SEED`, which the calibration never swept — see the seed-negative in `mod.rs`.
    fn control_run() -> Result<SetFitRun<EncoderTuned>, SetFitTrainError> {
        fx::prepared_run(fx::calibrated_variant(), None).tune_encoder()
    }

    /// CONTROL: a reference-defaults fixture run passes the gate and mints `EncoderTuned`.
    ///
    /// Without this the three negatives below would be satisfied by a gate that rejects
    /// everything, which is the failure mode that makes a rejection-only test suite worthless.
    #[test]
    fn negative_control_reference_run_passes_the_gate() {
        let run = control_run().expect("the reference-defaults fixture run must pass the gate");
        let passed = run.evidence();
        let summary = passed.summary();

        assert_eq!(summary.verdict, Verdict::Pass, "the control's verdict must be Pass");
        assert_eq!(summary.contract_version, EVIDENCE_CONTRACT_VERSION);
        assert!(
            Thresholds::frozen().is_calibrated(&summary.calibration_regime_id),
            "the control must run inside the calibrated regime, got `{}`",
            summary.calibration_regime_id,
        );

        // The per-class epsilon actually applied is the CONTRACT's, not a local literal.
        let frozen = Thresholds::frozen();
        let table = table_of_run(&frozen, &summary.calibration_regime_id);
        for class in ParameterClass::ALL {
            let entry = table.of(class);
            let Some(eps) = entry.eps else { continue };
            for row in passed.table().rows_of_class(class) {
                assert!(
                    row.relative_delta > eps,
                    "{}: {} passed the gate at relative delta {:e} which does not exceed the \
                     contracted epsilon {eps:e}",
                    class.tag(),
                    row.name,
                    row.relative_delta,
                );
            }
        }

        // The summary is BOUND to the table it summarizes.
        assert_eq!(
            summary.table_hash,
            hex::encode(passed.table().table_hash().expect("table hash")),
            "the summary must bind to its own table",
        );
        assert_eq!(summary.trainable_count, passed.table().rows.len());
    }

    /// NEGATIVE 1 — an all-frozen run cannot pass by being un-checkable (SAFE-03, D-09).
    #[test]
    fn negative_all_frozen_run_has_no_trainable_parameters() {
        // Calibrated coordinates on purpose: the regime check runs BEFORE the empty-trainable-set
        // check, so an uncalibrated fixture here would be refused for the wrong reason and this
        // test would stop being about SAFE-03 at all.
        let variant = fx::calibrated_variant();
        // Every group of every layer, plus the embeddings: the complete freeze.
        let mut policy = vec![FreezeGroup::Embeddings];
        for layer in 0..fx::slice_encoder(variant.root_seed).num_layers() {
            policy.push(FreezeGroup::LayerAttention(layer));
            policy.push(FreezeGroup::LayerFfn(layer));
            policy.push(FreezeGroup::LayerNorm(layer));
        }
        let run = fx::prepared_run_with_freeze(variant, policy);
        match run.tune_encoder() {
            Err(SetFitTrainError::NoTrainableParameters { trainable_count }) => {
                assert_eq!(trainable_count, 0, "the message must report the observed count");
                let rendered =
                    SetFitTrainError::NoTrainableParameters { trainable_count }.to_string();
                assert!(rendered.contains("trainable_count 0"), "rendered: {rendered}");
                assert!(rendered.contains("SAFE-03"), "the diagnosis must name what it enforces");
            }
            other => panic!("an all-frozen run must be rejected as unpassable, got {other:?}"),
        }
    }

    /// NEGATIVE 2 — a 1e-30-learning-rate run is rejected and the message NAMES the offender.
    #[test]
    fn negative_null_learning_rate_run_is_rejected_naming_the_offender() {
        // Calibrated coordinates: this test is about the EPSILON comparison, which only runs
        // once the regime check has passed.
        let run = fx::prepared_run(fx::calibrated_variant(), Some(1e-30));
        match run.tune_encoder() {
            Err(SetFitTrainError::EvidenceRejected { worst, summary, table }) => {
                // The offender is named by its DOTTED HF name, not an index.
                assert!(
                    worst.name.contains('.'),
                    "the offender must be named by its dotted HF name, got `{}`",
                    worst.name,
                );
                // Its class is one the contract actually gates.
                let frozen = Thresholds::frozen();
                // NOT `table`: that name is already bound by the match arm to the run's
                // UpdateEvidence. This is the THRESHOLD table for the regime the run recorded.
                let regime_table = table_of_run(&frozen, &summary.calibration_regime_id);
                let class = ParameterClass::ALL
                    .into_iter()
                    .find(|c| c.tag() == worst.class)
                    .unwrap_or_else(|| panic!("unknown class `{}`", worst.class));
                assert!(regime_table.of(class).gated, "an ungated class must never be blamed");

                // The measured delta and the CONTRACTED epsilon are both present and consistent.
                assert_eq!(
                    Some(worst.eps),
                    regime_table.of(class).eps,
                    "the blamed epsilon must be the contract's value for that class",
                );
                assert!(
                    worst.relative_delta <= worst.eps,
                    "the offender must actually have missed its threshold: {:e} vs {:e}",
                    worst.relative_delta,
                    worst.eps,
                );
                // At 1e-30 every update underflows the parameter ULP, so the delta is exactly 0.
                assert_eq!(worst.relative_delta, 0.0, "a 1e-30 run moves nothing at f32 scale");

                // AUDITABLE FAILURE: the COMPLETE record travels inside the error.
                assert_eq!(
                    table.rows.len(),
                    summary.trainable_count,
                    "the failed table must carry ONE ROW PER TRAINABLE PARAMETER so a rejection \
                     can be investigated rather than merely reported",
                );
                assert!(table.rows.len() > 1, "non-vacuity: the fixture has many parameters");
                assert!(table.rows.contains_key(&worst.name), "the offender must be IN the table");
                assert_eq!(summary.verdict, Verdict::Fail);
                // Run-level facts survive the rejection too.
                assert!(!table.loss_trace_hash.is_empty());
                assert!(!table.consumed_pair_digest.is_empty());
                assert!(!table.batch_boundary_list.is_empty());

                let rendered =
                    SetFitTrainError::EvidenceRejected { worst, summary, table }.to_string();
                assert!(rendered.contains("REJECTED"), "rendered: {rendered}");
                assert!(rendered.contains("TRN-03"), "the diagnosis must name its requirement");
            }
            other => panic!("a 1e-30 run must be rejected by the gate, got {other:?}"),
        }
    }

    /// NEGATIVE 2b — a run in which EVERY gated parameter MOVED is still rejected, by the
    /// epsilon and by nothing else.
    ///
    /// This test exists because mutating the gate exposed a hole in the set above. Deleting
    /// the `relative_delta > eps` comparison outright left every other negative GREEN: the
    /// 1e-30 run has bit-for-bit zero deltas so the strict movement predicate catches it
    /// before the threshold is ever consulted, and even a real 1e-8 run contains some
    /// parameters that did not move at all, so it too is rejected without the epsilon.
    ///
    /// The only witness that isolates the threshold is a table in which EVERYTHING moved and
    /// everything fell short. It is built from the CONTROL's real table, with every relative
    /// delta scaled below its class epsilon and `moved` left true, so the sole reason to
    /// reject it is the comparison this test exists to protect.
    #[test]
    fn negative_a_run_that_moved_everywhere_but_fell_short_is_rejected_by_the_epsilon() {
        let out = tune_output(fx::default_variant(), None);
        let frozen = Thresholds::frozen();

        // CONTROL: unmodified, this table passes.
        let good = UpdateEvidence::from_tune_output(&out, &regime_id()).expect("evidence");
        validate_evidence(&good, &frozen, out.trainable_count, out.frozen_count)
            .expect("CONTROL: the reference table passes before it is scaled down");

        // The poison: everything still MOVED, everything now falls short.
        let mut short = good.clone();
        for row in short.rows.values_mut() {
            row.relative_delta *= 1e-3;
            assert!(row.moved, "the scaling must not disturb the movement predicate");
            assert!(row.delta_norm > 0.0);
        }
        short.embedding_delta_median *= 1e-3;

        // Non-vacuity: EVERY gated row moved, so nothing here is rejectable by `moved`.
        let fixture_table = table_of_run(&frozen, &short.calibration_regime_id);
        let gated_rows: Vec<&EvidenceRow> =
            short.rows.values().filter(|r| fixture_table.of(r.class).gated).collect();
        assert!(!gated_rows.is_empty());
        assert!(
            gated_rows.iter().all(|r| r.moved && r.delta_norm > 0.0),
            "if any gated parameter failed to move, the strict predicate could reject this \
             table and the epsilon would again go untested",
        );
        assert!(
            gated_rows.iter().all(|r| r.grad_norm_max.is_finite()),
            "and every gradient is finite, so the finiteness predicate cannot reject it either",
        );

        match validate_evidence(&short, &frozen, out.trainable_count, out.frozen_count) {
            Err(SetFitTrainError::EvidenceRejected { worst, .. }) => {
                let class = ParameterClass::ALL
                    .into_iter()
                    .find(|c| c.tag() == worst.class)
                    .unwrap_or_else(|| panic!("unknown class `{}`", worst.class));
                assert_eq!(Some(worst.eps), fixture_table.of(class).eps);
                assert!(
                    worst.relative_delta > 0.0,
                    "the blamed parameter MOVED ({:e}); only the threshold rejected it",
                    worst.relative_delta,
                );
                assert!(worst.relative_delta <= worst.eps);
            }
            other => panic!(
                "a table in which everything moved but fell short of its epsilon MUST be \
                 rejected; got {other:?}. If this returned Ok, the epsilon comparison is not \
                 doing anything and every frozen threshold in the contract is decoration.",
            ),
        }
    }

    /// NEGATIVE 3 — an out-of-regime run is refused BEFORE any threshold is compared.
    ///
    /// Driven through `validate_evidence` directly with a doctored regime id, because the point
    /// is the ORDER of the checks: the table handed in is the CONTROL's, which passes every
    /// threshold. If the regime check ran second, this table would pass and the test would be
    /// green for the wrong reason.
    #[test]
    fn negative_uncalibrated_regime_is_refused_before_any_comparison() {
        let out = tune_output(fx::default_variant(), None);
        let frozen = Thresholds::frozen();

        let good = UpdateEvidence::from_tune_output(&out, &regime_id()).expect("evidence");
        validate_evidence(&good, &frozen, out.trainable_count, out.frozen_count)
            .expect("CONTROL: this very table passes when its regime is calibrated");

        let foreign =
            "minilm-full-h384-l6-a12-i1536-v30522@production|seeds=1,42,7|cells=s16e2b8,s8e1b4";
        let bad = UpdateEvidence::from_tune_output(&out, foreign).expect("evidence");
        match validate_evidence(&bad, &frozen, out.trainable_count, out.frozen_count) {
            Err(SetFitTrainError::UncalibratedRegime { observed, calibrated }) => {
                assert_eq!(observed, foreign);
                assert_eq!(
                    calibrated.len(),
                    2,
                    "two calibrated fingerprints since plan 05-03 — the fixture slice and the \
                     production encoder — and this run matches NEITHER",
                );
                assert!(!calibrated.contains(&foreign.to_string()));
                // The point of the id above: `minilm-full-...@production` differs from the
                // calibrated production entry `minilm-slice-...@1110a243` in BOTH the prefix and
                // the revision, so calibrating the production encoder must not have made it
                // resolvable. An architecture is matched for exact equality, never by family.
                assert!(
                    calibrated.iter().all(|c| !c.starts_with("minilm-full-")),
                    "no calibrated entry may carry the `minilm-full-` rendering: {calibrated:?}",
                );
                let rendered =
                    SetFitTrainError::UncalibratedRegime { observed, calibrated }.to_string();
                assert!(rendered.contains("D-10(c)"), "rendered: {rendered}");
            }
            other => panic!("an out-of-regime run must fail closed, got {other:?}"),
        }
    }

    /// NEGATIVE 3b — a trainable set of ONLY gradient-free parameters is unpassable.
    ///
    /// The hole the `attention_key_bias` exclusion would otherwise open: freeze everything the
    /// gate checks, leave only what it cannot check, and a naive `for p in gated { .. }` loop
    /// passes vacuously because it iterates over nothing.
    #[test]
    fn negative_only_gradient_free_parameters_cannot_testify() {
        let out = tune_output(fx::default_variant(), None);
        let full = UpdateEvidence::from_tune_output(&out, &regime_id()).expect("evidence");

        let mut ungated_only = full.clone();
        ungated_only.rows.retain(|_, r| r.class == ParameterClass::AttentionKeyBias);
        let kept = ungated_only.rows.len();
        assert!(kept > 0, "non-vacuity: the fixture must emit key biases");

        match validate_evidence(&ungated_only, &Thresholds::frozen(), kept, out.frozen_count) {
            Err(SetFitTrainError::NoTestifyingParameters { trainable_count, ungated_count }) => {
                assert_eq!(trainable_count, kept);
                assert_eq!(ungated_count, kept);
            }
            other => panic!("an all-ungated trainable set must be unpassable, got {other:?}"),
        }
    }

    /// MIRROR — arming the gate changed nothing about what the loop records.
    ///
    /// Two independent passing runs agree bit-for-bit on every recorded digest. The gate READS
    /// the evidence; if it had started to influence what the loop consumed, these would diverge.
    #[test]
    fn negative_mirror_two_passing_runs_agree_on_every_digest() {
        let first = control_run().expect("first control run");
        let second = control_run().expect("second control run");
        let (a, b) = (first.evidence().table(), second.evidence().table());

        assert_eq!(a.loss_trace_hash, b.loss_trace_hash, "loss trace");
        assert_eq!(a.consumed_pair_digest, b.consumed_pair_digest, "consumed pairs");
        assert_eq!(a.batch_boundary_digest, b.batch_boundary_digest, "batch boundaries");
        assert_eq!(a.batch_boundary_list, b.batch_boundary_list, "readable boundary list");
        assert_eq!(a.parameter_registry_hash, b.parameter_registry_hash, "registry");
        assert_eq!(a.step_count, b.step_count, "step count");

        // Non-vacuity: the digests are real, not two empty strings compared to each other.
        assert_eq!(a.loss_trace_hash.len(), 64, "a SHA-256 renders as 64 hex characters");
        assert!(a.step_count > 0, "the mirror must compare runs that actually ran");

        // And the whole table hash agrees, which covers every per-parameter measurement at once.
        assert_eq!(
            a.table_hash().expect("hash"),
            b.table_hash().expect("hash"),
            "two identical runs must produce identical evidence bytes",
        );
    }

    /// `PassedEvidence` retains the run-level fields every downstream accessor resolves to.
    ///
    /// W-06: anything dropped here becomes something a later plan RECOMPUTES, which is exactly
    /// the false-green the in-band digests exist to remove.
    #[test]
    fn evidence_gate_passed_evidence_carries_the_complete_record() {
        let run = control_run().expect("control");
        let table = run.evidence().table();

        assert!(!table.loss_trace_hash.is_empty());
        assert!(!table.consumed_pair_digest.is_empty());
        assert!(!table.batch_boundary_digest.is_empty());
        assert!(!table.batch_boundary_list.is_empty(), "03-08's recorded boundary source");
        assert!(!table.parameter_registry_hash.is_empty());
        assert!(table.step_count > 0);
        assert!(table.k > 0);
        assert!(table.first_k_mean.is_finite() && table.last_k_mean.is_finite());
        assert!(table.embedding_delta_min.is_finite());
        assert!(table.embedding_delta_median > 0.0);
        assert!(table.pre_clip_norm_max.is_finite());
        assert!(!table.calibration_regime_id.is_empty());
    }

    // -----------------------------------------------------------------------------------
    // The calibration matrix (measurement, not judgment)
    // -----------------------------------------------------------------------------------

    /// A run's recorded output for one cell.
    fn tune_output(
        variant: fx::CalibrationVariant,
        encoder_lr_override: Option<f64>,
    ) -> TuneOutput {
        let (encoder, dataset, selection, config) =
            fx::prepared_run(variant, encoder_lr_override).into_parts();
        run_tuning(encoder, &dataset, &selection, &config).expect("the fixture run must tune")
    }

    /// Its evidence table.
    fn evidence_for(
        variant: fx::CalibrationVariant,
        encoder_lr_override: Option<f64>,
    ) -> UpdateEvidence {
        let out = tune_output(variant, encoder_lr_override);
        UpdateEvidence::from_tune_output(&out, &regime_id()).expect("evidence")
    }

    /// The regime this matrix was measured under: architecture, seed set, boundary set.
    fn regime_id() -> String {
        let mut seeds: Vec<String> =
            fx::calibration_variants().iter().map(|v| v.root_seed.to_string()).collect();
        seeds.sort_unstable();
        seeds.dedup();
        let mut cells: Vec<String> =
            fx::calibration_variants().iter().map(|v| v.label.to_string()).collect();
        cells.sort_unstable();
        cells.dedup();
        format!(
            "minilm-slice-h64-l2-a2-i256-v97@1110a243|seeds={}|cells={}",
            seeds.join(","),
            cells.join(","),
        )
    }

    /// The largest `relative_delta` that pure `f32` REPRESENTATION ROUNDING can produce for
    /// this row, and therefore the floor a frozen epsilon has to clear to mean anything.
    ///
    /// Rigorous rather than estimated: rounding each element to the nearest `f32` perturbs it
    /// by at most half a ULP, i.e. `|dx_i| <= (EPSILON/2)|x_i|`, so
    /// `||dTheta||_2 <= (EPSILON/2)||theta||_2` and the ratio is bounded by
    /// `(EPSILON/2) * init_norm / max(denom_used, scale_floor_used)`. Every input is a field
    /// the evidence row already records, so this is MEASURED from the run rather than assumed
    /// from a nominal parameter magnitude.
    ///
    /// Plan 03-05 flagged `projection_bias` as unfreezable because `min/10` "sits close to f32
    /// resolution" while its predicate actually tested `median/min > 100` — a spread statistic
    /// that says nothing about resolution. This is the quantity that claim is about.
    fn rounding_noise_floor(row: &EvidenceRow) -> f64 {
        let denom = row.denom_used.max(row.scale_floor_used);
        (f64::from(f32::EPSILON) / 2.0) * row.init_norm / denom
    }

    /// The learning rate of the null control. Chosen by the plan; measured below to be far
    /// under the `f32` resolution of every parameter, which is what makes it a null.
    const CONTROL_LR: f64 = 1e-30;

    /// A SECOND control, at a small but fully REPRESENTABLE learning rate.
    ///
    /// `CONTROL_LR` is a numerical null: at 1e-30 every per-element update underflows the
    /// parameter's ULP, so the measured delta is bit-for-bit zero and the matrix bounds
    /// epsilon from ABOVE only (03-05's concern 3). 1e-8 is 2000x below the reference 2e-5 —
    /// a run that is not meaningfully training — but it is large enough that AdamW writes a
    /// different `f32` back. It therefore gives a REAL lower bound: whatever a
    /// nearly-not-training run can produce, a frozen epsilon must sit above.
    ///
    /// It is also the measurement that decides what the gradient-free class may assert. If a
    /// parameter's `||dTheta|| > 0` is true at 1e-8 just as it is at 2e-5, that predicate does
    /// not distinguish training from not-training FOR THAT PARAMETER, and arming it would be
    /// the same vacuity that disqualified the pair-loss endpoint statistic.
    const NEAR_NULL_LR: f64 = 1e-8;

    // -----------------------------------------------------------------------------------
    // THE epsilon window rule — ONE site, fail-closed (D-18, plan 05-14)
    // -----------------------------------------------------------------------------------

    /// The symmetric safety factor the window rule applies to both edges.
    ///
    /// Named once because the rule used to be inline ARITHMETIC in two places — the fixture
    /// matrix's basis table and the production matrix's — which is why plan 05-01's `eps/noise`
    /// reporting defect had to be fixed twice to stop it surviving in a sibling report. Plan
    /// 05-03 revisits this factor against 05-01's evidence; the verdict below does not depend on
    /// its value.
    const WINDOW_SAFETY_FACTOR: f64 = 10.0;

    /// WHICH measured quantity the window's LOWER edge is derived from, and at what factor.
    ///
    /// The upper edge is not a degree of freedom here: it is `best_real / WINDOW_SAFETY_FACTOR`
    /// under every variant, because nothing 05-01 measured challenges the upper factor. Only the
    /// lower edge is in question, and `contracts/setfit-train-lifecycle-v1.yaml` records TWO
    /// candidate lower bounds — this enum is those two, made selectable so a candidate's window
    /// status is obtained by RUNNING [`epsilon_basis`] rather than by open-coding a second
    /// emptiness comparison beside it (plan 05-03 Task 1's structural constraint; the
    /// `lower < upper` in `epsilon_basis` stays the only one in this file).
    ///
    /// # The bound is cited; the FACTOR is chosen
    ///
    /// [`Self::NotTrainingControl`] carries a factor the contract's DERIVATION invariant states
    /// (`10 * worst_control_delta <= eps`). [`Self::RoundingNoiseFloor`] does NOT: the contract
    /// states a clearance CONDITION ("no parameter can satisfy its threshold by rounding alone")
    /// and records 49x-315x as the clearance the fixture epsilons HAPPENED TO HAVE, never as a
    /// required `lower = k * floor` formula. Any factor on the noise floor is therefore CHOSEN by
    /// the plan that selects it and must be recorded as chosen.
    #[derive(Debug, Clone, Copy, PartialEq)]
    enum LowerBound {
        /// `factor * max(worst_ctrl, worst_nnull)` — the D-03 rule, whose factor IS contracted.
        NotTrainingControl {
            /// The safety factor applied to the worst not-really-training delta.
            factor: f64,
        },
        /// `factor * noise_floor` — the contracted f32 rounding-noise clearance condition. The
        /// factor is a CHOICE, not a citation; see the type's docs.
        RoundingNoiseFloor {
            /// The safety factor applied to the class's own f32 rounding-noise floor.
            factor: f64,
        },
    }

    impl LowerBound {
        /// This rule's lower edge for one class's measured aggregates.
        fn lower(self, agg: &ClassAggregates) -> f64 {
            match self {
                Self::NotTrainingControl { factor } => agg.worst_ctrl.max(agg.worst_nnull) * factor,
                Self::RoundingNoiseFloor { factor } => agg.noise_floor * factor,
            }
        }

        /// A short label for the report's candidate table.
        fn label(self) -> String {
            match self {
                Self::NotTrainingControl { factor } => {
                    format!("{factor:.0} x max(worst_ctrl, worst_nnull)")
                }
                Self::RoundingNoiseFloor { factor } => format!("{factor:.0} x noise_floor"),
            }
        }
    }

    /// The lower bound the contract's DERIVATION invariant states, at its contracted factor.
    ///
    /// This is the rule the FIXTURE regime's five gated epsilons were frozen under, and the rule
    /// plan 05-01 measured collapsing at the production envelope. It stays the default every
    /// existing call site passes, so nothing about the fixture derivation moves.
    const CONTRACTED_NEAR_NULL_LOWER_BOUND: LowerBound =
        LowerBound::NotTrainingControl { factor: WINDOW_SAFETY_FACTOR };

    /// The lower bound the PRODUCTION regime's windows are derived under.
    ///
    /// Held as its own constant because the production and fixture regimes are separate
    /// measurements and 05-01 proved the contracted near-null bound does not survive the
    /// production envelope (five of six classes have no legal epsilon at 1536 optimizer steps).
    /// Which bound replaces it was decided at plan 05-03's D-04 checkpoint, on the candidate
    /// tables this file renders, and is recorded in the contract's `calibration_regime`
    /// invariants: the contract's OTHER lower bound, the f32 rounding-noise clearance condition.
    ///
    /// # The bound is cited; the FACTOR is chosen
    ///
    /// The contract states the clearance CONDITION and records 49x-315x as what the fixture
    /// epsilons happened to have — never a required `lower = k * floor` formula. The `10.0`
    /// below is therefore this plan's CHOICE, made for two stated reasons: it mirrors the
    /// near-null leg's own factor, and it is strictly stricter than bare clearance. The contract
    /// records it as chosen and prints the bare-condition window beside it, so a reader can see
    /// what the factor bought.
    const PRODUCTION_LOWER_BOUND: LowerBound = LowerBound::RoundingNoiseFloor { factor: 10.0 };

    /// The measured aggregates one class contributes to the basis.
    ///
    /// The first four are what the window rule consumes; the last two are what the report
    /// prints beside it. They travel together so the report cannot be rendered from one set of
    /// numbers while the verdict is computed from another.
    #[derive(Debug, Clone, Copy, Default, PartialEq)]
    struct ClassAggregates {
        /// Worst (largest) relative delta the 1e-30 control produced, across measured cells.
        worst_ctrl: f64,
        /// Worst (largest) relative delta the 1e-8 near-null control produced.
        worst_nnull: f64,
        /// Best (smallest) relative delta a real run produced — the binding upper edge.
        best_real: f64,
        /// Worst (largest) per-cell MEDIAN real delta, for the `median/min` spread column.
        worst_median: f64,
        /// The f32 rounding-noise floor a frozen epsilon has to clear.
        noise_floor: f64,
        /// Whether EVERY near-null row still satisfied the strict `||dTheta|| > 0` predicate.
        near_null_moved_all: bool,
    }

    /// Whether a class's derived window is an interval at all.
    ///
    /// STRICT: a point is not an interval. Equal bounds carry no value epsilon can take, so
    /// they are [`WindowStatus::Empty`] with an exceed factor of exactly 1.
    #[derive(Debug, Clone, Copy, PartialEq)]
    enum WindowStatus {
        /// `lower < upper` — there is a value epsilon can take.
        Legal,
        /// The lower bound reached or exceeded the upper one.
        Empty {
            /// `lower / upper`, or infinity when `upper` is zero.
            exceed_factor: f64,
        },
    }

    /// How much of the boundary matrix a basis was derived from.
    ///
    /// TYPED rather than banner prose, so a consumer cannot read a provisional basis as a
    /// freezable one. Provisional coverage is NOT a refusal: within a fixed rule `lower` is
    /// monotone non-decreasing and `upper` monotone non-increasing in the measured cells, so a
    /// collapse observed on a subset can never be cleared by measuring more, while a
    /// provisional basis genuinely can still change.
    #[derive(Debug, Clone, PartialEq)]
    enum BasisCoverage {
        /// Every cell of the boundary matrix was measured.
        Complete,
        /// A subset was measured; the absent cells are named.
        Provisional {
            /// How many cells were measured.
            measured: usize,
            /// How many the full boundary matrix has.
            total: usize,
            /// The cells that were NOT measured, in the full matrix's order.
            missing: Vec<String>,
        },
    }

    /// One class's row of the derived basis.
    #[derive(Debug, Clone, PartialEq)]
    struct BasisRow {
        /// The class this row is about.
        class: ParameterClass,
        /// The measurements it was derived from.
        aggregates: ClassAggregates,
        /// The window's lower edge.
        lower: f64,
        /// The window's upper edge.
        upper: f64,
        /// Whether the window is an interval.
        window: WindowStatus,
        /// Whether this class must carry a legal window for the run to pass.
        required: bool,
        /// Whether a frozen table RESOLVED and declares this class ungated.
        ///
        /// Distinct from `!required`: with no table resolved every class is required and NO
        /// class is declared-ungated, because there is no declaration to read.
        declared_ungated: bool,
    }

    impl BasisRow {
        /// `upper / noise_floor`, suppressed to `n/a` for an EMPTY window.
        ///
        /// That column is only meaningful while `upper` is a legal epsilon. Printing it for an
        /// empty window yields a plausible-looking margin for a class that has no epsilon at
        /// all; plan 05-01 tracked `1.51e1` for `attention_key_bias` across several reports
        /// before noticing it was meaningless. Suppressed rather than fixed up, because there
        /// is no correct value to print.
        fn eps_over_noise(&self) -> String {
            match self.window {
                WindowStatus::Empty { .. } => "n/a".to_string(),
                WindowStatus::Legal if self.aggregates.noise_floor > 0.0 => {
                    format!("{:.2e}", self.upper / self.aggregates.noise_floor)
                }
                WindowStatus::Legal => "inf".to_string(),
            }
        }

        /// The `window` column: `EXISTS`, or `EMPTY` carrying its exceed factor.
        fn window_tag(&self) -> String {
            match self.window {
                WindowStatus::Legal => "EXISTS".to_string(),
                WindowStatus::Empty { exceed_factor } => format!("EMPTY {exceed_factor:.2}x"),
            }
        }

        /// The `gating` column — the annotation that makes an exclusion visible in the artifact
        /// rather than inferable only from the absence of a failure.
        fn gating_tag(&self) -> &'static str {
            if self.declared_ungated {
                "declared-ungated"
            } else {
                "required"
            }
        }

        /// `worst_median / best_real`, the spread statistic the report already printed.
        fn spread(&self) -> f64 {
            if self.aggregates.best_real > 0.0 {
                self.aggregates.worst_median / self.aggregates.best_real
            } else {
                f64::INFINITY
            }
        }
    }

    /// A derived epsilon basis: every class's window, what the regime required, and how much of
    /// the boundary matrix it covers.
    #[derive(Debug, Clone, PartialEq)]
    struct EpsilonBasis {
        /// The regime the basis was derived FOR — the id the gated set was looked up by.
        regime_id: String,
        /// Whether a frozen table resolved for that id.
        table_resolved: bool,
        /// The classes the resolved table gates, empty when no table resolved.
        gated: Vec<&'static str>,
        /// One row per [`ParameterClass::ALL`], in that order.
        rows: Vec<BasisRow>,
        /// How much of the boundary matrix was measured.
        coverage: BasisCoverage,
        /// WHICH lower bound the rows' `lower` edges were derived under.
        ///
        /// Carried so a basis is self-describing: a reader of a rendered table can tell which
        /// rule produced it without inferring the rule from the numbers.
        lower_bound: LowerBound,
    }

    impl EpsilonBasis {
        /// One class's row. Total over [`ParameterClass::ALL`] by construction.
        fn row(&self, class: ParameterClass) -> &BasisRow {
            self.rows
                .iter()
                .find(|r| r.class == class)
                .unwrap_or_else(|| panic!("every class has a basis row; {class} did not"))
        }

        /// The line that states the gated-set lookup's OWN answer.
        ///
        /// Printed so the declared-ungated annotations are traceable to a mechanism rather than
        /// to an assumption, and so any later control can read the verdict's precondition off
        /// the run instead of proxying it from source or contract text.
        fn regime_table_line(&self) -> String {
            if self.table_resolved {
                format!(
                    "REGIME TABLE FOR THE DERIVED REGIME: resolved for `{}` — gated classes: \
                     [{}]\n",
                    self.regime_id,
                    self.gated.join(", "),
                )
            } else {
                format!(
                    "REGIME TABLE FOR THE DERIVED REGIME: absent — no frozen table covers `{}`, \
                     so EVERY class is required to carry a legal window\n",
                    self.regime_id,
                )
            }
        }
    }

    /// One collapsed class, in the typed refusal.
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct CollapsedClass {
        /// The class tag.
        class: &'static str,
        /// Its window's lower edge.
        lower: f64,
        /// Its window's upper edge.
        upper: f64,
        /// `lower / upper`.
        exceed_factor: f64,
    }

    /// The typed refusal: at least one class the regime GATES has no legal epsilon window.
    ///
    /// Carries the whole basis so the report can still be RENDERED and written on refusal — a
    /// derivation whose numbers cannot be read is worse than one that fails loudly.
    #[derive(Debug, Clone, PartialEq)]
    struct CollapsedWindows {
        /// The basis that was derived, refusal notwithstanding.
        basis: EpsilonBasis,
        /// Every gated class whose window is empty.
        collapsed: Vec<CollapsedClass>,
    }

    impl fmt::Display for CollapsedWindows {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            writeln!(
                f,
                "EPSILON BASIS COLLAPSED: {} of {} classes have NO legal epsilon window, and \
                 this regime requires every one of them",
                self.collapsed.len(),
                ParameterClass::ALL.len(),
            )?;
            writeln!(f, "  regime: {}", self.basis.regime_id)?;
            writeln!(
                f,
                "  frozen table: {}",
                if self.basis.table_resolved {
                    format!("resolved — gated classes [{}]", self.basis.gated.join(", "))
                } else {
                    "absent — every class is required, because a class the contract has not \
                     recorded cannot be exempted by omission"
                        .to_string()
                },
            )?;
            for c in &self.collapsed {
                writeln!(
                    f,
                    "  {:<20} lower {:.3e} EXCEEDS upper {:.3e} by {:.2}x",
                    c.class, c.lower, c.upper, c.exceed_factor,
                )?;
            }
            write!(
                f,
                "  A run whose epsilon basis is empty for a gated class is not evidence that an \
                 epsilon basis exists (D-18). Do not freeze an epsilon for a class listed above."
            )
        }
    }

    /// THE window rule. One site, called by BOTH matrices, fail-closed on collapse.
    ///
    /// `aggregates` is keyed by [`ParameterClass::tag`]; an absent class contributes zeroes, so
    /// a class nothing was measured for is reported with a degenerate window rather than
    /// silently dropped from the table.
    ///
    /// `measured_cells` and `full_matrix_cells` are compared to derive the typed coverage;
    /// passing the same list twice states COMPLETE coverage by construction.
    ///
    /// # The required set is READ from the regime, never declared here
    ///
    /// The gated set comes from `Thresholds::frozen().table_for(regime_id)`. There is
    /// deliberately no allowlist, no local constant and no environment override: the only way
    /// to exempt a class from needing an epsilon is a recorded contract table, which is a
    /// human-approved `pv diff`-flagged edit. Where no table resolves, EVERY class is required.
    ///
    /// # Errors
    ///
    /// [`CollapsedWindows`] when any REQUIRED class's window is empty. The refusal carries the
    /// full basis so the caller can still render and write its report before failing — which is
    /// also why it is BOXED: the refusal is deliberately the larger of the two variants, and an
    /// unboxed `Result` would pay for it on every success return.
    fn epsilon_basis(
        regime_id: &str,
        aggregates: &BTreeMap<&'static str, ClassAggregates>,
        measured_cells: &[String],
        full_matrix_cells: &[String],
        lower_bound: LowerBound,
    ) -> Result<EpsilonBasis, Box<CollapsedWindows>> {
        let frozen = Thresholds::frozen();
        let table = frozen.table_for(regime_id);

        let gated: Vec<&'static str> = match table {
            Some(t) => ParameterClass::ALL
                .into_iter()
                .filter(|class| t.of(*class).gated)
                .map(ParameterClass::tag)
                .collect(),
            None => Vec::new(),
        };

        let mut rows = Vec::with_capacity(ParameterClass::ALL.len());
        for class in ParameterClass::ALL {
            let agg = aggregates.get(class.tag()).copied().unwrap_or_default();
            // The ONE strict-window comparison in this file. `10x` in the report's column
            // headers names THIS factor; 05-03 edits the two lines above the comparison, never
            // the comparison itself. The lower edge is now supplied by the caller's named
            // `LowerBound` rule, so a CANDIDATE bound's per-class status is obtained by calling
            // this function again rather than by open-coding a second emptiness determination.
            let lower = lower_bound.lower(&agg);
            let upper = agg.best_real / WINDOW_SAFETY_FACTOR;
            let window = if lower < upper {
                WindowStatus::Legal
            } else {
                WindowStatus::Empty {
                    exceed_factor: if upper > 0.0 { lower / upper } else { f64::INFINITY },
                }
            };
            let required = match table {
                Some(t) => t.of(class).gated,
                None => true,
            };
            rows.push(BasisRow {
                class,
                aggregates: agg,
                lower,
                upper,
                window,
                required,
                declared_ungated: table.is_some() && !required,
            });
        }

        // The missing-cell derivation the production matrix's PROVISIONAL banner already used,
        // moved here rather than reimplemented — it is the same question and must not be able
        // to answer it twice differently.
        let missing: Vec<String> =
            full_matrix_cells.iter().filter(|c| !measured_cells.contains(c)).cloned().collect();
        let coverage = if missing.is_empty() {
            BasisCoverage::Complete
        } else {
            BasisCoverage::Provisional {
                measured: measured_cells.len(),
                total: full_matrix_cells.len(),
                missing,
            }
        };

        let basis = EpsilonBasis {
            regime_id: regime_id.to_string(),
            table_resolved: table.is_some(),
            gated,
            rows,
            coverage,
            lower_bound,
        };

        let collapsed: Vec<CollapsedClass> = basis
            .rows
            .iter()
            .filter(|row| row.required)
            .filter_map(|row| match row.window {
                WindowStatus::Legal => None,
                WindowStatus::Empty { exceed_factor } => Some(CollapsedClass {
                    class: row.class.tag(),
                    lower: row.lower,
                    upper: row.upper,
                    exceed_factor,
                }),
            })
            .collect();

        if collapsed.is_empty() {
            Ok(basis)
        } else {
            Err(Box::new(CollapsedWindows { basis, collapsed }))
        }
    }

    /// Render the basis section of a derivation report.
    ///
    /// Both matrices call this, so the fixture and production tables can no longer disagree.
    /// `complete_heading` is the caller's own heading for a COMPLETE basis; a PROVISIONAL one
    /// renders 05-01's banner verbatim instead.
    ///
    /// The verdict is NOT consulted here. A class the resolved table excludes from gating still
    /// prints its row and, if empty, still appears in the `WINDOWS THAT DO NOT EXIST` block —
    /// annotated, not omitted. The report and the verdict are separate surfaces.
    fn render_epsilon_basis(basis: &EpsilonBasis, complete_heading: &str) -> String {
        let mut out = String::new();
        out.push('\n');
        out.push_str(&basis.regime_table_line());

        match &basis.coverage {
            BasisCoverage::Complete => out.push_str(&format!("\n{complete_heading}\n")),
            BasisCoverage::Provisional { measured, total, missing } => out.push_str(&format!(
                "\nCROSS-CELL EPSILON BASIS — PROVISIONAL, NOT THE FROZEN EPSILON.\n\
                 Derived from {measured} of {total} boundary cells. MISSING: {}.\n\
                 The window rule takes the worst control and the best real across ALL measured \
                 cells, so an unmeasured cell can still move best_real and shrink every window \
                 below. Plan 05-03 must NOT freeze epsilon from this table.\n",
                missing.join(", "),
            )),
        }

        out.push_str(
            "class                worst_ctrl    worst_nnull   best_real     10x_lower     \
             10x_upper     noise_floor   eps/noise     nnull_moved   window        \
             median/min    gating\n",
        );
        for row in &basis.rows {
            out.push_str(&format!(
                "{:<20} {:<13.3e} {:<13.3e} {:<13.3e} {:<13.3e} {:<13.3e} {:<13.3e} {:<13} \
                 {:<13} {:<13} {:<13.1e} {}\n",
                row.class.tag(),
                row.aggregates.worst_ctrl,
                row.aggregates.worst_nnull,
                row.aggregates.best_real,
                row.lower,
                row.upper,
                row.aggregates.noise_floor,
                row.eps_over_noise(),
                row.aggregates.near_null_moved_all,
                row.window_tag(),
                row.spread(),
                row.gating_tag(),
            ));
        }

        let empty: Vec<&BasisRow> =
            basis.rows.iter().filter(|r| !matches!(r.window, WindowStatus::Legal)).collect();
        if empty.is_empty() {
            out.push_str("\nEvery class above has a non-empty window (10x_lower < 10x_upper).\n");
        } else {
            out.push_str(&format!(
                "\nWINDOWS THAT DO NOT EXIST — {} of {} classes have NO legal epsilon\n",
                empty.len(),
                ParameterClass::ALL.len(),
            ));
            for row in &empty {
                let factor = match row.window {
                    WindowStatus::Empty { exceed_factor } => exceed_factor,
                    WindowStatus::Legal => f64::NAN,
                };
                out.push_str(&format!(
                    "  {:<20} lower {:.3e} EXCEEDS upper {:.3e} by {:.2}x{}\n",
                    row.class.tag(),
                    row.lower,
                    row.upper,
                    factor,
                    if row.declared_ungated {
                        " [declared-ungated by the resolved table — excluded from the VERDICT, \
                         never from this report]"
                    } else {
                        ""
                    },
                ));
            }
            out.push_str(
                "  The window rule is [10 x max(worst_ctrl, worst_nnull), best_real / 10]. When \
                 the\n  lower bound exceeds the upper there is no value epsilon can take: it \
                 cannot be both\n  10x above the strongest not-training signal and 10x below the \
                 weakest training signal.\n  `eps/noise` reads `n/a` for these classes BY DESIGN \
                 — that column divides `10x_upper`\n  by the noise floor, and `10x_upper` is not \
                 a legal epsilon here. Do not freeze an epsilon\n  for a class listed above, and \
                 do not read its suppressed margin as small rather\n  than absent.\n",
            );
        }
        out
    }

    // -----------------------------------------------------------------------------------
    // CANDIDATE LOWER BOUNDS — report only (plan 05-03, the D-04 decision input)
    // -----------------------------------------------------------------------------------

    /// The candidate lower bounds plan 05-03's D-04 checkpoint chooses between.
    ///
    /// Both BOUNDS are already written into `setfit-train-lifecycle-v1.yaml`; the question the
    /// checkpoint answers is which one binds the production regime, and — separately — what
    /// factor sits on it. The noise-floor bound is printed at TWO factors precisely because the
    /// contract supplies the condition and not the factor, so `L2-bare` is what the contract
    /// literally requires and `L2-10x` is a choice this plan makes and must own.
    const CANDIDATE_LOWER_BOUNDS: [(&str, LowerBound); 3] = [
        ("L1      (D-03 rule, being replaced)", LowerBound::NotTrainingControl { factor: 10.0 }),
        ("L2-bare (bare contracted clearance)", LowerBound::RoundingNoiseFloor { factor: 1.0 }),
        ("L2-10x  (CHOSEN safety factor)", LowerBound::RoundingNoiseFloor { factor: 10.0 }),
    ];

    /// Render the candidate lower-bound tables — REPORT ONLY, never a verdict.
    ///
    /// # Why this cannot open-code a comparison
    ///
    /// Plan 05-14 pins the non-comment count of `lower < upper` in this file at exactly ONE,
    /// because the duplicated inline arithmetic is why 05-01's `eps/noise` trap had to be fixed
    /// twice. Four candidate columns each need an emptiness determination, and the naive
    /// implementation would open-code four more and destroy that invariant by construction. So
    /// each candidate's status is obtained by CALLING [`epsilon_basis`] with that candidate's
    /// bounds and reading the per-class status it returns. The refusal carries the whole basis,
    /// which is what makes a collapsed candidate readable rather than fatal here.
    fn render_candidate_lower_bounds(
        regime_id: &str,
        aggregates: &BTreeMap<&'static str, ClassAggregates>,
        measured_cells: &[String],
        full_matrix_cells: &[String],
    ) -> String {
        let mut out = String::new();
        out.push_str(
            "\nCANDIDATE LOWER BOUNDS — report only, no rule is frozen by printing it.\n\
             Every candidate shares the contracted upper edge `best_real / 10`; only the LOWER\n\
             edge differs. `width` is upper/lower — how much room a frozen epsilon has, NOT a\n\
             margin over anything. A width at or below 1.00x is an EMPTY window.\n",
        );

        for (name, candidate) in CANDIDATE_LOWER_BOUNDS {
            let derived =
                epsilon_basis(regime_id, aggregates, measured_cells, full_matrix_cells, candidate);
            let basis = match &derived {
                Ok(basis) => basis,
                Err(collapse) => &collapse.basis,
            };
            out.push_str(&format!(
                "\n  {name}   lower = {}\n  {:<20} {:<13} {:<13} {:<13} {}\n",
                candidate.label(),
                "class",
                "lower",
                "upper",
                "width",
                "window",
            ));
            for row in &basis.rows {
                let width = if row.lower > 0.0 { row.upper / row.lower } else { f64::INFINITY };
                out.push_str(&format!(
                    "  {:<20} {:<13.3e} {:<13.3e} {:<13.2} {}\n",
                    row.class.tag(),
                    row.lower,
                    row.upper,
                    width,
                    row.window_tag(),
                ));
            }
        }

        // L3 — relaxing the near-null bound's safety factors, refuted BY ARITHMETIC rather than
        // by taste, so the numbers are on the record. These are divisions on two measured
        // quantities, not window-emptiness determinations, so they do not add a comparison site.
        out.push_str(
            "\n  L3 — RELAXING THE NEAR-NULL FACTORS. Largest admissible values, per class:\n\
             \x20   k_low_max  = best_real / (10 x worst_nnull)   (relax the lower factor alone)\n\
             \x20   product_max = best_real / worst_nnull          (relax BOTH factors)\n\
             \x20 The contracted product is 100. A product_max BELOW 1 means epsilon would sit\n\
             \x20 UNDER the near-null delta it must exceed — an INVERTED margin, not a reduced\n\
             \x20 one, under which a near-null run would PASS.\n",
        );
        out.push_str(&format!("  {:<20} {:<15} {}\n", "class", "k_low_max", "product_max"));
        let mut binding: Option<(&'static str, f64)> = None;
        for class in ParameterClass::ALL {
            let agg = aggregates.get(class.tag()).copied().unwrap_or_default();
            let worst = agg.worst_ctrl.max(agg.worst_nnull);
            let (k_low_max, product_max) = if worst > 0.0 {
                (agg.best_real / (WINDOW_SAFETY_FACTOR * worst), agg.best_real / worst)
            } else {
                (f64::INFINITY, f64::INFINITY)
            };
            out.push_str(&format!(
                "  {:<20} {:<15.3} {:.3}\n",
                class.tag(),
                k_low_max,
                product_max,
            ));
            if binding.is_none_or(|(_, worst_so_far)| product_max < worst_so_far) {
                binding = Some((class.tag(), product_max));
            }
        }
        if let Some((tag, product_max)) = binding {
            out.push_str(&format!(
                "  BINDING CLASS: {tag} — largest admissible factor PRODUCT {product_max:.3} \
                 against the contracted 100.\n",
            ));
        }
        out
    }

    // -----------------------------------------------------------------------------------
    // The DEFAULT-SUITE falsification of the window verdict (D-18, plan 05-14)
    //
    // Every test below runs in a plain `cargo test -p aprender-train --lib --features setfit`:
    // no `#[ignore]`, no env gate, no 86.7 MB checkout, no training. The aggregates are
    // ordinary `f64` inputs, which is the whole point — a gate whose falsification needs an
    // eight-hour job is a gate nobody falsifies.
    //
    // NAMING: every test here carries `epsilon_basis` in its name, PREFIXED (`evidence_...`)
    // so that a grep for the bare window-rule definition still finds exactly one function
    // rather than one per test. The verify block that gates this work filters on
    // `epsilon_basis` and asserts a count floor, so a test named outside this convention would
    // be a durable guard that the block never runs.
    // -----------------------------------------------------------------------------------

    /// A regime id no frozen table can ever cover.
    ///
    /// Its architecture component is not a real encoder fingerprint and never will be, so
    /// `table_for` resolves `None` for it whatever plan 05-03 lands. That is what makes the
    /// real-data regression test below RED BY CONSTRUCTION rather than state-dependent: it
    /// cannot flip green when the production table is frozen.
    const UNCALIBRATABLE_REGIME: &str =
        "no-frozen-table-by-construction-05-14@0000000|seeds=13|cells=s64e1b16";

    /// The production regime the 12 committed passes were measured in, verbatim from 05-01.
    ///
    /// Resolves NO table today (the architecture differs from the fixture slice's). Unlike
    /// [`UNCALIBRATABLE_REGIME`] this one is a real id that 05-03 may eventually calibrate,
    /// which is exactly why the DURABLE test does not use it.
    const PRODUCTION_REGIME_TODAY: &str =
        "minilm-slice-h384-l6-a12-i1536-v30522@1110a243|seeds=13|cells=s64e1b16";

    /// A basis in which every class's window is legal: `lower = 1e-6 < upper = 1e-5`.
    fn legal_aggregates() -> BTreeMap<&'static str, ClassAggregates> {
        let mut out = BTreeMap::new();
        for class in ParameterClass::ALL {
            out.insert(
                class.tag(),
                ClassAggregates {
                    worst_ctrl: 0.0,
                    worst_nnull: 1.0e-7,
                    best_real: 1.0e-4,
                    worst_median: 5.0e-4,
                    noise_floor: 1.0e-9,
                    near_null_moved_all: false,
                },
            );
        }
        out
    }

    /// Collapse ONE class: `lower = 1e-3` against `upper = 1e-5`, a 100x exceedance.
    fn collapse_one(
        aggregates: &mut BTreeMap<&'static str, ClassAggregates>,
        class: ParameterClass,
    ) {
        let entry = aggregates.get_mut(class.tag()).expect("every class has aggregates");
        entry.worst_nnull = 1.0e-4;
    }

    /// One cell, measured, out of one — COMPLETE coverage.
    fn one_cell() -> Vec<String> {
        vec!["s8:13".to_string()]
    }

    #[test]
    fn evidence_epsilon_basis_all_windows_legal_returns_the_success_value() {
        let basis = epsilon_basis(
            &regime_id(),
            &legal_aggregates(),
            &one_cell(),
            &one_cell(),
            CONTRACTED_NEAR_NULL_LOWER_BOUND,
        )
        .expect("a basis whose every class has a legal window is not a refusal");

        assert_eq!(basis.rows.len(), ParameterClass::ALL.len(), "one row per class");
        for class in ParameterClass::ALL {
            let row = basis.row(class);
            assert_eq!(row.window, WindowStatus::Legal, "{class}: window should be legal");
            // The rows carry their bounds and their noise-floor clearance, not just a verdict.
            assert!(row.lower < row.upper, "{class}: bounds are an interval");
            assert!(
                row.eps_over_noise().starts_with('1'),
                "{class}: eps/noise is 1e-5 / 1e-9 = 1.00e4, got {}",
                row.eps_over_noise(),
            );
            assert!(row.aggregates.noise_floor > 0.0, "{class}: the noise floor is carried");
        }
        assert_eq!(basis.coverage, BasisCoverage::Complete);
    }

    #[test]
    fn evidence_epsilon_basis_one_empty_gated_window_refuses_naming_class_and_factor() {
        let mut aggregates = legal_aggregates();
        collapse_one(&mut aggregates, ParameterClass::Embedding);

        let collapse = epsilon_basis(
            &regime_id(),
            &aggregates,
            &one_cell(),
            &one_cell(),
            CONTRACTED_NEAR_NULL_LOWER_BOUND,
        )
        .expect_err(
            "a class the fixture regime GATES with no legal window must refuse, not report",
        );

        assert_eq!(collapse.collapsed.len(), 1, "exactly one class collapsed");
        let only = collapse.collapsed[0];
        assert_eq!(only.class, "embedding");
        // lower = 1e-4 * 10 = 1e-3; upper = 1e-4 / 10 = 1e-5; the factor is 100.
        assert!(
            (only.exceed_factor - 100.0).abs() < 1e-9,
            "the refusal carries the exceed factor, got {}",
            only.exceed_factor,
        );
        let rendered = collapse.to_string();
        assert!(rendered.contains("embedding"), "the message names the class: {rendered}");
        assert!(rendered.contains("100.00x"), "the message names the factor: {rendered}");
    }

    #[test]
    fn evidence_epsilon_basis_exactly_equal_bounds_refuse_because_the_window_is_strict() {
        // Chosen to be EXACT in binary f64: 2.0 * 10.0 == 20.0 and 200.0 / 10.0 == 20.0, so
        // this is a genuine touching-bounds case and not a rounding artifact.
        let mut aggregates = legal_aggregates();
        let entry = aggregates
            .get_mut(ParameterClass::LayerNormWeight.tag())
            .expect("every class has aggregates");
        entry.worst_nnull = 2.0;
        entry.best_real = 200.0;

        let collapse = epsilon_basis(
            &regime_id(),
            &aggregates,
            &one_cell(),
            &one_cell(),
            CONTRACTED_NEAR_NULL_LOWER_BOUND,
        )
        .expect_err("equal bounds are a point, not an interval; there is no value to freeze");

        let only = collapse.collapsed[0];
        assert_eq!(only.class, "layer_norm_weight");
        assert!((only.lower - 20.0).abs() < f64::EPSILON, "lower is exactly 20.0");
        assert!((only.upper - 20.0).abs() < f64::EPSILON, "upper is exactly 20.0");
        assert!(
            (only.exceed_factor - 1.0).abs() < f64::EPSILON,
            "touching bounds exceed by exactly 1x, got {}",
            only.exceed_factor,
        );
    }

    #[test]
    fn evidence_epsilon_basis_declared_ungated_moves_the_verdict_without_removing_the_row() {
        // ONE empty window, placed on `attention_key_bias` — the class the fixture regime's
        // frozen table declares ungated (`gated: false`).
        let mut aggregates = legal_aggregates();
        collapse_one(&mut aggregates, ParameterClass::AttentionKeyBias);

        // (a) With the table RESOLVED, the declaration is read and the verdict is a success
        // value — but the row survives, EMPTY, with its factor and its annotation.
        let basis = epsilon_basis(
            &regime_id(),
            &aggregates,
            &one_cell(),
            &one_cell(),
            CONTRACTED_NEAR_NULL_LOWER_BOUND,
        )
        .expect("a class the resolved table declares ungated cannot collapse the verdict");
        let row = basis.row(ParameterClass::AttentionKeyBias);
        assert!(!row.required, "the table declares this class ungated");
        assert!(row.declared_ungated, "and the row records that it was DECLARED, not merely lax");
        assert!(matches!(row.window, WindowStatus::Empty { .. }), "the window is still empty");
        let report = render_epsilon_basis(&basis, "HEADING");
        assert!(report.contains("attention_key_bias"), "the row is not removed:\n{report}");
        assert!(report.contains("EMPTY 100.00x"), "with its EMPTY status and factor:\n{report}");
        assert!(report.contains("declared-ungated"), "and its annotation:\n{report}");

        // (b) The two-sided partner: the SAME empty window on a class the SAME table gates.
        let mut gated_side = legal_aggregates();
        collapse_one(&mut gated_side, ParameterClass::ProjectionBias);
        let collapse = epsilon_basis(
            &regime_id(),
            &gated_side,
            &one_cell(),
            &one_cell(),
            CONTRACTED_NEAR_NULL_LOWER_BOUND,
        )
        .expect_err("the identical window on a GATED class must refuse");
        assert_eq!(collapse.collapsed[0].class, "projection_bias");
        assert!(
            (collapse.collapsed[0].exceed_factor - 100.0).abs() < 1e-9,
            "the same window, so the same factor — only the declaration differs",
        );

        // (c) And the declaration is what moved it: the SAME aggregates as (a) under a regime
        // that resolves NO table refuse, because an unrecorded class is never ungated by
        // default.
        let strict = epsilon_basis(
            UNCALIBRATABLE_REGIME,
            &aggregates,
            &one_cell(),
            &one_cell(),
            CONTRACTED_NEAR_NULL_LOWER_BOUND,
        )
        .expect_err("with no table resolved, every class is required");
        assert_eq!(strict.collapsed[0].class, "attention_key_bias");
        assert!(
            !strict.basis.row(ParameterClass::AttentionKeyBias).declared_ungated,
            "no table resolved means there is no declaration to read",
        );
    }

    /// With NO table resolved, EVERY class is required — exemption by omission stays closed.
    ///
    /// # Why this now derives under `UNCALIBRATABLE_REGIME`
    ///
    /// Plan 05-14 wrote this test against `PRODUCTION_REGIME_TODAY` and said so explicitly: that
    /// id "is a real id that 05-03 may eventually calibrate, which is exactly why the DURABLE
    /// test does not use it". Plan 05-03 calibrated it. So the id no longer demonstrates the
    /// no-table branch, and the honest migration is to derive under the id that can never
    /// resolve — not to weaken the assertion.
    ///
    /// The state change is not silently dropped: it is asserted POSITIVELY below, so the fact
    /// that the production regime went from uncalibrated to calibrated is recorded by a test
    /// rather than by a test's disappearance.
    #[test]
    fn evidence_epsilon_basis_without_a_resolved_table_every_class_is_required() {
        let frozen = Thresholds::frozen();
        assert!(
            frozen.table_for(UNCALIBRATABLE_REGIME).is_none(),
            "this test's design is that no table resolves for `{UNCALIBRATABLE_REGIME}`",
        );
        // THE STATE CHANGE, asserted rather than assumed: plan 05-03 calibrated the production
        // regime, which is why this test no longer derives under it.
        assert!(
            frozen.table_for(PRODUCTION_REGIME_TODAY).is_some(),
            "since plan 05-03 the production regime IS calibrated; if this ever stops holding, \
             the F-10 unblock has been reverted and that must fail loudly here",
        );

        let basis_all_legal = epsilon_basis(
            UNCALIBRATABLE_REGIME,
            &legal_aggregates(),
            &one_cell(),
            &one_cell(),
            CONTRACTED_NEAR_NULL_LOWER_BOUND,
        )
        .expect("no table resolved is not by itself a refusal");
        assert!(!basis_all_legal.table_resolved);
        assert!(basis_all_legal.gated.is_empty(), "there is no gated set to report");
        for class in ParameterClass::ALL {
            assert!(basis_all_legal.row(class).required, "{class} is required with no table");
        }
        assert!(
            basis_all_legal.regime_table_line().contains("absent"),
            "the report states the lookup's own answer: {}",
            basis_all_legal.regime_table_line(),
        );

        // The class BOTH resolved tables declare ungated is required here, because there is no
        // declaration to read.
        let mut aggregates = legal_aggregates();
        collapse_one(&mut aggregates, ParameterClass::AttentionKeyBias);
        let collapse = epsilon_basis(
            UNCALIBRATABLE_REGIME,
            &aggregates,
            &one_cell(),
            &one_cell(),
            CONTRACTED_NEAR_NULL_LOWER_BOUND,
        )
        .expect_err("exemption by omission is exactly the hole this closes");
        assert_eq!(collapse.collapsed[0].class, "attention_key_bias");

        // CONTROL, so the refusal above cannot be read as "this id refuses everything": under
        // the now-CALIBRATED production regime the identical aggregates are a success value,
        // because that table DECLARES the class ungated. The declaration is what moves the
        // verdict, and only a recorded contract table can supply one.
        let declared = epsilon_basis(
            PRODUCTION_REGIME_TODAY,
            &aggregates,
            &one_cell(),
            &one_cell(),
            CONTRACTED_NEAR_NULL_LOWER_BOUND,
        )
        .expect("the production table declares attention_key_bias ungated");
        assert!(declared.table_resolved);
        assert!(declared.row(ParameterClass::AttentionKeyBias).declared_ungated);
    }

    #[test]
    fn evidence_epsilon_basis_coverage_is_typed_complete_or_provisional() {
        let full = vec!["s8:13".to_string(), "s64:13".to_string()];

        let complete = epsilon_basis(
            &regime_id(),
            &legal_aggregates(),
            &full,
            &full,
            CONTRACTED_NEAR_NULL_LOWER_BOUND,
        )
        .expect("all windows legal");
        assert_eq!(complete.coverage, BasisCoverage::Complete);

        // A partial measurement stays LEGAL — provisional coverage is not a refusal.
        let provisional = epsilon_basis(
            &regime_id(),
            &legal_aggregates(),
            &one_cell(),
            &full,
            CONTRACTED_NEAR_NULL_LOWER_BOUND,
        )
        .expect("a provisional basis with all windows legal is a success value");
        assert_eq!(
            provisional.coverage,
            BasisCoverage::Provisional {
                measured: 1,
                total: 2,
                missing: vec!["s64:13".to_string()],
            },
            "and a consumer can still tell it apart from a freezable one",
        );
        let report = render_epsilon_basis(&provisional, "HEADING");
        assert!(report.contains("PROVISIONAL, NOT THE FROZEN EPSILON"), "{report}");
        assert!(report.contains("MISSING: s64:13"), "{report}");
        assert!(
            !report.contains("HEADING"),
            "a provisional basis must not wear a complete heading"
        );
    }

    #[test]
    fn evidence_epsilon_basis_report_line_names_whether_a_table_resolved() {
        let resolved = epsilon_basis(
            &regime_id(),
            &legal_aggregates(),
            &one_cell(),
            &one_cell(),
            CONTRACTED_NEAR_NULL_LOWER_BOUND,
        )
        .expect("all windows legal");
        let line = resolved.regime_table_line();
        assert!(line.starts_with("REGIME TABLE FOR THE DERIVED REGIME: resolved"), "{line}");
        assert!(!line.contains("attention_key_bias"), "the gated set EXCLUDES it: {line}");
        for class in [
            "embedding",
            "layer_norm_weight",
            "layer_norm_bias",
            "projection_weight",
            "projection_bias",
        ] {
            assert!(line.contains(class), "the gated set names {class}: {line}");
        }

        let absent = epsilon_basis(
            UNCALIBRATABLE_REGIME,
            &legal_aggregates(),
            &one_cell(),
            &one_cell(),
            CONTRACTED_NEAR_NULL_LOWER_BOUND,
        )
        .expect("all windows legal");
        assert!(
            absent.regime_table_line().starts_with("REGIME TABLE FOR THE DERIVED REGIME: absent"),
            "{}",
            absent.regime_table_line(),
        );
    }

    /// Where the 12 banked passes of plan 05-01 live, as committed to the repository.
    fn committed_calibration_store() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../.planning/phases/05-benchmark-and-claims-gate/calibration-store")
    }

    /// The 12 committed passes still verify from their bytes, and the guard is NON-VACUOUS.
    ///
    /// Sidecars are COMPOUND-named — `<pass_stem>.evidence.json` / `<pass_stem>.meta.json` where
    /// `<pass_stem>` is `{cell_label}-seed{seed}-{condition}`. There is no file called
    /// `meta.json`, so an implementation that looked for one would discover zero sidecars,
    /// verify zero pairs and PASS. The PAIR COUNT is therefore asserted FIRST: a guard that
    /// verifies nothing is worse than no guard.
    #[test]
    fn evidence_epsilon_basis_committed_store_digests_verify() {
        let store = committed_calibration_store();
        let entries = std::fs::read_dir(&store).unwrap_or_else(|e| {
            panic!("the committed calibration store must be readable at {}: {e}", store.display())
        });

        let mut pairs: Vec<(String, std::path::PathBuf, std::path::PathBuf)> = Vec::new();
        for entry in entries {
            let path = entry.expect("a readable directory entry").path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else { continue };
            let Some(stem) = name.strip_suffix(".evidence.json") else { continue };
            let meta_path = store.join(format!("{stem}.meta.json"));
            assert!(
                meta_path.is_file(),
                "{stem}: the evidence file has no `{stem}.meta.json` beside it, so its digest \
                 cannot be checked against anything",
            );
            pairs.push((stem.to_string(), path.clone(), meta_path));
        }
        pairs.sort();
        assert_eq!(
            pairs.len(),
            12,
            "expected the 12 banked passes of plan 05-01 under {}; found {}. A count of ZERO \
             must FAIL here — a guard that discovers no pairs verifies nothing while reporting \
             success.",
            store.display(),
            pairs.len(),
        );

        for (stem, evidence_path, meta_path) in &pairs {
            let bytes = std::fs::read(evidence_path)
                .unwrap_or_else(|e| panic!("cannot read {}: {e}", evidence_path.display()));
            let meta: PassMeta = serde_json::from_slice(
                &std::fs::read(meta_path)
                    .unwrap_or_else(|e| panic!("cannot read {}: {e}", meta_path.display())),
            )
            .unwrap_or_else(|e| panic!("cannot parse {}: {e}", meta_path.display()));

            let mut hasher = Sha256::new();
            hasher.update(&bytes);
            assert_eq!(
                hex::encode(hasher.finalize()),
                meta.evidence_sha256,
                "{stem}: the committed bytes do not hash to the recorded digest — the store is \
                 corrupt, not stale",
            );

            let evidence: UpdateEvidence = serde_json::from_slice(&bytes)
                .unwrap_or_else(|e| panic!("cannot parse {}: {e}", evidence_path.display()));
            assert_eq!(
                evidence.to_canonical_bytes().expect("re-serialize committed evidence"),
                *bytes,
                "{stem}: the committed evidence does not round-trip; loading it lost information",
            );
        }
        println!("committed calibration store: {} of 12 pairs verified", pairs.len());
    }

    /// THE DURABLE REAL-DATA REGRESSION TEST — red by construction, forever.
    ///
    /// The aggregates are the ones plan 05-01 MEASURED over the four banked cells
    /// (`s8:13,s8:31,s8:53,s64:13`), copied from the PHASE-LEVEL FINDING table of
    /// `05-01-calibration-measurements.md`: `worst_ctrl` is `0.0` for every class (the 1e-30
    /// control wrote back bit-identical weights), so `worst_nnull` alone sets the lower bound.
    ///
    /// It derives under [`UNCALIBRATABLE_REGIME`], for which no table can ever resolve, so the
    /// strict branch always applies. That is what makes it independent of whatever plan 05-03
    /// lands: the `--ignored` four-cell combine is a state-dependent control on top of this;
    /// THIS is the permanent guard.
    #[test]
    fn evidence_epsilon_basis_real_four_cell_aggregates_refuse_under_no_resolved_table() {
        assert!(
            Thresholds::frozen().table_for(UNCALIBRATABLE_REGIME).is_none(),
            "this test's whole design is that no table resolves for `{UNCALIBRATABLE_REGIME}`; \
             if one now does, the guard has stopped being red by construction and must be \
             re-derived rather than relaxed",
        );

        // (worst_nnull, best_real) per class, measured — 05-01 PHASE-LEVEL FINDING.
        let measured: [(ParameterClass, f64, f64); 6] = [
            (ParameterClass::Embedding, 1.553e-4, 1.813e-3),
            (ParameterClass::LayerNormWeight, 4.802e-7, 1.891e-4),
            (ParameterClass::LayerNormBias, 9.844e-5, 7.112e-4),
            (ParameterClass::ProjectionWeight, 1.051e-4, 1.231e-3),
            (ParameterClass::ProjectionBias, 1.101e-4, 3.447e-4),
            (ParameterClass::AttentionKeyBias, 9.278e-9, 1.714e-7),
        ];
        let mut aggregates: BTreeMap<&'static str, ClassAggregates> = BTreeMap::new();
        for (class, worst_nnull, best_real) in measured {
            aggregates.insert(
                class.tag(),
                ClassAggregates {
                    worst_ctrl: 0.0,
                    worst_nnull,
                    best_real,
                    worst_median: best_real,
                    noise_floor: 0.0,
                    near_null_moved_all: true,
                },
            );
        }

        let collapse = epsilon_basis(
            UNCALIBRATABLE_REGIME,
            &aggregates,
            &["s8:13".to_string(), "s8:31".to_string(), "s8:53".to_string(), "s64:13".to_string()],
            &["s8:13".to_string(), "s8:31".to_string(), "s8:53".to_string(), "s64:13".to_string()],
            CONTRACTED_NEAR_NULL_LOWER_BOUND,
        )
        .expect_err(
            "five of six classes have NO legal epsilon on the measured four-cell basis; a run \
             that reports success on this data is the D-18 defect itself",
        );

        let named: Vec<&str> = collapse.collapsed.iter().map(|c| c.class).collect();
        assert_eq!(
            named,
            vec![
                "embedding",
                "layer_norm_bias",
                "projection_weight",
                "projection_bias",
                "attention_key_bias",
            ],
            "exactly the five 05-01 recorded; `layer_norm_weight` is the one that survives",
        );

        // The exceed factors 05-01 published, to two decimals.
        let expected = [
            ("embedding", 8.56),
            ("layer_norm_bias", 13.84),
            ("projection_weight", 8.54),
            ("projection_bias", 31.94),
            ("attention_key_bias", 5.41),
        ];
        for ((tag, want), got) in expected.iter().zip(collapse.collapsed.iter()) {
            assert_eq!(*tag, got.class);
            assert!(
                (got.exceed_factor - want).abs() < 0.01,
                "{tag}: 05-01 recorded {want}x, the shipped rule derives {}x",
                got.exceed_factor,
            );
        }
    }

    /// THE calibration matrix — `#[ignore]`d, and deliberately so.
    ///
    /// Twelve complete `run_tuning` passes over a real-weight MiniLM slice do not belong on
    /// the `evidence_` filter a developer types dozens of times a day. `#[ignore]` keeps it
    /// off every default run including `cargo test --workspace --lib`, while `pub(crate)`
    /// `run_tuning` and `#[cfg(test)]` `calibration_variants` both stay exactly as narrow as
    /// they are — the out-of-crate integration target an earlier draft specified could not
    /// have compiled against either.
    ///
    /// Invoke with:
    /// `cargo test -p aprender-train --lib --features setfit calibration_matrix -- --ignored --nocapture`
    #[test]
    #[ignore = "12 full tuning passes; run explicitly with --ignored (plan 03-05 epsilon basis)"]
    fn calibration_matrix_epsilon_basis() {
        let variants = fx::calibration_variants();
        assert!(variants.len() >= 6, "at least six cells");

        let mut report = String::new();
        report.push_str(&format!("\nCALIBRATION REGIME: {}\n", regime_id()));
        report.push_str(
            "\ncell             class                real_min      real_median   real_max      \
             ctrl_max      nnull_max     nnull_moved   noise_floor   support_frac  \
             all_moved\n",
        );

        // Cross-cell aggregates, per class, that plan 03-06 freezes epsilon from.
        let mut real_min_across: BTreeMap<&'static str, f64> = BTreeMap::new();
        let mut ctrl_max_across: BTreeMap<&'static str, f64> = BTreeMap::new();
        let mut median_max_across: BTreeMap<&'static str, f64> = BTreeMap::new();
        // The rounding-noise floor the frozen epsilon has to clear, per class, worst cell.
        let mut noise_floor_across: BTreeMap<&'static str, f64> = BTreeMap::new();
        // The near-null (1e-8) control's worst case per class — the REAL lower bound.
        let mut near_null_max_across: BTreeMap<&'static str, f64> = BTreeMap::new();
        // Whether EVERY near-null run still satisfies the strict `||dTheta|| > 0` predicate.
        let mut near_null_moved_all: BTreeMap<&'static str, bool> = BTreeMap::new();
        // WHICH parameter sets each class's lower bound. Without the name, 03-06 knows the
        // number but not what to widen if the margin turns out to be too narrow.
        let mut binding_param: BTreeMap<&'static str, String> = BTreeMap::new();
        let mut endpoint_rows: Vec<String> = Vec::new();
        let mut endpoint_deltas: Vec<f64> = Vec::new();
        let mut cells_run = 0_usize;

        for variant in &variants {
            let real = evidence_for(*variant, None);
            let control = evidence_for(*variant, Some(CONTROL_LR));
            let near_null = evidence_for(*variant, Some(NEAR_NULL_LR));
            cells_run += 3;

            let endpoint_delta = real.last_k_mean - real.first_k_mean;
            endpoint_deltas.push(endpoint_delta);
            endpoint_rows.push(format!(
                "  seed {:>3} cell {:<9} k={} first_k={:.9} last_k={:.9} delta={endpoint_delta:+.9}\n",
                variant.root_seed,
                variant.label,
                real.k,
                real.first_k_mean,
                real.last_k_mean,
            ));

            for class in ParameterClass::ALL {
                let real_rows = real.rows_of_class(class);
                let control_rows = control.rows_of_class(class);
                let near_null_rows = near_null.rows_of_class(class);
                assert!(!real_rows.is_empty(), "{class} has no rows");
                assert_eq!(real_rows.len(), control_rows.len());
                assert_eq!(real_rows.len(), near_null_rows.len());

                let real_values: Vec<f64> = real_rows.iter().map(|r| r.relative_delta).collect();
                let control_values: Vec<f64> =
                    control_rows.iter().map(|r| r.relative_delta).collect();
                let support: Vec<f64> =
                    real_rows.iter().map(|r| r.delta_support_fraction).collect();

                let near_null_values: Vec<f64> =
                    near_null_rows.iter().map(|r| r.relative_delta).collect();
                let real_min = min_of(&real_values);
                let control_max = max_of(&control_values);
                let near_null_max = max_of(&near_null_values);
                let near_null_moved = near_null_rows.iter().all(|r| r.moved);

                let cell_noise_floor = max_of(
                    &real_rows.iter().map(|r| rounding_noise_floor(r)).collect::<Vec<f64>>(),
                );

                report.push_str(&format!(
                    "seed{:<3} {:<9} {:<20} {:<13.3e} {:<13.3e} {:<13.3e} {:<13.3e} {:<13.3e} \
                     {:<13} {:<13.3e} {:<13.4} {}\n",
                    variant.root_seed,
                    variant.label,
                    class.tag(),
                    real_min,
                    median_of(&real_values),
                    max_of(&real_values),
                    control_max,
                    near_null_max,
                    near_null_moved,
                    cell_noise_floor,
                    median_of(&support),
                    real_rows.iter().all(|r| r.moved),
                ));

                // (c) SEPARATION, per class and per cell.
                assert!(
                    control_max < real_min,
                    // near_null is reported, not asserted: whether it separates is the
                    // question this matrix exists to answer, not a property to presume.
                    "seed {} cell {} class {}: the 1e-30 control's max relative delta \
                     ({control_max:e}) is not below the real run's min ({real_min:e})",
                    variant.root_seed,
                    variant.label,
                    class.tag(),
                );

                let slot = real_min_across.entry(class.tag()).or_insert(f64::INFINITY);
                if real_min < *slot {
                    *slot = real_min;
                    let binding = real_rows
                        .iter()
                        .min_by(|a, b| {
                            a.relative_delta
                                .partial_cmp(&b.relative_delta)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map_or_else(String::new, |r| {
                            format!(
                                "{} (seed {} cell {}) delta_norm={:.3e} init_norm={:.3e} \
                                 grad_norm_max={:.3e} grad_norm_mean={:.3e} steps_observed={} \
                                 noise_floor={:.3e}",
                                r.name,
                                variant.root_seed,
                                variant.label,
                                r.delta_norm,
                                r.init_norm,
                                r.grad_norm_max,
                                r.grad_norm_mean,
                                r.steps_observed,
                                rounding_noise_floor(r),
                            )
                        });
                    binding_param.insert(class.tag(), binding);
                }
                let slot = noise_floor_across.entry(class.tag()).or_insert(f64::NEG_INFINITY);
                *slot = slot.max(cell_noise_floor);
                let slot = near_null_max_across.entry(class.tag()).or_insert(f64::NEG_INFINITY);
                *slot = slot.max(near_null_max);
                let slot = near_null_moved_all.entry(class.tag()).or_insert(true);
                *slot = *slot && near_null_moved;
                let slot = ctrl_max_across.entry(class.tag()).or_insert(f64::NEG_INFINITY);
                *slot = slot.max(control_max);
                let slot = median_max_across.entry(class.tag()).or_insert(f64::NEG_INFINITY);
                *slot = slot.max(median_of(&real_values));
            }
        }

        report.push_str("\nENDPOINT MEANS (computed, NOT judged)\n");
        for row in &endpoint_rows {
            report.push_str(row);
        }
        report.push_str(&format!(
            "  CROSS-SEED SPREAD of (last_k - first_k): min={:+.9} max={:+.9} range={:.9}\n",
            min_of(&endpoint_deltas),
            max_of(&endpoint_deltas),
            max_of(&endpoint_deltas) - min_of(&endpoint_deltas),
        ));

        // The window rule and its verdict live in ONE place (D-18). The lower edge is the WORSE
        // of the two controls: a frozen epsilon has to sit above anything a not-really-training
        // run produced, and the 1e-8 control is the one that produces anything at all.
        let mut aggregates: BTreeMap<&'static str, ClassAggregates> = BTreeMap::new();
        for class in ParameterClass::ALL {
            aggregates.insert(
                class.tag(),
                ClassAggregates {
                    worst_ctrl: ctrl_max_across.get(class.tag()).copied().unwrap_or(0.0),
                    worst_nnull: near_null_max_across.get(class.tag()).copied().unwrap_or(0.0),
                    best_real: real_min_across.get(class.tag()).copied().unwrap_or(0.0),
                    worst_median: median_max_across.get(class.tag()).copied().unwrap_or(0.0),
                    noise_floor: noise_floor_across.get(class.tag()).copied().unwrap_or(0.0),
                    near_null_moved_all: near_null_moved_all
                        .get(class.tag())
                        .copied()
                        .unwrap_or(false),
                },
            );
        }
        // This matrix measures every cell it defines, so its coverage is COMPLETE by
        // construction — stated by passing the same list twice rather than asserted in prose.
        let fixture_cells: Vec<String> =
            variants.iter().map(|v| format!("seed{}:{}", v.root_seed, v.label)).collect();
        // The FIXTURE regime's five gated epsilons were frozen under the contracted near-null
        // rule and still satisfy it, so this call site does not move: plan 05-03 changes what
        // binds the PRODUCTION regime, and the fixture derivation must stay exactly what it was.
        let basis_verdict = epsilon_basis(
            &regime_id(),
            &aggregates,
            &fixture_cells,
            &fixture_cells,
            CONTRACTED_NEAR_NULL_LOWER_BOUND,
        );
        {
            let basis = match &basis_verdict {
                Ok(basis) => basis,
                Err(collapse) => &collapse.basis,
            };
            report.push_str(&render_epsilon_basis(
                basis,
                "CROSS-CELL EPSILON BASIS (03-06 freezes from these)",
            ));
        }

        report.push_str("\nWHAT BINDS EACH CLASS'S LOWER EDGE\n");
        for class in ParameterClass::ALL {
            report.push_str(&format!(
                "  {:<20} {}\n",
                class.tag(),
                binding_param.get(class.tag()).map_or("-", String::as_str),
            ));
        }

        // FLAG, not a silent narrowing: a class whose slowest member is orders of magnitude
        // below its own median cannot be gated by one epsilon without either failing that
        // member or admitting a nearly-frozen encoder. Reported here and carried into the
        // SUMMARY with its numbers; widening the fixture or the matrix is 03-06's call.
        report.push_str("\nFLAGS\n");
        let mut flagged = 0_usize;
        for class in ParameterClass::ALL {
            let best_real = real_min_across.get(class.tag()).copied().unwrap_or(0.0);
            let worst_median = median_max_across.get(class.tag()).copied().unwrap_or(0.0);
            if best_real > 0.0 && worst_median / best_real > 100.0 {
                flagged += 1;
                report.push_str(&format!(
                    "  WIDE-SPREAD {}: median {:.3e} is {:.1e}x its own class minimum {:.3e}; \
                     a single per-class epsilon at min/10 = {:.3e} sits close to f32 \
                     resolution\n",
                    class.tag(),
                    worst_median,
                    worst_median / best_real,
                    best_real,
                    best_real / 10.0,
                ));
            }
        }
        // The flag that actually tests the claim `WIDE-SPREAD` makes in prose. `median/min` is a
        // spread statistic; it can be large for a class whose slowest member still moves far
        // above rounding noise, and small for one that does not move at all. THIS is the
        // freeze blocker: an epsilon at or under the rounding-noise floor cannot reject any
        // parameter that moved by even one ULP, so the class's gate would be a restatement of
        // the strict `||dTheta|| > 0` predicate wearing a threshold's name.
        for class in ParameterClass::ALL {
            let best_real = real_min_across.get(class.tag()).copied().unwrap_or(0.0);
            let noise_floor = noise_floor_across.get(class.tag()).copied().unwrap_or(0.0);
            let eps = best_real / 10.0;
            if noise_floor > 0.0 && eps <= noise_floor {
                flagged += 1;
                report.push_str(&format!(
                    "  EPS-BELOW-NOISE {}: the 10x-margin epsilon {:.3e} is at or under the \
                     f32 rounding-noise floor {:.3e} (ratio {:.2e}); this class cannot be \
                     frozen at this width\n",
                    class.tag(),
                    eps,
                    noise_floor,
                    eps / noise_floor,
                ));
            }
        }
        // Does the strict `||dTheta|| > 0` predicate DISCRIMINATE for this class? If a
        // 1e-8 run — 2000x below the reference rate, not meaningfully training — still moves
        // every member of the class, then `moved()` is true for reasons that have nothing to
        // do with tuning and arming it for that class asserts something that cannot fail.
        for class in ParameterClass::ALL {
            if near_null_moved_all.get(class.tag()).copied().unwrap_or(false) {
                flagged += 1;
                report.push_str(&format!(
                    "  MOVED-ALONE-INSUFFICIENT {}: every member still satisfies \
                     ||dTheta|| > 0 under the 1e-8 near-null control (worst relative delta \
                     {:.3e}), so the strict predicate ALONE does not separate training from \
                     not-training for this class -- the class's epsilon is what does, and \
                     this is the measurement that shows the epsilon is not decoration\n",
                    class.tag(),
                    near_null_max_across.get(class.tag()).copied().unwrap_or(0.0),
                ));
            }
        }
        if flagged == 0 {
            report.push_str("  none\n");
        }

        assert!(cells_run >= 12, "the matrix must run >= 12 passes, ran {cells_run}");

        // The report goes to stdout AND to a file. The file is not belt-and-braces: a
        // summarizing wrapper around `cargo test` (this repo ships one) drops `--nocapture`
        // output entirely, and a calibration whose numbers cannot be read is a calibration
        // that did not happen. `target/` is NOT a safe destination — `.cargo/config.toml`
        // redirects the target directory and it may not exist relative to the test's cwd,
        // which is how the first run of this test failed.
        println!("{report}");
        let destination = std::env::var("SETFIT_CALIBRATION_REPORT").map_or_else(
            |_| std::env::temp_dir().join("setfit-calibration-matrix.txt"),
            std::path::PathBuf::from,
        );
        std::fs::write(&destination, &report).unwrap_or_else(|e| {
            panic!("the calibration report must be writable at {destination:?}: {e}")
        });
        println!("calibration report written to {}", destination.display());

        // THE VERDICT, asserted only after the report has been written: a derivation whose
        // numbers cannot be read is worse than one that fails loudly (D-18).
        if let Err(collapse) = basis_verdict {
            panic!("{collapse}");
        }
    }

    // -----------------------------------------------------------------------------------
    // The PRODUCTION calibration matrix (plan 05-01, the F-10 unblock) — measurement only
    // -----------------------------------------------------------------------------------
    //
    // This half of the file measures the SAME quantities as `calibration_matrix_epsilon_basis`
    // directly above, on the production `sentence-transformers/all-MiniLM-L6-v2` checkout
    // instead of the committed 437 KB slice. It edits no contract, no threshold and no gate:
    // the `UncalibratedRegime` refusal lives in `validate_evidence` (judgement time), never in
    // `run_tuning` / `UpdateEvidence::from_tune_output`, so a production regime can be MEASURED
    // on today's code with zero relaxation. Freezing anything from these numbers is plan 05-03's
    // deliberate three-place edit, behind the D-04 human checkpoint.

    /// Contrastive **body** epochs, FROZEN from the pinned `setfit 1.1.3` reference environment.
    ///
    /// Read from the hash-locked env rather than from documentation or memory:
    ///
    /// ```text
    /// $ cd scripts/setfit_fixtures && uv run python -c \
    ///     "from setfit import TrainingArguments; a = TrainingArguments(); \
    ///      print(a.num_epochs, a.batch_size, a.body_learning_rate)"
    /// (1, 16) (16, 2) (2e-05, 1e-05)
    /// ```
    ///
    /// Every one of those three is a `(body, head)` PAIR. `SetFitTrainConfig` configures the
    /// contrastive BODY stage — the only stage this evidence table measures — so the body
    /// member is the one that maps onto its knobs: epochs 1, batch 16, encoder lr 2e-5.
    ///
    /// These two values are baked into every cell label `s{shots}e1b16` and therefore into the
    /// D-02 contract entry, which is why they are frozen BEFORE any calibration pass rather
    /// than read off whichever run happened to be convenient afterwards.
    const PRODUCTION_EPOCHS: u32 = 1;

    /// Contrastive **body** batch size, frozen from the same command. See [`PRODUCTION_EPOCHS`].
    const PRODUCTION_BATCH: u32 = 16;

    /// Training rows per class in the production synthetic corpus.
    ///
    /// The committed fixture pool is 16 per class (`fx::TRAIN_PER_CLASS`) — exactly the
    /// largest shot count the FIXTURE matrix asks for. The production envelope's boundary is
    /// `s64`, so the corpus is regenerated here at 64 rather than reused: a selector asked for
    /// 64 rows from a 16-row pool does not measure an `s64` cell, it fails.
    const PRODUCTION_TRAIN_PER_CLASS: usize = 64;

    /// The boundary shot counts of the D-02 envelope `{8, 16, 32, 64}`.
    const PRODUCTION_SHOTS: [u32; 2] = [8, 64];

    /// The min / median / max of the ten contracted seeds
    /// `{13, 17, 23, 29, 31, 37, 41, 43, 47, 53}`.
    const PRODUCTION_SEEDS: [u64; 3] = [13, 31, 53];

    /// The production checkout, resolved exactly as `full_weight_parity.rs:54` resolves it.
    ///
    /// (Plan 05-01 cites that file under `aprender-train/tests/`; it actually lives at
    /// `crates/aprender-core/tests/setfit_conformance/full_weight_parity.rs` — same pattern,
    /// same default, same env var.)
    fn production_checkout_dir() -> std::path::PathBuf {
        std::env::var("APRENDER_MINILM_DIR").map_or_else(
            |_| {
                let home = std::env::var("HOME").expect("HOME");
                std::path::PathBuf::from(home).join(".cache/aprender/minilm-l6-v2-1110a243")
            },
            std::path::PathBuf::from,
        )
    }

    /// Per-class sentence material for the production corpus.
    ///
    /// Unconstrained by the slice's 97-row vocabulary — the production encoder carries the
    /// full 30522-token WordPiece table, so this text needs no `vocab_remap` gymnastics. It is
    /// still entirely SYNTHETIC: no row of the phase's source corpus appears here (T-3-18).
    const PRODUCTION_CLASS_WORDS: [(&str, &str); 3] =
        [("market", "rallied"), ("climate", "shifted"), ("athlete", "trained")];

    /// Eight modifiers times eight objects gives the sixty-four distinct rows each class needs.
    const PRODUCTION_MODIFIERS: [&str; 8] =
        ["quick", "brown", "lazy", "warm", "bright", "quiet", "steady", "distant"];

    /// The object half of the grid.
    const PRODUCTION_OBJECTS: [&str; 8] =
        ["mat", "rug", "line", "pad", "ledger", "harbor", "summit", "corridor"];

    /// Held-out material for validation and test, disjoint from every train row — a cross-split
    /// duplicate would be coalesced by the ingest ladder and silently shrink a class pool.
    const PRODUCTION_HELDOUT: [&str; 2] = ["hesitant", "reluctant"];

    /// The synthetic text for one production row.
    fn production_row_text(role_index: usize, label: usize, index: usize) -> String {
        let (subject, verb) = PRODUCTION_CLASS_WORDS[label];
        if role_index == 0 {
            let modifier = PRODUCTION_MODIFIERS[index % PRODUCTION_MODIFIERS.len()];
            let object =
                PRODUCTION_OBJECTS[(index / PRODUCTION_MODIFIERS.len()) % PRODUCTION_OBJECTS.len()];
            format!("the {modifier} {subject} {verb} over the {object} .")
        } else {
            let modifier = PRODUCTION_HELDOUT[(role_index - 1) % PRODUCTION_HELDOUT.len()];
            format!("the {modifier} {subject} {verb} again today .")
        }
    }

    /// The production synthetic corpus, built through the real `from_labeled_rows` ingest
    /// ladder — the same door `fx::synthetic_dataset` uses, at production pool size.
    fn production_dataset(ledger: &mut AccessLedger) -> PreparedDataset<Canonical> {
        let label_names: Vec<String> =
            (0..PRODUCTION_CLASS_WORDS.len()).map(|i| format!("class{i}")).collect();
        let rows = |role: &str, role_index: usize, per_class: usize| -> Vec<LabeledExample> {
            (0..PRODUCTION_CLASS_WORDS.len())
                .flat_map(|label| {
                    (0..per_class).map(move |index| LabeledExample {
                        id: format!("{role}:{label}-{index}"),
                        input: production_row_text(role_index, label, index),
                        label,
                        label_text: format!("class{label}"),
                        source_split: role.to_string(),
                    })
                })
                .collect()
        };
        let decl = |per_class: usize| SplitDeclaration {
            expected_class_counts: vec![per_class; PRODUCTION_CLASS_WORDS.len()],
            label_names: label_names.clone(),
        };
        PreparedDataset::<Canonical>::from_labeled_rows(
            rows("train", 0, PRODUCTION_TRAIN_PER_CLASS),
            rows("validation", 1, 1),
            rows("test", 2, 1),
            &CanonicalDeclarations {
                train: decl(PRODUCTION_TRAIN_PER_CLASS),
                validation: decl(1),
                test: decl(1),
                label_names,
            },
            ledger,
        )
        .expect("the production synthetic corpus must be a valid canonical dataset")
    }

    /// One cell of the production matrix.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct ProductionCell {
        shots: u32,
        seed: u64,
    }

    impl ProductionCell {
        /// The cell label this cell is expected to render, `s{shots}e{E}b{B}`.
        ///
        /// PREDICTED here for the report's ordering, and ASSERTED against what the run
        /// actually rendered. The id that reaches the contract is always the rendered one.
        fn label(self) -> String {
            format!("s{}e{PRODUCTION_EPOCHS}b{PRODUCTION_BATCH}", self.shots)
        }
    }

    /// The production configuration for one cell, at the frozen E/B.
    ///
    /// `encoder_lr_override` is how the two controls are expressed: a control differs from its
    /// cell in the learning rate and in NOTHING else, so a difference in the measured relative
    /// deltas is attributable to the learning rate alone. Identical in shape to
    /// `fx::config_for`, with two deliberate differences: the epochs/batch are the FROZEN
    /// production values rather than a shrunk fixture cell, and the pair budget is left to the
    /// contracted closed form (`PairConfig::budget = None`) because that is what a production
    /// benchmark cell will run.
    fn production_config(seed: u64, encoder_lr_override: Option<f64>) -> SetFitTrainConfig {
        let reference = SetFitTrainConfig::reference_defaults(seed);
        // Non-vacuity: the in-repo reference recipe must agree with the pinned setfit 1.1.3
        // environment this plan froze E/B from. If the two ever diverge, the cell labels this
        // harness measures stop describing the cells production runs in.
        assert_eq!(
            reference.epochs(),
            PRODUCTION_EPOCHS,
            "REFERENCE_EPOCHS disagrees with the pinned setfit 1.1.3 body epochs",
        );
        assert_eq!(
            reference.batch_size(),
            PRODUCTION_BATCH,
            "REFERENCE_BATCH_SIZE disagrees with the pinned setfit 1.1.3 body batch size",
        );
        SetFitTrainConfig::new(SetFitTrainRequest {
            encoder_lr: encoder_lr_override.unwrap_or_else(|| reference.encoder_lr()),
            epochs: PRODUCTION_EPOCHS,
            batch_size: PRODUCTION_BATCH,
            warmup_ratio: reference.warmup_ratio(),
            grad_clip_max_norm: reference.grad_clip_max_norm(),
            max_length: reference.max_length(),
            pair_config: PairConfig::new(seed),
            freeze_policy: Vec::new(),
            head_regularization: reference.head_regularization(),
            root_seed: seed,
            device: "cpu".to_string(),
            lr_schedule: reference.lr_schedule(),
        })
        .expect("the production configuration satisfies the twelve-knob table")
    }

    /// What one production pass measured.
    struct ProductionPass {
        evidence: UpdateEvidence,
        /// The regime id the RUN rendered — the string plan 05-03 copies byte-for-byte.
        regime: String,
        /// Wall-clock seconds. Lives on this stdout-report struct and NOWHERE in serialized
        /// evidence: a timing field inside `UpdateEvidence` would make two identical runs
        /// serialize differently and take TRN-06's bitwise claim with it.
        elapsed_secs: f64,
        steps: u64,
        /// The PROBED device this pass actually resolved to, read off `ResolvedSetFitConfig`
        /// after `prepare()` rather than echoed from the request string.
        ///
        /// CLAUDE.md verification rule 2: never label a run by intent. A report that says
        /// "cpu" because the config asked for "cpu" proves nothing about what executed; this
        /// is the resolved value, and the assertion below is what makes it load-bearing.
        device: String,
    }

    /// Run one production cell under one condition, through the SHIPPED doors only.
    fn production_pass(
        dir: &std::path::Path,
        cell: ProductionCell,
        encoder_lr_override: Option<f64>,
    ) -> ProductionPass {
        let mut ledger = AccessLedger::new();
        let dataset = production_dataset(&mut ledger);
        let selection = FewShotSelector::select(
            &dataset,
            &SelectionConfig { root_seed: cell.seed, shots_per_class: cell.shots },
            &mut ledger,
        )
        .expect("the production corpus must support this selection");
        let encoder = SetFitMiniLm::from_pretrained_dir(dir, cell.seed)
            .expect("the pinned production checkout must load through the bound constructor");

        let run = SetFitRun::prepare(
            encoder,
            dataset,
            selection,
            production_config(cell.seed, encoder_lr_override),
        )
        .expect("the production run must prepare");
        let (encoder, dataset, selection, config) = run.into_parts();

        // RENDERED by the production code path from the run's own coordinates, never composed
        // here. T-05-01-01: this is the string the contract entry is copied from.
        let regime = crate::train::setfit::calibration_regime_id(&encoder, &selection, &config);

        // The PROBED device, read off the resolved config. Phase 3 refuses any non-CPU
        // resolved device outright (`tune_rejects_a_non_cpu_resolved_device`: a
        // `Device::Cuda { index: 0 }` preflight is `UnsupportedDeviceForPhase3`), so this
        // measurement is CPU-bound by construction on ANY host — which is exactly why the
        // choice of host is an authorization question and not a speed one.
        let device = format!("{:?}", config.device());
        assert!(
            device.to_lowercase().contains("cpu"),
            "the production calibration must execute on CPU; resolved device was {device}",
        );

        let started = std::time::Instant::now();
        let out =
            run_tuning(encoder, &dataset, &selection, &config).expect("the production run tunes");
        let elapsed_secs = started.elapsed().as_secs_f64();

        let evidence =
            UpdateEvidence::from_tune_output(&out, &regime).expect("production evidence");
        let steps = evidence.step_count;
        ProductionPass { evidence, regime, elapsed_secs, steps, device }
    }

    /// Which cells and conditions this invocation runs.
    enum ProductionMode {
        /// `APRENDER_CALIBRATION_PROBE=1` — ONE cell (s8, seed 13), REAL condition only, so the
        /// boundary matrix's wall-clock can be PROJECTED before it is committed to (the
        /// CLAUDE.md >1hr compute check-in rule).
        Probe,
        /// `APRENDER_CALIBRATION_PROSPECTIVE="s16:41,s32:29"` — named contracted-but-unmeasured
        /// cells, REAL condition only, run AFTER epsilon is frozen. Validation, not derivation.
        Prospective(Vec<ProductionCell>),
        /// `APRENDER_CALIBRATION_CELLS="s8:13,s8:31"` — a named SUBSET of the boundary matrix
        /// with the FULL three-condition treatment, so the 18 passes can be executed in
        /// resumable chunks.
        ///
        /// This partitions the work; it does not shrink it. Each chunk runs exactly the same
        /// passes, with the same conditions and the same separation assertion, that the
        /// unchunked matrix would have run for those cells — so `{s8,s64} x {13,31,53}` split
        /// across several invocations is the SAME measurement as one invocation, not a trim.
        /// The cross-cell epsilon basis is then derived in plan 05-03 from the union of the
        /// per-cell tables, which is how the window rule is defined anyway (worst control and
        /// best real *across all measured cells*).
        CellSubset(Vec<ProductionCell>),
        /// `APRENDER_CALIBRATION_PASS="s64:13:real"` — ONE cell under ONE condition, persisted
        /// to the store and nothing else (D').
        ///
        /// The unit of work that actually fits the measured ~55.8 minute unattended window, and
        /// — more importantly — the unit that is RETRYABLE: a kill costs one pass rather than a
        /// whole three-pass cell. No separation assertion runs here, because a single condition
        /// cannot support one; the report this writes is always `STATUS: PARTIAL`.
        SinglePass(ProductionCell, Condition),
        /// `APRENDER_CALIBRATION_COMBINE="s64:13,s64:31"` — no training at all: load the three
        /// persisted conditions of each named cell and run the SAME per-class analysis and
        /// separation assertion the in-process path runs (D').
        ///
        /// The code below is shared, not parallel: the only thing this mode changes is where
        /// the three `ProductionPass` values come from. Everything downstream — the regime
        /// assertion, the per-class rows, `ctrl_max < real_min`, the epsilon accumulators — is
        /// literally the same lines executing on the same values.
        Combine(Vec<ProductionCell>),
        /// The full boundary matrix: `{s8, s64} x {13, 31, 53} x {real, control, near-null}`.
        BoundaryMatrix,
    }

    /// Parse a `s<shots>:<seed>,…` cell list from an environment variable.
    fn parse_cell_spec(var: &str, spec: &str) -> Vec<ProductionCell> {
        let cells: Vec<ProductionCell> =
            spec.split(',')
                .filter(|s| !s.trim().is_empty())
                .map(|entry| {
                    let (shots, seed) = entry.trim().split_once(':').unwrap_or_else(|| {
                        panic!("{var} entry `{entry}` is not `s<shots>:<seed>`")
                    });
                    let shots = shots
                        .trim()
                        .strip_prefix('s')
                        .unwrap_or_else(|| panic!("{var} shots `{shots}` must start with `s`"));
                    ProductionCell {
                        shots: shots.parse().expect("shot count"),
                        seed: seed.trim().parse().expect("seed"),
                    }
                })
                .collect();
        assert!(!cells.is_empty(), "{var} named no cells");
        cells
    }

    // =======================================================================================
    // D': per-CONDITION persistence
    //
    // A cell is three passes, and at s64 a pass is ~54 minutes against a measured ~55.8 minute
    // unattended window. Running a whole cell in one process therefore cannot complete: the
    // first attempt died 100 seconds after its first pass, losing the cell. The fix is to let
    // each PASS be its own invocation, persist what it measured, and perform the separation
    // assertion in a later cheap invocation over the persisted tables.
    //
    // This does not move the assertion off the measurement. `ctrl_max < real_min` compares
    // recorded numbers either way; the only question is whether the numbers a fresh process
    // records are the same numbers. That is not assumed here — `cross_process_determinism`
    // below runs one s8 pass twice in two fresh processes and requires the persisted tables to
    // be BIT-IDENTICAL. CLAUDE.md verification rule 4: widening a guard's scope requires
    // re-proving it in the new scope, and the old in-process proof does not transfer.
    // =======================================================================================

    /// One of the three conditions a cell is measured under.
    ///
    /// Previously implicit in three `production_pass` call sites with bare `Option<f64>`
    /// arguments; named here because a persisted pass has to say on disk which condition it is.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Condition {
        Real,
        Control,
        NearNull,
    }

    impl Condition {
        fn tag(self) -> &'static str {
            match self {
                Self::Real => "real",
                Self::Control => "control",
                Self::NearNull => "near-null",
            }
        }

        /// The ONLY thing that differs between conditions. A control differs from its cell in
        /// the learning rate and in nothing else, which is what makes a difference in the
        /// measured deltas attributable to the learning rate alone.
        fn lr_override(self) -> Option<f64> {
            match self {
                Self::Real => None,
                Self::Control => Some(CONTROL_LR),
                Self::NearNull => Some(NEAR_NULL_LR),
            }
        }

        fn parse(s: &str) -> Self {
            match s.trim() {
                "real" => Self::Real,
                "control" => Self::Control,
                "near-null" => Self::NearNull,
                other => panic!("unknown condition `{other}`; expected real|control|near-null"),
            }
        }
    }

    /// Where persisted passes live.
    fn calibration_store() -> std::path::PathBuf {
        std::env::var("APRENDER_CALIBRATION_STORE").map_or_else(
            |_| std::env::temp_dir().join("setfit-calibration-store"),
            std::path::PathBuf::from,
        )
    }

    /// The file stem one (cell, condition) pass is persisted under.
    fn pass_stem(cell: ProductionCell, condition: Condition) -> String {
        format!("{}-seed{}-{}", cell.label(), cell.seed, condition.tag())
    }

    /// The NON-deterministic half of a persisted pass: timing and the probed device.
    ///
    /// Split into its own file precisely BECAUSE it is not reproducible. `elapsed_secs` differs
    /// between two identical runs, so folding it into the evidence file would make the
    /// bit-identity check below unsatisfiable and there would be nothing left to check. The
    /// evidence file holds only what the assertion consumes; this holds only what the report
    /// prints.
    #[derive(Debug, Serialize, Deserialize)]
    struct PassMeta {
        cell: String,
        seed: u64,
        condition: String,
        device: String,
        elapsed_secs: f64,
        steps: u64,
        evidence_sha256: String,
    }

    /// Write one pass to the store.
    fn persist_pass(
        store: &std::path::Path,
        cell: ProductionCell,
        condition: Condition,
        pass: &ProductionPass,
    ) {
        std::fs::create_dir_all(store)
            .unwrap_or_else(|e| panic!("cannot create calibration store {}: {e}", store.display()));
        let stem = pass_stem(cell, condition);
        let bytes = pass.evidence.to_canonical_bytes().expect("canonical evidence bytes");
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let evidence_sha256 = hex::encode(hasher.finalize());

        let evidence_path = store.join(format!("{stem}.evidence.json"));
        std::fs::write(&evidence_path, &bytes)
            .unwrap_or_else(|e| panic!("cannot write {}: {e}", evidence_path.display()));
        let meta = PassMeta {
            cell: cell.label(),
            seed: cell.seed,
            condition: condition.tag().to_string(),
            device: pass.device.clone(),
            elapsed_secs: pass.elapsed_secs,
            steps: pass.steps,
            evidence_sha256: evidence_sha256.clone(),
        };
        let meta_path = store.join(format!("{stem}.meta.json"));
        std::fs::write(
            &meta_path,
            serde_json::to_vec_pretty(&meta).expect("pass metadata serializes"),
        )
        .unwrap_or_else(|e| panic!("cannot write {}: {e}", meta_path.display()));

        eprintln!(
            "[persist] {stem} evidence_sha256={evidence_sha256} bytes={} -> {}",
            bytes.len(),
            store.display(),
        );
    }

    /// Read one pass back out of the store.
    ///
    /// Two checks make the load lossless rather than merely successful: the bytes must still
    /// hash to the digest recorded when they were written, and the table PARSED BACK must
    /// re-serialize to those same bytes. Without the second, a field silently dropped by
    /// deserialization would sail through — and the assertion would then run on a table that is
    /// not the one measured.
    fn load_pass(
        store: &std::path::Path,
        cell: ProductionCell,
        condition: Condition,
    ) -> ProductionPass {
        let stem = pass_stem(cell, condition);
        let evidence_path = store.join(format!("{stem}.evidence.json"));
        let bytes = std::fs::read(&evidence_path).unwrap_or_else(|e| {
            panic!(
                "missing persisted pass {}: {e}\nRun it first with \
                 APRENDER_CALIBRATION_PASS=\"s{}:{}:{}\"",
                evidence_path.display(),
                cell.shots,
                cell.seed,
                condition.tag(),
            )
        });
        let meta_path = store.join(format!("{stem}.meta.json"));
        let meta: PassMeta = serde_json::from_slice(
            &std::fs::read(&meta_path)
                .unwrap_or_else(|e| panic!("cannot read {}: {e}", meta_path.display())),
        )
        .unwrap_or_else(|e| panic!("cannot parse {}: {e}", meta_path.display()));

        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let digest = hex::encode(hasher.finalize());
        assert_eq!(
            digest, meta.evidence_sha256,
            "{stem}: the persisted evidence does not hash to the digest recorded when it was \
             written — the store is corrupt, not stale",
        );

        let evidence: UpdateEvidence = serde_json::from_slice(&bytes)
            .unwrap_or_else(|e| panic!("cannot parse {}: {e}", evidence_path.display()));
        assert_eq!(
            evidence.to_canonical_bytes().expect("re-serialize persisted evidence"),
            bytes,
            "{stem}: the persisted evidence does not round-trip; loading it lost information",
        );

        ProductionPass {
            regime: evidence.calibration_regime_id.clone(),
            steps: evidence.step_count,
            evidence,
            elapsed_secs: meta.elapsed_secs,
            device: meta.device,
        }
    }

    /// Free space below which a pass must not be launched, in GiB.
    ///
    /// The volume filled to 100% during the first s64 attempt and every shell invocation began
    /// failing with `ENOSPC`. The final report write PANICS on failure, so exhausting the disk
    /// at the end of a ~54 minute pass destroys that pass at its last step. A preflight check
    /// costs milliseconds and protects the whole pass.
    const MIN_FREE_GIB: u64 = 10;

    /// Refuse to start a pass without room to write its result.
    ///
    /// Fails CLOSED on a measurement below the floor and OPEN if free space cannot be measured:
    /// an unparseable `df` is a reason to warn, not a reason to block an authorized run.
    fn assert_disk_headroom(path: &std::path::Path) {
        let out = match std::process::Command::new("df").arg("-Pk").arg(path).output() {
            Ok(o) if o.status.success() => o.stdout,
            _ => {
                eprintln!("[preflight] WARNING: could not run `df`; free-space check SKIPPED");
                return;
            }
        };
        let text = String::from_utf8_lossy(&out);
        let Some(avail_kib) = text
            .lines()
            .nth(1)
            .and_then(|l| l.split_whitespace().nth(3))
            .and_then(|f| f.parse::<u64>().ok())
        else {
            eprintln!("[preflight] WARNING: could not parse `df` output; check SKIPPED");
            return;
        };
        let avail_gib = avail_kib / (1024 * 1024);
        assert!(
            avail_gib >= MIN_FREE_GIB,
            "refusing to start a pass with {avail_gib} GiB free on {} (floor {MIN_FREE_GIB} \
             GiB): a ~54 minute pass whose final write hits ENOSPC is a pass destroyed at its \
             last step",
            path.display(),
        );
        eprintln!("[preflight] {avail_gib} GiB free on {} — OK", path.display());
    }

    /// Read the mode off the environment. Probe wins over the others if several are set.
    fn production_mode() -> ProductionMode {
        if std::env::var("APRENDER_CALIBRATION_PROBE").is_ok_and(|v| v == "1") {
            return ProductionMode::Probe;
        }
        if let Ok(spec) = std::env::var("APRENDER_CALIBRATION_PASS") {
            let (cell_spec, condition) = spec.trim().rsplit_once(':').unwrap_or_else(|| {
                panic!("APRENDER_CALIBRATION_PASS `{spec}` is not `s<shots>:<seed>:<condition>`")
            });
            let cells = parse_cell_spec("APRENDER_CALIBRATION_PASS", cell_spec);
            assert_eq!(cells.len(), 1, "APRENDER_CALIBRATION_PASS names exactly one cell");
            return ProductionMode::SinglePass(cells[0], Condition::parse(condition));
        }
        if let Ok(spec) = std::env::var("APRENDER_CALIBRATION_COMBINE") {
            return ProductionMode::Combine(parse_cell_spec("APRENDER_CALIBRATION_COMBINE", &spec));
        }
        if let Ok(spec) = std::env::var("APRENDER_CALIBRATION_CELLS") {
            return ProductionMode::CellSubset(parse_cell_spec(
                "APRENDER_CALIBRATION_CELLS",
                &spec,
            ));
        }
        if let Ok(spec) = std::env::var("APRENDER_CALIBRATION_PROSPECTIVE") {
            return ProductionMode::Prospective(parse_cell_spec(
                "APRENDER_CALIBRATION_PROSPECTIVE",
                &spec,
            ));
        }
        ProductionMode::BoundaryMatrix
    }

    /// THE production calibration matrix — `#[ignore]`d, env-gated, and measurement-only.
    ///
    /// Eighteen complete `run_tuning` passes over the full 22M-parameter production encoder do
    /// not belong on any default filter, and the 86.7 MB checkout they need is fetched rather
    /// than vendored (D-10 / SAFE-02). Absent the checkout the test PRINTS A SKIP naming
    /// `APRENDER_MINILM_DIR` and returns, exactly as `full_weight_parity.rs` does.
    ///
    /// Materialize the checkout:
    /// `cd scripts/setfit_fixtures && uv run python fetch_full_weights.py`
    ///
    /// Timed single-cell probe (run this FIRST — it is what the compute projection is built on):
    /// ```text
    /// CARGO_INCREMENTAL=0 APRENDER_CALIBRATION_PROBE=1 \
    ///   cargo test -p aprender-train --lib --features setfit production_calibration \
    ///   -- --ignored --nocapture
    /// ```
    ///
    /// The full boundary matrix (probe env var unset):
    /// ```text
    /// CARGO_INCREMENTAL=0 cargo test -p aprender-train --lib --features setfit \
    ///   production_calibration -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "18 full tuning passes on the 86.7 MB production checkout; run explicitly with --ignored (plan 05-01)"]
    #[allow(clippy::too_many_lines)]
    fn production_calibration_matrix() {
        let dir = production_checkout_dir();
        if !dir.join("full_manifest.json").is_file() {
            println!(
                "SKIP production_calibration_matrix: no pinned checkout at {} — set \
                 APRENDER_MINILM_DIR or run `cd scripts/setfit_fixtures && uv run python \
                 fetch_full_weights.py`",
                dir.display(),
            );
            return;
        }

        let mode = production_mode();
        let store = calibration_store();

        // D' SINGLE PASS: run one condition, persist it, stop. Handled before the report
        // machinery because a single condition supports no separation assertion and therefore
        // has no per-class table to render.
        if let ProductionMode::SinglePass(cell, condition) = mode {
            assert_disk_headroom(&store);
            let pass = production_pass(&dir, cell, condition.lr_override());
            eprintln!(
                "[progress] pass done: seed={} cell={} condition={} device={} steps={} \
                 wall_clock={:.1}s regime={}",
                cell.seed,
                cell.label(),
                condition.tag(),
                pass.device,
                pass.steps,
                pass.elapsed_secs,
                pass.regime,
            );
            assert!(
                pass.regime.contains(&format!("cells={}", cell.label())),
                "seed {} cell {}: rendered regime `{}` does not name the expected cell label",
                cell.seed,
                cell.label(),
                pass.regime,
            );
            persist_pass(&store, cell, condition, &pass);
            println!(
                "\nSTATUS: PARTIAL — one persisted pass (cell {} seed {} condition {}). No \
                 separation assertion has run; combine the cell's three conditions with \
                 APRENDER_CALIBRATION_COMBINE=\"s{}:{}\".",
                cell.label(),
                cell.seed,
                condition.tag(),
                cell.shots,
                cell.seed,
            );
            return;
        }

        let (cells, full_conditions, mode_name) = match mode {
            ProductionMode::Probe => (
                vec![ProductionCell { shots: PRODUCTION_SHOTS[0], seed: PRODUCTION_SEEDS[0] }],
                false,
                "PROBE (one cell, REAL only)",
            ),
            ProductionMode::Prospective(ref cells) => {
                (cells.clone(), false, "PROSPECTIVE VALIDATION (named cells, REAL only)")
            }
            ProductionMode::CellSubset(ref cells) => (
                cells.clone(),
                true,
                "CELL SUBSET (named cells, FULL 3 conditions — a resumable chunk of the matrix)",
            ),
            ProductionMode::BoundaryMatrix => {
                let mut out = Vec::with_capacity(PRODUCTION_SHOTS.len() * PRODUCTION_SEEDS.len());
                for &shots in &PRODUCTION_SHOTS {
                    for &seed in &PRODUCTION_SEEDS {
                        out.push(ProductionCell { shots, seed });
                    }
                }
                (out, true, "BOUNDARY MATRIX (2 cells x 3 seeds x 3 conditions)")
            }
            ProductionMode::Combine(ref cells) => (
                cells.clone(),
                true,
                "COMBINE (named cells, FULL 3 conditions loaded from the persisted store — no \
                 training)",
            ),
            ProductionMode::SinglePass(..) => unreachable!("handled above"),
        };

        // The ONE line that differs between measuring and combining. Everything downstream runs
        // on `ProductionPass` values and cannot tell which door they came through.
        let load_from_store = matches!(mode, ProductionMode::Combine(_));
        let obtain = |cell: ProductionCell, condition: Condition| -> ProductionPass {
            if load_from_store {
                load_pass(&store, cell, condition)
            } else {
                assert_disk_headroom(&store);
                let pass = production_pass(&dir, cell, condition.lr_override());
                // Persisted even on the in-process paths, so a chunk that dies later still
                // banks every pass it finished.
                persist_pass(&store, cell, condition, &pass);
                pass
            }
        };

        // The report file is resolved UP FRONT and rewritten after every cell, not written
        // once at the end.
        //
        // This is not belt-and-braces. The first attempt at the full matrix was killed
        // externally at pass 9 of 18, and because the report was only emitted after the final
        // pass, nine completed `run_tuning` passes produced no per-class numbers at all —
        // roughly 40 minutes of measurement lost to a signal, with nothing wrong in the code.
        // A long unattended job that keeps its findings only in memory is one interruption
        // away from having measured nothing.
        let destination = std::env::var("SETFIT_PRODUCTION_CALIBRATION_REPORT").map_or_else(
            |_| std::env::temp_dir().join("setfit-production-calibration.txt"),
            std::path::PathBuf::from,
        );
        let flush = |body: &str, complete: bool| {
            let banner = if complete {
                "STATUS: COMPLETE\n"
            } else {
                "STATUS: PARTIAL — this run had not finished when the file was written; the \
                 cells below are the ones that COMPLETED. Treat any absent cell as unmeasured, \
                 never as passing.\n"
            };
            let _ = std::fs::write(&destination, format!("{banner}{body}"));
        };

        let mut report = String::new();
        report.push_str(&format!("\nPRODUCTION CALIBRATION MODE: {mode_name}\n"));
        report.push_str(&format!("CHECKOUT: {}\n", dir.display()));
        report.push_str(&format!(
            "FROZEN PRODUCTION HYPERPARAMETERS: epochs={PRODUCTION_EPOCHS} \
             batch={PRODUCTION_BATCH} (pinned setfit 1.1.3 body defaults)\n",
        ));
        report.push_str(
            "\ncell             class                real_min      real_median   real_max      \
             ctrl_max      nnull_max     nnull_moved   noise_floor   support_frac  \
             all_moved\n",
        );

        let mut real_min_across: BTreeMap<&'static str, f64> = BTreeMap::new();
        let mut ctrl_max_across: BTreeMap<&'static str, f64> = BTreeMap::new();
        let mut median_max_across: BTreeMap<&'static str, f64> = BTreeMap::new();
        let mut noise_floor_across: BTreeMap<&'static str, f64> = BTreeMap::new();
        let mut near_null_max_across: BTreeMap<&'static str, f64> = BTreeMap::new();
        let mut near_null_moved_all: BTreeMap<&'static str, bool> = BTreeMap::new();
        let mut binding_param: BTreeMap<&'static str, String> = BTreeMap::new();
        let mut embedding_delta_min_across = f64::INFINITY;
        let mut embedding_delta_median_min_across = f64::INFINITY;
        let mut regimes: Vec<String> = Vec::new();
        let mut timing_rows: Vec<String> = Vec::new();
        let mut total_secs = 0.0_f64;
        let mut passes_run = 0_usize;

        for cell in &cells {
            let mut record = |name: &str, pass: &ProductionPass| {
                // STREAMED to stderr as each pass lands, not just accumulated into the
                // end-of-run report. The full matrix is an ~8-hour unattended job; if it dies
                // at pass 14 the operator must be able to say WHICH cells completed and at
                // what cost, rather than losing every measurement to one panic.
                eprintln!(
                    "[progress] pass done: seed={} cell={} condition={} device={} steps={} \
                     wall_clock={:.1}s regime={}",
                    cell.seed,
                    cell.label(),
                    name,
                    pass.device,
                    pass.steps,
                    pass.elapsed_secs,
                    pass.regime,
                );
                timing_rows.push(format!(
                    "  seed {:>3} cell {:<10} condition {:<10} device={:<8} steps={:<6} \
                     wall_clock={:.1}s\n",
                    cell.seed,
                    cell.label(),
                    name,
                    pass.device,
                    pass.steps,
                    pass.elapsed_secs,
                ));
            };

            let real = obtain(*cell, Condition::Real);
            passes_run += 1;
            total_secs += real.elapsed_secs;
            record("real", &real);

            let control = if full_conditions {
                let p = obtain(*cell, Condition::Control);
                passes_run += 1;
                total_secs += p.elapsed_secs;
                record("control", &p);
                Some(p)
            } else {
                None
            };
            let near_null = if full_conditions {
                let p = obtain(*cell, Condition::NearNull);
                passes_run += 1;
                total_secs += p.elapsed_secs;
                record("near-null", &p);
                Some(p)
            } else {
                None
            };

            // Every pass's OWN rendered id, recorded so the report PROVES all cells share one
            // architecture component rather than asserting it in prose.
            regimes.push(real.regime.clone());
            if let Some(p) = control.as_ref() {
                regimes.push(p.regime.clone());
            }
            if let Some(p) = near_null.as_ref() {
                regimes.push(p.regime.clone());
            }

            // The cell label the run RENDERED must be the one this cell claims to be — a
            // selection that drew a different number of rows would silently relabel the cell.
            assert!(
                real.regime.contains(&format!("cells={}", cell.label())),
                "seed {} cell {}: rendered regime `{}` does not name the expected cell label",
                cell.seed,
                cell.label(),
                real.regime,
            );

            embedding_delta_min_across =
                embedding_delta_min_across.min(real.evidence.embedding_delta_min);
            // The run-level `embedding_delta_floor` is derived from the smallest embedding-class
            // MEDIAN across measured cells, not the smallest MINIMUM — the contract's
            // embedding_delta_floor equation states the two are deliberately different questions
            // ("whether the table moved as a body"). Accumulated and printed so the frozen floor
            // is read off a RUN rather than recomputed by hand from the per-cell table.
            embedding_delta_median_min_across =
                embedding_delta_median_min_across.min(real.evidence.embedding_delta_median);

            for class in ParameterClass::ALL {
                let real_rows = real.evidence.rows_of_class(class);
                assert!(!real_rows.is_empty(), "{class} has no rows on the production encoder");

                let real_values: Vec<f64> = real_rows.iter().map(|r| r.relative_delta).collect();
                let support: Vec<f64> =
                    real_rows.iter().map(|r| r.delta_support_fraction).collect();
                let real_min = min_of(&real_values);
                let cell_noise_floor = max_of(
                    &real_rows.iter().map(|r| rounding_noise_floor(r)).collect::<Vec<f64>>(),
                );

                let control_max = control.as_ref().map(|p| {
                    let rows = p.evidence.rows_of_class(class);
                    assert_eq!(rows.len(), real_rows.len(), "{class}: control row count");
                    max_of(&rows.iter().map(|r| r.relative_delta).collect::<Vec<f64>>())
                });
                let (near_null_max, near_null_moved) =
                    near_null.as_ref().map_or((None, None), |p| {
                        let rows = p.evidence.rows_of_class(class);
                        assert_eq!(rows.len(), real_rows.len(), "{class}: near-null row count");
                        (
                            Some(max_of(
                                &rows.iter().map(|r| r.relative_delta).collect::<Vec<f64>>(),
                            )),
                            Some(rows.iter().all(|r| r.moved)),
                        )
                    });

                report.push_str(&format!(
                    "seed{:<3} {:<10} {:<20} {:<13.3e} {:<13.3e} {:<13.3e} {:<13} {:<13} \
                     {:<13} {:<13.3e} {:<13.4} {}\n",
                    cell.seed,
                    cell.label(),
                    class.tag(),
                    real_min,
                    median_of(&real_values),
                    max_of(&real_values),
                    control_max.map_or_else(|| "-".to_string(), |v| format!("{v:.3e}")),
                    near_null_max.map_or_else(|| "-".to_string(), |v| format!("{v:.3e}")),
                    near_null_moved.map_or_else(|| "-".to_string(), |v| v.to_string()),
                    cell_noise_floor,
                    median_of(&support),
                    real_rows.iter().all(|r| r.moved),
                ));

                // (c) SEPARATION, per class and per cell — the assertion whose failure means
                // the epsilon window does not exist and the D-04 checkpoint needs raw data
                // rather than a forced number. Asserted only where a control was run.
                if let Some(control_max) = control_max {
                    assert!(
                        control_max < real_min,
                        "seed {} cell {} class {}: the 1e-30 control's max relative delta \
                         ({control_max:e}) is not below the real run's min ({real_min:e})",
                        cell.seed,
                        cell.label(),
                        class.tag(),
                    );
                }

                let slot = real_min_across.entry(class.tag()).or_insert(f64::INFINITY);
                if real_min < *slot {
                    *slot = real_min;
                    let binding = real_rows
                        .iter()
                        .min_by(|a, b| {
                            a.relative_delta
                                .partial_cmp(&b.relative_delta)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map_or_else(String::new, |r| {
                            format!(
                                "{} (seed {} cell {}) delta_norm={:.3e} init_norm={:.3e} \
                                 grad_norm_max={:.3e} grad_norm_mean={:.3e} steps_observed={} \
                                 noise_floor={:.3e} support_frac={:.4}",
                                r.name,
                                cell.seed,
                                cell.label(),
                                r.delta_norm,
                                r.init_norm,
                                r.grad_norm_max,
                                r.grad_norm_mean,
                                r.steps_observed,
                                rounding_noise_floor(r),
                                r.delta_support_fraction,
                            )
                        });
                    binding_param.insert(class.tag(), binding);
                }
                let slot = noise_floor_across.entry(class.tag()).or_insert(f64::NEG_INFINITY);
                *slot = slot.max(cell_noise_floor);
                if let Some(v) = near_null_max {
                    let slot = near_null_max_across.entry(class.tag()).or_insert(f64::NEG_INFINITY);
                    *slot = slot.max(v);
                }
                if let Some(v) = near_null_moved {
                    let slot = near_null_moved_all.entry(class.tag()).or_insert(true);
                    *slot = *slot && v;
                }
                if let Some(v) = control_max {
                    let slot = ctrl_max_across.entry(class.tag()).or_insert(f64::NEG_INFINITY);
                    *slot = slot.max(v);
                }
                let slot = median_max_across.entry(class.tag()).or_insert(f64::NEG_INFINITY);
                *slot = slot.max(median_of(&real_values));
            }

            // Persist everything measured so far. A kill after this point costs at most the
            // cell in flight, never the cells already paid for.
            let mut partial = report.clone();
            partial.push_str("\nPASSES COMPLETED SO FAR\n");
            for row in &timing_rows {
                partial.push_str(row);
            }
            for id in &regimes {
                partial.push_str(&format!("  regime: {id}\n"));
            }
            flush(&partial, false);
            eprintln!(
                "[progress] cell {} seed {} complete; partial report written to {}",
                cell.label(),
                cell.seed,
                destination.display()
            );
        }

        // The architecture@revision component every cell rendered, under the SAME header the
        // fixture matrix prints — this is the line plan 05-03 copies its entry prefix from.
        let architecture = regimes
            .first()
            .and_then(|r| r.split('|').next())
            .map_or_else(String::new, str::to_string);
        report.push_str(&format!("\nCALIBRATION REGIME: {architecture}\n"));
        report.push_str("\nEVERY RENDERED REGIME ID (verbatim, one per pass)\n");
        for id in &regimes {
            report.push_str(&format!("  {id}\n"));
            assert!(
                id.starts_with(&architecture),
                "cells disagree on the architecture component: `{id}` vs `{architecture}`",
            );
        }

        report.push_str(
            "\nWALL CLOCK and RESOLVED DEVICE (stdout report only — never evidence fields)\n",
        );
        for row in &timing_rows {
            report.push_str(row);
        }
        report.push_str(&format!("  TOTAL: {total_secs:.1}s over {passes_run} passes\n"));

        report.push_str(&format!(
            "\nEMBEDDING DELTA MIN across measured cells: {embedding_delta_min_across:.3e}\n",
        ));
        report.push_str(&format!(
            "EMBEDDING DELTA MEDIAN MIN across measured cells: \
             {embedding_delta_median_min_across:.3e}\n  \
             (the run-level embedding_delta_floor's upper edge is this / 10, rounded DOWN to two \
             significant figures — the contract's embedding_delta_floor derivation)\n",
        ));

        // The verdict, held until the report has been written to disk (D-18).
        let mut basis_verdict: Option<Result<EpsilonBasis, Box<CollapsedWindows>>> = None;
        if full_conditions {
            // The header must say WHICH cells the basis covers, because the window rule is
            // defined across ALL measured cells and an unmeasured cell can still move
            // `best_real` and shrink every window below. A table headed "plan 05-03 freezes
            // from these" while half the boundary matrix is missing is an invitation to freeze
            // a number that the remaining cells would refute — which is exactly the false-green
            // this report exists to prevent. The banner and its missing-cell list are now
            // derived from the TYPED coverage value `epsilon_basis` returns.
            let full_matrix: Vec<ProductionCell> = PRODUCTION_SHOTS
                .iter()
                .flat_map(|&shots| {
                    PRODUCTION_SEEDS.iter().map(move |&seed| ProductionCell { shots, seed })
                })
                .collect();
            let measured_cells: Vec<String> =
                cells.iter().map(|c| format!("s{}:{}", c.shots, c.seed)).collect();
            let full_matrix_cells: Vec<String> =
                full_matrix.iter().map(|c| format!("s{}:{}", c.shots, c.seed)).collect();

            let mut aggregates: BTreeMap<&'static str, ClassAggregates> = BTreeMap::new();
            for class in ParameterClass::ALL {
                aggregates.insert(
                    class.tag(),
                    ClassAggregates {
                        worst_ctrl: ctrl_max_across.get(class.tag()).copied().unwrap_or(0.0),
                        worst_nnull: near_null_max_across.get(class.tag()).copied().unwrap_or(0.0),
                        best_real: real_min_across.get(class.tag()).copied().unwrap_or(0.0),
                        worst_median: median_max_across.get(class.tag()).copied().unwrap_or(0.0),
                        noise_floor: noise_floor_across.get(class.tag()).copied().unwrap_or(0.0),
                        near_null_moved_all: near_null_moved_all
                            .get(class.tag())
                            .copied()
                            .unwrap_or(false),
                    },
                );
            }

            // The regime the basis is DERIVED FOR, in the calibrated-entry grammar: the
            // architecture every pass rendered, plus every seed and every cell label this run
            // measured. That is the shape 05-03 would freeze as a table entry, so the lookup
            // below asks exactly the question the frozen table would have to answer.
            let mut seeds: Vec<String> = cells.iter().map(|c| c.seed.to_string()).collect();
            seeds.sort_unstable();
            seeds.dedup();
            let mut labels: Vec<String> = cells.iter().map(|c| c.label()).collect();
            labels.sort_unstable();
            labels.dedup();
            let derived_regime =
                format!("{architecture}|seeds={}|cells={}", seeds.join(","), labels.join(","));

            let verdict = epsilon_basis(
                &derived_regime,
                &aggregates,
                &measured_cells,
                &full_matrix_cells,
                PRODUCTION_LOWER_BOUND,
            );
            {
                let basis = match &verdict {
                    Ok(basis) => basis,
                    Err(collapse) => &collapse.basis,
                };
                report.push_str(&render_epsilon_basis(
                    basis,
                    "CROSS-CELL EPSILON BASIS over the COMPLETE boundary matrix (plan 05-03 \
                     freezes from these)",
                ));
            }

            // The D-04 decision input: every candidate lower bound, each derived by CALLING
            // `epsilon_basis` with that candidate's rule. Appended to the report before the
            // verdict is asserted, so the tables survive a refusal (D-18's "write the report
            // first" property) — which is the state this section was written to be read in.
            report.push_str(&render_candidate_lower_bounds(
                &derived_regime,
                &aggregates,
                &measured_cells,
                &full_matrix_cells,
            ));

            basis_verdict = Some(verdict);
        } else {
            report.push_str(
                "\n(no control conditions in this mode — no epsilon basis is derivable from \
                 this run, by construction)\n",
            );
        }

        report.push_str("\nWHAT BINDS EACH CLASS'S LOWEST REAL DELTA\n");
        for class in ParameterClass::ALL {
            report.push_str(&format!(
                "  {:<20} {}\n",
                class.tag(),
                binding_param.get(class.tag()).map_or("-", String::as_str),
            ));
        }

        assert!(passes_run > 0, "the matrix must run at least one pass");
        report.push_str(&format!("\nPASSES RUN: {passes_run}\n"));

        println!("{report}");
        // Final write, with the COMPLETE banner. Unlike the per-cell flushes this one is
        // allowed to fail loudly: a calibration whose numbers cannot be read is a calibration
        // that did not happen, and by this point every pass has already been paid for.
        std::fs::write(&destination, format!("STATUS: COMPLETE\n{report}")).unwrap_or_else(|e| {
            panic!("the production calibration report must be writable at {destination:?}: {e}")
        });
        println!("production calibration report written to {}", destination.display());

        // THE VERDICT, asserted only after the report has been written and its destination
        // announced. Before plan 05-14 an empty epsilon basis was REPORTED beside `rc=0`; the
        // four-cell combine printed `EMPTY` five times and exited zero (D-18 / 05-01 FINDING 3).
        if let Some(Err(collapse)) = basis_verdict {
            panic!("{collapse}");
        }
    }

    /// D''s load-bearing precondition: the same pass, run in two SEPARATE processes, must
    /// persist BIT-IDENTICAL evidence.
    ///
    /// D' moves `ctrl_max < real_min` from a comparison of values computed in one process to a
    /// comparison of values persisted from three. That the two are equivalent is exactly the
    /// kind of claim CLAUDE.md verification rule 4 forbids inheriting: widening a guard's scope
    /// requires re-proving it in the NEW scope, and the in-process proof does not transfer. If
    /// a fresh process could record even slightly different numbers — a different rayon
    /// reduction order, an unseeded map iteration, an ambient thread count reaching the
    /// arithmetic — then the persisted comparison would be measuring something the in-process
    /// one never measured, and every s64 number derived through it would be unfounded.
    ///
    /// This is a real cross-process test, not a same-process stand-in: it re-executes THIS test
    /// binary twice via `current_exe`, each child writing to its own store, and compares the
    /// bytes. A same-process double call would prove nothing about process boundaries, which is
    /// precisely where the doubt lives.
    ///
    /// `s8:13:real` is the cheapest pass in the matrix (~40 s), so the whole proof costs about
    /// two minutes — against the ~8 h it protects.
    #[test]
    #[ignore = "spawns two child processes running an s8 production pass each (~2 min); proves D' (plan 05-01)"]
    fn cross_process_determinism_of_persisted_evidence() {
        let dir = production_checkout_dir();
        if !dir.join("full_manifest.json").is_file() {
            println!(
                "SKIP cross_process_determinism_of_persisted_evidence: no pinned checkout at {}",
                dir.display(),
            );
            return;
        }

        let cell = ProductionCell { shots: PRODUCTION_SHOTS[0], seed: PRODUCTION_SEEDS[0] };
        let stem = pass_stem(cell, Condition::Real);
        let exe = std::env::current_exe().expect("the running test binary has a path");
        let base =
            std::env::temp_dir().join(format!("setfit-determinism-proof-{}", std::process::id()));

        // The FULL test path, derived rather than written out. `--exact` matches the whole
        // name, so the bare `production_calibration_matrix` filters to zero tests — and a child
        // that runs NOTHING still exits 0. That is the false-green this proof would be most
        // embarrassed by, so the child's own count is asserted below rather than inferred from
        // its exit status. Deriving from `module_path!()` also means a module rename cannot
        // quietly reintroduce the mismatch.
        let target = format!(
            "{}::production_calibration_matrix",
            module_path!().split_once("::").map_or(module_path!(), |(_, rest)| rest),
        );

        let run_child = |tag: &str| -> (Vec<u8>, PassMeta) {
            let store = base.join(tag);
            let out = std::process::Command::new(&exe)
                .args(["--exact", &target, "--ignored", "--nocapture"])
                .env("APRENDER_CALIBRATION_PASS", format!("s{}:{}:real", cell.shots, cell.seed))
                .env("APRENDER_CALIBRATION_STORE", &store)
                // The child must take its mode from PASS alone; an inherited mode variable
                // would silently run a different job than the one this proof describes.
                .env_remove("APRENDER_CALIBRATION_PROBE")
                .env_remove("APRENDER_CALIBRATION_CELLS")
                .env_remove("APRENDER_CALIBRATION_COMBINE")
                .env_remove("APRENDER_CALIBRATION_PROSPECTIVE")
                .output()
                .expect("the child test process spawns");
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            assert!(
                out.status.success(),
                "child pass `{tag}` did not succeed: {}\n--- stdout ---\n{stdout}\n\
                 --- stderr ---\n{stderr}",
                out.status,
            );
            // A child that filtered to zero tests exits 0 and measures nothing.
            assert!(
                stdout.contains("1 passed"),
                "child pass `{tag}` exited 0 but did not report exactly one test passing — it \
                 ran NOTHING and proved nothing. Filter was `{target}`.\n--- stdout ---\n{stdout}",
            );
            for line in
                stderr.lines().filter(|l| l.starts_with("[progress]") || l.starts_with("[persist]"))
            {
                println!("  child {tag}: {line}");
            }
            let evidence = std::fs::read(store.join(format!("{stem}.evidence.json")))
                .expect("the child persisted its evidence");
            let meta: PassMeta = serde_json::from_slice(
                &std::fs::read(store.join(format!("{stem}.meta.json")))
                    .expect("the child persisted its metadata"),
            )
            .expect("child metadata parses");
            (evidence, meta)
        };

        let (first, first_meta) = run_child("a");
        let (second, second_meta) = run_child("b");

        assert_eq!(
            first_meta.evidence_sha256, second_meta.evidence_sha256,
            "two fresh processes running {stem} recorded DIFFERENT evidence digests\n  \
             process a: {}\n  process b: {}\nD' is unsound on this host: the separation \
             assertion over persisted tables would not be the assertion the in-process path \
             makes. Do NOT widen a tolerance to hide this.",
            first_meta.evidence_sha256, second_meta.evidence_sha256,
        );
        assert_eq!(
            first, second,
            "two fresh processes running {stem} persisted evidence with equal digests but \
             unequal bytes — which would mean the digest is not injective over these tables",
        );

        // Non-vacuity: bytes that were empty, or a digest of nothing, would satisfy the
        // equalities above while proving nothing at all.
        assert!(
            first.len() > 1024,
            "the persisted evidence is implausibly small: {} bytes",
            first.len()
        );
        assert_eq!(first_meta.steps, second_meta.steps, "step counts differ across processes");
        assert!(first_meta.steps > 0, "a pass that took no optimizer steps proves nothing");

        println!(
            "\nD' CROSS-PROCESS DETERMINISM: PASS\n  pass:            {stem}\n  \
             evidence_sha256: {}\n  bytes:           {}\n  steps:           {}\n  \
             wall_clock:      {:.1}s / {:.1}s (differs by design; timing is NOT evidence)\n",
            first_meta.evidence_sha256,
            first.len(),
            first_meta.steps,
            first_meta.elapsed_secs,
            second_meta.elapsed_secs,
        );

        let _ = std::fs::remove_dir_all(&base);
    }
}
