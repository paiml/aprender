//! The D-08 classification request/response pair (OPS-04, OPS-06).
//!
//! ONE versioned envelope, owned by core, serialized byte-identically by every
//! surface: the Rust API, `apr classify --json` (04-07) and
//! `POST /v1/classify` (04-08). A surface-local response type would make "the
//! three surfaces agree" a claim about three independently-maintained structs;
//! here it is true by construction, because there is only one struct.
//!
//! # Why every field is private (review M1)
//!
//! The claim this module exists to make true is *"a non-finite value is
//! unrepresentable in a `ClassifyResponse`"*. A derived `Deserialize` over
//! public fields falsifies that claim twice over: any caller can build a value
//! with a struct literal, and any JSON body can produce one without passing the
//! validating constructor. So the fields are private with read accessors, the
//! only constructors validate, and BOTH serde directions route through a
//! PRIVATE wire struct via `#[serde(into = ..., try_from = ...)]` — the
//! `SetFitTrainConfig` precedent (aprender-train `config.rs:175-177`).
//!
//! The constructors are nonetheless `pub`, on purpose. The parity harness
//! (04-09) builds its in-band negative by constructing a response whose
//! probability is perturbed by 10x the contract tolerance — a FINITE value, and
//! therefore a legitimate construction. A test-only backdoor would have been a
//! shipped backdoor.
//!
//! # Why the request document is shared (review M2)
//!
//! [`ClassifyRequestDocument`] is the ONE document the CLI `--input` file and
//! the HTTP body both carry. The alternative — a line-delimited CLI format —
//! cannot represent a text containing a newline, so a whitespace probe would
//! arrive as two CLI texts and one HTTP text and the parity surfaces would
//! receive DIFFERENT ordered input sets while appearing to agree on everything
//! they did compare.

use serde::{Deserialize, Serialize};

/// The envelope's schema version.
///
/// Bumped only by a deliberate, `pv diff`-visible change to the field set. A
/// payload declaring any other version is a typed rejection rather than a
/// best-effort parse: silently accepting a v2 body as v1 is how a renamed field
/// becomes a missing field nobody notices.
pub const CLASSIFY_SCHEMA_VERSION: u32 = 1;

/// Contract bound `max_batch_texts` (`setfit-apr-v1` item 11).
///
/// Enforced in core, BEFORE tokenization: a batch bound checked after
/// tokenization is not a bound on the work an attacker can request (T-04-11).
pub const MAX_BATCH_TEXTS: usize = 256;

/// Contract bound `max_request_body_bytes` (`setfit-apr-v1` item 11), in bytes.
///
/// The sibling of [`MAX_BATCH_TEXTS`] in the same `bounds:` block, and it lives
/// here for the same reason: the request document is core-owned, so the number
/// every surface bounds a body against is core-owned too. Left in the contract
/// alone, the CLI's `--input` reader and the HTTP body extractor would each pick
/// their own, and "the two surfaces received the SAME ordered text set" would
/// stop being true at exactly the size where it matters.
///
/// It is a TRANSPORT bound and cannot be enforced here: this module never sees
/// bytes, only an already-parsed [`ClassifyRequestDocument`]. The surface that
/// reads the payload is the one that must apply it, BEFORE deserializing —
/// [`MAX_BATCH_TEXTS`] is checked after the document exists, so it bounds the
/// tokenization but not the parse.
///
/// # Enforcement is DEFERRED to the reading surface, and that is recorded, not implied
///
/// `setfit-apr-v1` is the only contract in `contracts/` that names a request-body
/// bound, and no crate in this workspace enforces one today — there is no
/// `DefaultBodyLimit`, no body-limit layer, and no sibling obligation to match.
/// So this constant does not inherit an established pattern; it ESTABLISHES the
/// number, so that when `apr classify --input` (04-07) and `POST /v1/classify`
/// (04-08) land they apply the same one instead of each picking their own. The
/// bound biting on a real oversized payload is those plans' obligation and is
/// deliberately not claimed here.
pub const MAX_REQUEST_BODY_BYTES: u64 = 1_048_576;

/// Absolute tolerance on one result row's probability mass.
///
/// Absolute and not relative: probabilities live in `[0, 1]`, and a relative
/// bound near zero would be unboundedly strict.
pub const PROBABILITY_MASS_ABS_TOLERANCE: f64 = 1e-6;

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

/// Everything the classify path and the envelope's invariants can refuse.
///
/// Every variant is small (no boxed payload is needed) and the enum is
/// `#[non_exhaustive]`, so 04-07/04-08/04-09 match with a wildcard arm and a
/// later variant is not a breaking change.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ClassifyError {
    /// The request carried zero texts.
    EmptyInput,
    /// The request carried more texts than [`MAX_BATCH_TEXTS`] allows.
    BatchTooLarge {
        /// The contract bound.
        max: usize,
        /// What the request asked for.
        got: usize,
    },
    /// A non-finite value reached a response field.
    NonFiniteResponse {
        /// Which field: `probabilities`, `logits`, `margin` or `latency_ms`.
        field: &'static str,
    },
    /// `latency_ms` was negative. Zero is legal — see [`ClassifyResponse::latency_ms`].
    NegativeLatency {
        /// The observed value.
        got: f64,
    },
    /// A probability row did not sum to 1 within [`PROBABILITY_MASS_ABS_TOLERANCE`].
    ProbabilityMassOutOfRange {
        /// The observed mass.
        mass: f64,
    },
    /// A vector's arity disagreed with the label count.
    LabelCountMismatch {
        /// The arity every row must have.
        expected: usize,
        /// The arity observed.
        got: usize,
    },
    /// A payload declared a schema version this build does not implement.
    UnsupportedSchemaVersion {
        /// [`CLASSIFY_SCHEMA_VERSION`].
        expected: u32,
        /// What the payload declared.
        got: u32,
    },
    /// Tokenization or the encoder forward pass failed.
    EncodeFailed {
        /// The typed error's rendering, naming what failed.
        reason: String,
    },
    /// The classifier head refused the embeddings it was handed.
    HeadFailed {
        /// The typed error's rendering, naming what failed.
        reason: String,
    },
}

impl std::fmt::Display for ClassifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "the request carried zero texts"),
            Self::BatchTooLarge { max, got } => {
                write!(f, "the request carried {got} texts; the bound is {max}")
            }
            Self::NonFiniteResponse { field } => {
                write!(f, "a non-finite value reached the `{field}` field")
            }
            Self::NegativeLatency { got } => write!(f, "latency_ms is {got}, which is negative"),
            Self::ProbabilityMassOutOfRange { mass } => write!(
                f,
                "a result row has probability mass {mass}, which is not 1 within \
                 {PROBABILITY_MASS_ABS_TOLERANCE}"
            ),
            Self::LabelCountMismatch { expected, got } => {
                write!(f, "expected {expected} entries per row, observed {got}")
            }
            Self::UnsupportedSchemaVersion { expected, got } => write!(
                f,
                "the payload declares classify schema version {got}; this build implements \
                 {expected}"
            ),
            Self::EncodeFailed { reason } => write!(f, "the encode step failed: {reason}"),
            Self::HeadFailed { reason } => write!(f, "the classifier head failed: {reason}"),
        }
    }
}

impl std::error::Error for ClassifyError {}

// ---------------------------------------------------------------------------
// The request document (review M2)
// ---------------------------------------------------------------------------

/// The ONE request document every surface parses.
///
/// `deny_unknown_fields`, so a key this schema does not model is a rejection
/// rather than a silently ignored knob. Fields are public because this is an
/// INPUT document: it carries no invariant a caller could break, and the bounds
/// that matter ([`MAX_BATCH_TEXTS`], non-emptiness) are enforced by
/// [`crate::setfit::artifact::VerifiedSetFitModel::classify`] where the work
/// would otherwise happen.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClassifyRequestDocument {
    /// The ordered texts to classify. Order is response order.
    pub texts: Vec<String>,
    /// Whether the response should carry per-class logits.
    ///
    /// Defaults to `false`, so a body that omits it is legal and means "no
    /// logits" rather than "missing field".
    #[serde(default)]
    pub include_logits: bool,
}

impl ClassifyRequestDocument {
    /// A request for `texts`, without logits.
    #[must_use]
    pub fn new<I, S>(texts: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            texts: texts.into_iter().map(Into::into).collect(),
            include_logits: false,
        }
    }

    /// The same request, with per-class logits requested.
    #[must_use]
    pub fn with_logits(mut self) -> Self {
        self.include_logits = true;
        self
    }
}

// ---------------------------------------------------------------------------
// One text's result
// ---------------------------------------------------------------------------

/// One text's classification, in input order.
///
/// Every field is private; [`Self::new`] is the only constructor and it
/// validates. `Deserialize` routes through the same validation via the private
/// [`ClassifyResultWire`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(into = "ClassifyResultWire", try_from = "ClassifyResultWire")]
pub struct ClassifyResult {
    label: String,
    probabilities: Vec<f64>,
    logits: Option<Vec<f64>>,
    margin: f64,
    token_count: u32,
    truncated: bool,
}

impl ClassifyResult {
    /// The validating constructor — the only way a `ClassifyResult` exists.
    ///
    /// # Errors
    ///
    /// [`ClassifyError::NonFiniteResponse`] naming `probabilities`, `logits` or
    /// `margin`; [`ClassifyError::LabelCountMismatch`] when a logit vector's
    /// arity differs from the probability vector's;
    /// [`ClassifyError::ProbabilityMassOutOfRange`] when the row does not sum to
    /// 1 within [`PROBABILITY_MASS_ABS_TOLERANCE`].
    pub fn new(
        label: String,
        probabilities: Vec<f64>,
        logits: Option<Vec<f64>>,
        margin: f64,
        token_count: u32,
        truncated: bool,
    ) -> Result<Self, ClassifyError> {
        if probabilities.iter().any(|p| !p.is_finite()) {
            return Err(ClassifyError::NonFiniteResponse {
                field: "probabilities",
            });
        }
        if let Some(values) = logits.as_ref() {
            if values.iter().any(|l| !l.is_finite()) {
                return Err(ClassifyError::NonFiniteResponse { field: "logits" });
            }
            if values.len() != probabilities.len() {
                return Err(ClassifyError::LabelCountMismatch {
                    expected: probabilities.len(),
                    got: values.len(),
                });
            }
        }
        if !margin.is_finite() {
            return Err(ClassifyError::NonFiniteResponse { field: "margin" });
        }
        // Mass LAST, and NaN-visible: the finiteness checks above already
        // excluded NaN, so this is defense in depth rather than the primary
        // guard — but it is written so that it would still refuse a NaN if a
        // future refactor moved it ahead of them.
        let mass: f64 = probabilities.iter().sum();
        if !within((mass - 1.0).abs(), PROBABILITY_MASS_ABS_TOLERANCE) {
            return Err(ClassifyError::ProbabilityMassOutOfRange { mass });
        }
        Ok(Self {
            label,
            probabilities,
            logits,
            margin,
            token_count,
            truncated,
        })
    }

    /// The winning label, compared EXACTLY by every downstream gate.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// The full ordered probability vector, one entry per ordered label.
    #[must_use]
    pub fn probabilities(&self) -> &[f64] {
        &self.probabilities
    }

    /// The per-class logits, when the request asked for them.
    #[must_use]
    pub fn logits(&self) -> Option<&[f64]> {
        self.logits.as_deref()
    }

    /// Top-1 minus top-2 probability.
    #[must_use]
    pub fn margin(&self) -> f64 {
        self.margin
    }

    /// The number of positions the model actually CONSUMED for this text.
    ///
    /// The attention-mask length after truncation (contract item 11), read off
    /// the tokenized batch — NOT a tokenizer-internal count, which varies with
    /// whether special tokens are counted. Under this definition a truncated
    /// text has `token_count == MAX_SEQUENCE_LENGTH` by construction.
    #[must_use]
    pub fn token_count(&self) -> u32 {
        self.token_count
    }

    /// Whether the input exceeded the pinned truncation bound.
    #[must_use]
    pub fn truncated(&self) -> bool {
        self.truncated
    }
}

/// [`ClassifyResult`]'s private wire form.
///
/// `deny_unknown_fields` rejects unknown KEYS; it says nothing about invalid
/// VALUES — that is what [`ClassifyResult::new`] is for, and routing through it
/// is this type's whole purpose. Field order here IS the serialized order
/// (`serde_json` emits declaration order), so the golden bytes are stable.
///
/// `logits` carries NO `skip_serializing_if`: an unrequested logit vector
/// serializes as an explicit `null`. A missing key is invisible both to a
/// reviewer's diff and to a null-walking guard; an explicit null is loud to
/// both. The adjacent artifact sub-documents are proscribed from the attribute
/// for the same reason.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ClassifyResultWire {
    label: String,
    probabilities: Vec<f64>,
    logits: Option<Vec<f64>>,
    margin: f64,
    token_count: u32,
    truncated: bool,
}

impl From<ClassifyResult> for ClassifyResultWire {
    fn from(value: ClassifyResult) -> Self {
        Self {
            label: value.label,
            probabilities: value.probabilities,
            logits: value.logits,
            margin: value.margin,
            token_count: value.token_count,
            truncated: value.truncated,
        }
    }
}

impl TryFrom<ClassifyResultWire> for ClassifyResult {
    type Error = ClassifyError;

    fn try_from(wire: ClassifyResultWire) -> Result<Self, Self::Error> {
        Self::new(
            wire.label,
            wire.probabilities,
            wire.logits,
            wire.margin,
            wire.token_count,
            wire.truncated,
        )
    }
}

// ---------------------------------------------------------------------------
// The envelope
// ---------------------------------------------------------------------------

/// The classification envelope OPS-04 specifies.
///
/// Every field is private; [`Self::new`] is the only constructor and it
/// validates. `Deserialize` routes through the same validation via the private
/// [`ClassifyResponseWire`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(into = "ClassifyResponseWire", try_from = "ClassifyResponseWire")]
pub struct ClassifyResponse {
    schema_version: u32,
    artifact_sha256: String,
    backend: String,
    latency_ms: f64,
    results: Vec<ClassifyResult>,
}

/// EQUALITY EXCLUDES `latency_ms`, because the contract says it must.
///
/// `classify_response_schema`'s postcondition is that `latency_ms` "participates
/// in no equality comparison", and its invariant repeats that it "is EXCLUDED
/// from every equality comparison across surfaces". A derived `PartialEq`
/// silently made both false: `==` compared two wall-clock measurements, so the
/// obvious `assert_eq!(from_cli, from_http)` in a parity gate was a guaranteed
/// red for a reason that has nothing to do with the model, and the non-obvious
/// workaround puts the exclusion rule back in every harness that compares — which
/// is exactly where a rule goes to be applied inconsistently.
///
/// The rule now lives in ONE place, on the type that owns it. Everything a
/// response asserts about the model — schema version, artifact identity, backend
/// and every result row — is compared; the one field that is a MEASUREMENT is not.
/// A caller that genuinely wants to inspect the timing reads
/// [`ClassifyResponse::latency_ms`], which is the honest way to ask about it.
///
/// `Eq` is deliberately NOT implemented: [`ClassifyResult`] carries `f64`
/// probabilities, so the relation is not reflexive in general even though the
/// validating constructor excludes `NaN` today.
impl PartialEq for ClassifyResponse {
    fn eq(&self, other: &Self) -> bool {
        self.schema_version == other.schema_version
            && self.artifact_sha256 == other.artifact_sha256
            && self.backend == other.backend
            && self.results == other.results
    }
}

impl ClassifyResponse {
    /// The validating constructor — the only way a `ClassifyResponse` exists.
    ///
    /// `schema_version` is not a parameter: it is [`CLASSIFY_SCHEMA_VERSION`],
    /// so no caller can mint a response claiming a version this build does not
    /// implement.
    ///
    /// # Errors
    ///
    /// [`ClassifyError::NonFiniteResponse`] for a non-finite `latency_ms`,
    /// [`ClassifyError::NegativeLatency`] for a negative one, and
    /// [`ClassifyError::LabelCountMismatch`] when two result rows disagree on
    /// their probability arity — two rows of different width describe two
    /// different label sets, which no single response can be about.
    pub fn new(
        artifact_sha256: String,
        backend: String,
        latency_ms: f64,
        results: Vec<ClassifyResult>,
    ) -> Result<Self, ClassifyError> {
        if !latency_ms.is_finite() {
            return Err(ClassifyError::NonFiniteResponse {
                field: "latency_ms",
            });
        }
        if latency_ms < 0.0 {
            return Err(ClassifyError::NegativeLatency { got: latency_ms });
        }
        if let Some(first) = results.first() {
            let expected = first.probabilities.len();
            for result in &results {
                if result.probabilities.len() != expected {
                    return Err(ClassifyError::LabelCountMismatch {
                        expected,
                        got: result.probabilities.len(),
                    });
                }
            }
        }
        Ok(Self {
            schema_version: CLASSIFY_SCHEMA_VERSION,
            artifact_sha256,
            backend,
            latency_ms,
            results,
        })
    }

    /// The envelope's schema version.
    #[must_use]
    pub fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// The artifact this response was produced from.
    ///
    /// Present on every response from every surface, so Phase 5 can attribute a
    /// measurement to an artifact without trusting the caller's bookkeeping.
    #[must_use]
    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }

    /// The execution-derived backend identity (D-12).
    ///
    /// Produced by the encode invocation that RAN — see
    /// [`crate::setfit::encoder::ExecutionBackend`]. There is no parameter, no
    /// setter and no configuration path that reaches this field.
    #[must_use]
    pub fn backend(&self) -> &str {
        &self.backend
    }

    /// Wall-clock milliseconds around the compute.
    ///
    /// A MEASUREMENT, not an identity: it is excluded from every cross-surface
    /// equality comparison, and NO gate may assert it is strictly positive. A
    /// fast operation under a coarse timer legitimately reports `0.0`, so a
    /// `> 0` assertion is a flake with a schedule.
    #[must_use]
    pub fn latency_ms(&self) -> f64 {
        self.latency_ms
    }

    /// The per-text results, in input order.
    #[must_use]
    pub fn results(&self) -> &[ClassifyResult] {
        &self.results
    }
}

/// [`ClassifyResponse`]'s private wire form. See [`ClassifyResultWire`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ClassifyResponseWire {
    schema_version: u32,
    artifact_sha256: String,
    backend: String,
    latency_ms: f64,
    results: Vec<ClassifyResultWire>,
}

impl From<ClassifyResponse> for ClassifyResponseWire {
    fn from(value: ClassifyResponse) -> Self {
        Self {
            schema_version: value.schema_version,
            artifact_sha256: value.artifact_sha256,
            backend: value.backend,
            latency_ms: value.latency_ms,
            results: value.results.into_iter().map(Into::into).collect(),
        }
    }
}

impl TryFrom<ClassifyResponseWire> for ClassifyResponse {
    type Error = ClassifyError;

    fn try_from(wire: ClassifyResponseWire) -> Result<Self, Self::Error> {
        if wire.schema_version != CLASSIFY_SCHEMA_VERSION {
            return Err(ClassifyError::UnsupportedSchemaVersion {
                expected: CLASSIFY_SCHEMA_VERSION,
                got: wire.schema_version,
            });
        }
        let results = wire
            .results
            .into_iter()
            .map(ClassifyResult::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(wire.artifact_sha256, wire.backend, wire.latency_ms, results)
    }
}

// ---------------------------------------------------------------------------
// The one comparison helper
// ---------------------------------------------------------------------------

/// The NaN-visible tolerance comparator, defined ONCE for the crate in
/// [`crate::setfit::artifact::within`].
///
/// A second copy here would be a second comparator: same five lines today, and
/// nothing forcing them to stay the same tomorrow. Since divergence between a
/// load-time and a classify-time tolerance check would be silent, the definition
/// lives in one place and `within_is_nan_visible_in_both_argument_positions`
/// scans that one place for the `partial_cmp` form.
use crate::setfit::artifact::within;

// ---------------------------------------------------------------------------
// The one classification path (OPS-04)
// ---------------------------------------------------------------------------

impl crate::setfit::artifact::VerifiedSetFitModel {
    /// Classify an ordered batch of texts.
    ///
    /// A method on the VERIFIED typestate only, so the door stays single: every
    /// response this repository can produce came from a model that passed all
    /// seven rungs of the load ladder, including probe replay.
    ///
    /// # The backend field cannot be misreported
    ///
    /// `backend` is [`crate::setfit::encoder::ExecutionBackend::identity`]
    /// called on the value the encode invocation RETURNED. This method never
    /// names the kernel constant,
    /// takes no backend or device parameter, and reads no configuration: a
    /// reader who wants to change the reported value must change the encode
    /// path, which is exactly the point (D-12, review B6).
    ///
    /// # Errors
    ///
    /// [`ClassifyError::EmptyInput`] for a request with no texts;
    /// [`ClassifyError::BatchTooLarge`] above [`MAX_BATCH_TEXTS`] — both checked
    /// BEFORE tokenization, because a bound applied after the work is not a
    /// bound on the work; [`ClassifyError::EncodeFailed`] and
    /// [`ClassifyError::HeadFailed`] for the two fallible compute steps; and
    /// anything the envelope's validating constructors refuse.
    pub fn classify(
        &self,
        request: &ClassifyRequestDocument,
    ) -> Result<ClassifyResponse, ClassifyError> {
        // (1) Bounds FIRST, before a single token is produced (T-04-11).
        if request.texts.is_empty() {
            return Err(ClassifyError::EmptyInput);
        }
        if request.texts.len() > MAX_BATCH_TEXTS {
            return Err(ClassifyError::BatchTooLarge {
                max: MAX_BATCH_TEXTS,
                got: request.texts.len(),
            });
        }

        let started = std::time::Instant::now();
        let borrowed: Vec<&str> = request.texts.iter().map(String::as_str).collect();

        // (2) Tokenize ONCE. The token facts below are read off THIS batch — the
        //     very one the encoder consumes in step 3 — rather than recomputed
        //     from the input strings, which would describe a different call.
        let batch =
            self.model()
                .tokenize_batch(&borrowed)
                .map_err(|e| ClassifyError::EncodeFailed {
                    reason: e.to_string(),
                })?;

        // (3) Encode, keeping the identity the invocation hands back.
        let (pooled, backend) =
            self.model()
                .encode_batch_traced(&batch)
                .map_err(|e| ClassifyError::EncodeFailed {
                    reason: e.to_string(),
                })?;
        let features = embedding_rows(&pooled, request.texts.len())?;

        // (4) The head, through its SINGLE logit implementation. Probabilities
        //     are that same computation plus the head's own softmax, so the two
        //     reported vectors cannot disagree with each other.
        let logit_rows =
            self.head()
                .predict_logits(&features)
                .map_err(|e| ClassifyError::HeadFailed {
                    reason: e.to_string(),
                })?;

        let labels = self.ordered_labels();
        let mut results = Vec::with_capacity(logit_rows.len());
        // `logit_rows` is owned and dead after this loop, so each row MOVES into the
        // result rather than being cloned into it — one fewer allocation per text on
        // the per-request path.
        for (row, logits) in logit_rows.into_iter().enumerate() {
            let mut probabilities = vec![0.0_f64; logits.len()];
            crate::classification::multinomial::softmax_into(&logits, &mut probabilities);

            // Ties break to the LOWEST index, the same rule the head's own
            // `predict` uses — so the API and the head can never name different
            // winners for the same row.
            let winner = crate::classification::multinomial::argmax_lowest_index(&probabilities);
            let label = labels
                .get(winner)
                .ok_or(ClassifyError::LabelCountMismatch {
                    expected: labels.len(),
                    got: probabilities.len(),
                })?
                .clone();

            // Top-1 minus top-2, in one pass. With fewer than two labels `top2`
            // stays -inf and the margin is non-finite, which the envelope's
            // constructor refuses — fail-closed rather than a fabricated 0.0.
            let mut top1 = f64::NEG_INFINITY;
            let mut top2 = f64::NEG_INFINITY;
            for &p in &probabilities {
                if p > top1 {
                    top2 = top1;
                    top1 = p;
                } else if p > top2 {
                    top2 = p;
                }
            }

            results.push(ClassifyResult::new(
                label,
                probabilities,
                if request.include_logits {
                    Some(logits)
                } else {
                    None
                },
                top1 - top2,
                consumed_positions(&batch, row)?,
                // A MISSING fact is a refusal, not a `false`. `is_some_and` here
                // reported "not truncated" for a batch whose per-input facts were
                // short — the one shape in which the answer is unknown — while the
                // adjacent `consumed_positions` refused the same batch. Two
                // neighbouring reads of the same batch must not disagree about
                // whether an absent row is an error.
                truncation_fact(&batch, row)?,
            )?);
        }

        // (5) A MEASUREMENT, taken around the compute. Excluded from every
        //     cross-surface comparison; no gate may assert it is > 0.
        let latency_ms = started.elapsed().as_secs_f64() * 1000.0;

        ClassifyResponse::new(
            self.artifact_sha256().to_string(),
            // The identity travels back from step 3. There is no other
            // expression in this function that could produce this string.
            backend.identity(),
            latency_ms,
            results,
        )
    }
}

/// Whether row `row` hit the pinned truncation bound.
///
/// Reported from the batch's per-input facts, and a row the batch does not carry
/// is a typed refusal rather than a `false`: `truncated` is contract-normative
/// (gates assert it TOGETHER with `token_count == MAX_SEQUENCE_LENGTH`), and a
/// fabricated `false` would make a truncated text look untruncated.
fn truncation_fact(
    batch: &crate::setfit::tokenizer::SentenceBatch,
    row: usize,
) -> Result<bool, ClassifyError> {
    batch
        .truncation()
        .get(row)
        .map(|fact| fact.truncated)
        .ok_or_else(|| ClassifyError::EncodeFailed {
            reason: format!(
                "row {row} has no truncation fact in a batch carrying {}",
                batch.truncation().len()
            ),
        })
}

/// The number of positions the model actually CONSUMED for row `row`.
///
/// The count of kept entries in that row's attention mask — the contract's
/// definition of `token_count`. Read off the encoded batch, never recomputed
/// from the input text: a tokenizer-internal count varies with whether special
/// tokens are counted, and two surfaces disagreeing about that would make the
/// field mean different things in the same schema.
fn consumed_positions(
    batch: &crate::setfit::tokenizer::SentenceBatch,
    row: usize,
) -> Result<u32, ClassifyError> {
    let seq = batch.seq();
    let start = row.saturating_mul(seq);
    let end = start.saturating_add(seq);
    // `get`, not a slice index: this path's contract is that it does not panic.
    let mask =
        batch
            .attention_mask()
            .get(start..end)
            .ok_or_else(|| ClassifyError::EncodeFailed {
                reason: format!(
                    "row {row} spans {start}..{end} of a {}-element attention mask",
                    batch.attention_mask().len()
                ),
            })?;
    let kept = mask.iter().filter(|&&m| m != 0).count();
    u32::try_from(kept).map_err(|_| ClassifyError::EncodeFailed {
        reason: format!("row {row} consumed {kept} positions, which does not fit a u32"),
    })
}

/// `[B, H]` -> one owned `f32` row per text, with the row count checked.
///
/// Separate from [`crate::setfit::artifact::VerifiedSetFitModel::embed`]'s
/// equivalent because that method deliberately re-encodes through the UNTRACED
/// path; this one works on the tensor the traced encode already returned, which
/// is what ties the reported backend to these exact embeddings.
fn embedding_rows(
    pooled: &crate::autograd::Tensor,
    expected_rows: usize,
) -> Result<Vec<Vec<f32>>, ClassifyError> {
    // The shape rule lives once, in artifact.rs. What differs between `embed` and
    // `classify` is which encode produced the tensor, not how a `[B, H]` tensor is
    // split — so only the refusal TYPE is chosen here.
    crate::setfit::artifact::split_embedding_rows(pooled, expected_rows)
        .map_err(|reason| ClassifyError::EncodeFailed { reason })
}

// ===========================================================================
// Test modules
//
// They live DIRECTLY in this file, not behind a `#[path]` include, because the
// three task filters name `setfit::classify::envelope`,
// `setfit::classify::backend` and `setfit::classify::classify_path`. A
// `#[path = "classify_tests.rs"] mod classify_tests;` wrapper would insert a
// segment and every one of those filters would select ZERO tests and still exit
// 0 — the CR-02 vacuous pass that bit plan 04-13's `bundle_nullable` filter.
// ===========================================================================

/// This file's PRODUCTION source, with the test modules cut off.
///
/// A source assertion must not scan its own needle. A guard greping for
/// `pub struct ...Wire` whose own text contains that literal can NEVER turn
/// green, and one greping for an attribute it also quotes passes vacuously on
/// its own text. Cutting at the first test module removes both failure modes at
/// once, and it is the same class of defect as orchestrator note F-05 (a
/// `skip_serializing_if` gate turning red on its own documentation) — which this
/// module's first run reproduced exactly.
#[cfg(test)]
fn production_source() -> &'static str {
    const SRC: &str = include_str!("classify.rs");
    // The cut is the banner ABOVE this function, so this function and all three
    // test modules fall outside it. `find` returns the FIRST occurrence, which
    // is the banner — not this literal, which lives below it.
    // `production_source_excludes_the_test_modules` pins that down rather than
    // assuming it.
    let cut = SRC.find("// Test modules").unwrap_or(SRC.len());
    &SRC[..cut]
}

/// The ONE model every classify suite runs against.
///
/// Built from `artifact::fixture`'s view through the REAL writer and the REAL
/// fail-closed loader, so these suites exercise a `VerifiedSetFitModel` that
/// passed all seven rungs — not a hand-assembled stand-in that skipped them.
#[cfg(test)]
fn fixture_verified_model() -> crate::setfit::artifact::VerifiedSetFitModel {
    let view = crate::setfit::artifact::fixture::fixture_view_full_pin_shape();
    let bytes =
        crate::setfit::artifact::write_setfit_apr(&view).expect("the fixture view is writable");
    crate::setfit::artifact::load_setfit_apr(&bytes).expect("the fixture artifact verifies")
}

/// The same fixture's encoder half, for the suites that need `SetFitMiniLm`
/// directly rather than through the verified typestate.
#[cfg(test)]
fn fixture_encoder_model() -> crate::setfit::SetFitMiniLm {
    let view = crate::setfit::artifact::fixture::fixture_view_full_pin_shape();
    crate::setfit::SetFitMiniLm::from_bundle_parts(
        &view.tokenizer_bytes,
        &view.architecture,
        view.tensors.clone(),
        view.root_seed,
    )
    .expect("the fixture parts rebuild a model")
}

/// Task 1: the envelope's invariants, on every path in and out.
#[cfg(test)]
mod envelope {
    use super::*;

    /// The exact bytes a fixture response serializes to.
    ///
    /// A committed LITERAL, not a re-serialization of the same struct: comparing
    /// a struct against itself proves only that serde is deterministic. This
    /// compares against bytes a human reviewed, so any field rename, reorder or
    /// removal is a loud diff in the D-08 schema.
    const GOLDEN_RESPONSE_JSON: &str = concat!(
        r#"{"schema_version":1,"#,
        r#""artifact_sha256":"9f2c7a1d4e8b60315a7c9e0d2f4b6813a5c7e9f1b3d50729468a0c2e4f6a8b1d","#,
        r#""backend":"cpu:setfit-core:fixture-kernel","#,
        r#""latency_ms":0.0,"#,
        r#""results":["#,
        r#"{"label":"positive","probabilities":[0.25,0.75],"logits":null,"#,
        r#""margin":0.5,"token_count":7,"truncated":false},"#,
        r#"{"label":"negative","probabilities":[0.875,0.125],"logits":null,"#,
        r#""margin":0.75,"token_count":5,"truncated":false}"#,
        r#"]}"#,
    );

    const GOLDEN_SHA: &str = "9f2c7a1d4e8b60315a7c9e0d2f4b6813a5c7e9f1b3d50729468a0c2e4f6a8b1d";

    /// A FIXTURE backend value, deliberately NOT the real v1 identity.
    ///
    /// This golden pins the SCHEMA — field names, declaration order, exact bytes
    /// — not the identity. Writing the real `<device>:setfit-core:<kernel>`
    /// value here would put the kernel literal in this file, and the D-12 gate
    /// requires that literal to live in `encoder.rs` and nowhere else: the
    /// identity must arrive as a VALUE returned by the encode call, never as a
    /// string this module knows how to spell.
    const GOLDEN_BACKEND: &str = "cpu:setfit-core:fixture-kernel";

    fn golden_response() -> ClassifyResponse {
        let a = ClassifyResult::new("positive".into(), vec![0.25, 0.75], None, 0.5, 7, false)
            .expect("fixture result a is valid");
        let b = ClassifyResult::new("negative".into(), vec![0.875, 0.125], None, 0.75, 5, false)
            .expect("fixture result b is valid");
        ClassifyResponse::new(GOLDEN_SHA.into(), GOLDEN_BACKEND.into(), 0.0, vec![a, b])
            .expect("fixture response is valid")
    }

    fn one_result(probabilities: Vec<f64>) -> Result<ClassifyResult, ClassifyError> {
        ClassifyResult::new("positive".into(), probabilities, None, 0.5, 7, false)
    }

    #[test]
    fn production_source_excludes_the_test_modules() {
        // The cut itself is load-bearing for five assertions below, so it is
        // asserted rather than assumed.
        let src = production_source();
        assert!(
            src.contains("pub struct ClassifyResponse {"),
            "the production half must still contain the declarations being scanned"
        );
        assert!(
            !src.contains("fn production_source"),
            "the cut must remove this test module, or every source assertion scans its own text"
        );
    }

    #[test]
    fn serialized_response_carries_exactly_the_contract_field_names() {
        let json = serde_json::to_string(&golden_response()).expect("serializes");
        let value: serde_json::Value = serde_json::from_str(&json).expect("reparses");
        let obj = value.as_object().expect("an object");
        let mut keys: Vec<&str> = obj.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            vec![
                "artifact_sha256",
                "backend",
                "latency_ms",
                "results",
                "schema_version"
            ],
            "the response's field set is the D-08 contract's field set"
        );

        let results = obj["results"].as_array().expect("results is an array");
        assert!(
            !results.is_empty(),
            "the fixture carries results to inspect"
        );
        for result in results {
            let robj = result.as_object().expect("a result object");
            let mut rkeys: Vec<&str> = robj.keys().map(String::as_str).collect();
            rkeys.sort_unstable();
            assert_eq!(
                rkeys,
                vec![
                    "label",
                    "logits",
                    "margin",
                    "probabilities",
                    "token_count",
                    "truncated"
                ],
                "each result's field set is the D-08 contract's field set"
            );
        }
    }

    #[test]
    fn serialized_response_matches_the_committed_golden_bytes() {
        let json = serde_json::to_string(&golden_response()).expect("serializes");
        assert_eq!(
            json, GOLDEN_RESPONSE_JSON,
            "the serialized envelope drifted from the committed golden"
        );
    }

    #[test]
    fn constructor_rejects_a_non_finite_probability() {
        let err = ClassifyResult::new("positive".into(), vec![f64::NAN, 0.5], None, 0.5, 7, false)
            .expect_err("NaN in probabilities must be refused");
        assert!(
            matches!(
                err,
                ClassifyError::NonFiniteResponse {
                    field: "probabilities"
                }
            ),
            "expected NonFiniteResponse{{probabilities}}, got {err:?}"
        );
    }

    #[test]
    fn constructor_rejects_a_non_finite_logit() {
        let err = ClassifyResult::new(
            "positive".into(),
            vec![0.25, 0.75],
            Some(vec![f64::INFINITY, 0.0]),
            0.5,
            7,
            false,
        )
        .expect_err("infinity in logits must be refused");
        assert!(
            matches!(err, ClassifyError::NonFiniteResponse { field: "logits" }),
            "expected NonFiniteResponse{{logits}}, got {err:?}"
        );
    }

    #[test]
    fn constructor_rejects_a_non_finite_margin() {
        let err = ClassifyResult::new(
            "positive".into(),
            vec![0.25, 0.75],
            None,
            f64::NEG_INFINITY,
            7,
            false,
        )
        .expect_err("a non-finite margin must be refused");
        assert!(
            matches!(err, ClassifyError::NonFiniteResponse { field: "margin" }),
            "expected NonFiniteResponse{{margin}}, got {err:?}"
        );
    }

    #[test]
    fn constructor_rejects_a_non_finite_latency() {
        let ok = one_result(vec![0.25, 0.75]).expect("a valid result");
        let err =
            ClassifyResponse::new(GOLDEN_SHA.into(), GOLDEN_BACKEND.into(), f64::NAN, vec![ok])
                .expect_err("NaN latency must be refused");
        assert!(
            matches!(
                err,
                ClassifyError::NonFiniteResponse {
                    field: "latency_ms"
                }
            ),
            "expected NonFiniteResponse{{latency_ms}}, got {err:?}"
        );
    }

    #[test]
    fn constructor_rejects_a_negative_latency() {
        let ok = one_result(vec![0.25, 0.75]).expect("a valid result");
        let err = ClassifyResponse::new(GOLDEN_SHA.into(), GOLDEN_BACKEND.into(), -1.0, vec![ok])
            .expect_err("a negative latency must be refused");
        assert!(
            matches!(err, ClassifyError::NegativeLatency { .. }),
            "expected NegativeLatency, got {err:?}"
        );
    }

    #[test]
    fn a_zero_latency_is_legal() {
        // The contract forbids any gate from asserting latency > 0: a fast
        // operation under a coarse timer legitimately reports exactly 0.
        let ok = one_result(vec![0.25, 0.75]).expect("a valid result");
        let response =
            ClassifyResponse::new(GOLDEN_SHA.into(), GOLDEN_BACKEND.into(), 0.0, vec![ok])
                .expect("zero latency is legal");
        assert!(response.latency_ms().is_finite() && response.latency_ms() >= 0.0);
    }

    #[test]
    fn probability_mass_must_be_one_within_tolerance() {
        let err = one_result(vec![0.25, 0.25]).expect_err("mass 0.5 must be refused");
        assert!(
            matches!(err, ClassifyError::ProbabilityMassOutOfRange { .. }),
            "expected ProbabilityMassOutOfRange, got {err:?}"
        );
        one_result(vec![0.25, 0.75]).expect("mass 1.0 is accepted");
        // Inside tolerance, so accepted; the bound is absolute, not relative.
        one_result(vec![0.25, 0.75 + 5e-7]).expect("mass within 1e-6 is accepted");
    }

    #[test]
    fn logits_arity_must_match_probabilities() {
        let err = ClassifyResult::new(
            "positive".into(),
            vec![0.25, 0.75],
            Some(vec![1.0]),
            0.5,
            7,
            false,
        )
        .expect_err("a 1-entry logit vector against 2 labels must be refused");
        assert!(
            matches!(
                err,
                ClassifyError::LabelCountMismatch {
                    expected: 2,
                    got: 1
                }
            ),
            "expected LabelCountMismatch{{2,1}}, got {err:?}"
        );
    }

    #[test]
    fn every_result_row_must_have_the_same_label_arity() {
        let two = one_result(vec![0.25, 0.75]).expect("valid");
        let three = ClassifyResult::new(
            "negative".into(),
            vec![0.25, 0.25, 0.5],
            None,
            0.25,
            5,
            false,
        )
        .expect("valid on its own");
        let err = ClassifyResponse::new(
            GOLDEN_SHA.into(),
            GOLDEN_BACKEND.into(),
            0.0,
            vec![two, three],
        )
        .expect_err("rows of differing arity describe two different label sets");
        assert!(
            matches!(
                err,
                ClassifyError::LabelCountMismatch {
                    expected: 2,
                    got: 3
                }
            ),
            "expected LabelCountMismatch{{2,3}}, got {err:?}"
        );
    }

    #[test]
    fn deserializing_a_null_probability_is_refused() {
        let body = GOLDEN_RESPONSE_JSON.replace("[0.25,0.75]", "[null,0.75]");
        assert!(
            serde_json::from_str::<ClassifyResponse>(&body).is_err(),
            "a null probability must not deserialize into a ClassifyResponse"
        );
    }

    #[test]
    fn deserializing_a_negative_latency_is_refused_typed() {
        // The TYPED assertion, on the conversion the Deserialize impl routes
        // through. Asserting only on serde_json's message would be a substring
        // test that a reworded error silently turns green.
        let wire = ClassifyResponseWire {
            schema_version: CLASSIFY_SCHEMA_VERSION,
            artifact_sha256: GOLDEN_SHA.into(),
            backend: GOLDEN_BACKEND.into(),
            latency_ms: -0.5,
            results: vec![ClassifyResultWire {
                label: "positive".into(),
                probabilities: vec![0.25, 0.75],
                logits: None,
                margin: 0.5,
                token_count: 7,
                truncated: false,
            }],
        };
        let err = ClassifyResponse::try_from(wire).expect_err("a negative latency is refused");
        assert!(
            matches!(err, ClassifyError::NegativeLatency { .. }),
            "expected NegativeLatency, got {err:?}"
        );

        // And the serde door genuinely uses that conversion.
        let body = GOLDEN_RESPONSE_JSON.replace(r#""latency_ms":0.0"#, r#""latency_ms":-0.5"#);
        assert!(
            serde_json::from_str::<ClassifyResponse>(&body).is_err(),
            "Deserialize must route through the validating constructor"
        );
    }

    #[test]
    fn deserializing_a_non_finite_probability_is_refused_typed() {
        let wire = ClassifyResultWire {
            label: "positive".into(),
            probabilities: vec![f64::NAN, 0.75],
            logits: None,
            margin: 0.5,
            token_count: 7,
            truncated: false,
        };
        let err = ClassifyResult::try_from(wire).expect_err("NaN is refused on the wire path too");
        assert!(
            matches!(
                err,
                ClassifyError::NonFiniteResponse {
                    field: "probabilities"
                }
            ),
            "expected NonFiniteResponse{{probabilities}}, got {err:?}"
        );
    }

    #[test]
    fn deserializing_an_unknown_schema_version_is_refused_typed() {
        let wire = ClassifyResponseWire {
            schema_version: 2,
            artifact_sha256: GOLDEN_SHA.into(),
            backend: GOLDEN_BACKEND.into(),
            latency_ms: 0.0,
            results: Vec::new(),
        };
        let err = ClassifyResponse::try_from(wire).expect_err("a v2 payload is not a v1 response");
        assert!(
            matches!(
                err,
                ClassifyError::UnsupportedSchemaVersion {
                    expected: 1,
                    got: 2
                }
            ),
            "expected UnsupportedSchemaVersion{{1,2}}, got {err:?}"
        );
    }

    #[test]
    fn deserializing_an_unknown_response_field_is_refused() {
        let body =
            GOLDEN_RESPONSE_JSON.replace(r#""latency_ms":0.0"#, r#""latency_ms":0.0,"tps":9"#);
        assert!(
            serde_json::from_str::<ClassifyResponse>(&body).is_err(),
            "deny_unknown_fields must refuse a key this schema does not model"
        );
    }

    #[test]
    fn a_valid_response_round_trips_through_serde() {
        let original = golden_response();
        let json = serde_json::to_string(&original).expect("serializes");
        let back: ClassifyResponse = serde_json::from_str(&json).expect("deserializes");
        assert_eq!(back, original, "the envelope round-trips unchanged");
        // `==` deliberately says nothing about `latency_ms` (see the type's
        // `PartialEq`), so the round trip asserts that field SEPARATELY and by
        // BITS. Bits and not `==` on the float: the wire carries a decimal
        // rendering, and a round trip that only compared `0.0 == 0.0` would also
        // have passed for `-0.0`, or for any value serde had renormalized. This
        // is the assertion that would otherwise have been quietly lost by
        // narrowing equality.
        assert_eq!(
            back.latency_ms().to_bits(),
            original.latency_ms().to_bits(),
            "latency_ms is excluded from `==` but must still survive the wire exactly"
        );
    }

    #[test]
    fn logits_key_is_present_and_null_when_not_requested() {
        // "Absent" is an explicit `null`, NOT a missing key. A missing key is
        // invisible to a diff and to a null-walking guard; an explicit null is
        // loud to both.
        let json = serde_json::to_string(&golden_response()).expect("serializes");
        assert!(
            json.contains(r#""logits":null"#),
            "an unrequested logits vector must serialize as an explicit null: {json}"
        );
        // The ATTRIBUTE form, not the bare token. The bare token appears in this
        // module's own doc comments explaining why the attribute is forbidden,
        // so a gate on it would turn red on its own documentation — orchestrator
        // note F-05, observed here on the first run of this very assertion.
        assert!(
            !production_source().contains("serde(skip_serializing_if"),
            "skip_serializing_if would make the absent-logits case invisible to a diff"
        );
    }

    #[test]
    fn requested_logits_serialize_as_an_array() {
        let result = ClassifyResult::new(
            "positive".into(),
            vec![0.25, 0.75],
            Some(vec![-0.5, 0.5]),
            0.5,
            7,
            false,
        )
        .expect("valid");
        let response =
            ClassifyResponse::new(GOLDEN_SHA.into(), GOLDEN_BACKEND.into(), 0.0, vec![result])
                .expect("valid");
        let json = serde_json::to_string(&response).expect("serializes");
        assert!(
            json.contains(r#""logits":[-0.5,0.5]"#),
            "requested logits serialize in place: {json}"
        );
    }

    #[test]
    fn request_document_round_trips_awkward_texts_byte_identically() {
        let doc = ClassifyRequestDocument {
            texts: vec![
                "a line\nand another".to_string(),
                "a\ttab".to_string(),
                String::new(),
                "café — 日本語 🌍".to_string(),
            ],
            include_logits: true,
        };
        let json = serde_json::to_string(&doc).expect("serializes");
        let back: ClassifyRequestDocument = serde_json::from_str(&json).expect("deserializes");
        assert_eq!(
            back, doc,
            "the shared request document must survive newlines, tabs, empties and non-ASCII"
        );
        // The defect this design exists to prevent: a line-delimited CLI format
        // would split the first text into two, so the CLI and HTTP surfaces
        // would receive DIFFERENT ordered input sets while appearing to agree.
        assert_eq!(back.texts.len(), 4, "four texts in, four texts out");
        assert_eq!(
            back.texts[0], "a line\nand another",
            "the embedded newline survives verbatim"
        );
    }

    #[test]
    fn request_document_defaults_include_logits_to_false() {
        let doc: ClassifyRequestDocument =
            serde_json::from_str(r#"{"texts":["hello"]}"#).expect("deserializes");
        assert!(!doc.include_logits, "include_logits defaults to false");
        assert!(!ClassifyRequestDocument::new(["hello"]).include_logits);
        assert!(
            ClassifyRequestDocument::new(["hello"])
                .with_logits()
                .include_logits
        );
    }

    #[test]
    fn request_document_rejects_an_unknown_field() {
        let err = serde_json::from_str::<ClassifyRequestDocument>(
            r#"{"texts":["hello"],"temperature":0.7}"#,
        )
        .expect_err("deny_unknown_fields refuses a field this schema does not model");
        assert!(
            err.to_string().contains("temperature"),
            "the rejection should name the unknown field: {err}"
        );
    }

    #[test]
    fn response_and_result_fields_are_private() {
        let src = production_source();
        for (marker, name) in [
            ("pub struct ClassifyResponse {", "ClassifyResponse"),
            ("pub struct ClassifyResult {", "ClassifyResult"),
        ] {
            let start = src
                .find(marker)
                .unwrap_or_else(|| panic!("{name} declaration not found"));
            let body = &src[start + marker.len()..];
            let end = body
                .find("\n}")
                .unwrap_or_else(|| panic!("{name} body not closed"));
            let block = &body[..end];
            assert!(
                !block.contains("pub "),
                "{name} must have NO public field — a public field falsifies \
                 \"a non-finite value is unrepresentable\" (review M1). Body was:\n{block}"
            );
        }
    }

    #[test]
    fn both_serde_directions_route_through_the_validating_constructor() {
        let src = production_source();
        assert!(
            src.contains(r#"try_from = "ClassifyResponseWire""#),
            "ClassifyResponse must deserialize via the validating wire conversion"
        );
        assert!(
            src.contains(r#"try_from = "ClassifyResultWire""#),
            "ClassifyResult must deserialize via the validating wire conversion"
        );
        assert!(
            src.contains(r#"into = "ClassifyResponseWire""#),
            "ClassifyResponse must serialize via the same wire form it parses"
        );
        assert!(
            !src.contains("pub struct ClassifyResponseWire"),
            "the wire struct must be private, or a caller could build one and skip validation"
        );
        assert!(
            !src.contains("pub struct ClassifyResultWire"),
            "the wire struct must be private, or a caller could build one and skip validation"
        );
    }

    #[test]
    fn within_is_nan_visible_in_both_argument_positions() {
        assert!(
            within(0.5, 1.0),
            "a finite delta inside the bound is inside"
        );
        assert!(within(1.0, 1.0), "the bound itself is inside");
        assert!(
            !within(1.5, 1.0),
            "a finite delta outside the bound is outside"
        );
        assert!(!within(f64::NAN, 1.0), "a NaN delta counts as OUTSIDE");
        assert!(!within(0.5, f64::NAN), "a NaN bound counts as OUTSIDE");
        assert!(
            !within(f64::from(f32::NAN), 1.0),
            "an f32 NaN widened to f64 counts as OUTSIDE"
        );
        // The idiom itself, not only its current behaviour: `delta <= bound`
        // happens to reject NaN, but refactors into `!(delta > bound)`, which
        // ACCEPTS it. `partial_cmp` cannot be refactored into acceptance.
        //
        // The scan follows the function: `within` is defined once for the crate,
        // in artifact.rs, and this is the guard that pins its shape. Scanning
        // classify.rs would now find only the `use` and prove nothing.
        let src = include_str!("artifact.rs");
        let start = src
            .find("pub(crate) fn within(")
            .expect("within is defined in artifact.rs");
        let body = &src[start..];
        let end = body.find("\n}").expect("within's body is closed");
        assert!(
            body[..end].contains("partial_cmp"),
            "within must be written through partial_cmp so the NaN case is visible"
        );
    }
}

/// Task 2: the execution-derived backend identity channel (D-12, review B6).
///
/// These suites live here rather than in `encoder.rs` because they exercise only
/// the re-exported surface — `SetFitMiniLm::encode_texts_traced` and
/// `ExecutionBackend::identity` — and because the plan fixes
/// `setfit::classify::backend` as this task's filter. No case here needs
/// encoder-private state.
#[cfg(test)]
mod backend {
    use super::*;
    use crate::setfit::encoder::ExecutionBackend;

    /// Item 12 of the contract, read from the contract rather than from a copy
    /// of it. `include_str!` and not a runtime read: a missing contract is a
    /// COMPILE error here, where a runtime read would be a silent skip.
    const CONTRACT: &str = include_str!("../../../../contracts/setfit-apr-v1.yaml");

    /// Item 12's row in the BINDING REGISTRY — the artifact a human or a gate
    /// reads to decide whether a contracted equation is implemented.
    ///
    /// `include_str!` and not a runtime read, for exactly the reason `CONTRACT`
    /// above is: a missing registry must be a COMPILE error here, where a
    /// runtime read would be a silent skip. A guard that silently skips is the
    /// same unfalsifiable claim this particular row's history is about — the
    /// row named `aprender::setfit::classify::backend_identity`, a path that
    /// exists at no point on this branch, and `pv audit` could not tell,
    /// because it reads the `status` field and does not resolve symbols.
    const BINDING: &str = include_str!("../../../../contracts/aprender/binding.yaml");

    /// The signature the registry records for the bound symbol.
    ///
    /// ONE literal with TWO obligations (test 6): the registry row must record
    /// it, and `encoder.rs` must declare it. A recorded signature that nothing
    /// reads back is a claim nothing checks.
    const ROW_SIGNATURE: &str = "fn identity(&self) -> String";

    /// Every `.rs` file in this directory, read at test time.
    ///
    /// `read_dir` and not a hardcoded list: a hardcoded list goes stale the
    /// moment a file is added, and a file added later is exactly where a
    /// capability-detection symbol would arrive unnoticed.
    fn setfit_sources() -> Vec<(String, String)> {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/setfit");
        let mut out = Vec::new();
        for entry in std::fs::read_dir(&dir).expect("the setfit source directory is readable") {
            let path = entry.expect("a readable directory entry").path();
            if path.extension().and_then(std::ffi::OsStr::to_str) == Some("rs") {
                let name = path
                    .file_name()
                    .and_then(std::ffi::OsStr::to_str)
                    .expect("a UTF-8 file name")
                    .to_string();
                let body = std::fs::read_to_string(&path).expect("a readable source file");
                out.push((name, body));
            }
        }
        // A non-zero MINIMUM, because a scan of zero files exits green and
        // proves nothing (orchestrator note F-04).
        assert!(
            out.len() >= 10,
            "expected at least 10 setfit source files, found {}: a path that resolved to \
             nothing would make every scan below vacuous",
            out.len()
        );
        out
    }

    /// The binding-registry row for `equation`, as raw YAML text: from the
    /// `- contract:` line that opens it to the next one.
    ///
    /// Returns `Option` rather than `expect`ing inside its callers so the
    /// non-vacuity arm (test 4) can assert the row was FOUND explicitly. A
    /// parser that silently matched nothing would make every pin below
    /// vacuously true — the zero-match class, one tier up from the code it
    /// guards.
    fn binding_row(equation: &str) -> Option<&'static str> {
        let key = format!("\n  equation: {equation}\n");
        let at = BINDING.find(&key)?;
        // Walk back to the `- contract:` line that opens this row, then forward
        // to the one that opens the next.
        let start = BINDING[..at].rfind("\n- contract:")? + 1;
        let tail = &BINDING[start..];
        let end = tail[1..]
            .find("\n- contract:")
            .map_or(tail.len(), |o| o + 1);
        Some(&tail[..end])
    }

    /// TEST 1 — the COMPILE witness.
    ///
    /// Coercing the inherent method into a typed function item makes rustc
    /// itself the witness that `ExecutionBackend::identity` exists at that path
    /// with that signature. Rename it, move it, or change its signature and
    /// this is a COMPILE error in the crate that owns it — not a registry row
    /// that drifted silently, which is the failure this row's history is about.
    #[test]
    fn the_registry_named_symbol_resolves_as_a_typed_function_item() {
        let f: fn(&ExecutionBackend) -> String = ExecutionBackend::identity;
        let model = fixture_encoder_model();
        let (_, backend) = model
            .encode_texts_traced(&["hello world"])
            .expect("the fixture encodes");
        assert_eq!(
            f(&backend),
            backend.identity(),
            "the function item bound here must BE the method the registry names, not a \
             same-named neighbour"
        );
    }

    /// TEST 2 — the registry pin, MUST-MATCH arm.
    #[test]
    fn the_registry_row_names_the_shipped_symbol_and_is_implemented() {
        let row = binding_row("backend_identity")
            .expect("the backend_identity row exists (test 4 is the dedicated non-vacuity arm)");
        for needle in [
            "\n  module_path: aprender::setfit::encoder\n",
            "\n  function: ExecutionBackend::identity\n",
            "\n  status: implemented\n",
        ] {
            assert!(
                row.contains(needle),
                "the backend_identity row does not carry {needle:?}. The `function` column is \
                 TYPE-QUALIFIED because `identity` is an INHERENT METHOD on ExecutionBackend, so \
                 a bare `identity` would be as unresolvable as the ghost it replaces. Row as \
                 found:\n{row}"
            );
        }
    }

    /// TEST 3 — the registry pin, MUST-NOT-MATCH arm.
    ///
    /// A correction that left the ghost behind in a SECOND row would satisfy
    /// test 2 and still ship an unresolvable claim.
    #[test]
    fn the_ghost_path_appears_nowhere_in_the_registry() {
        // Assembled from `concat!` fragments so this guard's OWN source text
        // does not carry the strings it forbids: `classify.rs` is inside the
        // directory `setfit_sources()` walks, the F-05 self-scan hazard the
        // sibling guard `the_setfit_surface_names_no_capability_detection_symbol`
        // documents in its own comment. `concat!` expands at compile time, so
        // the comparison is against the whole symbol.
        let ghost_path = concat!("aprender::setfit::classify", "::backend_identity");
        let bare_column = concat!("\n  function: ", "backend_identity\n");

        assert!(
            !BINDING.contains(ghost_path),
            "the registry still names {ghost_path:?}, which exists at no point on this branch"
        );
        assert!(
            !BINDING.contains(bare_column),
            "the registry still carries a bare {bare_column:?} column; an inherent method needs \
             its type to resolve"
        );

        // The case table is EXECUTED, not commented: both needles are shown able
        // to fire on text they WOULD reject.
        let planted = format!("  module_path: {ghost_path}{bare_column}  status: implemented\n");
        assert!(
            planted.contains(ghost_path) && planted.contains(bare_column),
            "both needles must match a planted violation, or the two arms above are theater"
        );
    }

    /// TEST 4 — NON-VACUITY. The arms in tests 2 and 3 are only evidence if the
    /// parser they share finds a row at all.
    #[test]
    fn the_registry_row_parser_finds_the_row_it_claims_to_pin() {
        // MUST-MATCH arm of the parser's own case table.
        let row = binding_row("backend_identity");
        assert!(
            row.is_some(),
            "no `equation: backend_identity` row was found in contracts/aprender/binding.yaml. A \
             parser that silently matched NOTHING would make every pin in this module vacuously \
             true"
        );
        let row = row.expect("is_some was just asserted");
        assert!(
            row.starts_with("- contract: setfit-apr-v1.yaml\n"),
            "the row must be found under the setfit-apr-v1.yaml list, not a neighbouring \
             contract's: {row}"
        );
        // MUST-NOT-MATCH arm: an equation that does not exist must yield None
        // rather than drifting onto a neighbouring row.
        assert!(
            binding_row("backend_identity_x").is_none(),
            "the parser returned a row for an equation that does not exist, so a renamed key \
             would go unnoticed"
        );
    }

    /// TEST 5 — CONTRACT AGREEMENT.
    ///
    /// Tests 1-4 prove the row names a symbol that EXISTS; this proves that
    /// symbol's OUTPUT is the thing item 12 describes, so the row binds the
    /// equation rather than merely something that compiles.
    #[test]
    fn the_bound_symbol_produces_the_identity_the_contract_describes() {
        // Obtained from the live encode path, never constructed: ExecutionBackend
        // has no public constructor and the sibling guard above exists to keep it
        // that way, so a value here is one an invocation RETURNED.
        let model = fixture_encoder_model();
        let (_, backend) = model
            .encode_texts_traced(&["hello world"])
            .expect("the fixture encodes");
        let identity = ExecutionBackend::identity(&backend);
        let segments: Vec<&str> = identity.split(':').collect();
        assert_eq!(
            segments.len(),
            3,
            "item 12's grammar is <device>:<implementation>:<kernel>: {identity:?}"
        );
        assert_eq!(
            segments[1], "setfit-core",
            "the middle segment names THIS implementation: {identity:?}"
        );
        assert!(
            CONTRACT.contains("backend = \"<device>:<implementation>:<kernel>\""),
            "item 12's grammar line moved, so the segment assertions above are pinned to a \
             formula that is no longer the contract's"
        );
    }

    /// TEST 6 — closes the `module_path` hole.
    ///
    /// Tests 1-3 witness the FUNCTION, but compare `module_path` against a
    /// hard-coded literal living inside `classify.rs` — a file that is NOT the
    /// module that literal names. A moved `ExecutionBackend` would leave the
    /// literal true and the row wrong. So reuse the sibling idiom from
    /// `execution_backend_has_no_public_constructor_or_setter`: read the file
    /// the path's last segment names and require the symbol to be declared
    /// THERE. That is the strongest tie available without a proc-macro.
    #[test]
    fn the_module_path_segment_names_the_file_the_symbol_lives_in() {
        let src = include_str!("encoder.rs");
        let row = binding_row("backend_identity").expect("the backend_identity row exists");
        assert!(
            row.contains("\n  module_path: aprender::setfit::encoder\n"),
            "the row's module_path must end in the `encoder` segment this test reads"
        );
        assert!(
            src.contains("impl ExecutionBackend {"),
            "encoder.rs does not declare `impl ExecutionBackend {{`, so the row's `encoder` \
             segment names a module the symbol does not live in"
        );
        assert!(
            row.contains(&format!("\n  signature: '{ROW_SIGNATURE}'\n")),
            "the row must record the signature {ROW_SIGNATURE:?}"
        );
        assert!(
            src.contains(&format!("pub {ROW_SIGNATURE}")),
            "encoder.rs must declare `pub {ROW_SIGNATURE}`; a signature the registry records but \
             nothing reads back is a claim nothing checks"
        );
    }

    #[test]
    fn encode_texts_traced_returns_the_identity_the_contract_pins() {
        let model = fixture_encoder_model();
        let (_, backend) = model
            .encode_texts_traced(&["hello world"])
            .expect("the fixture encodes");
        let identity = backend.identity();
        // Pinned against the CONTRACT's own text, not against a literal copied
        // into this file: a copy can drift from the contract silently, and the
        // D-12 gate additionally requires the kernel literal to appear nowhere
        // in this module.
        let pin = format!("v1 = \"{identity}\"");
        assert!(
            CONTRACT.contains(&pin),
            "the observed identity {identity:?} is not the value item 12 pins ({pin:?})"
        );
    }

    #[test]
    fn the_identity_carries_no_simd_capability_token() {
        let model = fixture_encoder_model();
        let (_, backend) = model
            .encode_texts_traced(&["hello world"])
            .expect("the fixture encodes");
        let identity = backend.identity().to_lowercase();
        for token in ["avx", "neon", "sse", "512"] {
            assert!(
                !identity.contains(token),
                "the identity {identity:?} names the capability token {token:?}. Detecting \
                 that a SIMD ISA is AVAILABLE is consistent with a SCALAR execution — \
                 trueno's Matrix::matmul dispatches on SIZE — so such a value describes the \
                 HOST, not the run (CLAUDE.md Verification Discipline rule 2)"
            );
        }
    }

    #[test]
    fn the_identity_has_exactly_three_colon_separated_segments() {
        let model = fixture_encoder_model();
        let (_, backend) = model
            .encode_texts_traced(&["hello world"])
            .expect("the fixture encodes");
        let identity = backend.identity();
        let segments: Vec<&str> = identity.split(':').collect();
        assert_eq!(
            segments.len(),
            3,
            "the grammar is <device>:<implementation>:<kernel> and stays three segments, so a \
             future GPU identity is COMPARABLE to this one: {identity:?}"
        );
        assert!(
            segments.iter().all(|s| !s.is_empty()),
            "no segment may be empty: {identity:?}"
        );
        assert_eq!(segments[0], backend.device());
        assert_eq!(segments[2], backend.kernel());
    }

    #[test]
    fn execution_backend_has_no_public_constructor_or_setter() {
        let src = include_str!("encoder.rs");
        let marker = "pub struct ExecutionBackend {";
        let start = src.find(marker).expect("ExecutionBackend is declared here");
        let body = &src[start + marker.len()..];
        let end = body.find("\n}").expect("its body is closed");
        assert!(
            !body[..end].contains("pub "),
            "ExecutionBackend must have no public field, or a caller could forge an identity"
        );

        let impl_marker = "impl ExecutionBackend {";
        let istart = src
            .find(impl_marker)
            .expect("ExecutionBackend has an inherent impl");
        let ibody = &src[istart + impl_marker.len()..];
        let iend = ibody.find("\n}").expect("the impl block is closed");
        let block = &ibody[..iend];
        for forbidden in ["pub fn new", "pub const fn new", "&mut self", "-> Self"] {
            assert!(
                !block.contains(forbidden),
                "ExecutionBackend's public impl must contain no {forbidden:?}: a constructor \
                 or a setter would let a value be MINTED rather than RETURNED by the encode \
                 invocation that ran (D-12)"
            );
        }
        assert!(
            !src.contains("pub const ENCODE_BACKEND")
                && !src.contains("pub(crate) const ENCODE_BACKEND"),
            "the constant instance stays module-private; only encode_with_backend hands it out"
        );
    }

    #[test]
    fn the_setfit_surface_names_no_capability_detection_symbol() {
        // The needles are assembled from fragments so this test's OWN text does
        // not contain them — otherwise scanning `classify.rs` would make the
        // guard permanently red, the F-05 failure mode. `concat!` is expanded at
        // compile time, so the comparison is against the whole symbol.
        let needles = [
            concat!("select_", "backend"),
            concat!("Backend::", "AVX"),
            concat!("detect_x86_", "backend"),
            concat!("detect_arm_", "backend"),
        ];
        let sources = setfit_sources();
        for (name, body) in &sources {
            for needle in needles {
                assert!(
                    !body.contains(needle),
                    "{name} names {needle:?}. A capability probe reports what the HOST CAN DO; \
                     the backend field must report what RAN (review B6, D-12)"
                );
            }
        }
        // The scan is proven able to fire, on a string it WOULD reject.
        let planted = format!("let b = {};", needles[1]);
        assert!(
            needles.iter().any(|n| planted.contains(n)),
            "the needle set must match a planted violation, or this gate is theater"
        );
    }

    #[test]
    fn the_kernel_literal_lives_only_in_encoder_rs() {
        // Assembled from fragments for the same reason as above: written whole,
        // this test would itself be a second home for the literal.
        let kernel = concat!("autograd-", "trueno-matmul");
        let sources = setfit_sources();
        let mut carriers: Vec<&str> = Vec::new();
        for (name, body) in &sources {
            if body.contains(kernel) {
                carriers.push(name.as_str());
            }
        }
        assert_eq!(
            carriers,
            vec!["encoder.rs"],
            "the kernel identity must be spelled in encoder.rs and nowhere else in this \
             directory — classify.rs included. It arrives everywhere else as a VALUE returned \
             by the encode call (review B6)"
        );
    }

    #[test]
    fn the_traced_and_untraced_encode_paths_agree_elementwise() {
        // `encode`/`encode_texts` were extended BESIDE, not modified: the
        // training path and every Phase 1 conformance fixture call them, and
        // their output must not have moved because a reporting channel was
        // added. Comparing the two paths' data is the behavioural half of that
        // claim; `git diff` on encoder.rs is the textual half.
        let model = fixture_encoder_model();
        let texts = ["hello world", "a much longer sentence for the batch"];
        let plain = model.encode_texts(&texts).expect("the untraced path runs");
        let (traced, _) = model
            .encode_texts_traced(&texts)
            .expect("the traced path runs");
        assert_eq!(plain.shape(), traced.shape(), "same shape");
        assert_eq!(
            plain.data(),
            traced.data(),
            "the traced path must return the SAME embeddings, bit for bit"
        );
    }
}

/// Task 3: `VerifiedSetFitModel::classify` — the one classification path.
#[cfg(test)]
mod classify_path {
    use super::*;
    use crate::setfit::tokenizer::MAX_SEQUENCE_LENGTH;

    /// The contract's `probe_truncation_boundary`, built by its OWN rule:
    /// `repeat_unit` repeated exactly `repeat_count` times, no separator.
    const TRUNCATION_PROBE_UNIT: &str = "few shot classification with contrastive pairs ";
    const TRUNCATION_PROBE_REPEATS: usize = 64;

    fn truncation_probe() -> String {
        TRUNCATION_PROBE_UNIT.repeat(TRUNCATION_PROBE_REPEATS)
    }

    #[test]
    fn the_truncation_probe_matches_the_contract_construction_rule() {
        // The probe is only evidence about truncation if it is the contract's
        // probe. Pinned against the contract text rather than against a comment.
        let contract = include_str!("../../../../contracts/setfit-apr-v1.yaml");
        assert!(
            contract.contains(&format!("repeat_unit: '{TRUNCATION_PROBE_UNIT}'")),
            "the repeat unit drifted from contract item probe_truncation_boundary"
        );
        assert!(
            contract.contains(&format!("repeat_count: {TRUNCATION_PROBE_REPEATS}")),
            "the repeat count drifted from contract item probe_truncation_boundary"
        );
    }

    #[test]
    fn classify_returns_one_result_per_text_in_input_order() {
        let model = fixture_verified_model();
        let labels = model.ordered_labels().to_vec();
        let request = ClassifyRequestDocument::new(["ok", "a longer sentence here", "x"]);
        let response = model.classify(&request).expect("the fixture classifies");

        assert_eq!(response.results().len(), 3, "one result per input text");
        for result in response.results() {
            assert_eq!(
                result.probabilities().len(),
                labels.len(),
                "every result carries the FULL ordered probability vector"
            );
            assert!(
                labels.iter().any(|l| l == result.label()),
                "the winning label {:?} is one of the head's ordered labels",
                result.label()
            );
        }
        assert_eq!(response.schema_version(), CLASSIFY_SCHEMA_VERSION);
    }

    #[test]
    fn classify_refuses_an_empty_request() {
        let model = fixture_verified_model();
        let err = model
            .classify(&ClassifyRequestDocument::new(Vec::<String>::new()))
            .expect_err("an empty batch is refused");
        assert!(
            matches!(err, ClassifyError::EmptyInput),
            "expected EmptyInput, got {err:?}"
        );
    }

    #[test]
    fn classify_refuses_a_batch_over_the_contract_bound() {
        let model = fixture_verified_model();
        let texts: Vec<String> = (0..=MAX_BATCH_TEXTS).map(|i| format!("text {i}")).collect();
        assert_eq!(texts.len(), 257, "the bound is 256, so this is one over");
        let err = model
            .classify(&ClassifyRequestDocument::new(texts))
            .expect_err("257 texts exceed the contract bound");
        assert!(
            matches!(
                err,
                ClassifyError::BatchTooLarge {
                    max: MAX_BATCH_TEXTS,
                    got: 257
                }
            ),
            "expected BatchTooLarge{{256,257}}, got {err:?}"
        );
    }

    #[test]
    fn classify_accepts_exactly_the_contract_bound() {
        // The boundary itself, so the refusal above is shown to be OFF-BY-NONE:
        // a bound that also rejected 256 would pass the test above for the
        // wrong reason.
        let model = fixture_verified_model();
        let texts: Vec<String> = (0..MAX_BATCH_TEXTS).map(|i| format!("t{i}")).collect();
        let response = model
            .classify(&ClassifyRequestDocument::new(texts))
            .expect("exactly 256 texts are legal");
        assert_eq!(response.results().len(), MAX_BATCH_TEXTS);
    }

    #[test]
    fn a_truncated_text_reports_truncated_and_the_pinned_token_count() {
        let model = fixture_verified_model();
        let response = model
            .classify(&ClassifyRequestDocument::new([truncation_probe()]))
            .expect("the truncation probe classifies");
        let result = response.results().first().expect("one result");
        assert!(
            result.truncated(),
            "the contract's truncation probe exceeds the pinned bound"
        );
        assert_eq!(
            usize::try_from(result.token_count()).expect("fits"),
            MAX_SEQUENCE_LENGTH,
            "a truncated text consumed exactly the pinned bound's worth of positions — \
             compared against the EXPORTED constant, never a literal 256"
        );
    }

    #[test]
    fn a_short_text_reports_not_truncated_and_a_token_count_below_the_bound() {
        let model = fixture_verified_model();
        let response = model
            .classify(&ClassifyRequestDocument::new(["ok"]))
            .expect("a short text classifies");
        let result = response.results().first().expect("one result");
        assert!(
            !result.truncated(),
            "a two-character input is not truncated"
        );
        assert!(
            usize::try_from(result.token_count()).expect("fits") < MAX_SEQUENCE_LENGTH,
            "token_count {} should be below the pinned bound",
            result.token_count()
        );
        assert!(
            result.token_count() > 0,
            "a non-empty text consumes at least one position"
        );
    }

    #[test]
    fn the_response_backend_equals_the_identity_encode_texts_traced_returns() {
        // D-12: the two values must be the SAME because they have the same
        // source — the encode invocation — not because two literals were kept
        // in sync.
        let model = fixture_verified_model();
        let encoder = fixture_encoder_model();
        let (_, backend) = encoder
            .encode_texts_traced(&["ok"])
            .expect("the traced encode runs");
        let response = model
            .classify(&ClassifyRequestDocument::new(["ok"]))
            .expect("classifies");
        assert_eq!(
            response.backend(),
            backend.identity(),
            "the response's backend is the identity the encode path returns"
        );
    }

    #[test]
    fn classify_exposes_no_parameter_that_can_set_the_backend() {
        let src = production_source();
        let marker = "pub fn classify(";
        let start = src.find(marker).expect("classify is defined here");
        let body = &src[start..];
        let end = body.find(") -> Result<").expect("its signature is closed");
        let signature = &body[..end];
        assert!(
            signature.contains("request: &ClassifyRequestDocument"),
            "classify takes the shared request document: {signature}"
        );
        for forbidden in ["backend", "device", "kernel"] {
            assert!(
                !signature.contains(forbidden),
                "classify's signature must expose no {forbidden:?} parameter — the identity \
                 is REPORTED FROM EXECUTION, never echoed from a caller (D-12): {signature}"
            );
        }
        // And the request document itself carries no such knob either, so the
        // absence above cannot be routed around through the one parameter it
        // does take.
        let doc_marker = "pub struct ClassifyRequestDocument {";
        let dstart = src.find(doc_marker).expect("the request document is here");
        let dbody = &src[dstart + doc_marker.len()..];
        let dend = dbody.find("\n}").expect("its body is closed");
        for forbidden in ["backend", "device", "kernel"] {
            assert!(
                !dbody[..dend].contains(forbidden),
                "ClassifyRequestDocument must carry no {forbidden:?} field"
            );
        }
    }

    #[test]
    fn latency_ms_is_finite_and_non_negative() {
        // Never asserted strictly positive: a fast operation under a coarse
        // timer legitimately reports 0.0, so `> 0` would be a flake with a
        // schedule (contract item 11).
        let model = fixture_verified_model();
        let response = model
            .classify(&ClassifyRequestDocument::new(["ok"]))
            .expect("classifies");
        assert!(
            response.latency_ms().is_finite() && response.latency_ms() >= 0.0,
            "latency_ms was {}",
            response.latency_ms()
        );
    }

    #[test]
    fn single_and_batched_classification_agree_within_tolerance() {
        // Padding invariance at the RESPONSE level: a mixed-length batch pads
        // its short rows, and the masked mean pool must make that invisible.
        let model = fixture_verified_model();
        let texts = ["ok", "a considerably longer sentence than the first one"];
        let batched = model
            .classify(&ClassifyRequestDocument::new(texts))
            .expect("the batch classifies");
        for (i, text) in texts.iter().enumerate() {
            let single = model
                .classify(&ClassifyRequestDocument::new([*text]))
                .expect("the singleton classifies");
            let a = single.results()[0].probabilities();
            let b = batched.results()[i].probabilities();
            assert_eq!(a.len(), b.len(), "same arity");
            for (j, (p, q)) in a.iter().zip(b.iter()).enumerate() {
                assert!(
                    within((p - q).abs(), 1e-6),
                    "text {i} class {j}: singleton {p} vs batched {q} differ by more than 1e-6"
                );
            }
            assert_eq!(
                single.results()[0].label(),
                batched.results()[i].label(),
                "labels are compared EXACTLY; a 'close enough' label is a wrong answer"
            );
        }
    }

    #[test]
    fn include_logits_controls_the_logits_field() {
        let model = fixture_verified_model();
        let without = model
            .classify(&ClassifyRequestDocument::new(["ok"]))
            .expect("classifies");
        assert!(
            without.results()[0].logits().is_none(),
            "logits are absent unless requested"
        );

        let with = model
            .classify(&ClassifyRequestDocument::new(["ok"]).with_logits())
            .expect("classifies");
        let logits = with.results()[0]
            .logits()
            .expect("logits are present when requested");
        assert_eq!(
            logits.len(),
            model.ordered_labels().len(),
            "one logit per ordered label"
        );
        assert!(
            logits.iter().all(|l| l.is_finite()),
            "every logit is finite"
        );
        // The two vectors come from ONE computation, so they cannot disagree.
        assert_eq!(
            with.results()[0].probabilities(),
            without.results()[0].probabilities(),
            "requesting logits must not change the probabilities"
        );
    }

    #[test]
    fn probabilities_match_the_heads_predict_proba_exactly() {
        // classify derives probabilities from `predict_logits` + the head's own
        // `softmax_into`. `predict_proba` is defined as exactly that pair, so
        // this asserts the equivalence rather than arguing for it.
        let model = fixture_verified_model();
        let texts = vec!["ok".to_string(), "another input".to_string()];
        let response = model
            .classify(&ClassifyRequestDocument::new(texts.clone()))
            .expect("classifies");
        let embeddings = model.embed(&texts).expect("the embed surface runs");
        let expected = model
            .head()
            .predict_proba(&embeddings)
            .expect("the head runs");
        for (row, result) in response.results().iter().enumerate() {
            assert_eq!(
                result.probabilities(),
                expected[row].as_slice(),
                "row {row} must match the head's own predict_proba bit for bit"
            );
        }
    }

    #[test]
    fn margin_is_top1_minus_top2_and_the_label_is_the_argmax() {
        let model = fixture_verified_model();
        let response = model
            .classify(&ClassifyRequestDocument::new(["ok", "another one"]))
            .expect("classifies");
        let labels = model.ordered_labels();
        for result in response.results() {
            let mut sorted = result.probabilities().to_vec();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal));
            assert!(
                within((result.margin() - (sorted[0] - sorted[1])).abs(), 1e-12),
                "margin {} should be top1 {} minus top2 {}",
                result.margin(),
                sorted[0],
                sorted[1]
            );
            assert!(result.margin() >= 0.0, "top1 is never below top2");
            let winner =
                crate::classification::multinomial::argmax_lowest_index(result.probabilities());
            assert_eq!(
                result.label(),
                labels[winner],
                "the label is the argmax, ties breaking to the lowest index"
            );
        }
    }

    #[test]
    fn the_response_carries_the_models_artifact_sha256() {
        let model = fixture_verified_model();
        let response = model
            .classify(&ClassifyRequestDocument::new(["ok"]))
            .expect("classifies");
        assert_eq!(
            response.artifact_sha256(),
            model.artifact_sha256(),
            "every response is attributable to its artifact without trusting the caller"
        );
        assert_eq!(
            response.artifact_sha256().len(),
            64,
            "a lowercase-hex sha256"
        );
    }

    #[test]
    fn a_classify_response_serializes_and_reparses_unchanged() {
        // The end-to-end D-08 claim: what classify produces is what the CLI and
        // the HTTP surface will emit, and it survives its own schema.
        let model = fixture_verified_model();
        let response = model
            .classify(&ClassifyRequestDocument::new(["ok", "two"]).with_logits())
            .expect("classifies");
        let json = serde_json::to_string(&response).expect("serializes");
        let back: ClassifyResponse = serde_json::from_str(&json).expect("reparses");
        assert_eq!(back, response, "the real response round-trips unchanged");
        // Same split as the golden round trip: `==` covers everything the response
        // CLAIMS about the model, and the one MEASUREMENT is asserted by bits
        // beside it rather than smuggled into equality.
        assert_eq!(
            back.latency_ms().to_bits(),
            response.latency_ms().to_bits(),
            "a measured latency must survive the wire exactly, even though `==` ignores it"
        );
    }

    /// The exclusion is a PROPERTY OF THE TYPE, not a convention harnesses follow.
    ///
    /// Two responses that agree about the model and disagree only about how long
    /// it took are EQUAL. Without this, the 04-09 parity gate comparing a CLI
    /// response against an HTTP one would be red on every run for a reason that
    /// has nothing to do with either surface — a flake with a schedule, which the
    /// contract names by that phrase.
    #[test]
    fn responses_differing_only_in_latency_are_equal_and_differing_in_substance_are_not() {
        let model = fixture_verified_model();
        let request = ClassifyRequestDocument::new(["ok", "two"]).with_logits();
        let fast = model.classify(&request).expect("classifies");

        // A response identical in every claim, rebuilt with a wildly different
        // measurement. Constructed through the shipped validating constructor, so
        // this is a value either surface could legitimately have produced.
        let slow = ClassifyResponse::new(
            fast.artifact_sha256().to_string(),
            fast.backend().to_string(),
            fast.latency_ms() + 4_096.0,
            fast.results().to_vec(),
        )
        .expect("a larger finite latency is a legal response");
        assert_ne!(
            slow.latency_ms().to_bits(),
            fast.latency_ms().to_bits(),
            "the fixture must actually differ in the excluded field, or this proves nothing"
        );
        assert_eq!(
            slow, fast,
            "latency_ms participates in no equality comparison (contract setfit-apr-v1, \
             classify_response_schema)"
        );

        // And the exclusion is NARROW: every other field still decides equality.
        let other_artifact = ClassifyResponse::new(
            "0".repeat(64),
            fast.backend().to_string(),
            fast.latency_ms(),
            fast.results().to_vec(),
        )
        .expect("a different artifact hash is still a legal response");
        assert_ne!(
            other_artifact, fast,
            "excluding the measurement must not excuse excluding the identity"
        );

        let other_backend = ClassifyResponse::new(
            fast.artifact_sha256().to_string(),
            format!("{}-elsewhere", fast.backend()),
            fast.latency_ms(),
            fast.results().to_vec(),
        )
        .expect("a different backend string is still a legal response");
        assert_ne!(
            other_backend, fast,
            "two surfaces reporting different backends are not the same response"
        );

        let fewer_rows = ClassifyResponse::new(
            fast.artifact_sha256().to_string(),
            fast.backend().to_string(),
            fast.latency_ms(),
            fast.results()[..1].to_vec(),
        )
        .expect("a one-row response is legal");
        assert_ne!(
            fewer_rows, fast,
            "the results are the answer; equality must still see them"
        );
    }
}
