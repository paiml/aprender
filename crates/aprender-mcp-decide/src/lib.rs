//! aprender-mcp-decide — a THIN, single-model MCP server for decision models (Phase 8 D-15).
//!
//! # One model, one tool, on purpose
//!
//! This crate is deliberately NOT a general-purpose ML server. A server that wraps a
//! machine-learning model serves ONE model behind ONE task-shaped tool, so the analyst
//! who curates the connector can reason about exactly what it does. The tool here is
//! [`TOOL_NAME`] (`classify`), and it is bound to ONE task: the question and the ordered
//! labels come from the served `decide-apr-v1` artifact's own `task.json` (D-09), never
//! from the caller. The caller sends only `texts`.
//!
//! # In-process, transport only
//!
//! Classification calls `aprender-decide`'s [`Model`] (a [`aprender_decide::Decider`])
//! directly: the model is minted only by the decide-apr-v1 load ladder, loads once at
//! startup and stays warm. There is no subprocess and no second inference path (OPS-03).
//! `crates/aprender-mcp-setfit` is the template this crate copies.
//!
//! # Bounds ownership
//!
//! Every request bound is owned by THIS transport and named from
//! `contracts/decide-tool-boundary-v1.yaml`; [`ClassifyLimits::CONTRACTED`] mirrors that
//! file and a unit test asserts each field equal to it. The bounds are split by cost:
//!
//! - [`precheck`] runs on the async handler path and checks only the text COUNT and each
//!   text's UTF-8 BYTE length. It takes no model, so it cannot tokenize: caller-controlled
//!   text never buys tokenizer CPU on the protocol loop.
//! - [`classify_blocking`] runs inside ONE admitted blocking section and does the rest:
//!   tokenize once ([`Model::prepare`]), check the built-row TOKEN BUDGET, then score.
//! - [`Admission`] bounds concurrent callers of [`ClassifyService::call`]: at most
//!   `classify_max_in_flight` computations run and at most `classify_max_pending` calls are
//!   admitted; the next is refused at once. The permits move into the blocking closure, so
//!   a slot is released only when the CPU work it paid for has ended. Through the shipped
//!   transports pmcp 2.19.3 dispatches one tool call at a time (stdio: one worker; streamable
//!   HTTP: the `Server` behind a mutex), so that refusal is not reachable from a transport
//!   (`classify_admission`, plan 08-28).
//!
//! Every bound refusal is [`pmcp::Error::tool_rejected`] (plan 08-28, B-iserror): pmcp
//! 2.19.3's tool dispatch sends it as a successful `tools/call` result with `isError: true`
//! and the message as its one text content, so a client reads the refusal as a tool answer
//! it can act on, not as a protocol failure. Model and internal failures stay
//! `pmcp::Error::internal` (JSON-RPC -32603). Every refusal names the contract and the
//! constant key, reports the OBSERVED value, and never echoes caller text (ASVS V7). Input
//! texts are never logged; counts and timings are.

use std::fmt;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use aprender_decide::artifact::read_decide_apr_bytes_bounded;
use aprender_decide::{ArtifactError, DecideError, Decision, LayaError, PreparedRow};
use pmcp::types::capabilities::ServerCapabilities;
use pmcp::{Server, SimpleToolExt as _};
use serde::Serialize;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

// Transport crates (the Lambda wrapper) hold the loaded model in their state;
// re-exported so they depend on this crate alone, not on aprender-decide.
pub use aprender_decide::Decider as Model;

/// The one tool this server advertises.
pub const TOOL_NAME: &str = "classify";

/// The contract every bound and refusal here is named from.
pub const CONTRACT: &str = "contracts/decide-tool-boundary-v1.yaml";

pub use args::ClassifyArgs;

/// The argument type and its schema derive — the ONLY place in this crate's non-test code
/// where `clippy::disallowed_methods` is allowed (IN-05).
///
/// schemars' `JsonSchema` derive expands to `.unwrap()` internally, and the generated impl
/// lands at module scope where a struct-level allow cannot reach it. The allow therefore
/// covers this module, which holds that one derive and nothing else, so an `.unwrap()`
/// written anywhere else in the crate still fails `clippy -D warnings`. Re-exported at the
/// crate root, so `aprender_mcp_decide::ClassifyArgs` is unchanged for the Lambda crate.
mod args {
    #![allow(clippy::disallowed_methods)]

    use schemars::JsonSchema;
    use serde::Deserialize;

    /// The MCP argument surface of [`crate::TOOL_NAME`] (D-09).
    ///
    /// `texts` is the ONLY field. The question and the labels are the served artifact's
    /// task; a caller-supplied `labels` (or any other key) is a rejection under
    /// `deny_unknown_fields`, never a silently ignored knob.
    #[derive(Debug, Clone, Deserialize, JsonSchema)]
    #[serde(deny_unknown_fields)]
    pub struct ClassifyArgs {
        /// The ordered texts to classify; order is response order. One COMPLETE document
        /// per element (a whole message, post or comment) — do not split one document
        /// across elements, and do not join several documents into one.
        pub texts: Vec<String>,
    }
}

/// The request and admission bounds, mirrored from `decide-tool-boundary-v1`.
///
/// The fields are public so tests (and the 08-07 accepted-region probe) can build a
/// shrunk instance; the served path always uses [`ClassifyLimits::CONTRACTED`]
/// ([`ClassifyService::served`]), so no served request runs under other values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClassifyLimits {
    /// `classify_min_texts`.
    pub min_texts: usize,
    /// `classify_max_texts`.
    pub max_texts: usize,
    /// `classify_max_text_bytes` — UTF-8 bytes per text, checked before tokenization.
    pub max_text_bytes: usize,
    /// `classify_max_total_tokens` — sum of BUILT row lengths over the request.
    pub max_total_tokens: usize,
    /// `classify_max_in_flight` — computations running at once, per process.
    pub max_in_flight: usize,
    /// `classify_max_pending` — requests admitted (running + waiting), per process.
    pub max_pending: usize,
}

impl ClassifyLimits {
    /// The contracted bounds (`contracts/decide-tool-boundary-v1.yaml` `constants`), priced
    /// for the contract's `lambda_memory_mb` tier (10 240 MB since v7.0.0: at most 8 texts and
    /// 800 built tokens; the superseded 3 008 MB tier was 2 and 120). The contract derives them.
    pub const CONTRACTED: Self = Self {
        min_texts: 1,
        max_texts: 8,
        max_text_bytes: 16_384,
        max_total_tokens: 800,
        max_in_flight: 1,
        max_pending: 4,
    };
}

/// A BOUND refusal: the caller's to fix, so it is an in-band tool result
/// (`isError: true`), never a JSON-RPC error (`refusal_names_bound`, plan 08-28 B-iserror).
/// No `details`: the message is the whole refusal, so no structured copy can drift from it.
fn refusal(message: String) -> pmcp::Error {
    pmcp::Error::tool_rejected(message, None)
}

/// `classify_count_bound`: the ONE count check and its ONE refusal text, shared by
/// [`parse_args`] (on the JSON array, before any element becomes a `String`) and
/// [`precheck`] (for callers of [`ClassifyService::call`] that never went through
/// `parse_args`), so the two messages cannot drift.
fn check_count(limits: &ClassifyLimits, n: usize) -> pmcp::Result<()> {
    if n < limits.min_texts {
        return Err(refusal(format!(
            "{TOOL_NAME}: {n} texts is fewer than classify_min_texts {} ({CONTRACT}); \
             send at least one text",
            limits.min_texts
        )));
    }
    if n > limits.max_texts {
        return Err(refusal(format!(
            "{TOOL_NAME}: {n} texts exceeds classify_max_texts {} ({CONTRACT}); split the batch",
            limits.max_texts
        )));
    }
    Ok(())
}

/// The cheap bounds, in contract order: COUNT, then per-text UTF-8 BYTES.
///
/// Runs on the async handler path. It takes no model, so it cannot tokenize — the
/// token budget is checked by [`classify_blocking`] inside the admitted blocking
/// section.
///
/// # Errors
///
/// A bound refusal (`pmcp::Error::tool_rejected`) naming the contract, the violated key and
/// the observed value; never the text.
pub fn precheck(limits: &ClassifyLimits, args: &ClassifyArgs) -> pmcp::Result<()> {
    check_count(limits, args.texts.len())?;
    for (index, text) in args.texts.iter().enumerate() {
        let bytes = text.len();
        if bytes > limits.max_text_bytes {
            return Err(refusal(format!(
                "{TOOL_NAME}: texts[{index}] is {bytes} UTF-8 bytes, over \
                 classify_max_text_bytes {} ({CONTRACT}); send a shorter document",
                limits.max_text_bytes
            )));
        }
    }
    Ok(())
}

/// Why the blocking section refused or failed.
#[derive(Debug, Clone, PartialEq)]
pub enum ClassifyFailure {
    /// The built rows exceed `classify_max_total_tokens` — the caller can fix this.
    TokenBudget {
        /// Sum of built-row lengths over the request.
        total: usize,
        /// Each text's built-row length, in input order.
        per_text: Vec<usize>,
        /// The budget that was exceeded.
        limit: usize,
    },
    /// The decision model refused (tokenizer, builder or forward) — internal.
    Model(DecideError),
    /// A decision named a label index outside the task — internal (a defect).
    LabelIndex {
        /// The out-of-range index.
        index: usize,
        /// The number of labels the task has.
        labels: usize,
    },
}

impl fmt::Display for ClassifyFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TokenBudget {
                total,
                per_text,
                limit,
            } => write!(
                f,
                "{TOOL_NAME}: built rows total {total} tokens (per text {per_text:?}), over \
                 classify_max_total_tokens {limit} ({CONTRACT}); send fewer or shorter texts"
            ),
            // The tokenizer's own message may quote its input, so it is withheld here:
            // no error this server returns may carry caller text (ASVS V7).
            Self::Model(DecideError::Laya(LayaError::Tokenizer(_))) => {
                write!(
                    f,
                    "{TOOL_NAME}: the tokenizer refused a text (detail withheld)"
                )
            }
            Self::Model(e) => write!(f, "{TOOL_NAME}: {e}"),
            Self::LabelIndex { index, labels } => write!(
                f,
                "{TOOL_NAME}: decision label index {index} is outside the task's {labels} labels"
            ),
        }
    }
}

impl std::error::Error for ClassifyFailure {}

impl ClassifyFailure {
    /// Map onto the MCP error taxonomy: the caller-fixable budget refusal is a bound
    /// refusal (an `isError` tool result); everything else is internal (JSON-RPC -32603).
    #[must_use]
    pub fn into_pmcp(self) -> pmcp::Error {
        match self {
            Self::TokenBudget { .. } => refusal(self.to_string()),
            other => pmcp::Error::internal(other.to_string()),
        }
    }
}

/// The token budget over already-built row lengths (pure; no model).
///
/// # Errors
///
/// [`ClassifyFailure::TokenBudget`] when the sum exceeds `limits.max_total_tokens`.
pub fn check_token_budget(
    limits: &ClassifyLimits,
    per_text: &[usize],
) -> Result<(), ClassifyFailure> {
    let total = per_text
        .iter()
        .fold(0usize, |acc, &n| acc.saturating_add(n));
    if total > limits.max_total_tokens {
        return Err(ClassifyFailure::TokenBudget {
            total,
            per_text: per_text.to_vec(),
            limit: limits.max_total_tokens,
        });
    }
    Ok(())
}

/// The admitted blocking section: tokenize ONCE, check the token budget, then score.
///
/// Call it only after [`precheck`] and inside an admitted [`Ticket`] — it is seconds of
/// CPU for a full-window row.
///
/// # Errors
///
/// [`ClassifyFailure::TokenBudget`] before any forward runs, or
/// [`ClassifyFailure::Model`] for a refusal from the model.
pub fn classify_blocking(
    model: &Model,
    limits: &ClassifyLimits,
    texts: &[String],
) -> Result<Vec<Decision>, ClassifyFailure> {
    let rows = model.prepare(texts).map_err(ClassifyFailure::Model)?;
    let per_text: Vec<usize> = rows.iter().map(PreparedRow::tokens).collect();
    check_token_budget(limits, &per_text)?;
    model
        .classify_prepared(&rows)
        .map_err(ClassifyFailure::Model)
}

/// The admission refusal: `classify_max_pending` requests are already admitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Busy {
    /// The pending bound that is full.
    pub max_pending: usize,
}

impl fmt::Display for Busy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{TOOL_NAME}: {} requests are already admitted, the classify_max_pending bound \
             ({CONTRACT}); retry later",
            self.max_pending
        )
    }
}

impl std::error::Error for Busy {}

impl Busy {
    /// The refusal as a bound refusal (an `isError` tool result) naming
    /// `classify_max_pending`.
    #[must_use]
    pub fn into_pmcp(self) -> pmcp::Error {
        refusal(self.to_string())
    }
}

/// Why an admitted computation did not produce a value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdmissionError {
    /// The in-flight semaphore was closed (never happens: nothing closes it).
    Closed,
    /// The blocking task panicked or was cancelled by the runtime.
    Join(String),
}

impl fmt::Display for AdmissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Closed => write!(f, "{TOOL_NAME}: admission closed"),
            Self::Join(e) => write!(f, "{TOOL_NAME}: classify task join: {e}"),
        }
    }
}

impl std::error::Error for AdmissionError {}

/// Process-wide admission (`decide-tool-boundary-v1` `classify_admission`).
///
/// Two semaphores sized from the contract: `pending` (running + waiting, taken
/// without waiting — a request past the cap is refused at once, never queued without
/// bound) and `in_flight` (awaited by an admitted request). Clones share the same
/// semaphores; create ONE per server process.
#[derive(Debug, Clone)]
pub struct Admission {
    in_flight: Arc<Semaphore>,
    pending: Arc<Semaphore>,
    max_pending: usize,
}

impl Admission {
    /// Admission sized from `limits.max_in_flight` and `limits.max_pending`.
    #[must_use]
    pub fn new(limits: &ClassifyLimits) -> Self {
        Self {
            in_flight: Arc::new(Semaphore::new(limits.max_in_flight)),
            pending: Arc::new(Semaphore::new(limits.max_pending)),
            max_pending: limits.max_pending,
        }
    }

    /// Take a pending slot now, or refuse now.
    ///
    /// # Errors
    ///
    /// [`Busy`] when `max_pending` requests are already admitted.
    pub fn try_admit(&self) -> Result<Ticket, Busy> {
        let pending = Arc::clone(&self.pending)
            .try_acquire_owned()
            .map_err(|_| Busy {
                max_pending: self.max_pending,
            })?;
        Ok(Ticket {
            pending,
            in_flight: Arc::clone(&self.in_flight),
        })
    }
}

/// An admitted request: it holds a pending slot and may wait for the in-flight slot.
///
/// Dropping a ticket (or the future of [`Ticket::run_blocking`] while it still waits)
/// releases the pending slot and runs nothing.
#[derive(Debug)]
pub struct Ticket {
    pending: OwnedSemaphorePermit,
    in_flight: Arc<Semaphore>,
}

impl Ticket {
    /// Wait for the in-flight slot, then run `work` on a blocking thread.
    ///
    /// BOTH owned permits move into the blocking closure and are dropped only when
    /// `work` returns. A caller that drops this future while `work` runs therefore does
    /// NOT free either slot early: the blocking task cannot be cancelled, its result is
    /// discarded, and the slots count the CPU work it is still doing.
    ///
    /// # Errors
    ///
    /// [`AdmissionError`] if the slot cannot be taken or the blocking task fails.
    pub async fn run_blocking<F, T>(self, work: F) -> Result<T, AdmissionError>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let slot = Arc::clone(&self.in_flight)
            .acquire_owned()
            .await
            .map_err(|_| AdmissionError::Closed)?;
        let pending = self.pending;
        tokio::task::spawn_blocking(move || {
            let held = (pending, slot);
            let out = work();
            drop(held);
            out
        })
        .await
        .map_err(|e| AdmissionError::Join(e.to_string()))
    }
}

/// The served artifact's identity (D-11), in every response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelView {
    /// sha256 of the WHOLE served `.apr` file.
    pub artifact_sha256: String,
    /// sha256 of the artifact's `recipe.json` blob.
    pub recipe_id: String,
    /// The decision method (`laya`).
    pub method: String,
    /// The declared base, e.g. `laya-en-root@55cf4c4e`.
    pub base: String,
}

/// One text's decision.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ResultView {
    /// The most probable label (argmax, first on ties).
    pub label: String,
    /// One calibrated probability per label, in the order of the top-level `labels`.
    pub probabilities: Vec<f32>,
    /// Tokens in the built row the model scored (what the budget charged).
    pub tokens: usize,
    /// True when the text was cut to fit the model's window (D-12).
    pub truncated: bool,
}

/// The `classify` response (D-11, D-12): ARRAYS in task order, never a label-keyed map.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ClassifyResponse {
    /// Which model answered.
    pub model: ModelView,
    /// The task's labels, in `task.json` criteria order.
    pub labels: Vec<String>,
    /// One result per input text, in input order.
    pub results: Vec<ResultView>,
}

impl ClassifyResponse {
    /// Build the response from `model`'s identity and labels and its decisions.
    ///
    /// # Errors
    ///
    /// [`ClassifyFailure::LabelIndex`] if a decision names a label outside the task.
    pub fn from_decisions(model: &Model, decisions: &[Decision]) -> Result<Self, ClassifyFailure> {
        let id = model.identity();
        let labels = model.task().owned_labels();
        let results = decisions
            .iter()
            .map(|d| {
                let label =
                    labels
                        .get(d.label_index)
                        .cloned()
                        .ok_or(ClassifyFailure::LabelIndex {
                            index: d.label_index,
                            labels: labels.len(),
                        })?;
                Ok(ResultView {
                    label,
                    probabilities: d.probabilities.clone(),
                    tokens: d.tokens,
                    truncated: d.truncated,
                })
            })
            .collect::<Result<Vec<_>, ClassifyFailure>>()?;
        Ok(Self {
            model: ModelView {
                artifact_sha256: id.artifact_sha256.clone(),
                recipe_id: id.recipe_id.clone(),
                method: id.method.clone(),
                base: id.base.clone(),
            },
            labels,
            results,
        })
    }
}

/// The description's truncation sentence, DERIVED from the served window and the tier's
/// budget (plan 08-28, A-derive; WR-03).
///
/// A truncated text builds a row of exactly `max_len` tokens (the artifact's
/// `agent.max_len`). When that row fits `classify_max_total_tokens`, truncation is
/// reachable and the sentence promises it (Laya-en's 512-token window at the contracted
/// 10 240 MB tier's 800-token budget). When it does not — the same window at the superseded
/// 3 008 MB tier's 120-token budget — every text long enough to be truncated is refused by
/// the budget first, so the sentence says that instead of promising `truncated: true`.
#[must_use]
pub fn truncation_sentence(max_len: usize, limits: &ClassifyLimits) -> String {
    if max_len <= limits.max_total_tokens {
        "Long texts are truncated by the model itself to its window, and each such result \
         reports `truncated: true`."
            .to_string()
    } else {
        format!(
            "A text whose built row would exceed the {budget}-token request budget is refused \
             (classify_max_total_tokens) rather than truncated, because the model's own window \
             ({max_len} tokens) is larger than that budget: send a shorter excerpt of a long \
             document instead.",
            budget = limits.max_total_tokens
        )
    }
}

/// The tool description, built FROM THE ARTIFACT: its question, its labels in order
/// (with their criteria), the bounds, the one-document guidance, and the truncation
/// sentence derived from the artifact's window and the contracted budget
/// ([`truncation_sentence`]).
#[must_use]
pub fn tool_description(model: &Model) -> String {
    tool_description_for(model, &ClassifyLimits::CONTRACTED)
}

/// [`tool_description`] under `limits`. Private: the served description is always the
/// contracted one; the unit tests shrink the budget to reach the other truncation branch.
fn tool_description_for(model: &Model, limits: &ClassifyLimits) -> String {
    let task = model.task();
    let truncation = truncation_sentence(model.manifest().agent.max_len, limits);
    let labels = task.labels().join(", ");
    let criteria: String = task
        .criteria()
        .iter()
        .map(|c| match &c.description {
            Some(d) => format!("\n- {}: {d}", c.name),
            None => format!("\n- {}", c.name),
        })
        .collect();
    format!(
        "Classify texts for ONE fixed decision, the task this server was deployed with: \
         \"{question}\" The labels, in this order: [{labels}].{criteria}\n\
         Each element of `texts` is ONE complete document (e.g. one whole customer message) \
         and yields exactly one decision — NEVER split a single document into multiple \
         elements (fragments decide worse than the whole) and never join separate documents \
         into one element. {truncation} Returns `model` (the served artifact's \
         identity: artifact_sha256, recipe_id, method, base), `labels` (the order above), and \
         `results`, one per text in input order: `label` (the most probable), \
         `probabilities` (calibrated, one per label in `labels` order), `tokens`, \
         `truncated`. Bounded per call ({CONTRACT}): {min}..={max} texts, at most \
         {bytes} UTF-8 bytes per text, at most {tokens} model tokens over the whole request.",
        question = task.instructions(),
        min = limits.min_texts,
        max = limits.max_texts,
        bytes = limits.max_text_bytes,
        tokens = limits.max_total_tokens,
    )
}

/// One classify service per server process: the model, its admission and its bounds.
#[derive(Debug, Clone)]
pub struct ClassifyService {
    model: Arc<Model>,
    admission: Admission,
    limits: ClassifyLimits,
}

impl ClassifyService {
    /// The SERVED service: always [`ClassifyLimits::CONTRACTED`], with one fresh
    /// [`Admission`] sized from it.
    #[must_use]
    pub fn served(model: Arc<Model>) -> Self {
        Self::with_limits(model, ClassifyLimits::CONTRACTED)
    }

    /// A service under other limits. Private: the only public door is
    /// [`ClassifyService::served`]; the unit tests use this to shrink a bound.
    fn with_limits(model: Arc<Model>, limits: ClassifyLimits) -> Self {
        Self {
            admission: Admission::new(&limits),
            model,
            limits,
        }
    }

    /// The bounds this service enforces.
    #[must_use]
    pub fn limits(&self) -> &ClassifyLimits {
        &self.limits
    }

    /// The whole handler path: precheck (count, bytes) -> admission -> the blocking
    /// section (tokenize, token budget, score) -> response.
    ///
    /// # Errors
    ///
    /// Bound refusals (`tool_rejected`) for every bound and for admission; internal errors
    /// for model failures.
    pub async fn call(&self, args: ClassifyArgs) -> pmcp::Result<ClassifyResponse> {
        precheck(&self.limits, &args)?;
        let ticket = self.admission.try_admit().map_err(Busy::into_pmcp)?;
        let model = Arc::clone(&self.model);
        let limits = self.limits;
        let count = args.texts.len();
        let started = Instant::now();
        let outcome = ticket
            .run_blocking(move || {
                let decisions = classify_blocking(&model, &limits, &args.texts)?;
                ClassifyResponse::from_decisions(&model, &decisions)
            })
            .await
            .map_err(|e| pmcp::Error::internal(e.to_string()))?;
        let ms = started.elapsed().as_millis();
        match outcome {
            Ok(response) => {
                let tokens: usize = response.results.iter().map(|r| r.tokens).sum();
                eprintln!("{TOOL_NAME}: ok texts={count} tokens={tokens} ms={ms}");
                Ok(response)
            }
            Err(failure) => {
                eprintln!("{TOOL_NAME}: refused texts={count} ms={ms}");
                Err(failure.into_pmcp())
            }
        }
    }
}

/// Why a model failed to load at startup.
#[derive(Debug)]
pub enum ModelLoadError {
    /// The artifact file could not be opened or statted.
    Io(std::io::Error),
    /// The bytes were refused by the decide-apr-v1 load ladder (the rung is named).
    Artifact(ArtifactError),
}

impl fmt::Display for ModelLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "cannot read model artifact: {e}"),
            Self::Artifact(e) => write!(f, "model artifact refused: {e}"),
        }
    }
}

impl std::error::Error for ModelLoadError {}

/// Load and fully verify a `decide-apr-v1` artifact from a file path.
///
/// The read is bounded by the file's declared length BEFORE the bytes land in memory
/// (ladder rung 1), then the bytes go through the whole ladder. This,
/// [`load_model_from_bytes`] and [`load_model_from_hashed`] are the only load doors; a
/// server never mints its own.
///
/// # Errors
///
/// [`ModelLoadError::Io`] if the file cannot be opened or statted;
/// [`ModelLoadError::Artifact`] for an oversized read or any refusing rung.
pub fn load_model_from_path(path: &Path) -> Result<Model, ModelLoadError> {
    let file = std::fs::File::open(path).map_err(ModelLoadError::Io)?;
    let declared = file.metadata().map_err(ModelLoadError::Io)?.len();
    let bytes =
        read_decide_apr_bytes_bounded(file, Some(declared)).map_err(ModelLoadError::Artifact)?;
    load_model_from_bytes(&bytes)
}

/// Load and fully verify a `decide-apr-v1` artifact already in memory — the door the
/// Lambda crate walks through after fetching the file from S3.
///
/// # Errors
///
/// [`ModelLoadError::Artifact`] for any refusing rung.
pub fn load_model_from_bytes(bytes: &[u8]) -> Result<Model, ModelLoadError> {
    Model::load_bytes(bytes).map_err(ModelLoadError::Artifact)
}

/// [`load_model_from_bytes`] over bytes already hashed by
/// [`aprender_decide::artifact::HashedArtifact::new`] — the door the Lambda crate walks
/// through after checking its pin against that digest, so the ladder mints the identity
/// from it instead of hashing ~0.85 GB a second time.
///
/// # Errors
///
/// [`ModelLoadError::Artifact`] for any refusing rung.
pub fn load_model_from_hashed(
    hashed: &aprender_decide::artifact::HashedArtifact<'_>,
) -> Result<Model, ModelLoadError> {
    Model::load_hashed(hashed).map_err(ModelLoadError::Artifact)
}

/// Deserialize the tool arguments, COUNT FIRST, refusing a malformed shape WITHOUT serde's
/// message.
///
/// When `texts` is a JSON array, its length is checked against
/// [`ClassifyLimits::CONTRACTED`] before anything is deserialized, so an oversized list is
/// refused naming `classify_max_texts` whatever its elements are, and costs one length
/// comparison instead of one `String` per element (V5-c). pmcp has already parsed the frame
/// into this `Value` — on stdio without a framing cap (pmcp 2.19.3), over HTTP inside the
/// stateless config's 4 MiB request cap — so the saving is the work AFTER that parse.
///
/// Only then is the strict shape deserialized: for a string where the list belongs, or an
/// unknown key, serde quotes the offending value or key — caller text — which
/// `refusal_names_bound` forbids (ASVS V7). The refusal names the contract and the one
/// accepted shape instead.
///
/// # Errors
///
/// A bound refusal (`pmcp::Error::tool_rejected`) naming the contract (and, for the count,
/// the violated key and the observed count); never the arguments.
pub fn parse_args(args: serde_json::Value) -> pmcp::Result<ClassifyArgs> {
    if let Some(texts) = args.get("texts").and_then(serde_json::Value::as_array) {
        check_count(&ClassifyLimits::CONTRACTED, texts.len())?;
    }
    serde_json::from_value(args).map_err(|_| {
        refusal(format!(
            "{TOOL_NAME}: the arguments must be exactly {{\"texts\": [string, ...]}} and nothing \
             else (classify_count_bound precondition: deny_unknown_fields, {CONTRACT}); \
             detail withheld so no caller text is echoed"
        ))
    })
}

/// decide-tool-boundary-v1 `classify_token_budget`: `classify_max_texts x` the served task's
/// shortest built row (an empty text: task prefix and markers) must fit
/// `classify_max_total_tokens` — otherwise the advertised count is one the budget can never
/// admit, and every call at that count would be refused. Returns that shortest row.
///
/// # Errors
///
/// `pmcp::Error::internal` when the task cannot be prepared; a bound refusal
/// (`tool_rejected`, naming `classify_max_total_tokens`) when the contracted count does not
/// fit the budget for this artifact's task. It is a BUILD-time refusal: it stops
/// [`build_server`], so it never reaches a `tools/call`.
pub fn check_served_task_fits(model: &Model, limits: &ClassifyLimits) -> pmcp::Result<usize> {
    let rows = model
        .prepare(&[String::new()])
        .map_err(|e| pmcp::Error::internal(format!("{TOOL_NAME}: {e}")))?;
    let min_row = rows.first().map_or(0, PreparedRow::tokens);
    let need = min_row.saturating_mul(limits.max_texts);
    if need > limits.max_total_tokens {
        return Err(refusal(format!(
            "{TOOL_NAME}: the served task's shortest built row is {min_row} tokens, so \
             classify_max_texts {} needs {need} > classify_max_total_tokens {} ({CONTRACT}); \
             this artifact's task cannot be served at the contracted tier",
            limits.max_texts, limits.max_total_tokens
        )));
    }
    Ok(min_row)
}

/// Build the MCP server: exactly one `classify` tool over one loaded model.
///
/// The tool runs [`ClassifyService::served`] — the contracted bounds and one
/// process-wide [`Admission`] — so the blocking section never stalls the protocol loop
/// and never oversubscribes the CPU. Its arguments are parsed by [`parse_args`] (the same
/// strict schema, advertised from [`ClassifyArgs`]) so no refusal echoes caller text.
///
/// # Errors
///
/// `pmcp::Error` if the served task cannot fit the contracted bounds
/// ([`check_served_task_fits`]) or the server builder refuses the configuration.
pub fn build_server(model: Arc<Model>, name: &str, version: &str) -> pmcp::Result<Server> {
    build_server_with_limits(model, ClassifyLimits::CONTRACTED, name, version)
}

/// [`build_server`] under other limits. Private: the only public door serves
/// [`ClassifyLimits::CONTRACTED`]; the unit tests shrink a bound here to show the
/// served-task fit check refuses at build time.
fn build_server_with_limits(
    model: Arc<Model>,
    limits: ClassifyLimits,
    name: &str,
    version: &str,
) -> pmcp::Result<Server> {
    check_served_task_fits(&model, &limits)?;
    let description = tool_description_for(&model, &limits);
    let service = ClassifyService::with_limits(model, limits);
    let tool = pmcp::SimpleTool::new(
        TOOL_NAME,
        move |args: serde_json::Value, _extra: pmcp::RequestHandlerExtra| -> ToolFuture {
            let service = service.clone();
            Box::pin(async move {
                let response = service.call(parse_args(args)?).await?;
                serde_json::to_value(&response)
                    .map_err(|e| pmcp::Error::internal(format!("response serialization: {e}")))
            })
        },
    )
    .with_description(description)
    .with_schema_from::<ClassifyArgs>();
    Server::builder()
        .name(name)
        .version(version)
        .capabilities(ServerCapabilities::tools_only())
        .tool(TOOL_NAME, tool)
        .build()
}

/// The boxed future a [`pmcp::SimpleTool`] handler returns.
type ToolFuture =
    std::pin::Pin<Box<dyn std::future::Future<Output = pmcp::Result<serde_json::Value>> + Send>>;

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // serde_json::json! / schema_for! expand to .unwrap()
mod tests;
