//! The receipt emitter — the half of the receipt rule that was missing.
//!
//! # The defect
//!
//! `scripts/perf_gate.sh` reads `drain_ms`, `tokenization.method`, `timeouts`
//! and `requested`/`completed` off a receipt, and **nothing in the workspace
//! wrote a receipt at all**. Measurement code can hold a perfect `drain_ms` in a
//! struct field forever; the gate never sees a struct. The only artefact with
//! `drain_ms` in it on `62d23d8d1` was the gate's own hand-typed selftest
//! fixture (`"drain_ms":12`), so the gate was green on fiction and red on every
//! measurement that could actually be taken.
//!
//! [`ReceiptInput::render`] closes that: it takes the **per-request terminal
//! records** and derives the receipt from them. There is deliberately no way to
//! hand it a `drain_ms`, a `timeouts` count, or an `agg_ratio` — every number it
//! emits is computed from samples that travel in the same receipt, which is the
//! rule `scripts/lib/bench_receipt.py` already applies to ratios.
//!
//! # What this refuses to emit
//!
//! - **§4.4.9's scheduler block.** `max_in_flight`, `admission_rejected`,
//!   `preempted_recompute`, `preempted_swap`, `kv_blocks_*`, `gpu_layers_*`,
//!   `backend_loaded[]`, `autofit_applied[]` are **server**-reported by
//!   construction — I-16 says `max_in_flight` "is reported by the **server**,
//!   not inferred by the harness", and I-2 says `gpu_layers_resolved` is read
//!   from the loader. A client-side estimate would be indistinguishable from a
//!   real answer, which is worse than a missing field. The block is omitted and
//!   named in `unproduced_fields`, with the reason.
//! - **Arm D's `kv` block**, for the same reason, unless a caller supplies one
//!   via [`KvBlock::from_server_report`]. Without it the receipt is legal at
//!   merge phase and honestly fails at release phase, which is the correct
//!   posture: Arm D is REPORTING on its threshold, not on its fields.
//! - **Any comparator ratio.** See [`super::drain::ComparatorStatus`]: there is
//!   no `Measured` variant, because a ratio needs a baseline object that itself
//!   passes every receipt rule (I-3) and a comparator lane driven by the same
//!   client binary (I-15).
//! - **A default `resolution`.** Every [`Provenance`] string is required and an
//!   empty one is refused. A `--resolution` that defaults to `scripts/apr_bin.sh`
//!   invents provenance, and invented provenance is indistinguishable from
//!   measured provenance.

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::str::FromStr;

use super::drain::{BandInput, ComparatorStatus, DerivedBand};

/// §4.4.9 fields a client cannot observe, and why. Emitted verbatim into the
/// receipt's `unproduced_fields` rather than guessed at.
pub const SERVER_ONLY_FIELDS: &str = "§4.4.9 scheduler block (max_in_flight, admission_rejected, \
     preempted_recompute, preempted_swap, kv_blocks_total, kv_blocks_peak_used, \
     kv_bytes_reserved, kv_bytes_used, gpu_layers_requested, gpu_layers_resolved, \
     gpu_layers_total, backend_loaded[], autofit_applied[]) — every one is reported by the \
     SERVER. I-16: max_in_flight is reported by the server, not inferred by the harness; I-2: \
     gpu_layers_resolved is read from the loader and never inferred. A client-side estimate \
     would read exactly like a measurement, so none is emitted.";

/// The dispatch path a run actually took. Mirrors `bench_receipt.py`'s
/// `COMPUTE_CLASSES`; I-2 requires this be the path **taken**, read from the
/// running process, never the hardware present.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputeClass {
    /// SIMD on the host CPU.
    Cpu,
    /// NVIDIA CUDA.
    Cuda,
    /// Apple Metal.
    Metal,
    /// wgpu.
    Wgpu,
    /// The run could not determine its own path. Legal, and never a ratio.
    Unknown,
}

impl ComputeClass {
    /// The wire token, matching `bench_receipt.py`.
    #[must_use]
    pub fn wire_token(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
            Self::Wgpu => "wgpu",
            Self::Unknown => "unknown",
        }
    }
}

/// Parse the wire token back.
///
/// The reverse of [`ComputeClass::wire_token`] rather than a second table:
/// `wire_token` is what `bench_receipt.py` matches against `COMPUTE_CLASSES`,
/// so a parser with its own spelling would let a receipt be written with a
/// class the validator then rejects — after the measurement had been taken.
/// `roundtrip_is_the_only_spelling` pins the two together.
impl FromStr for ComputeClass {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        [
            Self::Cpu,
            Self::Cuda,
            Self::Metal,
            Self::Wgpu,
            Self::Unknown,
        ]
        .into_iter()
        .find(|c| c.wire_token() == s)
        .ok_or_else(|| {
            format!(
                "compute_class {s:?}: expected one of cpu, cuda, metal, wgpu, unknown (I-2 \
                 requires the path TAKEN, not the hardware present)"
            )
        })
    }
}

/// §4.2.2 identity plus the §4.2.3 join key. **No `Default` impl**: every field
/// is a fact about a specific run, and a blank one that serialises is how a
/// receipt acquires provenance it never had.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Provenance {
    /// The binary that ran, as an absolute path.
    pub binary_path: String,
    /// Host-local anti-substitution fingerprint. 64 lowercase hex characters.
    pub binary_sha256: String,
    /// How that path was resolved. **No default** — see the module docs.
    pub resolution: String,
    /// The dispatch path taken (I-2).
    pub compute_class: ComputeClass,
    /// Join key: which host.
    pub host: String,
    /// Join key: which accelerator.
    pub accelerator: String,
    /// Join key: which model.
    pub model: String,
    /// Join key: which quantization.
    pub quantization: String,
    /// Cargo features read **from the built binary**, never from `Cargo.toml`.
    pub feature_set: Vec<String>,
}

impl Provenance {
    /// §4.2 checks that a receipt cannot be written without passing.
    ///
    /// # Errors
    /// When any field is empty, the digest is not 64 lowercase hex characters,
    /// or the declared `compute_class` is a path the build cannot reach.
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in self.required_strings() {
            if value.trim().is_empty() {
                return Err(format!(
                    "provenance.{name}: empty — this field has no default; a receipt that does \
                     not say {name} is an anonymous number, not evidence"
                ));
            }
        }
        if !is_sha256(&self.binary_sha256) {
            return Err(format!(
                "provenance.binary_sha256: {:?} is not 64 lowercase hex characters",
                self.binary_sha256
            ));
        }
        self.validate_feature_set()
    }

    fn required_strings(&self) -> [(&'static str, &str); 7] {
        [
            ("binary_path", &self.binary_path),
            ("binary_sha256", &self.binary_sha256),
            ("resolution", &self.resolution),
            ("host", &self.host),
            ("accelerator", &self.accelerator),
            ("model", &self.model),
            ("quantization", &self.quantization),
        ]
    }

    /// I-2's other half: a class the build cannot reach is a fabricated claim.
    fn validate_feature_set(&self) -> Result<(), String> {
        let needs_feature = matches!(self.compute_class, ComputeClass::Cuda | ComputeClass::Wgpu);
        let token = self.compute_class.wire_token();
        if needs_feature && !self.feature_set.iter().any(|f| f == token) {
            return Err(format!(
                "provenance.compute_class={token} but feature_set={:?} does not contain it — a \
                 build without the feature cannot take that path (I-2)",
                self.feature_set
            ));
        }
        Ok(())
    }

    fn to_json(&self) -> Value {
        json!({
            "binary_path": self.binary_path,
            "binary_sha256": self.binary_sha256,
            "resolution": self.resolution,
            "compute_class": self.compute_class.wire_token(),
            "host": self.host,
            "accelerator": self.accelerator,
            "model": self.model,
            "quantization": self.quantization,
            "feature_set": self.feature_set,
        })
    }
}

/// §4.4.6 — how tokens were counted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenCountingMethod {
    /// Counts taken from the server's own `usage` fields — two servers'
    /// `usage` fields are two implementations' opinions.
    ServerUsage,
    /// Counts computed client-side with the model's own tokenizer. Canonical.
    ClientTokenizer,
    /// Counts of the streamed content chunks the client observed, with **no
    /// tokenizer involved at all**.
    ///
    /// This is not a third thing an operator may declare — it is the third
    /// thing that can be *true*, and until PERF-048 it had no spelling, so a
    /// streaming run against `apr serve` recorded it as `server_usage`
    /// (#2754). `apr serve`'s SSE stream carries no `usage` object, so
    /// `llm::band::usage_counts` fell back to `token_timestamps.len()`: the
    /// number of SSE deltas with non-empty `content`.
    ///
    /// **That equals a token count only for a server that emits exactly one
    /// token per chunk.** PERF-045 measured 960 = 30 x 32 exactly against one
    /// build of `apr serve`, which is an observation about that server on that
    /// day and never an invariant: a server emitting multi-token chunks would
    /// make this count silently wrong. Recording it under its own name is what
    /// lets a reader — and the gate — know which of those two worlds a number
    /// came from. There is deliberately no `tokenizer_sha256` here, because no
    /// tokenizer ran and a digest would name one that did not.
    ClientChunkCount,
}

impl TokenCountingMethod {
    /// The wire token `perf_gate.sh` reads from `tokenization.method`.
    #[must_use]
    pub fn wire_token(self) -> &'static str {
        match self {
            Self::ServerUsage => "server_usage",
            Self::ClientTokenizer => "client_tokenizer",
            Self::ClientChunkCount => "client_chunk_count",
        }
    }

    /// True when this method's counts are comparable across two servers.
    ///
    /// §4.4.6 exists so a comparator ratio is only computed when both sides
    /// count the same way. [`Self::ClientChunkCount`] is not a token count at
    /// all unless the server happens to emit one token per chunk, so a ratio
    /// taken over it compares two servers' chunking policies.
    #[must_use]
    pub fn is_a_token_count(self) -> bool {
        !matches!(self, Self::ClientChunkCount)
    }
}

/// §4.4.6 — the `tokenization` block, required in every receipt.
///
/// `method` has **no default** (I-13). The variants below make
/// "`client_tokenizer` with no digest" unrepresentable rather than merely
/// rejected, and `counts_special_tokens` / `counts_prompt_echo` are plain
/// non-optional booleans the caller must state.
///
/// > `counts_*` under `server_usage` is an operator **declaration** about the
/// > server's counting semantics, not a client measurement — §4.4.6 is titled
/// > "Token counting must be **declared**". It is required rather than
/// > defaulted precisely so it cannot be silently wrong.
///
/// PERF-024's measurement side used to carry a second spelling of this block,
/// `perf_gate::protocol::Tokenization` — a struct with an `Option<String>`
/// digest and a `validate()`. The merge kept this enum, because it makes
/// "`client_tokenizer` with no digest" unrepresentable rather than merely
/// rejected, and moved that struct's one extra behaviour
/// ([`Self::require_counter`]) here. `protocol` now re-exports this type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizationBlock {
    /// Server-reported counts.
    ServerUsage {
        /// Whether the count includes special tokens.
        counts_special_tokens: bool,
        /// Whether the count includes the echoed prompt.
        counts_prompt_echo: bool,
    },
    /// Client-side counts with the model's own tokenizer.
    ClientTokenizer {
        /// Digest of the tokenizer actually used. 64 lowercase hex characters.
        tokenizer_sha256: String,
        /// Whether the count includes special tokens.
        counts_special_tokens: bool,
        /// Whether the count includes the echoed prompt.
        counts_prompt_echo: bool,
    },
}

impl TokenizationBlock {
    /// The declared counting method.
    #[must_use]
    pub fn method(&self) -> TokenCountingMethod {
        match self {
            Self::ServerUsage { .. } => TokenCountingMethod::ServerUsage,
            Self::ClientTokenizer { .. } => TokenCountingMethod::ClientTokenizer,
        }
    }

    /// # Errors
    /// When a `client_tokenizer` digest is not 64 lowercase hex characters.
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::ServerUsage { .. } => Ok(()),
            Self::ClientTokenizer {
                tokenizer_sha256, ..
            } if is_sha256(tokenizer_sha256) => Ok(()),
            Self::ClientTokenizer {
                tokenizer_sha256, ..
            } => Err(format!(
                "tokenization.tokenizer_sha256: {tokenizer_sha256:?} is not 64 lowercase hex \
                 characters — §4.4.6 requires it when method = client_tokenizer"
            )),
        }
    }

    /// Poka-yoke for transports: a declared method the transport cannot honour
    /// is refused at construction, not silently downgraded at measure time.
    ///
    /// Moved here from `protocol::Tokenization` when the two spellings of the
    /// §4.4.6 block were merged — the enum kept the shape, this kept its one
    /// behaviour the enum lacked.
    ///
    /// # Errors
    /// When the declared method and the available counting machinery disagree.
    pub fn require_counter(&self, has_client_counter: bool) -> Result<(), String> {
        match (self.method(), has_client_counter) {
            (TokenCountingMethod::ClientTokenizer, false) => Err(
                "tokenization.method = client_tokenizer but no client TokenCounter was supplied"
                    .to_string(),
            ),
            (TokenCountingMethod::ServerUsage, true) => Err(
                "tokenization.method = server_usage but a client TokenCounter was supplied; \
                 declare client_tokenizer or drop the counter"
                    .to_string(),
            ),
            _ => Ok(()),
        }
    }

    /// The digest, when this block carries one.
    fn digest(&self) -> Option<&str> {
        match self {
            Self::ServerUsage { .. } => None,
            Self::ClientTokenizer {
                tokenizer_sha256, ..
            } => Some(tokenizer_sha256),
        }
    }

    fn counts(&self) -> (bool, bool) {
        match self {
            Self::ServerUsage {
                counts_special_tokens,
                counts_prompt_echo,
            }
            | Self::ClientTokenizer {
                counts_special_tokens,
                counts_prompt_echo,
                ..
            } => (*counts_special_tokens, *counts_prompt_echo),
        }
    }
}

/// §4.4.6 — what the run OBSERVED about who counted the tokens.
///
/// # The defect this exists to make impossible
///
/// PERF-045 ran the §4.4 band protocol for the first time and every streaming
/// receipt said `tokenization.method: "server_usage"` while the tokens were
/// counted by the client (#2754). Nothing was lying on purpose:
/// `llm::band::usage_counts` takes a fallback arm when the response carries no
/// `usage` object, `apr serve`'s SSE stream carries none, and
/// [`TokenizationBlock::validate`] cross-checks the declaration against
/// *nothing*. All 30 samples carried `prompt_tokens: 0` and
/// `generated_tokens == token_times_s.len()`, and the receipt reported itself
/// conformant.
///
/// A declaration that no observation can contradict is decoration. This is the
/// observation: two counters, taken from the retained samples, over which
/// [`ResolvedTokenization::resolve`] computes the method that was **used**.
/// The used method is therefore a pure function of what happened, and cannot be
/// made to agree with the declaration by editing the declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizationObservation {
    /// Completed sampled responses that carried a server `usage` object.
    pub responses_with_server_usage: usize,
    /// Completed sampled responses whose tokens a **client-side tokenizer**
    /// counted.
    ///
    /// Zero for the `apr test llm bench --band` harness, and that is the point:
    /// it supplies no `TokenCounter` at all, so `--tokenization client_tokenizer`
    /// was as unfalsifiable as `server_usage` was. A counter, not a flag,
    /// because "some responses" is a mixture and a mixture is not a method.
    pub responses_counted_by_client_tokenizer: usize,
    /// Completed sampled responses considered. Failed and timed-out requests
    /// are excluded: they carry no response to have counted anything from.
    pub responses_counted: usize,
}

impl TokenizationObservation {
    /// Build the observation for a harness with **no client tokenizer**, from
    /// one `carried a server usage object` flag per completed response.
    #[must_use]
    pub fn from_server_usage_flags(flags: impl IntoIterator<Item = bool>) -> Self {
        let mut counted = 0_usize;
        let mut with_usage = 0_usize;
        for flag in flags {
            counted += 1;
            with_usage += usize::from(flag);
        }
        Self {
            responses_with_server_usage: with_usage,
            responses_counted_by_client_tokenizer: 0,
            responses_counted: counted,
        }
    }

    /// Sum two observations, so one receipt can span several bands.
    #[must_use]
    pub fn merged(self, other: Self) -> Self {
        Self {
            responses_with_server_usage: self.responses_with_server_usage
                + other.responses_with_server_usage,
            responses_counted_by_client_tokenizer: self.responses_counted_by_client_tokenizer
                + other.responses_counted_by_client_tokenizer,
            responses_counted: self.responses_counted + other.responses_counted,
        }
    }

    /// True only when **every** counted response carried a server `usage`
    /// object, and at least one response was counted.
    ///
    /// Not "at least one": a run in which some responses reported usage and
    /// some did not produced a mixture of two counting methods, and the mixture
    /// is the fallback class, not the server class.
    #[must_use]
    pub fn every_response_carried_server_usage(self) -> bool {
        self.responses_counted > 0 && self.responses_with_server_usage == self.responses_counted
    }

    /// True only when a client tokenizer counted **every** response.
    #[must_use]
    pub fn every_response_counted_by_client_tokenizer(self) -> bool {
        self.responses_counted > 0
            && self.responses_counted_by_client_tokenizer == self.responses_counted
    }
}

/// §4.4.6 — the declaration, the observation, and the method actually **used**.
///
/// # Why the downgrade is recorded rather than refused
///
/// No streaming receipt against `apr serve` can satisfy `server_usage` today:
/// the SSE stream carries no `usage` object at all. Refusing the run would not
/// remove the defect, it would remove the measurement — and against the
/// 2026-09-25 matrix expiry it would remove the campaign. A refusal is the
/// right answer to an *ambiguous* provenance, not to a *knowable* one.
///
/// # Why the downgrade is loud
///
/// A silent fallback is how #2754 happened, so a second silent fallback one
/// level up would repeat the pattern. [`Self::used`] is derived from the
/// observation and never from the declaration, both are written into the
/// receipt as distinct fields, and `scripts/perf_gate.sh`'s Arm C re-derives
/// the same predicate from the counts that travel beside them. Editing the
/// producer so the two agree does not make them agree at the gate: the counts
/// still say what happened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedTokenization {
    declared: TokenizationBlock,
    observed: TokenizationObservation,
    used: TokenCountingMethod,
}

impl ResolvedTokenization {
    /// Resolve the declaration against the observation.
    ///
    /// `used` is a **pure function of `observed`**. That is the whole design:
    /// there is no argument to this function that can make the receipt name a
    /// counting method the run did not use.
    #[must_use]
    pub fn resolve(declared: TokenizationBlock, observed: TokenizationObservation) -> Self {
        // Order matters and is the transport's, not a preference: `usage_counts`
        // takes the server's figures whenever a `usage` object is present, so a
        // response that had both was counted by the server.
        let used = if observed.every_response_carried_server_usage() {
            TokenCountingMethod::ServerUsage
        } else if observed.every_response_counted_by_client_tokenizer() {
            TokenCountingMethod::ClientTokenizer
        } else {
            TokenCountingMethod::ClientChunkCount
        };
        Self {
            declared,
            observed,
            used,
        }
    }

    /// The method the operator declared.
    #[must_use]
    pub fn requested(&self) -> TokenCountingMethod {
        self.declared.method()
    }

    /// The method the run actually used.
    #[must_use]
    pub fn used(&self) -> TokenCountingMethod {
        self.used
    }

    /// True when the run did not count the way the operator declared.
    #[must_use]
    pub fn downgraded(&self) -> bool {
        self.used != self.requested()
    }

    /// The observation the resolution was computed from.
    #[must_use]
    pub fn observed(&self) -> TokenizationObservation {
        self.observed
    }

    /// Prose naming what was declared, what happened, and the numbers, for the
    /// receipt and for the operator reading it a month later.
    #[must_use]
    pub fn downgrade_reason(&self) -> Option<String> {
        if !self.downgraded() {
            return None;
        }
        Some(format!(
            "§4.4.6 tokenization DOWNGRADED: `{}` was declared, `{}` was used. \
             {} of {} completed responses carried a server `usage` object and {} were \
             counted by a client tokenizer. The receipt records the method that was USED; \
             the declaration is kept under `method_requested` so the two cannot be \
             confused (#2754).",
            self.requested().wire_token(),
            self.used.wire_token(),
            self.observed.responses_with_server_usage,
            self.observed.responses_counted,
            self.observed.responses_counted_by_client_tokenizer,
        ))
    }

    /// # Errors
    /// When the declaration is malformed, when nothing was observed (a
    /// resolution over zero responses asserts a counting method for a run that
    /// counted nothing), or when the used method is `client_tokenizer` without
    /// a 64-hex digest.
    pub fn validate(&self) -> Result<(), String> {
        self.declared.validate()?;
        if self.observed.responses_counted == 0 {
            return Err(
                "tokenization: zero completed responses were observed, so no counting method \
                 was used — a receipt that names one is asserting something the run did not do"
                    .to_string(),
            );
        }
        if self.observed.responses_with_server_usage > self.observed.responses_counted
            || self.observed.responses_counted_by_client_tokenizer > self.observed.responses_counted
        {
            return Err(format!(
                "tokenization: {} server-counted and {} client-tokenizer-counted responses \
                 against {} counted -- a counter cannot exceed its own denominator",
                self.observed.responses_with_server_usage,
                self.observed.responses_counted_by_client_tokenizer,
                self.observed.responses_counted
            ));
        }
        // §4.4.6 requires the digest whenever the counts came from a client
        // tokenizer. The enum makes that unrepresentable for the DECLARATION;
        // this is the same rule applied to the method actually USED, which is
        // the one the receipt reports and the gate reads.
        if self.used == TokenCountingMethod::ClientTokenizer
            && !self.declared.digest().is_some_and(is_sha256)
        {
            return Err(
                "tokenization.method = client_tokenizer was USED with no 64-hex \
                 tokenizer_sha256 — §4.4.6 requires the digest of the tokenizer that counted"
                    .to_string(),
            );
        }
        Ok(())
    }

    fn to_json(&self) -> Value {
        let mut map = Map::new();
        // `method` is the USED method, because that is the one every reader —
        // perf_gate.sh included — treats as "how these numbers were counted".
        map.insert("method".into(), json!(self.used.wire_token()));
        map.insert(
            "method_requested".into(),
            json!(self.requested().wire_token()),
        );
        map.insert("downgraded".into(), json!(self.downgraded()));
        if let Some(reason) = self.downgrade_reason() {
            map.insert("downgrade_reason".into(), json!(reason));
        }
        if let Some(sha) = self.declared.digest() {
            map.insert("tokenizer_sha256".into(), json!(sha));
        }
        let (special, echo) = self.declared.counts();
        map.insert("counts_special_tokens".into(), json!(special));
        map.insert("counts_prompt_echo".into(), json!(echo));
        map.insert(
            "responses_with_server_usage".into(),
            json!(self.observed.responses_with_server_usage),
        );
        map.insert(
            "responses_counted_by_client_tokenizer".into(),
            json!(self.observed.responses_counted_by_client_tokenizer),
        );
        map.insert(
            "responses_counted".into(),
            json!(self.observed.responses_counted),
        );
        Value::Object(map)
    }
}

/// Arm D's memory block. **Server-reported**: the only constructor says so in
/// its name, so a client-side guess has nowhere to enter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvBlock {
    bytes_used: u64,
    bytes_reserved: u64,
    admission_rejected: u64,
    preempted_swap: u64,
}

impl KvBlock {
    /// Build the block from figures the **server** reported.
    #[must_use]
    pub fn from_server_report(
        bytes_used: u64,
        bytes_reserved: u64,
        admission_rejected: u64,
        preempted_swap: u64,
    ) -> Self {
        Self {
            bytes_used,
            bytes_reserved,
            admission_rejected,
            preempted_swap,
        }
    }

    fn to_json(self) -> Value {
        json!({
            "bytes_used": self.bytes_used,
            "bytes_reserved": self.bytes_reserved,
            "admission_rejected": self.admission_rejected,
            "preempted_swap": self.preempted_swap,
        })
    }
}

/// §4.3 — which workload the band set ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Workload {
    /// Homogeneous, `prompt_tokens = 512 ± 8`, `max_tokens = 128`, ignore-EOS.
    W1,
    /// Ragged prompt and generation mixture, with an injector at `window/2`.
    W2,
}

impl Workload {
    /// The wire token.
    #[must_use]
    pub fn wire_token(self) -> &'static str {
        match self {
            Self::W1 => "W1",
            Self::W2 => "W2",
        }
    }
}

/// Parse the wire token back. As [`ComputeClass`]'s, derived from
/// `wire_token` so the two cannot drift.
impl FromStr for Workload {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        [Self::W1, Self::W2]
            .into_iter()
            .find(|w| w.wire_token() == s)
            .ok_or_else(|| format!("workload {s:?}: expected W1 or W2 (§4.3)"))
    }
}

/// §4.3 — the prompt set that was actually sent, bound to the workload label.
///
/// # The defect this closes
///
/// `--workload` was free text (#2756). PERF-045 passed
/// `--workload W1 --profile short`; `short` is **one prompt sent 30 times**,
/// and the receipt recorded `"workload": "W1"`. `Workload::from_str` only
/// checked the string was `W1` or `W2`; the line `profile short (1 prompt(s))`
/// existed on stdout, which is not retained.
///
/// The real W1 corpus is in-tree at
/// `crates/aprender-serve/benchmarks/qwen-coder/prompts-w1.jsonl` — 256 distinct
/// prompts — and its own `_meta.distinctness_rationale` says why that matters:
///
/// > Identical prompts would let prefix caching, not the scheduler, drive Arm
/// > A's `scaling_efficiency`.
///
/// So the harness labelled as W1 exactly the degenerate single-prompt run the
/// corpus author documented as invalidating the measurement it feeds. This type
/// records what was sent — count, distinct count and a digest over the prompt
/// texts in issue order — so a reader can tell the two apart, and
/// [`Self::label_refusal`] refuses the label outright when the sent set cannot
/// bear it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkloadCorpus {
    /// Prompts in the set the harness cycled through.
    pub prompts: usize,
    /// Distinct prompt texts among them.
    pub distinct_prompts: usize,
    /// SHA-256 over the prompt texts in issue order, newline-separated.
    pub sha256: String,
    /// Where the set came from, verbatim — the profile name or the file path.
    pub source: String,
}

impl WorkloadCorpus {
    /// Digest and count the prompt texts in the order the harness will cycle
    /// them.
    #[must_use]
    pub fn from_prompt_texts(texts: &[String], source: impl Into<String>) -> Self {
        let mut hasher = Sha256::new();
        for t in texts {
            hasher.update(t.as_bytes());
            hasher.update(b"\n");
        }
        let distinct: std::collections::BTreeSet<&String> = texts.iter().collect();
        Self {
            prompts: texts.len(),
            distinct_prompts: distinct.len(),
            sha256: format!("{:x}", hasher.finalize()),
            source: source.into(),
        }
    }

    /// The sent set is too degenerate to carry `workload`'s label at all.
    ///
    /// `sample_floor` is §4.4.2's `max(30, 8c)` for the **narrowest** band that
    /// will run — the smallest number of sampled requests any legal band
    /// issues. A corpus with fewer distinct prompts than that guarantees the
    /// same prompt is issued twice inside even the smallest legal band, which
    /// is the prefix-caching condition the W1 corpus was built to avoid. The
    /// floor is an argument rather than a constant so the rule is visible at
    /// the call site instead of buried here.
    ///
    /// This is a refusal rather than a stated violation because, unlike the
    /// §4.4.6 downgrade, refusing does **not** remove the measurement: a
    /// conformant corpus is committed in-tree and the message names it.
    #[must_use]
    pub fn label_refusal(&self, workload: Workload, sample_floor: usize) -> Option<String> {
        if self.distinct_prompts >= sample_floor {
            return None;
        }
        Some(format!(
            "--workload {label} names §4.3's {label} corpus, but the prompt set actually sent \
             ({source}) holds {distinct} distinct prompt(s) — fewer than the {sample_floor} \
             sampled requests the narrowest band must issue, so the same prompt is served \
             repeatedly inside a single band. The {label} corpus's own \
             `_meta.distinctness_rationale` says why that invalidates the measurement it feeds: \
             \"Identical prompts would let prefix caching, not the scheduler, drive Arm A's \
             scaling_efficiency.\" Pass the committed corpus instead: --prompts \
             crates/aprender-serve/benchmarks/qwen-coder/prompts-{lower}.jsonl",
            label = workload.wire_token(),
            lower = workload.wire_token().to_lowercase(),
            source = self.source,
            distinct = self.distinct_prompts,
        ))
    }

    /// The set is legal but still repeats inside the **widest** band that ran.
    ///
    /// Stated rather than refused: the committed W2 corpus holds 99 prompts and
    /// the widest declared band issues `max(30, 8x16) = 128` sampled requests,
    /// so this condition is true of a corpus the project ships. Refusing it
    /// would make W2 unmeasurable; saying nothing would let prefix caching
    /// drive the number in silence.
    #[must_use]
    pub fn repetition_violation(
        &self,
        workload: Workload,
        widest_band_samples: usize,
    ) -> Option<String> {
        if self.distinct_prompts >= widest_band_samples {
            return None;
        }
        Some(format!(
            "§4.3 {label}: {distinct} distinct prompt(s) in the sent set ({source}) against \
             {widest_band_samples} sampled requests in the widest band — prompts repeat within \
             a band, so prefix caching contributes to Arm A's scaling_efficiency on this cell",
            label = workload.wire_token(),
            distinct = self.distinct_prompts,
            source = self.source,
        ))
    }

    fn to_json(&self) -> Value {
        json!({
            "prompts": self.prompts,
            "distinct_prompts": self.distinct_prompts,
            "sha256": self.sha256,
            "source": self.source,
        })
    }
}

/// §4.4.2 — which replicate this receipt is, and how many the cell ran.
///
/// # The defect this closes
///
/// `--replicates 1` produced a receipt **byte-indistinguishable from one
/// replicate of a spec `N = 3` cell**, and `is_conformant()` returned true
/// (#2755). `grep -ic replicate receipt.r1.json` was `0`. The warning went to
/// stdout, which is not retained, and two doc comments disagreed with each
/// other and with the code about where it went.
///
/// §4.4.2 fixes `N = 3` because the confidence interval is computed across
/// replicates. A receipt that does not record its own `N` cannot be checked
/// against the protocol it claims to follow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Replicates {
    /// 1-based index of this replicate.
    pub index: usize,
    /// Replicates the cell actually ran.
    pub effective: usize,
    /// §4.4.2's `N`. Carried so the receipt states the bar it is measured
    /// against rather than requiring the reader to know it.
    pub required: usize,
}

impl Replicates {
    /// True when the cell ran fewer replicates than §4.4.2 requires.
    #[must_use]
    pub fn below_spec(self) -> bool {
        self.effective < self.required
    }

    /// The stated violation a below-spec `N` must carry into the receipt.
    #[must_use]
    pub fn violation(self) -> Option<String> {
        if !self.below_spec() {
            return None;
        }
        Some(format!(
            "§4.4.2 replicates={} < N={}: the cell is under-replicated and its bootstrap \
             confidence interval is correspondingly weak. Stated here rather than on stdout, \
             which is not retained (#2755).",
            self.effective, self.required
        ))
    }

    fn to_json(self) -> Value {
        json!({
            "index": self.index,
            "effective": self.effective,
            "required": self.required,
            "below_spec": self.below_spec(),
        })
    }
}

/// SHA-256 of a file, as the 64 lowercase hex characters
/// [`Provenance::validate`] and `bench_receipt.py` both demand.
///
/// Lives beside the validator on purpose. The digest is the one field of §4.2.2
/// that a producer cannot type by hand, and a producer that formatted it
/// differently from the way the validator matches it would fail only after the
/// measurement had already been paid for.
///
/// # Errors
/// When `path` cannot be opened or read.
pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

/// Everything needed to render one host × workload receipt.
#[derive(Debug, Clone, PartialEq)]
pub struct ReceiptInput {
    /// §4.2.2 identity and §4.2.3 join key.
    pub provenance: Provenance,
    /// §4.4.6 counting **declaration** — what the operator asked for.
    ///
    /// This is not what the receipt's `tokenization.method` reports. That is
    /// derived from [`Self::tokenization_observed`], because a declaration no
    /// observation can contradict is decoration (#2754).
    pub tokenization: TokenizationBlock,
    /// §4.4.6 counting **observation** — what the run actually did.
    ///
    /// Required rather than optional: a receipt that could omit it would be
    /// free to keep asserting the declaration, which is the defect.
    pub tokenization_observed: TokenizationObservation,
    /// §4.3 workload.
    pub workload: Workload,
    /// §4.3 — the prompt set actually sent, which binds the label above to
    /// something falsifiable (#2756).
    pub workload_corpus: WorkloadCorpus,
    /// §4.4.2 — which replicate this is and how many the cell ran (#2755).
    pub replicates: Replicates,
    /// Every departure from §4.4 this run is stating rather than hiding.
    ///
    /// `scripts/perf_gate.sh` REPORTs these at merge phase and FAILs on them at
    /// release. The alternative — conformant, silent and indistinguishable —
    /// is the worst of the three.
    pub stated_violations: Vec<String>,
    /// The commit under test (I-10).
    pub commit: String,
    /// One entry per band, each carrying its own per-request records.
    pub bands: Vec<BandInput>,
    /// Arm D's server-reported memory block, when the server reported one.
    pub kv: Option<KvBlock>,
}

impl ReceiptInput {
    /// Derive and render the receipt.
    ///
    /// # Errors
    /// When provenance or the tokenization block is invalid, when there are no
    /// bands, when any band violates §4.4.3/§4.4.7 (see [`BandInput::derive`]),
    /// or when the retained samples are a constant — `bench_receipt.py` calls
    /// that the fabricated-measurement shape (F12) and so does this.
    pub fn render(&self) -> Result<Value, String> {
        self.provenance.validate()?;
        self.resolved_tokenization().validate()?;
        if self.bands.is_empty() {
            return Err(
                "receipt has no bands — a measurement over zero bands is a vacuous pass"
                    .to_string(),
            );
        }
        let bands = self
            .bands
            .iter()
            .map(BandInput::derive)
            .collect::<Result<Vec<_>, _>>()?;
        let samples = samples_ms(&bands);
        validate_samples(&samples)?;
        Ok(self.assemble(&bands, samples))
    }

    /// [`Self::render`], as pretty-printed JSON.
    ///
    /// # Errors
    /// As [`Self::render`]; serialisation itself cannot fail for this shape.
    pub fn render_string(&self) -> Result<String, String> {
        let value = self.render()?;
        serde_json::to_string_pretty(&value).map_err(|e| format!("serialising receipt: {e}"))
    }

    fn assemble(&self, bands: &[DerivedBand], samples: Vec<f64>) -> Value {
        let mut map = Map::new();
        map.insert("spec".into(), json!("APR-PERF-GATE-001 v2.2 §4.4"));
        map.insert("commit".into(), json!(self.commit));
        map.insert("workload".into(), json!(self.workload.wire_token()));
        map.insert("workload_corpus".into(), self.workload_corpus.to_json());
        map.insert("replicates".into(), self.replicates.to_json());
        map.insert("client_model".into(), json!("closed_loop"));
        map.insert("provenance".into(), self.provenance.to_json());
        map.insert(
            "tokenization".into(),
            self.resolved_tokenization().to_json(),
        );
        map.insert(
            "stated_violations".into(),
            json!(self.all_stated_violations()),
        );
        insert_counts(&mut map, bands);
        map.insert("drain_ms".into(), json!(receipt_drain_ms(bands)));
        map.insert("n".into(), json!(samples.len()));
        map.insert("samples_ms".into(), json!(samples));
        map.insert(
            "bands".into(),
            Value::Array(bands.iter().map(band_json).collect()),
        );
        if let Some(kv) = self.kv {
            map.insert("kv".into(), kv.to_json());
        }
        map.insert("unproduced_fields".into(), json!(self.unproduced(bands)));
        Value::Object(map)
    }

    /// §4.4.6 resolved: the declaration measured against the observation.
    #[must_use]
    pub fn resolved_tokenization(&self) -> ResolvedTokenization {
        ResolvedTokenization::resolve(self.tokenization.clone(), self.tokenization_observed)
    }

    /// Everything this receipt states about its own departures from §4.4.
    ///
    /// The caller's list plus the two the receipt can derive for itself. A
    /// below-spec `N` and a repeating prompt set are recorded here whether or
    /// not the caller remembered to say so: relying on the caller to state a
    /// violation is how the stdout warning in #2755 came to be the only record.
    fn all_stated_violations(&self) -> Vec<String> {
        let mut out = self.stated_violations.clone();
        if let Some(v) = self.replicates.violation() {
            if !out.contains(&v) {
                out.push(v);
            }
        }
        out
    }

    fn unproduced(&self, bands: &[DerivedBand]) -> Vec<String> {
        let mut out = vec![SERVER_ONLY_FIELDS.to_string()];
        if self.kv.is_none() {
            out.push(
                "Arm D `kv` block (bytes_used, bytes_reserved, admission_rejected, \
                 preempted_swap) — server-reported. Absent here, so this receipt is legal at \
                 merge phase and correctly FAILS at release phase rather than carrying invented \
                 memory figures."
                    .to_string(),
            );
        }
        out.extend(bands.iter().flat_map(|b| b.unproduced.clone()));
        out
    }
}

/// §4.4.7 at receipt level: the **maximum** band drain, not a mean or a sum.
///
/// The `SUSPECT` rule is per-band and asks whether *one request dominated a
/// window*. Averaging four bands hides the one that did; summing invents a
/// drain phase no band ran. The worst band is the one a reader must see.
fn receipt_drain_ms(bands: &[DerivedBand]) -> f64 {
    bands.iter().map(|b| b.drain_ms).fold(0.0_f64, f64::max)
}

fn insert_counts(map: &mut Map<String, Value>, bands: &[DerivedBand]) {
    map.insert("requested".into(), json!(sum(bands, |b| b.requested)));
    map.insert("completed".into(), json!(sum(bands, |b| b.completed)));
    map.insert("timeouts".into(), json!(sum(bands, |b| b.timeouts)));
    map.insert("truncated".into(), json!(sum(bands, |b| b.truncated)));
    map.insert("errors".into(), json!(sum(bands, |b| b.errors)));
}

fn sum(bands: &[DerivedBand], f: impl Fn(&DerivedBand) -> usize) -> usize {
    bands.iter().map(f).sum()
}

fn samples_ms(bands: &[DerivedBand]) -> Vec<f64> {
    bands.iter().flat_map(|b| b.latencies_ms.clone()).collect()
}

/// I-4 and F12, applied by the producer rather than discovered by the validator.
fn validate_samples(samples: &[f64]) -> Result<(), String> {
    if samples.is_empty() {
        return Err(
            "samples_ms would be empty — no band completed a single request, and a \
                    receipt with no retained samples permanently forecloses the bootstrap (I-4)"
                .to_string(),
        );
    }
    let first = samples[0];
    if samples.len() > 1 && samples.iter().all(|s| (s - first).abs() < f64::EPSILON) {
        return Err(format!(
            "samples_ms: all {} samples identical ({first}) — a real timing distribution is not \
             constant; this is the fabricated-measurement shape (F12)",
            samples.len()
        ));
    }
    Ok(())
}

fn band_json(b: &DerivedBand) -> Value {
    let mut map = Map::new();
    map.insert("concurrency".into(), json!(b.concurrency));
    map.insert(
        "aggregate_tok_per_sec".into(),
        json!(b.aggregate_tok_per_sec),
    );
    map.insert("tokens_total".into(), json!(b.tokens_total));
    map.insert("span_ms".into(), json!(b.span_ms));
    map.insert("window_ms".into(), json!(b.window_ms));
    map.insert("drain_ms".into(), json!(b.drain_ms));
    map.insert("requested".into(), json!(b.requested));
    map.insert("completed".into(), json!(b.completed));
    map.insert("timeouts".into(), json!(b.timeouts));
    map.insert("truncated".into(), json!(b.truncated));
    map.insert("errors".into(), json!(b.errors));
    map.insert("suspect".into(), json!(b.suspect));
    insert_optional_latency(&mut map, b);
    insert_comparator(&mut map, &b.comparator);
    Value::Object(map)
}

/// Streaming-only metrics. Absent means absent: no zero stands in for one.
fn insert_optional_latency(map: &mut Map<String, Value>, b: &DerivedBand) {
    for (key, value) in [
        ("decode_tok_per_sec", b.decode_tok_per_sec),
        ("ttft_p50_ms", b.ttft_p50_ms),
        ("ttft_p95_ms", b.ttft_p95_ms),
        ("itl_p50_ms", b.itl_p50_ms),
        ("itl_p95_ms", b.itl_p95_ms),
    ] {
        if let Some(v) = value {
            map.insert(key.into(), json!(v));
        }
    }
}

fn insert_comparator(map: &mut Map<String, Value>, status: &ComparatorStatus) {
    map.insert("comparator_status".into(), json!(status.wire_token()));
    match status {
        ComparatorStatus::NotApplicable { decided_by, reason } => {
            map.insert("comparator_decided_by".into(), json!(decided_by));
            map.insert("comparator_reason".into(), json!(reason));
        }
        ComparatorStatus::Unmeasured { owner, reason } => {
            map.insert("comparator_owner".into(), json!(owner));
            map.insert("comparator_reason".into(), json!(reason));
        }
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

#[cfg(test)]
mod producer_tests {
    //! The two conversions PERF-025's CLI needs, and the digest it cannot type.

    use super::*;
    use std::io::Write;

    /// A parser with its own spelling would let a receipt be written with a
    /// class `bench_receipt.py` then rejects — after the band had already run.
    #[test]
    fn compute_class_roundtrip_is_the_only_spelling() {
        for c in [
            ComputeClass::Cpu,
            ComputeClass::Cuda,
            ComputeClass::Metal,
            ComputeClass::Wgpu,
            ComputeClass::Unknown,
        ] {
            assert_eq!(
                ComputeClass::from_str(c.wire_token()).expect("wire token must parse"),
                c
            );
        }
    }

    /// `bench_receipt.py`'s `COMPUTE_CLASSES` tuple, spelled out here so that
    /// adding a variant without teaching the validator goes red.
    #[test]
    fn the_wire_tokens_are_bench_receipt_pys_compute_classes() {
        let tokens: Vec<&str> = ["cpu", "cuda", "metal", "wgpu", "unknown"].into();
        for t in &tokens {
            assert!(ComputeClass::from_str(t).is_ok(), "{t} must parse");
        }
        assert!(ComputeClass::from_str("tpu").is_err());
        assert!(ComputeClass::from_str("gpu").is_err());
        assert!(
            ComputeClass::from_str("CUDA").is_err(),
            "case is load-bearing"
        );
    }

    #[test]
    fn workload_roundtrips_and_refuses_anything_else() {
        for w in [Workload::W1, Workload::W2] {
            assert_eq!(Workload::from_str(w.wire_token()).expect("parses"), w);
        }
        assert!(Workload::from_str("W3").is_err());
        assert!(Workload::from_str("w1").is_err());
    }

    /// The digest must be the shape `Provenance::validate` accepts, or the
    /// producer writes a receipt its own validator rejects.
    #[test]
    fn sha256_file_produces_a_digest_provenance_accepts() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("payload.bin");
        let mut f = std::fs::File::create(&path).expect("create");
        f.write_all(b"abc").expect("write");
        drop(f);

        let digest = sha256_file(&path).expect("hashes");
        // The published SHA-256 of "abc".
        assert_eq!(
            digest,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(digest.len(), 64);
        assert!(is_sha256(&digest), "must satisfy the receipt's own check");
    }

    #[test]
    fn sha256_file_reports_a_missing_file_rather_than_a_digest() {
        assert!(sha256_file(Path::new("/nonexistent/perf-025")).is_err());
    }
}

#[cfg(test)]
mod perf048_tests {
    //! PERF-048 — the three provenance fields PERF-045's receipt asserted while
    //! the run did something else (#2754, #2755, #2756).
    //!
    //! Every test here states the mutation it is the RED for. The pattern that
    //! matters is the first block: `used` is a pure function of the
    //! **observation**, so no edit to a declaration can make the receipt name a
    //! method the run did not use.

    use super::*;

    fn observed(with_usage: usize, by_tokenizer: usize, counted: usize) -> TokenizationObservation {
        TokenizationObservation {
            responses_with_server_usage: with_usage,
            responses_counted_by_client_tokenizer: by_tokenizer,
            responses_counted: counted,
        }
    }

    fn declared_server_usage() -> TokenizationBlock {
        TokenizationBlock::ServerUsage {
            counts_special_tokens: true,
            counts_prompt_echo: false,
        }
    }

    fn declared_client_tokenizer() -> TokenizationBlock {
        TokenizationBlock::ClientTokenizer {
            tokenizer_sha256: "c".repeat(64),
            counts_special_tokens: true,
            counts_prompt_echo: false,
        }
    }

    /// **THE PERF-045 RECEIPT.** `server_usage` declared, 0 of 30 responses
    /// carried a `usage` object. The receipt must name what happened.
    ///
    /// RED for: making `used()` return `self.declared.method()`.
    #[test]
    fn the_receipt_records_the_method_that_was_used_not_the_one_requested() {
        let r = ResolvedTokenization::resolve(declared_server_usage(), observed(0, 0, 30));
        assert_eq!(r.requested(), TokenCountingMethod::ServerUsage);
        assert_eq!(r.used(), TokenCountingMethod::ClientChunkCount);
        assert!(r.downgraded());
        let json = r.to_json();
        assert_eq!(json["method"], "client_chunk_count");
        assert_eq!(json["method_requested"], "server_usage");
        assert_eq!(json["downgraded"], true);
        let reason = json["downgrade_reason"].as_str().expect("reason present");
        assert!(reason.contains("server_usage"), "{reason}");
        assert!(reason.contains("client_chunk_count"), "{reason}");
        assert!(reason.contains("0 of 30"), "{reason}");
    }

    /// THE DISCRIMINATION CASE. A server that does report usage is not
    /// downgraded, and carries no `downgrade_reason` at all.
    ///
    /// Without this, "always downgrade" would pass the test above.
    #[test]
    fn a_server_that_reports_usage_is_not_downgraded() {
        let r = ResolvedTokenization::resolve(declared_server_usage(), observed(30, 0, 30));
        assert_eq!(r.used(), TokenCountingMethod::ServerUsage);
        assert!(!r.downgraded());
        assert!(r.downgrade_reason().is_none());
        let json = r.to_json();
        assert_eq!(json["method"], "server_usage");
        assert_eq!(json["downgraded"], false);
        assert!(json.get("downgrade_reason").is_none());
    }

    /// The property that makes the pair impossible to silently equalise: for a
    /// fixed observation, **every** declaration yields the same used method.
    ///
    /// RED for: any edit that lets the declaration influence `used`.
    #[test]
    fn the_used_method_is_a_function_of_the_observation_alone() {
        for obs in [observed(0, 0, 30), observed(30, 0, 30), observed(0, 30, 30)] {
            let a = ResolvedTokenization::resolve(declared_server_usage(), obs).used();
            let b = ResolvedTokenization::resolve(declared_client_tokenizer(), obs).used();
            assert_eq!(
                a, b,
                "two declarations over the same observation disagreed: {obs:?}"
            );
        }
    }

    /// A mixture of two counting methods is neither of them.
    #[test]
    fn a_partial_server_usage_run_is_the_fallback_class() {
        let r = ResolvedTokenization::resolve(declared_server_usage(), observed(29, 0, 30));
        assert_eq!(r.used(), TokenCountingMethod::ClientChunkCount);
        assert!(r.downgraded());

        let r = ResolvedTokenization::resolve(declared_client_tokenizer(), observed(0, 29, 30));
        assert_eq!(r.used(), TokenCountingMethod::ClientChunkCount);
    }

    /// §4.4.6's digest rule, applied to the method that was USED.
    ///
    /// RED for: dropping the digest branch from `validate`.
    #[test]
    fn a_client_tokenizer_that_counted_must_name_itself() {
        let ok = ResolvedTokenization::resolve(declared_client_tokenizer(), observed(0, 30, 30));
        assert_eq!(ok.used(), TokenCountingMethod::ClientTokenizer);
        assert!(ok.validate().is_ok());
        assert_eq!(ok.to_json()["tokenizer_sha256"], "c".repeat(64));

        // A tokenizer counted every response, but the declaration beside it
        // names no tokenizer. The enum cannot express that, so the observation
        // is what has to carry the contradiction.
        let bad = ResolvedTokenization::resolve(declared_server_usage(), observed(0, 30, 30));
        assert_eq!(bad.used(), TokenCountingMethod::ClientTokenizer);
        let err = bad
            .validate()
            .expect_err("no digest for a tokenizer that counted");
        assert!(err.contains("tokenizer_sha256"), "{err}");
    }

    /// A resolution over zero responses names a method for a run that counted
    /// nothing.
    #[test]
    fn a_resolution_over_zero_responses_is_refused() {
        let r = ResolvedTokenization::resolve(declared_server_usage(), observed(0, 0, 0));
        let err = r.validate().expect_err("nothing was counted");
        assert!(err.contains("zero completed responses"), "{err}");
    }

    /// A counter above its own denominator is a fabricated observation.
    #[test]
    fn a_counter_cannot_exceed_its_own_denominator() {
        let r = ResolvedTokenization::resolve(declared_server_usage(), observed(31, 0, 30));
        assert!(r.validate().is_err());
        let r = ResolvedTokenization::resolve(declared_server_usage(), observed(0, 31, 30));
        assert!(r.validate().is_err());
    }

    /// Bands merge into one receipt-level observation, and the merge is a sum
    /// rather than a vote: one band whose server went quiet downgrades the
    /// receipt.
    #[test]
    fn merging_bands_keeps_a_mixture_a_mixture() {
        let merged = observed(30, 0, 30).merged(observed(0, 0, 30));
        assert_eq!(merged.responses_counted, 60);
        assert_eq!(merged.responses_with_server_usage, 30);
        assert!(!merged.every_response_carried_server_usage());
        assert_eq!(
            ResolvedTokenization::resolve(declared_server_usage(), merged).used(),
            TokenCountingMethod::ClientChunkCount
        );
    }

    /// Chunk counts are not token counts, and §4.4.6 exists so a ratio is only
    /// taken across two sides that counted the same way.
    #[test]
    fn a_chunk_count_is_not_a_token_count() {
        assert!(TokenCountingMethod::ServerUsage.is_a_token_count());
        assert!(TokenCountingMethod::ClientTokenizer.is_a_token_count());
        assert!(!TokenCountingMethod::ClientChunkCount.is_a_token_count());
        assert_eq!(
            TokenCountingMethod::ClientChunkCount.wire_token(),
            "client_chunk_count"
        );
    }

    // ------------------------------------------------------------ #2755 -----

    /// A below-spec `N` must be visible in the receipt, not on stdout.
    ///
    /// RED for: dropping `Replicates::violation`, or defaulting `required`.
    #[test]
    fn a_below_spec_replicate_count_states_itself() {
        let r = Replicates {
            index: 1,
            effective: 1,
            required: 3,
        };
        assert!(r.below_spec());
        let v = r.violation().expect("a stated violation");
        assert!(v.contains("replicates=1"), "{v}");
        assert!(v.contains("N=3"), "{v}");
        let json = r.to_json();
        assert_eq!(json["effective"], 1);
        assert_eq!(json["required"], 3);
        assert_eq!(json["below_spec"], true);
    }

    /// THE DISCRIMINATION CASE: a spec-N cell states nothing.
    #[test]
    fn a_spec_replicate_count_states_nothing() {
        let r = Replicates {
            index: 2,
            effective: 3,
            required: 3,
        };
        assert!(!r.below_spec());
        assert!(r.violation().is_none());
        assert_eq!(r.to_json()["below_spec"], false);
    }

    // ------------------------------------------------------------ #2756 -----

    fn texts(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("// w1-{i:04} body")).collect()
    }

    /// **THE PERF-045 WORKLOAD.** One prompt sent 30 times, labelled W1.
    ///
    /// RED for: relaxing `label_refusal` to compare against 1 rather than the
    /// band's sample floor.
    #[test]
    fn a_one_prompt_corpus_cannot_be_called_w1() {
        let c = WorkloadCorpus::from_prompt_texts(&texts(1), "profile short (1 prompt(s))");
        assert_eq!(c.prompts, 1);
        assert_eq!(c.distinct_prompts, 1);
        let refusal = c.label_refusal(Workload::W1, 30).expect("must refuse");
        assert!(refusal.contains("prefix caching"), "{refusal}");
        assert!(refusal.contains("profile short"), "{refusal}");
        assert!(refusal.contains("prompts-w1.jsonl"), "{refusal}");
    }

    /// THE DISCRIMINATION CASE: the committed corpus carries its own label.
    #[test]
    fn the_committed_corpus_size_carries_the_label() {
        let c = WorkloadCorpus::from_prompt_texts(&texts(256), "file prompts-w1.jsonl");
        assert!(c.label_refusal(Workload::W1, 30).is_none());
        assert!(c.repetition_violation(Workload::W1, 128).is_none());
    }

    /// The boundary, both sides. A floor test that only checks the far side of
    /// the boundary passes for an off-by-one comparison.
    #[test]
    fn the_label_floor_is_the_narrowest_bands_sample_count() {
        assert!(WorkloadCorpus::from_prompt_texts(&texts(30), "p")
            .label_refusal(Workload::W1, 30)
            .is_none());
        assert!(WorkloadCorpus::from_prompt_texts(&texts(29), "p")
            .label_refusal(Workload::W1, 30)
            .is_some());
    }

    /// A legal-but-repeating corpus is stated rather than refused: the shipped
    /// W2 corpus holds 99 prompts against a 128-sample widest band.
    #[test]
    fn a_corpus_that_repeats_inside_the_widest_band_states_it() {
        let c = WorkloadCorpus::from_prompt_texts(&texts(99), "file prompts-w2.jsonl");
        assert!(c.label_refusal(Workload::W2, 30).is_none());
        let v = c
            .repetition_violation(Workload::W2, 128)
            .expect("99 < 128 repeats");
        assert!(v.contains("99 distinct"), "{v}");
        assert!(v.contains("prefix caching"), "{v}");
    }

    /// The digest is over the prompts, so a different set is a different
    /// corpus even at the same count — which is the whole of "bind the label
    /// to something falsifiable".
    #[test]
    fn the_digest_distinguishes_two_corpora_of_the_same_size() {
        let a = WorkloadCorpus::from_prompt_texts(&texts(30), "a");
        let b = WorkloadCorpus::from_prompt_texts(&texts(30), "b");
        assert_eq!(a.sha256, b.sha256, "same prompts, same digest");
        let mut other = texts(30);
        other[7] = "// something else".to_string();
        let c = WorkloadCorpus::from_prompt_texts(&other, "c");
        assert_ne!(a.sha256, c.sha256);
        assert_eq!(a.sha256.len(), 64);
    }

    /// A repeated prompt is counted once as distinct and many times as sent —
    /// the two numbers together are what make a repeated corpus visible.
    #[test]
    fn distinct_and_sent_counts_are_recorded_separately() {
        let repeated = vec!["one".to_string(); 30];
        let c = WorkloadCorpus::from_prompt_texts(&repeated, "profile short");
        assert_eq!(c.prompts, 30);
        assert_eq!(c.distinct_prompts, 1);
        assert!(c.label_refusal(Workload::W1, 30).is_some());
    }
}
