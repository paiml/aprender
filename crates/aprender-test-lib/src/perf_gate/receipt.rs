//! PP-LLAMA-001 v3.0 Appendix B — the receipt emitter, and the typed reader
//! that can refuse one.
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
//! hand it a `drain_ms`, a `timeouts` count, or a ratio — every number it emits
//! is computed from samples that travel in the same receipt, which is the rule
//! `scripts/lib/bench_receipt.py` already applies to ratios.
//!
//! # What v3 added, and why each one is a field rather than a convention
//!
//! - **`run_id`, `started_utc`, `clock_source` (PP-30).** Two lanes of one
//!   invocation share a `run_id`, which is what makes PP-3's "the baseline is
//!   from the same run" checkable rather than asserted. The id is derived —
//!   `sha256(started_utc ‖ host ‖ client sha256 ‖ pid)[..32]` — so it is
//!   reproducible from the receipt's own contents, which a UUID is not.
//! - **A split `provenance` (PP-2, 18, 20, 25).** The v2.2 shape had ONE
//!   `binary_sha256`, and the producer filled it from `std::env::current_exe()`
//!   — the *client*, not the `apr serve` under test. PP-18 ("the subject was
//!   built from an ancestor commit") and PP-25 ("one client binary drove both
//!   lanes") are different claims about different binaries, and one field could
//!   not carry both. [`Provenance::subject`] and [`Provenance::client`] are now
//!   separate identities and the comparator has its own with a pin expiry.
//! - **`protocol` (§5.1).** Window, warmup, quiesce, cooldown, `n_predict`,
//!   replicate count, interleaving and the sampler pin. Two receipts written
//!   under different protocols are not comparable, so all of it is also in the
//!   PP-22 join key.
//! - **`ladder` (PP-24).** Bands are derived from what both servers *admitted*,
//!   not declared by the harness. A `c = 16` band against a subject that
//!   admitted 11 slots measured a queue, not a server.
//! - **A typed [`Receipt`] with `deny_unknown_fields`.** Until v3 the receipt
//!   had a serialiser and no deserialiser, so every "strip field X" must-fire
//!   was testable only in python inside `perf_gate.sh --selftest` — which does
//!   not run in `workspace-test` at all (that image has no python3).
//!
//! # What this still refuses to emit
//!
//! - **§4.4.9's scheduler block.** `max_in_flight`, `admission_rejected`,
//!   `preempted_recompute`, `preempted_swap`, `kv_blocks_*`, `gpu_layers_*`,
//!   `backend_loaded[]`, `autofit_applied[]` are **server**-reported by
//!   construction — PP-13 says `max_in_flight` "is reported by the **server**,
//!   not inferred by the harness". A client-side estimate would be
//!   indistinguishable from a real answer, which is worse than a missing field.
//!   The block is omitted and named in `unproduced_fields`, with the reason.
//! - **Arm D's `kv` block**, for the same reason, unless a caller supplies one
//!   via [`KvBlock::from_server_report`].
//! - **A ratio without a baseline.** [`super::drain::ComparatorStatus::Measured`]
//!   is constructible only through `BandInput::join_status`, which refuses a
//!   cross-run baseline, a join-key mismatch and a timed-out lane.
//! - **A default `resolution`.** Every [`Provenance`] string is required and an
//!   empty one is refused. A `--resolution` that defaults to `scripts/apr_bin.sh`
//!   invents provenance, and invented provenance is indistinguishable from
//!   measured provenance.

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::str::FromStr;

use super::drain::{
    AdmissionCap, BandContext, BandInput, BandStatus, ComparatorStatus, DerivedBand, SampleRow,
    StreamMode, StreamWitness, SCHEMA_VERSION,
};
use super::join::{BandRatios, JoinKey};
use super::protocol::ProtocolParams;
use super::samples::SamplesFile;
use super::witness::BatchInvarianceWitness;

/// The spec string every v3 receipt carries.
pub const SPEC_ID: &str = "PP-LLAMA-001 v3.0";

/// PP-30 — the clock a plain `std` producer reads.
pub const CLOCK_SOURCE_SYSTEM_REALTIME: &str = "std::time::SystemTime (CLOCK_REALTIME)";

/// §4.4.9 fields a client cannot observe, and why. Emitted verbatim into the
/// receipt's `unproduced_fields` rather than guessed at.
pub const SERVER_ONLY_FIELDS: &str = "§4.4.9 scheduler block (max_in_flight, admission_rejected, \
     preempted_recompute, preempted_swap, kv_blocks_total, kv_blocks_peak_used, \
     kv_bytes_reserved, kv_bytes_used, gpu_layers_requested, gpu_layers_resolved, \
     gpu_layers_total, backend_loaded[], autofit_applied[]) — every one is reported by the \
     SERVER. PP-13: max_in_flight is reported by the server, not inferred by the harness; PP-2: \
     gpu_layers_resolved is read from the loader and never inferred. A client-side estimate \
     would read exactly like a measurement, so none is emitted.";

/// PP-3 / PP-30 — the identifier both lanes of one harness invocation share.
///
/// 32 lowercase hex characters, **derived** rather than random:
/// `sha256(started_utc ‖ host ‖ client_sha256 ‖ pid)[..32]`. §1(d) requires a
/// receipt to be decidable from its own contents, and a UUID cannot be
/// recomputed from the receipt it sits in — so a receipt could claim any
/// `run_id` and nothing could check it. Every input here is already on the
/// receipt.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct RunId(String);

impl RunId {
    /// Derive the id from the four facts that identify an invocation.
    #[must_use]
    pub fn derive(started_utc: &str, host: &str, client_sha256: &str, pid: u32) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(started_utc.as_bytes());
        hasher.update(host.as_bytes());
        hasher.update(client_sha256.as_bytes());
        hasher.update(pid.to_string().as_bytes());
        let digest = format!("{:x}", hasher.finalize());
        Self(digest[..32].to_string())
    }

    /// The 32 hex characters.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for RunId {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        if value.len() == 32
            && value
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        {
            Ok(Self(value))
        } else {
            Err(format!(
                "run_id {value:?} is not 32 lowercase hex characters — PP-3 keys the baseline on \
                 it, so a malformed one would make every ratio unjoinable"
            ))
        }
    }
}

impl From<RunId> for String {
    fn from(id: RunId) -> Self {
        id.0
    }
}

/// PP-30 — the current instant as RFC3339 UTC with milliseconds and a literal
/// `Z`, the exact shape [`Provenance::validate`] accepts.
///
/// Not available on `wasm32`, where the crate has no clock dependency; a caller
/// there supplies the timestamp it observed.
#[cfg(not(target_arch = "wasm32"))]
#[must_use]
pub fn now_utc_millis() -> String {
    chrono::Utc::now()
        .format("%Y-%m-%dT%H:%M:%S%.3fZ")
        .to_string()
}

/// The dispatch path a run actually took. Mirrors `bench_receipt.py`'s
/// `COMPUTE_CLASSES`; PP-2 requires this be the path **taken**, read from the
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
                "compute_class {s:?}: expected one of cpu, cuda, metal, wgpu, unknown (PP-2 \
                 requires the path TAKEN, not the hardware present)"
            )
        })
    }
}

/// PP-18 — the subject binary (`apr serve`) that served the band.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SubjectIdentity {
    /// Absolute path to the binary that served.
    pub path: String,
    /// Its digest. 64 lowercase hex characters.
    pub sha256: String,
    /// The commit it was built from. PP-18 asserts this is an ancestor of the
    /// commit under test.
    pub commit: String,
    /// Cargo features read **from the built binary**, never from `Cargo.toml`.
    pub feature_set: Vec<String>,
}

/// PP-25 — the client binary that drove **both** lanes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClientIdentity {
    /// Absolute path to the client.
    pub path: String,
    /// Its digest. 64 lowercase hex characters.
    pub sha256: String,
    /// The commit it was built from.
    pub commit: String,
    /// PP-3 / PP-30 — the client process id, the fourth input to
    /// [`RunId::derive`].
    ///
    /// The id is documented as "reproducible from the receipt's own contents,
    /// which a UUID is not" — and the pid was **not on the receipt**, so the
    /// claim was false and no reader could check a `run_id` at all. Two
    /// invocations that read the same millisecond on the same host with the
    /// same client differ only here; without it they would share a `run_id` and
    /// PP-3's same-run rule would join two runs.
    pub pid: u32,
}

/// PP-20 — the pinned comparator build, and when the pin goes stale.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComparatorIdentity {
    /// The upstream commit the comparator was built from.
    pub commit: String,
    /// The `cmake` line it was configured with.
    pub cmake: String,
    /// The built binary's digest. 64 lowercase hex characters.
    pub sha256: String,
    /// RFC3339 UTC instant after which every ratio against this pin is
    /// `COMPARATOR_STALE`.
    pub pin_expiry: String,
    /// `GET /props` for this band, stored verbatim (§5.3).
    pub props: Value,
}

/// PP-23's input — the model file the band served.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelIdentity {
    /// Absolute path to the weights.
    pub path: String,
    /// Its digest. 64 lowercase hex characters.
    pub sha256: String,
    /// `stat -c %s` of the file — the roofline's numerator input.
    pub bytes: u64,
}

/// §4.2.2 identity plus the §4.2.3 join key. **No `Default` impl**: every field
/// is a fact about a specific run, and a blank one that serialises is how a
/// receipt acquires provenance it never had.
///
/// `Eq` is deliberately absent: `server_config` and `comparator.props` are
/// verbatim JSON, which contains floats.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    /// The binary that ran, as an absolute path. Kept from v2.2 so today's
    /// readers keep working; it is the **client** path, and
    /// [`Self::client`] says so in a field that cannot be misread.
    pub binary_path: String,
    /// Host-local anti-substitution fingerprint of that same binary.
    /// 64 lowercase hex characters.
    pub binary_sha256: String,
    /// How that path was resolved. **No default** — see the module docs.
    pub resolution: String,
    /// The dispatch path taken (PP-2).
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
    /// PP-30 — RFC3339 UTC with milliseconds and a literal `Z`.
    pub started_utc: String,
    /// PP-30 — which clock that instant came from.
    pub clock_source: String,
    /// PP-18 — the `apr serve` under test.
    pub subject: SubjectIdentity,
    /// PP-25 — the one client that drove both lanes.
    pub client: ClientIdentity,
    /// PP-20 — the comparator pin, when there was a comparator lane.
    pub comparator: Option<ComparatorIdentity>,
    /// PP-2 — `GET /v1/effective-config`, stored verbatim before the first
    /// request. Verbatim because a harness that re-shapes the server's answer
    /// is a harness that can lose the field the answer was needed for.
    pub server_config: Option<Value>,
    /// PP-23's input — the weights file.
    pub model_file: Option<ModelIdentity>,
}

impl Provenance {
    /// §4.2 checks that a receipt cannot be written without passing.
    ///
    /// # Errors
    /// When any required field is empty, when any digest is not 64 lowercase
    /// hex characters, when `started_utc` is not RFC3339 UTC (PP-30), or when
    /// the declared `compute_class` is a path the build cannot reach (PP-2).
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in self.required_strings() {
            if value.trim().is_empty() {
                return Err(format!(
                    "provenance.{name}: empty — this field has no default; a receipt that does \
                     not say {name} is an anonymous number, not evidence"
                ));
            }
        }
        for (name, digest) in self.digests() {
            if !is_sha256(digest) {
                return Err(format!(
                    "provenance.{name}: {digest:?} is not 64 lowercase hex characters"
                ));
            }
        }
        validate_rfc3339_utc_millis("provenance.started_utc", &self.started_utc)?;
        if let Some(c) = &self.comparator {
            validate_rfc3339_utc_millis("provenance.comparator.pin_expiry", &c.pin_expiry)?;
        }
        self.validate_feature_set()
    }

    /// PP-20 — did the comparator pin expire before this run started?
    ///
    /// Both instants are canonical RFC3339 UTC with the same field widths, so
    /// lexicographic order **is** chronological order; [`Self::validate`]
    /// refuses anything else, which is what makes the string comparison sound.
    #[must_use]
    pub fn comparator_is_stale(&self) -> bool {
        self.comparator
            .as_ref()
            .is_some_and(|c| c.pin_expiry < self.started_utc)
    }

    fn required_strings(&self) -> Vec<(&'static str, &str)> {
        let mut out = vec![
            ("binary_path", self.binary_path.as_str()),
            ("binary_sha256", self.binary_sha256.as_str()),
            ("resolution", self.resolution.as_str()),
            ("host", self.host.as_str()),
            ("accelerator", self.accelerator.as_str()),
            ("model", self.model.as_str()),
            ("quantization", self.quantization.as_str()),
            ("started_utc", self.started_utc.as_str()),
            ("clock_source", self.clock_source.as_str()),
            ("subject.path", self.subject.path.as_str()),
            ("subject.commit", self.subject.commit.as_str()),
            ("client.path", self.client.path.as_str()),
            ("client.commit", self.client.commit.as_str()),
        ];
        if let Some(c) = &self.comparator {
            out.push(("comparator.commit", c.commit.as_str()));
            out.push(("comparator.cmake", c.cmake.as_str()));
            out.push(("comparator.pin_expiry", c.pin_expiry.as_str()));
        }
        if let Some(m) = &self.model_file {
            out.push(("model_file.path", m.path.as_str()));
        }
        out
    }

    fn digests(&self) -> Vec<(&'static str, &str)> {
        let mut out = vec![
            ("binary_sha256", self.binary_sha256.as_str()),
            ("subject.sha256", self.subject.sha256.as_str()),
            ("client.sha256", self.client.sha256.as_str()),
        ];
        if let Some(c) = &self.comparator {
            out.push(("comparator.sha256", c.sha256.as_str()));
        }
        if let Some(m) = &self.model_file {
            out.push(("model_file.sha256", m.sha256.as_str()));
        }
        out
    }

    /// PP-2's other half: a class the build cannot reach is a fabricated claim.
    /// Checked against the **subject's** feature set, since the subject is the
    /// process that took the path.
    fn validate_feature_set(&self) -> Result<(), String> {
        let needs_feature = matches!(self.compute_class, ComputeClass::Cuda | ComputeClass::Wgpu);
        let token = self.compute_class.wire_token();
        if needs_feature && !self.subject.feature_set.iter().any(|f| f == token) {
            return Err(format!(
                "provenance.compute_class={token} but subject.feature_set={:?} does not contain \
                 it — a build without the feature cannot take that path (PP-2)",
                self.subject.feature_set
            ));
        }
        Ok(())
    }
}

/// PP-30 — RFC3339 UTC with exactly three fractional digits and a literal `Z`.
///
/// Hand-checked rather than parsed with a calendar library: the receipt needs a
/// *canonical* spelling, not merely a parseable instant, because PP-20 compares
/// `pin_expiry` with `started_utc` as strings. `2026-09-02T10:11:12+00:00` is a
/// valid RFC3339 timestamp and would sort wrongly against this shape, so it is
/// refused rather than normalised.
fn validate_rfc3339_utc_millis(field: &str, value: &str) -> Result<(), String> {
    const SHAPE: &str = "YYYY-MM-DDTHH:MM:SS.mmmZ";
    let bytes = value.as_bytes();
    let ok = bytes.len() == 24
        && bytes.iter().enumerate().all(|(i, b)| match i {
            4 | 7 => *b == b'-',
            10 => *b == b'T',
            13 | 16 => *b == b':',
            19 => *b == b'.',
            23 => *b == b'Z',
            _ => b.is_ascii_digit(),
        });
    if !ok {
        return Err(format!(
            "{field}: {value:?} is not {SHAPE} — PP-30 needs a canonical UTC instant, because \
             PP-20 compares a pin expiry against it as a string and any other spelling sorts \
             wrongly"
        ));
    }
    Ok(())
}

/// §4.4.6 — how tokens were counted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenCountingMethod {
    /// Counts taken from the server's own `usage` fields — two servers'
    /// `usage` fields are two implementations' opinions.
    ServerUsage,
    /// Counts computed client-side with the model's own tokenizer. Canonical.
    ClientTokenizer,
}

impl TokenCountingMethod {
    /// The wire token `perf_gate.sh` reads from `tokenization.method`.
    #[must_use]
    pub fn wire_token(self) -> &'static str {
        match self {
            Self::ServerUsage => "server_usage",
            Self::ClientTokenizer => "client_tokenizer",
        }
    }
}

/// §4.4.6 — the `tokenization` block, required in every receipt.
///
/// `method` has **no default** (PP-11). The variants below make
/// "`client_tokenizer` with no digest" unrepresentable rather than merely
/// rejected, and `counts_special_tokens` / `counts_prompt_echo` are plain
/// non-optional booleans the caller must state.
///
/// > `counts_*` under `server_usage` is an operator **declaration** about the
/// > server's counting semantics, not a client measurement — §4.4.6 is titled
/// > "Token counting must be **declared**". It is required rather than
/// > defaulted precisely so it cannot be silently wrong.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case", deny_unknown_fields)]
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
}

/// Arm D's memory block. **Server-reported**: the only constructor says so in
/// its name, so a client-side guess has nowhere to enter.
///
/// The two byte figures are required. The two counters are `Option`, because
/// `apr serve` reports `admission_rejected` and `preempted_swap` as `null` —
/// it has no KV-admission refusal path and no swap path, so there is no
/// quantity for either to denote. Three of the four numbers were therefore
/// being thrown away with the block: `kv` could never be produced at all.
///
/// `null` is the honest spelling and **not** `0`: "this server does not count
/// them" and "this server counted none" are different facts, and Arm D reads
/// `admission_rejected > 0` as evidence. `perf_gate.sh:752` already names a
/// null counter in its `missing` list, so a partial block is reported as
/// partial rather than read as a zeroed one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvBlock {
    bytes_used: u64,
    bytes_reserved: u64,
    admission_rejected: Option<u64>,
    preempted_swap: Option<u64>,
}

impl KvBlock {
    /// Build the block from figures the **server** reported. A counter the
    /// server did not report is `None`, never `0`.
    #[must_use]
    pub fn from_server_report(
        bytes_used: u64,
        bytes_reserved: u64,
        admission_rejected: Option<u64>,
        preempted_swap: Option<u64>,
    ) -> Self {
        Self {
            bytes_used,
            bytes_reserved,
            admission_rejected,
            preempted_swap,
        }
    }

    /// The counter names this server did not report, for `unproduced_fields`.
    ///
    /// Empty when the block is complete — which is the must-not-fire side: a
    /// complete block must name nothing.
    #[must_use]
    pub fn uncounted_fields(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if self.admission_rejected.is_none() {
            out.push("kv.admission_rejected");
        }
        if self.preempted_swap.is_none() {
            out.push("kv.preempted_swap");
        }
        out
    }
}

/// PP-24 — what each lane's server said it would admit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SlotsAdmitted {
    /// The subject's reported slot count. `None` when it reported none.
    pub apr: Option<u32>,
    /// The comparator's reported slot count. `None` when there was no lane.
    pub llama: Option<u32>,
}

/// PP-24 — the band ladder: declared, and what both servers actually admitted.
///
/// A `c = 16` band against a subject that admitted 11 slots measured a queue,
/// not a server. So the bands that may carry numbers are **derived** from the
/// admissions rather than declared by the harness.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ladder {
    /// The bands the matrix declares.
    pub declared: Vec<u32>,
    /// `{c ∈ declared : c ≤ min(slots_admitted)}`.
    pub derived: Vec<u32>,
    /// What each lane reported.
    pub slots_admitted: SlotsAdmitted,
}

impl Ladder {
    /// Derive the ladder from the declared bands and the two servers' reports.
    ///
    /// When neither lane reported a slot count the derived ladder is the
    /// declared one — and the caller names the absence in `unproduced_fields`,
    /// which [`ReceiptInput::render`] does. Silently narrowing the ladder on no
    /// evidence would drop bands that ran perfectly well.
    #[must_use]
    pub fn derive(declared: &[u32], slots_admitted: SlotsAdmitted) -> Self {
        let cap = match (slots_admitted.apr, slots_admitted.llama) {
            (Some(a), Some(l)) => Some(a.min(l)),
            (Some(a), None) => Some(a),
            (None, Some(l)) => Some(l),
            (None, None) => None,
        };
        let derived = declared
            .iter()
            .copied()
            .filter(|c| cap.admits(*c))
            .collect();
        Self {
            declared: declared.to_vec(),
            derived,
            slots_admitted,
        }
    }

    /// True when neither lane reported a slot count, so the ladder is the
    /// declared one on no evidence.
    #[must_use]
    pub fn is_underived(&self) -> bool {
        self.slots_admitted.apr.is_none() && self.slots_admitted.llama.is_none()
    }
}

/// `cap.is_none() || cap >= c`, as a named predicate so the ladder rule reads
/// the way PP-24 states it.
trait CapExt {
    fn admits(self, c: u32) -> bool;
}

impl CapExt for Option<u32> {
    fn admits(self, c: u32) -> bool {
        match self {
            None => true,
            Some(cap) => cap >= c,
        }
    }
}

/// PP-23 — the memory-bandwidth ceiling on **per-sequence** decode.
///
/// `bandwidth_bytes_per_sec / model_bytes` tokens per second: decoding one
/// token reads the whole model once. Compared to `dec(1)` and to nothing else —
/// an aggregate over `c` sequences amortises the read across them and is
/// legitimately above the single-sequence ceiling (gx10's c=8 aggregate did exactly
/// that and was correct; the figures live in
/// evidence/perf-gate-001-w1-gx10/receipt.r1.json and are not restated here, PP-12).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Roofline {
    /// Measured `[V]` memory bandwidth, bytes per second.
    pub bandwidth_bytes_per_sec: f64,
    /// `stat -c %s` of the weights.
    pub model_bytes: u64,
}

impl Roofline {
    /// The ceiling, in tokens per second. `None` when the model has no size.
    #[must_use]
    pub fn tok_per_sec(self) -> Option<f64> {
        if self.model_bytes == 0 || self.bandwidth_bytes_per_sec <= 0.0 {
            return None;
        }
        Some(self.bandwidth_bytes_per_sec / self.model_bytes as f64)
    }
}

/// §5.1 — which workload the band set ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Workload {
    /// Homogeneous, `prompt_tokens = 512 ± 8`, `n_predict = 128`, ignore-EOS.
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
            .ok_or_else(|| format!("workload {s:?}: expected W1 or W2 (§5.1)"))
    }
}

/// SHA-256 of a file, as the 64 lowercase hex characters
/// [`Provenance::validate`] and `bench_receipt.py` both demand.
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
    /// PP-4 — the wire schema version. `3` for PP-LLAMA-001 v3.0.
    pub schema_version: u32,
    /// PP-3 — the id both lanes of this invocation share.
    pub run_id: RunId,
    /// §4.2.2 identity and §4.2.3 join key.
    pub provenance: Provenance,
    /// §4.4.6 counting declaration.
    pub tokenization: TokenizationBlock,
    /// §5.1 workload.
    pub workload: Workload,
    /// §5.1 protocol parameters, from `perf-matrix.yaml`.
    pub protocol: ProtocolParams,
    /// The commit under test (PP-21).
    pub commit: String,
    /// PP-24 — the declared and derived band ladder.
    pub ladder: Ladder,
    /// One entry per band, each carrying its own per-request records.
    pub bands: Vec<BandInput>,
    /// Arm D's server-reported memory block, when the server reported one.
    pub kv: Option<KvBlock>,
    /// PP-23 — the memory-bandwidth ceiling, when a `[V]` bandwidth exists.
    pub roofline: Option<Roofline>,
}

impl ReceiptInput {
    /// A receipt at the current schema version with no `kv` block and no
    /// roofline, which is the shape a first conformant run has.
    #[must_use]
    pub fn new(
        run_id: RunId,
        provenance: Provenance,
        tokenization: TokenizationBlock,
        workload: Workload,
        protocol: ProtocolParams,
        commit: impl Into<String>,
        ladder: Ladder,
        bands: Vec<BandInput>,
    ) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            run_id,
            provenance,
            tokenization,
            workload,
            protocol,
            commit: commit.into(),
            ladder,
            bands,
            kv: None,
            roofline: None,
        }
    }

    /// The band-derivation context this receipt implies.
    #[must_use]
    pub fn band_context(&self) -> BandContext {
        BandContext {
            schema_version: self.schema_version,
            replicates: self.protocol.replicates,
            interleaved: self.protocol.interleaved,
            comparator_stale: self.provenance.comparator_is_stale(),
            ..BandContext::default()
        }
    }

    /// PP-22 — the join key for one of this receipt's bands.
    #[must_use]
    pub fn join_key(&self, band: &BandInput) -> JoinKey {
        JoinKey::of(self, band)
    }

    /// Derive and render the receipt.
    ///
    /// # Errors
    /// When provenance or the tokenization block is invalid, when there are no
    /// bands, when any band contradicts its own clock (see
    /// [`BandInput::derive`]), when a band sits above the derived ladder
    /// without an admission cap or a decision (PP-24), when `dec(1)` exceeds the
    /// roofline (PP-23), or when the retained samples are a constant —
    /// `bench_receipt.py` calls that the fabricated-measurement shape (F12).
    pub fn render(&self) -> Result<Value, String> {
        self.provenance.validate()?;
        self.tokenization.validate()?;
        self.check_ladder_is_derived()?;
        if self.bands.is_empty() {
            return Err(
                "receipt has no bands — a measurement over zero bands is a vacuous pass"
                    .to_string(),
            );
        }
        let ctx = self.band_context();
        let stale = ctx.comparator_stale;
        let mut bands = Vec::with_capacity(self.bands.len());
        for input in &self.bands {
            self.check_ladder(input)?;
            let mut derived = input.derive_in(&ctx)?.with_join_key(self.join_key(input));
            if stale {
                let expiry = self
                    .provenance
                    .comparator
                    .as_ref()
                    .map_or("", |c| c.pin_expiry.as_str());
                derived = derived.marked_comparator_stale(expiry, &self.provenance.started_utc);
            }
            bands.push(derived);
        }
        self.check_roofline(&bands)?;
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

    /// PP-24 — `ladder.derived` must be the one `declared` and `slots_admitted`
    /// produce.
    ///
    /// [`Ladder`] has public fields and travels through the producer as data,
    /// so `derived` was whatever the caller put there. Nothing recomputed it —
    /// which means a hand-written `derived: [1, 4, 8, 16]` beside
    /// `slots_admitted: {apr: 4}` excused every band from [`Self::check_ladder`]
    /// and put four bands on the wire that measured a queue. The ladder is a
    /// FUNCTION of two recorded inputs, and here it is applied.
    fn check_ladder_is_derived(&self) -> Result<(), String> {
        let recomputed = Ladder::derive(&self.ladder.declared, self.ladder.slots_admitted);
        if recomputed.derived == self.ladder.derived {
            return Ok(());
        }
        Err(format!(
            "PP-24: ladder.derived is {:?} but declared {:?} with slots_admitted apr={:?} \
             llama={:?} derives {:?} — `derived` is `{{c ∈ declared : c ≤ min(slots_admitted)}}`, \
             not a field a producer may state. A supplied ladder that disagrees with its own \
             inputs excuses exactly the bands PP-24 exists to exclude.",
            self.ladder.derived,
            self.ladder.declared,
            self.ladder.slots_admitted.apr,
            self.ladder.slots_admitted.llama,
            recomputed.derived
        ))
    }

    /// PP-24 — a band above the derived ladder must say who decided it could
    /// run, or which lane capped it.
    fn check_ladder(&self, band: &BandInput) -> Result<(), String> {
        if self.ladder.derived.contains(&band.concurrency) {
            return Ok(());
        }
        match &band.comparator {
            ComparatorStatus::NotApplicable { .. } => Ok(()),
            ComparatorStatus::Unmeasured {
                admission_capped: Some(_),
                ..
            } => Ok(()),
            _ => Err(format!(
                "PP-24: band c={} is not in the derived ladder {:?} (slots_admitted apr={:?} \
                 llama={:?}) and carries neither an admission cap nor a decision — a band above \
                 what the servers admitted measured a queue, not a server",
                band.concurrency,
                self.ladder.derived,
                self.ladder.slots_admitted.apr,
                self.ladder.slots_admitted.llama
            )),
        }
    }

    /// PP-23 — `dec(1)` above the ceiling is schema-fatal. The **aggregate** is
    /// never compared: over `c` sequences one weight read serves `c` tokens, so
    /// an aggregate above the single-sequence ceiling is expected, not a defect.
    fn check_roofline(&self, bands: &[DerivedBand]) -> Result<(), String> {
        let Some(ceiling) = self.roofline.and_then(Roofline::tok_per_sec) else {
            return Ok(());
        };
        for b in bands.iter().filter(|b| b.concurrency == 1) {
            if let Some(dec) = b.decode_tok_per_sec {
                if dec > ceiling {
                    return Err(format!(
                        "PP-23: decode_tok_per_sec={dec:.1} at c=1 exceeds the memory-bandwidth \
                         ceiling {ceiling:.1} tok/s — decoding a token reads the whole model \
                         once, so this is not a fast run, it is a wrong measurement"
                    ));
                }
            }
        }
        Ok(())
    }

    fn assemble(&self, bands: &[DerivedBand], samples: Vec<f64>) -> Value {
        let mut map = Map::new();
        map.insert("spec".into(), json!(SPEC_ID));
        map.insert("schema_version".into(), json!(self.schema_version));
        map.insert("run_id".into(), json!(self.run_id.as_str()));
        map.insert("commit".into(), json!(self.commit));
        map.insert("workload".into(), json!(self.workload.wire_token()));
        map.insert("protocol".into(), to_value(&self.protocol));
        map.insert("client_model".into(), json!("closed_loop"));
        map.insert("provenance".into(), to_value(&self.provenance));
        map.insert("tokenization".into(), to_value(&self.tokenization));
        insert_counts(&mut map, bands);
        map.insert(
            "short_of_n_predict".into(),
            json!(sum(bands, |b| b.short_of_n_predict)),
        );
        map.insert("drain_ms".into(), json!(receipt_drain_ms(bands)));
        map.insert("n".into(), json!(samples.len()));
        map.insert("samples_ms".into(), json!(samples));
        map.insert("ladder".into(), to_value(&self.ladder));
        let roofline = self.roofline.and_then(Roofline::tok_per_sec);
        let render_ctx = RenderContexts {
            subject: RenderContext {
                agg1: band_metric(bands, 1, |b| b.aggregate_tok_per_sec),
                dec1: band_metric(bands, 1, |b| b.decode_tok_per_sec),
                roofline,
            },
            // PP-3: the comparator lane's own c=1 band is the baseline of THIS
            // receipt's c=1 band. Its `agg(1)`/`dec(1)` are the only correct
            // denominators for the baselines' scaling_efficiency and
            // overhead_share.
            comparator: RenderContext {
                agg1: baseline_metric(bands, 1, |b| b.aggregate_tok_per_sec),
                dec1: baseline_metric(bands, 1, |b| b.decode_tok_per_sec),
                roofline,
            },
        };
        map.insert(
            "bands".into(),
            Value::Array(
                bands
                    .iter()
                    .map(|b| band_json(b, &render_ctx.subject, Some(&render_ctx)))
                    .collect(),
            ),
        );
        if let Some(kv) = self.kv {
            map.insert("kv".into(), to_value(&kv));
        }
        map.insert("unproduced_fields".into(), json!(self.unproduced(bands)));
        Value::Object(map)
    }

    fn unproduced(&self, bands: &[DerivedBand]) -> Vec<String> {
        let mut out = vec![SERVER_ONLY_FIELDS.to_string()];
        match &self.kv {
            None => out.push(
                "Arm D `kv` block (bytes_used, bytes_reserved, admission_rejected, \
                 preempted_swap) — server-reported. Absent here, so this receipt is legal at \
                 merge phase and correctly FAILS at release phase rather than carrying invented \
                 memory figures."
                    .to_string(),
            ),
            Some(kv) => {
                let uncounted = kv.uncounted_fields();
                if !uncounted.is_empty() {
                    out.push(format!(
                        "Arm D {uncounted:?} — the server reported the KV byte figures but not \
                         these counters: the mechanism they would count does not exist on this \
                         build. They are null rather than 0, because \"not counted\" and \
                         \"counted none\" are different facts and Arm D reads one of them as \
                         evidence."
                    ));
                }
            }
        }
        if self.roofline.is_none() {
            out.push(
                "PP-23 roofline_tok_per_sec — no `[V]` memory bandwidth is committed for this \
                 host, so the ceiling is null on every band. A vendor GB/s figure is not a \
                 measurement (PP-12)."
                    .to_string(),
            );
        }
        if self.ladder.is_underived() {
            out.push(format!(
                "PP-24 ladder.slots_admitted — neither lane reported a slot count, so \
                 ladder.derived is the declared set {:?} on no evidence. The band ceiling is \
                 server-reported (PP-13) and this run did not observe one.",
                self.ladder.declared
            ));
        }
        if self.provenance.server_config.is_none() {
            out.push(
                "PP-2 provenance.server_config — `GET /v1/effective-config` was not stored, so \
                 resolved max_batch, GpuProfile, scheduler identity and the memory fields are \
                 absent. Every one of them is server-reported; none is inferred here."
                    .to_string(),
            );
        }
        out.extend(bands.iter().flat_map(|b| b.unproduced.clone()));
        out
    }
}

/// Receipt-level figures the per-band renderer needs, **for one lane**.
///
/// `scaling_efficiency` is `agg(c) / (c · agg(1))` and `overhead_share` is
/// `agg(1) / dec(1)`: both divide a band by its OWN lane's `c = 1` figures.
/// Rendering the baseline with the subject's context computed
/// `llama_agg(c) / (c · apr_agg(1))` — a number that is not a scaling
/// efficiency of anything, and that moves when the subject gets faster. So the
/// comparator lane gets its own context, built from the baselines.
struct RenderContext {
    agg1: Option<f64>,
    dec1: Option<f64>,
    roofline: Option<f64>,
}

/// Both lanes' contexts, so `band_json` renders each band against its own.
struct RenderContexts {
    subject: RenderContext,
    comparator: RenderContext,
}

fn band_metric(
    bands: &[DerivedBand],
    concurrency: u32,
    f: impl Fn(&DerivedBand) -> Option<f64>,
) -> Option<f64> {
    bands
        .iter()
        .find(|b| b.concurrency == concurrency)
        .and_then(f)
}

/// The same figure, taken from the COMPARATOR lane: the baseline attached to
/// the band at `concurrency`. `None` when that band did not join.
fn baseline_metric(
    bands: &[DerivedBand],
    concurrency: u32,
    f: impl Fn(&DerivedBand) -> Option<f64>,
) -> Option<f64> {
    match &bands
        .iter()
        .find(|b| b.concurrency == concurrency)?
        .comparator
    {
        ComparatorStatus::Measured(join) => f(join.baseline()),
        ComparatorStatus::NotApplicable { .. } | ComparatorStatus::Unmeasured { .. } => None,
    }
}

/// PP-10 at receipt level: the **maximum** band drain, not a mean or a sum.
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

/// PP-7 and F12, applied by the producer rather than discovered by the validator.
fn validate_samples(samples: &[f64]) -> Result<(), String> {
    if samples.is_empty() {
        return Err(
            "samples_ms would be empty — no band completed a single request, and a \
                    receipt with no retained samples permanently forecloses the bootstrap (PP-7)"
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

/// Serialise a value that cannot fail to serialise (no maps with non-string
/// keys, no non-finite floats reachable from a validated receipt).
fn to_value<T: Serialize>(value: &T) -> Value {
    serde_json::to_value(value).unwrap_or(Value::Null)
}

/// One band as JSON, against `ctx` — **its own lane's** receipt-level figures.
///
/// `comparator` is `Some` for a subject band (and carries both lanes' contexts,
/// so the baseline can be rendered against the comparator's) and `None` for a
/// baseline: a baseline that carried a baseline of its own would be a chain,
/// which `ReceiptBand::validate` refuses.
fn band_json(b: &DerivedBand, ctx: &RenderContext, comparator: Option<&RenderContexts>) -> Value {
    let mut map = Map::new();
    map.insert("concurrency".into(), json!(b.concurrency));
    map.insert("replicate".into(), json!(b.replicate));
    map.insert("status".into(), json!(b.status.wire_token()));
    if let Some(agg) = b.aggregate_tok_per_sec {
        map.insert("aggregate_tok_per_sec".into(), json!(agg));
    }
    map.insert("tokens_total".into(), json!(b.tokens_total));
    map.insert("span_ms".into(), json!(b.span_ms));
    map.insert("window_ms".into(), json!(b.window_ms));
    map.insert("drain_ms".into(), json!(b.drain_ms));
    map.insert("requested".into(), json!(b.requested));
    map.insert("completed".into(), json!(b.completed));
    map.insert("timeouts".into(), json!(b.timeouts));
    map.insert("truncated".into(), json!(b.truncated));
    map.insert("errors".into(), json!(b.errors));
    map.insert("short_of_n_predict".into(), json!(b.short_of_n_predict));
    map.insert("suspect".into(), json!(b.suspect));
    map.insert(
        "stream_mode".into(),
        b.stream_mode.map_or(Value::Null, |m| to_value(&m)),
    );
    map.insert(
        "stream_witness".into(),
        b.stream_witness.map_or(Value::Null, |w| to_value(&w)),
    );
    map.insert(
        "witness".into(),
        b.witness.as_ref().map_or(Value::Null, to_value),
    );
    map.insert("scaling_efficiency".into(), scaling_efficiency(b, ctx));
    map.insert("overhead_share".into(), overhead_share(b, ctx));
    map.insert(
        "roofline_tok_per_sec".into(),
        ctx.roofline.map_or(Value::Null, |r| json!(r)),
    );
    map.insert(
        "samples_file".into(),
        b.samples_file.as_ref().map_or(Value::Null, to_value),
    );
    map.insert("samples".into(), to_value(&b.samples));
    map.insert(
        "join_key".into(),
        b.join_key.as_ref().map_or(Value::Null, to_value),
    );
    if let Some(run_id) = &b.run_id {
        map.insert("run_id".into(), json!(run_id.as_str()));
    }
    insert_optional_latency(&mut map, b);
    if let Some(contexts) = comparator {
        insert_comparator(&mut map, &b.comparator, contexts);
    }
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
    if let Some(prefill) = b.prefill_tok_per_sec {
        map.insert("prefill_tok_per_sec".into(), json!(prefill));
        // PP-13: the number is the SERVER's, and the receipt says so beside it.
        map.insert("prefill_source".into(), json!("server"));
    }
}

/// PP-31 — `agg(c) / (c · agg(1))`. Null at `c = 1` (where it is 1 by
/// construction) and null without an `agg(1)`. REPORTED, never ratcheted: an
/// improvement to `agg(1)` lowers it, and failing a build for getting faster is
/// the defect PP-31 exists to remove.
fn scaling_efficiency(b: &DerivedBand, ctx: &RenderContext) -> Value {
    if b.concurrency <= 1 {
        return Value::Null;
    }
    match (b.aggregate_tok_per_sec, ctx.agg1) {
        (Some(agg), Some(agg1)) if agg1 > 0.0 => {
            json!(agg / (f64::from(b.concurrency) * agg1))
        }
        _ => Value::Null,
    }
}

/// §3 `overhead_share` — `agg(1) / dec(1)`, per lane. Only meaningful at
/// `c = 1`, where the two are measurements of the same single stream: the gap
/// between them is everything that is not decode.
fn overhead_share(b: &DerivedBand, ctx: &RenderContext) -> Value {
    if b.concurrency != 1 {
        return Value::Null;
    }
    match (ctx.agg1, ctx.dec1) {
        (Some(agg1), Some(dec1)) if dec1 > 0.0 => json!(agg1 / dec1),
        _ => Value::Null,
    }
}

fn insert_comparator(
    map: &mut Map<String, Value>,
    status: &ComparatorStatus,
    ctx: &RenderContexts,
) {
    map.insert("comparator_status".into(), json!(status.wire_token()));
    match status {
        ComparatorStatus::NotApplicable {
            decided_by,
            reason,
            budget,
        } => {
            map.insert("comparator_decided_by".into(), json!(decided_by));
            map.insert("comparator_reason".into(), json!(reason));
            if let Some(b) = budget {
                map.insert("comparator_budget".into(), json!(b));
            }
            map.insert("baseline".into(), Value::Null);
            map.insert("ratios".into(), Value::Null);
        }
        ComparatorStatus::Unmeasured {
            owner,
            reason,
            admission_capped,
        } => {
            map.insert("comparator_owner".into(), json!(owner));
            map.insert("comparator_reason".into(), json!(reason));
            if let Some(cap) = admission_capped {
                map.insert("comparator_admission_capped".into(), to_value(cap));
            }
            map.insert("baseline".into(), Value::Null);
            map.insert("ratios".into(), Value::Null);
        }
        ComparatorStatus::Measured(join) => {
            // The baseline is rendered by the SAME function, minus its own
            // baseline/ratios: a comparator lane that did not have to satisfy
            // every receipt rule is not a baseline (PP-3). Its per-lane figures
            // come from `ctx.comparator`, never from the subject's.
            map.insert(
                "baseline".into(),
                band_json(join.baseline(), &ctx.comparator, None),
            );
            map.insert("ratios".into(), to_value(join.ratios()));
        }
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

// ---------------------------------------------------------------------------
// The typed reader
// ---------------------------------------------------------------------------

/// A receipt, read back. `deny_unknown_fields` at every level.
///
/// # Why the producer has a reader at all
///
/// Until v3 the receipt had a serialiser and no deserialiser. Every must-fire
/// of the form "strip field X and prove the gate reds" was therefore testable
/// only in python inside `perf_gate.sh --selftest` — and that runs only in the
/// `ci` job, never in `workspace-test`, whose image has no python3. So the Rust
/// half of those rules could not be turned red at all.
///
/// `deny_unknown_fields` is the other half: a receipt carrying a key this type
/// does not know is refused rather than silently ignored, which is what stops a
/// producer from inventing `agg_ratio` beside a band and a reader from quietly
/// dropping it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Receipt {
    /// `"PP-LLAMA-001 v3.0"`.
    pub spec: String,
    /// PP-4 — `3` for v3.
    pub schema_version: u32,
    /// PP-3 — shared by both lanes of one invocation.
    pub run_id: RunId,
    /// The commit under test.
    pub commit: String,
    /// §5.1 workload.
    pub workload: Workload,
    /// §5.1 protocol parameters.
    pub protocol: ProtocolParams,
    /// §4.4.1 — always `closed_loop` in this producer.
    pub client_model: String,
    /// §4.2.2 identity.
    pub provenance: Provenance,
    /// §4.4.6 counting declaration.
    pub tokenization: TokenizationBlock,
    /// Requests issued across every band.
    pub requested: usize,
    /// Requests that completed.
    pub completed: usize,
    /// Requests that hit the hard timeout.
    pub timeouts: usize,
    /// Requests abandoned at a drain deadline.
    pub truncated: usize,
    /// Requests that failed otherwise.
    pub errors: usize,
    /// PP-28 — completed requests short of `n_predict`, over every band.
    pub short_of_n_predict: usize,
    /// PP-10 — the worst band's drain phase.
    pub drain_ms: f64,
    /// Retained sample count.
    pub n: usize,
    /// Retained per-request end-to-end latencies.
    pub samples_ms: Vec<f64>,
    /// PP-24 — the ladder.
    pub ladder: Ladder,
    /// The bands.
    pub bands: Vec<ReceiptBand>,
    /// Arm D's server-reported memory block.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv: Option<KvBlock>,
    /// Fields this producer could not produce, each with its reason.
    pub unproduced_fields: Vec<String>,
    /// PP-21 — the detached HMAC block `scripts/perf_receipt_sign.sh` appends on
    /// the measuring host. The renderer never writes it (it must not be able to
    /// sign its own output), so the reader carries it opaquely: a signed receipt
    /// must parse, and `scripts/lib/receipt_sig.py --verify` is the verifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<serde_json::Value>,
}

/// One band, read back.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(clippy::struct_excessive_bools)]
pub struct ReceiptBand {
    /// Fixed concurrency `c`.
    pub concurrency: u32,
    /// Which replicate, 1-based.
    pub replicate: u32,
    /// §7.4 status token.
    pub status: String,
    /// §3 `agg`. Absent on an `INVALID-CORRECTNESS` band.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aggregate_tok_per_sec: Option<f64>,
    /// `agg`'s numerator.
    pub tokens_total: u64,
    /// `agg`'s denominator, in milliseconds.
    pub span_ms: f64,
    /// `T`.
    pub window_ms: f64,
    /// PP-10 drain phase.
    pub drain_ms: f64,
    /// Requests issued.
    pub requested: usize,
    /// Requests completed.
    pub completed: usize,
    /// Requests that timed out.
    pub timeouts: usize,
    /// Requests abandoned at the drain deadline.
    pub truncated: usize,
    /// Requests that failed otherwise.
    pub errors: usize,
    /// PP-28 count for this band.
    pub short_of_n_predict: usize,
    /// PP-10 `SUSPECT` annotations.
    pub suspect: Vec<String>,
    /// PP-27 — what the server declared.
    pub stream_mode: Option<StreamMode>,
    /// PP-27 — what the client observed.
    pub stream_witness: Option<StreamWitness>,
    /// PP-26 — the correctness witness.
    pub witness: Option<BatchInvarianceWitness>,
    /// PP-31 — reported, never ratcheted.
    pub scaling_efficiency: Option<f64>,
    /// §3 `overhead_share`, at c=1 only.
    pub overhead_share: Option<f64>,
    /// PP-23 ceiling.
    pub roofline_tok_per_sec: Option<f64>,
    /// PP-7 — the retained gz side file.
    pub samples_file: Option<SamplesFile>,
    /// PP-7 — the per-request rows.
    pub samples: Vec<SampleRow>,
    /// PP-22 — the key this band joins on.
    pub join_key: Option<JoinKey>,
    /// PP-3 — the run this band belongs to. Present on a baseline.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_id: Option<RunId>,
    /// §3 `dec`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_tok_per_sec: Option<f64>,
    /// p50 TTFT.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_p50_ms: Option<f64>,
    /// p95 TTFT.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_p95_ms: Option<f64>,
    /// p50 pooled ITL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub itl_p50_ms: Option<f64>,
    /// p95 pooled ITL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub itl_p95_ms: Option<f64>,
    /// §3 `prefill`, server-reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_tok_per_sec: Option<f64>,
    /// `"server"` whenever `prefill_tok_per_sec` is present (PP-13).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_source: Option<String>,
    /// Legacy comparator posture token.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub comparator_status: Option<String>,
    /// Who owes an `UNMEASURED` comparator.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub comparator_owner: Option<String>,
    /// Who decided a `NOT_APPLICABLE` comparator.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub comparator_decided_by: Option<String>,
    /// Why.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub comparator_reason: Option<String>,
    /// PP-24 — the server-reported ceiling behind an `NA`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub comparator_budget: Option<String>,
    /// PP-24 — which lane capped admission.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub comparator_admission_capped: Option<AdmissionCap>,
    /// PP-3 — the comparator lane's band. Absent on a baseline itself.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline: Option<Box<ReceiptBand>>,
    /// P-5 — the ratios. Absent on a baseline itself.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ratios: Option<BandRatios>,
}

impl Receipt {
    /// Parse a receipt, refusing any key this type does not know.
    ///
    /// # Errors
    /// On malformed JSON, a missing required field, or an unknown one.
    pub fn parse(text: &str) -> Result<Self, String> {
        serde_json::from_str(text).map_err(|e| format!("parsing receipt: {e}"))
    }

    /// The L1 rules a reader can apply to a receipt it did not produce.
    ///
    /// # Errors
    /// When the spec string or schema version is wrong, when provenance fails
    /// its own checks (PP-2, 18, 20, 25, 30), when a band's `status` is outside
    /// the §7.4 vocabulary, or when a band carries `ratios` without a
    /// `baseline` (PP-3, PP-17).
    pub fn validate(&self) -> Result<(), String> {
        if self.spec != SPEC_ID {
            return Err(format!(
                "receipt.spec is {:?}, expected {SPEC_ID:?}",
                self.spec
            ));
        }
        if self.schema_version != SCHEMA_VERSION {
            return Err(format!(
                "receipt.schema_version is {}, expected {SCHEMA_VERSION} — a receipt at another \
                 version is historical and is never a baseline (PP-4)",
                self.schema_version
            ));
        }
        self.provenance.validate()?;
        self.tokenization.validate()?;
        if self.bands.is_empty() {
            return Err(
                "receipt has no bands — a measurement over zero bands is a vacuous \
                        pass"
                    .to_string(),
            );
        }
        self.check_run_id()?;
        for band in &self.bands {
            band.validate()?;
        }
        Ok(())
    }

    /// PP-3 / §1(d) — the `run_id` a reader recomputes from this receipt's own
    /// four inputs must be the one written on it.
    ///
    /// Every input is now on the wire: `provenance.started_utc`,
    /// `provenance.host`, `provenance.client.sha256`, `provenance.client.pid`.
    /// Before `pid` was carried, "derived rather than random, so it is
    /// reproducible from the receipt" was a comment and not a checkable claim:
    /// a receipt could state any 32 hex characters and nothing could disagree.
    fn check_run_id(&self) -> Result<(), String> {
        let recomputed = RunId::derive(
            &self.provenance.started_utc,
            &self.provenance.host,
            &self.provenance.client.sha256,
            self.provenance.client.pid,
        );
        if recomputed == self.run_id {
            return Ok(());
        }
        Err(format!(
            "PP-3: run_id is {} but sha256(started_utc ‖ host ‖ client.sha256 ‖ client.pid)[..32]              over this receipt's own provenance is {} — the id is DERIVED, and one that its own              contents do not reproduce identifies nothing",
            self.run_id.as_str(),
            recomputed.as_str()
        ))
    }
}

impl ReceiptBand {
    /// The band-level half of [`Receipt::validate`].
    ///
    /// # Errors
    /// When `status` is outside the §7.4 vocabulary, when `ratios` appear
    /// without a `baseline` (PP-3), or when a `baseline` carries its own
    /// baseline (a baseline is one lane, not a chain).
    pub fn validate(&self) -> Result<(), String> {
        let known = BandStatus::vocabulary()
            .iter()
            .any(|s| s.wire_token() == self.status);
        if !known {
            return Err(format!(
                "band c={}: status {:?} is outside the §7.4 vocabulary {:?}",
                self.concurrency,
                self.status,
                BandStatus::vocabulary()
                    .iter()
                    .map(|s| s.wire_token())
                    .collect::<Vec<_>>()
            ));
        }
        if self.ratios.is_some() && self.baseline.is_none() {
            return Err(format!(
                "PP-3 band c={}: `ratios` without a `baseline` — a ratio is representable only \
                 against a baseline object that itself passes every receipt rule and shares the \
                 run_id",
                self.concurrency
            ));
        }
        if let Some(baseline) = &self.baseline {
            if baseline.baseline.is_some() || baseline.ratios.is_some() {
                return Err(format!(
                    "PP-3 band c={}: the baseline carries its own baseline/ratios — a baseline is \
                     one comparator lane, not a chain of them",
                    self.concurrency
                ));
            }
            baseline.validate()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod producer_tests {
    //! The conversions the CLI needs, the digest it cannot type, and the
    //! validators that stop a receipt from being written at all.

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

    /// PP-3 / PP-30 — the id is DERIVED, so a reader holding the receipt can
    /// recompute it. A random id could be claimed and never checked.
    #[test]
    fn the_run_id_is_derived_from_the_receipts_own_contents() {
        let a = RunId::derive("2026-09-02T10:11:12.345Z", "lambda", &"c".repeat(64), 4242);
        let b = RunId::derive("2026-09-02T10:11:12.345Z", "lambda", &"c".repeat(64), 4242);
        assert_eq!(a, b, "the same four facts give the same id");
        assert_eq!(a.as_str().len(), 32);
        assert!(a
            .as_str()
            .bytes()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));

        for changed in [
            RunId::derive("2026-09-02T10:11:12.346Z", "lambda", &"c".repeat(64), 4242),
            RunId::derive("2026-09-02T10:11:12.345Z", "gx10", &"c".repeat(64), 4242),
            RunId::derive("2026-09-02T10:11:12.345Z", "lambda", &"d".repeat(64), 4242),
            RunId::derive("2026-09-02T10:11:12.345Z", "lambda", &"c".repeat(64), 4243),
        ] {
            assert_ne!(a, changed, "every input must move the id");
        }
    }

    /// And a malformed one does not parse, so it cannot reach the join.
    #[test]
    fn a_malformed_run_id_is_refused() {
        assert!(RunId::try_from("abc".to_string()).is_err());
        assert!(RunId::try_from("A".repeat(32)).is_err(), "case matters");
        assert!(RunId::try_from("z".repeat(32)).is_err(), "hex only");
        assert!(RunId::try_from("a".repeat(33)).is_err());
        assert!(RunId::try_from("a".repeat(32)).is_ok());
    }

    /// PP-30 — the canonical spelling is the one PP-20 can compare as a string.
    #[test]
    fn started_utc_must_be_rfc3339_utc() {
        assert!(validate_rfc3339_utc_millis("t", "2026-09-02T10:11:12.345Z").is_ok());
        for bad in [
            "",
            "2026-09-02",
            "2026-09-02T10:11:12Z",
            "2026-09-02T10:11:12.345+00:00",
            "2026-09-02t10:11:12.345Z",
            "2026-09-02T10:11:12.3456Z",
            "not-a-time-at-all-....Z",
        ] {
            assert!(
                validate_rfc3339_utc_millis("t", bad).is_err(),
                "{bad:?} must be refused"
            );
        }
    }

    /// And the canonical spelling really does sort chronologically, which is
    /// what PP-20's string comparison rests on.
    #[test]
    fn canonical_timestamps_sort_chronologically() {
        let mut times = vec![
            "2026-12-01T00:00:00.000Z".to_string(),
            "2026-09-02T10:11:12.345Z".to_string(),
            "2026-09-02T10:11:12.344Z".to_string(),
            "2025-01-01T00:00:00.000Z".to_string(),
        ];
        times.sort();
        assert_eq!(
            times,
            vec![
                "2025-01-01T00:00:00.000Z",
                "2026-09-02T10:11:12.344Z",
                "2026-09-02T10:11:12.345Z",
                "2026-12-01T00:00:00.000Z",
            ]
        );
    }

    /// PP-24 — the ladder is the declared set capped by the SMALLER admission.
    #[test]
    fn ladder_derives_from_the_minimum_admission() {
        let declared = [1_u32, 4, 8, 16];
        let l = Ladder::derive(
            &declared,
            SlotsAdmitted {
                apr: Some(11),
                llama: Some(16),
            },
        );
        assert_eq!(l.derived, vec![1, 4, 8], "c=16 exceeds the subject's 11");
        assert!(!l.is_underived());

        let other_way = Ladder::derive(
            &declared,
            SlotsAdmitted {
                apr: Some(16),
                llama: Some(4),
            },
        );
        assert_eq!(other_way.derived, vec![1, 4], "the comparator caps too");

        let one_lane = Ladder::derive(
            &declared,
            SlotsAdmitted {
                apr: Some(8),
                llama: None,
            },
        );
        assert_eq!(one_lane.derived, vec![1, 4, 8]);

        let blind = Ladder::derive(
            &declared,
            SlotsAdmitted {
                apr: None,
                llama: None,
            },
        );
        assert_eq!(
            blind.derived,
            vec![1, 4, 8, 16],
            "no evidence does not narrow the ladder"
        );
        assert!(blind.is_underived(), "…but it is named as unevidenced");
    }

    /// PP-30 — the helper produces exactly the canonical shape the validator
    /// accepts. A producer whose own timestamp its validator rejects would fail
    /// only after the measurement had been paid for.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn now_utc_millis_is_the_shape_the_validator_accepts() {
        let now = now_utc_millis();
        validate_rfc3339_utc_millis("now", &now)
            .unwrap_or_else(|e| panic!("{now:?} must satisfy the receipt's own check: {e}"));
        assert!(now.ends_with('Z'));
        assert_eq!(now.len(), 24);
    }

    /// PP-23 — the ceiling is bandwidth over model bytes, and a model with no
    /// size has no ceiling rather than an infinite one.
    #[test]
    fn the_roofline_is_bandwidth_over_model_bytes() {
        let r = Roofline {
            bandwidth_bytes_per_sec: 1_008_000_000_000.0,
            model_bytes: 4_683_073_440,
        };
        let ceiling = r.tok_per_sec().expect("a sized model has a ceiling");
        assert!((ceiling - 215.2).abs() < 0.1, "{ceiling}");
        assert!(Roofline {
            bandwidth_bytes_per_sec: 1.0,
            model_bytes: 0
        }
        .tok_per_sec()
        .is_none());
        assert!(Roofline {
            bandwidth_bytes_per_sec: 0.0,
            model_bytes: 10
        }
        .tok_per_sec()
        .is_none());
    }
}
