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

    fn to_json(&self) -> Value {
        let mut map = Map::new();
        map.insert("method".into(), json!(self.method().wire_token()));
        let (special, echo) = match self {
            Self::ServerUsage {
                counts_special_tokens,
                counts_prompt_echo,
            } => (*counts_special_tokens, *counts_prompt_echo),
            Self::ClientTokenizer {
                tokenizer_sha256,
                counts_special_tokens,
                counts_prompt_echo,
            } => {
                map.insert("tokenizer_sha256".into(), json!(tokenizer_sha256));
                (*counts_special_tokens, *counts_prompt_echo)
            }
        };
        map.insert("counts_special_tokens".into(), json!(special));
        map.insert("counts_prompt_echo".into(), json!(echo));
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

/// Everything needed to render one host × workload receipt.
#[derive(Debug, Clone, PartialEq)]
pub struct ReceiptInput {
    /// §4.2.2 identity and §4.2.3 join key.
    pub provenance: Provenance,
    /// §4.4.6 counting declaration.
    pub tokenization: TokenizationBlock,
    /// §4.3 workload.
    pub workload: Workload,
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
        self.tokenization.validate()?;
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
        map.insert("client_model".into(), json!("closed_loop"));
        map.insert("provenance".into(), self.provenance.to_json());
        map.insert("tokenization".into(), self.tokenization.to_json());
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
