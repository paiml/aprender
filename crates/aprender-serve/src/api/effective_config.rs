//! `GET /v1/effective-config` — what THIS process resolved, read from the process.
//!
//! PP-LLAMA-001 §5.2 / PP-2. Every number a receipt records about the server
//! under test has to come from the server, not from a CLI flag the operator
//! typed and not from a `cfg!` the harness guessed. Before this endpoint the
//! three closest surfaces each reported something else:
//!
//! * `/health` collapsed every accelerator to `compute_mode: "cpu" | "gpu"`;
//! * `/v1/metrics` reported `gpu_memory_total_bytes` as the literal
//!   `24 * 1024^3` ("RTX 4090 has 24GB VRAM") — a constant, on any device;
//! * `apr test llm --band` took `--compute-class` and `--server-feature` as
//!   operator-typed CLI arguments and wrote them into the receipt verbatim.
//!
//! So the rules this module enforces:
//!
//! 1. **Residency, never `cfg!`.** `compute_class` and `backend_loaded` are
//!    derived from which backend is actually resident in [`AppState`]. A CUDA
//!    build serving a CPU model reports `cpu`. The `cfg!` list is reported
//!    SEPARATELY, as `build_features`, so a reader can see both and a validator
//!    can cross-check them (PP-2 must-fire: `compute_class: "cuda"` with no
//!    `"cuda"` in `build_features`).
//! 2. **Measured or absent — but "the mechanism does not exist" is a fact, not
//!    an absence.** Every field this server did not measure is `null`, never a
//!    plausible-looking default. `kv_blocks_total` is `null` with a stated
//!    `kv_layout` because this KV cache has no block table at all, so there is
//!    no quantity for the field to denote and a number would be a fabrication.
//!    Where a quantity DOES exist and this server knows it is zero, the honest
//!    report is `0` beside the reason: `admission_rejected` is counted, and
//!    `preempted_swap` is `0` because the layout named in `kv_layout` has no
//!    swap path. `null` there would say "this server does not count", which is
//!    a different statement and one a reader cannot act on.
//! 3. **One JSON shape on every build.** The `cuda` block is `null` on a
//!    non-CUDA build and an object on a CUDA build; no key appears or vanishes
//!    with a feature, so a receipt validator can assert the key set once.
//! 4. **Snapshot static, `try_read` live.** PMAT-073 records a ~2 s stall when
//!    a handler takes `cuda_model().read()` behind the scheduler's `write()`.
//!    This endpoint must never join that queue: everything static is snapshotted
//!    at `AppState` construction, and the live CUDA fields are read with
//!    `try_read`. A failed try sets `lock_contended: true` and reports the block
//!    as absent rather than blocking or inventing values.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use axum::{extract::State, Json};
use serde::Serialize;

use super::AppState;

/// Wire schema version of the `/v1/effective-config` body.
pub const EFFECTIVE_CONFIG_SCHEMA_VERSION: u32 = 1;

/// How this server tells the time, stated on the wire (PP-30).
///
/// A receipt carrying `started_utc` without saying which clock produced it
/// cannot be compared against a comparator lane's timestamps.
pub const CLOCK_SOURCE: &str =
    "chrono::Utc::now (CLOCK_REALTIME) + std::time::Instant (CLOCK_MONOTONIC)";

// ---------------------------------------------------------------------------
// Server clock (PP-30)
// ---------------------------------------------------------------------------

/// The one start time this process reports.
///
/// Before this there were three, and they disagreed about what "start" meant:
/// a `OnceLock<Instant>` latched on the first `/health` hit (monotonic only, so
/// no UTC to put in a receipt), `MetricsCollector.start_time`, and a
/// `OnceLock<i64>` latched on the FIRST CHAT REQUEST whose doc called itself
/// "model load time". `/health`'s `uptime_sec`, `/v1/models`' `created` and this
/// endpoint now read this one clock, so they cannot disagree.
///
/// It is PROCESS-wide ([`ServerClock::process`]), not per-[`AppState`]: two
/// `AppState`s built in one process describe one running server, and
/// `created` on `/v1/models` is required to be stable across requests even when
/// the caller rebuilds the router (FALSIFY-CRUX-C-33-004). It latches at the
/// first `AppState` construction, which in a real server is the moment the
/// model finished loading.
#[derive(Debug)]
pub struct ServerClock {
    started_utc: String,
    started_unix: i64,
    started: Instant,
    pid: u32,
}

impl ServerClock {
    /// Latch the clock now. Prefer [`ServerClock::process`].
    #[must_use]
    pub fn new() -> Self {
        let now = chrono::Utc::now();
        Self {
            started_utc: now.to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
            started_unix: now.timestamp(),
            started: Instant::now(),
            pid: std::process::id(),
        }
    }

    /// The process-wide clock, latched on first use.
    #[must_use]
    pub fn process() -> Arc<Self> {
        static CLOCK: std::sync::OnceLock<Arc<ServerClock>> = std::sync::OnceLock::new();
        CLOCK.get_or_init(|| Arc::new(ServerClock::new())).clone()
    }

    /// RFC 3339 UTC instant this server started, millisecond precision.
    #[must_use]
    pub fn started_utc(&self) -> &str {
        &self.started_utc
    }

    /// The same instant as Unix seconds, for the OpenAI `created` fields.
    #[must_use]
    pub fn started_unix_secs(&self) -> i64 {
        self.started_unix
    }

    /// Monotonic seconds since start. Never decreases (FALSIFY-CRUX-C-34-003).
    #[must_use]
    pub fn uptime_sec(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }

    /// OS process id of the server.
    #[must_use]
    pub fn pid(&self) -> u32 {
        self.pid
    }

    /// The clock as it appears on the wire.
    #[must_use]
    pub fn report(&self) -> ServerReport {
        ServerReport {
            version: crate::VERSION,
            build_commit: option_env!("APR_GIT_SHA").map(str::to_string),
            pid: self.pid,
            started_utc: self.started_utc.clone(),
            clock_source: CLOCK_SOURCE,
            uptime_sec: self.uptime_sec(),
        }
    }
}

impl Default for ServerClock {
    fn default() -> Self {
        Self::new()
    }
}

/// Identity and clock of the serving process (PP-30).
#[derive(Debug, Clone, Serialize)]
pub struct ServerReport {
    /// `realizar`'s crate version.
    pub version: &'static str,
    /// Commit this server binary was built from, when the build recorded one.
    ///
    /// `realizar`'s own build script does not set `APR_GIT_SHA`, so on an
    /// `apr serve` process this is the commit the CLI reported through
    /// [`OffloadReport::build_commit`]; `null` when nothing recorded one.
    /// Never the string `"unknown"` — an absent commit is a fact, a fake one
    /// is not.
    pub build_commit: Option<String>,
    /// OS process id — lets a harness match a receipt to an `nvidia-smi` row.
    pub pid: u32,
    /// RFC 3339 UTC start instant (PP-30).
    pub started_utc: String,
    /// Which clocks produced `started_utc` and `uptime_sec` (PP-30).
    pub clock_source: &'static str,
    /// Monotonic seconds since start.
    pub uptime_sec: f64,
}

// ---------------------------------------------------------------------------
// In-flight counter (PP-24)
// ---------------------------------------------------------------------------

/// How many requests the scheduler currently holds, and the high-water mark.
///
/// PP-24 derives the concurrency ladder from what the server will actually
/// admit. `slots_admitted` (the ceiling) is known before a band runs;
/// `peak_in_flight` is what the band achieved. Both are reported, because a
/// ladder derived from a ceiling the server never reached is a ladder derived
/// from an optimism.
#[derive(Debug, Default)]
pub struct InFlightCounter {
    now: AtomicUsize,
    peak: AtomicUsize,
}

impl InFlightCounter {
    /// A fresh counter.
    #[must_use]
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Record a request entering the scheduler; returns the new in-flight count.
    pub fn enter(&self) -> usize {
        let now = self.now.fetch_add(1, Ordering::AcqRel) + 1;
        self.peak.fetch_max(now, Ordering::AcqRel);
        now
    }

    /// Set the in-flight count to what the scheduler ITSELF holds — the number
    /// of slots it has live (`BatchState::m`) — and raise the peak to match.
    ///
    /// This is the producer the batch scheduler uses. Counting `enter`/`leave`
    /// per request double-counted a staggered prompt (once in the batch it
    /// arrived with, once when it joined) and never saw a recycled slot at all
    /// (PP-24 review, cross-vendor finding); the scheduler's own slot count is
    /// the fact, so the counter mirrors it instead of re-deriving it.
    pub fn set(&self, active: usize) {
        self.now.store(active, Ordering::Release);
        self.peak.fetch_max(active, Ordering::AcqRel);
    }

    /// Record a request leaving the scheduler.
    ///
    /// Saturating: an unmatched `leave` must not wrap the counter to
    /// `usize::MAX` and make every later reading nonsense.
    pub fn leave(&self) {
        let _ = self
            .now
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                Some(n.saturating_sub(1))
            });
    }

    /// Requests in flight right now.
    #[must_use]
    pub fn in_flight_now(&self) -> usize {
        self.now.load(Ordering::Acquire)
    }

    /// Highest simultaneous in-flight count since start.
    #[must_use]
    pub fn peak_in_flight(&self) -> usize {
        self.peak.load(Ordering::Acquire)
    }
}

// ---------------------------------------------------------------------------
// Admission (§5.2 `kv.admission_rejected`)
// ---------------------------------------------------------------------------

/// How this server treats a request it has no free slot for.
///
/// One wire token, so a reader of `admission_rejected: 0` can tell "nothing was
/// refused because nothing is ever refused" from "nothing was refused this
/// run". This scheduler QUEUES: a request beyond `slots_admitted` waits for the
/// next batch rather than being turned away.
pub const ADMISSION_POLICY: &str = "queue";

/// Requests this server REFUSED admission to.
///
/// Queueing is not the same as never refusing. The submission channel is
/// bounded, and a request that finds it full is answered `503`; that refusal is
/// the one admission event this server has, and it is the one this counts.
/// Counting it is what makes `admission_rejected` a measurement rather than the
/// constant `0` a reader would have to take on trust.
#[derive(Debug, Default)]
pub struct AdmissionCounter {
    rejected: AtomicU64,
}

impl AdmissionCounter {
    /// A fresh counter.
    #[must_use]
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Record one refused request; returns the new total.
    pub fn record_rejected(&self) -> u64 {
        self.rejected.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Requests refused since start.
    #[must_use]
    pub fn rejected(&self) -> u64 {
        self.rejected.load(Ordering::Acquire)
    }
}

// ---------------------------------------------------------------------------
// Offload (PP-14, PP-15)
// ---------------------------------------------------------------------------

/// What the loader did with `--gpu-layers`, stated by the process that did it.
///
/// PP-15: "a boolean accelerator flag has no observable resolution". The three
/// numbers below were computed in `apr-cli` and PRINTED ONLY, next to a
/// `backend=` label that was a `cfg!` — a build-time string, not what loaded.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct OffloadReport {
    /// What the operator asked for, as typed (`"all"`, `"0"`, `"auto"`, `"none"`).
    pub gpu_layers_requested: String,
    /// What the loader placed on the accelerator.
    pub gpu_layers_resolved: u32,
    /// How many layers the model has.
    pub gpu_layers_total: u32,
    /// The policy that produced `gpu_layers_resolved` from the request.
    ///
    /// `"all_or_nothing"` states the limitation the loader documents: no
    /// free-VRAM query is reachable before construction, so a partial `Exact(n)`
    /// is REFUSED rather than silently rounded, and `Auto` resolves to all.
    pub offload_policy: &'static str,
    /// Layer counts auto-fit changed. Empty when auto-fit changed nothing.
    pub autofit_applied: Vec<String>,
    /// Arguments the operator set explicitly on the command line.
    pub explicit_args: Vec<String>,
    /// `cfg!` feature list of the process that built the CLI (`cuda`,
    /// `cuda-batch`, `wgpu`, …). Surfaced at the top level of the response as
    /// `build_features_cli`, so it has one home on the wire.
    #[serde(skip_serializing)]
    pub build_features: Vec<String>,
    /// Commit the launching CLI was built from. Surfaced as `server.build_commit`.
    #[serde(skip_serializing)]
    pub build_commit: Option<String>,
}

/// PP-14: auto-fit must never overwrite something the operator set explicitly.
///
/// The two sets are disjoint or the report is a lie: a run that says both
/// "the operator pinned `gpu_layers`" and "auto-fit chose `gpu_layers`" cannot
/// be reproduced from its own receipt.
#[must_use]
pub fn pp14_holds(report: &OffloadReport) -> bool {
    !report
        .autofit_applied
        .iter()
        .any(|applied| report.explicit_args.iter().any(|arg| arg == applied))
}

// ---------------------------------------------------------------------------
// Scheduler (PP-13, PP-24)
// ---------------------------------------------------------------------------

/// The scheduler this server is running, and its admission ceiling.
///
/// The identity and its window/batch settings used to be printed to stdout and
/// then MOVED into the spawned scheduler, so nothing after startup could say
/// which of the two schedulers was running or what it would admit.
#[derive(Debug, Clone, Serialize)]
pub struct SchedulerReport {
    /// `"cuda_batch"`, `"iteration"`, `"parity052_batch"` or `"direct_rwlock"`
    /// (the serialized fallback taken when no scheduler channel is attached).
    pub kind: &'static str,
    /// Requests the scheduler will run concurrently.
    pub max_in_flight: usize,
    /// Batch-formation window in milliseconds.
    pub window_ms: u64,
    /// Prefill chunk size, for schedulers that chunk prefill.
    pub prefill_chunk_size: Option<usize>,
    /// Per-step token budget, for schedulers that have one.
    pub token_budget: Option<usize>,
    /// The admission ceiling PP-24 derives the ladder from.
    pub slots_admitted: usize,
    /// Where the ceiling came from: `"kv_budget"`, `"env"` or `"default"`.
    pub admission_ceiling_reason: &'static str,
    /// Live: requests in flight at the moment of this GET.
    ///
    /// `null` when this scheduler is not instrumented — a scheduler that does
    /// not count cannot report `0`, because `0` is also what an idle
    /// instrumented scheduler reports and PP-24 would read the two the same way.
    pub in_flight_now: Option<usize>,
    /// Live: highest simultaneous in-flight count since start. `null` as above.
    pub peak_in_flight: Option<usize>,
}

/// PP-24: where the admission ceiling came from, as a wire token.
///
/// `slots_admitted` is the number PP-24 derives the concurrency ladder from, so
/// a reader has to be able to tell "the KV budget allowed this many" from "the
/// operator pinned it" from "nobody decided and this is the default". The
/// `CUDA_MAX_BATCH` env transport erased exactly that distinction; this maps the
/// recovered `MaxBatchSizing::source` onto the vocabulary §5.2 uses.
#[must_use]
pub fn admission_ceiling_reason(max_batch_source: Option<&str>) -> &'static str {
    match max_batch_source {
        Some("env") => "env",
        Some("computed") => "kv_budget",
        _ => "default",
    }
}

// ---------------------------------------------------------------------------
// Model + KV
// ---------------------------------------------------------------------------

/// What the loader measured about the served model.
///
/// Every field is `Option`: `None` means this server does not know, and the
/// reader must not substitute a default (`ModelSourceInfo`'s rule).
#[derive(Debug, Clone, Serialize)]
pub struct ModelReport {
    /// Absolute path of the model file.
    pub path: Option<String>,
    /// Size of the model file in bytes.
    pub size_bytes: Option<u64>,
    /// Container format (`gguf`, `apr`, `safetensors`).
    pub format: Option<String>,
    /// Quantization of the loaded weights.
    pub quantization: Option<String>,
    /// Model architecture.
    pub architecture: Option<String>,
    /// Context length this server was CONFIGURED with.
    pub context_length: Option<usize>,
    /// The model's own advertised maximum context.
    pub model_max_context_length: Option<usize>,
    /// Computed content hash of the model bytes, when one was computed.
    pub content_hash: Option<String>,
    /// Parameter count the loader derived.
    pub parameter_count: Option<u64>,
    /// Whether a model is resident and servable right now.
    pub loaded: bool,
}

impl ModelReport {
    fn from_state(state: &AppState) -> Self {
        let source = state.model_source();
        Self {
            path: source.and_then(|s| s.path().map(str::to_string)),
            size_bytes: source.and_then(super::ModelSourceInfo::size_bytes),
            format: source.and_then(|s| s.format().map(str::to_string)),
            quantization: source.and_then(|s| s.quantization().map(str::to_string)),
            architecture: source
                .and_then(|s| s.architecture().map(str::to_string))
                .or_else(|| state.model_architecture()),
            context_length: source.and_then(super::ModelSourceInfo::context_length),
            model_max_context_length: source
                .and_then(super::ModelSourceInfo::model_max_context_length),
            content_hash: source.and_then(|s| s.content_hash().map(str::to_string)),
            parameter_count: source.and_then(super::ModelSourceInfo::parameter_count),
            loaded: state.model_loaded(),
        }
    }
}

/// KV-cache accounting (§5.2 memory fields, §9 #7).
///
/// `kv_blocks_total` is deliberately `Option` and deliberately `None` on this
/// backend: the CUDA path allocates one CONTIGUOUS KV buffer per slot and has
/// no block table, so there is no quantity for the field to denote. `kv_layout`
/// says so, and `kv_slots_allocated` / `kv_slots_max` carry the real shape.
///
/// The four figures Arm D reads — `bytes_used`, `bytes_reserved`,
/// `admission_rejected`, `preempted_swap` — are the reason the last two are
/// NUMBERS and not `Option`s. The band producer builds its memory block only
/// when the server reported both byte figures (`KvBlock::from_server_report`; the two
/// counters may be null and are then named in `unproduced_fields`), so a
/// `null` in either of the counters silently deleted Arm D from every receipt
/// this endpoint fed: three measured numbers and a `null` read exactly like a
/// server that predates the block. Where the mechanism does not exist, the
/// honest report is the number with its reason beside it — `admission_policy`
/// for the first, `kv_layout` for the second — and the type now makes `null`
/// unrepresentable rather than merely discouraged.
#[derive(Debug, Clone, Serialize)]
pub struct KvReport {
    /// Bytes of KV actually in use (single-sequence cache + allocated slots).
    pub bytes_used: Option<usize>,
    /// Bytes reserved for KV by the allocation.
    pub bytes_reserved: Option<usize>,
    /// Requests refused admission, counted by [`AdmissionCounter`].
    ///
    /// Normally `0`, because [`ADMISSION_POLICY`] is `"queue"`: a request
    /// beyond `slots_admitted` waits for the next batch. It is not always `0` —
    /// the bounded submission channel refuses with `503` when it is full, and
    /// that is what this counts.
    pub admission_rejected: u64,
    /// The policy the count above is a count OF: see [`ADMISSION_POLICY`].
    pub admission_policy: &'static str,
    /// Sequences swapped out under KV pressure.
    ///
    /// Always `0`, and stated rather than left `null`: the layout named in
    /// `kv_layout` is a contiguous per-slot allocation with no swap path, so
    /// no mechanism exists that could make this non-zero.
    pub preempted_swap: u64,
    /// Bytes one batched KV slot costs at the configured context length.
    pub kv_per_slot_bytes: Option<usize>,
    /// Batched KV slots currently allocated.
    pub kv_slots_allocated: Option<usize>,
    /// Hard ceiling on batched KV slots.
    pub kv_slots_max: Option<usize>,
    /// Always `null` here — see the type docs.
    pub kv_blocks_total: Option<usize>,
    /// The layout the numbers above describe.
    pub kv_layout: &'static str,
}

// ---------------------------------------------------------------------------
// The response
// ---------------------------------------------------------------------------

/// Body of `GET /v1/effective-config`.
///
/// The key set is IDENTICAL on every build. `cuda` is `null` on a build without
/// the `cuda` feature and an object on a build with it; it is carried as an
/// opaque [`serde_json::Value`] precisely so the presence of a key can never
/// depend on a `cfg!`.
#[derive(Debug, Clone, Serialize)]
pub struct EffectiveConfigResponse {
    /// Wire schema version of this body.
    pub schema_version: u32,
    /// REG-15 (PP-066 #2971): what the load-time parity gate measured for the loaded
    /// GPU model — never absent. `not-run` on a CPU-only server, `skipped` under the
    /// `SKIP_PARITY_GATE` override, `PASS` with the cosine when the gate admitted the model.
    pub parity: ParityReport,
    /// Identity and clock of the serving process.
    pub server: ServerReport,
    /// The dispatch path this process will take, from residency (PP-2).
    pub compute_class: &'static str,
    /// `realizar`'s own `cfg!` feature list.
    pub build_features: Vec<&'static str>,
    /// The launching CLI's `cfg!` feature list, when it supplied one (§9 #8).
    pub build_features_cli: Option<Vec<String>>,
    /// Backends actually resident, from `AppState` — never from `cfg!`.
    pub backend_loaded: Vec<&'static str>,
    /// What the loader measured about the model.
    pub model: ModelReport,
    /// Layer offload as the loader resolved it (PP-14/PP-15).
    pub offload: Option<OffloadReport>,
    /// Scheduler identity, window and admission ceiling (PP-13/PP-24).
    pub scheduler: Option<SchedulerReport>,
    /// CUDA-resolved configuration, or `null` on a non-CUDA build/model.
    pub cuda: Option<serde_json::Value>,
    /// KV-cache accounting, or `null` when no KV-owning backend is resident.
    pub kv: Option<KvReport>,
    /// `true` when a live field could not be read because the model lock was
    /// held (PMAT-073). The affected blocks are absent, never guessed.
    pub lock_contended: bool,
}

/// `realizar`'s own compile-time feature set.
///
/// Reported SEPARATELY from `compute_class`: this says what the binary CAN do,
/// `compute_class` says what it IS doing. A validator that only had one of the
/// two could not catch "reports `cuda` on a CPU build".
#[must_use]
pub fn build_features() -> Vec<&'static str> {
    let mut features: Vec<&'static str> = Vec::new();
    if cfg!(feature = "server") {
        features.push("server");
    }
    if cfg!(feature = "cli") {
        features.push("cli");
    }
    if cfg!(feature = "gpu") {
        features.push("gpu");
    }
    if cfg!(feature = "cuda") {
        features.push("cuda");
    }
    features
}

/// Backends resident in this `AppState`, in dispatch order.
///
/// Derived from what is actually loaded. `has_cuda_model()` and friends are the
/// same predicates the chat backend chain dispatches on, so this list cannot
/// disagree with the path a request will take.
#[must_use]
pub fn backend_loaded(state: &AppState) -> Vec<&'static str> {
    let mut loaded: Vec<&'static str> = Vec::new();
    #[cfg(feature = "cuda")]
    {
        if state.has_cuda_model()
            || state.safetensors_cuda_model().is_some()
            || state.apr_q4k_tx().is_some()
        {
            loaded.push("cuda");
        }
    }
    #[cfg(feature = "gpu")]
    if state.has_gpu_model() || state.has_cached_model() {
        loaded.push("wgpu");
    }
    if state.quantized_model().is_some() || state.apr_transformer().is_some() {
        loaded.push("cpu");
    }
    loaded
}

/// PP-2: the dispatch path this process will take, read from the process.
///
/// `cuda` when a CUDA backend is resident, `wgpu` when a wgpu one is,
/// `cpu` when only a CPU backend is, and `unknown` when nothing is loaded —
/// which is a fact, not a default.
#[must_use]
pub fn compute_class_from_residency(state: &AppState) -> &'static str {
    match backend_loaded(state).first() {
        Some(first) => first,
        None => "unknown",
    }
}

/// Build the whole body from an `AppState`.
#[must_use]
/// The `parity` block of `GET /v1/effective-config` (REG-15, PP-066 #2971).
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ParityReport {
    /// `PASS` | `skipped` | `not-run`
    pub status: String,
    /// The cosine the gate measured, when it ran.
    pub cosine: Option<f32>,
    /// Positions the gate compared (the load-time gate: one token).
    pub positions: usize,
    /// The threshold the verdict was judged against.
    pub threshold: f32,
    /// Where the threshold and the measurement come from.
    pub basis: String,
}

impl ParityReport {
    /// No GPU model is loaded, so no gate ran; the threshold reported is the gate's constant.
    #[must_use]
    pub fn not_run(why: &str) -> Self {
        Self {
            status: "not-run".into(),
            cosine: None,
            positions: 0,
            threshold: 0.98,
            basis: why.into(),
        }
    }
}

/// The loaded GPU model's gate record, or `not-run` when there is none.
#[cfg(feature = "cuda")]
fn parity_report(state: &AppState) -> ParityReport {
    let Some(model) = state.cuda_model() else {
        return ParityReport::not_run("no GPU model loaded (cpu residency)");
    };
    match model.try_read() {
        Ok(m) => ParityReport {
            status: m.parity.status.to_string(),
            cosine: m.parity.cosine,
            positions: m.parity.positions,
            threshold: m.parity.threshold,
            basis: m.parity.basis.to_string(),
        },
        Err(_) => {
            ParityReport::not_run("the GPU model lock is contended; the record is on the model")
        },
    }
}

/// A build without the `cuda` feature has no gate: `not-run`, with the reason.
#[cfg(not(feature = "cuda"))]
fn parity_report(_state: &AppState) -> ParityReport {
    ParityReport::not_run("cuda feature not compiled: no GPU gate exists in this build")
}

/// The body of `GET /v1/effective-config`, derived from residency and the loaded model —
/// never from `cfg!` (PP-2), and never without its `parity` block (REG-15).
pub fn effective_config(state: &AppState) -> EffectiveConfigResponse {
    let effective = state.effective_config_state();
    let cuda_snapshot = cuda_snapshot(state);
    EffectiveConfigResponse {
        schema_version: EFFECTIVE_CONFIG_SCHEMA_VERSION,
        parity: parity_report(state),
        server: {
            let mut server = effective.clock.report();
            if server.build_commit.is_none() {
                server.build_commit = effective
                    .offload
                    .as_ref()
                    .and_then(|o| o.build_commit.clone());
            }
            server
        },
        compute_class: compute_class_from_residency(state),
        build_features: build_features(),
        build_features_cli: effective.offload.as_ref().map(|o| o.build_features.clone()),
        backend_loaded: backend_loaded(state),
        model: ModelReport::from_state(state),
        offload: effective.offload.as_ref().map(|o| (**o).clone()),
        scheduler: effective.scheduler.as_ref().map(|report| {
            let mut report = (**report).clone();
            if let Some(counter) = effective.in_flight.as_ref() {
                report.in_flight_now = Some(counter.in_flight_now());
                report.peak_in_flight = Some(counter.peak_in_flight());
            }
            report
        }),
        cuda: cuda_snapshot.cuda,
        kv: cuda_snapshot.kv,
        lock_contended: cuda_snapshot.lock_contended,
    }
}

/// The live half of the body: everything that needs the model lock.
struct CudaSnapshot {
    cuda: Option<serde_json::Value>,
    kv: Option<KvReport>,
    lock_contended: bool,
}

#[cfg(not(feature = "cuda"))]
fn cuda_snapshot(_state: &AppState) -> CudaSnapshot {
    CudaSnapshot {
        cuda: None,
        kv: None,
        lock_contended: false,
    }
}

#[cfg(feature = "cuda")]
fn cuda_snapshot(state: &AppState) -> CudaSnapshot {
    let Some(lock) = state.cuda_model() else {
        return CudaSnapshot {
            cuda: None,
            kv: None,
            lock_contended: false,
        };
    };
    // PMAT-073: never join the scheduler's write queue for a metadata GET.
    let Ok(model) = lock.try_read() else {
        return CudaSnapshot {
            cuda: None,
            kv: None,
            lock_contended: true,
        };
    };
    let vram = model.vram_report();
    let kv = KvReport {
        bytes_used: Some(vram.kv_bytes_reserved),
        bytes_reserved: Some(vram.kv_bytes_reserved),
        admission_rejected: state.effective_config_state().admission.rejected(),
        admission_policy: ADMISSION_POLICY,
        preempted_swap: 0,
        kv_per_slot_bytes: Some(vram.kv_per_slot_bytes),
        kv_slots_allocated: Some(vram.kv_slots_allocated),
        kv_slots_max: Some(vram.kv_slots_max),
        kv_blocks_total: None,
        kv_layout: vram.kv_layout,
    };
    let block = serde_json::json!({
        "gpu_profile": model.executor().gpu_profile(),
        "graphs": model.executor().graph_config(),
        "prefill_path": model.prefill_path(),
        "max_batch": model.max_batch_sizing(),
        "vram": vram,
    });
    CudaSnapshot {
        cuda: Some(block),
        kv: Some(kv),
        lock_contended: false,
    }
}

/// `GET /v1/effective-config` (§12 row 6 / PP-2).
pub(crate) async fn effective_config_handler(
    State(state): State<AppState>,
) -> Json<EffectiveConfigResponse> {
    if state.is_verbose() {
        eprintln!("[VERBOSE] GET /v1/effective-config");
    }
    Json(effective_config(&state))
}

// ---------------------------------------------------------------------------
// Shared per-server state carried on AppState
// ---------------------------------------------------------------------------

/// Everything `/v1/effective-config` reports that is not already on `AppState`.
///
/// One field on `AppState` rather than four, so adding a reported fact does not
/// mean editing sixteen struct literals again.
#[derive(Clone)]
pub struct EffectiveConfigState {
    pub(crate) clock: Arc<ServerClock>,
    pub(crate) offload: Option<Arc<OffloadReport>>,
    pub(crate) scheduler: Option<Arc<SchedulerReport>>,
    pub(crate) in_flight: Option<Arc<InFlightCounter>>,
    /// Refused admissions. NOT an `Option`: unlike `in_flight`, which a
    /// scheduler may or may not instrument, every path that can refuse a
    /// request reports here, so the count is always a measurement.
    pub(crate) admission: Arc<AdmissionCounter>,
}

impl EffectiveConfigState {
    /// Attach this `AppState` to the process clock.
    #[must_use]
    pub fn new() -> Self {
        Self {
            clock: ServerClock::process(),
            offload: None,
            scheduler: None,
            in_flight: None,
            admission: AdmissionCounter::new(),
        }
    }
}

impl Default for EffectiveConfigState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod effective_config_tests {
    use super::*;

    fn report(autofit: &[&str], explicit: &[&str]) -> OffloadReport {
        OffloadReport {
            gpu_layers_requested: "all".to_string(),
            gpu_layers_resolved: 28,
            gpu_layers_total: 28,
            offload_policy: "all_or_nothing",
            autofit_applied: autofit.iter().map(|s| (*s).to_string()).collect(),
            explicit_args: explicit.iter().map(|s| (*s).to_string()).collect(),
            build_features: vec!["cuda".to_string()],
            build_commit: None,
        }
    }

    /// PP-14 must-fire: a run that says auto-fit chose the very argument the
    /// operator pinned cannot be reproduced from its own receipt.
    /// REG-15 (#2971): the parity block is never absent and always carries the five keys.
    #[test]
    fn parity_report_carries_the_five_keys_when_no_gate_ran() {
        let v = serde_json::to_value(ParityReport::not_run("test")).expect("serialises");
        let mut keys: Vec<&str> = v
            .as_object()
            .expect("object")
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            ["basis", "cosine", "positions", "status", "threshold"]
        );
        assert_eq!(v["status"], "not-run");
        assert!(v["cosine"].is_null());
    }

    #[test]
    fn autofit_override() {
        assert!(
            !pp14_holds(&report(&["gpu_layers"], &["gpu_layers", "context_length"])),
            "auto-fit and the operator both claiming `gpu_layers` must be REFUSED"
        );
    }

    /// PP-14 must-not-fire: disjoint sets hold, including both empty.
    #[test]
    fn autofit_ok() {
        assert!(pp14_holds(&report(&[], &["gpu_layers"])));
        assert!(pp14_holds(&report(&["context_length"], &["gpu_layers"])));
        assert!(pp14_holds(&report(&[], &[])));
    }

    /// PP-24: each source maps to its own token, and an unknown source is
    /// `"default"` rather than a guess at one of the other two.
    #[test]
    fn admission_ceiling_reason_table() {
        assert_eq!(admission_ceiling_reason(Some("env")), "env");
        assert_eq!(admission_ceiling_reason(Some("computed")), "kv_budget");
        assert_eq!(admission_ceiling_reason(None), "default");
        assert_eq!(admission_ceiling_reason(Some("something-else")), "default");
        assert_ne!(
            admission_ceiling_reason(Some("env")),
            admission_ceiling_reason(Some("computed")),
            "an operator ceiling and a KV-budget ceiling must not read the same"
        );
    }

    /// The counter's peak is a high-water mark, not the current value, and
    /// `leave` must not wrap below zero.
    #[test]
    /// `set` mirrors the scheduler's live slot count: the peak follows the
    /// highest value ever set, `now` follows the last one, and a set to zero
    /// (the batch ended) leaves the peak standing.
    #[test]
    fn in_flight_counter_set_mirrors_the_scheduler_and_keeps_the_peak() {
        let c = InFlightCounter::new();
        c.set(3);
        c.set(5);
        c.set(2);
        assert_eq!(c.in_flight_now(), 2);
        assert_eq!(c.peak_in_flight(), 5);
        c.set(0);
        assert_eq!(c.in_flight_now(), 0);
        assert_eq!(
            c.peak_in_flight(),
            5,
            "the peak is what the band achieved, not what is left"
        );
    }

    fn in_flight_counter_tracks_peak_and_saturates() {
        let counter = InFlightCounter::new();
        assert_eq!(counter.enter(), 1);
        assert_eq!(counter.enter(), 2);
        assert_eq!(counter.enter(), 3);
        counter.leave();
        counter.leave();
        assert_eq!(counter.in_flight_now(), 1);
        assert_eq!(counter.peak_in_flight(), 3, "peak is the high-water mark");
        counter.leave();
        counter.leave();
        assert_eq!(
            counter.in_flight_now(),
            0,
            "an unmatched leave must saturate, not wrap to usize::MAX"
        );
        assert_eq!(counter.peak_in_flight(), 3);
    }

    fn kv_fixture(admission_rejected: u64) -> KvReport {
        KvReport {
            bytes_used: Some(2_348_810_240),
            bytes_reserved: Some(2_348_810_240),
            admission_rejected,
            admission_policy: ADMISSION_POLICY,
            preempted_swap: 0,
            kv_per_slot_bytes: Some(469_762_048),
            kv_slots_allocated: Some(4),
            kv_slots_max: Some(32),
            kv_blocks_total: None,
            kv_layout: "contiguous_per_slot",
        }
    }

    /// Arm D's memory block is built ONLY when the server reported all FOUR
    /// figures — `KvBlock::from_server_report(bytes_used, bytes_reserved,
    /// admission_rejected, preempted_swap)`, each read with `as_u64(...)?`.
    ///
    /// This is the must-fire the endpoint used to fail: two of the four were
    /// `null`, so the `?` short-circuited and every receipt this endpoint fed
    /// carried no `kv` block at all. A reader could not tell that from a server
    /// predating the field. The assertion is the harness's own extraction, run
    /// against the serialized body.
    #[test]
    fn kv_block_reports_all_four_numbers_arm_d_reads() {
        let json = serde_json::to_value(kv_fixture(0)).expect("serialize");
        for field in [
            "bytes_used",
            "bytes_reserved",
            "admission_rejected",
            "preempted_swap",
        ] {
            assert!(
                json[field].as_u64().is_some(),
                "`kv.{field}` must be a number: Arm D's block is built only when \
                 both byte figures parse; a null counter is kept and named, a null byte figure DELETES the block:\n{json}"
            );
        }
        // ...and each number arrives with the reason a reader needs to act on it.
        assert_eq!(
            json["admission_policy"].as_str(),
            Some("queue"),
            "`admission_rejected: 0` is only readable beside the policy it counts \
             under:\n{json}"
        );
        assert_eq!(
            json["kv_layout"].as_str(),
            Some("contiguous_per_slot"),
            "`preempted_swap: 0` is only readable beside the layout that has no \
             swap path:\n{json}"
        );
        // The one field that genuinely denotes nothing stays null.
        assert!(
            json["kv_blocks_total"].is_null(),
            "there is no block table to count:\n{json}"
        );
    }

    /// A counted zero and a counted non-zero must not read the same, or the
    /// figure is a constant wearing a counter's name.
    #[test]
    fn admission_rejected_reports_what_was_counted() {
        let none = serde_json::to_value(kv_fixture(0)).expect("serialize");
        let some = serde_json::to_value(kv_fixture(3)).expect("serialize");
        assert_eq!(none["admission_rejected"].as_u64(), Some(0));
        assert_eq!(some["admission_rejected"].as_u64(), Some(3));
        assert_ne!(
            none["admission_rejected"], some["admission_rejected"],
            "a server that refused three requests must not report what an idle one does"
        );
    }

    /// The counter counts refusals, and the running total it returns is the
    /// total after this one — not the total before it.
    #[test]
    fn admission_counter_counts_each_refusal() {
        let counter = AdmissionCounter::new();
        assert_eq!(counter.rejected(), 0, "nothing refused yet");
        assert_eq!(counter.record_rejected(), 1, "the total AFTER this refusal");
        assert_eq!(counter.record_rejected(), 2);
        assert_eq!(counter.rejected(), 2);
        // Shared through the Arc the handlers clone with `AppState`.
        let shared = Arc::clone(&counter);
        shared.record_rejected();
        assert_eq!(
            counter.rejected(),
            3,
            "a clone of the handle must count into the same total"
        );
    }

    /// The clock is one clock: two handles in one process report the same
    /// start, which is what makes `/v1/models`' `created` stable across
    /// requests (FALSIFY-CRUX-C-33-004).
    #[test]
    fn process_clock_is_shared() {
        let a = ServerClock::process();
        let b = ServerClock::process();
        assert_eq!(a.started_utc(), b.started_utc());
        assert_eq!(a.started_unix_secs(), b.started_unix_secs());
        assert_eq!(a.pid(), std::process::id());
    }

    /// PP-30: the timestamp must be RFC 3339 UTC and parse as such.
    #[test]
    fn started_utc_is_rfc3339_utc() {
        let clock = ServerClock::new();
        let parsed = chrono::DateTime::parse_from_rfc3339(clock.started_utc())
            .expect("started_utc must be RFC 3339");
        assert_eq!(parsed.offset().local_minus_utc(), 0, "must be UTC");
        assert!(
            clock.started_utc().ends_with('Z'),
            "UTC is written with Z, got {}",
            clock.started_utc()
        );
        assert_eq!(parsed.timestamp(), clock.started_unix_secs());
    }

    /// Uptime advances and never goes backwards.
    #[test]
    fn uptime_is_monotonic() {
        let clock = ServerClock::new();
        let first = clock.uptime_sec();
        let second = clock.uptime_sec();
        assert!(second >= first, "{second} < {first}");
        assert!(first >= 0.0);
    }

    /// `build_features` states what the binary CAN do. It must contain `cuda`
    /// exactly when the crate was built with it — the cross-check a validator
    /// runs against `compute_class`.
    #[test]
    fn build_features_agree_with_cfg() {
        let features = build_features();
        assert_eq!(features.contains(&"server"), cfg!(feature = "server"));
        assert_eq!(features.contains(&"cli"), cfg!(feature = "cli"));
        assert_eq!(features.contains(&"gpu"), cfg!(feature = "gpu"));
        assert_eq!(features.contains(&"cuda"), cfg!(feature = "cuda"));
        assert!(
            features.len() <= 4,
            "a feature reported that this crate does not have: {features:?}"
        );
    }

    /// PP-2 must-fire, as a rule a validator can run: `compute_class` must be
    /// producible by the build. The endpoint reports both halves so this
    /// comparison is possible at all — before it, `compute_class` was an
    /// operator-typed CLI argument with nothing to check it against.
    #[test]
    fn a_cuda_class_is_only_valid_on_a_cuda_build() {
        let features = build_features();
        let valid = |class: &str| class != "cuda" || features.contains(&"cuda");
        assert!(valid("cpu"), "cpu is producible by every build");
        assert!(valid("unknown"));
        assert_eq!(
            valid("cuda"),
            cfg!(feature = "cuda"),
            "`compute_class: cuda` on a build without the cuda feature is \
             INVALID-BUILD, and this is the comparison that says so"
        );
    }
}
