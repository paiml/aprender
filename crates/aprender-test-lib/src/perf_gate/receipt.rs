//! §4.4 — the receipt, and the only writer of one (PERF-025).
//!
//! # The defect this closes
//!
//! PERF-024 built the §4.4 protocol and stopped one step short, saying so in its
//! own module doc: *"Receipt emission. Nothing here writes `receipt.json`."*
//! The consequence was that a conformant band could be **run** and could not be
//! **reported**: `scripts/perf_gate.sh`'s real mode
//! (`--host/--phase/--workload/--receipt`) had no caller anywhere in the repo,
//! because nothing in the repo could produce the file it takes. A protocol whose
//! output no gate can read is the same defect as a gate no protocol can feed.
//!
//! # Two consumers, one file
//!
//! The schema is not invented here. It is the **intersection of two existing
//! readers**, and every field below exists because one of them reads it:
//!
//! | Reader | Requires |
//! |---|---|
//! | `scripts/lib/bench_receipt.py` (the schema authority) | `provenance.{binary_path,binary_sha256,resolution,compute_class}`, join key `{host,accelerator,model,quantization}`, `samples_ms` non-empty **and not constant**, `n == len(samples_ms)` |
//! | `scripts/perf_gate.sh` Arm C | `requested == completed`, `timeouts == 0`, `tokenization.method`, `drain_ms` non-null, no band with `tokens_total == 0` |
//! | `scripts/perf_gate.sh` Arm A | `bands[].concurrency`, `bands[].aggregate_tok_per_sec`, band `c=1` present |
//! | `scripts/perf_gate.sh` Arm B | `bands[].agg_ratio`/`decode_ratio`, **or** `comparator_status` marking the cell |
//!
//! Arms D and E read `kv.*`, `itl.p95_w*_ms` and `injector.*`, which are
//! **server**-reported (§4.4.9) or belong to the W2 injector. A client cannot
//! synthesise them, so this writer emits neither rather than emitting a
//! plausible-looking null. They are `REPORT` at merge phase and only fatal at
//! release, which is the correct place for that conversation.
//!
//! # `feature_set` describes the SERVER, and is therefore not defaulted
//!
//! `bench_receipt.py` rejects `compute_class = cuda` when `feature_set` does not
//! contain `cuda` — "a build without the feature cannot take that path". That
//! check is about the **binary that decoded the tokens**, which for an HTTP
//! measurement is the *server*, not this client. Filling `feature_set` in from
//! the client's own `cfg!` flags would make the one check that catches the
//! fabricated-14x class compare the wrong build, and it would read green.
//! So it is [`Option`], supplied by the caller who knows how the server was
//! built, and omitted otherwise — an absent check that says it is absent beats
//! a present check pointed at the wrong thing.

use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::bootstrap::BootstrapCi;
use super::metrics::{percentile, BandMetrics, RequestSample};
use super::protocol::{
    ClientModel, Tokenization, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, MIN_WALL_CLOCK, QUIESCE,
    REPLICATES, REQUEST_TIMEOUT, WARMUP_MULTIPLIER,
};
use super::samples::{write_samples_gz, SamplesFile};
use super::window::WindowReport;

/// The spec version this receipt claims conformance to.
pub const SPEC: &str = "APR-PERF-GATE-001-v2.2";

/// `bench_receipt.py`'s `COMPUTE_CLASSES`, mirrored so a bad value is refused
/// where the operator typed it rather than by a Python traceback minutes later.
pub const COMPUTE_CLASSES: [&str; 5] = ["cpu", "cuda", "metal", "wgpu", "unknown"];

/// §4.4.8 — a cell with no comparator lane measured. Arm B reads this and
/// degrades to `REPORT`; Arm A still gates the cell.
pub const COMPARATOR_UNMEASURED: &str = "UNMEASURED";

/// `sha256` of a file's bytes, lowercase hex.
///
/// Used for `provenance.binary_sha256`, which `bench_receipt.py` requires to be
/// a 64-character lowercase hex digest. A receipt naming a binary it did not
/// hash is an anonymous number.
///
/// # Errors
/// On any read failure.
pub fn sha256_file(path: &Path) -> io::Result<String> {
    let bytes = std::fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Which binary ran, which dispatch path it took, and where.
///
/// The last four fields are `bench_receipt.py`'s `JOIN_KEY_REQUIRED`: without
/// them a receipt from one host on one model is structurally comparable with a
/// receipt from another host on another, which is how a 0.5B result on gx10 gets
/// compared with a 7B result on lambda.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    /// Absolute path of the measuring binary.
    pub binary_path: String,
    /// `sha256` of that binary, 64 lowercase hex characters.
    pub binary_sha256: String,
    /// How the binary was resolved, e.g. `current_exe` or `apr_bin.sh`.
    pub resolution: String,
    /// The dispatch path **taken**, not the hardware present.
    pub compute_class: String,
    /// Join key: which machine.
    pub host: String,
    /// Join key: which accelerator.
    pub accelerator: String,
    /// Join key: which model.
    pub model: String,
    /// Join key: which quantization.
    pub quantization: String,
    /// The **server's** build features. See the module doc: never defaulted
    /// from the client's own `cfg!`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feature_set: Option<Vec<String>>,
}

impl Provenance {
    /// Reject what `bench_receipt.py` would reject, at the point the operator
    /// typed it.
    ///
    /// # Errors
    /// When the digest is malformed, the compute class is unknown, or any join
    /// key field is empty.
    pub fn validate(&self) -> Result<(), String> {
        if self.binary_sha256.len() != 64
            || !self
                .binary_sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err(format!(
                "provenance.binary_sha256 must be 64 lowercase hex characters, got {:?}",
                self.binary_sha256
            ));
        }
        if !COMPUTE_CLASSES.contains(&self.compute_class.as_str()) {
            return Err(format!(
                "provenance.compute_class {:?} not in {COMPUTE_CLASSES:?}",
                self.compute_class
            ));
        }
        self.validate_join_key()?;
        self.validate_feature_set()
    }

    fn validate_join_key(&self) -> Result<(), String> {
        for (name, value) in [
            ("host", &self.host),
            ("accelerator", &self.accelerator),
            ("model", &self.model),
            ("quantization", &self.quantization),
        ] {
            if value.trim().is_empty() {
                return Err(format!(
                    "provenance.{name} is empty — a receipt that does not say WHERE and on WHAT \
                     cannot be compared to another"
                ));
            }
        }
        Ok(())
    }

    /// The `bench_receipt.py` rule, applied early: a class the declared build
    /// cannot reach is a fabricated claim, not a measurement.
    fn validate_feature_set(&self) -> Result<(), String> {
        let Some(features) = self.feature_set.as_ref() else {
            return Ok(());
        };
        let gated = self.compute_class == "cuda" || self.compute_class == "wgpu";
        if gated && !features.iter().any(|f| f == &self.compute_class) {
            return Err(format!(
                "provenance.compute_class={} but feature_set={features:?} does not contain it — \
                 a build without the feature cannot take that path",
                self.compute_class
            ));
        }
        Ok(())
    }
}

/// §4.4.4 interval, in a form a receipt can be read back out of.
///
/// [`BootstrapCi::resampling_unit`] is `&'static str`, which makes its derived
/// `Deserialize<'de>` require `'de: 'static`. A receipt embedding that type
/// verbatim can be **written and never read back** — and a receipt that cannot
/// be re-read is a receipt whose interval cannot be checked. Mirrored with
/// owned fields rather than edited upstream, so the round-trip is a property of
/// this schema and not of a sibling ticket's type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CiBlock {
    /// The statistic on the observed sample.
    pub point: f64,
    /// Lower percentile bound.
    pub lower: f64,
    /// Upper percentile bound.
    pub upper: f64,
    /// Nominal coverage, e.g. 0.95.
    pub confidence: f64,
    /// Resamples drawn.
    pub resamples: usize,
    /// The seed (§4.4.4).
    pub seed: u64,
    /// Whole requests, always.
    pub resampling_unit: String,
    /// Observations resampled.
    pub n: usize,
}

impl From<&BootstrapCi> for CiBlock {
    fn from(ci: &BootstrapCi) -> Self {
        Self {
            point: ci.point,
            lower: ci.lower,
            upper: ci.upper,
            confidence: ci.confidence,
            resamples: ci.resamples,
            seed: ci.seed,
            resampling_unit: ci.resampling_unit.to_string(),
            n: ci.n,
        }
    }
}

/// One replicate of one band, as produced by the §4.4 protocol.
///
/// Owned rather than borrowed: a receipt is written once at the end of a run
/// that took minutes, so the copy is free and the lifetimes are not.
#[derive(Debug, Clone)]
pub struct Replicate {
    /// §4.4.3 metrics for this replicate.
    pub metrics: BandMetrics,
    /// §4.4.2/§4.4.7 window and drain accounting.
    pub window: WindowReport,
    /// §4.4.5 raw per-request samples.
    pub samples: Vec<RequestSample>,
    /// §4.4.4 interval on `agg_tok_s`.
    pub agg_ci: Option<BootstrapCi>,
    /// Every departure from §4.4 this replicate carries.
    pub protocol_violations: Vec<String>,
}

/// One replicate as it appears in the receipt, with its retained samples named.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplicateReceipt {
    /// Zero-based replicate index within the cell.
    pub replicate: usize,
    /// §4.4.3 wall-clock aggregate.
    pub aggregate_tok_per_sec: f64,
    /// §4.4.3 median per-request decode rate.
    pub decode_tok_per_sec: f64,
    /// Tokens in `aggregate_tok_per_sec`'s numerator.
    pub tokens_total: u64,
    /// The denominator, so a reader can check the division.
    pub span_s: f64,
    /// Requests admitted inside the window.
    pub requested: usize,
    /// Requests that completed.
    pub completed: usize,
    /// Requests that hit the 120 s hard timeout.
    pub timeouts: usize,
    /// Requests abandoned at the drain deadline (§4.4.7).
    pub truncated: usize,
    /// Requests that failed for any other reason.
    pub errors: usize,
    /// `T` minus window open.
    pub window_ms: f64,
    /// §4.4.7 — last drained completion minus `T`.
    pub drain_ms: f64,
    /// Peak concurrency **the client** observed. Never the server's `max_in_flight`.
    pub client_peak_in_flight: usize,
    /// §4.4.5 — where the raw samples went, and what they hash to.
    pub samples_file: SamplesFile,
    /// §4.4.4 interval, `None` when fewer than two samples survived.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agg_ci: Option<CiBlock>,
    /// Departures from §4.4, empty when conformant.
    pub protocol_violations: Vec<String>,
}

/// One band (one concurrency level) across its replicates.
///
/// Headline rates are the **median across replicates**; counts are **sums**;
/// `drain_ms` is the **max**. Each rule is the conservative one for the quantity
/// it summarises, and every replicate's own value survives in `replicates` so a
/// reader is never forced to trust the summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BandReceipt {
    /// Fixed concurrency `c`.
    pub concurrency: usize,
    /// §4.4.3 wall-clock aggregate, median across replicates. Arm A reads this.
    pub aggregate_tok_per_sec: f64,
    /// Median across replicates of the per-replicate median decode rate.
    pub decode_tok_per_sec: f64,
    /// Sum across replicates. Arm C fails a band whose value is zero.
    pub tokens_total: u64,
    /// p50 time to first token, median across replicates.
    pub ttft_p50_ms: f64,
    /// p95 time to first token, median across replicates.
    pub ttft_p95_ms: f64,
    /// p50 of pooled inter-token gaps, median across replicates.
    pub itl_p50_ms: f64,
    /// p95 of pooled inter-token gaps, median across replicates.
    pub itl_p95_ms: f64,
    /// Sum across replicates.
    pub requested: usize,
    /// Sum across replicates.
    pub completed: usize,
    /// Sum across replicates.
    pub timeouts: usize,
    /// Sum across replicates.
    pub truncated: usize,
    /// Sum across replicates.
    pub errors: usize,
    /// Max across replicates (§4.4.7).
    pub drain_ms: f64,
    /// §4.4.8 — `UNMEASURED` until a comparator lane runs. Arm B reads this and
    /// degrades to `REPORT` rather than inventing a ratio.
    pub comparator_status: String,
    /// Arm B1. Absent without a comparator; a ratio is never synthesised.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agg_ratio: Option<f64>,
    /// Arm B2. Absent without a comparator.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_ratio: Option<f64>,
    /// Every replicate, unsummarised.
    pub replicates: Vec<ReplicateReceipt>,
    /// The union of every replicate's departures, plus the cell's own.
    pub protocol_violations: Vec<String>,
}

/// §4.4.2/§4.4.3 protocol parameters, recorded so a shrunken run is visible.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProtocolBlock {
    /// §4.4.2 `2 × c`.
    pub warmup_multiplier: usize,
    /// §4.4.2 quiesce, seconds.
    pub quiesce_s: f64,
    /// §4.4.2 minimum wall-clock per band, seconds.
    pub min_wall_clock_s: f64,
    /// §4.4.3 hard per-request timeout, seconds.
    pub request_timeout_s: f64,
    /// §4.4.2 replicates this run actually performed.
    pub replicates: usize,
    /// §4.4.2 replicates the spec requires.
    pub replicates_required: usize,
}

impl ProtocolBlock {
    /// The block for a run that performed `replicates` replicates per cell.
    #[must_use]
    pub fn new(replicates: usize) -> Self {
        Self {
            warmup_multiplier: WARMUP_MULTIPLIER,
            quiesce_s: QUIESCE.as_secs_f64(),
            min_wall_clock_s: MIN_WALL_CLOCK.as_secs_f64(),
            request_timeout_s: REQUEST_TIMEOUT.as_secs_f64(),
            replicates,
            replicates_required: REPLICATES,
        }
    }
}

/// §4.4.4 — everything needed to re-derive the interval from the retained samples.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BootstrapBlock {
    /// Always `percentile` (§4.4.4 excludes BCa).
    pub method: String,
    /// §4.4.4 resamples.
    pub resamples: usize,
    /// §4.4.4 seed.
    pub seed: u64,
    /// Whole requests; tokens within a request are not independent.
    pub resampling_unit: String,
}

impl Default for BootstrapBlock {
    fn default() -> Self {
        Self {
            method: "percentile".to_string(),
            resamples: BOOTSTRAP_RESAMPLES,
            seed: BOOTSTRAP_SEED,
            resampling_unit: "whole_requests".to_string(),
        }
    }
}

/// The receipt. One file, two readers, no second definition of the schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Receipt {
    /// The spec version claimed.
    pub spec: String,
    /// §4.3 workload identifier, `W1` or `W2`.
    pub workload: String,
    /// Which binary, which path, where.
    pub provenance: Provenance,
    /// §4.4.6, required, no default.
    pub tokenization: Tokenization,
    /// §4.4.1 — recorded, not assumed.
    pub client_model: ClientModel,
    /// §4.4.2/§4.4.3 parameters.
    pub protocol: ProtocolBlock,
    /// §4.4.4 parameters.
    pub bootstrap: BootstrapBlock,
    /// Arm C: total requests admitted across every band and replicate.
    pub requested: usize,
    /// Arm C: total completed. Must equal `requested` or the gate fails.
    pub completed: usize,
    /// Arm C: must be zero.
    pub timeouts: usize,
    /// Requests abandoned at a drain deadline.
    pub truncated: usize,
    /// Requests that failed for any other reason.
    pub errors: usize,
    /// Arm C: must be present. Max across every replicate's `WindowReport`.
    pub drain_ms: f64,
    /// `bench_receipt.py`: raw per-request latencies, pooled. Non-empty, and a
    /// real distribution rather than a constant.
    pub samples_ms: Vec<f64>,
    /// `bench_receipt.py`: must equal `samples_ms.len()`.
    pub n: usize,
    /// Arms A and B read this. Band `c=1` must be present for Arm A.
    pub bands: Vec<BandReceipt>,
    /// True when no band carried a departure from §4.4.
    pub conformant: bool,
    /// The union of every band's departures.
    pub protocol_violations: Vec<String>,
    /// Commit under measurement, for the release staleness arm.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
}

impl Receipt {
    /// True when Arm C's counting rules hold. Not a substitute for running the
    /// gate: it is the same predicate, checked early, so a run that cannot
    /// possibly pass says so before the operator waits for CI.
    #[must_use]
    pub fn arm_c_would_pass(&self) -> bool {
        self.requested == self.completed
            && self.timeouts == 0
            && self.bands.iter().all(|b| b.tokens_total > 0)
            && !self.samples_ms.is_empty()
    }
}

/// Where a written receipt lives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WrittenReceipt {
    /// The `receipt.json` path.
    pub receipt: PathBuf,
    /// Every gzipped JSONL sample file written beside it.
    pub sample_files: Vec<PathBuf>,
    /// Size of `receipt.json` in bytes.
    pub bytes: u64,
}

fn median_of(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    percentile(&v, 0.50).unwrap_or(0.0)
}

fn pick<F: Fn(&ReplicateReceipt) -> f64>(reps: &[ReplicateReceipt], f: F) -> f64 {
    median_of(&reps.iter().map(f).collect::<Vec<_>>())
}

/// §4.4.5 — write one replicate's raw samples beside the receipt.
fn retain_samples(
    dir: &Path,
    concurrency: usize,
    index: usize,
    rep: &Replicate,
) -> io::Result<SamplesFile> {
    let name = format!("samples-c{concurrency}-r{index}.jsonl.gz");
    write_samples_gz(&dir.join(name), &rep.samples)
}

fn replicate_receipt(index: usize, rep: &Replicate, samples_file: SamplesFile) -> ReplicateReceipt {
    let m = &rep.metrics;
    ReplicateReceipt {
        replicate: index,
        aggregate_tok_per_sec: m.agg_tok_s,
        decode_tok_per_sec: m.decode_tok_s,
        tokens_total: m.tokens_total,
        span_s: m.span_s,
        requested: m.requested,
        completed: m.completed,
        timeouts: m.timeouts,
        truncated: m.truncated,
        errors: m.errors,
        window_ms: rep.window.window_ms,
        drain_ms: rep.window.drain_ms,
        client_peak_in_flight: rep.window.client_peak_in_flight,
        samples_file,
        agg_ci: rep.agg_ci.as_ref().map(CiBlock::from),
        protocol_violations: rep.protocol_violations.clone(),
    }
}

/// Median rates, summed counts, max drain — see [`BandReceipt`].
fn fold_band(concurrency: usize, reps: Vec<ReplicateReceipt>) -> BandReceipt {
    let violations: Vec<String> = reps
        .iter()
        .flat_map(|r| r.protocol_violations.iter().cloned())
        .collect();
    BandReceipt {
        concurrency,
        aggregate_tok_per_sec: pick(&reps, |r| r.aggregate_tok_per_sec),
        decode_tok_per_sec: pick(&reps, |r| r.decode_tok_per_sec),
        tokens_total: reps.iter().map(|r| r.tokens_total).sum(),
        ttft_p50_ms: 0.0,
        ttft_p95_ms: 0.0,
        itl_p50_ms: 0.0,
        itl_p95_ms: 0.0,
        requested: reps.iter().map(|r| r.requested).sum(),
        completed: reps.iter().map(|r| r.completed).sum(),
        timeouts: reps.iter().map(|r| r.timeouts).sum(),
        truncated: reps.iter().map(|r| r.truncated).sum(),
        errors: reps.iter().map(|r| r.errors).sum(),
        drain_ms: reps.iter().map(|r| r.drain_ms).fold(0.0, f64::max),
        comparator_status: COMPARATOR_UNMEASURED.to_string(),
        agg_ratio: None,
        decode_ratio: None,
        replicates: reps,
        protocol_violations: violations,
    }
}

/// Fill the latency percentiles, which live on [`BandMetrics`] rather than on
/// the per-replicate receipt row.
fn with_latencies(mut band: BandReceipt, replicates: &[Replicate]) -> BandReceipt {
    let m: Vec<&BandMetrics> = replicates.iter().map(|r| &r.metrics).collect();
    let med = |f: fn(&BandMetrics) -> f64| median_of(&m.iter().map(|x| f(x)).collect::<Vec<_>>());
    band.ttft_p50_ms = med(|x| x.ttft_p50_ms);
    band.ttft_p95_ms = med(|x| x.ttft_p95_ms);
    band.itl_p50_ms = med(|x| x.itl_p50_ms);
    band.itl_p95_ms = med(|x| x.itl_p95_ms);
    band
}

/// Build one band's receipt row, retaining every replicate's raw samples.
///
/// # Errors
/// On any sample-retention failure. A retention failure is fatal, never
/// swallowed: a receipt whose samples silently failed to write is exactly the
/// summary-only receipt §4.4.5 rejects.
pub fn build_band(
    dir: &Path,
    concurrency: usize,
    replicates: &[Replicate],
) -> io::Result<(BandReceipt, Vec<PathBuf>)> {
    let mut rows = Vec::with_capacity(replicates.len());
    let mut paths = Vec::with_capacity(replicates.len());
    for (i, rep) in replicates.iter().enumerate() {
        let file = retain_samples(dir, concurrency, i, rep)?;
        paths.push(dir.join(&file.path));
        rows.push(replicate_receipt(i, rep, file));
    }
    Ok((
        with_latencies(fold_band(concurrency, rows), replicates),
        paths,
    ))
}

/// Per-request latencies in milliseconds, for `bench_receipt.py`'s `samples_ms`.
fn latencies_ms(replicates: &[Replicate]) -> Vec<f64> {
    replicates
        .iter()
        .flat_map(|r| r.samples.iter())
        .map(|s| (s.end_s - s.start_s) * 1000.0)
        .collect()
}

/// Everything a receipt needs that the protocol cannot know.
#[derive(Debug, Clone)]
pub struct ReceiptMeta {
    /// §4.3 workload identifier.
    pub workload: String,
    /// Which binary, which path, where.
    pub provenance: Provenance,
    /// §4.4.6, required.
    pub tokenization: Tokenization,
    /// Replicates performed per cell.
    pub replicates: usize,
    /// Commit under measurement.
    pub commit: Option<String>,
}

fn totals(bands: &[BandReceipt]) -> (usize, usize, usize, usize, usize) {
    (
        bands.iter().map(|b| b.requested).sum(),
        bands.iter().map(|b| b.completed).sum(),
        bands.iter().map(|b| b.timeouts).sum(),
        bands.iter().map(|b| b.truncated).sum(),
        bands.iter().map(|b| b.errors).sum(),
    )
}

/// §4.4.2 — a run that performed fewer than `REPLICATES` replicates per cell is
/// not conformant, and says so rather than looking identical to one that was.
fn replicate_violation(replicates: usize) -> Vec<String> {
    if replicates < REPLICATES {
        vec![format!(
            "§4.4.2 replicates={replicates} < N={REPLICATES} required per cell"
        )]
    } else {
        Vec::new()
    }
}

/// Assemble the receipt from finished bands.
#[must_use]
pub fn assemble(
    meta: &ReceiptMeta,
    bands: Vec<BandReceipt>,
    samples_ms: Vec<f64>,
    drain_ms: f64,
) -> Receipt {
    let (requested, completed, timeouts, truncated, errors) = totals(&bands);
    let mut violations = replicate_violation(meta.replicates);
    violations.extend(
        bands
            .iter()
            .flat_map(|b| b.protocol_violations.iter().cloned()),
    );
    Receipt {
        spec: SPEC.to_string(),
        workload: meta.workload.clone(),
        provenance: meta.provenance.clone(),
        tokenization: meta.tokenization.clone(),
        client_model: ClientModel::ClosedLoop,
        protocol: ProtocolBlock::new(meta.replicates),
        bootstrap: BootstrapBlock::default(),
        requested,
        completed,
        timeouts,
        truncated,
        errors,
        drain_ms,
        n: samples_ms.len(),
        samples_ms,
        bands,
        conformant: violations.is_empty(),
        protocol_violations: violations,
        commit: meta.commit.clone(),
    }
}

/// Write `receipt.json` and every band's gzipped JSONL samples into `dir`.
///
/// `cells` is `(concurrency, replicates)` in band order. Arm A requires band
/// `c = 1` to be present, and this writer does not silently supply it: a
/// receipt missing `c = 1` fails the gate, which is the correct outcome.
///
/// # Errors
/// When provenance or the §4.4.6 block does not validate, or on any I/O
/// failure. Validation happens **before** the first byte is written, so a
/// rejected run does not leave a half-written receipt behind.
pub fn write_receipt(
    dir: &Path,
    meta: &ReceiptMeta,
    cells: &[(usize, Vec<Replicate>)],
) -> io::Result<(Receipt, WrittenReceipt)> {
    let invalid = |e: String| io::Error::new(io::ErrorKind::InvalidInput, e);
    meta.provenance.validate().map_err(invalid)?;
    meta.tokenization.validate().map_err(invalid)?;
    if cells.is_empty() {
        return Err(invalid(
            "no bands were measured — an empty receipt is a vacuous pass, not a fast run".into(),
        ));
    }
    std::fs::create_dir_all(dir)?;

    let mut bands = Vec::with_capacity(cells.len());
    let mut sample_files = Vec::new();
    let mut samples_ms = Vec::new();
    let mut drain_ms = 0.0_f64;
    for (concurrency, replicates) in cells {
        let (band, paths) = build_band(dir, *concurrency, replicates)?;
        drain_ms = drain_ms.max(band.drain_ms);
        samples_ms.extend(latencies_ms(replicates));
        sample_files.extend(paths);
        bands.push(band);
    }

    let receipt = assemble(meta, bands, samples_ms, drain_ms);
    let path = dir.join("receipt.json");
    let json = serde_json::to_string_pretty(&receipt)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    std::fs::write(&path, &json)?;
    Ok((
        receipt,
        WrittenReceipt {
            receipt: path,
            sample_files,
            bytes: json.len() as u64,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perf_gate::bootstrap::bootstrap_agg_tok_s_ci;
    use crate::perf_gate::protocol::{BandConfig, Outcome};
    use crate::perf_gate::window::WindowController;

    fn provenance() -> Provenance {
        Provenance {
            binary_path: "/opt/apr".into(),
            binary_sha256: "a".repeat(64),
            resolution: "current_exe".into(),
            compute_class: "cpu".into(),
            host: "lambda".into(),
            accelerator: "rtx-4090".into(),
            model: "qwen2.5-coder-1.5b-instruct".into(),
            quantization: "Q4_K_M".into(),
            feature_set: None,
        }
    }

    fn meta(replicates: usize) -> ReceiptMeta {
        ReceiptMeta {
            workload: "W1".into(),
            provenance: provenance(),
            tokenization: Tokenization::server_usage(true, false),
            replicates,
            commit: Some("62d23d8d1".into()),
        }
    }

    /// A replicate whose per-request latencies differ, because a constant
    /// `samples_ms` is the fabricated-measurement shape `bench_receipt.py`
    /// rejects (F12).
    fn replicate(concurrency: usize, n: usize, offset: f64) -> Replicate {
        let cfg = BandConfig::conformant(concurrency);
        let mut w = WindowController::with_bounds(n, 0.0);
        let mut samples = Vec::new();
        let mut now = 0.0_f64;
        while let Some((index, in_flight)) = w.try_admit_with_in_flight(now) {
            let start = now;
            let end = now + 1.0 + offset + (index as f64) * 0.01;
            let drained = w.complete(end);
            samples.push(RequestSample {
                index,
                worker: index % concurrency,
                start_s: start,
                end_s: end,
                token_times_s: (1..=8).map(|k| start + f64::from(k) * 0.125).collect(),
                generated_tokens: 8,
                prompt_tokens: 512,
                outcome: Outcome::Completed,
                in_flight_at_start: in_flight,
                drained,
            });
            now += 0.25;
        }
        let window = w.report();
        let metrics = BandMetrics::from_samples(cfg.concurrency, &samples);
        let agg_ci = bootstrap_agg_tok_s_ci(&samples, 0.95);
        Replicate {
            metrics,
            window,
            samples,
            agg_ci,
            protocol_violations: Vec::new(),
        }
    }

    fn cells(replicates: usize) -> Vec<(usize, Vec<Replicate>)> {
        vec![(
            1,
            (0..replicates)
                .map(|k| replicate(1, 30, k as f64 * 0.05))
                .collect(),
        )]
    }

    #[test]
    fn a_written_receipt_carries_every_field_arm_c_reads() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (receipt, written) =
            write_receipt(dir.path(), &meta(REPLICATES), &cells(REPLICATES)).expect("write");

        assert!(written.receipt.exists(), "receipt.json must exist");
        assert_eq!(receipt.requested, receipt.completed, "Arm C");
        assert_eq!(receipt.timeouts, 0, "Arm C");
        assert!(receipt.drain_ms >= 0.0, "Arm C requires drain_ms present");
        assert!(!receipt.samples_ms.is_empty(), "bench_receipt.py");
        assert_eq!(receipt.n, receipt.samples_ms.len(), "bench_receipt.py");
        assert!(receipt.arm_c_would_pass(), "{receipt:?}");
    }

    /// §4.4.5 — a summary-only receipt is rejected. The gz files must exist,
    /// be non-empty, and round-trip back to the samples that produced them.
    #[test]
    fn raw_samples_are_retained_as_gzipped_jsonl_and_round_trip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (receipt, written) = write_receipt(dir.path(), &meta(3), &cells(3)).expect("write");

        assert_eq!(written.sample_files.len(), 3, "one file per replicate");
        for path in &written.sample_files {
            assert!(path.exists(), "{} must exist", path.display());
            let ext = path.to_string_lossy().to_string();
            assert!(ext.ends_with(".jsonl.gz"), "{ext} must be gzipped JSONL");
            let back = crate::perf_gate::samples::read_samples_gz(path).expect("round trip");
            assert!(!back.is_empty(), "retained samples must not be empty");
        }
        let band = &receipt.bands[0];
        for rep in &band.replicates {
            assert!(rep.samples_file.rows > 0);
            assert_eq!(rep.samples_file.sha256.len(), 64);
            assert!(rep.samples_file.bytes > 0);
        }
    }

    /// `bench_receipt.py` rejects a constant `samples_ms` as F12. The receipt
    /// this writer produces must not be able to trip that.
    #[test]
    fn samples_ms_is_a_distribution_not_a_constant() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (receipt, _) = write_receipt(dir.path(), &meta(3), &cells(3)).expect("write");
        let first = receipt.samples_ms[0];
        assert!(
            receipt.samples_ms.iter().any(|v| (v - first).abs() > 1e-9),
            "a constant samples_ms is the fabricated-measurement shape (F12)"
        );
    }

    /// The escape hatch must be visible from the receipt.
    #[test]
    fn a_run_with_too_few_replicates_says_so() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (receipt, _) = write_receipt(dir.path(), &meta(1), &cells(1)).expect("write");
        assert!(!receipt.conformant);
        assert!(
            receipt
                .protocol_violations
                .iter()
                .any(|v| v.contains("replicates=1")),
            "{:?}",
            receipt.protocol_violations
        );
    }

    #[test]
    fn a_full_replicate_set_is_conformant() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (receipt, _) =
            write_receipt(dir.path(), &meta(REPLICATES), &cells(REPLICATES)).expect("write");
        assert!(receipt.conformant, "{:?}", receipt.protocol_violations);
        assert!(receipt.protocol_violations.is_empty());
    }

    /// The distinguishing fields. `LoadTest::run` can produce none of these, so
    /// their presence is what proves which path wrote the file.
    #[test]
    fn the_receipt_names_the_protocol_that_produced_it() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (receipt, _) = write_receipt(dir.path(), &meta(3), &cells(3)).expect("write");
        let json = serde_json::to_string(&receipt).expect("serialize");
        assert!(json.contains("\"client_model\":\"closed_loop\""), "{json}");
        assert!(json.contains("\"seed\":2026"), "{json}");
        assert!(json.contains("\"resampling_unit\":\"whole_requests\""));
        assert!(json.contains("\"drain_ms\""));
        assert!(json.contains("APR-PERF-GATE-001-v2.2"));
    }

    /// A receipt must be readable back out, or its interval cannot be checked
    /// and its numbers cannot be audited.
    ///
    /// Floats are compared with a relative tolerance rather than for equality.
    /// `serde_json` is built here without its `float_roundtrip` feature, so the
    /// parser is fast rather than exact and can land **1 ULP** away from the
    /// value that was written — observed on `agg_tok_s`
    /// (`28.103044496487122` read back as `28.10304449648712`). That is far
    /// below any threshold this gate applies, but it is a real property of the
    /// receipt and is recorded here rather than hidden behind a rounded
    /// assertion: a future check that demands bit-identical re-derivation from
    /// the retained samples would be relying on something untrue.
    #[test]
    fn a_receipt_round_trips_through_json() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (receipt, _) = write_receipt(dir.path(), &meta(3), &cells(3)).expect("write");
        let json = serde_json::to_string(&receipt).expect("serialize");
        let back: Receipt = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(back.spec, receipt.spec);
        assert_eq!(back.provenance, receipt.provenance);
        assert_eq!(back.tokenization, receipt.tokenization);
        assert_eq!(back.client_model, receipt.client_model);
        assert_eq!(back.bootstrap, receipt.bootstrap);
        assert_eq!(back.requested, receipt.requested);
        assert_eq!(back.completed, receipt.completed);
        assert_eq!(back.timeouts, receipt.timeouts);
        assert_eq!(back.n, receipt.n);
        assert_eq!(back.conformant, receipt.conformant);
        assert_eq!(back.bands.len(), receipt.bands.len());

        let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * a.abs().max(1.0);
        assert!(close(back.drain_ms, receipt.drain_ms));
        for (got, want) in back.bands.iter().zip(&receipt.bands) {
            assert_eq!(got.concurrency, want.concurrency);
            assert_eq!(got.tokens_total, want.tokens_total);
            assert_eq!(got.comparator_status, want.comparator_status);
            assert_eq!(got.replicates.len(), want.replicates.len());
            assert!(
                close(got.aggregate_tok_per_sec, want.aggregate_tok_per_sec),
                "{} vs {}",
                got.aggregate_tok_per_sec,
                want.aggregate_tok_per_sec
            );
            for (g, w) in got.replicates.iter().zip(&want.replicates) {
                // The retention pointer must survive exactly: it is what makes
                // the samples re-derivable.
                assert_eq!(g.samples_file, w.samples_file);
            }
        }
    }

    #[test]
    fn a_malformed_binary_digest_is_refused_before_anything_is_written() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut m = meta(3);
        m.provenance.binary_sha256 = "deadbeef".into();
        let err = write_receipt(dir.path(), &m, &cells(3)).expect_err("must refuse");
        assert!(err.to_string().contains("binary_sha256"), "{err}");
        assert!(
            !dir.path().join("receipt.json").exists(),
            "a rejected run must not leave a half-written receipt"
        );
    }

    #[test]
    fn an_unknown_compute_class_is_refused() {
        let mut p = provenance();
        p.compute_class = "tpu".into();
        assert!(p.validate().is_err());
    }

    /// The `bench_receipt.py` rule that catches the fabricated-14x class.
    #[test]
    fn a_compute_class_the_declared_build_cannot_reach_is_refused() {
        let mut p = provenance();
        p.compute_class = "cuda".into();
        p.feature_set = Some(vec!["default".into()]);
        let err = p
            .validate()
            .expect_err("cuda without the feature must fail");
        assert!(err.contains("cannot take that path"), "{err}");

        p.feature_set = Some(vec!["cuda".into()]);
        assert!(p.validate().is_ok());
    }

    #[test]
    fn an_empty_join_key_field_is_refused() {
        for blank in ["host", "accelerator", "model", "quantization"] {
            let mut p = provenance();
            match blank {
                "host" => p.host = String::new(),
                "accelerator" => p.accelerator = String::new(),
                "model" => p.model = String::new(),
                _ => p.quantization = String::new(),
            }
            let err = p.validate().expect_err("blank join key must fail");
            assert!(err.contains(blank), "{err}");
        }
    }

    #[test]
    fn a_receipt_with_no_bands_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let err = write_receipt(dir.path(), &meta(3), &[]).expect_err("must refuse");
        assert!(err.to_string().contains("vacuous"), "{err}");
    }

    /// §4.4.8 — without a comparator lane the ratios are absent and the cell is
    /// marked, rather than a ratio being synthesised from one side.
    #[test]
    fn a_comparator_free_band_is_marked_not_ratioed() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (receipt, _) = write_receipt(dir.path(), &meta(3), &cells(3)).expect("write");
        let band = &receipt.bands[0];
        assert_eq!(band.comparator_status, COMPARATOR_UNMEASURED);
        assert!(band.agg_ratio.is_none());
        assert!(band.decode_ratio.is_none());
    }

    #[test]
    fn sha256_file_matches_a_known_digest() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("x");
        std::fs::write(&path, b"").expect("write");
        // sha256 of the empty string.
        let digest = sha256_file(&path).expect("hash");
        assert_eq!(
            digest,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(digest.len(), 64, "bench_receipt.py requires 64 hex chars");
    }
}
