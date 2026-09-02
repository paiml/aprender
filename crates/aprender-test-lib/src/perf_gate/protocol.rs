//! PP-LLAMA-001 v3.0 §5.1 — protocol parameters, band configuration, and the
//! conformance predicate that makes a shrunken run say so.
//!
//! # Where the numbers live (PP-33)
//!
//! Every protocol parameter is declared in `scripts/perf-matrix.yaml` under
//! `protocol:` and read from there by [`ProtocolParams::from_matrix`]. The
//! `pub const`s below are the **spec fallback**: they exist so this module's
//! own tests have a value to compare against when the matrix has not yet been
//! amended, and so [`ProtocolParams::from_matrix`] can be proven to *differ*
//! from them (`conformance_violations_read_the_loaded_params_not_the_consts`).
//! They are documented fallbacks, never a silent default: a matrix without a
//! `protocol:` block makes [`ProtocolParams::from_matrix`] return `Err` naming
//! the missing block.
//!
//! Nothing in this file is a threshold; they are all protocol parameters.

use std::time::Duration;

use serde::{Deserialize, Serialize};

/// §4.4.1 — the client model. Closed-loop is the only model this module
/// implements, and it is *recorded* rather than assumed so the choice is
/// falsifiable from the receipt alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClientModel {
    /// `c` workers; each issues a request, waits for completion, immediately
    /// issues the next.
    ClosedLoop,
}

/// §4.4.2 — warmup requests are `WARMUP_MULTIPLIER × c`, discarded.
pub const WARMUP_MULTIPLIER: usize = 2;
/// §4.4.2 — quiesce between warmup completion and the first sampled request.
pub const QUIESCE: Duration = Duration::from_secs(5);
/// §4.4.2 — minimum sampled requests is `max(MIN_SAMPLES_FLOOR, MIN_SAMPLES_PER_WORKER × c)`.
pub const MIN_SAMPLES_FLOOR: usize = 30;
/// §4.4.2 — the per-worker term of the minimum-sample rule.
pub const MIN_SAMPLES_PER_WORKER: usize = 8;
/// §4.4.2 — minimum wall-clock per band.
pub const MIN_WALL_CLOCK: Duration = Duration::from_secs(60);
/// §4.4.3 — hard per-request timeout. A request exceeding this increments `timeouts`.
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
/// §5.1 — cooldown between the two lanes of one interleaved replicate.
pub const COOLDOWN: Duration = Duration::from_secs(10);
/// §4.3 — full band replicates per cell. **Five**, not three: `n = 3` sizes an
/// effect and bounds no variance, so no σ-dependent status may change below 5.
pub const REPLICATES: usize = 5;
/// §4.3 — replicates are interleaved A,B,A,B,…; a non-interleaved receipt is
/// `NONCONFORMANT-VALID` (PP-9's key carries `interleaved: true`).
pub const INTERLEAVED: bool = true;
/// §5.1 W1 — generated tokens per request, on the wire as OpenAI `max_tokens`.
pub const N_PREDICT: u32 = 128;
/// §5.1 — the pinned sampler temperature for both lanes (PP-28).
pub const SAMPLER_TEMPERATURE: f64 = 0.0;
/// §5.1 — the pinned sampler seed for both lanes (PP-28).
pub const SAMPLER_SEED: u64 = 0;
/// §5.1 — `ignore_eos` on both lanes, so every retained sample runs to
/// `n_predict` (PP-28).
pub const SAMPLER_IGNORE_EOS: bool = true;
/// PP-27 — a live stream has `median(ttft / e2e)` well below 1; a replayed one
/// approaches 1 because the whole answer arrives at once. Matrix key
/// `stream.live_ttft_over_e2e_max`.
pub const STREAM_LIVE_TTFT_OVER_E2E_MAX: f64 = 0.95;
/// PP-26 — tokens that must agree between `m=1` and the batched run. Matrix key
/// `witness.min_agree_tokens`.
pub const WITNESS_MIN_AGREE_TOKENS: u32 = 64;
/// §4.4.4 — bootstrap resamples.
pub const BOOTSTRAP_RESAMPLES: usize = 10_000;
/// §4.4.4 — bootstrap seed. Goes in the receipt; the interval is reproducible
/// from the retained samples with this value and no other.
pub const BOOTSTRAP_SEED: u64 = 2026;

/// §4.4.2 — `max(30, 8 × c)`.
#[must_use]
pub fn min_sampled_requests(concurrency: usize) -> usize {
    MIN_SAMPLES_FLOOR.max(MIN_SAMPLES_PER_WORKER * concurrency)
}

/// §4.4.2 — `2 × c`.
#[must_use]
pub fn warmup_requests(concurrency: usize) -> usize {
    WARMUP_MULTIPLIER * concurrency
}

/// `scripts/perf-matrix.yaml`, compiled in.
///
/// PP-33 puts every number the gate compares against in that file. Reading it
/// at runtime would make the producer depend on a path that does not exist in a
/// published crate; `include_str!` binds the exact bytes of the checkout the
/// binary was built from, which is also what the receipt's `commit` claims.
pub const PERF_MATRIX_SOURCE: &str = include_str!("../../../../scripts/perf-matrix.yaml");

/// §5.1 / PP-28 — the sampler pinned on both lanes, on the wire in every receipt.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sampler {
    /// Greedy decoding. `0.0` in every conformant run.
    pub temperature: f64,
    /// The seed both lanes were given.
    pub seed: u64,
    /// `true` in W1, so `completion_tokens == n_predict` on every sample.
    pub ignore_eos: bool,
}

impl Sampler {
    /// The §5.1 pin, as the spec fallback.
    #[must_use]
    pub const fn spec_fallback() -> Self {
        Self {
            temperature: SAMPLER_TEMPERATURE,
            seed: SAMPLER_SEED,
            ignore_eos: SAMPLER_IGNORE_EOS,
        }
    }
}

/// §5.1 — the protocol block, read from `perf-matrix.yaml` and emitted verbatim
/// at the receipt's top level as `protocol`.
///
/// A reader that cannot see the window, the warmup, the cooldown, the sampler
/// and the replicate count cannot tell a 60 s conformant band from a 5 s one,
/// and two receipts written under different protocols are not comparable — which
/// is why the whole block is also in the PP-22 join key.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProtocolParams {
    /// `T` — the measurement window, in milliseconds.
    pub window_ms: u64,
    /// Warmup requests per worker, discarded (§4.4.2's `2 × c` as a per-worker
    /// count, so the band-level figure is `warmup_requests_per_worker × c`).
    pub warmup_requests_per_worker: u32,
    /// Quiesce between warmup completion and the first sampled request.
    pub quiesce_ms: u64,
    /// §5.1 — cooldown between the two lanes of an interleaved replicate.
    pub cooldown_ms: u64,
    /// Generated tokens per request (`max_tokens` on the wire).
    pub n_predict: u32,
    /// Replicates per cell. The matrix declares the FLOOR (`replicates_min`);
    /// a receipt records the number actually run and is `NONCONFORMANT-VALID`
    /// below it.
    pub replicates: u32,
    /// Whether the replicates alternated A,B,A,B,….
    pub interleaved: bool,
    /// The pinned sampler (PP-28).
    pub sampler: Sampler,
}

impl ProtocolParams {
    /// The spec literals, as a documented fallback for tests and for callers
    /// that must render a receipt before the matrix carries a `protocol:` block.
    ///
    /// Never reached silently by [`Self::from_matrix`], which returns `Err`.
    #[must_use]
    pub const fn spec_fallback() -> Self {
        Self {
            window_ms: MIN_WALL_CLOCK.as_millis() as u64,
            warmup_requests_per_worker: WARMUP_MULTIPLIER as u32,
            quiesce_ms: QUIESCE.as_millis() as u64,
            cooldown_ms: COOLDOWN.as_millis() as u64,
            n_predict: N_PREDICT,
            replicates: REPLICATES as u32,
            interleaved: INTERLEAVED,
            sampler: Sampler::spec_fallback(),
        }
    }

    /// Read the `protocol:` block out of the compiled-in `perf-matrix.yaml`.
    ///
    /// # Errors
    /// When the file does not parse as YAML, or when it carries no `protocol:`
    /// block, or when the block is missing a key. The error NAMES the missing
    /// thing: a protocol silently defaulted to the Rust consts is exactly the
    /// drift PP-33 exists to prevent.
    pub fn from_matrix() -> Result<Self, String> {
        Self::from_matrix_source(PERF_MATRIX_SOURCE)
    }

    /// [`Self::from_matrix`] against an explicit document, so a test can feed a
    /// matrix that differs from the shipped one.
    ///
    /// # Errors
    /// As [`Self::from_matrix`].
    pub fn from_matrix_source(source: &str) -> Result<Self, String> {
        let block = matrix_block(source, "protocol")?;
        let raw: MatrixProtocolBlock = serde_yaml_ng::from_value(block)
            .map_err(|e| format!("perf-matrix.yaml `protocol:` block: {e}"))?;
        Ok(Self {
            window_ms: raw.window_ms,
            warmup_requests_per_worker: raw.warmup_requests_per_worker,
            quiesce_ms: raw.quiesce_ms,
            cooldown_ms: raw.cooldown_ms,
            n_predict: raw.n_predict,
            replicates: raw.replicates_min,
            interleaved: raw.interleaved,
            sampler: raw.sampler,
        })
    }

    /// The parameters a receipt written on this checkout must use, **with the
    /// provenance of where they came from**.
    ///
    /// [`Self::from_matrix`] is the honest reader and returns `Err`; this is the
    /// one place the fallback is taken, and the returned [`ProtocolSource`] is
    /// what the producer prints once and — on the fallback — names in
    /// `unproduced_fields`.
    ///
    /// The silent version of this ([`Self::effective`], which discarded the
    /// error) put the Rust consts on the wire under a `protocol:` block the
    /// receipt then claimed came from the matrix. PP-33's whole point is that
    /// every gated number lives in `perf-matrix.yaml`; a receipt that quietly
    /// substituted a compiled-in copy is the drift it exists to prevent, and it
    /// was invisible because nothing ever called `source()`.
    #[must_use]
    pub fn effective_with_source() -> (Self, ProtocolSource) {
        match Self::from_matrix() {
            Ok(params) => (params, ProtocolSource::Matrix),
            Err(reason) => (Self::spec_fallback(), ProtocolSource::SpecFallback(reason)),
        }
    }

    /// [`Self::effective_with_source`] without the provenance, for callers that
    /// record it separately (the CLI producer) or do not write a receipt at all
    /// (tests, `BandConfig`).
    #[must_use]
    pub fn effective() -> Self {
        Self::effective_with_source().0
    }

    /// `"perf-matrix.yaml"` when the matrix declares a `protocol:` block, and
    /// the reason the fallback was taken otherwise.
    ///
    /// # Errors
    /// When the matrix has no `protocol:` block; the error is the provenance
    /// note a caller puts in `unproduced_fields`.
    pub fn source() -> Result<&'static str, String> {
        Self::from_matrix().map(|_| "perf-matrix.yaml `protocol:`")
    }
}

/// PP-33 — where a [`ProtocolParams`] came from.
///
/// Not a boolean: the fallback carries the reason it was taken, because "the
/// matrix has no `protocol:` block" and "the matrix does not parse as YAML" ask
/// for different fixes and a receipt that says only "fallback" tells the reader
/// neither.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProtocolSource {
    /// Read from the compiled-in `scripts/perf-matrix.yaml`.
    Matrix,
    /// The matrix could not supply them; the reason is carried verbatim.
    SpecFallback(String),
}

impl ProtocolSource {
    /// The single line the producer prints before the first request.
    #[must_use]
    pub fn announcement(&self) -> String {
        match self {
            Self::Matrix => "protocol: matrix (scripts/perf-matrix.yaml `protocol:`)".to_string(),
            Self::SpecFallback(reason) => {
                format!("protocol: spec fallback because {reason}")
            }
        }
    }

    /// The `unproduced_fields` entry, or `None` when the matrix supplied the
    /// parameters and there is nothing unproduced.
    #[must_use]
    pub fn unproduced_note(&self) -> Option<String> {
        match self {
            Self::Matrix => None,
            Self::SpecFallback(reason) => Some(format!(
                "PP-33 protocol — the `protocol:` block on this receipt is the compiled-in Rust \
                 spec fallback, NOT scripts/perf-matrix.yaml: {reason}. Every protocol parameter \
                 a gate compares against must live in the matrix; these came from consts and are \
                 unverifiable against it."
            )),
        }
    }
}

/// PP-27 — `stream.live_ttft_over_e2e_max` from the matrix.
///
/// # Errors
/// When the matrix has no `stream:` block or no `live_ttft_over_e2e_max` key.
pub fn stream_live_ttft_over_e2e_max_from(source: &str) -> Result<f64, String> {
    let block = matrix_block(source, "stream")?;
    let raw: MatrixStreamBlock = serde_yaml_ng::from_value(block)
        .map_err(|e| format!("perf-matrix.yaml `stream:` block: {e}"))?;
    Ok(raw.live_ttft_over_e2e_max)
}

/// [`stream_live_ttft_over_e2e_max_from`] over the compiled-in matrix, falling
/// back to [`STREAM_LIVE_TTFT_OVER_E2E_MAX`] when the block is absent.
#[must_use]
pub fn stream_live_ttft_over_e2e_max() -> f64 {
    stream_live_ttft_over_e2e_max_from(PERF_MATRIX_SOURCE).unwrap_or(STREAM_LIVE_TTFT_OVER_E2E_MAX)
}

/// PP-26 — `witness.min_agree_tokens` from the matrix.
///
/// # Errors
/// When the matrix has no `witness:` block or no `min_agree_tokens` key.
pub fn witness_min_agree_tokens_from(source: &str) -> Result<u32, String> {
    let block = matrix_block(source, "witness")?;
    let raw: MatrixWitnessBlock = serde_yaml_ng::from_value(block)
        .map_err(|e| format!("perf-matrix.yaml `witness:` block: {e}"))?;
    Ok(raw.min_agree_tokens)
}

/// [`witness_min_agree_tokens_from`] over the compiled-in matrix, falling back
/// to [`WITNESS_MIN_AGREE_TOKENS`] when the block is absent.
#[must_use]
pub fn witness_min_agree_tokens() -> u32 {
    witness_min_agree_tokens_from(PERF_MATRIX_SOURCE).unwrap_or(WITNESS_MIN_AGREE_TOKENS)
}

/// Pull one top-level block out of the matrix, naming what is missing.
fn matrix_block(source: &str, key: &str) -> Result<serde_yaml_ng::Value, String> {
    let doc: serde_yaml_ng::Value = serde_yaml_ng::from_str(source)
        .map_err(|e| format!("perf-matrix.yaml does not parse as YAML: {e}"))?;
    doc.get(key).cloned().ok_or_else(|| {
        format!(
            "perf-matrix.yaml has no `{key}:` block — PP-33 requires every protocol parameter and \
             threshold to live there; refusing to substitute the Rust spec fallback silently"
        )
    })
}

/// The matrix spelling of the protocol block. Governance keys
/// (`threshold_class`, `author`, `prompt_tokens`) are ignored here rather than
/// mirrored, so adding one does not have to touch this crate.
///
/// These three `Matrix*` readers are the deliberate exception to the
/// `deny_unknown_fields` rule every other `Deserialize` type in `perf_gate/`
/// carries: PP-33 requires the matrix to hold governance metadata beside each
/// number, and refusing an unknown key here would make adding an `author:` to
/// `perf-matrix.yaml` fail this crate's build. The receipt types are the ones
/// that must refuse a key they do not understand — the receipt is the evidence,
/// the matrix is the policy.
#[derive(Debug, Deserialize)]
struct MatrixProtocolBlock {
    window_ms: u64,
    warmup_requests_per_worker: u32,
    quiesce_ms: u64,
    cooldown_ms: u64,
    n_predict: u32,
    replicates_min: u32,
    interleaved: bool,
    sampler: Sampler,
}

#[derive(Debug, Deserialize)]
struct MatrixStreamBlock {
    live_ttft_over_e2e_max: f64,
}

#[derive(Debug, Deserialize)]
struct MatrixWitnessBlock {
    min_agree_tokens: u32,
}

/// One band's measurement parameters.
///
/// [`BandConfig::conformant`] is the only constructor that produces §4.4-legal
/// values. [`BandConfig::relaxed`] exists so unit tests can exercise the driver
/// in milliseconds instead of minutes — and every run carries
/// [`BandConfig::conformance_violations`] into its receipt, so a relaxed run is
/// self-identifying rather than indistinguishable from a real one. A knob that
/// lets you shrink the window without saying so is how a gate stops being able
/// to fail.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BandConfig {
    /// Fixed concurrency `c`.
    pub concurrency: usize,
    /// Warmup requests, discarded and never written to the receipt.
    pub warmup_requests: usize,
    /// Quiesce between warmup completion and the first sampled request.
    pub quiesce: Duration,
    /// Minimum sampled requests to issue before the window may close.
    pub min_samples: usize,
    /// Minimum wall-clock the window must stay open.
    pub min_wall_clock: Duration,
    /// Hard per-request timeout.
    pub request_timeout: Duration,
    /// §5.1 — the pause between the two lanes of one interleaved replicate.
    /// Without it the second lane inherits the first lane's thermal and VRAM
    /// state, which is the drift interleaving exists to cancel.
    pub cooldown: Duration,
    /// Recorded, not assumed.
    pub client_model: ClientModel,
}

impl BandConfig {
    /// The §4.4-conformant configuration for concurrency `c`.
    ///
    /// # Panics
    /// Never; `concurrency` of 0 is clamped to 1 so the worker pool always has
    /// a worker. A zero-worker band is unrepresentable rather than empty.
    #[must_use]
    pub fn conformant(concurrency: usize) -> Self {
        let concurrency = concurrency.max(1);
        Self {
            concurrency,
            warmup_requests: warmup_requests(concurrency),
            quiesce: QUIESCE,
            min_samples: min_sampled_requests(concurrency),
            min_wall_clock: MIN_WALL_CLOCK,
            request_timeout: REQUEST_TIMEOUT,
            cooldown: COOLDOWN,
            client_model: ClientModel::ClosedLoop,
        }
    }

    /// A shrunken configuration for tests and smoke runs. **Never conformant**:
    /// [`Self::conformance_violations`] is non-empty by construction unless the
    /// caller happens to pass the spec values back in.
    #[must_use]
    pub fn relaxed(
        concurrency: usize,
        min_samples: usize,
        min_wall_clock: Duration,
        quiesce: Duration,
    ) -> Self {
        let concurrency = concurrency.max(1);
        Self {
            concurrency,
            warmup_requests: warmup_requests(concurrency),
            quiesce,
            min_samples,
            min_wall_clock,
            request_timeout: REQUEST_TIMEOUT,
            cooldown: COOLDOWN,
            client_model: ClientModel::ClosedLoop,
        }
    }

    /// [`Self::relaxed`] with the §5.1 cooldown shrunk too, for tests that must
    /// exercise the cooldown departure itself.
    #[must_use]
    pub fn relaxed_with_cooldown(
        concurrency: usize,
        min_samples: usize,
        min_wall_clock: Duration,
        quiesce: Duration,
        cooldown: Duration,
    ) -> Self {
        Self {
            cooldown,
            ..Self::relaxed(concurrency, min_samples, min_wall_clock, quiesce)
        }
    }

    /// Every way this configuration departs from the protocol, in prose, for
    /// the receipt.
    ///
    /// Compares against [`ProtocolParams::effective`] — the `protocol:` block of
    /// `perf-matrix.yaml` when it exists (PP-33), the spec fallback otherwise —
    /// so amending the matrix moves this predicate rather than a Rust literal.
    #[must_use]
    pub fn conformance_violations(&self) -> Vec<String> {
        self.conformance_violations_against(&ProtocolParams::effective())
    }

    /// [`Self::conformance_violations`] against explicit parameters, so a test
    /// can prove the predicate reads the loaded block and not a constant.
    #[must_use]
    pub fn conformance_violations_against(&self, params: &ProtocolParams) -> Vec<String> {
        let mut out = Vec::new();
        let want_warmup = params.warmup_requests_per_worker as usize * self.concurrency;
        if self.warmup_requests < want_warmup {
            out.push(format!(
                "§4.4.2 warmup_requests={} < {}*c={want_warmup}",
                self.warmup_requests, params.warmup_requests_per_worker
            ));
        }
        let want_quiesce = Duration::from_millis(params.quiesce_ms);
        if self.quiesce < want_quiesce {
            out.push(format!(
                "§4.4.2 quiesce={:?} < {want_quiesce:?}",
                self.quiesce
            ));
        }
        let want_samples = min_sampled_requests(self.concurrency);
        if self.min_samples < want_samples {
            out.push(format!(
                "§4.4.2 min_samples={} < max(30, 8*c)={want_samples}",
                self.min_samples
            ));
        }
        let want_window = Duration::from_millis(params.window_ms);
        if self.min_wall_clock < want_window {
            out.push(format!(
                "§5.1 min_wall_clock={:?} < window_ms={want_window:?}",
                self.min_wall_clock
            ));
        }
        let want_cooldown = Duration::from_millis(params.cooldown_ms);
        if self.cooldown < want_cooldown {
            out.push(format!(
                "§5.1 cooldown={:?} < cooldown_ms={want_cooldown:?}",
                self.cooldown
            ));
        }
        if self.request_timeout != REQUEST_TIMEOUT {
            out.push(format!(
                "§4.4.3 request_timeout={:?} != 120s",
                self.request_timeout
            ));
        }
        out
    }

    /// True when [`Self::conformance_violations`] is empty.
    #[must_use]
    pub fn is_conformant(&self) -> bool {
        self.conformance_violations().is_empty()
    }
}

/// §4.4.3 / §4.4.7 — how one request ended, and the §4.4.7 `SUSPECT` fraction.
///
/// Both are defined once, in [`super::drain`], and re-exported here so
/// §4.4.1-§4.4.5 code keeps its `protocol::` path. The four outcomes are the
/// four counters the receipt must carry, and they are mutually exclusive, so
/// `requested = completed + timeouts + truncated + errors` holds by
/// construction rather than by convention.
///
/// **[`Outcome::AbandonedAtDrain`] is the §4.4.7 sense of `truncated`, NOT
/// `finish_reason == "length"`.** Under W1 every request stops at
/// `max_tokens = 128` with EOS ignored, so reading `truncated` as the
/// finish-reason sense would exclude the entire workload from `agg_tok_s`'s
/// numerator and report zero throughput for a healthy server. The variant is
/// spelled `AbandonedAtDrain` rather than `Truncated` precisely so the two
/// senses cannot be conflated; conflating them is how two conformant harnesses
/// produce incomparable receipts.
pub use super::drain::{Outcome, DRAIN_SUSPECT_FRACTION};

/// §4.4.6 — the `tokenization` block lives in [`super::receipt`].
///
/// This module used to carry its own `Tokenization` struct with an
/// `Option<String>` digest and a `validate()`.
/// [`super::receipt::TokenizationBlock`] is the same §4.4.6 block as an enum,
/// in which "`client_tokenizer` with no digest" is unrepresentable rather than
/// merely rejected, and it is the one wired into the emitter
/// `scripts/perf_gate.sh` reads. Keeping both would be two spellings of one
/// schema, free to drift.
pub use super::receipt::{TokenCountingMethod, TokenizationBlock};

#[cfg(test)]
mod tests {
    use super::*;

    /// A `perf-matrix.yaml` fragment carrying exactly the keys PP-33 puts
    /// there. Used so the loader is exercised whether or not the shipped matrix
    /// has been amended yet.
    const FIXTURE_MATRIX: &str = "\
schema_version: 2
protocol:
  window_ms: 60000
  warmup_requests_per_worker: 2
  quiesce_ms: 5000
  cooldown_ms: 10000
  n_predict: 128
  prompt_tokens: 512
  replicates_min: 5
  interleaved: true
  sampler: {temperature: 0.0, seed: 0, ignore_eos: true}
  threshold_class: policy
  author: spec-owner
stream:
  live_ttft_over_e2e_max: 0.95
  threshold_class: policy
  author: spec-owner
witness:
  min_agree_tokens: 64
  threshold_class: policy
  author: spec-owner
";

    #[test]
    fn min_sampled_requests_is_max_30_or_8c() {
        assert_eq!(min_sampled_requests(1), 30);
        assert_eq!(min_sampled_requests(3), 30);
        assert_eq!(min_sampled_requests(4), 32);
        assert_eq!(min_sampled_requests(8), 64);
        assert_eq!(min_sampled_requests(16), 128);
    }

    #[test]
    fn warmup_is_two_per_worker() {
        for c in [1_usize, 4, 8, 16] {
            assert_eq!(warmup_requests(c), 2 * c);
        }
    }

    /// The shipped defaults ARE the spec values. If someone edits a constant,
    /// this is the test that reds.
    #[test]
    fn conformant_config_matches_the_spec_literals() {
        for c in [1_usize, 4, 8, 16] {
            let cfg = BandConfig::conformant(c);
            assert_eq!(cfg.concurrency, c);
            assert_eq!(cfg.warmup_requests, 2 * c);
            assert_eq!(cfg.quiesce, Duration::from_secs(5));
            assert_eq!(cfg.min_samples, 30.max(8 * c));
            assert_eq!(cfg.min_wall_clock, Duration::from_secs(60));
            assert_eq!(cfg.request_timeout, Duration::from_secs(120));
            assert_eq!(cfg.cooldown, Duration::from_secs(10));
            assert_eq!(cfg.client_model, ClientModel::ClosedLoop);
            assert!(
                cfg.is_conformant(),
                "violations: {:?}",
                cfg.conformance_violations()
            );
        }
    }

    /// A matrix source with no `protocol:` block does not quietly become the
    /// Rust constants: the reader says which block is missing (PP-33).
    #[test]
    fn a_matrix_without_a_protocol_block_is_an_error_not_a_default() {
        let err = ProtocolParams::from_matrix_source("schema_version: 2\nbands: [1, 4]\n")
            .expect_err("no protocol block");
        assert!(err.contains("`protocol:`"), "{err}");
        assert!(err.contains("PP-33"), "{err}");
    }

    /// And a block that is present is read field by field.
    #[test]
    fn the_protocol_block_is_read_from_the_matrix() {
        let p = ProtocolParams::from_matrix_source(FIXTURE_MATRIX).expect("block parses");
        assert_eq!(p.window_ms, 60_000);
        assert_eq!(p.warmup_requests_per_worker, 2);
        assert_eq!(p.quiesce_ms, 5_000);
        assert_eq!(p.cooldown_ms, 10_000);
        assert_eq!(p.n_predict, 128);
        assert_eq!(
            p.replicates, 5,
            "matrix `replicates_min` is the receipt's n floor"
        );
        assert!(p.interleaved);
        assert_eq!(p.sampler.temperature, 0.0);
        assert_eq!(p.sampler.seed, 0);
        assert!(p.sampler.ignore_eos);
    }

    /// THE POINT: the conformance predicate reads the LOADED parameters. Feed a
    /// matrix declaring a 120 s window and a 60 s band stops being conformant —
    /// which a predicate hard-coded to `MIN_WALL_CLOCK` could not do.
    #[test]
    fn conformance_violations_read_the_loaded_params_not_the_consts() {
        let cfg = BandConfig::conformant(4);
        let spec = ProtocolParams::spec_fallback();
        assert!(cfg.conformance_violations_against(&spec).is_empty());

        let wider = ProtocolParams {
            window_ms: 120_000,
            ..spec
        };
        let v = cfg.conformance_violations_against(&wider);
        assert_eq!(v.len(), 1, "{v:?}");
        assert!(v[0].contains("min_wall_clock"), "{v:?}");
    }

    /// The cooldown is a departure like any other, and it is checked.
    #[test]
    fn a_missing_cooldown_is_a_conformance_violation() {
        let cfg = BandConfig::relaxed_with_cooldown(
            4,
            32,
            Duration::from_secs(60),
            Duration::from_secs(5),
            Duration::ZERO,
        );
        let v = cfg.conformance_violations();
        assert_eq!(v.len(), 1, "{v:?}");
        assert!(v[0].contains("cooldown"), "{v:?}");
    }

    /// `effective()` takes the matrix when it has the block and the documented
    /// fallback otherwise, and `source()` says which — so a caller can name the
    /// substitution in `unproduced_fields` rather than have it be silent.
    #[test]
    fn the_effective_params_say_where_they_came_from() {
        let effective = ProtocolParams::effective();
        match ProtocolParams::source() {
            Ok(where_from) => {
                assert_eq!(where_from, "perf-matrix.yaml `protocol:`");
                assert_eq!(effective, ProtocolParams::from_matrix().expect("block"));
            }
            Err(reason) => {
                assert!(reason.contains("`protocol:`"), "{reason}");
                assert_eq!(effective, ProtocolParams::spec_fallback());
            }
        }
    }

    /// PP-27's and PP-26's numbers come out of the matrix too.
    #[test]
    fn the_stream_and_witness_thresholds_are_read_from_the_matrix() {
        assert_eq!(
            stream_live_ttft_over_e2e_max_from(FIXTURE_MATRIX).expect("stream block"),
            0.95
        );
        assert_eq!(
            witness_min_agree_tokens_from(FIXTURE_MATRIX).expect("witness block"),
            64
        );
        assert!(stream_live_ttft_over_e2e_max_from("bands: [1]\n").is_err());
        assert!(witness_min_agree_tokens_from("bands: [1]\n").is_err());
    }

    /// When `scripts/perf-matrix.yaml` DOES declare the block, it must agree
    /// with the fallback the tests compare against — otherwise the two drift
    /// and a receipt is written under one protocol and validated under another.
    #[test]
    fn the_shipped_matrix_block_when_present_agrees_with_the_spec_fallback() {
        match ProtocolParams::from_matrix() {
            Ok(loaded) => assert_eq!(
                loaded,
                ProtocolParams::spec_fallback(),
                "scripts/perf-matrix.yaml `protocol:` disagrees with protocol.rs's fallback"
            ),
            Err(reason) => assert!(
                reason.contains("`protocol:`"),
                "the only acceptable absence is a named one: {reason}"
            ),
        }
    }

    /// PP-33 — `effective_with_source` says WHICH of the two sources supplied
    /// the block, and the fallback carries the reason.
    ///
    /// `effective()` swallowed the error and `source()` was never called by
    /// anything, so a run whose matrix did not parse put the compiled-in Rust
    /// constants on the wire under a `protocol:` block the receipt then
    /// presented as the matrix's. PP-33's whole point is that every number a
    /// gate compares against lives in `perf-matrix.yaml`; a silent compiled-in
    /// substitute is the drift it exists to prevent.
    #[test]
    fn the_protocol_source_is_reported_and_the_fallback_says_why() {
        let (params, source) = ProtocolParams::effective_with_source();
        match ProtocolParams::from_matrix() {
            Ok(from_matrix) => {
                assert_eq!(source, ProtocolSource::Matrix);
                assert_eq!(params, from_matrix);
                assert!(source.announcement().contains("matrix"), "{source:?}");
                assert!(
                    source.unproduced_note().is_none(),
                    "the matrix supplied them; nothing is unproduced"
                );
            }
            Err(reason) => {
                assert_eq!(source, ProtocolSource::SpecFallback(reason.clone()));
                assert_eq!(params, ProtocolParams::spec_fallback());
            }
        }

        // The fallback's own two obligations, whatever the shipped matrix does:
        // it names the reason on stdout, and it names itself in the receipt.
        let fallback =
            ProtocolSource::SpecFallback("perf-matrix.yaml has no `protocol:` block".to_string());
        assert_eq!(
            fallback.announcement(),
            "protocol: spec fallback because perf-matrix.yaml has no `protocol:` block"
        );
        let note = fallback
            .unproduced_note()
            .expect("a fallback is an unproduced field");
        assert!(note.contains("PP-33"), "{note}");
        assert!(note.contains("no `protocol:` block"), "{note}");
        assert!(
            note.contains("NOT scripts/perf-matrix.yaml"),
            "the note must say the block on the wire is not the matrix's: {note}"
        );
    }

    /// The escape hatch must be visible from the receipt. A relaxed run that
    /// reported itself conformant is exactly the fabricated-baseline class.
    #[test]
    fn relaxed_config_reports_every_departure() {
        let cfg = BandConfig::relaxed(4, 8, Duration::from_millis(50), Duration::ZERO);
        assert!(!cfg.is_conformant());
        let v = cfg.conformance_violations();
        assert_eq!(
            v.len(),
            3,
            "expected quiesce+min_samples+min_wall, got {v:?}"
        );
        assert!(v.iter().any(|s| s.contains("quiesce")));
        assert!(v.iter().any(|s| s.contains("min_samples")));
        assert!(v.iter().any(|s| s.contains("min_wall_clock")));
    }

    #[test]
    fn client_model_serializes_as_closed_loop() {
        let j =
            serde_json::to_string(&ClientModel::ClosedLoop).expect("ClientModel must serialize");
        assert_eq!(j, "\"closed_loop\"");
    }

    /// `protocol::REQUEST_TIMEOUT` (a `Duration`) and `drain::REQUEST_TIMEOUT_MS`
    /// (an `f64`) are the same §4.4.3 limit in two types the compiler cannot
    /// unify. Editing one without the other is the drift this pins.
    #[test]
    fn the_two_request_timeout_spellings_agree() {
        assert_eq!(
            REQUEST_TIMEOUT.as_millis(),
            u128::from(super::super::drain::REQUEST_TIMEOUT_MS as u64)
        );
    }

    /// The §4.4.6 block is `receipt::TokenizationBlock`; the poka-yoke that used
    /// to live on `protocol::Tokenization` moved with it and is still enforced.
    #[test]
    fn declared_method_and_available_counter_must_agree() {
        let ct = TokenizationBlock::ClientTokenizer {
            tokenizer_sha256: "c".repeat(64),
            counts_special_tokens: true,
            counts_prompt_echo: false,
        };
        assert!(ct.validate().is_ok());
        assert!(ct.require_counter(false).is_err());
        assert!(ct.require_counter(true).is_ok());

        let su = TokenizationBlock::ServerUsage {
            counts_special_tokens: true,
            counts_prompt_echo: false,
        };
        assert!(su.require_counter(true).is_err());
        assert!(su.require_counter(false).is_ok());
    }
}
