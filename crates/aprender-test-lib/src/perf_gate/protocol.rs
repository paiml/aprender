//! §4.4.1, §4.4.2, §4.4.6 — protocol constants, band configuration, and the
//! conformance predicate that makes a shrunken run say so.
//!
//! Every literal here is quoted from `docs/specifications/APR-PERF-GATE-001-v2.2.md`.
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
/// §4.4.2 — full band replicates per cell.
pub const REPLICATES: usize = 3;
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
            client_model: ClientModel::ClosedLoop,
        }
    }

    /// Every way this configuration departs from §4.4.2, in prose, for the receipt.
    #[must_use]
    pub fn conformance_violations(&self) -> Vec<String> {
        let mut out = Vec::new();
        let want_warmup = warmup_requests(self.concurrency);
        if self.warmup_requests < want_warmup {
            out.push(format!(
                "§4.4.2 warmup_requests={} < 2*c={want_warmup}",
                self.warmup_requests
            ));
        }
        if self.quiesce < QUIESCE {
            out.push(format!("§4.4.2 quiesce={:?} < 5s", self.quiesce));
        }
        let want_samples = min_sampled_requests(self.concurrency);
        if self.min_samples < want_samples {
            out.push(format!(
                "§4.4.2 min_samples={} < max(30, 8*c)={want_samples}",
                self.min_samples
            ));
        }
        if self.min_wall_clock < MIN_WALL_CLOCK {
            out.push(format!(
                "§4.4.2 min_wall_clock={:?} < 60s",
                self.min_wall_clock
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
            assert_eq!(cfg.client_model, ClientModel::ClosedLoop);
            assert!(
                cfg.is_conformant(),
                "violations: {:?}",
                cfg.conformance_violations()
            );
        }
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
        assert!(ct.require_counter(None).is_err());
        assert!(ct.require_counter(Some(&"c".repeat(64))).is_ok());
        // The arm a `bool` could never express: a counter IS present, and it is
        // not the one the block names.
        let borrowed = ct
            .require_counter(Some(&"d".repeat(64)))
            .expect_err("a digest for another file must be refused");
        assert!(borrowed.contains("did not open"), "{borrowed}");

        let su = TokenizationBlock::ServerUsage {
            counts_special_tokens: true,
            counts_prompt_echo: false,
        };
        assert!(su.require_counter(Some(&"c".repeat(64))).is_err());
        assert!(su.require_counter(None).is_ok());
    }
}
