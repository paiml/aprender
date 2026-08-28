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
/// §4.4.7 — `drain_ms > DRAIN_SUSPECT_FRACTION × window` is annotated `SUSPECT`.
pub const DRAIN_SUSPECT_FRACTION: f64 = 0.5;

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

/// §4.4.3 / §4.4.7 — how one request ended. The four values are the four
/// counters the receipt must carry, and they are mutually exclusive so
/// `requested = completed + timeouts + truncated + errors` holds by construction.
///
/// **`Truncated` means "abandoned at the drain deadline" (§4.4.7), NOT
/// `finish_reason == "length"`.** Under W1 every request stops at
/// `max_tokens = 128` with EOS ignored, so reading `truncated` as the
/// finish-reason sense would exclude the entire workload from `agg_tok_s`'s
/// numerator and report zero throughput for a healthy server. The two senses are
/// named differently on purpose; conflating them is how two conformant harnesses
/// produce incomparable receipts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    /// Finished inside the window or during the drain, with a usable response.
    Completed,
    /// Hit the 120 s hard per-request timeout (§4.4.3).
    Timeout,
    /// Abandoned at the drain deadline (§4.4.7).
    Truncated,
    /// Failed for any other reason (transport, non-2xx, unparseable body).
    Error,
}

/// §4.4.6 — how tokens were counted. **No `Default` impl, deliberately:** the
/// spec says `method` has no default and its absence is schema-fatal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenizationMethod {
    /// Token counts taken from the server's own `usage` fields.
    ServerUsage,
    /// Token counts computed client-side with the model's tokenizer. Canonical.
    ClientTokenizer,
}

/// §4.4.6 — the `tokenization` block, required in every receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tokenization {
    /// REQUIRED, no default.
    pub method: TokenizationMethod,
    /// REQUIRED when `method = client_tokenizer`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokenizer_sha256: Option<String>,
    /// REQUIRED.
    pub counts_special_tokens: bool,
    /// REQUIRED.
    pub counts_prompt_echo: bool,
}

impl Tokenization {
    /// The canonical method: the model's own tokenizer, applied identically to
    /// the measured server and its comparator.
    ///
    /// # Errors
    /// When `tokenizer_sha256` is not a 64-character lowercase hex digest.
    pub fn client_tokenizer(
        tokenizer_sha256: impl Into<String>,
        counts_special_tokens: bool,
        counts_prompt_echo: bool,
    ) -> Result<Self, String> {
        let sha = tokenizer_sha256.into();
        let block = Self {
            method: TokenizationMethod::ClientTokenizer,
            tokenizer_sha256: Some(sha),
            counts_special_tokens,
            counts_prompt_echo,
        };
        block.validate()?;
        Ok(block)
    }

    /// Server-reported `usage` counts. Legal, but two servers' `usage` fields
    /// are two implementations' opinions — §4.4.6 prefers `client_tokenizer`.
    #[must_use]
    pub fn server_usage(counts_special_tokens: bool, counts_prompt_echo: bool) -> Self {
        Self {
            method: TokenizationMethod::ServerUsage,
            tokenizer_sha256: None,
            counts_special_tokens,
            counts_prompt_echo,
        }
    }

    /// §4.4.6 schema check.
    ///
    /// # Errors
    /// When `client_tokenizer` carries no digest, or the digest is malformed,
    /// or `server_usage` carries a digest it cannot have produced.
    pub fn validate(&self) -> Result<(), String> {
        match self.method {
            TokenizationMethod::ClientTokenizer => {
                let sha = self.tokenizer_sha256.as_deref().ok_or_else(|| {
                    "tokenization.tokenizer_sha256 is REQUIRED when method = client_tokenizer"
                        .to_string()
                })?;
                if sha.len() != 64 || !sha.bytes().all(|b| b.is_ascii_hexdigit()) {
                    return Err(format!(
                        "tokenization.tokenizer_sha256 must be 64 hex characters, got {:?}",
                        sha
                    ));
                }
                Ok(())
            }
            TokenizationMethod::ServerUsage => {
                if self.tokenizer_sha256.is_some() {
                    return Err(
                        "tokenization.tokenizer_sha256 is meaningless when method = server_usage"
                            .to_string(),
                    );
                }
                Ok(())
            }
        }
    }

    /// Poka-yoke for transports: a declared method the transport cannot honour
    /// is refused at construction, not silently downgraded at measure time.
    ///
    /// # Errors
    /// When the declared method and the available counting machinery disagree.
    pub fn require_counter(&self, has_client_counter: bool) -> Result<(), String> {
        match (self.method, has_client_counter) {
            (TokenizationMethod::ClientTokenizer, false) => Err(
                "tokenization.method = client_tokenizer but no client TokenCounter was supplied"
                    .to_string(),
            ),
            (TokenizationMethod::ServerUsage, true) => Err(
                "tokenization.method = server_usage but a client TokenCounter was supplied; \
                 declare client_tokenizer or drop the counter"
                    .to_string(),
            ),
            _ => Ok(()),
        }
    }
}

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

    #[test]
    fn client_tokenizer_without_a_digest_is_refused() {
        let bad = Tokenization {
            method: TokenizationMethod::ClientTokenizer,
            tokenizer_sha256: None,
            counts_special_tokens: true,
            counts_prompt_echo: false,
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn client_tokenizer_digest_must_be_hex64() {
        assert!(Tokenization::client_tokenizer("deadbeef", true, false).is_err());
        let good = "a".repeat(64);
        assert!(Tokenization::client_tokenizer(good, true, false).is_ok());
    }

    #[test]
    fn server_usage_may_not_carry_a_tokenizer_digest() {
        let bad = Tokenization {
            method: TokenizationMethod::ServerUsage,
            tokenizer_sha256: Some("b".repeat(64)),
            counts_special_tokens: false,
            counts_prompt_echo: false,
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn declared_method_and_available_counter_must_agree() {
        let ct = Tokenization::client_tokenizer("c".repeat(64), true, false).expect("valid digest");
        assert!(ct.require_counter(false).is_err());
        assert!(ct.require_counter(true).is_ok());

        let su = Tokenization::server_usage(true, false);
        assert!(su.require_counter(true).is_err());
        assert!(su.require_counter(false).is_ok());
    }

    #[test]
    fn tokenization_block_round_trips_with_method_present() {
        let t = Tokenization::server_usage(true, false);
        let j = serde_json::to_string(&t).expect("serialize");
        assert!(j.contains("\"method\":\"server_usage\""), "{j}");
        assert!(!j.contains("tokenizer_sha256"), "{j}");
        let back: Tokenization = serde_json::from_str(&j).expect("deserialize");
        assert_eq!(back, t);
    }

    /// `method` has no default: a block missing it must fail to deserialize.
    #[test]
    fn tokenization_without_method_is_schema_fatal() {
        let j = r#"{"counts_special_tokens":true,"counts_prompt_echo":false}"#;
        let r: Result<Tokenization, _> = serde_json::from_str(j);
        assert!(r.is_err(), "absent method must be fatal, not defaulted");
    }
}
