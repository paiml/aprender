//! PP-LLAMA-001 v3.0 PP-3 / PP-22 / P-5 — the join key, and the only shape a
//! ratio may take.
//!
//! # Why a ratio is a type and not an `f64`
//!
//! `scripts/lib/perf_receipt.py` used to emit `agg_ratio` and `decode_ratio` as
//! bare scalars beside a band. A bare scalar cannot say which comparator run it
//! divided by, whether that run was the same invocation, whether its band was
//! the same `c`, its window the same length, or whether either lane timed out.
//! Every one of those is a way the number is wrong, and none of them is visible
//! in the number.
//!
//! So there is no `From<f64> for Ratio` and no public field assignment that
//! makes one: [`super::drain::ComparatorStatus::Measured`] is constructible only
//! through [`super::drain::BandInput::join`], which refuses
//!
//! - a comparator lane from a different `run_id` (PP-3 — "shares `run_id`"),
//! - a [`JoinKey`] mismatch on any of the fourteen fields (PP-22),
//! - `timeouts > 0` on either lane (PP-5),
//! - a comparator configured with `n_batch = 1` (§5.3's recorded dissent: a
//!   `-b 1` comparator is a cripple, and once manufactured a 2.39×
//!   overstatement).
//!
//! # The two estimators (§4.3)
//!
//! | unit | metrics | estimator | [`RatioMethod`] |
//! |---|---|---|---|
//! | replicate (window statistics) | `agg`, `prefill` | mean of per-replicate `ln(subject/comparator)`, one-sided t lower bound, exponentiated | [`RatioMethod::ReplicateTLower`] |
//! | request (per-request statistics) | `dec`, `ttft`, `itl_p95` | paired percentile bootstrap, 10 000 resamples, seed 2026 | [`RatioMethod::PairedPercentileBootstrap`] |
//!
//! `lcb95` is `None` — not `0.0`, and not the point estimate — when the design
//! cannot support a bound (`n < 5` replicates, fewer than two requests). §4.3:
//! "`n = 3` sizes an effect and bounds no variance."

use serde::{Deserialize, Serialize};

use super::drain::BandInput;
use super::receipt::{ReceiptInput, TokenCountingMethod, Workload};

/// Which estimator produced a [`Ratio`] (§4.3). On the wire as a snake_case
/// token so a reader can tell a window statistic from a request statistic
/// without knowing the metric's name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RatioMethod {
    /// §4.3 request unit: paired percentile bootstrap, 5th percentile.
    PairedPercentileBootstrap,
    /// §4.3 replicate unit: one-sided t lower bound on the mean log-ratio.
    ReplicateTLower,
}

impl RatioMethod {
    /// The wire token.
    #[must_use]
    pub fn wire_token(self) -> &'static str {
        match self {
            Self::PairedPercentileBootstrap => "paired_percentile_bootstrap",
            Self::ReplicateTLower => "replicate_t_lower",
        }
    }
}

/// P-5 — one metric's ratio, with the bound the verdict is taken on.
///
/// `point` is `x_subject / x_comparator`. `lcb95` is the one-sided 95% lower
/// confidence bound; `None` means the design could not support one and the
/// ratio is REPORTING only.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ratio {
    /// The ratio on the observed data.
    pub point: f64,
    /// One-sided 95% lower bound, or `None` when the design cannot support one.
    pub lcb95: Option<f64>,
    /// Which estimator produced it.
    pub method: RatioMethod,
    /// Observations the estimator used — replicates or requests, per `method`.
    pub n: usize,
}

impl Ratio {
    /// A ratio with no bound, for a design that cannot support one (`n < 5`
    /// replicates). REPORTING only: P-5's verdict needs `lcb95`.
    #[must_use]
    pub fn reporting_only(point: f64, method: RatioMethod, n: usize) -> Self {
        Self {
            point,
            lcb95: None,
            method,
            n,
        }
    }

    /// P-5 — does this ratio PASS at non-inferiority margin `delta`?
    ///
    /// `false` when there is no bound: a ratio without a lower bound has not
    /// been shown to be anything, and "no evidence" is not a pass.
    #[must_use]
    pub fn passes(&self, delta: f64) -> bool {
        self.lcb95.is_some_and(|l| l >= 1.0 - delta)
    }
}

/// The estimators' return type. The same struct as [`Ratio`] under the name the
/// statistics modules use for it, so `bootstrap::paired_ratio_lcb` and
/// `replicate::log_ratio_lcb` produce a value that goes on the wire unchanged
/// rather than through a lossy conversion.
pub type RatioBound = Ratio;

/// P-3 — the three ratios a band may carry. `agg` is always present when the
/// band joined at all; `dec` and `prefill` are `None` when the lane could not
/// produce the metric (no streaming, no server timings).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BandRatios {
    /// Aggregate throughput ratio. Gated at c>1 (§7.2).
    pub agg: Ratio,
    /// Per-request decode ratio. Gated at c=1 (§7.2).
    pub dec: Option<Ratio>,
    /// Server-reported prefill ratio. Gated at c=1 (§7.2).
    pub prefill: Option<Ratio>,
}

/// PP-22 — the fourteen fields two bands must agree on before their numbers may
/// be divided.
///
/// Each field is a way two measurements can look comparable and not be. `c=4`
/// against `c=16` compares different offered loads; a 30 s window against a 60 s
/// one compares different amounts of thermal drift; `n_batch = 1` against a
/// served comparator compares a working server against a crippled one.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JoinKey {
    /// Which host.
    pub host: String,
    /// Which workload.
    pub workload: Workload,
    /// The band's concurrency `c`.
    pub band: u32,
    /// Which model.
    pub model: String,
    /// Which quantization.
    pub quant: String,
    /// How tokens were counted (§4.4.6).
    pub tokenization: TokenCountingMethod,
    /// The measurement window, in milliseconds.
    pub window_ms: u64,
    /// Replicates run.
    pub replicates: u32,
    /// Whether those replicates alternated.
    pub interleaved: bool,
    /// Comparator `-c` per slot. `None` when the lane did not report one.
    pub n_ctx_slot: Option<u32>,
    /// KV cache type, e.g. `f16`.
    pub kv_type: Option<String>,
    /// Flash attention.
    pub fa: Option<bool>,
    /// Comparator `-b`. `Some(1)` is refused outright (§5.3).
    pub n_batch: Option<u32>,
    /// Generated tokens per request.
    pub n_predict: u32,
}

impl JoinKey {
    /// Build the key for one band of one receipt.
    #[must_use]
    pub fn of(receipt: &ReceiptInput, band: &BandInput) -> Self {
        Self {
            host: receipt.provenance.host.clone(),
            workload: receipt.workload,
            band: band.concurrency,
            model: receipt.provenance.model.clone(),
            quant: receipt.provenance.quantization.clone(),
            tokenization: receipt.tokenization.method(),
            // PP-22 keys on the DECLARED window (the protocol's), never the measured
            // close instant: two lanes never close on the same millisecond.
            window_ms: receipt.protocol.window_ms,
            replicates: receipt.protocol.replicates,
            interleaved: receipt.protocol.interleaved,
            n_ctx_slot: band.lane.n_ctx_slot,
            kv_type: band.lane.kv_type.clone(),
            fa: band.lane.fa,
            n_batch: band.lane.n_batch,
            n_predict: receipt.protocol.n_predict,
        }
    }

    /// §5.3 — a comparator serving one request at a time is not serving the
    /// band. Checked on its own so a key can be rejected before it is compared
    /// with anything.
    ///
    /// # Errors
    /// When `n_batch == Some(1)`.
    pub fn refuse_cripple(&self) -> Result<(), String> {
        if self.n_batch == Some(1) {
            return Err(format!(
                "PP-22 join key at c={}: n_batch=1 — §5.3 refuses a `-b 1` comparator as a \
                 cripple; that configuration manufactured a 2.39x overstatement once \
                 (llama_pin.toml:129-165) and it is not a lane serving the band",
                self.band
            ));
        }
        Ok(())
    }

    /// PP-22 — refuse the join, naming **every** differing field.
    ///
    /// All of them, not the first: a caller told only "band differs" re-runs,
    /// discovers the window differs too, and re-runs again. Both keys are also
    /// checked for the `-b 1` cripple.
    ///
    /// # Errors
    /// When any field differs, or when either key is a `-b 1` comparator.
    pub fn refuse_mismatch(&self, other: &Self) -> Result<(), String> {
        self.refuse_cripple()?;
        other.refuse_cripple()?;
        let mut differing = Vec::new();
        let mut note = |name: &str, a: String, b: String| {
            if a != b {
                differing.push(format!("{name}: {a} != {b}"));
            }
        };
        note("host", self.host.clone(), other.host.clone());
        note(
            "workload",
            self.workload.wire_token().to_string(),
            other.workload.wire_token().to_string(),
        );
        note("band", self.band.to_string(), other.band.to_string());
        note("model", self.model.clone(), other.model.clone());
        note("quant", self.quant.clone(), other.quant.clone());
        note(
            "tokenization",
            self.tokenization.wire_token().to_string(),
            other.tokenization.wire_token().to_string(),
        );
        note(
            "window_ms",
            self.window_ms.to_string(),
            other.window_ms.to_string(),
        );
        note(
            "replicates",
            self.replicates.to_string(),
            other.replicates.to_string(),
        );
        note(
            "interleaved",
            self.interleaved.to_string(),
            other.interleaved.to_string(),
        );
        note(
            "n_ctx_slot",
            opt(self.n_ctx_slot.as_ref()),
            opt(other.n_ctx_slot.as_ref()),
        );
        note(
            "kv_type",
            opt(self.kv_type.as_ref()),
            opt(other.kv_type.as_ref()),
        );
        note("fa", opt(self.fa.as_ref()), opt(other.fa.as_ref()));
        note(
            "n_batch",
            opt(self.n_batch.as_ref()),
            opt(other.n_batch.as_ref()),
        );
        note(
            "n_predict",
            self.n_predict.to_string(),
            other.n_predict.to_string(),
        );
        if differing.is_empty() {
            return Ok(());
        }
        Err(format!(
            "PP-22 join refused: {} — two bands that differ on any join field are not two \
             measurements of the same thing, and their quotient is not a ratio",
            differing.join("; ")
        ))
    }
}

fn opt<T: std::fmt::Display>(v: Option<&T>) -> String {
    v.map_or_else(|| "null".to_string(), std::string::ToString::to_string)
}

#[cfg(test)]
mod tests {
    // The `<selftest-name>__<sentence>` spelling is load-bearing: PP-29's
    // `scripts/spec_conformance.sh` joins the §6 invariant table to the test
    // list on the prefix before the double underscore, so renaming these to
    // single-underscore snake case would silently unjoin the rows they arm.
    #![allow(non_snake_case)]
    use super::*;

    fn key(band: u32) -> JoinKey {
        JoinKey {
            host: "lambda".to_string(),
            workload: Workload::W1,
            band,
            model: "qwen2.5-coder-7b-apache-q4k-v1".to_string(),
            quant: "Q4_K_M".to_string(),
            tokenization: TokenCountingMethod::ClientTokenizer,
            window_ms: 60_000,
            replicates: 5,
            interleaved: true,
            n_ctx_slot: Some(1024),
            kv_type: Some("f16".to_string()),
            fa: Some(true),
            n_batch: Some(2048),
            n_predict: 128,
        }
    }

    /// PP-22 must-not-fire: identical keys join.
    #[test]
    fn join_ok__matching_keys_join() {
        assert!(key(4).refuse_mismatch(&key(4)).is_ok());
    }

    /// PP-22 must-fire, first spelling: two different offered loads.
    #[test]
    fn join_mismatch__c4_against_c16_is_refused() {
        let err = key(4).refuse_mismatch(&key(16)).expect_err("c differs");
        assert!(err.contains("band: 4 != 16"), "{err}");
        assert!(err.contains("PP-22"), "{err}");
    }

    /// PP-22 must-fire, second spelling: two different amounts of drift.
    #[test]
    fn joining_a_30s_window_with_a_60s_window_is_refused() {
        let short = JoinKey {
            window_ms: 30_000,
            ..key(4)
        };
        let err = short.refuse_mismatch(&key(4)).expect_err("window differs");
        assert!(err.contains("window_ms: 30000 != 60000"), "{err}");
    }

    /// PP-22 must-fire, third spelling: §5.3's recorded dissent.
    #[test]
    fn a_b1_comparator_is_refused_as_a_cripple() {
        let crippled = JoinKey {
            n_batch: Some(1),
            ..key(4)
        };
        let err = key(4)
            .refuse_mismatch(&crippled)
            .expect_err("-b 1 comparator");
        assert!(err.contains("cripple"), "{err}");
        // And it is refused even when BOTH lanes were crippled identically —
        // two crippled lanes agree on every field and are still not a parity
        // measurement.
        assert!(crippled.refuse_mismatch(&crippled).is_err());
    }

    /// Every differing field is named, not just the first one found.
    #[test]
    fn a_mismatch_names_every_differing_field() {
        let other = JoinKey {
            host: "gx10".to_string(),
            window_ms: 30_000,
            interleaved: false,
            kv_type: Some("q8_0".to_string()),
            fa: None,
            ..key(16)
        };
        let err = key(4).refuse_mismatch(&other).expect_err("many differ");
        for field in ["host", "band", "window_ms", "interleaved", "kv_type", "fa"] {
            assert!(err.contains(field), "{field} missing from: {err}");
        }
    }

    /// A `None` on one side and a value on the other is a difference, not a
    /// wildcard: "the comparator did not report `n_ctx_slot`" is exactly the
    /// case where the ratio must not be formed.
    #[test]
    fn an_absent_field_does_not_match_a_present_one() {
        let unreported = JoinKey {
            n_ctx_slot: None,
            ..key(4)
        };
        let err = key(4)
            .refuse_mismatch(&unreported)
            .expect_err("null vs 1024");
        assert!(err.contains("n_ctx_slot: 1024 != null"), "{err}");
    }

    /// P-5: a ratio with no bound has not been shown to be anything.
    #[test]
    fn a_ratio_without_a_bound_never_passes() {
        let reporting = Ratio::reporting_only(1.42, RatioMethod::ReplicateTLower, 3);
        assert!(!reporting.passes(0.0));
        assert!(
            !reporting.passes(0.5),
            "no bound is not a pass at any delta"
        );

        let bounded = Ratio {
            point: 1.02,
            lcb95: Some(1.005),
            method: RatioMethod::ReplicateTLower,
            n: 5,
        };
        assert!(bounded.passes(0.0), "lcb95 >= 1 - 0 passes at parity");

        let below = Ratio {
            lcb95: Some(0.98),
            ..bounded.clone()
        };
        assert!(!below.passes(0.0));
        assert!(below.passes(0.05), "delta 0.05 admits an lcb95 of 0.98");
    }

    #[test]
    fn ratio_method_wire_tokens_are_the_schema_spelling() {
        assert_eq!(
            RatioMethod::PairedPercentileBootstrap.wire_token(),
            "paired_percentile_bootstrap"
        );
        assert_eq!(
            RatioMethod::ReplicateTLower.wire_token(),
            "replicate_t_lower"
        );
        let j = serde_json::to_string(&RatioMethod::ReplicateTLower).expect("serialises");
        assert_eq!(j, "\"replicate_t_lower\"");
    }
}
