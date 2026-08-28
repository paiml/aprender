//! §4.4.3 — the metric definitions, computed from retained per-request samples.
//!
//! The whole point of this file is that `agg_tok_s` is a **wall-clock** quantity
//! and the mean of per-request rates is a different number. `mean_of_rates` is
//! implemented here *deliberately*, next to it, so the difference is asserted in
//! a test rather than argued about in review. It is never used to produce a
//! receipt figure.

use serde::{Deserialize, Serialize};

use super::protocol::Outcome;

/// One retained per-request sample (§4.4.5). Every time is an offset in seconds
/// from a single band-wide origin, so the wall-clock span is computable across
/// workers without reconstructing anything.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RequestSample {
    /// Monotone index in issue order.
    pub index: usize,
    /// Which closed-loop worker issued it.
    pub worker: usize,
    /// Offset of the request start from the band origin.
    pub start_s: f64,
    /// Offset of the request's completion (or abandonment) from the band origin.
    pub end_s: f64,
    /// Arrival offsets of each streamed token. Empty when not streaming.
    #[serde(default)]
    pub token_times_s: Vec<f64>,
    /// Generated (completion) tokens, counted per the receipt's `tokenization` block.
    pub generated_tokens: u32,
    /// Prompt tokens, when the server reports them.
    #[serde(default)]
    pub prompt_tokens: u32,
    /// How the request ended.
    pub outcome: Outcome,
    /// Concurrent requests in flight at the instant this one was issued.
    /// The direct, per-request evidence that the client was actually concurrent.
    #[serde(default)]
    pub in_flight_at_start: usize,
    /// True when this request completed after the measurement window closed,
    /// i.e. during the §4.4.7 drain.
    #[serde(default)]
    pub drained: bool,
}

impl RequestSample {
    /// §4.4.3 `ttft_ms` — request start to first token byte at the client.
    /// `None` when no token was ever observed.
    #[must_use]
    pub fn ttft_ms(&self) -> Option<f64> {
        self.token_times_s
            .first()
            .map(|t| (t - self.start_s) * 1000.0)
    }

    /// §4.4.3 `decode_tok_s` for this request:
    /// `(generated tokens - 1) / (last token time - first token time)`.
    ///
    /// `None` when fewer than two tokens were observed, where the quantity is
    /// undefined rather than zero.
    #[must_use]
    pub fn decode_tok_s(&self) -> Option<f64> {
        if self.token_times_s.len() < 2 || self.generated_tokens < 2 {
            return None;
        }
        let first = self.token_times_s[0];
        let last = self.token_times_s[self.token_times_s.len() - 1];
        let span = last - first;
        if span <= 0.0 {
            return None;
        }
        Some(f64::from(self.generated_tokens - 1) / span)
    }

    /// §4.4.3 `itl_ms` — this request's inter-token gaps, for pooling.
    #[must_use]
    pub fn itl_gaps_ms(&self) -> Vec<f64> {
        self.token_times_s
            .windows(2)
            .map(|w| (w[1] - w[0]) * 1000.0)
            .collect()
    }

    /// True when this sample contributes to `agg_tok_s`'s numerator (§4.4.3:
    /// completed and non-truncated).
    #[must_use]
    pub fn counts_toward_aggregate(&self) -> bool {
        self.outcome == Outcome::Completed
    }
}

/// §4.4.3 — one band's metrics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BandMetrics {
    /// Fixed concurrency `c`.
    pub concurrency: usize,
    /// (Σ generated tokens over completed, non-truncated sampled requests)
    /// ÷ (last completion − first request start). **Wall-clock.**
    pub agg_tok_s: f64,
    /// Median across sampled requests of per-request `(tokens-1)/(last-first)`.
    pub decode_tok_s: f64,
    /// p50 of `ttft_ms`.
    pub ttft_p50_ms: f64,
    /// p95 of `ttft_ms`.
    pub ttft_p95_ms: f64,
    /// p50 of the pooled inter-token gaps.
    pub itl_p50_ms: f64,
    /// p95 of the pooled inter-token gaps.
    pub itl_p95_ms: f64,
    /// Requests issued inside the window.
    pub requested: usize,
    /// Requests that completed.
    pub completed: usize,
    /// Requests that hit the 120 s hard timeout.
    pub timeouts: usize,
    /// Requests abandoned at the drain deadline (§4.4.7).
    pub truncated: usize,
    /// Requests that failed for any other reason.
    pub errors: usize,
    /// Σ generated tokens over the requests in the numerator.
    pub tokens_total: u64,
    /// The denominator actually used, in seconds. Present so a reader can
    /// re-derive `agg_tok_s` without the samples.
    pub span_s: f64,
}

/// §4.4.3 — linear-interpolated percentile of an ascending slice.
///
/// Defined once, in [`super::drain`], and re-exported here so the §4.4.3 metric
/// code keeps its `metrics::percentile` path. The two modules shipped
/// byte-identical copies on their respective branches; one of them had to go.
pub use super::drain::percentile;

fn sorted(mut v: Vec<f64>) -> Vec<f64> {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v
}

fn median(values: Vec<f64>) -> f64 {
    percentile(&sorted(values), 0.50).unwrap_or(0.0)
}

/// §4.4.3 `agg_tok_s`, exactly as specified.
///
/// Numerator: generated tokens over completed, non-truncated samples.
/// Denominator: last completion minus first request start, where "first request
/// start" is over **every** sampled request (§4.4.7) — including ones that
/// timed out, because they occupied the server for that time.
///
/// Returns `0.0` when the span is non-positive; a band with no elapsed time
/// produced no evidence and must not read as infinite throughput.
#[must_use]
pub fn agg_tok_s(samples: &[RequestSample]) -> f64 {
    let (tokens, span) = aggregate_terms(samples);
    if span <= 0.0 {
        return 0.0;
    }
    tokens as f64 / span
}

/// The numerator and denominator of [`agg_tok_s`], separately, so a receipt can
/// carry both and a reader can check the division.
#[must_use]
pub fn aggregate_terms(samples: &[RequestSample]) -> (u64, f64) {
    if samples.is_empty() {
        return (0, 0.0);
    }
    let tokens: u64 = samples
        .iter()
        .filter(|s| s.counts_toward_aggregate())
        .map(|s| u64::from(s.generated_tokens))
        .sum();
    let first_start = samples
        .iter()
        .map(|s| s.start_s)
        .fold(f64::INFINITY, f64::min);
    let last_end = samples
        .iter()
        .filter(|s| s.counts_toward_aggregate())
        .map(|s| s.end_s)
        .fold(f64::NEG_INFINITY, f64::max);
    if !first_start.is_finite() || !last_end.is_finite() {
        return (tokens, 0.0);
    }
    (tokens, last_end - first_start)
}

/// The arithmetic mean of per-request token rates.
///
/// **This is not `agg_tok_s` and must never be reported as it.** It exists so
/// the test `agg_tok_s_is_wall_clock_not_the_mean_of_rates` can assert the two
/// differ on a fixture with hand-computed values. Under a
/// serialising server with idle gaps between requests, this number can be many
/// times the true aggregate.
#[must_use]
pub fn mean_of_rates(samples: &[RequestSample]) -> f64 {
    let rates: Vec<f64> = samples
        .iter()
        .filter(|s| s.counts_toward_aggregate())
        .filter_map(|s| {
            let dur = s.end_s - s.start_s;
            if dur > 0.0 {
                Some(f64::from(s.generated_tokens) / dur)
            } else {
                None
            }
        })
        .collect();
    if rates.is_empty() {
        return 0.0;
    }
    rates.iter().sum::<f64>() / rates.len() as f64
}

impl BandMetrics {
    /// Compute every §4.4.3 metric from one band's retained samples.
    #[must_use]
    pub fn from_samples(concurrency: usize, samples: &[RequestSample]) -> Self {
        let (tokens_total, span_s) = aggregate_terms(samples);
        let agg = if span_s > 0.0 {
            tokens_total as f64 / span_s
        } else {
            0.0
        };

        let ttfts = sorted(samples.iter().filter_map(RequestSample::ttft_ms).collect());
        let itls = sorted(
            samples
                .iter()
                .flat_map(RequestSample::itl_gaps_ms)
                .collect::<Vec<f64>>(),
        );
        let decodes: Vec<f64> = samples
            .iter()
            .filter_map(RequestSample::decode_tok_s)
            .collect();

        let count = |o: Outcome| samples.iter().filter(|s| s.outcome == o).count();

        Self {
            concurrency,
            agg_tok_s: agg,
            decode_tok_s: median(decodes),
            ttft_p50_ms: percentile(&ttfts, 0.50).unwrap_or(0.0),
            ttft_p95_ms: percentile(&ttfts, 0.95).unwrap_or(0.0),
            itl_p50_ms: percentile(&itls, 0.50).unwrap_or(0.0),
            itl_p95_ms: percentile(&itls, 0.95).unwrap_or(0.0),
            requested: samples.len(),
            completed: count(Outcome::Completed),
            timeouts: count(Outcome::Timeout),
            truncated: count(Outcome::AbandonedAtDrain),
            errors: count(Outcome::Failed),
            tokens_total,
            span_s,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a sample whose token arrivals are evenly spaced across
    /// `[start, end]`, so `decode_tok_s` and `itl` are hand-computable.
    fn sample(index: usize, start_s: f64, end_s: f64, tokens: u32) -> RequestSample {
        let n = tokens as usize;
        let token_times_s = if n == 0 {
            Vec::new()
        } else {
            let step = (end_s - start_s) / n as f64;
            (1..=n).map(|i| start_s + step * i as f64).collect()
        };
        RequestSample {
            index,
            worker: 0,
            start_s,
            end_s,
            token_times_s,
            generated_tokens: tokens,
            prompt_tokens: 0,
            outcome: Outcome::Completed,
            in_flight_at_start: 1,
            drained: false,
        }
    }

    /// THE fixture this ticket exists for.
    ///
    /// Four requests, 100 tokens each, run two-at-a-time:
    ///   r0 [0,1]  r1 [0,2]  r2 [1,3]  r3 [2,4]
    /// Wall-clock span = 4.0 s, tokens = 400  =>  agg_tok_s = 100.0
    /// Per-request rates = 100, 50, 50, 50    =>  mean       =  62.5
    #[test]
    fn agg_tok_s_is_wall_clock_not_the_mean_of_rates() {
        let s = vec![
            sample(0, 0.0, 1.0, 100),
            sample(1, 0.0, 2.0, 100),
            sample(2, 1.0, 3.0, 100),
            sample(3, 2.0, 4.0, 100),
        ];
        let (tokens, span) = aggregate_terms(&s);
        assert_eq!(tokens, 400);
        assert!((span - 4.0).abs() < 1e-12, "span={span}");

        let agg = agg_tok_s(&s);
        let mean = mean_of_rates(&s);
        assert!((agg - 100.0).abs() < 1e-9, "agg={agg}, want 100.0");
        assert!((mean - 62.5).abs() < 1e-9, "mean={mean}, want 62.5");
        assert!(
            (agg - mean).abs() > 1.0,
            "agg {agg} and mean-of-rates {mean} must not coincide"
        );
    }

    /// The dangerous direction: a serialising server with idle gaps. The mean of
    /// per-request rates reports a throughput the machine never delivered; the
    /// assertions below pin both figures, so this comment does not restate them.
    /// This is the shape of the number the epic exists to refuse.
    #[test]
    fn mean_of_rates_overstates_a_serialising_server() {
        let s = vec![
            sample(0, 0.0, 1.0, 100),
            sample(1, 2.0, 3.0, 100),
            sample(2, 4.0, 5.0, 100),
            sample(3, 6.0, 7.0, 100),
        ];
        let agg = agg_tok_s(&s);
        let mean = mean_of_rates(&s);
        assert!((agg - 400.0 / 7.0).abs() < 1e-9, "agg={agg}");
        assert!((mean - 100.0).abs() < 1e-9, "mean={mean}");
        assert!(mean > agg * 1.7, "mean {mean} must overstate agg {agg}");
    }

    /// §4.4.3: the numerator counts only completed, non-truncated requests, but
    /// the denominator starts at the FIRST request start whatever its outcome.
    #[test]
    fn timeouts_lengthen_the_span_but_add_no_tokens() {
        let mut timed_out = sample(0, 0.0, 1.0, 100);
        timed_out.outcome = Outcome::Timeout;
        timed_out.generated_tokens = 100; // partial output must not be credited
        let s = vec![timed_out, sample(1, 0.5, 2.5, 100)];

        let (tokens, span) = aggregate_terms(&s);
        assert_eq!(tokens, 100, "a timed-out request contributes no tokens");
        assert!(
            (span - 2.5).abs() < 1e-12,
            "span must start at 0.0, got {span}"
        );

        let m = BandMetrics::from_samples(2, &s);
        assert_eq!(m.requested, 2);
        assert_eq!(m.completed, 1);
        assert_eq!(m.timeouts, 1);
        assert!((m.agg_tok_s - 40.0).abs() < 1e-9, "{}", m.agg_tok_s);
    }

    #[test]
    fn decode_tok_s_is_the_median_of_per_request_rates() {
        // Token arrivals evenly spaced: 100 tokens over a 1.0 s request means
        // step 0.01 s, first at 0.01, last at 1.00 -> span 0.99, rate 99/0.99 = 100.
        let one = sample(0, 0.0, 1.0, 100);
        assert!((one.decode_tok_s().expect("two+ tokens") - 100.0).abs() < 1e-9);

        // Three requests at 100, 50, 25 tok/s -> median 50.
        let s = vec![
            sample(0, 0.0, 1.0, 100),
            sample(1, 0.0, 2.0, 100),
            sample(2, 0.0, 4.0, 100),
        ];
        let m = BandMetrics::from_samples(1, &s);
        assert!((m.decode_tok_s - 50.0).abs() < 1e-9, "{}", m.decode_tok_s);
    }

    #[test]
    fn single_token_request_has_no_decode_rate_and_no_gaps() {
        let s = sample(0, 0.0, 1.0, 1);
        assert_eq!(s.decode_tok_s(), None);
        assert!(s.itl_gaps_ms().is_empty());
        assert!(s.ttft_ms().is_some(), "one token still has a TTFT");
    }

    #[test]
    fn ttft_is_start_to_first_token() {
        let s = sample(0, 10.0, 11.0, 4); // step 0.25 -> first at 10.25
        assert!((s.ttft_ms().expect("has tokens") - 250.0).abs() < 1e-9);
    }

    #[test]
    fn itl_gaps_are_pooled_across_requests() {
        // r0: 4 tokens over 1.0 s -> 3 gaps of 250 ms
        // r1: 3 tokens over 3.0 s -> 2 gaps of 1000 ms
        let s = vec![sample(0, 0.0, 1.0, 4), sample(1, 0.0, 3.0, 3)];
        let pooled: Vec<f64> = s.iter().flat_map(RequestSample::itl_gaps_ms).collect();
        assert_eq!(
            pooled.len(),
            5,
            "3 + 2 gaps pooled, not 2 per-request means"
        );
        let m = BandMetrics::from_samples(2, &s);
        // sorted: 250,250,250,1000,1000 -> p50 = 250
        assert!((m.itl_p50_ms - 250.0).abs() < 1e-9, "{}", m.itl_p50_ms);
        assert!(m.itl_p95_ms > 900.0, "{}", m.itl_p95_ms);
    }

    #[test]
    fn percentile_of_nothing_is_none_not_zero() {
        assert_eq!(percentile(&[], 0.5), None);
        assert_eq!(percentile(&[7.0], 0.95), Some(7.0));
    }

    #[test]
    fn percentile_interpolates_between_order_statistics() {
        let v = vec![0.0, 10.0, 20.0, 30.0];
        assert_eq!(percentile(&v, 0.0), Some(0.0));
        assert_eq!(percentile(&v, 1.0), Some(30.0));
        assert_eq!(percentile(&v, 0.5), Some(15.0));
    }

    #[test]
    fn empty_band_is_zero_not_infinite() {
        let m = BandMetrics::from_samples(4, &[]);
        assert_eq!(m.agg_tok_s, 0.0);
        assert_eq!(m.requested, 0);
        assert_eq!(m.span_s, 0.0);
    }

    #[test]
    fn samples_round_trip_as_jsonl_rows() {
        let s = sample(3, 1.5, 2.5, 8);
        let line = serde_json::to_string(&s).expect("serialize");
        let back: RequestSample = serde_json::from_str(&line).expect("deserialize");
        assert_eq!(back, s);
    }
}
