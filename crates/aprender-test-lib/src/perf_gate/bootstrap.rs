//! §4.4.4 — bootstrap percentile confidence intervals.
//!
//! Percentile method, 10 000 resamples, seed 2026, resampling **whole
//! requests**. Tokens within a request are not independent, so resampling
//! tokens (or per-token latencies) would understate the interval by pretending
//! each token is its own observation.
//!
//! BCa is deliberately not implemented: §4.4.4 rejects it as an undocumented
//! degree of freedom at this dispersion.
//!
//! The PRNG is a `SplitMix64` written out in full rather than taken from a
//! crate. §4.4.4 requires the interval to be *reproducible from the retained
//! samples*, and a dependency's internal stream is free to change across a
//! semver-compatible bump — which would silently move every published interval.
//! The stream is pinned by [`tests::splitmix64_stream_is_pinned`].

use serde::{Deserialize, Serialize};

use super::join::{Ratio, RatioMethod};
use super::metrics::{percentile, RequestSample};
use super::protocol::{BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED};

/// `SplitMix64`, exactly as published by Steele et al. Deterministic across
/// platforms: `u64` wrapping arithmetic only, no floats in the state.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Seed the generator.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next 64 bits.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform index in `[0, n)` by Lemire's multiply-shift. Unbiased enough for
    /// resampling and, unlike modulo, free of the low-bit bias that would skew
    /// which requests get picked.
    pub fn index_below(&mut self, n: usize) -> usize {
        debug_assert!(n > 0, "index_below(0) is undefined");
        ((u128::from(self.next_u64()) * n as u128) >> 64) as usize
    }
}

/// A bootstrap percentile interval, with everything needed to re-derive it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BootstrapCi {
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
    /// The seed. In the receipt, per §4.4.4.
    pub seed: u64,
    /// Whole requests, always. Recorded so the unit of resampling is on the page.
    pub resampling_unit: &'static str,
    /// Observations resampled.
    pub n: usize,
}

/// A statistic of a set of whole requests, e.g. [`super::metrics::agg_tok_s`].
///
/// A function pointer rather than a generic `F: Fn(..)`: a fn item passed to a
/// generic higher-ranked bound makes rustc infer a fresh lifetime and reject the
/// call, so every call site would need a wrapping closure — which clippy then
/// correctly flags as redundant. The pointer type takes the fn item directly.
pub type Statistic = fn(&[RequestSample]) -> f64;

/// §4.4.4 — bootstrap percentile CI for any statistic of a set of whole requests.
///
/// `statistic` is applied to the observed samples for the point estimate and to
/// each resample for the distribution. Passing
/// [`super::metrics::agg_tok_s`] gives the aggregate's interval; passing a
/// median gives the median's.
///
/// Returns `None` for fewer than two observations, where a bootstrap interval is
/// not defined. Returning a degenerate `[x, x]` there would read as a
/// measurement of impossible precision.
pub fn bootstrap_ci(
    samples: &[RequestSample],
    confidence: f64,
    statistic: Statistic,
) -> Option<BootstrapCi> {
    bootstrap_ci_with(
        samples,
        confidence,
        BOOTSTRAP_RESAMPLES,
        BOOTSTRAP_SEED,
        statistic,
    )
}

/// [`bootstrap_ci`] with the resample count and seed spelled out. The public
/// entry point pins both to the §4.4.4 values; this exists for the tests that
/// prove the pinning matters.
pub fn bootstrap_ci_with(
    samples: &[RequestSample],
    confidence: f64,
    resamples: usize,
    seed: u64,
    statistic: Statistic,
) -> Option<BootstrapCi> {
    let n = samples.len();
    if n < 2 || resamples == 0 || !(0.0..1.0).contains(&confidence) {
        return None;
    }

    let mut rng = SplitMix64::new(seed);
    let mut draws = Vec::with_capacity(resamples);
    // One reusable buffer: resampling WHOLE requests means cloning records, and
    // 10 000 fresh allocations of n records is the difference between a CI that
    // is cheap enough to always compute and one people turn off.
    let mut resample: Vec<RequestSample> = Vec::with_capacity(n);
    for _ in 0..resamples {
        resample.clear();
        for _ in 0..n {
            resample.push(samples[rng.index_below(n)].clone());
        }
        draws.push(statistic(&resample));
    }
    draws.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let alpha = 1.0 - confidence;
    let lower = percentile(&draws, alpha / 2.0)?;
    let upper = percentile(&draws, 1.0 - alpha / 2.0)?;

    Some(BootstrapCi {
        point: statistic(samples),
        lower,
        upper,
        confidence,
        resamples,
        seed,
        resampling_unit: "whole_request",
        n,
    })
}

/// [`bootstrap_ci`] specialised to §4.4.3's `agg_tok_s`.
///
/// A named wrapper rather than a bare function reference at each call site:
/// passing `agg_tok_s` directly makes rustc infer a fresh lifetime instead of
/// the higher-ranked one the bound wants, and the resulting error is opaque.
///
/// Returns `None` under the same conditions as [`bootstrap_ci`].
#[must_use]
pub fn bootstrap_agg_tok_s_ci(samples: &[RequestSample], confidence: f64) -> Option<BootstrapCi> {
    bootstrap_ci(samples, confidence, super::metrics::agg_tok_s)
}

/// PP-LLAMA-001 v3.0 §4.3 — the **request-unit** estimator: a paired percentile
/// bootstrap of the ratio of a per-request statistic across the two lanes.
///
/// # What "paired" means here, said explicitly
///
/// The two lanes issue their own requests; there is no per-request pairing key,
/// and there cannot be one — request 7 of the subject and request 7 of the
/// comparator are not the same event. "Paired" is therefore *pairing of the
/// resample index*: draw `k` runs one resample of the subject lane and one of
/// the comparator lane from **one** `SplitMix64(2026)` stream, and the ratio of
/// the two statistics is draw `k` of the ratio distribution. The lanes are
/// independent within a draw; what is shared is the seed and the draw index, so
/// the whole distribution is reproducible from the retained samples of both
/// lanes and nothing else (§4.4.4's requirement, applied to two lanes).
///
/// # The verdict statistic
///
/// The **5th percentile** of the ratio draws — a one-sided 95% lower bound.
/// Not the 2.5th: P-5 asks "is the lower bound at or above `1 − δ`", which is a
/// one-sided question, and taking `alpha/2` there would report a looser bound
/// as if it were the same guarantee.
///
/// Returns `None` when either lane has fewer than two retained requests, when
/// `confidence` is not in `(0, 1)`, or when the comparator's statistic is not
/// positive — a ratio with a zero denominator is not a large ratio.
#[must_use]
pub fn paired_ratio_lcb(
    subject: &[RequestSample],
    comparator: &[RequestSample],
    statistic: Statistic,
    confidence: f64,
) -> Option<Ratio> {
    let draws = paired_ratio_draws(
        subject,
        comparator,
        statistic,
        BOOTSTRAP_RESAMPLES,
        BOOTSTRAP_SEED,
        confidence,
    )?;
    let denominator = statistic(comparator);
    Some(Ratio {
        point: statistic(subject) / denominator,
        lcb95: percentile(&draws, 1.0 - confidence),
        method: RatioMethod::PairedPercentileBootstrap,
        n: subject.len() + comparator.len(),
    })
}

/// The sorted ratio draws behind [`paired_ratio_lcb`].
///
/// Public so a test can assert *which* percentile the bound is, rather than
/// re-implementing the draw loop and proving only that two copies of the same
/// code agree.
///
/// Returns `None` under the same conditions as [`paired_ratio_lcb`].
#[must_use]
pub fn paired_ratio_draws(
    subject: &[RequestSample],
    comparator: &[RequestSample],
    statistic: Statistic,
    resamples: usize,
    seed: u64,
    confidence: f64,
) -> Option<Vec<f64>> {
    let (n_s, n_c) = (subject.len(), comparator.len());
    if n_s < 2 || n_c < 2 || resamples == 0 || !(0.0..1.0).contains(&confidence) {
        return None;
    }
    if statistic(comparator) <= 0.0 {
        return None;
    }
    let mut rng = SplitMix64::new(seed);
    let mut draws = Vec::with_capacity(resamples);
    let mut lane_s: Vec<RequestSample> = Vec::with_capacity(n_s);
    let mut lane_c: Vec<RequestSample> = Vec::with_capacity(n_c);
    for _ in 0..resamples {
        lane_s.clear();
        for _ in 0..n_s {
            lane_s.push(subject[rng.index_below(n_s)].clone());
        }
        lane_c.clear();
        for _ in 0..n_c {
            lane_c.push(comparator[rng.index_below(n_c)].clone());
        }
        let denominator = statistic(&lane_c);
        if denominator > 0.0 {
            draws.push(statistic(&lane_s) / denominator);
        }
    }
    if draws.is_empty() {
        return None;
    }
    draws.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(draws)
}

/// §3 `dec` — the band's per-request decode rate: the **median** over retained
/// requests of `(completion_tokens − 1) / (e2e − ttft)`.
///
/// `0.0` for a set with no streamed request, where the quantity is undefined;
/// [`paired_ratio_lcb`] then refuses the ratio rather than dividing by it.
#[must_use]
pub fn median_decode_tok_s(samples: &[RequestSample]) -> f64 {
    let mut rates: Vec<f64> = samples
        .iter()
        .filter(|s| s.counts_toward_aggregate())
        .filter_map(RequestSample::decode_tok_s)
        .collect();
    rates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    percentile(&rates, 0.50).unwrap_or(0.0)
}

/// §3 `ttft` — p50 time-to-first-token, in milliseconds.
#[must_use]
pub fn ttft_p50_ms(samples: &[RequestSample]) -> f64 {
    ttft_percentile_ms(samples, 0.50)
}

/// §3 `itl_p95` — the 95th percentile of the **pooled** inter-token intervals.
///
/// Pooled across requests, not a percentile of per-request percentiles: the
/// tail this metric exists to expose is a few very late tokens, and averaging
/// each request's own p95 first hides exactly those.
#[must_use]
pub fn itl_p95_ms(samples: &[RequestSample]) -> f64 {
    let mut gaps: Vec<f64> = samples
        .iter()
        .filter(|s| s.counts_toward_aggregate())
        .flat_map(RequestSample::itl_gaps_ms)
        .collect();
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    percentile(&gaps, 0.95).unwrap_or(0.0)
}

fn ttft_percentile_ms(samples: &[RequestSample], p: f64) -> f64 {
    let mut v: Vec<f64> = samples
        .iter()
        .filter(|s| s.counts_toward_aggregate())
        .filter_map(RequestSample::ttft_ms)
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    percentile(&v, p).unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perf_gate::metrics::agg_tok_s;
    use crate::perf_gate::protocol::Outcome;

    fn sample(index: usize, start_s: f64, end_s: f64, tokens: u32) -> RequestSample {
        RequestSample {
            index,
            worker: index % 4,
            start_s,
            end_s,
            token_times_s: vec![start_s + 0.01, end_s],
            generated_tokens: tokens,
            prompt_tokens: 512,
            outcome: Outcome::Completed,
            in_flight_at_start: 4,
            drained: false,
        }
    }

    fn deck(n: usize) -> Vec<RequestSample> {
        (0..n)
            .map(|i| {
                let start = i as f64 * 0.25;
                let jitter = f64::from((i % 7) as u32) * 0.05;
                sample(i, start, start + 1.0 + jitter, 100 + (i % 5) as u32)
            })
            .collect()
    }

    /// The stream is pinned. If this reds, every previously published interval
    /// moved, and that must be a deliberate, versioned decision.
    #[test]
    fn splitmix64_stream_is_pinned() {
        let mut r = SplitMix64::new(2026);
        let got: Vec<u64> = (0..4).map(|_| r.next_u64()).collect();
        assert_eq!(
            got,
            vec![
                15_824_617_304_438_902_051,
                8_699_989_649_721_214_301,
                12_310_341_597_754_734_734,
                7_097_835_237_234_771_186,
            ],
            "SplitMix64(2026) stream changed"
        );
    }

    #[test]
    fn index_below_stays_in_range_and_covers_it() {
        let mut r = SplitMix64::new(BOOTSTRAP_SEED);
        let mut seen = [false; 5];
        for _ in 0..500 {
            let i = r.index_below(5);
            assert!(i < 5, "index {i} out of range");
            seen[i] = true;
        }
        assert!(seen.iter().all(|&s| s), "every index must be reachable");
    }

    /// §4.4.4's whole point: same samples + seed 2026 => the identical interval,
    /// bit for bit, twice.
    #[test]
    fn same_samples_and_seed_give_the_identical_interval_twice() {
        let s = deck(40);
        let a = bootstrap_agg_tok_s_ci(&s, 0.95).expect("n >= 2");
        let b = bootstrap_agg_tok_s_ci(&s, 0.95).expect("n >= 2");
        assert_eq!(
            a, b,
            "the CI must be reproducible from the retained samples"
        );
        assert_eq!(a.seed, 2026);
        assert_eq!(a.resamples, 10_000);
        assert_eq!(a.resampling_unit, "whole_request");
        assert_eq!(a.n, 40);
    }

    /// And the seed is load-bearing, not decoration: a different seed must move
    /// the interval, or "seed 2026" would be an unfalsifiable claim.
    #[test]
    fn a_different_seed_gives_a_different_interval() {
        let s = deck(40);
        let a = bootstrap_ci_with(&s, 0.95, 10_000, 2026, agg_tok_s).expect("n >= 2");
        let b = bootstrap_ci_with(&s, 0.95, 10_000, 2027, agg_tok_s).expect("n >= 2");
        assert_eq!(
            a.point, b.point,
            "the point estimate does not depend on the seed"
        );
        assert!(
            (a.lower - b.lower).abs() > f64::EPSILON || (a.upper - b.upper).abs() > f64::EPSILON,
            "seed had no effect: {a:?} vs {b:?}"
        );
    }

    #[test]
    fn the_interval_brackets_the_point_estimate() {
        let s = deck(60);
        let ci = bootstrap_agg_tok_s_ci(&s, 0.95).expect("n >= 2");
        assert!(ci.lower <= ci.point, "{ci:?}");
        assert!(ci.point <= ci.upper, "{ci:?}");
        assert!(
            ci.lower < ci.upper,
            "a non-degenerate sample needs width: {ci:?}"
        );
    }

    /// Whole requests, not tokens. Resampling n whole records from n records
    /// must be able to draw the same record twice — that is what makes it a
    /// bootstrap. A permutation would give zero width.
    #[test]
    fn resampling_is_with_replacement_over_whole_requests() {
        let mut rng = SplitMix64::new(BOOTSTRAP_SEED);
        let n = 8;
        let mut counts = vec![0_usize; n];
        for _ in 0..n {
            counts[rng.index_below(n)] += 1;
        }
        assert!(
            counts.iter().any(|&c| c >= 2),
            "a with-replacement draw of n from n must duplicate: {counts:?}"
        );
    }

    /// A wider spread of per-request behaviour must widen the interval. An
    /// undersized or noisy `n` should FAIL a gate by widening, never pass
    /// silently (§4.4.2).
    #[test]
    fn more_dispersion_widens_the_interval() {
        let tight: Vec<RequestSample> = (0..40)
            .map(|i| sample(i, i as f64 * 0.25, i as f64 * 0.25 + 1.0, 100))
            .collect();
        let loose: Vec<RequestSample> = (0..40)
            .map(|i| {
                let start = i as f64 * 0.25;
                let dur = if i % 2 == 0 { 0.2 } else { 4.0 };
                sample(i, start, start + dur, 100)
            })
            .collect();
        let a = bootstrap_agg_tok_s_ci(&tight, 0.95).expect("n >= 2");
        let b = bootstrap_agg_tok_s_ci(&loose, 0.95).expect("n >= 2");
        assert!(
            (b.upper - b.lower) > (a.upper - a.lower),
            "dispersed: {:?} must be wider than tight: {:?}",
            b,
            a
        );
    }

    #[test]
    fn fewer_than_two_observations_has_no_interval() {
        assert!(bootstrap_agg_tok_s_ci(&[], 0.95).is_none());
        assert!(bootstrap_agg_tok_s_ci(&deck(1), 0.95).is_none());
        assert!(bootstrap_agg_tok_s_ci(&deck(2), 0.95).is_some());
    }

    #[test]
    fn a_nonsense_confidence_has_no_interval() {
        let s = deck(10);
        assert!(bootstrap_ci(&s, 1.0, agg_tok_s).is_none());
        assert!(bootstrap_ci(&s, -0.1, agg_tok_s).is_none());
    }

    /// A lane whose per-request rate is `rate` tok/s, `n` requests.
    fn lane(n: usize, rate: f64, jitter: f64) -> Vec<RequestSample> {
        (0..n)
            .map(|i| {
                let start = i as f64 * 0.25;
                // Distinct per-request rates: a median over a handful of
                // repeated values is a step function, and its bootstrap
                // percentile would be identical under every seed — which would
                // make the seed test pass for the wrong reason.
                let r = rate + jitter * (((i * 7 + 3) % 23) as f64 / 23.0 - 0.5);
                // 128 tokens; 127 gaps at 1/r seconds each.
                let step = 1.0 / r;
                let times: Vec<f64> = (0..128)
                    .map(|k| start + 0.05 + f64::from(k) * step)
                    .collect();
                let end = times[127] + 0.01;
                RequestSample {
                    index: i,
                    worker: i % 4,
                    start_s: start,
                    end_s: end,
                    token_times_s: times,
                    generated_tokens: 128,
                    prompt_tokens: 512,
                    outcome: Outcome::Completed,
                    in_flight_at_start: 1,
                    drained: false,
                }
            })
            .collect()
    }

    /// §4.4.4 applied to two lanes: the same two sample sets and seed 2026 give
    /// the identical bound, bit for bit, twice.
    #[test]
    fn paired_ratio_lcb_is_reproducible_bit_for_bit_at_seed_2026() {
        let subject = lane(30, 100.0, 3.0);
        let comparator = lane(30, 90.0, 3.0);
        let a = paired_ratio_lcb(&subject, &comparator, median_decode_tok_s, 0.95)
            .expect("both lanes have n >= 2");
        let b = paired_ratio_lcb(&subject, &comparator, median_decode_tok_s, 0.95)
            .expect("both lanes have n >= 2");
        assert_eq!(a, b, "the bound must be reproducible from the samples");
        assert_eq!(BOOTSTRAP_SEED, 2026);
        assert_eq!(BOOTSTRAP_RESAMPLES, 10_000);
        assert_eq!(a.method, RatioMethod::PairedPercentileBootstrap);
        assert_eq!(a.n, 60, "both lanes' retained requests");

        // And the seed is load-bearing: a different stream moves the bound.
        let other = paired_ratio_draws(
            &subject,
            &comparator,
            median_decode_tok_s,
            10_000,
            2027,
            0.95,
        )
        .expect("draws");
        let mine = paired_ratio_draws(
            &subject,
            &comparator,
            median_decode_tok_s,
            10_000,
            2026,
            0.95,
        )
        .expect("draws");
        assert_ne!(
            percentile(&other, 0.05),
            percentile(&mine, 0.05),
            "seed 2026 must not be decoration"
        );
    }

    /// P-5 is a ONE-SIDED question. Taking `alpha/2` would report a looser
    /// bound under the same name.
    #[test]
    fn lcb95_is_the_fifth_percentile_not_the_2_5th() {
        let subject = lane(30, 100.0, 8.0);
        let comparator = lane(30, 90.0, 8.0);
        let draws = paired_ratio_draws(
            &subject,
            &comparator,
            median_decode_tok_s,
            BOOTSTRAP_RESAMPLES,
            BOOTSTRAP_SEED,
            0.95,
        )
        .expect("draws");
        let bound = paired_ratio_lcb(&subject, &comparator, median_decode_tok_s, 0.95)
            .expect("bound")
            .lcb95
            .expect("lcb95");
        let p05 = percentile(&draws, 0.05).expect("p05");
        let p025 = percentile(&draws, 0.025).expect("p025");
        assert_eq!(bound, p05, "the bound is the 5th percentile");
        assert_ne!(
            p05, p025,
            "the two percentiles must differ, or this test proves nothing"
        );
        assert!(p025 < p05, "the 2.5th percentile is the looser bound");
    }

    /// A lane divided by itself is 1.0 exactly, and its bound brackets it from
    /// below. If the point estimate drifted off 1.0 the statistic would be
    /// resample-dependent, which it must not be.
    #[test]
    fn identical_lanes_give_point_one() {
        let l = lane(30, 100.0, 4.0);
        for statistic in [
            median_decode_tok_s as Statistic,
            ttft_p50_ms as Statistic,
            itl_p95_ms as Statistic,
        ] {
            let r = paired_ratio_lcb(&l, &l, statistic, 0.95).expect("n >= 2");
            assert_eq!(r.point, 1.0, "a lane against itself is parity");
            let lcb = r.lcb95.expect("bounded");
            assert!(lcb <= r.point, "{r:?}");
            assert!(lcb > 0.0, "{r:?}");
        }
    }

    /// The point estimate tracks the lanes: a faster subject raises it, and the
    /// direction is subject-over-comparator, never the reverse.
    #[test]
    fn the_ratio_is_subject_over_comparator() {
        let fast = lane(30, 120.0, 2.0);
        let slow = lane(30, 60.0, 2.0);
        let up = paired_ratio_lcb(&fast, &slow, median_decode_tok_s, 0.95).expect("n >= 2");
        let down = paired_ratio_lcb(&slow, &fast, median_decode_tok_s, 0.95).expect("n >= 2");
        assert!(up.point > 1.5, "{up:?}");
        assert!(down.point < 0.7, "{down:?}");
        assert!(
            (up.point * down.point - 1.0).abs() < 1e-9,
            "{up:?} {down:?}"
        );
    }

    /// A lane with one retained request supports no bootstrap, and a lane whose
    /// statistic is zero is not a denominator.
    #[test]
    fn a_degenerate_lane_has_no_paired_bound() {
        let ok = lane(30, 100.0, 2.0);
        assert!(paired_ratio_lcb(&ok[..1], &ok, median_decode_tok_s, 0.95).is_none());
        assert!(paired_ratio_lcb(&ok, &ok[..1], median_decode_tok_s, 0.95).is_none());
        assert!(paired_ratio_lcb(&ok, &ok, median_decode_tok_s, 1.0).is_none());

        let unstreamed: Vec<RequestSample> = ok
            .iter()
            .map(|s| RequestSample {
                token_times_s: Vec::new(),
                ..s.clone()
            })
            .collect();
        assert!(
            paired_ratio_lcb(&ok, &unstreamed, median_decode_tok_s, 0.95).is_none(),
            "a zero denominator is not a large ratio"
        );
    }

    /// The three request-unit statistics are the §3 definitions, computed over
    /// completed requests only.
    #[test]
    fn the_request_unit_statistics_are_the_section_3_definitions() {
        let l = lane(4, 100.0, 0.0);
        // 128 tokens at 100 tok/s: 127 gaps of 10 ms, decode = 127/1.27 = 100.
        assert!(
            (median_decode_tok_s(&l) - 100.0).abs() < 1e-6,
            "{}",
            median_decode_tok_s(&l)
        );
        assert!((ttft_p50_ms(&l) - 50.0).abs() < 1e-6, "{}", ttft_p50_ms(&l));
        assert!((itl_p95_ms(&l) - 10.0).abs() < 1e-6, "{}", itl_p95_ms(&l));
        assert_eq!(median_decode_tok_s(&[]), 0.0);
        assert_eq!(ttft_p50_ms(&[]), 0.0);
        assert_eq!(itl_p95_ms(&[]), 0.0);
    }
}
