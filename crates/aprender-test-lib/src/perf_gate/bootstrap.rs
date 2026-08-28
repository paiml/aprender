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
}
