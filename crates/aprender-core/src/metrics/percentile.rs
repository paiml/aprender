//! Latency percentile computation (CRUX-E-07).
//!
//! Pure-Rust nearest-rank percentile implementation with invariant classifiers.
//! Used by `apr bench --percentiles 50,95,99` to report latency P50/P95/P99.
//!
//! Contract: `contracts/crux-E-07-v1.yaml` §`latency_percentile`.
//! arXiv grounding: arXiv:2505.02502 — deployment-framework latency capability.

use std::cmp::Ordering;

/// Outcome of computing a single percentile (CRUX-E-07 FALSIFY-005).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PercentileOutcome {
    /// Percentile computed successfully (value in ms).
    Ok(f64),
    /// Input sample set was empty.
    EmptySamples,
    /// Percentile point outside `(0.0, 100.0]`.
    InvalidPercentile(f64),
    /// A sample was non-finite (NaN or ±inf).
    NonFiniteSample,
    /// A sample was negative (wall-clock latency cannot be negative).
    NegativeSample(f64),
}

/// Outcome of computing a ladder of percentiles (monotonicity-aware).
#[derive(Debug, Clone, PartialEq)]
pub enum PercentileLadderOutcome {
    /// All percentiles computed; monotonically non-decreasing as p increases.
    Ok(Vec<f64>),
    /// One of the sub-computations failed — carries the first failing outcome.
    SubFailure(PercentileOutcome),
    /// Percentile points not strictly increasing (e.g., `[95, 50]`).
    PointsNotSorted,
    /// Percentiles computed but monotonicity `p99 >= p95 >= p50` violated — bug signal.
    MonotonicityViolated { points: Vec<f64>, values: Vec<f64> },
}

const PERCENTILE_MIN_EXCLUSIVE: f64 = 0.0;
const PERCENTILE_MAX_INCLUSIVE: f64 = 100.0;

/// Compute a single percentile using the nearest-rank method.
///
/// Given samples `x_1..x_n` sorted ascending, the percentile `p ∈ (0, 100]`
/// is the value at rank `ceil(p/100 * n)` (1-indexed).
///
/// Nearest-rank was chosen over linear interpolation to match
/// `benchmarks/benchmark_serving.py` (vLLM) and llama.cpp convention.
///
/// # Invariants (verified by unit tests below)
/// - `p > 0 && p <= 100` → `Ok(v)` where `v` is one of the input samples
/// - `p1 < p2` → `compute_percentile(xs, p1) <= compute_percentile(xs, p2)`
/// - Empty samples → `EmptySamples`
/// - NaN/Inf/negative → distinct `Outcome` variant (no silent pass)
#[must_use]
pub fn compute_percentile(samples: &[f64], percentile: f64) -> PercentileOutcome {
    if samples.is_empty() {
        return PercentileOutcome::EmptySamples;
    }
    if !percentile.is_finite()
        || percentile <= PERCENTILE_MIN_EXCLUSIVE
        || percentile > PERCENTILE_MAX_INCLUSIVE
    {
        return PercentileOutcome::InvalidPercentile(percentile);
    }
    for &s in samples {
        if !s.is_finite() {
            return PercentileOutcome::NonFiniteSample;
        }
        if s < 0.0 {
            return PercentileOutcome::NegativeSample(s);
        }
    }

    let mut sorted: Vec<f64> = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let n = sorted.len() as f64;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let rank_1idx = (percentile / PERCENTILE_MAX_INCLUSIVE * n).ceil() as usize;
    let rank_0idx = rank_1idx.saturating_sub(1).min(sorted.len() - 1);
    PercentileOutcome::Ok(sorted[rank_0idx])
}

/// Compute a ladder of percentiles and enforce monotonicity.
///
/// `points` MUST be strictly ascending (e.g., `[50.0, 95.0, 99.0]`).
/// Returns `Ok(values)` where `values[i]` is the percentile for `points[i]`,
/// or a `SubFailure` / `PointsNotSorted` / `MonotonicityViolated` outcome.
#[must_use]
pub fn compute_percentile_ladder(samples: &[f64], points: &[f64]) -> PercentileLadderOutcome {
    for window in points.windows(2) {
        if window[0] >= window[1] {
            return PercentileLadderOutcome::PointsNotSorted;
        }
    }
    let mut values = Vec::with_capacity(points.len());
    for &p in points {
        match compute_percentile(samples, p) {
            PercentileOutcome::Ok(v) => values.push(v),
            other => return PercentileLadderOutcome::SubFailure(other),
        }
    }
    for window in values.windows(2) {
        if window[0] > window[1] {
            return PercentileLadderOutcome::MonotonicityViolated {
                points: points.to_vec(),
                values,
            };
        }
    }
    PercentileLadderOutcome::Ok(values)
}

// ─── Falsification classifiers (CRUX-E-07 evidence_discharged_by) ───

/// Classify: nearest-rank p100 equals max(samples).
#[must_use]
pub fn classify_p100_is_max(samples: &[f64]) -> bool {
    match compute_percentile(samples, 100.0) {
        PercentileOutcome::Ok(v) => samples
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
            .eq(&v),
        _ => false,
    }
}

/// Classify: monotonicity holds for `[50, 95, 99]` on all-positive samples.
#[must_use]
pub fn classify_ladder_monotone(samples: &[f64]) -> bool {
    matches!(
        compute_percentile_ladder(samples, &[50.0, 95.0, 99.0]),
        PercentileLadderOutcome::Ok(_)
    )
}

/// Classify: empty input maps to `EmptySamples` (not panic, not zero).
#[must_use]
pub fn classify_empty_is_distinct() -> bool {
    matches!(
        compute_percentile(&[], 50.0),
        PercentileOutcome::EmptySamples
    )
}

/// Classify: NaN sample maps to `NonFiniteSample`.
#[must_use]
pub fn classify_nan_sample_rejected() -> bool {
    matches!(
        compute_percentile(&[1.0, f64::NAN, 3.0], 50.0),
        PercentileOutcome::NonFiniteSample
    )
}

/// Classify: negative sample maps to `NegativeSample`.
#[must_use]
pub fn classify_negative_sample_rejected() -> bool {
    matches!(
        compute_percentile(&[1.0, -2.0, 3.0], 50.0),
        PercentileOutcome::NegativeSample(_)
    )
}

/// Classify: out-of-range percentile point maps to `InvalidPercentile`.
#[must_use]
pub fn classify_invalid_percentile_rejected() -> bool {
    matches!(
        compute_percentile(&[1.0, 2.0, 3.0], 0.0),
        PercentileOutcome::InvalidPercentile(_)
    ) && matches!(
        compute_percentile(&[1.0, 2.0, 3.0], 100.01),
        PercentileOutcome::InvalidPercentile(_)
    )
}

/// Classify: unsorted ladder points rejected.
#[must_use]
pub fn classify_unsorted_points_rejected() -> bool {
    matches!(
        compute_percentile_ladder(&[1.0, 2.0, 3.0], &[95.0, 50.0]),
        PercentileLadderOutcome::PointsNotSorted
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn p50_of_sorted_1_to_10_is_5() {
        let xs: Vec<f64> = (1..=10).map(f64::from).collect();
        match compute_percentile(&xs, 50.0) {
            PercentileOutcome::Ok(v) => assert!(approx_eq(v, 5.0), "p50 = {v}"),
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    #[test]
    fn p99_of_100_samples_is_99th_element() {
        let xs: Vec<f64> = (1..=100).map(f64::from).collect();
        match compute_percentile(&xs, 99.0) {
            PercentileOutcome::Ok(v) => assert!(approx_eq(v, 99.0), "p99 = {v}"),
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    #[test]
    fn p100_equals_max() {
        let xs = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        assert!(classify_p100_is_max(&xs));
    }

    #[test]
    fn ladder_50_95_99_monotone_on_positive() {
        let xs: Vec<f64> = (1..=100).map(f64::from).collect();
        match compute_percentile_ladder(&xs, &[50.0, 95.0, 99.0]) {
            PercentileLadderOutcome::Ok(vs) => {
                assert_eq!(vs.len(), 3);
                assert!(vs[0] <= vs[1]);
                assert!(vs[1] <= vs[2]);
            }
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    #[test]
    fn empty_samples_distinct_outcome() {
        assert!(matches!(
            compute_percentile(&[], 50.0),
            PercentileOutcome::EmptySamples
        ));
    }

    #[test]
    fn nan_sample_rejected() {
        assert!(classify_nan_sample_rejected());
    }

    #[test]
    fn negative_sample_rejected() {
        assert!(classify_negative_sample_rejected());
    }

    #[test]
    fn infinite_sample_rejected() {
        assert!(matches!(
            compute_percentile(&[1.0, f64::INFINITY, 3.0], 50.0),
            PercentileOutcome::NonFiniteSample
        ));
    }

    #[test]
    fn percentile_zero_rejected() {
        assert!(matches!(
            compute_percentile(&[1.0, 2.0, 3.0], 0.0),
            PercentileOutcome::InvalidPercentile(_)
        ));
    }

    #[test]
    fn percentile_over_100_rejected() {
        assert!(matches!(
            compute_percentile(&[1.0, 2.0, 3.0], 100.01),
            PercentileOutcome::InvalidPercentile(_)
        ));
        assert!(matches!(
            compute_percentile(&[1.0, 2.0, 3.0], 200.0),
            PercentileOutcome::InvalidPercentile(_)
        ));
    }

    #[test]
    fn percentile_nan_point_rejected() {
        assert!(matches!(
            compute_percentile(&[1.0, 2.0, 3.0], f64::NAN),
            PercentileOutcome::InvalidPercentile(_)
        ));
    }

    #[test]
    fn ladder_unsorted_rejected() {
        assert!(classify_unsorted_points_rejected());
    }

    #[test]
    fn ladder_duplicate_rejected() {
        assert!(matches!(
            compute_percentile_ladder(&[1.0, 2.0, 3.0], &[50.0, 50.0]),
            PercentileLadderOutcome::PointsNotSorted
        ));
    }

    #[test]
    fn single_sample_is_every_percentile() {
        for p in [1.0, 50.0, 95.0, 99.0, 100.0] {
            match compute_percentile(&[42.0], p) {
                PercentileOutcome::Ok(v) => assert!(approx_eq(v, 42.0), "p{p} = {v}"),
                other => panic!("expected Ok, got {other:?}"),
            }
        }
    }

    #[test]
    fn duplicate_samples_handled() {
        let xs = vec![5.0; 10];
        for p in [50.0, 95.0, 99.0, 100.0] {
            match compute_percentile(&xs, p) {
                PercentileOutcome::Ok(v) => assert!(approx_eq(v, 5.0)),
                other => panic!("expected Ok, got {other:?}"),
            }
        }
    }

    #[test]
    fn unsorted_input_still_correct() {
        let xs = vec![9.0, 1.0, 5.0, 3.0, 7.0];
        match compute_percentile(&xs, 50.0) {
            PercentileOutcome::Ok(v) => assert!(approx_eq(v, 5.0)),
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    #[test]
    fn ladder_sub_failure_propagates() {
        let outcome = compute_percentile_ladder(&[1.0, -2.0, 3.0], &[50.0, 95.0]);
        assert!(matches!(
            outcome,
            PercentileLadderOutcome::SubFailure(PercentileOutcome::NegativeSample(_))
        ));
    }

    #[test]
    fn small_sample_count_5_p99() {
        let xs = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        match compute_percentile(&xs, 99.0) {
            PercentileOutcome::Ok(v) => assert!(approx_eq(v, 50.0), "p99 of 5 samples = {v}"),
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    #[test]
    fn classify_empty_is_distinct_passes() {
        assert!(classify_empty_is_distinct());
    }

    #[test]
    fn classify_invalid_percentile_rejected_passes() {
        assert!(classify_invalid_percentile_rejected());
    }

    #[test]
    fn classify_ladder_monotone_passes() {
        let xs: Vec<f64> = (1..=50).map(f64::from).collect();
        assert!(classify_ladder_monotone(&xs));
    }
}
