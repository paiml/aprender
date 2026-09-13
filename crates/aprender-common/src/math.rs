//! Shared mathematical functions for the Batuta stack.
//!
//! Provides common math operations (statistics, special functions) used across
//! pmat, trueno, aprender, and trueno-viz.

// =============================================================================
// ERROR FUNCTION (Abramowitz & Stegun approximation)
// =============================================================================

/// Compute the error function erf(x) using the Abramowitz & Stegun approximation.
///
/// Maximum error: |ε| < 1.5 × 10⁻⁷
///
/// # Examples
/// ```
/// use batuta_common::math::erf;
/// assert!((erf(0.0) - 0.0).abs() < 1e-6);
/// assert!((erf(1.0) - 0.842_700_8).abs() < 1e-5);
/// assert!((erf(-1.0) + 0.842_700_8).abs() < 1e-5);
/// ```
#[must_use]
pub fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun formula 7.1.26
    const A1: f64 = 0.254_829_592;
    const A2: f64 = -0.284_496_736;
    const A3: f64 = 1.421_413_741;
    const A4: f64 = -1.453_152_027;
    const A5: f64 = 1.061_405_429;
    const P: f64 = 0.327_591_1;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();

    sign * y
}

/// Compute erf(x) with f32 precision.
///
/// Convenience wrapper for f32 callers; internally delegates to the f64 version.
#[must_use]
pub fn erf_f32(x: f32) -> f32 {
    erf(f64::from(x)) as f32
}

// =============================================================================
// HIGH-ACCURACY ERROR FUNCTION (W. J. Cody rational Chebyshev approximation)
// =============================================================================
//
// Why this exists alongside `erf` above (plan 01-09, amendment A-03):
//
// `erf` uses Abramowitz & Stegun 7.1.26, whose max ABSOLUTE error is 1.5e-7. That is
// fine for the statistics callers it was written for, but it is NOT sufficient for
// `gelu_exact`, the FFN activation on the ENC-03 parity path. Measured: A&S-based
// gelu_exact deviates from a high-precision reference by up to 4.77e-7 absolute, and —
// far worse — by 129 f32 ulps of the LOCAL VALUE near x = -2.67, because
// `1 + erf(x/sqrt(2))` cancels catastrophically in the negative tail (two ~1.0
// quantities leaving ~0.0077). That is a SYSTEMATIC bias, not noise, and it would
// compound across six FFN layers.
//
// `erf` above is deliberately left untouched so its existing callers keep their exact
// current behavior.
//
// Reference: W. J. Cody, "Rational Chebyshev Approximation for the Error Function",
// Math. Comp. 23 (1969), 631-637 — the CALERF algorithm also used by fdlibm/Cephes.
// Accuracy is near machine precision in f64 (~1e-16 relative).

/// 1/sqrt(pi), used by the large-argument asymptotic branch.
const SQRPI: f64 = 0.564_189_583_547_756_3;

/// Cody's branch threshold between the direct erf series and the erfc branches.
const CODY_THRESH: f64 = 0.468_75;

/// Above this magnitude erfc(x) underflows to 0 in f64.
const CODY_XBIG: f64 = 26.543;

// The six coefficient tables below are transcribed from Cody's published CALERF
// values, written as the shortest round-tripping f64 literals so the stored bit
// patterns are exactly the published constants.
//
// `unreadable_literal` is allowed here rather than adding digit separators: these are
// a transcribed numerical table, and any grouping that satisfies `unreadable_literal`
// trips `inconsistent_digit_grouping` (the integer and fractional parts have
// different digit counts per row). Ungrouped literals keep them diffable against the
// published source, which is the property that actually matters for a coefficient table.
#[allow(clippy::unreadable_literal)]
const CODY_A: [f64; 5] = [
    3.1611237438705655,
    113.86415415105016,
    377.485237685302,
    3209.3775891384694,
    0.18577770618460315,
];
#[allow(clippy::unreadable_literal)]
const CODY_B: [f64; 4] = [
    23.601290952344122,
    244.02463793444417,
    1282.6165260773723,
    2844.236833439171,
];
#[allow(clippy::unreadable_literal)]
const CODY_C: [f64; 9] = [
    0.5641884969886701,
    8.883149794388377,
    66.11919063714163,
    298.6351381974001,
    881.952221241769,
    1712.0476126340707,
    2051.0783778260716,
    1230.3393547979972,
    2.1531153547440383e-8,
];
#[allow(clippy::unreadable_literal)]
const CODY_D: [f64; 8] = [
    15.744926110709835,
    117.6939508913125,
    537.1811018620099,
    1621.3895745666903,
    3290.7992357334597,
    4362.619090143247,
    3439.3676741437216,
    1230.3393548037495,
];
#[allow(clippy::unreadable_literal)]
const CODY_P: [f64; 6] = [
    0.30532663496123236,
    0.36034489994980445,
    0.12578172611122926,
    0.016083785148742275,
    0.0006587491615298378,
    0.016315387137302097,
];
#[allow(clippy::unreadable_literal)]
const CODY_Q: [f64; 5] = [
    2.568520192289822,
    1.8729528499234604,
    0.5279051029514285,
    0.06051834131244132,
    0.0023352049762686918,
];

/// erf(x) for |x| <= `CODY_THRESH`, via the direct rational approximation.
fn cody_erf_small(x: f64) -> f64 {
    let z = x * x;
    let mut xnum = CODY_A[4] * z;
    let mut xden = z;
    for i in 0..3 {
        xnum = (xnum + CODY_A[i]) * z;
        xden = (xden + CODY_B[i]) * z;
    }
    x * (xnum + CODY_A[3]) / (xden + CODY_B[3])
}

/// erfc(y) for y > `CODY_THRESH` (y strictly positive).
///
/// Uses Cody's middle branch for y <= 4 and the asymptotic branch beyond. Both apply
/// Cody's split-exponential trick (`ysq` truncated to 1/16) so that `exp(-y*y)` is
/// evaluated without losing low-order bits.
fn cody_erfc_pos(y: f64) -> f64 {
    if y >= CODY_XBIG {
        return 0.0;
    }

    let result = if y <= 4.0 {
        let mut xnum = CODY_C[8] * y;
        let mut xden = y;
        for i in 0..7 {
            xnum = (xnum + CODY_C[i]) * y;
            xden = (xden + CODY_D[i]) * y;
        }
        (xnum + CODY_C[7]) / (xden + CODY_D[7])
    } else {
        let z = 1.0 / (y * y);
        let mut xnum = CODY_P[5] * z;
        let mut xden = z;
        for i in 0..4 {
            xnum = (xnum + CODY_P[i]) * z;
            xden = (xden + CODY_Q[i]) * z;
        }
        let r = z * (xnum + CODY_P[4]) / (xden + CODY_Q[4]);
        (SQRPI - r) / y
    };

    // Split exp(-y^2) = exp(-ysq^2) * exp(-del) with ysq truncated to a 1/16 grid.
    let ysq = (y * 16.0).trunc() / 16.0;
    let del = (y - ysq) * (y + ysq);
    (-ysq * ysq).exp() * (-del).exp() * result
}

/// High-accuracy error function, accurate to near f64 machine precision.
///
/// Use this instead of [`erf`] wherever the result feeds a numerical-parity gate.
/// [`erf`] (Abramowitz & Stegun 7.1.26) is only accurate to 1.5e-7 absolute.
///
/// # Examples
/// ```
/// use batuta_common::math::erf_precise;
/// assert!((erf_precise(1.0) - 0.842_700_792_949_714_9).abs() < 1e-15);
/// assert!((erf_precise(-1.0) + 0.842_700_792_949_714_9).abs() < 1e-15);
/// assert_eq!(erf_precise(0.0), 0.0);
/// ```
#[must_use]
pub fn erf_precise(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    let y = x.abs();
    if y <= CODY_THRESH {
        return cody_erf_small(x);
    }
    let v = 1.0 - cody_erfc_pos(y);
    if x < 0.0 { -v } else { v }
}

/// High-accuracy complementary error function `erfc(x) = 1 - erf(x)`.
///
/// Computing `1.0 - erf(x)` directly loses precision for large positive `x`, where
/// erfc is tiny; this routine keeps full relative accuracy there. That matters for
/// `gelu_exact(x) = 0.5 * x * erfc(-x / sqrt(2))`, whose negative tail is exactly
/// that regime.
///
/// # Examples
/// ```
/// use batuta_common::math::erfc_precise;
/// assert!((erfc_precise(0.0) - 1.0).abs() < 1e-15);
/// // erfc stays accurate where 1 - erf(x) would cancel to nothing.
/// assert!((erfc_precise(3.0) - 2.209_049_699_858_544e-5).abs() < 1e-19);
/// ```
#[must_use]
pub fn erfc_precise(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    let y = x.abs();
    if y <= CODY_THRESH {
        return 1.0 - cody_erf_small(x);
    }
    if x > 0.0 {
        cody_erfc_pos(y)
    } else {
        2.0 - cody_erfc_pos(y)
    }
}

// =============================================================================
// STANDARD DEVIATION
// =============================================================================

/// Compute sample standard deviation of a slice (Bessel's correction, n-1).
///
/// Returns 0.0 if fewer than 2 elements.
///
/// # Examples
/// ```
/// use batuta_common::math::std_dev;
/// let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
/// assert!((std_dev(&data) - 2.138).abs() < 0.01);
/// assert_eq!(std_dev(&[1.0]), 0.0);
/// assert_eq!(std_dev(&[]), 0.0);
/// ```
#[must_use]
pub fn std_dev(samples: &[f64]) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    variance.sqrt()
}

/// Compute sample standard deviation for f32 data.
///
/// Returns 0.0 if fewer than 2 elements.
#[must_use]
pub fn std_dev_f32(samples: &[f32]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }
    let n = samples.len() as f32;
    let mean = samples.iter().sum::<f32>() / n;
    let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (n - 1.0);
    variance.sqrt()
}

/// Compute sample standard deviation given a pre-computed mean.
///
/// Useful when the mean has already been calculated separately.
#[must_use]
pub fn std_dev_with_mean(samples: &[f64], mean: f64) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }
    let variance =
        samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (samples.len() - 1) as f64;
    variance.sqrt()
}

/// Compute sample standard deviation for f32 data given a pre-computed mean.
#[must_use]
pub fn std_dev_f32_with_mean(samples: &[f32], mean: f32) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }
    let variance =
        samples.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (samples.len() - 1) as f32;
    variance.sqrt()
}

// =============================================================================
// COSINE SIMILARITY
// =============================================================================

/// Compute cosine similarity between two f32 vectors.
///
/// Returns 0.0 if either vector has zero norm.
///
/// # Examples
/// ```
/// use batuta_common::math::cosine_similarity_f32;
/// let a = [1.0f32, 0.0, 0.0];
/// let b = [0.0f32, 1.0, 0.0];
/// assert!((cosine_similarity_f32(&a, &b) - 0.0).abs() < 1e-6);
///
/// let c = [1.0f32, 2.0, 3.0];
/// assert!((cosine_similarity_f32(&c, &c) - 1.0).abs() < 1e-6);
/// ```
#[must_use]
pub fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Compute cosine similarity between two f64 vectors.
///
/// Returns 0.0 if either vector has zero norm.
#[must_use]
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

// =============================================================================
// USAGE PERCENT
// =============================================================================

/// Compute usage percentage from used/total byte counts.
///
/// Returns 0.0 if `total` is 0 (avoids divide-by-zero).
///
/// # Examples
/// ```
/// use batuta_common::math::usage_percent;
/// assert!((usage_percent(750, 1000) - 75.0).abs() < 1e-10);
/// assert_eq!(usage_percent(0, 0), 0.0);
/// assert!((usage_percent(1024, 4096) - 25.0).abs() < 1e-10);
/// ```
#[must_use]
pub fn usage_percent(used: u64, total: u64) -> f64 {
    if total == 0 {
        return 0.0;
    }
    (used as f64 / total as f64) * 100.0
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- erf ---

    #[test]
    fn test_erf_zero() {
        assert!((erf(0.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_erf_positive() {
        assert!((erf(1.0) - 0.842_700_793).abs() < 1e-6);
    }

    #[test]
    fn test_erf_negative_symmetry() {
        assert!((erf(-1.0) + erf(1.0)).abs() < 1e-10);
    }

    // --- erf_precise / erfc_precise (plan 01-09, amendment A-03) ---
    //
    // These are checked against an INDEPENDENTLY derived oracle (Maclaurin series for
    // |t| <= 2, Laplace continued fraction beyond), not against `erf` above — that
    // would only prove the two agree, which is precisely what must NOT be assumed.

    fn oracle_erf_series(t: f64) -> f64 {
        let mut u = t;
        let mut sum = t;
        let mut n = 1.0_f64;
        while n <= 200.0 {
            u *= -(t * t) / n;
            let add = u / (2.0 * n + 1.0);
            sum += add;
            if add == 0.0 || add.abs() < 1e-18 * sum.abs() {
                break;
            }
            n += 1.0;
        }
        sum * 2.0 / std::f64::consts::PI.sqrt()
    }

    fn oracle_erfc_cf(t: f64) -> f64 {
        let mut cf = 0.0_f64;
        let mut k = 80_i32;
        while k >= 1 {
            cf = (f64::from(k) / 2.0) / (t + cf);
            k -= 1;
        }
        (-t * t).exp() / std::f64::consts::PI.sqrt() / (t + cf)
    }

    fn oracle_erf(t: f64) -> f64 {
        let a = t.abs();
        let v = if a <= 2.0 {
            oracle_erf_series(a)
        } else {
            1.0 - oracle_erfc_cf(a)
        };
        if t < 0.0 { -v } else { v }
    }

    #[test]
    fn erf_precise_matches_independent_oracle_across_the_range() {
        let mut worst = 0.0_f64;
        let mut worst_at = 0.0_f64;
        for i in 0..=1200 {
            let x = -6.0 + 0.01 * f64::from(i);
            let d = (erf_precise(x) - oracle_erf(x)).abs();
            if d > worst {
                worst = d;
                worst_at = x;
            }
        }
        assert!(
            worst < 1e-14,
            "erf_precise deviates from the independent oracle by {worst:.3e} at x={worst_at}"
        );
    }

    #[test]
    fn erf_precise_is_far_more_accurate_than_the_abramowitz_stegun_erf() {
        // Pins WHY erf_precise was added: A&S is ~1e-7, Cody is ~1e-15.
        let mut worst_as = 0.0_f64;
        let mut worst_precise = 0.0_f64;
        for i in 0..=1200 {
            let x = -6.0 + 0.01 * f64::from(i);
            let want = oracle_erf(x);
            worst_as = worst_as.max((erf(x) - want).abs());
            worst_precise = worst_precise.max((erf_precise(x) - want).abs());
        }
        assert!(
            worst_as > 1e-9,
            "A&S erf unexpectedly accurate ({worst_as:.3e}) — the premise for erf_precise changed"
        );
        assert!(
            worst_precise * 1e6 < worst_as,
            "erf_precise (worst {worst_precise:.3e}) must be orders of magnitude better \
             than A&S erf (worst {worst_as:.3e})"
        );
    }

    #[test]
    fn erfc_precise_keeps_relative_accuracy_in_the_far_tail() {
        // The whole point: 1 - erf(x) cancels to nothing out here, erfc does not.
        for &(x, want) in &[
            (2.0_f64, 4.677_734_981_047_265e-3_f64),
            (3.0, 2.209_049_699_858_544e-5),
            (4.0, 1.541_725_790_028_002_6e-8),
            (5.0, 1.537_459_794_428_035_4e-12),
        ] {
            let got = erfc_precise(x);
            let rel = (got - want).abs() / want;
            assert!(
                rel < 1e-13,
                "erfc_precise({x}) = {got:e}, expected {want:e} (rel {rel:.3e})"
            );
        }
    }

    #[test]
    fn erfc_precise_and_erf_precise_are_consistent_and_symmetric() {
        for i in 0..=120 {
            let x = -6.0 + 0.1 * f64::from(i);
            assert!(
                (erfc_precise(x) - (1.0 - erf_precise(x))).abs() < 1e-14,
                "erfc_precise({x}) inconsistent with 1 - erf_precise({x})"
            );
            assert!(
                (erf_precise(-x) + erf_precise(x)).abs() < 1e-15,
                "erf_precise must be odd, failed at {x}"
            );
        }
        assert_eq!(erf_precise(0.0), 0.0);
        assert!((erfc_precise(0.0) - 1.0).abs() < 1e-15);
        assert!(erf_precise(f64::NAN).is_nan());
        assert!(erfc_precise(f64::NAN).is_nan());
        assert_eq!(erfc_precise(30.0), 0.0, "erfc underflows to 0 past XBIG");
    }

    #[test]
    fn test_erf_large() {
        assert!((erf(5.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_erf_f32_matches() {
        let f32_val = erf_f32(1.0_f32);
        let f64_val = erf(1.0) as f32;
        assert!((f32_val - f64_val).abs() < 1e-6);
    }

    // --- std_dev ---

    #[test]
    fn test_std_dev_known_value() {
        // Sample std_dev with Bessel's correction (n-1):
        // Mean = 5.0, sum_sq_diff = 32, variance = 32/7 ≈ 4.571, sd ≈ 2.138
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((std_dev(&data) - 2.138).abs() < 0.01);
    }

    #[test]
    fn test_std_dev_single_element() {
        assert_eq!(std_dev(&[42.0]), 0.0);
    }

    #[test]
    fn test_std_dev_empty() {
        assert_eq!(std_dev(&[]), 0.0);
    }

    #[test]
    fn test_std_dev_identical_values() {
        assert_eq!(std_dev(&[5.0, 5.0, 5.0, 5.0]), 0.0);
    }

    #[test]
    fn test_std_dev_f32() {
        let data: Vec<f32> = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((std_dev_f32(&data) - 2.138).abs() < 0.02);
    }

    #[test]
    fn test_std_dev_with_mean_matches() {
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let sd1 = std_dev(&data);
        let sd2 = std_dev_with_mean(&data, mean);
        assert!((sd1 - sd2).abs() < 1e-10);
    }

    // --- cosine_similarity ---

    #[test]
    fn test_cosine_identical() {
        let a = [1.0, 2.0, 3.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_opposite() {
        let a = [1.0, 0.0];
        let b = [-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_zero_vector() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_f32() {
        let a = [1.0f32, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0];
        assert!(cosine_similarity_f32(&a, &b).abs() < 1e-6);
    }

    // --- usage_percent ---

    #[test]
    fn test_usage_percent_normal() {
        assert!((usage_percent(750, 1000) - 75.0).abs() < 1e-10);
    }

    #[test]
    fn test_usage_percent_zero_total() {
        assert_eq!(usage_percent(0, 0), 0.0);
    }

    #[test]
    fn test_usage_percent_full() {
        assert!((usage_percent(1000, 1000) - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_usage_percent_empty() {
        assert!((usage_percent(0, 1000) - 0.0).abs() < 1e-10);
    }
}
