use super::super::super::super::*;
use proptest::prelude::*;

proptest! {
    /// Property test: Cov(X,X) = Var(X) (covariance with self equals variance)
    #[test]
    fn test_covariance_self_equals_variance(
        a in prop::collection::vec(-50.0f32..50.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let cov = va.covariance(&va).unwrap();
        let var = va.variance().unwrap();

        let tolerance = 1e-3 * var.abs().max(1e-5);
        prop_assert!(
            (cov - var).abs() < tolerance,
            "Cov(X,X) = {} != Var(X) = {}",
            cov, var
        );
    }

    /// Property test: Cov(X,Y) = Cov(Y,X) (symmetry/commutativity)
    #[test]
    fn test_covariance_symmetric(
        ab in prop::collection::vec((-50.0f32..50.0, -50.0f32..50.0), 1..100)
    ) {
        let a: Vec<f32> = ab.iter().map(|(x, _)| *x).collect();
        let b: Vec<f32> = ab.iter().map(|(_, y)| *y).collect();
        let va = Vector::from_slice(&a);
        let vb = Vector::from_slice(&b);

        let cov_ab = va.covariance(&vb).unwrap();
        let cov_ba = vb.covariance(&va).unwrap();

        let tolerance = 1e-4 * cov_ab.abs().max(1e-5);
        prop_assert!(
            (cov_ab - cov_ba).abs() < tolerance,
            "Cov(X,Y) = {} != Cov(Y,X) = {}",
            cov_ab, cov_ba
        );
    }

    /// Property test: Cov(aX, bY) = ab*Cov(X,Y) (bilinearity)
    #[test]
    fn test_covariance_bilinearity(
        ab in prop::collection::vec((-20.0f32..20.0, -20.0f32..20.0), 1..50),
        scale_a in -3.0f32..3.0,
        scale_b in -3.0f32..3.0
    ) {
        let a: Vec<f32> = ab.iter().map(|(x, _)| *x).collect();
        let b: Vec<f32> = ab.iter().map(|(_, y)| *y).collect();
        let va = Vector::from_slice(&a);
        let vb = Vector::from_slice(&b);

        let cov_original = va.covariance(&vb).unwrap();

        let scaled_a = va.scale(scale_a).unwrap();
        let scaled_b = vb.scale(scale_b).unwrap();
        let cov_scaled = scaled_a.covariance(&scaled_b).unwrap();

        let expected = scale_a * scale_b * cov_original;
        // Use relative tolerance accounting for compounding floating-point errors
        // Small covariances need larger relative tolerance due to precision limits
        let tolerance = 0.5 * expected.abs().max(1e-3);
        prop_assert!(
            (cov_scaled - expected).abs() < tolerance,
            "Cov({}*X, {}*Y) = {} != {}*{}*Cov(X,Y) = {}",
            scale_a, scale_b, cov_scaled, scale_a, scale_b, expected
        );
    }

    /// Property test: -1 <= rho(X,Y) <= 1 (correlation is bounded)
    #[test]
    fn test_correlation_bounded(
        ab in prop::collection::vec((-50.0f32..50.0, -50.0f32..50.0), 2..100)
    ) {
        let a: Vec<f32> = ab.iter().map(|(x, _)| *x).collect();
        let b: Vec<f32> = ab.iter().map(|(_, y)| *y).collect();

        // Ensure vectors are not constant
        let std_a: f32 = a.iter().map(|x| x * x).sum::<f32>() / a.len() as f32
                       - (a.iter().sum::<f32>() / a.len() as f32).powi(2);
        let std_b: f32 = b.iter().map(|y| y * y).sum::<f32>() / b.len() as f32
                       - (b.iter().sum::<f32>() / b.len() as f32).powi(2);

        if std_a < 1e-6 || std_b < 1e-6 {
            return Ok(());  // Skip constant vectors
        }

        let va = Vector::from_slice(&a);
        let vb = Vector::from_slice(&b);
        let corr = va.correlation(&vb).unwrap();

        prop_assert!(
            (-1.0 - 1e-5..=1.0 + 1e-5).contains(&corr),
            "correlation = {} not in range [-1, 1]",
            corr
        );
    }

    /// Property test: rho(X,Y) = rho(Y,X) (symmetry)
    #[test]
    fn test_correlation_symmetric(
        ab in prop::collection::vec((-50.0f32..50.0, -50.0f32..50.0), 2..100)
    ) {
        let a: Vec<f32> = ab.iter().map(|(x, _)| *x).collect();
        let b: Vec<f32> = ab.iter().map(|(_, y)| *y).collect();

        // Ensure vectors are not constant
        let std_a: f32 = a.iter().map(|x| x * x).sum::<f32>() / a.len() as f32
                       - (a.iter().sum::<f32>() / a.len() as f32).powi(2);
        let std_b: f32 = b.iter().map(|y| y * y).sum::<f32>() / b.len() as f32
                       - (b.iter().sum::<f32>() / b.len() as f32).powi(2);

        if std_a < 1e-6 || std_b < 1e-6 {
            return Ok(());  // Skip constant vectors
        }

        let va = Vector::from_slice(&a);
        let vb = Vector::from_slice(&b);

        let corr_ab = va.correlation(&vb).unwrap();
        let corr_ba = vb.correlation(&va).unwrap();

        let tolerance = 1e-5;
        prop_assert!(
            (corr_ab - corr_ba).abs() < tolerance,
            "rho(X,Y) = {} != rho(Y,X) = {}",
            corr_ab, corr_ba
        );
    }

    /// Property test: rho(X,X) = 1 (perfect self-correlation)
    #[test]
    fn test_correlation_self_is_one(
        a in prop::collection::vec(-50.0f32..50.0, 2..100)
    ) {
        // Ensure vector is not constant
        let std_a: f32 = a.iter().map(|x| x * x).sum::<f32>() / a.len() as f32
                       - (a.iter().sum::<f32>() / a.len() as f32).powi(2);

        if std_a < 1e-6 {
            return Ok(());  // Skip constant vectors
        }

        let va = Vector::from_slice(&a);
        let corr = va.correlation(&va).unwrap();

        prop_assert!(
            (corr - 1.0).abs() < 1e-5,
            "rho(X,X) = {} != 1.0",
            corr
        );
    }
}

// ========================================================================
// Property tests for zscore() - Z-score normalization
// ========================================================================

proptest! {
    /// Property test: zscore() produces mean ~ 0
    #[test]
    fn test_zscore_produces_zero_mean(
        a in prop::collection::vec(-100.0f32..100.0, 2..100)
    ) {
        // Ensure vector is not constant. Two-pass, because E[x^2] - E[x]^2
        // catastrophically cancels: on the input that broke this test in CI
        // (a = [75.395966, 75.466225]) it returns 0.001953125 against a true
        // variance of 0.001234085 -- 58% high. A skip criterion that wrong is
        // unreliable in both directions.
        let n = a.len() as f32;
        let mean_a: f32 = a.iter().sum::<f32>() / n;
        let var_a: f32 = a.iter().map(|x| (x - mean_a) * (x - mean_a)).sum::<f32>() / n;

        if var_a < 1e-6 {
            return Ok(());  // Skip constant vectors
        }

        let va = Vector::from_slice(&a);
        let z = va.zscore().unwrap();
        let mean = z.mean().unwrap();

        // The achievable precision here is NOT absolute. z_i = (x_i - mu) / sigma
        // carries a rounding residue of about eps*|x_i|, so |mean(z)| is bounded
        // by eps * max|a| / sigma -- which grows without limit as the data
        // approaches constant. The old fixed 1e-4 asserted a precision f32
        // cannot deliver: for the CI input, sigma = 0.0351 and max|a| = 75.47,
        // giving a bound of 2.56e-4, so the test was ~2.6x too tight and failed
        // at 1.086e-4 after 114 successes. The floor keeps the assertion
        // meaningful for well-conditioned inputs.
        let sigma = var_a.sqrt();
        let max_abs = a.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let tol = (10.0 * f32::EPSILON * max_abs / sigma).max(1e-4);

        prop_assert!(
            mean.abs() < tol,
            "zscore mean = {}, tolerance = {} (sigma = {}, max|a| = {})",
            mean, tol, sigma, max_abs
        );
    }
}

/// Regression, aprender CI 2026-08-28: this exact pair broke
/// `test_zscore_produces_zero_mean` after 114 successes. Two large,
/// nearly-equal values make sigma tiny relative to |a|, so the rounding
/// residue in mean(zscore(a)) is ~eps*max|a|/sigma = 2.56e-4 -- larger than
/// the fixed 1e-4 the test used to assert. Deterministic, so it cannot
/// depend on proptest drawing the seed again.
#[test]
fn zscore_zero_mean_tolerates_ill_conditioned_input() {
    let a = [75.395966_f32, 75.466225_f32];
    let n = a.len() as f32;
    let mean_a: f32 = a.iter().sum::<f32>() / n;
    let var_a: f32 = a.iter().map(|x| (x - mean_a) * (x - mean_a)).sum::<f32>() / n;
    let sigma = var_a.sqrt();
    let max_abs = a.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    let tol = (10.0 * f32::EPSILON * max_abs / sigma).max(1e-4);

    let mean = Vector::from_slice(&a).zscore().unwrap().mean().unwrap();

    // The point of the regression: the residue really does exceed the old
    // fixed bound, so this input would still fail the pre-fix assertion.
    assert!(
        tol > 1e-4,
        "input must be ill-conditioned enough to need a scaled tolerance, tol = {tol}"
    );
    assert!(mean.abs() < tol, "zscore mean = {mean}, tolerance = {tol}");
}

proptest! {
    /// Property test: zscore() produces stddev ~ 1
    #[test]
    fn test_zscore_produces_unit_stddev(
        a in prop::collection::vec(-100.0f32..100.0, 3..100)
    ) {
        // Ensure vector is not constant
        let std_a: f32 = a.iter().map(|x| x * x).sum::<f32>() / a.len() as f32
                       - (a.iter().sum::<f32>() / a.len() as f32).powi(2);

        if std_a < 1e-6 {
            return Ok(());  // Skip constant vectors
        }

        let va = Vector::from_slice(&a);
        let z = va.zscore().unwrap();
        let std = z.stddev().unwrap();

        // Use relaxed tolerance for floating-point precision, especially for small n
        prop_assert!(
            (std - 1.0).abs() < 1e-3,
            "zscore stddev = {}, expected ~ 1",
            std
        );
    }
}

proptest! {
    /// Property test: zscore() preserves correlation structure
    /// rho(zscore(X), zscore(Y)) = rho(X, Y)
    #[test]
    fn test_zscore_preserves_correlation(
        ab in prop::collection::vec((-50.0f32..50.0, -50.0f32..50.0), 2..100)
    ) {
        let a: Vec<f32> = ab.iter().map(|(x, _)| *x).collect();
        let b: Vec<f32> = ab.iter().map(|(_, y)| *y).collect();

        // Ensure vectors have sufficient variance for stable correlation
        // Small variances cause numerical instability in zscore normalization
        let std_a: f32 = a.iter().map(|x| x * x).sum::<f32>() / a.len() as f32
                       - (a.iter().sum::<f32>() / a.len() as f32).powi(2);
        let std_b: f32 = b.iter().map(|x| x * x).sum::<f32>() / b.len() as f32
                       - (b.iter().sum::<f32>() / b.len() as f32).powi(2);

        if std_a < 0.1 || std_b < 0.1 {
            return Ok(());  // Skip near-constant vectors (variance < 0.1)
        }

        let va = Vector::from_slice(&a);
        let vb = Vector::from_slice(&b);

        // Original correlation
        let corr_orig = va.correlation(&vb).unwrap();

        // Correlation after zscore
        let za = va.zscore().unwrap();
        let zb = vb.zscore().unwrap();
        let corr_zscore = za.correlation(&zb).unwrap();

        let tolerance = 1e-3;
        prop_assert!(
            (corr_orig - corr_zscore).abs() < tolerance,
            "rho(X,Y) = {} != rho(zscore(X), zscore(Y)) = {}",
            corr_orig, corr_zscore
        );
    }
}

// ========================================================================
// Property tests for minmax_normalize() - Min-max normalization
// ========================================================================

proptest! {
    /// Property test: minmax_normalize() produces min = 0
    #[test]
    fn test_minmax_normalize_produces_zero_min(
        a in prop::collection::vec(-100.0f32..100.0, 2..100)
    ) {
        // Ensure vector is not constant
        let min_a = a.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_a = a.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        if (max_a - min_a).abs() < 1e-6 {
            return Ok(());  // Skip constant vectors
        }

        let va = Vector::from_slice(&a);
        let normalized = va.minmax_normalize().unwrap();
        let min = normalized.min().unwrap();

        prop_assert!(
            min.abs() < 1e-4,
            "minmax min = {}, expected ~ 0",
            min
        );
    }
}

proptest! {
    /// Property test: minmax_normalize() produces max = 1
    #[test]
    fn test_minmax_normalize_produces_one_max(
        a in prop::collection::vec(-100.0f32..100.0, 2..100)
    ) {
        // Ensure vector is not constant
        let min_a = a.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_a = a.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        if (max_a - min_a).abs() < 1e-6 {
            return Ok(());  // Skip constant vectors
        }

        let va = Vector::from_slice(&a);
        let normalized = va.minmax_normalize().unwrap();
        let max = normalized.max().unwrap();

        prop_assert!(
            (max - 1.0).abs() < 1e-4,
            "minmax max = {}, expected ~ 1",
            max
        );
    }
}

proptest! {
    /// Property test: minmax_normalize() preserves order (monotonicity)
    /// If a\[i\] <= a\[j\], then normalized\[i\] <= normalized\[j\]
    #[test]
    fn test_minmax_normalize_preserves_order(
        a in prop::collection::vec(-100.0f32..100.0, 2..100)
    ) {
        // Ensure vector is not constant
        let min_a = a.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_a = a.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        if (max_a - min_a).abs() < 1e-6 {
            return Ok(());  // Skip constant vectors
        }

        let va = Vector::from_slice(&a);
        let normalized = va.minmax_normalize().unwrap();

        // Check that order is preserved for all pairs
        for i in 0..a.len() {
            for j in 0..a.len() {
                if a[i] <= a[j] {
                    prop_assert!(
                        normalized.data[i] <= normalized.data[j] + 1e-5,
                        "Order not preserved: a[{}]={} <= a[{}]={}, but norm[{}]={} > norm[{}]={}",
                        i, a[i], j, a[j], i, normalized.data[i], j, normalized.data[j]
                    );
                }
            }
        }
    }
}

// ========================================================================
// Property tests for clip() - Range clipping
// ========================================================================

proptest! {
    /// Property test: clip() produces values within [min_val, max_val]
    #[test]
    fn test_clip_within_bounds(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        min_val in -50.0f32..50.0,
        max_val in -50.0f32..50.0
    ) {
        // Ensure min <= max
        let (min_val, max_val) = if min_val <= max_val {
            (min_val, max_val)
        } else {
            (max_val, min_val)
        };

        let va = Vector::from_slice(&a);
        let clipped = va.clip(min_val, max_val).unwrap();

        // All values must be within [min_val, max_val]
        for &val in clipped.as_slice() {
            prop_assert!(
                (min_val..=max_val).contains(&val),
                "Value {} not in range [{}, {}]",
                val, min_val, max_val
            );
        }
    }
}

proptest! {
    /// Property test: clip() preserves order (monotonicity)
    /// If a\[i\] <= a\[j\], then clip(a)[i] <= clip(a)[j]
    #[test]
    fn test_clip_preserves_order(
        a in prop::collection::vec(-100.0f32..100.0, 2..100),
        min_val in -50.0f32..50.0,
        max_val in -50.0f32..50.0
    ) {
        // Ensure min <= max
        let (min_val, max_val) = if min_val <= max_val {
            (min_val, max_val)
        } else {
            (max_val, min_val)
        };

        let va = Vector::from_slice(&a);
        let clipped = va.clip(min_val, max_val).unwrap();

        // Check order preservation
        for i in 0..a.len() {
            for j in 0..a.len() {
                if a[i] <= a[j] {
                    prop_assert!(
                        clipped.data[i] <= clipped.data[j] + 1e-5,
                        "Order not preserved: a[{}]={} <= a[{}]={}, but clip[{}]={} > clip[{}]={}",
                        i, a[i], j, a[j], i, clipped.data[i], j, clipped.data[j]
                    );
                }
            }
        }
    }
}

proptest! {
    /// Property test: clip() is idempotent
    /// clip(clip(X)) = clip(X)
    #[test]
    fn test_clip_idempotent(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        min_val in -50.0f32..50.0,
        max_val in -50.0f32..50.0
    ) {
        // Ensure min <= max
        let (min_val, max_val) = if min_val <= max_val {
            (min_val, max_val)
        } else {
            (max_val, min_val)
        };

        let va = Vector::from_slice(&a);
        let clipped1 = va.clip(min_val, max_val).unwrap();
        let clipped2 = clipped1.clip(min_val, max_val).unwrap();

        // Clipping twice should give same result
        for i in 0..clipped1.len() {
            prop_assert!(
                (clipped1.data[i] - clipped2.data[i]).abs() < 1e-5,
                "Idempotency violated at index {}: clip_once={}, clip_twice={}",
                i, clipped1.data[i], clipped2.data[i]
            );
        }
    }
}
