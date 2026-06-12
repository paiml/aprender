//! Regression metrics matching `sklearn.metrics`.
//!
//! sklearn-named, slice-based, `(y_true, y_pred)` argument order. apr already had
//! `Vector`-based `mse`/`mae` (in the parent module) but no `r2_score` and no
//! sklearn-named slice API — this closes that Pillar-1 (beat scikit-learn) gap.

/// R² — the coefficient of determination, matching `sklearn.metrics.r2_score`.
///
/// `R² = 1 − SS_res / SS_tot`. When the target has zero variance (`SS_tot = 0`),
/// returns 1.0 for a perfect fit and 0.0 otherwise, as sklearn does.
///
/// # Panics
/// Panics if `y_true` and `y_pred` differ in length.
#[must_use]
pub fn r2_score(y_true: &[f32], y_pred: &[f32]) -> f32 {
    assert_eq!(y_true.len(), y_pred.len(), "r2_score: length mismatch");
    let n = y_true.len();
    if n == 0 {
        return f32::NAN;
    }
    let mean = y_true.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64;
    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred)
        .map(|(&t, &p)| (f64::from(t) - f64::from(p)).powi(2))
        .sum();
    let ss_tot: f64 = y_true.iter().map(|&t| (f64::from(t) - mean).powi(2)).sum();
    if ss_tot == 0.0 {
        return if ss_res == 0.0 { 1.0 } else { 0.0 };
    }
    (1.0 - ss_res / ss_tot) as f32
}

/// Mean squared error, matching `sklearn.metrics.mean_squared_error`.
///
/// # Panics
/// Panics if `y_true` and `y_pred` differ in length.
#[must_use]
pub fn mean_squared_error(y_true: &[f32], y_pred: &[f32]) -> f32 {
    assert_eq!(
        y_true.len(),
        y_pred.len(),
        "mean_squared_error: length mismatch"
    );
    let n = y_true.len();
    if n == 0 {
        return 0.0;
    }
    let s: f64 = y_true
        .iter()
        .zip(y_pred)
        .map(|(&t, &p)| (f64::from(t) - f64::from(p)).powi(2))
        .sum();
    (s / n as f64) as f32
}

/// Mean absolute error, matching `sklearn.metrics.mean_absolute_error`.
///
/// # Panics
/// Panics if `y_true` and `y_pred` differ in length.
#[must_use]
pub fn mean_absolute_error(y_true: &[f32], y_pred: &[f32]) -> f32 {
    assert_eq!(
        y_true.len(),
        y_pred.len(),
        "mean_absolute_error: length mismatch"
    );
    let n = y_true.len();
    if n == 0 {
        return 0.0;
    }
    let s: f64 = y_true
        .iter()
        .zip(y_pred)
        .map(|(&t, &p)| (f64::from(t) - f64::from(p)).abs())
        .sum();
    (s / n as f64) as f32
}

/// Maximum residual error `max|yᵢ − ŷᵢ|`, matching `sklearn.metrics.max_error`.
///
/// # Panics
/// Panics if `y_true` and `y_pred` differ in length.
#[must_use]
pub fn max_error(y_true: &[f32], y_pred: &[f32]) -> f32 {
    assert_eq!(y_true.len(), y_pred.len(), "max_error: length mismatch");
    y_true
        .iter()
        .zip(y_pred)
        .map(|(&t, &p)| (t - p).abs())
        .fold(0.0, f32::max)
}

/// Median absolute error, matching `sklearn.metrics.median_absolute_error`
/// (even-length: mean of the two central order statistics).
///
/// # Panics
/// Panics if `y_true` and `y_pred` differ in length.
#[must_use]
pub fn median_absolute_error(y_true: &[f32], y_pred: &[f32]) -> f32 {
    assert_eq!(
        y_true.len(),
        y_pred.len(),
        "median_absolute_error: length mismatch"
    );
    let n = y_true.len();
    if n == 0 {
        return 0.0;
    }
    let mut errs: Vec<f32> = y_true
        .iter()
        .zip(y_pred)
        .map(|(&t, &p)| (t - p).abs())
        .collect();
    errs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    if n % 2 == 1 {
        errs[n / 2]
    } else {
        (errs[n / 2 - 1] + errs[n / 2]) / 2.0
    }
}

/// Mean squared logarithmic error, matching `sklearn.metrics.mean_squared_log_error`.
/// Requires non-negative `y_true`/`y_pred` (uses `ln(1 + ·)`).
///
/// # Panics
/// Panics if `y_true` and `y_pred` differ in length.
#[must_use]
pub fn mean_squared_log_error(y_true: &[f32], y_pred: &[f32]) -> f32 {
    assert_eq!(
        y_true.len(),
        y_pred.len(),
        "mean_squared_log_error: length mismatch"
    );
    let n = y_true.len();
    if n == 0 {
        return 0.0;
    }
    let s: f64 = y_true
        .iter()
        .zip(y_pred)
        .map(|(&t, &p)| {
            let d = (1.0 + f64::from(t)).ln() - (1.0 + f64::from(p)).ln();
            d * d
        })
        .sum();
    (s / n as f64) as f32
}

/// Mean absolute percentage error, matching
/// `sklearn.metrics.mean_absolute_percentage_error` (denominator floored at
/// machine epsilon to avoid division by zero, as sklearn does).
///
/// # Panics
/// Panics if `y_true` and `y_pred` differ in length.
#[must_use]
pub fn mean_absolute_percentage_error(y_true: &[f32], y_pred: &[f32]) -> f32 {
    assert_eq!(
        y_true.len(),
        y_pred.len(),
        "mean_absolute_percentage_error: length mismatch"
    );
    let n = y_true.len();
    if n == 0 {
        return 0.0;
    }
    const EPS: f64 = 1e-15;
    let s: f64 = y_true
        .iter()
        .zip(y_pred)
        .map(|(&t, &p)| (f64::from(t) - f64::from(p)).abs() / f64::from(t).abs().max(EPS))
        .sum();
    (s / n as f64) as f32
}

/// Explained variance regression score, matching `sklearn.metrics.explained_variance_score`.
/// `1 - Var(y_true - y_pred) / Var(y_true)` (population variance). Differs from
/// R² when the residuals are biased (non-zero mean).
///
/// # Panics
/// Panics if `y_true` and `y_pred` differ in length.
#[must_use]
pub fn explained_variance_score(y_true: &[f32], y_pred: &[f32]) -> f32 {
    assert_eq!(
        y_true.len(),
        y_pred.len(),
        "explained_variance_score: length mismatch"
    );
    let n = y_true.len();
    if n == 0 {
        return f32::NAN;
    }
    let resid: Vec<f64> = y_true
        .iter()
        .zip(y_pred)
        .map(|(&t, &p)| f64::from(t) - f64::from(p))
        .collect();
    let mean_r = resid.iter().sum::<f64>() / n as f64;
    let var_r = resid.iter().map(|r| (r - mean_r).powi(2)).sum::<f64>() / n as f64;
    let mean_t = y_true.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64;
    let var_t = y_true
        .iter()
        .map(|&v| (f64::from(v) - mean_t).powi(2))
        .sum::<f64>()
        / n as f64;
    if var_t == 0.0 {
        return if var_r == 0.0 { 1.0 } else { 0.0 };
    }
    (1.0 - var_r / var_t) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    // Oracle values pinned from scikit-learn 2026-06-11.
    const YT: [f32; 5] = [3.0, -0.5, 2.0, 7.0, 4.2];
    const YP: [f32; 5] = [2.5, 0.0, 2.1, 7.8, 3.9];

    /// FT-METRIC-R2 / MSE / MAE: match sklearn within 1e-4.
    #[test]
    fn regression_metrics_match_sklearn() {
        assert!((r2_score(&YT, &YP) - 0.959_467).abs() < 1e-4);
        assert!((mean_squared_error(&YT, &YP) - 0.248).abs() < 1e-4);
        assert!((mean_absolute_error(&YT, &YP) - 0.440).abs() < 1e-4);
        // perfect fit
        assert!((r2_score(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]) - 1.0).abs() < 1e-6);
        // zero-variance target, imperfect -> 0.0 (sklearn convention)
        assert!(r2_score(&[5.0, 5.0], &[4.0, 6.0]).abs() < 1e-6);
        // explained_variance_score (differs from R² under biased residuals)
        assert!(
            (explained_variance_score(&[3.0, -0.5, 2.0, 7.0], &[2.5, 0.0, 2.0, 8.0]) - 0.957_173)
                .abs()
                < 1e-4
        );
    }

    /// FT-METRIC-MAXERR/MEDAE/MSLE/MAPE: match sklearn within 1e-4.
    #[test]
    fn extended_regression_metrics_match_sklearn() {
        assert!((max_error(&YT, &YP) - 0.8).abs() < 1e-4);
        assert!((median_absolute_error(&YT, &YP) - 0.5).abs() < 1e-4);
        let ytp = [3.0f32, 5.0, 2.5, 7.0, 1.0];
        let ypp = [2.5f32, 5.0, 3.0, 8.0, 1.2];
        assert!((mean_squared_log_error(&ytp, &ypp) - 0.011_724).abs() < 1e-4);
        assert!((mean_absolute_percentage_error(&ytp, &ypp) - 0.141_905).abs() < 1e-4);
    }
}
