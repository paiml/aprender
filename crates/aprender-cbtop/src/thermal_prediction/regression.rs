//! Linear regression and Pearson correlation helpers.

/// Ordinary least squares on (x, y) pairs.
///
/// Returns `(slope, intercept, r_squared)` or `None` if fewer than 2 pairs.
pub(crate) fn ols_fit(pairs: &[(f64, f64)]) -> Option<(f64, f64, f64)> {
    let n = pairs.len() as f64;
    if n < 2.0 {
        return None;
    }

    let (mut sx, mut sy, mut sxy, mut sxx) = (0.0, 0.0, 0.0, 0.0);
    for &(x, y) in pairs {
        sx += x;
        sy += y;
        sxy += x * y;
        sxx += x * x;
    }

    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-10 {
        return Some((0.0, sy / n, 1.0));
    }

    let slope = (n * sxy - sx * sy) / denom;
    let intercept = (sy - slope * sx) / n;

    // R-squared
    let mean_y = sy / n;
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for &(x, y) in pairs {
        ss_res += (y - intercept - slope * x).powi(2);
        ss_tot += (y - mean_y).powi(2);
    }
    let r_sq = if ss_tot < 1e-10 {
        1.0
    } else {
        (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
    };

    Some((slope, intercept, r_sq))
}

/// Pearson correlation coefficient on (x, y) pairs.
///
/// Returns `(pearson_r, slope)` or `None` if fewer than 2 pairs.
pub(crate) fn pearson_r(pairs: &[(f64, f64)]) -> Option<(f64, f64)> {
    let n = pairs.len() as f64;
    if n < 2.0 {
        return None;
    }

    let (mut sx, mut sy, mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for &(x, y) in pairs {
        sx += x;
        sy += y;
        sxy += x * y;
        sxx += x * x;
        syy += y * y;
    }

    let denom_r = ((n * sxx - sx * sx) * (n * syy - sy * sy)).sqrt();
    let r = if denom_r.abs() < 1e-10 {
        0.0
    } else {
        (n * sxy - sx * sy) / denom_r
    };

    let slope_denom = n * sxx - sx * sx;
    let slope = if slope_denom.abs() < 1e-10 {
        0.0
    } else {
        (n * sxy - sx * sy) / slope_denom
    };

    Some((r, slope))
}
