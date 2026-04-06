//! Statistical helper functions for drift detection.

/// Count samples in bins defined by edges
pub fn bin_counts(data: &[f64], edges: &[f64]) -> Vec<usize> {
    let mut counts = vec![0; edges.len() - 1];
    for &val in data {
        for i in 0..counts.len() {
            if val > edges[i] && val <= edges[i + 1] {
                counts[i] += 1;
                break;
            }
        }
    }
    counts
}

/// Approximate p-value for KS statistic using Kolmogorov distribution
pub fn ks_p_value(lambda: f64) -> f64 {
    if lambda <= 0.0 {
        return 1.0;
    }
    // Asymptotic approximation: P(D > d) ≈ 2 * sum_{k=1}^∞ (-1)^{k+1} * exp(-2 * k^2 * λ^2)
    let mut p = 0.0;
    for k in 1..=100 {
        let sign = if k % 2 == 1 { 1.0 } else { -1.0 };
        let term = sign * (-2.0 * f64::from(k).powi(2) * lambda.powi(2)).exp();
        p += term;
        if term.abs() < 1e-10 {
            break;
        }
    }
    (2.0 * p).clamp(0.0, 1.0)
}

/// Approximate chi-square p-value using Wilson-Hilferty approximation
pub fn chi_square_p_value(chi_sq: f64, df: usize) -> f64 {
    if df == 0 || chi_sq <= 0.0 {
        return 1.0;
    }
    let k = df as f64;
    // Wilson-Hilferty transformation to normal
    let z = ((chi_sq / k).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / (2.0 / (9.0 * k)).sqrt();
    // Convert z to p-value (upper tail)
    0.5 * (1.0 - erf(z / std::f64::consts::SQRT_2))
}

/// Error function approximation
pub fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}
