//! Statistical helper functions for optimization analysis.

/// Calculate mean of samples
pub(crate) fn mean(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.iter().sum::<f64>() / samples.len() as f64
}

/// Calculate standard deviation of samples (delegates to batuta-common).
pub(crate) fn std_dev(samples: &[f64]) -> f64 {
    batuta_common::math::std_dev(samples)
}

/// Calculate coefficient of variation (%)
pub(crate) fn cv(samples: &[f64]) -> f64 {
    let m = mean(samples);
    if m <= 0.0 || samples.len() < 2 {
        return 0.0;
    }
    (std_dev(samples) / m) * 100.0
}

/// Welch's t-test for unequal variances (two-tailed)
/// Returns approximate p-value
pub(crate) fn t_test(a: &[f64], b: &[f64]) -> f64 {
    if a.len() < 2 || b.len() < 2 {
        return 1.0; // Not significant if insufficient samples
    }

    let mean_a = mean(a);
    let mean_b = mean(b);
    let var_a = std_dev(a).powi(2);
    let var_b = std_dev(b).powi(2);
    let n_a = a.len() as f64;
    let n_b = b.len() as f64;

    // Welch's t-statistic
    let se = ((var_a / n_a) + (var_b / n_b)).sqrt();
    if se == 0.0 {
        return 1.0;
    }

    let t = (mean_a - mean_b).abs() / se;

    // Welch-Satterthwaite degrees of freedom
    let num = ((var_a / n_a) + (var_b / n_b)).powi(2);
    let denom = (var_a / n_a).powi(2) / (n_a - 1.0) + (var_b / n_b).powi(2) / (n_b - 1.0);
    let df = if denom > 0.0 { num / denom } else { 1.0 };

    // Approximate p-value using normal distribution for large df
    // For small df, this is an approximation
    if df > 30.0 {
        // Use normal approximation
        2.0 * (1.0 - normal_cdf(t))
    } else {
        // Simple approximation for small df
        // Real implementation would use t-distribution CDF
        let adjusted_t = t * (1.0 + 0.5 / df).sqrt();
        2.0 * (1.0 - normal_cdf(adjusted_t))
    }
}

/// Standard normal CDF approximation (Abramowitz and Stegun)
fn normal_cdf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs() / std::f64::consts::SQRT_2;

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    0.5 * (1.0 + sign * y)
}
