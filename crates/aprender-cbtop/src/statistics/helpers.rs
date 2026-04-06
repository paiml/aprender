//! Statistical helper functions: bootstrap, percentile, normal CDF, etc.

use std::cmp::Ordering;

/// Bootstrap confidence interval calculation
pub fn bootstrap_ci(samples: &[f64], confidence_level: f64, iterations: usize) -> (f64, f64) {
    if samples.is_empty() {
        return (0.0, 0.0);
    }

    if samples.len() == 1 {
        return (samples[0], samples[0]);
    }

    let n = samples.len();
    let mut bootstrap_means: Vec<f64> = Vec::with_capacity(iterations);

    // Simple LCG for deterministic but varied resampling
    let mut rng_state: u64 = 12345;
    let lcg = |state: &mut u64| -> usize {
        *state = state.wrapping_mul(1103515245).wrapping_add(12345);
        ((*state >> 16) as usize) % n
    };

    for _ in 0..iterations {
        let mut sum = 0.0;
        for _ in 0..n {
            sum += samples[lcg(&mut rng_state)];
        }
        bootstrap_means.push(sum / n as f64);
    }

    bootstrap_means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let alpha = 1.0 - confidence_level;
    let lower_idx = ((alpha / 2.0) * iterations as f64) as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * iterations as f64) as usize;

    let ci_lower = bootstrap_means
        .get(lower_idx)
        .copied()
        .unwrap_or(bootstrap_means[0]);
    let ci_upper = bootstrap_means
        .get(upper_idx.min(iterations - 1))
        .copied()
        .unwrap_or(*bootstrap_means.last().unwrap_or(&0.0));

    (ci_lower, ci_upper)
}

/// Calculate percentile (0.0 to 1.0)
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Standard normal CDF approximation
pub(super) fn normal_cdf(x: f64) -> f64 {
    // Approximation using error function
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation (delegates to batuta-common).
fn erf(x: f64) -> f64 {
    batuta_common::math::erf(x)
}

/// Calculate robust mean (trimmed mean, removing top/bottom 10%)
pub fn trimmed_mean(samples: &[f64], trim_percent: f64) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }

    let mut sorted: Vec<f64> = samples.iter().copied().filter(|x| x.is_finite()).collect();

    if sorted.is_empty() {
        return None;
    }

    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let trim_count = (sorted.len() as f64 * trim_percent) as usize;
    let start = trim_count;
    let end = sorted.len() - trim_count;

    if start >= end {
        return Some(sorted[sorted.len() / 2]);
    }

    let trimmed = &sorted[start..end];
    Some(trimmed.iter().sum::<f64>() / trimmed.len() as f64)
}
