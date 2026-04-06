//! TPE sampling utilities - KDE and EI ratio calculations

use rand::Rng;

use crate::optim::hpo::types::{ParameterValue, Trial};

/// Sample continuous parameter with EI ratio
pub fn sample_ei_ratio_continuous<R: Rng>(
    good_values: &[f64],
    bad_values: &[f64],
    low: f64,
    high: f64,
    kde_bandwidth: f64,
    rng: &mut R,
) -> f64 {
    if good_values.is_empty() {
        return low + rng.random::<f64>() * (high - low);
    }

    // Generate candidate samples
    let n_candidates = 24;
    let mut best_value = low;
    let mut best_ei = f64::NEG_INFINITY;

    let bandwidth = kde_bandwidth * (high - low) / 10.0;

    for _ in 0..n_candidates {
        // Sample from good distribution (KDE)
        let idx = (rng.random::<f64>() * good_values.len() as f64).floor() as usize;
        let idx = idx.min(good_values.len() - 1);
        let base = good_values[idx];
        // Box-Muller transform for Gaussian noise
        let u1: f64 = rng.random::<f64>().max(1e-10);
        let u2: f64 = rng.random::<f64>();
        let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos() * bandwidth;
        let candidate = (base + noise).clamp(low, high);

        // Compute l(x) / g(x) approximately
        let l_score = kde_score(candidate, good_values, bandwidth);
        let g_score = kde_score(candidate, bad_values, bandwidth);
        let ei = l_score / (g_score + 1e-10);

        if ei > best_ei {
            best_ei = ei;
            best_value = candidate;
        }
    }

    best_value
}

/// Simple KDE score (internal helper)
fn kde_score(x: f64, values: &[f64], bandwidth: f64) -> f64 {
    if values.is_empty() {
        return 1.0;
    }
    values.iter().map(|&v| (-(x - v).powi(2) / (2.0 * bandwidth.powi(2))).exp()).sum::<f64>()
        / values.len() as f64
}

/// Sample discrete parameter with EI ratio
pub fn sample_ei_ratio_discrete<R: Rng>(
    good_values: &[i64],
    bad_values: &[i64],
    low: i64,
    high: i64,
    rng: &mut R,
) -> i64 {
    if good_values.is_empty() {
        let range = (high - low + 1) as usize;
        let offset = (rng.random::<f64>() * range as f64).floor() as i64;
        return (low + offset).min(high);
    }

    // Count occurrences with Laplace smoothing
    let range = (high - low + 1) as usize;
    let mut good_counts = vec![1.0; range]; // Laplace smoothing
    let mut bad_counts = vec![1.0; range];

    for &v in good_values {
        good_counts[(v - low) as usize] += 1.0;
    }
    for &v in bad_values {
        bad_counts[(v - low) as usize] += 1.0;
    }

    // Compute weights (l/g)
    let mut weights: Vec<f64> =
        good_counts.iter().zip(bad_counts.iter()).map(|(l, g)| l / g).collect();

    // Normalize
    let total: f64 = weights.iter().sum();
    for w in &mut weights {
        *w /= total;
    }

    // Sample
    let r: f64 = rng.random();
    let mut cumsum = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        cumsum += w;
        if r < cumsum {
            return low + i as i64;
        }
    }

    high
}

/// Count categorical occurrences
pub fn count_categorical(name: &str, trials: &[&Trial], choices: &[String]) -> Vec<usize> {
    let mut counts = vec![0usize; choices.len()];
    for trial in trials {
        if let Some(ParameterValue::Categorical(s)) = trial.config.get(name) {
            if let Some(idx) = choices.iter().position(|c| c == s) {
                counts[idx] += 1;
            }
        }
    }
    counts
}
