//! Utility functions for differential privacy.

use super::accountant::RdpAccountant;

/// Estimate required noise multiplier for target epsilon
///
/// Given target (epsilon, delta), dataset size, batch size, and epochs,
/// estimate the noise multiplier needed.
pub fn estimate_noise_multiplier(
    target_epsilon: f64,
    delta: f64,
    dataset_size: usize,
    batch_size: usize,
    epochs: usize,
) -> f64 {
    if target_epsilon <= 0.0 || delta <= 0.0 {
        return f64::INFINITY;
    }

    let sample_rate = batch_size as f64 / dataset_size as f64;
    let steps = (epochs * dataset_size) / batch_size;

    // Binary search for noise multiplier
    let mut low = 0.1;
    let mut high = 100.0;

    for _ in 0..100 {
        let mid = f64::midpoint(low, high);

        // Simulate privacy accounting
        let mut accountant = RdpAccountant::new();
        for _ in 0..steps {
            accountant.step(mid, sample_rate);
        }

        let (epsilon, _) = accountant.get_privacy_spent(delta);

        if epsilon < target_epsilon {
            high = mid;
        } else {
            low = mid;
        }

        if (high - low) < 0.01 {
            break;
        }
    }

    high * 1.1 // Add 10% safety margin
}

/// Privacy cost per step for given parameters
pub fn privacy_cost_per_step(noise_multiplier: f64, sample_rate: f64, delta: f64) -> f64 {
    let mut accountant = RdpAccountant::new();
    accountant.step(noise_multiplier, sample_rate);
    accountant.get_privacy_spent(delta).0
}
