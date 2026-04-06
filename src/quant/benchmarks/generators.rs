//! Weight generators for benchmarks
//!
//! Functions for generating synthetic weight distributions for testing.

/// Generate Gaussian-like weight distribution (common in neural networks)
pub fn generate_gaussian_weights(n: usize, mean: f32, std_dev: f32, seed: u64) -> Vec<f32> {
    // Simple LCG for reproducibility
    let mut state = seed;
    let a: u64 = 1103515245;
    let c: u64 = 12345;
    let m: u64 = 1 << 31;

    (0..n)
        .map(|_| {
            // Box-Muller transform (simplified)
            state = (a.wrapping_mul(state).wrapping_add(c)) % m;
            let u1 = (state as f32) / (m as f32);
            state = (a.wrapping_mul(state).wrapping_add(c)) % m;
            let u2 = (state as f32) / (m as f32);

            let z = (-2.0 * u1.max(1e-10).ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            mean + std_dev * z
        })
        .collect()
}

/// Generate uniform weights in range
pub fn generate_uniform_weights(n: usize, min: f32, max: f32, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let a: u64 = 1103515245;
    let c: u64 = 12345;
    let m: u64 = 1 << 31;

    (0..n)
        .map(|_| {
            state = (a.wrapping_mul(state).wrapping_add(c)) % m;
            let t = (state as f32) / (m as f32);
            min + t * (max - min)
        })
        .collect()
}

/// Generate weights with outliers (to test robustness)
pub fn generate_weights_with_outliers(
    n: usize,
    outlier_ratio: f32,
    outlier_magnitude: f32,
    seed: u64,
) -> Vec<f32> {
    let mut weights = generate_gaussian_weights(n, 0.0, 1.0, seed);
    let num_outliers = (n as f32 * outlier_ratio) as usize;

    let mut state = seed.wrapping_add(12345);
    let a: u64 = 1103515245;
    let c: u64 = 12345;
    let m: u64 = 1 << 31;

    for _ in 0..num_outliers {
        state = (a.wrapping_mul(state).wrapping_add(c)) % m;
        let idx = (state as usize) % n;
        state = (a.wrapping_mul(state).wrapping_add(c)) % m;
        let sign = if state.is_multiple_of(2) { 1.0 } else { -1.0 };
        weights[idx] = sign * outlier_magnitude;
    }

    weights
}

/// Generate multi-channel weights (like conv/linear layer)
pub fn generate_multi_channel_weights(
    num_channels: usize,
    features_per_channel: usize,
    scale_variance: f32,
    seed: u64,
) -> Vec<f32> {
    let mut weights = Vec::with_capacity(num_channels * features_per_channel);
    let mut state = seed;
    let a: u64 = 1103515245;
    let c: u64 = 12345;
    let m: u64 = 1 << 31;

    for ch in 0..num_channels {
        state = (a.wrapping_mul(state).wrapping_add(c)) % m;
        let channel_scale = 1.0 + (ch as f32 / num_channels as f32) * scale_variance;

        let channel_weights =
            generate_gaussian_weights(features_per_channel, 0.0, channel_scale, state);
        weights.extend(channel_weights);
    }

    weights
}
