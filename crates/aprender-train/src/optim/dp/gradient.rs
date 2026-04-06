//! Gradient operations for differential privacy.

use rand::Rng;
use std::f64::consts::PI;

/// Clip gradient to max norm (per-sample)
pub fn clip_gradient(grad: &[f64], max_norm: f64) -> Vec<f64> {
    let norm: f64 = grad.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();

    if norm > max_norm {
        let scale = max_norm / norm;
        grad.iter().map(|x| x * scale).collect()
    } else {
        grad.to_vec()
    }
}

/// Compute L2 norm of gradient
pub fn grad_norm(grad: &[f64]) -> f64 {
    grad.iter().map(|x| x.powi(2)).sum::<f64>().sqrt()
}

/// Add Gaussian noise to gradient
pub fn add_gaussian_noise<R: Rng>(grad: &[f64], std_dev: f64, rng: &mut R) -> Vec<f64> {
    grad.iter()
        .map(|&x| {
            // Box-Muller transform for Gaussian noise
            let u1: f64 = rng.random::<f64>().max(1e-10);
            let u2: f64 = rng.random::<f64>();
            let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos() * std_dev;
            x + noise
        })
        .collect()
}
