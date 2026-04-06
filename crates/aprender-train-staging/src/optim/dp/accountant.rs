//! RDP (Renyi Differential Privacy) accountant.
//!
//! Provides tighter privacy bounds than basic composition.
//! Based on Mironov (2017).

use serde::{Deserialize, Serialize};

/// RDP (Renyi Differential Privacy) accountant
///
/// Provides tighter privacy bounds than basic composition.
/// Based on Mironov (2017).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdpAccountant {
    /// RDP orders to track
    orders: Vec<f64>,
    /// Accumulated RDP values for each order
    pub(crate) rdp: Vec<f64>,
    /// Number of steps taken
    steps: usize,
}

impl RdpAccountant {
    /// Create a new RDP accountant
    pub fn new() -> Self {
        // Standard orders for RDP accounting
        let orders: Vec<f64> = (2..=256).map(f64::from).collect();
        let rdp = vec![0.0; orders.len()];
        Self { orders, rdp, steps: 0 }
    }

    /// Create with custom orders
    pub fn with_orders(orders: Vec<f64>) -> Self {
        let rdp = vec![0.0; orders.len()];
        Self { orders, rdp, steps: 0 }
    }

    /// Record a training step
    pub fn step(&mut self, noise_multiplier: f64, sample_rate: f64) {
        for (i, &alpha) in self.orders.iter().enumerate() {
            let rdp_step = compute_rdp_gaussian(noise_multiplier, sample_rate, alpha);
            self.rdp[i] += rdp_step;
        }
        self.steps += 1;
    }

    /// Get privacy spent as (epsilon, delta)
    pub fn get_privacy_spent(&self, delta: f64) -> (f64, f64) {
        rdp_to_dp(&self.orders, &self.rdp, delta)
    }

    /// Get number of steps
    pub fn n_steps(&self) -> usize {
        self.steps
    }

    /// Reset the accountant
    pub fn reset(&mut self) {
        for r in &mut self.rdp {
            *r = 0.0;
        }
        self.steps = 0;
    }
}

impl Default for RdpAccountant {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute RDP of Gaussian mechanism for subsampled data
pub fn compute_rdp_gaussian(noise_multiplier: f64, sample_rate: f64, alpha: f64) -> f64 {
    if noise_multiplier <= 0.0 || sample_rate <= 0.0 {
        return f64::INFINITY;
    }

    // RDP of Gaussian mechanism: alpha / (2 * sigma^2)
    // For subsampled mechanism, we use Poisson subsampling bound
    let sigma = noise_multiplier;

    if sample_rate >= 1.0 {
        // Full batch
        alpha / (2.0 * sigma.powi(2))
    } else {
        // Subsampled: upper bound using privacy amplification
        // Simplified bound: q^2 * alpha / (2 * sigma^2)
        // where q is the sampling rate
        let q = sample_rate;

        // More accurate bound using log1p for numerical stability
        if alpha <= 1.0 {
            return f64::INFINITY;
        }

        // Approximate subsampled RDP
        let log_a = (alpha - 1.0) * ((alpha * q.powi(2)) / (2.0 * sigma.powi(2))).min(1.0).ln_1p();
        log_a / (alpha - 1.0)
    }
}

/// Convert RDP to (epsilon, delta)-DP
pub fn rdp_to_dp(orders: &[f64], rdp: &[f64], delta: f64) -> (f64, f64) {
    if delta <= 0.0 || orders.is_empty() {
        return (f64::INFINITY, delta);
    }

    let log_delta = delta.max(f64::MIN_POSITIVE).ln();

    // Find optimal order
    let mut min_epsilon = f64::INFINITY;
    for (&alpha, &rdp_alpha) in orders.iter().zip(rdp.iter()) {
        if alpha <= 1.0 {
            continue;
        }
        // epsilon = rdp_alpha - (log(delta) + log(alpha - 1)) / (alpha - 1) + log(alpha / (alpha - 1))
        let epsilon = rdp_alpha
            + (1.0 / (alpha - 1.0)) * ((alpha - 1.0) / alpha).max(f64::MIN_POSITIVE).ln()
            - (log_delta + (alpha - 1.0).max(f64::MIN_POSITIVE).ln()) / (alpha - 1.0);

        if epsilon < min_epsilon && epsilon >= 0.0 {
            min_epsilon = epsilon;
        }
    }

    (min_epsilon, delta)
}
