//! Dataset generators and loaders (Pillar 1: beat scikit-learn).
//!
//! Mirrors `sklearn.datasets`. This module provides the synthetic generators
//! (`make_blobs`, `make_regression`) needed to run classification/regression
//! benchmarks and correctness tests without any external data files. Embedded
//! real datasets (`load_iris`/`load_digits`/`load_california_housing`) are a
//! follow-up (PMAT-720 continuation).
//!
//! All generators are **deterministic** given a seed — the same `seed` yields
//! byte-identical output, so benchmarks and falsifiers are reproducible.

use crate::primitives::{Matrix, Vector};

/// Small, fast, well-distributed seeded PRNG (SplitMix64) — no external dep,
/// reproducible across platforms.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    fn next_f32(&mut self) -> f32 {
        // top 24 bits -> [0,1) with full f32 mantissa precision
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Standard-normal sample via Box–Muller.
    fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-9);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (core::f32::consts::TAU * u2).cos()
    }
}

/// Generate isotropic Gaussian blobs for clustering/classification.
///
/// Mirrors `sklearn.datasets.make_blobs`. `centers` defines the cluster means
/// (and thus `n_features = centers[0].len()` and `n_classes = centers.len()`);
/// samples are assigned round-robin to centers, so classes are balanced.
///
/// # Panics
/// Panics if `centers` is empty or centers have inconsistent lengths.
#[must_use]
pub fn make_blobs(
    n_samples: usize,
    centers: &[Vec<f32>],
    cluster_std: f32,
    seed: u64,
) -> (Matrix<f32>, Vec<usize>) {
    assert!(!centers.is_empty(), "make_blobs: needs >= 1 center");
    let n_features = centers[0].len();
    assert!(
        centers.iter().all(|c| c.len() == n_features),
        "make_blobs: all centers must have the same length"
    );
    let n_centers = centers.len();

    let mut rng = SplitMix64::new(seed);
    let mut data = Vec::with_capacity(n_samples * n_features);
    let mut labels = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let c = i % n_centers;
        for f in 0..n_features {
            data.push(centers[c][f] + cluster_std * rng.next_gaussian());
        }
        labels.push(c);
    }
    let x = Matrix::from_vec(n_samples, n_features, data).expect("make_blobs: valid dims");
    (x, labels)
}

/// Generate a random linear regression problem with Gaussian noise.
///
/// Mirrors `sklearn.datasets.make_regression`. Returns features `X`
/// (`n_samples × n_features`, standard-normal) and targets `y = X·w + noise·ε`
/// for a fixed random ground-truth `w`.
///
/// # Panics
/// Panics if the produced matrix dimensions are inconsistent (never for valid
/// inputs).
#[must_use]
pub fn make_regression(
    n_samples: usize,
    n_features: usize,
    noise: f32,
    seed: u64,
) -> (Matrix<f32>, Vector<f32>) {
    let mut rng = SplitMix64::new(seed);
    let weights: Vec<f32> = (0..n_features).map(|_| rng.next_gaussian()).collect();

    let mut data = Vec::with_capacity(n_samples * n_features);
    let mut targets = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let mut y = 0.0f32;
        for &w in &weights {
            let x = rng.next_gaussian();
            data.push(x);
            y += w * x;
        }
        y += noise * rng.next_gaussian();
        targets.push(y);
    }
    let x = Matrix::from_vec(n_samples, n_features, data).expect("make_regression: valid dims");
    (x, Vector::from_vec(targets))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FT-DATA-001: make_blobs is deterministic (same seed -> identical output).
    #[test]
    fn make_blobs_is_deterministic() {
        let centers = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
        let (x1, y1) = make_blobs(20, &centers, 0.5, 42);
        let (x2, y2) = make_blobs(20, &centers, 0.5, 42);
        assert_eq!(
            x1.as_slice(),
            x2.as_slice(),
            "same seed must give identical X"
        );
        assert_eq!(y1, y2, "same seed must give identical labels");
        // a different seed must differ
        let (x3, _) = make_blobs(20, &centers, 0.5, 43);
        assert_ne!(x1.as_slice(), x3.as_slice(), "different seed must differ");
    }

    /// FT-DATA-002: make_blobs shapes + balanced round-robin labels.
    #[test]
    fn make_blobs_shapes_and_labels() {
        let centers = vec![vec![0.0, 0.0, 0.0], vec![5.0, 5.0, 5.0]];
        let (x, y) = make_blobs(10, &centers, 0.3, 7);
        assert_eq!(x.n_rows(), 10);
        assert_eq!(x.n_cols(), 3);
        assert_eq!(y.len(), 10);
        assert!(y.iter().all(|&c| c < 2));
        // round-robin -> balanced
        assert_eq!(y.iter().filter(|&&c| c == 0).count(), 5);
    }

    /// FT-DATA-003: well-separated blobs are linearly separable — each sample is
    /// nearer its own center than the other (cluster_std << center gap).
    #[test]
    fn make_blobs_clusters_are_separable() {
        let centers = vec![vec![0.0, 0.0], vec![20.0, 20.0]];
        let (x, y) = make_blobs(40, &centers, 0.5, 99);
        for i in 0..x.n_rows() {
            let p0 = (x.get(i, 0).powi(2) + x.get(i, 1).powi(2)).sqrt();
            let p1 = ((x.get(i, 0) - 20.0).powi(2) + (x.get(i, 1) - 20.0).powi(2)).sqrt();
            let nearest = usize::from(p1 < p0);
            assert_eq!(nearest, y[i], "sample {i} must be nearest its own center");
        }
    }

    /// FT-DATA-004: make_regression shapes + determinism + signal (target
    /// variance >> 0 so it's a real regression problem, not noise).
    #[test]
    fn make_regression_shapes_and_signal() {
        let (x, y) = make_regression(100, 4, 0.1, 5);
        assert_eq!(x.n_rows(), 100);
        assert_eq!(x.n_cols(), 4);
        assert_eq!(y.len(), 100);
        let (x2, y2) = make_regression(100, 4, 0.1, 5);
        assert_eq!(x.as_slice(), x2.as_slice());
        assert_eq!(y.as_slice(), y2.as_slice());
        let mean = y.as_slice().iter().sum::<f32>() / y.len() as f32;
        let var = y.as_slice().iter().map(|v| (v - mean).powi(2)).sum::<f32>() / y.len() as f32;
        assert!(
            var > 0.1,
            "regression target must carry signal, var = {var}"
        );
    }
}
