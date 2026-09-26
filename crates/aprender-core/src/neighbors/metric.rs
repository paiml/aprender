//! Distance metrics for the neighbor indexes.

use serde::{Deserialize, Serialize};

/// A distance between two points of equal dimension.
///
/// Every formula accumulates left to right in `f32`, one coordinate at a time.
/// That is the arithmetic the pre-#3149 consumers used (KNN's `compute_distance`,
/// LOF's `compute_knn`, DBSCAN's `euclidean_distance`), so rewiring them onto an
/// index changes no distance by a single bit.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Metric {
    /// `sqrt(Σ (a_i - b_i)²)`
    Euclidean,
    /// `Σ |a_i - b_i|`
    Manhattan,
    /// `(Σ |a_i - b_i|^p)^(1/p)`. A metric only for `p >= 1`.
    Minkowski(f32),
    /// `max |a_i - b_i|`
    Chebyshev,
    /// `1 - a·b / (‖a‖ ‖b‖)`; `1` when either vector is zero.
    ///
    /// NOT a metric: it violates the triangle inequality, so no tree can prune
    /// with it and it is served by brute force only.
    Cosine,
}

impl Metric {
    /// The distance between `a` and `b`.
    ///
    /// # Panics
    ///
    /// Debug builds assert `a.len() == b.len()`.
    #[must_use]
    pub fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "points must have equal dimension");
        match *self {
            Metric::Euclidean => {
                let mut sum = 0.0_f32;
                for (&x, &y) in a.iter().zip(b) {
                    let diff = x - y;
                    sum += diff * diff;
                }
                sum.sqrt()
            }
            Metric::Manhattan => {
                let mut sum = 0.0_f32;
                for (&x, &y) in a.iter().zip(b) {
                    sum += (x - y).abs();
                }
                sum
            }
            Metric::Minkowski(p) => {
                let mut sum = 0.0_f32;
                for (&x, &y) in a.iter().zip(b) {
                    sum += (x - y).abs().powf(p);
                }
                sum.powf(1.0 / p)
            }
            Metric::Chebyshev => {
                let mut max = 0.0_f32;
                for (&x, &y) in a.iter().zip(b) {
                    max = max.max((x - y).abs());
                }
                max
            }
            Metric::Cosine => {
                let (mut dot, mut na, mut nb) = (0.0_f32, 0.0_f32, 0.0_f32);
                for (&x, &y) in a.iter().zip(b) {
                    dot += x * y;
                    na += x * x;
                    nb += y * y;
                }
                let denom = na.sqrt() * nb.sqrt();
                if denom == 0.0 {
                    1.0
                } else {
                    // Clamp: rounding can push |cos| a hair past 1.
                    (1.0 - (dot / denom).clamp(-1.0, 1.0)).max(0.0)
                }
            }
        }
    }

    /// True when a space-partitioning tree may prune with this metric: it must
    /// satisfy the triangle inequality (ball tree) and dominate every single
    /// coordinate difference (kd-tree). Minkowski needs a finite `p >= 1`.
    #[must_use]
    pub fn is_tree_compatible(&self) -> bool {
        match *self {
            Metric::Euclidean | Metric::Manhattan | Metric::Chebyshev => true,
            Metric::Minkowski(p) => p.is_finite() && p >= 1.0,
            Metric::Cosine => false,
        }
    }
}
