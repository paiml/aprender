//! Exact spatial neighbor search (#3149).
//!
//! One index abstraction for every algorithm that asks "which points are near
//! this one": [`KNearestNeighbors`](crate::classification::KNearestNeighbors),
//! [`DBSCAN`](crate::cluster::DBSCAN) and
//! [`LocalOutlierFactor`](crate::cluster::LocalOutlierFactor) all query through
//! it. Search is EXACT: a tree returns precisely the set brute force returns,
//! ties included. Approximate search (HNSW) lives in [`crate::index`].
//!
//! # Indexes
//!
//! | Index | Build | Query | Metrics |
//! |-------|-------|-------|---------|
//! | [`BruteForce`] | O(n) copy | O(n) | all |
//! | [`KdTree`] | O(n log n) | ~O(log n) for small d | tree-compatible |
//! | [`BallTree`] | O(n log n) | ~O(log n), degrades slower with d | tree-compatible |
//!
//! "Tree-compatible" ([`Metric::is_tree_compatible`]) means Euclidean,
//! Manhattan, Chebyshev and Minkowski with `p >= 1`. Cosine distance breaks
//! the triangle inequality, and so does Minkowski with `p < 1`; both are
//! brute-force only, and asking a tree for them is an error, never a silently
//! wrong answer.
//!
//! # Result order
//!
//! - [`NeighborIndex::k_nearest`] returns at most `k` neighbors in ascending
//!   `(distance, index)` order. Equal distances are broken by the lower point
//!   index, so the result is a pure function of the data, never of tree shape.
//! - [`NeighborIndex::within_radius`] returns every point with
//!   `distance <= radius`, in ascending index order.
//!
//! # `NeighborAlgorithm::Auto`
//!
//! Resolved from `(n, d, metric)`, in this order:
//!
//! 1. metric not tree-compatible → [`BruteForce`];
//! 2. `n <= 64` → [`BruteForce`]: a tree's traversal costs more than the
//!    handful of distances it would skip;
//! 3. `d <= 16` → [`KdTree`]: axis splits stay selective in low dimension;
//! 4. `d <= 64` → [`BallTree`]: spheres keep pruning after axis planes stop;
//! 5. otherwise → [`BruteForce`]: past that, both trees visit nearly every
//!    point and only add overhead.
//!
//! These thresholds pick a speed, never an answer: every choice returns the
//! same neighbors.
//!
//! # Example
//!
//! ```
//! use aprender::neighbors::{Metric, NeighborAlgorithm, NeighborIndex, SpatialIndex};
//!
//! let data = [0.0, 0.0, 1.0, 0.0, 0.0, 3.0, 5.0, 5.0];
//! let index = SpatialIndex::build(&data, 4, 2, Metric::Euclidean, NeighborAlgorithm::Auto)
//!     .expect("euclidean is supported by every index");
//! let nearest = index.k_nearest(&[0.9, 0.1], 2);
//! assert_eq!(nearest.iter().map(|nb| nb.index).collect::<Vec<_>>(), vec![1, 0]);
//! let close = index.within_radius(&[0.0, 0.0], 1.0);
//! assert_eq!(close.iter().map(|nb| nb.index).collect::<Vec<_>>(), vec![0, 1]);
//! ```

mod ball_tree;
mod brute;
mod kd_tree;
mod metric;

pub use ball_tree::BallTree;
pub use brute::BruteForce;
pub use kd_tree::KdTree;
pub use metric::Metric;

use crate::primitives::Matrix;
use std::cmp::Ordering;

/// Relative rounding slack in tree pruning bounds. It can only make a tree
/// visit MORE nodes, never skip one that holds an answer.
const SLACK: f32 = 1e-4;

/// One neighbor: a point index into the indexed data and its distance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neighbor {
    /// Row of the point in the data the index was built from.
    pub index: usize,
    /// Distance from the query under the index's metric.
    pub distance: f32,
}

/// Work done by one query. Counts, not wall-clock, so a test of sub-linear
/// scaling does not depend on the machine.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QueryStats {
    /// Tree nodes entered (1 for brute force).
    pub nodes_visited: usize,
    /// Metric evaluations, including distances to ball centers.
    pub distance_evaluations: usize,
}

/// Exact neighbor queries over a fixed point set.
pub trait NeighborIndex {
    /// Number of indexed points.
    fn len(&self) -> usize;

    /// True when no points are indexed.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Dimension of the indexed points.
    fn n_features(&self) -> usize;

    /// The metric distances are measured in.
    fn metric(&self) -> Metric;

    /// [`k_nearest`](Self::k_nearest), plus the work it took.
    fn k_nearest_with_stats(&self, point: &[f32], k: usize) -> (Vec<Neighbor>, QueryStats);

    /// [`within_radius`](Self::within_radius), plus the work it took.
    fn within_radius_with_stats(&self, point: &[f32], radius: f32) -> (Vec<Neighbor>, QueryStats);

    /// The `min(k, len)` nearest points, ascending by `(distance, index)`.
    ///
    /// # Panics
    ///
    /// Panics if `point.len() != self.n_features()`.
    fn k_nearest(&self, point: &[f32], k: usize) -> Vec<Neighbor> {
        self.k_nearest_with_stats(point, k).0
    }

    /// Every point with `distance <= radius`, ascending by index.
    ///
    /// # Panics
    ///
    /// Panics if `point.len() != self.n_features()`.
    fn within_radius(&self, point: &[f32], radius: f32) -> Vec<Neighbor> {
        self.within_radius_with_stats(point, radius).0
    }
}

/// Which index to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NeighborAlgorithm {
    /// Chosen from `(n, d, metric)`; see the module docs for the rule.
    #[default]
    Auto,
    /// [`BruteForce`].
    BruteForce,
    /// [`KdTree`].
    KdTree,
    /// [`BallTree`].
    BallTree,
}

impl NeighborAlgorithm {
    /// The concrete algorithm `self` means for `n` points of dimension `d`.
    /// Never returns `Auto`.
    #[must_use]
    pub fn resolve(self, n: usize, d: usize, metric: Metric) -> Self {
        match self {
            Self::Auto if !metric.is_tree_compatible() || n <= 64 => Self::BruteForce,
            Self::Auto if d <= 16 => Self::KdTree,
            Self::Auto if d <= 64 => Self::BallTree,
            Self::Auto => Self::BruteForce,
            explicit => explicit,
        }
    }
}

/// Any of the three indexes behind one concrete, cloneable type.
#[derive(Debug, Clone)]
pub enum SpatialIndex {
    /// Exhaustive search.
    BruteForce(BruteForce),
    /// Kd-tree.
    KdTree(KdTree),
    /// Ball tree.
    BallTree(BallTree),
}

impl SpatialIndex {
    /// Builds the index `algorithm` resolves to over `n` row-major points of
    /// dimension `d`.
    ///
    /// # Errors
    ///
    /// Returns an error when `data.len() != n * d`, or when a tree is requested
    /// explicitly for a metric it cannot prune with.
    pub fn build(
        data: &[f32],
        n: usize,
        d: usize,
        metric: Metric,
        algorithm: NeighborAlgorithm,
    ) -> crate::error::Result<Self> {
        if data.len() != n * d {
            return Err("data length must equal n * d".into());
        }
        Ok(match algorithm.resolve(n, d, metric) {
            NeighborAlgorithm::KdTree => Self::KdTree(KdTree::new(data, n, d, metric)?),
            NeighborAlgorithm::BallTree => Self::BallTree(BallTree::new(data, n, d, metric)?),
            _ => Self::BruteForce(BruteForce::new(data, n, d, metric)),
        })
    }

    /// Builds over the rows of `x`.
    ///
    /// # Errors
    ///
    /// As [`SpatialIndex::build`].
    pub fn from_matrix(
        x: &Matrix<f32>,
        metric: Metric,
        algorithm: NeighborAlgorithm,
    ) -> crate::error::Result<Self> {
        let (n, d) = x.shape();
        Self::build(x.as_slice(), n, d, metric, algorithm)
    }

    /// The concrete algorithm this index uses.
    #[must_use]
    pub fn algorithm(&self) -> NeighborAlgorithm {
        match self {
            Self::BruteForce(_) => NeighborAlgorithm::BruteForce,
            Self::KdTree(_) => NeighborAlgorithm::KdTree,
            Self::BallTree(_) => NeighborAlgorithm::BallTree,
        }
    }

    fn inner(&self) -> &dyn NeighborIndex {
        match self {
            Self::BruteForce(i) => i,
            Self::KdTree(i) => i,
            Self::BallTree(i) => i,
        }
    }
}

impl NeighborIndex for SpatialIndex {
    fn len(&self) -> usize {
        self.inner().len()
    }

    fn n_features(&self) -> usize {
        self.inner().n_features()
    }

    fn metric(&self) -> Metric {
        self.inner().metric()
    }

    fn k_nearest_with_stats(&self, point: &[f32], k: usize) -> (Vec<Neighbor>, QueryStats) {
        self.inner().k_nearest_with_stats(point, k)
    }

    fn within_radius_with_stats(&self, point: &[f32], radius: f32) -> (Vec<Neighbor>, QueryStats) {
        self.inner().within_radius_with_stats(point, radius)
    }
}

/// The total order results are reported in: distance, then point index.
fn key_cmp(a: (f32, usize), b: (f32, usize)) -> Ordering {
    a.0.total_cmp(&b.0).then(a.1.cmp(&b.1))
}

/// The `k` smallest `(distance, index)` pairs seen so far, kept sorted.
struct KBest {
    k: usize,
    items: Vec<(f32, usize)>,
}

impl KBest {
    fn new(k: usize) -> Self {
        Self {
            k,
            // One spare slot: `offer` inserts before it truncates.
            items: Vec::with_capacity(k.min(1024) + 1),
        }
    }

    fn offer(&mut self, distance: f32, index: usize) {
        let key = (distance, index);
        if self.items.len() == self.k {
            match self.items.last() {
                Some(&last) if key_cmp(key, last) == Ordering::Less => {}
                _ => return,
            }
        }
        let at = self
            .items
            .partition_point(|&it| key_cmp(it, key) == Ordering::Less);
        self.items.insert(at, key);
        self.items.truncate(self.k);
    }

    /// The distance a candidate must not exceed to still matter. `+inf` until
    /// `k` candidates are held; `-inf` when `k == 0`, so everything prunes.
    fn worst(&self) -> f32 {
        if self.k == 0 {
            f32::NEG_INFINITY
        } else if self.items.len() < self.k {
            f32::INFINITY
        } else {
            self.items.last().map_or(f32::INFINITY, |&(d, _)| d)
        }
    }

    fn into_sorted(self) -> Vec<Neighbor> {
        self.items
            .into_iter()
            .map(|(distance, index)| Neighbor { index, distance })
            .collect()
    }
}

/// True when a region no closer than `lower_bound` can be skipped against
/// `bound`. Strict `>`: a region AT the bound may hold a tie with a lower
/// index, so it is searched.
fn prunable(lower_bound: f32, bound: f32) -> bool {
    lower_bound * (1.0 - SLACK) > bound
}

/// The coordinate with the largest spread over `rows`, or `None` when every
/// coordinate is constant (or `d == 0`) and there is nothing to split on.
fn spread_dim(data: &[f32], d: usize, rows: &[usize]) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for dim in 0..d {
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &i in rows {
            let v = data[i * d + dim];
            lo = lo.min(v);
            hi = hi.max(v);
        }
        let spread = hi - lo;
        if spread > 0.0 && best.map_or(true, |(_, s)| spread > s) {
            best = Some((dim, spread));
        }
    }
    best.map(|(dim, _)| dim)
}

fn check_dim(point: &[f32], d: usize) {
    assert_eq!(
        point.len(),
        d,
        "query has {} coordinates, index has {d}",
        point.len()
    );
}

#[cfg(kani)]
mod kani_proofs;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod tests_rewire;
