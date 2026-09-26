//! Kd-tree: axis-aligned median splits, exact search.

use super::{prunable, spread_dim, KBest, Metric, Neighbor, NeighborIndex, QueryStats};

/// Points per leaf. Below this a linear scan beats another split.
const LEAF_SIZE: usize = 16;

#[derive(Debug, Clone)]
enum Node {
    Leaf {
        start: usize,
        end: usize,
    },
    /// Every point under `left` has `x[dim] <= value`; every point under
    /// `right` has `x[dim] >= value`. `value` is a data coordinate, so it is
    /// exactly representable and the plane bound needs no rounding slack for
    /// the Euclidean, Manhattan and Chebyshev metrics.
    Split {
        dim: usize,
        value: f32,
        left: usize,
        right: usize,
    },
}

/// Exact kd-tree over a Minkowski-family metric (`p >= 1`) or Chebyshev.
///
/// Pruning uses the gap between the query and a splitting plane: every
/// supported metric is at least the absolute difference in any one coordinate,
/// so a subtree whose plane is farther than the current k-th neighbor cannot
/// contain a closer point.
#[derive(Debug, Clone)]
pub struct KdTree {
    data: Vec<f32>,
    n: usize,
    d: usize,
    metric: Metric,
    order: Vec<usize>,
    nodes: Vec<Node>,
}

impl KdTree {
    /// Builds a kd-tree over `n` row-major points of dimension `d`.
    ///
    /// # Errors
    ///
    /// Returns an error when `metric` cannot prune a tree (`Cosine`,
    /// `Minkowski(p)` with `p < 1`) or when `data.len() != n * d`.
    pub fn new(data: &[f32], n: usize, d: usize, metric: Metric) -> crate::error::Result<Self> {
        if !metric.is_tree_compatible() {
            return Err(format!("KdTree cannot prune with {metric:?}; use BruteForce").into());
        }
        if data.len() != n * d {
            return Err("data length must equal n * d".into());
        }
        let mut tree = Self {
            data: data.to_vec(),
            n,
            d,
            metric,
            order: (0..n).collect(),
            nodes: Vec::new(),
        };
        if n > 0 {
            tree.build(0, n);
        }
        Ok(tree)
    }

    fn point(&self, i: usize) -> &[f32] {
        &self.data[i * self.d..(i + 1) * self.d]
    }

    /// Builds the subtree over `order[start..end]`; returns its node id.
    fn build(&mut self, start: usize, end: usize) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node::Leaf { start, end });
        if end - start <= LEAF_SIZE {
            return id;
        }
        // No coordinate varies (all points identical, or d == 0): nothing to
        // split on, so the range stays one leaf.
        let Some(dim) = spread_dim(&self.data, self.d, &self.order[start..end]) else {
            return id;
        };
        let mid = start + (end - start) / 2;
        let (data, d) = (&self.data, self.d);
        self.order[start..end].select_nth_unstable_by(mid - start, |&a, &b| {
            data[a * d + dim].total_cmp(&data[b * d + dim])
        });
        let value = self.data[self.order[mid] * self.d + dim];
        let left = self.build(start, mid);
        let right = self.build(mid, end);
        self.nodes[id] = Node::Split {
            dim,
            value,
            left,
            right,
        };
        id
    }

    fn knn(&self, node: usize, q: &[f32], best: &mut KBest, stats: &mut QueryStats) {
        stats.nodes_visited += 1;
        match self.nodes[node] {
            Node::Leaf { start, end } => {
                for &i in &self.order[start..end] {
                    stats.distance_evaluations += 1;
                    best.offer(self.metric.distance(q, self.point(i)), i);
                }
            }
            Node::Split {
                dim,
                value,
                left,
                right,
            } => {
                let gap = q[dim] - value;
                let (near, far) = if gap < 0.0 {
                    (left, right)
                } else {
                    (right, left)
                };
                self.knn(near, q, best, stats);
                if !prunable(gap.abs(), best.worst()) {
                    self.knn(far, q, best, stats);
                }
            }
        }
    }

    fn radius(
        &self,
        node: usize,
        q: &[f32],
        r: f32,
        out: &mut Vec<Neighbor>,
        stats: &mut QueryStats,
    ) {
        stats.nodes_visited += 1;
        match self.nodes[node] {
            Node::Leaf { start, end } => {
                for &i in &self.order[start..end] {
                    stats.distance_evaluations += 1;
                    let distance = self.metric.distance(q, self.point(i));
                    if distance <= r {
                        out.push(Neighbor { index: i, distance });
                    }
                }
            }
            Node::Split {
                dim,
                value,
                left,
                right,
            } => {
                let gap = q[dim] - value;
                // The child on the query's side of the plane has lower bound 0.
                if gap < 0.0 || !prunable(gap.abs(), r) {
                    self.radius(left, q, r, out, stats);
                }
                if gap >= 0.0 || !prunable(gap.abs(), r) {
                    self.radius(right, q, r, out, stats);
                }
            }
        }
    }
}

impl NeighborIndex for KdTree {
    fn len(&self) -> usize {
        self.n
    }

    fn n_features(&self) -> usize {
        self.d
    }

    fn metric(&self) -> Metric {
        self.metric
    }

    fn k_nearest_with_stats(&self, point: &[f32], k: usize) -> (Vec<Neighbor>, QueryStats) {
        super::check_dim(point, self.d);
        let mut best = KBest::new(k);
        let mut stats = QueryStats::default();
        if self.n > 0 && k > 0 {
            self.knn(0, point, &mut best, &mut stats);
        }
        (best.into_sorted(), stats)
    }

    fn within_radius_with_stats(&self, point: &[f32], radius: f32) -> (Vec<Neighbor>, QueryStats) {
        super::check_dim(point, self.d);
        let mut out = Vec::new();
        let mut stats = QueryStats::default();
        if self.n > 0 {
            self.radius(0, point, radius, &mut out, &mut stats);
        }
        out.sort_unstable_by_key(|nb| nb.index);
        (out, stats)
    }
}
