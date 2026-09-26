//! Ball tree: nested hyperspheres, exact search.

use super::{spread_dim, KBest, Metric, Neighbor, NeighborIndex, QueryStats, SLACK};

/// Points per leaf. Below this a linear scan beats another split.
const LEAF_SIZE: usize = 16;

#[derive(Debug, Clone)]
struct Ball {
    start: usize,
    end: usize,
    center: Vec<f32>,
    /// `max metric(center, x)` over the points in the ball.
    radius: f32,
    children: Option<(usize, usize)>,
}

/// Exact ball tree over any true metric this module offers (everything but
/// `Cosine` and `Minkowski(p < 1)`).
///
/// Pruning uses the triangle inequality: no point in a ball of radius `r`
/// around `c` is closer to `q` than `d(q, c) - r`.
#[derive(Debug, Clone)]
pub struct BallTree {
    data: Vec<f32>,
    n: usize,
    d: usize,
    metric: Metric,
    order: Vec<usize>,
    nodes: Vec<Ball>,
}

impl BallTree {
    /// Builds a ball tree over `n` row-major points of dimension `d`.
    ///
    /// # Errors
    ///
    /// Returns an error when `metric` violates the triangle inequality
    /// (`Cosine`, `Minkowski(p)` with `p < 1`) or when `data.len() != n * d`.
    pub fn new(data: &[f32], n: usize, d: usize, metric: Metric) -> crate::error::Result<Self> {
        if !metric.is_tree_compatible() {
            return Err(format!("BallTree cannot prune with {metric:?}; use BruteForce").into());
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

    fn build(&mut self, start: usize, end: usize) -> usize {
        // Centroid, accumulated in f64 so a large ball's center is not biased
        // by f32 rounding; any center is correct, a good one prunes more.
        let mut acc = vec![0.0_f64; self.d];
        for &i in &self.order[start..end] {
            for (a, &x) in acc.iter_mut().zip(self.point(i)) {
                *a += f64::from(x);
            }
        }
        let len = (end - start) as f64;
        let center: Vec<f32> = acc.iter().map(|&a| (a / len) as f32).collect();
        let radius = self.order[start..end]
            .iter()
            .map(|&i| self.metric.distance(&center, self.point(i)))
            .fold(0.0_f32, f32::max);

        let id = self.nodes.len();
        self.nodes.push(Ball {
            start,
            end,
            center,
            radius,
            children: None,
        });
        if end - start <= LEAF_SIZE {
            return id;
        }
        let Some(dim) = spread_dim(&self.data, self.d, &self.order[start..end]) else {
            return id;
        };
        let mid = start + (end - start) / 2;
        let (data, d) = (&self.data, self.d);
        self.order[start..end].select_nth_unstable_by(mid - start, |&a, &b| {
            data[a * d + dim].total_cmp(&data[b * d + dim])
        });
        let left = self.build(start, mid);
        let right = self.build(mid, end);
        self.nodes[id].children = Some((left, right));
        id
    }

    /// `d(q, center)` for ball `node`.
    fn center_distance(&self, node: usize, q: &[f32], stats: &mut QueryStats) -> f32 {
        stats.distance_evaluations += 1;
        self.metric.distance(q, &self.nodes[node].center)
    }

    /// True when ball `node`, whose center is `dqc` from the query, cannot
    /// hold a point within `bound`. The slack absorbs f32 rounding in the three
    /// distances the triangle inequality relates; it only ever visits more.
    fn excluded(&self, node: usize, dqc: f32, bound: f32) -> bool {
        let r = self.nodes[node].radius;
        (dqc - r) - SLACK * (dqc + r) > bound
    }

    fn knn(&self, node: usize, q: &[f32], best: &mut KBest, stats: &mut QueryStats) {
        stats.nodes_visited += 1;
        let ball = &self.nodes[node];
        match ball.children {
            None => {
                for &i in &self.order[ball.start..ball.end] {
                    stats.distance_evaluations += 1;
                    best.offer(self.metric.distance(q, self.point(i)), i);
                }
            }
            Some((l, r)) => {
                let dl = self.center_distance(l, q, stats);
                let dr = self.center_distance(r, q, stats);
                let (first, df, second, ds) = if dl <= dr {
                    (l, dl, r, dr)
                } else {
                    (r, dr, l, dl)
                };
                if !self.excluded(first, df, best.worst()) {
                    self.knn(first, q, best, stats);
                }
                if !self.excluded(second, ds, best.worst()) {
                    self.knn(second, q, best, stats);
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
        let ball = &self.nodes[node];
        match ball.children {
            None => {
                for &i in &self.order[ball.start..ball.end] {
                    stats.distance_evaluations += 1;
                    let distance = self.metric.distance(q, self.point(i));
                    if distance <= r {
                        out.push(Neighbor { index: i, distance });
                    }
                }
            }
            Some((l, rt)) => {
                for child in [l, rt] {
                    let dqc = self.center_distance(child, q, stats);
                    if !self.excluded(child, dqc, r) {
                        self.radius(child, q, r, out, stats);
                    }
                }
            }
        }
    }
}

impl NeighborIndex for BallTree {
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
            let dqc = self.center_distance(0, point, &mut stats);
            if !self.excluded(0, dqc, radius) {
                self.radius(0, point, radius, &mut out, &mut stats);
            }
        }
        out.sort_unstable_by_key(|nb| nb.index);
        (out, stats)
    }
}
