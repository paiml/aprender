//! Exhaustive search: the reference every tree is proven against.

use super::{KBest, Metric, Neighbor, NeighborIndex, QueryStats};

/// Exhaustive nearest-neighbor search. O(n) distance evaluations per query,
/// any metric, and the oracle the tree indexes are tested against.
#[derive(Debug, Clone)]
pub struct BruteForce {
    data: Vec<f32>,
    n: usize,
    d: usize,
    metric: Metric,
}

impl BruteForce {
    /// Builds an index over `n` row-major points of dimension `d`.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != n * d`.
    #[must_use]
    pub fn new(data: &[f32], n: usize, d: usize, metric: Metric) -> Self {
        assert_eq!(data.len(), n * d, "data length must equal n * d");
        Self {
            data: data.to_vec(),
            n,
            d,
            metric,
        }
    }

    fn point(&self, i: usize) -> &[f32] {
        &self.data[i * self.d..(i + 1) * self.d]
    }
}

impl NeighborIndex for BruteForce {
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
        for i in 0..self.n {
            best.offer(self.metric.distance(point, self.point(i)), i);
        }
        let stats = QueryStats {
            nodes_visited: 1,
            distance_evaluations: self.n,
        };
        (best.into_sorted(), stats)
    }

    fn within_radius_with_stats(&self, point: &[f32], radius: f32) -> (Vec<Neighbor>, QueryStats) {
        super::check_dim(point, self.d);
        let found = (0..self.n)
            .filter_map(|i| {
                let distance = self.metric.distance(point, self.point(i));
                (distance <= radius).then_some(Neighbor { index: i, distance })
            })
            .collect();
        let stats = QueryStats {
            nodes_visited: 1,
            distance_evaluations: self.n,
        };
        (found, stats)
    }
}
