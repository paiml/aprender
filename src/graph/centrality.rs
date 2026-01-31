//! Graph centrality measures.
//!
//! This module provides various centrality algorithms for measuring node importance:
//! - Degree centrality (Freeman's normalization)
//! - PageRank (power iteration with Kahan summation)
//! - Betweenness centrality (Brandes' parallel algorithm)
//! - Closeness centrality (geodesic distances)
//! - Eigenvector centrality (power iteration)
//! - Katz centrality (with attenuation factor)
//! - Harmonic centrality (Boldi & Vigna)

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::collections::{HashMap, VecDeque};

use super::{Graph, NodeId};

/// Kahan summation for computing L1 distance between two vectors.
///
/// Uses compensated summation to prevent floating-point drift.
fn kahan_diff(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut c = 0.0; // Compensation term

    for (ai, bi) in a.iter().zip(b.iter()) {
        let y = (ai - bi).abs() - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    sum
}

/// Extension trait for graph centrality measures.
///
/// Provides methods to compute various centrality metrics that measure
/// the importance or influence of nodes in a graph.
pub trait GraphCentrality {
    /// Compute degree centrality for all nodes.
    ///
    /// Uses Freeman's normalization (1978): `C_D(v)` = deg(v) / (n - 1)
    ///
    /// # Returns
    /// `HashMap` mapping `NodeId` to centrality score in [0, 1]
    ///
    /// # Performance
    /// O(n + m) where n = nodes, m = edges
    fn degree_centrality(&self) -> HashMap<NodeId, f64>;

    /// Compute `PageRank` using power iteration with Kahan summation.
    ///
    /// Uses the `PageRank` algorithm (Page et al. 1999) with numerically
    /// stable Kahan summation (Higham 1993) to prevent floating-point
    /// drift in large graphs (>10K nodes).
    ///
    /// # Arguments
    /// * `damping` - Damping factor (typically 0.85)
    /// * `max_iter` - Maximum iterations (default 100)
    /// * `tol` - Convergence tolerance (default 1e-6)
    ///
    /// # Returns
    /// Vector of `PageRank` scores (one per node)
    fn pagerank(&self, damping: f64, max_iter: usize, tol: f64) -> Result<Vec<f64>, String>;

    /// Compute betweenness centrality using parallel Brandes' algorithm.
    ///
    /// Uses Brandes' algorithm (2001) with Rayon parallelization for the outer loop.
    ///
    /// # Returns
    /// Vector of betweenness centrality scores (one per node)
    fn betweenness_centrality(&self) -> Vec<f64>;

    /// Compute closeness centrality for all nodes.
    ///
    /// Closeness is the reciprocal of the sum of shortest path distances.
    ///
    /// # Returns
    /// Vector of closeness centrality scores
    fn closeness_centrality(&self) -> Vec<f64>;

    /// Compute eigenvector centrality using power iteration.
    ///
    /// # Arguments
    /// * `max_iter` - Maximum power iterations
    /// * `tol` - Convergence tolerance
    fn eigenvector_centrality(&self, max_iter: usize, tol: f64) -> Result<Vec<f64>, String>;

    /// Compute Katz centrality with attenuation factor.
    ///
    /// # Arguments
    /// * `alpha` - Attenuation factor (must be in (0, 1))
    /// * `max_iter` - Maximum iterations
    /// * `tol` - Convergence tolerance
    fn katz_centrality(&self, alpha: f64, max_iter: usize, tol: f64) -> Result<Vec<f64>, String>;

    /// Compute harmonic centrality for all nodes.
    ///
    /// Harmonic centrality is the sum of reciprocal distances to all other nodes.
    fn harmonic_centrality(&self) -> Vec<f64>;
}

impl GraphCentrality for Graph {
    fn degree_centrality(&self) -> HashMap<NodeId, f64> {
        let mut centrality = HashMap::with_capacity(self.num_nodes());

        if self.num_nodes() <= 1 {
            for v in 0..self.num_nodes() {
                centrality.insert(v, 0.0);
            }
            return centrality;
        }

        let norm = (self.num_nodes() - 1) as f64;

        #[allow(clippy::needless_range_loop)]
        for v in 0..self.num_nodes() {
            let degree = self.neighbors(v).len() as f64;
            centrality.insert(v, degree / norm);
        }

        centrality
    }

    fn pagerank(&self, damping: f64, max_iter: usize, tol: f64) -> Result<Vec<f64>, String> {
        if self.num_nodes() == 0 {
            return Ok(Vec::new());
        }

        let n = self.num_nodes();
        let mut ranks = vec![1.0 / n as f64; n];
        let mut new_ranks = vec![0.0; n];

        for _ in 0..max_iter {
            // Handle dangling nodes (nodes with no outgoing edges)
            let mut dangling_sum = 0.0;
            #[allow(clippy::needless_range_loop)]
            for v in 0..n {
                if self.neighbors(v).is_empty() {
                    dangling_sum += ranks[v];
                }
            }
            let dangling_contribution = damping * dangling_sum / n as f64;

            // Compute new ranks with Kahan summation
            #[allow(clippy::needless_range_loop)]
            for v in 0..n {
                let incoming_neighbors = self.incoming_neighbors(v);

                let mut sum = 0.0;
                let mut c = 0.0; // Kahan compensation term

                for u in &incoming_neighbors {
                    let out_degree = self.neighbors(*u).len() as f64;
                    if out_degree > 0.0 {
                        let y = (ranks[*u] / out_degree) - c;
                        let t = sum + y;
                        c = (t - sum) - y;
                        sum = t;
                    }
                }

                new_ranks[v] = (1.0 - damping) / n as f64 + damping * sum + dangling_contribution;
            }

            // Convergence check using Kahan for diff calculation
            let diff = kahan_diff(&ranks, &new_ranks);
            if diff < tol {
                return Ok(new_ranks);
            }

            std::mem::swap(&mut ranks, &mut new_ranks);
        }

        Ok(ranks)
    }

    fn betweenness_centrality(&self) -> Vec<f64> {
        if self.num_nodes() == 0 {
            return Vec::new();
        }

        // Compute partial betweenness from each source (parallel when available)
        #[cfg(feature = "parallel")]
        let partial_scores: Vec<Vec<f64>> = (0..self.num_nodes())
            .into_par_iter()
            .map(|source| brandes_bfs_from_source(self, source))
            .collect();

        #[cfg(not(feature = "parallel"))]
        let partial_scores: Vec<Vec<f64>> = (0..self.num_nodes())
            .map(|source| brandes_bfs_from_source(self, source))
            .collect();

        // Reduce partial scores
        let mut centrality = vec![0.0; self.num_nodes()];
        for partial in partial_scores {
            for (i, &score) in partial.iter().enumerate() {
                centrality[i] += score;
            }
        }

        // Normalize for undirected graphs
        if !self.is_directed() {
            for score in &mut centrality {
                *score /= 2.0;
            }
        }

        centrality
    }

    fn closeness_centrality(&self) -> Vec<f64> {
        if self.num_nodes() == 0 {
            return Vec::new();
        }

        let mut centrality = vec![0.0; self.num_nodes()];

        #[allow(clippy::needless_range_loop)]
        for v in 0..self.num_nodes() {
            let distances = bfs_distances(self, v);

            let sum: usize = distances.iter().filter(|&&d| d != usize::MAX).sum();
            let reachable = distances
                .iter()
                .filter(|&&d| d != usize::MAX && d > 0)
                .count();

            if reachable > 0 && sum > 0 {
                centrality[v] = reachable as f64 / sum as f64;
            }
        }

        centrality
    }

    fn eigenvector_centrality(&self, max_iter: usize, tol: f64) -> Result<Vec<f64>, String> {
        if self.num_nodes() == 0 {
            return Ok(Vec::new());
        }

        let n = self.num_nodes();
        let mut x = vec![1.0 / (n as f64).sqrt(); n];
        let mut x_new = vec![0.0; n];

        for _ in 0..max_iter {
            #[allow(clippy::needless_range_loop)]
            for v in 0..n {
                x_new[v] = self.neighbors(v).iter().map(|&u| x[u]).sum();
            }

            let norm: f64 = x_new.iter().map(|&val| val * val).sum::<f64>().sqrt();

            if norm < 1e-10 {
                return Ok(vec![0.0; n]);
            }

            for val in &mut x_new {
                *val /= norm;
            }

            let diff: f64 = x.iter().zip(&x_new).map(|(a, b)| (a - b).abs()).sum();

            if diff < tol {
                return Ok(x_new);
            }

            std::mem::swap(&mut x, &mut x_new);
        }

        Ok(x)
    }

    fn katz_centrality(&self, alpha: f64, max_iter: usize, tol: f64) -> Result<Vec<f64>, String> {
        if self.num_nodes() == 0 {
            return Ok(Vec::new());
        }

        if alpha <= 0.0 || alpha >= 1.0 {
            return Err("Alpha must be in (0, 1)".to_string());
        }

        let n = self.num_nodes();
        let mut x = vec![1.0; n];
        let mut x_new = vec![0.0; n];

        for _ in 0..max_iter {
            #[allow(clippy::needless_range_loop)]
            for v in 0..n {
                let incoming = self.incoming_neighbors(v);
                let neighbors_sum: f64 = incoming.iter().map(|&u| x[u]).sum();
                x_new[v] = 1.0 + alpha * neighbors_sum;
            }

            let diff: f64 = x.iter().zip(&x_new).map(|(a, b)| (a - b).abs()).sum();

            if diff < tol {
                return Ok(x_new);
            }

            std::mem::swap(&mut x, &mut x_new);
        }

        Ok(x)
    }

    fn harmonic_centrality(&self) -> Vec<f64> {
        if self.num_nodes() == 0 {
            return Vec::new();
        }

        let mut centrality = vec![0.0; self.num_nodes()];

        #[allow(clippy::needless_range_loop)]
        for v in 0..self.num_nodes() {
            let distances = bfs_distances(self, v);

            for &dist in &distances {
                if dist > 0 && dist != usize::MAX {
                    centrality[v] += 1.0 / dist as f64;
                }
            }
        }

        centrality
    }
}

/// Brandes' BFS from a single source node.
fn brandes_bfs_from_source(graph: &Graph, source: NodeId) -> Vec<f64> {
    let n = graph.num_nodes();
    let mut stack = Vec::new();
    let mut paths = vec![0u64; n];
    let mut distance = vec![i32::MAX; n];
    let mut predecessors: Vec<Vec<NodeId>> = vec![Vec::new(); n];
    let mut dependency = vec![0.0; n];

    paths[source] = 1;
    distance[source] = 0;
    let mut queue = VecDeque::new();
    queue.push_back(source);

    while let Some(v) = queue.pop_front() {
        stack.push(v);
        for &w in graph.neighbors(v) {
            if distance[w] == i32::MAX {
                distance[w] = distance[v] + 1;
                queue.push_back(w);
            }
            if distance[w] == distance[v] + 1 {
                paths[w] = paths[w].saturating_add(paths[v]);
                predecessors[w].push(v);
            }
        }
    }

    while let Some(w) = stack.pop() {
        for &v in &predecessors[w] {
            let contrib = (paths[v] as f64 / paths[w] as f64) * (1.0 + dependency[w]);
            dependency[v] += contrib;
        }
    }

    dependency
}

/// BFS to compute shortest path distances from a source node.
fn bfs_distances(graph: &Graph, source: NodeId) -> Vec<usize> {
    let mut distances = vec![usize::MAX; graph.num_nodes()];
    distances[source] = 0;

    let mut queue = VecDeque::new();
    queue.push_back(source);

    while let Some(v) = queue.pop_front() {
        for &w in graph.neighbors(v) {
            if distances[w] == usize::MAX {
                distances[w] = distances[v] + 1;
                queue.push_back(w);
            }
        }
    }

    distances
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_degree_centrality_triangle() {
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], false);
        let dc = g.degree_centrality();
        assert_eq!(dc.len(), 3);
        // In triangle, all nodes have degree 2, normalized = 2/2 = 1.0
        assert!((dc[&0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pagerank_cycle() {
        let g = Graph::from_edges(&[(0, 1), (1, 2), (2, 0)], true);
        let pr = g.pagerank(0.85, 100, 1e-6).expect("should converge");
        // In a cycle, all nodes should have equal PageRank
        let sum: f64 = pr.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_betweenness_path() {
        let g = Graph::from_edges(&[(0, 1), (1, 2)], false);
        let bc = g.betweenness_centrality();
        // Middle node has highest betweenness
        assert!(bc[1] > bc[0]);
        assert!(bc[1] > bc[2]);
    }

    #[test]
    fn test_closeness_star() {
        let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], false);
        let cc = g.closeness_centrality();
        assert!(cc[0] > cc[1]); // center has highest closeness
    }

    #[test]
    fn test_harmonic_centrality() {
        let g = Graph::from_edges(&[(0, 1), (0, 2), (0, 3)], false);
        let hc = g.harmonic_centrality();
        assert!(hc[0] > hc[1]); // center most central
    }
}
