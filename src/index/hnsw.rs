//! Hierarchical Navigable Small World (HNSW) index.
//!
//! HNSW is a graph-based approximate nearest neighbor search algorithm
//! achieving O(log n) query complexity with high recall (>95%).
//!
//! # Algorithm
//!
//! - Multi-layer graph structure (like skip lists)
//! - Each node connects to M neighbors per layer
//! - Top layers: sparse (navigation), bottom layer: dense (all elements)
//! - Search: greedy descent from top to bottom
//!
//! # References
//!
//! Malkov & Yashunin (2018). "Efficient and robust approximate nearest
//! neighbor search using Hierarchical Navigable Small World graphs."
//! IEEE TPAMI. <https://arxiv.org/abs/1603.09320>
//!
//! # Examples
//!
//! ```
//! use aprender::index::hnsw::HNSWIndex;
//! use aprender::primitives::Vector;
//!
//! let mut index = HNSWIndex::new(16, 200, 0.0);
//!
//! // Add items
//! index.add("doc1", Vector::from_slice(&[1.0, 0.0, 0.0]));
//! index.add("doc2", Vector::from_slice(&[0.0, 1.0, 0.0]));
//! index.add("doc3", Vector::from_slice(&[0.8, 0.2, 0.0]));
//!
//! // Search
//! let query = Vector::from_slice(&[0.9, 0.1, 0.0]);
//! let results = index.search(&query, 2);
//!
//! assert_eq!(results[0].0, "doc1");  // Closest
//! ```

use crate::primitives::Vector;
use rand::Rng;
use std::collections::{HashMap, HashSet};

/// HNSW index for approximate nearest neighbor search.
///
/// # Configuration
///
/// - `m`: Max connections per node (typical: 12-48)
/// - `ef_construction`: Size of dynamic candidate list during construction (typical: 100-200)
/// - `ml`: Level multiplier for probabilistic layer assignment (default: 1/ln(2))
#[derive(Debug)]
pub struct HNSWIndex {
    /// Maximum number of connections per node
    m: usize,
    /// Maximum number of connections for layer 0 (2*M)
    max_m0: usize,
    /// Size of dynamic candidate list during construction
    ef_construction: usize,
    /// Level multiplier (1/ln(2) ≈ 1.44)
    ml: f64,
    /// All nodes in the graph
    nodes: Vec<Node>,
    /// Mapping from item ID to node index
    item_to_node: HashMap<String, usize>,
    /// Entry point (top layer node)
    entry_point: Option<usize>,
    /// Random number generator
    rng: rand::rngs::ThreadRng,
}

/// Node in the HNSW graph.
#[derive(Debug, Clone)]
struct Node {
    /// Item identifier
    item_id: String,
    /// Feature vector
    vector: Vector<f64>,
    /// Connections per layer (layer -> neighbor indices)
    connections: Vec<Vec<usize>>,
}

impl HNSWIndex {
    /// Create a new HNSW index.
    ///
    /// # Arguments
    ///
    /// * `m` - Maximum connections per node (12-48 recommended)
    /// * `ef_construction` - Construction parameter (100-200 recommended)
    /// * `seed` - Random seed (0.0 for default `ThreadRng`)
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::index::hnsw::HNSWIndex;
    ///
    /// let index = HNSWIndex::new(16, 200, 0.0);
    /// ```
    #[must_use]
    pub fn new(m: usize, ef_construction: usize, _seed: f64) -> Self {
        Self {
            m,
            max_m0: 2 * m,
            ef_construction,
            ml: 1.0 / (2.0_f64).ln(), // 1/ln(2)
            nodes: Vec::new(),
            item_to_node: HashMap::new(),
            entry_point: None,
            rng: rand::thread_rng(),
        }
    }

    /// Add an item to the index.
    ///
    /// # Arguments
    ///
    /// * `item_id` - Unique identifier for the item
    /// * `vector` - Feature vector
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::index::hnsw::HNSWIndex;
    /// use aprender::primitives::Vector;
    ///
    /// let mut index = HNSWIndex::new(16, 200, 0.0);
    /// index.add("item1", Vector::from_slice(&[1.0, 2.0, 3.0]));
    /// ```
    pub fn add(&mut self, item_id: impl Into<String>, vector: Vector<f64>) {
        let item_id = item_id.into();

        // Determine layer for this node
        let layer = self.random_layer();

        // Create node
        let node_idx = self.nodes.len();
        let connections = vec![Vec::new(); layer + 1];
        let node = Node {
            item_id: item_id.clone(),
            vector,
            connections,
        };

        self.nodes.push(node);
        self.item_to_node.insert(item_id, node_idx);

        // If first node, set as entry point
        if self.entry_point.is_none() {
            self.entry_point = Some(node_idx);
            return;
        }

        // Insert node into graph
        self.insert_node(node_idx, layer);
    }

    /// Search for k nearest neighbors.
    ///
    /// # Arguments
    ///
    /// * `query` - Query vector
    /// * `k` - Number of neighbors to return
    ///
    /// # Returns
    ///
    /// List of (`item_id`, distance) pairs, sorted by distance (closest first)
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::index::hnsw::HNSWIndex;
    /// use aprender::primitives::Vector;
    ///
    /// let mut index = HNSWIndex::new(16, 200, 0.0);
    /// index.add("a", Vector::from_slice(&[1.0, 0.0]));
    /// index.add("b", Vector::from_slice(&[0.0, 1.0]));
    ///
    /// let query = Vector::from_slice(&[0.9, 0.1]);
    /// let results = index.search(&query, 1);
    ///
    /// assert_eq!(results[0].0, "a");
    /// ```
    #[must_use]
    pub fn search(&self, query: &Vector<f64>, k: usize) -> Vec<(String, f64)> {
        if self.nodes.is_empty() || self.entry_point.is_none() {
            return Vec::new();
        }

        let ep = self.entry_point.expect("Entry point exists");
        let top_layer = self.nodes[ep].connections.len().saturating_sub(1);

        // Search from top layer to layer 1
        let mut curr = ep;
        for lc in (1..=top_layer).rev() {
            curr = self
                .search_layer(query, curr, 1, lc)
                .into_iter()
                .next()
                .unwrap_or(curr);
        }

        // Search layer 0 with ef
        let candidates = self.search_layer(query, curr, k.max(self.ef_construction), 0);

        // Return top-k with item IDs, sorted by distance (closest first)
        let mut results: Vec<(String, f64)> = candidates
            .into_iter()
            .map(|idx| {
                let node = &self.nodes[idx];
                let dist = Self::distance(query, &node.vector);
                (node.item_id.clone(), dist)
            })
            .collect();

        // Sort by distance (ascending - closest first)
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top-k
        results.into_iter().take(k).collect()
    }

    /// Number of items in the index.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::index::hnsw::HNSWIndex;
    /// use aprender::primitives::Vector;
    ///
    /// let mut index = HNSWIndex::new(16, 200, 0.0);
    /// assert_eq!(index.len(), 0);
    ///
    /// index.add("item1", Vector::from_slice(&[1.0]));
    /// assert_eq!(index.len(), 1);
    /// ```
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if index is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::index::hnsw::HNSWIndex;
    ///
    /// let index = HNSWIndex::new(16, 200, 0.0);
    /// assert!(index.is_empty());
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get the M parameter (max connections per node).
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::index::hnsw::HNSWIndex;
    ///
    /// let index = HNSWIndex::new(16, 200, 0.0);
    /// assert_eq!(index.m(), 16);
    /// ```
    #[must_use]
    pub fn m(&self) -> usize {
        self.m
    }

    /// Get the `ef_construction` parameter.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::index::hnsw::HNSWIndex;
    ///
    /// let index = HNSWIndex::new(16, 200, 0.0);
    /// assert_eq!(index.ef_construction(), 200);
    /// ```
    #[must_use]
    pub fn ef_construction(&self) -> usize {
        self.ef_construction
    }

    /// Randomly select layer for new node.
    ///
    /// Uses exponential decay: P(layer = l) ~ exp(-l / ml)
    fn random_layer(&mut self) -> usize {
        let r: f64 = self.rng.gen_range(0.0..1.0);
        (-r.ln() * self.ml).floor() as usize
    }

    /// Insert node into the graph at specified layer.
    fn insert_node(&mut self, node_idx: usize, layer: usize) {
        let ep = self.entry_point.expect("Entry point exists");
        let top_layer = self.nodes[ep].connections.len().saturating_sub(1);

        // Search from top to target layer
        let mut curr = ep;
        for lc in (layer + 1..=top_layer).rev() {
            curr = self
                .search_layer_node(node_idx, curr, 1, lc)
                .into_iter()
                .next()
                .unwrap_or(curr);
        }

        // Insert at each layer from top down to 0
        for lc in (0..=layer).rev() {
            let candidates = self.search_layer_node(node_idx, curr, self.ef_construction, lc);

            // Select M nearest neighbors
            let m = if lc == 0 { self.max_m0 } else { self.m };
            let neighbors: Vec<usize> = candidates.into_iter().take(m).collect();

            // Add bidirectional connections
            for &neighbor in &neighbors {
                // Add neighbor to new node
                self.nodes[node_idx].connections[lc].push(neighbor);

                // Add new node to neighbor (with pruning if needed)
                // Only if neighbor has this layer
                if lc < self.nodes[neighbor].connections.len() {
                    self.nodes[neighbor].connections[lc].push(node_idx);
                    self.prune_connections(neighbor, lc, m);
                }
            }

            if let Some(&first) = neighbors.first() {
                curr = first;
            }
        }

        // Update entry point if new node has higher layer
        if layer > top_layer {
            self.entry_point = Some(node_idx);
        }
    }

    /// Search a single layer for nearest neighbors to query.
    fn search_layer(
        &self,
        query: &Vector<f64>,
        entry: usize,
        ef: usize,
        layer: usize,
    ) -> Vec<usize> {
        let mut visited = HashSet::new();
        let mut candidates = Vec::new();
        let mut best = Vec::new();

        // Initialize with entry point
        let entry_dist = Self::distance(query, &self.nodes[entry].vector);
        candidates.push((entry, entry_dist));
        best.push((entry, entry_dist));
        visited.insert(entry);

        while let Some((curr, _)) = candidates.pop() {
            // Get worst distance in best set
            let worst_best_dist = best
                .iter()
                .map(|(_, d)| *d)
                .fold(f64::NEG_INFINITY, f64::max);

            // Stop if current is farther than worst in best
            let curr_dist = Self::distance(query, &self.nodes[curr].vector);
            if curr_dist > worst_best_dist && best.len() >= ef {
                break;
            }

            // Explore neighbors
            if layer < self.nodes[curr].connections.len() {
                for &neighbor in &self.nodes[curr].connections[layer] {
                    if visited.insert(neighbor) {
                        let neighbor_dist = Self::distance(query, &self.nodes[neighbor].vector);

                        if neighbor_dist < worst_best_dist || best.len() < ef {
                            candidates.push((neighbor, neighbor_dist));
                            best.push((neighbor, neighbor_dist));

                            // Sort by distance (ascending)
                            candidates.sort_by(|a, b| {
                                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            best.sort_by(|a, b| {
                                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                            });

                            // Keep only ef best
                            if best.len() > ef {
                                best.truncate(ef);
                            }
                        }
                    }
                }
            }
        }

        best.into_iter().map(|(idx, _)| idx).collect()
    }

    /// Search a single layer for nearest neighbors to a node.
    fn search_layer_node(
        &self,
        node_idx: usize,
        entry: usize,
        ef: usize,
        layer: usize,
    ) -> Vec<usize> {
        self.search_layer(&self.nodes[node_idx].vector, entry, ef, layer)
    }

    /// Prune connections to maintain max M neighbors.
    fn prune_connections(&mut self, node_idx: usize, layer: usize, max_m: usize) {
        if self.nodes[node_idx].connections[layer].len() <= max_m {
            return;
        }

        // Get distances to all neighbors
        let node_vec = self.nodes[node_idx].vector.clone();
        let mut neighbors: Vec<(usize, f64)> = self.nodes[node_idx].connections[layer]
            .iter()
            .map(|&neighbor| {
                let dist = Self::distance(&node_vec, &self.nodes[neighbor].vector);
                (neighbor, dist)
            })
            .collect();

        // Sort by distance (ascending)
        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Keep only max_m closest
        self.nodes[node_idx].connections[layer] = neighbors
            .into_iter()
            .take(max_m)
            .map(|(idx, _)| idx)
            .collect();
    }

    /// Compute cosine distance (1 - cosine similarity).
    ///
    /// Returns values in [0, 2]:
    /// - 0: identical vectors
    /// - 1: orthogonal
    /// - 2: opposite direction
    fn distance(a: &Vector<f64>, b: &Vector<f64>) -> f64 {
        if a.len() != b.len() {
            return f64::INFINITY;
        }

        let dot: f64 = a
            .as_slice()
            .iter()
            .zip(b.as_slice().iter())
            .map(|(x, y)| x * y)
            .sum();
        let norm_a: f64 = a.as_slice().iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.as_slice().iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return f64::INFINITY;
        }

        let cos_sim = dot / (norm_a * norm_b);
        1.0 - cos_sim.clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_index() {
        let index = HNSWIndex::new(16, 200, 0.0);
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_add_single_item() {
        let mut index = HNSWIndex::new(16, 200, 0.0);
        index.add("item1", Vector::from_slice(&[1.0, 2.0, 3.0]));
        assert_eq!(index.len(), 1);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_search_single_item() {
        let mut index = HNSWIndex::new(16, 200, 0.0);
        index.add("item1", Vector::from_slice(&[1.0, 2.0, 3.0]));

        let query = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let results = index.search(&query, 1);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "item1");
        assert!(results[0].1 < 1e-6); // Distance ~0
    }

    #[test]
    fn test_search_multiple_items() {
        let mut index = HNSWIndex::new(16, 200, 0.0);
        index.add("a", Vector::from_slice(&[1.0, 0.0, 0.0]));
        index.add("b", Vector::from_slice(&[0.0, 1.0, 0.0]));
        index.add("c", Vector::from_slice(&[0.0, 0.0, 1.0]));

        let query = Vector::from_slice(&[0.9, 0.1, 0.0]);
        let results = index.search(&query, 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "a"); // Closest to [1,0,0]
    }

    #[test]
    fn test_search_k_larger_than_index() {
        let mut index = HNSWIndex::new(16, 200, 0.0);
        index.add("a", Vector::from_slice(&[1.0]));
        index.add("b", Vector::from_slice(&[2.0]));

        let query = Vector::from_slice(&[1.5]);
        let results = index.search(&query, 10);

        assert_eq!(results.len(), 2); // Only 2 items available
    }

    #[test]
    fn test_cosine_distance() {
        // Identical vectors
        let a = Vector::from_slice(&[1.0, 0.0]);
        let b = Vector::from_slice(&[1.0, 0.0]);
        assert!(HNSWIndex::distance(&a, &b) < 1e-10);

        // Orthogonal vectors
        let a = Vector::from_slice(&[1.0, 0.0]);
        let b = Vector::from_slice(&[0.0, 1.0]);
        assert!((HNSWIndex::distance(&a, &b) - 1.0).abs() < 1e-10);

        // Opposite vectors
        let a = Vector::from_slice(&[1.0, 0.0]);
        let b = Vector::from_slice(&[-1.0, 0.0]);
        assert!((HNSWIndex::distance(&a, &b) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_search_order_by_similarity() {
        let mut index = HNSWIndex::new(16, 200, 0.0);

        // Add items at different angles (cosine distance measures angle, not magnitude)
        index.add("closest", Vector::from_slice(&[1.0, 0.1])); // ~6 degrees from x-axis
        index.add("medium", Vector::from_slice(&[1.0, 1.0])); // 45 degrees from x-axis
        index.add("farthest", Vector::from_slice(&[0.1, 1.0])); // ~84 degrees from x-axis

        let query = Vector::from_slice(&[1.0, 0.0]); // 0 degrees from x-axis
        let results = index.search(&query, 3);

        assert_eq!(results.len(), 3);
        // Results should be ordered by cosine distance (ascending)
        assert!(
            results[0].1 <= results[1].1,
            "First result should be closest"
        );
        assert!(
            results[1].1 <= results[2].1,
            "Second result should be closer than or equal to third"
        );

        // Verify that "farthest" has the largest distance
        let farthest_dist = results
            .iter()
            .find(|(id, _)| id == "farthest")
            .expect("farthest should be in results")
            .1;
        let closest_dist = results
            .iter()
            .find(|(id, _)| id == "closest")
            .expect("closest should be in results")
            .1;
        assert!(
            farthest_dist > closest_dist,
            "Farthest should have larger distance than closest"
        );
    }

    #[test]
    fn test_empty_search() {
        let index = HNSWIndex::new(16, 200, 0.0);
        let query = Vector::from_slice(&[1.0, 2.0]);
        let results = index.search(&query, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_debug() {
        let mut index = HNSWIndex::new(16, 200, 0.0);

        index.add("a", Vector::from_slice(&[1.0, 1.0]));
        index.add("b", Vector::from_slice(&[2.0, 2.0]));
        index.add("c", Vector::from_slice(&[10.0, 10.0]));

        let query = Vector::from_slice(&[0.9, 0.9]);
        let results = index.search(&query, 3);

        // Print results for debugging
        for (i, (id, dist)) in results.iter().enumerate() {
            eprintln!("[{i}] id={id}, dist={dist:.6}");
        }

        // Just check that we get 3 results
        assert_eq!(results.len(), 3);
    }
}
