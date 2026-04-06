//! CSR (Compressed Sparse Row) graph representation
//!
//! Based on `GraphBLAST` (Yang et al., ACM `ToMS` 2022) for GPU-optimized sparse matrix operations.
//!
//! # CSR Format
//!
//! ```text
//! Graph: 0 → 1, 0 → 2, 1 → 2
//!
//! CSR:
//!   row_offsets: [0, 2, 3, 3]  // Node 0: edges [0..2), Node 1: [2..3), Node 2: [3..3)
//!   col_indices: [1, 2, 2]      // Edge 0 → node 1, edge 1 → node 2, edge 2 → node 2
//!   edge_weights: [1.0, 1.0, 1.0]
//! ```

use anyhow::{anyhow, Result};
use std::collections::HashMap;

/// Node identifier (zero-indexed)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u32);

/// CSR (Compressed Sparse Row) graph
///
/// Optimized for:
/// - O(1) access to outgoing edges (via forward CSR)
/// - O(1) access to incoming edges (via reverse CSR)
/// - GPU-friendly memory layout
/// - Sparse matrix operations (via aprender)
///
/// # Example
///
/// ```
/// use trueno_graph::{CsrGraph, NodeId};
///
/// let mut graph = CsrGraph::new();
/// graph.add_edge(NodeId(0), NodeId(1), 1.0).unwrap();
/// graph.add_edge(NodeId(0), NodeId(2), 1.0).unwrap();
///
/// let neighbors = graph.outgoing_neighbors(NodeId(0)).unwrap();
/// assert_eq!(neighbors.len(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct CsrGraph {
    /// Forward CSR: Row offsets for outgoing edges
    /// node i's edges start at `row_offsets`[i]
    /// Length: `num_nodes` + 1
    row_offsets: Vec<u32>,

    /// Forward CSR: Column indices (edge targets)
    /// Length: `num_edges`
    col_indices: Vec<u32>,

    /// Forward CSR: Edge weights
    /// Length: `num_edges`
    edge_weights: Vec<f32>,

    /// Reverse CSR: Row offsets for incoming edges
    /// node i's incoming edges start at `rev_row_offsets`[i]
    /// Length: `num_nodes` + 1
    rev_row_offsets: Vec<u32>,

    /// Reverse CSR: Column indices (edge sources)
    /// Length: `num_edges`
    rev_col_indices: Vec<u32>,

    /// Reverse CSR: Edge weights (same as forward, but reordered)
    /// Length: `num_edges`
    rev_edge_weights: Vec<f32>,

    /// Node names (for debugging/export)
    node_names: HashMap<NodeId, String>,

    /// Number of nodes
    num_nodes: usize,
}

impl CsrGraph {
    /// Create new empty graph
    #[must_use]
    pub fn new() -> Self {
        Self {
            row_offsets: vec![0], // Start with single offset
            col_indices: Vec::new(),
            edge_weights: Vec::new(),
            rev_row_offsets: vec![0], // Reverse CSR offsets
            rev_col_indices: Vec::new(),
            rev_edge_weights: Vec::new(),
            node_names: HashMap::new(),
            num_nodes: 0,
        }
    }

    /// Create graph from edge list
    ///
    /// # Arguments
    ///
    /// * `edges` - List of (source, target, weight) tuples
    ///
    /// # Errors
    ///
    /// Returns error if edge list is invalid (e.g., negative node IDs)
    pub fn from_edge_list(edges: &[(NodeId, NodeId, f32)]) -> Result<Self> {
        if edges.is_empty() {
            return Ok(Self::new());
        }

        // Find max node ID to determine graph size
        let max_node = edges
            .iter()
            .flat_map(|(src, dst, _)| [src.0, dst.0])
            .max()
            .ok_or_else(|| anyhow!("Empty edge list"))?;

        let num_nodes = (max_node + 1) as usize;

        // Build intermediate adjacency lists for both forward and reverse directions
        let mut adj_list: Vec<Vec<(u32, f32)>> = vec![Vec::new(); num_nodes];
        let mut rev_adj_list: Vec<Vec<(u32, f32)>> = vec![Vec::new(); num_nodes];

        for (src, dst, weight) in edges {
            adj_list[src.0 as usize].push((dst.0, *weight));
            rev_adj_list[dst.0 as usize].push((src.0, *weight)); // Reverse: dst ← src
        }

        // Build forward CSR
        let mut row_offsets = Vec::with_capacity(num_nodes + 1);
        let mut col_indices = Vec::new();
        let mut edge_weights_vec = Vec::new();

        let mut offset = 0_u32;
        row_offsets.push(offset);

        for neighbors in &adj_list {
            #[allow(clippy::cast_possible_truncation)] // Graphs >4B nodes not supported yet
            let len_u32 = neighbors.len() as u32;
            offset += len_u32;
            row_offsets.push(offset);

            for (target, weight) in neighbors {
                col_indices.push(*target);
                edge_weights_vec.push(*weight);
            }
        }

        // Build reverse CSR
        let mut rev_row_offsets = Vec::with_capacity(num_nodes + 1);
        let mut rev_col_indices = Vec::new();
        let mut rev_edge_weights_vec = Vec::new();

        let mut rev_offset = 0_u32;
        rev_row_offsets.push(rev_offset);

        for rev_neighbors in &rev_adj_list {
            #[allow(clippy::cast_possible_truncation)]
            let len_u32 = rev_neighbors.len() as u32;
            rev_offset += len_u32;
            rev_row_offsets.push(rev_offset);

            for (source, weight) in rev_neighbors {
                rev_col_indices.push(*source);
                rev_edge_weights_vec.push(*weight);
            }
        }

        Ok(Self {
            row_offsets,
            col_indices,
            edge_weights: edge_weights_vec,
            rev_row_offsets,
            rev_col_indices,
            rev_edge_weights: rev_edge_weights_vec,
            node_names: HashMap::new(),
            num_nodes,
        })
    }

    /// Add edge to graph (dynamic insertion)
    ///
    /// Note: For large graphs, use `from_edge_list` for better performance.
    ///
    /// # Errors
    ///
    /// Returns error if graph is already finalized (immutable CSR)
    pub fn add_edge(&mut self, src: NodeId, dst: NodeId, weight: f32) -> Result<()> {
        // Expand graph if needed
        let max_node = src.0.max(dst.0) as usize;
        if max_node >= self.num_nodes {
            self.expand_to(max_node + 1);
        }

        // Insert forward edge (src → dst)
        let src_idx = src.0 as usize;
        let end = self.row_offsets[src_idx + 1] as usize;

        self.col_indices.insert(end, dst.0);
        self.edge_weights.insert(end, weight);

        // Update row offsets for all nodes after src
        for offset in &mut self.row_offsets[src_idx + 1..] {
            *offset += 1;
        }

        // Insert reverse edge (dst ← src)
        let dst_idx = dst.0 as usize;
        let rev_end = self.rev_row_offsets[dst_idx + 1] as usize;

        self.rev_col_indices.insert(rev_end, src.0);
        self.rev_edge_weights.insert(rev_end, weight);

        // Update reverse row offsets for all nodes after dst
        for offset in &mut self.rev_row_offsets[dst_idx + 1..] {
            *offset += 1;
        }

        Ok(())
    }

    /// Get outgoing neighbors (callees) of a node
    ///
    /// # Errors
    ///
    /// Returns error if node ID is out of bounds
    pub fn outgoing_neighbors(&self, node: NodeId) -> Result<&[u32]> {
        if (node.0 as usize) >= self.num_nodes {
            return Err(anyhow!("Node ID {} out of bounds", node.0));
        }

        let idx = node.0 as usize;
        let start = self.row_offsets[idx] as usize;
        let end = self.row_offsets[idx + 1] as usize;

        Ok(&self.col_indices[start..end])
    }

    /// Get incoming neighbors (callers) of a node
    ///
    /// Returns O(1) access to incoming edges via reverse CSR.
    ///
    /// # Errors
    ///
    /// Returns error if node ID is out of bounds
    pub fn incoming_neighbors(&self, target: NodeId) -> Result<&[u32]> {
        if (target.0 as usize) >= self.num_nodes {
            return Err(anyhow!("Node ID {} out of bounds", target.0));
        }

        let idx = target.0 as usize;
        let start = self.rev_row_offsets[idx] as usize;
        let end = self.rev_row_offsets[idx + 1] as usize;

        Ok(&self.rev_col_indices[start..end])
    }

    /// Set node name (for debugging/export)
    pub fn set_node_name(&mut self, node: NodeId, name: String) {
        self.node_names.insert(node, name);
    }

    /// Get node name
    #[must_use]
    pub fn get_node_name(&self, node: NodeId) -> Option<&str> {
        self.node_names.get(&node).map(String::as_str)
    }

    /// Get number of nodes
    #[must_use]
    pub const fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Get number of edges
    #[must_use]
    pub fn num_edges(&self) -> usize {
        self.col_indices.len()
    }

    /// Get row offsets as slice (for GPU upload)
    #[must_use]
    pub fn row_offsets_slice(&self) -> &[u32] {
        &self.row_offsets
    }

    /// Get column indices as slice (for GPU upload)
    #[must_use]
    pub fn col_indices_slice(&self) -> &[u32] {
        &self.col_indices
    }

    /// Get edge weights as slice (for GPU upload)
    #[must_use]
    pub fn edge_weights_slice(&self) -> &[f32] {
        &self.edge_weights
    }

    /// Get adjacency list for a specific node
    ///
    /// Returns (targets, weights) slices for the node's outgoing edges
    #[must_use]
    pub fn adjacency(&self, node_id: NodeId) -> (&[u32], &[f32]) {
        let idx = node_id.0 as usize;
        if idx >= self.num_nodes {
            return (&[], &[]);
        }

        let start = self.row_offsets[idx] as usize;
        let end = self.row_offsets[idx + 1] as usize;

        (&self.col_indices[start..end], &self.edge_weights[start..end])
    }

    /// Iterate over all nodes and their adjacency lists
    ///
    /// Yields (`node_id`, `targets`, `weights`) tuples
    pub fn iter_adjacency(&self) -> impl Iterator<Item = (NodeId, &[u32], &[f32])> + '_ {
        (0..self.num_nodes).map(move |node_id| {
            let start = self.row_offsets[node_id] as usize;
            let end = self.row_offsets[node_id + 1] as usize;

            #[allow(clippy::cast_possible_truncation)]
            (NodeId(node_id as u32), &self.col_indices[start..end], &self.edge_weights[start..end])
        })
    }

    /// Expand graph to accommodate new nodes
    fn expand_to(&mut self, new_size: usize) {
        if new_size <= self.num_nodes {
            return;
        }

        // Add row offsets for new nodes (all point to same offset = no edges)
        let last_offset = *self.row_offsets.last().unwrap_or(&0);
        for _ in self.num_nodes..new_size {
            self.row_offsets.push(last_offset);
        }

        // Add reverse row offsets for new nodes
        let rev_last_offset = *self.rev_row_offsets.last().unwrap_or(&0);
        for _ in self.num_nodes..new_size {
            self.rev_row_offsets.push(rev_last_offset);
        }

        self.num_nodes = new_size;
    }

    /// Get CSR components (for aprender integration)
    #[must_use]
    pub fn csr_components(&self) -> (&[u32], &[u32], &[f32]) {
        (&self.row_offsets, &self.col_indices, &self.edge_weights)
    }
}

impl Default for CsrGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_graph() {
        let graph = CsrGraph::new();
        assert_eq!(graph.num_nodes(), 0);
        assert_eq!(graph.num_edges(), 0);
    }

    #[test]
    fn test_from_edge_list_simple() {
        let edges = vec![
            (NodeId(0), NodeId(1), 1.0),
            (NodeId(0), NodeId(2), 1.0),
            (NodeId(1), NodeId(2), 1.0),
        ];

        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        assert_eq!(graph.num_nodes(), 3);
        assert_eq!(graph.num_edges(), 3);

        // Check CSR structure
        assert_eq!(graph.row_offsets, vec![0, 2, 3, 3]);
        assert_eq!(graph.col_indices, vec![1, 2, 2]);
        assert_eq!(graph.edge_weights, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_outgoing_neighbors() {
        let edges = vec![(NodeId(0), NodeId(1), 1.0), (NodeId(0), NodeId(2), 2.0)];

        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        let neighbors = graph.outgoing_neighbors(NodeId(0)).unwrap();
        assert_eq!(neighbors, &[1, 2]);

        let neighbors = graph.outgoing_neighbors(NodeId(1)).unwrap();
        let empty: &[u32] = &[];
        assert_eq!(neighbors, empty);
    }

    #[test]
    fn test_incoming_neighbors() {
        let edges = vec![(NodeId(0), NodeId(2), 1.0), (NodeId(1), NodeId(2), 1.0)];

        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        let callers = graph.incoming_neighbors(NodeId(2)).unwrap();
        assert_eq!(callers.len(), 2);
        assert!(callers.contains(&0));
        assert!(callers.contains(&1));
    }

    #[test]
    fn test_reverse_csr_structure() {
        // Build a simple graph to verify reverse CSR structure
        let edges = vec![
            (NodeId(0), NodeId(1), 1.0), // 0 → 1
            (NodeId(0), NodeId(2), 2.0), // 0 → 2
            (NodeId(1), NodeId(2), 3.0), // 1 → 2
        ];

        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        // Node 0: no incoming edges
        let empty: &[u32] = &[];
        assert_eq!(graph.incoming_neighbors(NodeId(0)).unwrap(), empty);

        // Node 1: incoming from 0
        assert_eq!(graph.incoming_neighbors(NodeId(1)).unwrap(), &[0]);

        // Node 2: incoming from 0 and 1
        let node2_incoming = graph.incoming_neighbors(NodeId(2)).unwrap();
        assert_eq!(node2_incoming.len(), 2);
        assert!(node2_incoming.contains(&0));
        assert!(node2_incoming.contains(&1));
    }

    #[test]
    fn test_reverse_csr_multi_edges() {
        // Test that reverse CSR correctly handles multi-edges (duplicate edges)
        let edges = vec![
            (NodeId(0), NodeId(1), 1.0),
            (NodeId(0), NodeId(1), 2.0), // Duplicate edge with different weight
            (NodeId(2), NodeId(1), 3.0),
        ];

        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        // Node 1 should have 3 incoming edges (2 from node 0, 1 from node 2)
        let incoming = graph.incoming_neighbors(NodeId(1)).unwrap();
        assert_eq!(incoming.len(), 3);

        // Count occurrences
        let count_0 = incoming.iter().filter(|&&x| x == 0).count();
        let count_2 = incoming.iter().filter(|&&x| x == 2).count();

        assert_eq!(count_0, 2, "Should have 2 edges from node 0");
        assert_eq!(count_2, 1, "Should have 1 edge from node 2");
    }

    #[test]
    fn test_reverse_csr_with_add_edge() {
        // Test that reverse CSR is correctly updated when using add_edge
        let mut graph = CsrGraph::new();

        graph.add_edge(NodeId(0), NodeId(1), 1.0).unwrap();
        graph.add_edge(NodeId(2), NodeId(1), 2.0).unwrap();

        // Node 1 should have incoming edges from 0 and 2
        let incoming = graph.incoming_neighbors(NodeId(1)).unwrap();
        assert_eq!(incoming.len(), 2);
        assert!(incoming.contains(&0));
        assert!(incoming.contains(&2));

        // Add another edge
        graph.add_edge(NodeId(3), NodeId(1), 3.0).unwrap();

        // Now node 1 should have 3 incoming edges
        let incoming = graph.incoming_neighbors(NodeId(1)).unwrap();
        assert_eq!(incoming.len(), 3);
        assert!(incoming.contains(&0));
        assert!(incoming.contains(&2));
        assert!(incoming.contains(&3));
    }

    #[test]
    fn test_add_edge_dynamic() {
        let mut graph = CsrGraph::new();

        graph.add_edge(NodeId(0), NodeId(1), 1.0).unwrap();
        graph.add_edge(NodeId(0), NodeId(2), 1.0).unwrap();

        assert_eq!(graph.num_nodes(), 3);
        assert_eq!(graph.num_edges(), 2);

        let neighbors = graph.outgoing_neighbors(NodeId(0)).unwrap();
        assert_eq!(neighbors, &[1, 2]);
    }

    #[test]
    fn test_node_names() {
        let mut graph = CsrGraph::new();
        graph.add_edge(NodeId(0), NodeId(1), 1.0).unwrap();

        graph.set_node_name(NodeId(0), "main".to_string());
        graph.set_node_name(NodeId(1), "parse_args".to_string());

        assert_eq!(graph.get_node_name(NodeId(0)), Some("main"));
        assert_eq!(graph.get_node_name(NodeId(1)), Some("parse_args"));
    }

    #[test]
    fn test_csr_components() {
        let edges = vec![(NodeId(0), NodeId(1), 1.0), (NodeId(0), NodeId(2), 2.0)];

        let graph = CsrGraph::from_edge_list(&edges).unwrap();
        let (row_offsets, col_indices, weights) = graph.csr_components();

        assert_eq!(row_offsets, &[0, 2, 2, 2]);
        assert_eq!(col_indices, &[1, 2]);
        assert_eq!(weights, &[1.0, 2.0]);
    }

    #[test]
    fn test_adjacency() {
        let edges = vec![
            (NodeId(0), NodeId(1), 1.5),
            (NodeId(0), NodeId(2), 2.5),
            (NodeId(1), NodeId(2), 3.5),
        ];

        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        // Node 0 has two outgoing edges
        let (targets, weights) = graph.adjacency(NodeId(0));
        assert_eq!(targets, &[1, 2]);
        assert_eq!(weights, &[1.5, 2.5]);

        // Node 1 has one outgoing edge
        let (targets, weights) = graph.adjacency(NodeId(1));
        assert_eq!(targets, &[2]);
        assert_eq!(weights, &[3.5]);

        // Node 2 has no outgoing edges
        let (targets, weights) = graph.adjacency(NodeId(2));
        let empty_u32: &[u32] = &[];
        let empty_f32: &[f32] = &[];
        assert_eq!(targets, empty_u32);
        assert_eq!(weights, empty_f32);
    }

    #[test]
    fn test_adjacency_out_of_bounds() {
        let edges = vec![(NodeId(0), NodeId(1), 1.0)];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        // Out of bounds node should return empty slices
        let (targets, weights) = graph.adjacency(NodeId(999));
        let empty_u32: &[u32] = &[];
        let empty_f32: &[f32] = &[];
        assert_eq!(targets, empty_u32);
        assert_eq!(weights, empty_f32);
    }

    #[test]
    fn test_iter_adjacency() {
        let edges = vec![
            (NodeId(0), NodeId(1), 1.0),
            (NodeId(0), NodeId(2), 2.0),
            (NodeId(1), NodeId(2), 3.0),
        ];

        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        let adjacencies: Vec<_> = graph.iter_adjacency().collect();

        assert_eq!(adjacencies.len(), 3);

        // Node 0
        assert_eq!(adjacencies[0].0, NodeId(0));
        assert_eq!(adjacencies[0].1, &[1, 2]);
        assert_eq!(adjacencies[0].2, &[1.0, 2.0]);

        // Node 1
        assert_eq!(adjacencies[1].0, NodeId(1));
        assert_eq!(adjacencies[1].1, &[2]);
        assert_eq!(adjacencies[1].2, &[3.0]);

        // Node 2
        assert_eq!(adjacencies[2].0, NodeId(2));
        let empty_u32: &[u32] = &[];
        let empty_f32: &[f32] = &[];
        assert_eq!(adjacencies[2].1, empty_u32);
        assert_eq!(adjacencies[2].2, empty_f32);
    }

    #[test]
    fn test_slice_methods() {
        let edges = vec![(NodeId(0), NodeId(1), 1.0), (NodeId(0), NodeId(2), 2.0)];
        let graph = CsrGraph::from_edge_list(&edges).unwrap();

        // Test slice methods
        assert_eq!(graph.row_offsets_slice(), &[0, 2, 2, 2]);
        assert_eq!(graph.col_indices_slice(), &[1, 2]);
        assert_eq!(graph.edge_weights_slice(), &[1.0, 2.0]);
    }

    #[test]
    fn test_get_node_name_nonexistent() {
        let graph = CsrGraph::new();
        assert_eq!(graph.get_node_name(NodeId(0)), None);
    }

    #[test]
    fn test_empty_adjacency_iterator() {
        let graph = CsrGraph::new();
        let adjacencies: Vec<_> = graph.iter_adjacency().collect();
        assert_eq!(adjacencies.len(), 0);
    }
}
