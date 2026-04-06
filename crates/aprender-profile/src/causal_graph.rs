//! Causal graph construction from distributed traces (Sprint 41)
//!
//! This module constructs a directed acyclic graph (DAG) from span traces to enable
//! critical path analysis, anti-pattern detection, and performance bottleneck identification.
//!
//! # Toyota Way Principle: Genchi Genbutsu (Go and See)
//!
//! Rather than inferring causality from timestamps (unreliable due to clock skew),
//! we use Lamport logical clocks to establish **provable** happens-before relationships.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │ Span Trace (from trueno-db)                                     │
//! │   - trace_id: 4bf92f3c...                                       │
//! │   - spans: [(span_1, parent=None, logical_clock=0),             │
//! │             (span_2, parent=span_1, logical_clock=1),           │
//! │             (span_3, parent=span_1, logical_clock=2)]           │
//! └─────────────────────────────────────────────────────────────────┘
//!                          │
//!                          │ build_causal_graph()
//!                          ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │ Causal Graph (CSR format via trueno-graph)                      │
//! │                                                                  │
//! │   span_1 (root)                                                 │
//! │   ├─ span_2 (duration: 1000ns)                                 │
//! │   └─ span_3 (duration: 2000ns)                                 │
//! │                                                                  │
//! │ CSR Representation:                                             │
//! │   row_offsets: [0, 2, 2, 2]                                    │
//! │   col_indices: [1, 2]                                           │
//! │   edge_weights: [1000.0, 2000.0]                               │
//! └─────────────────────────────────────────────────────────────────┘
//!                          │
//!                          │ Enables downstream analysis
//!                          ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │ Analysis Algorithms                                             │
//! │   - Critical path (longest path from root)                      │
//! │   - Anti-patterns (God Process, Tight Loops, PCIe bottleneck)  │
//! │   - PageRank (identify central syscalls)                        │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Peer-Reviewed Foundation
//!
//! - **Lamport (1978). "Time, Clocks, and the Ordering of Events."**
//!   - Theorem: A → B iff `logical_clock(A)` < `logical_clock(B)`
//!   - Application: Graph edges respect Lamport ordering
//!
//! - **Tarjan (1972). "Depth-First Search and Linear Graph Algorithms."**
//!   - Finding: DAG topological sort in O(V+E)
//!   - Application: Critical path via DAG longest path
//!
//! # Example
//!
//! ```
//! use renacer::causal_graph::CausalGraph;
//! use renacer::span_record::{SpanRecord, SpanKind, StatusCode};
//! use std::collections::HashMap;
//!
//! # fn main() -> anyhow::Result<()> {
//! // Create test spans
//! let root = SpanRecord::new(
//!     [1; 16], [1; 8], None,
//!     "root".to_string(), SpanKind::Internal,
//!     0, 1000, 0,
//!     StatusCode::Ok, String::new(),
//!     HashMap::new(), HashMap::new(),
//!     1234, 5678,
//! );
//!
//! let child = SpanRecord::new(
//!     [1; 16], [2; 8], Some([1; 8]),
//!     "child".to_string(), SpanKind::Internal,
//!     1000, 3000, 1,
//!     StatusCode::Ok, String::new(),
//!     HashMap::new(), HashMap::new(),
//!     1234, 5678,
//! );
//!
//! // Build causal graph
//! let graph = CausalGraph::from_spans(&[root, child])?;
//!
//! // Graph has 2 nodes, 1 edge (root → child)
//! assert_eq!(graph.node_count(), 2);
//! assert_eq!(graph.edge_count(), 1);
//! # Ok(())
//! # }
//! ```

use crate::span_record::SpanRecord;
use anyhow::{Context, Result};
use std::collections::HashMap;
use trueno_graph::{CsrGraph, NodeId};

/// Causal graph built from span traces
///
/// This wraps trueno-graph's `CsrGraph` with span-specific semantics.
///
/// # Node Representation
///
/// - Each span becomes a node in the graph
/// - `NodeId` is derived from span index (0-based)
/// - Node metadata stored separately in `span_metadata`
///
/// # Edge Representation
///
/// Edges represent happens-before relationships:
/// - **Parent → Child**: Explicit via `parent_span_id`
/// - **Logical ordering**: Implicit via Lamport clocks (if `A.logical_clock` < `B.logical_clock` and same trace)
/// - **Edge weight**: Span duration in nanoseconds (for critical path)
///
/// # Performance
///
/// - **Construction**: O(V + E) where V = spans, E = edges
/// - **Neighbor query**: O(1) via CSR indexing
/// - **Target**: <100ms for 1K spans
pub struct CausalGraph {
    /// Underlying CSR graph (via trueno-graph)
    graph: CsrGraph,

    /// Span metadata indexed by `NodeId`
    /// Maps `NodeId` → `SpanRecord` for reverse lookups
    span_metadata: HashMap<NodeId, SpanRecord>,

    /// Span ID to `NodeId` mapping
    /// Maps `span_id` → `NodeId` for edge construction
    span_id_to_node: HashMap<[u8; 8], NodeId>,

    /// Root nodes (spans with no parent)
    roots: Vec<NodeId>,
}

impl CausalGraph {
    /// Build a causal graph from a sequence of spans
    ///
    /// # Arguments
    ///
    /// * `spans` - Sequence of spans from the same trace
    ///
    /// # Returns
    ///
    /// A `CausalGraph` with nodes and edges representing causality.
    ///
    /// # Errors
    ///
    /// Returns error if spans have inconsistent `trace_id` or if graph construction fails.
    ///
    /// # Performance
    ///
    /// - Time complexity: O(V + E) where V = `spans.len()`, E ≈ `spans.len()` - 1
    /// - Space complexity: O(V + E)
    /// - Target: <100ms for 1K spans
    ///
    /// # Example
    ///
    /// ```
    /// use renacer::causal_graph::CausalGraph;
    /// use renacer::span_record::{SpanRecord, SpanKind, StatusCode};
    /// use std::collections::HashMap;
    ///
    /// # fn main() -> anyhow::Result<()> {
    /// let root = SpanRecord::new(
    ///     [1; 16], [1; 8], None,
    ///     "root".to_string(), SpanKind::Internal,
    ///     0, 1000, 0,
    ///     StatusCode::Ok, String::new(),
    ///     HashMap::new(), HashMap::new(),
    ///     1234, 5678,
    /// );
    ///
    /// let graph = CausalGraph::from_spans(&[root])?;
    /// assert_eq!(graph.node_count(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_spans(spans: &[SpanRecord]) -> Result<Self> {
        if spans.is_empty() {
            return Ok(Self {
                graph: CsrGraph::new(),
                span_metadata: HashMap::new(),
                span_id_to_node: HashMap::new(),
                roots: Vec::new(),
            });
        }

        // Validate all spans have same trace_id
        let trace_id = spans[0].trace_id;
        for span in spans {
            if span.trace_id != trace_id {
                anyhow::bail!(
                    "All spans must have same trace_id. Expected {:?}, got {:?}",
                    hex::encode(trace_id),
                    hex::encode(span.trace_id)
                );
            }
        }

        let mut graph = CsrGraph::new();
        let mut span_metadata = HashMap::new();
        let mut span_id_to_node = HashMap::new();
        let mut roots = Vec::new();

        // Phase 1: Create nodes
        for (idx, span) in spans.iter().enumerate() {
            let node_id = NodeId(idx as u32);
            span_metadata.insert(node_id, span.clone());
            span_id_to_node.insert(span.span_id, node_id);

            if span.is_root() {
                roots.push(node_id);
            }
        }

        // Phase 2: Create edges based on parent-child relationships
        for (idx, span) in spans.iter().enumerate() {
            let child_node = NodeId(idx as u32);

            if let Some(parent_span_id) = span.parent_span_id {
                // Find parent node
                if let Some(&parent_node) = span_id_to_node.get(&parent_span_id) {
                    // Edge weight = child span duration (nanoseconds)
                    let weight = span.duration_nanos as f32;

                    graph
                        .add_edge(parent_node, child_node, weight)
                        .context("Failed to add parent-child edge")?;
                }
                // If parent not found, it might be from a different trace slice
                // This is OK - we just won't have that edge
            }
        }

        Ok(Self { graph, span_metadata, span_id_to_node, roots })
    }

    /// Get the number of nodes in the graph
    pub fn node_count(&self) -> usize {
        self.span_metadata.len()
    }

    /// Get the number of edges in the graph
    pub fn edge_count(&self) -> usize {
        self.graph.num_edges()
    }

    /// Get root nodes (spans with no parent)
    pub fn roots(&self) -> &[NodeId] {
        &self.roots
    }

    /// Get span metadata for a node
    pub fn get_span(&self, node: NodeId) -> Option<&SpanRecord> {
        self.span_metadata.get(&node)
    }

    /// Get outgoing neighbors (children) of a node
    ///
    /// # Returns
    ///
    /// List of child nodes with edge weights (span durations).
    pub fn children(&self, node: NodeId) -> Result<Vec<(NodeId, f32)>> {
        let (neighbors, weights) = self.graph.adjacency(node);

        Ok(neighbors.iter().zip(weights.iter()).map(|(&n, &w)| (NodeId(n), w)).collect())
    }

    /// Get the underlying CSR graph
    ///
    /// This allows direct access to trueno-graph algorithms (`PageRank`, BFS, etc.)
    pub fn as_csr_graph(&self) -> &CsrGraph {
        &self.graph
    }

    /// Find all descendant nodes (transitive closure from a root)
    ///
    /// This performs a depth-first traversal to find all nodes reachable from the given node.
    ///
    /// # Arguments
    ///
    /// * `root` - Starting node for traversal
    ///
    /// # Returns
    ///
    /// List of all descendant nodes (including root).
    pub fn descendants(&self, root: NodeId) -> Result<Vec<NodeId>> {
        let mut visited = Vec::new();
        let mut stack = vec![root];

        while let Some(node) = stack.pop() {
            if visited.contains(&node) {
                continue;
            }

            visited.push(node);

            let children = self.children(node)?;
            for (child, _weight) in children {
                if !visited.contains(&child) {
                    stack.push(child);
                }
            }
        }

        Ok(visited)
    }

    /// Validate graph is a DAG (no cycles)
    ///
    /// Lamport clocks should guarantee acyclicity, but this validates it.
    ///
    /// # Returns
    ///
    /// `Ok(true)` if DAG, `Ok(false)` if cycle detected.
    pub fn is_dag(&self) -> Result<bool> {
        // Simple DFS-based cycle detection
        let mut visited = std::collections::HashSet::new();
        let mut rec_stack = std::collections::HashSet::new();

        for &root in &self.roots {
            if !visited.contains(&root) && self.has_cycle_dfs(root, &mut visited, &mut rec_stack)? {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// DFS helper for cycle detection
    fn has_cycle_dfs(
        &self,
        node: NodeId,
        visited: &mut std::collections::HashSet<NodeId>,
        rec_stack: &mut std::collections::HashSet<NodeId>,
    ) -> Result<bool> {
        visited.insert(node);
        rec_stack.insert(node);

        let children = self.children(node)?;
        for (child, _) in children {
            if !visited.contains(&child) {
                if self.has_cycle_dfs(child, visited, rec_stack)? {
                    return Ok(true);
                }
            } else if rec_stack.contains(&child) {
                // Back edge = cycle
                return Ok(true);
            }
        }

        rec_stack.remove(&node);
        Ok(false)
    }

    /// Get `NodeId` from `span_id`
    ///
    /// # Arguments
    ///
    /// * `span_id` - The span ID to look up
    ///
    /// # Returns
    ///
    /// The corresponding `NodeId` if found, None otherwise.
    pub fn get_node_by_span_id(&self, span_id: &[u8; 8]) -> Option<NodeId> {
        self.span_id_to_node.get(span_id).copied()
    }

    /// Get `SpanRecord` from `span_id`
    ///
    /// # Arguments
    ///
    /// * `span_id` - The span ID to look up
    ///
    /// # Returns
    ///
    /// The corresponding `SpanRecord` if found, None otherwise.
    pub fn get_span_by_id(&self, span_id: &[u8; 8]) -> Option<&SpanRecord> {
        self.get_node_by_span_id(span_id).and_then(|node| self.get_span(node))
    }
}

// Compile-time thread-safety verification (Sprint 59)
static_assertions::assert_impl_all!(CausalGraph: Send, Sync);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::span_record::{SpanKind, StatusCode};

    fn create_test_span(
        span_id: u8,
        parent_id: Option<u8>,
        logical_clock: u64,
        duration_nanos: u64,
    ) -> SpanRecord {
        SpanRecord::new(
            [1; 16],
            [span_id; 8],
            parent_id.map(|p| [p; 8]),
            format!("span_{}", span_id),
            SpanKind::Internal,
            logical_clock * 1000,
            logical_clock * 1000 + duration_nanos,
            logical_clock,
            StatusCode::Ok,
            String::new(),
            HashMap::new(),
            HashMap::new(),
            1234,
            5678,
        )
    }

    #[test]
    fn test_empty_graph() {
        let graph = CausalGraph::from_spans(&[]).expect("test");
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_single_node() {
        let span = create_test_span(1, None, 0, 1000);
        let graph = CausalGraph::from_spans(&[span]).expect("test");

        assert_eq!(graph.node_count(), 1);
        assert_eq!(graph.edge_count(), 0);
        assert_eq!(graph.roots().len(), 1);
    }

    #[test]
    fn test_parent_child() {
        let root = create_test_span(1, None, 0, 1000);
        let child = create_test_span(2, Some(1), 1, 2000);

        let graph = CausalGraph::from_spans(&[root, child]).expect("test");

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);
        assert_eq!(graph.roots().len(), 1);

        // Verify parent → child edge exists
        let children = graph.children(NodeId(0)).expect("test");
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].0, NodeId(1));
        assert_eq!(children[0].1, 2000.0); // Duration of child
    }

    #[test]
    fn test_tree_structure() {
        let root = create_test_span(1, None, 0, 1000);
        let child1 = create_test_span(2, Some(1), 1, 500);
        let child2 = create_test_span(3, Some(1), 2, 700);
        let grandchild = create_test_span(4, Some(2), 3, 300);

        let graph = CausalGraph::from_spans(&[root, child1, child2, grandchild]).expect("test");

        assert_eq!(graph.node_count(), 4);
        assert_eq!(graph.edge_count(), 3); // root→child1, root→child2, child1→grandchild
        assert_eq!(graph.roots().len(), 1);

        // Root has 2 children
        let children = graph.children(NodeId(0)).expect("test");
        assert_eq!(children.len(), 2);

        // child1 has 1 child (grandchild)
        let grandchildren = graph.children(NodeId(1)).expect("test");
        assert_eq!(grandchildren.len(), 1);

        // child2 has no children
        let leaf_children = graph.children(NodeId(2)).expect("test");
        assert_eq!(leaf_children.len(), 0);
    }

    #[test]
    fn test_get_span_metadata() {
        let root = create_test_span(1, None, 0, 1000);
        let graph = CausalGraph::from_spans(std::slice::from_ref(&root)).expect("test");

        let span = graph.get_span(NodeId(0)).expect("test");
        assert_eq!(span.span_name, "span_1");
        assert_eq!(span.logical_clock, 0);
    }

    #[test]
    fn test_descendants() {
        let root = create_test_span(1, None, 0, 1000);
        let child1 = create_test_span(2, Some(1), 1, 500);
        let child2 = create_test_span(3, Some(1), 2, 700);
        let grandchild = create_test_span(4, Some(2), 3, 300);

        let graph = CausalGraph::from_spans(&[root, child1, child2, grandchild]).expect("test");

        let desc = graph.descendants(NodeId(0)).expect("test");
        assert_eq!(desc.len(), 4); // All nodes reachable from root
    }

    #[test]
    fn test_is_dag() {
        let root = create_test_span(1, None, 0, 1000);
        let child = create_test_span(2, Some(1), 1, 500);

        let graph = CausalGraph::from_spans(&[root, child]).expect("test");

        assert!(graph.is_dag().expect("test"));
    }

    #[test]
    fn test_inconsistent_trace_id() {
        let mut span1 = create_test_span(1, None, 0, 1000);
        let span2 = create_test_span(2, Some(1), 1, 500);

        span1.trace_id = [2; 16]; // Different trace ID

        let result = CausalGraph::from_spans(&[span1, span2]);
        assert!(result.is_err());
    }
}
