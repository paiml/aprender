//! ExecutionGraph core: struct definition, basic graph operations, scope management.

use std::collections::HashMap;

use crate::brick::exec_graph::node::{
    EdgeType, ExecutionEdge, ExecutionNode, ExecutionNodeId, TransferDirection,
};

/// Execution path graph for tracking brick → kernel → PTX relationships.
///
/// PAR-201: Captures the full execution hierarchy for profiling analysis.
///
/// # Example
///
/// ```rust,ignore
/// use trueno::brick::{ExecutionGraph, ExecutionNode, EdgeType};
///
/// let mut graph = ExecutionGraph::new();
///
/// // Add layer scope
/// let layer_id = graph.add_node(ExecutionNode::Layer { index: 0 });
///
/// // Add brick within layer
/// let brick_id = graph.add_node(ExecutionNode::Brick {
///     id: BrickId::QkvProjection,
///     timing_ns: 1000,
///     elements: 4096,
/// });
/// graph.add_edge(layer_id, brick_id, EdgeType::Contains);
///
/// // Add kernel launched by brick
/// let kernel_id = graph.add_node(ExecutionNode::Kernel {
///     name: "batched_q4k_gemv".into(),
///     ptx_hash: 0x7a3b1c2d,
///     grid: (32, 1, 1),
///     block: (256, 1, 1),
///     shared_mem: 4096,
/// });
/// graph.add_edge(brick_id, kernel_id, EdgeType::Launches);
///
/// // Export to trueno-graph for analysis
/// #[cfg(feature = "execution-graph")]
/// let csr = graph.to_csr();
/// ```
#[derive(Debug, Default)]
pub struct ExecutionGraph {
    /// All nodes in the graph
    pub(crate) nodes: Vec<ExecutionNode>,
    /// All edges in the graph
    pub(crate) edges: Vec<ExecutionEdge>,
    /// Scope stack for hierarchical recording
    pub(crate) scope_stack: Vec<ExecutionNodeId>,
    /// Node name → ID mapping for fast lookup
    pub(crate) name_to_id: HashMap<String, ExecutionNodeId>,
}

impl ExecutionGraph {
    /// Create a new empty execution graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node to the graph, returning its ID.
    pub fn add_node(&mut self, node: ExecutionNode) -> ExecutionNodeId {
        let id = ExecutionNodeId(self.nodes.len() as u32);
        let name = node.name();
        self.name_to_id.insert(name, id);
        self.nodes.push(node);
        id
    }

    /// Add an edge between two nodes.
    pub fn add_edge(&mut self, src: ExecutionNodeId, dst: ExecutionNodeId, edge_type: EdgeType) {
        debug_assert!(
            (src.0 as usize) < self.nodes.len(),
            "CB-BUDGET: src node {} does not exist (graph has {} nodes)",
            src.0,
            self.nodes.len()
        );
        debug_assert!(
            (dst.0 as usize) < self.nodes.len(),
            "CB-BUDGET: dst node {} does not exist (graph has {} nodes)",
            dst.0,
            self.nodes.len()
        );
        self.edges.push(ExecutionEdge { src, dst, edge_type, weight: 1.0 });
    }

    /// Add an edge with a weight.
    pub fn add_weighted_edge(
        &mut self,
        src: ExecutionNodeId,
        dst: ExecutionNodeId,
        edge_type: EdgeType,
        weight: f32,
    ) {
        self.edges.push(ExecutionEdge { src, dst, edge_type, weight });
    }

    /// Push a scope for hierarchical recording.
    /// All subsequent nodes will be children of this scope.
    pub fn push_scope(&mut self, node: ExecutionNode) -> ExecutionNodeId {
        let id = self.add_node(node);
        if let Some(&parent) = self.scope_stack.last() {
            self.add_edge(parent, id, EdgeType::Contains);
        }
        self.scope_stack.push(id);
        id
    }

    /// Pop the current scope.
    pub fn pop_scope(&mut self) -> Option<ExecutionNodeId> {
        self.scope_stack.pop()
    }

    /// Get the current scope (if any).
    pub fn current_scope(&self) -> Option<ExecutionNodeId> {
        self.scope_stack.last().copied()
    }

    /// Add a node under the current scope.
    pub fn add_node_in_scope(&mut self, node: ExecutionNode) -> ExecutionNodeId {
        let id = self.add_node(node);
        if let Some(&parent) = self.scope_stack.last() {
            self.add_edge(parent, id, EdgeType::Contains);
        }
        id
    }

    /// Record a kernel launch under the current scope.
    pub fn record_kernel_launch(
        &mut self,
        name: &str,
        ptx_hash: u64,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
    ) -> ExecutionNodeId {
        debug_assert!(grid.0 > 0 && grid.1 > 0 && grid.2 > 0, "CB-BUDGET: grid dims must be > 0");
        debug_assert!(
            block.0 > 0 && block.1 > 0 && block.2 > 0,
            "CB-BUDGET: block dims must be > 0"
        );
        let kernel = ExecutionNode::Kernel {
            name: name.to_string(),
            ptx_hash,
            grid,
            block,
            shared_mem,
            timing_ns: None,
            arithmetic_intensity: None,
            achieved_tflops: None,
        };
        let kernel_id = self.add_node(kernel);

        // Link from current scope with Launches edge
        if let Some(&parent) = self.scope_stack.last() {
            self.add_edge(parent, kernel_id, EdgeType::Launches);
        }

        kernel_id
    }

    /// Record a kernel launch with roofline metrics (Phase 9).
    #[allow(clippy::too_many_arguments)]
    pub fn record_kernel_launch_with_metrics(
        &mut self,
        name: &str,
        ptx_hash: u64,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
        timing_ns: u64,
        arithmetic_intensity: f32,
        achieved_tflops: f32,
    ) -> ExecutionNodeId {
        let kernel = ExecutionNode::Kernel {
            name: name.to_string(),
            ptx_hash,
            grid,
            block,
            shared_mem,
            timing_ns: Some(timing_ns),
            arithmetic_intensity: Some(arithmetic_intensity),
            achieved_tflops: Some(achieved_tflops),
        };
        let kernel_id = self.add_node(kernel);

        if let Some(&parent) = self.scope_stack.last() {
            self.add_edge(parent, kernel_id, EdgeType::Launches);
        }

        kernel_id
    }

    /// Record a memory transfer (Phase 9: data movement topology).
    pub fn record_transfer(
        &mut self,
        src: &str,
        dst: &str,
        bytes: u64,
        direction: TransferDirection,
        timing_ns: Option<u64>,
    ) -> ExecutionNodeId {
        let transfer = ExecutionNode::Transfer {
            src: src.to_string(),
            dst: dst.to_string(),
            bytes,
            direction,
            timing_ns,
        };
        let transfer_id = self.add_node(transfer);

        if let Some(&parent) = self.scope_stack.last() {
            self.add_edge(parent, transfer_id, EdgeType::Contains);
        }

        transfer_id
    }

    /// Add a dependency edge for critical path analysis (Phase 9).
    pub fn add_dependency(&mut self, from: ExecutionNodeId, to: ExecutionNodeId) {
        self.add_edge(from, to, EdgeType::DependsOn);
    }

    /// Get a node by ID.
    pub fn node(&self, id: ExecutionNodeId) -> Option<&ExecutionNode> {
        self.nodes.get(id.0 as usize)
    }

    /// Get a node by name.
    pub fn node_by_name(&self, name: &str) -> Option<(ExecutionNodeId, &ExecutionNode)> {
        self.name_to_id.get(name).and_then(|&id| self.nodes.get(id.0 as usize).map(|n| (id, n)))
    }

    /// Get all nodes.
    pub fn nodes(&self) -> &[ExecutionNode] {
        &self.nodes
    }

    /// Get all edges.
    pub fn edges(&self) -> &[ExecutionEdge] {
        &self.edges
    }

    /// Number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges.
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Get outgoing edges for a node.
    pub fn outgoing_edges(&self, node: ExecutionNodeId) -> impl Iterator<Item = &ExecutionEdge> {
        self.edges.iter().filter(move |e| e.src == node)
    }

    /// Get incoming edges for a node.
    pub fn incoming_edges(&self, node: ExecutionNodeId) -> impl Iterator<Item = &ExecutionEdge> {
        self.edges.iter().filter(move |e| e.dst == node)
    }

    /// Find all kernel nodes.
    pub fn kernel_nodes(&self) -> impl Iterator<Item = (ExecutionNodeId, &ExecutionNode)> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.is_kernel())
            .map(|(i, n)| (ExecutionNodeId(i as u32), n))
    }

    /// Find the slowest kernel (by parent brick timing).
    pub fn slowest_kernel(&self) -> Option<(ExecutionNodeId, &ExecutionNode, u64)> {
        let mut slowest: Option<(ExecutionNodeId, &ExecutionNode, u64)> = None;

        for (id, node) in self.nodes.iter().enumerate() {
            if let ExecutionNode::Brick { timing_ns, .. } = node {
                // Check if this brick has kernel children
                let node_id = ExecutionNodeId(id as u32);
                let has_kernel =
                    self.outgoing_edges(node_id).any(|e| e.edge_type == EdgeType::Launches);

                if has_kernel {
                    match &slowest {
                        None => slowest = Some((node_id, node, *timing_ns)),
                        Some((_, _, t)) if *timing_ns > *t => {
                            slowest = Some((node_id, node, *timing_ns))
                        }
                        Some(_) => {} // Keep existing slowest
                    }
                }
            }
        }

        slowest
    }

    /// Clear the graph.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.scope_stack.clear();
        self.name_to_id.clear();
    }

    /// Check if scope stack is balanced (empty).
    pub fn is_scope_balanced(&self) -> bool {
        self.scope_stack.is_empty()
    }
}
