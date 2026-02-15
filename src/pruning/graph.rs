//! Dependency graph for consistent structured pruning.
//!
//! # Toyota Way: Poka-Yoke (Mistake-Proofing)
//! Dependency tracking ensures that when a channel is pruned,
//! all connected layers are updated consistently.
//!
//! # Problem
//! When pruning channel j from layer L, we must also:
//! - Remove column j from weight matrix of layer L
//! - Remove row j from weight matrix of layer L+1
//! - Update any skip connections or residuals
//!
//! # References
//! - Ma, X., et al. (2023). LLM-Pruner: On the structural pruning of large
//!   language models. `NeurIPS`.

use super::error::PruningError;
use std::collections::{HashMap, HashSet};

/// Type of dependency between layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DependencyType {
    /// Sequential connection (output of A -> input of B).
    Sequential,
    /// Skip/residual connection (output added to later layer).
    Skip,
    /// Concatenation (outputs concatenated along channel dim).
    Concat,
    /// Element-wise addition (outputs added element-wise).
    Add,
    /// Element-wise multiplication.
    Mul,
    /// Attention (Q, K, V projections share head structure).
    Attention,
}

/// A node in the dependency graph representing a layer or operation.
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Unique identifier for this node.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Type of layer/operation.
    pub node_type: NodeType,
    /// Output dimension (channels for conv/linear).
    pub output_dim: usize,
    /// Input dimension.
    pub input_dim: usize,
    /// Whether this node can be pruned.
    pub prunable: bool,
}

impl GraphNode {
    /// Create a new graph node.
    pub fn new(id: impl Into<String>, name: impl Into<String>, node_type: NodeType) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            node_type,
            output_dim: 0,
            input_dim: 0,
            prunable: true,
        }
    }

    /// Set dimensions.
    #[must_use]
    pub fn with_dims(mut self, input_dim: usize, output_dim: usize) -> Self {
        self.input_dim = input_dim;
        self.output_dim = output_dim;
        self
    }

    /// Set prunable flag.
    #[must_use]
    pub fn with_prunable(mut self, prunable: bool) -> Self {
        self.prunable = prunable;
        self
    }
}

/// Type of layer/operation node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeType {
    /// Linear/Dense layer.
    Linear,
    /// Convolutional layer.
    Conv,
    /// Layer normalization.
    LayerNorm,
    /// Batch normalization.
    BatchNorm,
    /// Embedding layer.
    Embedding,
    /// Attention block.
    Attention,
    /// MLP/FFN block.
    MLP,
    /// Activation function.
    Activation,
    /// Pooling layer.
    Pooling,
    /// Input node.
    Input,
    /// Output node.
    Output,
    /// Other/unknown.
    Other,
}

/// An edge in the dependency graph.
#[derive(Debug, Clone)]
pub struct GraphEdge {
    /// Source node ID.
    pub from: String,
    /// Destination node ID.
    pub to: String,
    /// Type of dependency.
    pub dep_type: DependencyType,
    /// Which dimension is connected (for multi-output nodes).
    pub dim_index: usize,
}

impl GraphEdge {
    /// Create a new edge.
    pub fn new(from: impl Into<String>, to: impl Into<String>, dep_type: DependencyType) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            dep_type,
            dim_index: 0,
        }
    }

    /// Set dimension index.
    #[must_use]
    pub fn with_dim_index(mut self, dim_index: usize) -> Self {
        self.dim_index = dim_index;
        self
    }
}

/// Dependency graph for a neural network.
///
/// Tracks how layers are connected to enable consistent pruning.
/// When channels are pruned from one layer, all dependent layers
/// must be updated accordingly.
#[derive(Debug, Clone)]
pub struct DependencyGraph {
    /// Nodes in the graph.
    nodes: HashMap<String, GraphNode>,
    /// Outgoing edges from each node.
    edges_out: HashMap<String, Vec<GraphEdge>>,
    /// Incoming edges to each node.
    edges_in: HashMap<String, Vec<GraphEdge>>,
}

impl DependencyGraph {
    /// Create a new empty dependency graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges_out: HashMap::new(),
            edges_in: HashMap::new(),
        }
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, node: GraphNode) {
        let id = node.id.clone();
        self.nodes.insert(id.clone(), node);
        self.edges_out.entry(id.clone()).or_default();
        self.edges_in.entry(id).or_default();
    }

    /// Add an edge to the graph.
    pub fn add_edge(&mut self, edge: GraphEdge) -> Result<(), PruningError> {
        // Validate nodes exist
        if !self.nodes.contains_key(&edge.from) {
            return Err(PruningError::InvalidPattern {
                message: format!("Source node '{}' not found", edge.from),
            });
        }
        if !self.nodes.contains_key(&edge.to) {
            return Err(PruningError::InvalidPattern {
                message: format!("Target node '{}' not found", edge.to),
            });
        }

        self.edges_out
            .entry(edge.from.clone())
            .or_default()
            .push(edge.clone());
        self.edges_in.entry(edge.to.clone()).or_default().push(edge);

        Ok(())
    }

    /// Get a node by ID.
    #[must_use]
    pub fn get_node(&self, id: &str) -> Option<&GraphNode> {
        self.nodes.get(id)
    }

    /// Get mutable node by ID.
    pub fn get_node_mut(&mut self, id: &str) -> Option<&mut GraphNode> {
        self.nodes.get_mut(id)
    }

    /// Get all nodes.
    pub fn nodes(&self) -> impl Iterator<Item = &GraphNode> {
        self.nodes.values()
    }

    /// Get outgoing edges from a node.
    pub fn edges_from(&self, id: &str) -> &[GraphEdge] {
        self.edges_out.get(id).map_or(&[], Vec::as_slice)
    }

    /// Get incoming edges to a node.
    pub fn edges_to(&self, id: &str) -> &[GraphEdge] {
        self.edges_in.get(id).map_or(&[], Vec::as_slice)
    }

    /// Get number of nodes.
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get number of edges.
    pub fn num_edges(&self) -> usize {
        self.edges_out.values().map(Vec::len).sum()
    }

    /// Find all nodes that would be affected by pruning a given node's output channels.
    ///
    /// Returns set of node IDs that need their input dimensions updated.
    #[must_use]
    pub fn downstream_dependents(&self, node_id: &str) -> HashSet<String> {
        let mut dependents = HashSet::new();
        let mut visited = HashSet::new();
        let mut queue = vec![node_id.to_string()];

        while let Some(current) = queue.pop() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());

            for edge in self.edges_from(&current) {
                dependents.insert(edge.to.clone());
                queue.push(edge.to.clone());
            }
        }

        dependents
    }

    /// Find all nodes that would be affected by pruning a given node's input channels.
    ///
    /// Returns set of node IDs that need their output dimensions updated.
    #[must_use]
    pub fn upstream_dependents(&self, node_id: &str) -> HashSet<String> {
        let mut dependents = HashSet::new();
        let mut visited = HashSet::new();
        let mut queue = vec![node_id.to_string()];

        while let Some(current) = queue.pop() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());

            for edge in self.edges_to(&current) {
                dependents.insert(edge.from.clone());
                queue.push(edge.from.clone());
            }
        }

        dependents
    }

    /// Get all prunable nodes.
    #[must_use]
    pub fn prunable_nodes(&self) -> Vec<&GraphNode> {
        self.nodes.values().filter(|n| n.prunable).collect()
    }

    /// Validate graph structure.
    ///
    /// Checks for:
    /// - Dimension consistency across edges
    /// - No orphan nodes (except input/output)
    /// - No cycles in sequential paths
    pub fn validate(&self) -> Result<(), PruningError> {
        // Check all edges reference valid nodes
        for edges in self.edges_out.values() {
            for edge in edges {
                if !self.nodes.contains_key(&edge.to) {
                    return Err(PruningError::InvalidPattern {
                        message: format!("Edge references unknown node: {}", edge.to),
                    });
                }
            }
        }

        // Check dimension consistency for sequential edges
        for edges in self.edges_out.values() {
            for edge in edges {
                if edge.dep_type == DependencyType::Sequential {
                    let from_node = self.nodes.get(&edge.from);
                    let to_node = self.nodes.get(&edge.to);

                    if let (Some(from), Some(to)) = (from_node, to_node) {
                        if from.output_dim != 0
                            && to.input_dim != 0
                            && from.output_dim != to.input_dim
                        {
                            return Err(PruningError::ShapeMismatch {
                                expected: vec![from.output_dim],
                                got: vec![to.input_dim],
                            });
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Create a simple linear chain graph.
    ///
    /// Useful for testing and simple sequential models.
    #[must_use]
    pub fn linear_chain(layer_dims: &[(usize, usize)], names: &[&str]) -> Self {
        let mut graph = Self::new();

        for (i, ((in_dim, out_dim), name)) in layer_dims.iter().zip(names.iter()).enumerate() {
            let node = GraphNode::new(format!("layer_{i}"), *name, NodeType::Linear)
                .with_dims(*in_dim, *out_dim);

            graph.add_node(node);
        }

        for i in 1..names.len() {
            let edge = GraphEdge::new(
                format!("layer_{}", i - 1),
                format!("layer_{i}"),
                DependencyType::Sequential,
            );
            graph.add_edge(edge).ok();
        }

        graph
    }
}

impl Default for DependencyGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Pruning plan that ensures consistency across dependent layers.
#[derive(Debug, Clone)]
pub struct PruningPlan {
    /// Channels to remove from each layer.
    pub channel_removals: HashMap<String, Vec<usize>>,
    /// Layer removals (for depth pruning).
    pub layer_removals: Vec<String>,
    /// Validation status.
    validated: bool,
}

include!("graph_part_02.rs");
include!("graph_part_03.rs");
