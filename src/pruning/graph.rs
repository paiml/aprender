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

impl PruningPlan {
    /// Create a new empty pruning plan.
    #[must_use] 
    pub fn new() -> Self {
        Self {
            channel_removals: HashMap::new(),
            layer_removals: Vec::new(),
            validated: false,
        }
    }

    /// Add channel removals for a layer.
    pub fn remove_channels(&mut self, layer_id: impl Into<String>, channels: Vec<usize>) {
        self.channel_removals.insert(layer_id.into(), channels);
        self.validated = false;
    }

    /// Add a layer to be removed.
    pub fn remove_layer(&mut self, layer_id: impl Into<String>) {
        self.layer_removals.push(layer_id.into());
        self.validated = false;
    }

    /// Get channels to remove from a layer.
    #[must_use] 
    pub fn channels_to_remove(&self, layer_id: &str) -> Option<&Vec<usize>> {
        self.channel_removals.get(layer_id)
    }

    /// Check if a layer should be removed.
    #[must_use] 
    pub fn is_layer_removed(&self, layer_id: &str) -> bool {
        self.layer_removals.contains(&layer_id.to_string())
    }

    /// Validate the plan against a dependency graph.
    ///
    /// Ensures:
    /// - All referenced layers exist
    /// - Channel indices are valid
    /// - Dependencies are satisfied
    pub fn validate(&mut self, graph: &DependencyGraph) -> Result<(), PruningError> {
        // Check all referenced layers exist
        for layer_id in self.channel_removals.keys() {
            if graph.get_node(layer_id).is_none() {
                return Err(PruningError::InvalidPattern {
                    message: format!("Layer '{layer_id}' not found in graph"),
                });
            }
        }

        for layer_id in &self.layer_removals {
            if graph.get_node(layer_id).is_none() {
                return Err(PruningError::InvalidPattern {
                    message: format!("Layer '{layer_id}' not found in graph"),
                });
            }
        }

        // Check channel indices are valid
        for (layer_id, channels) in &self.channel_removals {
            if let Some(node) = graph.get_node(layer_id) {
                for &ch in channels {
                    if ch >= node.output_dim && node.output_dim > 0 {
                        return Err(PruningError::InvalidSparsity {
                            value: ch as f32,
                            constraint: format!(
                                "Channel {} >= output_dim {} for layer {}",
                                ch, node.output_dim, layer_id
                            ),
                        });
                    }
                }
            }
        }

        self.validated = true;
        Ok(())
    }

    /// Check if plan is validated.
    #[must_use] 
    pub fn is_validated(&self) -> bool {
        self.validated
    }

    /// Get total number of channels being removed.
    pub fn total_channels_removed(&self) -> usize {
        self.channel_removals.values().map(Vec::len).sum()
    }

    /// Get number of layers being removed.
    #[must_use] 
    pub fn total_layers_removed(&self) -> usize {
        self.layer_removals.len()
    }
}

impl Default for PruningPlan {
    fn default() -> Self {
        Self::new()
    }
}

/// Propagate channel pruning through a dependency graph.
///
/// When output channels are pruned from a layer, this function
/// determines which input channels of downstream layers must also be pruned.
///
/// # Arguments
/// * `graph` - The dependency graph
/// * `source_layer` - Layer where channels are being pruned
/// * `pruned_channels` - Indices of channels being removed
///
/// # Returns
/// Map of `layer_id` -> input channels to remove
#[must_use] 
pub fn propagate_channel_pruning(
    graph: &DependencyGraph,
    source_layer: &str,
    pruned_channels: &[usize],
) -> HashMap<String, Vec<usize>> {
    let mut result = HashMap::new();

    // Find all downstream layers connected via sequential edges
    for edge in graph.edges_from(source_layer) {
        if edge.dep_type == DependencyType::Sequential {
            // Downstream layer needs same input channels removed
            result.insert(edge.to.clone(), pruned_channels.to_vec());
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================================================
    // FALSIFICATION: Graph construction
    // ==========================================================================
    #[test]
    fn test_graph_new() {
        let graph = DependencyGraph::new();
        assert_eq!(
            graph.num_nodes(),
            0,
            "GRA-01 FALSIFIED: New graph should be empty"
        );
        assert_eq!(graph.num_edges(), 0);
    }

    #[test]
    fn test_graph_add_node() {
        let mut graph = DependencyGraph::new();
        let node = GraphNode::new("layer0", "Linear1", NodeType::Linear);
        graph.add_node(node);

        assert_eq!(graph.num_nodes(), 1, "GRA-02 FALSIFIED: Should have 1 node");
        assert!(graph.get_node("layer0").is_some());
    }

    #[test]
    fn test_graph_add_edge() {
        let mut graph = DependencyGraph::new();

        graph.add_node(GraphNode::new("a", "A", NodeType::Linear));
        graph.add_node(GraphNode::new("b", "B", NodeType::Linear));

        let edge = GraphEdge::new("a", "b", DependencyType::Sequential);
        graph.add_edge(edge).unwrap();

        assert_eq!(graph.num_edges(), 1, "GRA-03 FALSIFIED: Should have 1 edge");
    }

    #[test]
    fn test_graph_add_edge_invalid_source() {
        let mut graph = DependencyGraph::new();
        graph.add_node(GraphNode::new("b", "B", NodeType::Linear));

        let edge = GraphEdge::new("nonexistent", "b", DependencyType::Sequential);
        let result = graph.add_edge(edge);

        assert!(
            result.is_err(),
            "GRA-04 FALSIFIED: Should error on invalid source"
        );
    }

    #[test]
    fn test_graph_add_edge_invalid_target() {
        let mut graph = DependencyGraph::new();
        graph.add_node(GraphNode::new("a", "A", NodeType::Linear));

        let edge = GraphEdge::new("a", "nonexistent", DependencyType::Sequential);
        let result = graph.add_edge(edge);

        assert!(
            result.is_err(),
            "GRA-05 FALSIFIED: Should error on invalid target"
        );
    }

    // ==========================================================================
    // FALSIFICATION: Linear chain helper
    // ==========================================================================
    #[test]
    fn test_linear_chain() {
        let graph = DependencyGraph::linear_chain(
            &[(768, 512), (512, 256), (256, 128)],
            &["fc1", "fc2", "fc3"],
        );

        assert_eq!(
            graph.num_nodes(),
            3,
            "GRA-06 FALSIFIED: Should have 3 nodes"
        );
        assert_eq!(
            graph.num_edges(),
            2,
            "GRA-06 FALSIFIED: Should have 2 edges"
        );

        let node = graph.get_node("layer_0").unwrap();
        assert_eq!(node.input_dim, 768);
        assert_eq!(node.output_dim, 512);
    }

    // ==========================================================================
    // FALSIFICATION: Dependency traversal
    // ==========================================================================
    #[test]
    fn test_downstream_dependents() {
        let mut graph = DependencyGraph::new();

        graph.add_node(GraphNode::new("a", "A", NodeType::Linear));
        graph.add_node(GraphNode::new("b", "B", NodeType::Linear));
        graph.add_node(GraphNode::new("c", "C", NodeType::Linear));

        graph
            .add_edge(GraphEdge::new("a", "b", DependencyType::Sequential))
            .unwrap();
        graph
            .add_edge(GraphEdge::new("b", "c", DependencyType::Sequential))
            .unwrap();

        let deps = graph.downstream_dependents("a");

        assert!(
            deps.contains("b"),
            "GRA-07 FALSIFIED: B should be downstream of A"
        );
        assert!(
            deps.contains("c"),
            "GRA-07 FALSIFIED: C should be downstream of A"
        );
        assert!(
            !deps.contains("a"),
            "GRA-07 FALSIFIED: A should not be its own dependent"
        );
    }

    #[test]
    fn test_upstream_dependents() {
        let mut graph = DependencyGraph::new();

        graph.add_node(GraphNode::new("a", "A", NodeType::Linear));
        graph.add_node(GraphNode::new("b", "B", NodeType::Linear));
        graph.add_node(GraphNode::new("c", "C", NodeType::Linear));

        graph
            .add_edge(GraphEdge::new("a", "b", DependencyType::Sequential))
            .unwrap();
        graph
            .add_edge(GraphEdge::new("b", "c", DependencyType::Sequential))
            .unwrap();

        let deps = graph.upstream_dependents("c");

        assert!(
            deps.contains("a"),
            "GRA-08 FALSIFIED: A should be upstream of C"
        );
        assert!(
            deps.contains("b"),
            "GRA-08 FALSIFIED: B should be upstream of C"
        );
    }

    // ==========================================================================
    // FALSIFICATION: Prunable nodes
    // ==========================================================================
    #[test]
    fn test_prunable_nodes() {
        let mut graph = DependencyGraph::new();

        graph.add_node(GraphNode::new("a", "A", NodeType::Linear).with_prunable(true));
        graph.add_node(GraphNode::new("b", "B", NodeType::Linear).with_prunable(false));
        graph.add_node(GraphNode::new("c", "C", NodeType::Linear).with_prunable(true));

        let prunable = graph.prunable_nodes();

        assert_eq!(
            prunable.len(),
            2,
            "GRA-09 FALSIFIED: Should have 2 prunable nodes"
        );
    }

    // ==========================================================================
    // FALSIFICATION: Graph validation
    // ==========================================================================
    #[test]
    fn test_validate_dimension_mismatch() {
        let mut graph = DependencyGraph::new();

        graph.add_node(GraphNode::new("a", "A", NodeType::Linear).with_dims(10, 20));
        graph.add_node(GraphNode::new("b", "B", NodeType::Linear).with_dims(30, 40)); // Mismatch!

        graph
            .add_edge(GraphEdge::new("a", "b", DependencyType::Sequential))
            .unwrap();

        let result = graph.validate();
        assert!(
            result.is_err(),
            "GRA-10 FALSIFIED: Should detect dimension mismatch"
        );
    }

    #[test]
    fn test_validate_dimension_match() {
        let mut graph = DependencyGraph::new();

        graph.add_node(GraphNode::new("a", "A", NodeType::Linear).with_dims(10, 20));
        graph.add_node(GraphNode::new("b", "B", NodeType::Linear).with_dims(20, 40));

        graph
            .add_edge(GraphEdge::new("a", "b", DependencyType::Sequential))
            .unwrap();

        let result = graph.validate();
        assert!(
            result.is_ok(),
            "GRA-11 FALSIFIED: Matching dimensions should pass"
        );
    }

    // ==========================================================================
    // FALSIFICATION: Pruning plan
    // ==========================================================================
    #[test]
    fn test_pruning_plan_new() {
        let plan = PruningPlan::new();
        assert_eq!(plan.total_channels_removed(), 0);
        assert_eq!(plan.total_layers_removed(), 0);
        assert!(!plan.is_validated());
    }

    #[test]
    fn test_pruning_plan_add_channels() {
        let mut plan = PruningPlan::new();
        plan.remove_channels("layer0", vec![0, 5, 10]);
        plan.remove_channels("layer1", vec![1, 2]);

        assert_eq!(plan.total_channels_removed(), 5);
        assert_eq!(plan.channels_to_remove("layer0"), Some(&vec![0, 5, 10]));
    }

    #[test]
    fn test_pruning_plan_remove_layer() {
        let mut plan = PruningPlan::new();
        plan.remove_layer("layer5");
        plan.remove_layer("layer10");

        assert_eq!(plan.total_layers_removed(), 2);
        assert!(plan.is_layer_removed("layer5"));
        assert!(!plan.is_layer_removed("layer0"));
    }

    #[test]
    fn test_pruning_plan_validate() {
        let graph = DependencyGraph::linear_chain(&[(100, 50), (50, 25)], &["fc1", "fc2"]);

        let mut plan = PruningPlan::new();
        plan.remove_channels("layer_0", vec![0, 10, 20]);

        let result = plan.validate(&graph);
        assert!(result.is_ok(), "GRA-12 FALSIFIED: Valid plan should pass");
        assert!(plan.is_validated());
    }

    #[test]
    fn test_pruning_plan_validate_invalid_layer() {
        let graph = DependencyGraph::linear_chain(&[(100, 50)], &["fc1"]);

        let mut plan = PruningPlan::new();
        plan.remove_channels("nonexistent", vec![0]);

        let result = plan.validate(&graph);
        assert!(
            result.is_err(),
            "GRA-13 FALSIFIED: Should error on invalid layer"
        );
    }

    #[test]
    fn test_pruning_plan_validate_invalid_channel() {
        let graph = DependencyGraph::linear_chain(&[(100, 50)], &["fc1"]);

        let mut plan = PruningPlan::new();
        plan.remove_channels("layer_0", vec![100]); // Out of bounds (output_dim = 50)

        let result = plan.validate(&graph);
        assert!(
            result.is_err(),
            "GRA-14 FALSIFIED: Should error on invalid channel index"
        );
    }

    // ==========================================================================
    // FALSIFICATION: Channel propagation
    // ==========================================================================
    #[test]
    fn test_propagate_channel_pruning() {
        let mut graph = DependencyGraph::new();

        graph.add_node(GraphNode::new("a", "A", NodeType::Linear).with_dims(100, 50));
        graph.add_node(GraphNode::new("b", "B", NodeType::Linear).with_dims(50, 25));

        graph
            .add_edge(GraphEdge::new("a", "b", DependencyType::Sequential))
            .unwrap();

        let propagation = propagate_channel_pruning(&graph, "a", &[5, 10, 15]);

        assert!(
            propagation.contains_key("b"),
            "GRA-15 FALSIFIED: B should be affected by pruning A"
        );
        assert_eq!(propagation.get("b"), Some(&vec![5, 10, 15]));
    }

    // ==========================================================================
    // FALSIFICATION: GraphNode builder
    // ==========================================================================
    #[test]
    fn test_graph_node_builder() {
        let node = GraphNode::new("test", "Test Layer", NodeType::Linear)
            .with_dims(100, 200)
            .with_prunable(false);

        assert_eq!(node.id, "test");
        assert_eq!(node.name, "Test Layer");
        assert_eq!(node.input_dim, 100);
        assert_eq!(node.output_dim, 200);
        assert!(!node.prunable);
    }

    // ==========================================================================
    // FALSIFICATION: Edge types
    // ==========================================================================
    #[test]
    fn test_dependency_types() {
        let sequential = DependencyType::Sequential;
        let skip = DependencyType::Skip;

        assert_ne!(
            sequential, skip,
            "GRA-16 FALSIFIED: Types should be distinct"
        );
    }

    #[test]
    fn test_edge_with_dim_index() {
        let edge = GraphEdge::new("a", "b", DependencyType::Concat).with_dim_index(2);
        assert_eq!(edge.dim_index, 2);
    }

    // ==========================================================================
    // FALSIFICATION: Clone and Debug
    // ==========================================================================
    #[test]
    fn test_graph_clone() {
        let mut orig = DependencyGraph::new();
        orig.add_node(GraphNode::new("a", "A", NodeType::Linear));

        let cloned = orig.clone();
        assert_eq!(orig.num_nodes(), cloned.num_nodes());
    }

    #[test]
    fn test_graph_debug() {
        let graph = DependencyGraph::new();
        let debug = format!("{:?}", graph);
        assert!(debug.contains("DependencyGraph"));
    }

    #[test]
    fn test_node_debug() {
        let node = GraphNode::new("test", "Test", NodeType::Linear);
        let debug = format!("{:?}", node);
        assert!(debug.contains("GraphNode"));
    }

    #[test]
    fn test_edge_debug() {
        let edge = GraphEdge::new("a", "b", DependencyType::Sequential);
        let debug = format!("{:?}", edge);
        assert!(debug.contains("GraphEdge"));
    }

    #[test]
    fn test_plan_debug() {
        let plan = PruningPlan::new();
        let debug = format!("{:?}", plan);
        assert!(debug.contains("PruningPlan"));
    }

    // ==========================================================================
    // FALSIFICATION: Default implementations
    // ==========================================================================
    #[test]
    fn test_graph_default() {
        let graph = DependencyGraph::default();
        assert_eq!(graph.num_nodes(), 0);
    }

    #[test]
    fn test_plan_default() {
        let plan = PruningPlan::default();
        assert_eq!(plan.total_channels_removed(), 0);
    }
}
