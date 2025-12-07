//! Message Passing Neural Network for Code Graphs
//!
//! Implements MPNN layers specifically designed for code analysis:
//! - Type propagation through data flow edges
//! - Lifetime analysis through ownership edges
//! - Control flow analysis through CFG edges

use std::collections::HashMap;

/// Edge types in a code graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CodeEdgeType {
    /// Control flow edge (CFG)
    ControlFlow,
    /// Data dependency edge (def-use)
    DataFlow,
    /// AST child relationship
    AstChild,
    /// Type annotation edge
    TypeAnnotation,
    /// Ownership/borrow edge
    Ownership,
    /// Call edge (function invocation)
    Call,
    /// Return edge
    Return,
}

/// A node in the code graph
#[derive(Debug, Clone)]
pub struct CodeGraphNode {
    /// Node identifier
    pub id: usize,
    /// Node features (embedding)
    pub features: Vec<f64>,
    /// Node type (e.g., "variable", "function", "type")
    pub node_type: String,
}

impl CodeGraphNode {
    /// Create a new code graph node
    #[must_use]
    pub fn new(id: usize, features: Vec<f64>, node_type: impl Into<String>) -> Self {
        Self {
            id,
            features,
            node_type: node_type.into(),
        }
    }

    /// Get the feature dimension
    #[must_use]
    pub fn dim(&self) -> usize {
        self.features.len()
    }
}

/// An edge in the code graph
#[derive(Debug, Clone)]
pub struct CodeGraphEdge {
    /// Source node index
    pub source: usize,
    /// Target node index
    pub target: usize,
    /// Edge type
    pub edge_type: CodeEdgeType,
    /// Optional edge features
    pub features: Option<Vec<f64>>,
}

impl CodeGraphEdge {
    /// Create a new edge
    #[must_use]
    pub fn new(source: usize, target: usize, edge_type: CodeEdgeType) -> Self {
        Self {
            source,
            target,
            edge_type,
            features: None,
        }
    }

    /// Create edge with features
    #[must_use]
    pub fn with_features(mut self, features: Vec<f64>) -> Self {
        self.features = Some(features);
        self
    }
}

/// Code graph representation
#[derive(Debug, Clone)]
pub struct CodeGraph {
    /// Nodes in the graph
    nodes: Vec<CodeGraphNode>,
    /// Edges in the graph
    edges: Vec<CodeGraphEdge>,
    /// Adjacency list (node -> [(neighbor, edge_idx)])
    adj_list: Vec<Vec<(usize, usize)>>,
}

impl CodeGraph {
    /// Create a new empty code graph
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            adj_list: Vec::new(),
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: CodeGraphNode) -> usize {
        let id = self.nodes.len();
        self.nodes.push(node);
        self.adj_list.push(Vec::new());
        id
    }

    /// Add an edge to the graph
    pub fn add_edge(&mut self, edge: CodeGraphEdge) {
        let edge_idx = self.edges.len();
        let source = edge.source;
        let target = edge.target;
        self.edges.push(edge);

        // Add to adjacency list (undirected for message passing)
        if source < self.adj_list.len() {
            self.adj_list[source].push((target, edge_idx));
        }
        if target < self.adj_list.len() {
            self.adj_list[target].push((source, edge_idx));
        }
    }

    /// Get the number of nodes
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of edges
    #[must_use]
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Get a node by index
    #[must_use]
    pub fn node(&self, idx: usize) -> Option<&CodeGraphNode> {
        self.nodes.get(idx)
    }

    /// Get neighbors of a node
    #[must_use]
    pub fn neighbors(&self, node_idx: usize) -> &[(usize, usize)] {
        self.adj_list.get(node_idx).map_or(&[], Vec::as_slice)
    }

    /// Get all nodes
    #[must_use]
    pub fn nodes(&self) -> &[CodeGraphNode] {
        &self.nodes
    }

    /// Get all edges
    #[must_use]
    pub fn edges(&self) -> &[CodeGraphEdge] {
        &self.edges
    }
}

impl Default for CodeGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Message Passing Neural Network layer for code graphs
///
/// This layer performs message passing between nodes in a code graph,
/// with edge-type-aware aggregation.
#[derive(Debug, Clone)]
pub struct CodeMPNNLayer {
    /// Input feature dimension
    in_dim: usize,
    /// Output feature dimension
    out_dim: usize,
    /// Hidden dimension for message computation
    hidden_dim: usize,
    /// Edge type embeddings
    edge_type_weights: HashMap<CodeEdgeType, Vec<f64>>,
    /// Message weights
    message_weights: Vec<f64>,
    /// Update weights
    update_weights: Vec<f64>,
    /// Random seed for initialization
    seed: u64,
}

impl CodeMPNNLayer {
    /// Create a new MPNN layer
    #[must_use]
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let hidden_dim = (in_dim + out_dim) / 2;
        let seed = 42;

        // Initialize edge type weights
        let mut edge_type_weights = HashMap::new();
        for edge_type in [
            CodeEdgeType::ControlFlow,
            CodeEdgeType::DataFlow,
            CodeEdgeType::AstChild,
            CodeEdgeType::TypeAnnotation,
            CodeEdgeType::Ownership,
            CodeEdgeType::Call,
            CodeEdgeType::Return,
        ] {
            edge_type_weights.insert(edge_type, Self::init_weights(hidden_dim, seed));
        }

        Self {
            in_dim,
            out_dim,
            hidden_dim,
            edge_type_weights,
            message_weights: Self::init_weights(in_dim * hidden_dim, seed),
            update_weights: Self::init_weights(hidden_dim * out_dim, seed),
            seed,
        }
    }

    /// Set random seed
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        // Re-initialize weights with new seed
        self.message_weights = Self::init_weights(self.in_dim * self.hidden_dim, seed);
        self.update_weights = Self::init_weights(self.hidden_dim * self.out_dim, seed);
        self
    }

    /// Initialize weights using Xavier initialization
    fn init_weights(size: usize, seed: u64) -> Vec<f64> {
        let scale = (2.0 / size as f64).sqrt();
        let mut weights = Vec::with_capacity(size);
        let mut hash = seed;
        for _ in 0..size {
            hash = hash.wrapping_mul(0x5851_f42d_4c95_7f2d).wrapping_add(1);
            let val = ((hash >> 32) as f64) / f64::from(u32::MAX) * 2.0 - 1.0;
            weights.push(val * scale);
        }
        weights
    }

    /// Compute message from source to target
    fn compute_message(&self, source_features: &[f64], edge_type: CodeEdgeType) -> Vec<f64> {
        // Get edge type weights
        let edge_weights = self
            .edge_type_weights
            .get(&edge_type)
            .expect("Edge type not found");

        // Simple message: element-wise product of source features and edge weights
        let mut message = vec![0.0; self.hidden_dim];
        for i in 0..self.hidden_dim.min(source_features.len()) {
            message[i] = source_features[i] * edge_weights[i % edge_weights.len()];
        }
        message
    }

    /// Aggregate messages from neighbors
    fn aggregate_messages(&self, messages: &[Vec<f64>]) -> Vec<f64> {
        if messages.is_empty() {
            return vec![0.0; self.hidden_dim];
        }

        // Mean aggregation
        let mut aggregated = vec![0.0; self.hidden_dim];
        for msg in messages {
            for (i, &val) in msg.iter().enumerate() {
                if i < self.hidden_dim {
                    aggregated[i] += val;
                }
            }
        }
        let n = messages.len() as f64;
        for val in &mut aggregated {
            *val /= n;
        }
        aggregated
    }

    /// Update node features based on aggregated messages
    fn update_features(&self, node_features: &[f64], aggregated: &[f64]) -> Vec<f64> {
        // Combine node features with aggregated messages
        let mut updated = vec![0.0; self.out_dim];
        for i in 0..self.out_dim {
            // Simple combination: sum of weighted features
            let node_contrib = if i < node_features.len() {
                node_features[i]
            } else {
                0.0
            };
            let msg_contrib = if i < aggregated.len() {
                aggregated[i]
            } else {
                0.0
            };
            // ReLU activation
            updated[i] = (node_contrib + msg_contrib).max(0.0);
        }
        updated
    }

    /// Forward pass through the MPNN layer
    #[must_use]
    pub fn forward(&self, graph: &CodeGraph) -> Vec<Vec<f64>> {
        let n = graph.num_nodes();
        let mut output = Vec::with_capacity(n);

        for node_idx in 0..n {
            // Collect messages from all neighbors
            let mut messages = Vec::new();
            for &(neighbor_idx, edge_idx) in graph.neighbors(node_idx) {
                if let Some(neighbor) = graph.node(neighbor_idx) {
                    let edge = &graph.edges()[edge_idx];
                    let msg = self.compute_message(&neighbor.features, edge.edge_type);
                    messages.push(msg);
                }
            }

            // Aggregate messages
            let aggregated = self.aggregate_messages(&messages);

            // Update node features
            let node = graph.node(node_idx).expect("Node not found");
            let updated = self.update_features(&node.features, &aggregated);
            output.push(updated);
        }

        output
    }

    /// Get input dimension
    #[must_use]
    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    /// Get output dimension
    #[must_use]
    pub fn out_dim(&self) -> usize {
        self.out_dim
    }
}

/// Stack of MPNN layers for deep code analysis
#[derive(Debug)]
pub struct CodeMPNN {
    /// MPNN layers
    layers: Vec<CodeMPNNLayer>,
}

impl CodeMPNN {
    /// Create a new MPNN with the given layer dimensions
    #[must_use]
    pub fn new(dims: &[usize]) -> Self {
        assert!(dims.len() >= 2, "Need at least input and output dimensions");
        let mut layers = Vec::new();
        for i in 0..dims.len() - 1 {
            layers.push(CodeMPNNLayer::new(dims[i], dims[i + 1]));
        }
        Self { layers }
    }

    /// Forward pass through all layers
    #[must_use]
    pub fn forward(&self, graph: &CodeGraph) -> Vec<Vec<f64>> {
        if self.layers.is_empty() {
            return graph.nodes().iter().map(|n| n.features.clone()).collect();
        }

        // Create a mutable graph for layer-by-layer processing
        let mut current_features: Vec<Vec<f64>> =
            graph.nodes().iter().map(|n| n.features.clone()).collect();

        for layer in &self.layers {
            // Create temporary graph with current features
            let mut temp_graph = CodeGraph::new();
            for (i, features) in current_features.iter().enumerate() {
                let node_type = graph.node(i).map_or("unknown", |n| &n.node_type);
                temp_graph.add_node(CodeGraphNode::new(i, features.clone(), node_type));
            }
            for edge in graph.edges() {
                temp_graph.add_edge(edge.clone());
            }

            current_features = layer.forward(&temp_graph);
        }

        current_features
    }

    /// Get the number of layers
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get output dimension
    #[must_use]
    pub fn out_dim(&self) -> usize {
        self.layers.last().map_or(0, CodeMPNNLayer::out_dim)
    }
}

/// Global pooling operations for code graphs
pub mod pooling {
    /// Mean pooling over node embeddings
    #[must_use]
    pub fn mean_pool(embeddings: &[Vec<f64>]) -> Vec<f64> {
        if embeddings.is_empty() {
            return Vec::new();
        }

        let dim = embeddings[0].len();
        let mut result = vec![0.0; dim];
        for emb in embeddings {
            for (i, &val) in emb.iter().enumerate() {
                result[i] += val;
            }
        }
        let n = embeddings.len() as f64;
        for val in &mut result {
            *val /= n;
        }
        result
    }

    /// Max pooling over node embeddings
    #[must_use]
    pub fn max_pool(embeddings: &[Vec<f64>]) -> Vec<f64> {
        if embeddings.is_empty() {
            return Vec::new();
        }

        let dim = embeddings[0].len();
        let mut result = vec![f64::NEG_INFINITY; dim];
        for emb in embeddings {
            for (i, &val) in emb.iter().enumerate() {
                if val > result[i] {
                    result[i] = val;
                }
            }
        }
        result
    }

    /// Sum pooling over node embeddings
    #[must_use]
    pub fn sum_pool(embeddings: &[Vec<f64>]) -> Vec<f64> {
        if embeddings.is_empty() {
            return Vec::new();
        }

        let dim = embeddings[0].len();
        let mut result = vec![0.0; dim];
        for emb in embeddings {
            for (i, &val) in emb.iter().enumerate() {
                result[i] += val;
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> CodeGraph {
        let mut graph = CodeGraph::new();

        // Add nodes: x (variable), y (variable), add (function)
        graph.add_node(CodeGraphNode::new(0, vec![1.0, 0.0, 0.0], "variable"));
        graph.add_node(CodeGraphNode::new(1, vec![0.0, 1.0, 0.0], "variable"));
        graph.add_node(CodeGraphNode::new(2, vec![0.0, 0.0, 1.0], "function"));

        // Add edges: x -> add (data flow), y -> add (data flow)
        graph.add_edge(CodeGraphEdge::new(0, 2, CodeEdgeType::DataFlow));
        graph.add_edge(CodeGraphEdge::new(1, 2, CodeEdgeType::DataFlow));

        graph
    }

    #[test]
    fn test_code_graph_creation() {
        let graph = create_test_graph();
        assert_eq!(graph.num_nodes(), 3);
        assert_eq!(graph.num_edges(), 2);
    }

    #[test]
    fn test_code_graph_neighbors() {
        let graph = create_test_graph();

        // Node 0 (x) should have one neighbor (add)
        assert_eq!(graph.neighbors(0).len(), 1);
        assert_eq!(graph.neighbors(0)[0].0, 2); // Neighbor is node 2

        // Node 2 (add) should have two neighbors (x and y)
        assert_eq!(graph.neighbors(2).len(), 2);
    }

    #[test]
    fn test_mpnn_layer_creation() {
        let layer = CodeMPNNLayer::new(3, 4);
        assert_eq!(layer.in_dim(), 3);
        assert_eq!(layer.out_dim(), 4);
    }

    #[test]
    fn test_mpnn_layer_forward() {
        let graph = create_test_graph();
        let layer = CodeMPNNLayer::new(3, 4);

        let output = layer.forward(&graph);

        assert_eq!(output.len(), 3);
        for node_output in &output {
            assert_eq!(node_output.len(), 4);
        }
    }

    #[test]
    fn test_mpnn_stack() {
        let graph = create_test_graph();
        let mpnn = CodeMPNN::new(&[3, 8, 4]);

        let output = mpnn.forward(&graph);

        assert_eq!(output.len(), 3);
        assert_eq!(mpnn.num_layers(), 2);
        assert_eq!(mpnn.out_dim(), 4);
    }

    #[test]
    fn test_mean_pooling() {
        let embeddings = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        let pooled = pooling::mean_pool(&embeddings);
        assert_eq!(pooled.len(), 3);
        assert!((pooled[0] - 4.0).abs() < 1e-6);
        assert!((pooled[1] - 5.0).abs() < 1e-6);
        assert!((pooled[2] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_max_pooling() {
        let embeddings = vec![
            vec![1.0, 8.0, 3.0],
            vec![4.0, 5.0, 9.0],
            vec![7.0, 2.0, 6.0],
        ];

        let pooled = pooling::max_pool(&embeddings);
        assert_eq!(pooled.len(), 3);
        assert!((pooled[0] - 7.0).abs() < 1e-6);
        assert!((pooled[1] - 8.0).abs() < 1e-6);
        assert!((pooled[2] - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_sum_pooling() {
        let embeddings = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

        let pooled = pooling::sum_pool(&embeddings);
        assert_eq!(pooled.len(), 3);
        assert!((pooled[0] - 5.0).abs() < 1e-6);
        assert!((pooled[1] - 7.0).abs() < 1e-6);
        assert!((pooled[2] - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_empty_graph() {
        let graph = CodeGraph::new();
        let layer = CodeMPNNLayer::new(3, 4);

        let output = layer.forward(&graph);
        assert!(output.is_empty());
    }

    #[test]
    fn test_deterministic_weights() {
        let layer1 = CodeMPNNLayer::new(3, 4).with_seed(123);
        let layer2 = CodeMPNNLayer::new(3, 4).with_seed(123);

        // Same seed should produce same weights
        assert_eq!(layer1.message_weights.len(), layer2.message_weights.len());
        for (w1, w2) in layer1
            .message_weights
            .iter()
            .zip(layer2.message_weights.iter())
        {
            assert!((w1 - w2).abs() < 1e-10);
        }
    }

    #[test]
    fn test_edge_with_features() {
        let mut graph = CodeGraph::new();
        graph.add_node(CodeGraphNode::new(0, vec![1.0], "a"));
        graph.add_node(CodeGraphNode::new(1, vec![2.0], "b"));

        let edge = CodeGraphEdge::new(0, 1, CodeEdgeType::DataFlow).with_features(vec![0.5, 0.5]);
        graph.add_edge(edge);

        assert_eq!(graph.edges()[0].features, Some(vec![0.5, 0.5]));
    }

    #[test]
    fn test_code_graph_default() {
        let graph = CodeGraph::default();
        assert_eq!(graph.num_nodes(), 0);
        assert_eq!(graph.num_edges(), 0);
    }
}
