//! Provenance graph for causal analysis.

use super::edge::ProvenanceEdge;
use super::node::{NodeId, ProvenanceNode};
use crate::monitor::inference::path::DecisionPath;
use crate::monitor::inference::trace::DecisionTrace;
use std::collections::HashMap;

/// Provenance graph for causal analysis
///
/// # Features
/// - DAG structure for causal relationships
/// - Backward traversal for root cause analysis
/// - Anomaly detection in decision chains
pub struct ProvenanceGraph {
    /// Nodes indexed by ID
    nodes: HashMap<NodeId, ProvenanceNode>,
    /// Edges
    edges: Vec<ProvenanceEdge>,
    /// Adjacency list (forward): node -> outgoing edges
    forward: HashMap<NodeId, Vec<usize>>,
    /// Adjacency list (backward): node -> incoming edges
    backward: HashMap<NodeId, Vec<usize>>,
    /// Next node ID
    next_id: NodeId,
}

impl ProvenanceGraph {
    /// Create a new empty graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            forward: HashMap::new(),
            backward: HashMap::new(),
            next_id: 0,
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: ProvenanceNode) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.insert(id, node);
        id
    }

    /// Add an edge to the graph
    pub fn add_edge(&mut self, edge: ProvenanceEdge) {
        let edge_idx = self.edges.len();
        self.forward.entry(edge.from).or_default().push(edge_idx);
        self.backward.entry(edge.to).or_default().push(edge_idx);
        self.edges.push(edge);
    }

    /// Get a node by ID
    pub fn get_node(&self, id: NodeId) -> Option<&ProvenanceNode> {
        self.nodes.get(&id)
    }

    /// Get all nodes
    pub fn nodes(&self) -> &HashMap<NodeId, ProvenanceNode> {
        &self.nodes
    }

    /// Get all edges
    pub fn edges(&self) -> &[ProvenanceEdge] {
        &self.edges
    }

    /// Get incoming edges for a node
    pub fn incoming_edges(&self, id: NodeId) -> Vec<&ProvenanceEdge> {
        self.backward
            .get(&id)
            .map(|indices| indices.iter().map(|&i| &self.edges[i]).collect())
            .unwrap_or_default()
    }

    /// Get outgoing edges for a node
    pub fn outgoing_edges(&self, id: NodeId) -> Vec<&ProvenanceEdge> {
        self.forward
            .get(&id)
            .map(|indices| indices.iter().map(|&i| &self.edges[i]).collect())
            .unwrap_or_default()
    }

    /// Get predecessor nodes
    pub fn predecessors(&self, id: NodeId) -> Vec<NodeId> {
        self.incoming_edges(id).into_iter().map(|e| e.from).collect()
    }

    /// Get successor nodes
    pub fn successors(&self, id: NodeId) -> Vec<NodeId> {
        self.outgoing_edges(id).into_iter().map(|e| e.to).collect()
    }

    /// Number of nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Add an inference node from a decision trace
    pub fn add_inference<P: DecisionPath>(
        &mut self,
        trace: &DecisionTrace<P>,
        model_id: &str,
        model_version: &str,
    ) -> NodeId {
        self.add_node(ProvenanceNode::Inference {
            model_id: model_id.to_string(),
            model_version: model_version.to_string(),
            confidence: trace.confidence(),
            output: trace.output,
        })
    }
}

impl Default for ProvenanceGraph {
    fn default() -> Self {
        Self::new()
    }
}
