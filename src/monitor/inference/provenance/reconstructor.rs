//! Incident reconstructor for forensic analysis.

use super::attack::{Anomaly, AttackPath};
use super::graph::ProvenanceGraph;
use super::node::{NodeId, ProvenanceNode};

/// Incident reconstructor using provenance graph
pub struct IncidentReconstructor<'a> {
    graph: &'a ProvenanceGraph,
}

impl<'a> IncidentReconstructor<'a> {
    /// Create a new reconstructor
    pub fn new(graph: &'a ProvenanceGraph) -> Self {
        Self { graph }
    }

    /// Trace backwards from incident node to root causes
    pub fn reconstruct_path(&self, incident_node: NodeId, max_depth: usize) -> AttackPath {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();

        queue.push_back((incident_node, 0usize));
        visited.insert(incident_node);

        while let Some((node_id, depth)) = queue.pop_front() {
            if depth > max_depth {
                continue;
            }

            if let Some(node) = self.graph.get_node(node_id) {
                nodes.push((node_id, node.clone()));
            }

            for edge in self.graph.incoming_edges(node_id) {
                edges.push(edge.clone());

                if !visited.contains(&edge.from) {
                    visited.insert(edge.from);
                    queue.push_back((edge.from, depth + 1));
                }
            }
        }

        // Reverse to get causal order (root → incident)
        nodes.reverse();

        // Calculate duration
        let duration_ns = self.calculate_duration(&nodes);

        AttackPath { nodes, edges, duration_ns, anomaly_indices: Vec::new() }
    }

    /// Calculate duration from timestamps
    fn calculate_duration(&self, nodes: &[(NodeId, ProvenanceNode)]) -> u64 {
        let timestamps: Vec<u64> = nodes.iter().filter_map(|(_, n)| n.timestamp_ns()).collect();

        if timestamps.len() < 2 {
            return 0;
        }

        let min = *timestamps.iter().min().unwrap_or(&0);
        let max = *timestamps.iter().max().unwrap_or(&0);
        max - min
    }

    /// Identify anomalies in a path
    pub fn identify_anomalies(&self, path: &AttackPath, confidence_threshold: f32) -> Vec<Anomaly> {
        let mut anomalies = Vec::new();

        for (idx, (node_id, node)) in path.nodes.iter().enumerate() {
            // Check for low confidence inferences
            if let ProvenanceNode::Inference { confidence, .. } = node {
                if *confidence < confidence_threshold {
                    anomalies.push(Anomaly {
                        node_id: *node_id,
                        description: format!(
                            "Low confidence inference: {:.1}% (threshold: {:.1}%)",
                            confidence * 100.0,
                            confidence_threshold * 100.0
                        ),
                        severity: 1.0 - *confidence,
                    });
                }
            }

            // Check for suspicious fusion with many inputs
            if let ProvenanceNode::Fusion { input_refs, .. } = node {
                if input_refs.len() > 10 {
                    anomalies.push(Anomaly {
                        node_id: *node_id,
                        description: format!(
                            "Unusually many fusion inputs: {} (expected <10)",
                            input_refs.len()
                        ),
                        severity: 0.3,
                    });
                }
            }

            // Flag nodes with no predecessors (except inputs)
            if !matches!(node, ProvenanceNode::Input { .. }) {
                let preds = self.graph.predecessors(*node_id);
                if preds.is_empty() {
                    anomalies.push(Anomaly {
                        node_id: *node_id,
                        description: format!("{} node has no predecessors", node.type_name()),
                        severity: 0.5,
                    });
                }
            }

            let _ = idx; // Used in future enhancements
        }

        anomalies
    }
}
