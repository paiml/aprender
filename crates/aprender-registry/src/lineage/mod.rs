//! Model lineage tracking.
//!
//! Tracks how models are derived from other models through various operations.

use crate::model::ModelId;
use crate::recipe::RecipeId;
use serde::{Deserialize, Serialize};

/// Types of model lineage relationships.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ModelLineageEdge {
    /// Model was fine-tuned from parent.
    FineTuned {
        /// Parent model ID.
        parent: ModelId,
        /// Recipe used for fine-tuning.
        recipe: RecipeId,
    },
    /// Model was distilled from teacher.
    Distilled {
        /// Teacher model ID.
        teacher: ModelId,
        /// Distillation temperature.
        temperature: f32,
    },
    /// Model was merged from multiple sources.
    Merged {
        /// Source model IDs.
        sources: Vec<ModelId>,
        /// Merge weights.
        weights: Vec<f32>,
    },
    /// Model was quantized from source.
    Quantized {
        /// Source model ID.
        source: ModelId,
        /// Quantization type.
        quantization: QuantizationType,
    },
    /// Model was pruned from source.
    Pruned {
        /// Source model ID.
        source: ModelId,
        /// Target sparsity (0.0 to 1.0).
        sparsity: f32,
    },
}

/// Types of quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuantizationType {
    /// 8-bit integer quantization.
    Int8,
    /// 4-bit integer quantization.
    Int4,
    /// 16-bit floating point.
    Fp16,
    /// Brain floating point 16.
    Bf16,
    /// Dynamic quantization.
    Dynamic,
}

impl std::fmt::Display for QuantizationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Int8 => "int8",
            Self::Int4 => "int4",
            Self::Fp16 => "fp16",
            Self::Bf16 => "bf16",
            Self::Dynamic => "dynamic",
        };
        write!(f, "{s}")
    }
}

/// A node in the lineage graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageNode {
    /// Model ID.
    pub model_id: ModelId,
    /// Model name.
    pub model_name: String,
    /// Model version string.
    pub model_version: String,
}

/// A lineage graph showing model derivation history.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LineageGraph {
    /// Nodes in the graph.
    pub nodes: Vec<LineageNode>,
    /// Edges representing derivation relationships.
    pub edges: Vec<LineageEdgeRecord>,
}

/// A recorded edge in the lineage graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageEdgeRecord {
    /// Source node index.
    pub from_idx: usize,
    /// Target node index.
    pub to_idx: usize,
    /// Edge type and metadata.
    pub edge: ModelLineageEdge,
}

impl LineageGraph {
    /// Create an empty lineage graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, node: LineageNode) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        idx
    }

    /// Add an edge to the graph.
    pub fn add_edge(&mut self, from_idx: usize, to_idx: usize, edge: ModelLineageEdge) {
        self.edges.push(LineageEdgeRecord { from_idx, to_idx, edge });
    }

    /// Get the number of nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of edges.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Get ancestors of a node (models it was derived from).
    #[must_use]
    pub fn ancestors(&self, node_idx: usize) -> Vec<usize> {
        self.edges.iter().filter(|e| e.to_idx == node_idx).map(|e| e.from_idx).collect()
    }

    /// Get descendants of a node (models derived from it).
    #[must_use]
    pub fn descendants(&self, node_idx: usize) -> Vec<usize> {
        self.edges.iter().filter(|e| e.from_idx == node_idx).map(|e| e.to_idx).collect()
    }

    /// Find node index by model ID.
    #[must_use]
    pub fn find_node(&self, model_id: &ModelId) -> Option<usize> {
        self.nodes.iter().position(|n| &n.model_id == model_id)
    }

    /// Get all ancestors of a node (full transitive closure).
    ///
    /// Returns all nodes from which this model was derived, recursively.
    #[must_use]
    pub fn all_ancestors(&self, node_idx: usize) -> Vec<usize> {
        let mut visited = std::collections::HashSet::new();
        let mut result = Vec::new();
        self.collect_ancestors(node_idx, &mut visited, &mut result);
        result
    }

    fn collect_ancestors(
        &self,
        node_idx: usize,
        visited: &mut std::collections::HashSet<usize>,
        result: &mut Vec<usize>,
    ) {
        for parent_idx in self.ancestors(node_idx) {
            if visited.insert(parent_idx) {
                result.push(parent_idx);
                self.collect_ancestors(parent_idx, visited, result);
            }
        }
    }

    /// Get all descendants of a node (full transitive closure).
    ///
    /// Returns all nodes derived from this model, recursively.
    #[must_use]
    pub fn all_descendants(&self, node_idx: usize) -> Vec<usize> {
        let mut visited = std::collections::HashSet::new();
        let mut result = Vec::new();
        self.collect_descendants(node_idx, &mut visited, &mut result);
        result
    }

    fn collect_descendants(
        &self,
        node_idx: usize,
        visited: &mut std::collections::HashSet<usize>,
        result: &mut Vec<usize>,
    ) {
        for child_idx in self.descendants(node_idx) {
            if visited.insert(child_idx) {
                result.push(child_idx);
                self.collect_descendants(child_idx, visited, result);
            }
        }
    }

    /// Get root models (models with no parents).
    #[must_use]
    pub fn root_nodes(&self) -> Vec<usize> {
        (0..self.nodes.len()).filter(|&idx| self.ancestors(idx).is_empty()).collect()
    }

    /// Get leaf models (models with no children).
    #[must_use]
    pub fn leaf_nodes(&self) -> Vec<usize> {
        (0..self.nodes.len()).filter(|&idx| self.descendants(idx).is_empty()).collect()
    }

    /// Find path between two nodes (returns node indices from source to target).
    ///
    /// Uses BFS to find shortest path. Returns `None` if no path exists.
    #[must_use]
    pub fn path_between(&self, from_idx: usize, to_idx: usize) -> Option<Vec<usize>> {
        use std::collections::{HashMap, VecDeque};

        if from_idx == to_idx {
            return Some(vec![from_idx]);
        }

        let mut queue = VecDeque::new();
        let mut parent_map: HashMap<usize, usize> = HashMap::new();

        queue.push_back(from_idx);

        while let Some(current) = queue.pop_front() {
            for child_idx in self.descendants(current) {
                if !parent_map.contains_key(&child_idx) {
                    parent_map.insert(child_idx, current);
                    if child_idx == to_idx {
                        // Reconstruct path
                        let mut path = vec![to_idx];
                        let mut node = to_idx;
                        while let Some(&parent) = parent_map.get(&node) {
                            path.push(parent);
                            node = parent;
                        }
                        path.reverse();
                        return Some(path);
                    }
                    queue.push_back(child_idx);
                }
            }
        }

        None
    }

    /// Perform topological sort of the graph.
    ///
    /// Returns nodes in order such that parents come before children.
    /// Returns `None` if the graph has a cycle.
    #[must_use]
    pub fn topological_sort(&self) -> Option<Vec<usize>> {
        use std::collections::HashMap;

        let n = self.nodes.len();
        if n == 0 {
            return Some(Vec::new());
        }

        // Calculate in-degree for each node
        let mut in_degree: HashMap<usize, usize> = (0..n).map(|i| (i, 0)).collect();
        for edge in &self.edges {
            *in_degree.entry(edge.to_idx).or_insert(0) += 1;
        }

        // Start with nodes that have no incoming edges
        let mut queue: Vec<usize> = in_degree
            .iter()
            .filter_map(|(&node, &degree)| if degree == 0 { Some(node) } else { None })
            .collect();

        let mut result = Vec::with_capacity(n);

        while let Some(node) = queue.pop() {
            result.push(node);

            for child in self.descendants(node) {
                if let Some(degree) = in_degree.get_mut(&child) {
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push(child);
                    }
                }
            }
        }

        // If we didn't visit all nodes, there's a cycle
        if result.len() == n {
            Some(result)
        } else {
            None
        }
    }

    /// Get depth of a node (longest path from any root).
    #[must_use]
    pub fn depth(&self, node_idx: usize) -> usize {
        let ancestors = self.ancestors(node_idx);
        if ancestors.is_empty() {
            0
        } else {
            ancestors.iter().map(|&a| self.depth(a) + 1).max().unwrap_or(0)
        }
    }

    /// Get the edges connecting two specific nodes.
    #[must_use]
    pub fn edges_between(&self, from_idx: usize, to_idx: usize) -> Vec<&LineageEdgeRecord> {
        self.edges.iter().filter(|e| e.from_idx == from_idx && e.to_idx == to_idx).collect()
    }

    /// Check if the graph is a DAG (directed acyclic graph).
    #[must_use]
    pub fn is_dag(&self) -> bool {
        self.topological_sort().is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_type_display() {
        assert_eq!(QuantizationType::Int8.to_string(), "int8");
        assert_eq!(QuantizationType::Fp16.to_string(), "fp16");
    }

    #[test]
    fn test_lineage_graph_basic() {
        let mut graph = LineageGraph::new();

        let base_id = ModelId::new();
        let finetuned_id = ModelId::new();

        let base_idx = graph.add_node(LineageNode {
            model_id: base_id.clone(),
            model_name: "base-model".to_string(),
            model_version: "1.0.0".to_string(),
        });

        let finetuned_idx = graph.add_node(LineageNode {
            model_id: finetuned_id.clone(),
            model_name: "finetuned-model".to_string(),
            model_version: "1.0.0".to_string(),
        });

        graph.add_edge(
            base_idx,
            finetuned_idx,
            ModelLineageEdge::FineTuned { parent: base_id.clone(), recipe: RecipeId::new() },
        );

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);
        assert_eq!(graph.ancestors(finetuned_idx), vec![base_idx]);
        assert_eq!(graph.descendants(base_idx), vec![finetuned_idx]);
    }

    #[test]
    fn test_lineage_graph_find_node() {
        let mut graph = LineageGraph::new();
        let model_id = ModelId::new();

        graph.add_node(LineageNode {
            model_id: model_id.clone(),
            model_name: "test-model".to_string(),
            model_version: "1.0.0".to_string(),
        });

        assert_eq!(graph.find_node(&model_id), Some(0));
        assert_eq!(graph.find_node(&ModelId::new()), None);
    }

    #[test]
    fn test_lineage_edge_serialization() {
        let edge = ModelLineageEdge::Quantized {
            source: ModelId::new(),
            quantization: QuantizationType::Int8,
        };

        let json = serde_json::to_string(&edge).unwrap();
        assert!(json.contains("quantized"));
        assert!(json.contains("int8"));

        let deserialized: ModelLineageEdge = serde_json::from_str(&json).unwrap();
        if let ModelLineageEdge::Quantized { quantization, .. } = deserialized {
            assert_eq!(quantization, QuantizationType::Int8);
        } else {
            panic!("Wrong variant");
        }
    }

    #[test]
    fn test_merged_lineage() {
        let sources = vec![ModelId::new(), ModelId::new(), ModelId::new()];
        let weights = vec![0.5, 0.3, 0.2];

        let edge = ModelLineageEdge::Merged { sources: sources.clone(), weights: weights.clone() };

        let json = serde_json::to_string(&edge).unwrap();
        let deserialized: ModelLineageEdge = serde_json::from_str(&json).unwrap();

        if let ModelLineageEdge::Merged { sources: s, weights: w } = deserialized {
            assert_eq!(s.len(), 3);
            assert_eq!(w.len(), 3);
        } else {
            panic!("Wrong variant");
        }
    }

    // -------------------------------------------------------------------------
    // Full Traversal Tests
    // -------------------------------------------------------------------------

    fn build_chain_graph() -> (LineageGraph, Vec<ModelId>) {
        // Creates: A -> B -> C -> D
        let mut graph = LineageGraph::new();
        let ids: Vec<ModelId> = (0..4).map(|_| ModelId::new()).collect();

        for (i, id) in ids.iter().enumerate() {
            graph.add_node(LineageNode {
                model_id: id.clone(),
                model_name: format!("model-{i}"),
                model_version: "1.0.0".to_string(),
            });
        }

        for (i, id) in ids.iter().enumerate().take(3) {
            graph.add_edge(
                i,
                i + 1,
                ModelLineageEdge::FineTuned { parent: id.clone(), recipe: RecipeId::new() },
            );
        }

        (graph, ids)
    }

    fn build_diamond_graph() -> (LineageGraph, Vec<ModelId>) {
        // Creates:
        //     A
        //    / \
        //   B   C
        //    \ /
        //     D
        let mut graph = LineageGraph::new();
        let ids: Vec<ModelId> = (0..4).map(|_| ModelId::new()).collect();

        let names = ["A", "B", "C", "D"];
        for (i, (id, name)) in ids.iter().zip(names.iter()).enumerate() {
            graph.add_node(LineageNode {
                model_id: id.clone(),
                model_name: (*name).to_string(),
                model_version: format!("1.{i}.0"),
            });
        }

        // A -> B
        graph.add_edge(
            0,
            1,
            ModelLineageEdge::FineTuned { parent: ids[0].clone(), recipe: RecipeId::new() },
        );
        // A -> C
        graph.add_edge(
            0,
            2,
            ModelLineageEdge::Quantized {
                source: ids[0].clone(),
                quantization: QuantizationType::Int8,
            },
        );
        // B -> D
        graph.add_edge(
            1,
            3,
            ModelLineageEdge::FineTuned { parent: ids[1].clone(), recipe: RecipeId::new() },
        );
        // C -> D
        graph.add_edge(
            2,
            3,
            ModelLineageEdge::Merged {
                sources: vec![ids[1].clone(), ids[2].clone()],
                weights: vec![0.5, 0.5],
            },
        );

        (graph, ids)
    }

    #[test]
    fn test_all_ancestors_chain() {
        let (graph, _) = build_chain_graph();

        // D (idx 3) should have ancestors A, B, C
        let ancestors = graph.all_ancestors(3);
        assert_eq!(ancestors.len(), 3);
        assert!(ancestors.contains(&0));
        assert!(ancestors.contains(&1));
        assert!(ancestors.contains(&2));

        // A (idx 0) should have no ancestors
        assert!(graph.all_ancestors(0).is_empty());

        // B (idx 1) should have only A
        let ancestors = graph.all_ancestors(1);
        assert_eq!(ancestors.len(), 1);
        assert!(ancestors.contains(&0));
    }

    #[test]
    fn test_all_descendants_chain() {
        let (graph, _) = build_chain_graph();

        // A (idx 0) should have descendants B, C, D
        let descendants = graph.all_descendants(0);
        assert_eq!(descendants.len(), 3);
        assert!(descendants.contains(&1));
        assert!(descendants.contains(&2));
        assert!(descendants.contains(&3));

        // D (idx 3) should have no descendants
        assert!(graph.all_descendants(3).is_empty());
    }

    #[test]
    fn test_all_ancestors_diamond() {
        let (graph, _) = build_diamond_graph();

        // D has ancestors A, B, C
        let ancestors = graph.all_ancestors(3);
        assert_eq!(ancestors.len(), 3);
        assert!(ancestors.contains(&0));
        assert!(ancestors.contains(&1));
        assert!(ancestors.contains(&2));
    }

    #[test]
    fn test_root_nodes() {
        let (chain, _) = build_chain_graph();
        assert_eq!(chain.root_nodes(), vec![0]);

        let (diamond, _) = build_diamond_graph();
        assert_eq!(diamond.root_nodes(), vec![0]);
    }

    #[test]
    fn test_leaf_nodes() {
        let (chain, _) = build_chain_graph();
        assert_eq!(chain.leaf_nodes(), vec![3]);

        let (diamond, _) = build_diamond_graph();
        assert_eq!(diamond.leaf_nodes(), vec![3]);
    }

    #[test]
    fn test_path_between() {
        let (graph, _) = build_chain_graph();

        // Path from A to D
        let path = graph.path_between(0, 3).unwrap();
        assert_eq!(path, vec![0, 1, 2, 3]);

        // Path from B to D
        let path = graph.path_between(1, 3).unwrap();
        assert_eq!(path, vec![1, 2, 3]);

        // Same node
        let path = graph.path_between(2, 2).unwrap();
        assert_eq!(path, vec![2]);

        // No path (wrong direction)
        assert!(graph.path_between(3, 0).is_none());
    }

    #[test]
    fn test_path_between_diamond() {
        let (graph, _) = build_diamond_graph();

        // Path from A to D (could go through B or C)
        let path = graph.path_between(0, 3).unwrap();
        assert!(path.len() == 3); // A -> B/C -> D
        assert_eq!(path[0], 0);
        assert_eq!(*path.last().unwrap(), 3);
    }

    #[test]
    fn test_topological_sort() {
        let (graph, _) = build_chain_graph();
        let sorted = graph.topological_sort().unwrap();

        // A should come before B, B before C, C before D
        let pos_a = sorted.iter().position(|&x| x == 0).unwrap();
        let pos_b = sorted.iter().position(|&x| x == 1).unwrap();
        let pos_c = sorted.iter().position(|&x| x == 2).unwrap();
        let pos_d = sorted.iter().position(|&x| x == 3).unwrap();

        assert!(pos_a < pos_b);
        assert!(pos_b < pos_c);
        assert!(pos_c < pos_d);
    }

    #[test]
    fn test_topological_sort_diamond() {
        let (graph, _) = build_diamond_graph();
        let sorted = graph.topological_sort().unwrap();

        let pos_a = sorted.iter().position(|&x| x == 0).unwrap();
        let pos_b = sorted.iter().position(|&x| x == 1).unwrap();
        let pos_c = sorted.iter().position(|&x| x == 2).unwrap();
        let pos_d = sorted.iter().position(|&x| x == 3).unwrap();

        // A should come before B and C
        assert!(pos_a < pos_b);
        assert!(pos_a < pos_c);
        // B and C should come before D
        assert!(pos_b < pos_d);
        assert!(pos_c < pos_d);
    }

    #[test]
    fn test_topological_sort_empty() {
        let graph = LineageGraph::new();
        assert_eq!(graph.topological_sort(), Some(vec![]));
    }

    #[test]
    fn test_depth() {
        let (graph, _) = build_chain_graph();

        assert_eq!(graph.depth(0), 0); // A is root
        assert_eq!(graph.depth(1), 1); // B
        assert_eq!(graph.depth(2), 2); // C
        assert_eq!(graph.depth(3), 3); // D
    }

    #[test]
    fn test_depth_diamond() {
        let (graph, _) = build_diamond_graph();

        assert_eq!(graph.depth(0), 0); // A is root
        assert_eq!(graph.depth(1), 1); // B
        assert_eq!(graph.depth(2), 1); // C
        assert_eq!(graph.depth(3), 2); // D (longest path is A->B->D or A->C->D)
    }

    #[test]
    fn test_edges_between() {
        let (graph, ids) = build_diamond_graph();

        // A -> B has one edge
        let edges = graph.edges_between(0, 1);
        assert_eq!(edges.len(), 1);
        assert!(matches!(edges[0].edge, ModelLineageEdge::FineTuned { .. }));

        // A -> C has one edge
        let edges = graph.edges_between(0, 2);
        assert_eq!(edges.len(), 1);
        assert!(matches!(edges[0].edge, ModelLineageEdge::Quantized { .. }));

        // No edge between B and C
        assert!(graph.edges_between(1, 2).is_empty());

        // D has edges from both B and C
        assert_eq!(graph.edges_between(1, 3).len(), 1);
        assert_eq!(graph.edges_between(2, 3).len(), 1);

        let _ = ids; // suppress unused warning
    }

    #[test]
    fn test_is_dag() {
        let (graph, _) = build_chain_graph();
        assert!(graph.is_dag());

        let (graph, _) = build_diamond_graph();
        assert!(graph.is_dag());

        // Empty graph is a DAG
        let empty = LineageGraph::new();
        assert!(empty.is_dag());
    }

    #[test]
    fn test_lineage_edge_pruned() {
        let edge = ModelLineageEdge::Pruned { source: ModelId::new(), sparsity: 0.5 };

        let json = serde_json::to_string(&edge).unwrap();
        assert!(json.contains("pruned"));
        assert!(json.contains("0.5"));

        let deserialized: ModelLineageEdge = serde_json::from_str(&json).unwrap();
        if let ModelLineageEdge::Pruned { sparsity, .. } = deserialized {
            assert!((sparsity - 0.5).abs() < f32::EPSILON);
        } else {
            panic!("Wrong variant");
        }
    }

    #[test]
    fn test_lineage_edge_distilled() {
        let edge = ModelLineageEdge::Distilled { teacher: ModelId::new(), temperature: 2.0 };

        let json = serde_json::to_string(&edge).unwrap();
        assert!(json.contains("distilled"));
        assert!(json.contains("2.0") || json.contains("2"));

        let deserialized: ModelLineageEdge = serde_json::from_str(&json).unwrap();
        if let ModelLineageEdge::Distilled { temperature, .. } = deserialized {
            assert!((temperature - 2.0).abs() < f32::EPSILON);
        } else {
            panic!("Wrong variant");
        }
    }

    #[test]
    fn test_all_quantization_types() {
        let types = [
            QuantizationType::Int8,
            QuantizationType::Int4,
            QuantizationType::Fp16,
            QuantizationType::Bf16,
            QuantizationType::Dynamic,
        ];

        for qt in types {
            let edge = ModelLineageEdge::Quantized { source: ModelId::new(), quantization: qt };

            let json = serde_json::to_string(&edge).unwrap();
            let _: ModelLineageEdge = serde_json::from_str(&json).unwrap();
        }
    }
}
