//! Louvain community detection algorithm
//!
//! Community detection for identifying code modules based on graph structure.
//! Based on the Louvain method (Blondel et al., 2008) via aprender integration.
//!
//! # References
//! - Blondel et al. (2008): "Fast unfolding of communities in large networks"
//! - Girvan & Newman (2002): "Community structure in social and biological networks"

use crate::{CsrGraph, NodeId};
use anyhow::Result;
use aprender::graph::Graph as AprenderGraph;

/// Community detection result
#[derive(Debug, Clone)]
pub struct CommunityDetectionResult {
    /// Communities, each containing a list of node IDs
    pub communities: Vec<Vec<NodeId>>,

    /// Total number of communities found
    pub num_communities: usize,

    /// Modularity score (quality metric for community structure)
    pub modularity: f64,
}

impl CommunityDetectionResult {
    /// Get the community ID for a given node
    ///
    /// Returns None if node not found in any community
    #[must_use]
    pub fn get_community(&self, node: NodeId) -> Option<usize> {
        for (comm_id, community) in self.communities.iter().enumerate() {
            if community.contains(&node) {
                return Some(comm_id);
            }
        }
        None
    }

    /// Get all nodes in a specific community
    #[must_use]
    pub fn get_community_nodes(&self, comm_id: usize) -> Option<&[NodeId]> {
        self.communities.get(comm_id).map(Vec::as_slice)
    }

    /// Get size of a specific community
    #[must_use]
    pub fn community_size(&self, comm_id: usize) -> Option<usize> {
        self.communities.get(comm_id).map(Vec::len)
    }
}

/// Detect communities using the Louvain algorithm
///
/// The Louvain method is a greedy modularity optimization algorithm that:
/// 1. Starts with each node in its own community
/// 2. Iteratively moves nodes to neighboring communities to maximize modularity
/// 3. Aggregates communities into super-nodes and repeats
///
/// # Arguments
///
/// * `graph` - The graph to analyze
///
/// # Returns
///
/// `CommunityDetectionResult` containing communities and modularity score
///
/// # Errors
///
/// Returns error if graph conversion fails (typically won't happen)
///
/// # Example
///
/// ```ignore
/// # use trueno_graph::{CsrGraph, NodeId};
/// # use trueno_graph::louvain;
/// let mut graph = CsrGraph::new();
/// graph.add_edge(NodeId(0), NodeId(1), 1.0)?;
/// graph.add_edge(NodeId(1), NodeId(2), 1.0)?;
/// graph.add_edge(NodeId(3), NodeId(4), 1.0)?; // Separate component
///
/// let result = louvain(&graph)?;
/// println!("Found {} communities", result.num_communities);
/// println!("Modularity: {:.3}", result.modularity);
/// ```
pub fn louvain(graph: &CsrGraph) -> Result<CommunityDetectionResult> {
    // Convert trueno-graph CSR to aprender Graph
    let aprender_graph = convert_to_aprender(graph);

    // Run Louvain algorithm
    let communities = aprender_graph.louvain();

    // Calculate modularity
    let modularity = aprender_graph.modularity(&communities);

    // Convert aprender NodeId (usize) to trueno NodeId (u32)
    let converted_communities: Vec<Vec<NodeId>> = communities
        .into_iter()
        .map(|community| {
            community
                .into_iter()
                .filter_map(|node_id| u32::try_from(node_id).ok().map(NodeId))
                .collect()
        })
        .filter(|community: &Vec<NodeId>| !community.is_empty()) // Remove empty communities
        .collect();

    let num_communities = converted_communities.len();

    Ok(CommunityDetectionResult { communities: converted_communities, num_communities, modularity })
}

/// Convert `CsrGraph` to aprender `Graph` format
fn convert_to_aprender(graph: &CsrGraph) -> AprenderGraph {
    // Build edge list
    let mut edges = Vec::new();

    for (src, targets, weights) in graph.iter_adjacency() {
        for (dst, weight) in targets.iter().zip(weights.iter()) {
            edges.push((src.0 as usize, *dst as usize, f64::from(*weight)));
        }
    }

    // Create undirected graph from edge list
    AprenderGraph::from_weighted_edges(&edges, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_louvain_empty_graph() {
        let graph = CsrGraph::new();
        let result = louvain(&graph).unwrap();

        assert_eq!(result.num_communities, 0);
        assert_eq!(result.communities.len(), 0);
    }

    #[test]
    fn test_louvain_single_triangle() {
        // Single triangle - should form one community
        let mut graph = CsrGraph::new();
        graph.add_edge(NodeId(0), NodeId(1), 1.0).unwrap();
        graph.add_edge(NodeId(1), NodeId(2), 1.0).unwrap();
        graph.add_edge(NodeId(2), NodeId(0), 1.0).unwrap();

        let result = louvain(&graph).unwrap();

        assert_eq!(result.num_communities, 1, "Triangle should form 1 community");

        // All nodes should be in the same community
        let comm_0 = result.get_community(NodeId(0));
        let comm_1 = result.get_community(NodeId(1));
        let comm_2 = result.get_community(NodeId(2));

        assert_eq!(comm_0, comm_1);
        assert_eq!(comm_1, comm_2);
    }

    #[test]
    fn test_louvain_two_triangles_connected() {
        // Two triangles with a single connecting edge
        // Triangle 1: 0-1-2
        // Triangle 2: 3-4-5
        // Bridge: 2-3
        let mut graph = CsrGraph::new();

        // Triangle 1
        graph.add_edge(NodeId(0), NodeId(1), 1.0).unwrap();
        graph.add_edge(NodeId(1), NodeId(2), 1.0).unwrap();
        graph.add_edge(NodeId(2), NodeId(0), 1.0).unwrap();

        // Triangle 2
        graph.add_edge(NodeId(3), NodeId(4), 1.0).unwrap();
        graph.add_edge(NodeId(4), NodeId(5), 1.0).unwrap();
        graph.add_edge(NodeId(5), NodeId(3), 1.0).unwrap();

        // Bridge edge
        graph.add_edge(NodeId(2), NodeId(3), 1.0).unwrap();

        let result = louvain(&graph).unwrap();

        // Should find 2 communities (one per triangle)
        assert!(result.num_communities >= 1, "Should find at least 1 community");

        // Modularity should be positive (indicates good community structure)
        assert!(result.modularity > 0.0, "Modularity should be positive for community structure");
    }

    #[test]
    fn test_louvain_disconnected_components() {
        // Two completely disconnected triangles
        // Triangle 1: 0-1-2
        // Triangle 2: 3-4-5
        let mut graph = CsrGraph::new();

        // Triangle 1
        graph.add_edge(NodeId(0), NodeId(1), 1.0).unwrap();
        graph.add_edge(NodeId(1), NodeId(2), 1.0).unwrap();
        graph.add_edge(NodeId(2), NodeId(0), 1.0).unwrap();

        // Triangle 2
        graph.add_edge(NodeId(3), NodeId(4), 1.0).unwrap();
        graph.add_edge(NodeId(4), NodeId(5), 1.0).unwrap();
        graph.add_edge(NodeId(5), NodeId(3), 1.0).unwrap();

        let result = louvain(&graph).unwrap();

        // Should find 2 communities (one per disconnected component)
        assert_eq!(
            result.num_communities, 2,
            "Should find 2 communities for 2 disconnected components"
        );

        // Verify nodes 0,1,2 are in different community than 3,4,5
        let comm_0 = result.get_community(NodeId(0));
        let comm_3 = result.get_community(NodeId(3));

        assert!(comm_0.is_some() && comm_3.is_some());
        assert_ne!(comm_0, comm_3, "Disconnected components should be in different communities");
    }

    #[test]
    fn test_community_detection_result_api() {
        // Test CommunityDetectionResult helper methods
        let result = CommunityDetectionResult {
            communities: vec![vec![NodeId(0), NodeId(1), NodeId(2)], vec![NodeId(3), NodeId(4)]],
            num_communities: 2,
            modularity: 0.42,
        };

        // Test get_community
        assert_eq!(result.get_community(NodeId(0)), Some(0));
        assert_eq!(result.get_community(NodeId(3)), Some(1));
        assert_eq!(result.get_community(NodeId(99)), None);

        // Test get_community_nodes
        assert_eq!(
            result.get_community_nodes(0),
            Some(&[NodeId(0), NodeId(1), NodeId(2)] as &[NodeId])
        );
        assert_eq!(result.get_community_nodes(1), Some(&[NodeId(3), NodeId(4)] as &[NodeId]));
        assert_eq!(result.get_community_nodes(2), None);

        // Test community_size
        assert_eq!(result.community_size(0), Some(3));
        assert_eq!(result.community_size(1), Some(2));
        assert_eq!(result.community_size(2), None);
    }

    #[test]
    fn test_louvain_all_nodes_assigned() {
        // Verify every node gets assigned to exactly one community
        let mut graph = CsrGraph::new();

        // Create a simple graph: 0-1-2-3-4
        graph.add_edge(NodeId(0), NodeId(1), 1.0).unwrap();
        graph.add_edge(NodeId(1), NodeId(2), 1.0).unwrap();
        graph.add_edge(NodeId(2), NodeId(3), 1.0).unwrap();
        graph.add_edge(NodeId(3), NodeId(4), 1.0).unwrap();

        let result = louvain(&graph).unwrap();

        // Collect all assigned nodes
        let mut assigned_nodes = std::collections::HashSet::new();
        for community in &result.communities {
            for &node in community {
                assigned_nodes.insert(node);
            }
        }

        // All 5 nodes should be assigned
        assert_eq!(assigned_nodes.len(), 5);
        assert!(assigned_nodes.contains(&NodeId(0)));
        assert!(assigned_nodes.contains(&NodeId(1)));
        assert!(assigned_nodes.contains(&NodeId(2)));
        assert!(assigned_nodes.contains(&NodeId(3)));
        assert!(assigned_nodes.contains(&NodeId(4)));
    }
}
