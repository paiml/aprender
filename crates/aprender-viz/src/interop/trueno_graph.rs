//! trueno-graph library integration.
//!
//! Provides visualization extensions for trueno-graph types (CsrGraph, NodeId).
//!
//! # Examples
//!
//! ```rust,ignore
//! use trueno_graph::{CsrGraph, NodeId};
//! use trueno_viz::interop::trueno_graph::GraphViz;
//!
//! let mut graph = CsrGraph::new();
//! graph.add_edge(NodeId(0), NodeId(1), 1.0)?;
//! graph.add_edge(NodeId(1), NodeId(2), 1.0)?;
//!
//! // Visualize as force-directed graph
//! let fb = graph.to_force_graph()?;
//! ```

use batuta_common::display::WithDimensions;
use trueno_graph::{louvain, pagerank, CommunityDetectionResult, CsrGraph, NodeId};

use crate::color::Rgba;
use crate::error::Result;
use crate::framebuffer::Framebuffer;
use crate::plots::{ForceGraph, GraphEdge, GraphNode, Histogram, ScatterPlot};

/// Default community colors (distinct, colorblind-friendly palette).
const COMMUNITY_COLORS: &[Rgba] = &[
    Rgba::new(66, 133, 244, 255),  // Blue
    Rgba::new(234, 67, 53, 255),   // Red
    Rgba::new(52, 168, 83, 255),   // Green
    Rgba::new(251, 188, 5, 255),   // Yellow
    Rgba::new(154, 160, 166, 255), // Gray
    Rgba::new(171, 71, 188, 255),  // Purple
    Rgba::new(255, 112, 67, 255),  // Orange
    Rgba::new(0, 172, 193, 255),   // Cyan
];

// ============================================================================
// CsrGraph Visualization Extensions
// ============================================================================

/// Visualization extensions for trueno-graph CsrGraph.
pub trait GraphViz {
    /// Create a force-directed graph visualization.
    fn to_force_graph(&self) -> Result<Framebuffer>;

    /// Create a force-directed graph with custom dimensions.
    fn to_force_graph_with(&self, width: u32, height: u32) -> Result<Framebuffer>;

    /// Create a force-directed graph with nodes colored by community.
    fn to_community_graph(&self) -> Result<Framebuffer>;

    /// Create a force-directed graph with nodes sized by PageRank.
    fn to_pagerank_graph(&self) -> Result<Framebuffer>;

    /// Create a force-directed graph with both community colors and PageRank sizing.
    fn to_analysis_graph(&self) -> Result<Framebuffer>;

    /// Create a histogram of node degrees (outgoing).
    fn degree_histogram(&self) -> Result<Framebuffer>;

    /// Create a scatter plot of in-degree vs out-degree.
    fn degree_scatter(&self) -> Result<Framebuffer>;
}

impl GraphViz for CsrGraph {
    fn to_force_graph(&self) -> Result<Framebuffer> {
        self.to_force_graph_with(600, 500)
    }

    fn to_force_graph_with(&self, width: u32, height: u32) -> Result<Framebuffer> {
        let mut fg = ForceGraph::new().dimensions(width, height).iterations(100);

        // Add nodes
        for i in 0..self.num_nodes() {
            let mut node = GraphNode::new(i);
            if let Some(name) = self.get_node_name(NodeId(i as u32)) {
                node = node.label(name);
            }
            fg = fg.add_node(node);
        }

        // Add edges
        for (src, targets, weights) in self.iter_adjacency() {
            for (dst, weight) in targets.iter().zip(weights.iter()) {
                fg = fg.add_edge(GraphEdge::new(src.0 as usize, *dst as usize).weight(*weight));
            }
        }

        let built = fg.build()?;
        built.to_framebuffer()
    }

    fn to_community_graph(&self) -> Result<Framebuffer> {
        let communities = louvain(self)
            .map_err(|e| crate::error::Error::Rendering(format!("Louvain failed: {e}")))?;

        graph_with_communities(self, &communities, 600, 500)
    }

    fn to_pagerank_graph(&self) -> Result<Framebuffer> {
        let scores = pagerank(self, 20, 1e-6)
            .map_err(|e| crate::error::Error::Rendering(format!("PageRank failed: {e}")))?;

        graph_with_pagerank(self, &scores, 600, 500)
    }

    fn to_analysis_graph(&self) -> Result<Framebuffer> {
        let communities = louvain(self)
            .map_err(|e| crate::error::Error::Rendering(format!("Louvain failed: {e}")))?;

        let scores = pagerank(self, 20, 1e-6)
            .map_err(|e| crate::error::Error::Rendering(format!("PageRank failed: {e}")))?;

        graph_with_analysis(self, &communities, &scores, 600, 500)
    }

    fn degree_histogram(&self) -> Result<Framebuffer> {
        let degrees: Vec<f32> = (0..self.num_nodes())
            .map(|i| {
                self.outgoing_neighbors(NodeId(i as u32)).map(|n| n.len() as f32).unwrap_or(0.0)
            })
            .collect();

        let plot = Histogram::new()
            .data(&degrees)
            .color(Rgba::new(66, 133, 244, 255))
            .dimensions(600, 400)
            .build()?;

        plot.to_framebuffer()
    }

    fn degree_scatter(&self) -> Result<Framebuffer> {
        let n = self.num_nodes();
        let mut in_degrees = vec![0.0f32; n];
        let mut out_degrees = vec![0.0f32; n];

        for i in 0..n {
            out_degrees[i] =
                self.outgoing_neighbors(NodeId(i as u32)).map(|n| n.len() as f32).unwrap_or(0.0);

            in_degrees[i] =
                self.incoming_neighbors(NodeId(i as u32)).map(|n| n.len() as f32).unwrap_or(0.0);
        }

        let plot = ScatterPlot::new()
            .x(&in_degrees)
            .y(&out_degrees)
            .color(Rgba::new(66, 133, 244, 255))
            .size(6.0)
            .dimensions(600, 500)
            .build()?;

        plot.to_framebuffer()
    }
}

// ============================================================================
// Internal Helper Functions
// ============================================================================

fn graph_with_communities(
    graph: &CsrGraph,
    communities: &CommunityDetectionResult,
    width: u32,
    height: u32,
) -> Result<Framebuffer> {
    let mut fg = ForceGraph::new().dimensions(width, height).iterations(100);

    // Add nodes with community colors
    for i in 0..graph.num_nodes() {
        let node_id = NodeId(i as u32);
        let comm_id = communities.get_community(node_id).unwrap_or(0);
        let color = COMMUNITY_COLORS[comm_id % COMMUNITY_COLORS.len()];

        let mut node = GraphNode::new(i).color(color);
        if let Some(name) = graph.get_node_name(node_id) {
            node = node.label(name);
        }
        fg = fg.add_node(node);
    }

    // Add edges
    for (src, targets, weights) in graph.iter_adjacency() {
        for (dst, weight) in targets.iter().zip(weights.iter()) {
            fg = fg.add_edge(GraphEdge::new(src.0 as usize, *dst as usize).weight(*weight));
        }
    }

    let built = fg.build()?;
    built.to_framebuffer()
}

fn graph_with_pagerank(
    graph: &CsrGraph,
    scores: &[f32],
    width: u32,
    height: u32,
) -> Result<Framebuffer> {
    let mut fg = ForceGraph::new().dimensions(width, height).iterations(100);

    // Find min/max for normalization
    let max_score = scores.iter().copied().fold(0.0f32, f32::max);
    let min_score = scores.iter().copied().fold(f32::MAX, f32::min);
    let score_range = (max_score - min_score).max(0.001);

    // Add nodes with PageRank-based sizing
    for i in 0..graph.num_nodes() {
        let score = scores.get(i).copied().unwrap_or(0.0);
        let normalized = (score - min_score) / score_range;
        let radius = 5.0 + normalized * 20.0; // 5-25 pixel radius

        let mut node = GraphNode::new(i).radius(radius);
        if let Some(name) = graph.get_node_name(NodeId(i as u32)) {
            node = node.label(name);
        }
        fg = fg.add_node(node);
    }

    // Add edges
    for (src, targets, weights) in graph.iter_adjacency() {
        for (dst, weight) in targets.iter().zip(weights.iter()) {
            fg = fg.add_edge(GraphEdge::new(src.0 as usize, *dst as usize).weight(*weight));
        }
    }

    let built = fg.build()?;
    built.to_framebuffer()
}

fn graph_with_analysis(
    graph: &CsrGraph,
    communities: &CommunityDetectionResult,
    scores: &[f32],
    width: u32,
    height: u32,
) -> Result<Framebuffer> {
    let mut fg = ForceGraph::new().dimensions(width, height).iterations(100);

    // Find min/max for normalization
    let max_score = scores.iter().copied().fold(0.0f32, f32::max);
    let min_score = scores.iter().copied().fold(f32::MAX, f32::min);
    let score_range = (max_score - min_score).max(0.001);

    // Add nodes with both community colors and PageRank sizing
    for i in 0..graph.num_nodes() {
        let node_id = NodeId(i as u32);

        // Community color
        let comm_id = communities.get_community(node_id).unwrap_or(0);
        let color = COMMUNITY_COLORS[comm_id % COMMUNITY_COLORS.len()];

        // PageRank size
        let score = scores.get(i).copied().unwrap_or(0.0);
        let normalized = (score - min_score) / score_range;
        let radius = 5.0 + normalized * 20.0;

        let mut node = GraphNode::new(i).color(color).radius(radius);
        if let Some(name) = graph.get_node_name(node_id) {
            node = node.label(name);
        }
        fg = fg.add_node(node);
    }

    // Add edges
    for (src, targets, weights) in graph.iter_adjacency() {
        for (dst, weight) in targets.iter().zip(weights.iter()) {
            fg = fg.add_edge(GraphEdge::new(src.0 as usize, *dst as usize).weight(*weight));
        }
    }

    let built = fg.build()?;
    built.to_framebuffer()
}

// ============================================================================
// PageRank Visualization
// ============================================================================

/// Visualization extensions for PageRank results.
pub trait PageRankViz {
    /// Create a histogram of PageRank scores.
    fn to_histogram(&self) -> Result<Framebuffer>;

    /// Create a bar chart of top N PageRank scores.
    fn top_n_bar(&self, n: usize) -> Result<Framebuffer>;
}

impl PageRankViz for Vec<f32> {
    fn to_histogram(&self) -> Result<Framebuffer> {
        let plot = Histogram::new()
            .data(self)
            .color(Rgba::new(66, 133, 244, 255))
            .dimensions(600, 400)
            .build()?;

        plot.to_framebuffer()
    }

    fn top_n_bar(&self, n: usize) -> Result<Framebuffer> {
        // Get indices sorted by score descending
        let mut indexed: Vec<(usize, f32)> = self.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_n: Vec<f32> = indexed.iter().take(n).map(|(_, s)| *s).collect();
        let x: Vec<f32> = (0..top_n.len()).map(|i| i as f32).collect();

        let plot = ScatterPlot::new()
            .x(&x)
            .y(&top_n)
            .color(Rgba::new(66, 133, 244, 255))
            .size(10.0)
            .dimensions(600, 400)
            .build()?;

        plot.to_framebuffer()
    }
}

// ============================================================================
// Community Visualization
// ============================================================================

/// Visualization extensions for community detection results.
pub trait CommunityViz {
    /// Create a histogram of community sizes.
    fn size_histogram(&self) -> Result<Framebuffer>;

    /// Get the modularity score.
    fn modularity_score(&self) -> f64;
}

impl CommunityViz for CommunityDetectionResult {
    fn size_histogram(&self) -> Result<Framebuffer> {
        let sizes: Vec<f32> = self.communities.iter().map(|c| c.len() as f32).collect();

        let plot = Histogram::new()
            .data(&sizes)
            .color(Rgba::new(52, 168, 83, 255))
            .dimensions(600, 400)
            .build()?;

        plot.to_framebuffer()
    }

    fn modularity_score(&self) -> f64 {
        self.modularity
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Visualize a graph using force-directed layout.
pub fn visualize_graph(graph: &CsrGraph) -> Result<Framebuffer> {
    graph.to_force_graph()
}

/// Visualize a graph with community detection coloring.
pub fn visualize_communities(graph: &CsrGraph) -> Result<Framebuffer> {
    graph.to_community_graph()
}

/// Visualize a graph with PageRank-based node sizing.
pub fn visualize_pagerank(graph: &CsrGraph) -> Result<Framebuffer> {
    graph.to_pagerank_graph()
}

/// Full analysis visualization (communities + PageRank).
pub fn visualize_analysis(graph: &CsrGraph) -> Result<Framebuffer> {
    graph.to_analysis_graph()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> CsrGraph {
        let edges = vec![
            (NodeId(0), NodeId(1), 1.0),
            (NodeId(1), NodeId(2), 1.0),
            (NodeId(2), NodeId(0), 1.0),
            (NodeId(2), NodeId(3), 1.0),
            (NodeId(3), NodeId(4), 1.0),
            (NodeId(4), NodeId(3), 1.0),
        ];
        CsrGraph::from_edge_list(&edges).expect("Failed to create graph")
    }

    #[test]
    fn test_to_force_graph() {
        let graph = create_test_graph();
        let fb = graph.to_force_graph().expect("operation should succeed");
        assert_eq!(fb.width(), 600);
        assert_eq!(fb.height(), 500);
    }

    #[test]
    fn test_to_force_graph_with() {
        let graph = create_test_graph();
        let fb = graph.to_force_graph_with(800, 600).expect("operation should succeed");
        assert_eq!(fb.width(), 800);
        assert_eq!(fb.height(), 600);
    }

    #[test]
    fn test_to_community_graph() {
        let graph = create_test_graph();
        let fb = graph.to_community_graph().expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_to_pagerank_graph() {
        let graph = create_test_graph();
        let fb = graph.to_pagerank_graph().expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_to_analysis_graph() {
        let graph = create_test_graph();
        let fb = graph.to_analysis_graph().expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_degree_histogram() {
        let graph = create_test_graph();
        let fb = graph.degree_histogram().expect("operation should succeed");
        assert_eq!(fb.width(), 600);
        assert_eq!(fb.height(), 400);
    }

    #[test]
    fn test_degree_scatter() {
        let graph = create_test_graph();
        let fb = graph.degree_scatter().expect("operation should succeed");
        assert_eq!(fb.width(), 600);
        assert_eq!(fb.height(), 500);
    }

    #[test]
    fn test_pagerank_histogram() {
        let graph = create_test_graph();
        let scores = pagerank(&graph, 20, 1e-6).expect("operation should succeed");
        let fb = scores.to_histogram().expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_pagerank_top_n() {
        let graph = create_test_graph();
        let scores = pagerank(&graph, 20, 1e-6).expect("operation should succeed");
        let fb = scores.top_n_bar(3).expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_community_size_histogram() {
        let graph = create_test_graph();
        let communities = louvain(&graph).expect("operation should succeed");
        let fb = communities.size_histogram().expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_convenience_functions() {
        let graph = create_test_graph();

        let fb = visualize_graph(&graph).expect("operation should succeed");
        assert!(fb.width() > 0);

        let fb = visualize_communities(&graph).expect("operation should succeed");
        assert!(fb.width() > 0);

        let fb = visualize_pagerank(&graph).expect("operation should succeed");
        assert!(fb.width() > 0);

        let fb = visualize_analysis(&graph).expect("operation should succeed");
        assert!(fb.width() > 0);
    }
}
