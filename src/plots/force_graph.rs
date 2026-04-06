//! Force-directed graph layout visualization.
//!
//! Implements the Fruchterman-Reingold algorithm for graph layout.
//! Reference: Fruchterman & Reingold (1991), "Graph Drawing by Force-directed Placement"

use crate::color::Rgba;
use crate::error::{Error, Result};
use crate::framebuffer::Framebuffer;
use crate::render::{draw_circle, draw_line};

/// A node in the graph.
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Node identifier
    pub id: usize,
    /// Node label (optional)
    pub label: Option<String>,
    /// Node color
    pub color: Rgba,
    /// Node radius
    pub radius: f32,
    /// Current x position
    x: f32,
    /// Current y position
    y: f32,
    /// Velocity x (for simulation)
    vx: f32,
    /// Velocity y (for simulation)
    vy: f32,
}

impl GraphNode {
    /// Create a new node.
    #[must_use]
    pub fn new(id: usize) -> Self {
        Self {
            id,
            label: None,
            color: Rgba::new(66, 133, 244, 255),
            radius: 8.0,
            x: 0.0,
            y: 0.0,
            vx: 0.0,
            vy: 0.0,
        }
    }

    /// Set the node label.
    #[must_use]
    pub fn label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }

    /// Set the node color.
    #[must_use]
    pub fn color(mut self, color: Rgba) -> Self {
        self.color = color;
        self
    }

    /// Set the node radius.
    #[must_use]
    pub fn radius(mut self, radius: f32) -> Self {
        self.radius = radius;
        self
    }

    /// Get node position.
    #[must_use]
    pub fn position(&self) -> (f32, f32) {
        (self.x, self.y)
    }
}

/// An edge in the graph.
#[derive(Debug, Clone)]
pub struct GraphEdge {
    /// Source node index
    pub source: usize,
    /// Target node index
    pub target: usize,
    /// Edge weight (affects spring strength)
    pub weight: f32,
    /// Edge color
    pub color: Rgba,
    /// Edge width
    pub width: f32,
}

impl GraphEdge {
    /// Create a new edge.
    #[must_use]
    pub fn new(source: usize, target: usize) -> Self {
        Self { source, target, weight: 1.0, color: Rgba::new(150, 150, 150, 200), width: 1.0 }
    }

    /// Set edge weight.
    #[must_use]
    pub fn weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Set edge color.
    #[must_use]
    pub fn color(mut self, color: Rgba) -> Self {
        self.color = color;
        self
    }

    /// Set edge width.
    #[must_use]
    pub fn width(mut self, width: f32) -> Self {
        self.width = width;
        self
    }
}

/// Force-directed graph layout using Fruchterman-Reingold algorithm.
#[derive(Debug, Clone)]
pub struct ForceGraph {
    /// Graph nodes
    nodes: Vec<GraphNode>,
    /// Graph edges
    edges: Vec<GraphEdge>,
    /// Image width
    width: u32,
    /// Image height
    height: u32,
    /// Margin around the graph
    margin: u32,
    /// Number of simulation iterations
    iterations: usize,
    /// Repulsion strength (k² term)
    repulsion: f32,
    /// Attraction strength
    attraction: f32,
    /// Temperature (cooling factor)
    temperature: f32,
    /// Background color
    background: Rgba,
}

impl Default for ForceGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl ForceGraph {
    /// Create a new force-directed graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            width: 600,
            height: 400,
            margin: 40,
            iterations: 100,
            repulsion: 10000.0,
            attraction: 0.01,
            temperature: 100.0,
            background: Rgba::WHITE,
        }
    }

    /// Add a node to the graph.
    #[must_use]
    pub fn add_node(mut self, node: GraphNode) -> Self {
        self.nodes.push(node);
        self
    }

    /// Add an edge to the graph.
    #[must_use]
    pub fn add_edge(mut self, edge: GraphEdge) -> Self {
        self.edges.push(edge);
        self
    }

    /// Add a simple edge by node indices.
    #[must_use]
    pub fn edge(mut self, source: usize, target: usize) -> Self {
        self.edges.push(GraphEdge::new(source, target));
        self
    }

    /// Set margin.
    #[must_use]
    pub fn margin(mut self, margin: u32) -> Self {
        self.margin = margin;
        self
    }

    /// Set number of simulation iterations.
    #[must_use]
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Set repulsion strength.
    #[must_use]
    pub fn repulsion(mut self, repulsion: f32) -> Self {
        self.repulsion = repulsion;
        self
    }

    /// Set attraction strength.
    #[must_use]
    pub fn attraction(mut self, attraction: f32) -> Self {
        self.attraction = attraction;
        self
    }

    /// Set background color.
    #[must_use]
    pub fn background(mut self, color: Rgba) -> Self {
        self.background = color;
        self
    }

    /// Build and run the force simulation.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph is empty.
    pub fn build(mut self) -> Result<BuiltForceGraph> {
        if self.nodes.is_empty() {
            return Err(Error::EmptyData);
        }

        // Validate edges
        for edge in &self.edges {
            if edge.source >= self.nodes.len() || edge.target >= self.nodes.len() {
                return Err(Error::Rendering(format!(
                    "Invalid edge: {} -> {} (only {} nodes)",
                    edge.source,
                    edge.target,
                    self.nodes.len()
                )));
            }
        }

        // Initialize node positions randomly
        let area_width = (self.width - 2 * self.margin) as f32;
        let area_height = (self.height - 2 * self.margin) as f32;

        for (i, node) in self.nodes.iter_mut().enumerate() {
            // Deterministic pseudo-random initial positions
            let seed = i.wrapping_mul(1_103_515_245).wrapping_add(12345);
            node.x = self.margin as f32 + (seed % 1000) as f32 / 1000.0 * area_width;
            node.y = self.margin as f32 + ((seed / 1000) % 1000) as f32 / 1000.0 * area_height;
        }

        // Run Fruchterman-Reingold simulation
        self.run_simulation(area_width, area_height);

        Ok(BuiltForceGraph {
            nodes: self.nodes,
            edges: self.edges,
            width: self.width,
            height: self.height,
            margin: self.margin,
            background: self.background,
        })
    }

    /// Run the force-directed layout simulation.
    fn run_simulation(&mut self, area_width: f32, area_height: f32) {
        let n = self.nodes.len();
        if n == 0 {
            return;
        }

        // Optimal distance factor (not directly used but kept for reference)
        let _k = (area_width * area_height / n as f32).sqrt();
        let mut temp = self.temperature;

        for _ in 0..self.iterations {
            // Calculate repulsive forces between all pairs
            for i in 0..n {
                self.nodes[i].vx = 0.0;
                self.nodes[i].vy = 0.0;

                for j in 0..n {
                    if i == j {
                        continue;
                    }

                    let dx = self.nodes[i].x - self.nodes[j].x;
                    let dy = self.nodes[i].y - self.nodes[j].y;
                    let dist = (dx * dx + dy * dy).sqrt().max(0.01);

                    // Repulsive force: k² / d
                    let force = self.repulsion / dist;
                    self.nodes[i].vx += dx / dist * force;
                    self.nodes[i].vy += dy / dist * force;
                }
            }

            // Calculate attractive forces along edges
            for edge in &self.edges {
                let dx = self.nodes[edge.target].x - self.nodes[edge.source].x;
                let dy = self.nodes[edge.target].y - self.nodes[edge.source].y;
                let dist = (dx * dx + dy * dy).sqrt().max(0.01);

                // Attractive force: d² / k
                let force = dist * self.attraction * edge.weight;

                let fx = dx / dist * force;
                let fy = dy / dist * force;

                self.nodes[edge.source].vx += fx;
                self.nodes[edge.source].vy += fy;
                self.nodes[edge.target].vx -= fx;
                self.nodes[edge.target].vy -= fy;
            }

            // Update positions with temperature-limited displacement
            for node in &mut self.nodes {
                let disp = (node.vx * node.vx + node.vy * node.vy).sqrt().max(0.01);
                let capped_disp = disp.min(temp);

                node.x += node.vx / disp * capped_disp;
                node.y += node.vy / disp * capped_disp;

                // Keep nodes within bounds
                node.x = node.x.clamp(self.margin as f32, self.margin as f32 + area_width);
                node.y = node.y.clamp(self.margin as f32, self.margin as f32 + area_height);
            }

            // Cool down
            temp *= 0.95;
        }
    }
}

impl batuta_common::display::WithDimensions for ForceGraph {
    fn set_dimensions(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }
}

/// A built force-directed graph ready for rendering.
#[derive(Debug)]
pub struct BuiltForceGraph {
    nodes: Vec<GraphNode>,
    edges: Vec<GraphEdge>,
    width: u32,
    height: u32,
    margin: u32,
    background: Rgba,
}

impl BuiltForceGraph {
    /// Get number of nodes.
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get number of edges.
    #[must_use]
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Get the margin around the graph.
    #[must_use]
    pub fn margin(&self) -> u32 {
        self.margin
    }

    /// Get node positions.
    #[must_use]
    pub fn positions(&self) -> Vec<(f32, f32)> {
        self.nodes.iter().map(|n| (n.x, n.y)).collect()
    }

    /// Render to a new framebuffer.
    ///
    /// # Errors
    ///
    /// Returns an error if framebuffer creation fails.
    pub fn to_framebuffer(&self) -> Result<Framebuffer> {
        let mut fb = Framebuffer::new(self.width, self.height)?;
        fb.clear(self.background);
        self.render(&mut fb)?;
        Ok(fb)
    }

    /// Render onto an existing framebuffer.
    ///
    /// # Errors
    ///
    /// Returns an error if rendering fails.
    pub fn render(&self, fb: &mut Framebuffer) -> Result<()> {
        // Draw edges first (below nodes)
        for edge in &self.edges {
            let source = &self.nodes[edge.source];
            let target = &self.nodes[edge.target];

            draw_line(
                fb,
                source.x as i32,
                source.y as i32,
                target.x as i32,
                target.y as i32,
                edge.color,
            );
        }

        // Draw nodes
        for node in &self.nodes {
            draw_circle(fb, node.x as i32, node.y as i32, node.radius as i32, node.color);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use batuta_common::display::WithDimensions;

    #[test]
    fn test_graph_node_builder() {
        let node = GraphNode::new(0).label("Test").color(Rgba::RED).radius(10.0);

        assert_eq!(node.id, 0);
        assert_eq!(node.label, Some("Test".to_string()));
        assert_eq!(node.color, Rgba::RED);
        assert!((node.radius - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_graph_edge_builder() {
        let edge = GraphEdge::new(0, 1).weight(2.0).color(Rgba::BLUE).width(3.0);

        assert_eq!(edge.source, 0);
        assert_eq!(edge.target, 1);
        assert!((edge.weight - 2.0).abs() < 0.01);
        assert_eq!(edge.color, Rgba::BLUE);
        assert!((edge.width - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_force_graph_empty_error() {
        let result = ForceGraph::new().build();
        assert!(result.is_err());
    }

    #[test]
    fn test_force_graph_invalid_edge() {
        let result = ForceGraph::new()
            .add_node(GraphNode::new(0))
            .edge(0, 5) // Invalid: node 5 doesn't exist
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_force_graph_single_node() {
        let graph = ForceGraph::new()
            .add_node(GraphNode::new(0))
            .dimensions(200, 200)
            .build()
            .expect("builder should produce valid result");

        assert_eq!(graph.num_nodes(), 1);
        assert_eq!(graph.num_edges(), 0);
    }

    #[test]
    fn test_force_graph_simple() {
        let graph = ForceGraph::new()
            .add_node(GraphNode::new(0))
            .add_node(GraphNode::new(1))
            .add_node(GraphNode::new(2))
            .edge(0, 1)
            .edge(1, 2)
            .edge(2, 0)
            .dimensions(300, 300)
            .iterations(50)
            .build()
            .expect("operation should succeed");

        assert_eq!(graph.num_nodes(), 3);
        assert_eq!(graph.num_edges(), 3);

        // Positions should be within bounds
        for (x, y) in graph.positions() {
            assert!((0.0..=300.0).contains(&x));
            assert!((0.0..=300.0).contains(&y));
        }
    }

    #[test]
    fn test_force_graph_render() {
        let graph = ForceGraph::new()
            .add_node(GraphNode::new(0))
            .add_node(GraphNode::new(1))
            .edge(0, 1)
            .dimensions(200, 150)
            .build()
            .expect("operation should succeed");

        let fb = graph.to_framebuffer().expect("operation should succeed");
        assert_eq!(fb.width(), 200);
        assert_eq!(fb.height(), 150);
    }

    #[test]
    fn test_force_graph_separation() {
        // Two unconnected nodes should repel each other
        let graph = ForceGraph::new()
            .add_node(GraphNode::new(0))
            .add_node(GraphNode::new(1))
            .dimensions(400, 400)
            .iterations(100)
            .build()
            .expect("operation should succeed");

        let positions = graph.positions();
        let (x0, y0) = positions[0];
        let (x1, y1) = positions[1];

        // Calculate distance
        let dist = ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt();

        // Unconnected nodes should be far apart
        assert!(dist > 100.0, "Unconnected nodes should repel: dist={dist}");
    }

    #[test]
    fn test_force_graph_attraction() {
        // Connected nodes should be closer than unconnected ones
        // Build two graphs: one with edge, one without
        let graph_connected = ForceGraph::new()
            .add_node(GraphNode::new(0))
            .add_node(GraphNode::new(1))
            .edge(0, 1)
            .dimensions(400, 400)
            .iterations(100)
            .repulsion(5000.0)
            .attraction(0.05)
            .build()
            .expect("operation should succeed");

        let graph_disconnected = ForceGraph::new()
            .add_node(GraphNode::new(0))
            .add_node(GraphNode::new(1))
            .dimensions(400, 400)
            .iterations(100)
            .repulsion(5000.0)
            .build()
            .expect("operation should succeed");

        let pos_conn = graph_connected.positions();
        let pos_disc = graph_disconnected.positions();

        let dist_conn = ((pos_conn[1].0 - pos_conn[0].0).powi(2)
            + (pos_conn[1].1 - pos_conn[0].1).powi(2))
        .sqrt();

        let dist_disc = ((pos_disc[1].0 - pos_disc[0].0).powi(2)
            + (pos_disc[1].1 - pos_disc[0].1).powi(2))
        .sqrt();

        // Connected nodes should be closer than disconnected
        assert!(
            dist_conn < dist_disc,
            "Connected ({dist_conn}) should be closer than disconnected ({dist_disc})"
        );
    }
}
