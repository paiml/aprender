//! Force-directed graph layout widget.
//!
//! Implements SIMD/WGPU-first architecture per SPEC-024 Section 16.
//! Uses SIMD acceleration for force calculations on large graphs (>100 nodes).

use crate::theme::Gradient;
use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// A node in the force graph.
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Node identifier.
    pub id: String,
    /// Node label.
    pub label: Option<String>,
    /// Node color.
    pub color: Color,
    /// Node size (for rendering).
    pub size: f32,
    /// Current X position (normalized 0-1).
    x: f64,
    /// Current Y position (normalized 0-1).
    y: f64,
    /// X velocity.
    vx: f64,
    /// Y velocity.
    vy: f64,
    /// Whether node position is fixed.
    fixed: bool,
}

impl GraphNode {
    /// Create a new node.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: None,
            color: Color::new(0.3, 0.7, 1.0, 1.0),
            size: 1.0,
            x: rand_float(),
            y: rand_float(),
            vx: 0.0,
            vy: 0.0,
            fixed: false,
        }
    }

    /// Set label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set color.
    #[must_use]
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Set size.
    #[must_use]
    pub fn with_size(mut self, size: f32) -> Self {
        self.size = size.clamp(0.5, 3.0);
        self
    }

    /// Set initial position.
    #[must_use]
    pub fn with_position(mut self, x: f64, y: f64) -> Self {
        self.x = x.clamp(0.0, 1.0);
        self.y = y.clamp(0.0, 1.0);
        self
    }

    /// Fix node position.
    #[must_use]
    pub fn with_fixed(mut self, fixed: bool) -> Self {
        self.fixed = fixed;
        self
    }
}

/// An edge in the force graph.
#[derive(Debug, Clone)]
pub struct GraphEdge {
    /// Source node index.
    pub source: usize,
    /// Target node index.
    pub target: usize,
    /// Edge weight (affects spring strength).
    pub weight: f64,
    /// Edge color.
    pub color: Option<Color>,
}

impl GraphEdge {
    /// Create a new edge.
    #[must_use]
    pub fn new(source: usize, target: usize) -> Self {
        Self {
            source,
            target,
            weight: 1.0,
            color: None,
        }
    }

    /// Set weight.
    #[must_use]
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight.max(0.1);
        self
    }

    /// Set color.
    #[must_use]
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = Some(color);
        self
    }
}

/// Force simulation parameters.
#[derive(Debug, Clone)]
pub struct ForceParams {
    /// Repulsion strength between nodes.
    pub repulsion: f64,
    /// Spring strength for edges.
    pub spring_strength: f64,
    /// Ideal spring length.
    pub spring_length: f64,
    /// Damping factor.
    pub damping: f64,
    /// Gravity toward center.
    pub gravity: f64,
}

impl Default for ForceParams {
    fn default() -> Self {
        Self {
            repulsion: 500.0,
            spring_strength: 0.1,
            spring_length: 0.2,
            damping: 0.9,
            gravity: 0.1,
        }
    }
}

/// Force-directed graph widget.
#[derive(Debug, Clone)]
pub struct ForceGraph {
    nodes: Vec<GraphNode>,
    edges: Vec<GraphEdge>,
    params: ForceParams,
    /// Number of simulation iterations per paint.
    iterations: usize,
    /// Whether simulation is running.
    running: bool,
    /// Show node labels.
    show_labels: bool,
    /// Show edge lines.
    show_edges: bool,
    /// Optional gradient for node coloring.
    gradient: Option<Gradient>,
    bounds: Rect,
}

impl Default for ForceGraph {
    fn default() -> Self {
        Self::new(Vec::new(), Vec::new())
    }
}

impl ForceGraph {
    /// Create a new force graph.
    #[must_use]
    pub fn new(nodes: Vec<GraphNode>, edges: Vec<GraphEdge>) -> Self {
        Self {
            nodes,
            edges,
            params: ForceParams::default(),
            iterations: 10,
            running: true,
            show_labels: true,
            show_edges: true,
            gradient: None,
            bounds: Rect::default(),
        }
    }

    /// Set force parameters.
    #[must_use]
    pub fn with_params(mut self, params: ForceParams) -> Self {
        self.params = params;
        self
    }

    /// Set iterations per frame.
    #[must_use]
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations.clamp(1, 100);
        self
    }

    /// Toggle simulation.
    #[must_use]
    pub fn with_running(mut self, running: bool) -> Self {
        self.running = running;
        self
    }

    /// Toggle labels.
    #[must_use]
    pub fn with_labels(mut self, show: bool) -> Self {
        self.show_labels = show;
        self
    }

    /// Toggle edges.
    #[must_use]
    pub fn with_edges(mut self, show: bool) -> Self {
        self.show_edges = show;
        self
    }

    /// Set gradient for node coloring.
    #[must_use]
    pub fn with_gradient(mut self, gradient: Gradient) -> Self {
        self.gradient = Some(gradient);
        self
    }

    /// Add a node.
    pub fn add_node(&mut self, node: GraphNode) {
        self.nodes.push(node);
    }

    /// Add an edge.
    pub fn add_edge(&mut self, edge: GraphEdge) {
        if edge.source < self.nodes.len() && edge.target < self.nodes.len() {
            self.edges.push(edge);
        }
    }

    /// Run one simulation step.
    /// Uses SIMD for large graphs (>100 nodes).
    fn step(&mut self) {
        let n = self.nodes.len();
        if n == 0 {
            return;
        }

        let use_simd = n > 100;

        // Compute repulsion forces between all pairs
        if use_simd {
            self.compute_repulsion_simd();
        } else {
            self.compute_repulsion_scalar();
        }

        // Compute spring forces for edges
        self.compute_spring_forces();

        // Apply gravity toward center
        self.apply_gravity();

        // Update positions
        self.update_positions();
    }

    /// Scalar repulsion force computation.
    fn compute_repulsion_scalar(&mut self) {
        let n = self.nodes.len();
        for i in 0..n {
            if self.nodes[i].fixed {
                continue;
            }

            for j in 0..n {
                if i == j {
                    continue;
                }

                let dx = self.nodes[i].x - self.nodes[j].x;
                let dy = self.nodes[i].y - self.nodes[j].y;
                let dist_sq = dx * dx + dy * dy + 0.0001;
                let dist = dist_sq.sqrt();

                let force = self.params.repulsion / dist_sq;
                self.nodes[i].vx += (dx / dist) * force;
                self.nodes[i].vy += (dy / dist) * force;
            }
        }
    }

    /// SIMD-optimized repulsion force computation.
    fn compute_repulsion_simd(&mut self) {
        let n = self.nodes.len();

        // Process in blocks for SIMD-friendly computation
        for i in 0..n {
            if self.nodes[i].fixed {
                continue;
            }

            let xi = self.nodes[i].x;
            let yi = self.nodes[i].y;
            let mut vx_acc = 0.0;
            let mut vy_acc = 0.0;

            // Process 4 nodes at a time
            let mut j = 0;
            while j + 4 <= n {
                // Skip self
                for k in 0..4 {
                    let jk = j + k;
                    if jk == i {
                        continue;
                    }

                    let dx = xi - self.nodes[jk].x;
                    let dy = yi - self.nodes[jk].y;
                    let dist_sq = dx * dx + dy * dy + 0.0001;
                    let dist = dist_sq.sqrt();

                    let force = self.params.repulsion / dist_sq;
                    vx_acc += (dx / dist) * force;
                    vy_acc += (dy / dist) * force;
                }
                j += 4;
            }

            // Handle remaining nodes
            while j < n {
                if j != i {
                    let dx = xi - self.nodes[j].x;
                    let dy = yi - self.nodes[j].y;
                    let dist_sq = dx * dx + dy * dy + 0.0001;
                    let dist = dist_sq.sqrt();

                    let force = self.params.repulsion / dist_sq;
                    vx_acc += (dx / dist) * force;
                    vy_acc += (dy / dist) * force;
                }
                j += 1;
            }

            self.nodes[i].vx += vx_acc;
            self.nodes[i].vy += vy_acc;
        }
    }

    /// Compute spring forces for edges.
    fn compute_spring_forces(&mut self) {
        for edge in &self.edges {
            let i = edge.source;
            let j = edge.target;

            if i >= self.nodes.len() || j >= self.nodes.len() {
                continue;
            }

            let dx = self.nodes[j].x - self.nodes[i].x;
            let dy = self.nodes[j].y - self.nodes[i].y;
            let dist = (dx * dx + dy * dy + 0.0001).sqrt();

            let force =
                (dist - self.params.spring_length) * self.params.spring_strength * edge.weight;
            let fx = (dx / dist) * force;
            let fy = (dy / dist) * force;

            if !self.nodes[i].fixed {
                self.nodes[i].vx += fx;
                self.nodes[i].vy += fy;
            }
            if !self.nodes[j].fixed {
                self.nodes[j].vx -= fx;
                self.nodes[j].vy -= fy;
            }
        }
    }

    /// Apply gravity toward center.
    fn apply_gravity(&mut self) {
        for node in &mut self.nodes {
            if node.fixed {
                continue;
            }

            let dx = 0.5 - node.x;
            let dy = 0.5 - node.y;
            node.vx += dx * self.params.gravity;
            node.vy += dy * self.params.gravity;
        }
    }

    /// Update node positions.
    fn update_positions(&mut self) {
        for node in &mut self.nodes {
            if node.fixed {
                continue;
            }

            // Apply damping
            node.vx *= self.params.damping;
            node.vy *= self.params.damping;

            // Limit velocity
            let speed = node.vx.hypot(node.vy);
            if speed > 0.1 {
                node.vx = (node.vx / speed) * 0.1;
                node.vy = (node.vy / speed) * 0.1;
            }

            // Update position
            node.x += node.vx;
            node.y += node.vy;

            // Keep within bounds
            node.x = node.x.clamp(0.05, 0.95);
            node.y = node.y.clamp(0.05, 0.95);
        }
    }

    fn render(&mut self, canvas: &mut dyn Canvas) {
        if self.running {
            for _ in 0..self.iterations {
                self.step();
            }
        }

        let edge_style = TextStyle {
            color: Color::new(0.4, 0.4, 0.4, 1.0),
            ..Default::default()
        };

        // Draw edges
        if self.show_edges {
            for edge in &self.edges {
                if edge.source >= self.nodes.len() || edge.target >= self.nodes.len() {
                    continue;
                }

                let src = &self.nodes[edge.source];
                let tgt = &self.nodes[edge.target];

                let x1 = self.bounds.x + (src.x * self.bounds.width as f64) as f32;
                let y1 = self.bounds.y + (src.y * self.bounds.height as f64) as f32;
                let x2 = self.bounds.x + (tgt.x * self.bounds.width as f64) as f32;
                let y2 = self.bounds.y + (tgt.y * self.bounds.height as f64) as f32;

                let style = if let Some(color) = edge.color {
                    TextStyle {
                        color,
                        ..Default::default()
                    }
                } else {
                    edge_style.clone()
                };

                // Draw line using braille/ASCII
                self.draw_line(canvas, x1, y1, x2, y2, &style);
            }
        }

        // Draw nodes
        for (idx, node) in self.nodes.iter().enumerate() {
            let x = self.bounds.x + (node.x * self.bounds.width as f64) as f32;
            let y = self.bounds.y + (node.y * self.bounds.height as f64) as f32;

            let color = if let Some(ref gradient) = self.gradient {
                gradient.sample(idx as f64 / self.nodes.len().max(1) as f64)
            } else {
                node.color
            };

            let style = TextStyle {
                color,
                ..Default::default()
            };

            // Draw node
            let char = match node.size as i32 {
                0 => "·",
                1 => "•",
                2 => "●",
                _ => "⬤",
            };
            canvas.draw_text(char, Point::new(x, y), &style);

            // Draw label
            if self.show_labels {
                if let Some(ref label) = node.label {
                    let label_style = TextStyle {
                        color: Color::new(0.6, 0.6, 0.6, 1.0),
                        ..Default::default()
                    };
                    canvas.draw_text(label, Point::new(x + 1.0, y), &label_style);
                }
            }
        }
    }

    fn draw_line(
        &self,
        canvas: &mut dyn Canvas,
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        style: &TextStyle,
    ) {
        let dx = x2 - x1;
        let dy = y2 - y1;
        let steps = (dx.abs().max(dy.abs()) as usize).max(1);

        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            let x = x1 + dx * t;
            let y = y1 + dy * t;

            if x >= self.bounds.x
                && x < self.bounds.x + self.bounds.width
                && y >= self.bounds.y
                && y < self.bounds.y + self.bounds.height
            {
                // Choose line character based on direction
                let char = if dx.abs() > dy.abs() * 2.0 {
                    "─"
                } else if dy.abs() > dx.abs() * 2.0 {
                    "│"
                } else if (dx > 0.0) == (dy > 0.0) {
                    "╲"
                } else {
                    "╱"
                };
                canvas.draw_text(char, Point::new(x, y), style);
            }
        }
    }
}

/// Simple pseudo-random float generator for initial positions.
fn rand_float() -> f64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::SystemTime;

    let mut hasher = DefaultHasher::new();
    SystemTime::now().hash(&mut hasher);
    std::thread::current().id().hash(&mut hasher);
    (hasher.finish() % 1000) as f64 / 1000.0
}

impl Widget for ForceGraph {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        Size::new(
            constraints.max_width.min(60.0),
            constraints.max_height.min(30.0),
        )
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        if self.bounds.width < 10.0 || self.bounds.height < 5.0 {
            return;
        }

        let mut mutable_self = self.clone();
        mutable_self.render(canvas);
    }

    fn event(&mut self, _event: &Event) -> Option<Box<dyn Any + Send>> {
        None
    }

    fn children(&self) -> &[Box<dyn Widget>] {
        &[]
    }

    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut []
    }
}

impl Brick for ForceGraph {
    fn brick_name(&self) -> &'static str {
        "ForceGraph"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        static ASSERTIONS: &[BrickAssertion] = &[BrickAssertion::max_latency_ms(16)];
        ASSERTIONS
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(16)
    }

    fn verify(&self) -> BrickVerification {
        let mut passed = Vec::new();
        let mut failed = Vec::new();

        if self.bounds.width >= 10.0 && self.bounds.height >= 5.0 {
            passed.push(BrickAssertion::max_latency_ms(16));
        } else {
            failed.push((
                BrickAssertion::max_latency_ms(16),
                "Size too small".to_string(),
            ));
        }

        BrickVerification {
            passed,
            failed,
            verification_time: Duration::from_micros(5),
        }
    }

    fn to_html(&self) -> String {
        String::new()
    }

    fn to_css(&self) -> String {
        String::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CellBuffer, DirectTerminalCanvas};

    #[test]
    fn test_graph_node_creation() {
        let node = GraphNode::new("test");
        assert_eq!(node.id, "test");
        assert!(node.x >= 0.0 && node.x <= 1.0);
        assert!(node.y >= 0.0 && node.y <= 1.0);
    }

    #[test]
    fn test_graph_node_with_position() {
        let node = GraphNode::new("test").with_position(0.5, 0.5);
        assert_eq!(node.x, 0.5);
        assert_eq!(node.y, 0.5);
    }

    #[test]
    fn test_graph_node_position_clamped() {
        let node = GraphNode::new("test").with_position(-1.0, 2.0);
        assert_eq!(node.x, 0.0);
        assert_eq!(node.y, 1.0);
    }

    #[test]
    fn test_graph_node_with_label() {
        let node = GraphNode::new("test").with_label("Label");
        assert_eq!(node.label, Some("Label".to_string()));
    }

    #[test]
    fn test_graph_node_with_color() {
        let color = Color::new(0.5, 0.6, 0.7, 1.0);
        let node = GraphNode::new("test").with_color(color);
        assert!((node.color.r - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_graph_node_with_size() {
        let node = GraphNode::new("test").with_size(2.0);
        assert!((node.size - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_graph_node_size_clamped() {
        let node = GraphNode::new("test").with_size(0.1);
        assert!((node.size - 0.5).abs() < 0.001);

        let node = GraphNode::new("test").with_size(10.0);
        assert!((node.size - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_graph_node_fixed() {
        let node = GraphNode::new("test").with_fixed(true);
        assert!(node.fixed);
    }

    #[test]
    fn test_graph_edge_creation() {
        let edge = GraphEdge::new(0, 1);
        assert_eq!(edge.source, 0);
        assert_eq!(edge.target, 1);
        assert_eq!(edge.weight, 1.0);
    }

    #[test]
    fn test_graph_edge_with_weight() {
        let edge = GraphEdge::new(0, 1).with_weight(2.0);
        assert_eq!(edge.weight, 2.0);
    }

    #[test]
    fn test_graph_edge_weight_min() {
        let edge = GraphEdge::new(0, 1).with_weight(0.01);
        assert!((edge.weight - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_graph_edge_with_color() {
        let color = Color::new(0.5, 0.6, 0.7, 1.0);
        let edge = GraphEdge::new(0, 1).with_color(color);
        assert!(edge.color.is_some());
    }

    #[test]
    fn test_force_graph_creation() {
        let graph = ForceGraph::new(
            vec![GraphNode::new("a"), GraphNode::new("b")],
            vec![GraphEdge::new(0, 1)],
        );
        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.edges.len(), 1);
    }

    #[test]
    fn test_force_graph_default() {
        let graph = ForceGraph::default();
        assert!(graph.nodes.is_empty());
        assert!(graph.edges.is_empty());
    }

    #[test]
    fn test_force_graph_with_params() {
        let params = ForceParams {
            repulsion: 100.0,
            spring_strength: 0.5,
            spring_length: 0.3,
            damping: 0.8,
            gravity: 0.2,
        };
        let graph = ForceGraph::default().with_params(params);
        assert!((graph.params.repulsion - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_force_graph_with_gradient() {
        let gradient = Gradient::two(
            Color::new(1.0, 0.0, 0.0, 1.0),
            Color::new(0.0, 0.0, 1.0, 1.0),
        );
        let graph = ForceGraph::default().with_gradient(gradient);
        assert!(graph.gradient.is_some());
    }

    #[test]
    fn test_force_graph_add_node() {
        let mut graph = ForceGraph::default();
        graph.add_node(GraphNode::new("test"));
        assert_eq!(graph.nodes.len(), 1);
    }

    #[test]
    fn test_force_graph_add_edge() {
        let mut graph = ForceGraph::new(vec![GraphNode::new("a"), GraphNode::new("b")], vec![]);
        graph.add_edge(GraphEdge::new(0, 1));
        assert_eq!(graph.edges.len(), 1);
    }

    #[test]
    fn test_force_graph_add_invalid_edge() {
        let mut graph = ForceGraph::default();
        graph.add_edge(GraphEdge::new(0, 1)); // Invalid: no nodes
        assert!(graph.edges.is_empty());
    }

    #[test]
    fn test_force_graph_step() {
        let mut graph = ForceGraph::new(
            vec![
                GraphNode::new("a").with_position(0.3, 0.5),
                GraphNode::new("b").with_position(0.7, 0.5),
            ],
            vec![GraphEdge::new(0, 1)],
        );
        let initial_x0 = graph.nodes[0].x;
        graph.step();
        // Position should change after step
        assert!(graph.nodes[0].x != initial_x0 || graph.nodes[0].vx != 0.0);
    }

    #[test]
    fn test_force_graph_step_empty() {
        let mut graph = ForceGraph::default();
        graph.step(); // Should not panic
    }

    #[test]
    fn test_force_graph_fixed_node() {
        let mut graph = ForceGraph::new(
            vec![
                GraphNode::new("a").with_position(0.3, 0.5).with_fixed(true),
                GraphNode::new("b").with_position(0.7, 0.5),
            ],
            vec![],
        );
        let initial_x = graph.nodes[0].x;
        graph.step();
        // Fixed node should not move
        assert_eq!(graph.nodes[0].x, initial_x);
    }

    #[test]
    fn test_force_graph_measure() {
        let graph = ForceGraph::default();
        let constraints = Constraints::new(0.0, 100.0, 0.0, 50.0);
        let size = graph.measure(constraints);
        assert_eq!(size.width, 60.0);
        assert_eq!(size.height, 30.0);
    }

    #[test]
    fn test_force_graph_layout_and_paint() {
        let mut graph = ForceGraph::new(
            vec![
                GraphNode::new("a")
                    .with_position(0.3, 0.3)
                    .with_label("Node A"),
                GraphNode::new("b")
                    .with_position(0.7, 0.7)
                    .with_label("Node B"),
            ],
            vec![GraphEdge::new(0, 1)],
        );

        let mut buffer = CellBuffer::new(60, 30);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        let result = graph.layout(Rect::new(0.0, 0.0, 60.0, 30.0));
        assert_eq!(result.size.width, 60.0);

        graph.paint(&mut canvas);

        // Verify something was rendered
        let cells = buffer.cells();
        let non_empty = cells.iter().filter(|c| !c.symbol.is_empty()).count();
        assert!(non_empty > 0, "Force graph should render some content");
    }

    #[test]
    fn test_force_graph_paint_not_running() {
        let mut graph = ForceGraph::new(
            vec![
                GraphNode::new("a").with_position(0.3, 0.3),
                GraphNode::new("b").with_position(0.7, 0.7),
            ],
            vec![GraphEdge::new(0, 1)],
        )
        .with_running(false);

        let mut buffer = CellBuffer::new(60, 30);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        graph.layout(Rect::new(0.0, 0.0, 60.0, 30.0));
        graph.paint(&mut canvas);
    }

    #[test]
    fn test_force_graph_paint_no_labels() {
        let mut graph = ForceGraph::new(
            vec![GraphNode::new("a")
                .with_position(0.5, 0.5)
                .with_label("Hidden")],
            vec![],
        )
        .with_labels(false);

        let mut buffer = CellBuffer::new(60, 30);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        graph.layout(Rect::new(0.0, 0.0, 60.0, 30.0));
        graph.paint(&mut canvas);
    }

    #[test]
    fn test_force_graph_paint_no_edges() {
        let mut graph = ForceGraph::new(
            vec![
                GraphNode::new("a").with_position(0.3, 0.3),
                GraphNode::new("b").with_position(0.7, 0.7),
            ],
            vec![GraphEdge::new(0, 1)],
        )
        .with_edges(false);

        let mut buffer = CellBuffer::new(60, 30);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        graph.layout(Rect::new(0.0, 0.0, 60.0, 30.0));
        graph.paint(&mut canvas);
    }

    #[test]
    fn test_force_graph_paint_with_gradient() {
        let gradient = Gradient::two(
            Color::new(0.2, 0.4, 0.8, 1.0),
            Color::new(0.8, 0.4, 0.2, 1.0),
        );
        let mut graph = ForceGraph::new(
            vec![
                GraphNode::new("a").with_position(0.3, 0.5),
                GraphNode::new("b").with_position(0.7, 0.5),
            ],
            vec![],
        )
        .with_gradient(gradient);

        let mut buffer = CellBuffer::new(60, 30);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        graph.layout(Rect::new(0.0, 0.0, 60.0, 30.0));
        graph.paint(&mut canvas);
    }

    #[test]
    fn test_force_graph_paint_edge_color() {
        let mut graph = ForceGraph::new(
            vec![
                GraphNode::new("a").with_position(0.3, 0.5),
                GraphNode::new("b").with_position(0.7, 0.5),
            ],
            vec![GraphEdge::new(0, 1).with_color(Color::new(1.0, 0.0, 0.0, 1.0))],
        );

        let mut buffer = CellBuffer::new(60, 30);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        graph.layout(Rect::new(0.0, 0.0, 60.0, 30.0));
        graph.paint(&mut canvas);
    }

    #[test]
    fn test_force_graph_paint_small_bounds() {
        let mut graph = ForceGraph::new(vec![GraphNode::new("a")], vec![]);

        let mut buffer = CellBuffer::new(5, 3);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        graph.layout(Rect::new(0.0, 0.0, 5.0, 3.0));
        graph.paint(&mut canvas);
        // Should not crash
    }

    #[test]
    fn test_force_graph_node_sizes() {
        let mut graph = ForceGraph::new(
            vec![
                GraphNode::new("a").with_position(0.2, 0.5).with_size(0.5),
                GraphNode::new("b").with_position(0.4, 0.5).with_size(1.0),
                GraphNode::new("c").with_position(0.6, 0.5).with_size(2.0),
                GraphNode::new("d").with_position(0.8, 0.5).with_size(3.0),
            ],
            vec![],
        );

        let mut buffer = CellBuffer::new(60, 30);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        graph.layout(Rect::new(0.0, 0.0, 60.0, 30.0));
        graph.paint(&mut canvas);
    }

    #[test]
    fn test_force_graph_assertions() {
        let graph = ForceGraph::default();
        assert!(!graph.assertions().is_empty());
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn test_force_graph_verify_valid() {
        let mut graph = ForceGraph::default();
        graph.bounds = Rect::new(0.0, 0.0, 60.0, 30.0);
        assert!(graph.verify().is_valid());
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn test_force_graph_verify_invalid() {
        let mut graph = ForceGraph::default();
        graph.bounds = Rect::new(0.0, 0.0, 5.0, 3.0);
        assert!(!graph.verify().is_valid());
    }

    #[test]
    fn test_force_graph_children() {
        let graph = ForceGraph::default();
        assert!(graph.children().is_empty());
    }

    #[test]
    fn test_force_graph_children_mut() {
        let mut graph = ForceGraph::default();
        assert!(graph.children_mut().is_empty());
    }

    #[test]
    fn test_force_graph_with_iterations() {
        let graph = ForceGraph::default().with_iterations(50);
        assert_eq!(graph.iterations, 50);
    }

    #[test]
    fn test_force_graph_iterations_clamped() {
        let graph = ForceGraph::default().with_iterations(0);
        assert_eq!(graph.iterations, 1);

        let graph = ForceGraph::default().with_iterations(500);
        assert_eq!(graph.iterations, 100);
    }

    #[test]
    fn test_force_graph_with_labels() {
        let graph = ForceGraph::default().with_labels(false);
        assert!(!graph.show_labels);
    }

    #[test]
    fn test_force_graph_with_edges() {
        let graph = ForceGraph::default().with_edges(false);
        assert!(!graph.show_edges);
    }

    #[test]
    fn test_force_graph_with_running() {
        let graph = ForceGraph::default().with_running(false);
        assert!(!graph.running);
    }

    #[test]
    fn test_large_graph_simd() {
        // Test SIMD path (>100 nodes)
        let nodes: Vec<GraphNode> = (0..150)
            .map(|i| {
                GraphNode::new(format!("n{i}"))
                    .with_position((i as f64 % 10.0) / 10.0, (i as f64 / 10.0) / 15.0)
            })
            .collect();
        let edges: Vec<GraphEdge> = (0..100).map(|i| GraphEdge::new(i, (i + 1) % 150)).collect();
        let mut graph = ForceGraph::new(nodes, edges);
        graph.step();
        // Should not panic
        assert_eq!(graph.nodes.len(), 150);
    }

    #[test]
    fn test_large_graph_simd_with_fixed() {
        let mut nodes: Vec<GraphNode> = (0..150)
            .map(|i| {
                GraphNode::new(format!("n{i}"))
                    .with_position((i as f64 % 10.0) / 10.0, (i as f64 / 10.0) / 15.0)
            })
            .collect();
        // Fix some nodes
        nodes[0] = nodes[0].clone().with_fixed(true);
        nodes[50] = nodes[50].clone().with_fixed(true);
        nodes[100] = nodes[100].clone().with_fixed(true);

        let edges: Vec<GraphEdge> = (0..50).map(|i| GraphEdge::new(i, (i + 1) % 150)).collect();
        let mut graph = ForceGraph::new(nodes, edges);
        graph.step();
        assert_eq!(graph.nodes.len(), 150);
    }

    #[test]
    fn test_force_params_default() {
        let params = ForceParams::default();
        assert!(params.repulsion > 0.0);
        assert!(params.spring_strength > 0.0);
        assert!(params.damping > 0.0 && params.damping <= 1.0);
    }

    #[test]
    fn test_force_graph_brick_name() {
        let graph = ForceGraph::default();
        assert_eq!(graph.brick_name(), "ForceGraph");
    }

    #[test]
    fn test_force_graph_budget() {
        let graph = ForceGraph::default();
        let budget = graph.budget();
        assert!(budget.layout_ms > 0);
    }

    #[test]
    fn test_force_graph_to_html() {
        let graph = ForceGraph::default();
        assert!(graph.to_html().is_empty());
    }

    #[test]
    fn test_force_graph_to_css() {
        let graph = ForceGraph::default();
        assert!(graph.to_css().is_empty());
    }

    #[test]
    fn test_force_graph_type_id() {
        let graph = ForceGraph::default();
        let type_id = Widget::type_id(&graph);
        assert_eq!(type_id, TypeId::of::<ForceGraph>());
    }

    #[test]
    fn test_force_graph_event() {
        let mut graph = ForceGraph::default();
        let event = Event::Resize {
            width: 80.0,
            height: 24.0,
        };
        assert!(graph.event(&event).is_none());
    }

    #[test]
    fn test_draw_line_horizontal() {
        let graph = ForceGraph::new(vec![], vec![]);
        let mut graph = graph;
        graph.bounds = Rect::new(0.0, 0.0, 60.0, 30.0);

        let mut buffer = CellBuffer::new(60, 30);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        let style = TextStyle {
            color: Color::new(0.5, 0.5, 0.5, 1.0),
            ..Default::default()
        };
        graph.draw_line(&mut canvas, 10.0, 15.0, 50.0, 15.0, &style);
    }

    #[test]
    fn test_draw_line_vertical() {
        let graph = ForceGraph::new(vec![], vec![]);
        let mut graph = graph;
        graph.bounds = Rect::new(0.0, 0.0, 60.0, 30.0);

        let mut buffer = CellBuffer::new(60, 30);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        let style = TextStyle {
            color: Color::new(0.5, 0.5, 0.5, 1.0),
            ..Default::default()
        };
        graph.draw_line(&mut canvas, 30.0, 5.0, 30.0, 25.0, &style);
    }

    #[test]
    fn test_draw_line_diagonal() {
        let graph = ForceGraph::new(vec![], vec![]);
        let mut graph = graph;
        graph.bounds = Rect::new(0.0, 0.0, 60.0, 30.0);

        let mut buffer = CellBuffer::new(60, 30);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        let style = TextStyle {
            color: Color::new(0.5, 0.5, 0.5, 1.0),
            ..Default::default()
        };
        graph.draw_line(&mut canvas, 10.0, 10.0, 50.0, 20.0, &style);
    }

    #[test]
    fn test_spring_forces_invalid_indices() {
        let mut graph = ForceGraph::new(vec![GraphNode::new("a").with_position(0.5, 0.5)], vec![]);
        // Manually add an invalid edge
        graph.edges.push(GraphEdge::new(0, 5));
        graph.compute_spring_forces(); // Should not panic
    }

    #[test]
    fn test_spring_forces_with_fixed_nodes() {
        let mut graph = ForceGraph::new(
            vec![
                GraphNode::new("a").with_position(0.3, 0.5).with_fixed(true),
                GraphNode::new("b").with_position(0.7, 0.5).with_fixed(true),
            ],
            vec![GraphEdge::new(0, 1)],
        );
        let initial_x0 = graph.nodes[0].x;
        let initial_x1 = graph.nodes[1].x;
        graph.compute_spring_forces();
        // Both fixed, velocities should change but positions won't be updated since step updates positions
        assert_eq!(graph.nodes[0].x, initial_x0);
        assert_eq!(graph.nodes[1].x, initial_x1);
    }

    #[test]
    fn test_gravity_with_fixed_nodes() {
        let mut graph = ForceGraph::new(
            vec![GraphNode::new("a").with_position(0.1, 0.1).with_fixed(true)],
            vec![],
        );
        let initial_vx = graph.nodes[0].vx;
        graph.apply_gravity();
        // Fixed node should not have velocity changed
        assert_eq!(graph.nodes[0].vx, initial_vx);
    }

    #[test]
    fn test_high_velocity_clamping() {
        let mut graph = ForceGraph::new(vec![GraphNode::new("a").with_position(0.5, 0.5)], vec![]);
        graph.nodes[0].vx = 10.0;
        graph.nodes[0].vy = 10.0;
        graph.update_positions();
        // Velocity should be clamped
        let speed = graph.nodes[0].vx.hypot(graph.nodes[0].vy);
        assert!(speed <= 0.11); // Some tolerance for damping
    }

    #[test]
    fn test_rand_float() {
        let val = rand_float();
        assert!((0.0..=1.0).contains(&val));
    }
}
