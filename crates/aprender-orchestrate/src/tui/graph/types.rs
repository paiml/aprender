//! Core types for graph visualization
//!
//! Contains `NodeShape`, `NodeStatus`, `Position`, `Node`, and `Edge` types.

// ============================================================================
// GRAPH-001: Core Types
// ============================================================================

/// Maximum nodes for TUI rendering (Muri prevention per peer review #3)
pub const MAX_TUI_NODES: usize = 500;

/// Default visible nodes (Mieruka per peer review #9)
pub const DEFAULT_VISIBLE_NODES: usize = 20;

/// Node shape for accessibility (Respect for People per peer review #6)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NodeShape {
    /// Circle: default node
    #[default]
    Circle,
    /// Diamond: input/source
    Diamond,
    /// Square: transform/process
    Square,
    /// Triangle: output/sink
    Triangle,
    /// Star: highlighted/selected
    Star,
}

impl NodeShape {
    /// Get Unicode character for shape
    #[must_use]
    pub fn unicode(&self) -> char {
        match self {
            Self::Circle => '●',
            Self::Diamond => '◆',
            Self::Square => '■',
            Self::Triangle => '▲',
            Self::Star => '★',
        }
    }

    /// Get ASCII fallback character (Standardized Work per peer review #5)
    #[must_use]
    pub fn ascii(&self) -> char {
        match self {
            Self::Circle => 'o',
            Self::Diamond => '<',
            Self::Square => '#',
            Self::Triangle => '^',
            Self::Star => '*',
        }
    }
}

/// Node status with accessible visual encoding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NodeStatus {
    /// Healthy/success - green + circle
    #[default]
    Healthy,
    /// Warning/pending - yellow + triangle
    Warning,
    /// Error/critical - red + diamond (not just red per peer review #6)
    Error,
    /// Info/selected - cyan + star
    Info,
    /// Neutral/unknown - gray + square
    Neutral,
}

impl NodeStatus {
    /// Get shape for status (accessibility - not just color)
    #[must_use]
    pub fn shape(&self) -> NodeShape {
        match self {
            Self::Healthy => NodeShape::Circle,
            Self::Warning => NodeShape::Triangle,
            Self::Error => NodeShape::Diamond,
            Self::Info => NodeShape::Star,
            Self::Neutral => NodeShape::Square,
        }
    }

    /// Get ANSI color code
    #[must_use]
    pub fn color_code(&self) -> &'static str {
        match self {
            Self::Healthy => "\x1b[32m", // Green
            Self::Warning => "\x1b[33m", // Yellow
            Self::Error => "\x1b[31m",   // Red
            Self::Info => "\x1b[36m",    // Cyan
            Self::Neutral => "\x1b[90m", // Gray
        }
    }
}

/// 2D position for layout
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Position {
    pub x: f32,
    pub y: f32,
}

impl Position {
    /// Create new position
    #[must_use]
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Euclidean distance to another position
    #[must_use]
    pub fn distance(&self, other: &Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

/// Node in the graph
#[derive(Debug, Clone)]
pub struct Node<T> {
    /// Node identifier
    pub id: String,
    /// Node data
    pub data: T,
    /// Visual status
    pub status: NodeStatus,
    /// Display label
    pub label: Option<String>,
    /// Computed position (after layout)
    pub position: Position,
    /// Node importance (for Mieruka filtering)
    pub importance: f32,
}

impl<T> Node<T> {
    /// Create new node
    pub fn new(id: impl Into<String>, data: T) -> Self {
        Self {
            id: id.into(),
            data,
            status: NodeStatus::default(),
            label: None,
            position: Position::default(),
            importance: 1.0,
        }
    }

    /// Set status
    #[must_use]
    pub fn with_status(mut self, status: NodeStatus) -> Self {
        self.status = status;
        self
    }

    /// Set label
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set importance
    #[must_use]
    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance;
        self
    }
}

/// Edge in the graph
#[derive(Debug, Clone)]
pub struct Edge<E> {
    /// Source node ID
    pub from: String,
    /// Target node ID
    pub to: String,
    /// Edge data
    pub data: E,
    /// Edge weight (for layout algorithms and visualization)
    pub weight: f32,
    /// Optional edge label (Neo4j/Cytoscape pattern)
    pub label: Option<String>,
}

impl<E> Edge<E> {
    /// Create new edge
    pub fn new(from: impl Into<String>, to: impl Into<String>, data: E) -> Self {
        Self { from: from.into(), to: to.into(), data, weight: 1.0, label: None }
    }

    /// Set weight
    #[must_use]
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Set label (Neo4j pattern)
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // NodeShape Tests
    // =========================================================================

    #[test]
    fn test_node_shape_default() {
        assert_eq!(NodeShape::default(), NodeShape::Circle);
    }

    #[test]
    fn test_node_shape_unicode() {
        assert_eq!(NodeShape::Circle.unicode(), '●');
        assert_eq!(NodeShape::Diamond.unicode(), '◆');
        assert_eq!(NodeShape::Square.unicode(), '■');
        assert_eq!(NodeShape::Triangle.unicode(), '▲');
        assert_eq!(NodeShape::Star.unicode(), '★');
    }

    #[test]
    fn test_node_shape_ascii() {
        assert_eq!(NodeShape::Circle.ascii(), 'o');
        assert_eq!(NodeShape::Diamond.ascii(), '<');
        assert_eq!(NodeShape::Square.ascii(), '#');
        assert_eq!(NodeShape::Triangle.ascii(), '^');
        assert_eq!(NodeShape::Star.ascii(), '*');
    }

    // =========================================================================
    // NodeStatus Tests
    // =========================================================================

    #[test]
    fn test_node_status_default() {
        assert_eq!(NodeStatus::default(), NodeStatus::Healthy);
    }

    #[test]
    fn test_node_status_shape() {
        assert_eq!(NodeStatus::Healthy.shape(), NodeShape::Circle);
        assert_eq!(NodeStatus::Warning.shape(), NodeShape::Triangle);
        assert_eq!(NodeStatus::Error.shape(), NodeShape::Diamond);
        assert_eq!(NodeStatus::Info.shape(), NodeShape::Star);
        assert_eq!(NodeStatus::Neutral.shape(), NodeShape::Square);
    }

    #[test]
    fn test_node_status_color_code() {
        assert!(NodeStatus::Healthy.color_code().contains("32m"));
        assert!(NodeStatus::Warning.color_code().contains("33m"));
        assert!(NodeStatus::Error.color_code().contains("31m"));
        assert!(NodeStatus::Info.color_code().contains("36m"));
        assert!(NodeStatus::Neutral.color_code().contains("90m"));
    }

    // =========================================================================
    // Position Tests
    // =========================================================================

    #[test]
    fn test_position_default() {
        let pos = Position::default();
        assert_eq!(pos.x, 0.0);
        assert_eq!(pos.y, 0.0);
    }

    #[test]
    fn test_position_new() {
        let pos = Position::new(10.0, 20.0);
        assert_eq!(pos.x, 10.0);
        assert_eq!(pos.y, 20.0);
    }

    #[test]
    fn test_position_distance() {
        let p1 = Position::new(0.0, 0.0);
        let p2 = Position::new(3.0, 4.0);
        assert!((p1.distance(&p2) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_position_distance_to_self() {
        let p = Position::new(5.0, 5.0);
        assert!((p.distance(&p)).abs() < 0.001);
    }

    // =========================================================================
    // Node Tests
    // =========================================================================

    #[test]
    fn test_node_new() {
        let node = Node::new("test", 42);
        assert_eq!(node.id, "test");
        assert_eq!(node.data, 42);
        assert_eq!(node.status, NodeStatus::Healthy);
        assert!(node.label.is_none());
        assert_eq!(node.importance, 1.0);
    }

    #[test]
    fn test_node_with_status() {
        let node = Node::new("test", ()).with_status(NodeStatus::Error);
        assert_eq!(node.status, NodeStatus::Error);
    }

    #[test]
    fn test_node_with_label() {
        let node = Node::new("test", ()).with_label("My Label");
        assert_eq!(node.label, Some("My Label".to_string()));
    }

    #[test]
    fn test_node_with_importance() {
        let node = Node::new("test", ()).with_importance(0.5);
        assert_eq!(node.importance, 0.5);
    }

    #[test]
    fn test_node_builder_chain() {
        let node = Node::new("test", "data")
            .with_status(NodeStatus::Warning)
            .with_label("Label")
            .with_importance(0.75);
        assert_eq!(node.status, NodeStatus::Warning);
        assert_eq!(node.label, Some("Label".to_string()));
        assert_eq!(node.importance, 0.75);
    }

    // =========================================================================
    // Edge Tests
    // =========================================================================

    #[test]
    fn test_edge_new() {
        let edge = Edge::new("A", "B", 100);
        assert_eq!(edge.from, "A");
        assert_eq!(edge.to, "B");
        assert_eq!(edge.data, 100);
        assert_eq!(edge.weight, 1.0);
        assert!(edge.label.is_none());
    }

    #[test]
    fn test_edge_with_weight() {
        let edge = Edge::new("A", "B", ()).with_weight(2.5);
        assert_eq!(edge.weight, 2.5);
    }

    #[test]
    fn test_edge_with_label() {
        let edge = Edge::new("A", "B", ()).with_label("depends_on");
        assert_eq!(edge.label, Some("depends_on".to_string()));
    }

    #[test]
    fn test_edge_builder_chain() {
        let edge =
            Edge::new("source", "target", "edge_data").with_weight(3.0).with_label("relation");
        assert_eq!(edge.from, "source");
        assert_eq!(edge.to, "target");
        assert_eq!(edge.weight, 3.0);
        assert_eq!(edge.label, Some("relation".to_string()));
    }

    // =========================================================================
    // Constants Tests
    // =========================================================================

    #[test]
    fn test_constants() {
        assert_eq!(MAX_TUI_NODES, 500);
        assert_eq!(DEFAULT_VISIBLE_NODES, 20);
    }
}
