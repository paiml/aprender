//! TUI (Terminal User Interface) Module
//!
//! Provides terminal-based visualization components for the batuta stack.
//!
//! ## Modules
//!
//! - `graph`: Core graph data structures
//! - `graph_analytics`: PageRank, community detection, centrality metrics
//! - `graph_layout`: Layout algorithms for graph visualization
//!
//! ## Design Principles
//!
//! Follows Toyota Way principles:
//! - **Mieruka**: Visual management for instant status recognition
//! - **Jidoka**: Built-in quality via comprehensive tests
//! - **Respect for People**: Accessibility via shapes (not just colors)

pub mod graph;
pub mod graph_analytics;
pub mod graph_layout;

pub use graph::{
    Edge, Graph, GraphRenderer, Node, NodeShape, NodeStatus, Position, RenderMode, RenderedGraph,
    DEFAULT_VISIBLE_NODES, MAX_TUI_NODES,
};
pub use graph_analytics::{GraphAnalytics, GraphAnalyticsExt, COMMUNITY_COLORS};
pub use graph_layout::{LayoutAlgorithm, LayoutConfig, LayoutEngine};
