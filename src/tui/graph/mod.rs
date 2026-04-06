//! Graph TUI Visualization Module
//!
//! Terminal-based graph visualization with Toyota Way principles.
//!
//! ## Design Principles (per spec docs/batuta-graph-viz-tui-spec.md)
//!
//! - **Genchi Genbutsu**: Graph data at source, no duplication
//! - **Jidoka**: Built-in quality via layout convergence tests
//! - **Heijunka**: Single layout engine, multiple output backends
//! - **Mieruka**: Semantic colors for instant status recognition
//! - **Respect for People**: Accessibility via shapes (not just color)
//!
//! ## Academic References
//!
//! - Fruchterman & Reingold (1991) - Force-directed layout
//! - Kamada & Kawai (1989) - Spring-based layout
//! - Sugiyama et al. (1981) - Hierarchical DAG layout
//!
//! ## Performance Constraints (Muri Prevention)
//!
//! - Hard limit: 500 nodes maximum for TUI rendering
//! - Default visible: Top 20 nodes by centrality (Mieruka)

mod filtering;
mod graph_core;
mod rendering;
mod types;

#[cfg(test)]
mod tests_analytics;
#[cfg(test)]
mod tests_core;
#[cfg(test)]
mod tests_filtering;
#[cfg(test)]
mod tests_layout;

// Re-export all public types from types module
pub use types::{
    Edge, Node, NodeShape, NodeStatus, Position, DEFAULT_VISIBLE_NODES, MAX_TUI_NODES,
};

// Re-export Graph from graph_core module
pub use graph_core::Graph;

// Re-export rendering types
pub use rendering::{GraphRenderer, RenderMode, RenderedGraph};

// Re-export layout types from graph_layout module (sibling module)
pub use super::graph_layout::{LayoutAlgorithm, LayoutConfig, LayoutEngine};

// Re-export analytics types from graph_analytics module (sibling module)
pub use super::graph_analytics::{GraphAnalytics, GraphAnalyticsExt, COMMUNITY_COLORS};
