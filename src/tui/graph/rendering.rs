//! Graph rendering for TUI
//!
//! Contains `RenderMode`, `RenderedGraph`, and `GraphRenderer`.

use super::graph_core::Graph;
use super::types::{Node, Position};

// ============================================================================
// GRAPH-004: TUI Rendering
// ============================================================================

/// Render mode for terminal compatibility
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RenderMode {
    /// Unicode with colors (default)
    #[default]
    Unicode,
    /// ASCII fallback for legacy terminals (per peer review #5)
    Ascii,
    /// Plain text without colors
    Plain,
}

/// Rendered graph as string buffer
#[derive(Debug, Clone)]
pub struct RenderedGraph {
    /// Width in characters
    pub width: usize,
    /// Height in characters
    pub height: usize,
    /// Character buffer
    pub buffer: Vec<Vec<char>>,
    /// Color buffer (ANSI codes per cell)
    pub colors: Vec<Vec<Option<&'static str>>>,
}

impl RenderedGraph {
    /// Create new render buffer
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            buffer: vec![vec![' '; width]; height],
            colors: vec![vec![None; width]; height],
        }
    }

    /// Set character at position
    pub fn set(&mut self, x: usize, y: usize, ch: char, color: Option<&'static str>) {
        if x < self.width && y < self.height {
            self.buffer[y][x] = ch;
            self.colors[y][x] = color;
        }
    }

    /// Render to string with ANSI colors
    #[must_use]
    pub fn to_string_colored(&self) -> String {
        let mut result = String::new();
        for y in 0..self.height {
            for x in 0..self.width {
                if let Some(color) = self.colors[y][x] {
                    result.push_str(color);
                    result.push(self.buffer[y][x]);
                    result.push_str("\x1b[0m");
                } else {
                    result.push(self.buffer[y][x]);
                }
            }
            result.push('\n');
        }
        result
    }

    /// Render to plain string (no colors)
    #[must_use]
    pub fn to_string_plain(&self) -> String {
        self.buffer.iter().map(|row| row.iter().collect::<String>()).collect::<Vec<_>>().join("\n")
    }
}

/// Graph renderer
pub struct GraphRenderer {
    /// Render mode
    pub mode: RenderMode,
    /// Show labels
    pub show_labels: bool,
    /// Show edges
    pub show_edges: bool,
}

impl Default for GraphRenderer {
    fn default() -> Self {
        Self { mode: RenderMode::Unicode, show_labels: true, show_edges: true }
    }
}

impl GraphRenderer {
    /// Create new renderer
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set render mode
    #[must_use]
    pub fn with_mode(mut self, mode: RenderMode) -> Self {
        self.mode = mode;
        self
    }

    /// Render graph to buffer
    pub fn render<N, E>(&self, graph: &Graph<N, E>, width: usize, height: usize) -> RenderedGraph {
        let mut output = RenderedGraph::new(width, height);

        // Draw edges first (underneath nodes)
        if self.show_edges {
            for edge in graph.edges() {
                if let (Some(from), Some(to)) =
                    (graph.get_node(&edge.from), graph.get_node(&edge.to))
                {
                    self.draw_edge(&mut output, &from.position, &to.position, width, height);
                }
            }
        }

        // Draw nodes
        for node in graph.nodes() {
            self.draw_node(&mut output, node, width, height);
        }

        output
    }

    fn draw_node<N>(
        &self,
        output: &mut RenderedGraph,
        node: &Node<N>,
        width: usize,
        height: usize,
    ) {
        let x = (node.position.x / 80.0 * width as f32) as usize;
        let y = (node.position.y / 24.0 * height as f32) as usize;

        if x < width && y < height {
            let ch = match self.mode {
                RenderMode::Unicode => node.status.shape().unicode(),
                RenderMode::Ascii | RenderMode::Plain => node.status.shape().ascii(),
            };

            let color = match self.mode {
                RenderMode::Unicode | RenderMode::Ascii => Some(node.status.color_code()),
                RenderMode::Plain => None,
            };

            output.set(x, y, ch, color);

            // Draw label if enabled
            if self.show_labels {
                if let Some(ref label) = node.label {
                    let label_start = x.saturating_add(2);
                    for (i, c) in label.chars().take(10).enumerate() {
                        if label_start + i < width {
                            output.set(label_start + i, y, c, color);
                        }
                    }
                }
            }
        }
    }

    fn draw_edge(
        &self,
        output: &mut RenderedGraph,
        from: &Position,
        to: &Position,
        width: usize,
        height: usize,
    ) {
        let x1 = (from.x / 80.0 * width as f32) as i32;
        let y1 = (from.y / 24.0 * height as f32) as i32;
        let x2 = (to.x / 80.0 * width as f32) as i32;
        let y2 = (to.y / 24.0 * height as f32) as i32;

        // Bresenham's line algorithm
        let dx = (x2 - x1).abs();
        let dy = (y2 - y1).abs();
        let sx = if x1 < x2 { 1 } else { -1 };
        let sy = if y1 < y2 { 1 } else { -1 };
        let mut err = dx - dy;

        let mut x = x1;
        let mut y = y1;

        let edge_char = match self.mode {
            RenderMode::Unicode => '·',
            RenderMode::Ascii | RenderMode::Plain => '.',
        };

        while x != x2 || y != y2 {
            if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
                // Don't overwrite nodes
                if output.buffer[y as usize][x as usize] == ' ' {
                    output.set(x as usize, y as usize, edge_char, Some("\x1b[90m"));
                }
            }

            let e2 = 2 * err;
            if e2 > -dy {
                err -= dy;
                x += sx;
            }
            if e2 < dx {
                err += dx;
                y += sy;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_mode_default() {
        assert_eq!(RenderMode::default(), RenderMode::Unicode);
    }

    #[test]
    fn test_render_mode_equality() {
        assert_eq!(RenderMode::Unicode, RenderMode::Unicode);
        assert_eq!(RenderMode::Ascii, RenderMode::Ascii);
        assert_eq!(RenderMode::Plain, RenderMode::Plain);
        assert_ne!(RenderMode::Unicode, RenderMode::Ascii);
    }

    #[test]
    fn test_rendered_graph_new() {
        let graph = RenderedGraph::new(80, 24);
        assert_eq!(graph.width, 80);
        assert_eq!(graph.height, 24);
        assert_eq!(graph.buffer.len(), 24);
        assert_eq!(graph.buffer[0].len(), 80);
        assert_eq!(graph.colors.len(), 24);
    }

    #[test]
    fn test_rendered_graph_set() {
        let mut graph = RenderedGraph::new(10, 10);
        graph.set(5, 5, 'X', Some("\x1b[32m"));
        assert_eq!(graph.buffer[5][5], 'X');
        assert_eq!(graph.colors[5][5], Some("\x1b[32m"));
    }

    #[test]
    fn test_rendered_graph_set_out_of_bounds() {
        let mut graph = RenderedGraph::new(10, 10);
        graph.set(15, 15, 'X', None);
        // Should not crash, buffer unchanged
        assert_eq!(graph.buffer[0][0], ' ');
    }

    #[test]
    fn test_rendered_graph_to_string_plain() {
        let mut graph = RenderedGraph::new(5, 3);
        graph.set(2, 1, '*', None);
        let output = graph.to_string_plain();
        assert!(output.contains('*'));
    }

    #[test]
    fn test_rendered_graph_to_string_colored() {
        let mut graph = RenderedGraph::new(5, 3);
        graph.set(2, 1, '*', Some("\x1b[32m"));
        let output = graph.to_string_colored();
        assert!(output.contains("\x1b[32m"));
        assert!(output.contains("\x1b[0m"));
    }

    #[test]
    fn test_graph_renderer_default() {
        let renderer = GraphRenderer::default();
        assert_eq!(renderer.mode, RenderMode::Unicode);
        assert!(renderer.show_labels);
        assert!(renderer.show_edges);
    }

    #[test]
    fn test_graph_renderer_new() {
        let renderer = GraphRenderer::new();
        assert_eq!(renderer.mode, RenderMode::Unicode);
    }

    #[test]
    fn test_graph_renderer_with_mode() {
        let renderer = GraphRenderer::new().with_mode(RenderMode::Ascii);
        assert_eq!(renderer.mode, RenderMode::Ascii);
    }

    #[test]
    fn test_graph_renderer_render_empty_graph() {
        let graph: Graph<(), ()> = Graph::new();
        let renderer = GraphRenderer::new();
        let output = renderer.render(&graph, 40, 12);
        assert_eq!(output.width, 40);
        assert_eq!(output.height, 12);
    }
}
