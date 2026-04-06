//! Time-series graph widget with multiple rendering modes.
//!
//! Supports three rendering modes for terminal compatibility:
//!
//! - **Braille**: Highest resolution using Unicode braille patterns (U+2800-28FF)
//! - **Block**: Medium resolution using block characters (▗▄▖▟▌▙█)
//! - **TTY**: ASCII-only for pure TTY environments (░▒█)
//!
//! # Performance
//!
//! - Rendering is O(width × height) (Falsification criterion #2)
//! - Double-buffered to prevent flicker

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::widgets::Widget;

/// Rendering mode for the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GraphMode {
    /// Braille patterns (U+2800-28FF) - highest resolution.
    #[default]
    Braille,
    /// Block characters (▗▄▖▟▌▙█) - medium resolution.
    Block,
    /// ASCII characters (░▒█) - TTY compatible.
    Tty,
}

/// A time-series graph widget.
#[derive(Debug, Clone)]
pub struct Graph<'a> {
    /// Data points to display (0.0 - 1.0 normalized).
    data: &'a [f64],
    /// Rendering mode.
    mode: GraphMode,
    /// Graph color.
    color: Color,
    /// Whether to invert the graph (for upload graphs).
    inverted: bool,
}

impl<'a> Graph<'a> {
    /// Creates a new graph with the given data.
    #[must_use]
    pub fn new(data: &'a [f64]) -> Self {
        Self { data, mode: GraphMode::default(), color: Color::Cyan, inverted: false }
    }

    /// Sets the rendering mode.
    #[must_use]
    pub fn mode(mut self, mode: GraphMode) -> Self {
        self.mode = mode;
        self
    }

    /// Sets the graph color.
    #[must_use]
    pub fn color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Sets whether the graph is inverted.
    #[must_use]
    pub fn inverted(mut self, inverted: bool) -> Self {
        self.inverted = inverted;
        self
    }

    /// Renders braille characters for the data.
    fn render_braille(&self, area: Rect, buf: &mut Buffer) {
        if self.data.is_empty() || area.width == 0 || area.height == 0 {
            return;
        }

        let width = area.width as usize;
        let height = area.height as usize;

        // Each braille character represents 2x4 dots
        let _dots_per_char_x = 2; // Only using left column currently
        let dots_per_char_y = 4;

        for x in 0..width {
            // Map x position to data index
            let data_idx = (x * self.data.len()) / width;
            let value = self.data.get(data_idx).copied().unwrap_or(0.0).clamp(0.0, 1.0);

            // Calculate the height in dots
            let max_dots = height * dots_per_char_y;
            let filled_dots = if self.inverted {
                ((1.0 - value) * max_dots as f64) as usize
            } else {
                (value * max_dots as f64) as usize
            };

            // Render each row
            for y in 0..height {
                let char_y = if self.inverted { y } else { height - 1 - y };
                let dot_start = y * dots_per_char_y;

                // Determine which dots in this character should be filled
                let mut pattern: u8 = 0;

                for dot in 0..dots_per_char_y {
                    let dot_pos = dot_start + dot;
                    let should_fill =
                        if self.inverted { dot_pos >= filled_dots } else { dot_pos < filled_dots };

                    if should_fill {
                        // Braille dot pattern (column 0)
                        // Dots are numbered: 1,2,3,7 in left column, 4,5,6,8 in right
                        let bit = match dot {
                            0 => 0x01, // dot 1
                            1 => 0x02, // dot 2
                            2 => 0x04, // dot 3
                            3 => 0x40, // dot 7
                            _ => 0,
                        };
                        pattern |= bit;
                    }
                }

                // Convert pattern to braille character (U+2800 base)
                let braille = char::from_u32(0x2800 + u32::from(pattern)).unwrap_or(' ');

                let cell_x = area.x + x as u16;
                let cell_y = area.y + char_y as u16;

                if cell_x < area.x + area.width && cell_y < area.y + area.height {
                    buf.set_string(
                        cell_x,
                        cell_y,
                        braille.to_string(),
                        Style::default().fg(self.color),
                    );
                }
            }
        }
    }

    /// Renders block characters for the data.
    fn render_block(&self, area: Rect, buf: &mut Buffer) {
        if self.data.is_empty() || area.width == 0 || area.height == 0 {
            return;
        }

        let width = area.width as usize;
        let height = area.height as usize;

        // Block characters for different fill levels
        let blocks = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

        for x in 0..width {
            let data_idx = (x * self.data.len()) / width;
            let value = self.data.get(data_idx).copied().unwrap_or(0.0).clamp(0.0, 1.0);

            // Full blocks to render
            let full_height = (value * height as f64) as usize;
            let partial = ((value * height as f64) - full_height as f64) * 8.0;
            let partial_idx = (partial as usize).min(8);

            for y in 0..height {
                let char_y = if self.inverted { y } else { height - 1 - y };
                let block_char = if y < full_height {
                    '█'
                } else if y == full_height && partial_idx > 0 {
                    blocks[partial_idx]
                } else {
                    ' '
                };

                let cell_x = area.x + x as u16;
                let cell_y = area.y + char_y as u16;

                if cell_x < area.x + area.width && cell_y < area.y + area.height {
                    buf.set_string(
                        cell_x,
                        cell_y,
                        block_char.to_string(),
                        Style::default().fg(self.color),
                    );
                }
            }
        }
    }

    /// Renders TTY-compatible ASCII characters.
    fn render_tty(&self, area: Rect, buf: &mut Buffer) {
        if self.data.is_empty() || area.width == 0 || area.height == 0 {
            return;
        }

        let width = area.width as usize;
        let height = area.height as usize;

        // TTY characters: space, light shade, medium shade, full block
        let shades = [' ', '░', '▒', '█'];

        for x in 0..width {
            let data_idx = (x * self.data.len()) / width;
            let value = self.data.get(data_idx).copied().unwrap_or(0.0).clamp(0.0, 1.0);

            let filled_height = (value * height as f64) as usize;

            for y in 0..height {
                let char_y = if self.inverted { y } else { height - 1 - y };
                let shade_char = if y < filled_height {
                    '█'
                } else if y == filled_height {
                    let partial = (value * height as f64) - filled_height as f64;
                    let shade_idx = (partial * 3.0) as usize;
                    shades[shade_idx.min(3)]
                } else {
                    ' '
                };

                let cell_x = area.x + x as u16;
                let cell_y = area.y + char_y as u16;

                if cell_x < area.x + area.width && cell_y < area.y + area.height {
                    buf.set_string(
                        cell_x,
                        cell_y,
                        shade_char.to_string(),
                        Style::default().fg(self.color),
                    );
                }
            }
        }
    }
}

impl Widget for Graph<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        match self.mode {
            GraphMode::Braille => self.render_braille(area, buf),
            GraphMode::Block => self.render_block(area, buf),
            GraphMode::Tty => self.render_tty(area, buf),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn create_test_terminal() -> Terminal<TestBackend> {
        let backend = TestBackend::new(80, 24);
        Terminal::new(backend).expect("Failed to create terminal")
    }

    #[test]
    fn test_graph_new() {
        let data = vec![0.5; 10];
        let graph = Graph::new(&data);

        assert_eq!(graph.mode, GraphMode::Braille);
        assert_eq!(graph.color, Color::Cyan);
        assert!(!graph.inverted);
    }

    #[test]
    fn test_graph_builder() {
        let data = vec![0.5; 10];
        let graph = Graph::new(&data).mode(GraphMode::Block).color(Color::Red).inverted(true);

        assert_eq!(graph.mode, GraphMode::Block);
        assert_eq!(graph.color, Color::Red);
        assert!(graph.inverted);
    }

    #[test]
    fn test_graph_braille_rendering() {
        let mut terminal = create_test_terminal();
        let data = vec![0.0, 0.5, 1.0, 0.5, 0.0];

        terminal
            .draw(|frame| {
                let graph = Graph::new(&data).mode(GraphMode::Braille);
                frame.render_widget(graph, frame.area());
            })
            .expect("Failed to draw");

        let buffer = terminal.backend().buffer();
        let content: String =
            buffer.content().iter().map(|c| c.symbol().chars().next().unwrap_or(' ')).collect();

        // Verify some braille characters are present
        assert!(
            content.chars().any(|c| ('\u{2800}'..='\u{28FF}').contains(&c)),
            "Should contain braille characters"
        );
    }

    #[test]
    fn test_graph_tty_no_unicode_extended() {
        let mut terminal = create_test_terminal();
        let data = vec![0.5; 10];

        terminal
            .draw(|frame| {
                let graph = Graph::new(&data).mode(GraphMode::Tty);
                frame.render_widget(graph, frame.area());
            })
            .expect("Failed to draw");

        let buffer = terminal.backend().buffer();
        let content: String =
            buffer.content().iter().map(|c| c.symbol().chars().next().unwrap_or(' ')).collect();

        // TTY mode should only use basic characters (space, shades, full block)
        // All characters should be in the basic set
        for c in content.chars() {
            assert!(
                c == ' ' || c == '░' || c == '▒' || c == '█',
                "TTY mode should only use basic shade characters, found: {c:?}"
            );
        }
    }

    #[test]
    fn test_graph_empty_data() {
        let mut terminal = create_test_terminal();
        let data: Vec<f64> = vec![];

        terminal
            .draw(|frame| {
                let graph = Graph::new(&data);
                frame.render_widget(graph, frame.area());
            })
            .expect("Should handle empty data without panic");
    }

    #[test]
    fn test_graph_single_value() {
        let mut terminal = create_test_terminal();
        let data = vec![0.75];

        terminal
            .draw(|frame| {
                let graph = Graph::new(&data);
                frame.render_widget(graph, frame.area());
            })
            .expect("Should handle single value");
    }

    #[test]
    fn test_graph_mode_default() {
        assert_eq!(GraphMode::default(), GraphMode::Braille);
    }

    #[test]
    fn test_graph_block_rendering() {
        let mut terminal = create_test_terminal();
        let data = vec![0.0, 0.25, 0.5, 0.75, 1.0];

        terminal
            .draw(|frame| {
                let graph = Graph::new(&data).mode(GraphMode::Block);
                frame.render_widget(graph, frame.area());
            })
            .expect("Failed to draw block graph");

        let buffer = terminal.backend().buffer();
        let content: String =
            buffer.content().iter().map(|c| c.symbol().chars().next().unwrap_or(' ')).collect();

        // Block mode should use block characters
        assert!(content.chars().any(|c| c == '█' || c == '▇' || c == '▆'));
    }

    #[test]
    fn test_graph_tty_rendering() {
        let mut terminal = create_test_terminal();
        let data = vec![0.1, 0.5, 0.9, 0.5, 0.1];

        terminal
            .draw(|frame| {
                let graph = Graph::new(&data).mode(GraphMode::Tty);
                frame.render_widget(graph, frame.area());
            })
            .expect("Failed to draw TTY graph");

        // Should complete without panic
    }

    #[test]
    fn test_graph_inverted_braille() {
        let mut terminal = create_test_terminal();
        let data = vec![0.3, 0.6, 0.9];

        terminal
            .draw(|frame| {
                let graph = Graph::new(&data).mode(GraphMode::Braille).inverted(true);
                frame.render_widget(graph, frame.area());
            })
            .expect("Failed to draw inverted braille graph");
    }

    #[test]
    fn test_graph_inverted_block() {
        let mut terminal = create_test_terminal();
        let data = vec![0.2, 0.4, 0.8];

        terminal
            .draw(|frame| {
                let graph = Graph::new(&data).mode(GraphMode::Block).inverted(true);
                frame.render_widget(graph, frame.area());
            })
            .expect("Failed to draw inverted block graph");
    }

    #[test]
    fn test_graph_inverted_tty() {
        let mut terminal = create_test_terminal();
        let data = vec![0.5, 0.5, 0.5];

        terminal
            .draw(|frame| {
                let graph = Graph::new(&data).mode(GraphMode::Tty).inverted(true);
                frame.render_widget(graph, frame.area());
            })
            .expect("Failed to draw inverted TTY graph");
    }

    #[test]
    fn test_graph_full_values() {
        let mut terminal = create_test_terminal();
        let data = vec![1.0; 20];

        terminal
            .draw(|frame| {
                let graph = Graph::new(&data);
                frame.render_widget(graph, frame.area());
            })
            .expect("Failed to draw full graph");
    }

    #[test]
    fn test_graph_zero_values() {
        let mut terminal = create_test_terminal();
        let data = vec![0.0; 20];

        terminal
            .draw(|frame| {
                let graph = Graph::new(&data);
                frame.render_widget(graph, frame.area());
            })
            .expect("Failed to draw zero graph");
    }

    #[test]
    fn test_graph_out_of_range_clamping() {
        let mut terminal = create_test_terminal();
        // Values outside 0-1 should be clamped
        let data = vec![-0.5, 1.5, 2.0, -1.0];

        terminal
            .draw(|frame| {
                let graph = Graph::new(&data);
                frame.render_widget(graph, frame.area());
            })
            .expect("Should handle out of range values");
    }

    #[test]
    fn test_graph_mode_clone_debug() {
        let mode = GraphMode::Block;
        let cloned = mode;
        assert_eq!(mode, cloned);

        let debug_str = format!("{:?}", GraphMode::Tty);
        assert!(debug_str.contains("Tty"));
    }

    #[test]
    fn test_graph_clone() {
        let data = vec![0.5; 5];
        let graph = Graph::new(&data).color(Color::Yellow);
        let cloned = graph.clone();

        assert_eq!(graph.color, cloned.color);
    }

    #[test]
    fn test_graph_various_colors() {
        let mut terminal = create_test_terminal();
        let data = vec![0.5; 10];

        for color in [Color::Red, Color::Green, Color::Blue, Color::Yellow] {
            terminal
                .draw(|frame| {
                    let graph = Graph::new(&data).color(color);
                    frame.render_widget(graph, frame.area());
                })
                .expect("Should render with different colors");
        }
    }
}
