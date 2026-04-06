//! Heatmap widget for grid visualization.
//!
//! Displays a grid of values with color-coded cells.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::widgets::Widget;

/// A cell in the heatmap.
#[derive(Debug, Clone)]
pub struct HeatmapCell {
    /// Cell value (0.0 - 1.0 normalized).
    pub value: f64,
    /// Optional label.
    pub label: Option<String>,
}

impl HeatmapCell {
    /// Creates a new cell with the given value.
    #[must_use]
    pub fn new(value: f64) -> Self {
        Self { value: value.clamp(0.0, 1.0), label: None }
    }

    /// Creates a cell with a label.
    #[must_use]
    pub fn with_label(value: f64, label: impl Into<String>) -> Self {
        Self { value: value.clamp(0.0, 1.0), label: Some(label.into()) }
    }
}

/// Color palette for the heatmap.
#[derive(Debug, Clone)]
pub struct HeatmapPalette {
    /// Colors from low to high.
    colors: Vec<Color>,
}

impl HeatmapPalette {
    /// Creates a palette with the given colors.
    #[must_use]
    pub fn new(colors: Vec<Color>) -> Self {
        Self {
            colors: if colors.is_empty() {
                vec![Color::Blue, Color::Cyan, Color::Green, Color::Yellow, Color::Red]
            } else {
                colors
            },
        }
    }

    /// Returns a cool-to-warm palette.
    #[must_use]
    pub fn cool_warm() -> Self {
        Self::new(vec![Color::Blue, Color::Cyan, Color::Green, Color::Yellow, Color::Red])
    }

    /// Returns a grayscale palette.
    #[must_use]
    pub fn grayscale() -> Self {
        Self::new(vec![
            Color::Rgb(32, 32, 32),
            Color::Rgb(64, 64, 64),
            Color::Rgb(128, 128, 128),
            Color::Rgb(192, 192, 192),
            Color::Rgb(255, 255, 255),
        ])
    }

    /// Returns a temperature palette (cyan to red).
    #[must_use]
    pub fn temperature() -> Self {
        Self::new(vec![
            Color::Cyan,
            Color::Green,
            Color::Yellow,
            Color::Rgb(255, 165, 0), // Orange
            Color::Red,
        ])
    }

    /// Gets the color for a value (0.0 - 1.0).
    #[must_use]
    pub fn color_for(&self, value: f64) -> Color {
        if self.colors.is_empty() {
            return Color::White;
        }

        let value = value.clamp(0.0, 1.0);

        if self.colors.len() == 1 {
            return self.colors[0];
        }

        // Find the two colors to interpolate between
        let scaled = value * (self.colors.len() - 1) as f64;
        let idx = (scaled as usize).min(self.colors.len() - 2);
        let t = scaled - idx as f64;

        // For simple terminals, just pick the nearest color
        if t < 0.5 {
            self.colors[idx]
        } else {
            self.colors[idx + 1]
        }
    }
}

impl Default for HeatmapPalette {
    fn default() -> Self {
        Self::cool_warm()
    }
}

/// A heatmap widget for grid visualization.
#[derive(Debug, Clone)]
pub struct Heatmap {
    /// Grid of cells (row-major).
    cells: Vec<Vec<HeatmapCell>>,
    /// Color palette.
    palette: HeatmapPalette,
    /// Show cell labels.
    show_labels: bool,
    /// Cell width in characters.
    cell_width: u16,
    /// Cell height in characters.
    cell_height: u16,
    /// Title.
    title: Option<String>,
}

impl Heatmap {
    /// Creates a new heatmap with the given cells.
    #[must_use]
    pub fn new(cells: Vec<Vec<HeatmapCell>>) -> Self {
        Self {
            cells,
            palette: HeatmapPalette::default(),
            show_labels: true,
            cell_width: 4,
            cell_height: 2,
            title: None,
        }
    }

    /// Creates a heatmap from a 2D array of values.
    #[must_use]
    pub fn from_values(values: &[&[f64]]) -> Self {
        let cells =
            values.iter().map(|row| row.iter().map(|&v| HeatmapCell::new(v)).collect()).collect();

        Self::new(cells)
    }

    /// Sets the color palette.
    #[must_use]
    pub fn palette(mut self, palette: HeatmapPalette) -> Self {
        self.palette = palette;
        self
    }

    /// Sets whether to show cell labels.
    #[must_use]
    pub fn show_labels(mut self, show: bool) -> Self {
        self.show_labels = show;
        self
    }

    /// Sets the cell dimensions.
    #[must_use]
    pub fn cell_size(mut self, width: u16, height: u16) -> Self {
        self.cell_width = width.max(2);
        self.cell_height = height.max(1);
        self
    }

    /// Sets the title.
    #[must_use]
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Returns the number of rows.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.cells.len()
    }

    /// Returns the number of columns.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cells.first().map_or(0, std::vec::Vec::len)
    }

    /// Gets a cell at the given position.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> Option<&HeatmapCell> {
        self.cells.get(row).and_then(|r| r.get(col))
    }

    /// Renders a single cell.
    fn render_cell(&self, cell: &HeatmapCell, x: u16, y: u16, buf: &mut Buffer) {
        let color = self.palette.color_for(cell.value);

        // Fill cell with color
        for dy in 0..self.cell_height {
            for dx in 0..self.cell_width {
                buf.set_string(x + dx, y + dy, "█", Style::default().fg(color));
            }
        }

        // Add label if enabled and available
        if self.show_labels {
            let label = cell.label.as_deref().map_or_else(
                || format!("{:.0}", cell.value * 100.0),
                std::string::ToString::to_string,
            );

            let label_len = label.len() as u16;
            if label_len <= self.cell_width {
                let label_x = x + (self.cell_width - label_len) / 2;
                let label_y = y + self.cell_height / 2;

                // Use contrasting color for text
                let text_color = if cell.value > 0.5 { Color::Black } else { Color::White };

                buf.set_string(label_x, label_y, &label, Style::default().fg(text_color));
            }
        }
    }
}

impl Widget for Heatmap {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.width < 2 || area.height < 1 || self.cells.is_empty() {
            return;
        }

        let mut y = area.y;

        // Render title
        if let Some(ref title) = self.title {
            buf.set_string(area.x, y, title, Style::default().fg(Color::White));
            y += 1;
        }

        // Calculate how many cells fit
        let cols = ((area.width) / self.cell_width) as usize;
        let rows = ((area.height.saturating_sub(y - area.y)) / self.cell_height) as usize;

        // Render cells
        for (row_idx, row) in self.cells.iter().enumerate().take(rows) {
            let cell_y = y + (row_idx as u16) * self.cell_height;

            for (col_idx, cell) in row.iter().enumerate().take(cols) {
                let cell_x = area.x + (col_idx as u16) * self.cell_width;

                if cell_x + self.cell_width <= area.x + area.width
                    && cell_y + self.cell_height <= area.y + area.height
                {
                    self.render_cell(cell, cell_x, cell_y, buf);
                }
            }
        }
    }
}

// ============================================================================
// Tests (TDD - Written First)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn create_test_terminal() -> Terminal<TestBackend> {
        let backend = TestBackend::new(40, 20);
        Terminal::new(backend).expect("Failed to create terminal")
    }

    #[test]
    fn test_heatmap_cell_new() {
        let cell = HeatmapCell::new(0.5);
        assert!((cell.value - 0.5).abs() < 0.01);
        assert!(cell.label.is_none());
    }

    #[test]
    fn test_heatmap_cell_with_label() {
        let cell = HeatmapCell::with_label(0.75, "Core 0");
        assert!((cell.value - 0.75).abs() < 0.01);
        assert_eq!(cell.label.as_deref(), Some("Core 0"));
    }

    #[test]
    fn test_heatmap_cell_clamping() {
        let cell_low = HeatmapCell::new(-0.5);
        assert!((cell_low.value - 0.0).abs() < 0.01);

        let cell_high = HeatmapCell::new(1.5);
        assert!((cell_high.value - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_heatmap_palette_default() {
        let palette = HeatmapPalette::default();
        assert!(!palette.colors.is_empty());
    }

    #[test]
    fn test_heatmap_palette_color_for() {
        let palette = HeatmapPalette::cool_warm();

        // Low values should be cool colors
        let low_color = palette.color_for(0.0);
        assert_eq!(low_color, Color::Blue);

        // High values should be warm colors
        let high_color = palette.color_for(1.0);
        assert_eq!(high_color, Color::Red);
    }

    #[test]
    fn test_heatmap_palette_temperature() {
        let palette = HeatmapPalette::temperature();
        let low = palette.color_for(0.0);
        assert_eq!(low, Color::Cyan);
    }

    #[test]
    fn test_heatmap_palette_grayscale() {
        let palette = HeatmapPalette::grayscale();
        assert_eq!(palette.colors.len(), 5);
    }

    #[test]
    fn test_heatmap_new() {
        let cells = vec![
            vec![HeatmapCell::new(0.1), HeatmapCell::new(0.2)],
            vec![HeatmapCell::new(0.3), HeatmapCell::new(0.4)],
        ];

        let heatmap = Heatmap::new(cells);
        assert_eq!(heatmap.rows(), 2);
        assert_eq!(heatmap.cols(), 2);
    }

    #[test]
    fn test_heatmap_from_values() {
        let values: &[&[f64]] = &[&[0.1, 0.2, 0.3], &[0.4, 0.5, 0.6]];

        let heatmap = Heatmap::from_values(values);
        assert_eq!(heatmap.rows(), 2);
        assert_eq!(heatmap.cols(), 3);
    }

    #[test]
    fn test_heatmap_builder() {
        let heatmap = Heatmap::from_values(&[&[0.5]])
            .palette(HeatmapPalette::temperature())
            .show_labels(false)
            .cell_size(6, 3)
            .title("Test");

        assert!(!heatmap.show_labels);
        assert_eq!(heatmap.cell_width, 6);
        assert_eq!(heatmap.cell_height, 3);
        assert_eq!(heatmap.title.as_deref(), Some("Test"));
    }

    #[test]
    fn test_heatmap_get() {
        let heatmap = Heatmap::from_values(&[&[0.1, 0.2], &[0.3, 0.4]]);

        assert!(heatmap.get(0, 0).is_some());
        assert!((heatmap.get(0, 0).expect("key should exist").value - 0.1).abs() < 0.01);
        assert!((heatmap.get(1, 1).expect("key should exist").value - 0.4).abs() < 0.01);
        assert!(heatmap.get(5, 5).is_none());
    }

    #[test]
    fn test_heatmap_render() {
        let mut terminal = create_test_terminal();

        terminal
            .draw(|frame| {
                let heatmap =
                    Heatmap::from_values(&[&[0.0, 0.25, 0.5], &[0.75, 1.0, 0.5]]).title("Temps");

                frame.render_widget(heatmap, frame.area());
            })
            .expect("Failed to draw");

        // Just verify it doesn't panic
    }

    #[test]
    fn test_heatmap_render_empty() {
        let mut terminal = create_test_terminal();

        terminal
            .draw(|frame| {
                let heatmap = Heatmap::new(vec![]);
                frame.render_widget(heatmap, frame.area());
            })
            .expect("Should handle empty heatmap");
    }

    #[test]
    fn test_heatmap_render_single_cell() {
        let mut terminal = create_test_terminal();

        terminal
            .draw(|frame| {
                let heatmap = Heatmap::from_values(&[&[0.5]]);
                frame.render_widget(heatmap, frame.area());
            })
            .expect("Should handle single cell");
    }

    #[test]
    fn test_heatmap_render_with_labels() {
        let mut terminal = create_test_terminal();

        terminal
            .draw(|frame| {
                let cells = vec![vec![
                    HeatmapCell::with_label(0.3, "C0"),
                    HeatmapCell::with_label(0.7, "C1"),
                ]];

                let heatmap = Heatmap::new(cells).show_labels(true);
                frame.render_widget(heatmap, frame.area());
            })
            .expect("Should render with labels");
    }

    #[test]
    fn test_heatmap_palette_empty_defaults() {
        // Empty colors should use default palette
        let palette = HeatmapPalette::new(vec![]);
        assert_eq!(palette.colors.len(), 5);
        assert_eq!(palette.colors[0], Color::Blue);
        assert_eq!(palette.colors[4], Color::Red);
    }

    #[test]
    fn test_heatmap_palette_color_for_empty() {
        // Create palette then manually empty it
        let mut palette = HeatmapPalette::new(vec![Color::Red]);
        palette.colors.clear();
        // Should return White for empty palette
        assert_eq!(palette.color_for(0.5), Color::White);
    }

    #[test]
    fn test_heatmap_palette_color_for_single() {
        let palette = HeatmapPalette::new(vec![Color::Magenta]);
        // Single color should always return that color
        assert_eq!(palette.color_for(0.0), Color::Magenta);
        assert_eq!(palette.color_for(0.5), Color::Magenta);
        assert_eq!(palette.color_for(1.0), Color::Magenta);
    }

    #[test]
    fn test_heatmap_render_larger_with_labels() {
        // Render with larger cells to ensure label text_color branch is hit
        let backend = TestBackend::new(60, 15);
        let mut terminal = Terminal::new(backend).expect("operation should succeed");

        terminal
            .draw(|frame| {
                let cells = vec![
                    vec![HeatmapCell::with_label(0.1, "Low"), HeatmapCell::with_label(0.9, "High")],
                    vec![HeatmapCell::with_label(0.5, "Mid"), HeatmapCell::with_label(0.3, "Med")],
                ];
                let heatmap = Heatmap::new(cells).show_labels(true).cell_size(12, 4);
                frame.render_widget(heatmap, frame.area());
            })
            .expect("Should render larger heatmap with labels");
    }
}
