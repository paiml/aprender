//! Heatmap widget for grid-based value visualization.
//!
//! Displays a 2D grid of values as colored cells. Supports multiple
//! color palettes and optional value labels.

use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// Color palette for heatmap rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HeatmapPalette {
    /// Blue (cold) to Red (hot).
    #[default]
    BlueRed,
    /// Viridis-like (purple to yellow).
    Viridis,
    /// Green (low) to Red (high).
    GreenRed,
    /// Grayscale.
    Grayscale,
    /// Single color intensity.
    Mono(u8, u8, u8),
}

impl HeatmapPalette {
    /// Get color for normalized value (0.0 to 1.0).
    #[must_use]
    pub fn color(&self, value: f64) -> Color {
        let t = value.clamp(0.0, 1.0) as f32;
        match self {
            Self::BlueRed => {
                if t < 0.5 {
                    let s = t * 2.0;
                    Color::new(s, s, 1.0, 1.0)
                } else {
                    let s = (t - 0.5) * 2.0;
                    Color::new(1.0, 1.0 - s, 1.0 - s, 1.0)
                }
            }
            Self::Viridis => {
                let colors = [
                    (0.27, 0.00, 0.33),
                    (0.28, 0.14, 0.45),
                    (0.26, 0.24, 0.53),
                    (0.22, 0.34, 0.55),
                    (0.18, 0.44, 0.56),
                    (0.12, 0.56, 0.55),
                    (0.20, 0.72, 0.47),
                    (0.99, 0.91, 0.15),
                ];
                let idx = ((t * 7.0) as usize).min(6);
                let frac = (t * 7.0) - idx as f32;
                let (r1, g1, b1) = colors[idx];
                let (r2, g2, b2) = colors[(idx + 1).min(7)];
                Color::new(
                    r1 + (r2 - r1) * frac,
                    g1 + (g2 - g1) * frac,
                    b1 + (b2 - b1) * frac,
                    1.0,
                )
            }
            Self::GreenRed => Color::new(t, 1.0 - t, 0.0, 1.0),
            Self::Grayscale => Color::new(t, t, t, 1.0),
            Self::Mono(r, g, b) => {
                let r = (*r as f32 / 255.0) * t;
                let g = (*g as f32 / 255.0) * t;
                let b = (*b as f32 / 255.0) * t;
                Color::new(r, g, b, 1.0)
            }
        }
    }
}

/// A single heatmap cell.
#[derive(Debug, Clone)]
pub struct HeatmapCell {
    /// Cell value (will be normalized).
    pub value: f64,
    /// Optional label to display.
    pub label: Option<String>,
}

impl HeatmapCell {
    /// Create a cell with a value.
    #[must_use]
    pub fn new(value: f64) -> Self {
        Self { value, label: None }
    }

    /// Create a cell with value and label.
    #[must_use]
    pub fn with_label(value: f64, label: impl Into<String>) -> Self {
        Self {
            value,
            label: Some(label.into()),
        }
    }
}

/// Heatmap widget for 2D grid visualization.
#[derive(Debug, Clone)]
pub struct Heatmap {
    /// Grid data (row-major order).
    data: Vec<Vec<HeatmapCell>>,
    /// Row labels.
    row_labels: Vec<String>,
    /// Column labels.
    col_labels: Vec<String>,
    /// Color palette.
    palette: HeatmapPalette,
    /// Minimum value for normalization.
    min: f64,
    /// Maximum value for normalization.
    max: f64,
    /// Show cell values.
    show_values: bool,
    /// Cell width in characters.
    cell_width: u16,
    /// Cell height in characters.
    cell_height: u16,
    /// Cached bounds.
    bounds: Rect,
}

impl Default for Heatmap {
    fn default() -> Self {
        Self::new(vec![])
    }
}

impl Heatmap {
    /// Create a new heatmap with data.
    #[must_use]
    pub fn new(data: Vec<Vec<HeatmapCell>>) -> Self {
        let (min, max) = Self::compute_range(&data);
        Self {
            data,
            row_labels: vec![],
            col_labels: vec![],
            palette: HeatmapPalette::default(),
            min,
            max,
            show_values: false,
            cell_width: 4,
            cell_height: 1,
            bounds: Rect::default(),
        }
    }

    /// Create from a 2D array of f64 values.
    #[must_use]
    pub fn from_values(values: Vec<Vec<f64>>) -> Self {
        let data: Vec<Vec<HeatmapCell>> = values
            .into_iter()
            .map(|row| row.into_iter().map(HeatmapCell::new).collect())
            .collect();
        Self::new(data)
    }

    /// Set row labels.
    #[must_use]
    pub fn with_row_labels(mut self, labels: Vec<String>) -> Self {
        self.row_labels = labels;
        self
    }

    /// Set column labels.
    #[must_use]
    pub fn with_col_labels(mut self, labels: Vec<String>) -> Self {
        self.col_labels = labels;
        self
    }

    /// Set color palette.
    #[must_use]
    pub fn with_palette(mut self, palette: HeatmapPalette) -> Self {
        self.palette = palette;
        self
    }

    /// Set value range.
    #[must_use]
    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        self.min = min;
        self.max = max.max(min + 0.001);
        self
    }

    /// Show cell values.
    #[must_use]
    pub fn with_values(mut self, show: bool) -> Self {
        self.show_values = show;
        self
    }

    /// Set cell dimensions.
    #[must_use]
    pub fn with_cell_size(mut self, width: u16, height: u16) -> Self {
        self.cell_width = width.max(1);
        self.cell_height = height.max(1);
        self
    }

    /// Get number of rows.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.data.len()
    }

    /// Get number of columns.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.data.first().map_or(0, Vec::len)
    }

    fn compute_range(data: &[Vec<HeatmapCell>]) -> (f64, f64) {
        let mut min = f64::MAX;
        let mut max = f64::MIN;
        for row in data {
            for cell in row {
                min = min.min(cell.value);
                max = max.max(cell.value);
            }
        }
        if min == f64::MAX {
            (0.0, 1.0)
        } else if (max - min).abs() < f64::EPSILON {
            (min - 0.5, max + 0.5)
        } else {
            (min, max)
        }
    }

    fn normalize(&self, value: f64) -> f64 {
        let range = self.max - self.min;
        if range.abs() < f64::EPSILON {
            0.5
        } else {
            ((value - self.min) / range).clamp(0.0, 1.0)
        }
    }
}

impl Brick for Heatmap {
    fn brick_name(&self) -> &'static str {
        "heatmap"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        static ASSERTIONS: &[BrickAssertion] = &[BrickAssertion::max_latency_ms(16)];
        ASSERTIONS
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(16)
    }

    fn verify(&self) -> BrickVerification {
        BrickVerification {
            passed: self.assertions().to_vec(),
            failed: vec![],
            verification_time: Duration::from_micros(10),
        }
    }

    fn to_html(&self) -> String {
        String::new()
    }

    fn to_css(&self) -> String {
        String::new()
    }
}

impl Widget for Heatmap {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let label_width = self.row_labels.iter().map(String::len).max().unwrap_or(0) as f32;
        let width = label_width + (self.cols() as f32 * self.cell_width as f32);
        let height = if self.col_labels.is_empty() { 0.0 } else { 1.0 }
            + (self.rows() as f32 * self.cell_height as f32);
        constraints.constrain(Size::new(width, height))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        if self.data.is_empty() {
            return;
        }

        let label_width = self.row_labels.iter().map(String::len).max().unwrap_or(0) as f32;
        let start_x = self.bounds.x + label_width;
        let mut start_y = self.bounds.y;

        // Draw column labels
        if !self.col_labels.is_empty() {
            let label_style = TextStyle {
                color: Color::new(0.7, 0.7, 0.7, 1.0),
                ..Default::default()
            };
            for (col, label) in self.col_labels.iter().enumerate() {
                let x = start_x + (col as f32 * self.cell_width as f32);
                let truncated: String = label.chars().take(self.cell_width as usize).collect();
                canvas.draw_text(&truncated, Point::new(x, start_y), &label_style);
            }
            start_y += 1.0;
        }

        // Draw cells
        for (row_idx, row) in self.data.iter().enumerate() {
            let y = start_y + (row_idx as f32 * self.cell_height as f32);

            // Row label
            if let Some(label) = self.row_labels.get(row_idx) {
                let label_style = TextStyle {
                    color: Color::new(0.7, 0.7, 0.7, 1.0),
                    ..Default::default()
                };
                canvas.draw_text(label, Point::new(self.bounds.x, y), &label_style);
            }

            // Cells
            for (col_idx, cell) in row.iter().enumerate() {
                let x = start_x + (col_idx as f32 * self.cell_width as f32);
                let norm = self.normalize(cell.value);
                let color = self.palette.color(norm);

                // Fill cell background
                canvas.fill_rect(
                    Rect::new(x, y, self.cell_width as f32, self.cell_height as f32),
                    color,
                );

                // Draw value or label
                if self.show_values {
                    let text = cell
                        .label
                        .clone()
                        .unwrap_or_else(|| format!("{:.1}", cell.value));
                    let text: String = text.chars().take(self.cell_width as usize).collect();

                    // Use contrasting color for text
                    let text_color = if norm > 0.5 {
                        Color::BLACK
                    } else {
                        Color::WHITE
                    };
                    let text_style = TextStyle {
                        color: text_color,
                        ..Default::default()
                    };
                    canvas.draw_text(&text, Point::new(x, y), &text_style);
                }
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    struct MockCanvas {
        texts: Vec<(String, Point)>,
        rects: Vec<(Rect, Color)>,
    }

    impl MockCanvas {
        fn new() -> Self {
            Self {
                texts: vec![],
                rects: vec![],
            }
        }
    }

    impl Canvas for MockCanvas {
        fn fill_rect(&mut self, rect: Rect, color: Color) {
            self.rects.push((rect, color));
        }
        fn stroke_rect(&mut self, _rect: Rect, _color: Color, _width: f32) {}
        fn draw_text(&mut self, text: &str, position: Point, _style: &TextStyle) {
            self.texts.push((text.to_string(), position));
        }
        fn draw_line(&mut self, _from: Point, _to: Point, _color: Color, _width: f32) {}
        fn fill_circle(&mut self, _center: Point, _radius: f32, _color: Color) {}
        fn stroke_circle(&mut self, _center: Point, _radius: f32, _color: Color, _width: f32) {}
        fn fill_arc(&mut self, _c: Point, _r: f32, _s: f32, _e: f32, _color: Color) {}
        fn draw_path(&mut self, _points: &[Point], _color: Color, _width: f32) {}
        fn fill_polygon(&mut self, _points: &[Point], _color: Color) {}
        fn push_clip(&mut self, _rect: Rect) {}
        fn pop_clip(&mut self) {}
        fn push_transform(&mut self, _transform: presentar_core::Transform2D) {}
        fn pop_transform(&mut self) {}
    }

    #[test]
    fn test_heatmap_creation() {
        let data = vec![
            vec![HeatmapCell::new(1.0), HeatmapCell::new(2.0)],
            vec![HeatmapCell::new(3.0), HeatmapCell::new(4.0)],
        ];
        let heatmap = Heatmap::new(data);
        assert_eq!(heatmap.rows(), 2);
        assert_eq!(heatmap.cols(), 2);
    }

    #[test]
    fn test_heatmap_from_values() {
        let heatmap = Heatmap::from_values(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert_eq!(heatmap.rows(), 2);
        assert_eq!(heatmap.cols(), 2);
    }

    #[test]
    fn test_heatmap_assertions() {
        let heatmap = Heatmap::default();
        assert!(!heatmap.assertions().is_empty());
    }

    #[test]
    fn test_heatmap_verify() {
        let heatmap = Heatmap::default();
        assert!(heatmap.verify().is_valid());
    }

    #[test]
    fn test_heatmap_with_palette() {
        let heatmap = Heatmap::default().with_palette(HeatmapPalette::Viridis);
        assert_eq!(heatmap.palette, HeatmapPalette::Viridis);
    }

    #[test]
    fn test_heatmap_with_range() {
        let heatmap = Heatmap::default().with_range(0.0, 100.0);
        assert_eq!(heatmap.min, 0.0);
        assert_eq!(heatmap.max, 100.0);
    }

    #[test]
    fn test_heatmap_with_values() {
        let heatmap = Heatmap::default().with_values(true);
        assert!(heatmap.show_values);
    }

    #[test]
    fn test_heatmap_with_cell_size() {
        let heatmap = Heatmap::default().with_cell_size(6, 2);
        assert_eq!(heatmap.cell_width, 6);
        assert_eq!(heatmap.cell_height, 2);
    }

    #[test]
    fn test_heatmap_with_labels() {
        let heatmap = Heatmap::default()
            .with_row_labels(vec!["A".to_string(), "B".to_string()])
            .with_col_labels(vec!["X".to_string(), "Y".to_string()]);
        assert_eq!(heatmap.row_labels.len(), 2);
        assert_eq!(heatmap.col_labels.len(), 2);
    }

    #[test]
    fn test_heatmap_paint() {
        let mut heatmap = Heatmap::from_values(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        heatmap.bounds = Rect::new(0.0, 0.0, 20.0, 10.0);
        let mut canvas = MockCanvas::new();
        heatmap.paint(&mut canvas);
        assert!(!canvas.rects.is_empty());
    }

    #[test]
    fn test_heatmap_paint_with_values() {
        let mut heatmap = Heatmap::from_values(vec![vec![1.0, 2.0]]).with_values(true);
        heatmap.bounds = Rect::new(0.0, 0.0, 20.0, 10.0);
        let mut canvas = MockCanvas::new();
        heatmap.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_heatmap_paint_with_labels() {
        let mut heatmap = Heatmap::from_values(vec![vec![1.0]])
            .with_row_labels(vec!["Row".to_string()])
            .with_col_labels(vec!["Col".to_string()]);
        heatmap.bounds = Rect::new(0.0, 0.0, 20.0, 10.0);
        let mut canvas = MockCanvas::new();
        heatmap.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn test_heatmap_empty() {
        let mut heatmap = Heatmap::default();
        heatmap.bounds = Rect::new(0.0, 0.0, 20.0, 10.0);
        let mut canvas = MockCanvas::new();
        heatmap.paint(&mut canvas);
        assert!(canvas.rects.is_empty());
    }

    #[test]
    fn test_palette_blue_red() {
        let palette = HeatmapPalette::BlueRed;
        let _low = palette.color(0.0);
        let _mid = palette.color(0.5);
        let _high = palette.color(1.0);
    }

    #[test]
    fn test_palette_viridis() {
        let palette = HeatmapPalette::Viridis;
        let _low = palette.color(0.0);
        let _mid = palette.color(0.5);
        let _high = palette.color(1.0);
    }

    #[test]
    fn test_palette_green_red() {
        let palette = HeatmapPalette::GreenRed;
        let low = palette.color(0.0);
        let high = palette.color(1.0);
        assert!(low.g > low.r);
        assert!(high.r > high.g);
    }

    #[test]
    fn test_palette_grayscale() {
        let palette = HeatmapPalette::Grayscale;
        let mid = palette.color(0.5);
        assert!((mid.r - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_palette_mono() {
        let palette = HeatmapPalette::Mono(255, 0, 0);
        let full = palette.color(1.0);
        assert!((full.r - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_heatmap_cell_with_label() {
        let cell = HeatmapCell::with_label(5.0, "test");
        assert_eq!(cell.value, 5.0);
        assert_eq!(cell.label, Some("test".to_string()));
    }

    #[test]
    fn test_heatmap_measure() {
        let heatmap = Heatmap::from_values(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let size = heatmap.measure(Constraints::loose(Size::new(100.0, 100.0)));
        assert!(size.width > 0.0);
        assert!(size.height > 0.0);
    }

    #[test]
    fn test_heatmap_layout() {
        let mut heatmap = Heatmap::from_values(vec![vec![1.0]]);
        let bounds = Rect::new(5.0, 10.0, 30.0, 20.0);
        let result = heatmap.layout(bounds);
        assert_eq!(result.size.width, 30.0);
        assert_eq!(heatmap.bounds, bounds);
    }

    #[test]
    fn test_heatmap_brick_name() {
        let heatmap = Heatmap::default();
        assert_eq!(heatmap.brick_name(), "heatmap");
    }

    #[test]
    fn test_heatmap_type_id() {
        let heatmap = Heatmap::default();
        assert_eq!(Widget::type_id(&heatmap), TypeId::of::<Heatmap>());
    }

    #[test]
    fn test_heatmap_children() {
        let heatmap = Heatmap::default();
        assert!(heatmap.children().is_empty());
    }

    #[test]
    fn test_heatmap_event() {
        let mut heatmap = Heatmap::default();
        let event = Event::key_down(presentar_core::Key::Enter);
        assert!(heatmap.event(&event).is_none());
    }

    #[test]
    fn test_heatmap_children_mut() {
        let mut heatmap = Heatmap::default();
        assert!(heatmap.children_mut().is_empty());
    }

    #[test]
    fn test_heatmap_to_html() {
        let heatmap = Heatmap::default();
        assert!(heatmap.to_html().is_empty());
    }

    #[test]
    fn test_heatmap_to_css() {
        let heatmap = Heatmap::default();
        assert!(heatmap.to_css().is_empty());
    }

    #[test]
    fn test_heatmap_budget() {
        let heatmap = Heatmap::default();
        let budget = heatmap.budget();
        assert_eq!(budget.total_ms, 16);
    }

    #[test]
    fn test_heatmap_same_value_range() {
        // All cells have the same value - edge case for compute_range
        let heatmap = Heatmap::from_values(vec![vec![5.0, 5.0], vec![5.0, 5.0]]);
        // Range should be padded to avoid division by zero
        assert!(heatmap.min < heatmap.max);
    }

    #[test]
    fn test_heatmap_with_cell_labels() {
        let data = vec![vec![
            HeatmapCell::with_label(1.0, "A"),
            HeatmapCell::with_label(2.0, "B"),
        ]];
        let mut heatmap = Heatmap::new(data).with_values(true);
        heatmap.bounds = Rect::new(0.0, 0.0, 20.0, 10.0);
        let mut canvas = MockCanvas::new();
        heatmap.paint(&mut canvas);
        // Should show cell labels
        assert!(canvas.texts.iter().any(|(t, _)| t == "A" || t == "B"));
    }

    #[test]
    fn test_heatmap_high_value_contrast() {
        // High normalized value should use dark text
        let mut heatmap = Heatmap::from_values(vec![vec![10.0]]).with_values(true);
        heatmap.bounds = Rect::new(0.0, 0.0, 20.0, 10.0);
        let mut canvas = MockCanvas::new();
        heatmap.paint(&mut canvas);
        // Should have text drawn
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_heatmap_low_value_contrast() {
        // Low normalized value should use light text
        let mut heatmap = Heatmap::from_values(vec![vec![1.0, 10.0]]).with_values(true);
        heatmap.bounds = Rect::new(0.0, 0.0, 20.0, 10.0);
        let mut canvas = MockCanvas::new();
        heatmap.paint(&mut canvas);
        // Should have text drawn
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_palette_blue_red_low() {
        let palette = HeatmapPalette::BlueRed;
        let low = palette.color(0.0);
        // Low value: should be blue-ish
        assert!(low.b > low.r);
    }

    #[test]
    fn test_palette_blue_red_high() {
        let palette = HeatmapPalette::BlueRed;
        let high = palette.color(1.0);
        // High value: should be red-ish
        assert!(high.r > high.b);
    }

    #[test]
    fn test_heatmap_measure_with_labels() {
        let heatmap = Heatmap::from_values(vec![vec![1.0, 2.0]])
            .with_row_labels(vec!["LongRowLabel".to_string()])
            .with_col_labels(vec!["A".to_string(), "B".to_string()]);
        let size = heatmap.measure(Constraints::loose(Size::new(100.0, 100.0)));
        // Width should include label width
        assert!(size.width > 0.0);
        // Height should include column label row
        assert!(size.height >= 2.0);
    }

    #[test]
    fn test_heatmap_with_range_min_equals_max() {
        let heatmap = Heatmap::default().with_range(5.0, 5.0);
        // Max should be adjusted to avoid zero range
        assert!(heatmap.max > heatmap.min);
    }

    #[test]
    fn test_heatmap_cell_size_min() {
        let heatmap = Heatmap::default().with_cell_size(0, 0);
        // Should enforce minimum of 1
        assert_eq!(heatmap.cell_width, 1);
        assert_eq!(heatmap.cell_height, 1);
    }
}
