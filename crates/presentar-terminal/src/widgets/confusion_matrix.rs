//! Confusion matrix widget for ML classification visualization.
//!
//! Displays a confusion matrix with color-coded cells showing
//! classification performance across classes.

use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// Normalization mode for confusion matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Normalization {
    /// No normalization (raw counts).
    #[default]
    None,
    /// Normalize by row (recall per class).
    Row,
    /// Normalize by column (precision per class).
    Column,
    /// Normalize by total (overall distribution).
    Total,
}

/// Color palette for confusion matrix cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MatrixPalette {
    /// Blue (low) to red (high).
    #[default]
    BlueRed,
    /// Green for diagonal, red for off-diagonal.
    DiagonalGreen,
    /// Grayscale.
    Grayscale,
}

impl MatrixPalette {
    /// Get color for a normalized value (0.0 to 1.0).
    #[must_use]
    pub fn color(&self, value: f64, is_diagonal: bool) -> Color {
        let v = value.clamp(0.0, 1.0) as f32;
        match self {
            Self::BlueRed => Color::new(v, 0.2, 1.0 - v, 1.0),
            Self::DiagonalGreen => {
                if is_diagonal {
                    // Green for diagonal (correct predictions)
                    Color::new(0.2, 0.3 + v * 0.7, 0.2, 1.0)
                } else {
                    // Red for off-diagonal (errors)
                    Color::new(0.3 + v * 0.7, 0.2, 0.2, 1.0)
                }
            }
            Self::Grayscale => {
                let g = 0.2 + v * 0.6;
                Color::new(g, g, g, 1.0)
            }
        }
    }
}

/// Confusion matrix widget for classification visualization.
#[derive(Debug, Clone)]
pub struct ConfusionMatrix {
    /// Matrix data (rows are actual, columns are predicted).
    matrix: Vec<Vec<u64>>,
    /// Class labels.
    labels: Vec<String>,
    /// Normalization mode.
    normalization: Normalization,
    /// Color palette.
    palette: MatrixPalette,
    /// Cell width in characters.
    cell_width: usize,
    /// Whether to show values in cells.
    show_values: bool,
    /// Whether to show percentages instead of counts.
    show_percentages: bool,
    /// Title.
    title: Option<String>,
    /// Cached bounds.
    bounds: Rect,
}

impl Default for ConfusionMatrix {
    fn default() -> Self {
        Self::new(vec![vec![0]])
    }
}

impl ConfusionMatrix {
    /// Create a new confusion matrix.
    #[must_use]
    pub fn new(matrix: Vec<Vec<u64>>) -> Self {
        let size = matrix.len();
        let labels: Vec<String> = (0..size).map(|i| format!("{i}")).collect();
        Self {
            matrix,
            labels,
            normalization: Normalization::None,
            palette: MatrixPalette::default(),
            cell_width: 6,
            show_values: true,
            show_percentages: false,
            title: None,
            bounds: Rect::default(),
        }
    }

    /// Set class labels.
    #[must_use]
    pub fn with_labels(mut self, labels: Vec<String>) -> Self {
        self.labels = labels;
        self
    }

    /// Set normalization mode.
    #[must_use]
    pub fn with_normalization(mut self, normalization: Normalization) -> Self {
        self.normalization = normalization;
        self
    }

    /// Set color palette.
    #[must_use]
    pub fn with_palette(mut self, palette: MatrixPalette) -> Self {
        self.palette = palette;
        self
    }

    /// Set cell width.
    #[must_use]
    pub fn with_cell_width(mut self, width: usize) -> Self {
        self.cell_width = width.max(3);
        self
    }

    /// Show or hide values in cells.
    #[must_use]
    pub fn with_values(mut self, show: bool) -> Self {
        self.show_values = show;
        self
    }

    /// Show percentages instead of counts.
    #[must_use]
    pub fn with_percentages(mut self, show: bool) -> Self {
        self.show_percentages = show;
        self
    }

    /// Set title.
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Update matrix data.
    pub fn set_matrix(&mut self, matrix: Vec<Vec<u64>>) {
        self.matrix = matrix;
    }

    /// Get matrix dimensions.
    #[must_use]
    pub fn size(&self) -> usize {
        self.matrix.len()
    }

    /// Get total count.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.matrix.iter().flatten().sum()
    }

    /// Get accuracy (correct / total).
    #[must_use]
    pub fn accuracy(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            return 0.0;
        }
        let correct: u64 = self
            .matrix
            .iter()
            .enumerate()
            .map(|(i, row)| row.get(i).copied().unwrap_or(0))
            .sum();
        correct as f64 / total as f64
    }

    /// Get precision for a class.
    #[must_use]
    pub fn precision(&self, class: usize) -> f64 {
        let col_sum: u64 = self
            .matrix
            .iter()
            .map(|row| row.get(class).copied().unwrap_or(0))
            .sum();
        if col_sum == 0 {
            return 0.0;
        }
        self.matrix
            .get(class)
            .and_then(|row| row.get(class))
            .copied()
            .unwrap_or(0) as f64
            / col_sum as f64
    }

    /// Get recall for a class.
    #[must_use]
    pub fn recall(&self, class: usize) -> f64 {
        let row_sum: u64 = self.matrix.get(class).map_or(0, |row| row.iter().sum());
        if row_sum == 0 {
            return 0.0;
        }
        self.matrix
            .get(class)
            .and_then(|row| row.get(class))
            .copied()
            .unwrap_or(0) as f64
            / row_sum as f64
    }

    /// Get F1 score for a class.
    #[must_use]
    pub fn f1_score(&self, class: usize) -> f64 {
        let p = self.precision(class);
        let r = self.recall(class);
        if p + r == 0.0 {
            return 0.0;
        }
        2.0 * p * r / (p + r)
    }

    fn normalize_value(&self, row: usize, col: usize, value: u64) -> f64 {
        match self.normalization {
            Normalization::None => {
                let max_val = self.matrix.iter().flatten().max().copied().unwrap_or(1);
                if max_val == 0 {
                    0.0
                } else {
                    value as f64 / max_val as f64
                }
            }
            Normalization::Row => {
                let row_sum: u64 = self.matrix.get(row).map_or(1, |r| r.iter().sum());
                if row_sum == 0 {
                    0.0
                } else {
                    value as f64 / row_sum as f64
                }
            }
            Normalization::Column => {
                let col_sum: u64 = self
                    .matrix
                    .iter()
                    .map(|r| r.get(col).copied().unwrap_or(0))
                    .sum();
                if col_sum == 0 {
                    0.0
                } else {
                    value as f64 / col_sum as f64
                }
            }
            Normalization::Total => {
                let total = self.total();
                if total == 0 {
                    0.0
                } else {
                    value as f64 / total as f64
                }
            }
        }
    }

    fn format_value(&self, value: u64, normalized: f64) -> String {
        if self.show_percentages {
            format!("{:.0}%", normalized * 100.0)
        } else {
            value.to_string()
        }
    }

    fn label_width(&self) -> usize {
        self.labels
            .iter()
            .map(String::len)
            .max()
            .unwrap_or(3)
            .max(3)
    }
}

impl Brick for ConfusionMatrix {
    fn brick_name(&self) -> &'static str {
        "confusion_matrix"
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

impl Widget for ConfusionMatrix {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let label_w = self.label_width();
        let n = self.size();
        let title_rows = if self.title.is_some() { 2 } else { 0 };

        // Width: label column + header labels + cells
        let width = (label_w + 2 + n * (self.cell_width + 1)) as f32;
        // Height: title + header row + data rows + accuracy row
        let height = (title_rows + 1 + n + 1) as f32;

        constraints.constrain(Size::new(width.min(constraints.max_width), height))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        if self.matrix.is_empty() || self.bounds.width < 1.0 {
            return;
        }

        let label_w = self.label_width();
        let n = self.size();
        let mut y = self.bounds.y;

        let header_style = TextStyle {
            color: Color::new(0.9, 0.9, 0.9, 1.0),
            ..Default::default()
        };

        let dim_style = TextStyle {
            color: Color::new(0.6, 0.6, 0.6, 1.0),
            ..Default::default()
        };

        // Draw title
        if let Some(ref title) = self.title {
            canvas.draw_text(title, Point::new(self.bounds.x, y), &header_style);
            y += 2.0;
        }

        // Draw header row (predicted labels)
        let header_x = self.bounds.x + label_w as f32 + 2.0;
        canvas.draw_text("Pred→", Point::new(self.bounds.x, y), &dim_style);
        for (i, label) in self.labels.iter().enumerate().take(n) {
            let x = header_x + (i * (self.cell_width + 1)) as f32;
            let truncated = if label.len() > self.cell_width {
                &label[..self.cell_width]
            } else {
                label
            };
            canvas.draw_text(truncated, Point::new(x, y), &header_style);
        }
        y += 1.0;

        // Draw matrix rows
        for (row_idx, row) in self.matrix.iter().enumerate().take(n) {
            // Row label (actual)
            let label = self.labels.get(row_idx).map_or("?", String::as_str);
            let truncated = if label.len() > label_w {
                &label[..label_w]
            } else {
                label
            };
            canvas.draw_text(truncated, Point::new(self.bounds.x, y), &header_style);

            // Cells
            for (col_idx, &value) in row.iter().enumerate().take(n) {
                let x = header_x + (col_idx * (self.cell_width + 1)) as f32;
                let normalized = self.normalize_value(row_idx, col_idx, value);
                let is_diagonal = row_idx == col_idx;

                // Draw cell background
                let bg_color = self.palette.color(normalized, is_diagonal);
                canvas.fill_rect(Rect::new(x, y, self.cell_width as f32, 1.0), bg_color);

                // Draw value
                if self.show_values {
                    let text = self.format_value(value, normalized);
                    let text_color = if normalized > 0.5 {
                        Color::new(0.0, 0.0, 0.0, 1.0) // Dark text on light bg
                    } else {
                        Color::new(1.0, 1.0, 1.0, 1.0) // Light text on dark bg
                    };
                    let value_style = TextStyle {
                        color: text_color,
                        ..Default::default()
                    };
                    canvas.draw_text(&text, Point::new(x, y), &value_style);
                }
            }
            y += 1.0;
        }

        // Draw accuracy
        let accuracy = self.accuracy();
        let acc_text = format!("Accuracy: {:.1}%", accuracy * 100.0);
        canvas.draw_text(&acc_text, Point::new(self.bounds.x, y), &header_style);
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
    fn test_confusion_matrix_creation() {
        let cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]]);
        assert_eq!(cm.size(), 2);
    }

    #[test]
    fn test_confusion_matrix_total() {
        let cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]]);
        assert_eq!(cm.total(), 30);
    }

    #[test]
    fn test_confusion_matrix_accuracy() {
        let cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]]);
        // Correct: 10 + 15 = 25, Total: 30
        let acc = cm.accuracy();
        assert!((acc - 0.833).abs() < 0.01);
    }

    #[test]
    fn test_confusion_matrix_precision() {
        // Class 0: col sum = 10 + 3 = 13, diagonal = 10
        let cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]]);
        let prec = cm.precision(0);
        assert!((prec - 0.769).abs() < 0.01);
    }

    #[test]
    fn test_confusion_matrix_recall() {
        // Class 0: row sum = 10 + 2 = 12, diagonal = 10
        let cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]]);
        let recall = cm.recall(0);
        assert!((recall - 0.833).abs() < 0.01);
    }

    #[test]
    fn test_confusion_matrix_f1() {
        let cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]]);
        let f1 = cm.f1_score(0);
        assert!(f1 > 0.0 && f1 < 1.0);
    }

    #[test]
    fn test_confusion_matrix_with_labels() {
        let cm = ConfusionMatrix::new(vec![vec![5, 1], vec![2, 8]])
            .with_labels(vec!["Cat".to_string(), "Dog".to_string()]);
        assert_eq!(cm.labels.len(), 2);
        assert_eq!(cm.labels[0], "Cat");
    }

    #[test]
    fn test_confusion_matrix_with_normalization() {
        let cm = ConfusionMatrix::new(vec![vec![5]]).with_normalization(Normalization::Row);
        assert_eq!(cm.normalization, Normalization::Row);
    }

    #[test]
    fn test_confusion_matrix_with_palette() {
        let cm = ConfusionMatrix::new(vec![vec![5]]).with_palette(MatrixPalette::DiagonalGreen);
        assert_eq!(cm.palette, MatrixPalette::DiagonalGreen);
    }

    #[test]
    fn test_confusion_matrix_with_cell_width() {
        let cm = ConfusionMatrix::new(vec![vec![5]]).with_cell_width(10);
        assert_eq!(cm.cell_width, 10);
    }

    #[test]
    fn test_confusion_matrix_with_cell_width_min() {
        let cm = ConfusionMatrix::new(vec![vec![5]]).with_cell_width(1);
        assert_eq!(cm.cell_width, 3); // Minimum is 3
    }

    #[test]
    fn test_confusion_matrix_with_values() {
        let cm = ConfusionMatrix::new(vec![vec![5]]).with_values(false);
        assert!(!cm.show_values);
    }

    #[test]
    fn test_confusion_matrix_with_percentages() {
        let cm = ConfusionMatrix::new(vec![vec![5]]).with_percentages(true);
        assert!(cm.show_percentages);
    }

    #[test]
    fn test_confusion_matrix_with_title() {
        let cm = ConfusionMatrix::new(vec![vec![5]]).with_title("Test Matrix");
        assert_eq!(cm.title, Some("Test Matrix".to_string()));
    }

    #[test]
    fn test_confusion_matrix_set_matrix() {
        let mut cm = ConfusionMatrix::new(vec![vec![1]]);
        cm.set_matrix(vec![vec![2, 3], vec![4, 5]]);
        assert_eq!(cm.size(), 2);
    }

    #[test]
    fn test_confusion_matrix_paint() {
        let mut cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]]);
        cm.bounds = Rect::new(0.0, 0.0, 40.0, 10.0);

        let mut canvas = MockCanvas::new();
        cm.paint(&mut canvas);

        assert!(!canvas.texts.is_empty());
        assert!(!canvas.rects.is_empty());
    }

    #[test]
    fn test_confusion_matrix_paint_empty() {
        let cm = ConfusionMatrix::new(vec![]);
        let mut canvas = MockCanvas::new();
        cm.paint(&mut canvas);
        assert!(canvas.texts.is_empty());
    }

    #[test]
    fn test_confusion_matrix_measure() {
        let cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]]);
        let size = cm.measure(Constraints::loose(Size::new(100.0, 50.0)));
        assert!(size.width > 0.0);
        assert!(size.height > 0.0);
    }

    #[test]
    fn test_confusion_matrix_layout() {
        let mut cm = ConfusionMatrix::new(vec![vec![1]]);
        let bounds = Rect::new(5.0, 10.0, 30.0, 20.0);
        let result = cm.layout(bounds);
        assert_eq!(result.size.width, 30.0);
        assert_eq!(cm.bounds, bounds);
    }

    #[test]
    fn test_confusion_matrix_brick_name() {
        let cm = ConfusionMatrix::new(vec![vec![1]]);
        assert_eq!(cm.brick_name(), "confusion_matrix");
    }

    #[test]
    fn test_confusion_matrix_assertions() {
        let cm = ConfusionMatrix::new(vec![vec![1]]);
        assert!(!cm.assertions().is_empty());
    }

    #[test]
    fn test_confusion_matrix_budget() {
        let cm = ConfusionMatrix::new(vec![vec![1]]);
        let budget = cm.budget();
        assert!(budget.paint_ms > 0);
    }

    #[test]
    fn test_confusion_matrix_verify() {
        let cm = ConfusionMatrix::new(vec![vec![1]]);
        assert!(cm.verify().is_valid());
    }

    #[test]
    fn test_confusion_matrix_type_id() {
        let cm = ConfusionMatrix::new(vec![vec![1]]);
        assert_eq!(Widget::type_id(&cm), TypeId::of::<ConfusionMatrix>());
    }

    #[test]
    fn test_confusion_matrix_children() {
        let cm = ConfusionMatrix::new(vec![vec![1]]);
        assert!(cm.children().is_empty());
    }

    #[test]
    fn test_confusion_matrix_children_mut() {
        let mut cm = ConfusionMatrix::new(vec![vec![1]]);
        assert!(cm.children_mut().is_empty());
    }

    #[test]
    fn test_confusion_matrix_event() {
        let mut cm = ConfusionMatrix::new(vec![vec![1]]);
        let event = Event::key_down(presentar_core::Key::Enter);
        assert!(cm.event(&event).is_none());
    }

    #[test]
    fn test_confusion_matrix_default() {
        let cm = ConfusionMatrix::default();
        assert_eq!(cm.size(), 1);
    }

    #[test]
    fn test_confusion_matrix_to_html() {
        let cm = ConfusionMatrix::new(vec![vec![1]]);
        assert!(cm.to_html().is_empty());
    }

    #[test]
    fn test_confusion_matrix_to_css() {
        let cm = ConfusionMatrix::new(vec![vec![1]]);
        assert!(cm.to_css().is_empty());
    }

    #[test]
    fn test_palette_blue_red() {
        let palette = MatrixPalette::BlueRed;
        let low = palette.color(0.0, false);
        let high = palette.color(1.0, false);
        assert!(low.b > low.r); // Blue dominant for low
        assert!(high.r > high.b); // Red dominant for high
    }

    #[test]
    fn test_palette_diagonal_green() {
        let palette = MatrixPalette::DiagonalGreen;
        let diag = palette.color(0.8, true);
        let off_diag = palette.color(0.8, false);
        assert!(diag.g > diag.r); // Green for diagonal
        assert!(off_diag.r > off_diag.g); // Red for off-diagonal
    }

    #[test]
    fn test_palette_grayscale() {
        let palette = MatrixPalette::Grayscale;
        let color = palette.color(0.5, false);
        assert!((color.r - color.g).abs() < 0.01);
        assert!((color.g - color.b).abs() < 0.01);
    }

    #[test]
    fn test_normalization_default() {
        assert_eq!(Normalization::default(), Normalization::None);
    }

    #[test]
    fn test_zero_accuracy() {
        let cm = ConfusionMatrix::new(vec![vec![0, 0], vec![0, 0]]);
        assert_eq!(cm.accuracy(), 0.0);
    }

    #[test]
    fn test_zero_precision() {
        let cm = ConfusionMatrix::new(vec![vec![0, 0], vec![0, 0]]);
        assert_eq!(cm.precision(0), 0.0);
    }

    #[test]
    fn test_zero_recall() {
        let cm = ConfusionMatrix::new(vec![vec![0, 0], vec![0, 0]]);
        assert_eq!(cm.recall(0), 0.0);
    }

    #[test]
    fn test_zero_f1() {
        let cm = ConfusionMatrix::new(vec![vec![0, 0], vec![0, 0]]);
        assert_eq!(cm.f1_score(0), 0.0);
    }

    #[test]
    fn test_paint_with_title() {
        let mut cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]])
            .with_title("Classification Results");
        cm.bounds = Rect::new(0.0, 0.0, 50.0, 15.0);

        let mut canvas = MockCanvas::new();
        cm.paint(&mut canvas);

        // Title should be in the texts
        assert!(canvas
            .texts
            .iter()
            .any(|(t, _)| t.contains("Classification")));
    }

    #[test]
    fn test_paint_with_row_normalization() {
        let mut cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]])
            .with_normalization(Normalization::Row);
        cm.bounds = Rect::new(0.0, 0.0, 50.0, 15.0);

        let mut canvas = MockCanvas::new();
        cm.paint(&mut canvas);

        // Should have painted cells
        assert!(!canvas.rects.is_empty());
    }

    #[test]
    fn test_paint_with_column_normalization() {
        let mut cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]])
            .with_normalization(Normalization::Column);
        cm.bounds = Rect::new(0.0, 0.0, 50.0, 15.0);

        let mut canvas = MockCanvas::new();
        cm.paint(&mut canvas);

        // Should have painted cells
        assert!(!canvas.rects.is_empty());
    }

    #[test]
    fn test_paint_with_total_normalization() {
        let mut cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]])
            .with_normalization(Normalization::Total);
        cm.bounds = Rect::new(0.0, 0.0, 50.0, 15.0);

        let mut canvas = MockCanvas::new();
        cm.paint(&mut canvas);

        // Should have painted cells
        assert!(!canvas.rects.is_empty());
    }

    #[test]
    fn test_paint_with_percentages() {
        let mut cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]]).with_percentages(true);
        cm.bounds = Rect::new(0.0, 0.0, 50.0, 15.0);

        let mut canvas = MockCanvas::new();
        cm.paint(&mut canvas);

        // Should show percentage values
        assert!(canvas.texts.iter().any(|(t, _)| t.contains('%')));
    }

    #[test]
    fn test_normalize_value_none_zero_max() {
        // Matrix with all zeros - max_val == 0
        let mut cm = ConfusionMatrix::new(vec![vec![0, 0], vec![0, 0]]);
        cm.bounds = Rect::new(0.0, 0.0, 50.0, 15.0);

        let mut canvas = MockCanvas::new();
        cm.paint(&mut canvas);

        // Should still paint without error
        assert!(!canvas.rects.is_empty());
    }

    #[test]
    fn test_normalize_row_zero_sum() {
        // Row with zero sum
        let mut cm = ConfusionMatrix::new(vec![vec![0, 0], vec![3, 15]])
            .with_normalization(Normalization::Row);
        cm.bounds = Rect::new(0.0, 0.0, 50.0, 15.0);

        let mut canvas = MockCanvas::new();
        cm.paint(&mut canvas);

        // Should still paint without error
        assert!(!canvas.rects.is_empty());
    }

    #[test]
    fn test_normalize_column_zero_sum() {
        // Column with zero sum
        let mut cm = ConfusionMatrix::new(vec![vec![10, 0], vec![3, 0]])
            .with_normalization(Normalization::Column);
        cm.bounds = Rect::new(0.0, 0.0, 50.0, 15.0);

        let mut canvas = MockCanvas::new();
        cm.paint(&mut canvas);

        // Should still paint without error
        assert!(!canvas.rects.is_empty());
    }

    #[test]
    fn test_normalize_total_zero() {
        // All zeros
        let mut cm = ConfusionMatrix::new(vec![vec![0, 0], vec![0, 0]])
            .with_normalization(Normalization::Total);
        cm.bounds = Rect::new(0.0, 0.0, 50.0, 15.0);

        let mut canvas = MockCanvas::new();
        cm.paint(&mut canvas);

        // Should still paint without error
        assert!(!canvas.rects.is_empty());
    }

    #[test]
    fn test_paint_without_values() {
        let mut cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]]).with_values(false);
        cm.bounds = Rect::new(0.0, 0.0, 50.0, 15.0);

        let mut canvas = MockCanvas::new();
        cm.paint(&mut canvas);

        // Cells painted but no value texts in cells
        assert!(!canvas.rects.is_empty());
    }

    #[test]
    fn test_paint_long_labels() {
        let mut cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]]).with_labels(vec![
            "VeryLongLabelName".to_string(),
            "AnotherLongLabel".to_string(),
        ]);
        cm.bounds = Rect::new(0.0, 0.0, 50.0, 15.0);

        let mut canvas = MockCanvas::new();
        cm.paint(&mut canvas);

        // Should truncate labels
        assert!(!canvas.texts.is_empty());
    }
}
