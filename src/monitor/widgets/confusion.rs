//! Confusion matrix widget for ML classification visualization.
//!
//! Displays a confusion matrix with color-coded cells showing
//! classification performance across classes. Supports multiple
//! normalization modes and color palettes.
//!
//! ## ML Metrics
//! - Accuracy: Overall correct predictions / total
//! - Precision: True positives / predicted positives per class
//! - Recall: True positives / actual positives per class
//! - F1 Score: Harmonic mean of precision and recall

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Style},
    widgets::Widget,
};

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
    /// Cool blues (for a calmer visualization).
    Blues,
}

impl MatrixPalette {
    /// Get color for a normalized value (0.0 to 1.0).
    #[must_use]
    pub fn color(&self, value: f64, is_diagonal: bool) -> Color {
        let v = value.clamp(0.0, 1.0);
        match self {
            Self::BlueRed => {
                // Blue to red gradient
                let r = (v * 255.0) as u8;
                let b = ((1.0 - v) * 255.0) as u8;
                Color::Rgb(r, 50, b)
            }
            Self::DiagonalGreen => {
                if is_diagonal {
                    // Green for correct predictions (diagonal)
                    let g = 80 + (v * 175.0) as u8;
                    Color::Rgb(50, g, 50)
                } else {
                    // Red for errors (off-diagonal)
                    let r = 80 + (v * 175.0) as u8;
                    Color::Rgb(r, 50, 50)
                }
            }
            Self::Grayscale => {
                let g = (50.0 + v * 150.0) as u8;
                Color::Rgb(g, g, g)
            }
            Self::Blues => {
                // Light blue to dark blue
                let intensity = (50.0 + v * 180.0) as u8;
                Color::Rgb(50, intensity, 150 + (v * 105.0) as u8)
            }
        }
    }

    /// Get text color (for contrast) based on cell background.
    #[must_use]
    pub fn text_color(&self, value: f64) -> Color {
        if value > 0.5 {
            Color::Black // Dark text on light background
        } else {
            Color::White // Light text on dark background
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
    /// Show accuracy footer.
    show_accuracy: bool,
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
            show_accuracy: true,
        }
    }

    /// Set class labels.
    #[must_use]
    pub fn labels(mut self, labels: Vec<String>) -> Self {
        self.labels = labels;
        self
    }

    /// Set normalization mode.
    #[must_use]
    pub fn normalization(mut self, normalization: Normalization) -> Self {
        self.normalization = normalization;
        self
    }

    /// Set color palette.
    #[must_use]
    pub fn palette(mut self, palette: MatrixPalette) -> Self {
        self.palette = palette;
        self
    }

    /// Set cell width (minimum 3).
    #[must_use]
    pub fn cell_width(mut self, width: usize) -> Self {
        self.cell_width = width.max(3);
        self
    }

    /// Show or hide values in cells.
    #[must_use]
    pub fn show_values(mut self, show: bool) -> Self {
        self.show_values = show;
        self
    }

    /// Show percentages instead of counts.
    #[must_use]
    pub fn show_percentages(mut self, show: bool) -> Self {
        self.show_percentages = show;
        self
    }

    /// Set title.
    #[must_use]
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Show or hide accuracy footer.
    #[must_use]
    pub fn show_accuracy(mut self, show: bool) -> Self {
        self.show_accuracy = show;
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
        let correct: u64 =
            self.matrix.iter().enumerate().map(|(i, row)| row.get(i).copied().unwrap_or(0)).sum();
        correct as f64 / total as f64
    }

    /// Get precision for a class (true positives / predicted positives).
    #[must_use]
    pub fn precision(&self, class: usize) -> f64 {
        let col_sum: u64 = self.matrix.iter().map(|row| row.get(class).copied().unwrap_or(0)).sum();
        if col_sum == 0 {
            return 0.0;
        }
        self.matrix.get(class).and_then(|row| row.get(class)).copied().unwrap_or(0) as f64
            / col_sum as f64
    }

    /// Get recall for a class (true positives / actual positives).
    #[must_use]
    pub fn recall(&self, class: usize) -> f64 {
        let row_sum: u64 = self.matrix.get(class).map_or(0, |row| row.iter().sum());
        if row_sum == 0 {
            return 0.0;
        }
        self.matrix.get(class).and_then(|row| row.get(class)).copied().unwrap_or(0) as f64
            / row_sum as f64
    }

    /// Get F1 score for a class (harmonic mean of precision and recall).
    #[must_use]
    pub fn f1_score(&self, class: usize) -> f64 {
        let p = self.precision(class);
        let r = self.recall(class);
        if p + r == 0.0 {
            return 0.0;
        }
        2.0 * p * r / (p + r)
    }

    /// Get macro-averaged F1 score across all classes.
    #[must_use]
    pub fn macro_f1(&self) -> f64 {
        if self.matrix.is_empty() {
            return 0.0;
        }
        let sum: f64 = (0..self.size()).map(|i| self.f1_score(i)).sum();
        sum / self.size() as f64
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
                let col_sum: u64 =
                    self.matrix.iter().map(|r| r.get(col).copied().unwrap_or(0)).sum();
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
        self.labels.iter().map(String::len).max().unwrap_or(3).max(3)
    }
}

impl Widget for ConfusionMatrix {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if self.matrix.is_empty() || area.width < 5 || area.height < 3 {
            return;
        }

        let label_w = self.label_width();
        let n = self.size();
        let mut y = area.y;

        let header_style = Style::default().fg(Color::White);
        let dim_style = Style::default().fg(Color::DarkGray);

        // Draw title
        if let Some(ref title) = self.title {
            for (i, ch) in title.chars().enumerate() {
                let x = area.x + i as u16;
                if x < area.x + area.width {
                    if let Some(cell) = buf.cell_mut((x, y)) {
                        cell.set_char(ch).set_style(header_style);
                    }
                }
            }
            y += 1;
            if y >= area.y + area.height {
                return;
            }
        }

        // Draw header row (predicted labels)
        let header_x = area.x + label_w as u16 + 2;
        let pred_label = "Pred→";
        for (i, ch) in pred_label.chars().enumerate() {
            let x = area.x + i as u16;
            if x < area.x + area.width {
                if let Some(cell) = buf.cell_mut((x, y)) {
                    cell.set_char(ch).set_style(dim_style);
                }
            }
        }

        for (i, label) in self.labels.iter().enumerate().take(n) {
            let x = header_x + (i * (self.cell_width + 1)) as u16;
            let truncated: String = label.chars().take(self.cell_width).collect();
            for (j, ch) in truncated.chars().enumerate() {
                let cell_x = x + j as u16;
                if cell_x < area.x + area.width {
                    if let Some(cell) = buf.cell_mut((cell_x, y)) {
                        cell.set_char(ch).set_style(header_style);
                    }
                }
            }
        }
        y += 1;

        // Draw matrix rows
        for (row_idx, row) in self.matrix.iter().enumerate().take(n) {
            if y >= area.y + area.height {
                break;
            }

            // Row label (actual)
            let label = self.labels.get(row_idx).map_or("?", String::as_str);
            let truncated: String = label.chars().take(label_w).collect();
            for (i, ch) in truncated.chars().enumerate() {
                let x = area.x + i as u16;
                if x < area.x + area.width {
                    if let Some(cell) = buf.cell_mut((x, y)) {
                        cell.set_char(ch).set_style(header_style);
                    }
                }
            }

            // Cells
            for (col_idx, &value) in row.iter().enumerate().take(n) {
                let x = header_x + (col_idx * (self.cell_width + 1)) as u16;
                let normalized = self.normalize_value(row_idx, col_idx, value);
                let is_diagonal = row_idx == col_idx;

                // Draw cell background
                let bg_color = self.palette.color(normalized, is_diagonal);
                let text_color = self.palette.text_color(normalized);
                let cell_style = Style::default().bg(bg_color).fg(text_color);

                // Fill cell with background
                for cell_offset in 0..self.cell_width {
                    let cell_x = x + cell_offset as u16;
                    if cell_x < area.x + area.width {
                        if let Some(cell) = buf.cell_mut((cell_x, y)) {
                            cell.set_char(' ').set_style(cell_style);
                        }
                    }
                }

                // Draw value
                if self.show_values {
                    let text = self.format_value(value, normalized);
                    let text_truncated: String = text.chars().take(self.cell_width).collect();
                    for (j, ch) in text_truncated.chars().enumerate() {
                        let cell_x = x + j as u16;
                        if cell_x < area.x + area.width {
                            if let Some(cell) = buf.cell_mut((cell_x, y)) {
                                cell.set_char(ch).set_style(cell_style);
                            }
                        }
                    }
                }
            }
            y += 1;
        }

        // Draw accuracy footer
        if self.show_accuracy && y < area.y + area.height {
            let accuracy = self.accuracy();
            let acc_text = format!("Accuracy: {:.1}%", accuracy * 100.0);
            for (i, ch) in acc_text.chars().enumerate() {
                let x = area.x + i as u16;
                if x < area.x + area.width {
                    if let Some(cell) = buf.cell_mut((x, y)) {
                        cell.set_char(ch).set_style(header_style);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod normalization_tests {
        use super::*;

        #[test]
        fn test_default() {
            assert_eq!(Normalization::default(), Normalization::None);
        }

        #[test]
        fn test_equality() {
            assert_eq!(Normalization::Row, Normalization::Row);
            assert_ne!(Normalization::Row, Normalization::Column);
        }
    }

    mod palette_tests {
        use super::*;

        #[test]
        fn test_default() {
            assert_eq!(MatrixPalette::default(), MatrixPalette::BlueRed);
        }

        #[test]
        fn test_blue_red_low() {
            let color = MatrixPalette::BlueRed.color(0.0, false);
            match color {
                Color::Rgb(r, _, b) => {
                    assert!(b > r, "Low value should be more blue");
                }
                _ => panic!("Expected RGB color"),
            }
        }

        #[test]
        fn test_blue_red_high() {
            let color = MatrixPalette::BlueRed.color(1.0, false);
            match color {
                Color::Rgb(r, _, b) => {
                    assert!(r > b, "High value should be more red");
                }
                _ => panic!("Expected RGB color"),
            }
        }

        #[test]
        fn test_diagonal_green_diagonal() {
            let color = MatrixPalette::DiagonalGreen.color(0.8, true);
            match color {
                Color::Rgb(r, g, _) => {
                    assert!(g > r, "Diagonal should be green");
                }
                _ => panic!("Expected RGB color"),
            }
        }

        #[test]
        fn test_diagonal_green_off_diagonal() {
            let color = MatrixPalette::DiagonalGreen.color(0.8, false);
            match color {
                Color::Rgb(r, g, _) => {
                    assert!(r > g, "Off-diagonal should be red");
                }
                _ => panic!("Expected RGB color"),
            }
        }

        #[test]
        fn test_grayscale() {
            let color = MatrixPalette::Grayscale.color(0.5, false);
            match color {
                Color::Rgb(r, g, b) => {
                    assert_eq!(r, g);
                    assert_eq!(g, b);
                }
                _ => panic!("Expected RGB color"),
            }
        }

        #[test]
        fn test_blues() {
            let color = MatrixPalette::Blues.color(0.5, false);
            match color {
                Color::Rgb(r, _, b) => {
                    assert!(b > r, "Blues should have more blue");
                }
                _ => panic!("Expected RGB color"),
            }
        }

        #[test]
        fn test_text_color_low() {
            let color = MatrixPalette::BlueRed.text_color(0.2);
            assert_eq!(color, Color::White);
        }

        #[test]
        fn test_text_color_high() {
            let color = MatrixPalette::BlueRed.text_color(0.8);
            assert_eq!(color, Color::Black);
        }

        #[test]
        fn test_color_clamps_value() {
            // Values outside [0, 1] should be clamped
            let low = MatrixPalette::BlueRed.color(-0.5, false);
            let high = MatrixPalette::BlueRed.color(1.5, false);
            // Should not panic and produce valid colors
            match (low, high) {
                (Color::Rgb(_, _, _), Color::Rgb(_, _, _)) => {}
                _ => panic!("Expected RGB colors"),
            }
        }
    }

    mod matrix_construction_tests {
        use super::*;

        #[test]
        fn test_new() {
            let cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]]);
            assert_eq!(cm.size(), 2);
        }

        #[test]
        fn test_default() {
            let cm = ConfusionMatrix::default();
            assert_eq!(cm.size(), 1);
        }

        #[test]
        fn test_labels() {
            let cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]])
                .labels(vec!["Cat".to_string(), "Dog".to_string()]);
            assert_eq!(cm.labels.len(), 2);
            assert_eq!(cm.labels[0], "Cat");
        }

        #[test]
        fn test_normalization() {
            let cm = ConfusionMatrix::new(vec![vec![5]]).normalization(Normalization::Row);
            assert_eq!(cm.normalization, Normalization::Row);
        }

        #[test]
        fn test_palette() {
            let cm = ConfusionMatrix::new(vec![vec![5]]).palette(MatrixPalette::DiagonalGreen);
            assert_eq!(cm.palette, MatrixPalette::DiagonalGreen);
        }

        #[test]
        fn test_cell_width() {
            let cm = ConfusionMatrix::new(vec![vec![5]]).cell_width(10);
            assert_eq!(cm.cell_width, 10);
        }

        #[test]
        fn test_cell_width_minimum() {
            let cm = ConfusionMatrix::new(vec![vec![5]]).cell_width(1);
            assert_eq!(cm.cell_width, 3);
        }

        #[test]
        fn test_show_values() {
            let cm = ConfusionMatrix::new(vec![vec![5]]).show_values(false);
            assert!(!cm.show_values);
        }

        #[test]
        fn test_show_percentages() {
            let cm = ConfusionMatrix::new(vec![vec![5]]).show_percentages(true);
            assert!(cm.show_percentages);
        }

        #[test]
        fn test_title() {
            let cm = ConfusionMatrix::new(vec![vec![5]]).title("Test");
            assert_eq!(cm.title.as_deref(), Some("Test"));
        }

        #[test]
        fn test_show_accuracy() {
            let cm = ConfusionMatrix::new(vec![vec![5]]).show_accuracy(false);
            assert!(!cm.show_accuracy);
        }

        #[test]
        fn test_set_matrix() {
            let mut cm = ConfusionMatrix::new(vec![vec![1]]);
            cm.set_matrix(vec![vec![2, 3], vec![4, 5]]);
            assert_eq!(cm.size(), 2);
        }

        #[test]
        fn test_builder_chaining() {
            let cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]])
                .labels(vec!["A".to_string(), "B".to_string()])
                .normalization(Normalization::Row)
                .palette(MatrixPalette::Blues)
                .cell_width(8)
                .show_values(true)
                .show_percentages(true)
                .title("My Matrix")
                .show_accuracy(true);

            assert_eq!(cm.labels.len(), 2);
            assert_eq!(cm.normalization, Normalization::Row);
            assert_eq!(cm.palette, MatrixPalette::Blues);
            assert_eq!(cm.cell_width, 8);
        }
    }

    mod metrics_tests {
        use super::*;

        #[test]
        fn test_total() {
            let cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]]);
            assert_eq!(cm.total(), 30);
        }

        #[test]
        fn test_total_empty() {
            let cm = ConfusionMatrix::new(vec![]);
            assert_eq!(cm.total(), 0);
        }

        #[test]
        fn test_accuracy() {
            let cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]]);
            // Correct: 10 + 15 = 25, Total: 30
            let acc = cm.accuracy();
            assert!((acc - 0.833).abs() < 0.01);
        }

        #[test]
        fn test_accuracy_zero_total() {
            let cm = ConfusionMatrix::new(vec![vec![0, 0], vec![0, 0]]);
            assert_eq!(cm.accuracy(), 0.0);
        }

        #[test]
        fn test_precision() {
            // Class 0: col sum = 10 + 3 = 13, diagonal = 10
            let cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]]);
            let prec = cm.precision(0);
            assert!((prec - 0.769).abs() < 0.01);
        }

        #[test]
        fn test_precision_zero_column() {
            let cm = ConfusionMatrix::new(vec![vec![0, 5], vec![0, 10]]);
            assert_eq!(cm.precision(0), 0.0);
        }

        #[test]
        fn test_recall() {
            // Class 0: row sum = 10 + 2 = 12, diagonal = 10
            let cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]]);
            let recall = cm.recall(0);
            assert!((recall - 0.833).abs() < 0.01);
        }

        #[test]
        fn test_recall_zero_row() {
            let cm = ConfusionMatrix::new(vec![vec![0, 0], vec![3, 15]]);
            assert_eq!(cm.recall(0), 0.0);
        }

        #[test]
        fn test_f1_score() {
            let cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]]);
            let f1 = cm.f1_score(0);
            assert!(f1 > 0.0 && f1 < 1.0);
        }

        #[test]
        fn test_f1_score_zero() {
            let cm = ConfusionMatrix::new(vec![vec![0, 0], vec![0, 0]]);
            assert_eq!(cm.f1_score(0), 0.0);
        }

        #[test]
        fn test_macro_f1() {
            let cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]]);
            let macro_f1 = cm.macro_f1();
            assert!(macro_f1 > 0.0 && macro_f1 < 1.0);
        }

        #[test]
        fn test_macro_f1_empty() {
            let cm = ConfusionMatrix::new(vec![]);
            assert_eq!(cm.macro_f1(), 0.0);
        }

        #[test]
        fn test_precision_out_of_bounds() {
            let cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]]);
            // Should handle gracefully
            assert_eq!(cm.precision(5), 0.0);
        }

        #[test]
        fn test_recall_out_of_bounds() {
            let cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]]);
            assert_eq!(cm.recall(5), 0.0);
        }
    }

    mod normalization_value_tests {
        use super::*;

        #[test]
        fn test_normalize_none() {
            let cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]]);
            // Max is 15, so 10/15 ≈ 0.667
            let normalized = cm.normalize_value(0, 0, 10);
            assert!((normalized - 0.667).abs() < 0.01);
        }

        #[test]
        fn test_normalize_none_zero_max() {
            let cm = ConfusionMatrix::new(vec![vec![0, 0], vec![0, 0]]);
            assert_eq!(cm.normalize_value(0, 0, 0), 0.0);
        }

        #[test]
        fn test_normalize_row() {
            let cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]])
                .normalization(Normalization::Row);
            // Row 0 sum = 12, so 10/12 ≈ 0.833
            let normalized = cm.normalize_value(0, 0, 10);
            assert!((normalized - 0.833).abs() < 0.01);
        }

        #[test]
        fn test_normalize_row_zero_sum() {
            let cm = ConfusionMatrix::new(vec![vec![0, 0], vec![3, 15]])
                .normalization(Normalization::Row);
            assert_eq!(cm.normalize_value(0, 0, 0), 0.0);
        }

        #[test]
        fn test_normalize_column() {
            let cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]])
                .normalization(Normalization::Column);
            // Col 0 sum = 13, so 10/13 ≈ 0.769
            let normalized = cm.normalize_value(0, 0, 10);
            assert!((normalized - 0.769).abs() < 0.01);
        }

        #[test]
        fn test_normalize_column_zero_sum() {
            let cm = ConfusionMatrix::new(vec![vec![10, 0], vec![3, 0]])
                .normalization(Normalization::Column);
            assert_eq!(cm.normalize_value(0, 1, 0), 0.0);
        }

        #[test]
        fn test_normalize_total() {
            let cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]])
                .normalization(Normalization::Total);
            // Total = 30, so 10/30 ≈ 0.333
            let normalized = cm.normalize_value(0, 0, 10);
            assert!((normalized - 0.333).abs() < 0.01);
        }

        #[test]
        fn test_normalize_total_zero() {
            let cm = ConfusionMatrix::new(vec![vec![0, 0], vec![0, 0]])
                .normalization(Normalization::Total);
            assert_eq!(cm.normalize_value(0, 0, 0), 0.0);
        }
    }

    mod format_tests {
        use super::*;

        #[test]
        fn test_format_value_count() {
            let cm = ConfusionMatrix::new(vec![vec![123]]);
            assert_eq!(cm.format_value(123, 0.5), "123");
        }

        #[test]
        fn test_format_value_percentage() {
            let cm = ConfusionMatrix::new(vec![vec![10]]).show_percentages(true);
            assert_eq!(cm.format_value(10, 0.5), "50%");
        }

        #[test]
        fn test_label_width() {
            let cm = ConfusionMatrix::new(vec![vec![1, 2], vec![3, 4]])
                .labels(vec!["Short".to_string(), "VeryLongLabel".to_string()]);
            assert_eq!(cm.label_width(), 13);
        }

        #[test]
        fn test_label_width_minimum() {
            let cm = ConfusionMatrix::new(vec![vec![1]]).labels(vec!["A".to_string()]);
            assert_eq!(cm.label_width(), 3);
        }
    }

    mod rendering_tests {
        use super::*;

        fn create_test_buffer(width: u16, height: u16) -> (Rect, Buffer) {
            let area = Rect::new(0, 0, width, height);
            let buf = Buffer::empty(area);
            (area, buf)
        }

        #[test]
        fn test_render_empty() {
            let (area, mut buf) = create_test_buffer(50, 20);
            let cm = ConfusionMatrix::new(vec![]);
            cm.render(area, &mut buf);
            // Should not panic
        }

        #[test]
        fn test_render_small_area() {
            let (area, mut buf) = create_test_buffer(3, 2);
            let cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]]);
            cm.render(area, &mut buf);
            // Should not panic with small area
        }

        #[test]
        fn test_render_basic() {
            let (area, mut buf) = create_test_buffer(50, 20);
            let cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]]);
            cm.render(area, &mut buf);
        }

        #[test]
        fn test_render_with_title() {
            let (area, mut buf) = create_test_buffer(50, 20);
            let cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]]).title("Test Matrix");
            cm.render(area, &mut buf);
        }

        #[test]
        fn test_render_with_labels() {
            let (area, mut buf) = create_test_buffer(50, 20);
            let cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]])
                .labels(vec!["Cat".to_string(), "Dog".to_string()]);
            cm.render(area, &mut buf);
        }

        #[test]
        fn test_render_no_values() {
            let (area, mut buf) = create_test_buffer(50, 20);
            let cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]]).show_values(false);
            cm.render(area, &mut buf);
        }

        #[test]
        fn test_render_percentages() {
            let (area, mut buf) = create_test_buffer(50, 20);
            let cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]]).show_percentages(true);
            cm.render(area, &mut buf);
        }

        #[test]
        fn test_render_no_accuracy() {
            let (area, mut buf) = create_test_buffer(50, 20);
            let cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]]).show_accuracy(false);
            cm.render(area, &mut buf);
        }

        #[test]
        fn test_render_all_palettes() {
            let (area, mut buf) = create_test_buffer(50, 20);
            for palette in [
                MatrixPalette::BlueRed,
                MatrixPalette::DiagonalGreen,
                MatrixPalette::Grayscale,
                MatrixPalette::Blues,
            ] {
                let cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]]).palette(palette);
                cm.render(area, &mut buf);
            }
        }

        #[test]
        fn test_render_all_normalizations() {
            let (area, mut buf) = create_test_buffer(50, 20);
            for norm in [
                Normalization::None,
                Normalization::Row,
                Normalization::Column,
                Normalization::Total,
            ] {
                let cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]]).normalization(norm);
                cm.render(area, &mut buf);
            }
        }

        #[test]
        fn test_render_3x3() {
            let (area, mut buf) = create_test_buffer(60, 25);
            let cm = ConfusionMatrix::new(vec![vec![45, 3, 2], vec![5, 40, 5], vec![1, 4, 45]])
                .labels(vec!["A".to_string(), "B".to_string(), "C".to_string()]);
            cm.render(area, &mut buf);
        }

        #[test]
        fn test_render_tight_height() {
            // Height just enough for header + 2 rows (no accuracy)
            let (area, mut buf) = create_test_buffer(50, 4);
            let cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]]).show_accuracy(false);
            cm.render(area, &mut buf);
        }

        #[test]
        fn test_render_narrow_width() {
            // Very narrow - tests boundary clipping
            let (area, mut buf) = create_test_buffer(15, 10);
            let cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]])
                .labels(vec!["Cat".to_string(), "Dog".to_string()])
                .title("Matrix");
            cm.render(area, &mut buf);
        }

        #[test]
        fn test_render_very_narrow() {
            // Extremely narrow - cell clipping
            let (area, mut buf) = create_test_buffer(10, 10);
            let cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]]);
            cm.render(area, &mut buf);
        }

        #[test]
        fn test_render_title_clipping() {
            // Title longer than width
            let (area, mut buf) = create_test_buffer(20, 10);
            let cm = ConfusionMatrix::new(vec![vec![10, 2], vec![3, 15]])
                .title("Very Long Title That Will Be Clipped");
            cm.render(area, &mut buf);
        }

        #[test]
        fn test_render_height_truncation() {
            // Not enough height for all rows
            let (area, mut buf) = create_test_buffer(50, 3);
            let cm = ConfusionMatrix::new(vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]])
                .title("Test");
            cm.render(area, &mut buf);
        }

        #[test]
        fn test_render_many_classes() {
            // 5x5 matrix in medium area
            let (area, mut buf) = create_test_buffer(60, 12);
            let cm = ConfusionMatrix::new(vec![
                vec![10, 1, 0, 0, 0],
                vec![1, 10, 1, 0, 0],
                vec![0, 1, 10, 1, 0],
                vec![0, 0, 1, 10, 1],
                vec![0, 0, 0, 1, 10],
            ]);
            cm.render(area, &mut buf);
        }

        #[test]
        fn test_render_with_all_options() {
            let (area, mut buf) = create_test_buffer(70, 15);
            let cm = ConfusionMatrix::new(vec![vec![50, 10], vec![5, 35]])
                .labels(vec!["Positive".to_string(), "Negative".to_string()])
                .title("Classification Results")
                .normalization(Normalization::Row)
                .show_percentages(true)
                .show_accuracy(true)
                .cell_width(10);
            cm.render(area, &mut buf);
        }
    }
}
