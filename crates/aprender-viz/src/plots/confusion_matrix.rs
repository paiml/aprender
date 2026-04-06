//! Confusion matrix visualization for classification evaluation.
//!
//! A confusion matrix is a table used to describe the performance of a
//! classification model on a set of test data for which the true values
//! are known.
//!
//! # References
//!
//! - Stehman, S. V. (1997). "Selecting and interpreting measures of thematic
//!   classification accuracy." Remote Sensing of Environment, 62(1), 77-89.

use crate::color::Rgba;
use crate::error::{Error, Result};
use crate::framebuffer::Framebuffer;
use crate::scale::{ColorScale, Scale};

/// Normalization mode for confusion matrix values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Normalization {
    /// No normalization (raw counts).
    #[default]
    None,
    /// Normalize by row (shows recall/sensitivity).
    Row,
    /// Normalize by column (shows precision).
    Column,
    /// Normalize by total (shows overall distribution).
    All,
}

/// Builder for creating confusion matrix visualizations.
#[derive(Debug, Clone)]
pub struct ConfusionMatrix {
    /// Confusion matrix data (row-major, shape: classes x classes).
    /// Row = actual class, Column = predicted class.
    data: Vec<u32>,
    /// Number of classes.
    num_classes: usize,
    /// Class labels (optional).
    labels: Vec<String>,
    /// Normalization mode.
    normalization: Normalization,
    /// Output width in pixels.
    width: u32,
    /// Output height in pixels.
    height: u32,
    /// Margin around the matrix.
    margin: u32,
    /// Show cell borders.
    show_borders: bool,
    /// Border color.
    border_color: Rgba,
    /// Border width.
    border_width: u32,
    /// Color scale for cells.
    color_scale: Option<ColorScale>,
}

impl Default for ConfusionMatrix {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfusionMatrix {
    /// Create a new confusion matrix builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            num_classes: 0,
            labels: Vec::new(),
            normalization: Normalization::default(),
            width: 600,
            height: 600,
            margin: 60,
            show_borders: true,
            border_color: Rgba::rgb(100, 100, 100),
            border_width: 1,
            color_scale: None,
        }
    }

    /// Set the confusion matrix data from a flat array.
    ///
    /// Data should be in row-major order where:
    /// - Rows represent actual/true classes
    /// - Columns represent predicted classes
    #[must_use]
    pub fn data(mut self, data: &[u32], num_classes: usize) -> Self {
        self.data = data.to_vec();
        self.num_classes = num_classes;
        self
    }

    /// Set the confusion matrix data from predictions and true labels.
    ///
    /// Both arrays should contain class indices (0 to num_classes-1).
    #[must_use]
    pub fn from_predictions(
        mut self,
        y_true: &[usize],
        y_pred: &[usize],
        num_classes: usize,
    ) -> Self {
        self.num_classes = num_classes;
        self.data = vec![0; num_classes * num_classes];

        for (&true_class, &pred_class) in y_true.iter().zip(y_pred.iter()) {
            if true_class < num_classes && pred_class < num_classes {
                let idx = true_class * num_classes + pred_class;
                self.data[idx] += 1;
            }
        }

        self
    }

    /// Set the confusion matrix data from a 2D vector.
    #[must_use]
    pub fn data_2d(mut self, matrix: &[Vec<u32>]) -> Self {
        if matrix.is_empty() {
            return self;
        }

        self.num_classes = matrix.len();
        self.data = matrix.iter().flatten().copied().collect();
        self
    }

    /// Set class labels.
    #[must_use]
    pub fn labels(mut self, labels: &[impl AsRef<str>]) -> Self {
        self.labels = labels.iter().map(|s| s.as_ref().to_string()).collect();
        self
    }

    /// Set the normalization mode.
    #[must_use]
    pub fn normalize(mut self, mode: Normalization) -> Self {
        self.normalization = mode;
        self
    }

    /// Set the margin around the matrix.
    #[must_use]
    pub fn margin(mut self, margin: u32) -> Self {
        self.margin = margin;
        self
    }

    /// Enable or disable cell borders.
    #[must_use]
    pub fn borders(mut self, show: bool) -> Self {
        self.show_borders = show;
        self
    }

    /// Set a custom color scale.
    #[must_use]
    pub fn color_scale(mut self, scale: ColorScale) -> Self {
        self.color_scale = Some(scale);
        self
    }

    /// Build and validate the confusion matrix.
    ///
    /// # Errors
    ///
    /// Returns an error if data is empty or dimensions don't match.
    pub fn build(self) -> Result<Self> {
        if self.data.is_empty() || self.num_classes == 0 {
            return Err(Error::EmptyData);
        }

        let expected_len = self.num_classes * self.num_classes;
        if self.data.len() != expected_len {
            return Err(Error::DataLengthMismatch { x_len: expected_len, y_len: self.data.len() });
        }

        Ok(self)
    }

    /// Get the normalized values based on the normalization mode.
    fn normalized_values(&self) -> Vec<f32> {
        let n = self.num_classes;
        let mut normalized = vec![0.0; n * n];

        match self.normalization {
            Normalization::None => {
                // Just convert to f32
                for (i, &v) in self.data.iter().enumerate() {
                    normalized[i] = v as f32;
                }
            }
            Normalization::Row => {
                // Normalize by row (divide each cell by row sum)
                for row in 0..n {
                    let row_sum: u32 = (0..n).map(|col| self.data[row * n + col]).sum();
                    if row_sum > 0 {
                        for col in 0..n {
                            normalized[row * n + col] =
                                self.data[row * n + col] as f32 / row_sum as f32;
                        }
                    }
                }
            }
            Normalization::Column => {
                // Normalize by column (divide each cell by column sum)
                for col in 0..n {
                    let col_sum: u32 = (0..n).map(|row| self.data[row * n + col]).sum();
                    if col_sum > 0 {
                        for row in 0..n {
                            normalized[row * n + col] =
                                self.data[row * n + col] as f32 / col_sum as f32;
                        }
                    }
                }
            }
            Normalization::All => {
                // Normalize by total
                let total: u32 = self.data.iter().sum();
                if total > 0 {
                    for (i, &v) in self.data.iter().enumerate() {
                        normalized[i] = v as f32 / total as f32;
                    }
                }
            }
        }

        normalized
    }

    /// Get the value extent for color scaling.
    fn value_extent(&self) -> (f32, f32) {
        let normalized = self.normalized_values();
        let min = normalized.iter().copied().fold(f32::INFINITY, f32::min);
        let max = normalized.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Ensure we have a valid range
        if (max - min).abs() < f32::EPSILON {
            (0.0, max.max(1.0))
        } else {
            (min, max)
        }
    }

    /// Create a color scale for the matrix.
    fn create_color_scale(&self) -> Option<ColorScale> {
        if let Some(ref scale) = self.color_scale {
            return Some(scale.clone());
        }

        let (min, max) = self.value_extent();
        ColorScale::blues((min, max))
    }

    /// Calculate derived metrics from the confusion matrix.
    #[must_use]
    pub fn metrics(&self) -> ConfusionMatrixMetrics {
        let n = self.num_classes;

        // True positives for each class
        let true_positives: Vec<u32> = (0..n).map(|i| self.data[i * n + i]).collect();

        // False positives for each class (column sum - diagonal)
        let false_positives: Vec<u32> = (0..n)
            .map(|col| {
                let col_sum: u32 = (0..n).map(|row| self.data[row * n + col]).sum();
                col_sum - self.data[col * n + col]
            })
            .collect();

        // False negatives for each class (row sum - diagonal)
        let false_negatives: Vec<u32> = (0..n)
            .map(|row| {
                let row_sum: u32 = (0..n).map(|col| self.data[row * n + col]).sum();
                row_sum - self.data[row * n + row]
            })
            .collect();

        // Overall accuracy
        let total: u32 = self.data.iter().sum();
        let correct: u32 = true_positives.iter().sum();
        let accuracy = if total > 0 { correct as f32 / total as f32 } else { 0.0 };

        // Per-class precision and recall
        let precision: Vec<f32> = (0..n)
            .map(|i| {
                let tp = true_positives[i] as f32;
                let fp = false_positives[i] as f32;
                if tp + fp > 0.0 {
                    tp / (tp + fp)
                } else {
                    0.0
                }
            })
            .collect();

        let recall: Vec<f32> = (0..n)
            .map(|i| {
                let tp = true_positives[i] as f32;
                let fn_ = false_negatives[i] as f32;
                if tp + fn_ > 0.0 {
                    tp / (tp + fn_)
                } else {
                    0.0
                }
            })
            .collect();

        ConfusionMatrixMetrics {
            accuracy,
            precision,
            recall,
            true_positives,
            false_positives,
            false_negatives,
        }
    }

    /// Render the confusion matrix to a framebuffer.
    ///
    /// # Errors
    ///
    /// Returns an error if rendering fails.
    pub fn render(&self, fb: &mut Framebuffer) -> Result<()> {
        let color_scale = self.create_color_scale().ok_or(Error::EmptyData)?;
        let normalized = self.normalized_values();

        let n = self.num_classes;

        // Calculate cell dimensions (use saturating subtraction to prevent overflow)
        let plot_width = self.width.saturating_sub(2 * self.margin);
        let plot_height = self.height.saturating_sub(2 * self.margin);

        // Ensure we have valid dimensions
        if plot_width == 0 || plot_height == 0 {
            return Ok(()); // Nothing to render
        }

        let cell_width = plot_width / n as u32;
        let cell_height = plot_height / n as u32;

        if cell_width == 0 || cell_height == 0 {
            return Ok(()); // Cells too small to render
        }

        // Render cells
        for row in 0..n {
            for col in 0..n {
                let value = normalized[row * n + col];
                let color = color_scale.scale(value);

                let x = self.margin + (col as u32) * cell_width;
                let y = self.margin + (row as u32) * cell_height;

                // Draw filled cell
                fb.fill_rect(x, y, cell_width, cell_height, color);

                // Draw border if enabled
                if self.show_borders && self.border_width > 0 {
                    self.draw_cell_border(fb, x, y, cell_width, cell_height);
                }
            }
        }

        // Highlight diagonal (true positives) with a subtle overlay
        self.highlight_diagonal(fb, cell_width, cell_height);

        Ok(())
    }

    /// Draw a cell border.
    fn draw_cell_border(&self, fb: &mut Framebuffer, x: u32, y: u32, width: u32, height: u32) {
        let bw = self.border_width;

        // Right border
        if x + width <= fb.width() {
            fb.fill_rect(x + width - bw, y, bw, height, self.border_color);
        }

        // Bottom border
        if y + height <= fb.height() {
            fb.fill_rect(x, y + height - bw, width, bw, self.border_color);
        }
    }

    /// Highlight diagonal cells with a subtle border.
    fn highlight_diagonal(&self, fb: &mut Framebuffer, cell_width: u32, cell_height: u32) {
        let n = self.num_classes;
        let highlight_color = Rgba::rgb(50, 50, 50);

        for i in 0..n {
            let x = self.margin + (i as u32) * cell_width;
            let y = self.margin + (i as u32) * cell_height;

            // Draw a darker border around diagonal cells
            fb.fill_rect(x, y, cell_width, 2, highlight_color);
            fb.fill_rect(x, y + cell_height - 2, cell_width, 2, highlight_color);
            fb.fill_rect(x, y, 2, cell_height, highlight_color);
            fb.fill_rect(x + cell_width - 2, y, 2, cell_height, highlight_color);
        }
    }

    /// Render to a new framebuffer.
    ///
    /// # Errors
    ///
    /// Returns an error if rendering fails.
    pub fn to_framebuffer(&self) -> Result<Framebuffer> {
        let mut fb = Framebuffer::new(self.width, self.height)?;
        fb.clear(Rgba::WHITE);
        self.render(&mut fb)?;
        Ok(fb)
    }

    /// Get the number of classes.
    #[must_use]
    pub const fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// Get the total count.
    #[must_use]
    pub fn total(&self) -> u32 {
        self.data.iter().sum()
    }
}

impl batuta_common::display::WithDimensions for ConfusionMatrix {
    fn set_dimensions(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }
}

/// Metrics derived from a confusion matrix.
#[derive(Debug, Clone)]
pub struct ConfusionMatrixMetrics {
    /// Overall accuracy.
    pub accuracy: f32,
    /// Per-class precision.
    pub precision: Vec<f32>,
    /// Per-class recall (sensitivity).
    pub recall: Vec<f32>,
    /// Per-class true positives.
    pub true_positives: Vec<u32>,
    /// Per-class false positives.
    pub false_positives: Vec<u32>,
    /// Per-class false negatives.
    pub false_negatives: Vec<u32>,
}

impl ConfusionMatrixMetrics {
    /// Calculate F1 score for each class.
    #[must_use]
    pub fn f1_scores(&self) -> Vec<f32> {
        self.precision
            .iter()
            .zip(&self.recall)
            .map(|(&p, &r)| if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 })
            .collect()
    }

    /// Calculate macro-averaged F1 score.
    #[must_use]
    pub fn macro_f1(&self) -> f32 {
        let scores = self.f1_scores();
        if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f32>() / scores.len() as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use batuta_common::display::WithDimensions;

    #[test]
    fn test_confusion_matrix_builder() {
        let data = vec![50, 10, 5, 45];
        let cm = ConfusionMatrix::new()
            .data(&data, 2)
            .build()
            .expect("builder should produce valid result");

        assert_eq!(cm.num_classes(), 2);
        assert_eq!(cm.total(), 110);
    }

    #[test]
    fn test_confusion_matrix_from_predictions() {
        let y_true = vec![0, 0, 1, 1, 1, 0];
        let y_pred = vec![0, 1, 1, 1, 0, 0];

        let cm = ConfusionMatrix::new()
            .from_predictions(&y_true, &y_pred, 2)
            .build()
            .expect("builder should produce valid result");

        assert_eq!(cm.num_classes(), 2);
        assert_eq!(cm.total(), 6);
    }

    #[test]
    fn test_confusion_matrix_empty_data() {
        let result = ConfusionMatrix::new().build();
        assert!(result.is_err());
    }

    #[test]
    fn test_confusion_matrix_dimension_mismatch() {
        let data = vec![1, 2, 3]; // 3 elements, but 2x2 = 4 expected
        let result = ConfusionMatrix::new().data(&data, 2).build();
        assert!(result.is_err());
    }

    #[test]
    fn test_confusion_matrix_render() {
        let data = vec![50, 10, 5, 45];
        let cm = ConfusionMatrix::new()
            .data(&data, 2)
            .dimensions(200, 200)
            .margin(20)
            .build()
            .expect("builder should produce valid result");

        let fb = cm.to_framebuffer();
        assert!(fb.is_ok());
    }

    #[test]
    fn test_confusion_matrix_normalization_row() {
        let data = vec![80, 20, 10, 90];
        let cm = ConfusionMatrix::new()
            .data(&data, 2)
            .normalize(Normalization::Row)
            .build()
            .expect("builder should produce valid result");

        let normalized = cm.normalized_values();
        // Row 0: 80/(80+20) = 0.8, 20/(80+20) = 0.2
        assert!((normalized[0] - 0.8).abs() < 0.001);
        assert!((normalized[1] - 0.2).abs() < 0.001);
        // Row 1: 10/(10+90) = 0.1, 90/(10+90) = 0.9
        assert!((normalized[2] - 0.1).abs() < 0.001);
        assert!((normalized[3] - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_confusion_matrix_normalization_column() {
        let data = vec![80, 20, 10, 90];
        let cm = ConfusionMatrix::new()
            .data(&data, 2)
            .normalize(Normalization::Column)
            .build()
            .expect("builder should produce valid result");

        let normalized = cm.normalized_values();
        // Col 0: 80/(80+10) ≈ 0.889, 10/(80+10) ≈ 0.111
        assert!((normalized[0] - 80.0 / 90.0).abs() < 0.001);
        assert!((normalized[2] - 10.0 / 90.0).abs() < 0.001);
    }

    #[test]
    fn test_confusion_matrix_metrics() {
        // Binary classification: TP=50, FP=10, FN=5, TN=35
        let data = vec![50, 10, 5, 35];
        let cm = ConfusionMatrix::new()
            .data(&data, 2)
            .build()
            .expect("builder should produce valid result");

        let metrics = cm.metrics();

        // Accuracy = (50 + 35) / 100 = 0.85
        assert!((metrics.accuracy - 0.85).abs() < 0.001);

        // Class 0: precision = 50/(50+5) ≈ 0.909
        assert!((metrics.precision[0] - 50.0 / 55.0).abs() < 0.001);

        // Class 0: recall = 50/(50+10) ≈ 0.833
        assert!((metrics.recall[0] - 50.0 / 60.0).abs() < 0.001);
    }

    #[test]
    fn test_confusion_matrix_f1_scores() {
        let data = vec![50, 10, 5, 35];
        let cm = ConfusionMatrix::new()
            .data(&data, 2)
            .build()
            .expect("builder should produce valid result");

        let metrics = cm.metrics();
        let f1_scores = metrics.f1_scores();

        // F1 = 2 * precision * recall / (precision + recall)
        let expected_f1 = 2.0 * (50.0 / 55.0) * (50.0 / 60.0) / ((50.0 / 55.0) + (50.0 / 60.0));
        assert!((f1_scores[0] - expected_f1).abs() < 0.001);
    }

    #[test]
    fn test_confusion_matrix_data_2d() {
        let matrix = vec![vec![10, 2], vec![3, 15]];
        let cm = ConfusionMatrix::new()
            .data_2d(&matrix)
            .build()
            .expect("builder should produce valid result");

        assert_eq!(cm.num_classes(), 2);
        assert_eq!(cm.total(), 30);
    }

    #[test]
    fn test_confusion_matrix_with_labels() {
        let data = vec![50, 10, 5, 35];
        let cm = ConfusionMatrix::new()
            .data(&data, 2)
            .labels(&["Positive", "Negative"])
            .build()
            .expect("operation should succeed");

        assert_eq!(cm.num_classes(), 2);
    }

    #[test]
    fn test_confusion_matrix_multiclass() {
        let data = vec![
            30, 5, 2, // Class 0: 30 correct, 5 predicted as 1, 2 predicted as 2
            3, 28, 4, // Class 1: 3 predicted as 0, 28 correct, 4 predicted as 2
            1, 2, 25, // Class 2: 1 predicted as 0, 2 predicted as 1, 25 correct
        ];

        let cm = ConfusionMatrix::new()
            .data(&data, 3)
            .dimensions(150, 150)
            .build()
            .expect("builder should produce valid result");

        assert_eq!(cm.num_classes(), 3);

        let metrics = cm.metrics();
        // Overall accuracy = (30 + 28 + 25) / 100 = 0.83
        assert!((metrics.accuracy - 0.83).abs() < 0.001);

        let fb = cm.to_framebuffer();
        assert!(fb.is_ok());
    }

    #[test]
    fn test_normalization_all() {
        let data = vec![50, 10, 5, 35];
        let cm = ConfusionMatrix::new()
            .data(&data, 2)
            .normalize(Normalization::All)
            .build()
            .expect("builder should produce valid result");

        let normalized = cm.normalized_values();
        // Total = 100, so each cell is divided by 100
        assert!((normalized[0] - 0.50).abs() < 0.001);
        assert!((normalized[1] - 0.10).abs() < 0.001);
        assert!((normalized[2] - 0.05).abs() < 0.001);
        assert!((normalized[3] - 0.35).abs() < 0.001);
    }

    #[test]
    fn test_normalization_none() {
        let data = vec![50, 10, 5, 35];
        let cm = ConfusionMatrix::new()
            .data(&data, 2)
            .normalize(Normalization::None)
            .build()
            .expect("builder should produce valid result");

        let normalized = cm.normalized_values();
        // No normalization - values should be as-is (converted to f64)
        assert!((normalized[0] - 50.0).abs() < 0.001);
        assert!((normalized[1] - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_confusion_matrix_border_options() {
        let data = vec![50, 10, 5, 35];
        let cm = ConfusionMatrix::new()
            .data(&data, 2)
            .borders(false)
            .build()
            .expect("builder should produce valid result");

        let fb = cm.to_framebuffer();
        assert!(fb.is_ok());
    }

    #[test]
    fn test_confusion_matrix_color_scale() {
        let data = vec![50, 10, 5, 35];
        let scale = ColorScale::viridis((0.0, 100.0)).expect("operation should succeed");
        let cm = ConfusionMatrix::new()
            .data(&data, 2)
            .color_scale(scale)
            .build()
            .expect("builder should produce valid result");

        let fb = cm.to_framebuffer();
        assert!(fb.is_ok());
    }

    #[test]
    fn test_normalization_default() {
        assert_eq!(Normalization::default(), Normalization::None);
    }

    #[test]
    fn test_confusion_matrix_default() {
        let cm = ConfusionMatrix::default();
        // Default should have empty data
        let result = cm.build();
        assert!(result.is_err());
    }

    #[test]
    fn test_confusion_matrix_clone_debug() {
        let data = vec![50, 10, 5, 35];
        let cm = ConfusionMatrix::new().data(&data, 2);
        let cloned = cm.clone();
        let debug = format!("{cloned:?}");
        assert!(debug.contains("ConfusionMatrix"));
    }

    #[test]
    fn test_normalization_debug_clone_eq() {
        let norm = Normalization::Row;
        let cloned = norm;
        assert_eq!(norm, cloned);

        let debug = format!("{norm:?}");
        assert!(debug.contains("Row"));
    }

    #[test]
    fn test_predictions_valid() {
        let y_true = vec![0, 1, 2, 0, 1];
        let y_pred = vec![0, 1, 2, 1, 1];

        let cm = ConfusionMatrix::new()
            .from_predictions(&y_true, &y_pred, 3)
            .build()
            .expect("builder should produce valid result");

        assert_eq!(cm.num_classes(), 3);
        assert_eq!(cm.total(), 5);
    }

    #[test]
    fn test_empty_2d_matrix() {
        let matrix: Vec<Vec<u32>> = vec![];
        let result = ConfusionMatrix::new().data_2d(&matrix).build();
        assert!(result.is_err());
    }
}
