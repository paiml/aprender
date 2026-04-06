//! Aprender ML library integration.
//!
//! Provides visualization extensions for aprender types (Vector, Matrix, DataFrame).
//!
//! # Examples
//!
//! ```rust,ignore
//! use aprender::primitives::{Vector, Matrix};
//! use trueno_viz::interop::aprender::VectorViz;
//!
//! let predictions = Vector::from_slice(&[2.5, 4.1, 3.9, 5.2]);
//! let actual = Vector::from_slice(&[2.0, 4.0, 4.0, 5.0]);
//!
//! // Visualize predictions vs actual
//! let fb = predictions.scatter_vs(&actual)?;
//! ```

use aprender::data::DataFrame as AprenderDataFrame;
use aprender::primitives::{Matrix, Vector};
use batuta_common::display::WithDimensions;

use crate::color::Rgba;
use crate::error::Result;
use crate::framebuffer::Framebuffer;
use crate::plots::{
    BoxPlot, Heatmap, HeatmapPalette, Histogram, LineChart, LineSeries, ScatterPlot,
};

// ============================================================================
// Vector Visualization Extensions
// ============================================================================

/// Visualization extensions for aprender Vector.
pub trait VectorViz {
    /// Create a histogram of the vector values.
    fn to_histogram(&self) -> Result<Framebuffer>;

    /// Create a histogram with custom options.
    fn to_histogram_with(&self, width: u32, height: u32, color: Rgba) -> Result<Framebuffer>;

    /// Create a scatter plot comparing two vectors (self = predictions, other = actual).
    fn scatter_vs(&self, other: &Vector<f32>) -> Result<Framebuffer>;

    /// Create a scatter plot with custom options.
    fn scatter_vs_with(
        &self,
        other: &Vector<f32>,
        width: u32,
        height: u32,
        color: Rgba,
    ) -> Result<Framebuffer>;

    /// Create a line plot of the vector values (index as x-axis).
    fn to_line(&self) -> Result<Framebuffer>;

    /// Create a residual plot (predicted - actual vs actual).
    fn residual_plot(&self, actual: &Vector<f32>) -> Result<Framebuffer>;
}

impl VectorViz for Vector<f32> {
    fn to_histogram(&self) -> Result<Framebuffer> {
        self.to_histogram_with(600, 400, Rgba::new(70, 130, 180, 255))
    }

    fn to_histogram_with(&self, width: u32, height: u32, color: Rgba) -> Result<Framebuffer> {
        let plot = Histogram::new()
            .data(self.as_slice())
            .color(color)
            .dimensions(width, height)
            .build()?;

        plot.to_framebuffer()
    }

    fn scatter_vs(&self, other: &Vector<f32>) -> Result<Framebuffer> {
        self.scatter_vs_with(other, 600, 600, Rgba::new(66, 133, 244, 255))
    }

    fn scatter_vs_with(
        &self,
        other: &Vector<f32>,
        width: u32,
        height: u32,
        color: Rgba,
    ) -> Result<Framebuffer> {
        let plot = ScatterPlot::new()
            .x(other.as_slice()) // actual on x-axis
            .y(self.as_slice()) // predicted on y-axis
            .color(color)
            .size(5.0)
            .dimensions(width, height)
            .build()?;

        plot.to_framebuffer()
    }

    fn to_line(&self) -> Result<Framebuffer> {
        let x: Vec<f32> = (0..self.len()).map(|i| i as f32).collect();

        let plot = LineChart::new()
            .add_series(
                LineSeries::new("data")
                    .data(&x, self.as_slice())
                    .color(Rgba::new(66, 133, 244, 255)),
            )
            .dimensions(600, 400)
            .build()?;

        plot.to_framebuffer()
    }

    fn residual_plot(&self, actual: &Vector<f32>) -> Result<Framebuffer> {
        let n = self.len().min(actual.len());
        let residuals: Vec<f32> = self.as_slice()[..n]
            .iter()
            .zip(actual.as_slice()[..n].iter())
            .map(|(p, a)| p - a)
            .collect();

        let plot = ScatterPlot::new()
            .x(&actual.as_slice()[..n])
            .y(&residuals)
            .color(Rgba::new(234, 67, 53, 255))
            .size(5.0)
            .dimensions(600, 400)
            .build()?;

        plot.to_framebuffer()
    }
}

// ============================================================================
// Matrix Visualization Extensions
// ============================================================================

/// Visualization extensions for aprender Matrix.
pub trait MatrixViz {
    /// Create a heatmap of the matrix.
    fn to_heatmap(&self) -> Result<Framebuffer>;

    /// Create a heatmap with custom palette.
    fn to_heatmap_with(&self, palette: HeatmapPalette) -> Result<Framebuffer>;

    /// Create a correlation heatmap (assumes square correlation matrix).
    fn correlation_heatmap(&self) -> Result<Framebuffer>;
}

impl MatrixViz for Matrix<f32> {
    fn to_heatmap(&self) -> Result<Framebuffer> {
        self.to_heatmap_with(HeatmapPalette::Viridis)
    }

    fn to_heatmap_with(&self, palette: HeatmapPalette) -> Result<Framebuffer> {
        let (rows, cols) = self.shape();

        let plot = Heatmap::new()
            .data(self.as_slice(), rows, cols)
            .palette(palette)
            .dimensions(600, 500)
            .build()?;

        plot.to_framebuffer()
    }

    fn correlation_heatmap(&self) -> Result<Framebuffer> {
        let (rows, cols) = self.shape();

        let plot = Heatmap::new()
            .data(self.as_slice(), rows, cols)
            .palette(HeatmapPalette::RedBlue)
            .dimensions(600, 600)
            .build()?;

        plot.to_framebuffer()
    }
}

// ============================================================================
// DataFrame Visualization Extensions
// ============================================================================

/// Visualization extensions for aprender DataFrame.
pub trait DataFrameViz {
    /// Create a scatter plot of two columns.
    fn scatter(&self, x_col: &str, y_col: &str) -> Result<Framebuffer>;

    /// Create a histogram of a column.
    fn histogram(&self, col: &str) -> Result<Framebuffer>;

    /// Create a box plot of multiple columns.
    fn boxplot(&self, cols: &[&str]) -> Result<Framebuffer>;

    /// Create a line chart of a column (index as x-axis).
    fn line(&self, col: &str) -> Result<Framebuffer>;

    /// Create a correlation matrix heatmap.
    fn correlation_matrix(&self) -> Result<Framebuffer>;
}

impl DataFrameViz for AprenderDataFrame {
    fn scatter(&self, x_col: &str, y_col: &str) -> Result<Framebuffer> {
        let x = self
            .column(x_col)
            .map_err(|e| crate::error::Error::Rendering(format!("Column '{x_col}': {e}")))?;
        let y = self
            .column(y_col)
            .map_err(|e| crate::error::Error::Rendering(format!("Column '{y_col}': {e}")))?;

        let plot = ScatterPlot::new()
            .x(x.as_slice())
            .y(y.as_slice())
            .color(Rgba::new(66, 133, 244, 255))
            .size(5.0)
            .dimensions(600, 500)
            .build()?;

        plot.to_framebuffer()
    }

    fn histogram(&self, col: &str) -> Result<Framebuffer> {
        let data = self
            .column(col)
            .map_err(|e| crate::error::Error::Rendering(format!("Column '{col}': {e}")))?;

        let plot = Histogram::new()
            .data(data.as_slice())
            .color(Rgba::new(70, 130, 180, 255))
            .dimensions(600, 400)
            .build()?;

        plot.to_framebuffer()
    }

    fn boxplot(&self, cols: &[&str]) -> Result<Framebuffer> {
        let mut plot = BoxPlot::new().dimensions(600, 400);

        for col_name in cols {
            if let Ok(col) = self.column(col_name) {
                plot = plot.add_group(col.as_slice(), col_name);
            }
        }

        let built = plot.build()?;
        built.to_framebuffer()
    }

    fn line(&self, col: &str) -> Result<Framebuffer> {
        let data = self
            .column(col)
            .map_err(|e| crate::error::Error::Rendering(format!("Column '{col}': {e}")))?;

        let x: Vec<f32> = (0..data.len()).map(|i| i as f32).collect();

        let plot = LineChart::new()
            .add_series(
                LineSeries::new(col).data(&x, data.as_slice()).color(Rgba::new(66, 133, 244, 255)),
            )
            .dimensions(600, 400)
            .build()?;

        plot.to_framebuffer()
    }

    fn correlation_matrix(&self) -> Result<Framebuffer> {
        // Compute correlation matrix
        let n_cols = self.n_cols();
        let n_rows = self.n_rows();

        if n_rows < 2 {
            return Err(crate::error::Error::Rendering(
                "Need at least 2 rows for correlation".into(),
            ));
        }

        // Collect columns into a vec for indexed access
        let columns: Vec<(&str, &Vector<f32>)> = self.iter_columns().collect();

        let mut corr_data = vec![0.0f32; n_cols * n_cols];

        for i in 0..n_cols {
            for j in 0..n_cols {
                let corr = if i == j {
                    1.0
                } else {
                    let (_, col_i) = columns[i];
                    let (_, col_j) = columns[j];
                    pearson_correlation(col_i.as_slice(), col_j.as_slice())
                };
                corr_data[i * n_cols + j] = corr;
            }
        }

        let plot = Heatmap::new()
            .data(&corr_data, n_cols, n_cols)
            .palette(HeatmapPalette::RedBlue)
            .dimensions(600, 600)
            .build()?;

        plot.to_framebuffer()
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute Pearson correlation coefficient.
fn pearson_correlation(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }

    let x_mean: f32 = x[..n].iter().sum::<f32>() / n as f32;
    let y_mean: f32 = y[..n].iter().sum::<f32>() / n as f32;

    let mut cov = 0.0f32;
    let mut var_x = 0.0f32;
    let mut var_y = 0.0f32;

    for i in 0..n {
        let dx = x[i] - x_mean;
        let dy = y[i] - y_mean;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < f32::EPSILON {
        0.0
    } else {
        cov / denom
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Create a predictions vs actual scatter plot.
///
/// This is the most common visualization for regression model evaluation.
pub fn predictions_vs_actual(
    predictions: &Vector<f32>,
    actual: &Vector<f32>,
) -> Result<Framebuffer> {
    predictions.scatter_vs(actual)
}

/// Create a residual plot.
///
/// Shows residuals (predicted - actual) vs actual values.
/// Useful for detecting heteroscedasticity and non-linearity.
pub fn residuals(predictions: &Vector<f32>, actual: &Vector<f32>) -> Result<Framebuffer> {
    predictions.residual_plot(actual)
}

/// Create a training loss curve from a vector of loss values.
pub fn loss_curve(losses: &Vector<f32>) -> Result<Framebuffer> {
    losses.to_line()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_histogram() {
        let v = Vector::from_slice(&[1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0]);
        let fb = v.to_histogram().expect("operation should succeed");
        assert_eq!(fb.width(), 600);
        assert_eq!(fb.height(), 400);
    }

    #[test]
    fn test_vector_histogram_with() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let fb = v.to_histogram_with(400, 300, Rgba::RED).expect("operation should succeed");
        assert_eq!(fb.width(), 400);
        assert_eq!(fb.height(), 300);
    }

    #[test]
    fn test_vector_scatter_vs() {
        let pred = Vector::from_slice(&[2.0, 4.0, 3.0, 5.0]);
        let actual = Vector::from_slice(&[2.1, 3.9, 3.1, 4.8]);
        let fb = pred.scatter_vs(&actual).expect("operation should succeed");
        assert_eq!(fb.width(), 600);
        assert_eq!(fb.height(), 600);
    }

    #[test]
    fn test_vector_scatter_vs_with() {
        let pred = Vector::from_slice(&[2.0, 4.0, 3.0, 5.0]);
        let actual = Vector::from_slice(&[2.1, 3.9, 3.1, 4.8]);
        let fb =
            pred.scatter_vs_with(&actual, 500, 500, Rgba::GREEN).expect("operation should succeed");
        assert_eq!(fb.width(), 500);
        assert_eq!(fb.height(), 500);
    }

    #[test]
    fn test_vector_to_line() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 2.5, 4.0, 3.5]);
        let fb = v.to_line().expect("operation should succeed");
        assert_eq!(fb.width(), 600);
        assert_eq!(fb.height(), 400);
    }

    #[test]
    fn test_vector_residual_plot() {
        let pred = Vector::from_slice(&[2.0, 4.0, 3.0, 5.0]);
        let actual = Vector::from_slice(&[2.1, 3.9, 3.1, 4.8]);
        let fb = pred.residual_plot(&actual).expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_matrix_heatmap() {
        let m = Matrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .expect("operation should succeed");
        let fb = m.to_heatmap().expect("operation should succeed");
        assert_eq!(fb.width(), 600);
        assert_eq!(fb.height(), 500);
    }

    #[test]
    fn test_matrix_heatmap_with_palette() {
        let m = Matrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .expect("operation should succeed");
        let fb = m.to_heatmap_with(HeatmapPalette::Magma).expect("operation should succeed");
        assert_eq!(fb.width(), 600);
    }

    #[test]
    fn test_matrix_correlation_heatmap() {
        let m = Matrix::from_vec(3, 3, vec![1.0, 0.5, 0.3, 0.5, 1.0, 0.7, 0.3, 0.7, 1.0])
            .expect("operation should succeed");
        let fb = m.correlation_heatmap().expect("operation should succeed");
        assert_eq!(fb.width(), 600);
        assert_eq!(fb.height(), 600);
    }

    #[test]
    fn test_pearson_correlation() {
        // Perfect positive correlation
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = pearson_correlation(&x, &y);
        assert!((corr - 1.0).abs() < 0.001);

        // Perfect negative correlation
        let y_neg = [10.0, 8.0, 6.0, 4.0, 2.0];
        let corr_neg = pearson_correlation(&x, &y_neg);
        assert!((corr_neg + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_pearson_correlation_short() {
        // Less than 2 elements
        let x = [1.0];
        let y = [2.0];
        let corr = pearson_correlation(&x, &y);
        assert_eq!(corr, 0.0);
    }

    #[test]
    fn test_pearson_correlation_zero_variance() {
        // Constant values = zero variance
        let x = [5.0, 5.0, 5.0, 5.0];
        let y = [1.0, 2.0, 3.0, 4.0];
        let corr = pearson_correlation(&x, &y);
        assert_eq!(corr, 0.0);
    }

    #[test]
    fn test_dataframe_scatter() {
        let columns = vec![
            ("x".to_string(), Vector::from_slice(&[1.0, 2.0, 3.0, 4.0])),
            ("y".to_string(), Vector::from_slice(&[2.0, 4.0, 3.0, 5.0])),
        ];
        let df = AprenderDataFrame::new(columns).expect("rendering should succeed");
        let fb = df.scatter("x", "y").expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_dataframe_scatter_missing_column() {
        let columns = vec![("x".to_string(), Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]))];
        let df = AprenderDataFrame::new(columns).expect("rendering should succeed");
        let result = df.scatter("x", "missing");
        assert!(result.is_err());
    }

    #[test]
    fn test_dataframe_histogram() {
        let columns = vec![(
            "values".to_string(),
            Vector::from_slice(&[1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0]),
        )];
        let df = AprenderDataFrame::new(columns).expect("rendering should succeed");
        let fb = df.histogram("values").expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_dataframe_histogram_missing_column() {
        let columns = vec![("x".to_string(), Vector::from_slice(&[1.0, 2.0, 3.0]))];
        let df = AprenderDataFrame::new(columns).expect("rendering should succeed");
        let result = df.histogram("missing");
        assert!(result.is_err());
    }

    #[test]
    fn test_dataframe_boxplot() {
        let columns = vec![
            ("a".to_string(), Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0])),
            ("b".to_string(), Vector::from_slice(&[2.0, 3.0, 4.0, 5.0, 6.0])),
        ];
        let df = AprenderDataFrame::new(columns).expect("rendering should succeed");
        let fb = df.boxplot(&["a", "b"]).expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_dataframe_line() {
        let columns = vec![("values".to_string(), Vector::from_slice(&[1.0, 2.0, 3.0, 2.5, 4.0]))];
        let df = AprenderDataFrame::new(columns).expect("rendering should succeed");
        let fb = df.line("values").expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_dataframe_line_missing_column() {
        let columns = vec![("x".to_string(), Vector::from_slice(&[1.0, 2.0, 3.0]))];
        let df = AprenderDataFrame::new(columns).expect("rendering should succeed");
        let result = df.line("missing");
        assert!(result.is_err());
    }

    #[test]
    fn test_dataframe_correlation_matrix() {
        let columns = vec![
            ("a".to_string(), Vector::from_slice(&[1.0, 2.0, 3.0, 4.0])),
            ("b".to_string(), Vector::from_slice(&[2.0, 4.0, 6.0, 8.0])),
            ("c".to_string(), Vector::from_slice(&[8.0, 6.0, 4.0, 2.0])),
        ];
        let df = AprenderDataFrame::new(columns).expect("rendering should succeed");
        let fb = df.correlation_matrix().expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_dataframe_correlation_matrix_single_row() {
        let columns = vec![
            ("a".to_string(), Vector::from_slice(&[1.0])),
            ("b".to_string(), Vector::from_slice(&[2.0])),
        ];
        let df = AprenderDataFrame::new(columns).expect("rendering should succeed");
        let result = df.correlation_matrix();
        assert!(result.is_err()); // Need at least 2 rows
    }

    #[test]
    fn test_convenience_predictions_vs_actual() {
        let pred = Vector::from_slice(&[2.0, 4.0, 3.0, 5.0]);
        let actual = Vector::from_slice(&[2.1, 3.9, 3.1, 4.8]);
        let fb = predictions_vs_actual(&pred, &actual).expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_convenience_residuals() {
        let pred = Vector::from_slice(&[2.0, 4.0, 3.0, 5.0]);
        let actual = Vector::from_slice(&[2.1, 3.9, 3.1, 4.8]);
        let fb = residuals(&pred, &actual).expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_convenience_loss_curve() {
        let losses = Vector::from_slice(&[1.0, 0.8, 0.6, 0.4, 0.3, 0.25]);
        let fb = loss_curve(&losses).expect("operation should succeed");
        assert!(fb.width() > 0);
    }
}
