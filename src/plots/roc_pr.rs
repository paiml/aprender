//! ROC and Precision-Recall curve visualization.
//!
//! ROC (Receiver Operating Characteristic) and PR (Precision-Recall) curves
//! are essential tools for evaluating binary classifiers across different
//! decision thresholds.
//!
//! # References
//!
//! - Fawcett, T. (2006). "An introduction to ROC analysis." Pattern Recognition
//!   Letters, 27(8), 861-874.
//! - Davis, J., & Goadrich, M. (2006). "The relationship between Precision-Recall
//!   and ROC curves." ICML '06.

use crate::color::Rgba;
use crate::error::{Error, Result};
use crate::framebuffer::Framebuffer;
use crate::render::draw_line_aa;
use crate::scale::{LinearScale, Scale};

/// A point on a curve (x, y coordinates with associated threshold).
#[derive(Debug, Clone, Copy)]
pub struct CurvePoint {
    /// X coordinate (e.g., FPR for ROC, Recall for PR).
    pub x: f32,
    /// Y coordinate (e.g., TPR for ROC, Precision for PR).
    pub y: f32,
    /// Threshold value that produced this point.
    pub threshold: f32,
}

/// Computed ROC curve data.
#[derive(Debug, Clone)]
pub struct RocData {
    /// Points on the ROC curve (FPR, TPR pairs).
    pub points: Vec<CurvePoint>,
    /// Area under the ROC curve.
    pub auc: f32,
}

/// Computed Precision-Recall curve data.
#[derive(Debug, Clone)]
pub struct PrData {
    /// Points on the PR curve (Recall, Precision pairs).
    pub points: Vec<CurvePoint>,
    /// Average Precision (area under PR curve).
    pub average_precision: f32,
}

/// Compute ROC curve from prediction scores and binary labels.
///
/// # Arguments
///
/// * `y_true` - Binary ground truth labels (0 or 1)
/// * `y_scores` - Prediction scores (higher = more likely positive)
///
/// # Returns
///
/// ROC curve data including points and AUC.
pub fn compute_roc(y_true: &[u8], y_scores: &[f32]) -> Result<RocData> {
    if y_true.len() != y_scores.len() {
        return Err(Error::DataLengthMismatch { x_len: y_true.len(), y_len: y_scores.len() });
    }

    if y_true.is_empty() {
        return Err(Error::EmptyData);
    }

    // Count total positives and negatives
    let total_positives = y_true.iter().fold(0u32, |acc, &b| acc + u32::from(b == 1)) as f32;
    let total_negatives = y_true.len() as f32 - total_positives;

    if total_positives == 0.0 || total_negatives == 0.0 {
        return Err(Error::ScaleDomain("Need both positive and negative samples".to_string()));
    }

    // Get sorted indices by score (descending)
    let mut indices: Vec<usize> = (0..y_scores.len()).collect();
    indices.sort_by(|&a, &b| {
        y_scores[b].partial_cmp(&y_scores[a]).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Calculate TPR and FPR at each threshold
    let mut points = Vec::with_capacity(y_scores.len() + 2);
    let mut tp = 0.0;
    let mut fp = 0.0;

    // Start at (0, 0) with threshold above max
    let max_score = y_scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    points.push(CurvePoint { x: 0.0, y: 0.0, threshold: max_score + 1.0 });

    for &idx in &indices {
        if y_true[idx] == 1 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }

        let tpr = tp / total_positives;
        let fpr = fp / total_negatives;

        points.push(CurvePoint { x: fpr, y: tpr, threshold: y_scores[idx] });
    }

    // Ensure we end at (1, 1)
    if let Some(last) = points.last() {
        if last.x < 1.0 || last.y < 1.0 {
            points.push(CurvePoint { x: 1.0, y: 1.0, threshold: f32::NEG_INFINITY });
        }
    }

    // Calculate AUC using trapezoidal rule
    let auc = calculate_auc(&points);

    Ok(RocData { points, auc })
}

/// Compute Precision-Recall curve from prediction scores and binary labels.
///
/// # Arguments
///
/// * `y_true` - Binary ground truth labels (0 or 1)
/// * `y_scores` - Prediction scores (higher = more likely positive)
///
/// # Returns
///
/// PR curve data including points and average precision.
pub fn compute_pr(y_true: &[u8], y_scores: &[f32]) -> Result<PrData> {
    if y_true.len() != y_scores.len() {
        return Err(Error::DataLengthMismatch { x_len: y_true.len(), y_len: y_scores.len() });
    }

    if y_true.is_empty() {
        return Err(Error::EmptyData);
    }

    // Count total positives
    let total_positives = y_true.iter().fold(0u32, |acc, &b| acc + u32::from(b == 1)) as f32;

    if total_positives == 0.0 {
        return Err(Error::ScaleDomain("Need at least one positive sample".to_string()));
    }

    // Get sorted indices by score (descending)
    let mut indices: Vec<usize> = (0..y_scores.len()).collect();
    indices.sort_by(|&a, &b| {
        y_scores[b].partial_cmp(&y_scores[a]).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Calculate Precision and Recall at each threshold
    let mut points = Vec::with_capacity(y_scores.len() + 1);
    let mut tp = 0.0;
    let mut fp = 0.0;

    // Start at (0, 1) - at highest threshold, precision is undefined but we use 1
    let max_score = y_scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    points.push(CurvePoint { x: 0.0, y: 1.0, threshold: max_score + 1.0 });

    for &idx in &indices {
        if y_true[idx] == 1 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }

        let recall = tp / total_positives;
        let precision = tp / (tp + fp);

        points.push(CurvePoint { x: recall, y: precision, threshold: y_scores[idx] });
    }

    // Calculate Average Precision (area under interpolated PR curve)
    let average_precision = calculate_average_precision(&points);

    Ok(PrData { points, average_precision })
}

/// Calculate AUC using trapezoidal rule.
fn calculate_auc(points: &[CurvePoint]) -> f32 {
    if points.len() < 2 {
        return 0.0;
    }

    let mut auc = 0.0;
    for i in 1..points.len() {
        let dx = points[i].x - points[i - 1].x;
        let avg_y = (points[i].y + points[i - 1].y) / 2.0;
        auc += dx * avg_y;
    }

    auc.clamp(0.0, 1.0)
}

/// Calculate Average Precision for PR curve.
fn calculate_average_precision(points: &[CurvePoint]) -> f32 {
    if points.len() < 2 {
        return 0.0;
    }

    let mut ap = 0.0;
    for i in 1..points.len() {
        let delta_recall = points[i].x - points[i - 1].x;
        // Use precision at point i (right-hand rule)
        ap += delta_recall * points[i].y;
    }

    ap.clamp(0.0, 1.0)
}

// ============================================================================
// ROC Curve Visualization
// ============================================================================

/// Builder for ROC curve visualization.
#[derive(Debug, Clone)]
pub struct RocCurve {
    /// ROC data to visualize.
    data: Option<RocData>,
    /// Line color.
    color: Rgba,
    /// Show diagonal reference line.
    show_diagonal: bool,
    /// Diagonal line color.
    diagonal_color: Rgba,
    /// Output width.
    width: u32,
    /// Output height.
    height: u32,
    /// Margin.
    margin: u32,
}

impl Default for RocCurve {
    fn default() -> Self {
        Self::new()
    }
}

impl RocCurve {
    /// Create a new ROC curve builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: None,
            color: Rgba::BLUE,
            show_diagonal: true,
            diagonal_color: Rgba::rgb(200, 200, 200),
            width: 600,
            height: 600,
            margin: 40,
        }
    }

    /// Set the ROC data directly.
    #[must_use]
    pub fn data(mut self, roc_data: RocData) -> Self {
        self.data = Some(roc_data);
        self
    }

    /// Compute ROC from predictions and labels.
    ///
    /// # Errors
    ///
    /// Returns error if computation fails.
    pub fn from_predictions(mut self, y_true: &[u8], y_scores: &[f32]) -> Result<Self> {
        self.data = Some(compute_roc(y_true, y_scores)?);
        Ok(self)
    }

    /// Set the line color.
    #[must_use]
    pub fn color(mut self, color: Rgba) -> Self {
        self.color = color;
        self
    }

    /// Show or hide the diagonal reference line.
    #[must_use]
    pub fn diagonal(mut self, show: bool) -> Self {
        self.show_diagonal = show;
        self
    }

    /// Build and validate.
    pub fn build(self) -> Result<Self> {
        if self.data.is_none() {
            return Err(Error::EmptyData);
        }
        Ok(self)
    }

    /// Get the AUC value.
    #[must_use]
    pub fn auc(&self) -> f32 {
        self.data.as_ref().map_or(0.0, |d| d.auc)
    }

    /// Render the ROC curve.
    pub fn render(&self, fb: &mut Framebuffer) -> Result<()> {
        let roc_data = self.data.as_ref().ok_or(Error::EmptyData)?;

        let plot_size = self.width.min(self.height) - 2 * self.margin;
        let x_scale =
            LinearScale::new((0.0, 1.0), (self.margin as f32, (self.margin + plot_size) as f32))?;
        let y_scale =
            LinearScale::new((0.0, 1.0), ((self.margin + plot_size) as f32, self.margin as f32))?;

        // Draw diagonal reference line
        if self.show_diagonal {
            let x0 = x_scale.scale(0.0);
            let y0 = y_scale.scale(0.0);
            let x1 = x_scale.scale(1.0);
            let y1 = y_scale.scale(1.0);
            draw_line_aa(fb, x0, y0, x1, y1, self.diagonal_color);
        }

        // Draw ROC curve
        for i in 1..roc_data.points.len() {
            let p0 = &roc_data.points[i - 1];
            let p1 = &roc_data.points[i];

            let x0 = x_scale.scale(p0.x);
            let y0 = y_scale.scale(p0.y);
            let x1 = x_scale.scale(p1.x);
            let y1 = y_scale.scale(p1.y);

            draw_line_aa(fb, x0, y0, x1, y1, self.color);
        }

        Ok(())
    }

    /// Render to a new framebuffer.
    pub fn to_framebuffer(&self) -> Result<Framebuffer> {
        let mut fb = Framebuffer::new(self.width, self.height)?;
        fb.clear(Rgba::WHITE);
        self.render(&mut fb)?;
        Ok(fb)
    }
}

// ============================================================================
// PR Curve Visualization
// ============================================================================

/// Builder for Precision-Recall curve visualization.
#[derive(Debug, Clone)]
pub struct PrCurve {
    /// PR data to visualize.
    data: Option<PrData>,
    /// Line color.
    color: Rgba,
    /// Show no-skill reference line (horizontal at positive rate).
    show_baseline: bool,
    /// Baseline color.
    baseline_color: Rgba,
    /// Positive class rate (for baseline).
    positive_rate: f32,
    /// Output width.
    width: u32,
    /// Output height.
    height: u32,
    /// Margin.
    margin: u32,
}

impl Default for PrCurve {
    fn default() -> Self {
        Self::new()
    }
}

impl PrCurve {
    /// Create a new PR curve builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: None,
            color: Rgba::rgb(0, 128, 0), // Green
            show_baseline: true,
            baseline_color: Rgba::rgb(200, 200, 200),
            positive_rate: 0.5,
            width: 600,
            height: 600,
            margin: 40,
        }
    }

    /// Set the PR data directly.
    #[must_use]
    pub fn data(mut self, pr_data: PrData) -> Self {
        self.data = Some(pr_data);
        self
    }

    /// Compute PR curve from predictions and labels.
    ///
    /// # Errors
    ///
    /// Returns error if computation fails.
    pub fn from_predictions(mut self, y_true: &[u8], y_scores: &[f32]) -> Result<Self> {
        // Calculate positive rate for baseline
        let total_positives = y_true.iter().fold(0u32, |acc, &b| acc + u32::from(b == 1)) as f32;
        self.positive_rate = total_positives / y_true.len() as f32;

        self.data = Some(compute_pr(y_true, y_scores)?);
        Ok(self)
    }

    /// Set the line color.
    #[must_use]
    pub fn color(mut self, color: Rgba) -> Self {
        self.color = color;
        self
    }

    /// Show or hide the baseline reference line.
    #[must_use]
    pub fn baseline(mut self, show: bool) -> Self {
        self.show_baseline = show;
        self
    }

    /// Build and validate.
    pub fn build(self) -> Result<Self> {
        if self.data.is_none() {
            return Err(Error::EmptyData);
        }
        Ok(self)
    }

    /// Get the average precision value.
    #[must_use]
    pub fn average_precision(&self) -> f32 {
        self.data.as_ref().map_or(0.0, |d| d.average_precision)
    }

    /// Render the PR curve.
    pub fn render(&self, fb: &mut Framebuffer) -> Result<()> {
        let pr_data = self.data.as_ref().ok_or(Error::EmptyData)?;

        let plot_size = self.width.min(self.height) - 2 * self.margin;
        let x_scale =
            LinearScale::new((0.0, 1.0), (self.margin as f32, (self.margin + plot_size) as f32))?;
        let y_scale =
            LinearScale::new((0.0, 1.0), ((self.margin + plot_size) as f32, self.margin as f32))?;

        // Draw baseline reference line (horizontal at positive rate)
        if self.show_baseline {
            let x0 = x_scale.scale(0.0);
            let y = y_scale.scale(self.positive_rate);
            let x1 = x_scale.scale(1.0);
            draw_line_aa(fb, x0, y, x1, y, self.baseline_color);
        }

        // Draw PR curve
        for i in 1..pr_data.points.len() {
            let p0 = &pr_data.points[i - 1];
            let p1 = &pr_data.points[i];

            let x0 = x_scale.scale(p0.x);
            let y0 = y_scale.scale(p0.y);
            let x1 = x_scale.scale(p1.x);
            let y1 = y_scale.scale(p1.y);

            draw_line_aa(fb, x0, y0, x1, y1, self.color);
        }

        Ok(())
    }

    /// Render to a new framebuffer.
    pub fn to_framebuffer(&self) -> Result<Framebuffer> {
        let mut fb = Framebuffer::new(self.width, self.height)?;
        fb.clear(Rgba::WHITE);
        self.render(&mut fb)?;
        Ok(fb)
    }
}

// ============================================================================
// Tests
// ============================================================================

impl batuta_common::display::WithDimensions for RocCurve {
    fn set_dimensions(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }
}

impl batuta_common::display::WithDimensions for PrCurve {
    fn set_dimensions(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use batuta_common::display::WithDimensions;

    #[test]
    fn test_compute_roc_basic() {
        // Simple test case
        let y_true = vec![0, 0, 1, 1, 1];
        let y_scores = vec![0.1, 0.4, 0.35, 0.8, 0.9];

        let roc = compute_roc(&y_true, &y_scores).expect("operation should succeed");

        // AUC should be between 0 and 1
        assert!(roc.auc >= 0.0 && roc.auc <= 1.0);
        // Should have multiple points
        assert!(roc.points.len() > 2);
        // First point should be (0, 0) or close
        assert!(roc.points[0].x < 0.01);
        assert!(roc.points[0].y < 0.01);
    }

    #[test]
    fn test_compute_roc_perfect() {
        // Perfect classifier
        let y_true = vec![0, 0, 0, 1, 1, 1];
        let y_scores = vec![0.1, 0.2, 0.3, 0.7, 0.8, 0.9];

        let roc = compute_roc(&y_true, &y_scores).expect("operation should succeed");

        // Perfect classifier should have AUC = 1.0
        assert!((roc.auc - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_roc_random() {
        // Random classifier (scores don't correlate with labels)
        let y_true = vec![0, 1, 0, 1, 0, 1, 0, 1];
        let y_scores = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];

        let roc = compute_roc(&y_true, &y_scores).expect("operation should succeed");

        // Random classifier should have AUC ≈ 0.5
        assert!((roc.auc - 0.5).abs() < 0.2);
    }

    #[test]
    fn test_compute_pr_basic() {
        let y_true = vec![0, 0, 1, 1, 1];
        let y_scores = vec![0.1, 0.4, 0.35, 0.8, 0.9];

        let pr = compute_pr(&y_true, &y_scores).expect("operation should succeed");

        // Average precision should be between 0 and 1
        assert!(pr.average_precision >= 0.0 && pr.average_precision <= 1.0);
        // Should have multiple points
        assert!(pr.points.len() > 2);
    }

    #[test]
    fn test_compute_roc_empty() {
        let y_true: Vec<u8> = vec![];
        let y_scores: Vec<f32> = vec![];

        let result = compute_roc(&y_true, &y_scores);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_roc_length_mismatch() {
        let y_true = vec![0, 1, 1];
        let y_scores = vec![0.5, 0.6];

        let result = compute_roc(&y_true, &y_scores);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_roc_no_positives() {
        let y_true = vec![0, 0, 0, 0];
        let y_scores = vec![0.1, 0.2, 0.3, 0.4];

        let result = compute_roc(&y_true, &y_scores);
        assert!(result.is_err());
    }

    #[test]
    fn test_roc_curve_render() {
        let y_true = vec![0, 0, 1, 1, 1, 0, 1, 0];
        let y_scores = vec![0.1, 0.3, 0.4, 0.8, 0.9, 0.2, 0.6, 0.35];

        let roc = RocCurve::new()
            .from_predictions(&y_true, &y_scores)
            .expect("operation should succeed")
            .dimensions(200, 200)
            .build()
            .expect("operation should succeed");

        let fb = roc.to_framebuffer();
        assert!(fb.is_ok());
    }

    #[test]
    fn test_pr_curve_render() {
        let y_true = vec![0, 0, 1, 1, 1, 0, 1, 0];
        let y_scores = vec![0.1, 0.3, 0.4, 0.8, 0.9, 0.2, 0.6, 0.35];

        let pr = PrCurve::new()
            .from_predictions(&y_true, &y_scores)
            .expect("operation should succeed")
            .dimensions(200, 200)
            .build()
            .expect("operation should succeed");

        let fb = pr.to_framebuffer();
        assert!(fb.is_ok());
    }

    #[test]
    fn test_roc_auc_getter() {
        let y_true = vec![0, 0, 1, 1];
        let y_scores = vec![0.1, 0.2, 0.8, 0.9];

        let roc = RocCurve::new()
            .from_predictions(&y_true, &y_scores)
            .expect("builder should produce valid result")
            .build()
            .expect("builder should produce valid result");

        let auc = roc.auc();
        assert!(auc > 0.5);
    }

    #[test]
    fn test_pr_average_precision_getter() {
        let y_true = vec![0, 0, 1, 1];
        let y_scores = vec![0.1, 0.2, 0.8, 0.9];

        let pr = PrCurve::new()
            .from_predictions(&y_true, &y_scores)
            .expect("builder should produce valid result")
            .build()
            .expect("builder should produce valid result");

        let ap = pr.average_precision();
        assert!(ap > 0.5);
    }

    #[test]
    fn test_auc_calculation() {
        // Simple test: points forming a triangle with known area
        let points = vec![
            CurvePoint { x: 0.0, y: 0.0, threshold: 1.0 },
            CurvePoint { x: 1.0, y: 1.0, threshold: 0.0 },
        ];

        let auc = calculate_auc(&points);
        // Triangle area = 0.5
        assert!((auc - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_roc_curve_with_data() {
        let roc_data = RocData {
            points: vec![
                CurvePoint { x: 0.0, y: 0.0, threshold: 1.0 },
                CurvePoint { x: 0.0, y: 0.5, threshold: 0.8 },
                CurvePoint { x: 0.5, y: 1.0, threshold: 0.5 },
                CurvePoint { x: 1.0, y: 1.0, threshold: 0.0 },
            ],
            auc: 0.875,
        };

        let roc = RocCurve::new()
            .data(roc_data)
            .dimensions(200, 200)
            .build()
            .expect("builder should produce valid result");

        assert!((roc.auc() - 0.875).abs() < 0.001);
    }

    #[test]
    fn test_roc_curve_color() {
        let y_true = vec![0, 0, 1, 1];
        let y_scores = vec![0.1, 0.2, 0.8, 0.9];

        let roc = RocCurve::new()
            .from_predictions(&y_true, &y_scores)
            .expect("operation should succeed")
            .color(Rgba::RED)
            .dimensions(200, 200)
            .build()
            .expect("operation should succeed");

        let fb = roc.to_framebuffer();
        assert!(fb.is_ok());
    }

    #[test]
    fn test_roc_curve_diagonal() {
        let y_true = vec![0, 0, 1, 1];
        let y_scores = vec![0.1, 0.2, 0.8, 0.9];

        let roc = RocCurve::new()
            .from_predictions(&y_true, &y_scores)
            .expect("operation should succeed")
            .diagonal(false)
            .dimensions(200, 200)
            .build()
            .expect("operation should succeed");

        let fb = roc.to_framebuffer();
        assert!(fb.is_ok());
    }

    #[test]
    fn test_pr_curve_color() {
        let y_true = vec![0, 0, 1, 1];
        let y_scores = vec![0.1, 0.2, 0.8, 0.9];

        let pr = PrCurve::new()
            .from_predictions(&y_true, &y_scores)
            .expect("operation should succeed")
            .color(Rgba::GREEN)
            .dimensions(200, 200)
            .build()
            .expect("operation should succeed");

        let fb = pr.to_framebuffer();
        assert!(fb.is_ok());
    }

    #[test]
    fn test_pr_curve_baseline() {
        let y_true = vec![0, 0, 1, 1];
        let y_scores = vec![0.1, 0.2, 0.8, 0.9];

        let pr = PrCurve::new()
            .from_predictions(&y_true, &y_scores)
            .expect("operation should succeed")
            .baseline(false)
            .dimensions(200, 200)
            .build()
            .expect("operation should succeed");

        let fb = pr.to_framebuffer();
        assert!(fb.is_ok());
    }

    #[test]
    fn test_pr_curve_with_data() {
        let pr_data = PrData {
            points: vec![
                CurvePoint { x: 0.0, y: 1.0, threshold: 1.0 },
                CurvePoint { x: 0.5, y: 0.8, threshold: 0.5 },
                CurvePoint { x: 1.0, y: 0.5, threshold: 0.0 },
            ],
            average_precision: 0.75,
        };

        let pr = PrCurve::new()
            .data(pr_data)
            .dimensions(200, 200)
            .build()
            .expect("builder should produce valid result");

        assert!((pr.average_precision() - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_curve_point_clone_debug() {
        let point = CurvePoint { x: 0.5, y: 0.8, threshold: 0.7 };
        let cloned = point;
        let debug = format!("{cloned:?}");
        assert!(debug.contains("CurvePoint"));
        assert!((point.x - cloned.x).abs() < f32::EPSILON);
    }

    #[test]
    fn test_roc_data_clone_debug() {
        let data =
            RocData { points: vec![CurvePoint { x: 0.0, y: 0.0, threshold: 1.0 }], auc: 0.8 };
        let cloned = data.clone();
        let debug = format!("{cloned:?}");
        assert!(debug.contains("RocData"));
    }

    #[test]
    fn test_pr_data_clone_debug() {
        let data = PrData {
            points: vec![CurvePoint { x: 0.0, y: 1.0, threshold: 1.0 }],
            average_precision: 0.9,
        };
        let cloned = data.clone();
        let debug = format!("{cloned:?}");
        assert!(debug.contains("PrData"));
    }

    #[test]
    fn test_roc_no_negatives() {
        let y_true = vec![1, 1, 1, 1];
        let y_scores = vec![0.1, 0.2, 0.8, 0.9];

        let result = compute_roc(&y_true, &y_scores);
        assert!(result.is_err());
    }

    #[test]
    fn test_pr_empty() {
        let y_true: Vec<u8> = vec![];
        let y_scores: Vec<f32> = vec![];

        let result = compute_pr(&y_true, &y_scores);
        assert!(result.is_err());
    }

    #[test]
    fn test_pr_length_mismatch() {
        let y_true = vec![0, 1, 1];
        let y_scores = vec![0.5, 0.6];

        let result = compute_pr(&y_true, &y_scores);
        assert!(result.is_err());
    }

    #[test]
    fn test_pr_no_positives() {
        let y_true = vec![0, 0, 0, 0];
        let y_scores = vec![0.1, 0.2, 0.3, 0.4];

        let result = compute_pr(&y_true, &y_scores);
        assert!(result.is_err());
    }
}
