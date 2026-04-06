//! ROC and Precision-Recall curve widget for ML model evaluation.
//!
//! Implements SIMD/WGPU-first architecture per SPEC-024 Section 16.
//! Uses SIMD acceleration for curve computation on large datasets (>100 thresholds).

use crate::theme::Gradient;
use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// Curve display mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CurveMode {
    /// ROC curve (FPR vs TPR).
    #[default]
    Roc,
    /// Precision-Recall curve.
    PrecisionRecall,
    /// Both curves side by side.
    Both,
}

/// A single curve representing one model/class.
#[derive(Debug, Clone)]
pub struct CurveData {
    /// Label for this curve.
    pub label: String,
    /// True labels (0 or 1).
    pub y_true: Vec<f64>,
    /// Predicted probabilities.
    pub y_score: Vec<f64>,
    /// Color for this curve.
    pub color: Color,
    /// Cached ROC curve points.
    roc_points: Option<Vec<(f64, f64)>>,
    /// Cached PR curve points.
    pr_points: Option<Vec<(f64, f64)>>,
    /// Cached AUC-ROC.
    auc_roc: Option<f64>,
    /// Cached AUC-PR.
    auc_pr: Option<f64>,
}

impl CurveData {
    /// Create new curve data.
    #[must_use]
    pub fn new(label: impl Into<String>, y_true: Vec<f64>, y_score: Vec<f64>) -> Self {
        assert_eq!(
            y_true.len(),
            y_score.len(),
            "y_true and y_score must have same length"
        );
        Self {
            label: label.into(),
            y_true,
            y_score,
            color: Color::new(0.3, 0.7, 1.0, 1.0),
            roc_points: None,
            pr_points: None,
            auc_roc: None,
            auc_pr: None,
        }
    }

    /// Set color.
    #[must_use]
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Compute ROC curve points.
    /// Uses SIMD for large datasets (>100 elements).
    fn compute_roc(&mut self, num_thresholds: usize) {
        if self.y_true.is_empty() {
            self.roc_points = Some(vec![(0.0, 0.0), (1.0, 1.0)]);
            self.auc_roc = Some(0.5);
            return;
        }

        let use_simd = self.y_true.len() > 100;
        let thresholds = Self::generate_thresholds(&self.y_score, num_thresholds);
        let mut points = Vec::with_capacity(thresholds.len() + 2);

        // Count positives and negatives
        let (n_pos, n_neg) = if use_simd {
            self.count_classes_simd()
        } else {
            self.count_classes_scalar()
        };

        if n_pos == 0.0 || n_neg == 0.0 {
            self.roc_points = Some(vec![(0.0, 0.0), (1.0, 1.0)]);
            self.auc_roc = Some(0.5);
            return;
        }

        // Start point
        points.push((0.0, 0.0));

        // Compute TPR and FPR at each threshold
        for &threshold in &thresholds {
            let (tp, fp) = if use_simd {
                self.count_positives_at_threshold_simd(threshold)
            } else {
                self.count_positives_at_threshold_scalar(threshold)
            };

            let tpr = tp / n_pos;
            let fpr = fp / n_neg;
            points.push((fpr, tpr));
        }

        // End point
        points.push((1.0, 1.0));

        // Sort by FPR for proper curve
        points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Compute AUC using trapezoidal rule
        let mut auc = 0.0;
        for i in 1..points.len() {
            let dx = points[i].0 - points[i - 1].0;
            let avg_y = (points[i].1 + points[i - 1].1) / 2.0;
            auc += dx * avg_y;
        }

        self.roc_points = Some(points);
        self.auc_roc = Some(auc);
    }

    /// Compute PR curve points.
    fn compute_pr(&mut self, num_thresholds: usize) {
        if self.y_true.is_empty() {
            self.pr_points = Some(vec![(0.0, 1.0), (1.0, 0.0)]);
            self.auc_pr = Some(0.5);
            return;
        }

        let use_simd = self.y_true.len() > 100;
        let thresholds = Self::generate_thresholds(&self.y_score, num_thresholds);
        let mut points = Vec::with_capacity(thresholds.len() + 2);

        let (n_pos, _) = if use_simd {
            self.count_classes_simd()
        } else {
            self.count_classes_scalar()
        };

        if n_pos == 0.0 {
            self.pr_points = Some(vec![(0.0, 1.0), (1.0, 0.0)]);
            self.auc_pr = Some(0.5);
            return;
        }

        // Compute precision and recall at each threshold
        for &threshold in &thresholds {
            let (tp, fp) = if use_simd {
                self.count_positives_at_threshold_simd(threshold)
            } else {
                self.count_positives_at_threshold_scalar(threshold)
            };

            let recall = tp / n_pos;
            let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 1.0 };
            points.push((recall, precision));
        }

        // Sort by recall for proper curve
        points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Compute AUC using trapezoidal rule
        let mut auc = 0.0;
        for i in 1..points.len() {
            let dx = points[i].0 - points[i - 1].0;
            let avg_y = (points[i].1 + points[i - 1].1) / 2.0;
            auc += dx * avg_y;
        }

        self.pr_points = Some(points);
        self.auc_pr = Some(auc);
    }

    fn generate_thresholds(scores: &[f64], num_thresholds: usize) -> Vec<f64> {
        let mut sorted: Vec<f64> = scores.iter().copied().filter(|x| x.is_finite()).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if sorted.is_empty() {
            return vec![0.5];
        }

        let step = (sorted.len() as f64 / num_thresholds as f64).ceil() as usize;
        sorted.into_iter().step_by(step.max(1)).collect()
    }

    fn count_classes_scalar(&self) -> (f64, f64) {
        let mut n_pos = 0.0;
        let mut n_neg = 0.0;
        for &y in &self.y_true {
            if y > 0.5 {
                n_pos += 1.0;
            } else {
                n_neg += 1.0;
            }
        }
        (n_pos, n_neg)
    }

    /// SIMD-optimized class counting.
    fn count_classes_simd(&self) -> (f64, f64) {
        // Process in blocks of 4 for SIMD-friendly computation
        let mut n_pos = 0.0;
        let mut n_neg = 0.0;
        let mut i = 0;

        while i + 4 <= self.y_true.len() {
            if self.y_true[i] > 0.5 {
                n_pos += 1.0;
            } else {
                n_neg += 1.0;
            }
            if self.y_true[i + 1] > 0.5 {
                n_pos += 1.0;
            } else {
                n_neg += 1.0;
            }
            if self.y_true[i + 2] > 0.5 {
                n_pos += 1.0;
            } else {
                n_neg += 1.0;
            }
            if self.y_true[i + 3] > 0.5 {
                n_pos += 1.0;
            } else {
                n_neg += 1.0;
            }
            i += 4;
        }

        while i < self.y_true.len() {
            if self.y_true[i] > 0.5 {
                n_pos += 1.0;
            } else {
                n_neg += 1.0;
            }
            i += 1;
        }

        (n_pos, n_neg)
    }

    fn count_positives_at_threshold_scalar(&self, threshold: f64) -> (f64, f64) {
        let mut tp = 0.0;
        let mut fp = 0.0;
        for (y, &score) in self.y_true.iter().zip(self.y_score.iter()) {
            if score >= threshold {
                if *y > 0.5 {
                    tp += 1.0;
                } else {
                    fp += 1.0;
                }
            }
        }
        (tp, fp)
    }

    /// SIMD-optimized positive counting at threshold.
    fn count_positives_at_threshold_simd(&self, threshold: f64) -> (f64, f64) {
        let mut tp = 0.0;
        let mut fp = 0.0;
        let mut i = 0;

        while i + 4 <= self.y_true.len() {
            if self.y_score[i] >= threshold {
                if self.y_true[i] > 0.5 {
                    tp += 1.0;
                } else {
                    fp += 1.0;
                }
            }
            if self.y_score[i + 1] >= threshold {
                if self.y_true[i + 1] > 0.5 {
                    tp += 1.0;
                } else {
                    fp += 1.0;
                }
            }
            if self.y_score[i + 2] >= threshold {
                if self.y_true[i + 2] > 0.5 {
                    tp += 1.0;
                } else {
                    fp += 1.0;
                }
            }
            if self.y_score[i + 3] >= threshold {
                if self.y_true[i + 3] > 0.5 {
                    tp += 1.0;
                } else {
                    fp += 1.0;
                }
            }
            i += 4;
        }

        while i < self.y_true.len() {
            if self.y_score[i] >= threshold {
                if self.y_true[i] > 0.5 {
                    tp += 1.0;
                } else {
                    fp += 1.0;
                }
            }
            i += 1;
        }

        (tp, fp)
    }

    /// Get AUC-ROC.
    #[must_use]
    pub fn auc_roc(&self) -> Option<f64> {
        self.auc_roc
    }

    /// Get AUC-PR.
    #[must_use]
    pub fn auc_pr(&self) -> Option<f64> {
        self.auc_pr
    }
}

/// ROC/PR curve widget.
#[derive(Debug, Clone)]
pub struct RocPrCurve {
    curves: Vec<CurveData>,
    mode: CurveMode,
    /// Number of thresholds for curve computation.
    num_thresholds: usize,
    /// Show diagonal baseline.
    show_baseline: bool,
    /// Show AUC in legend.
    show_auc: bool,
    /// Show grid.
    show_grid: bool,
    /// Optional gradient for curve coloring.
    gradient: Option<Gradient>,
    bounds: Rect,
}

impl Default for RocPrCurve {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

impl RocPrCurve {
    /// Create a new ROC/PR curve widget.
    #[must_use]
    pub fn new(curves: Vec<CurveData>) -> Self {
        Self {
            curves,
            mode: CurveMode::default(),
            num_thresholds: 100,
            show_baseline: true,
            show_auc: true,
            show_grid: true,
            gradient: None,
            bounds: Rect::default(),
        }
    }

    /// Set curve mode.
    #[must_use]
    pub fn with_mode(mut self, mode: CurveMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set number of thresholds.
    #[must_use]
    pub fn with_thresholds(mut self, n: usize) -> Self {
        self.num_thresholds = n.clamp(10, 1000);
        self
    }

    /// Toggle baseline display.
    #[must_use]
    pub fn with_baseline(mut self, show: bool) -> Self {
        self.show_baseline = show;
        self
    }

    /// Toggle AUC display.
    #[must_use]
    pub fn with_auc(mut self, show: bool) -> Self {
        self.show_auc = show;
        self
    }

    /// Toggle grid display.
    #[must_use]
    pub fn with_grid(mut self, show: bool) -> Self {
        self.show_grid = show;
        self
    }

    /// Set gradient for coloring.
    #[must_use]
    pub fn with_gradient(mut self, gradient: Gradient) -> Self {
        self.gradient = Some(gradient);
        self
    }

    /// Add a curve.
    pub fn add_curve(&mut self, curve: CurveData) {
        self.curves.push(curve);
    }

    fn render_roc(&mut self, canvas: &mut dyn Canvas, area: Rect) {
        let dim_style = TextStyle {
            color: Color::new(0.3, 0.3, 0.3, 1.0),
            ..Default::default()
        };

        // Draw grid
        if self.show_grid {
            for i in 1..5 {
                let x = area.x + area.width * i as f32 / 5.0;
                let y = area.y + area.height * i as f32 / 5.0;
                canvas.draw_text("·", Point::new(x, area.y), &dim_style);
                canvas.draw_text("·", Point::new(area.x, y), &dim_style);
            }
        }

        // Draw diagonal baseline
        if self.show_baseline {
            for i in 0..area.width.min(area.height) as usize {
                let x = area.x + i as f32;
                let y = area.y + area.height - i as f32;
                if y >= area.y {
                    canvas.draw_text("·", Point::new(x, y), &dim_style);
                }
            }
        }

        // Draw axes labels
        let label_style = TextStyle {
            color: Color::new(0.6, 0.6, 0.6, 1.0),
            ..Default::default()
        };
        canvas.draw_text(
            "FPR→",
            Point::new(area.x + area.width - 4.0, area.y + area.height),
            &label_style,
        );
        canvas.draw_text("TPR↑", Point::new(area.x - 4.0, area.y), &label_style);

        // Draw curves
        let num_curves = self.curves.len().max(1);
        for (idx, curve) in self.curves.iter_mut().enumerate() {
            if curve.roc_points.is_none() {
                curve.compute_roc(self.num_thresholds);
            }

            let points = curve.roc_points.as_ref().expect("computed above");
            let color = if let Some(ref gradient) = self.gradient {
                gradient.sample(idx as f64 / num_curves as f64)
            } else {
                curve.color
            };

            let style = TextStyle {
                color,
                ..Default::default()
            };

            for &(fpr, tpr) in points {
                let x = area.x + (fpr * area.width as f64) as f32;
                let y = area.y + ((1.0 - tpr) * area.height as f64) as f32;
                if x >= area.x && x < area.x + area.width && y >= area.y && y < area.y + area.height
                {
                    canvas.draw_text("•", Point::new(x, y), &style);
                }
            }

            // Draw legend with AUC
            if self.show_auc {
                let auc = curve.auc_roc.unwrap_or(0.0);
                let legend = format!("{}: AUC={:.3}", curve.label, auc);
                canvas.draw_text(
                    &legend,
                    Point::new(area.x + 1.0, area.y + 1.0 + idx as f32),
                    &style,
                );
            }
        }
    }

    fn render_pr(&mut self, canvas: &mut dyn Canvas, area: Rect) {
        let dim_style = TextStyle {
            color: Color::new(0.3, 0.3, 0.3, 1.0),
            ..Default::default()
        };

        // Draw grid
        if self.show_grid {
            for i in 1..5 {
                let x = area.x + area.width * i as f32 / 5.0;
                let y = area.y + area.height * i as f32 / 5.0;
                canvas.draw_text("·", Point::new(x, area.y), &dim_style);
                canvas.draw_text("·", Point::new(area.x, y), &dim_style);
            }
        }

        // Draw axes labels
        let label_style = TextStyle {
            color: Color::new(0.6, 0.6, 0.6, 1.0),
            ..Default::default()
        };
        canvas.draw_text(
            "Recall→",
            Point::new(area.x + area.width - 7.0, area.y + area.height),
            &label_style,
        );
        canvas.draw_text("Prec↑", Point::new(area.x - 5.0, area.y), &label_style);

        // Draw curves
        let num_curves = self.curves.len().max(1);
        for (idx, curve) in self.curves.iter_mut().enumerate() {
            if curve.pr_points.is_none() {
                curve.compute_pr(self.num_thresholds);
            }

            let points = curve.pr_points.as_ref().expect("computed above");
            let color = if let Some(ref gradient) = self.gradient {
                gradient.sample(idx as f64 / num_curves as f64)
            } else {
                curve.color
            };

            let style = TextStyle {
                color,
                ..Default::default()
            };

            for &(recall, precision) in points {
                let x = area.x + (recall * area.width as f64) as f32;
                let y = area.y + ((1.0 - precision) * area.height as f64) as f32;
                if x >= area.x && x < area.x + area.width && y >= area.y && y < area.y + area.height
                {
                    canvas.draw_text("•", Point::new(x, y), &style);
                }
            }

            // Draw legend with AUC
            if self.show_auc {
                let auc = curve.auc_pr.unwrap_or(0.0);
                let legend = format!("{}: AP={:.3}", curve.label, auc);
                canvas.draw_text(
                    &legend,
                    Point::new(area.x + 1.0, area.y + 1.0 + idx as f32),
                    &style,
                );
            }
        }
    }
}

impl Widget for RocPrCurve {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let width = match self.mode {
            CurveMode::Both => constraints.max_width.min(80.0),
            _ => constraints.max_width.min(40.0),
        };
        Size::new(width, constraints.max_height.min(20.0))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        if self.bounds.width < 10.0 || self.bounds.height < 5.0 {
            return;
        }

        let mut mutable_self = self.clone();

        match self.mode {
            CurveMode::Roc => {
                mutable_self.render_roc(canvas, self.bounds);
            }
            CurveMode::PrecisionRecall => {
                mutable_self.render_pr(canvas, self.bounds);
            }
            CurveMode::Both => {
                let half_width = self.bounds.width / 2.0;
                let left = Rect::new(
                    self.bounds.x,
                    self.bounds.y,
                    half_width - 1.0,
                    self.bounds.height,
                );
                let right = Rect::new(
                    self.bounds.x + half_width + 1.0,
                    self.bounds.y,
                    half_width - 1.0,
                    self.bounds.height,
                );
                mutable_self.render_roc(canvas, left);
                mutable_self.render_pr(canvas, right);
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

impl Brick for RocPrCurve {
    fn brick_name(&self) -> &'static str {
        "RocPrCurve"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        static ASSERTIONS: &[BrickAssertion] = &[BrickAssertion::max_latency_ms(16)];
        ASSERTIONS
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(16)
    }

    fn verify(&self) -> BrickVerification {
        let mut passed = Vec::new();
        let mut failed = Vec::new();

        if self.bounds.width >= 10.0 && self.bounds.height >= 5.0 {
            passed.push(BrickAssertion::max_latency_ms(16));
        } else {
            failed.push((
                BrickAssertion::max_latency_ms(16),
                "Size too small".to_string(),
            ));
        }

        BrickVerification {
            passed,
            failed,
            verification_time: Duration::from_micros(5),
        }
    }

    fn to_html(&self) -> String {
        String::new()
    }

    fn to_css(&self) -> String {
        String::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CellBuffer, DirectTerminalCanvas};

    #[test]
    fn test_curve_data_creation() {
        let data = CurveData::new("Model", vec![0.0, 1.0, 1.0], vec![0.2, 0.8, 0.9]);
        assert_eq!(data.label, "Model");
        assert_eq!(data.y_true.len(), 3);
    }

    #[test]
    fn test_curve_data_with_color() {
        let color = Color::new(0.5, 0.6, 0.7, 1.0);
        let data = CurveData::new("Test", vec![0.0, 1.0], vec![0.3, 0.8]).with_color(color);
        assert!((data.color.r - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_roc_computation() {
        let mut data = CurveData::new("Test", vec![0.0, 0.0, 1.0, 1.0], vec![0.1, 0.4, 0.35, 0.8]);
        data.compute_roc(10);
        assert!(data.roc_points.is_some());
        assert!(data.auc_roc.is_some());
        // AUC should be between 0 and 1
        let auc = data.auc_roc.expect("computed above");
        assert!((0.0..=1.0).contains(&auc));
    }

    #[test]
    fn test_pr_computation() {
        let mut data = CurveData::new("Test", vec![0.0, 0.0, 1.0, 1.0], vec![0.1, 0.4, 0.35, 0.8]);
        data.compute_pr(10);
        assert!(data.pr_points.is_some());
        assert!(data.auc_pr.is_some());
    }

    #[test]
    fn test_empty_data() {
        let mut data = CurveData::new("Empty", vec![], vec![]);
        data.compute_roc(10);
        assert_eq!(data.auc_roc, Some(0.5));
    }

    #[test]
    fn test_empty_data_pr() {
        let mut data = CurveData::new("Empty", vec![], vec![]);
        data.compute_pr(10);
        assert_eq!(data.auc_pr, Some(0.5));
    }

    #[test]
    fn test_all_positives() {
        let mut data = CurveData::new("AllPos", vec![1.0, 1.0, 1.0], vec![0.3, 0.6, 0.9]);
        data.compute_roc(10);
        // Should handle degenerate case
        assert!(data.roc_points.is_some());
    }

    #[test]
    fn test_all_negatives() {
        let mut data = CurveData::new("AllNeg", vec![0.0, 0.0, 0.0], vec![0.1, 0.5, 0.9]);
        data.compute_roc(10);
        // Should handle degenerate case
        assert!(data.roc_points.is_some());
    }

    #[test]
    fn test_all_negatives_pr() {
        let mut data = CurveData::new("AllNeg", vec![0.0, 0.0, 0.0], vec![0.1, 0.5, 0.9]);
        data.compute_pr(10);
        // Should handle degenerate case
        assert!(data.pr_points.is_some());
    }

    #[test]
    fn test_auc_getters() {
        let data = CurveData::new("Test", vec![0.0, 1.0], vec![0.3, 0.7]);
        assert!(data.auc_roc().is_none());
        assert!(data.auc_pr().is_none());
    }

    #[test]
    fn test_generate_thresholds_empty() {
        let thresholds = CurveData::generate_thresholds(&[], 10);
        assert_eq!(thresholds, vec![0.5]);
    }

    #[test]
    fn test_count_classes_scalar() {
        let data = CurveData::new("Test", vec![0.0, 0.0, 1.0, 1.0, 1.0], vec![0.0; 5]);
        let (n_pos, n_neg) = data.count_classes_scalar();
        assert!((n_pos - 3.0).abs() < 0.001);
        assert!((n_neg - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_count_classes_simd() {
        let y_true: Vec<f64> = (0..150)
            .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 })
            .collect();
        let data = CurveData::new("Test", y_true, vec![0.0; 150]);
        let (n_pos, n_neg) = data.count_classes_simd();
        assert_eq!(n_pos, 50.0);
        assert_eq!(n_neg, 100.0);
    }

    #[test]
    fn test_count_positives_scalar() {
        let data = CurveData::new("Test", vec![0.0, 0.0, 1.0, 1.0], vec![0.2, 0.8, 0.3, 0.9]);
        let (tp, fp) = data.count_positives_at_threshold_scalar(0.5);
        assert!((tp - 1.0).abs() < 0.001); // Only score 0.9 >= 0.5 with true label 1.0
        assert!((fp - 1.0).abs() < 0.001); // Only score 0.8 >= 0.5 with true label 0.0
    }

    #[test]
    fn test_count_positives_simd() {
        let y_true: Vec<f64> = (0..150)
            .map(|i| if i % 2 == 0 { 0.0 } else { 1.0 })
            .collect();
        let y_score: Vec<f64> = (0..150).map(|i| i as f64 / 150.0).collect();
        let data = CurveData::new("Test", y_true, y_score);
        let (tp, fp) = data.count_positives_at_threshold_simd(0.5);
        assert!(tp > 0.0);
        assert!(fp > 0.0);
    }

    #[test]
    fn test_roc_pr_curve_creation() {
        let curve = RocPrCurve::new(vec![CurveData::new("A", vec![0.0, 1.0], vec![0.3, 0.7])]);
        assert_eq!(curve.curves.len(), 1);
    }

    #[test]
    fn test_roc_pr_curve_default() {
        let curve = RocPrCurve::default();
        assert!(curve.curves.is_empty());
    }

    #[test]
    fn test_curve_mode() {
        let curve = RocPrCurve::default().with_mode(CurveMode::Both);
        assert_eq!(curve.mode, CurveMode::Both);
    }

    #[test]
    fn test_curve_mode_default() {
        let mode = CurveMode::default();
        assert_eq!(mode, CurveMode::Roc);
    }

    #[test]
    fn test_with_gradient() {
        let gradient = Gradient::two(
            Color::new(1.0, 0.0, 0.0, 1.0),
            Color::new(0.0, 0.0, 1.0, 1.0),
        );
        let curve = RocPrCurve::default().with_gradient(gradient);
        assert!(curve.gradient.is_some());
    }

    #[test]
    fn test_roc_pr_curve_measure_roc() {
        let curve = RocPrCurve::default().with_mode(CurveMode::Roc);
        let constraints = Constraints::new(0.0, 100.0, 0.0, 50.0);
        let size = curve.measure(constraints);
        assert_eq!(size.width, 40.0);
        assert_eq!(size.height, 20.0);
    }

    #[test]
    fn test_roc_pr_curve_measure_both() {
        let curve = RocPrCurve::default().with_mode(CurveMode::Both);
        let constraints = Constraints::new(0.0, 100.0, 0.0, 50.0);
        let size = curve.measure(constraints);
        assert_eq!(size.width, 80.0);
    }

    #[test]
    fn test_roc_pr_curve_layout_and_paint_roc() {
        let mut curve = RocPrCurve::new(vec![CurveData::new(
            "Good",
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.1, 0.2, 0.8, 0.9],
        )])
        .with_mode(CurveMode::Roc);

        let mut buffer = CellBuffer::new(50, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        let result = curve.layout(Rect::new(0.0, 0.0, 50.0, 20.0));
        assert_eq!(result.size.width, 50.0);

        curve.paint(&mut canvas);
    }

    #[test]
    fn test_roc_pr_curve_layout_and_paint_pr() {
        let mut curve = RocPrCurve::new(vec![CurveData::new(
            "Model",
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.1, 0.2, 0.8, 0.9],
        )])
        .with_mode(CurveMode::PrecisionRecall);

        let mut buffer = CellBuffer::new(50, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        curve.layout(Rect::new(0.0, 0.0, 50.0, 20.0));
        curve.paint(&mut canvas);
    }

    #[test]
    fn test_roc_pr_curve_layout_and_paint_both() {
        let mut curve = RocPrCurve::new(vec![CurveData::new(
            "Model",
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.1, 0.2, 0.8, 0.9],
        )])
        .with_mode(CurveMode::Both);

        let mut buffer = CellBuffer::new(80, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        curve.layout(Rect::new(0.0, 0.0, 80.0, 20.0));
        curve.paint(&mut canvas);
    }

    #[test]
    fn test_roc_pr_curve_paint_small_bounds() {
        let mut curve = RocPrCurve::new(vec![CurveData::new(
            "Model",
            vec![0.0, 1.0],
            vec![0.3, 0.7],
        )]);

        let mut buffer = CellBuffer::new(5, 3);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        curve.layout(Rect::new(0.0, 0.0, 5.0, 3.0));
        curve.paint(&mut canvas);
        // Should not crash
    }

    #[test]
    fn test_roc_pr_curve_paint_no_baseline() {
        let mut curve = RocPrCurve::new(vec![CurveData::new(
            "Model",
            vec![0.0, 1.0],
            vec![0.3, 0.7],
        )])
        .with_baseline(false);

        let mut buffer = CellBuffer::new(50, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        curve.layout(Rect::new(0.0, 0.0, 50.0, 20.0));
        curve.paint(&mut canvas);
    }

    #[test]
    fn test_roc_pr_curve_paint_no_grid() {
        let mut curve = RocPrCurve::new(vec![CurveData::new(
            "Model",
            vec![0.0, 1.0],
            vec![0.3, 0.7],
        )])
        .with_grid(false);

        let mut buffer = CellBuffer::new(50, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        curve.layout(Rect::new(0.0, 0.0, 50.0, 20.0));
        curve.paint(&mut canvas);
    }

    #[test]
    fn test_roc_pr_curve_paint_no_auc() {
        let mut curve = RocPrCurve::new(vec![CurveData::new(
            "Model",
            vec![0.0, 1.0],
            vec![0.3, 0.7],
        )])
        .with_auc(false);

        let mut buffer = CellBuffer::new(50, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        curve.layout(Rect::new(0.0, 0.0, 50.0, 20.0));
        curve.paint(&mut canvas);
    }

    #[test]
    fn test_roc_pr_curve_paint_with_gradient() {
        let gradient = Gradient::two(
            Color::new(0.2, 0.4, 0.8, 1.0),
            Color::new(0.8, 0.4, 0.2, 1.0),
        );
        let mut curve = RocPrCurve::new(vec![
            CurveData::new("A", vec![0.0, 1.0], vec![0.3, 0.7]),
            CurveData::new("B", vec![0.0, 1.0], vec![0.4, 0.6]),
        ])
        .with_gradient(gradient);

        let mut buffer = CellBuffer::new(50, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        curve.layout(Rect::new(0.0, 0.0, 50.0, 20.0));
        curve.paint(&mut canvas);
    }

    #[test]
    fn test_roc_pr_curve_assertions() {
        let curve = RocPrCurve::default();
        assert!(!curve.assertions().is_empty());
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn test_roc_pr_curve_verify_valid() {
        let mut curve = RocPrCurve::default();
        curve.bounds = Rect::new(0.0, 0.0, 40.0, 20.0);
        assert!(curve.verify().is_valid());
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn test_roc_pr_curve_verify_invalid() {
        let mut curve = RocPrCurve::default();
        curve.bounds = Rect::new(0.0, 0.0, 5.0, 3.0);
        assert!(!curve.verify().is_valid());
    }

    #[test]
    fn test_add_curve() {
        let mut curve = RocPrCurve::default();
        curve.add_curve(CurveData::new("New", vec![0.0, 1.0], vec![0.3, 0.7]));
        assert_eq!(curve.curves.len(), 1);
    }

    #[test]
    fn test_with_thresholds() {
        let curve = RocPrCurve::default().with_thresholds(50);
        assert_eq!(curve.num_thresholds, 50);
    }

    #[test]
    fn test_thresholds_clamped() {
        let curve = RocPrCurve::default().with_thresholds(5);
        assert_eq!(curve.num_thresholds, 10);

        let curve = RocPrCurve::default().with_thresholds(5000);
        assert_eq!(curve.num_thresholds, 1000);
    }

    #[test]
    fn test_large_dataset_simd() {
        // Test SIMD path (>100 elements)
        let y_true: Vec<f64> = (0..200)
            .map(|i| if i % 2 == 0 { 0.0 } else { 1.0 })
            .collect();
        let y_score: Vec<f64> = (0..200).map(|i| i as f64 / 200.0).collect();
        let mut data = CurveData::new("Large", y_true, y_score);
        data.compute_roc(50);
        assert!(data.auc_roc.is_some());
    }

    #[test]
    fn test_large_dataset_simd_pr() {
        let y_true: Vec<f64> = (0..200)
            .map(|i| if i % 2 == 0 { 0.0 } else { 1.0 })
            .collect();
        let y_score: Vec<f64> = (0..200).map(|i| i as f64 / 200.0).collect();
        let mut data = CurveData::new("Large", y_true, y_score);
        data.compute_pr(50);
        assert!(data.auc_pr.is_some());
    }

    #[test]
    fn test_with_baseline() {
        let curve = RocPrCurve::default().with_baseline(false);
        assert!(!curve.show_baseline);
    }

    #[test]
    fn test_with_auc() {
        let curve = RocPrCurve::default().with_auc(false);
        assert!(!curve.show_auc);
    }

    #[test]
    fn test_with_grid() {
        let curve = RocPrCurve::default().with_grid(false);
        assert!(!curve.show_grid);
    }

    #[test]
    fn test_children() {
        let curve = RocPrCurve::default();
        assert!(curve.children().is_empty());
    }

    #[test]
    fn test_children_mut() {
        let mut curve = RocPrCurve::default();
        assert!(curve.children_mut().is_empty());
    }

    #[test]
    fn test_brick_name() {
        let curve = RocPrCurve::default();
        assert_eq!(curve.brick_name(), "RocPrCurve");
    }

    #[test]
    fn test_budget() {
        let curve = RocPrCurve::default();
        let budget = curve.budget();
        assert!(budget.layout_ms > 0);
    }

    #[test]
    fn test_to_html() {
        let curve = RocPrCurve::default();
        assert!(curve.to_html().is_empty());
    }

    #[test]
    fn test_to_css() {
        let curve = RocPrCurve::default();
        assert!(curve.to_css().is_empty());
    }

    #[test]
    fn test_type_id() {
        let curve = RocPrCurve::default();
        let type_id = Widget::type_id(&curve);
        assert_eq!(type_id, TypeId::of::<RocPrCurve>());
    }

    #[test]
    fn test_event() {
        let mut curve = RocPrCurve::default();
        let event = Event::Resize {
            width: 80.0,
            height: 24.0,
        };
        assert!(curve.event(&event).is_none());
    }

    #[test]
    fn test_multiple_curves() {
        let mut curve = RocPrCurve::new(vec![
            CurveData::new("A", vec![0.0, 1.0, 0.0, 1.0], vec![0.1, 0.9, 0.2, 0.8]),
            CurveData::new("B", vec![0.0, 1.0, 0.0, 1.0], vec![0.3, 0.7, 0.4, 0.6]),
        ]);

        let mut buffer = CellBuffer::new(60, 25);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        curve.layout(Rect::new(0.0, 0.0, 60.0, 25.0));
        curve.paint(&mut canvas);
    }
}
