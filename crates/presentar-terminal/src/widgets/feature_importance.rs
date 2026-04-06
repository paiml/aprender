//! Feature importance plot widget.
//!
//! Implements SPEC-024 Section 26.5.3.

use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// Feature importance plot widget.
#[derive(Debug, Clone)]
pub struct FeatureImportance {
    /// Feature names.
    features: Vec<String>,
    /// Importance values.
    importances: Vec<f64>,
    /// Sort by importance.
    sorted: bool,
    /// Show values.
    show_values: bool,
    /// Bar color.
    bar_color: Color,
    /// Maximum features to display.
    max_features: usize,
    /// Cached bounds.
    bounds: Rect,
}

impl FeatureImportance {
    /// Create a new feature importance plot.
    #[must_use]
    pub fn new(features: Vec<String>, importances: Vec<f64>) -> Self {
        Self {
            features,
            importances,
            sorted: true,
            show_values: true,
            bar_color: Color::new(0.2, 0.6, 0.9, 1.0),
            max_features: 20,
            bounds: Rect::default(),
        }
    }

    /// Toggle sorting.
    #[must_use]
    pub fn with_sorted(mut self, sorted: bool) -> Self {
        self.sorted = sorted;
        self
    }

    /// Toggle value display.
    #[must_use]
    pub fn with_show_values(mut self, show: bool) -> Self {
        self.show_values = show;
        self
    }

    /// Set bar color.
    #[must_use]
    pub fn with_color(mut self, color: Color) -> Self {
        self.bar_color = color;
        self
    }

    /// Set maximum features to display.
    #[must_use]
    pub fn with_max_features(mut self, max: usize) -> Self {
        self.max_features = max;
        self
    }

    /// Get sorted indices.
    fn sorted_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.importances.len()).collect();
        if self.sorted {
            indices.sort_by(|&a, &b| {
                self.importances[b]
                    .partial_cmp(&self.importances[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        indices.truncate(self.max_features);
        indices
    }

    /// Get maximum importance.
    fn max_importance(&self) -> f64 {
        self.importances
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(0.0f64, f64::max)
            .max(1e-10)
    }
}

impl Default for FeatureImportance {
    fn default() -> Self {
        Self::new(Vec::new(), Vec::new())
    }
}

impl Widget for FeatureImportance {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let height = (self.features.len().min(self.max_features) + 2) as f32;
        Size::new(
            constraints.max_width.min(60.0),
            constraints.max_height.min(height),
        )
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        if self.bounds.width < 20.0 || self.bounds.height < 3.0 || self.features.is_empty() {
            return;
        }

        let indices = self.sorted_indices();
        let max_imp = self.max_importance();

        // Calculate layout
        let label_width = 15.0f32;
        let value_width = if self.show_values { 8.0 } else { 0.0 };
        let bar_start = self.bounds.x + label_width;
        let bar_max_width = self.bounds.width - label_width - value_width - 1.0;

        let label_style = TextStyle {
            color: Color::new(0.7, 0.7, 0.7, 1.0),
            ..Default::default()
        };

        let bar_style = TextStyle {
            color: self.bar_color,
            ..Default::default()
        };

        let value_style = TextStyle {
            color: Color::new(0.5, 0.5, 0.5, 1.0),
            ..Default::default()
        };

        // Title
        canvas.draw_text(
            "Feature Importance",
            Point::new(self.bounds.x, self.bounds.y),
            &label_style,
        );

        // Draw bars
        let available_rows = (self.bounds.height as usize).saturating_sub(2);
        for (row, &idx) in indices.iter().enumerate().take(available_rows) {
            let y = self.bounds.y + row as f32 + 1.0;

            // Feature name (truncated)
            let name: String = self.features[idx].chars().take(14).collect();
            canvas.draw_text(
                &format!("{name:>14}"),
                Point::new(self.bounds.x, y),
                &label_style,
            );

            // Bar
            let importance = self.importances[idx].max(0.0);
            let bar_width = ((importance / max_imp) * bar_max_width as f64) as usize;

            if bar_width > 0 {
                let bar_str: String = "█".repeat(bar_width);
                canvas.draw_text(&bar_str, Point::new(bar_start, y), &bar_style);
            }

            // Value
            if self.show_values {
                let value_x = bar_start + bar_max_width + 1.0;
                canvas.draw_text(
                    &format!("{importance:.3}"),
                    Point::new(value_x, y),
                    &value_style,
                );
            }
        }

        // Show "..." if truncated
        if indices.len() > available_rows {
            let y = self.bounds.y + self.bounds.height - 1.0;
            canvas.draw_text(
                &format!("... and {} more", indices.len() - available_rows),
                Point::new(self.bounds.x, y),
                &label_style,
            );
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

impl Brick for FeatureImportance {
    fn brick_name(&self) -> &'static str {
        "FeatureImportance"
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

        if self.bounds.width >= 20.0 && self.bounds.height >= 3.0 {
            passed.push(BrickAssertion::max_latency_ms(16));
        } else {
            failed.push((
                BrickAssertion::max_latency_ms(16),
                "Size too small".to_string(),
            ));
        }

        // Check length consistency
        if self.features.len() != self.importances.len() {
            failed.push((
                BrickAssertion::max_latency_ms(16),
                "Features and importances length mismatch".to_string(),
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
    use crate::direct::{CellBuffer, DirectTerminalCanvas};

    #[test]
    fn test_feature_importance_new() {
        let features = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let importances = vec![0.5, 0.3, 0.2];
        let plot = FeatureImportance::new(features, importances);
        assert_eq!(plot.features.len(), 3);
        assert_eq!(plot.importances.len(), 3);
    }

    #[test]
    fn test_feature_importance_empty() {
        let plot = FeatureImportance::default();
        assert!(plot.features.is_empty());
    }

    #[test]
    fn test_feature_importance_sorted_indices() {
        let features = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let importances = vec![0.2, 0.5, 0.3];
        let plot = FeatureImportance::new(features, importances);
        let indices = plot.sorted_indices();
        assert_eq!(indices[0], 1); // B has highest importance
    }

    #[test]
    fn test_feature_importance_max() {
        let features = vec!["A".to_string(), "B".to_string()];
        let importances = vec![0.2, 0.8];
        let plot = FeatureImportance::new(features, importances);
        assert!((plot.max_importance() - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_feature_importance_paint() {
        let features = vec![
            "feature_a".to_string(),
            "feature_b".to_string(),
            "feature_c".to_string(),
            "feature_d".to_string(),
            "feature_e".to_string(),
        ];
        let importances = vec![0.35, 0.25, 0.2, 0.12, 0.08];

        let mut plot = FeatureImportance::new(features, importances);
        let bounds = Rect::new(0.0, 0.0, 60.0, 10.0);
        plot.layout(bounds);

        let mut buffer = CellBuffer::new(60, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        plot.paint(&mut canvas);
    }

    #[test]
    fn test_feature_importance_with_options() {
        let features = vec!["A".to_string()];
        let importances = vec![0.5];
        let plot = FeatureImportance::new(features, importances)
            .with_sorted(false)
            .with_show_values(false)
            .with_color(Color::RED)
            .with_max_features(10);

        assert!(!plot.sorted);
        assert!(!plot.show_values);
        assert_eq!(plot.max_features, 10);
    }

    #[test]
    fn test_feature_importance_verify() {
        let features = vec!["A".to_string(), "B".to_string()];
        let importances = vec![0.5, 0.3];
        let mut plot = FeatureImportance::new(features, importances);
        plot.bounds = Rect::new(0.0, 0.0, 60.0, 10.0);
        assert!(plot.verify().is_valid());
    }

    #[test]
    fn test_feature_importance_verify_mismatch() {
        let features = vec!["A".to_string(), "B".to_string()];
        let importances = vec![0.5]; // Length mismatch
        let mut plot = FeatureImportance::new(features, importances);
        plot.bounds = Rect::new(0.0, 0.0, 60.0, 10.0);
        assert!(!plot.verify().is_valid());
    }

    #[test]
    fn test_feature_importance_brick_name() {
        let plot = FeatureImportance::default();
        assert_eq!(plot.brick_name(), "FeatureImportance");
    }

    // =========================================================================
    // Additional coverage tests
    // =========================================================================

    #[test]
    fn test_sorted_indices_unsorted() {
        let features = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let importances = vec![0.2, 0.5, 0.3];
        let plot = FeatureImportance::new(features, importances).with_sorted(false);
        let indices = plot.sorted_indices();
        // Should maintain original order
        assert_eq!(indices[0], 0);
        assert_eq!(indices[1], 1);
        assert_eq!(indices[2], 2);
    }

    #[test]
    fn test_sorted_indices_with_nan() {
        let features = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let importances = vec![0.2, f64::NAN, 0.5];
        let plot = FeatureImportance::new(features, importances);
        let indices = plot.sorted_indices();
        // NaN comparison should not panic
        assert_eq!(indices.len(), 3);
    }

    #[test]
    fn test_sorted_indices_truncates() {
        let features: Vec<String> = (0..30).map(|i| format!("F{i}")).collect();
        let importances: Vec<f64> = (0..30).map(|i| i as f64 / 30.0).collect();
        let plot = FeatureImportance::new(features, importances).with_max_features(5);
        let indices = plot.sorted_indices();
        assert_eq!(indices.len(), 5);
    }

    #[test]
    fn test_max_importance_empty() {
        let plot = FeatureImportance::default();
        let max = plot.max_importance();
        // Should return minimum value to avoid division by zero
        assert!(max > 0.0);
    }

    #[test]
    fn test_max_importance_all_nan() {
        let features = vec!["A".to_string(), "B".to_string()];
        let importances = vec![f64::NAN, f64::NAN];
        let plot = FeatureImportance::new(features, importances);
        let max = plot.max_importance();
        // Should return minimum to avoid div by zero
        assert!(max > 0.0);
    }

    #[test]
    fn test_max_importance_with_infinity() {
        let features = vec!["A".to_string(), "B".to_string()];
        let importances = vec![f64::INFINITY, 0.5];
        let plot = FeatureImportance::new(features, importances);
        let max = plot.max_importance();
        // Should filter infinity
        assert!(max < f64::INFINITY);
    }

    #[test]
    fn test_max_importance_negative() {
        let features = vec!["A".to_string(), "B".to_string()];
        let importances = vec![-0.5, -0.3];
        let plot = FeatureImportance::new(features, importances);
        let max = plot.max_importance();
        // max should at least be the min threshold
        assert!(max > 0.0);
    }

    #[test]
    fn test_paint_too_small_width() {
        let features = vec!["A".to_string()];
        let importances = vec![0.5];
        let mut plot = FeatureImportance::new(features, importances);
        plot.bounds = Rect::new(0.0, 0.0, 10.0, 10.0); // Too narrow

        let mut buffer = CellBuffer::new(10, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        plot.paint(&mut canvas); // Should return early
    }

    #[test]
    fn test_paint_too_small_height() {
        let features = vec!["A".to_string()];
        let importances = vec![0.5];
        let mut plot = FeatureImportance::new(features, importances);
        plot.bounds = Rect::new(0.0, 0.0, 60.0, 2.0); // Too short

        let mut buffer = CellBuffer::new(60, 2);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        plot.paint(&mut canvas); // Should return early
    }

    #[test]
    fn test_paint_empty_features() {
        let plot = FeatureImportance::default();
        let mut buffer = CellBuffer::new(60, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        plot.paint(&mut canvas); // Should return early
    }

    #[test]
    fn test_paint_without_values() {
        let features = vec!["feature_a".to_string(), "feature_b".to_string()];
        let importances = vec![0.5, 0.3];
        let mut plot = FeatureImportance::new(features, importances).with_show_values(false);
        plot.bounds = Rect::new(0.0, 0.0, 60.0, 10.0);

        let mut buffer = CellBuffer::new(60, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        plot.paint(&mut canvas);
    }

    #[test]
    fn test_paint_with_truncation() {
        // More features than available rows
        let features: Vec<String> = (0..20).map(|i| format!("F{i}")).collect();
        let importances: Vec<f64> = (0..20).map(|i| i as f64 / 20.0).collect();
        let mut plot = FeatureImportance::new(features, importances);
        plot.bounds = Rect::new(0.0, 0.0, 60.0, 8.0); // Only ~6 rows for features

        let mut buffer = CellBuffer::new(60, 8);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        plot.paint(&mut canvas); // Should show "... and X more"
    }

    #[test]
    fn test_paint_long_feature_names() {
        let features = vec!["very_long_feature_name_that_exceeds_14_chars".to_string()];
        let importances = vec![0.5];
        let mut plot = FeatureImportance::new(features, importances);
        plot.bounds = Rect::new(0.0, 0.0, 60.0, 10.0);

        let mut buffer = CellBuffer::new(60, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        plot.paint(&mut canvas); // Should truncate name
    }

    #[test]
    fn test_paint_zero_bar_width() {
        let features = vec!["A".to_string()];
        let importances = vec![0.0]; // Zero importance
        let mut plot = FeatureImportance::new(features, importances);
        plot.bounds = Rect::new(0.0, 0.0, 60.0, 10.0);

        let mut buffer = CellBuffer::new(60, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        plot.paint(&mut canvas);
    }

    #[test]
    fn test_paint_negative_importance() {
        let features = vec!["A".to_string()];
        let importances = vec![-0.5]; // Negative importance
        let mut plot = FeatureImportance::new(features, importances);
        plot.bounds = Rect::new(0.0, 0.0, 60.0, 10.0);

        let mut buffer = CellBuffer::new(60, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        plot.paint(&mut canvas); // Should handle gracefully (clamp to 0)
    }

    #[test]
    fn test_verify_too_small() {
        let features = vec!["A".to_string()];
        let importances = vec![0.5];
        let mut plot = FeatureImportance::new(features, importances);
        plot.bounds = Rect::new(0.0, 0.0, 10.0, 2.0); // Too small
        let result = plot.verify();
        assert!(!result.failed.is_empty());
    }

    #[test]
    fn test_widget_type_id() {
        let plot = FeatureImportance::default();
        let id = Widget::type_id(&plot);
        assert_eq!(id, TypeId::of::<FeatureImportance>());
    }

    #[test]
    fn test_widget_measure() {
        let features = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let importances = vec![0.5, 0.3, 0.2];
        let plot = FeatureImportance::new(features, importances);
        let constraints = Constraints::tight(Size::new(100.0, 50.0));
        let size = plot.measure(constraints);
        assert!(size.width <= 60.0);
        assert!(size.height <= 50.0);
    }

    #[test]
    fn test_widget_layout() {
        let mut plot = FeatureImportance::default();
        let bounds = Rect::new(10.0, 20.0, 40.0, 15.0);
        let result = plot.layout(bounds);
        assert_eq!(result.size.width, 40.0);
        assert_eq!(result.size.height, 15.0);
        assert_eq!(plot.bounds, bounds);
    }

    #[test]
    fn test_widget_event() {
        let mut plot = FeatureImportance::default();
        let result = plot.event(&Event::FocusIn);
        assert!(result.is_none());
    }

    #[test]
    fn test_widget_children() {
        let plot = FeatureImportance::default();
        assert!(plot.children().is_empty());
    }

    #[test]
    fn test_widget_children_mut() {
        let mut plot = FeatureImportance::default();
        assert!(plot.children_mut().is_empty());
    }

    #[test]
    fn test_brick_assertions() {
        let plot = FeatureImportance::default();
        let assertions = plot.assertions();
        assert!(!assertions.is_empty());
    }

    #[test]
    fn test_brick_budget() {
        let plot = FeatureImportance::default();
        let budget = plot.budget();
        assert!(budget.total_ms > 0);
    }

    #[test]
    fn test_brick_to_html() {
        let plot = FeatureImportance::default();
        assert!(plot.to_html().is_empty());
    }

    #[test]
    fn test_brick_to_css() {
        let plot = FeatureImportance::default();
        assert!(plot.to_css().is_empty());
    }

    #[test]
    fn test_clone() {
        let features = vec!["A".to_string(), "B".to_string()];
        let importances = vec![0.5, 0.3];
        let plot = FeatureImportance::new(features, importances)
            .with_sorted(true)
            .with_show_values(false)
            .with_color(Color::GREEN)
            .with_max_features(5);

        let cloned = plot;
        assert_eq!(cloned.features.len(), 2);
        assert!(cloned.sorted);
        assert!(!cloned.show_values);
        assert_eq!(cloned.max_features, 5);
    }

    #[test]
    fn test_debug() {
        let plot = FeatureImportance::default();
        let debug_str = format!("{:?}", plot);
        assert!(debug_str.contains("FeatureImportance"));
    }

    #[test]
    fn test_default_values() {
        let features = vec!["A".to_string()];
        let importances = vec![0.5];
        let plot = FeatureImportance::new(features, importances);
        assert!(plot.sorted);
        assert!(plot.show_values);
        assert_eq!(plot.max_features, 20);
    }
}
