//! Tests for chart components.

use super::*;
use crate::train::tui::capability::TerminalMode;

// FeatureImportanceChart tests
#[test]
fn test_feature_importance_chart_new() {
    let chart = FeatureImportanceChart::new(5, 20);
    assert_eq!(chart.top_k, 5);
    assert_eq!(chart.bar_width, 20);
}

#[test]
fn test_feature_importance_chart_update() {
    let mut chart = FeatureImportanceChart::new(3, 20);
    let importances = vec![(0, 0.5), (1, 0.8), (2, 0.3), (3, 0.9)];
    chart.update(&importances, None);

    assert_eq!(chart.names.len(), 3);
    assert_eq!(chart.scores.len(), 3);
    // Should be sorted by importance: 0.9, 0.8, 0.5
    assert_eq!(chart.scores[0], 0.9);
    assert_eq!(chart.scores[1], 0.8);
    assert_eq!(chart.scores[2], 0.5);
}

#[test]
fn test_feature_importance_chart_with_names() {
    let mut chart = FeatureImportanceChart::new(2, 20);
    let importances = vec![(0, 0.5), (1, 0.8)];
    let names = vec!["feature_a".to_string(), "feature_b".to_string()];
    chart.update(&importances, Some(&names));

    assert!(chart.names.contains(&"feature_b".to_string()));
}

#[test]
fn test_feature_importance_chart_render_empty() {
    let chart = FeatureImportanceChart::new(5, 20);
    assert!(chart.render().contains("No feature importance"));
}

#[test]
fn test_feature_importance_chart_render() {
    let mut chart = FeatureImportanceChart::new(2, 10);
    let importances = vec![(0, 1.0), (1, 0.5)];
    chart.update(&importances, None);

    let rendered = chart.render();
    assert!(rendered.contains("Feature Importance"));
    assert!(rendered.contains("█"));
}

// GradientFlowHeatmap tests
#[test]
fn test_gradient_flow_heatmap_new() {
    let layers = vec!["layer_0".to_string(), "layer_1".to_string()];
    let cols = vec!["Q".to_string(), "K".to_string(), "V".to_string()];
    let heatmap = GradientFlowHeatmap::new(layers, cols);

    assert_eq!(heatmap.layer_names.len(), 2);
    assert_eq!(heatmap.column_labels.len(), 3);
    assert_eq!(heatmap.gradients.len(), 2);
    assert_eq!(heatmap.gradients[0].len(), 3);
}

#[test]
fn test_gradient_flow_heatmap_update() {
    let layers = vec!["layer_0".to_string()];
    let cols = vec!["Q".to_string(), "K".to_string()];
    let mut heatmap = GradientFlowHeatmap::new(layers, cols);

    heatmap.update(0, 0, 1.0);
    heatmap.update(0, 1, 0.1);

    // Values stored as log scale
    assert!(heatmap.gradients[0][0] > heatmap.gradients[0][1]);
}

#[test]
fn test_gradient_flow_heatmap_update_out_of_bounds() {
    let layers = vec!["layer_0".to_string()];
    let cols = vec!["Q".to_string()];
    let mut heatmap = GradientFlowHeatmap::new(layers, cols);

    // Should not panic
    heatmap.update(10, 10, 1.0);
}

#[test]
fn test_gradient_flow_heatmap_render() {
    let layers = vec!["layer_0".to_string()];
    let cols = vec!["Q".to_string(), "K".to_string()];
    let mut heatmap = GradientFlowHeatmap::new(layers, cols);
    heatmap.update(0, 0, 1.0);
    heatmap.update(0, 1, 0.01);

    let rendered = heatmap.render();
    assert!(rendered.contains("Gradient Flow"));
    assert!(rendered.contains("layer_0"));
}

// LossCurveDisplay tests
#[test]
fn test_loss_curve_display_new() {
    let display = LossCurveDisplay::new(80, 20);
    assert_eq!(display.epochs(), 0);
}

#[test]
fn test_loss_curve_display_push() {
    let mut display = LossCurveDisplay::new(80, 20);
    display.push_train_loss(1.0);
    display.push_val_loss(1.2);
    assert_eq!(display.epochs(), 1);
}

#[test]
fn test_loss_curve_display_push_losses() {
    let mut display = LossCurveDisplay::new(80, 20);
    display.push_losses(1.0, 1.2);
    assert_eq!(display.epochs(), 1);
}

#[test]
fn test_loss_curve_display_render_waiting() {
    let display = LossCurveDisplay::new(80, 20);
    assert!(display.render_terminal().contains("waiting"));
}

#[test]
fn test_loss_curve_display_terminal_mode() {
    let display = LossCurveDisplay::new(80, 20).terminal_mode(TerminalMode::Ascii);
    assert_eq!(display.terminal_mode, TerminalMode::Ascii);
}

#[test]
fn test_loss_curve_display_smoothing() {
    let display = LossCurveDisplay::new(80, 20).smoothing(0.9);
    // Just verify it doesn't panic
    assert_eq!(display.epochs(), 0);
}

#[test]
fn test_loss_curve_display_render_with_data() {
    let mut display = LossCurveDisplay::new(80, 20);
    // Need at least 2 epochs for rendering
    display.push_train_loss(2.0);
    display.push_val_loss(2.5);
    display.push_train_loss(1.5);
    display.push_val_loss(1.8);
    display.push_train_loss(1.2);
    display.push_val_loss(1.4);

    let rendered = display.render_terminal();
    // Should not be waiting message
    assert!(!rendered.contains("waiting"));
}

#[test]
fn test_loss_curve_display_summary() {
    let mut display = LossCurveDisplay::new(80, 20);
    display.push_train_loss(2.0);
    display.push_val_loss(2.5);
    display.push_train_loss(1.5);
    display.push_val_loss(1.8);

    let summary = display.summary();
    assert_eq!(summary.len(), 2);
    assert_eq!(summary[0].0, "Train");
    assert_eq!(summary[1].0, "Val");
}

#[test]
fn test_loss_curve_display_ascii_mode() {
    let mut display = LossCurveDisplay::new(80, 20).terminal_mode(TerminalMode::Ascii);
    display.push_train_loss(2.0);
    display.push_val_loss(2.5);
    display.push_train_loss(1.5);
    display.push_val_loss(1.8);
    display.push_train_loss(1.2);
    display.push_val_loss(1.4);

    let rendered = display.render_terminal();
    assert!(!rendered.contains("waiting"));
}

#[test]
fn test_loss_curve_display_ansi_mode() {
    let mut display = LossCurveDisplay::new(80, 20).terminal_mode(TerminalMode::Ansi);
    display.push_train_loss(2.0);
    display.push_val_loss(2.5);
    display.push_train_loss(1.5);
    display.push_val_loss(1.8);
    display.push_train_loss(1.2);
    display.push_val_loss(1.4);

    let rendered = display.render_terminal();
    assert!(!rendered.contains("waiting"));
}

#[test]
fn test_feature_importance_chart_zero_max_score() {
    let mut chart = FeatureImportanceChart::new(3, 20);
    let importances = vec![(0, 0.0), (1, 0.0), (2, 0.0)];
    chart.update(&importances, None);

    let rendered = chart.render();
    // Should render without panic even with zero scores
    assert!(rendered.contains("Feature Importance"));
}

#[test]
fn test_gradient_flow_heatmap_uniform_values() {
    let layers = vec!["layer_0".to_string(), "layer_1".to_string()];
    let cols = vec!["Q".to_string(), "K".to_string()];
    let mut heatmap = GradientFlowHeatmap::new(layers, cols);

    // All same values
    heatmap.update(0, 0, 1.0);
    heatmap.update(0, 1, 1.0);
    heatmap.update(1, 0, 1.0);
    heatmap.update(1, 1, 1.0);

    let rendered = heatmap.render();
    assert!(rendered.contains("Gradient Flow"));
}

#[test]
fn test_feature_importance_chart_clone() {
    let chart = FeatureImportanceChart::new(5, 20);
    let cloned = chart.clone();
    assert_eq!(chart.top_k, cloned.top_k);
    assert_eq!(chart.bar_width, cloned.bar_width);
}

#[test]
fn test_gradient_flow_heatmap_clone() {
    let layers = vec!["layer_0".to_string()];
    let cols = vec!["Q".to_string()];
    let heatmap = GradientFlowHeatmap::new(layers, cols);
    let cloned = heatmap.clone();
    assert_eq!(heatmap.layer_names, cloned.layer_names);
}

#[test]
fn test_feature_importance_chart_debug() {
    let chart = FeatureImportanceChart::new(5, 20);
    let debug_str = format!("{chart:?}");
    assert!(debug_str.contains("FeatureImportanceChart"));
}

#[test]
fn test_gradient_flow_heatmap_debug() {
    let layers = vec!["layer_0".to_string()];
    let cols = vec!["Q".to_string()];
    let heatmap = GradientFlowHeatmap::new(layers, cols);
    let debug_str = format!("{heatmap:?}");
    assert!(debug_str.contains("GradientFlowHeatmap"));
}

#[test]
fn test_loss_curve_display_multiple_pushes() {
    let mut display = LossCurveDisplay::new(80, 20);
    for i in 0..10 {
        display.push_train_loss(2.0 - i as f32 * 0.1);
        display.push_val_loss(2.5 - i as f32 * 0.1);
    }
    assert_eq!(display.epochs(), 10);
}
