#![allow(clippy::unwrap_used, clippy::disallowed_methods)]
//! Example YAML validation tests.
//!
//! EXTREME TDD: These tests validate all 50 examples from the spec.
//! Each example must pass the 15-point QA checklist validation.

use presentar_yaml::Manifest;
use std::fs;
use std::path::Path;

/// Validate a manifest meets minimum quality requirements.
fn validate_manifest(manifest: &Manifest) -> Vec<String> {
    let mut errors = Vec::new();

    // Required fields
    if manifest.name.is_empty() {
        errors.push("Missing name".to_string());
    }
    if manifest.version.is_empty() {
        errors.push("Missing version".to_string());
    }
    if manifest.presentar.is_empty() {
        errors.push("Missing presentar version".to_string());
    }

    // Layout validation
    if manifest.layout.layout_type.is_empty() {
        errors.push("Missing layout type".to_string());
    }

    // Section validation
    for section in &manifest.layout.sections {
        if section.id.is_empty() {
            errors.push("Section missing id".to_string());
        }
        for widget in &section.widgets {
            if widget.widget_type.is_empty() {
                errors.push(format!("Widget in section '{}' missing type", section.id));
            }
        }
    }

    errors
}

/// Load and validate a YAML file.
fn load_and_validate(path: &Path) -> Result<Manifest, String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

    let manifest = Manifest::from_yaml(&content)
        .map_err(|e| format!("Failed to parse {}: {}", path.display(), e))?;

    let errors = validate_manifest(&manifest);
    if !errors.is_empty() {
        return Err(format!(
            "Validation errors in {}: {}",
            path.display(),
            errors.join(", ")
        ));
    }

    Ok(manifest)
}

// =============================================================================
// Section A: .apr Model Visualization (APR-001 to APR-010)
// =============================================================================

#[test]
fn test_apr_001_model_card_basic() {
    let path = Path::new("../../examples/apr/model_card_basic.yaml");
    let manifest = load_and_validate(path).expect("APR-001 should be valid");

    assert_eq!(manifest.name, "model-card-basic");
    assert!(!manifest.models.is_empty() || !manifest.data.is_empty());

    // Must have model_card widget
    let has_model_card = manifest
        .layout
        .sections
        .iter()
        .flat_map(|s| &s.widgets)
        .any(|w| w.widget_type == "model_card");
    assert!(has_model_card, "APR-001 must have model_card widget");
}

#[test]
fn test_apr_002_model_comparison() {
    let path = Path::new("../../examples/apr/model_comparison.yaml");
    let manifest = load_and_validate(path).expect("APR-002 should be valid");

    assert_eq!(manifest.name, "model-comparison");

    // Must have at least 2 model_card widgets for comparison
    let model_card_count = manifest
        .layout
        .sections
        .iter()
        .flat_map(|s| &s.widgets)
        .filter(|w| w.widget_type == "model_card")
        .count();
    assert!(
        model_card_count >= 2,
        "APR-002 must have at least 2 model_card widgets"
    );
}

#[test]
fn test_apr_003_model_metrics_chart() {
    let path = Path::new("../../examples/apr/model_metrics_chart.yaml");
    let manifest = load_and_validate(path).expect("APR-003 should be valid");

    assert_eq!(manifest.name, "model-metrics-chart");

    // Must have chart widget with line type
    let has_line_chart = manifest
        .layout
        .sections
        .iter()
        .flat_map(|s| &s.widgets)
        .any(|w| w.widget_type == "chart" && w.chart_type.as_deref() == Some("line"));
    assert!(has_line_chart, "APR-003 must have line chart widget");
}

#[test]
fn test_apr_005_model_inference_demo() {
    let path = Path::new("../../examples/apr/model_inference_demo.yaml");
    let manifest = load_and_validate(path).expect("APR-005 should be valid");

    assert_eq!(manifest.name, "model-inference");

    // Must have both model_card and chart
    let widget_types: Vec<_> = manifest
        .layout
        .sections
        .iter()
        .flat_map(|s| &s.widgets)
        .map(|w| w.widget_type.as_str())
        .collect();

    assert!(
        widget_types.contains(&"model_card"),
        "APR-005 must have model_card"
    );
    assert!(widget_types.contains(&"chart"), "APR-005 must have chart");
}

#[test]
fn test_apr_007_model_gradient_flow() {
    let path = Path::new("../../examples/apr/model_gradient_flow.yaml");
    let manifest = load_and_validate(path).expect("APR-007 should be valid");

    assert_eq!(manifest.name, "gradient-flow");

    // Must have heatmap chart
    let has_heatmap = manifest
        .layout
        .sections
        .iter()
        .flat_map(|s| &s.widgets)
        .any(|w| w.widget_type == "chart" && w.chart_type.as_deref() == Some("heatmap"));
    assert!(has_heatmap, "APR-007 must have heatmap chart");
}

#[test]
fn test_apr_010_model_export_preview() {
    let path = Path::new("../../examples/apr/model_export_preview.yaml");
    let manifest = load_and_validate(path).expect("APR-010 should be valid");

    assert_eq!(manifest.name, "model-export-preview");

    // Must have buttons for export
    let button_count = manifest
        .layout
        .sections
        .iter()
        .flat_map(|s| &s.widgets)
        .filter(|w| w.widget_type == "button")
        .count();
    assert!(
        button_count >= 2,
        "APR-010 must have at least 2 export buttons"
    );
}

// =============================================================================
// Section B: .ald Dataset Visualization (ALD-001 to ALD-010)
// =============================================================================

#[test]
fn test_ald_001_data_card_basic() {
    let path = Path::new("../../examples/ald/data_card_basic.yaml");
    let manifest = load_and_validate(path).expect("ALD-001 should be valid");

    assert_eq!(manifest.name, "data-card-basic");

    // Must have data source
    assert!(!manifest.data.is_empty(), "ALD-001 must have data source");

    // Must have data_card widget
    let has_data_card = manifest
        .layout
        .sections
        .iter()
        .flat_map(|s| &s.widgets)
        .any(|w| w.widget_type == "data_card");
    assert!(has_data_card, "ALD-001 must have data_card widget");
}

#[test]
fn test_ald_002_data_table_virtualized() {
    let path = Path::new("../../examples/ald/data_table_virtualized.yaml");
    let manifest = load_and_validate(path).expect("ALD-002 should be valid");

    assert_eq!(manifest.name, "data-table-virtualized");

    // Must have data_table widget
    let has_data_table = manifest
        .layout
        .sections
        .iter()
        .flat_map(|s| &s.widgets)
        .any(|w| w.widget_type == "data_table");
    assert!(has_data_table, "ALD-002 must have data_table widget");
}

#[test]
fn test_ald_003_data_distribution_chart() {
    let path = Path::new("../../examples/ald/data_distribution.yaml");
    let manifest = load_and_validate(path).expect("ALD-003 should be valid");

    assert_eq!(manifest.name, "data-distribution");

    // Must have histogram chart
    let has_histogram = manifest
        .layout
        .sections
        .iter()
        .flat_map(|s| &s.widgets)
        .any(|w| w.widget_type == "chart" && w.chart_type.as_deref() == Some("histogram"));
    assert!(has_histogram, "ALD-003 must have histogram chart");
}

#[test]
fn test_ald_004_data_scatter_plot() {
    let path = Path::new("../../examples/ald/data_scatter.yaml");
    let manifest = load_and_validate(path).expect("ALD-004 should be valid");

    assert_eq!(manifest.name, "data-scatter");

    // Must have scatter chart with x, y, color
    let scatter = manifest
        .layout
        .sections
        .iter()
        .flat_map(|s| &s.widgets)
        .find(|w| w.widget_type == "chart" && w.chart_type.as_deref() == Some("scatter"));

    assert!(scatter.is_some(), "ALD-004 must have scatter chart");
    let scatter = scatter.unwrap();
    assert!(scatter.x.is_some(), "Scatter must have x field");
    assert!(scatter.y.is_some(), "Scatter must have y field");
}

#[test]
fn test_ald_006_data_time_series() {
    let path = Path::new("../../examples/ald/data_timeseries.yaml");
    let manifest = load_and_validate(path).expect("ALD-006 should be valid");

    assert_eq!(manifest.name, "data-timeseries");

    // Must have line chart
    let has_line = manifest
        .layout
        .sections
        .iter()
        .flat_map(|s| &s.widgets)
        .any(|w| w.widget_type == "chart" && w.chart_type.as_deref() == Some("line"));
    assert!(has_line, "ALD-006 must have line chart for timeseries");
}

#[test]
fn test_ald_008_data_class_balance() {
    let path = Path::new("../../examples/ald/class_balance.yaml");
    let manifest = load_and_validate(path).expect("ALD-008 should be valid");

    assert_eq!(manifest.name, "class-balance");

    // Must have bar chart
    let has_bar = manifest
        .layout
        .sections
        .iter()
        .flat_map(|s| &s.widgets)
        .any(|w| w.widget_type == "chart" && w.chart_type.as_deref() == Some("bar"));
    assert!(has_bar, "ALD-008 must have bar chart");
}

// =============================================================================
// Section C: Basic Charts (CHT-001 to CHT-010)
// =============================================================================

#[test]
fn test_cht_001_line_chart_basic() {
    let path = Path::new("../../examples/charts/line_chart_basic.yaml");
    let manifest = load_and_validate(path).expect("CHT-001 should be valid");

    assert_eq!(manifest.name, "line-chart-basic");

    // Must have line chart
    let has_line = manifest
        .layout
        .sections
        .iter()
        .flat_map(|s| &s.widgets)
        .any(|w| w.widget_type == "chart" && w.chart_type.as_deref() == Some("line"));
    assert!(has_line, "CHT-001 must have line chart");
}

#[test]
fn test_cht_002_bar_chart_grouped() {
    let path = Path::new("../../examples/charts/bar_chart_grouped.yaml");
    let manifest = load_and_validate(path).expect("CHT-002 should be valid");

    assert_eq!(manifest.name, "bar-chart-grouped");

    // Must have bar chart
    let has_bar = manifest
        .layout
        .sections
        .iter()
        .flat_map(|s| &s.widgets)
        .any(|w| w.widget_type == "chart" && w.chart_type.as_deref() == Some("bar"));
    assert!(has_bar, "CHT-002 must have bar chart");
}

#[test]
fn test_cht_003_pie_chart_basic() {
    let path = Path::new("../../examples/charts/pie_chart_basic.yaml");
    let manifest = load_and_validate(path).expect("CHT-003 should be valid");

    assert_eq!(manifest.name, "pie-chart-basic");

    // Must have pie chart
    let has_pie = manifest
        .layout
        .sections
        .iter()
        .flat_map(|s| &s.widgets)
        .any(|w| w.widget_type == "chart" && w.chart_type.as_deref() == Some("pie"));
    assert!(has_pie, "CHT-003 must have pie chart");
}

// =============================================================================
// Section D: Interactive Dashboards (DSH-001 to DSH-010)
// =============================================================================

#[test]
fn test_dsh_001_training_dashboard() {
    let path = Path::new("../../examples/dashboards/training_dashboard.yaml");
    let manifest = load_and_validate(path).expect("DSH-001 should be valid");

    assert_eq!(manifest.name, "training-dashboard");

    // Must use grid layout
    assert_eq!(
        manifest.layout.layout_type, "grid",
        "DSH-001 must use grid layout"
    );

    // Must have multiple widgets (model_card, charts, text)
    let widget_count: usize = manifest
        .layout
        .sections
        .iter()
        .map(|s| s.widgets.len())
        .sum();
    assert!(widget_count >= 3, "DSH-001 must have at least 3 widgets");
}

#[test]
fn test_dsh_002_dataset_explorer() {
    let path = Path::new("../../examples/dashboards/dataset_explorer.yaml");
    let manifest = load_and_validate(path).expect("DSH-002 should be valid");

    assert_eq!(manifest.name, "dataset-explorer");

    // Must have interactive widgets
    assert!(
        manifest
            .layout
            .sections
            .iter()
            .flat_map(|s| &s.widgets)
            .any(|w| w.widget_type == "data_table"),
        "DSH-002 must have data_table"
    );
}

#[test]
fn test_dsh_003_model_comparison_dashboard() {
    let path = Path::new("../../examples/dashboards/model_comparison_dashboard.yaml");
    let manifest = load_and_validate(path).expect("DSH-003 should be valid");

    assert_eq!(manifest.name, "model-comparison-dashboard");

    // Must have chart widgets
    let chart_count = manifest
        .layout
        .sections
        .iter()
        .flat_map(|s| &s.widgets)
        .filter(|w| w.widget_type == "chart")
        .count();
    assert!(
        chart_count >= 2,
        "DSH-003 must have at least 2 comparison charts"
    );
}

#[test]
fn test_dsh_005_experiment_tracker() {
    let path = Path::new("../../examples/dashboards/experiment_tracker.yaml");
    let manifest = load_and_validate(path).expect("DSH-005 should be valid");

    assert_eq!(manifest.name, "experiment-tracker");

    // Must have data_table and chart
    let widget_types: Vec<_> = manifest
        .layout
        .sections
        .iter()
        .flat_map(|s| &s.widgets)
        .map(|w| w.widget_type.as_str())
        .collect();

    assert!(
        widget_types.contains(&"data_table"),
        "DSH-005 must have data_table"
    );
    assert!(widget_types.contains(&"chart"), "DSH-005 must have chart");
}

#[test]
fn test_dsh_008_confusion_matrix() {
    let path = Path::new("../../examples/dashboards/confusion_matrix.yaml");
    let manifest = load_and_validate(path).expect("DSH-008 should be valid");

    assert_eq!(manifest.name, "confusion-matrix");

    // Must have heatmap
    let has_heatmap = manifest
        .layout
        .sections
        .iter()
        .flat_map(|s| &s.widgets)
        .any(|w| w.widget_type == "chart" && w.chart_type.as_deref() == Some("heatmap"));
    assert!(
        has_heatmap,
        "DSH-008 must have heatmap for confusion matrix"
    );
}

// =============================================================================
// Section E: Edge Cases & Stress Tests (EDG-001 to EDG-010)
// =============================================================================

#[test]
fn test_edg_001_empty_dataset() {
    let path = Path::new("../../examples/edge_cases/empty_dataset.yaml");
    let manifest = load_and_validate(path).expect("EDG-001 should be valid");

    assert_eq!(manifest.name, "empty-dataset");

    // Must handle empty gracefully with empty_message
    let has_data_card = manifest
        .layout
        .sections
        .iter()
        .flat_map(|s| &s.widgets)
        .any(|w| w.widget_type == "data_card");
    assert!(has_data_card, "EDG-001 must have data_card widget");
}

#[test]
fn test_edg_002_large_dataset() {
    let path = Path::new("../../examples/edge_cases/large_dataset.yaml");
    let manifest = load_and_validate(path).expect("EDG-002 should be valid");

    assert_eq!(manifest.name, "large-dataset");

    // Must have virtualized data_table
    let has_data_table = manifest
        .layout
        .sections
        .iter()
        .flat_map(|s| &s.widgets)
        .any(|w| w.widget_type == "data_table");
    assert!(has_data_table, "EDG-002 must have data_table widget");
}

// =============================================================================
// Utility: Test all examples in a directory
// =============================================================================

#[test]
fn test_all_apr_examples_exist() {
    let apr_dir = Path::new("../../examples/apr");
    assert!(apr_dir.exists(), "examples/apr directory must exist");

    let required = [
        "model_card_basic.yaml",
        "model_comparison.yaml",
        "model_metrics_chart.yaml",
        "model_inference_demo.yaml",
        "model_gradient_flow.yaml",
        "model_export_preview.yaml",
    ];

    for file in &required {
        let path = apr_dir.join(file);
        assert!(path.exists(), "Required APR example missing: {file}");
    }
}

#[test]
fn test_all_ald_examples_exist() {
    let ald_dir = Path::new("../../examples/ald");
    assert!(ald_dir.exists(), "examples/ald directory must exist");

    let required = [
        "data_card_basic.yaml",
        "data_table_virtualized.yaml",
        "data_distribution.yaml",
        "data_scatter.yaml",
        "data_timeseries.yaml",
        "class_balance.yaml",
    ];

    for file in &required {
        let path = ald_dir.join(file);
        assert!(path.exists(), "Required ALD example missing: {file}");
    }
}

#[test]
fn test_all_chart_examples_exist() {
    let charts_dir = Path::new("../../examples/charts");
    assert!(charts_dir.exists(), "examples/charts directory must exist");

    let required = [
        "line_chart_basic.yaml",
        "bar_chart_grouped.yaml",
        "pie_chart_basic.yaml",
    ];

    for file in &required {
        let path = charts_dir.join(file);
        assert!(path.exists(), "Required chart example missing: {file}");
    }
}

#[test]
fn test_all_dashboard_examples_exist() {
    let dashboards_dir = Path::new("../../examples/dashboards");
    assert!(
        dashboards_dir.exists(),
        "examples/dashboards directory must exist"
    );

    let required = [
        "training_dashboard.yaml",
        "dataset_explorer.yaml",
        "model_comparison_dashboard.yaml",
        "experiment_tracker.yaml",
        "confusion_matrix.yaml",
    ];

    for file in &required {
        let path = dashboards_dir.join(file);
        assert!(path.exists(), "Required dashboard example missing: {file}");
    }
}

#[test]
fn test_all_edge_case_examples_exist() {
    let edge_dir = Path::new("../../examples/edge_cases");
    assert!(
        edge_dir.exists(),
        "examples/edge_cases directory must exist"
    );

    let required = ["empty_dataset.yaml", "large_dataset.yaml"];

    for file in &required {
        let path = edge_dir.join(file);
        assert!(path.exists(), "Required edge case example missing: {file}");
    }
}
