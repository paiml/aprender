//! YAML manifest types for Presentar applications.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Application manifest loaded from app.yaml.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// Presentar version
    pub presentar: String,
    /// Application name
    pub name: String,
    /// Application version
    pub version: String,
    /// Application description
    #[serde(default)]
    pub description: String,
    /// Quality score (auto-computed)
    #[serde(default)]
    pub score: Option<Score>,
    /// Data sources
    #[serde(default)]
    pub data: HashMap<String, DataSource>,
    /// Model references
    #[serde(default)]
    pub models: HashMap<String, ModelRef>,
    /// Layout configuration
    pub layout: LayoutConfig,
    /// Interactions
    #[serde(default)]
    pub interactions: Vec<Interaction>,
    /// Theme configuration
    #[serde(default)]
    pub theme: Option<ThemeConfig>,
}

/// Quality score metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Score {
    /// Letter grade (F-A+)
    pub grade: String,
    /// Numeric score (0-100)
    pub value: f64,
    /// Test coverage percentage
    #[serde(default)]
    pub coverage: Option<f64>,
}

/// Data source configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSource {
    /// Source URI (pacha://, file://, https://)
    pub source: String,
    /// Data format (ald, csv, json)
    #[serde(default = "default_format")]
    pub format: String,
    /// Refresh interval
    #[serde(default)]
    pub refresh: Option<String>,
}

fn default_format() -> String {
    "ald".to_string()
}

/// Model reference configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRef {
    /// Source URI (pacha://, file://)
    pub source: String,
    /// Model format (apr)
    #[serde(default = "default_model_format")]
    pub format: String,
}

fn default_model_format() -> String {
    "apr".to_string()
}

/// Layout configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutConfig {
    /// Layout type (dashboard, app, custom)
    #[serde(rename = "type")]
    pub layout_type: String,
    /// Number of columns for grid layout
    #[serde(default = "default_columns")]
    pub columns: u32,
    /// Row configuration
    #[serde(default)]
    pub rows: String,
    /// Gap between sections
    #[serde(default = "default_gap")]
    pub gap: u32,
    /// Layout sections
    #[serde(default)]
    pub sections: Vec<Section>,
}

const fn default_columns() -> u32 {
    12
}

const fn default_gap() -> u32 {
    16
}

/// Layout section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Section {
    /// Section ID
    pub id: String,
    /// Grid span [start, end]
    #[serde(default)]
    pub span: Option<[u32; 2]>,
    /// Widgets in this section
    #[serde(default)]
    pub widgets: Vec<WidgetConfig>,
}

/// Widget configuration from YAML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetConfig {
    /// Widget type (text, button, chart, data-table, etc.)
    #[serde(rename = "type")]
    pub widget_type: String,
    /// Widget ID
    #[serde(default)]
    pub id: Option<String>,
    /// Content (for text widgets)
    #[serde(default)]
    pub content: Option<String>,
    /// Style name
    #[serde(default)]
    pub style: Option<String>,
    /// Data binding expression
    #[serde(default)]
    pub data: Option<String>,
    /// Chart type (for chart widgets)
    #[serde(default)]
    pub chart_type: Option<String>,
    /// X axis field
    #[serde(default)]
    pub x: Option<String>,
    /// Y axis field
    #[serde(default)]
    pub y: Option<String>,
    /// Color field
    #[serde(default)]
    pub color: Option<String>,
    /// Model source (for inference widgets)
    #[serde(default)]
    pub model_source: Option<String>,
    /// Inference engine (for inference widgets, e.g., "ngram-v1", "onnx-simd")
    #[serde(default)]
    pub engine: Option<String>,
    /// Acceleration preference (for inference widgets, e.g., "auto", "simd", "wgpu")
    #[serde(default)]
    pub acceleration: Option<String>,
    /// Additional properties
    #[serde(flatten)]
    pub extra: HashMap<String, serde_yaml_ng::Value>,
}

/// Interaction configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Interaction {
    /// Event trigger
    pub trigger: String,
    /// Action type
    pub action: String,
    /// Target (for navigation)
    #[serde(default)]
    pub target: Option<String>,
    /// Content (for tooltips)
    #[serde(default)]
    pub content: Option<String>,
    /// Script (for custom actions)
    #[serde(default)]
    pub script: Option<String>,
}

/// Theme configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThemeConfig {
    /// Theme preset (light, dark)
    #[serde(default)]
    pub preset: Option<String>,
    /// Custom colors
    #[serde(default)]
    pub colors: HashMap<String, String>,
}

impl Manifest {
    /// Parse a manifest from YAML string.
    ///
    /// # Errors
    ///
    /// Returns an error if the YAML is invalid.
    pub fn from_yaml(yaml: &str) -> Result<Self, serde_yaml_ng::Error> {
        serde_yaml_ng::from_str(yaml)
    }

    /// Serialize manifest to YAML string.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn to_yaml(&self) -> Result<String, serde_yaml_ng::Error> {
        serde_yaml_ng::to_string(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXAMPLE_YAML: &str = r#"
presentar: "0.1"
name: "test-app"
version: "1.0.0"
description: "Test application"

layout:
  type: "dashboard"
  columns: 12
  gap: 16
  sections:
    - id: "header"
      span: [1, 12]
      widgets:
        - type: "text"
          content: "Hello World"
          style: "heading-1"
"#;

    #[test]
    fn test_parse_manifest() {
        let manifest = Manifest::from_yaml(EXAMPLE_YAML).unwrap();
        assert_eq!(manifest.name, "test-app");
        assert_eq!(manifest.version, "1.0.0");
        assert_eq!(manifest.layout.columns, 12);
        assert_eq!(manifest.layout.sections.len(), 1);
    }

    #[test]
    fn test_parse_section() {
        let manifest = Manifest::from_yaml(EXAMPLE_YAML).unwrap();
        let section = &manifest.layout.sections[0];
        assert_eq!(section.id, "header");
        assert_eq!(section.span, Some([1, 12]));
        assert_eq!(section.widgets.len(), 1);
    }

    #[test]
    fn test_parse_widget() {
        let manifest = Manifest::from_yaml(EXAMPLE_YAML).unwrap();
        let widget = &manifest.layout.sections[0].widgets[0];
        assert_eq!(widget.widget_type, "text");
        assert_eq!(widget.content, Some("Hello World".to_string()));
        assert_eq!(widget.style, Some("heading-1".to_string()));
    }

    #[test]
    fn test_roundtrip() {
        let manifest = Manifest::from_yaml(EXAMPLE_YAML).unwrap();
        let yaml = manifest.to_yaml().unwrap();
        let manifest2 = Manifest::from_yaml(&yaml).unwrap();
        assert_eq!(manifest.name, manifest2.name);
        assert_eq!(manifest.version, manifest2.version);
    }

    #[test]
    fn test_data_source() {
        let yaml = r#"
presentar: "0.1"
name: "test"
version: "1.0.0"
data:
  transactions:
    source: "pacha://datasets/transactions:latest"
    format: "ald"
    refresh: "5m"
layout:
  type: "app"
"#;

        let manifest = Manifest::from_yaml(yaml).unwrap();
        assert!(manifest.data.contains_key("transactions"));
        let ds = &manifest.data["transactions"];
        assert_eq!(ds.format, "ald");
        assert_eq!(ds.refresh, Some("5m".to_string()));
    }

    #[test]
    fn test_model_ref() {
        let yaml = r#"
presentar: "0.1"
name: "test"
version: "1.0.0"
models:
  classifier:
    source: "pacha://models/classifier:1.0.0"
    format: "apr"
layout:
  type: "app"
"#;

        let manifest = Manifest::from_yaml(yaml).unwrap();
        assert!(manifest.models.contains_key("classifier"));
        let model = &manifest.models["classifier"];
        assert_eq!(model.format, "apr");
    }

    // =========================================================================
    // Theme Config Tests
    // =========================================================================

    #[test]
    fn test_theme_preset() {
        let yaml = r#"
presentar: "0.1"
name: "test"
version: "1.0.0"
layout:
  type: "app"
theme:
  preset: "dark"
"#;

        let manifest = Manifest::from_yaml(yaml).unwrap();
        assert!(manifest.theme.is_some());
        let theme = manifest.theme.unwrap();
        assert_eq!(theme.preset, Some("dark".to_string()));
    }

    #[test]
    fn test_theme_custom_colors() {
        let yaml = r##"
presentar: "0.1"
name: "test"
version: "1.0.0"
layout:
  type: "app"
theme:
  preset: "light"
  colors:
    primary: "#6366f1"
    danger: "#ef4444"
    success: "#10b981"
"##;

        let manifest = Manifest::from_yaml(yaml).unwrap();
        let theme = manifest.theme.unwrap();
        assert_eq!(theme.colors.get("primary"), Some(&"#6366f1".to_string()));
        assert_eq!(theme.colors.get("danger"), Some(&"#ef4444".to_string()));
        assert_eq!(theme.colors.get("success"), Some(&"#10b981".to_string()));
    }

    // =========================================================================
    // Interaction Tests
    // =========================================================================

    #[test]
    fn test_interaction_navigate() {
        let yaml = r#"
presentar: "0.1"
name: "test"
version: "1.0.0"
layout:
  type: "app"
interactions:
  - trigger: "table.row.click"
    action: "navigate"
    target: "/details/{{ row.id }}"
"#;

        let manifest = Manifest::from_yaml(yaml).unwrap();
        assert_eq!(manifest.interactions.len(), 1);
        let interaction = &manifest.interactions[0];
        assert_eq!(interaction.trigger, "table.row.click");
        assert_eq!(interaction.action, "navigate");
        assert_eq!(
            interaction.target,
            Some("/details/{{ row.id }}".to_string())
        );
    }

    #[test]
    fn test_interaction_tooltip() {
        let yaml = r#"
presentar: "0.1"
name: "test"
version: "1.0.0"
layout:
  type: "app"
interactions:
  - trigger: "chart.point.hover"
    action: "tooltip"
    content: "Value: {{ point.value }}"
"#;

        let manifest = Manifest::from_yaml(yaml).unwrap();
        let interaction = &manifest.interactions[0];
        assert_eq!(interaction.action, "tooltip");
        assert_eq!(
            interaction.content,
            Some("Value: {{ point.value }}".to_string())
        );
    }

    #[test]
    fn test_interaction_script() {
        let yaml = r#"
presentar: "0.1"
name: "test"
version: "1.0.0"
layout:
  type: "app"
interactions:
  - trigger: "button.click"
    action: "custom"
    script: |
      let x = state.count + 1
      set_state("count", x)
"#;

        let manifest = Manifest::from_yaml(yaml).unwrap();
        let interaction = &manifest.interactions[0];
        assert_eq!(interaction.action, "custom");
        assert!(interaction.script.is_some());
        assert!(interaction.script.as_ref().unwrap().contains("set_state"));
    }

    // =========================================================================
    // Score Tests
    // =========================================================================

    #[test]
    fn test_score_metadata() {
        let yaml = r#"
presentar: "0.1"
name: "test"
version: "1.0.0"
score:
  grade: "A"
  value: 92.3
  coverage: 94.1
layout:
  type: "app"
"#;

        let manifest = Manifest::from_yaml(yaml).unwrap();
        assert!(manifest.score.is_some());
        let score = manifest.score.unwrap();
        assert_eq!(score.grade, "A");
        assert!((score.value - 92.3).abs() < 0.01);
        assert_eq!(score.coverage, Some(94.1));
    }

    #[test]
    fn test_score_without_coverage() {
        let yaml = r#"
presentar: "0.1"
name: "test"
version: "1.0.0"
score:
  grade: "B+"
  value: 82.0
layout:
  type: "app"
"#;

        let manifest = Manifest::from_yaml(yaml).unwrap();
        let score = manifest.score.unwrap();
        assert_eq!(score.grade, "B+");
        assert_eq!(score.coverage, None);
    }

    // =========================================================================
    // Default Value Tests
    // =========================================================================

    #[test]
    fn test_default_columns() {
        let yaml = r#"
presentar: "0.1"
name: "test"
version: "1.0.0"
layout:
  type: "dashboard"
"#;

        let manifest = Manifest::from_yaml(yaml).unwrap();
        assert_eq!(manifest.layout.columns, 12); // Default value
    }

    #[test]
    fn test_default_gap() {
        let yaml = r#"
presentar: "0.1"
name: "test"
version: "1.0.0"
layout:
  type: "dashboard"
"#;

        let manifest = Manifest::from_yaml(yaml).unwrap();
        assert_eq!(manifest.layout.gap, 16); // Default value
    }

    #[test]
    fn test_default_data_format() {
        let yaml = r#"
presentar: "0.1"
name: "test"
version: "1.0.0"
data:
  test_data:
    source: "file://data.csv"
layout:
  type: "app"
"#;

        let manifest = Manifest::from_yaml(yaml).unwrap();
        let ds = &manifest.data["test_data"];
        assert_eq!(ds.format, "ald"); // Default format
    }

    #[test]
    fn test_default_model_format() {
        let yaml = r#"
presentar: "0.1"
name: "test"
version: "1.0.0"
models:
  test_model:
    source: "file://model.bin"
layout:
  type: "app"
"#;

        let manifest = Manifest::from_yaml(yaml).unwrap();
        let model = &manifest.models["test_model"];
        assert_eq!(model.format, "apr"); // Default format
    }

    // =========================================================================
    // Widget Config Tests
    // =========================================================================

    #[test]
    fn test_chart_widget_config() {
        let yaml = r#"
presentar: "0.1"
name: "test"
version: "1.0.0"
layout:
  type: "dashboard"
  sections:
    - id: "chart-section"
      widgets:
        - type: "chart"
          chart_type: "line"
          data: "{{ data.transactions }}"
          x: "timestamp"
          y: "amount"
          color: "{{ predictions.fraud }}"
"#;

        let manifest = Manifest::from_yaml(yaml).unwrap();
        let widget = &manifest.layout.sections[0].widgets[0];
        assert_eq!(widget.widget_type, "chart");
        assert_eq!(widget.chart_type, Some("line".to_string()));
        assert_eq!(widget.x, Some("timestamp".to_string()));
        assert_eq!(widget.y, Some("amount".to_string()));
        assert!(widget.color.is_some());
    }

    #[test]
    fn test_widget_extra_properties() {
        let yaml = r#"
presentar: "0.1"
name: "test"
version: "1.0.0"
layout:
  type: "app"
  sections:
    - id: "main"
      widgets:
        - type: "data-table"
          data: "{{ data.items }}"
          pagination: 50
          sortable: true
          filterable: true
"#;

        let manifest = Manifest::from_yaml(yaml).unwrap();
        let widget = &manifest.layout.sections[0].widgets[0];
        assert_eq!(widget.widget_type, "data-table");
        assert!(widget.extra.contains_key("pagination"));
        assert!(widget.extra.contains_key("sortable"));
        assert!(widget.extra.contains_key("filterable"));
    }

    // =========================================================================
    // Multiple Sections Tests
    // =========================================================================

    #[test]
    fn test_multiple_sections() {
        let yaml = r#"
presentar: "0.1"
name: "dashboard"
version: "1.0.0"
layout:
  type: "dashboard"
  columns: 12
  sections:
    - id: "header"
      span: [1, 12]
    - id: "sidebar"
      span: [1, 3]
    - id: "main"
      span: [4, 12]
    - id: "footer"
      span: [1, 12]
"#;

        let manifest = Manifest::from_yaml(yaml).unwrap();
        assert_eq!(manifest.layout.sections.len(), 4);
        assert_eq!(manifest.layout.sections[0].id, "header");
        assert_eq!(manifest.layout.sections[1].span, Some([1, 3]));
        assert_eq!(manifest.layout.sections[2].span, Some([4, 12]));
    }

    // =========================================================================
    // Error Cases
    // =========================================================================

    #[test]
    fn test_missing_required_fields() {
        let yaml = r#"
presentar: "0.1"
name: "test"
"#;

        let result = Manifest::from_yaml(yaml);
        assert!(result.is_err()); // Missing version and layout
    }

    #[test]
    fn test_invalid_yaml() {
        let yaml = "this is not valid yaml: [}";
        let result = Manifest::from_yaml(yaml);
        assert!(result.is_err());
    }

    // =========================================================================
    // Full Integration Test
    // =========================================================================

    #[test]
    fn test_complex_manifest() {
        let yaml = r##"
presentar: "0.1"
name: "fraud-detection-dashboard"
version: "1.0.0"
description: "Real-time fraud detection monitoring"

score:
  grade: "A"
  value: 92.3
  coverage: 94.1

data:
  transactions:
    source: "pacha://datasets/transactions:latest"
    format: "ald"
    refresh: "5m"
  predictions:
    source: "./predictions.ald"

models:
  fraud_detector:
    source: "pacha://models/fraud-detector:1.2.0"

layout:
  type: "dashboard"
  columns: 12
  gap: 16
  sections:
    - id: "header"
      span: [1, 12]
      widgets:
        - type: "text"
          content: "Fraud Detection Dashboard"
          style: "heading-1"
        - type: "model-card"
          id: "model-info"

    - id: "metrics"
      span: [1, 4]
      widgets:
        - type: "metric"
          data: "{{ data.transactions | count | rate(1m) }}"

    - id: "chart"
      span: [5, 12]
      widgets:
        - type: "chart"
          chart_type: "line"
          data: "{{ data.transactions }}"
          x: "timestamp"
          y: "amount"

interactions:
  - trigger: "chart.point.hover"
    action: "tooltip"
    content: "Amount: {{ point.amount }}"

theme:
  preset: "dark"
  colors:
    primary: "#6366f1"
    danger: "#ef4444"
"##;

        let manifest = Manifest::from_yaml(yaml).unwrap();

        // Basic info
        assert_eq!(manifest.name, "fraud-detection-dashboard");
        assert_eq!(manifest.version, "1.0.0");
        assert!(!manifest.description.is_empty());

        // Score
        assert!(manifest.score.is_some());

        // Data sources
        assert_eq!(manifest.data.len(), 2);
        assert!(manifest.data.contains_key("transactions"));
        assert!(manifest.data.contains_key("predictions"));

        // Models
        assert_eq!(manifest.models.len(), 1);
        assert!(manifest.models.contains_key("fraud_detector"));

        // Layout
        assert_eq!(manifest.layout.layout_type, "dashboard");
        assert_eq!(manifest.layout.columns, 12);
        assert_eq!(manifest.layout.sections.len(), 3);

        // Interactions
        assert_eq!(manifest.interactions.len(), 1);

        // Theme
        assert!(manifest.theme.is_some());
        let theme = manifest.theme.unwrap();
        assert_eq!(theme.preset, Some("dark".to_string()));
        assert_eq!(theme.colors.len(), 2);
    }
}
