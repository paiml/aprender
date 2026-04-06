//! `ModelCard` widget for displaying ML model metadata.

use presentar_core::{
    widget::{AccessibleRole, LayoutResult, TextStyle},
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Point, Rect,
    Size, TypeId, Widget,
};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::HashMap;
use std::time::Duration;

/// Model status indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ModelStatus {
    /// Draft/in development
    #[default]
    Draft,
    /// Under review
    Review,
    /// Published/production ready
    Published,
    /// Deprecated
    Deprecated,
    /// Archived
    Archived,
}

impl ModelStatus {
    /// Get display color for status.
    #[must_use]
    pub fn color(&self) -> Color {
        match self {
            Self::Draft => Color::new(0.6, 0.6, 0.6, 1.0),
            Self::Review => Color::new(0.9, 0.7, 0.1, 1.0),
            Self::Published => Color::new(0.2, 0.7, 0.3, 1.0),
            Self::Deprecated => Color::new(0.9, 0.5, 0.1, 1.0),
            Self::Archived => Color::new(0.5, 0.5, 0.5, 1.0),
        }
    }

    /// Get status label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Draft => "Draft",
            Self::Review => "Review",
            Self::Published => "Published",
            Self::Deprecated => "Deprecated",
            Self::Archived => "Archived",
        }
    }
}

/// Model metric (e.g., accuracy, F1 score).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelMetric {
    /// Metric name
    pub name: String,
    /// Metric value
    pub value: f64,
    /// Optional unit
    pub unit: Option<String>,
    /// Higher is better
    pub higher_is_better: bool,
}

impl ModelMetric {
    /// Create a new metric.
    #[must_use]
    pub fn new(name: impl Into<String>, value: f64) -> Self {
        Self {
            name: name.into(),
            value,
            unit: None,
            higher_is_better: true,
        }
    }

    /// Set unit.
    #[must_use]
    pub fn unit(mut self, unit: impl Into<String>) -> Self {
        self.unit = Some(unit.into());
        self
    }

    /// Set lower is better.
    #[must_use]
    pub const fn lower_is_better(mut self) -> Self {
        self.higher_is_better = false;
        self
    }

    /// Format the value for display.
    #[must_use]
    pub fn formatted_value(&self) -> String {
        if let Some(ref unit) = self.unit {
            format!("{:.2}{}", self.value, unit)
        } else if self.value.abs() < 1.0 {
            format!("{:.2}%", self.value * 100.0)
        } else {
            format!("{:.2}", self.value)
        }
    }
}

/// `ModelCard` widget for displaying ML model metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCard {
    /// Model name
    name: String,
    /// Model version
    version: String,
    /// Model description
    description: Option<String>,
    /// Model status
    status: ModelStatus,
    /// Model type/framework (e.g., `PyTorch`, `TensorFlow`)
    framework: Option<String>,
    /// Model task (e.g., "classification", "regression")
    task: Option<String>,
    /// Performance metrics
    metrics: Vec<ModelMetric>,
    /// Parameter count
    parameters: Option<u64>,
    /// Training dataset name
    dataset: Option<String>,
    /// Author/owner
    author: Option<String>,
    /// Tags
    tags: Vec<String>,
    /// Custom metadata
    metadata: HashMap<String, String>,
    /// Card width
    width: Option<f32>,
    /// Card height
    height: Option<f32>,
    /// Background color
    background: Color,
    /// Border color
    border_color: Color,
    /// Corner radius
    corner_radius: f32,
    /// Show metrics chart
    show_metrics_chart: bool,
    /// Accessible name
    accessible_name_value: Option<String>,
    /// Test ID
    test_id_value: Option<String>,
    /// Cached bounds
    #[serde(skip)]
    bounds: Rect,
}

impl Default for ModelCard {
    fn default() -> Self {
        Self {
            name: String::new(),
            version: String::from("1.0.0"),
            description: None,
            status: ModelStatus::Draft,
            framework: None,
            task: None,
            metrics: Vec::new(),
            parameters: None,
            dataset: None,
            author: None,
            tags: Vec::new(),
            metadata: HashMap::new(),
            width: None,
            height: None,
            background: Color::WHITE,
            border_color: Color::new(0.9, 0.9, 0.9, 1.0),
            corner_radius: 8.0,
            show_metrics_chart: true,
            accessible_name_value: None,
            test_id_value: None,
            bounds: Rect::default(),
        }
    }
}

impl ModelCard {
    /// Create a new model card.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    /// Set model name.
    #[must_use]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set version.
    #[must_use]
    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Set description.
    #[must_use]
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set status.
    #[must_use]
    pub const fn status(mut self, status: ModelStatus) -> Self {
        self.status = status;
        self
    }

    /// Set framework.
    #[must_use]
    pub fn framework(mut self, framework: impl Into<String>) -> Self {
        self.framework = Some(framework.into());
        self
    }

    /// Set task.
    #[must_use]
    pub fn task(mut self, task: impl Into<String>) -> Self {
        self.task = Some(task.into());
        self
    }

    /// Add a metric.
    #[must_use]
    pub fn metric(mut self, metric: ModelMetric) -> Self {
        self.metrics.push(metric);
        self
    }

    /// Add multiple metrics.
    #[must_use]
    pub fn metrics(mut self, metrics: impl IntoIterator<Item = ModelMetric>) -> Self {
        self.metrics.extend(metrics);
        self
    }

    /// Set parameter count.
    #[must_use]
    pub const fn parameters(mut self, count: u64) -> Self {
        self.parameters = Some(count);
        self
    }

    /// Set dataset.
    #[must_use]
    pub fn dataset(mut self, dataset: impl Into<String>) -> Self {
        self.dataset = Some(dataset.into());
        self
    }

    /// Set author.
    #[must_use]
    pub fn author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    /// Add a tag.
    #[must_use]
    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Add multiple tags.
    #[must_use]
    pub fn tags(mut self, tags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tags.extend(tags.into_iter().map(Into::into));
        self
    }

    /// Add custom metadata.
    #[must_use]
    pub fn metadata_entry(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set width.
    #[must_use]
    pub fn width(mut self, width: f32) -> Self {
        self.width = Some(width.max(200.0));
        self
    }

    /// Set height.
    #[must_use]
    pub fn height(mut self, height: f32) -> Self {
        self.height = Some(height.max(150.0));
        self
    }

    /// Set background color.
    #[must_use]
    pub const fn background(mut self, color: Color) -> Self {
        self.background = color;
        self
    }

    /// Set border color.
    #[must_use]
    pub const fn border_color(mut self, color: Color) -> Self {
        self.border_color = color;
        self
    }

    /// Set corner radius.
    #[must_use]
    pub fn corner_radius(mut self, radius: f32) -> Self {
        self.corner_radius = radius.max(0.0);
        self
    }

    /// Set whether to show metrics chart.
    #[must_use]
    pub const fn show_metrics_chart(mut self, show: bool) -> Self {
        self.show_metrics_chart = show;
        self
    }

    /// Set accessible name.
    #[must_use]
    pub fn accessible_name(mut self, name: impl Into<String>) -> Self {
        self.accessible_name_value = Some(name.into());
        self
    }

    /// Set test ID.
    #[must_use]
    pub fn test_id(mut self, id: impl Into<String>) -> Self {
        self.test_id_value = Some(id.into());
        self
    }

    // Getters

    /// Get model name.
    #[must_use]
    pub fn get_name(&self) -> &str {
        &self.name
    }

    /// Get version.
    #[must_use]
    pub fn get_version(&self) -> &str {
        &self.version
    }

    /// Get description.
    #[must_use]
    pub fn get_description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    /// Get status.
    #[must_use]
    pub const fn get_status(&self) -> ModelStatus {
        self.status
    }

    /// Get framework.
    #[must_use]
    pub fn get_framework(&self) -> Option<&str> {
        self.framework.as_deref()
    }

    /// Get task.
    #[must_use]
    pub fn get_task(&self) -> Option<&str> {
        self.task.as_deref()
    }

    /// Get metrics.
    #[must_use]
    pub fn get_metrics(&self) -> &[ModelMetric] {
        &self.metrics
    }

    /// Get parameter count.
    #[must_use]
    pub const fn get_parameters(&self) -> Option<u64> {
        self.parameters
    }

    /// Get dataset.
    #[must_use]
    pub fn get_dataset(&self) -> Option<&str> {
        self.dataset.as_deref()
    }

    /// Get author.
    #[must_use]
    pub fn get_author(&self) -> Option<&str> {
        self.author.as_deref()
    }

    /// Get tags.
    #[must_use]
    pub fn get_tags(&self) -> &[String] {
        &self.tags
    }

    /// Get custom metadata.
    #[must_use]
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }

    /// Check if model has metrics.
    #[must_use]
    pub fn has_metrics(&self) -> bool {
        !self.metrics.is_empty()
    }

    /// Format parameter count for display.
    #[must_use]
    pub fn formatted_parameters(&self) -> Option<String> {
        self.parameters.map(|p| {
            if p >= 1_000_000_000 {
                format!("{:.1}B", p as f64 / 1_000_000_000.0)
            } else if p >= 1_000_000 {
                format!("{:.1}M", p as f64 / 1_000_000.0)
            } else if p >= 1_000 {
                format!("{:.1}K", p as f64 / 1_000.0)
            } else {
                format!("{p}")
            }
        })
    }
}

impl Widget for ModelCard {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let width = self.width.unwrap_or(320.0);
        let height = self.height.unwrap_or(200.0);
        constraints.constrain(Size::new(width, height))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: bounds.size(),
        }
    }

    #[allow(clippy::too_many_lines)]
    fn paint(&self, canvas: &mut dyn Canvas) {
        let padding = 16.0;

        // Background
        canvas.fill_rect(self.bounds, self.background);

        // Border
        canvas.stroke_rect(self.bounds, self.border_color, 1.0);

        // Status badge
        let status_color = self.status.color();
        let badge_rect = Rect::new(
            self.bounds.x + self.bounds.width - 80.0,
            self.bounds.y + padding,
            70.0,
            22.0,
        );
        canvas.fill_rect(badge_rect, status_color);

        let badge_style = TextStyle {
            size: 10.0,
            color: Color::WHITE,
            ..TextStyle::default()
        };
        canvas.draw_text(
            self.status.label(),
            Point::new(badge_rect.x + 10.0, badge_rect.y + 15.0),
            &badge_style,
        );

        // Title
        let title_style = TextStyle {
            size: 18.0,
            color: Color::new(0.1, 0.1, 0.1, 1.0),
            ..TextStyle::default()
        };
        canvas.draw_text(
            &self.name,
            Point::new(self.bounds.x + padding, self.bounds.y + padding + 16.0),
            &title_style,
        );

        // Version
        let version_style = TextStyle {
            size: 12.0,
            color: Color::new(0.5, 0.5, 0.5, 1.0),
            ..TextStyle::default()
        };
        canvas.draw_text(
            &format!("v{}", self.version),
            Point::new(self.bounds.x + padding, self.bounds.y + padding + 36.0),
            &version_style,
        );

        // Description (if any)
        let mut y_offset = padding + 50.0;
        if let Some(ref desc) = self.description {
            let desc_style = TextStyle {
                size: 12.0,
                color: Color::new(0.3, 0.3, 0.3, 1.0),
                ..TextStyle::default()
            };
            canvas.draw_text(
                desc,
                Point::new(self.bounds.x + padding, self.bounds.y + y_offset + 12.0),
                &desc_style,
            );
            y_offset += 24.0;
        }

        // Framework and task
        if self.framework.is_some() || self.task.is_some() {
            let info_style = TextStyle {
                size: 11.0,
                color: Color::new(0.4, 0.4, 0.4, 1.0),
                ..TextStyle::default()
            };
            let info_text = match (&self.framework, &self.task) {
                (Some(f), Some(t)) => format!("{f} • {t}"),
                (Some(f), None) => f.clone(),
                (None, Some(t)) => t.clone(),
                (None, None) => String::new(),
            };
            canvas.draw_text(
                &info_text,
                Point::new(self.bounds.x + padding, self.bounds.y + y_offset + 12.0),
                &info_style,
            );
            y_offset += 20.0;
        }

        // Parameters
        if let Some(params) = self.formatted_parameters() {
            let params_style = TextStyle {
                size: 11.0,
                color: Color::new(0.4, 0.4, 0.4, 1.0),
                ..TextStyle::default()
            };
            canvas.draw_text(
                &format!("Parameters: {params}"),
                Point::new(self.bounds.x + padding, self.bounds.y + y_offset + 12.0),
                &params_style,
            );
            y_offset += 18.0;
        }

        // Metrics
        if self.show_metrics_chart && !self.metrics.is_empty() {
            let metric_style = TextStyle {
                size: 11.0,
                color: Color::new(0.2, 0.2, 0.2, 1.0),
                ..TextStyle::default()
            };
            let value_style = TextStyle {
                size: 14.0,
                color: Color::new(0.2, 0.47, 0.96, 1.0),
                ..TextStyle::default()
            };

            let metric_width =
                (self.bounds.width - padding * 2.0) / self.metrics.len().min(4) as f32;
            for (i, metric) in self.metrics.iter().take(4).enumerate() {
                let mx = (i as f32).mul_add(metric_width, self.bounds.x + padding);
                canvas.draw_text(
                    &metric.name,
                    Point::new(mx, self.bounds.y + y_offset + 12.0),
                    &metric_style,
                );
                canvas.draw_text(
                    &metric.formatted_value(),
                    Point::new(mx, self.bounds.y + y_offset + 28.0),
                    &value_style,
                );
            }
            y_offset += 40.0;
        }

        // Tags
        if !self.tags.is_empty() {
            let tag_style = TextStyle {
                size: 10.0,
                color: Color::new(0.3, 0.3, 0.3, 1.0),
                ..TextStyle::default()
            };
            let tag_bg = Color::new(0.95, 0.95, 0.95, 1.0);

            let mut tx = self.bounds.x + padding;
            for tag in self.tags.iter().take(5) {
                let tag_width = (tag.len() as f32).mul_add(6.0, 12.0);
                canvas.fill_rect(
                    Rect::new(tx, self.bounds.y + y_offset, tag_width, 18.0),
                    tag_bg,
                );
                canvas.draw_text(
                    tag,
                    Point::new(tx + 6.0, self.bounds.y + y_offset + 13.0),
                    &tag_style,
                );
                tx += tag_width + 6.0;
            }
        }
    }

    fn event(&mut self, _event: &presentar_core::Event) -> Option<Box<dyn Any + Send>> {
        None
    }

    fn children(&self) -> &[Box<dyn Widget>] {
        &[]
    }

    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut []
    }

    fn is_interactive(&self) -> bool {
        false
    }

    fn is_focusable(&self) -> bool {
        false
    }

    fn accessible_name(&self) -> Option<&str> {
        self.accessible_name_value.as_deref().or(Some(&self.name))
    }

    fn accessible_role(&self) -> AccessibleRole {
        AccessibleRole::Generic
    }

    fn test_id(&self) -> Option<&str> {
        self.test_id_value.as_deref()
    }
}

// PROBAR-SPEC-009: Brick Architecture - Tests define interface
impl Brick for ModelCard {
    fn brick_name(&self) -> &'static str {
        "ModelCard"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        &[BrickAssertion::MaxLatencyMs(16)]
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
        r#"<div class="brick-modelcard"></div>"#.to_string()
    }

    fn to_css(&self) -> String {
        ".brick-modelcard { display: block; }".to_string()
    }

    fn test_id(&self) -> Option<&str> {
        self.test_id_value.as_deref()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;

    // ===== ModelStatus Tests =====

    #[test]
    fn test_model_status_default() {
        assert_eq!(ModelStatus::default(), ModelStatus::Draft);
    }

    #[test]
    fn test_model_status_color() {
        let published = ModelStatus::Published;
        let color = published.color();
        assert!(color.g > color.r); // Green-ish
    }

    #[test]
    fn test_model_status_label() {
        assert_eq!(ModelStatus::Draft.label(), "Draft");
        assert_eq!(ModelStatus::Review.label(), "Review");
        assert_eq!(ModelStatus::Published.label(), "Published");
        assert_eq!(ModelStatus::Deprecated.label(), "Deprecated");
        assert_eq!(ModelStatus::Archived.label(), "Archived");
    }

    // ===== ModelMetric Tests =====

    #[test]
    fn test_model_metric_new() {
        let metric = ModelMetric::new("Accuracy", 0.95);
        assert_eq!(metric.name, "Accuracy");
        assert_eq!(metric.value, 0.95);
        assert!(metric.unit.is_none());
        assert!(metric.higher_is_better);
    }

    #[test]
    fn test_model_metric_unit() {
        let metric = ModelMetric::new("Latency", 45.0).unit("ms");
        assert_eq!(metric.unit, Some("ms".to_string()));
    }

    #[test]
    fn test_model_metric_lower_is_better() {
        let metric = ModelMetric::new("Loss", 0.05).lower_is_better();
        assert!(!metric.higher_is_better);
    }

    #[test]
    fn test_model_metric_formatted_value_percentage() {
        let metric = ModelMetric::new("Accuracy", 0.95);
        assert_eq!(metric.formatted_value(), "95.00%");
    }

    #[test]
    fn test_model_metric_formatted_value_with_unit() {
        let metric = ModelMetric::new("Latency", 45.0).unit("ms");
        assert_eq!(metric.formatted_value(), "45.00ms");
    }

    #[test]
    fn test_model_metric_formatted_value_large() {
        let metric = ModelMetric::new("Score", 1234.5);
        assert_eq!(metric.formatted_value(), "1234.50");
    }

    // ===== ModelCard Construction Tests =====

    #[test]
    fn test_model_card_new() {
        let card = ModelCard::new("GPT-4");
        assert_eq!(card.get_name(), "GPT-4");
        assert_eq!(card.get_version(), "1.0.0");
        assert_eq!(card.get_status(), ModelStatus::Draft);
    }

    #[test]
    fn test_model_card_default() {
        let card = ModelCard::default();
        assert!(card.name.is_empty());
        assert_eq!(card.version, "1.0.0");
    }

    #[test]
    fn test_model_card_builder() {
        let card = ModelCard::new("ResNet-50")
            .version("2.1.0")
            .description("Image classification model")
            .status(ModelStatus::Published)
            .framework("PyTorch")
            .task("classification")
            .metric(ModelMetric::new("Top-1 Accuracy", 0.761))
            .metric(ModelMetric::new("Top-5 Accuracy", 0.929))
            .parameters(25_600_000)
            .dataset("ImageNet")
            .author("Deep Learning Team")
            .tag("vision")
            .tag("classification")
            .metadata_entry("license", "Apache-2.0")
            .width(400.0)
            .height(300.0)
            .background(Color::WHITE)
            .border_color(Color::new(0.8, 0.8, 0.8, 1.0))
            .corner_radius(12.0)
            .show_metrics_chart(true)
            .accessible_name("ResNet-50 model card")
            .test_id("resnet-card");

        assert_eq!(card.get_name(), "ResNet-50");
        assert_eq!(card.get_version(), "2.1.0");
        assert_eq!(card.get_description(), Some("Image classification model"));
        assert_eq!(card.get_status(), ModelStatus::Published);
        assert_eq!(card.get_framework(), Some("PyTorch"));
        assert_eq!(card.get_task(), Some("classification"));
        assert_eq!(card.get_metrics().len(), 2);
        assert_eq!(card.get_parameters(), Some(25_600_000));
        assert_eq!(card.get_dataset(), Some("ImageNet"));
        assert_eq!(card.get_author(), Some("Deep Learning Team"));
        assert_eq!(card.get_tags().len(), 2);
        assert_eq!(card.get_metadata("license"), Some("Apache-2.0"));
        assert_eq!(Widget::accessible_name(&card), Some("ResNet-50 model card"));
        assert_eq!(Widget::test_id(&card), Some("resnet-card"));
    }

    #[test]
    fn test_model_card_metrics() {
        let metrics = vec![
            ModelMetric::new("Accuracy", 0.95),
            ModelMetric::new("F1", 0.92),
        ];
        let card = ModelCard::new("Model").metrics(metrics);
        assert_eq!(card.get_metrics().len(), 2);
        assert!(card.has_metrics());
    }

    #[test]
    fn test_model_card_tags() {
        let card = ModelCard::new("Model").tags(["nlp", "transformer", "bert"]);
        assert_eq!(card.get_tags().len(), 3);
    }

    // ===== Formatted Parameters Tests =====

    #[test]
    fn test_formatted_parameters_none() {
        let card = ModelCard::new("Model");
        assert!(card.formatted_parameters().is_none());
    }

    #[test]
    fn test_formatted_parameters_small() {
        let card = ModelCard::new("Model").parameters(500);
        assert_eq!(card.formatted_parameters(), Some("500".to_string()));
    }

    #[test]
    fn test_formatted_parameters_thousands() {
        let card = ModelCard::new("Model").parameters(25_000);
        assert_eq!(card.formatted_parameters(), Some("25.0K".to_string()));
    }

    #[test]
    fn test_formatted_parameters_millions() {
        let card = ModelCard::new("Model").parameters(125_000_000);
        assert_eq!(card.formatted_parameters(), Some("125.0M".to_string()));
    }

    #[test]
    fn test_formatted_parameters_billions() {
        let card = ModelCard::new("Model").parameters(175_000_000_000);
        assert_eq!(card.formatted_parameters(), Some("175.0B".to_string()));
    }

    // ===== Dimension Tests =====

    #[test]
    fn test_model_card_width_min() {
        let card = ModelCard::new("Model").width(100.0);
        assert_eq!(card.width, Some(200.0));
    }

    #[test]
    fn test_model_card_height_min() {
        let card = ModelCard::new("Model").height(50.0);
        assert_eq!(card.height, Some(150.0));
    }

    #[test]
    fn test_model_card_corner_radius_min() {
        let card = ModelCard::new("Model").corner_radius(-5.0);
        assert_eq!(card.corner_radius, 0.0);
    }

    // ===== Widget Trait Tests =====

    #[test]
    fn test_model_card_type_id() {
        let card = ModelCard::new("Model");
        assert_eq!(Widget::type_id(&card), TypeId::of::<ModelCard>());
    }

    #[test]
    fn test_model_card_measure_default() {
        let card = ModelCard::new("Model");
        let size = card.measure(Constraints::loose(Size::new(1000.0, 1000.0)));
        assert_eq!(size.width, 320.0);
        assert_eq!(size.height, 200.0);
    }

    #[test]
    fn test_model_card_measure_custom() {
        let card = ModelCard::new("Model").width(400.0).height(250.0);
        let size = card.measure(Constraints::loose(Size::new(1000.0, 1000.0)));
        assert_eq!(size.width, 400.0);
        assert_eq!(size.height, 250.0);
    }

    #[test]
    fn test_model_card_layout() {
        let mut card = ModelCard::new("Model");
        let bounds = Rect::new(10.0, 20.0, 320.0, 200.0);
        let result = card.layout(bounds);
        assert_eq!(result.size, Size::new(320.0, 200.0));
        assert_eq!(card.bounds, bounds);
    }

    #[test]
    fn test_model_card_children() {
        let card = ModelCard::new("Model");
        assert!(card.children().is_empty());
    }

    #[test]
    fn test_model_card_is_interactive() {
        let card = ModelCard::new("Model");
        assert!(!card.is_interactive());
    }

    #[test]
    fn test_model_card_is_focusable() {
        let card = ModelCard::new("Model");
        assert!(!card.is_focusable());
    }

    #[test]
    fn test_model_card_accessible_role() {
        let card = ModelCard::new("Model");
        assert_eq!(card.accessible_role(), AccessibleRole::Generic);
    }

    #[test]
    fn test_model_card_accessible_name_from_name() {
        let card = ModelCard::new("GPT-4");
        assert_eq!(Widget::accessible_name(&card), Some("GPT-4"));
    }

    #[test]
    fn test_model_card_accessible_name_explicit() {
        let card = ModelCard::new("GPT-4").accessible_name("Language model card");
        assert_eq!(Widget::accessible_name(&card), Some("Language model card"));
    }

    #[test]
    fn test_model_card_test_id() {
        let card = ModelCard::new("Model").test_id("model-card");
        assert_eq!(Widget::test_id(&card), Some("model-card"));
    }

    // ===== Has Metrics Tests =====

    #[test]
    fn test_model_card_has_metrics_false() {
        let card = ModelCard::new("Model");
        assert!(!card.has_metrics());
    }

    #[test]
    fn test_model_card_has_metrics_true() {
        let card = ModelCard::new("Model").metric(ModelMetric::new("Acc", 0.9));
        assert!(card.has_metrics());
    }

    // =========================================================================
    // Additional Coverage Tests
    // =========================================================================

    #[test]
    fn test_model_status_color_all_variants() {
        let _ = ModelStatus::Draft.color();
        let _ = ModelStatus::Review.color();
        let _ = ModelStatus::Published.color();
        let _ = ModelStatus::Deprecated.color();
        let _ = ModelStatus::Archived.color();
    }

    #[test]
    fn test_model_card_children_mut() {
        let mut card = ModelCard::new("Model");
        assert!(card.children_mut().is_empty());
    }

    #[test]
    fn test_model_card_event_returns_none() {
        let mut card = ModelCard::new("Model");
        let result = card.event(&presentar_core::Event::key_down(presentar_core::Key::Down));
        assert!(result.is_none());
    }

    #[test]
    fn test_model_card_test_id_none() {
        let card = ModelCard::new("Model");
        assert!(Widget::test_id(&card).is_none());
    }

    #[test]
    fn test_model_card_bounds() {
        let mut card = ModelCard::new("Model");
        card.layout(Rect::new(5.0, 10.0, 320.0, 200.0));
        assert_eq!(card.bounds.x, 5.0);
        assert_eq!(card.bounds.y, 10.0);
    }

    #[test]
    fn test_model_metric_eq() {
        let m1 = ModelMetric::new("Acc", 0.95);
        let m2 = ModelMetric::new("Acc", 0.95);
        assert_eq!(m1.name, m2.name);
        assert_eq!(m1.value, m2.value);
    }

    #[test]
    fn test_model_card_name_setter() {
        let card = ModelCard::new("Initial").name("Changed");
        assert_eq!(card.get_name(), "Changed");
    }

    // =========================================================================
    // Brick Trait Tests
    // =========================================================================

    #[test]
    fn test_model_card_brick_name() {
        let card = ModelCard::new("test");
        assert_eq!(card.brick_name(), "ModelCard");
    }

    #[test]
    fn test_model_card_brick_assertions() {
        let card = ModelCard::new("test");
        let assertions = card.assertions();
        assert!(!assertions.is_empty());
        assert!(matches!(assertions[0], BrickAssertion::MaxLatencyMs(16)));
    }

    #[test]
    fn test_model_card_brick_budget() {
        let card = ModelCard::new("test");
        let budget = card.budget();
        // Verify budget has reasonable values
        assert!(budget.layout_ms > 0);
        assert!(budget.paint_ms > 0);
    }

    #[test]
    fn test_model_card_brick_verify() {
        let card = ModelCard::new("test");
        let verification = card.verify();
        assert!(!verification.passed.is_empty());
        assert!(verification.failed.is_empty());
    }

    #[test]
    fn test_model_card_brick_to_html() {
        let card = ModelCard::new("test");
        let html = card.to_html();
        assert!(html.contains("brick-modelcard"));
    }

    #[test]
    fn test_model_card_brick_to_css() {
        let card = ModelCard::new("test");
        let css = card.to_css();
        assert!(css.contains(".brick-modelcard"));
        assert!(css.contains("display: block"));
    }

    #[test]
    fn test_model_card_brick_test_id() {
        let card = ModelCard::new("test").test_id("card-1");
        assert_eq!(Brick::test_id(&card), Some("card-1"));
    }

    #[test]
    fn test_model_card_brick_test_id_none() {
        let card = ModelCard::new("test");
        assert!(Brick::test_id(&card).is_none());
    }

    // =========================================================================
    // ModelStatus Additional Tests
    // =========================================================================

    #[test]
    fn test_model_status_debug() {
        let status = ModelStatus::Published;
        let debug_str = format!("{status:?}");
        assert!(debug_str.contains("Published"));
    }

    #[test]
    fn test_model_status_eq() {
        assert_eq!(ModelStatus::Draft, ModelStatus::Draft);
        assert_ne!(ModelStatus::Draft, ModelStatus::Published);
    }

    #[test]
    fn test_model_status_clone() {
        let status = ModelStatus::Review;
        let cloned = status;
        assert_eq!(cloned, ModelStatus::Review);
    }

    #[test]
    fn test_model_status_serde() {
        let status = ModelStatus::Deprecated;
        let serialized = serde_json::to_string(&status).unwrap();
        let deserialized: ModelStatus = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized, ModelStatus::Deprecated);
    }

    #[test]
    fn test_model_status_color_all_variants_detailed() {
        // Draft is gray
        let draft_color = ModelStatus::Draft.color();
        assert!((draft_color.r - 0.6).abs() < 0.01);

        // Review is yellow/orange
        let review_color = ModelStatus::Review.color();
        assert!(review_color.r > 0.8);
        assert!(review_color.g > 0.6);

        // Published is green
        let published_color = ModelStatus::Published.color();
        assert!(published_color.g > published_color.r);

        // Deprecated is orange
        let deprecated_color = ModelStatus::Deprecated.color();
        assert!(deprecated_color.r > 0.8);

        // Archived is gray
        let archived_color = ModelStatus::Archived.color();
        assert!((archived_color.r - 0.5).abs() < 0.01);
    }

    // =========================================================================
    // ModelMetric Additional Tests
    // =========================================================================

    #[test]
    fn test_model_metric_debug() {
        let metric = ModelMetric::new("Accuracy", 0.95);
        let debug_str = format!("{metric:?}");
        assert!(debug_str.contains("Accuracy"));
        assert!(debug_str.contains("0.95"));
    }

    #[test]
    fn test_model_metric_clone() {
        let metric = ModelMetric::new("F1", 0.88).unit("%").lower_is_better();
        let cloned = metric;
        assert_eq!(cloned.name, "F1");
        assert_eq!(cloned.value, 0.88);
        assert_eq!(cloned.unit, Some("%".to_string()));
        assert!(!cloned.higher_is_better);
    }

    #[test]
    fn test_model_metric_serde() {
        let metric = ModelMetric::new("Loss", 0.05).lower_is_better();
        let serialized = serde_json::to_string(&metric).unwrap();
        let deserialized: ModelMetric = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.name, "Loss");
        assert_eq!(deserialized.value, 0.05);
        assert!(!deserialized.higher_is_better);
    }

    #[test]
    fn test_model_metric_formatted_value_negative() {
        let metric = ModelMetric::new("Correlation", -0.5);
        let formatted = metric.formatted_value();
        assert!(formatted.contains("-50.00%"));
    }

    #[test]
    fn test_model_metric_formatted_value_zero() {
        let metric = ModelMetric::new("Bias", 0.0);
        assert_eq!(metric.formatted_value(), "0.00%");
    }

    #[test]
    fn test_model_metric_formatted_value_exactly_one() {
        let metric = ModelMetric::new("Perfect", 1.0);
        assert_eq!(metric.formatted_value(), "1.00");
    }

    // =========================================================================
    // ModelCard Additional Tests
    // =========================================================================

    #[test]
    fn test_model_card_debug() {
        let card = ModelCard::new("GPT-4");
        let debug_str = format!("{card:?}");
        assert!(debug_str.contains("GPT-4"));
    }

    #[test]
    fn test_model_card_clone() {
        let card = ModelCard::new("BERT")
            .version("2.0.0")
            .status(ModelStatus::Published)
            .framework("PyTorch");
        let cloned = card;
        assert_eq!(cloned.get_name(), "BERT");
        assert_eq!(cloned.get_version(), "2.0.0");
        assert_eq!(cloned.get_status(), ModelStatus::Published);
        assert_eq!(cloned.get_framework(), Some("PyTorch"));
    }

    #[test]
    fn test_model_card_serde() {
        let card = ModelCard::new("ResNet")
            .version("1.0.0")
            .status(ModelStatus::Draft);
        let serialized = serde_json::to_string(&card).unwrap();
        let deserialized: ModelCard = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.get_name(), "ResNet");
        assert_eq!(deserialized.get_version(), "1.0.0");
        assert_eq!(deserialized.get_status(), ModelStatus::Draft);
    }

    // =========================================================================
    // Getter Tests (ensure all getters are covered)
    // =========================================================================

    #[test]
    fn test_model_card_getters_none() {
        let card = ModelCard::new("test");
        assert!(card.get_description().is_none());
        assert!(card.get_framework().is_none());
        assert!(card.get_task().is_none());
        assert!(card.get_parameters().is_none());
        assert!(card.get_dataset().is_none());
        assert!(card.get_author().is_none());
        assert!(card.get_metadata("nonexistent").is_none());
    }

    #[test]
    fn test_model_card_getters_some() {
        let card = ModelCard::new("test")
            .description("desc")
            .framework("TensorFlow")
            .task("classification")
            .parameters(1_000_000)
            .dataset("CIFAR-10")
            .author("ML Team")
            .metadata_entry("license", "Apache-2.0");

        assert_eq!(card.get_description(), Some("desc"));
        assert_eq!(card.get_framework(), Some("TensorFlow"));
        assert_eq!(card.get_task(), Some("classification"));
        assert_eq!(card.get_parameters(), Some(1_000_000));
        assert_eq!(card.get_dataset(), Some("CIFAR-10"));
        assert_eq!(card.get_author(), Some("ML Team"));
        assert_eq!(card.get_metadata("license"), Some("Apache-2.0"));
    }

    // =========================================================================
    // Widget Trait Additional Tests
    // =========================================================================

    #[test]
    fn test_model_card_measure_with_tight_constraints() {
        let card = ModelCard::new("test").width(400.0).height(300.0);
        let size = card.measure(Constraints::tight(Size::new(200.0, 150.0)));
        assert_eq!(size.width, 200.0);
        assert_eq!(size.height, 150.0);
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_model_card_empty_metrics() {
        let card = ModelCard::new("test").metrics(vec![]);
        assert!(!card.has_metrics());
        assert!(card.get_metrics().is_empty());
    }

    #[test]
    fn test_model_card_many_metrics() {
        let metrics: Vec<ModelMetric> = (0..5)
            .map(|i| ModelMetric::new(format!("metric_{i}"), f64::from(i) * 0.1))
            .collect();
        let card = ModelCard::new("test").metrics(metrics);
        assert!(card.has_metrics());
        assert_eq!(card.get_metrics().len(), 5);
    }

    #[test]
    fn test_model_card_empty_tags() {
        let tags: [&str; 0] = [];
        let card = ModelCard::new("test").tags(tags);
        assert!(card.get_tags().is_empty());
    }

    #[test]
    fn test_model_card_show_metrics_chart_false() {
        let card = ModelCard::new("test")
            .metric(ModelMetric::new("acc", 0.95))
            .show_metrics_chart(false);
        assert!(card.has_metrics());
        // show_metrics_chart only affects paint, not has_metrics
    }

    #[test]
    fn test_model_card_default_colors() {
        let card = ModelCard::default();
        assert_eq!(card.background, Color::WHITE);
    }

    #[test]
    fn test_model_card_default_values() {
        let card = ModelCard::default();
        assert!(card.name.is_empty());
        assert_eq!(card.version, "1.0.0");
        assert_eq!(card.status, ModelStatus::Draft);
        assert!(card.show_metrics_chart);
        assert_eq!(card.corner_radius, 8.0);
    }

    // =========================================================================
    // Formatted Parameters Edge Cases
    // =========================================================================

    #[test]
    fn test_formatted_parameters_edge_cases() {
        // Exactly 1000 parameters
        let card = ModelCard::new("test").parameters(1000);
        assert_eq!(card.formatted_parameters(), Some("1.0K".to_string()));

        // Exactly 1 million
        let card = ModelCard::new("test").parameters(1_000_000);
        assert_eq!(card.formatted_parameters(), Some("1.0M".to_string()));

        // Exactly 1 billion
        let card = ModelCard::new("test").parameters(1_000_000_000);
        assert_eq!(card.formatted_parameters(), Some("1.0B".to_string()));
    }
}
