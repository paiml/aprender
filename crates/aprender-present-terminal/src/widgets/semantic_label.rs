//! `SemanticLabel` atomic widget.
//!
//! Text label with automatic color binding based on `HealthStatus`.
//! Reference: SPEC-024 Appendix I (Atomic Widget Mandate).
//!
//! # Falsification Criteria
//! - F-ATOM-SEM-001: Critical status MUST render Red (or theme equivalent).
//! - F-ATOM-SEM-002: Warning status MUST render Yellow/Orange.
//! - F-ATOM-SEM-003: Healthy status MUST render Green.

use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

use super::ux::HealthStatus;

/// Semantic status levels for automatic color binding.
/// More granular than `HealthStatus` for fine-grained visualization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SemanticStatus {
    /// Normal/healthy - green
    #[default]
    Normal,
    /// Good - cyan/blue
    Good,
    /// Warning - yellow
    Warning,
    /// High/elevated - orange
    High,
    /// Critical - red
    Critical,
    /// Unknown/disabled - gray
    Unknown,
    /// Custom color (bypass semantic)
    Custom(u8, u8, u8),
}

impl SemanticStatus {
    /// Get the color for this status level.
    #[must_use]
    pub fn color(&self) -> Color {
        match self {
            Self::Normal => Color::new(0.3, 0.9, 0.4, 1.0), // Green
            Self::Good => Color::new(0.3, 0.8, 1.0, 1.0),   // Cyan
            Self::Warning => Color::new(1.0, 0.85, 0.2, 1.0), // Yellow
            Self::High => Color::new(1.0, 0.6, 0.2, 1.0),   // Orange
            Self::Critical => Color::new(1.0, 0.3, 0.3, 1.0), // Red
            Self::Unknown => Color::new(0.5, 0.5, 0.5, 1.0), // Gray
            Self::Custom(r, g, b) => {
                Color::new(*r as f32 / 255.0, *g as f32 / 255.0, *b as f32 / 255.0, 1.0)
            }
        }
    }

    /// Create from a percentage (0-100).
    /// - 80-100: Normal (green)
    /// - 60-80: Good (cyan)
    /// - 40-60: Warning (yellow)
    /// - 20-40: High (orange)
    /// - 0-20: Critical (red)
    #[must_use]
    pub fn from_percentage(pct: f64) -> Self {
        if pct.is_nan() {
            Self::Unknown
        } else if pct >= 80.0 {
            Self::Normal
        } else if pct >= 60.0 {
            Self::Good
        } else if pct >= 40.0 {
            Self::Warning
        } else if pct >= 20.0 {
            Self::High
        } else {
            Self::Critical
        }
    }

    /// Create from inverted percentage (high = bad, like CPU usage).
    /// - 0-20: Normal (green)
    /// - 20-40: Good (cyan)
    /// - 40-60: Warning (yellow)
    /// - 60-80: High (orange)
    /// - 80-100: Critical (red)
    #[must_use]
    pub fn from_usage(pct: f64) -> Self {
        if pct.is_nan() {
            Self::Unknown
        } else if pct <= 20.0 {
            Self::Normal
        } else if pct <= 40.0 {
            Self::Good
        } else if pct <= 60.0 {
            Self::Warning
        } else if pct <= 80.0 {
            Self::High
        } else {
            Self::Critical
        }
    }

    /// Create from temperature in Celsius.
    #[must_use]
    pub fn from_temperature(temp_c: f64) -> Self {
        if temp_c.is_nan() {
            Self::Unknown
        } else if temp_c <= 50.0 {
            Self::Normal
        } else if temp_c <= 65.0 {
            Self::Good
        } else if temp_c <= 80.0 {
            Self::Warning
        } else if temp_c <= 90.0 {
            Self::High
        } else {
            Self::Critical
        }
    }

    /// Convert from `HealthStatus`.
    #[must_use]
    pub fn from_health_status(status: HealthStatus) -> Self {
        match status {
            HealthStatus::Healthy => Self::Normal,
            HealthStatus::Warning => Self::Warning,
            HealthStatus::Critical => Self::Critical,
            HealthStatus::Unknown => Self::Unknown,
        }
    }
}

/// `SemanticLabel` - text with automatic semantic coloring.
///
/// Automatically colors text based on status level, ensuring consistent
/// visual language across all panels.
#[derive(Debug, Clone)]
pub struct SemanticLabel {
    /// Label text.
    text: String,
    /// Semantic status (determines color).
    status: SemanticStatus,
    /// Optional prefix (e.g., "CPU: ").
    prefix: Option<String>,
    /// Optional suffix (e.g., "%").
    suffix: Option<String>,
    /// Whether to show status symbol (✓ ⚠ ✗).
    show_symbol: bool,
    /// Max width for truncation.
    max_width: Option<usize>,
    /// Cached bounds.
    bounds: Rect,
}

impl Default for SemanticLabel {
    fn default() -> Self {
        Self::new("")
    }
}

impl SemanticLabel {
    /// Create a new semantic label.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            status: SemanticStatus::Normal,
            prefix: None,
            suffix: None,
            show_symbol: false,
            max_width: None,
            bounds: Rect::default(),
        }
    }

    /// Create a label with percentage-based coloring.
    #[must_use]
    pub fn percentage(value: f64) -> Self {
        Self::new(format!("{value:.1}%")).with_status(SemanticStatus::from_usage(value))
    }

    /// Create a label with temperature-based coloring.
    #[must_use]
    pub fn temperature(temp_c: f64) -> Self {
        Self::new(format!("{temp_c:.0}°C")).with_status(SemanticStatus::from_temperature(temp_c))
    }

    /// Set semantic status.
    #[must_use]
    pub fn with_status(mut self, status: SemanticStatus) -> Self {
        self.status = status;
        self
    }

    /// Set prefix text.
    #[must_use]
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = Some(prefix.into());
        self
    }

    /// Set suffix text.
    #[must_use]
    pub fn with_suffix(mut self, suffix: impl Into<String>) -> Self {
        self.suffix = Some(suffix.into());
        self
    }

    /// Enable status symbol display.
    #[must_use]
    pub fn with_symbol(mut self) -> Self {
        self.show_symbol = true;
        self
    }

    /// Set max width for truncation.
    #[must_use]
    pub fn with_max_width(mut self, width: usize) -> Self {
        self.max_width = Some(width);
        self
    }

    /// Get the full display text.
    fn display_text(&self) -> String {
        let mut result = String::new();

        if self.show_symbol {
            let symbol = match self.status {
                SemanticStatus::Normal | SemanticStatus::Good => "✓",
                SemanticStatus::Warning => "⚠",
                SemanticStatus::High | SemanticStatus::Critical => "✗",
                SemanticStatus::Unknown | SemanticStatus::Custom(_, _, _) => "?",
            };
            result.push_str(symbol);
            result.push(' ');
        }

        if let Some(ref prefix) = self.prefix {
            result.push_str(prefix);
        }

        result.push_str(&self.text);

        if let Some(ref suffix) = self.suffix {
            result.push_str(suffix);
        }

        result
    }

    /// Truncate display text if `max_width` is set.
    fn truncated_text(&self) -> String {
        let full = self.display_text();
        if let Some(max) = self.max_width {
            let char_count = full.chars().count();
            if char_count > max {
                if max <= 1 {
                    return "…".to_string();
                }
                let truncated: String = full.chars().take(max - 1).collect();
                return format!("{truncated}…");
            }
        }
        full
    }
}

impl Widget for SemanticLabel {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let text = self.truncated_text();
        let width = text.chars().count() as f32;
        constraints.constrain(Size::new(width.min(constraints.max_width), 1.0))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        if self.bounds.width < 1.0 || self.bounds.height < 1.0 {
            return;
        }

        let text = self.truncated_text();
        let style = TextStyle {
            color: self.status.color(),
            ..Default::default()
        };

        canvas.draw_text(&text, Point::new(self.bounds.x, self.bounds.y), &style);
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

impl Brick for SemanticLabel {
    fn brick_name(&self) -> &'static str {
        "semantic_label"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        static ASSERTIONS: &[BrickAssertion] = &[BrickAssertion::max_latency_ms(1)];
        ASSERTIONS
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(1)
    }

    fn verify(&self) -> BrickVerification {
        // Verify color matches status
        let color = self.status.color();
        let color_valid = match self.status {
            // F-ATOM-SEM-001: Critical = Red
            SemanticStatus::Critical => color.r > 0.8 && color.g < 0.5,
            // F-ATOM-SEM-002: Warning = Yellow
            SemanticStatus::Warning => color.r > 0.8 && color.g > 0.7,
            // F-ATOM-SEM-003: Normal = Green
            SemanticStatus::Normal => color.g > 0.7 && color.r < 0.5,
            _ => true, // Other statuses don't have strict color requirements
        };

        if color_valid {
            BrickVerification {
                passed: self.assertions().to_vec(),
                failed: vec![],
                verification_time: Duration::from_micros(1),
            }
        } else {
            BrickVerification {
                passed: vec![],
                failed: self
                    .assertions()
                    .iter()
                    .map(|a| (a.clone(), "Color does not match status".to_string()))
                    .collect(),
                verification_time: Duration::from_micros(1),
            }
        }
    }

    fn to_html(&self) -> String {
        let class = match self.status {
            SemanticStatus::Normal => "status-normal",
            SemanticStatus::Good => "status-good",
            SemanticStatus::Warning => "status-warning",
            SemanticStatus::High => "status-high",
            SemanticStatus::Critical => "status-critical",
            SemanticStatus::Unknown => "status-unknown",
            SemanticStatus::Custom(_, _, _) => "status-custom",
        };
        format!(
            "<span class=\"semantic-label {}\">{}</span>",
            class,
            self.display_text()
        )
    }

    fn to_css(&self) -> String {
        String::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;
    use crate::direct::{CellBuffer, DirectTerminalCanvas};

    // F-ATOM-SEM-001: Critical renders red
    #[test]
    fn test_critical_is_red() {
        let label = SemanticLabel::new("Error").with_status(SemanticStatus::Critical);
        let color = label.status.color();
        assert!(color.r > 0.8, "Critical should be red");
        assert!(color.g < 0.5, "Critical should not be green");
    }

    // F-ATOM-SEM-002: Warning renders yellow
    #[test]
    fn test_warning_is_yellow() {
        let label = SemanticLabel::new("Warn").with_status(SemanticStatus::Warning);
        let color = label.status.color();
        assert!(color.r > 0.8, "Warning should have high red");
        assert!(color.g > 0.7, "Warning should have high green (yellow)");
    }

    // F-ATOM-SEM-003: Normal renders green
    #[test]
    fn test_normal_is_green() {
        let label = SemanticLabel::new("OK").with_status(SemanticStatus::Normal);
        let color = label.status.color();
        assert!(color.g > 0.7, "Normal should be green");
        assert!(color.r < 0.5, "Normal should not be red");
    }

    #[test]
    fn test_good_is_cyan() {
        let color = SemanticStatus::Good.color();
        assert!(color.b > 0.8, "Good should have high blue");
        assert!(color.g > 0.7, "Good should have high green");
    }

    #[test]
    fn test_high_is_orange() {
        let color = SemanticStatus::High.color();
        assert!(color.r > 0.8, "High should have high red");
        assert!(color.g > 0.5, "High should have medium green (orange)");
    }

    #[test]
    fn test_unknown_is_gray() {
        let color = SemanticStatus::Unknown.color();
        assert!((color.r - 0.5).abs() < 0.1, "Unknown should be gray");
        assert!((color.g - 0.5).abs() < 0.1, "Unknown should be gray");
    }

    #[test]
    fn test_custom_color() {
        let color = SemanticStatus::Custom(255, 128, 64).color();
        assert!((color.r - 1.0).abs() < 0.01);
        assert!((color.g - 0.502).abs() < 0.01);
        assert!((color.b - 0.251).abs() < 0.01);
    }

    // Percentage-based coloring
    #[test]
    fn test_percentage_coloring() {
        let low = SemanticLabel::percentage(10.0);
        assert_eq!(low.status, SemanticStatus::Normal);

        let high = SemanticLabel::percentage(95.0);
        assert_eq!(high.status, SemanticStatus::Critical);
    }

    #[test]
    fn test_from_percentage_all_ranges() {
        assert_eq!(
            SemanticStatus::from_percentage(90.0),
            SemanticStatus::Normal
        );
        assert_eq!(SemanticStatus::from_percentage(70.0), SemanticStatus::Good);
        assert_eq!(
            SemanticStatus::from_percentage(50.0),
            SemanticStatus::Warning
        );
        assert_eq!(SemanticStatus::from_percentage(30.0), SemanticStatus::High);
        assert_eq!(
            SemanticStatus::from_percentage(10.0),
            SemanticStatus::Critical
        );
        assert_eq!(
            SemanticStatus::from_percentage(f64::NAN),
            SemanticStatus::Unknown
        );
    }

    #[test]
    fn test_from_usage_all_ranges() {
        assert_eq!(SemanticStatus::from_usage(10.0), SemanticStatus::Normal);
        assert_eq!(SemanticStatus::from_usage(30.0), SemanticStatus::Good);
        assert_eq!(SemanticStatus::from_usage(50.0), SemanticStatus::Warning);
        assert_eq!(SemanticStatus::from_usage(70.0), SemanticStatus::High);
        assert_eq!(SemanticStatus::from_usage(90.0), SemanticStatus::Critical);
        assert_eq!(
            SemanticStatus::from_usage(f64::NAN),
            SemanticStatus::Unknown
        );
    }

    #[test]
    fn test_from_temperature_all_ranges() {
        assert_eq!(
            SemanticStatus::from_temperature(40.0),
            SemanticStatus::Normal
        );
        assert_eq!(SemanticStatus::from_temperature(60.0), SemanticStatus::Good);
        assert_eq!(
            SemanticStatus::from_temperature(75.0),
            SemanticStatus::Warning
        );
        assert_eq!(SemanticStatus::from_temperature(85.0), SemanticStatus::High);
        assert_eq!(
            SemanticStatus::from_temperature(95.0),
            SemanticStatus::Critical
        );
        assert_eq!(
            SemanticStatus::from_temperature(f64::NAN),
            SemanticStatus::Unknown
        );
    }

    // Temperature-based coloring
    #[test]
    fn test_temperature_coloring() {
        let cool = SemanticLabel::temperature(40.0);
        assert_eq!(cool.status, SemanticStatus::Normal);

        let hot = SemanticLabel::temperature(95.0);
        assert_eq!(hot.status, SemanticStatus::Critical);
    }

    // Display text composition
    #[test]
    fn test_display_text() {
        let label = SemanticLabel::new("50")
            .with_prefix("CPU: ")
            .with_suffix("%");
        assert_eq!(label.display_text(), "CPU: 50%");
    }

    // Symbol display
    #[test]
    fn test_symbol_display() {
        let label = SemanticLabel::new("OK")
            .with_status(SemanticStatus::Normal)
            .with_symbol();
        assert!(label.display_text().starts_with("✓"));
    }

    #[test]
    fn test_symbol_warning() {
        let label = SemanticLabel::new("Warn")
            .with_status(SemanticStatus::Warning)
            .with_symbol();
        assert!(label.display_text().starts_with("⚠"));
    }

    #[test]
    fn test_symbol_critical() {
        let label = SemanticLabel::new("Err")
            .with_status(SemanticStatus::Critical)
            .with_symbol();
        assert!(label.display_text().starts_with("✗"));
    }

    #[test]
    fn test_symbol_unknown() {
        let label = SemanticLabel::new("?")
            .with_status(SemanticStatus::Unknown)
            .with_symbol();
        assert!(label.display_text().starts_with('?'));
    }

    // Truncation
    #[test]
    fn test_truncation() {
        let label = SemanticLabel::new("Very long text here").with_max_width(10);
        let text = label.truncated_text();
        assert_eq!(text.chars().count(), 10);
        assert!(text.ends_with('…'));
    }

    #[test]
    fn test_truncation_width_1() {
        let label = SemanticLabel::new("Very long text").with_max_width(1);
        let text = label.truncated_text();
        assert_eq!(text, "…");
    }

    #[test]
    fn test_no_truncation_when_fits() {
        let label = SemanticLabel::new("Short");
        let text = label.truncated_text();
        assert_eq!(text, "Short");
    }

    // Widget trait tests
    #[test]
    fn test_measure() {
        let label = SemanticLabel::new("Hello");
        let constraints = Constraints {
            min_width: 0.0,
            max_width: 100.0,
            min_height: 0.0,
            max_height: 10.0,
        };
        let size = label.measure(constraints);
        assert_eq!(size.width, 5.0);
        assert_eq!(size.height, 1.0);
    }

    #[test]
    fn test_layout() {
        let mut label = SemanticLabel::new("Test");
        let result = label.layout(Rect::new(5.0, 10.0, 20.0, 1.0));
        assert_eq!(result.size.width, 20.0);
        assert_eq!(result.size.height, 1.0);
    }

    #[test]
    fn test_paint() {
        let mut label = SemanticLabel::new("Hello").with_status(SemanticStatus::Normal);
        let mut buffer = CellBuffer::new(20, 1);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        label.layout(Rect::new(0.0, 0.0, 20.0, 1.0));
        label.paint(&mut canvas);
        assert_eq!(buffer.get(0, 0).unwrap().symbol, "H");
    }

    #[test]
    fn test_paint_zero_width() {
        let mut label = SemanticLabel::new("Hello");
        let mut buffer = CellBuffer::new(20, 1);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        label.layout(Rect::new(0.0, 0.0, 0.0, 1.0));
        label.paint(&mut canvas); // Should not panic
    }

    #[test]
    fn test_paint_zero_height() {
        let mut label = SemanticLabel::new("Hello");
        let mut buffer = CellBuffer::new(20, 1);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        label.layout(Rect::new(0.0, 0.0, 20.0, 0.0));
        label.paint(&mut canvas); // Should not panic
    }

    #[test]
    fn test_event_returns_none() {
        let mut label = SemanticLabel::new("Test");
        let result = label.event(&Event::FocusIn);
        assert!(result.is_none());
    }

    #[test]
    fn test_children_empty() {
        let label = SemanticLabel::new("Test");
        assert!(label.children().is_empty());
    }

    #[test]
    fn test_children_mut_empty() {
        let mut label = SemanticLabel::new("Test");
        assert!(label.children_mut().is_empty());
    }

    // Brick verification
    #[test]
    fn test_brick_verification() {
        let label = SemanticLabel::new("Test").with_status(SemanticStatus::Critical);
        let v = label.verify();
        assert!(
            v.failed.is_empty(),
            "Critical label should pass verification"
        );
    }

    #[test]
    fn test_brick_verification_warning() {
        let label = SemanticLabel::new("Test").with_status(SemanticStatus::Warning);
        let v = label.verify();
        assert!(v.failed.is_empty());
    }

    #[test]
    fn test_brick_verification_normal() {
        let label = SemanticLabel::new("Test").with_status(SemanticStatus::Normal);
        let v = label.verify();
        assert!(v.failed.is_empty());
    }

    #[test]
    fn test_brick_name() {
        let label = SemanticLabel::new("Test");
        assert_eq!(label.brick_name(), "semantic_label");
    }

    #[test]
    fn test_brick_assertions() {
        let label = SemanticLabel::new("Test");
        assert!(!label.assertions().is_empty());
    }

    #[test]
    fn test_brick_budget() {
        let label = SemanticLabel::new("Test");
        let budget = label.budget();
        // uniform(1) => phase_ms = 1/3 = 0, but total_ms = 1
        assert!(budget.total_ms > 0);
    }

    #[test]
    fn test_to_html() {
        let label = SemanticLabel::new("Test").with_status(SemanticStatus::Critical);
        let html = label.to_html();
        assert!(html.contains("semantic-label"));
        assert!(html.contains("status-critical"));
        assert!(html.contains("Test"));
    }

    #[test]
    fn test_to_html_all_statuses() {
        assert!(SemanticLabel::new("")
            .with_status(SemanticStatus::Normal)
            .to_html()
            .contains("status-normal"));
        assert!(SemanticLabel::new("")
            .with_status(SemanticStatus::Good)
            .to_html()
            .contains("status-good"));
        assert!(SemanticLabel::new("")
            .with_status(SemanticStatus::Warning)
            .to_html()
            .contains("status-warning"));
        assert!(SemanticLabel::new("")
            .with_status(SemanticStatus::High)
            .to_html()
            .contains("status-high"));
        assert!(SemanticLabel::new("")
            .with_status(SemanticStatus::Unknown)
            .to_html()
            .contains("status-unknown"));
        assert!(SemanticLabel::new("")
            .with_status(SemanticStatus::Custom(0, 0, 0))
            .to_html()
            .contains("status-custom"));
    }

    #[test]
    fn test_to_css() {
        let label = SemanticLabel::new("Test");
        let css = label.to_css();
        assert!(css.is_empty()); // Currently returns empty
    }

    // From HealthStatus conversion
    #[test]
    fn test_from_health_status() {
        assert_eq!(
            SemanticStatus::from_health_status(HealthStatus::Critical),
            SemanticStatus::Critical
        );
        assert_eq!(
            SemanticStatus::from_health_status(HealthStatus::Healthy),
            SemanticStatus::Normal
        );
    }

    #[test]
    fn test_from_health_status_all() {
        assert_eq!(
            SemanticStatus::from_health_status(HealthStatus::Healthy),
            SemanticStatus::Normal
        );
        assert_eq!(
            SemanticStatus::from_health_status(HealthStatus::Warning),
            SemanticStatus::Warning
        );
        assert_eq!(
            SemanticStatus::from_health_status(HealthStatus::Critical),
            SemanticStatus::Critical
        );
        assert_eq!(
            SemanticStatus::from_health_status(HealthStatus::Unknown),
            SemanticStatus::Unknown
        );
    }

    #[test]
    fn test_default() {
        let label = SemanticLabel::default();
        assert_eq!(label.text, "");
        assert_eq!(label.status, SemanticStatus::Normal);
    }

    #[test]
    fn test_semantic_status_default() {
        let status = SemanticStatus::default();
        assert_eq!(status, SemanticStatus::Normal);
    }
}
