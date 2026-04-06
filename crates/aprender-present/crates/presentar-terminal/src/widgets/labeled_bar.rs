//! `LabeledBar` molecule widget.
//!
//! Composition of `FlexCell` (label) + `ProportionalBar` (bar) + `FlexCell` (value).
//! Reference: SPEC-024 Appendix I (Atomic Widget Mandate - Molecules).
//!
//! This is the fundamental building block for memory bars, CPU meters,
//! disk usage, GPU utilization, etc.

use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

use super::flex_cell::{Alignment, FlexCell, Overflow};
use super::proportional_bar::{BarSegment, ProportionalBar};
use super::semantic_label::SemanticStatus;

/// Layout mode for the labeled bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LabeledBarLayout {
    /// Label | Bar | Value (horizontal)
    #[default]
    Horizontal,
    /// Label on top, Bar below
    Stacked,
    /// Bar only with label overlay
    Overlay,
}

/// `LabeledBar` - composition of label + bar + value.
///
/// Example renders:
/// ```text
/// Horizontal: CPU   ████████░░░░░░░░░░  45%
/// Stacked:    Memory
///             ████████████░░░░░░░░  8.2G / 16G
/// Overlay:    ███45%████░░░░░░░░░░
/// ```
#[derive(Debug, Clone)]
pub struct LabeledBar {
    /// Label text (e.g., "CPU", "Memory").
    label: String,
    /// Value text (e.g., "45%", "8.2G / 16G").
    value: String,
    /// Bar segments.
    segments: Vec<BarSegment>,
    /// Label color.
    label_color: Color,
    /// Value color (or semantic).
    value_status: Option<SemanticStatus>,
    /// Value color override (if not using semantic).
    value_color: Color,
    /// Bar background color.
    bar_background: Color,
    /// Layout mode.
    layout_mode: LabeledBarLayout,
    /// Label width (fixed characters).
    label_width: usize,
    /// Value width (fixed characters).
    value_width: usize,
    /// Cached bounds.
    bounds: Rect,
}

impl Default for LabeledBar {
    fn default() -> Self {
        Self::new("Label", 0.0)
    }
}

impl LabeledBar {
    /// Create a new labeled bar with a single value.
    #[must_use]
    pub fn new(label: impl Into<String>, value: f64) -> Self {
        let clamped = value.clamp(0.0, 1.0);
        Self {
            label: label.into(),
            value: format!("{:.0}%", clamped * 100.0),
            segments: vec![BarSegment {
                value: clamped,
                color: Color::new(0.3, 0.8, 1.0, 1.0), // Default cyan
            }],
            label_color: Color::new(0.8, 0.8, 0.8, 1.0),
            value_status: None,
            value_color: Color::new(0.9, 0.9, 0.9, 1.0),
            bar_background: Color::new(0.15, 0.15, 0.15, 1.0),
            layout_mode: LabeledBarLayout::Horizontal,
            label_width: 8,
            value_width: 6,
            bounds: Rect::default(),
        }
    }

    /// Create a memory-style bar (used/total format).
    #[must_use]
    pub fn memory(label: impl Into<String>, used: u64, total: u64) -> Self {
        let ratio = if total > 0 {
            (used as f64 / total as f64).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let used_str = format_bytes(used);
        let total_str = format_bytes(total);

        Self::new(label, ratio)
            .with_value(format!("{used_str} / {total_str}"))
            .with_value_width(14)
            .with_semantic_value(SemanticStatus::from_usage(ratio * 100.0))
    }

    /// Create a percentage bar with semantic coloring.
    #[must_use]
    pub fn percentage(label: impl Into<String>, pct: f64) -> Self {
        let clamped = pct.clamp(0.0, 100.0);
        Self::new(label, clamped / 100.0)
            .with_value(format!("{clamped:.1}%"))
            .with_semantic_value(SemanticStatus::from_usage(clamped))
    }

    /// Create a temperature bar.
    #[must_use]
    pub fn temperature(label: impl Into<String>, temp_c: f64, max_temp: f64) -> Self {
        let ratio = if max_temp > 0.0 {
            (temp_c / max_temp).clamp(0.0, 1.0)
        } else {
            0.0
        };

        Self::new(label, ratio)
            .with_value(format!("{temp_c:.0}°C"))
            .with_semantic_value(SemanticStatus::from_temperature(temp_c))
            .with_value_width(5)
    }

    /// Set label text.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// Set value text.
    #[must_use]
    pub fn with_value(mut self, value: impl Into<String>) -> Self {
        self.value = value.into();
        self
    }

    /// Set semantic coloring for value.
    #[must_use]
    pub fn with_semantic_value(mut self, status: SemanticStatus) -> Self {
        self.value_status = Some(status);
        self
    }

    /// Set label color.
    #[must_use]
    pub fn with_label_color(mut self, color: Color) -> Self {
        self.label_color = color;
        self
    }

    /// Set value color (overrides semantic).
    #[must_use]
    pub fn with_value_color(mut self, color: Color) -> Self {
        self.value_color = color;
        self.value_status = None;
        self
    }

    /// Set bar color.
    #[must_use]
    pub fn with_bar_color(mut self, color: Color) -> Self {
        if let Some(seg) = self.segments.first_mut() {
            seg.color = color;
        }
        self
    }

    /// Set bar background color.
    #[must_use]
    pub fn with_bar_background(mut self, color: Color) -> Self {
        self.bar_background = color;
        self
    }

    /// Add a segment to the bar.
    #[must_use]
    pub fn with_segment(mut self, value: f64, color: Color) -> Self {
        self.segments.push(BarSegment {
            value: value.clamp(0.0, 1.0),
            color,
        });
        self
    }

    /// Set multiple segments (replaces existing).
    #[must_use]
    pub fn with_segments(mut self, segments: Vec<BarSegment>) -> Self {
        self.segments = segments;
        self
    }

    /// Set layout mode.
    #[must_use]
    pub fn with_layout(mut self, mode: LabeledBarLayout) -> Self {
        self.layout_mode = mode;
        self
    }

    /// Set label width.
    #[must_use]
    pub fn with_label_width(mut self, width: usize) -> Self {
        self.label_width = width;
        self
    }

    /// Set value width.
    #[must_use]
    pub fn with_value_width(mut self, width: usize) -> Self {
        self.value_width = width;
        self
    }

    /// Get effective value color.
    fn effective_value_color(&self) -> Color {
        self.value_status.map_or(self.value_color, |s| s.color())
    }
}

impl Widget for LabeledBar {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let height = match self.layout_mode {
            LabeledBarLayout::Horizontal | LabeledBarLayout::Overlay => 1.0,
            LabeledBarLayout::Stacked => 2.0,
        };
        constraints.constrain(Size::new(constraints.max_width, height))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        if self.bounds.width < 3.0 || self.bounds.height < 1.0 {
            return;
        }

        match self.layout_mode {
            LabeledBarLayout::Horizontal => self.paint_horizontal(canvas),
            LabeledBarLayout::Stacked => self.paint_stacked(canvas),
            LabeledBarLayout::Overlay => self.paint_overlay(canvas),
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

impl LabeledBar {
    fn paint_horizontal(&self, canvas: &mut dyn Canvas) {
        let total_width = self.bounds.width as usize;
        let bar_width = total_width.saturating_sub(self.label_width + self.value_width + 2);

        if bar_width < 1 {
            return;
        }

        let y = self.bounds.y;
        let mut x = self.bounds.x;

        // Label (left-aligned, fixed width)
        let mut label_cell = FlexCell::new(&self.label)
            .with_color(self.label_color)
            .with_overflow(Overflow::Ellipsis)
            .with_alignment(Alignment::Left);
        label_cell.layout(Rect::new(x, y, self.label_width as f32, 1.0));
        label_cell.paint(canvas);
        x += self.label_width as f32 + 1.0;

        // Bar
        let mut bar = ProportionalBar::new().with_background(self.bar_background);
        for seg in &self.segments {
            bar = bar.with_segment(seg.value, seg.color);
        }
        bar.layout(Rect::new(x, y, bar_width as f32, 1.0));
        bar.paint(canvas);
        x += bar_width as f32 + 1.0;

        // Value (right-aligned, fixed width)
        let mut value_cell = FlexCell::new(&self.value)
            .with_color(self.effective_value_color())
            .with_overflow(Overflow::Ellipsis)
            .with_alignment(Alignment::Right);
        value_cell.layout(Rect::new(x, y, self.value_width as f32, 1.0));
        value_cell.paint(canvas);
    }

    fn paint_stacked(&self, canvas: &mut dyn Canvas) {
        if self.bounds.height < 2.0 {
            return;
        }

        let x = self.bounds.x;
        let y = self.bounds.y;
        let width = self.bounds.width;

        // Label on first line
        let mut label_cell = FlexCell::new(&self.label)
            .with_color(self.label_color)
            .with_overflow(Overflow::Ellipsis);
        label_cell.layout(Rect::new(x, y, width, 1.0));
        label_cell.paint(canvas);

        // Bar on second line (with value at end)
        let bar_width = (width as usize).saturating_sub(self.value_width + 1);
        if bar_width < 1 {
            return;
        }

        let mut bar = ProportionalBar::new().with_background(self.bar_background);
        for seg in &self.segments {
            bar = bar.with_segment(seg.value, seg.color);
        }
        bar.layout(Rect::new(x, y + 1.0, bar_width as f32, 1.0));
        bar.paint(canvas);

        // Value at end of bar line
        let mut value_cell = FlexCell::new(&self.value)
            .with_color(self.effective_value_color())
            .with_overflow(Overflow::Ellipsis)
            .with_alignment(Alignment::Right);
        value_cell.layout(Rect::new(
            x + bar_width as f32 + 1.0,
            y + 1.0,
            self.value_width as f32,
            1.0,
        ));
        value_cell.paint(canvas);
    }

    fn paint_overlay(&self, canvas: &mut dyn Canvas) {
        let x = self.bounds.x;
        let y = self.bounds.y;
        let width = self.bounds.width;

        // Draw bar first
        let mut bar = ProportionalBar::new().with_background(self.bar_background);
        for seg in &self.segments {
            bar = bar.with_segment(seg.value, seg.color);
        }
        bar.layout(Rect::new(x, y, width, 1.0));
        bar.paint(canvas);

        // Overlay value text in center
        let text = format!("{} {}", self.label, self.value);
        let text_len = text.chars().count();
        let text_x = x + ((width as usize).saturating_sub(text_len) / 2) as f32;

        canvas.draw_text(
            &text,
            Point::new(text_x, y),
            &TextStyle {
                color: Color::new(1.0, 1.0, 1.0, 1.0), // White overlay
                ..Default::default()
            },
        );
    }
}

impl Brick for LabeledBar {
    fn brick_name(&self) -> &'static str {
        "labeled_bar"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        static ASSERTIONS: &[BrickAssertion] = &[
            BrickAssertion::max_latency_ms(2), // Composition of 3 widgets
        ];
        ASSERTIONS
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(2)
    }

    fn verify(&self) -> BrickVerification {
        // Verify segments don't exceed 1.0 total
        let total: f64 = self.segments.iter().map(|s| s.value).sum();
        let valid = total <= 1.0 + f64::EPSILON && !total.is_nan();

        if valid {
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
                    .map(|a| (a.clone(), "Segment total exceeds 1.0".to_string()))
                    .collect(),
                verification_time: Duration::from_micros(1),
            }
        }
    }

    fn to_html(&self) -> String {
        format!(
            r#"<div class="labeled-bar">
                <span class="label">{}</span>
                <div class="bar"></div>
                <span class="value">{}</span>
            </div>"#,
            self.label, self.value
        )
    }

    fn to_css(&self) -> String {
        String::new()
    }
}

/// Format bytes to human-readable string.
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    const TB: u64 = GB * 1024;

    if bytes >= TB {
        format!("{:.1}T", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.1}G", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1}M", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1}K", bytes as f64 / KB as f64)
    } else {
        format!("{bytes}B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::direct::{CellBuffer, DirectTerminalCanvas};

    // =========================================================================
    // CREATION TESTS
    // =========================================================================

    #[test]
    fn test_basic_creation() {
        let bar = LabeledBar::new("CPU", 0.5);
        assert_eq!(bar.label, "CPU");
        assert_eq!(bar.value, "50%");
    }

    #[test]
    fn test_default() {
        let bar = LabeledBar::default();
        assert_eq!(bar.label, "Label");
        assert_eq!(bar.value, "0%");
    }

    #[test]
    fn test_layout_mode_default() {
        assert_eq!(LabeledBarLayout::default(), LabeledBarLayout::Horizontal);
    }

    #[test]
    fn test_percentage_bar() {
        let bar = LabeledBar::percentage("Memory", 75.0);
        assert_eq!(bar.value, "75.0%");
        assert_eq!(bar.value_status, Some(SemanticStatus::High));
    }

    #[test]
    fn test_memory_bar() {
        let bar = LabeledBar::memory("RAM", 8 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024);
        assert!(bar.value.contains("8.0G"));
        assert!(bar.value.contains("16.0G"));
    }

    #[test]
    fn test_memory_bar_zero_total() {
        let bar = LabeledBar::memory("RAM", 100, 0);
        assert!(bar.segments[0].value >= 0.0);
    }

    #[test]
    fn test_temperature_bar() {
        let bar = LabeledBar::temperature("Core 0", 75.0, 100.0);
        assert_eq!(bar.value, "75°C");
        assert_eq!(bar.value_status, Some(SemanticStatus::Warning));
    }

    #[test]
    fn test_temperature_bar_zero_max() {
        let bar = LabeledBar::temperature("Core 0", 50.0, 0.0);
        assert_eq!(bar.segments[0].value, 0.0);
    }

    // =========================================================================
    // BUILDER TESTS
    // =========================================================================

    #[test]
    fn test_with_label() {
        let bar = LabeledBar::new("Old", 0.5).with_label("New");
        assert_eq!(bar.label, "New");
    }

    #[test]
    fn test_with_value() {
        let bar = LabeledBar::new("CPU", 0.5).with_value("Custom Value");
        assert_eq!(bar.value, "Custom Value");
    }

    #[test]
    fn test_with_semantic_value() {
        let bar = LabeledBar::new("CPU", 0.5).with_semantic_value(SemanticStatus::Warning);
        assert_eq!(bar.value_status, Some(SemanticStatus::Warning));
    }

    #[test]
    fn test_with_label_color() {
        let color = Color::new(1.0, 0.0, 0.0, 1.0);
        let bar = LabeledBar::new("CPU", 0.5).with_label_color(color);
        assert_eq!(bar.label_color, color);
    }

    #[test]
    fn test_with_value_color() {
        let color = Color::new(0.0, 1.0, 0.0, 1.0);
        let bar = LabeledBar::new("CPU", 0.5).with_value_color(color);
        assert_eq!(bar.value_color, color);
        assert!(bar.value_status.is_none()); // Clears semantic
    }

    #[test]
    fn test_with_bar_color() {
        let color = Color::new(0.0, 0.0, 1.0, 1.0);
        let bar = LabeledBar::new("CPU", 0.5).with_bar_color(color);
        assert_eq!(bar.segments[0].color, color);
    }

    #[test]
    fn test_with_bar_background() {
        let color = Color::new(0.1, 0.1, 0.1, 1.0);
        let bar = LabeledBar::new("CPU", 0.5).with_bar_background(color);
        assert_eq!(bar.bar_background, color);
    }

    #[test]
    fn test_with_segment() {
        let bar = LabeledBar::new("Disk", 0.3).with_segment(0.2, Color::GREEN);
        assert_eq!(bar.segments.len(), 2);
    }

    #[test]
    fn test_with_segments() {
        let segments = vec![
            BarSegment {
                value: 0.3,
                color: Color::BLUE,
            },
            BarSegment {
                value: 0.2,
                color: Color::GREEN,
            },
        ];
        let bar = LabeledBar::new("Disk", 0.0).with_segments(segments);
        assert_eq!(bar.segments.len(), 2);
    }

    #[test]
    fn test_with_layout() {
        let bar = LabeledBar::new("CPU", 0.5).with_layout(LabeledBarLayout::Stacked);
        assert_eq!(bar.layout_mode, LabeledBarLayout::Stacked);
    }

    #[test]
    fn test_with_label_width() {
        let bar = LabeledBar::new("CPU", 0.5).with_label_width(15);
        assert_eq!(bar.label_width, 15);
    }

    #[test]
    fn test_with_value_width() {
        let bar = LabeledBar::new("CPU", 0.5).with_value_width(10);
        assert_eq!(bar.value_width, 10);
    }

    // =========================================================================
    // WIDGET TESTS
    // =========================================================================

    #[test]
    fn test_type_id() {
        let bar = LabeledBar::new("CPU", 0.5);
        let id = Widget::type_id(&bar);
        assert_eq!(id, TypeId::of::<LabeledBar>());
    }

    #[test]
    fn test_measure_horizontal() {
        let bar = LabeledBar::new("CPU", 0.5);
        let constraints = Constraints::loose(Size::new(100.0, 50.0));
        let size = bar.measure(constraints);
        assert_eq!(size.height, 1.0);
    }

    #[test]
    fn test_measure_stacked() {
        let bar = LabeledBar::new("CPU", 0.5).with_layout(LabeledBarLayout::Stacked);
        let constraints = Constraints::loose(Size::new(100.0, 50.0));
        let size = bar.measure(constraints);
        assert_eq!(size.height, 2.0);
    }

    #[test]
    fn test_measure_overlay() {
        let bar = LabeledBar::new("CPU", 0.5).with_layout(LabeledBarLayout::Overlay);
        let constraints = Constraints::loose(Size::new(100.0, 50.0));
        let size = bar.measure(constraints);
        assert_eq!(size.height, 1.0);
    }

    #[test]
    fn test_layout() {
        let mut bar = LabeledBar::new("CPU", 0.5);
        let bounds = Rect::new(0.0, 0.0, 50.0, 1.0);
        let result = bar.layout(bounds);
        assert_eq!(result.size.width, 50.0);
        assert_eq!(bar.bounds, bounds);
    }

    #[test]
    fn test_horizontal_paint() {
        let mut bar = LabeledBar::new("Test", 0.5);
        bar.layout(Rect::new(0.0, 0.0, 40.0, 1.0));

        let mut buffer = CellBuffer::new(40, 1);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        bar.paint(&mut canvas);
    }

    #[test]
    fn test_horizontal_paint_narrow() {
        let mut bar = LabeledBar::new("Test", 0.5);
        bar.layout(Rect::new(0.0, 0.0, 10.0, 1.0));

        let mut buffer = CellBuffer::new(10, 1);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        bar.paint(&mut canvas);
    }

    #[test]
    fn test_horizontal_paint_too_narrow() {
        let mut bar = LabeledBar::new("Test", 0.5);
        bar.layout(Rect::new(0.0, 0.0, 2.0, 1.0));

        let mut buffer = CellBuffer::new(2, 1);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        bar.paint(&mut canvas); // Should return early
    }

    #[test]
    fn test_stacked_paint() {
        let mut bar = LabeledBar::new("Test", 0.5).with_layout(LabeledBarLayout::Stacked);
        bar.layout(Rect::new(0.0, 0.0, 40.0, 2.0));

        let mut buffer = CellBuffer::new(40, 2);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        bar.paint(&mut canvas);
    }

    #[test]
    fn test_stacked_paint_too_short() {
        let mut bar = LabeledBar::new("Test", 0.5).with_layout(LabeledBarLayout::Stacked);
        bar.layout(Rect::new(0.0, 0.0, 40.0, 1.0));

        let mut buffer = CellBuffer::new(40, 1);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        bar.paint(&mut canvas); // Should return early
    }

    #[test]
    fn test_overlay_paint() {
        let mut bar = LabeledBar::new("Test", 0.5).with_layout(LabeledBarLayout::Overlay);
        bar.layout(Rect::new(0.0, 0.0, 40.0, 1.0));

        let mut buffer = CellBuffer::new(40, 1);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);
        bar.paint(&mut canvas);
    }

    #[test]
    fn test_event() {
        let mut bar = LabeledBar::new("Test", 0.5);
        let event = Event::key_down(presentar_core::Key::Enter);
        let result = bar.event(&event);
        assert!(result.is_none());
    }

    #[test]
    fn test_children() {
        let bar = LabeledBar::new("Test", 0.5);
        assert!(bar.children().is_empty());
    }

    #[test]
    fn test_children_mut() {
        let mut bar = LabeledBar::new("Test", 0.5);
        assert!(bar.children_mut().is_empty());
    }

    // =========================================================================
    // BRICK TESTS
    // =========================================================================

    #[test]
    fn test_brick_name() {
        let bar = LabeledBar::new("Test", 0.5);
        assert_eq!(bar.brick_name(), "labeled_bar");
    }

    #[test]
    fn test_assertions() {
        let bar = LabeledBar::new("Test", 0.5);
        let assertions = bar.assertions();
        assert!(!assertions.is_empty());
    }

    #[test]
    fn test_budget() {
        let bar = LabeledBar::new("Test", 0.5);
        let budget = bar.budget();
        assert!(budget.total_ms > 0);
    }

    #[test]
    fn test_multi_segment() {
        let bar = LabeledBar::new("Disk", 0.0).with_segments(vec![
            BarSegment {
                value: 0.3,
                color: Color::BLUE,
            },
            BarSegment {
                value: 0.2,
                color: Color::GREEN,
            },
        ]);

        let v = bar.verify();
        assert!(v.failed.is_empty());
    }

    #[test]
    fn test_verify_exceeds_one() {
        let bar = LabeledBar::new("Disk", 0.0).with_segments(vec![
            BarSegment {
                value: 0.7,
                color: Color::BLUE,
            },
            BarSegment {
                value: 0.5,
                color: Color::GREEN,
            },
        ]);

        let v = bar.verify();
        assert!(!v.failed.is_empty());
    }

    #[test]
    fn test_to_html() {
        let bar = LabeledBar::new("CPU", 0.5);
        let html = bar.to_html();
        assert!(html.contains("CPU"));
        assert!(html.contains("50%"));
        assert!(html.contains("labeled-bar"));
    }

    #[test]
    fn test_to_css() {
        let bar = LabeledBar::new("CPU", 0.5);
        let css = bar.to_css();
        assert!(css.is_empty());
    }

    // =========================================================================
    // HELPER FUNCTION TESTS
    // =========================================================================

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500B");
        assert_eq!(format_bytes(1024), "1.0K");
        assert_eq!(format_bytes(1024 * 1024), "1.0M");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.0G");
        assert_eq!(format_bytes(1024u64 * 1024 * 1024 * 1024), "1.0T");
    }

    #[test]
    fn test_brick_verification() {
        let bar = LabeledBar::new("Valid", 0.5);
        let v = bar.verify();
        assert!(v.failed.is_empty());
    }

    #[test]
    fn test_semantic_coloring() {
        let critical = LabeledBar::percentage("High", 95.0);
        assert_eq!(critical.value_status, Some(SemanticStatus::Critical));

        let normal = LabeledBar::percentage("Low", 10.0);
        assert_eq!(normal.value_status, Some(SemanticStatus::Normal));
    }

    #[test]
    fn test_effective_value_color_with_semantic() {
        let bar = LabeledBar::new("CPU", 0.5).with_semantic_value(SemanticStatus::Critical);
        let color = bar.effective_value_color();
        // Critical should be red-ish
        assert!(color.r > 0.5);
    }

    #[test]
    fn test_effective_value_color_without_semantic() {
        let bar = LabeledBar::new("CPU", 0.5);
        let color = bar.effective_value_color();
        // Default value color is grayish
        assert!(color.r > 0.8);
    }
}
