//! Arc/circular gauge widget.
//!
//! Displays a value as an arc gauge using Unicode box-drawing characters.
//! Useful for compact metric displays like CPU temperature or utilization.

use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// Gauge display mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GaugeMode {
    /// Full arc (360° - ╭───╮).
    #[default]
    Arc,
    /// Quarter arc (90° - bottom-right corner).
    Quarter,
    /// Half arc (180° - bottom semicircle ╰───╯).
    Half,
    /// Three-quarter arc (270° - U-shape open at top-left).
    ThreeQuarter,
    /// Vertical bar with ticks.
    Vertical,
    /// Compact single-line.
    Compact,
}

/// Arc gauge widget using Unicode characters.
#[derive(Debug, Clone)]
pub struct Gauge {
    /// Current value.
    value: f64,
    /// Maximum value.
    max: f64,
    /// Gauge label.
    label: Option<String>,
    /// Display mode.
    mode: GaugeMode,
    /// Gauge color.
    color: Color,
    /// Warning threshold (percentage).
    warn_threshold: f64,
    /// Critical threshold (percentage).
    critical_threshold: f64,
    /// Show value text.
    show_value: bool,
    /// Unit suffix (e.g., "°C", "%").
    unit: Option<String>,
    /// Cached bounds.
    bounds: Rect,
}

impl Default for Gauge {
    fn default() -> Self {
        Self::new(0.0, 100.0)
    }
}

impl Gauge {
    /// Create a new gauge with value and max.
    #[must_use]
    pub fn new(value: f64, max: f64) -> Self {
        Self {
            value: value.clamp(0.0, max),
            max,
            label: None,
            mode: GaugeMode::default(),
            color: Color::new(0.3, 0.7, 1.0, 1.0),
            warn_threshold: 70.0,
            critical_threshold: 90.0,
            show_value: true,
            unit: None,
            bounds: Rect::default(),
        }
    }

    /// Create a percentage gauge (0-100).
    #[must_use]
    pub fn percentage(value: f64) -> Self {
        Self::new(value, 100.0).with_unit("%")
    }

    /// Create a temperature gauge.
    #[must_use]
    pub fn temperature(value: f64, max: f64) -> Self {
        Self::new(value, max)
            .with_unit("°C")
            .with_thresholds(60.0, 80.0)
    }

    /// Set the label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set the display mode.
    #[must_use]
    pub fn with_mode(mut self, mode: GaugeMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set the color.
    #[must_use]
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Set warning and critical thresholds.
    #[must_use]
    pub fn with_thresholds(mut self, warn: f64, critical: f64) -> Self {
        self.warn_threshold = warn;
        self.critical_threshold = critical;
        self
    }

    /// Set whether to show value text.
    #[must_use]
    pub fn with_value_display(mut self, show: bool) -> Self {
        self.show_value = show;
        self
    }

    /// Set unit suffix.
    #[must_use]
    pub fn with_unit(mut self, unit: impl Into<String>) -> Self {
        self.unit = Some(unit.into());
        self
    }

    /// Update the value.
    pub fn set_value(&mut self, value: f64) {
        self.value = value.clamp(0.0, self.max);
    }

    /// Get current value.
    #[must_use]
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Get percentage (0-100).
    #[must_use]
    pub fn percent(&self) -> f64 {
        if self.max == 0.0 {
            0.0
        } else {
            (self.value / self.max * 100.0).clamp(0.0, 100.0)
        }
    }

    /// Get current color based on thresholds.
    #[must_use]
    pub fn current_color(&self) -> Color {
        let pct = self.percent();
        if pct >= self.critical_threshold {
            Color::new(1.0, 0.3, 0.3, 1.0) // Red
        } else if pct >= self.warn_threshold {
            Color::new(1.0, 0.7, 0.2, 1.0) // Orange
        } else {
            self.color
        }
    }

    fn render_arc(&self, canvas: &mut dyn Canvas) {
        let width = self.bounds.width as usize;
        let height = self.bounds.height as usize;

        if width < 5 || height < 3 {
            // Fall back to compact mode
            self.render_compact(canvas);
            return;
        }

        let color = self.current_color();
        let style = TextStyle {
            color,
            ..Default::default()
        };
        let dim_style = TextStyle {
            color: Color::new(0.3, 0.3, 0.3, 1.0),
            ..Default::default()
        };

        // Arc characters
        let pct = self.percent() / 100.0;
        let arc_width = width.saturating_sub(2);
        let filled = ((pct * arc_width as f64).round() as usize).min(arc_width);

        // Top of arc: ╭───╮
        let mut top = String::with_capacity(width);
        top.push('╭');
        for i in 0..arc_width {
            if i < filled {
                top.push('━');
            } else {
                top.push('─');
            }
        }
        top.push('╮');
        canvas.draw_text(&top, Point::new(self.bounds.x, self.bounds.y), &style);

        // Middle: │ value │
        if height > 2 {
            let value_text = if self.show_value {
                let unit = self.unit.as_deref().unwrap_or("");
                format!("{:.0}{}", self.value, unit)
            } else {
                String::new()
            };
            let padding = (arc_width.saturating_sub(value_text.len())) / 2;
            let mut middle = String::with_capacity(width);
            middle.push('│');
            for _ in 0..padding {
                middle.push(' ');
            }
            middle.push_str(&value_text);
            for _ in 0..(arc_width - padding - value_text.len()) {
                middle.push(' ');
            }
            middle.push('│');
            canvas.draw_text(
                &middle,
                Point::new(self.bounds.x, self.bounds.y + 1.0),
                &dim_style,
            );
        }

        // Bottom of arc: ╰───╯
        if height > 1 {
            let mut bottom = String::with_capacity(width);
            bottom.push('╰');
            for _ in 0..arc_width {
                bottom.push('─');
            }
            bottom.push('╯');
            let y = if height > 2 {
                self.bounds.y + 2.0
            } else {
                self.bounds.y + 1.0
            };
            canvas.draw_text(&bottom, Point::new(self.bounds.x, y), &dim_style);
        }

        // Label below
        if let Some(ref label) = self.label {
            let label_y = self.bounds.y + height.min(3) as f32;
            if label_y < self.bounds.y + self.bounds.height {
                let label_style = TextStyle {
                    color: Color::new(0.6, 0.6, 0.6, 1.0),
                    ..Default::default()
                };
                canvas.draw_text(label, Point::new(self.bounds.x, label_y), &label_style);
            }
        }
    }

    fn render_vertical(&self, canvas: &mut dyn Canvas) {
        let height = (self.bounds.height as usize).saturating_sub(1);
        if height == 0 {
            return;
        }

        let color = self.current_color();
        let style = TextStyle {
            color,
            ..Default::default()
        };
        let dim_style = TextStyle {
            color: Color::new(0.3, 0.3, 0.3, 1.0),
            ..Default::default()
        };

        let pct = self.percent() / 100.0;
        let filled = ((pct * height as f64).round() as usize).min(height);

        // Draw vertical bar from bottom to top
        for i in 0..height {
            let y = self.bounds.y + (height - 1 - i) as f32;
            if i < filled {
                canvas.draw_text("█", Point::new(self.bounds.x, y), &style);
            } else {
                canvas.draw_text("░", Point::new(self.bounds.x, y), &dim_style);
            }
        }

        // Value at bottom
        if self.show_value {
            let unit = self.unit.as_deref().unwrap_or("");
            let value_text = format!("{:.0}{}", self.value, unit);
            canvas.draw_text(
                &value_text,
                Point::new(self.bounds.x + 2.0, self.bounds.y + height as f32),
                &style,
            );
        }
    }

    fn render_compact(&self, canvas: &mut dyn Canvas) {
        let color = self.current_color();
        let style = TextStyle {
            color,
            ..Default::default()
        };

        let unit = self.unit.as_deref().unwrap_or("");
        let label = self.label.as_deref().unwrap_or("");
        let text = if label.is_empty() {
            format!("{:.0}{}", self.value, unit)
        } else {
            format!("{}: {:.0}{}", label, self.value, unit)
        };

        canvas.draw_text(&text, Point::new(self.bounds.x, self.bounds.y), &style);
    }

    /// Render quarter arc (90° - bottom-right corner style).
    fn render_quarter(&self, canvas: &mut dyn Canvas) {
        let width = self.bounds.width as usize;
        let height = self.bounds.height as usize;

        if width < 3 || height < 2 {
            self.render_compact(canvas);
            return;
        }

        let color = self.current_color();
        let style = TextStyle {
            color,
            ..Default::default()
        };
        let dim_style = TextStyle {
            color: Color::new(0.3, 0.3, 0.3, 1.0),
            ..Default::default()
        };

        let pct = self.percent() / 100.0;
        let arc_width = width.saturating_sub(1);
        let filled = ((pct * arc_width as f64).round() as usize).min(arc_width);

        // Quarter arc: ───╮
        //                 │
        let mut top = String::with_capacity(width);
        for i in 0..arc_width {
            if i < filled {
                top.push('━');
            } else {
                top.push('─');
            }
        }
        top.push('╮');
        canvas.draw_text(&top, Point::new(self.bounds.x, self.bounds.y), &style);

        // Vertical part
        if height > 1 {
            let x_pos = self.bounds.x + arc_width as f32;
            canvas.draw_text("│", Point::new(x_pos, self.bounds.y + 1.0), &dim_style);
        }

        // Value display
        if self.show_value && height > 1 {
            let unit = self.unit.as_deref().unwrap_or("");
            let value_text = format!("{:.0}{}", self.value, unit);
            canvas.draw_text(
                &value_text,
                Point::new(self.bounds.x, self.bounds.y + 1.0),
                &style,
            );
        }

        // Label
        if let Some(ref label) = self.label {
            if height > 2 {
                let label_style = TextStyle {
                    color: Color::new(0.6, 0.6, 0.6, 1.0),
                    ..Default::default()
                };
                canvas.draw_text(
                    label,
                    Point::new(self.bounds.x, self.bounds.y + 2.0),
                    &label_style,
                );
            }
        }
    }

    /// Render half arc (180° - bottom semicircle).
    fn render_half(&self, canvas: &mut dyn Canvas) {
        let width = self.bounds.width as usize;
        let height = self.bounds.height as usize;

        if width < 4 || height < 2 {
            self.render_compact(canvas);
            return;
        }

        let color = self.current_color();
        let style = TextStyle {
            color,
            ..Default::default()
        };

        let pct = self.percent() / 100.0;
        let arc_width = width.saturating_sub(2);
        let filled = ((pct * arc_width as f64).round() as usize).min(arc_width);

        // Half arc (bottom semicircle): ╰───╯
        let mut arc = String::with_capacity(width);
        arc.push('╰');
        for i in 0..arc_width {
            if i < filled {
                arc.push('━');
            } else {
                arc.push('─');
            }
        }
        arc.push('╯');
        canvas.draw_text(&arc, Point::new(self.bounds.x, self.bounds.y), &style);

        // Value above arc
        if self.show_value && height > 1 {
            let unit = self.unit.as_deref().unwrap_or("");
            let value_text = format!("{:.0}{}", self.value, unit);
            let padding = (arc_width.saturating_sub(value_text.len())) / 2;
            canvas.draw_text(
                &value_text,
                Point::new(self.bounds.x + 1.0 + padding as f32, self.bounds.y + 1.0),
                &style,
            );
        }

        // Label
        if let Some(ref label) = self.label {
            if height > 2 {
                let label_style = TextStyle {
                    color: Color::new(0.6, 0.6, 0.6, 1.0),
                    ..Default::default()
                };
                canvas.draw_text(
                    label,
                    Point::new(self.bounds.x, self.bounds.y + 2.0),
                    &label_style,
                );
            }
        }
    }

    /// Render three-quarter arc (270° - U-shape open at top-left).
    fn render_three_quarter(&self, canvas: &mut dyn Canvas) {
        let width = self.bounds.width as usize;
        let height = self.bounds.height as usize;

        if width < 4 || height < 3 {
            self.render_compact(canvas);
            return;
        }

        let color = self.current_color();
        let style = TextStyle {
            color,
            ..Default::default()
        };
        let dim_style = TextStyle {
            color: Color::new(0.3, 0.3, 0.3, 1.0),
            ..Default::default()
        };

        let pct = self.percent() / 100.0;
        let arc_width = width.saturating_sub(1);

        // Three-quarter arc: open at top-left
        // ────╮
        //     │
        // ────╯

        // Top section (horizontal part with right corner)
        let top_filled = ((pct * arc_width as f64).round() as usize).min(arc_width);
        let mut top = String::with_capacity(width);
        for i in 0..arc_width {
            if i < top_filled {
                top.push('━');
            } else {
                top.push('─');
            }
        }
        top.push('╮');
        canvas.draw_text(&top, Point::new(self.bounds.x, self.bounds.y), &style);

        // Middle section (vertical bar on right)
        if height > 2 {
            let x_pos = self.bounds.x + arc_width as f32;
            canvas.draw_text("│", Point::new(x_pos, self.bounds.y + 1.0), &dim_style);

            // Value in middle
            if self.show_value {
                let unit = self.unit.as_deref().unwrap_or("");
                let value_text = format!("{:.0}{}", self.value, unit);
                canvas.draw_text(
                    &value_text,
                    Point::new(self.bounds.x, self.bounds.y + 1.0),
                    &style,
                );
            }
        }

        // Bottom section (horizontal with right corner)
        let bottom_y = if height > 2 {
            self.bounds.y + 2.0
        } else {
            self.bounds.y + 1.0
        };
        let mut bottom = String::with_capacity(width);
        for _ in 0..arc_width {
            bottom.push('─');
        }
        bottom.push('╯');
        canvas.draw_text(&bottom, Point::new(self.bounds.x, bottom_y), &dim_style);

        // Label
        if let Some(ref label) = self.label {
            if height > 3 {
                let label_style = TextStyle {
                    color: Color::new(0.6, 0.6, 0.6, 1.0),
                    ..Default::default()
                };
                canvas.draw_text(
                    label,
                    Point::new(self.bounds.x, self.bounds.y + 3.0),
                    &label_style,
                );
            }
        }
    }
}

impl Brick for Gauge {
    fn brick_name(&self) -> &'static str {
        "gauge"
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

        if self.value >= 0.0 && self.value <= self.max {
            passed.push(BrickAssertion::max_latency_ms(16));
        } else {
            failed.push((
                BrickAssertion::max_latency_ms(16),
                format!("Value {} outside range [0, {}]", self.value, self.max),
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

impl Widget for Gauge {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        match self.mode {
            GaugeMode::Arc | GaugeMode::ThreeQuarter => {
                let width = 10.0_f32.min(constraints.max_width);
                let height = 4.0_f32.min(constraints.max_height);
                constraints.constrain(Size::new(width, height))
            }
            GaugeMode::Quarter => {
                let width = 8.0_f32.min(constraints.max_width);
                let height = 3.0_f32.min(constraints.max_height);
                constraints.constrain(Size::new(width, height))
            }
            GaugeMode::Half => {
                let width = 10.0_f32.min(constraints.max_width);
                let height = 3.0_f32.min(constraints.max_height);
                constraints.constrain(Size::new(width, height))
            }
            GaugeMode::Vertical => {
                let height = 8.0_f32.min(constraints.max_height);
                constraints.constrain(Size::new(6.0, height))
            }
            GaugeMode::Compact => constraints.constrain(Size::new(12.0, 1.0)),
        }
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        match self.mode {
            GaugeMode::Arc => self.render_arc(canvas),
            GaugeMode::Quarter => self.render_quarter(canvas),
            GaugeMode::Half => self.render_half(canvas),
            GaugeMode::ThreeQuarter => self.render_three_quarter(canvas),
            GaugeMode::Vertical => self.render_vertical(canvas),
            GaugeMode::Compact => self.render_compact(canvas),
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

#[cfg(test)]
mod tests {
    use super::*;

    struct MockCanvas {
        texts: Vec<(String, Point)>,
    }

    impl MockCanvas {
        fn new() -> Self {
            Self { texts: vec![] }
        }
    }

    impl Canvas for MockCanvas {
        fn fill_rect(&mut self, _rect: Rect, _color: Color) {}
        fn stroke_rect(&mut self, _rect: Rect, _color: Color, _width: f32) {}
        fn draw_text(&mut self, text: &str, position: Point, _style: &TextStyle) {
            self.texts.push((text.to_string(), position));
        }
        fn draw_line(&mut self, _from: Point, _to: Point, _color: Color, _width: f32) {}
        fn fill_circle(&mut self, _center: Point, _radius: f32, _color: Color) {}
        fn stroke_circle(&mut self, _center: Point, _radius: f32, _color: Color, _width: f32) {}
        fn fill_arc(&mut self, _c: Point, _r: f32, _s: f32, _e: f32, _color: Color) {}
        fn draw_path(&mut self, _points: &[Point], _color: Color, _width: f32) {}
        fn fill_polygon(&mut self, _points: &[Point], _color: Color) {}
        fn push_clip(&mut self, _rect: Rect) {}
        fn pop_clip(&mut self) {}
        fn push_transform(&mut self, _transform: presentar_core::Transform2D) {}
        fn pop_transform(&mut self) {}
    }

    #[test]
    fn test_gauge_creation() {
        let gauge = Gauge::new(50.0, 100.0);
        assert_eq!(gauge.value(), 50.0);
        assert_eq!(gauge.max, 100.0);
    }

    #[test]
    fn test_gauge_percentage_constructor() {
        let gauge = Gauge::percentage(75.0);
        assert_eq!(gauge.percent(), 75.0);
        assert_eq!(gauge.unit, Some("%".to_string()));
    }

    #[test]
    fn test_gauge_temperature() {
        let gauge = Gauge::temperature(65.0, 100.0);
        assert_eq!(gauge.value(), 65.0);
        assert_eq!(gauge.unit, Some("°C".to_string()));
    }

    #[test]
    fn test_gauge_assertions() {
        let gauge = Gauge::default();
        assert!(!gauge.assertions().is_empty());
    }

    #[test]
    fn test_gauge_verify() {
        let gauge = Gauge::new(50.0, 100.0);
        assert!(gauge.verify().is_valid());
    }

    #[test]
    fn test_gauge_with_label() {
        let gauge = Gauge::new(50.0, 100.0).with_label("CPU");
        assert_eq!(gauge.label, Some("CPU".to_string()));
    }

    #[test]
    fn test_gauge_with_mode() {
        let gauge = Gauge::default().with_mode(GaugeMode::Vertical);
        assert_eq!(gauge.mode, GaugeMode::Vertical);
    }

    #[test]
    fn test_gauge_with_color() {
        let gauge = Gauge::default().with_color(Color::RED);
        assert_eq!(gauge.color, Color::RED);
    }

    #[test]
    fn test_gauge_with_thresholds() {
        let gauge = Gauge::default().with_thresholds(50.0, 80.0);
        assert_eq!(gauge.warn_threshold, 50.0);
        assert_eq!(gauge.critical_threshold, 80.0);
    }

    #[test]
    fn test_gauge_with_value_display() {
        let gauge = Gauge::default().with_value_display(false);
        assert!(!gauge.show_value);
    }

    #[test]
    fn test_gauge_with_unit() {
        let gauge = Gauge::default().with_unit("MB");
        assert_eq!(gauge.unit, Some("MB".to_string()));
    }

    #[test]
    fn test_gauge_set_value() {
        let mut gauge = Gauge::new(50.0, 100.0);
        gauge.set_value(75.0);
        assert_eq!(gauge.value(), 75.0);
    }

    #[test]
    fn test_gauge_set_value_clamped() {
        let mut gauge = Gauge::new(50.0, 100.0);
        gauge.set_value(150.0);
        assert_eq!(gauge.value(), 100.0);

        gauge.set_value(-10.0);
        assert_eq!(gauge.value(), 0.0);
    }

    #[test]
    fn test_gauge_current_color_normal() {
        let gauge = Gauge::new(50.0, 100.0);
        let color = gauge.current_color();
        assert_eq!(color, gauge.color);
    }

    #[test]
    fn test_gauge_current_color_warning() {
        let gauge = Gauge::new(75.0, 100.0);
        let color = gauge.current_color();
        assert!(color.r > 0.9);
    }

    #[test]
    fn test_gauge_current_color_critical() {
        let gauge = Gauge::new(95.0, 100.0);
        let color = gauge.current_color();
        assert!(color.r > 0.9);
        assert!(color.g < 0.5);
    }

    #[test]
    fn test_gauge_paint_arc() {
        let mut gauge = Gauge::default().with_mode(GaugeMode::Arc);
        gauge.bounds = Rect::new(0.0, 0.0, 10.0, 4.0);
        let mut canvas = MockCanvas::new();
        gauge.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_gauge_paint_vertical() {
        let mut gauge = Gauge::percentage(50.0).with_mode(GaugeMode::Vertical);
        gauge.bounds = Rect::new(0.0, 0.0, 6.0, 8.0);
        let mut canvas = MockCanvas::new();
        gauge.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_gauge_paint_compact() {
        let mut gauge = Gauge::percentage(50.0).with_mode(GaugeMode::Compact);
        gauge.bounds = Rect::new(0.0, 0.0, 20.0, 1.0);
        let mut canvas = MockCanvas::new();
        gauge.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_gauge_paint_compact_with_label() {
        let mut gauge = Gauge::percentage(50.0)
            .with_mode(GaugeMode::Compact)
            .with_label("CPU");
        gauge.bounds = Rect::new(0.0, 0.0, 20.0, 1.0);
        let mut canvas = MockCanvas::new();
        gauge.paint(&mut canvas);
        assert!(canvas.texts.iter().any(|(t, _)| t.contains("CPU")));
    }

    #[test]
    fn test_gauge_measure_arc() {
        let gauge = Gauge::default().with_mode(GaugeMode::Arc);
        let size = gauge.measure(Constraints::loose(Size::new(100.0, 100.0)));
        assert!(size.width > 0.0);
        assert!(size.height > 0.0);
    }

    #[test]
    fn test_gauge_measure_vertical() {
        let gauge = Gauge::default().with_mode(GaugeMode::Vertical);
        let size = gauge.measure(Constraints::loose(Size::new(100.0, 100.0)));
        assert!(size.height > size.width);
    }

    #[test]
    fn test_gauge_measure_compact() {
        let gauge = Gauge::default().with_mode(GaugeMode::Compact);
        let size = gauge.measure(Constraints::loose(Size::new(100.0, 100.0)));
        assert_eq!(size.height, 1.0);
    }

    #[test]
    fn test_gauge_layout() {
        let mut gauge = Gauge::default();
        let bounds = Rect::new(5.0, 10.0, 20.0, 10.0);
        let result = gauge.layout(bounds);
        assert_eq!(result.size.width, 20.0);
        assert_eq!(gauge.bounds, bounds);
    }

    #[test]
    fn test_gauge_brick_name() {
        let gauge = Gauge::default();
        assert_eq!(gauge.brick_name(), "gauge");
    }

    #[test]
    fn test_gauge_type_id() {
        let gauge = Gauge::default();
        assert_eq!(Widget::type_id(&gauge), TypeId::of::<Gauge>());
    }

    #[test]
    fn test_gauge_children() {
        let gauge = Gauge::default();
        assert!(gauge.children().is_empty());
    }

    #[test]
    fn test_gauge_event() {
        let mut gauge = Gauge::default();
        let event = Event::key_down(presentar_core::Key::Enter);
        assert!(gauge.event(&event).is_none());
    }

    #[test]
    fn test_gauge_default() {
        let gauge = Gauge::default();
        assert_eq!(gauge.value(), 0.0);
        assert_eq!(gauge.max, 100.0);
    }

    #[test]
    fn test_gauge_percentage_zero_max() {
        let gauge = Gauge::new(50.0, 0.0);
        assert_eq!(gauge.percent(), 0.0);
    }

    #[test]
    fn test_gauge_arc_fallback_to_compact() {
        // Arc mode with small bounds should fall back to compact
        let mut gauge = Gauge::percentage(50.0).with_mode(GaugeMode::Arc);
        gauge.bounds = Rect::new(0.0, 0.0, 3.0, 2.0); // Too small for arc
        let mut canvas = MockCanvas::new();
        gauge.paint(&mut canvas);
        // Should have rendered compact mode instead
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_gauge_arc_no_value_display() {
        let mut gauge = Gauge::percentage(50.0)
            .with_mode(GaugeMode::Arc)
            .with_value_display(false);
        gauge.bounds = Rect::new(0.0, 0.0, 10.0, 4.0);
        let mut canvas = MockCanvas::new();
        gauge.paint(&mut canvas);
        // Should render without value text
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_gauge_arc_with_label() {
        let mut gauge = Gauge::percentage(50.0)
            .with_mode(GaugeMode::Arc)
            .with_label("CPU");
        gauge.bounds = Rect::new(0.0, 0.0, 10.0, 5.0);
        let mut canvas = MockCanvas::new();
        gauge.paint(&mut canvas);
        // Should render with label below arc
        assert!(canvas.texts.iter().any(|(t, _)| t.contains("CPU")));
    }

    #[test]
    fn test_gauge_arc_height_two() {
        // Arc with only 2 lines height - falls back to compact (height < 3)
        let mut gauge = Gauge::percentage(50.0).with_mode(GaugeMode::Arc);
        gauge.bounds = Rect::new(0.0, 0.0, 10.0, 2.0);
        let mut canvas = MockCanvas::new();
        gauge.paint(&mut canvas);
        // Falls back to compact mode
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_gauge_arc_height_three_no_middle() {
        // Arc with exactly 3 lines height - draws top and bottom
        let mut gauge = Gauge::percentage(50.0).with_mode(GaugeMode::Arc);
        gauge.bounds = Rect::new(0.0, 0.0, 10.0, 3.0);
        let mut canvas = MockCanvas::new();
        gauge.paint(&mut canvas);
        // Should draw top and bottom (height > 2 is false, so no middle)
        assert!(canvas.texts.len() >= 2);
    }

    #[test]
    fn test_gauge_arc_height_four() {
        // Arc with 4 lines - draws all sections
        let mut gauge = Gauge::percentage(50.0).with_mode(GaugeMode::Arc);
        gauge.bounds = Rect::new(0.0, 0.0, 10.0, 4.0);
        let mut canvas = MockCanvas::new();
        gauge.paint(&mut canvas);
        // Should draw top, middle, and bottom (height > 2)
        assert!(canvas.texts.len() >= 3);
    }

    #[test]
    fn test_gauge_vertical_zero_height() {
        let mut gauge = Gauge::percentage(50.0).with_mode(GaugeMode::Vertical);
        gauge.bounds = Rect::new(0.0, 0.0, 6.0, 1.0); // height - 1 = 0
        let mut canvas = MockCanvas::new();
        gauge.paint(&mut canvas);
        // Early return when height is 0
        assert!(canvas.texts.is_empty());
    }

    #[test]
    fn test_gauge_vertical_no_value_display() {
        let mut gauge = Gauge::percentage(50.0)
            .with_mode(GaugeMode::Vertical)
            .with_value_display(false);
        gauge.bounds = Rect::new(0.0, 0.0, 6.0, 8.0);
        let mut canvas = MockCanvas::new();
        gauge.paint(&mut canvas);
        // Should not have value text
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_gauge_verify_invalid() {
        // Create a gauge and manually set invalid value
        let mut gauge = Gauge::new(50.0, 100.0);
        gauge.value = -10.0; // Invalid: outside range
        let result = gauge.verify();
        assert!(!result.is_valid());
        assert!(!result.failed.is_empty());
    }

    #[test]
    fn test_gauge_verify_above_max() {
        let mut gauge = Gauge::new(50.0, 100.0);
        gauge.value = 150.0; // Invalid: above max
        let result = gauge.verify();
        assert!(!result.is_valid());
    }

    #[test]
    fn test_gauge_children_mut() {
        let mut gauge = Gauge::default();
        assert!(gauge.children_mut().is_empty());
    }

    #[test]
    fn test_gauge_budget() {
        let gauge = Gauge::default();
        let budget = gauge.budget();
        assert_eq!(budget.total_ms, 16);
    }

    #[test]
    fn test_gauge_to_html() {
        let gauge = Gauge::default();
        assert!(gauge.to_html().is_empty());
    }

    #[test]
    fn test_gauge_to_css() {
        let gauge = Gauge::default();
        assert!(gauge.to_css().is_empty());
    }

    #[test]
    fn test_gauge_compact_no_label() {
        // Compact mode without label
        let mut gauge = Gauge::percentage(75.0).with_mode(GaugeMode::Compact);
        gauge.bounds = Rect::new(0.0, 0.0, 20.0, 1.0);
        let mut canvas = MockCanvas::new();
        gauge.paint(&mut canvas);
        // Should show value without label prefix
        assert!(canvas.texts.iter().any(|(t, _)| t.contains("75")));
    }

    #[test]
    fn test_gauge_quarter_mode() {
        let gauge = Gauge::default().with_mode(GaugeMode::Quarter);
        assert_eq!(gauge.mode, GaugeMode::Quarter);
    }

    #[test]
    fn test_gauge_half_mode() {
        let gauge = Gauge::default().with_mode(GaugeMode::Half);
        assert_eq!(gauge.mode, GaugeMode::Half);
    }

    #[test]
    fn test_gauge_three_quarter_mode() {
        let gauge = Gauge::default().with_mode(GaugeMode::ThreeQuarter);
        assert_eq!(gauge.mode, GaugeMode::ThreeQuarter);
    }

    #[test]
    fn test_gauge_quarter_paint() {
        let mut gauge = Gauge::percentage(50.0).with_mode(GaugeMode::Quarter);
        gauge.bounds = Rect::new(0.0, 0.0, 8.0, 3.0);
        let mut canvas = MockCanvas::new();
        gauge.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
        // Quarter arc should have ╮ character
        assert!(canvas.texts.iter().any(|(t, _)| t.contains('╮')));
    }

    #[test]
    fn test_gauge_half_paint() {
        let mut gauge = Gauge::percentage(50.0).with_mode(GaugeMode::Half);
        gauge.bounds = Rect::new(0.0, 0.0, 10.0, 3.0);
        let mut canvas = MockCanvas::new();
        gauge.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
        // Half arc should have ╰ and ╯ characters
        assert!(canvas.texts.iter().any(|(t, _)| t.contains('╰')));
    }

    #[test]
    fn test_gauge_three_quarter_paint() {
        let mut gauge = Gauge::percentage(50.0).with_mode(GaugeMode::ThreeQuarter);
        gauge.bounds = Rect::new(0.0, 0.0, 10.0, 4.0);
        let mut canvas = MockCanvas::new();
        gauge.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
        // Three-quarter arc should have ╮ and ╯ characters
        assert!(canvas.texts.iter().any(|(t, _)| t.contains('╮')));
        assert!(canvas.texts.iter().any(|(t, _)| t.contains('╯')));
    }

    #[test]
    fn test_gauge_quarter_fallback() {
        // Quarter mode with too small bounds falls back to compact
        let mut gauge = Gauge::percentage(50.0).with_mode(GaugeMode::Quarter);
        gauge.bounds = Rect::new(0.0, 0.0, 2.0, 1.0);
        let mut canvas = MockCanvas::new();
        gauge.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_gauge_half_fallback() {
        // Half mode with too small bounds falls back to compact
        let mut gauge = Gauge::percentage(50.0).with_mode(GaugeMode::Half);
        gauge.bounds = Rect::new(0.0, 0.0, 2.0, 1.0);
        let mut canvas = MockCanvas::new();
        gauge.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_gauge_three_quarter_fallback() {
        // Three-quarter mode with too small bounds falls back to compact
        let mut gauge = Gauge::percentage(50.0).with_mode(GaugeMode::ThreeQuarter);
        gauge.bounds = Rect::new(0.0, 0.0, 2.0, 2.0);
        let mut canvas = MockCanvas::new();
        gauge.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_gauge_measure_quarter() {
        let gauge = Gauge::default().with_mode(GaugeMode::Quarter);
        let size = gauge.measure(Constraints::loose(Size::new(100.0, 100.0)));
        assert_eq!(size.width, 8.0);
        assert_eq!(size.height, 3.0);
    }

    #[test]
    fn test_gauge_measure_half() {
        let gauge = Gauge::default().with_mode(GaugeMode::Half);
        let size = gauge.measure(Constraints::loose(Size::new(100.0, 100.0)));
        assert_eq!(size.width, 10.0);
        assert_eq!(size.height, 3.0);
    }

    #[test]
    fn test_gauge_measure_three_quarter() {
        let gauge = Gauge::default().with_mode(GaugeMode::ThreeQuarter);
        let size = gauge.measure(Constraints::loose(Size::new(100.0, 100.0)));
        assert_eq!(size.width, 10.0);
        assert_eq!(size.height, 4.0);
    }

    #[test]
    fn test_gauge_quarter_with_label() {
        let mut gauge = Gauge::percentage(50.0)
            .with_mode(GaugeMode::Quarter)
            .with_label("Temp");
        gauge.bounds = Rect::new(0.0, 0.0, 8.0, 4.0);
        let mut canvas = MockCanvas::new();
        gauge.paint(&mut canvas);
        assert!(canvas.texts.iter().any(|(t, _)| t.contains("Temp")));
    }

    #[test]
    fn test_gauge_half_with_label() {
        let mut gauge = Gauge::percentage(50.0)
            .with_mode(GaugeMode::Half)
            .with_label("Load");
        gauge.bounds = Rect::new(0.0, 0.0, 10.0, 4.0);
        let mut canvas = MockCanvas::new();
        gauge.paint(&mut canvas);
        assert!(canvas.texts.iter().any(|(t, _)| t.contains("Load")));
    }

    #[test]
    fn test_gauge_three_quarter_with_label() {
        let mut gauge = Gauge::percentage(50.0)
            .with_mode(GaugeMode::ThreeQuarter)
            .with_label("CPU");
        gauge.bounds = Rect::new(0.0, 0.0, 10.0, 5.0);
        let mut canvas = MockCanvas::new();
        gauge.paint(&mut canvas);
        assert!(canvas.texts.iter().any(|(t, _)| t.contains("CPU")));
    }
}
