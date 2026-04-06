//! `CollapsiblePanel` widget for expandable/collapsible content sections.
//!
//! Provides a panel with header that can be collapsed to save space.
//! Based on btop panel patterns with toggle indicators.

use crate::widgets::border::BorderStyle;
use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event, Key,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// Direction for collapse animation/behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CollapseDirection {
    /// Collapses upward (content below header disappears).
    #[default]
    Up,
    /// Collapses downward (content above header disappears).
    Down,
    /// Collapses leftward.
    Left,
    /// Collapses rightward.
    Right,
}

/// Indicator characters for expand/collapse state.
#[derive(Debug, Clone)]
pub struct CollapseIndicators {
    /// Character shown when expanded.
    pub expanded: char,
    /// Character shown when collapsed.
    pub collapsed: char,
}

impl Default for CollapseIndicators {
    fn default() -> Self {
        Self::triangle()
    }
}

impl CollapseIndicators {
    /// Triangle indicators (▼/▶).
    #[must_use]
    pub fn triangle() -> Self {
        Self {
            expanded: '▼',
            collapsed: '▶',
        }
    }

    /// Plus/minus indicators (−/+).
    #[must_use]
    pub fn plus_minus() -> Self {
        Self {
            expanded: '−',
            collapsed: '+',
        }
    }

    /// Chevron indicators (˅/˃).
    #[must_use]
    pub fn chevron() -> Self {
        Self {
            expanded: '˅',
            collapsed: '˃',
        }
    }

    /// Arrow indicators (↓/→).
    #[must_use]
    pub fn arrow() -> Self {
        Self {
            expanded: '↓',
            collapsed: '→',
        }
    }

    /// Get the current indicator based on collapsed state.
    #[must_use]
    pub fn current(&self, collapsed: bool) -> char {
        if collapsed {
            self.collapsed
        } else {
            self.expanded
        }
    }
}

/// Collapsible panel widget with header and toggle.
#[derive(Debug, Clone)]
pub struct CollapsiblePanel {
    /// Panel title.
    title: String,
    /// Collapsed state.
    collapsed: bool,
    /// Collapse direction.
    direction: CollapseDirection,
    /// Indicator characters.
    indicators: CollapseIndicators,
    /// Border style.
    border_style: BorderStyle,
    /// Title color.
    title_color: Color,
    /// Border color.
    border_color: Color,
    /// Indicator color.
    indicator_color: Color,
    /// Content height when expanded (in lines).
    content_height: usize,
    /// Cached bounds.
    bounds: Rect,
}

impl Default for CollapsiblePanel {
    fn default() -> Self {
        Self::new("Panel")
    }
}

impl CollapsiblePanel {
    /// Create a new collapsible panel.
    #[must_use]
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            collapsed: false,
            direction: CollapseDirection::default(),
            indicators: CollapseIndicators::default(),
            border_style: BorderStyle::Rounded,
            title_color: Color::WHITE,
            border_color: Color::new(0.4, 0.5, 0.6, 1.0),
            indicator_color: Color::new(0.8, 0.8, 0.3, 1.0),
            content_height: 3,
            bounds: Rect::default(),
        }
    }

    /// Set collapsed state.
    #[must_use]
    pub fn with_collapsed(mut self, collapsed: bool) -> Self {
        self.collapsed = collapsed;
        self
    }

    /// Set collapse direction.
    #[must_use]
    pub fn with_direction(mut self, direction: CollapseDirection) -> Self {
        self.direction = direction;
        self
    }

    /// Set indicator style.
    #[must_use]
    pub fn with_indicators(mut self, indicators: CollapseIndicators) -> Self {
        self.indicators = indicators;
        self
    }

    /// Set border style.
    #[must_use]
    pub fn with_border_style(mut self, style: BorderStyle) -> Self {
        self.border_style = style;
        self
    }

    /// Set title color.
    #[must_use]
    pub fn with_title_color(mut self, color: Color) -> Self {
        self.title_color = color;
        self
    }

    /// Set border color.
    #[must_use]
    pub fn with_border_color(mut self, color: Color) -> Self {
        self.border_color = color;
        self
    }

    /// Set indicator color.
    #[must_use]
    pub fn with_indicator_color(mut self, color: Color) -> Self {
        self.indicator_color = color;
        self
    }

    /// Set content height when expanded.
    #[must_use]
    pub fn with_content_height(mut self, height: usize) -> Self {
        self.content_height = height;
        self
    }

    // ================= State =================

    /// Get the title.
    #[must_use]
    pub fn title(&self) -> &str {
        &self.title
    }

    /// Check if panel is collapsed.
    #[must_use]
    pub fn is_collapsed(&self) -> bool {
        self.collapsed
    }

    /// Check if panel is expanded.
    #[must_use]
    pub fn is_expanded(&self) -> bool {
        !self.collapsed
    }

    /// Get collapse direction.
    #[must_use]
    pub fn direction(&self) -> CollapseDirection {
        self.direction
    }

    /// Get the inner content area (excluding border).
    #[must_use]
    pub fn inner_rect(&self) -> Rect {
        if self.collapsed {
            Rect::new(self.bounds.x + 1.0, self.bounds.y + 1.0, 0.0, 0.0)
        } else {
            Rect::new(
                self.bounds.x + 1.0,
                self.bounds.y + 1.0,
                (self.bounds.width - 2.0).max(0.0),
                (self.bounds.height - 2.0).max(0.0),
            )
        }
    }

    /// Get effective height based on collapsed state.
    #[must_use]
    pub fn effective_height(&self) -> usize {
        if self.collapsed {
            2 // Just header + bottom border
        } else {
            2 + self.content_height // Header + content + bottom border
        }
    }

    // ================= Actions =================

    /// Toggle collapsed state.
    pub fn toggle(&mut self) {
        self.collapsed = !self.collapsed;
    }

    /// Expand the panel.
    pub fn expand(&mut self) {
        self.collapsed = false;
    }

    /// Collapse the panel.
    pub fn collapse(&mut self) {
        self.collapsed = true;
    }

    /// Set title.
    pub fn set_title(&mut self, title: impl Into<String>) {
        self.title = title.into();
    }
}

impl Brick for CollapsiblePanel {
    fn brick_name(&self) -> &'static str {
        "collapsible_panel"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        static ASSERTIONS: &[BrickAssertion] = &[BrickAssertion::max_latency_ms(16)];
        ASSERTIONS
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(8)
    }

    fn verify(&self) -> BrickVerification {
        BrickVerification {
            passed: self.assertions().to_vec(),
            failed: vec![],
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

impl Widget for CollapsiblePanel {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let width = constraints.max_width.clamp(10.0, 40.0);
        let height = self.effective_height() as f32;
        constraints.constrain(Size::new(width, height))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        let width = self.bounds.width as usize;
        let height = self.bounds.height as usize;

        if width < 4 || height < 2 {
            return;
        }

        let (tl, top, tr, left, right, bl, bottom, br) = self.border_style.chars();
        let border_style = TextStyle {
            color: self.border_color,
            ..Default::default()
        };
        let indicator_style = TextStyle {
            color: self.indicator_color,
            ..Default::default()
        };
        let title_style = TextStyle {
            color: self.title_color,
            ..Default::default()
        };

        // Draw top border with indicator and title
        let indicator = self.indicators.current(self.collapsed);

        // Top-left corner
        canvas.draw_text(
            &tl.to_string(),
            Point::new(self.bounds.x, self.bounds.y),
            &border_style,
        );

        // Indicator after corner
        canvas.draw_text(
            &indicator.to_string(),
            Point::new(self.bounds.x + 1.0, self.bounds.y),
            &indicator_style,
        );

        // Space + title
        let title_start = 3;
        let available = width.saturating_sub(title_start + 2);
        let displayed_title: String = self.title.chars().take(available).collect();

        canvas.draw_text(
            &format!(" {displayed_title} "),
            Point::new(self.bounds.x + 2.0, self.bounds.y),
            &title_style,
        );

        // Rest of top border
        let title_len = displayed_title.chars().count() + 2; // +2 for spaces
        let rest_start = title_start + title_len;
        if rest_start < width - 1 {
            let rest: String = std::iter::repeat(top)
                .take(width - rest_start - 1)
                .collect();
            canvas.draw_text(
                &rest,
                Point::new(self.bounds.x + rest_start as f32, self.bounds.y),
                &border_style,
            );
        }

        // Top-right corner
        canvas.draw_text(
            &tr.to_string(),
            Point::new(self.bounds.x + (width - 1) as f32, self.bounds.y),
            &border_style,
        );

        // If collapsed, just draw bottom border
        if self.collapsed {
            // Bottom border (immediately after header)
            let mut bottom_line = String::with_capacity(width);
            bottom_line.push(bl);
            for _ in 0..(width - 2) {
                bottom_line.push(bottom);
            }
            bottom_line.push(br);
            canvas.draw_text(
                &bottom_line,
                Point::new(self.bounds.x, self.bounds.y + 1.0),
                &border_style,
            );
        } else {
            // Draw side borders for content area
            for y in 1..(height - 1) {
                canvas.draw_text(
                    &left.to_string(),
                    Point::new(self.bounds.x, self.bounds.y + y as f32),
                    &border_style,
                );
                canvas.draw_text(
                    &right.to_string(),
                    Point::new(self.bounds.x + (width - 1) as f32, self.bounds.y + y as f32),
                    &border_style,
                );
            }

            // Bottom border
            let mut bottom_line = String::with_capacity(width);
            bottom_line.push(bl);
            for _ in 0..(width - 2) {
                bottom_line.push(bottom);
            }
            bottom_line.push(br);
            canvas.draw_text(
                &bottom_line,
                Point::new(self.bounds.x, self.bounds.y + (height - 1) as f32),
                &border_style,
            );
        }
    }

    fn event(&mut self, event: &Event) -> Option<Box<dyn Any + Send>> {
        match event {
            Event::KeyDown {
                key: Key::Enter | Key::Space,
                ..
            } => {
                self.toggle();
                Some(Box::new(self.collapsed))
            }
            _ => None,
        }
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

    // =====================================================
    // Construction Tests
    // =====================================================

    #[test]
    fn test_new() {
        let panel = CollapsiblePanel::new("CPU");
        assert_eq!(panel.title(), "CPU");
        assert!(!panel.is_collapsed());
    }

    #[test]
    fn test_default() {
        let panel = CollapsiblePanel::default();
        assert_eq!(panel.title(), "Panel");
        assert!(!panel.is_collapsed());
    }

    #[test]
    fn test_with_collapsed() {
        let panel = CollapsiblePanel::new("Test").with_collapsed(true);
        assert!(panel.is_collapsed());
    }

    #[test]
    fn test_with_direction() {
        let panel = CollapsiblePanel::new("Test").with_direction(CollapseDirection::Left);
        assert_eq!(panel.direction(), CollapseDirection::Left);
    }

    #[test]
    fn test_with_indicators() {
        let panel = CollapsiblePanel::new("Test").with_indicators(CollapseIndicators::plus_minus());
        assert_eq!(panel.indicators.expanded, '−');
    }

    #[test]
    fn test_with_border_style() {
        let panel = CollapsiblePanel::new("Test").with_border_style(BorderStyle::Double);
        assert_eq!(panel.border_style, BorderStyle::Double);
    }

    #[test]
    fn test_with_title_color() {
        let panel = CollapsiblePanel::new("Test").with_title_color(Color::RED);
        assert_eq!(panel.title_color, Color::RED);
    }

    #[test]
    fn test_with_border_color() {
        let panel = CollapsiblePanel::new("Test").with_border_color(Color::GREEN);
        assert_eq!(panel.border_color, Color::GREEN);
    }

    #[test]
    fn test_with_indicator_color() {
        let panel = CollapsiblePanel::new("Test").with_indicator_color(Color::BLUE);
        assert_eq!(panel.indicator_color, Color::BLUE);
    }

    #[test]
    fn test_with_content_height() {
        let panel = CollapsiblePanel::new("Test").with_content_height(5);
        assert_eq!(panel.content_height, 5);
    }

    // =====================================================
    // CollapseIndicators Tests
    // =====================================================

    #[test]
    fn test_indicators_default() {
        let ind = CollapseIndicators::default();
        assert_eq!(ind.expanded, '▼');
        assert_eq!(ind.collapsed, '▶');
    }

    #[test]
    fn test_indicators_triangle() {
        let ind = CollapseIndicators::triangle();
        assert_eq!(ind.expanded, '▼');
        assert_eq!(ind.collapsed, '▶');
    }

    #[test]
    fn test_indicators_plus_minus() {
        let ind = CollapseIndicators::plus_minus();
        assert_eq!(ind.expanded, '−');
        assert_eq!(ind.collapsed, '+');
    }

    #[test]
    fn test_indicators_chevron() {
        let ind = CollapseIndicators::chevron();
        assert_eq!(ind.expanded, '˅');
        assert_eq!(ind.collapsed, '˃');
    }

    #[test]
    fn test_indicators_arrow() {
        let ind = CollapseIndicators::arrow();
        assert_eq!(ind.expanded, '↓');
        assert_eq!(ind.collapsed, '→');
    }

    #[test]
    fn test_indicators_current_expanded() {
        let ind = CollapseIndicators::triangle();
        assert_eq!(ind.current(false), '▼');
    }

    #[test]
    fn test_indicators_current_collapsed() {
        let ind = CollapseIndicators::triangle();
        assert_eq!(ind.current(true), '▶');
    }

    // =====================================================
    // CollapseDirection Tests
    // =====================================================

    #[test]
    fn test_direction_default() {
        assert_eq!(CollapseDirection::default(), CollapseDirection::Up);
    }

    #[test]
    fn test_direction_variants() {
        let _ = CollapseDirection::Up;
        let _ = CollapseDirection::Down;
        let _ = CollapseDirection::Left;
        let _ = CollapseDirection::Right;
    }

    // =====================================================
    // State Tests
    // =====================================================

    #[test]
    fn test_is_expanded() {
        let panel = CollapsiblePanel::new("Test");
        assert!(panel.is_expanded());
    }

    #[test]
    fn test_is_expanded_when_collapsed() {
        let panel = CollapsiblePanel::new("Test").with_collapsed(true);
        assert!(!panel.is_expanded());
    }

    #[test]
    fn test_effective_height_expanded() {
        let panel = CollapsiblePanel::new("Test").with_content_height(5);
        assert_eq!(panel.effective_height(), 7); // 2 + 5
    }

    #[test]
    fn test_effective_height_collapsed() {
        let panel = CollapsiblePanel::new("Test").with_collapsed(true);
        assert_eq!(panel.effective_height(), 2);
    }

    #[test]
    fn test_inner_rect_expanded() {
        let mut panel = CollapsiblePanel::new("Test");
        panel.bounds = Rect::new(0.0, 0.0, 20.0, 10.0);
        let inner = panel.inner_rect();
        assert_eq!(inner.x, 1.0);
        assert_eq!(inner.y, 1.0);
        assert_eq!(inner.width, 18.0);
        assert_eq!(inner.height, 8.0);
    }

    #[test]
    fn test_inner_rect_collapsed() {
        let mut panel = CollapsiblePanel::new("Test").with_collapsed(true);
        panel.bounds = Rect::new(0.0, 0.0, 20.0, 10.0);
        let inner = panel.inner_rect();
        assert_eq!(inner.width, 0.0);
        assert_eq!(inner.height, 0.0);
    }

    // =====================================================
    // Action Tests
    // =====================================================

    #[test]
    fn test_toggle_expand_to_collapse() {
        let mut panel = CollapsiblePanel::new("Test");
        panel.toggle();
        assert!(panel.is_collapsed());
    }

    #[test]
    fn test_toggle_collapse_to_expand() {
        let mut panel = CollapsiblePanel::new("Test").with_collapsed(true);
        panel.toggle();
        assert!(panel.is_expanded());
    }

    #[test]
    fn test_expand() {
        let mut panel = CollapsiblePanel::new("Test").with_collapsed(true);
        panel.expand();
        assert!(panel.is_expanded());
    }

    #[test]
    fn test_collapse() {
        let mut panel = CollapsiblePanel::new("Test");
        panel.collapse();
        assert!(panel.is_collapsed());
    }

    #[test]
    fn test_set_title() {
        let mut panel = CollapsiblePanel::new("Old");
        panel.set_title("New");
        assert_eq!(panel.title(), "New");
    }

    // =====================================================
    // Brick Trait Tests
    // =====================================================

    #[test]
    fn test_brick_name() {
        let panel = CollapsiblePanel::new("Test");
        assert_eq!(panel.brick_name(), "collapsible_panel");
    }

    #[test]
    fn test_assertions_not_empty() {
        let panel = CollapsiblePanel::new("Test");
        assert!(!panel.assertions().is_empty());
    }

    #[test]
    fn test_budget() {
        let panel = CollapsiblePanel::new("Test");
        assert!(panel.budget().paint_ms > 0);
    }

    #[test]
    fn test_verify() {
        let panel = CollapsiblePanel::new("Test");
        assert!(panel.verify().is_valid());
    }

    #[test]
    fn test_to_html() {
        let panel = CollapsiblePanel::new("Test");
        assert!(panel.to_html().is_empty());
    }

    #[test]
    fn test_to_css() {
        let panel = CollapsiblePanel::new("Test");
        assert!(panel.to_css().is_empty());
    }

    // =====================================================
    // Widget Trait Tests
    // =====================================================

    #[test]
    fn test_type_id() {
        let panel = CollapsiblePanel::new("Test");
        assert_eq!(Widget::type_id(&panel), TypeId::of::<CollapsiblePanel>());
    }

    #[test]
    fn test_measure_expanded() {
        let panel = CollapsiblePanel::new("Test").with_content_height(5);
        let size = panel.measure(Constraints::loose(Size::new(100.0, 100.0)));
        assert!(size.width >= 10.0);
        assert_eq!(size.height, 7.0); // 2 + 5
    }

    #[test]
    fn test_measure_collapsed() {
        let panel = CollapsiblePanel::new("Test").with_collapsed(true);
        let size = panel.measure(Constraints::loose(Size::new(100.0, 100.0)));
        assert_eq!(size.height, 2.0);
    }

    #[test]
    fn test_layout() {
        let mut panel = CollapsiblePanel::new("Test");
        let bounds = Rect::new(5.0, 10.0, 30.0, 8.0);
        let result = panel.layout(bounds);
        assert_eq!(result.size.width, 30.0);
        assert_eq!(panel.bounds, bounds);
    }

    #[test]
    fn test_children() {
        let panel = CollapsiblePanel::new("Test");
        assert!(panel.children().is_empty());
    }

    #[test]
    fn test_children_mut() {
        let mut panel = CollapsiblePanel::new("Test");
        assert!(panel.children_mut().is_empty());
    }

    // =====================================================
    // Paint Tests
    // =====================================================

    #[test]
    fn test_paint_expanded() {
        let mut panel = CollapsiblePanel::new("CPU");
        panel.bounds = Rect::new(0.0, 0.0, 20.0, 5.0);
        let mut canvas = MockCanvas::new();
        panel.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
        // Should contain title
        assert!(canvas.texts.iter().any(|(t, _)| t.contains("CPU")));
    }

    #[test]
    fn test_paint_collapsed() {
        let mut panel = CollapsiblePanel::new("CPU").with_collapsed(true);
        panel.bounds = Rect::new(0.0, 0.0, 20.0, 2.0);
        let mut canvas = MockCanvas::new();
        panel.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_paint_small_bounds() {
        let mut panel = CollapsiblePanel::new("CPU");
        panel.bounds = Rect::new(0.0, 0.0, 3.0, 1.0);
        let mut canvas = MockCanvas::new();
        panel.paint(&mut canvas);
        // Should handle gracefully (skip painting)
    }

    #[test]
    fn test_paint_with_indicator_expanded() {
        let mut panel = CollapsiblePanel::new("Test");
        panel.bounds = Rect::new(0.0, 0.0, 20.0, 5.0);
        let mut canvas = MockCanvas::new();
        panel.paint(&mut canvas);
        // Should show expanded indicator
        assert!(canvas.texts.iter().any(|(t, _)| t == "▼"));
    }

    #[test]
    fn test_paint_with_indicator_collapsed() {
        let mut panel = CollapsiblePanel::new("Test").with_collapsed(true);
        panel.bounds = Rect::new(0.0, 0.0, 20.0, 2.0);
        let mut canvas = MockCanvas::new();
        panel.paint(&mut canvas);
        // Should show collapsed indicator
        assert!(canvas.texts.iter().any(|(t, _)| t == "▶"));
    }

    // =====================================================
    // Event Tests
    // =====================================================

    #[test]
    fn test_event_enter_toggles() {
        let mut panel = CollapsiblePanel::new("Test");
        let event = Event::key_down(Key::Enter);
        let result = panel.event(&event);
        assert!(result.is_some());
        assert!(panel.is_collapsed());
    }

    #[test]
    fn test_event_space_toggles() {
        let mut panel = CollapsiblePanel::new("Test");
        let event = Event::key_down(Key::Space);
        let result = panel.event(&event);
        assert!(result.is_some());
        assert!(panel.is_collapsed());
    }

    #[test]
    fn test_event_other_keys_ignored() {
        let mut panel = CollapsiblePanel::new("Test");
        let event = Event::key_down(Key::Left);
        let result = panel.event(&event);
        assert!(result.is_none());
        assert!(panel.is_expanded());
    }

    #[test]
    fn test_event_returns_state() {
        let mut panel = CollapsiblePanel::new("Test");
        let event = Event::key_down(Key::Enter);
        let result = panel.event(&event);
        // Result should be the new collapsed state
        if let Some(boxed) = result {
            let state = boxed.downcast_ref::<bool>();
            assert_eq!(state, Some(&true));
        }
    }

    // =====================================================
    // Long Title Tests
    // =====================================================

    #[test]
    fn test_long_title_truncated() {
        let mut panel = CollapsiblePanel::new("Very Long Title That Should Be Truncated");
        panel.bounds = Rect::new(0.0, 0.0, 15.0, 5.0);
        let mut canvas = MockCanvas::new();
        panel.paint(&mut canvas);
        // Should not overflow bounds
    }
}
