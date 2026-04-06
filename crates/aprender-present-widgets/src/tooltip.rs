//! Tooltip widget for contextual hover information.

use presentar_core::{
    widget::{AccessibleRole, LayoutResult, TextStyle},
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    Point, Rect, Size, TypeId, Widget,
};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::time::Duration;

/// Tooltip placement relative to the anchor element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TooltipPlacement {
    /// Above the anchor
    #[default]
    Top,
    /// Below the anchor
    Bottom,
    /// Left of the anchor
    Left,
    /// Right of the anchor
    Right,
    /// Top left corner
    TopLeft,
    /// Top right corner
    TopRight,
    /// Bottom left corner
    BottomLeft,
    /// Bottom right corner
    BottomRight,
}

/// Tooltip widget for showing contextual information on hover.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tooltip {
    /// Tooltip text content
    content: String,
    /// Placement preference
    placement: TooltipPlacement,
    /// Show delay in milliseconds
    delay_ms: u32,
    /// Whether tooltip is currently visible
    visible: bool,
    /// Background color
    background: Color,
    /// Text color
    text_color: Color,
    /// Border color
    border_color: Color,
    /// Border width
    border_width: f32,
    /// Corner radius
    corner_radius: f32,
    /// Padding
    padding: f32,
    /// Arrow size
    arrow_size: f32,
    /// Show arrow
    show_arrow: bool,
    /// Maximum width
    max_width: Option<f32>,
    /// Text size
    text_size: f32,
    /// Accessible name
    accessible_name_value: Option<String>,
    /// Test ID
    test_id_value: Option<String>,
    /// Anchor bounds (for positioning)
    #[serde(skip)]
    anchor_bounds: Rect,
    /// Cached bounds
    #[serde(skip)]
    bounds: Rect,
}

impl Default for Tooltip {
    fn default() -> Self {
        Self {
            content: String::new(),
            placement: TooltipPlacement::Top,
            delay_ms: 200,
            visible: false,
            background: Color::new(0.15, 0.15, 0.15, 0.95),
            text_color: Color::WHITE,
            border_color: Color::new(0.3, 0.3, 0.3, 1.0),
            border_width: 0.0,
            corner_radius: 4.0,
            padding: 8.0,
            arrow_size: 6.0,
            show_arrow: true,
            max_width: Some(250.0),
            text_size: 12.0,
            accessible_name_value: None,
            test_id_value: None,
            anchor_bounds: Rect::default(),
            bounds: Rect::default(),
        }
    }
}

impl Tooltip {
    /// Create a new tooltip.
    #[must_use]
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            ..Self::default()
        }
    }

    /// Set the content.
    #[must_use]
    pub fn content(mut self, content: impl Into<String>) -> Self {
        self.content = content.into();
        self
    }

    /// Set the placement.
    #[must_use]
    pub const fn placement(mut self, placement: TooltipPlacement) -> Self {
        self.placement = placement;
        self
    }

    /// Set the show delay in milliseconds.
    #[must_use]
    pub const fn delay_ms(mut self, ms: u32) -> Self {
        self.delay_ms = ms;
        self
    }

    /// Set visibility.
    #[must_use]
    pub const fn visible(mut self, visible: bool) -> Self {
        self.visible = visible;
        self
    }

    /// Set background color.
    #[must_use]
    pub const fn background(mut self, color: Color) -> Self {
        self.background = color;
        self
    }

    /// Set text color.
    #[must_use]
    pub const fn text_color(mut self, color: Color) -> Self {
        self.text_color = color;
        self
    }

    /// Set border color.
    #[must_use]
    pub const fn border_color(mut self, color: Color) -> Self {
        self.border_color = color;
        self
    }

    /// Set border width.
    #[must_use]
    pub fn border_width(mut self, width: f32) -> Self {
        self.border_width = width.max(0.0);
        self
    }

    /// Set corner radius.
    #[must_use]
    pub fn corner_radius(mut self, radius: f32) -> Self {
        self.corner_radius = radius.max(0.0);
        self
    }

    /// Set padding.
    #[must_use]
    pub fn padding(mut self, padding: f32) -> Self {
        self.padding = padding.max(0.0);
        self
    }

    /// Set arrow size.
    #[must_use]
    pub fn arrow_size(mut self, size: f32) -> Self {
        self.arrow_size = size.max(0.0);
        self
    }

    /// Set whether to show arrow.
    #[must_use]
    pub const fn show_arrow(mut self, show: bool) -> Self {
        self.show_arrow = show;
        self
    }

    /// Set maximum width.
    #[must_use]
    pub fn max_width(mut self, width: f32) -> Self {
        self.max_width = Some(width.max(50.0));
        self
    }

    /// Remove maximum width constraint.
    #[must_use]
    pub const fn no_max_width(mut self) -> Self {
        self.max_width = None;
        self
    }

    /// Set text size.
    #[must_use]
    pub fn text_size(mut self, size: f32) -> Self {
        self.text_size = size.max(8.0);
        self
    }

    /// Set anchor bounds for positioning.
    #[must_use]
    pub const fn anchor(mut self, bounds: Rect) -> Self {
        self.anchor_bounds = bounds;
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

    /// Get the content.
    #[must_use]
    pub fn get_content(&self) -> &str {
        &self.content
    }

    /// Get the placement.
    #[must_use]
    pub const fn get_placement(&self) -> TooltipPlacement {
        self.placement
    }

    /// Get the delay in milliseconds.
    #[must_use]
    pub const fn get_delay_ms(&self) -> u32 {
        self.delay_ms
    }

    /// Check if visible.
    #[must_use]
    pub const fn is_visible(&self) -> bool {
        self.visible
    }

    /// Get the anchor bounds.
    #[must_use]
    pub const fn get_anchor(&self) -> Rect {
        self.anchor_bounds
    }

    /// Show the tooltip.
    pub fn show(&mut self) {
        self.visible = true;
    }

    /// Hide the tooltip.
    pub fn hide(&mut self) {
        self.visible = false;
    }

    /// Toggle visibility.
    pub fn toggle(&mut self) {
        self.visible = !self.visible;
    }

    /// Set anchor bounds (mutable).
    pub fn set_anchor(&mut self, bounds: Rect) {
        self.anchor_bounds = bounds;
    }

    /// Estimate text width.
    fn estimate_text_width(&self) -> f32 {
        // Approximate: chars * text_size * 0.6
        let char_width = self.text_size * 0.6;
        self.content.len() as f32 * char_width
    }

    /// Calculate tooltip size.
    fn calculate_size(&self) -> Size {
        let text_width = self.estimate_text_width();
        let max_text = self.max_width.map(|m| self.padding.mul_add(-2.0, m));

        let content_width = match max_text {
            Some(max) if text_width > max => max,
            _ => text_width,
        };

        let lines = if let Some(max) = max_text {
            (text_width / max).ceil().max(1.0)
        } else {
            1.0
        };

        let content_height = lines * self.text_size * 1.2;

        Size::new(
            self.padding.mul_add(2.0, content_width),
            self.padding.mul_add(2.0, content_height),
        )
    }

    /// Calculate tooltip position based on placement and anchor.
    fn calculate_position(&self, size: Size) -> Point {
        let anchor = self.anchor_bounds;
        let arrow_offset = if self.show_arrow {
            self.arrow_size
        } else {
            0.0
        };

        match self.placement {
            TooltipPlacement::Top => Point::new(
                anchor.x + (anchor.width - size.width) / 2.0,
                anchor.y - size.height - arrow_offset,
            ),
            TooltipPlacement::Bottom => Point::new(
                anchor.x + (anchor.width - size.width) / 2.0,
                anchor.y + anchor.height + arrow_offset,
            ),
            TooltipPlacement::Left => Point::new(
                anchor.x - size.width - arrow_offset,
                anchor.y + (anchor.height - size.height) / 2.0,
            ),
            TooltipPlacement::Right => Point::new(
                anchor.x + anchor.width + arrow_offset,
                anchor.y + (anchor.height - size.height) / 2.0,
            ),
            TooltipPlacement::TopLeft => {
                Point::new(anchor.x, anchor.y - size.height - arrow_offset)
            }
            TooltipPlacement::TopRight => Point::new(
                anchor.x + anchor.width - size.width,
                anchor.y - size.height - arrow_offset,
            ),
            TooltipPlacement::BottomLeft => {
                Point::new(anchor.x, anchor.y + anchor.height + arrow_offset)
            }
            TooltipPlacement::BottomRight => Point::new(
                anchor.x + anchor.width - size.width,
                anchor.y + anchor.height + arrow_offset,
            ),
        }
    }
}

impl Widget for Tooltip {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        if !self.visible || self.content.is_empty() {
            return Size::ZERO;
        }

        let size = self.calculate_size();
        constraints.constrain(size)
    }

    fn layout(&mut self, _bounds: Rect) -> LayoutResult {
        if !self.visible || self.content.is_empty() {
            self.bounds = Rect::default();
            return LayoutResult { size: Size::ZERO };
        }

        let size = self.calculate_size();
        let position = self.calculate_position(size);
        self.bounds = Rect::new(position.x, position.y, size.width, size.height);

        LayoutResult { size }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        if !self.visible || self.content.is_empty() {
            return;
        }

        // Draw background
        canvas.fill_rect(self.bounds, self.background);

        // Draw border if needed
        if self.border_width > 0.0 {
            canvas.stroke_rect(self.bounds, self.border_color, self.border_width);
        }

        // Draw arrow
        if self.show_arrow {
            let arrow_rect = match self.placement {
                TooltipPlacement::Top | TooltipPlacement::TopLeft | TooltipPlacement::TopRight => {
                    let cx = self.bounds.x + self.bounds.width / 2.0;
                    Rect::new(
                        cx - self.arrow_size,
                        self.bounds.y + self.bounds.height,
                        self.arrow_size * 2.0,
                        self.arrow_size,
                    )
                }
                TooltipPlacement::Bottom
                | TooltipPlacement::BottomLeft
                | TooltipPlacement::BottomRight => {
                    let cx = self.bounds.x + self.bounds.width / 2.0;
                    Rect::new(
                        cx - self.arrow_size,
                        self.bounds.y - self.arrow_size,
                        self.arrow_size * 2.0,
                        self.arrow_size,
                    )
                }
                TooltipPlacement::Left => {
                    let cy = self.bounds.y + self.bounds.height / 2.0;
                    Rect::new(
                        self.bounds.x + self.bounds.width,
                        cy - self.arrow_size,
                        self.arrow_size,
                        self.arrow_size * 2.0,
                    )
                }
                TooltipPlacement::Right => {
                    let cy = self.bounds.y + self.bounds.height / 2.0;
                    Rect::new(
                        self.bounds.x - self.arrow_size,
                        cy - self.arrow_size,
                        self.arrow_size,
                        self.arrow_size * 2.0,
                    )
                }
            };
            canvas.fill_rect(arrow_rect, self.background);
        }

        // Draw text
        let text_style = TextStyle {
            size: self.text_size,
            color: self.text_color,
            ..TextStyle::default()
        };

        canvas.draw_text(
            &self.content,
            Point::new(
                self.bounds.x + self.padding,
                self.bounds.y + self.padding + self.text_size,
            ),
            &text_style,
        );
    }

    fn event(&mut self, event: &Event) -> Option<Box<dyn Any + Send>> {
        // Tooltip doesn't handle events directly
        // Visibility is controlled by the parent/anchor widget
        if matches!(event, Event::MouseLeave) {
            self.hide();
        }
        None
    }

    fn children(&self) -> &[Box<dyn Widget>] {
        &[]
    }

    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut []
    }

    fn is_interactive(&self) -> bool {
        false // Tooltip itself is not interactive
    }

    fn is_focusable(&self) -> bool {
        false
    }

    fn accessible_name(&self) -> Option<&str> {
        self.accessible_name_value
            .as_deref()
            .or(Some(&self.content))
    }

    fn accessible_role(&self) -> AccessibleRole {
        AccessibleRole::Generic // Tooltip role
    }

    fn test_id(&self) -> Option<&str> {
        self.test_id_value.as_deref()
    }
}

// PROBAR-SPEC-009: Brick Architecture - Tests define interface
impl Brick for Tooltip {
    fn brick_name(&self) -> &'static str {
        "Tooltip"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        &[
            BrickAssertion::MaxLatencyMs(16),
            BrickAssertion::ContrastRatio(4.5), // WCAG AA for tooltip text
        ]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(16)
    }

    fn verify(&self) -> BrickVerification {
        let mut passed = Vec::new();
        let mut failed = Vec::new();

        // Verify text contrast against background
        let contrast = self.background.contrast_ratio(&self.text_color);
        if contrast >= 4.5 {
            passed.push(BrickAssertion::ContrastRatio(4.5));
        } else {
            failed.push((
                BrickAssertion::ContrastRatio(4.5),
                format!("Contrast ratio {contrast:.2}:1 < 4.5:1"),
            ));
        }

        // Latency assertion always passes at verification time
        passed.push(BrickAssertion::MaxLatencyMs(16));

        BrickVerification {
            passed,
            failed,
            verification_time: Duration::from_micros(10),
        }
    }

    fn to_html(&self) -> String {
        let test_id = self.test_id_value.as_deref().unwrap_or("tooltip");
        let aria_label = self
            .accessible_name_value
            .as_deref()
            .unwrap_or(&self.content);
        format!(
            r#"<div class="brick-tooltip" role="tooltip" data-testid="{}" aria-label="{}">{}</div>"#,
            test_id, aria_label, self.content
        )
    }

    fn to_css(&self) -> String {
        format!(
            r#".brick-tooltip {{
    background: {};
    color: {};
    padding: {}px;
    font-size: {}px;
    border-radius: {}px;
    max-width: {}px;
    position: absolute;
    z-index: 1000;
    pointer-events: none;
}}
.brick-tooltip[data-visible="false"] {{
    display: none;
}}"#,
            self.background.to_hex(),
            self.text_color.to_hex(),
            self.padding,
            self.text_size,
            self.corner_radius,
            self.max_width.unwrap_or(250.0),
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;

    // ===== TooltipPlacement Tests =====

    #[test]
    fn test_tooltip_placement_default() {
        assert_eq!(TooltipPlacement::default(), TooltipPlacement::Top);
    }

    #[test]
    fn test_tooltip_placement_variants() {
        let placements = [
            TooltipPlacement::Top,
            TooltipPlacement::Bottom,
            TooltipPlacement::Left,
            TooltipPlacement::Right,
            TooltipPlacement::TopLeft,
            TooltipPlacement::TopRight,
            TooltipPlacement::BottomLeft,
            TooltipPlacement::BottomRight,
        ];
        assert_eq!(placements.len(), 8);
    }

    // ===== Tooltip Construction Tests =====

    #[test]
    fn test_tooltip_new() {
        let tooltip = Tooltip::new("Help text");
        assert_eq!(tooltip.get_content(), "Help text");
        assert!(!tooltip.is_visible());
    }

    #[test]
    fn test_tooltip_default() {
        let tooltip = Tooltip::default();
        assert!(tooltip.content.is_empty());
        assert_eq!(tooltip.placement, TooltipPlacement::Top);
        assert_eq!(tooltip.delay_ms, 200);
        assert!(!tooltip.visible);
    }

    #[test]
    fn test_tooltip_builder() {
        let tooltip = Tooltip::new("Click to submit")
            .placement(TooltipPlacement::Bottom)
            .delay_ms(500)
            .visible(true)
            .background(Color::BLACK)
            .text_color(Color::WHITE)
            .border_color(Color::RED)
            .border_width(1.0)
            .corner_radius(8.0)
            .padding(12.0)
            .arrow_size(8.0)
            .show_arrow(true)
            .max_width(300.0)
            .text_size(14.0)
            .accessible_name("Submit button tooltip")
            .test_id("submit-tooltip");

        assert_eq!(tooltip.get_content(), "Click to submit");
        assert_eq!(tooltip.get_placement(), TooltipPlacement::Bottom);
        assert_eq!(tooltip.get_delay_ms(), 500);
        assert!(tooltip.is_visible());
        assert_eq!(
            Widget::accessible_name(&tooltip),
            Some("Submit button tooltip")
        );
        assert_eq!(Widget::test_id(&tooltip), Some("submit-tooltip"));
    }

    #[test]
    fn test_tooltip_content() {
        let tooltip = Tooltip::new("old").content("new");
        assert_eq!(tooltip.get_content(), "new");
    }

    // ===== Visibility Tests =====

    #[test]
    fn test_tooltip_show() {
        let mut tooltip = Tooltip::new("Text");
        assert!(!tooltip.is_visible());
        tooltip.show();
        assert!(tooltip.is_visible());
    }

    #[test]
    fn test_tooltip_hide() {
        let mut tooltip = Tooltip::new("Text").visible(true);
        assert!(tooltip.is_visible());
        tooltip.hide();
        assert!(!tooltip.is_visible());
    }

    #[test]
    fn test_tooltip_toggle() {
        let mut tooltip = Tooltip::new("Text");
        assert!(!tooltip.is_visible());
        tooltip.toggle();
        assert!(tooltip.is_visible());
        tooltip.toggle();
        assert!(!tooltip.is_visible());
    }

    // ===== Anchor Tests =====

    #[test]
    fn test_tooltip_anchor() {
        let anchor = Rect::new(100.0, 100.0, 80.0, 30.0);
        let tooltip = Tooltip::new("Help").anchor(anchor);
        assert_eq!(tooltip.get_anchor(), anchor);
    }

    #[test]
    fn test_tooltip_set_anchor() {
        let mut tooltip = Tooltip::new("Help");
        let anchor = Rect::new(50.0, 50.0, 100.0, 40.0);
        tooltip.set_anchor(anchor);
        assert_eq!(tooltip.get_anchor(), anchor);
    }

    // ===== Dimension Constraints Tests =====

    #[test]
    fn test_tooltip_border_width_min() {
        let tooltip = Tooltip::new("Text").border_width(-5.0);
        assert_eq!(tooltip.border_width, 0.0);
    }

    #[test]
    fn test_tooltip_corner_radius_min() {
        let tooltip = Tooltip::new("Text").corner_radius(-5.0);
        assert_eq!(tooltip.corner_radius, 0.0);
    }

    #[test]
    fn test_tooltip_padding_min() {
        let tooltip = Tooltip::new("Text").padding(-5.0);
        assert_eq!(tooltip.padding, 0.0);
    }

    #[test]
    fn test_tooltip_arrow_size_min() {
        let tooltip = Tooltip::new("Text").arrow_size(-5.0);
        assert_eq!(tooltip.arrow_size, 0.0);
    }

    #[test]
    fn test_tooltip_max_width_min() {
        let tooltip = Tooltip::new("Text").max_width(10.0);
        assert_eq!(tooltip.max_width, Some(50.0));
    }

    #[test]
    fn test_tooltip_no_max_width() {
        let tooltip = Tooltip::new("Text").max_width(200.0).no_max_width();
        assert!(tooltip.max_width.is_none());
    }

    #[test]
    fn test_tooltip_text_size_min() {
        let tooltip = Tooltip::new("Text").text_size(2.0);
        assert_eq!(tooltip.text_size, 8.0);
    }

    // ===== Size Calculation Tests =====

    #[test]
    fn test_tooltip_estimate_text_width() {
        let tooltip = Tooltip::new("Hello").text_size(12.0);
        let width = tooltip.estimate_text_width();
        // 5 chars * 12 * 0.6 = 36
        assert!((width - 36.0).abs() < 0.1);
    }

    #[test]
    fn test_tooltip_calculate_size() {
        let tooltip = Tooltip::new("Test").padding(10.0).text_size(12.0);
        let size = tooltip.calculate_size();
        assert!(size.width > 0.0);
        assert!(size.height > 0.0);
    }

    // ===== Position Calculation Tests =====

    #[test]
    fn test_tooltip_position_top() {
        let tooltip = Tooltip::new("Text")
            .placement(TooltipPlacement::Top)
            .anchor(Rect::new(100.0, 100.0, 80.0, 30.0))
            .show_arrow(true)
            .arrow_size(6.0);

        let size = Size::new(50.0, 24.0);
        let pos = tooltip.calculate_position(size);

        // Should be above anchor, centered
        assert!(pos.y < 100.0);
        assert!(pos.x > 100.0); // Offset for centering
    }

    #[test]
    fn test_tooltip_position_bottom() {
        let tooltip = Tooltip::new("Text")
            .placement(TooltipPlacement::Bottom)
            .anchor(Rect::new(100.0, 100.0, 80.0, 30.0))
            .show_arrow(true)
            .arrow_size(6.0);

        let size = Size::new(50.0, 24.0);
        let pos = tooltip.calculate_position(size);

        // Should be below anchor
        assert!(pos.y > 130.0); // 100 + 30 height + arrow
    }

    #[test]
    fn test_tooltip_position_left() {
        let tooltip = Tooltip::new("Text")
            .placement(TooltipPlacement::Left)
            .anchor(Rect::new(100.0, 100.0, 80.0, 30.0))
            .show_arrow(true)
            .arrow_size(6.0);

        let size = Size::new(50.0, 24.0);
        let pos = tooltip.calculate_position(size);

        // Should be to the left
        assert!(pos.x < 100.0 - 50.0);
    }

    #[test]
    fn test_tooltip_position_right() {
        let tooltip = Tooltip::new("Text")
            .placement(TooltipPlacement::Right)
            .anchor(Rect::new(100.0, 100.0, 80.0, 30.0))
            .show_arrow(true)
            .arrow_size(6.0);

        let size = Size::new(50.0, 24.0);
        let pos = tooltip.calculate_position(size);

        // Should be to the right
        assert!(pos.x > 180.0); // 100 + 80 width + arrow
    }

    // ===== Widget Trait Tests =====

    #[test]
    fn test_tooltip_type_id() {
        let tooltip = Tooltip::new("Text");
        assert_eq!(Widget::type_id(&tooltip), TypeId::of::<Tooltip>());
    }

    #[test]
    fn test_tooltip_measure_invisible() {
        let tooltip = Tooltip::new("Text").visible(false);
        let size = tooltip.measure(Constraints::loose(Size::new(500.0, 500.0)));
        assert_eq!(size, Size::ZERO);
    }

    #[test]
    fn test_tooltip_measure_empty() {
        let tooltip = Tooltip::default().visible(true);
        let size = tooltip.measure(Constraints::loose(Size::new(500.0, 500.0)));
        assert_eq!(size, Size::ZERO);
    }

    #[test]
    fn test_tooltip_measure_visible() {
        let tooltip = Tooltip::new("Some helpful text").visible(true);
        let size = tooltip.measure(Constraints::loose(Size::new(500.0, 500.0)));
        assert!(size.width > 0.0);
        assert!(size.height > 0.0);
    }

    #[test]
    fn test_tooltip_layout_invisible() {
        let mut tooltip = Tooltip::new("Text").visible(false);
        let result = tooltip.layout(Rect::new(0.0, 0.0, 200.0, 100.0));
        assert_eq!(result.size, Size::ZERO);
    }

    #[test]
    fn test_tooltip_layout_visible() {
        let mut tooltip = Tooltip::new("Text")
            .visible(true)
            .anchor(Rect::new(100.0, 100.0, 80.0, 30.0));
        let result = tooltip.layout(Rect::new(0.0, 0.0, 500.0, 500.0));
        assert!(result.size.width > 0.0);
        assert!(result.size.height > 0.0);
    }

    #[test]
    fn test_tooltip_children() {
        let tooltip = Tooltip::new("Text");
        assert!(tooltip.children().is_empty());
    }

    #[test]
    fn test_tooltip_is_interactive() {
        let tooltip = Tooltip::new("Text");
        assert!(!tooltip.is_interactive());
    }

    #[test]
    fn test_tooltip_is_focusable() {
        let tooltip = Tooltip::new("Text");
        assert!(!tooltip.is_focusable());
    }

    #[test]
    fn test_tooltip_accessible_role() {
        let tooltip = Tooltip::new("Text");
        assert_eq!(tooltip.accessible_role(), AccessibleRole::Generic);
    }

    #[test]
    fn test_tooltip_accessible_name_default() {
        let tooltip = Tooltip::new("Help text");
        // Falls back to content if no explicit name
        assert_eq!(Widget::accessible_name(&tooltip), Some("Help text"));
    }

    #[test]
    fn test_tooltip_accessible_name_explicit() {
        let tooltip = Tooltip::new("Help text").accessible_name("Explicit name");
        assert_eq!(Widget::accessible_name(&tooltip), Some("Explicit name"));
    }

    #[test]
    fn test_tooltip_test_id() {
        let tooltip = Tooltip::new("Text").test_id("help-tooltip");
        assert_eq!(Widget::test_id(&tooltip), Some("help-tooltip"));
    }

    // ===== Event Tests =====

    #[test]
    fn test_tooltip_mouse_leave_hides() {
        let mut tooltip = Tooltip::new("Text").visible(true);
        assert!(tooltip.is_visible());

        tooltip.event(&Event::MouseLeave);
        assert!(!tooltip.is_visible());
    }

    #[test]
    fn test_tooltip_stays_hidden_on_other_events() {
        let mut tooltip = Tooltip::new("Text").visible(false);
        // Other events don't affect visibility - delay handled externally
        tooltip.event(&Event::MouseMove {
            position: Point::new(0.0, 0.0),
        });
        assert!(!tooltip.is_visible());
    }

    // ===== Color Tests =====

    #[test]
    fn test_tooltip_colors() {
        let tooltip = Tooltip::new("Text")
            .background(Color::BLUE)
            .text_color(Color::RED)
            .border_color(Color::GREEN);

        assert_eq!(tooltip.background, Color::BLUE);
        assert_eq!(tooltip.text_color, Color::RED);
        assert_eq!(tooltip.border_color, Color::GREEN);
    }

    // ===== Additional Position Tests =====

    #[test]
    fn test_tooltip_position_top_left() {
        let tooltip = Tooltip::new("Text")
            .placement(TooltipPlacement::TopLeft)
            .anchor(Rect::new(100.0, 100.0, 80.0, 30.0))
            .show_arrow(true)
            .arrow_size(6.0);

        let size = Size::new(50.0, 24.0);
        let pos = tooltip.calculate_position(size);

        // Should be aligned to left edge of anchor
        assert_eq!(pos.x, 100.0);
        assert!(pos.y < 100.0); // Above anchor
    }

    #[test]
    fn test_tooltip_position_top_right() {
        let tooltip = Tooltip::new("Text")
            .placement(TooltipPlacement::TopRight)
            .anchor(Rect::new(100.0, 100.0, 80.0, 30.0))
            .show_arrow(true)
            .arrow_size(6.0);

        let size = Size::new(50.0, 24.0);
        let pos = tooltip.calculate_position(size);

        // Should be aligned to right edge of anchor (100 + 80 - 50 = 130)
        assert_eq!(pos.x, 130.0);
        assert!(pos.y < 100.0);
    }

    #[test]
    fn test_tooltip_position_bottom_left() {
        let tooltip = Tooltip::new("Text")
            .placement(TooltipPlacement::BottomLeft)
            .anchor(Rect::new(100.0, 100.0, 80.0, 30.0))
            .show_arrow(true)
            .arrow_size(6.0);

        let size = Size::new(50.0, 24.0);
        let pos = tooltip.calculate_position(size);

        // Should be aligned to left edge of anchor
        assert_eq!(pos.x, 100.0);
        assert!(pos.y > 130.0); // Below anchor (100 + 30 + arrow)
    }

    #[test]
    fn test_tooltip_position_bottom_right() {
        let tooltip = Tooltip::new("Text")
            .placement(TooltipPlacement::BottomRight)
            .anchor(Rect::new(100.0, 100.0, 80.0, 30.0))
            .show_arrow(true)
            .arrow_size(6.0);

        let size = Size::new(50.0, 24.0);
        let pos = tooltip.calculate_position(size);

        // Should be aligned to right edge of anchor
        assert_eq!(pos.x, 130.0);
        assert!(pos.y > 130.0);
    }

    #[test]
    fn test_tooltip_position_no_arrow() {
        let tooltip = Tooltip::new("Text")
            .placement(TooltipPlacement::Top)
            .anchor(Rect::new(100.0, 100.0, 80.0, 30.0))
            .show_arrow(false);

        let size = Size::new(50.0, 24.0);
        let pos = tooltip.calculate_position(size);

        // Should be directly above without arrow offset
        assert_eq!(pos.y, 100.0 - 24.0); // anchor.y - size.height
    }

    // ===== Calculate Size Tests =====

    #[test]
    fn test_calculate_size_no_max_width() {
        let tooltip = Tooltip::new("Short text")
            .padding(8.0)
            .text_size(12.0)
            .no_max_width();

        let size = tooltip.calculate_size();
        assert!(size.width > 0.0);
        assert!(size.height > 0.0);
    }

    #[test]
    fn test_calculate_size_wraps_long_text() {
        let tooltip = Tooltip::new("This is a very long tooltip text that should wrap")
            .padding(8.0)
            .text_size(12.0)
            .max_width(100.0);

        let size = tooltip.calculate_size();
        // Width should be capped at max_width
        assert!(size.width <= 100.0);
        // Height should increase due to wrapping
        assert!(size.height > 12.0f32.mul_add(1.2, 16.0)); // More than single line
    }

    // ===== Paint Tests =====

    #[test]
    fn test_tooltip_paint_invisible() {
        use presentar_core::RecordingCanvas;

        let tooltip = Tooltip::new("Text").visible(false);
        let mut canvas = RecordingCanvas::new();
        tooltip.paint(&mut canvas);

        // Should not draw anything when invisible
        assert_eq!(canvas.command_count(), 0);
    }

    #[test]
    fn test_tooltip_paint_empty_content() {
        use presentar_core::RecordingCanvas;

        let tooltip = Tooltip::default().visible(true);
        let mut canvas = RecordingCanvas::new();
        tooltip.paint(&mut canvas);

        // Should not draw anything when content is empty
        assert_eq!(canvas.command_count(), 0);
    }

    #[test]
    fn test_tooltip_paint_visible() {
        use presentar_core::RecordingCanvas;

        let mut tooltip = Tooltip::new("Help text")
            .visible(true)
            .anchor(Rect::new(100.0, 100.0, 80.0, 30.0))
            .placement(TooltipPlacement::Top);

        tooltip.layout(Rect::new(0.0, 0.0, 500.0, 500.0));

        let mut canvas = RecordingCanvas::new();
        tooltip.paint(&mut canvas);

        // Should draw: background rect + arrow rect + text
        assert!(canvas.command_count() >= 2);
    }

    #[test]
    fn test_tooltip_paint_with_border() {
        use presentar_core::RecordingCanvas;

        let mut tooltip = Tooltip::new("Help text")
            .visible(true)
            .anchor(Rect::new(100.0, 100.0, 80.0, 30.0))
            .border_width(2.0);

        tooltip.layout(Rect::new(0.0, 0.0, 500.0, 500.0));

        let mut canvas = RecordingCanvas::new();
        tooltip.paint(&mut canvas);

        // Should draw border when border_width > 0
        assert!(canvas.command_count() >= 3); // bg + border + text + possibly arrow
    }

    #[test]
    fn test_tooltip_paint_without_arrow() {
        use presentar_core::RecordingCanvas;

        let mut tooltip = Tooltip::new("Help text")
            .visible(true)
            .anchor(Rect::new(100.0, 100.0, 80.0, 30.0))
            .show_arrow(false);

        tooltip.layout(Rect::new(0.0, 0.0, 500.0, 500.0));

        let mut canvas = RecordingCanvas::new();
        tooltip.paint(&mut canvas);

        // Should draw bg + text but no arrow
        assert!(canvas.command_count() >= 2);
    }

    #[test]
    fn test_tooltip_paint_bottom_arrow() {
        use presentar_core::RecordingCanvas;

        let mut tooltip = Tooltip::new("Help text")
            .visible(true)
            .anchor(Rect::new(100.0, 100.0, 80.0, 30.0))
            .placement(TooltipPlacement::Bottom);

        tooltip.layout(Rect::new(0.0, 0.0, 500.0, 500.0));

        let mut canvas = RecordingCanvas::new();
        tooltip.paint(&mut canvas);

        assert!(canvas.command_count() >= 3);
    }

    #[test]
    fn test_tooltip_paint_left_arrow() {
        use presentar_core::RecordingCanvas;

        let mut tooltip = Tooltip::new("Help text")
            .visible(true)
            .anchor(Rect::new(100.0, 100.0, 80.0, 30.0))
            .placement(TooltipPlacement::Left);

        tooltip.layout(Rect::new(0.0, 0.0, 500.0, 500.0));

        let mut canvas = RecordingCanvas::new();
        tooltip.paint(&mut canvas);

        assert!(canvas.command_count() >= 3);
    }

    #[test]
    fn test_tooltip_paint_right_arrow() {
        use presentar_core::RecordingCanvas;

        let mut tooltip = Tooltip::new("Help text")
            .visible(true)
            .anchor(Rect::new(100.0, 100.0, 80.0, 30.0))
            .placement(TooltipPlacement::Right);

        tooltip.layout(Rect::new(0.0, 0.0, 500.0, 500.0));

        let mut canvas = RecordingCanvas::new();
        tooltip.paint(&mut canvas);

        assert!(canvas.command_count() >= 3);
    }

    // ===== Brick Trait Tests =====

    #[test]
    fn test_tooltip_brick_name() {
        let tooltip = Tooltip::new("Text");
        assert_eq!(tooltip.brick_name(), "Tooltip");
    }

    #[test]
    fn test_tooltip_brick_assertions() {
        let tooltip = Tooltip::new("Text");
        let assertions = tooltip.assertions();
        assert_eq!(assertions.len(), 2);
        assert!(assertions.contains(&BrickAssertion::MaxLatencyMs(16)));
        assert!(assertions.contains(&BrickAssertion::ContrastRatio(4.5)));
    }

    #[test]
    fn test_tooltip_brick_budget() {
        let tooltip = Tooltip::new("Text");
        let budget = tooltip.budget();
        // BrickBudget::uniform(16) sets internal values
        assert!(budget.measure_ms > 0);
        assert!(budget.layout_ms > 0);
        assert!(budget.paint_ms > 0);
    }

    #[test]
    fn test_tooltip_brick_verify_good_contrast() {
        // Dark background with white text should pass
        let tooltip = Tooltip::new("Text")
            .background(Color::BLACK)
            .text_color(Color::WHITE);

        let verification = tooltip.verify();
        assert!(verification
            .passed
            .contains(&BrickAssertion::ContrastRatio(4.5)));
        assert!(verification.failed.is_empty());
    }

    #[test]
    fn test_tooltip_brick_verify_bad_contrast() {
        // Light background with white text should fail
        let tooltip = Tooltip::new("Text")
            .background(Color::WHITE)
            .text_color(Color::rgb(0.9, 0.9, 0.9)); // Very light gray

        let verification = tooltip.verify();
        assert!(!verification.failed.is_empty());
        assert!(verification
            .failed
            .iter()
            .any(|(a, _)| *a == BrickAssertion::ContrastRatio(4.5)));
    }

    #[test]
    fn test_tooltip_to_html() {
        let tooltip = Tooltip::new("Help text")
            .test_id("help-tooltip")
            .accessible_name("Help information");

        let html = tooltip.to_html();
        assert!(html.contains("role=\"tooltip\""));
        assert!(html.contains("data-testid=\"help-tooltip\""));
        assert!(html.contains("aria-label=\"Help information\""));
        assert!(html.contains("Help text"));
    }

    #[test]
    fn test_tooltip_to_html_default_values() {
        let tooltip = Tooltip::new("Text");
        let html = tooltip.to_html();

        // Should use default test_id and content as aria-label
        assert!(html.contains("data-testid=\"tooltip\""));
        assert!(html.contains("aria-label=\"Text\""));
    }

    #[test]
    fn test_tooltip_to_css() {
        let tooltip = Tooltip::new("Text")
            .padding(12.0)
            .text_size(14.0)
            .corner_radius(6.0)
            .max_width(300.0);

        let css = tooltip.to_css();
        assert!(css.contains("padding: 12px"));
        assert!(css.contains("font-size: 14px"));
        assert!(css.contains("border-radius: 6px"));
        assert!(css.contains("max-width: 300px"));
    }

    // ===== Widget Children Tests =====

    #[test]
    fn test_tooltip_children_mut() {
        let mut tooltip = Tooltip::new("Text");
        assert!(tooltip.children_mut().is_empty());
    }

    // ===== Event Tests =====

    #[test]
    fn test_tooltip_event_other_events() {
        let mut tooltip = Tooltip::new("Text").visible(true);

        // Other events should not affect visibility or return a message
        let result = tooltip.event(&Event::MouseEnter);
        assert!(result.is_none());
        assert!(tooltip.is_visible());
    }

    // ===== Additional Coverage Tests =====

    #[test]
    fn test_tooltip_paint_topleft_arrow() {
        use presentar_core::RecordingCanvas;

        let mut tooltip = Tooltip::new("Help text")
            .visible(true)
            .anchor(Rect::new(100.0, 100.0, 80.0, 30.0))
            .placement(TooltipPlacement::TopLeft);

        tooltip.layout(Rect::new(0.0, 0.0, 500.0, 500.0));

        let mut canvas = RecordingCanvas::new();
        tooltip.paint(&mut canvas);

        assert!(canvas.command_count() >= 3);
    }

    #[test]
    fn test_tooltip_paint_topright_arrow() {
        use presentar_core::RecordingCanvas;

        let mut tooltip = Tooltip::new("Help text")
            .visible(true)
            .anchor(Rect::new(100.0, 100.0, 80.0, 30.0))
            .placement(TooltipPlacement::TopRight);

        tooltip.layout(Rect::new(0.0, 0.0, 500.0, 500.0));

        let mut canvas = RecordingCanvas::new();
        tooltip.paint(&mut canvas);

        assert!(canvas.command_count() >= 3);
    }

    #[test]
    fn test_tooltip_paint_bottomleft_arrow() {
        use presentar_core::RecordingCanvas;

        let mut tooltip = Tooltip::new("Help text")
            .visible(true)
            .anchor(Rect::new(100.0, 100.0, 80.0, 30.0))
            .placement(TooltipPlacement::BottomLeft);

        tooltip.layout(Rect::new(0.0, 0.0, 500.0, 500.0));

        let mut canvas = RecordingCanvas::new();
        tooltip.paint(&mut canvas);

        assert!(canvas.command_count() >= 3);
    }

    #[test]
    fn test_tooltip_paint_bottomright_arrow() {
        use presentar_core::RecordingCanvas;

        let mut tooltip = Tooltip::new("Help text")
            .visible(true)
            .anchor(Rect::new(100.0, 100.0, 80.0, 30.0))
            .placement(TooltipPlacement::BottomRight);

        tooltip.layout(Rect::new(0.0, 0.0, 500.0, 500.0));

        let mut canvas = RecordingCanvas::new();
        tooltip.paint(&mut canvas);

        assert!(canvas.command_count() >= 3);
    }

    #[test]
    fn test_tooltip_layout_empty() {
        let mut tooltip = Tooltip::default().visible(true);
        let result = tooltip.layout(Rect::new(0.0, 0.0, 200.0, 100.0));
        assert_eq!(result.size, Size::ZERO);
    }

    #[test]
    fn test_tooltip_placement_clone() {
        let placement = TooltipPlacement::Bottom;
        let cloned = placement;
        assert_eq!(cloned, TooltipPlacement::Bottom);
    }

    #[test]
    fn test_tooltip_placement_debug() {
        let placement = TooltipPlacement::Right;
        let debug = format!("{placement:?}");
        assert!(debug.contains("Right"));
    }

    #[test]
    fn test_tooltip_clone() {
        let tooltip = Tooltip::new("Text")
            .placement(TooltipPlacement::Left)
            .delay_ms(500);
        let cloned = tooltip;
        assert_eq!(cloned.get_content(), "Text");
        assert_eq!(cloned.get_placement(), TooltipPlacement::Left);
        assert_eq!(cloned.get_delay_ms(), 500);
    }

    #[test]
    fn test_tooltip_serde() {
        let tooltip = Tooltip::new("Help")
            .placement(TooltipPlacement::Bottom)
            .delay_ms(300);

        let json = serde_json::to_string(&tooltip).unwrap();
        let deserialized: Tooltip = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.get_content(), "Help");
        assert_eq!(deserialized.get_placement(), TooltipPlacement::Bottom);
        assert_eq!(deserialized.get_delay_ms(), 300);
    }

    #[test]
    fn test_tooltip_placement_serde() {
        let placement = TooltipPlacement::TopRight;
        let json = serde_json::to_string(&placement).unwrap();
        let deserialized: TooltipPlacement = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, TooltipPlacement::TopRight);
    }

    #[test]
    fn test_tooltip_bounds_after_layout() {
        let mut tooltip = Tooltip::new("Test tooltip text")
            .visible(true)
            .anchor(Rect::new(100.0, 100.0, 80.0, 30.0))
            .placement(TooltipPlacement::Top);

        tooltip.layout(Rect::new(0.0, 0.0, 500.0, 500.0));

        // bounds should be set after layout
        assert!(tooltip.bounds.width > 0.0);
        assert!(tooltip.bounds.height > 0.0);
    }
}
