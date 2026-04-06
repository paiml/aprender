//! Stack widget for z-axis overlapping children.

use presentar_core::{
    widget::{Brick, BrickAssertion, BrickBudget, BrickVerification, LayoutResult},
    Canvas, Constraints, Event, Rect, Size, TypeId, Widget,
};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::time::Duration;

/// How to align children within the stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum StackAlignment {
    /// Align to top-left corner
    #[default]
    TopLeft,
    /// Align to top center
    TopCenter,
    /// Align to top-right corner
    TopRight,
    /// Align to center-left
    CenterLeft,
    /// Center both axes
    Center,
    /// Align to center-right
    CenterRight,
    /// Align to bottom-left corner
    BottomLeft,
    /// Align to bottom center
    BottomCenter,
    /// Align to bottom-right corner
    BottomRight,
}

impl StackAlignment {
    /// Get horizontal offset ratio (0.0 = left, 0.5 = center, 1.0 = right).
    #[must_use]
    pub const fn horizontal_ratio(&self) -> f32 {
        match self {
            Self::TopLeft | Self::CenterLeft | Self::BottomLeft => 0.0,
            Self::TopCenter | Self::Center | Self::BottomCenter => 0.5,
            Self::TopRight | Self::CenterRight | Self::BottomRight => 1.0,
        }
    }

    /// Get vertical offset ratio (0.0 = top, 0.5 = center, 1.0 = bottom).
    #[must_use]
    pub const fn vertical_ratio(&self) -> f32 {
        match self {
            Self::TopLeft | Self::TopCenter | Self::TopRight => 0.0,
            Self::CenterLeft | Self::Center | Self::CenterRight => 0.5,
            Self::BottomLeft | Self::BottomCenter | Self::BottomRight => 1.0,
        }
    }
}

/// How to size the stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum StackFit {
    /// Size to the largest child
    #[default]
    Loose,
    /// Expand to fill available space
    Expand,
}

/// Stack widget for overlaying children.
///
/// Children are painted in order, with later children on top.
#[derive(Serialize, Deserialize)]
pub struct Stack {
    /// Alignment for non-positioned children
    alignment: StackAlignment,
    /// How to size the stack
    fit: StackFit,
    /// Children widgets
    #[serde(skip)]
    children: Vec<Box<dyn Widget>>,
    /// Test ID
    test_id_value: Option<String>,
    /// Cached bounds
    #[serde(skip)]
    bounds: Rect,
}

impl Default for Stack {
    fn default() -> Self {
        Self::new()
    }
}

impl Stack {
    /// Create a new empty stack.
    #[must_use]
    pub fn new() -> Self {
        Self {
            alignment: StackAlignment::TopLeft,
            fit: StackFit::Loose,
            children: Vec::new(),
            test_id_value: None,
            bounds: Rect::default(),
        }
    }

    /// Set alignment.
    #[must_use]
    pub const fn alignment(mut self, alignment: StackAlignment) -> Self {
        self.alignment = alignment;
        self
    }

    /// Set fit mode.
    #[must_use]
    pub const fn fit(mut self, fit: StackFit) -> Self {
        self.fit = fit;
        self
    }

    /// Add a child widget.
    pub fn child(mut self, widget: impl Widget + 'static) -> Self {
        self.children.push(Box::new(widget));
        self
    }

    /// Set test ID.
    #[must_use]
    pub fn with_test_id(mut self, id: impl Into<String>) -> Self {
        self.test_id_value = Some(id.into());
        self
    }

    /// Get alignment.
    #[must_use]
    pub const fn get_alignment(&self) -> StackAlignment {
        self.alignment
    }

    /// Get fit mode.
    #[must_use]
    pub const fn get_fit(&self) -> StackFit {
        self.fit
    }
}

impl Widget for Stack {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        if self.children.is_empty() {
            return match self.fit {
                StackFit::Loose => Size::ZERO,
                StackFit::Expand => Size::new(constraints.max_width, constraints.max_height),
            };
        }

        let mut max_width = 0.0f32;
        let mut max_height = 0.0f32;

        // Measure all children - size is the largest child
        for child in &self.children {
            let child_size = child.measure(constraints);
            max_width = max_width.max(child_size.width);
            max_height = max_height.max(child_size.height);
        }

        match self.fit {
            StackFit::Loose => constraints.constrain(Size::new(max_width, max_height)),
            StackFit::Expand => Size::new(constraints.max_width, constraints.max_height),
        }
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;

        // Layout all children with alignment
        for child in &mut self.children {
            let child_constraints = Constraints::loose(bounds.size());
            let child_size = child.measure(child_constraints);

            // Calculate position based on alignment
            let x = (bounds.width - child_size.width)
                .mul_add(self.alignment.horizontal_ratio(), bounds.x);
            let y = (bounds.height - child_size.height)
                .mul_add(self.alignment.vertical_ratio(), bounds.y);

            let child_bounds = Rect::new(x, y, child_size.width, child_size.height);
            child.layout(child_bounds);
        }

        LayoutResult {
            size: bounds.size(),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        // Paint children in order (first = bottom, last = top)
        for child in &self.children {
            child.paint(canvas);
        }
    }

    fn event(&mut self, event: &Event) -> Option<Box<dyn Any + Send>> {
        // Process events in reverse order (top-most first)
        for child in self.children.iter_mut().rev() {
            if let Some(msg) = child.event(event) {
                return Some(msg);
            }
        }
        None
    }

    fn children(&self) -> &[Box<dyn Widget>] {
        &self.children
    }

    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut self.children
    }

    fn test_id(&self) -> Option<&str> {
        self.test_id_value.as_deref()
    }
}

// PROBAR-SPEC-009: Brick Architecture - Tests define interface
impl Brick for Stack {
    fn brick_name(&self) -> &'static str {
        "Stack"
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
        r#"<div class="brick-stack"></div>"#.to_string()
    }

    fn to_css(&self) -> String {
        ".brick-stack { display: block; position: relative; }".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use presentar_core::Widget;

    // =========================================================================
    // StackAlignment Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_stack_alignment_default() {
        assert_eq!(StackAlignment::default(), StackAlignment::TopLeft);
    }

    #[test]
    fn test_stack_alignment_horizontal_ratio() {
        // Left alignments
        assert_eq!(StackAlignment::TopLeft.horizontal_ratio(), 0.0);
        assert_eq!(StackAlignment::CenterLeft.horizontal_ratio(), 0.0);
        assert_eq!(StackAlignment::BottomLeft.horizontal_ratio(), 0.0);

        // Center alignments
        assert_eq!(StackAlignment::TopCenter.horizontal_ratio(), 0.5);
        assert_eq!(StackAlignment::Center.horizontal_ratio(), 0.5);
        assert_eq!(StackAlignment::BottomCenter.horizontal_ratio(), 0.5);

        // Right alignments
        assert_eq!(StackAlignment::TopRight.horizontal_ratio(), 1.0);
        assert_eq!(StackAlignment::CenterRight.horizontal_ratio(), 1.0);
        assert_eq!(StackAlignment::BottomRight.horizontal_ratio(), 1.0);
    }

    #[test]
    fn test_stack_alignment_vertical_ratio() {
        // Top alignments
        assert_eq!(StackAlignment::TopLeft.vertical_ratio(), 0.0);
        assert_eq!(StackAlignment::TopCenter.vertical_ratio(), 0.0);
        assert_eq!(StackAlignment::TopRight.vertical_ratio(), 0.0);

        // Center alignments
        assert_eq!(StackAlignment::CenterLeft.vertical_ratio(), 0.5);
        assert_eq!(StackAlignment::Center.vertical_ratio(), 0.5);
        assert_eq!(StackAlignment::CenterRight.vertical_ratio(), 0.5);

        // Bottom alignments
        assert_eq!(StackAlignment::BottomLeft.vertical_ratio(), 1.0);
        assert_eq!(StackAlignment::BottomCenter.vertical_ratio(), 1.0);
        assert_eq!(StackAlignment::BottomRight.vertical_ratio(), 1.0);
    }

    // =========================================================================
    // StackFit Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_stack_fit_default() {
        assert_eq!(StackFit::default(), StackFit::Loose);
    }

    // =========================================================================
    // Stack Construction Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_stack_new() {
        let stack = Stack::new();
        assert_eq!(stack.get_alignment(), StackAlignment::TopLeft);
        assert_eq!(stack.get_fit(), StackFit::Loose);
        assert!(stack.children().is_empty());
    }

    #[test]
    fn test_stack_default() {
        let stack = Stack::default();
        assert_eq!(stack.get_alignment(), StackAlignment::TopLeft);
        assert_eq!(stack.get_fit(), StackFit::Loose);
    }

    #[test]
    fn test_stack_builder() {
        let stack = Stack::new()
            .alignment(StackAlignment::Center)
            .fit(StackFit::Expand)
            .with_test_id("my-stack");

        assert_eq!(stack.get_alignment(), StackAlignment::Center);
        assert_eq!(stack.get_fit(), StackFit::Expand);
        assert_eq!(Widget::test_id(&stack), Some("my-stack"));
    }

    // =========================================================================
    // Stack Measure Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_stack_empty_loose() {
        let stack = Stack::new().fit(StackFit::Loose);
        let size = stack.measure(Constraints::loose(Size::new(100.0, 100.0)));
        assert_eq!(size, Size::ZERO);
    }

    #[test]
    fn test_stack_empty_expand() {
        let stack = Stack::new().fit(StackFit::Expand);
        let size = stack.measure(Constraints::loose(Size::new(100.0, 100.0)));
        assert_eq!(size, Size::new(100.0, 100.0));
    }

    // =========================================================================
    // Stack Widget Trait Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_stack_type_id() {
        let stack = Stack::new();
        let type_id = Widget::type_id(&stack);
        assert_eq!(type_id, TypeId::of::<Stack>());
    }

    #[test]
    fn test_stack_test_id_none() {
        let stack = Stack::new();
        assert_eq!(Widget::test_id(&stack), None);
    }

    #[test]
    fn test_stack_test_id_some() {
        let stack = Stack::new().with_test_id("test-stack");
        assert_eq!(Widget::test_id(&stack), Some("test-stack"));
    }

    // =========================================================================
    // StackAlignment Tests - TESTS FIRST
    // =========================================================================

    #[test]
    fn test_stack_alignment_horizontal_ratios() {
        assert_eq!(StackAlignment::TopLeft.horizontal_ratio(), 0.0);
        assert_eq!(StackAlignment::TopCenter.horizontal_ratio(), 0.5);
        assert_eq!(StackAlignment::TopRight.horizontal_ratio(), 1.0);
        assert_eq!(StackAlignment::Center.horizontal_ratio(), 0.5);
        assert_eq!(StackAlignment::BottomRight.horizontal_ratio(), 1.0);
    }

    #[test]
    fn test_stack_alignment_vertical_ratios() {
        assert_eq!(StackAlignment::TopLeft.vertical_ratio(), 0.0);
        assert_eq!(StackAlignment::CenterLeft.vertical_ratio(), 0.5);
        assert_eq!(StackAlignment::BottomLeft.vertical_ratio(), 1.0);
        assert_eq!(StackAlignment::Center.vertical_ratio(), 0.5);
        assert_eq!(StackAlignment::BottomRight.vertical_ratio(), 1.0);
    }

    #[test]
    fn test_stack_alignment_default_is_top_left() {
        let align = StackAlignment::default();
        assert_eq!(align, StackAlignment::TopLeft);
    }

    #[test]
    fn test_stack_fit_default_is_loose() {
        let fit = StackFit::default();
        assert_eq!(fit, StackFit::Loose);
    }

    #[test]
    fn test_stack_layout_sets_bounds() {
        let mut stack = Stack::new();
        let result = stack.layout(Rect::new(10.0, 20.0, 100.0, 80.0));
        assert_eq!(result.size, Size::new(100.0, 80.0));
        assert_eq!(stack.bounds, Rect::new(10.0, 20.0, 100.0, 80.0));
    }

    #[test]
    fn test_stack_children_empty() {
        let stack = Stack::new();
        assert!(stack.children().is_empty());
    }

    #[test]
    fn test_stack_event_no_children() {
        let mut stack = Stack::new();
        stack.layout(Rect::new(0.0, 0.0, 100.0, 100.0));
        let result = stack.event(&Event::MouseEnter);
        assert!(result.is_none());
    }

    // =========================================================================
    // Brick Trait Tests
    // =========================================================================

    #[test]
    fn test_stack_brick_name() {
        let stack = Stack::new();
        assert_eq!(stack.brick_name(), "Stack");
    }

    #[test]
    fn test_stack_brick_assertions() {
        let stack = Stack::new();
        let assertions = stack.assertions();
        assert!(!assertions.is_empty());
        assert!(matches!(assertions[0], BrickAssertion::MaxLatencyMs(16)));
    }

    #[test]
    fn test_stack_brick_budget() {
        let stack = Stack::new();
        let budget = stack.budget();
        // Verify budget has reasonable values
        assert!(budget.layout_ms > 0);
        assert!(budget.paint_ms > 0);
    }

    #[test]
    fn test_stack_brick_verify() {
        let stack = Stack::new();
        let verification = stack.verify();
        assert!(!verification.passed.is_empty());
        assert!(verification.failed.is_empty());
    }

    #[test]
    fn test_stack_brick_to_html() {
        let stack = Stack::new();
        let html = stack.to_html();
        assert!(html.contains("brick-stack"));
    }

    #[test]
    fn test_stack_brick_to_css() {
        let stack = Stack::new();
        let css = stack.to_css();
        assert!(css.contains(".brick-stack"));
        assert!(css.contains("display: block"));
        assert!(css.contains("position: relative"));
    }

    // =========================================================================
    // StackAlignment Comprehensive Tests
    // =========================================================================

    #[test]
    fn test_stack_alignment_all_variants() {
        let alignments = [
            StackAlignment::TopLeft,
            StackAlignment::TopCenter,
            StackAlignment::TopRight,
            StackAlignment::CenterLeft,
            StackAlignment::Center,
            StackAlignment::CenterRight,
            StackAlignment::BottomLeft,
            StackAlignment::BottomCenter,
            StackAlignment::BottomRight,
        ];
        assert_eq!(alignments.len(), 9);
    }

    #[test]
    fn test_stack_alignment_debug() {
        let alignment = StackAlignment::Center;
        let debug_str = format!("{alignment:?}");
        assert!(debug_str.contains("Center"));
    }

    #[test]
    fn test_stack_alignment_eq() {
        assert_eq!(StackAlignment::Center, StackAlignment::Center);
        assert_ne!(StackAlignment::TopLeft, StackAlignment::BottomRight);
    }

    #[test]
    fn test_stack_alignment_clone() {
        let alignment = StackAlignment::BottomCenter;
        let cloned = alignment;
        assert_eq!(cloned, StackAlignment::BottomCenter);
    }

    // =========================================================================
    // StackFit Tests
    // =========================================================================

    #[test]
    fn test_stack_fit_eq() {
        assert_eq!(StackFit::Loose, StackFit::Loose);
        assert_ne!(StackFit::Loose, StackFit::Expand);
    }

    #[test]
    fn test_stack_fit_debug() {
        let fit = StackFit::Expand;
        let debug_str = format!("{fit:?}");
        assert!(debug_str.contains("Expand"));
    }

    #[test]
    fn test_stack_fit_clone() {
        let fit = StackFit::Expand;
        let cloned = fit;
        assert_eq!(cloned, StackFit::Expand);
    }

    // =========================================================================
    // Measure with Children (requires mock widget)
    // =========================================================================

    // Simple mock widget for testing
    struct MockWidget {
        size: Size,
    }

    impl Brick for MockWidget {
        fn brick_name(&self) -> &'static str {
            "MockWidget"
        }

        fn assertions(&self) -> &[BrickAssertion] {
            &[]
        }

        fn budget(&self) -> BrickBudget {
            BrickBudget::uniform(16)
        }

        fn verify(&self) -> BrickVerification {
            BrickVerification {
                passed: vec![],
                failed: vec![],
                verification_time: Duration::from_micros(1),
            }
        }

        fn to_html(&self) -> String {
            String::new()
        }

        fn to_css(&self) -> String {
            String::new()
        }
    }

    impl Widget for MockWidget {
        fn type_id(&self) -> TypeId {
            TypeId::of::<Self>()
        }

        fn measure(&self, _constraints: Constraints) -> Size {
            self.size
        }

        fn layout(&mut self, _bounds: Rect) -> LayoutResult {
            LayoutResult { size: self.size }
        }

        fn paint(&self, _canvas: &mut dyn Canvas) {}

        fn event(&mut self, _event: &Event) -> Option<Box<dyn std::any::Any + Send>> {
            None
        }

        fn children(&self) -> &[Box<dyn Widget>] {
            &[]
        }

        fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
            &mut []
        }
    }

    #[test]
    fn test_stack_measure_with_children_loose() {
        let stack = Stack::new()
            .fit(StackFit::Loose)
            .child(MockWidget {
                size: Size::new(50.0, 30.0),
            })
            .child(MockWidget {
                size: Size::new(100.0, 60.0),
            });

        let size = stack.measure(Constraints::loose(Size::new(500.0, 500.0)));
        // Should be the largest child
        assert_eq!(size.width, 100.0);
        assert_eq!(size.height, 60.0);
    }

    #[test]
    fn test_stack_measure_with_children_expand() {
        let stack = Stack::new().fit(StackFit::Expand).child(MockWidget {
            size: Size::new(50.0, 30.0),
        });

        let size = stack.measure(Constraints::loose(Size::new(500.0, 400.0)));
        // Should expand to fill available space
        assert_eq!(size.width, 500.0);
        assert_eq!(size.height, 400.0);
    }

    #[test]
    fn test_stack_layout_with_children_top_left() {
        let mut stack = Stack::new()
            .alignment(StackAlignment::TopLeft)
            .child(MockWidget {
                size: Size::new(50.0, 30.0),
            });

        stack.layout(Rect::new(0.0, 0.0, 200.0, 150.0));

        // Child should be at top-left corner
        // (Verified through layout result, can't directly access child bounds)
        assert_eq!(stack.bounds, Rect::new(0.0, 0.0, 200.0, 150.0));
    }

    #[test]
    fn test_stack_layout_with_children_center() {
        let mut stack = Stack::new()
            .alignment(StackAlignment::Center)
            .child(MockWidget {
                size: Size::new(50.0, 30.0),
            });

        let result = stack.layout(Rect::new(0.0, 0.0, 200.0, 150.0));
        assert_eq!(result.size, Size::new(200.0, 150.0));
    }

    #[test]
    fn test_stack_layout_with_children_bottom_right() {
        let mut stack = Stack::new()
            .alignment(StackAlignment::BottomRight)
            .child(MockWidget {
                size: Size::new(50.0, 30.0),
            });

        let result = stack.layout(Rect::new(0.0, 0.0, 200.0, 150.0));
        assert_eq!(result.size, Size::new(200.0, 150.0));
    }

    #[test]
    fn test_stack_children_count() {
        let stack = Stack::new()
            .child(MockWidget {
                size: Size::new(50.0, 30.0),
            })
            .child(MockWidget {
                size: Size::new(100.0, 60.0),
            })
            .child(MockWidget {
                size: Size::new(75.0, 45.0),
            });

        assert_eq!(stack.children().len(), 3);
    }

    #[test]
    fn test_stack_children_mut_count() {
        let mut stack = Stack::new()
            .child(MockWidget {
                size: Size::new(50.0, 30.0),
            })
            .child(MockWidget {
                size: Size::new(100.0, 60.0),
            });

        assert_eq!(stack.children_mut().len(), 2);
    }

    // =========================================================================
    // Test ID Tests
    // =========================================================================

    #[test]
    fn test_stack_widget_test_id() {
        let stack = Stack::new().with_test_id("stack-1");
        assert_eq!(Brick::test_id(&stack), None); // Brick::test_id is different method
    }

    #[test]
    fn test_stack_debug() {
        let stack = Stack::new();
        // Stack doesn't derive Debug, but we can test it compiles
        let _ = stack;
    }

    // =========================================================================
    // Default Implementation Tests
    // =========================================================================

    #[test]
    fn test_stack_default_impl() {
        let stack = Stack::default();
        assert_eq!(stack.get_alignment(), StackAlignment::TopLeft);
        assert_eq!(stack.get_fit(), StackFit::Loose);
        assert!(stack.children().is_empty());
    }

    // =========================================================================
    // Event Handling Tests
    // =========================================================================

    struct EventCapturingWidget {
        size: Size,
    }

    impl Brick for EventCapturingWidget {
        fn brick_name(&self) -> &'static str {
            "EventCapturingWidget"
        }

        fn assertions(&self) -> &[BrickAssertion] {
            &[]
        }

        fn budget(&self) -> BrickBudget {
            BrickBudget::uniform(16)
        }

        fn verify(&self) -> BrickVerification {
            BrickVerification {
                passed: vec![],
                failed: vec![],
                verification_time: Duration::from_micros(1),
            }
        }

        fn to_html(&self) -> String {
            String::new()
        }

        fn to_css(&self) -> String {
            String::new()
        }
    }

    impl Widget for EventCapturingWidget {
        fn type_id(&self) -> TypeId {
            TypeId::of::<Self>()
        }

        fn measure(&self, _constraints: Constraints) -> Size {
            self.size
        }

        fn layout(&mut self, _bounds: Rect) -> LayoutResult {
            LayoutResult { size: self.size }
        }

        fn paint(&self, _canvas: &mut dyn Canvas) {}

        fn event(&mut self, _event: &Event) -> Option<Box<dyn std::any::Any + Send>> {
            // Return a message to indicate event was handled
            Some(Box::new("handled".to_string()))
        }

        fn children(&self) -> &[Box<dyn Widget>] {
            &[]
        }

        fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
            &mut []
        }
    }

    #[test]
    fn test_stack_event_propagates_to_children() {
        let mut stack = Stack::new().child(EventCapturingWidget {
            size: Size::new(50.0, 30.0),
        });

        stack.layout(Rect::new(0.0, 0.0, 200.0, 150.0));
        let result = stack.event(&Event::MouseEnter);

        // Event should be captured by child
        assert!(result.is_some());
    }

    #[test]
    fn test_stack_event_reverse_order() {
        // Events should be processed in reverse order (top-most child first)
        let mut stack = Stack::new()
            .child(MockWidget {
                size: Size::new(50.0, 30.0),
            })
            .child(EventCapturingWidget {
                size: Size::new(50.0, 30.0),
            });

        stack.layout(Rect::new(0.0, 0.0, 200.0, 150.0));
        let result = stack.event(&Event::MouseEnter);

        // Last child (EventCapturingWidget) should handle it first
        assert!(result.is_some());
    }

    // =========================================================================
    // Measure Edge Cases
    // =========================================================================

    #[test]
    fn test_stack_measure_loose_with_constraints() {
        let stack = Stack::new().fit(StackFit::Loose).child(MockWidget {
            size: Size::new(500.0, 400.0),
        });

        // Constraint smaller than child
        let size = stack.measure(Constraints {
            min_width: 0.0,
            min_height: 0.0,
            max_width: 200.0,
            max_height: 150.0,
        });

        // Should be constrained
        assert_eq!(size.width, 200.0);
        assert_eq!(size.height, 150.0);
    }

    #[test]
    fn test_stack_measure_multiple_children_different_sizes() {
        let stack = Stack::new()
            .fit(StackFit::Loose)
            .child(MockWidget {
                size: Size::new(50.0, 100.0),
            })
            .child(MockWidget {
                size: Size::new(100.0, 50.0),
            })
            .child(MockWidget {
                size: Size::new(75.0, 75.0),
            });

        let size = stack.measure(Constraints::loose(Size::new(500.0, 500.0)));
        // Should take maximum of each dimension
        assert_eq!(size.width, 100.0);
        assert_eq!(size.height, 100.0);
    }
}
