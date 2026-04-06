//! Integration tests for presentar-layout.
//!
//! These tests verify the layout engine works correctly with widget trees.

use presentar_core::widget::{AccessibleRole, LayoutResult};
use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Constraints, Event, Rect, Size,
    TypeId, Widget,
};
use presentar_layout::LayoutEngine;
use std::any::Any;
use std::time::Duration;

// =============================================================================
// Test Widgets
// =============================================================================

/// A simple box widget for testing
struct Box {
    min_size: Size,
    children: Vec<std::boxed::Box<dyn Widget>>,
}

impl Box {
    fn new(width: f32, height: f32) -> Self {
        Self {
            min_size: Size::new(width, height),
            children: Vec::new(),
        }
    }

    fn with_child(mut self, child: impl Widget + 'static) -> Self {
        self.children.push(std::boxed::Box::new(child));
        self
    }
}

impl Brick for Box {
    fn brick_name(&self) -> &'static str {
        "Box"
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

impl Widget for Box {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        constraints.constrain(self.min_size)
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        LayoutResult {
            size: bounds.size(),
        }
    }

    fn paint(&self, _canvas: &mut dyn Canvas) {}

    fn event(&mut self, _event: &Event) -> Option<std::boxed::Box<dyn Any + Send>> {
        None
    }

    fn children(&self) -> &[std::boxed::Box<dyn Widget>] {
        &self.children
    }

    fn children_mut(&mut self) -> &mut [std::boxed::Box<dyn Widget>] {
        &mut self.children
    }

    fn accessible_role(&self) -> AccessibleRole {
        AccessibleRole::Generic
    }
}

/// A flexible widget that fills available space
#[allow(dead_code)]
struct Flexible {
    flex: f32,
    children: Vec<std::boxed::Box<dyn Widget>>,
}

impl Flexible {
    fn new(flex: f32) -> Self {
        Self {
            flex,
            children: Vec::new(),
        }
    }
}

impl Brick for Flexible {
    fn brick_name(&self) -> &'static str {
        "Flexible"
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

impl Widget for Flexible {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        // Flexible widgets take all available space scaled by flex factor
        constraints.biggest()
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        LayoutResult {
            size: bounds.size(),
        }
    }

    fn paint(&self, _canvas: &mut dyn Canvas) {}

    fn event(&mut self, _event: &Event) -> Option<std::boxed::Box<dyn Any + Send>> {
        None
    }

    fn children(&self) -> &[std::boxed::Box<dyn Widget>] {
        &self.children
    }

    fn children_mut(&mut self) -> &mut [std::boxed::Box<dyn Widget>] {
        &mut self.children
    }

    fn accessible_role(&self) -> AccessibleRole {
        AccessibleRole::Generic
    }
}

// =============================================================================
// Layout Engine Integration Tests
// =============================================================================

#[test]
fn test_layout_single_widget() {
    let mut engine = LayoutEngine::new();
    let mut widget = Box::new(100.0, 50.0);
    let viewport = Size::new(800.0, 600.0);

    let tree = engine.compute(&mut widget, viewport);

    assert_eq!(tree.widget_count(), 1);
    let size = tree.get_size(0).expect("should have size");
    assert_eq!(size.width, 100.0);
    assert_eq!(size.height, 50.0);
}

#[test]
fn test_layout_widget_tree() {
    let mut engine = LayoutEngine::new();
    let mut widget = Box::new(400.0, 300.0)
        .with_child(Box::new(100.0, 100.0))
        .with_child(Box::new(100.0, 100.0))
        .with_child(Box::new(100.0, 100.0));

    let viewport = Size::new(800.0, 600.0);
    let tree = engine.compute(&mut widget, viewport);

    // Root + 3 children = 4 widgets
    assert_eq!(tree.widget_count(), 4);
}

#[test]
fn test_layout_nested_widgets() {
    let mut engine = LayoutEngine::new();
    let mut widget = Box::new(500.0, 400.0).with_child(
        Box::new(300.0, 200.0)
            .with_child(Box::new(100.0, 100.0))
            .with_child(Box::new(100.0, 100.0)),
    );

    let viewport = Size::new(800.0, 600.0);
    let tree = engine.compute(&mut widget, viewport);

    // Root -> Child -> 2 grandchildren = 4 widgets
    assert_eq!(tree.widget_count(), 4);
}

#[test]
fn test_layout_constrained_by_viewport() {
    let mut engine = LayoutEngine::new();
    // Widget wants more than viewport
    let mut widget = Box::new(1000.0, 800.0);
    let viewport = Size::new(400.0, 300.0);

    let tree = engine.compute(&mut widget, viewport);

    let size = tree.get_size(0).expect("should have size");
    assert!(size.width <= viewport.width);
    assert!(size.height <= viewport.height);
}

#[test]
fn test_layout_readonly_mode() {
    let mut engine = LayoutEngine::new();
    let widget = Box::new(200.0, 150.0);
    let viewport = Size::new(800.0, 600.0);

    // Read-only layout (for measurement without mutation)
    let tree = engine.compute_readonly(&widget, viewport);

    assert_eq!(tree.widget_count(), 1);
    let size = tree.get_size(0).expect("should have size");
    assert_eq!(size.width, 200.0);
    assert_eq!(size.height, 150.0);
}

#[test]
fn test_layout_cache_clear() {
    let mut engine = LayoutEngine::new();
    let mut widget = Box::new(100.0, 100.0);
    let viewport = Size::new(800.0, 600.0);

    // First layout
    engine.compute(&mut widget, viewport);
    let (hits1, _misses1) = engine.cache_stats();

    // Clear and layout again
    engine.clear_cache();
    engine.compute(&mut widget, viewport);
    let (hits2, _misses2) = engine.cache_stats();

    // After clear, should have no hits from previous layout
    assert_eq!(hits1, 0);
    assert_eq!(hits2, 0);
}

#[test]
fn test_layout_positions_at_origin() {
    let mut engine = LayoutEngine::new();
    let mut widget = Box::new(100.0, 100.0);
    let viewport = Size::new(800.0, 600.0);

    let tree = engine.compute(&mut widget, viewport);

    let pos = tree.get_position(0).expect("should have position");
    assert_eq!(pos.x, 0.0);
    assert_eq!(pos.y, 0.0);
}

// =============================================================================
// Complex Layout Scenarios
// =============================================================================

#[test]
fn test_layout_deep_nesting() {
    let mut engine = LayoutEngine::new();

    // Create deeply nested structure: 5 levels
    let mut widget = Box::new(500.0, 500.0).with_child(
        Box::new(400.0, 400.0).with_child(
            Box::new(300.0, 300.0)
                .with_child(Box::new(200.0, 200.0).with_child(Box::new(100.0, 100.0))),
        ),
    );

    let viewport = Size::new(800.0, 600.0);
    let tree = engine.compute(&mut widget, viewport);

    // 5 levels of nesting
    assert_eq!(tree.widget_count(), 5);
}

#[test]
fn test_layout_wide_tree() {
    let mut engine = LayoutEngine::new();

    // Create wide tree: 1 root with 20 children
    let mut widget = Box::new(800.0, 600.0);
    for _ in 0..20 {
        widget = widget.with_child(Box::new(50.0, 50.0));
    }

    let viewport = Size::new(800.0, 600.0);
    let tree = engine.compute(&mut widget, viewport);

    // 1 root + 20 children
    assert_eq!(tree.widget_count(), 21);
}

#[test]
fn test_layout_empty_widget() {
    let mut engine = LayoutEngine::new();
    let mut widget = Box::new(0.0, 0.0);
    let viewport = Size::new(800.0, 600.0);

    let tree = engine.compute(&mut widget, viewport);

    let size = tree.get_size(0).expect("should have size");
    assert_eq!(size.width, 0.0);
    assert_eq!(size.height, 0.0);
}

#[test]
fn test_layout_flexible_widget() {
    let mut engine = LayoutEngine::new();
    let mut widget = Flexible::new(1.0);
    let viewport = Size::new(400.0, 300.0);

    let tree = engine.compute(&mut widget, viewport);

    let size = tree.get_size(0).expect("should have size");
    // Flexible widget should fill available space
    assert_eq!(size.width, 400.0);
    assert_eq!(size.height, 300.0);
}

// =============================================================================
// Regression Tests
// =============================================================================

#[test]
fn test_layout_zero_viewport() {
    let mut engine = LayoutEngine::new();
    let mut widget = Box::new(100.0, 100.0);
    let viewport = Size::new(0.0, 0.0);

    let tree = engine.compute(&mut widget, viewport);

    let size = tree.get_size(0).expect("should have size");
    assert_eq!(size.width, 0.0);
    assert_eq!(size.height, 0.0);
}

#[test]
fn test_layout_very_large_viewport() {
    let mut engine = LayoutEngine::new();
    let mut widget = Box::new(100.0, 100.0);
    let viewport = Size::new(10000.0, 10000.0);

    let tree = engine.compute(&mut widget, viewport);

    let size = tree.get_size(0).expect("should have size");
    // Widget should keep its size, not expand
    assert_eq!(size.width, 100.0);
    assert_eq!(size.height, 100.0);
}
