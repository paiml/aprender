//! Layout engine implementation.

use presentar_core::{Constraints, Rect, Size, Widget};
use std::collections::HashMap;

use crate::cache::LayoutCache;

/// Layout tree containing computed positions.
#[derive(Debug, Default)]
pub struct LayoutTree {
    /// Computed sizes for each widget
    pub sizes: HashMap<u64, Size>,
    /// Computed positions for each widget
    pub positions: HashMap<u64, Rect>,
}

impl LayoutTree {
    /// Get the size for a widget by ID.
    #[must_use]
    pub fn get_size(&self, id: u64) -> Option<Size> {
        self.sizes.get(&id).copied()
    }

    /// Get the position for a widget by ID.
    #[must_use]
    pub fn get_position(&self, id: u64) -> Option<Rect> {
        self.positions.get(&id).copied()
    }

    /// Get widget count.
    #[must_use]
    pub fn widget_count(&self) -> usize {
        self.positions.len()
    }
}

/// Layout engine with memoization.
#[derive(Debug, Default)]
pub struct LayoutEngine {
    cache: LayoutCache,
    next_id: u64,
}

impl LayoutEngine {
    /// Create a new layout engine.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute layout for the widget tree.
    ///
    /// This performs a two-phase layout:
    /// 1. Measure phase (bottom-up): Determine intrinsic sizes
    /// 2. Layout phase (top-down): Assign final positions and sizes
    pub fn compute(&mut self, root: &mut dyn Widget, viewport: Size) -> LayoutTree {
        self.cache.clear();
        self.next_id = 0;

        let constraints = Constraints::loose(viewport);

        // Phase 1: Measure (bottom-up)
        let mut sizes = HashMap::new();
        self.measure_tree(root, constraints, &mut sizes);

        // Reset ID counter for layout phase
        self.next_id = 0;

        // Phase 2: Layout (top-down)
        let mut positions = HashMap::new();
        let bounds = Rect::from_size(viewport);
        self.layout_tree(root, bounds, &mut positions);

        LayoutTree { sizes, positions }
    }

    /// Compute layout with read-only widget tree (for measurement only).
    pub fn compute_readonly(&mut self, root: &dyn Widget, viewport: Size) -> LayoutTree {
        self.cache.clear();
        self.next_id = 0;

        let constraints = Constraints::loose(viewport);

        // Phase 1: Measure (bottom-up)
        let mut sizes = HashMap::new();
        self.measure_tree(root, constraints, &mut sizes);

        // Reset ID counter
        self.next_id = 0;

        // Phase 2: Position (simplified for read-only)
        let mut positions = HashMap::new();
        let bounds = Rect::from_size(viewport);
        self.position_tree_readonly(root, bounds, &mut positions);

        LayoutTree { sizes, positions }
    }

    fn measure_tree(
        &mut self,
        widget: &dyn Widget,
        constraints: Constraints,
        sizes: &mut HashMap<u64, Size>,
    ) -> Size {
        let id = self.next_id;
        self.next_id += 1;

        // Measure children first (bottom-up)
        for child in widget.children() {
            self.measure_tree(child.as_ref(), constraints, sizes);
        }

        // Then measure self
        let size = widget.measure(constraints);
        sizes.insert(id, size);
        size
    }

    fn layout_tree(
        &mut self,
        widget: &mut dyn Widget,
        bounds: Rect,
        positions: &mut HashMap<u64, Rect>,
    ) {
        let id = self.next_id;
        self.next_id += 1;

        // Call layout on the widget - this allows it to position its children
        let result = widget.layout(bounds);
        positions.insert(
            id,
            Rect::new(bounds.x, bounds.y, result.size.width, result.size.height),
        );

        // Recursively layout children (they should already be positioned by parent's layout)
        for child in widget.children_mut() {
            // Children get their bounds from the parent's layout
            // We still need to traverse to record positions
            self.collect_child_positions(child.as_mut(), positions);
        }
    }

    fn collect_child_positions(
        &mut self,
        widget: &mut dyn Widget,
        positions: &mut HashMap<u64, Rect>,
    ) {
        let id = self.next_id;
        self.next_id += 1;

        // The widget should already have been laid out by its parent
        // We just record its current bounds
        // Note: In a real implementation, we'd need to track bounds per widget
        // For now, we assume the widget stores its own bounds

        // Get bounds from recent layout (stored in widget)
        // Since we can't easily get this, we'll use a placeholder
        positions.insert(id, Rect::default());

        for child in widget.children_mut() {
            self.collect_child_positions(child.as_mut(), positions);
        }
    }

    fn position_tree_readonly(
        &mut self,
        widget: &dyn Widget,
        bounds: Rect,
        positions: &mut HashMap<u64, Rect>,
    ) {
        let id = self.next_id;
        self.next_id += 1;

        positions.insert(id, bounds);

        // For read-only, we estimate child positions based on measurement
        for child in widget.children() {
            // Give each child the parent bounds (simplified)
            self.position_tree_readonly(child.as_ref(), bounds, positions);
        }
    }

    /// Clear the layout cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get cache statistics.
    #[must_use]
    pub const fn cache_stats(&self) -> (usize, usize) {
        (self.cache.hits(), self.cache.misses())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use presentar_core::widget::{AccessibleRole, LayoutResult};
    use presentar_core::{
        Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Event, TypeId,
    };
    use std::any::Any;
    use std::time::Duration;

    // Test widget for layout testing
    struct TestWidget {
        size: Size,
        children: Vec<Box<dyn Widget>>,
    }

    impl TestWidget {
        fn new(width: f32, height: f32) -> Self {
            Self {
                size: Size::new(width, height),
                children: Vec::new(),
            }
        }

        fn with_child(mut self, child: Self) -> Self {
            self.children.push(Box::new(child));
            self
        }
    }

    impl Brick for TestWidget {
        fn brick_name(&self) -> &'static str {
            "TestWidget"
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

    impl Widget for TestWidget {
        fn type_id(&self) -> TypeId {
            TypeId::of::<Self>()
        }

        fn measure(&self, constraints: Constraints) -> Size {
            constraints.constrain(self.size)
        }

        fn layout(&mut self, bounds: Rect) -> LayoutResult {
            LayoutResult {
                size: bounds.size(),
            }
        }

        fn paint(&self, _canvas: &mut dyn Canvas) {}

        fn event(&mut self, _event: &Event) -> Option<Box<dyn Any + Send>> {
            None
        }

        fn children(&self) -> &[Box<dyn Widget>] {
            &self.children
        }

        fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
            &mut self.children
        }

        fn accessible_role(&self) -> AccessibleRole {
            AccessibleRole::Generic
        }
    }

    #[test]
    fn test_layout_engine_new() {
        let engine = LayoutEngine::new();
        assert_eq!(engine.next_id, 0);
    }

    #[test]
    fn test_layout_tree_default() {
        let tree = LayoutTree::default();
        assert!(tree.sizes.is_empty());
        assert!(tree.positions.is_empty());
        assert_eq!(tree.widget_count(), 0);
    }

    #[test]
    fn test_layout_tree_get_size() {
        let mut tree = LayoutTree::default();
        tree.sizes.insert(0, Size::new(100.0, 50.0));
        assert_eq!(tree.get_size(0), Some(Size::new(100.0, 50.0)));
        assert_eq!(tree.get_size(1), None);
    }

    #[test]
    fn test_layout_tree_get_position() {
        let mut tree = LayoutTree::default();
        tree.positions.insert(0, Rect::new(10.0, 20.0, 100.0, 50.0));
        assert_eq!(
            tree.get_position(0),
            Some(Rect::new(10.0, 20.0, 100.0, 50.0))
        );
        assert_eq!(tree.get_position(1), None);
    }

    #[test]
    fn test_layout_single_widget() {
        let mut engine = LayoutEngine::new();
        let mut widget = TestWidget::new(100.0, 50.0);
        let viewport = Size::new(800.0, 600.0);

        let tree = engine.compute(&mut widget, viewport);

        assert_eq!(tree.widget_count(), 1);
        assert!(tree.get_size(0).is_some());
    }

    #[test]
    fn test_layout_widget_with_children() {
        let mut engine = LayoutEngine::new();
        let mut widget = TestWidget::new(200.0, 100.0)
            .with_child(TestWidget::new(50.0, 50.0))
            .with_child(TestWidget::new(50.0, 50.0));
        let viewport = Size::new(800.0, 600.0);

        let tree = engine.compute(&mut widget, viewport);

        // Root + 2 children = 3 widgets
        assert_eq!(tree.widget_count(), 3);
    }

    #[test]
    fn test_layout_nested_children() {
        let mut engine = LayoutEngine::new();
        let mut widget = TestWidget::new(300.0, 200.0)
            .with_child(TestWidget::new(100.0, 100.0).with_child(TestWidget::new(30.0, 30.0)));
        let viewport = Size::new(800.0, 600.0);

        let tree = engine.compute(&mut widget, viewport);

        // Root + child + grandchild = 3 widgets
        assert_eq!(tree.widget_count(), 3);
    }

    #[test]
    fn test_layout_readonly() {
        let mut engine = LayoutEngine::new();
        let widget = TestWidget::new(100.0, 50.0);
        let viewport = Size::new(800.0, 600.0);

        let tree = engine.compute_readonly(&widget, viewport);

        assert_eq!(tree.widget_count(), 1);
        assert_eq!(tree.get_size(0), Some(Size::new(100.0, 50.0)));
    }

    #[test]
    fn test_layout_cache_clear() {
        let mut engine = LayoutEngine::new();
        engine.clear_cache();
        let (hits, misses) = engine.cache_stats();
        assert_eq!(hits, 0);
        assert_eq!(misses, 0);
    }

    #[test]
    fn test_layout_viewport_constraint() {
        let mut engine = LayoutEngine::new();
        let mut widget = TestWidget::new(1000.0, 1000.0); // Larger than viewport
        let viewport = Size::new(400.0, 300.0);

        let tree = engine.compute(&mut widget, viewport);

        // Size should be constrained to viewport
        let size = tree.get_size(0).unwrap();
        assert!(size.width <= viewport.width);
        assert!(size.height <= viewport.height);
    }

    #[test]
    fn test_layout_position_at_origin() {
        let mut engine = LayoutEngine::new();
        let mut widget = TestWidget::new(100.0, 50.0);
        let viewport = Size::new(800.0, 600.0);

        let tree = engine.compute(&mut widget, viewport);

        let pos = tree.get_position(0).unwrap();
        assert_eq!(pos.x, 0.0);
        assert_eq!(pos.y, 0.0);
    }

    // =========================================================================
    // LayoutTree Tests
    // =========================================================================

    #[test]
    fn test_layout_tree_widget_count() {
        let mut tree = LayoutTree::default();
        assert_eq!(tree.widget_count(), 0);

        tree.positions.insert(0, Rect::default());
        tree.positions.insert(1, Rect::default());
        tree.positions.insert(2, Rect::default());

        assert_eq!(tree.widget_count(), 3);
    }

    #[test]
    fn test_layout_tree_sizes_and_positions() {
        let mut tree = LayoutTree::default();

        tree.sizes.insert(0, Size::new(100.0, 50.0));
        tree.positions.insert(0, Rect::new(10.0, 20.0, 100.0, 50.0));

        tree.sizes.insert(1, Size::new(200.0, 100.0));
        tree.positions
            .insert(1, Rect::new(120.0, 20.0, 200.0, 100.0));

        assert_eq!(tree.sizes.len(), 2);
        assert_eq!(tree.positions.len(), 2);
    }

    #[test]
    fn test_layout_tree_debug() {
        let tree = LayoutTree::default();
        let debug = format!("{:?}", tree);
        assert!(debug.contains("LayoutTree"));
    }

    // =========================================================================
    // LayoutEngine Default Tests
    // =========================================================================

    #[test]
    fn test_layout_engine_default() {
        let engine = LayoutEngine::default();
        assert_eq!(engine.next_id, 0);
    }

    #[test]
    fn test_layout_engine_debug() {
        let engine = LayoutEngine::new();
        let debug = format!("{:?}", engine);
        assert!(debug.contains("LayoutEngine"));
    }

    // =========================================================================
    // Multiple Compute Calls
    // =========================================================================

    #[test]
    fn test_layout_multiple_computes() {
        let mut engine = LayoutEngine::new();
        let viewport = Size::new(800.0, 600.0);

        // First compute
        let mut widget1 = TestWidget::new(100.0, 50.0);
        let tree1 = engine.compute(&mut widget1, viewport);
        assert_eq!(tree1.widget_count(), 1);

        // Second compute (should reset IDs)
        let mut widget2 = TestWidget::new(200.0, 100.0);
        let tree2 = engine.compute(&mut widget2, viewport);
        assert_eq!(tree2.widget_count(), 1);
    }

    #[test]
    fn test_layout_cache_cleared_on_compute() {
        let mut engine = LayoutEngine::new();
        let viewport = Size::new(800.0, 600.0);

        let mut widget = TestWidget::new(100.0, 50.0);
        engine.compute(&mut widget, viewport);

        let (hits, misses) = engine.cache_stats();
        assert_eq!(hits, 0);
        assert_eq!(misses, 0);
    }

    // =========================================================================
    // Viewport Size Variations
    // =========================================================================

    #[test]
    fn test_layout_zero_viewport() {
        let mut engine = LayoutEngine::new();
        let mut widget = TestWidget::new(100.0, 50.0);
        let viewport = Size::new(0.0, 0.0);

        let tree = engine.compute(&mut widget, viewport);
        assert_eq!(tree.widget_count(), 1);
    }

    #[test]
    fn test_layout_very_large_viewport() {
        let mut engine = LayoutEngine::new();
        let mut widget = TestWidget::new(100.0, 50.0);
        let viewport = Size::new(10000.0, 10000.0);

        let tree = engine.compute(&mut widget, viewport);
        let size = tree.get_size(0).unwrap();

        // Widget should keep its intrinsic size within loose constraints
        assert!(size.width <= 100.0);
        assert!(size.height <= 50.0);
    }

    #[test]
    fn test_layout_square_viewport() {
        let mut engine = LayoutEngine::new();
        let mut widget = TestWidget::new(100.0, 100.0);
        let viewport = Size::new(500.0, 500.0);

        let tree = engine.compute(&mut widget, viewport);
        assert_eq!(tree.widget_count(), 1);
    }

    // =========================================================================
    // Complex Widget Trees
    // =========================================================================

    #[test]
    fn test_layout_deeply_nested() {
        let mut engine = LayoutEngine::new();
        let viewport = Size::new(800.0, 600.0);

        let mut widget =
            TestWidget::new(100.0, 100.0).with_child(TestWidget::new(80.0, 80.0).with_child(
                TestWidget::new(60.0, 60.0).with_child(
                    TestWidget::new(40.0, 40.0).with_child(TestWidget::new(20.0, 20.0)),
                ),
            ));

        let tree = engine.compute(&mut widget, viewport);
        assert_eq!(tree.widget_count(), 5); // 5 levels of nesting
    }

    #[test]
    fn test_layout_wide_tree() {
        let mut engine = LayoutEngine::new();
        let viewport = Size::new(800.0, 600.0);

        let mut widget = TestWidget::new(200.0, 100.0)
            .with_child(TestWidget::new(30.0, 30.0))
            .with_child(TestWidget::new(30.0, 30.0))
            .with_child(TestWidget::new(30.0, 30.0))
            .with_child(TestWidget::new(30.0, 30.0))
            .with_child(TestWidget::new(30.0, 30.0));

        let tree = engine.compute(&mut widget, viewport);
        assert_eq!(tree.widget_count(), 6); // Root + 5 children
    }

    #[test]
    fn test_layout_mixed_tree() {
        let mut engine = LayoutEngine::new();
        let viewport = Size::new(800.0, 600.0);

        let mut widget = TestWidget::new(300.0, 200.0)
            .with_child(
                TestWidget::new(100.0, 100.0)
                    .with_child(TestWidget::new(30.0, 30.0))
                    .with_child(TestWidget::new(30.0, 30.0)),
            )
            .with_child(TestWidget::new(100.0, 100.0))
            .with_child(TestWidget::new(100.0, 100.0).with_child(TestWidget::new(30.0, 30.0)));

        let tree = engine.compute(&mut widget, viewport);
        // Root + 3 children + 2 grandchildren + 1 grandchild = 7
        assert_eq!(tree.widget_count(), 7);
    }

    // =========================================================================
    // Read-only Compute Tests
    // =========================================================================

    #[test]
    fn test_layout_readonly_with_children() {
        let mut engine = LayoutEngine::new();
        let viewport = Size::new(800.0, 600.0);

        let widget = TestWidget::new(200.0, 100.0)
            .with_child(TestWidget::new(50.0, 50.0))
            .with_child(TestWidget::new(50.0, 50.0));

        let tree = engine.compute_readonly(&widget, viewport);
        assert_eq!(tree.widget_count(), 3);
    }

    #[test]
    fn test_layout_readonly_nested() {
        let mut engine = LayoutEngine::new();
        let viewport = Size::new(800.0, 600.0);

        let widget = TestWidget::new(100.0, 100.0)
            .with_child(TestWidget::new(80.0, 80.0).with_child(TestWidget::new(60.0, 60.0)));

        let tree = engine.compute_readonly(&widget, viewport);
        assert_eq!(tree.widget_count(), 3);
    }

    // =========================================================================
    // Cache Stats Tests
    // =========================================================================

    #[test]
    fn test_cache_stats_initial() {
        let engine = LayoutEngine::new();
        let (hits, misses) = engine.cache_stats();
        assert_eq!(hits, 0);
        assert_eq!(misses, 0);
    }

    #[test]
    fn test_cache_stats_after_clear() {
        let mut engine = LayoutEngine::new();
        engine.clear_cache();
        let (hits, misses) = engine.cache_stats();
        assert_eq!(hits, 0);
        assert_eq!(misses, 0);
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    #[test]
    fn test_layout_widget_larger_than_viewport() {
        let mut engine = LayoutEngine::new();
        let mut widget = TestWidget::new(2000.0, 1500.0);
        let viewport = Size::new(800.0, 600.0);

        let tree = engine.compute(&mut widget, viewport);
        let size = tree.get_size(0).unwrap();

        // Should be constrained
        assert!(size.width <= viewport.width);
        assert!(size.height <= viewport.height);
    }

    #[test]
    fn test_layout_widget_fractional_size() {
        let mut engine = LayoutEngine::new();
        let mut widget = TestWidget::new(100.5, 50.25);
        let viewport = Size::new(800.0, 600.0);

        let tree = engine.compute(&mut widget, viewport);
        assert_eq!(tree.widget_count(), 1);
    }
}
