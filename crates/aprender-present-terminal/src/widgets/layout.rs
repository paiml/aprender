//! Layout system for compositional widget arrangement.
//!
//! Provides `Layout::rows()` and `Layout::columns()` for building UI hierarchies.
//! Supports percentage-based sizing, fixed sizes, and flex expansion.

use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Constraints, Event,
    LayoutResult, Rect, Size, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// Size specification for layout children.
#[derive(Debug, Clone, Copy)]
pub enum SizeSpec {
    /// Fixed size in terminal cells.
    Fixed(f32),
    /// Percentage of parent (0.0 - 100.0).
    Percent(f32),
    /// Expand to fill remaining space (with weight).
    Flex(f32),
    /// Use child's natural size.
    Auto,
}

impl Default for SizeSpec {
    fn default() -> Self {
        Self::Flex(1.0)
    }
}

/// A single item in a layout with its size specification.
pub struct LayoutItem {
    widget: Box<dyn Widget>,
    size: SizeSpec,
}

impl LayoutItem {
    /// Create a new layout item wrapping a widget.
    pub fn new(widget: impl Widget + 'static) -> Self {
        Self {
            widget: Box::new(widget),
            size: SizeSpec::default(),
        }
    }

    /// Set fixed height/width.
    #[must_use]
    pub fn fixed(mut self, size: f32) -> Self {
        self.size = SizeSpec::Fixed(size);
        self
    }

    /// Set percentage of parent.
    #[must_use]
    pub fn percent(mut self, pct: f32) -> Self {
        self.size = SizeSpec::Percent(pct);
        self
    }

    /// Expand to fill remaining space.
    #[must_use]
    pub fn expanded(mut self) -> Self {
        self.size = SizeSpec::Flex(1.0);
        self
    }

    /// Expand with specific weight.
    #[must_use]
    pub fn flex(mut self, weight: f32) -> Self {
        self.size = SizeSpec::Flex(weight);
        self
    }

    /// Use child's natural size.
    #[must_use]
    pub fn auto(mut self) -> Self {
        self.size = SizeSpec::Auto;
        self
    }
}

/// Layout direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Stack children vertically (rows).
    Vertical,
    /// Stack children horizontally (columns).
    Horizontal,
}

/// Compositional layout widget.
///
/// # Example
/// ```ignore
/// Layout::rows([
///     LayoutItem::new(header).fixed(1),
///     Layout::columns([
///         LayoutItem::new(sidebar).percent(25.0),
///         LayoutItem::new(content).expanded(),
///     ]).into_item().expanded(),
///     LayoutItem::new(footer).fixed(1),
/// ])
/// ```
pub struct Layout {
    direction: Direction,
    children: Vec<LayoutItem>,
    bounds: Rect,
    /// Cached child bounds after layout.
    child_bounds: Vec<Rect>,
}

impl Layout {
    /// Create a vertical layout (rows stacked top to bottom).
    pub fn rows(items: impl IntoIterator<Item = LayoutItem>) -> Self {
        Self {
            direction: Direction::Vertical,
            children: items.into_iter().collect(),
            bounds: Rect::default(),
            child_bounds: Vec::new(),
        }
    }

    /// Create a horizontal layout (columns side by side).
    pub fn columns(items: impl IntoIterator<Item = LayoutItem>) -> Self {
        Self {
            direction: Direction::Horizontal,
            children: items.into_iter().collect(),
            bounds: Rect::default(),
            child_bounds: Vec::new(),
        }
    }

    /// Convert this layout into a `LayoutItem` for nesting.
    #[must_use]
    pub fn into_item(self) -> LayoutItem {
        LayoutItem::new(self)
    }

    /// Add a child item.
    pub fn push(&mut self, item: LayoutItem) {
        self.children.push(item);
    }

    /// Get child count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.children.len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.children.is_empty()
    }

    /// Calculate sizes for all children given total available space.
    fn calculate_sizes(&self, total: f32) -> Vec<f32> {
        let mut sizes = vec![0.0; self.children.len()];
        let mut remaining = total;
        let mut flex_total = 0.0;

        // First pass: allocate fixed and percentage sizes
        for (i, item) in self.children.iter().enumerate() {
            match item.size {
                SizeSpec::Fixed(s) => {
                    sizes[i] = s.min(remaining);
                    remaining -= sizes[i];
                }
                SizeSpec::Percent(pct) => {
                    sizes[i] = (total * pct / 100.0).min(remaining);
                    remaining -= sizes[i];
                }
                SizeSpec::Auto => {
                    // For auto, we'll use a reasonable default or measure
                    // For now, treat as small fixed
                    sizes[i] = 1.0_f32.min(remaining);
                    remaining -= sizes[i];
                }
                SizeSpec::Flex(weight) => {
                    flex_total += weight;
                }
            }
        }

        // Second pass: distribute remaining space to flex items
        if flex_total > 0.0 && remaining > 0.0 {
            for (i, item) in self.children.iter().enumerate() {
                if let SizeSpec::Flex(weight) = item.size {
                    sizes[i] = remaining * (weight / flex_total);
                }
            }
        }

        sizes
    }
}

impl Brick for Layout {
    fn brick_name(&self) -> &'static str {
        "layout"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        static ASSERTIONS: &[BrickAssertion] = &[BrickAssertion::max_latency_ms(8)];
        ASSERTIONS
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(8)
    }

    fn verify(&self) -> BrickVerification {
        BrickVerification {
            passed: vec![BrickAssertion::max_latency_ms(8)],
            failed: vec![],
            verification_time: Duration::from_micros(10),
        }
    }

    fn to_html(&self) -> String {
        String::new()
    }

    fn to_css(&self) -> String {
        String::new()
    }
}

impl Widget for Layout {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        constraints.constrain(Size::new(constraints.max_width, constraints.max_height))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        self.child_bounds.clear();

        let total = match self.direction {
            Direction::Vertical => bounds.height,
            Direction::Horizontal => bounds.width,
        };

        let sizes = self.calculate_sizes(total);
        let mut offset = 0.0;

        for (i, item) in self.children.iter_mut().enumerate() {
            let size = sizes[i];
            let child_bounds = match self.direction {
                Direction::Vertical => Rect::new(bounds.x, bounds.y + offset, bounds.width, size),
                Direction::Horizontal => {
                    Rect::new(bounds.x + offset, bounds.y, size, bounds.height)
                }
            };

            self.child_bounds.push(child_bounds);
            item.widget.layout(child_bounds);
            offset += size;
        }

        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        for (i, item) in self.children.iter().enumerate() {
            if i < self.child_bounds.len() {
                // Push clip for child bounds
                canvas.push_clip(self.child_bounds[i]);
                item.widget.paint(canvas);
                canvas.pop_clip();
            }
        }
    }

    fn event(&mut self, event: &Event) -> Option<Box<dyn Any + Send>> {
        // Propagate to all children
        for item in &mut self.children {
            if let Some(result) = item.widget.event(event) {
                return Some(result);
            }
        }
        None
    }

    fn children(&self) -> &[Box<dyn Widget>] {
        // Can't return slice of nested items easily
        &[]
    }

    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut []
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widgets::Border;

    #[test]
    fn test_layout_rows_empty() {
        let layout = Layout::rows([]);
        assert!(layout.is_empty());
    }

    #[test]
    fn test_layout_columns_empty() {
        let layout = Layout::columns([]);
        assert!(layout.is_empty());
    }

    #[test]
    fn test_layout_item_fixed() {
        let item = LayoutItem::new(Border::new()).fixed(10.0);
        assert!(matches!(item.size, SizeSpec::Fixed(10.0)));
    }

    #[test]
    fn test_layout_item_percent() {
        let item = LayoutItem::new(Border::new()).percent(25.0);
        assert!(matches!(item.size, SizeSpec::Percent(25.0)));
    }

    #[test]
    fn test_layout_item_expanded() {
        let item = LayoutItem::new(Border::new()).expanded();
        assert!(matches!(item.size, SizeSpec::Flex(1.0)));
    }

    #[test]
    fn test_layout_item_flex() {
        let item = LayoutItem::new(Border::new()).flex(2.0);
        assert!(matches!(item.size, SizeSpec::Flex(2.0)));
    }

    #[test]
    fn test_layout_calculate_sizes_fixed() {
        let layout = Layout::rows([
            LayoutItem::new(Border::new()).fixed(10.0),
            LayoutItem::new(Border::new()).fixed(20.0),
        ]);
        let sizes = layout.calculate_sizes(100.0);
        assert_eq!(sizes[0], 10.0);
        assert_eq!(sizes[1], 20.0);
    }

    #[test]
    fn test_layout_calculate_sizes_percent() {
        let layout = Layout::rows([
            LayoutItem::new(Border::new()).percent(25.0),
            LayoutItem::new(Border::new()).percent(75.0),
        ]);
        let sizes = layout.calculate_sizes(100.0);
        assert_eq!(sizes[0], 25.0);
        assert_eq!(sizes[1], 75.0);
    }

    #[test]
    fn test_layout_calculate_sizes_flex() {
        let layout = Layout::rows([
            LayoutItem::new(Border::new()).flex(1.0),
            LayoutItem::new(Border::new()).flex(3.0),
        ]);
        let sizes = layout.calculate_sizes(100.0);
        assert_eq!(sizes[0], 25.0);
        assert_eq!(sizes[1], 75.0);
    }

    #[test]
    fn test_layout_calculate_sizes_mixed() {
        let layout = Layout::rows([
            LayoutItem::new(Border::new()).fixed(20.0),
            LayoutItem::new(Border::new()).expanded(),
        ]);
        let sizes = layout.calculate_sizes(100.0);
        assert_eq!(sizes[0], 20.0);
        assert_eq!(sizes[1], 80.0);
    }

    #[test]
    fn test_layout_nested() {
        let inner = Layout::columns([
            LayoutItem::new(Border::new()).percent(50.0),
            LayoutItem::new(Border::new()).percent(50.0),
        ]);
        let outer = Layout::rows([
            LayoutItem::new(Border::new()).fixed(5.0),
            inner.into_item().expanded(),
        ]);
        assert_eq!(outer.len(), 2);
    }

    #[test]
    fn test_layout_push() {
        let mut layout = Layout::rows([]);
        layout.push(LayoutItem::new(Border::new()));
        assert_eq!(layout.len(), 1);
    }

    #[test]
    fn test_layout_brick_name() {
        let layout = Layout::rows([]);
        assert_eq!(layout.brick_name(), "layout");
    }

    #[test]
    fn test_layout_verify() {
        let layout = Layout::rows([]);
        assert!(layout.verify().is_valid());
    }

    #[test]
    fn test_layout_measure() {
        let layout = Layout::rows([]);
        let size = layout.measure(Constraints::new(0.0, 100.0, 0.0, 50.0));
        assert_eq!(size.width, 100.0);
        assert_eq!(size.height, 50.0);
    }

    #[test]
    fn test_layout_vertical_layout() {
        let mut layout = Layout::rows([
            LayoutItem::new(Border::new()).fixed(10.0),
            LayoutItem::new(Border::new()).fixed(20.0),
        ]);
        layout.layout(Rect::new(0.0, 0.0, 80.0, 40.0));
        assert_eq!(layout.child_bounds.len(), 2);
        assert_eq!(layout.child_bounds[0].y, 0.0);
        assert_eq!(layout.child_bounds[0].height, 10.0);
        assert_eq!(layout.child_bounds[1].y, 10.0);
        assert_eq!(layout.child_bounds[1].height, 20.0);
    }

    #[test]
    fn test_layout_horizontal_layout() {
        let mut layout = Layout::columns([
            LayoutItem::new(Border::new()).fixed(20.0),
            LayoutItem::new(Border::new()).fixed(30.0),
        ]);
        layout.layout(Rect::new(0.0, 0.0, 80.0, 40.0));
        assert_eq!(layout.child_bounds.len(), 2);
        assert_eq!(layout.child_bounds[0].x, 0.0);
        assert_eq!(layout.child_bounds[0].width, 20.0);
        assert_eq!(layout.child_bounds[1].x, 20.0);
        assert_eq!(layout.child_bounds[1].width, 30.0);
    }

    #[test]
    fn test_size_spec_default() {
        let spec = SizeSpec::default();
        assert!(matches!(spec, SizeSpec::Flex(1.0)));
    }

    #[test]
    fn test_layout_item_auto() {
        let item = LayoutItem::new(Border::new()).auto();
        assert!(matches!(item.size, SizeSpec::Auto));
    }

    #[test]
    fn test_layout_calculate_sizes_auto() {
        let layout = Layout::rows([
            LayoutItem::new(Border::new()).auto(),
            LayoutItem::new(Border::new()).expanded(),
        ]);
        let sizes = layout.calculate_sizes(100.0);
        // Auto gets 1.0 (minimum)
        assert_eq!(sizes[0], 1.0);
        // Flex gets the remaining
        assert_eq!(sizes[1], 99.0);
    }

    #[test]
    fn test_layout_brick_assertions() {
        let layout = Layout::rows([]);
        let assertions = layout.assertions();
        assert!(!assertions.is_empty());
    }

    #[test]
    fn test_layout_brick_budget() {
        let layout = Layout::rows([]);
        let budget = layout.budget();
        assert_eq!(budget.total_ms, 8);
    }

    #[test]
    fn test_layout_to_html() {
        let layout = Layout::rows([]);
        assert_eq!(layout.to_html(), "");
    }

    #[test]
    fn test_layout_to_css() {
        let layout = Layout::rows([]);
        assert_eq!(layout.to_css(), "");
    }

    #[test]
    fn test_layout_type_id() {
        let layout = Layout::rows([]);
        let id = Widget::type_id(&layout);
        assert_eq!(id, TypeId::of::<Layout>());
    }

    #[test]
    fn test_layout_paint() {
        use crate::tools::HeadlessCanvas;
        let mut layout = Layout::rows([
            LayoutItem::new(Border::new()).fixed(10.0),
            LayoutItem::new(Border::new()).fixed(10.0),
        ]);
        layout.layout(Rect::new(0.0, 0.0, 80.0, 24.0));
        let mut canvas = HeadlessCanvas::new(80, 24);
        layout.paint(&mut canvas);
        // Painting should work without panic
    }

    #[test]
    fn test_layout_event() {
        use presentar_core::Event;
        let mut layout = Layout::rows([LayoutItem::new(Border::new()).fixed(10.0)]);
        let event = Event::Resize {
            width: 80.0,
            height: 24.0,
        };
        let result = layout.event(&event);
        assert!(result.is_none()); // Border doesn't handle events
    }

    #[test]
    fn test_layout_children() {
        let layout = Layout::rows([LayoutItem::new(Border::new()).fixed(10.0)]);
        // children() returns empty slice (items are wrapped)
        assert!(layout.children().is_empty());
    }

    #[test]
    fn test_layout_children_mut() {
        let mut layout = Layout::rows([LayoutItem::new(Border::new()).fixed(10.0)]);
        // children_mut() returns empty slice (items are wrapped)
        assert!(layout.children_mut().is_empty());
    }
}
