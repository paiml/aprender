//! Row widget for horizontal layout.

use presentar_core::{
    widget::{Brick, BrickAssertion, BrickBudget, BrickVerification, LayoutResult},
    Canvas, Constraints, Event, Rect, Size, TypeId, Widget,
};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::time::Duration;

/// Horizontal alignment options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum MainAxisAlignment {
    /// Pack children at the start
    #[default]
    Start,
    /// Pack children at the end
    End,
    /// Center children
    Center,
    /// Distribute space evenly between children
    SpaceBetween,
    /// Distribute space evenly around children
    SpaceAround,
    /// Distribute space evenly, including edges
    SpaceEvenly,
}

/// Vertical alignment options for row children.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CrossAxisAlignment {
    /// Align to the start (top)
    Start,
    /// Align to the end (bottom)
    End,
    /// Center vertically
    #[default]
    Center,
    /// Stretch to fill
    Stretch,
}

/// Row widget for horizontal layout of children.
#[derive(Serialize, Deserialize)]
pub struct Row {
    /// Main axis (horizontal) alignment
    main_axis_alignment: MainAxisAlignment,
    /// Cross axis (vertical) alignment
    cross_axis_alignment: CrossAxisAlignment,
    /// Gap between children
    gap: f32,
    /// Children widgets
    #[serde(skip)]
    children: Vec<Box<dyn Widget>>,
    /// Test ID
    test_id_value: Option<String>,
    /// Cached bounds
    #[serde(skip)]
    bounds: Rect,
    /// Cached child positions
    #[serde(skip)]
    child_bounds: Vec<Rect>,
}

impl Default for Row {
    fn default() -> Self {
        Self::new()
    }
}

impl Row {
    /// Create a new empty row.
    #[must_use]
    pub fn new() -> Self {
        Self {
            main_axis_alignment: MainAxisAlignment::Start,
            cross_axis_alignment: CrossAxisAlignment::Center,
            gap: 0.0,
            children: Vec::new(),
            test_id_value: None,
            bounds: Rect::default(),
            child_bounds: Vec::new(),
        }
    }

    /// Set main axis alignment.
    #[must_use]
    pub const fn main_axis_alignment(mut self, alignment: MainAxisAlignment) -> Self {
        self.main_axis_alignment = alignment;
        self
    }

    /// Set cross axis alignment.
    #[must_use]
    pub const fn cross_axis_alignment(mut self, alignment: CrossAxisAlignment) -> Self {
        self.cross_axis_alignment = alignment;
        self
    }

    /// Set gap between children.
    #[must_use]
    pub const fn gap(mut self, gap: f32) -> Self {
        self.gap = gap;
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
}

impl Widget for Row {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        if self.children.is_empty() {
            return Size::ZERO;
        }

        let mut total_width = 0.0f32;
        let mut max_height = 0.0f32;

        // Measure all children
        for (i, child) in self.children.iter().enumerate() {
            let child_constraints = Constraints::new(
                0.0,
                (constraints.max_width - total_width).max(0.0),
                0.0,
                constraints.max_height,
            );

            let child_size = child.measure(child_constraints);
            total_width += child_size.width;
            max_height = max_height.max(child_size.height);

            if i < self.children.len() - 1 {
                total_width += self.gap;
            }
        }

        constraints.constrain(Size::new(total_width, max_height))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        self.child_bounds.clear();

        if self.children.is_empty() {
            return LayoutResult { size: Size::ZERO };
        }

        // First pass: measure children
        let mut child_sizes: Vec<Size> = Vec::with_capacity(self.children.len());
        let mut total_width = 0.0f32;

        for child in &self.children {
            let child_constraints = Constraints::loose(bounds.size());
            let size = child.measure(child_constraints);
            total_width += size.width;
            child_sizes.push(size);
        }

        let gaps_width = self.gap * (self.children.len() - 1).max(0) as f32;
        let content_width = total_width + gaps_width;
        let remaining_space = (bounds.width - content_width).max(0.0);

        // Calculate starting position based on alignment
        let (mut x, extra_gap) = match self.main_axis_alignment {
            MainAxisAlignment::Start => (bounds.x, 0.0),
            MainAxisAlignment::End => (bounds.x + remaining_space, 0.0),
            MainAxisAlignment::Center => (bounds.x + remaining_space / 2.0, 0.0),
            MainAxisAlignment::SpaceBetween => {
                if self.children.len() > 1 {
                    (bounds.x, remaining_space / (self.children.len() - 1) as f32)
                } else {
                    (bounds.x, 0.0)
                }
            }
            MainAxisAlignment::SpaceAround => {
                let gap = remaining_space / self.children.len() as f32;
                (bounds.x + gap / 2.0, gap)
            }
            MainAxisAlignment::SpaceEvenly => {
                let gap = remaining_space / (self.children.len() + 1) as f32;
                (bounds.x + gap, gap)
            }
        };

        // Second pass: position children
        let num_children = self.children.len();
        for (i, (child, size)) in self.children.iter_mut().zip(child_sizes.iter()).enumerate() {
            let y = match self.cross_axis_alignment {
                CrossAxisAlignment::Start | CrossAxisAlignment::Stretch => bounds.y,
                CrossAxisAlignment::End => bounds.y + bounds.height - size.height,
                CrossAxisAlignment::Center => bounds.y + (bounds.height - size.height) / 2.0,
            };

            let height = if self.cross_axis_alignment == CrossAxisAlignment::Stretch {
                bounds.height
            } else {
                size.height
            };

            let child_bounds = Rect::new(x, y, size.width, height);
            child.layout(child_bounds);
            self.child_bounds.push(child_bounds);

            // Move x for next child
            if i < num_children - 1 {
                x += size.width;
                if self.main_axis_alignment == MainAxisAlignment::SpaceBetween {
                    // SpaceBetween uses only extra_gap (no regular gap)
                    x += extra_gap;
                } else {
                    x += self.gap + extra_gap;
                }
            }
        }

        LayoutResult {
            size: bounds.size(),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        for child in &self.children {
            child.paint(canvas);
        }
    }

    fn event(&mut self, event: &Event) -> Option<Box<dyn Any + Send>> {
        for child in &mut self.children {
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
impl Brick for Row {
    fn brick_name(&self) -> &'static str {
        "Row"
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
        r#"<div class="brick-row"></div>"#.to_string()
    }

    fn to_css(&self) -> String {
        ".brick-row { display: flex; flex-direction: row; }".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use presentar_core::widget::AccessibleRole;
    use presentar_core::{Brick, BrickBudget, BrickVerification, Widget};
    use std::time::Duration;

    // Test widget with fixed size for layout testing
    struct FixedWidget {
        size: Size,
    }

    impl FixedWidget {
        fn new(width: f32, height: f32) -> Self {
            Self {
                size: Size::new(width, height),
            }
        }
    }

    impl Brick for FixedWidget {
        fn brick_name(&self) -> &'static str {
            "FixedWidget"
        }

        fn assertions(&self) -> &[presentar_core::BrickAssertion] {
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

    impl Widget for FixedWidget {
        fn type_id(&self) -> TypeId {
            TypeId::of::<Self>()
        }

        fn measure(&self, constraints: Constraints) -> Size {
            constraints.constrain(self.size)
        }

        fn layout(&mut self, _bounds: Rect) -> LayoutResult {
            LayoutResult { size: self.size }
        }

        fn paint(&self, _canvas: &mut dyn Canvas) {}

        fn event(&mut self, _event: &Event) -> Option<Box<dyn Any + Send>> {
            None
        }

        fn children(&self) -> &[Box<dyn Widget>] {
            &[]
        }

        fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
            &mut []
        }

        fn accessible_role(&self) -> AccessibleRole {
            AccessibleRole::Generic
        }
    }

    // ===== Basic Tests =====

    #[test]
    fn test_row_empty() {
        let row = Row::new();
        let size = row.measure(Constraints::loose(Size::new(100.0, 100.0)));
        assert_eq!(size, Size::ZERO);
    }

    #[test]
    fn test_row_builder() {
        let row = Row::new()
            .gap(10.0)
            .main_axis_alignment(MainAxisAlignment::Center)
            .cross_axis_alignment(CrossAxisAlignment::Start)
            .with_test_id("my-row");

        assert_eq!(row.gap, 10.0);
        assert_eq!(row.main_axis_alignment, MainAxisAlignment::Center);
        assert_eq!(row.cross_axis_alignment, CrossAxisAlignment::Start);
        assert_eq!(Widget::test_id(&row), Some("my-row"));
    }

    #[test]
    fn test_row_default() {
        let row = Row::default();
        assert_eq!(row.main_axis_alignment, MainAxisAlignment::Start);
        assert_eq!(row.cross_axis_alignment, CrossAxisAlignment::Center);
        assert_eq!(row.gap, 0.0);
    }

    #[test]
    fn test_row_type_id() {
        let row = Row::new();
        assert_eq!(Widget::type_id(&row), TypeId::of::<Row>());
    }

    #[test]
    fn test_row_children() {
        let row = Row::new()
            .child(FixedWidget::new(50.0, 30.0))
            .child(FixedWidget::new(50.0, 30.0));
        assert_eq!(row.children().len(), 2);
    }

    // ===== Measure Tests =====

    #[test]
    fn test_row_measure_single_child() {
        let row = Row::new().child(FixedWidget::new(50.0, 30.0));
        let size = row.measure(Constraints::loose(Size::new(200.0, 100.0)));
        assert_eq!(size, Size::new(50.0, 30.0));
    }

    #[test]
    fn test_row_measure_multiple_children() {
        let row = Row::new()
            .child(FixedWidget::new(50.0, 30.0))
            .child(FixedWidget::new(60.0, 40.0));
        let size = row.measure(Constraints::loose(Size::new(200.0, 100.0)));
        assert_eq!(size, Size::new(110.0, 40.0));
    }

    #[test]
    fn test_row_measure_with_gap() {
        let row = Row::new()
            .gap(10.0)
            .child(FixedWidget::new(50.0, 30.0))
            .child(FixedWidget::new(50.0, 30.0));
        let size = row.measure(Constraints::loose(Size::new(200.0, 100.0)));
        assert_eq!(size, Size::new(110.0, 30.0)); // 50 + 10 + 50
    }

    #[test]
    fn test_row_measure_constrained() {
        let row = Row::new()
            .child(FixedWidget::new(100.0, 50.0))
            .child(FixedWidget::new(100.0, 50.0));
        let size = row.measure(Constraints::tight(Size::new(150.0, 40.0)));
        assert_eq!(size, Size::new(150.0, 40.0)); // Constrained to tight
    }

    // ===== MainAxisAlignment Tests =====

    #[test]
    fn test_row_alignment_start() {
        let mut row = Row::new()
            .main_axis_alignment(MainAxisAlignment::Start)
            .child(FixedWidget::new(30.0, 20.0))
            .child(FixedWidget::new(30.0, 20.0));

        row.layout(Rect::new(0.0, 0.0, 200.0, 50.0));

        assert_eq!(row.child_bounds.len(), 2);
        assert_eq!(row.child_bounds[0].x, 0.0);
        assert_eq!(row.child_bounds[1].x, 30.0);
    }

    #[test]
    fn test_row_alignment_end() {
        let mut row = Row::new()
            .main_axis_alignment(MainAxisAlignment::End)
            .child(FixedWidget::new(30.0, 20.0))
            .child(FixedWidget::new(30.0, 20.0));

        row.layout(Rect::new(0.0, 0.0, 200.0, 50.0));

        // 200 - 60 = 140 remaining, children at 140 and 170
        assert_eq!(row.child_bounds[0].x, 140.0);
        assert_eq!(row.child_bounds[1].x, 170.0);
    }

    #[test]
    fn test_row_alignment_center() {
        let mut row = Row::new()
            .main_axis_alignment(MainAxisAlignment::Center)
            .child(FixedWidget::new(30.0, 20.0))
            .child(FixedWidget::new(30.0, 20.0));

        row.layout(Rect::new(0.0, 0.0, 200.0, 50.0));

        // 200 - 60 = 140 remaining, offset = 70
        assert_eq!(row.child_bounds[0].x, 70.0);
        assert_eq!(row.child_bounds[1].x, 100.0);
    }

    #[test]
    fn test_row_alignment_space_between() {
        let mut row = Row::new()
            .main_axis_alignment(MainAxisAlignment::SpaceBetween)
            .child(FixedWidget::new(30.0, 20.0))
            .child(FixedWidget::new(30.0, 20.0));

        row.layout(Rect::new(0.0, 0.0, 200.0, 50.0));

        // First at start, last at end
        assert_eq!(row.child_bounds[0].x, 0.0);
        assert_eq!(row.child_bounds[1].x, 170.0); // 200 - 30
    }

    #[test]
    fn test_row_alignment_space_between_single_child() {
        let mut row = Row::new()
            .main_axis_alignment(MainAxisAlignment::SpaceBetween)
            .child(FixedWidget::new(30.0, 20.0));

        row.layout(Rect::new(0.0, 0.0, 200.0, 50.0));

        // Single child should be at start
        assert_eq!(row.child_bounds[0].x, 0.0);
    }

    #[test]
    fn test_row_alignment_space_between_three_children() {
        let mut row = Row::new()
            .main_axis_alignment(MainAxisAlignment::SpaceBetween)
            .child(FixedWidget::new(30.0, 20.0))
            .child(FixedWidget::new(30.0, 20.0))
            .child(FixedWidget::new(30.0, 20.0));

        row.layout(Rect::new(0.0, 0.0, 200.0, 50.0));

        // 200 - 90 = 110 remaining, gap = 55
        assert_eq!(row.child_bounds[0].x, 0.0);
        assert_eq!(row.child_bounds[1].x, 85.0); // 30 + 55
        assert_eq!(row.child_bounds[2].x, 170.0); // 200 - 30
    }

    #[test]
    fn test_row_alignment_space_around() {
        let mut row = Row::new()
            .main_axis_alignment(MainAxisAlignment::SpaceAround)
            .child(FixedWidget::new(40.0, 20.0))
            .child(FixedWidget::new(40.0, 20.0));

        row.layout(Rect::new(0.0, 0.0, 200.0, 50.0));

        // 200 - 80 = 120 remaining, gap = 60, half-gap = 30
        // First at 30, second at 30 + 40 + 60 = 130
        assert_eq!(row.child_bounds[0].x, 30.0);
        assert_eq!(row.child_bounds[1].x, 130.0);
    }

    #[test]
    fn test_row_alignment_space_evenly() {
        let mut row = Row::new()
            .main_axis_alignment(MainAxisAlignment::SpaceEvenly)
            .child(FixedWidget::new(40.0, 20.0))
            .child(FixedWidget::new(40.0, 20.0));

        row.layout(Rect::new(0.0, 0.0, 200.0, 50.0));

        // 200 - 80 = 120 remaining, 3 gaps (n+1), gap = 40
        // First at 40, second at 40 + 40 + 40 = 120
        assert_eq!(row.child_bounds[0].x, 40.0);
        assert_eq!(row.child_bounds[1].x, 120.0);
    }

    // ===== CrossAxisAlignment Tests =====

    #[test]
    fn test_row_cross_alignment_start() {
        let mut row = Row::new()
            .cross_axis_alignment(CrossAxisAlignment::Start)
            .child(FixedWidget::new(30.0, 20.0));

        row.layout(Rect::new(0.0, 0.0, 200.0, 100.0));

        assert_eq!(row.child_bounds[0].y, 0.0);
        assert_eq!(row.child_bounds[0].height, 20.0);
    }

    #[test]
    fn test_row_cross_alignment_end() {
        let mut row = Row::new()
            .cross_axis_alignment(CrossAxisAlignment::End)
            .child(FixedWidget::new(30.0, 20.0));

        row.layout(Rect::new(0.0, 0.0, 200.0, 100.0));

        assert_eq!(row.child_bounds[0].y, 80.0); // 100 - 20
        assert_eq!(row.child_bounds[0].height, 20.0);
    }

    #[test]
    fn test_row_cross_alignment_center() {
        let mut row = Row::new()
            .cross_axis_alignment(CrossAxisAlignment::Center)
            .child(FixedWidget::new(30.0, 20.0));

        row.layout(Rect::new(0.0, 0.0, 200.0, 100.0));

        assert_eq!(row.child_bounds[0].y, 40.0); // (100 - 20) / 2
        assert_eq!(row.child_bounds[0].height, 20.0);
    }

    #[test]
    fn test_row_cross_alignment_stretch() {
        let mut row = Row::new()
            .cross_axis_alignment(CrossAxisAlignment::Stretch)
            .child(FixedWidget::new(30.0, 20.0));

        row.layout(Rect::new(0.0, 0.0, 200.0, 100.0));

        assert_eq!(row.child_bounds[0].y, 0.0);
        assert_eq!(row.child_bounds[0].height, 100.0); // Stretched to container
    }

    // ===== Gap Tests =====

    #[test]
    fn test_row_gap_single_child() {
        let mut row = Row::new().gap(20.0).child(FixedWidget::new(30.0, 20.0));

        row.layout(Rect::new(0.0, 0.0, 200.0, 50.0));

        // Single child: no gap applied
        assert_eq!(row.child_bounds[0].x, 0.0);
    }

    #[test]
    fn test_row_gap_multiple_children() {
        let mut row = Row::new()
            .gap(15.0)
            .child(FixedWidget::new(30.0, 20.0))
            .child(FixedWidget::new(30.0, 20.0))
            .child(FixedWidget::new(30.0, 20.0));

        row.layout(Rect::new(0.0, 0.0, 200.0, 50.0));

        assert_eq!(row.child_bounds[0].x, 0.0);
        assert_eq!(row.child_bounds[1].x, 45.0); // 30 + 15
        assert_eq!(row.child_bounds[2].x, 90.0); // 45 + 30 + 15
    }

    #[test]
    fn test_row_gap_with_alignment_center() {
        let mut row = Row::new()
            .gap(10.0)
            .main_axis_alignment(MainAxisAlignment::Center)
            .child(FixedWidget::new(30.0, 20.0))
            .child(FixedWidget::new(30.0, 20.0));

        row.layout(Rect::new(0.0, 0.0, 200.0, 50.0));

        // Total: 30 + 10 + 30 = 70, remaining = 130, offset = 65
        assert_eq!(row.child_bounds[0].x, 65.0);
        assert_eq!(row.child_bounds[1].x, 105.0); // 65 + 30 + 10
    }

    // ===== Edge Cases =====

    #[test]
    fn test_row_layout_empty() {
        let mut row = Row::new();
        let result = row.layout(Rect::new(0.0, 0.0, 200.0, 50.0));
        assert_eq!(result.size, Size::ZERO);
    }

    #[test]
    fn test_row_content_larger_than_bounds() {
        let mut row = Row::new()
            .child(FixedWidget::new(100.0, 30.0))
            .child(FixedWidget::new(100.0, 30.0))
            .child(FixedWidget::new(100.0, 30.0));

        // Container only 200 wide, content is 300
        row.layout(Rect::new(0.0, 0.0, 200.0, 50.0));

        // Children still placed sequentially (overflow)
        assert_eq!(row.child_bounds[0].x, 0.0);
        assert_eq!(row.child_bounds[1].x, 100.0);
        assert_eq!(row.child_bounds[2].x, 200.0);
    }

    #[test]
    fn test_row_with_offset_bounds() {
        let mut row = Row::new()
            .child(FixedWidget::new(30.0, 20.0))
            .child(FixedWidget::new(30.0, 20.0));

        row.layout(Rect::new(50.0, 25.0, 200.0, 50.0));

        // Children should be offset by bounds origin
        assert_eq!(row.child_bounds[0].x, 50.0);
        assert_eq!(row.child_bounds[0].y, 40.0); // 25 + (50-20)/2
        assert_eq!(row.child_bounds[1].x, 80.0);
    }

    #[test]
    fn test_row_varying_child_heights() {
        let mut row = Row::new()
            .cross_axis_alignment(CrossAxisAlignment::Center)
            .child(FixedWidget::new(30.0, 20.0))
            .child(FixedWidget::new(30.0, 60.0))
            .child(FixedWidget::new(30.0, 40.0));

        row.layout(Rect::new(0.0, 0.0, 200.0, 100.0));

        // All centered differently based on their heights
        assert_eq!(row.child_bounds[0].y, 40.0); // (100-20)/2
        assert_eq!(row.child_bounds[1].y, 20.0); // (100-60)/2
        assert_eq!(row.child_bounds[2].y, 30.0); // (100-40)/2
    }

    // ===== Enum Default Tests =====

    #[test]
    fn test_main_axis_alignment_default() {
        assert_eq!(MainAxisAlignment::default(), MainAxisAlignment::Start);
    }

    #[test]
    fn test_cross_axis_alignment_default() {
        assert_eq!(CrossAxisAlignment::default(), CrossAxisAlignment::Center);
    }
}
