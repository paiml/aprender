//! Column widget for vertical layout.

use presentar_core::{
    widget::LayoutResult, Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas,
    Constraints, Event, Rect, Size, TypeId, Widget,
};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::time::Duration;

use crate::row::{CrossAxisAlignment, MainAxisAlignment};

/// Column widget for vertical layout of children.
#[derive(Serialize, Deserialize)]
pub struct Column {
    /// Main axis (vertical) alignment
    main_axis_alignment: MainAxisAlignment,
    /// Cross axis (horizontal) alignment
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

impl Default for Column {
    fn default() -> Self {
        Self::new()
    }
}

impl Column {
    /// Create a new empty column.
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

impl Widget for Column {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        if self.children.is_empty() {
            return Size::ZERO;
        }

        let mut max_width = 0.0f32;
        let mut total_height = 0.0f32;

        // Measure all children
        for (i, child) in self.children.iter().enumerate() {
            let child_constraints = Constraints::new(
                0.0,
                constraints.max_width,
                0.0,
                (constraints.max_height - total_height).max(0.0),
            );

            let child_size = child.measure(child_constraints);
            max_width = max_width.max(child_size.width);
            total_height += child_size.height;

            if i < self.children.len() - 1 {
                total_height += self.gap;
            }
        }

        constraints.constrain(Size::new(max_width, total_height))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        self.child_bounds.clear();

        if self.children.is_empty() {
            return LayoutResult { size: Size::ZERO };
        }

        // First pass: measure children
        let mut child_sizes: Vec<Size> = Vec::with_capacity(self.children.len());
        let mut total_height = 0.0f32;

        for child in &self.children {
            let child_constraints = Constraints::loose(bounds.size());
            let size = child.measure(child_constraints);
            total_height += size.height;
            child_sizes.push(size);
        }

        let gaps_height = self.gap * (self.children.len() - 1).max(0) as f32;
        let content_height = total_height + gaps_height;
        let remaining_space = (bounds.height - content_height).max(0.0);

        // Calculate starting position based on alignment
        let (mut y, extra_gap) = match self.main_axis_alignment {
            MainAxisAlignment::Start => (bounds.y, 0.0),
            MainAxisAlignment::End => (bounds.y + remaining_space, 0.0),
            MainAxisAlignment::Center => (bounds.y + remaining_space / 2.0, 0.0),
            MainAxisAlignment::SpaceBetween => {
                if self.children.len() > 1 {
                    (bounds.y, remaining_space / (self.children.len() - 1) as f32)
                } else {
                    (bounds.y, 0.0)
                }
            }
            MainAxisAlignment::SpaceAround => {
                let gap = remaining_space / self.children.len() as f32;
                (bounds.y + gap / 2.0, gap)
            }
            MainAxisAlignment::SpaceEvenly => {
                let gap = remaining_space / (self.children.len() + 1) as f32;
                (bounds.y + gap, gap)
            }
        };

        // Second pass: position children
        let num_children = self.children.len();
        for (i, (child, size)) in self.children.iter_mut().zip(child_sizes.iter()).enumerate() {
            let x = match self.cross_axis_alignment {
                CrossAxisAlignment::Start | CrossAxisAlignment::Stretch => bounds.x,
                CrossAxisAlignment::End => bounds.x + bounds.width - size.width,
                CrossAxisAlignment::Center => bounds.x + (bounds.width - size.width) / 2.0,
            };

            let width = if self.cross_axis_alignment == CrossAxisAlignment::Stretch {
                bounds.width
            } else {
                size.width
            };

            let child_bounds = Rect::new(x, y, width, size.height);
            child.layout(child_bounds);
            self.child_bounds.push(child_bounds);

            // Move y for next child
            if i < num_children - 1 {
                y += size.height;
                if self.main_axis_alignment == MainAxisAlignment::SpaceBetween {
                    // SpaceBetween uses only extra_gap (no regular gap)
                    y += extra_gap;
                } else {
                    y += self.gap + extra_gap;
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
impl Brick for Column {
    fn brick_name(&self) -> &'static str {
        "Column"
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
        let test_id = self.test_id_value.as_deref().unwrap_or("column");
        format!(r#"<div class="brick-column" data-testid="{test_id}"></div>"#)
    }

    fn to_css(&self) -> String {
        ".brick-column { display: flex; flex-direction: column; }".into()
    }

    fn test_id(&self) -> Option<&str> {
        self.test_id_value.as_deref()
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
    fn test_column_empty() {
        let col = Column::new();
        let size = col.measure(Constraints::loose(Size::new(100.0, 100.0)));
        assert_eq!(size, Size::ZERO);
    }

    #[test]
    fn test_column_builder() {
        let col = Column::new()
            .gap(10.0)
            .main_axis_alignment(MainAxisAlignment::Center)
            .cross_axis_alignment(CrossAxisAlignment::Start)
            .with_test_id("my-column");

        assert_eq!(col.gap, 10.0);
        assert_eq!(col.main_axis_alignment, MainAxisAlignment::Center);
        assert_eq!(col.cross_axis_alignment, CrossAxisAlignment::Start);
        assert_eq!(Widget::test_id(&col), Some("my-column"));
    }

    #[test]
    fn test_column_default() {
        let col = Column::default();
        assert_eq!(col.main_axis_alignment, MainAxisAlignment::Start);
        assert_eq!(col.cross_axis_alignment, CrossAxisAlignment::Center);
        assert_eq!(col.gap, 0.0);
    }

    #[test]
    fn test_column_type_id() {
        let col = Column::new();
        assert_eq!(Widget::type_id(&col), TypeId::of::<Column>());
    }

    #[test]
    fn test_column_children() {
        let col = Column::new()
            .child(FixedWidget::new(50.0, 30.0))
            .child(FixedWidget::new(50.0, 30.0));
        assert_eq!(col.children().len(), 2);
    }

    // ===== Measure Tests =====

    #[test]
    fn test_column_measure_single_child() {
        let col = Column::new().child(FixedWidget::new(50.0, 30.0));
        let size = col.measure(Constraints::loose(Size::new(200.0, 200.0)));
        assert_eq!(size, Size::new(50.0, 30.0));
    }

    #[test]
    fn test_column_measure_multiple_children() {
        let col = Column::new()
            .child(FixedWidget::new(50.0, 30.0))
            .child(FixedWidget::new(60.0, 40.0));
        let size = col.measure(Constraints::loose(Size::new(200.0, 200.0)));
        assert_eq!(size, Size::new(60.0, 70.0)); // max width, sum heights
    }

    #[test]
    fn test_column_measure_with_gap() {
        let col = Column::new()
            .gap(10.0)
            .child(FixedWidget::new(50.0, 30.0))
            .child(FixedWidget::new(50.0, 30.0));
        let size = col.measure(Constraints::loose(Size::new(200.0, 200.0)));
        assert_eq!(size, Size::new(50.0, 70.0)); // 30 + 10 + 30
    }

    #[test]
    fn test_column_measure_constrained() {
        let col = Column::new()
            .child(FixedWidget::new(100.0, 100.0))
            .child(FixedWidget::new(100.0, 100.0));
        let size = col.measure(Constraints::tight(Size::new(80.0, 150.0)));
        assert_eq!(size, Size::new(80.0, 150.0)); // Constrained to tight
    }

    // ===== MainAxisAlignment Tests =====

    #[test]
    fn test_column_alignment_start() {
        let mut col = Column::new()
            .main_axis_alignment(MainAxisAlignment::Start)
            .child(FixedWidget::new(30.0, 20.0))
            .child(FixedWidget::new(30.0, 20.0));

        col.layout(Rect::new(0.0, 0.0, 100.0, 200.0));

        assert_eq!(col.child_bounds.len(), 2);
        assert_eq!(col.child_bounds[0].y, 0.0);
        assert_eq!(col.child_bounds[1].y, 20.0);
    }

    #[test]
    fn test_column_alignment_end() {
        let mut col = Column::new()
            .main_axis_alignment(MainAxisAlignment::End)
            .child(FixedWidget::new(30.0, 20.0))
            .child(FixedWidget::new(30.0, 20.0));

        col.layout(Rect::new(0.0, 0.0, 100.0, 200.0));

        // 200 - 40 = 160 remaining, children at 160 and 180
        assert_eq!(col.child_bounds[0].y, 160.0);
        assert_eq!(col.child_bounds[1].y, 180.0);
    }

    #[test]
    fn test_column_alignment_center() {
        let mut col = Column::new()
            .main_axis_alignment(MainAxisAlignment::Center)
            .child(FixedWidget::new(30.0, 20.0))
            .child(FixedWidget::new(30.0, 20.0));

        col.layout(Rect::new(0.0, 0.0, 100.0, 200.0));

        // 200 - 40 = 160 remaining, offset = 80
        assert_eq!(col.child_bounds[0].y, 80.0);
        assert_eq!(col.child_bounds[1].y, 100.0);
    }

    #[test]
    fn test_column_alignment_space_between() {
        let mut col = Column::new()
            .main_axis_alignment(MainAxisAlignment::SpaceBetween)
            .child(FixedWidget::new(30.0, 20.0))
            .child(FixedWidget::new(30.0, 20.0));

        col.layout(Rect::new(0.0, 0.0, 100.0, 200.0));

        // First at start, last at end
        assert_eq!(col.child_bounds[0].y, 0.0);
        assert_eq!(col.child_bounds[1].y, 180.0); // 200 - 20
    }

    #[test]
    fn test_column_alignment_space_between_single_child() {
        let mut col = Column::new()
            .main_axis_alignment(MainAxisAlignment::SpaceBetween)
            .child(FixedWidget::new(30.0, 20.0));

        col.layout(Rect::new(0.0, 0.0, 100.0, 200.0));

        // Single child should be at start
        assert_eq!(col.child_bounds[0].y, 0.0);
    }

    #[test]
    fn test_column_alignment_space_between_three_children() {
        let mut col = Column::new()
            .main_axis_alignment(MainAxisAlignment::SpaceBetween)
            .child(FixedWidget::new(30.0, 20.0))
            .child(FixedWidget::new(30.0, 20.0))
            .child(FixedWidget::new(30.0, 20.0));

        col.layout(Rect::new(0.0, 0.0, 100.0, 200.0));

        // 200 - 60 = 140 remaining, gap = 70
        assert_eq!(col.child_bounds[0].y, 0.0);
        assert_eq!(col.child_bounds[1].y, 90.0); // 20 + 70
        assert_eq!(col.child_bounds[2].y, 180.0); // 200 - 20
    }

    #[test]
    fn test_column_alignment_space_around() {
        let mut col = Column::new()
            .main_axis_alignment(MainAxisAlignment::SpaceAround)
            .child(FixedWidget::new(30.0, 40.0))
            .child(FixedWidget::new(30.0, 40.0));

        col.layout(Rect::new(0.0, 0.0, 100.0, 200.0));

        // 200 - 80 = 120 remaining, gap = 60, half-gap = 30
        // First at 30, second at 30 + 40 + 60 = 130
        assert_eq!(col.child_bounds[0].y, 30.0);
        assert_eq!(col.child_bounds[1].y, 130.0);
    }

    #[test]
    fn test_column_alignment_space_evenly() {
        let mut col = Column::new()
            .main_axis_alignment(MainAxisAlignment::SpaceEvenly)
            .child(FixedWidget::new(30.0, 40.0))
            .child(FixedWidget::new(30.0, 40.0));

        col.layout(Rect::new(0.0, 0.0, 100.0, 200.0));

        // 200 - 80 = 120 remaining, 3 gaps (n+1), gap = 40
        // First at 40, second at 40 + 40 + 40 = 120
        assert_eq!(col.child_bounds[0].y, 40.0);
        assert_eq!(col.child_bounds[1].y, 120.0);
    }

    // ===== CrossAxisAlignment Tests =====

    #[test]
    fn test_column_cross_alignment_start() {
        let mut col = Column::new()
            .cross_axis_alignment(CrossAxisAlignment::Start)
            .child(FixedWidget::new(30.0, 20.0));

        col.layout(Rect::new(0.0, 0.0, 100.0, 200.0));

        assert_eq!(col.child_bounds[0].x, 0.0);
        assert_eq!(col.child_bounds[0].width, 30.0);
    }

    #[test]
    fn test_column_cross_alignment_end() {
        let mut col = Column::new()
            .cross_axis_alignment(CrossAxisAlignment::End)
            .child(FixedWidget::new(30.0, 20.0));

        col.layout(Rect::new(0.0, 0.0, 100.0, 200.0));

        assert_eq!(col.child_bounds[0].x, 70.0); // 100 - 30
        assert_eq!(col.child_bounds[0].width, 30.0);
    }

    #[test]
    fn test_column_cross_alignment_center() {
        let mut col = Column::new()
            .cross_axis_alignment(CrossAxisAlignment::Center)
            .child(FixedWidget::new(30.0, 20.0));

        col.layout(Rect::new(0.0, 0.0, 100.0, 200.0));

        assert_eq!(col.child_bounds[0].x, 35.0); // (100 - 30) / 2
        assert_eq!(col.child_bounds[0].width, 30.0);
    }

    #[test]
    fn test_column_cross_alignment_stretch() {
        let mut col = Column::new()
            .cross_axis_alignment(CrossAxisAlignment::Stretch)
            .child(FixedWidget::new(30.0, 20.0));

        col.layout(Rect::new(0.0, 0.0, 100.0, 200.0));

        assert_eq!(col.child_bounds[0].x, 0.0);
        assert_eq!(col.child_bounds[0].width, 100.0); // Stretched to container
    }

    // ===== Gap Tests =====

    #[test]
    fn test_column_gap_single_child() {
        let mut col = Column::new().gap(20.0).child(FixedWidget::new(30.0, 20.0));

        col.layout(Rect::new(0.0, 0.0, 100.0, 200.0));

        // Single child: no gap applied
        assert_eq!(col.child_bounds[0].y, 0.0);
    }

    #[test]
    fn test_column_gap_multiple_children() {
        let mut col = Column::new()
            .gap(15.0)
            .child(FixedWidget::new(30.0, 20.0))
            .child(FixedWidget::new(30.0, 20.0))
            .child(FixedWidget::new(30.0, 20.0));

        col.layout(Rect::new(0.0, 0.0, 100.0, 200.0));

        assert_eq!(col.child_bounds[0].y, 0.0);
        assert_eq!(col.child_bounds[1].y, 35.0); // 20 + 15
        assert_eq!(col.child_bounds[2].y, 70.0); // 35 + 20 + 15
    }

    #[test]
    fn test_column_gap_with_alignment_center() {
        let mut col = Column::new()
            .gap(10.0)
            .main_axis_alignment(MainAxisAlignment::Center)
            .child(FixedWidget::new(30.0, 20.0))
            .child(FixedWidget::new(30.0, 20.0));

        col.layout(Rect::new(0.0, 0.0, 100.0, 200.0));

        // Total: 20 + 10 + 20 = 50, remaining = 150, offset = 75
        assert_eq!(col.child_bounds[0].y, 75.0);
        assert_eq!(col.child_bounds[1].y, 105.0); // 75 + 20 + 10
    }

    // ===== Edge Cases =====

    #[test]
    fn test_column_layout_empty() {
        let mut col = Column::new();
        let result = col.layout(Rect::new(0.0, 0.0, 100.0, 200.0));
        assert_eq!(result.size, Size::ZERO);
    }

    #[test]
    fn test_column_content_larger_than_bounds() {
        let mut col = Column::new()
            .child(FixedWidget::new(30.0, 100.0))
            .child(FixedWidget::new(30.0, 100.0))
            .child(FixedWidget::new(30.0, 100.0));

        // Container only 200 tall, content is 300
        col.layout(Rect::new(0.0, 0.0, 100.0, 200.0));

        // Children still placed sequentially (overflow)
        assert_eq!(col.child_bounds[0].y, 0.0);
        assert_eq!(col.child_bounds[1].y, 100.0);
        assert_eq!(col.child_bounds[2].y, 200.0);
    }

    #[test]
    fn test_column_with_offset_bounds() {
        let mut col = Column::new()
            .child(FixedWidget::new(30.0, 20.0))
            .child(FixedWidget::new(30.0, 20.0));

        col.layout(Rect::new(25.0, 50.0, 100.0, 200.0));

        // Children should be offset by bounds origin
        assert_eq!(col.child_bounds[0].y, 50.0);
        assert_eq!(col.child_bounds[0].x, 60.0); // 25 + (100-30)/2
        assert_eq!(col.child_bounds[1].y, 70.0);
    }

    #[test]
    fn test_column_varying_child_widths() {
        let mut col = Column::new()
            .cross_axis_alignment(CrossAxisAlignment::Center)
            .child(FixedWidget::new(20.0, 30.0))
            .child(FixedWidget::new(60.0, 30.0))
            .child(FixedWidget::new(40.0, 30.0));

        col.layout(Rect::new(0.0, 0.0, 100.0, 200.0));

        // All centered differently based on their widths
        assert_eq!(col.child_bounds[0].x, 40.0); // (100-20)/2
        assert_eq!(col.child_bounds[1].x, 20.0); // (100-60)/2
        assert_eq!(col.child_bounds[2].x, 30.0); // (100-40)/2
    }
}
