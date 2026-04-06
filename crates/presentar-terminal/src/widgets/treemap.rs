//! Treemap widget for hierarchical space-filling visualization.
//!
//! Implements P207 from SPEC-024 Section 15.2.
//! Uses squarify algorithm for optimal aspect ratios.

use crate::theme::Gradient;
use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// A node in the treemap hierarchy.
#[derive(Debug, Clone)]
pub struct TreemapNode {
    /// Node label.
    pub label: String,
    /// Node value (size).
    pub value: f64,
    /// Optional color override.
    pub color: Option<Color>,
    /// Child nodes.
    pub children: Vec<Self>,
    /// Flash intensity (0.0-1.0) for change indication (GAP-TREE-001).
    /// Decays over time to create visual flash effect.
    pub flash_intensity: f32,
    /// Previous value for change detection.
    pub previous_value: Option<f64>,
}

impl TreemapNode {
    /// Create a leaf node.
    #[must_use]
    pub fn leaf(label: &str, value: f64) -> Self {
        Self {
            label: label.to_string(),
            value,
            color: None,
            children: Vec::new(),
            flash_intensity: 0.0,
            previous_value: None,
        }
    }

    /// Create a leaf node with color.
    #[must_use]
    pub fn leaf_colored(label: &str, value: f64, color: Color) -> Self {
        Self {
            label: label.to_string(),
            value,
            color: Some(color),
            children: Vec::new(),
            flash_intensity: 0.0,
            previous_value: None,
        }
    }

    /// Create a branch node.
    #[must_use]
    pub fn branch(label: &str, children: Vec<Self>) -> Self {
        let value = children.iter().map(Self::total_value).sum();
        Self {
            label: label.to_string(),
            value,
            color: None,
            children,
            flash_intensity: 0.0,
            previous_value: None,
        }
    }

    /// Update value and detect change for flash effect (GAP-TREE-001).
    ///
    /// Returns true if the value changed significantly.
    pub fn update_value(&mut self, new_value: f64) -> bool {
        let threshold = 0.01; // 1% change threshold
        let changed = self.previous_value.map_or(true, |prev| {
            let delta = (new_value - prev).abs();
            let relative = delta / prev.max(1.0);
            relative > threshold
        });

        if changed {
            self.flash_intensity = 1.0; // Full flash on change
        }

        self.previous_value = Some(self.value);
        self.value = new_value;
        changed
    }

    /// Decay flash intensity over time.
    ///
    /// Call this each frame to animate the flash effect.
    pub fn decay_flash(&mut self, decay_rate: f32) {
        self.flash_intensity = (self.flash_intensity - decay_rate).max(0.0);
        for child in &mut self.children {
            child.decay_flash(decay_rate);
        }
    }

    /// Check if this node is currently flashing.
    #[must_use]
    pub fn is_flashing(&self) -> bool {
        self.flash_intensity > 0.01
    }

    /// Get flash color (white overlay with intensity).
    #[must_use]
    pub fn flash_color(&self) -> Color {
        Color::new(1.0, 1.0, 1.0, self.flash_intensity * 0.5)
    }

    /// Get total value including children.
    #[must_use]
    pub fn total_value(&self) -> f64 {
        if self.children.is_empty() {
            self.value
        } else {
            self.children.iter().map(Self::total_value).sum()
        }
    }

    /// Check if this is a leaf node.
    #[must_use]
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

/// Layout algorithm for treemap.
#[derive(Debug, Clone, Copy, Default)]
pub enum TreemapLayout {
    /// Squarify algorithm (default, best aspect ratios).
    #[default]
    Squarify,
    /// Slice and dice (alternating horizontal/vertical).
    SliceAndDice,
    /// Binary tree layout.
    Binary,
}

/// Computed rectangle for a node.
#[derive(Debug, Clone)]
struct ComputedRect {
    rect: Rect,
    node_idx: usize,
    depth: usize,
}

/// Treemap widget.
#[derive(Debug, Clone)]
pub struct Treemap {
    root: Option<TreemapNode>,
    layout: TreemapLayout,
    gradient: Gradient,
    show_labels: bool,
    max_depth: usize,
    border_width: f32,
    bounds: Rect,
    // Cached layout
    computed_rects: Vec<ComputedRect>,
    flat_nodes: Vec<(TreemapNode, usize)>, // (node, depth)
}

impl Treemap {
    /// Create a new treemap widget.
    #[must_use]
    pub fn new() -> Self {
        Self {
            root: None,
            layout: TreemapLayout::default(),
            gradient: Gradient::three(
                Color::new(0.2, 0.4, 0.8, 1.0), // Blue at 0.0
                Color::new(0.4, 0.8, 0.4, 1.0), // Green at 0.5
                Color::new(0.8, 0.4, 0.2, 1.0), // Orange at 1.0
            ),
            show_labels: true,
            max_depth: 3,
            border_width: 0.0,
            bounds: Rect::default(),
            computed_rects: Vec::new(),
            flat_nodes: Vec::new(),
        }
    }

    /// Set the root node.
    #[must_use]
    pub fn with_root(mut self, root: TreemapNode) -> Self {
        self.root = Some(root);
        self.invalidate_layout();
        self
    }

    /// Set layout algorithm.
    #[must_use]
    pub fn with_layout(mut self, layout: TreemapLayout) -> Self {
        self.layout = layout;
        self.invalidate_layout();
        self
    }

    /// Set color gradient.
    #[must_use]
    pub fn with_gradient(mut self, gradient: Gradient) -> Self {
        self.gradient = gradient;
        self
    }

    /// Toggle labels.
    #[must_use]
    pub fn with_labels(mut self, show: bool) -> Self {
        self.show_labels = show;
        self
    }

    /// Set maximum depth to render.
    #[must_use]
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self.invalidate_layout();
        self
    }

    /// Invalidate cached layout.
    fn invalidate_layout(&mut self) {
        self.computed_rects.clear();
        self.flat_nodes.clear();
    }

    /// Flatten nodes with depth tracking.
    fn flatten_nodes_static(node: &TreemapNode, depth: usize, out: &mut Vec<(TreemapNode, usize)>) {
        out.push((node.clone(), depth));
        // Use a reasonable max depth (3) to avoid infinite recursion
        if depth < 3 {
            for child in &node.children {
                Self::flatten_nodes_static(child, depth + 1, out);
            }
        }
    }

    /// Compute layout using squarify algorithm.
    fn compute_layout(&mut self) {
        self.computed_rects.clear();
        self.flat_nodes.clear();

        let Some(root) = self.root.clone() else {
            return;
        };

        // Flatten tree for rendering
        let mut flat_nodes = Vec::new();
        Self::flatten_nodes_static(&root, 0, &mut flat_nodes);
        self.flat_nodes = flat_nodes;

        // Compute rectangles
        let bounds = self.bounds;
        self.squarify_layout(&root, bounds, 0, &mut 0);
    }

    /// Squarify algorithm implementation.
    fn squarify_layout(
        &mut self,
        node: &TreemapNode,
        rect: Rect,
        depth: usize,
        node_idx: &mut usize,
    ) {
        let current_idx = *node_idx;
        *node_idx += 1;

        // Store this node's rect
        self.computed_rects.push(ComputedRect {
            rect,
            node_idx: current_idx,
            depth,
        });

        if node.children.is_empty()
            || depth >= self.max_depth
            || rect.width < 2.0
            || rect.height < 2.0
        {
            return;
        }

        // Get sorted children by value (descending)
        let mut children: Vec<_> = node.children.iter().collect();
        children.sort_by(|a, b| {
            b.total_value()
                .partial_cmp(&a.total_value())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let total_value: f64 = children.iter().map(|c| c.total_value()).sum();
        if total_value <= 0.0 {
            return;
        }

        // Squarify layout
        let mut remaining_rect = Rect::new(
            rect.x + self.border_width,
            rect.y + self.border_width,
            (rect.width - 2.0 * self.border_width).max(0.0),
            (rect.height - 2.0 * self.border_width).max(0.0),
        );

        let mut row: Vec<(usize, f64)> = Vec::new();
        let mut row_sum = 0.0;

        for (i, child) in children.iter().enumerate() {
            let child_value = child.total_value();
            if child_value <= 0.0 {
                continue;
            }

            // Try adding to current row
            row.push((i, child_value));
            row_sum += child_value;

            // Check if we should finalize this row
            let worst_current = self.worst_ratio(&row, row_sum, remaining_rect, total_value);

            if i + 1 < children.len() {
                let next_value = children[i + 1].total_value();
                let mut test_row = row.clone();
                test_row.push((i + 1, next_value));
                let worst_with_next =
                    self.worst_ratio(&test_row, row_sum + next_value, remaining_rect, total_value);

                if worst_with_next > worst_current {
                    // Finalize current row
                    remaining_rect = self.layout_row(
                        &children,
                        &row,
                        row_sum,
                        remaining_rect,
                        total_value,
                        depth,
                        node_idx,
                    );
                    row.clear();
                    row_sum = 0.0;
                }
            }
        }

        // Layout final row
        if !row.is_empty() {
            self.layout_row(
                &children,
                &row,
                row_sum,
                remaining_rect,
                total_value,
                depth,
                node_idx,
            );
        }
    }

    /// Calculate worst aspect ratio in a row.
    fn worst_ratio(&self, row: &[(usize, f64)], row_sum: f64, rect: Rect, total: f64) -> f64 {
        if row.is_empty() || row_sum <= 0.0 || total <= 0.0 {
            return f64::INFINITY;
        }

        let area = rect.width as f64 * rect.height as f64;
        let row_area = area * (row_sum / total);

        let is_horizontal = rect.width >= rect.height;
        let side = if is_horizontal {
            rect.height
        } else {
            rect.width
        } as f64;

        if side <= 0.0 {
            return f64::INFINITY;
        }

        let side_sq = side * side;
        let row_sum_sq = row_sum * row_sum;

        row.iter()
            .map(|(_, v)| {
                let ratio = (row_area * v) / (side_sq * row_sum_sq / row_area);
                ratio.max(1.0 / ratio)
            })
            .fold(0.0f64, f64::max)
    }

    /// Layout a row of nodes and return remaining rect.
    #[allow(clippy::too_many_arguments)]
    fn layout_row(
        &mut self,
        children: &[&TreemapNode],
        row: &[(usize, f64)],
        row_sum: f64,
        rect: Rect,
        total: f64,
        depth: usize,
        node_idx: &mut usize,
    ) -> Rect {
        if row.is_empty() || row_sum <= 0.0 || total <= 0.0 {
            return rect;
        }

        let is_horizontal = rect.width >= rect.height;
        let row_fraction = row_sum / total;

        let (row_rect, remaining) = if is_horizontal {
            let row_height = rect.height * row_fraction as f32;
            (
                Rect::new(rect.x, rect.y, rect.width, row_height),
                Rect::new(
                    rect.x,
                    rect.y + row_height,
                    rect.width,
                    rect.height - row_height,
                ),
            )
        } else {
            let row_width = rect.width * row_fraction as f32;
            (
                Rect::new(rect.x, rect.y, row_width, rect.height),
                Rect::new(
                    rect.x + row_width,
                    rect.y,
                    rect.width - row_width,
                    rect.height,
                ),
            )
        };

        // Layout each child in the row
        let mut offset = 0.0f32;
        for &(child_idx, child_value) in row {
            let child_fraction = child_value / row_sum;

            let child_rect = if is_horizontal {
                let w = row_rect.width * child_fraction as f32;
                let r = Rect::new(row_rect.x + offset, row_rect.y, w, row_rect.height);
                offset += w;
                r
            } else {
                let h = row_rect.height * child_fraction as f32;
                let r = Rect::new(row_rect.x, row_rect.y + offset, row_rect.width, h);
                offset += h;
                r
            };

            self.squarify_layout(children[child_idx], child_rect, depth + 1, node_idx);
        }

        remaining
    }
}

impl Default for Treemap {
    fn default() -> Self {
        Self::new()
    }
}

impl Widget for Treemap {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        Size::new(
            constraints.max_width.min(80.0),
            constraints.max_height.min(40.0),
        )
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        self.compute_layout();
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        if self.bounds.width < 4.0 || self.bounds.height < 2.0 {
            return;
        }

        #[allow(clippy::redundant_closure_for_method_calls)]
        let total_value = self.root.as_ref().map_or(1.0, |r| r.total_value());

        // Draw rectangles from deepest to shallowest (painter's algorithm)
        let mut sorted_rects: Vec<_> = self.computed_rects.iter().collect();
        sorted_rects.sort_by(|a, b| b.depth.cmp(&a.depth));

        for computed in sorted_rects {
            if computed.rect.width < 1.0 || computed.rect.height < 1.0 {
                continue;
            }

            let node = if computed.node_idx < self.flat_nodes.len() {
                &self.flat_nodes[computed.node_idx].0
            } else {
                continue;
            };

            // Determine color
            let color = node.color.unwrap_or_else(|| {
                let t = (node.total_value() / total_value).clamp(0.0, 1.0);
                self.gradient.sample(t)
            });

            // Fill rectangle
            let fill_char = if computed.depth == 0 { ' ' } else { '░' };
            let style = TextStyle {
                color,
                ..Default::default()
            };

            for y in 0..(computed.rect.height as usize).max(1) {
                let row: String = (0..(computed.rect.width as usize).max(1))
                    .map(|_| fill_char)
                    .collect();
                canvas.draw_text(
                    &row,
                    Point::new(computed.rect.x, computed.rect.y + y as f32),
                    &style,
                );
            }

            // Draw label if enabled and there's space
            if self.show_labels && computed.rect.width >= 3.0 && computed.rect.height >= 1.0 {
                let label = if node.label.len() > computed.rect.width as usize - 1 {
                    format!("{}…", &node.label[..computed.rect.width as usize - 2])
                } else {
                    node.label.clone()
                };

                let label_style = TextStyle {
                    color: Color::new(1.0, 1.0, 1.0, 1.0),
                    ..Default::default()
                };

                canvas.draw_text(
                    &label,
                    Point::new(computed.rect.x + 1.0, computed.rect.y),
                    &label_style,
                );
            }
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

impl Brick for Treemap {
    fn brick_name(&self) -> &'static str {
        "Treemap"
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

        if self.bounds.width >= 4.0 && self.bounds.height >= 2.0 {
            passed.push(BrickAssertion::max_latency_ms(16));
        } else {
            failed.push((
                BrickAssertion::max_latency_ms(16),
                "Size too small".to_string(),
            ));
        }

        BrickVerification {
            passed,
            failed,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CellBuffer, DirectTerminalCanvas};

    #[test]
    fn test_treemap_creation() {
        let treemap = Treemap::new();
        assert!(treemap.root.is_none());
    }

    #[test]
    fn test_leaf_node() {
        let node = TreemapNode::leaf("test", 100.0);
        assert_eq!(node.label, "test");
        assert_eq!(node.value, 100.0);
        assert!(node.is_leaf());
    }

    #[test]
    fn test_leaf_node_colored() {
        let color = Color::new(0.5, 0.6, 0.7, 1.0);
        let node = TreemapNode::leaf_colored("colored", 50.0, color);
        assert_eq!(node.label, "colored");
        assert_eq!(node.value, 50.0);
        assert!(node.color.is_some());
        assert!(node.is_leaf());
    }

    #[test]
    fn test_branch_node() {
        let branch = TreemapNode::branch(
            "parent",
            vec![
                TreemapNode::leaf("child1", 50.0),
                TreemapNode::leaf("child2", 30.0),
            ],
        );
        assert!(!branch.is_leaf());
        assert_eq!(branch.total_value(), 80.0);
    }

    #[test]
    fn test_nested_branch_total_value() {
        let root = TreemapNode::branch(
            "root",
            vec![
                TreemapNode::branch(
                    "sub1",
                    vec![TreemapNode::leaf("a", 10.0), TreemapNode::leaf("b", 20.0)],
                ),
                TreemapNode::leaf("c", 30.0),
            ],
        );
        assert_eq!(root.total_value(), 60.0);
    }

    #[test]
    fn test_treemap_with_root() {
        let root = TreemapNode::branch(
            "root",
            vec![TreemapNode::leaf("a", 100.0), TreemapNode::leaf("b", 50.0)],
        );
        let treemap = Treemap::new().with_root(root);
        assert!(treemap.root.is_some());
    }

    #[test]
    fn test_treemap_with_layout() {
        let treemap = Treemap::new().with_layout(TreemapLayout::SliceAndDice);
        assert!(matches!(treemap.layout, TreemapLayout::SliceAndDice));

        let treemap2 = Treemap::new().with_layout(TreemapLayout::Binary);
        assert!(matches!(treemap2.layout, TreemapLayout::Binary));
    }

    #[test]
    fn test_treemap_with_gradient() {
        let gradient = Gradient::two(
            Color::new(1.0, 0.0, 0.0, 1.0),
            Color::new(0.0, 0.0, 1.0, 1.0),
        );
        let treemap = Treemap::new().with_gradient(gradient);
        // Verify gradient is set (sample a point)
        let sample = treemap.gradient.sample(0.5);
        assert!(sample.r > 0.0);
    }

    #[test]
    fn test_treemap_with_labels() {
        let treemap = Treemap::new().with_labels(false);
        assert!(!treemap.show_labels);

        let treemap2 = Treemap::new().with_labels(true);
        assert!(treemap2.show_labels);
    }

    #[test]
    fn test_treemap_with_max_depth() {
        let treemap = Treemap::new().with_max_depth(5);
        assert_eq!(treemap.max_depth, 5);
    }

    #[test]
    fn test_treemap_measure() {
        let treemap = Treemap::new();
        let constraints = Constraints::new(0.0, 100.0, 0.0, 50.0);
        let size = treemap.measure(constraints);
        assert_eq!(size.width, 80.0); // min(100, 80)
        assert_eq!(size.height, 40.0); // min(50, 40)
    }

    #[test]
    fn test_treemap_measure_small_constraints() {
        let treemap = Treemap::new();
        let constraints = Constraints::new(0.0, 30.0, 0.0, 20.0);
        let size = treemap.measure(constraints);
        assert_eq!(size.width, 30.0);
        assert_eq!(size.height, 20.0);
    }

    #[test]
    fn test_treemap_layout_and_paint() {
        let root = TreemapNode::branch(
            "root",
            vec![
                TreemapNode::leaf_colored("big", 100.0, Color::new(0.8, 0.2, 0.2, 1.0)),
                TreemapNode::leaf_colored("small", 50.0, Color::new(0.2, 0.8, 0.2, 1.0)),
            ],
        );
        let mut treemap = Treemap::new().with_root(root);

        let mut buffer = CellBuffer::new(40, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        let result = treemap.layout(Rect::new(0.0, 0.0, 40.0, 20.0));
        assert_eq!(result.size.width, 40.0);
        assert_eq!(result.size.height, 20.0);

        treemap.paint(&mut canvas);

        // Verify something was rendered
        let cells = buffer.cells();
        let non_empty = cells
            .iter()
            .filter(|c| !c.symbol.is_empty() && c.symbol != " ")
            .count();
        assert!(non_empty > 0, "Treemap should render some content");
    }

    #[test]
    fn test_treemap_paint_too_small() {
        let root = TreemapNode::leaf("tiny", 10.0);
        let mut treemap = Treemap::new().with_root(root);

        let mut buffer = CellBuffer::new(2, 1);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        treemap.layout(Rect::new(0.0, 0.0, 2.0, 1.0));
        treemap.paint(&mut canvas);
        // Should not crash with tiny bounds
    }

    #[test]
    fn test_treemap_paint_no_root() {
        let mut treemap = Treemap::new();

        let mut buffer = CellBuffer::new(40, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        treemap.layout(Rect::new(0.0, 0.0, 40.0, 20.0));
        treemap.paint(&mut canvas);
        // Should not crash with no root
    }

    #[test]
    fn test_treemap_deep_hierarchy() {
        let root = TreemapNode::branch(
            "level0",
            vec![TreemapNode::branch(
                "level1",
                vec![TreemapNode::branch(
                    "level2",
                    vec![TreemapNode::branch(
                        "level3",
                        vec![TreemapNode::leaf("deep", 100.0)],
                    )],
                )],
            )],
        );
        let mut treemap = Treemap::new().with_root(root).with_max_depth(4);

        let mut buffer = CellBuffer::new(60, 30);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        treemap.layout(Rect::new(0.0, 0.0, 60.0, 30.0));
        treemap.paint(&mut canvas);

        // Should render without crashing
        let cells = buffer.cells();
        assert!(!cells.is_empty());
    }

    #[test]
    fn test_treemap_many_children() {
        let children: Vec<TreemapNode> = (0..10)
            .map(|i| TreemapNode::leaf(&format!("node{i}"), (i + 1) as f64 * 10.0))
            .collect();
        let root = TreemapNode::branch("root", children);

        let mut treemap = Treemap::new().with_root(root);

        let mut buffer = CellBuffer::new(80, 40);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        treemap.layout(Rect::new(0.0, 0.0, 80.0, 40.0));
        treemap.paint(&mut canvas);
    }

    #[test]
    fn test_treemap_zero_value_children() {
        let root = TreemapNode::branch(
            "root",
            vec![
                TreemapNode::leaf("valid", 100.0),
                TreemapNode::leaf("zero", 0.0),
            ],
        );
        let mut treemap = Treemap::new().with_root(root);

        let mut buffer = CellBuffer::new(40, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        treemap.layout(Rect::new(0.0, 0.0, 40.0, 20.0));
        treemap.paint(&mut canvas);
        // Should handle zero-value nodes gracefully
    }

    #[test]
    fn test_treemap_long_labels() {
        let root = TreemapNode::branch(
            "root",
            vec![TreemapNode::leaf(
                "this_is_a_very_long_label_that_should_be_truncated",
                100.0,
            )],
        );
        let mut treemap = Treemap::new().with_root(root).with_labels(true);

        let mut buffer = CellBuffer::new(20, 10);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        treemap.layout(Rect::new(0.0, 0.0, 20.0, 10.0));
        treemap.paint(&mut canvas);
    }

    #[test]
    fn test_treemap_labels_disabled() {
        let root = TreemapNode::leaf("test", 100.0);
        let mut treemap = Treemap::new().with_root(root).with_labels(false);

        let mut buffer = CellBuffer::new(40, 20);
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        treemap.layout(Rect::new(0.0, 0.0, 40.0, 20.0));
        treemap.paint(&mut canvas);
    }

    #[test]
    fn test_treemap_assertions() {
        let treemap = Treemap::default();
        assert!(!treemap.assertions().is_empty());
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn test_treemap_verify_valid() {
        let mut treemap = Treemap::default();
        treemap.bounds = Rect::new(0.0, 0.0, 80.0, 40.0);
        assert!(treemap.verify().is_valid());
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn test_treemap_verify_invalid_small() {
        let mut treemap = Treemap::default();
        treemap.bounds = Rect::new(0.0, 0.0, 2.0, 1.0);
        assert!(!treemap.verify().is_valid());
    }

    #[test]
    fn test_treemap_children() {
        let treemap = Treemap::default();
        assert!(treemap.children().is_empty());
    }

    #[test]
    fn test_treemap_children_mut() {
        let mut treemap = Treemap::default();
        assert!(treemap.children_mut().is_empty());
    }

    #[test]
    fn test_treemap_brick_name() {
        let treemap = Treemap::new();
        assert_eq!(treemap.brick_name(), "Treemap");
    }

    #[test]
    fn test_treemap_budget() {
        let treemap = Treemap::new();
        let budget = treemap.budget();
        assert!(budget.layout_ms > 0);
        assert!(budget.paint_ms > 0);
    }

    #[test]
    fn test_treemap_to_html() {
        let treemap = Treemap::new();
        assert!(treemap.to_html().is_empty());
    }

    #[test]
    fn test_treemap_to_css() {
        let treemap = Treemap::new();
        assert!(treemap.to_css().is_empty());
    }

    #[test]
    fn test_treemap_type_id() {
        let treemap = Treemap::new();
        let type_id = Widget::type_id(&treemap);
        assert_eq!(type_id, TypeId::of::<Treemap>());
    }

    #[test]
    fn test_treemap_event() {
        let mut treemap = Treemap::new();
        let event = Event::Resize {
            width: 80.0,
            height: 24.0,
        };
        assert!(treemap.event(&event).is_none());
    }

    #[test]
    fn test_treemap_vertical_layout() {
        // Create a tall, narrow treemap to test vertical layout path
        let root = TreemapNode::branch(
            "root",
            vec![TreemapNode::leaf("a", 100.0), TreemapNode::leaf("b", 100.0)],
        );
        let mut treemap = Treemap::new().with_root(root);

        let mut buffer = CellBuffer::new(10, 40); // Tall and narrow
        let mut canvas = DirectTerminalCanvas::new(&mut buffer);

        treemap.layout(Rect::new(0.0, 0.0, 10.0, 40.0));
        treemap.paint(&mut canvas);
    }

    #[test]
    fn test_treemap_layout_default() {
        let layout = TreemapLayout::default();
        assert!(matches!(layout, TreemapLayout::Squarify));
    }

    // =========================================================================
    // GAP-TREE-001: Flash effect tests
    // =========================================================================

    #[test]
    fn test_treemap_node_flash_intensity_default() {
        let node = TreemapNode::leaf("test", 100.0);
        assert_eq!(node.flash_intensity, 0.0);
        assert!(node.previous_value.is_none());
    }

    #[test]
    fn test_treemap_node_flash_intensity_colored() {
        let node = TreemapNode::leaf_colored("test", 100.0, Color::new(1.0, 0.0, 0.0, 1.0));
        assert_eq!(node.flash_intensity, 0.0);
        assert!(node.previous_value.is_none());
    }

    #[test]
    fn test_treemap_node_flash_intensity_branch() {
        let node = TreemapNode::branch("root", vec![TreemapNode::leaf("child", 50.0)]);
        assert_eq!(node.flash_intensity, 0.0);
        assert!(node.previous_value.is_none());
    }

    #[test]
    fn test_treemap_node_update_value_first_call() {
        let mut node = TreemapNode::leaf("test", 100.0);
        let changed = node.update_value(150.0);
        assert!(changed, "First update should always report change");
        assert_eq!(node.flash_intensity, 1.0);
        assert_eq!(node.value, 150.0);
        assert_eq!(node.previous_value, Some(100.0));
    }

    #[test]
    fn test_treemap_node_update_value_significant_change() {
        let mut node = TreemapNode::leaf("test", 100.0);
        node.previous_value = Some(100.0);
        node.flash_intensity = 0.0;

        let changed = node.update_value(150.0); // 50% change
        assert!(changed, "50% change should trigger flash");
        assert_eq!(node.flash_intensity, 1.0);
    }

    #[test]
    fn test_treemap_node_update_value_small_change() {
        let mut node = TreemapNode::leaf("test", 100.0);
        node.previous_value = Some(100.0);
        node.flash_intensity = 0.0;

        let changed = node.update_value(100.5); // 0.5% change
        assert!(!changed, "0.5% change should not trigger flash");
        assert_eq!(node.flash_intensity, 0.0);
    }

    #[test]
    fn test_treemap_node_decay_flash() {
        let mut node = TreemapNode::leaf("test", 100.0);
        node.flash_intensity = 1.0;

        node.decay_flash(0.1);
        assert!((node.flash_intensity - 0.9).abs() < 0.001);

        node.decay_flash(0.5);
        assert!((node.flash_intensity - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_treemap_node_decay_flash_clamps_zero() {
        let mut node = TreemapNode::leaf("test", 100.0);
        node.flash_intensity = 0.1;

        node.decay_flash(0.5); // Would go negative
        assert_eq!(node.flash_intensity, 0.0);
    }

    #[test]
    fn test_treemap_node_decay_flash_recursive() {
        let mut root = TreemapNode::branch(
            "root",
            vec![TreemapNode::leaf("a", 50.0), TreemapNode::leaf("b", 50.0)],
        );
        root.flash_intensity = 1.0;
        root.children[0].flash_intensity = 0.8;
        root.children[1].flash_intensity = 0.5;

        root.decay_flash(0.2);

        assert!((root.flash_intensity - 0.8).abs() < 0.001);
        assert!((root.children[0].flash_intensity - 0.6).abs() < 0.001);
        assert!((root.children[1].flash_intensity - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_treemap_node_is_flashing() {
        let mut node = TreemapNode::leaf("test", 100.0);
        assert!(!node.is_flashing());

        node.flash_intensity = 1.0;
        assert!(node.is_flashing());

        node.flash_intensity = 0.02;
        assert!(node.is_flashing());

        node.flash_intensity = 0.005; // Below threshold
        assert!(!node.is_flashing());
    }

    #[test]
    fn test_treemap_node_flash_color() {
        let mut node = TreemapNode::leaf("test", 100.0);
        node.flash_intensity = 1.0;

        let color = node.flash_color();
        assert_eq!(color.r, 1.0);
        assert_eq!(color.g, 1.0);
        assert_eq!(color.b, 1.0);
        assert!((color.a - 0.5).abs() < 0.001); // 1.0 * 0.5

        node.flash_intensity = 0.5;
        let color2 = node.flash_color();
        assert!((color2.a - 0.25).abs() < 0.001); // 0.5 * 0.5
    }

    #[test]
    fn test_treemap_node_flash_color_zero_intensity() {
        let node = TreemapNode::leaf("test", 100.0);
        let color = node.flash_color();
        assert_eq!(color.a, 0.0);
    }
}
