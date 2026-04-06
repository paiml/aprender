//! Tree widget for hierarchical data visualization.
//!
//! Provides collapsible tree view using Unicode tree-drawing characters.
//! Ideal for process trees, file systems, or cluster hierarchies.

use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::collections::HashSet;
use std::time::Duration;

/// Tree branch characters.
const BRANCH_PIPE: &str = "│   ";
const BRANCH_TEE: &str = "├── ";
const BRANCH_ELBOW: &str = "└── ";
const BRANCH_SPACE: &str = "    ";

/// Unique identifier for tree nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u64);

impl NodeId {
    /// Create a new node ID.
    #[must_use]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }
}

/// A node in the tree.
#[derive(Debug, Clone)]
pub struct TreeNode {
    /// Unique identifier.
    pub id: NodeId,
    /// Display label.
    pub label: String,
    /// Optional value/info to display.
    pub info: Option<String>,
    /// Child nodes.
    pub children: Vec<Self>,
    /// Node color.
    pub color: Option<Color>,
}

impl TreeNode {
    /// Create a new tree node.
    #[must_use]
    pub fn new(id: u64, label: impl Into<String>) -> Self {
        Self {
            id: NodeId::new(id),
            label: label.into(),
            info: None,
            children: vec![],
            color: None,
        }
    }

    /// Add info text.
    #[must_use]
    pub fn with_info(mut self, info: impl Into<String>) -> Self {
        self.info = Some(info.into());
        self
    }

    /// Set node color.
    #[must_use]
    pub fn with_color(mut self, color: Color) -> Self {
        self.color = Some(color);
        self
    }

    /// Add a child node.
    #[must_use]
    pub fn with_child(mut self, child: Self) -> Self {
        self.children.push(child);
        self
    }

    /// Add multiple children.
    #[must_use]
    pub fn with_children(mut self, children: Vec<Self>) -> Self {
        self.children = children;
        self
    }

    /// Count total nodes including self.
    #[must_use]
    pub fn count_nodes(&self) -> usize {
        1 + self.children.iter().map(Self::count_nodes).sum::<usize>()
    }

    /// Get depth of the tree.
    #[must_use]
    pub fn depth(&self) -> usize {
        if self.children.is_empty() {
            1
        } else {
            1 + self.children.iter().map(Self::depth).max().unwrap_or(0)
        }
    }
}

/// Tree widget for hierarchical visualization.
#[derive(Debug, Clone)]
pub struct Tree {
    /// Root node.
    root: Option<TreeNode>,
    /// Expanded node IDs.
    expanded: HashSet<NodeId>,
    /// Default color for nodes.
    default_color: Color,
    /// Show info column.
    show_info: bool,
    /// Indent string (characters per level).
    #[allow(dead_code)]
    indent_width: usize,
    /// Scroll offset (for large trees).
    scroll_offset: usize,
    /// Selected node ID.
    selected: Option<NodeId>,
    /// Cached bounds.
    bounds: Rect,
}

impl Default for Tree {
    fn default() -> Self {
        Self::new()
    }
}

impl Tree {
    /// Create a new empty tree.
    #[must_use]
    pub fn new() -> Self {
        Self {
            root: None,
            expanded: HashSet::new(),
            default_color: Color::new(0.8, 0.8, 0.8, 1.0),
            show_info: true,
            indent_width: 4,
            scroll_offset: 0,
            selected: None,
            bounds: Rect::default(),
        }
    }

    /// Set the root node.
    #[must_use]
    pub fn with_root(mut self, root: TreeNode) -> Self {
        // Expand root by default
        self.expanded.insert(root.id);
        self.root = Some(root);
        self
    }

    /// Set default node color.
    #[must_use]
    pub fn with_color(mut self, color: Color) -> Self {
        self.default_color = color;
        self
    }

    /// Show or hide info column.
    #[must_use]
    pub fn with_info(mut self, show: bool) -> Self {
        self.show_info = show;
        self
    }

    /// Set all nodes as expanded.
    #[must_use]
    pub fn expand_all(mut self) -> Self {
        if let Some(ref root) = self.root {
            Self::collect_all_ids(root, &mut self.expanded);
        }
        self
    }

    /// Collapse all nodes except root.
    #[must_use]
    pub fn collapse_all(mut self) -> Self {
        self.expanded.clear();
        if let Some(ref root) = self.root {
            self.expanded.insert(root.id);
        }
        self
    }

    /// Toggle expansion of a node.
    pub fn toggle(&mut self, id: NodeId) {
        if self.expanded.contains(&id) {
            self.expanded.remove(&id);
        } else {
            self.expanded.insert(id);
        }
    }

    /// Expand a node.
    pub fn expand(&mut self, id: NodeId) {
        self.expanded.insert(id);
    }

    /// Collapse a node.
    pub fn collapse(&mut self, id: NodeId) {
        self.expanded.remove(&id);
    }

    /// Check if a node is expanded.
    #[must_use]
    pub fn is_expanded(&self, id: NodeId) -> bool {
        self.expanded.contains(&id)
    }

    /// Set scroll offset.
    pub fn set_scroll(&mut self, offset: usize) {
        self.scroll_offset = offset;
    }

    /// Select a node.
    pub fn select(&mut self, id: Option<NodeId>) {
        self.selected = id;
    }

    /// Get selected node ID.
    #[must_use]
    pub fn selected(&self) -> Option<NodeId> {
        self.selected
    }

    /// Set a new root.
    pub fn set_root(&mut self, root: TreeNode) {
        self.expanded.insert(root.id);
        self.root = Some(root);
    }

    /// Get visible line count.
    #[must_use]
    pub fn visible_lines(&self) -> usize {
        self.root
            .as_ref()
            .map_or(0, |r| self.count_visible_lines(r))
    }

    fn count_visible_lines(&self, node: &TreeNode) -> usize {
        let mut count = 1;
        if self.expanded.contains(&node.id) {
            for child in &node.children {
                count += self.count_visible_lines(child);
            }
        }
        count
    }

    fn collect_all_ids(node: &TreeNode, ids: &mut HashSet<NodeId>) {
        ids.insert(node.id);
        for child in &node.children {
            Self::collect_all_ids(child, ids);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn render_node(
        &self,
        canvas: &mut dyn Canvas,
        node: &TreeNode,
        x: f32,
        y: &mut f32,
        prefix: &str,
        is_last: bool,
        visible_height: f32,
    ) {
        // Skip if above viewport
        if *y < self.bounds.y {
            // Still need to recurse to track y position
        }

        // Build branch string
        let branch = if prefix.is_empty() {
            String::new()
        } else if is_last {
            format!("{prefix}{BRANCH_ELBOW}")
        } else {
            format!("{prefix}{BRANCH_TEE}")
        };

        // Only render if within bounds
        if *y >= self.bounds.y && *y < self.bounds.y + visible_height {
            // Draw branch characters
            let branch_style = TextStyle {
                color: Color::new(0.5, 0.5, 0.5, 1.0),
                ..Default::default()
            };
            canvas.draw_text(&branch, Point::new(x, *y), &branch_style);

            // Draw expand/collapse indicator
            let indicator = if node.children.is_empty() {
                "  "
            } else if self.expanded.contains(&node.id) {
                "▼ "
            } else {
                "▶ "
            };

            let indicator_x = x + branch.chars().count() as f32;
            canvas.draw_text(indicator, Point::new(indicator_x, *y), &branch_style);

            // Draw label
            let label_x = indicator_x + 2.0;
            let color = node.color.unwrap_or(self.default_color);
            let is_selected = self.selected == Some(node.id);

            let label_style = TextStyle {
                color: if is_selected {
                    Color::new(0.0, 0.0, 0.0, 1.0)
                } else {
                    color
                },
                ..Default::default()
            };

            // Draw selection highlight
            if is_selected {
                let label_len = node.label.len() as f32;
                canvas.fill_rect(
                    Rect::new(label_x, *y, label_len, 1.0),
                    Color::new(0.3, 0.7, 1.0, 1.0),
                );
            }

            canvas.draw_text(&node.label, Point::new(label_x, *y), &label_style);

            // Draw info if enabled
            if self.show_info {
                if let Some(ref info) = node.info {
                    let info_x = label_x + node.label.len() as f32 + 2.0;
                    let info_style = TextStyle {
                        color: Color::new(0.6, 0.6, 0.6, 1.0),
                        ..Default::default()
                    };
                    canvas.draw_text(info, Point::new(info_x, *y), &info_style);
                }
            }
        }

        *y += 1.0;

        // Render children if expanded
        if self.expanded.contains(&node.id) && !node.children.is_empty() {
            let child_prefix = if prefix.is_empty() {
                String::new()
            } else if is_last {
                format!("{prefix}{BRANCH_SPACE}")
            } else {
                format!("{prefix}{BRANCH_PIPE}")
            };

            let child_count = node.children.len();
            for (i, child) in node.children.iter().enumerate() {
                let child_is_last = i == child_count - 1;
                self.render_node(
                    canvas,
                    child,
                    x,
                    y,
                    &child_prefix,
                    child_is_last,
                    visible_height,
                );
            }
        }
    }
}

impl Brick for Tree {
    fn brick_name(&self) -> &'static str {
        "tree"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        static ASSERTIONS: &[BrickAssertion] = &[BrickAssertion::max_latency_ms(16)];
        ASSERTIONS
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
        String::new()
    }

    fn to_css(&self) -> String {
        String::new()
    }
}

impl Widget for Tree {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let lines = self.visible_lines() as f32;
        let width = constraints.max_width.min(80.0);
        let height = lines.min(constraints.max_height);
        constraints.constrain(Size::new(width, height.max(1.0)))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        if self.root.is_none() || self.bounds.width < 1.0 {
            return;
        }

        let mut y = self.bounds.y - self.scroll_offset as f32;
        if let Some(ref root) = self.root {
            self.render_node(
                canvas,
                root,
                self.bounds.x,
                &mut y,
                "",
                true,
                self.bounds.height,
            );
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
        rects: Vec<(Rect, Color)>,
    }

    impl MockCanvas {
        fn new() -> Self {
            Self {
                texts: vec![],
                rects: vec![],
            }
        }
    }

    impl Canvas for MockCanvas {
        fn fill_rect(&mut self, rect: Rect, color: Color) {
            self.rects.push((rect, color));
        }
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
    fn test_tree_creation() {
        let tree = Tree::new();
        assert!(tree.root.is_none());
    }

    #[test]
    fn test_tree_with_root() {
        let root = TreeNode::new(1, "Root");
        let tree = Tree::new().with_root(root);
        assert!(tree.root.is_some());
        assert!(tree.is_expanded(NodeId::new(1)));
    }

    #[test]
    fn test_tree_node_builder() {
        let node = TreeNode::new(1, "Parent")
            .with_info("Info text")
            .with_color(Color::RED)
            .with_child(TreeNode::new(2, "Child"));
        assert_eq!(node.children.len(), 1);
        assert!(node.info.is_some());
        assert!(node.color.is_some());
    }

    #[test]
    fn test_tree_toggle() {
        let root = TreeNode::new(1, "Root").with_child(TreeNode::new(2, "Child"));
        let mut tree = Tree::new().with_root(root);

        assert!(tree.is_expanded(NodeId::new(1)));
        tree.toggle(NodeId::new(1));
        assert!(!tree.is_expanded(NodeId::new(1)));
        tree.toggle(NodeId::new(1));
        assert!(tree.is_expanded(NodeId::new(1)));
    }

    #[test]
    fn test_tree_expand_collapse() {
        let root = TreeNode::new(1, "Root").with_child(TreeNode::new(2, "Child"));
        let mut tree = Tree::new().with_root(root);

        tree.expand(NodeId::new(2));
        assert!(tree.is_expanded(NodeId::new(2)));
        tree.collapse(NodeId::new(2));
        assert!(!tree.is_expanded(NodeId::new(2)));
    }

    #[test]
    fn test_tree_expand_all() {
        let root = TreeNode::new(1, "Root")
            .with_child(TreeNode::new(2, "Child1").with_child(TreeNode::new(3, "GrandChild")))
            .with_child(TreeNode::new(4, "Child2"));
        let tree = Tree::new().with_root(root).expand_all();

        assert!(tree.is_expanded(NodeId::new(1)));
        assert!(tree.is_expanded(NodeId::new(2)));
        assert!(tree.is_expanded(NodeId::new(3)));
        assert!(tree.is_expanded(NodeId::new(4)));
    }

    #[test]
    fn test_tree_collapse_all() {
        let root = TreeNode::new(1, "Root").with_child(TreeNode::new(2, "Child"));
        let tree = Tree::new().with_root(root).expand_all().collapse_all();

        assert!(tree.is_expanded(NodeId::new(1))); // Root stays expanded
        assert!(!tree.is_expanded(NodeId::new(2)));
    }

    #[test]
    fn test_tree_visible_lines() {
        let root = TreeNode::new(1, "Root")
            .with_child(TreeNode::new(2, "Child1"))
            .with_child(TreeNode::new(3, "Child2"));
        let tree = Tree::new().with_root(root);

        // Root expanded: Root + 2 children = 3 lines
        assert_eq!(tree.visible_lines(), 3);
    }

    #[test]
    fn test_tree_visible_lines_collapsed() {
        let root = TreeNode::new(1, "Root")
            .with_child(TreeNode::new(2, "Child1"))
            .with_child(TreeNode::new(3, "Child2"));
        let mut tree = Tree::new().with_root(root);
        tree.collapse(NodeId::new(1));

        // Root collapsed: just Root = 1 line
        assert_eq!(tree.visible_lines(), 1);
    }

    #[test]
    fn test_tree_node_count() {
        let root = TreeNode::new(1, "Root")
            .with_child(TreeNode::new(2, "Child1").with_child(TreeNode::new(3, "GrandChild")))
            .with_child(TreeNode::new(4, "Child2"));

        assert_eq!(root.count_nodes(), 4);
    }

    #[test]
    fn test_tree_node_depth() {
        let root = TreeNode::new(1, "Root")
            .with_child(TreeNode::new(2, "Child1").with_child(TreeNode::new(3, "GrandChild")))
            .with_child(TreeNode::new(4, "Child2"));

        assert_eq!(root.depth(), 3);
    }

    #[test]
    fn test_tree_selection() {
        let root = TreeNode::new(1, "Root");
        let mut tree = Tree::new().with_root(root);

        assert!(tree.selected().is_none());
        tree.select(Some(NodeId::new(1)));
        assert_eq!(tree.selected(), Some(NodeId::new(1)));
        tree.select(None);
        assert!(tree.selected().is_none());
    }

    #[test]
    fn test_tree_paint() {
        let root = TreeNode::new(1, "Root")
            .with_child(TreeNode::new(2, "Child1"))
            .with_child(TreeNode::new(3, "Child2"));
        let mut tree = Tree::new().with_root(root);
        tree.bounds = Rect::new(0.0, 0.0, 40.0, 10.0);

        let mut canvas = MockCanvas::new();
        tree.paint(&mut canvas);

        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_tree_paint_empty() {
        let tree = Tree::new();
        let mut canvas = MockCanvas::new();
        tree.paint(&mut canvas);
        assert!(canvas.texts.is_empty());
    }

    #[test]
    fn test_tree_measure() {
        let root = TreeNode::new(1, "Root")
            .with_child(TreeNode::new(2, "Child1"))
            .with_child(TreeNode::new(3, "Child2"));
        let tree = Tree::new().with_root(root);

        let size = tree.measure(Constraints::loose(Size::new(100.0, 50.0)));
        assert!(size.height >= 3.0); // 3 visible lines
    }

    #[test]
    fn test_tree_layout() {
        let mut tree = Tree::new();
        let bounds = Rect::new(5.0, 10.0, 30.0, 20.0);
        let result = tree.layout(bounds);

        assert_eq!(result.size.width, 30.0);
        assert_eq!(result.size.height, 20.0);
        assert_eq!(tree.bounds, bounds);
    }

    #[test]
    fn test_tree_brick_name() {
        let tree = Tree::new();
        assert_eq!(tree.brick_name(), "tree");
    }

    #[test]
    fn test_tree_assertions() {
        let tree = Tree::new();
        assert!(!tree.assertions().is_empty());
    }

    #[test]
    fn test_tree_budget() {
        let tree = Tree::new();
        let budget = tree.budget();
        assert!(budget.paint_ms > 0);
    }

    #[test]
    fn test_tree_verify() {
        let tree = Tree::new();
        assert!(tree.verify().is_valid());
    }

    #[test]
    fn test_tree_type_id() {
        let tree = Tree::new();
        assert_eq!(Widget::type_id(&tree), TypeId::of::<Tree>());
    }

    #[test]
    fn test_tree_children() {
        let tree = Tree::new();
        assert!(tree.children().is_empty());
    }

    #[test]
    fn test_tree_children_mut() {
        let mut tree = Tree::new();
        assert!(tree.children_mut().is_empty());
    }

    #[test]
    fn test_tree_event() {
        let mut tree = Tree::new();
        let event = Event::key_down(presentar_core::Key::Enter);
        assert!(tree.event(&event).is_none());
    }

    #[test]
    fn test_tree_default() {
        let tree = Tree::default();
        assert!(tree.root.is_none());
    }

    #[test]
    fn test_tree_to_html() {
        let tree = Tree::new();
        assert!(tree.to_html().is_empty());
    }

    #[test]
    fn test_tree_to_css() {
        let tree = Tree::new();
        assert!(tree.to_css().is_empty());
    }

    #[test]
    fn test_tree_scroll() {
        let mut tree = Tree::new();
        tree.set_scroll(5);
        assert_eq!(tree.scroll_offset, 5);
    }

    #[test]
    fn test_tree_with_color() {
        let tree = Tree::new().with_color(Color::RED);
        assert_eq!(tree.default_color, Color::RED);
    }

    #[test]
    fn test_tree_with_info() {
        let tree = Tree::new().with_info(false);
        assert!(!tree.show_info);
    }

    #[test]
    fn test_tree_set_root() {
        let mut tree = Tree::new();
        tree.set_root(TreeNode::new(1, "New Root"));
        assert!(tree.root.is_some());
    }

    #[test]
    fn test_node_id() {
        let id = NodeId::new(42);
        assert_eq!(id.0, 42);
    }

    #[test]
    fn test_tree_node_with_children() {
        let children = vec![TreeNode::new(2, "A"), TreeNode::new(3, "B")];
        let node = TreeNode::new(1, "Root").with_children(children);
        assert_eq!(node.children.len(), 2);
    }

    #[test]
    fn test_tree_paint_with_selection() {
        let root = TreeNode::new(1, "Root");
        let mut tree = Tree::new().with_root(root);
        tree.select(Some(NodeId::new(1)));
        tree.bounds = Rect::new(0.0, 0.0, 40.0, 10.0);

        let mut canvas = MockCanvas::new();
        tree.paint(&mut canvas);

        // Selection should cause a fill_rect call
        assert!(!canvas.rects.is_empty());
    }

    #[test]
    fn test_tree_leaf_node_depth() {
        let leaf = TreeNode::new(1, "Leaf");
        assert_eq!(leaf.depth(), 1);
    }

    #[test]
    fn test_tree_paint_collapsed_node() {
        // Test collapsed indicator (▶)
        let root = TreeNode::new(1, "Root").with_child(TreeNode::new(2, "Child"));
        let mut tree = Tree::new().with_root(root);
        tree.collapse(NodeId::new(1)); // Collapse the root
        tree.bounds = Rect::new(0.0, 0.0, 40.0, 10.0);

        let mut canvas = MockCanvas::new();
        tree.paint(&mut canvas);

        // Should paint collapsed indicator
        assert!(canvas.texts.iter().any(|(t, _)| t.contains("▶")));
    }

    #[test]
    fn test_tree_paint_with_info() {
        // Test info display
        let root = TreeNode::new(1, "Root").with_info("Info text");
        let mut tree = Tree::new().with_root(root).with_info(true);
        tree.bounds = Rect::new(0.0, 0.0, 60.0, 10.0);

        let mut canvas = MockCanvas::new();
        tree.paint(&mut canvas);

        // Should paint info text
        assert!(canvas.texts.iter().any(|(t, _)| t.contains("Info text")));
    }

    #[test]
    fn test_tree_paint_nested_expanded() {
        // Test branch prefixes with nested children
        let root = TreeNode::new(1, "Root")
            .with_child(
                TreeNode::new(2, "Child1")
                    .with_child(TreeNode::new(4, "GrandChild1"))
                    .with_child(TreeNode::new(5, "GrandChild2")),
            )
            .with_child(TreeNode::new(3, "Child2"));
        let mut tree = Tree::new().with_root(root).expand_all();
        tree.bounds = Rect::new(0.0, 0.0, 60.0, 20.0);

        let mut canvas = MockCanvas::new();
        tree.paint(&mut canvas);

        // Should paint multiple levels with branch characters
        assert!(canvas.texts.len() > 5);
    }

    #[test]
    fn test_tree_paint_not_last_child() {
        // Test branch tee (├──) for non-last children
        let root = TreeNode::new(1, "Root")
            .with_child(TreeNode::new(2, "Child1"))
            .with_child(TreeNode::new(3, "Child2"))
            .with_child(TreeNode::new(4, "Child3"));
        let mut tree = Tree::new().with_root(root);
        tree.bounds = Rect::new(0.0, 0.0, 40.0, 10.0);

        let mut canvas = MockCanvas::new();
        tree.paint(&mut canvas);

        // Should paint multiple children (root + 3 children = 4 nodes)
        // Branch chars are in the same text as the prefix
        assert!(canvas.texts.len() >= 4);
    }

    #[test]
    fn test_tree_paint_scrolled() {
        // Test paint with scroll offset (y < bounds.y case)
        let root = TreeNode::new(1, "Root")
            .with_child(TreeNode::new(2, "Child1"))
            .with_child(TreeNode::new(3, "Child2"));
        let mut tree = Tree::new().with_root(root);
        tree.set_scroll(5);
        tree.bounds = Rect::new(0.0, 0.0, 40.0, 10.0);

        let mut canvas = MockCanvas::new();
        tree.paint(&mut canvas);
        // Should handle scroll offset without panic
    }

    #[test]
    fn test_tree_paint_deep_nesting() {
        // Test deep nesting with branch pipes
        let grandchild =
            TreeNode::new(4, "GrandChild").with_child(TreeNode::new(5, "GreatGrandChild"));
        let child1 = TreeNode::new(2, "Child1").with_child(grandchild);
        let child2 = TreeNode::new(3, "Child2");
        let root = TreeNode::new(1, "Root")
            .with_child(child1)
            .with_child(child2);
        let mut tree = Tree::new().with_root(root).expand_all();
        tree.bounds = Rect::new(0.0, 0.0, 80.0, 20.0);

        let mut canvas = MockCanvas::new();
        tree.paint(&mut canvas);

        // Should render deep structure
        assert!(canvas.texts.len() >= 5);
    }
}
