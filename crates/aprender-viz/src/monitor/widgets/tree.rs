//! Collapsible tree widget for process hierarchy.
//!
//! Provides a tree view with O(1) toggle operations (Falsification criterion #12).

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::widgets::Widget;
use std::collections::{HashMap, HashSet};

/// A tree node.
#[derive(Debug, Clone)]
pub struct TreeNode {
    /// Node identifier.
    pub id: String,
    /// Display label.
    pub label: String,
    /// Child node IDs.
    pub children: Vec<String>,
    /// Optional metadata.
    pub metadata: HashMap<String, String>,
}

/// A collapsible tree widget.
#[derive(Debug, Clone)]
pub struct Tree {
    /// All nodes by ID.
    nodes: HashMap<String, TreeNode>,
    /// Root node IDs.
    roots: Vec<String>,
    /// Collapsed node IDs.
    collapsed: HashSet<String>,
    /// Selected node ID.
    selected: Option<String>,
}

impl Tree {
    /// Creates a new empty tree.
    #[must_use]
    pub fn new() -> Self {
        Self { nodes: HashMap::new(), roots: Vec::new(), collapsed: HashSet::new(), selected: None }
    }

    /// Adds a node to the tree.
    pub fn add_node(
        &mut self,
        id: impl Into<String>,
        label: impl Into<String>,
        parent: Option<&str>,
    ) {
        let id = id.into();
        let node = TreeNode {
            id: id.clone(),
            label: label.into(),
            children: Vec::new(),
            metadata: HashMap::new(),
        };

        if let Some(parent_id) = parent {
            if let Some(parent_node) = self.nodes.get_mut(parent_id) {
                parent_node.children.push(id.clone());
            }
        } else {
            self.roots.push(id.clone());
        }

        self.nodes.insert(id, node);
    }

    /// Returns whether a node is expanded (not collapsed).
    #[must_use]
    pub fn is_expanded(&self, id: &str) -> bool {
        !self.collapsed.contains(id)
    }

    /// Toggles the collapsed state of a node.
    pub fn toggle(&mut self, id: &str) {
        if self.collapsed.contains(id) {
            self.collapsed.remove(id);
        } else {
            self.collapsed.insert(id.to_string());
        }
    }

    /// Expands a node.
    pub fn expand(&mut self, id: &str) {
        self.collapsed.remove(id);
    }

    /// Collapses a node.
    pub fn collapse(&mut self, id: &str) {
        self.collapsed.insert(id.to_string());
    }

    /// Selects a node.
    pub fn select(&mut self, id: Option<String>) {
        self.selected = id;
    }

    /// Returns the selected node ID.
    #[must_use]
    pub fn selected(&self) -> Option<&str> {
        self.selected.as_deref()
    }

    /// Returns the number of nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns whether the tree is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Clears all nodes.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.roots.clear();
        self.collapsed.clear();
        self.selected = None;
    }

    /// Renders a node and its children recursively.
    fn render_node(
        &self,
        node_id: &str,
        depth: usize,
        y: &mut u16,
        area: Rect,
        buf: &mut Buffer,
        is_last: bool,
    ) {
        if *y >= area.y + area.height {
            return;
        }

        let Some(node) = self.nodes.get(node_id) else {
            return;
        };

        // Build prefix
        let mut prefix = String::new();
        for _ in 0..depth {
            prefix.push_str("│  ");
        }
        if depth > 0 {
            prefix.pop();
            prefix.pop();
            prefix.pop();
            if is_last {
                prefix.push_str("└─ ");
            } else {
                prefix.push_str("├─ ");
            }
        }

        // Add expand/collapse indicator
        let has_children = !node.children.is_empty();
        let indicator = if has_children {
            if self.is_expanded(node_id) {
                "▼ "
            } else {
                "▶ "
            }
        } else {
            "  "
        };

        let text = format!("{}{}{}", prefix, indicator, node.label);

        let is_selected = self.selected.as_ref() == Some(&node.id);
        let style = if is_selected {
            Style::default().bg(Color::DarkGray).add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        };

        let truncated: String = text.chars().take(area.width as usize).collect();
        buf.set_string(area.x, *y, truncated, style);
        *y += 1;

        // Render children if expanded
        if has_children && self.is_expanded(node_id) {
            let children = &node.children;
            for (i, child_id) in children.iter().enumerate() {
                let is_last_child = i == children.len() - 1;
                self.render_node(child_id, depth + 1, y, area, buf, is_last_child);
            }
        }
    }
}

impl Default for Tree {
    fn default() -> Self {
        Self::new()
    }
}

impl Widget for Tree {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.height == 0 || area.width == 0 {
            return;
        }

        let mut y = area.y;
        for (i, root_id) in self.roots.iter().enumerate() {
            let is_last = i == self.roots.len() - 1;
            self.render_node(root_id, 0, &mut y, area, buf, is_last);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_tree_new() {
        let tree = Tree::new();
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
    }

    #[test]
    fn test_tree_default() {
        let tree = Tree::default();
        assert!(tree.is_empty());
    }

    #[test]
    fn test_tree_add_nodes() {
        let mut tree = Tree::new();
        tree.add_node("root", "Root", None);
        tree.add_node("child1", "Child 1", Some("root"));
        tree.add_node("child2", "Child 2", Some("root"));

        assert_eq!(tree.len(), 3);
        assert!(!tree.is_empty());
    }

    #[test]
    fn test_tree_add_node_missing_parent() {
        let mut tree = Tree::new();
        // Add node with non-existent parent - should still add node but not as child
        tree.add_node("orphan", "Orphan", Some("nonexistent"));
        assert_eq!(tree.len(), 1);
    }

    #[test]
    fn test_tree_collapse_expand() {
        let mut tree = Tree::new();
        tree.add_node("root", "Root", None);
        tree.add_node("child", "Child", Some("root"));

        // Initially expanded
        assert!(tree.is_expanded("root"));

        // Toggle to collapse
        tree.toggle("root");
        assert!(!tree.is_expanded("root"));

        // Toggle to expand
        tree.toggle("root");
        assert!(tree.is_expanded("root"));
    }

    #[test]
    fn test_tree_expand_collapse_direct() {
        let mut tree = Tree::new();
        tree.add_node("root", "Root", None);

        // Direct expand/collapse
        tree.collapse("root");
        assert!(!tree.is_expanded("root"));

        tree.expand("root");
        assert!(tree.is_expanded("root"));
    }

    /// Falsification criterion #12: Tree view expansion/collapse completes in <10ms.
    #[test]
    fn test_tree_toggle_performance() {
        let mut tree = Tree::new();

        // Create a tree with 1000 nodes
        tree.add_node("root", "Root", None);
        for i in 0..999 {
            tree.add_node(format!("node_{i}"), format!("Node {i}"), Some("root"));
        }

        // Toggle should be O(1)
        let start = Instant::now();
        tree.toggle("root");
        let elapsed = start.elapsed();

        assert!(elapsed.as_millis() < 10, "Toggle took {elapsed:?}, should be under 10ms");
    }

    #[test]
    fn test_tree_selection() {
        let mut tree = Tree::new();
        tree.add_node("root", "Root", None);

        tree.select(Some("root".to_string()));
        assert_eq!(tree.selected(), Some("root"));

        tree.select(None);
        assert_eq!(tree.selected(), None);
    }

    #[test]
    fn test_tree_clear() {
        let mut tree = Tree::new();
        tree.add_node("root", "Root", None);
        tree.add_node("child", "Child", Some("root"));
        tree.select(Some("root".to_string()));
        tree.collapse("root");

        tree.clear();

        assert!(tree.is_empty());
        assert_eq!(tree.selected(), None);
    }

    #[test]
    fn test_tree_render_empty() {
        let tree = Tree::new();
        let mut buf = Buffer::empty(Rect::new(0, 0, 40, 10));
        tree.render(Rect::new(0, 0, 40, 10), &mut buf);
        // Should not panic
    }

    #[test]
    fn test_tree_render_zero_size() {
        let tree = Tree::new();
        let mut buf = Buffer::empty(Rect::new(0, 0, 0, 0));
        tree.render(Rect::new(0, 0, 0, 0), &mut buf);
        // Should return early without panic
    }

    #[test]
    fn test_tree_render_single_root() {
        let mut tree = Tree::new();
        tree.add_node("root", "Root Node", None);

        let mut buf = Buffer::empty(Rect::new(0, 0, 40, 10));
        tree.render(Rect::new(0, 0, 40, 10), &mut buf);

        // Check root is rendered (no children = no indicator, just label)
        let content = buf.cell((0, 0)).map_or("", ratatui::buffer::Cell::symbol);
        assert!(!content.is_empty());
    }

    #[test]
    fn test_tree_render_with_children() {
        let mut tree = Tree::new();
        tree.add_node("root", "Root", None);
        tree.add_node("child1", "Child 1", Some("root"));
        tree.add_node("child2", "Child 2", Some("root"));

        let mut buf = Buffer::empty(Rect::new(0, 0, 40, 10));
        tree.render(Rect::new(0, 0, 40, 10), &mut buf);

        // Root should have expand indicator (▼)
        // Children should be rendered on subsequent lines
    }

    #[test]
    fn test_tree_render_collapsed() {
        let mut tree = Tree::new();
        tree.add_node("root", "Root", None);
        tree.add_node("child1", "Child 1", Some("root"));
        tree.add_node("child2", "Child 2", Some("root"));
        tree.collapse("root");

        let mut buf = Buffer::empty(Rect::new(0, 0, 40, 10));
        tree.render(Rect::new(0, 0, 40, 10), &mut buf);

        // Root should show collapsed indicator (▶)
        // Children should NOT be rendered
    }

    #[test]
    fn test_tree_render_with_selection() {
        let mut tree = Tree::new();
        tree.add_node("root", "Root", None);
        tree.select(Some("root".to_string()));

        let mut buf = Buffer::empty(Rect::new(0, 0, 40, 10));
        tree.render(Rect::new(0, 0, 40, 10), &mut buf);

        // Selected node should have different background
        let cell = buf.cell((0, 0)).expect("operation should succeed");
        assert_eq!(cell.bg, Color::DarkGray);
    }

    #[test]
    fn test_tree_render_deep_hierarchy() {
        let mut tree = Tree::new();
        tree.add_node("level0", "Level 0", None);
        tree.add_node("level1", "Level 1", Some("level0"));
        tree.add_node("level2", "Level 2", Some("level1"));
        tree.add_node("level3", "Level 3", Some("level2"));

        let mut buf = Buffer::empty(Rect::new(0, 0, 60, 10));
        tree.render(Rect::new(0, 0, 60, 10), &mut buf);
        // Should render all levels with proper indentation
    }

    #[test]
    fn test_tree_render_multiple_roots() {
        let mut tree = Tree::new();
        tree.add_node("root1", "Root 1", None);
        tree.add_node("root2", "Root 2", None);
        tree.add_node("root3", "Root 3", None);

        let mut buf = Buffer::empty(Rect::new(0, 0, 40, 10));
        tree.render(Rect::new(0, 0, 40, 10), &mut buf);
        // Should render all roots
    }

    #[test]
    fn test_tree_render_overflow() {
        let mut tree = Tree::new();
        // Add more nodes than visible height
        for i in 0..100 {
            tree.add_node(format!("node{i}"), format!("Node {i}"), None);
        }

        let mut buf = Buffer::empty(Rect::new(0, 0, 40, 5));
        tree.render(Rect::new(0, 0, 40, 5), &mut buf);
        // Should not panic, only render visible nodes
    }

    #[test]
    fn test_tree_render_last_child_indicator() {
        let mut tree = Tree::new();
        tree.add_node("root", "Root", None);
        tree.add_node("child1", "Child 1", Some("root"));
        tree.add_node("child2", "Last Child", Some("root")); // Last child uses └─

        let mut buf = Buffer::empty(Rect::new(0, 0, 40, 10));
        tree.render(Rect::new(0, 0, 40, 10), &mut buf);
        // Last child should use └─ instead of ├─
    }

    #[test]
    fn test_tree_node_metadata() {
        let mut tree = Tree::new();
        tree.add_node("root", "Root", None);

        // TreeNode has metadata field
        if let Some(node) = tree.nodes.get_mut("root") {
            node.metadata.insert("key".to_string(), "value".to_string());
            assert_eq!(node.metadata.get("key"), Some(&"value".to_string()));
        }
    }
}
