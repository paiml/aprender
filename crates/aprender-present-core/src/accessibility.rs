#![allow(clippy::unwrap_used, clippy::disallowed_methods)]
//! Accessibility support for screen readers and assistive technology.
//!
//! This module provides:
//! - Hit testing for finding accessible elements at a point
//! - Accessibility tree building for screen reader navigation
//! - Focus order calculation
//! - ARIA-like properties for widgets

use crate::geometry::{Point, Rect};
use crate::widget::{AccessibleRole, WidgetId};
use std::collections::HashMap;

/// An accessible element in the accessibility tree.
#[derive(Debug, Clone)]
pub struct AccessibleNode {
    /// Unique identifier.
    pub id: AccessibleNodeId,
    /// Widget ID if associated with a widget.
    pub widget_id: Option<WidgetId>,
    /// Accessible name (label).
    pub name: Option<String>,
    /// Accessible description.
    pub description: Option<String>,
    /// Accessible role.
    pub role: AccessibleRole,
    /// Bounding rectangle.
    pub bounds: Rect,
    /// Whether the element is focusable.
    pub focusable: bool,
    /// Whether the element is currently focused.
    pub focused: bool,
    /// Whether the element is enabled.
    pub enabled: bool,
    /// Whether the element is visible.
    pub visible: bool,
    /// Whether the element is expanded (for expandable elements).
    pub expanded: Option<bool>,
    /// Whether the element is checked (for checkboxes/radios).
    pub checked: Option<CheckedState>,
    /// Current value (for sliders, inputs).
    pub value: Option<String>,
    /// Minimum value (for sliders).
    pub value_min: Option<f64>,
    /// Maximum value (for sliders).
    pub value_max: Option<f64>,
    /// Value text (human-readable value).
    pub value_text: Option<String>,
    /// Children node IDs.
    pub children: Vec<AccessibleNodeId>,
    /// Parent node ID.
    pub parent: Option<AccessibleNodeId>,
    /// Tab index for focus order (-1 = not focusable, 0 = natural order, >0 = explicit order).
    pub tab_index: i32,
    /// Level in heading hierarchy (1-6 for headings).
    pub level: Option<u8>,
    /// Live region type for dynamic content.
    pub live: LiveRegion,
    /// Custom properties.
    pub properties: HashMap<String, String>,
}

impl AccessibleNode {
    /// Create a new accessible node.
    pub fn new(id: AccessibleNodeId, role: AccessibleRole, bounds: Rect) -> Self {
        Self {
            id,
            widget_id: None,
            name: None,
            description: None,
            role,
            bounds,
            focusable: false,
            focused: false,
            enabled: true,
            visible: true,
            expanded: None,
            checked: None,
            value: None,
            value_min: None,
            value_max: None,
            value_text: None,
            children: Vec::new(),
            parent: None,
            tab_index: -1,
            level: None,
            live: LiveRegion::Off,
            properties: HashMap::new(),
        }
    }

    /// Create a new button node.
    pub fn button(id: AccessibleNodeId, name: &str, bounds: Rect) -> Self {
        let mut node = Self::new(id, AccessibleRole::Button, bounds);
        node.name = Some(name.to_string());
        node.focusable = true;
        node.tab_index = 0;
        node
    }

    /// Create a new checkbox node.
    pub fn checkbox(id: AccessibleNodeId, name: &str, checked: bool, bounds: Rect) -> Self {
        let mut node = Self::new(id, AccessibleRole::Checkbox, bounds);
        node.name = Some(name.to_string());
        node.focusable = true;
        node.tab_index = 0;
        node.checked = Some(if checked {
            CheckedState::Checked
        } else {
            CheckedState::Unchecked
        });
        node
    }

    /// Create a new text input node.
    pub fn text_input(id: AccessibleNodeId, label: &str, value: &str, bounds: Rect) -> Self {
        let mut node = Self::new(id, AccessibleRole::TextInput, bounds);
        node.name = Some(label.to_string());
        node.value = Some(value.to_string());
        node.focusable = true;
        node.tab_index = 0;
        node
    }

    /// Create a new heading node.
    pub fn heading(id: AccessibleNodeId, text: &str, level: u8, bounds: Rect) -> Self {
        let mut node = Self::new(id, AccessibleRole::Heading, bounds);
        node.name = Some(text.to_string());
        node.level = Some(level.min(6).max(1));
        node
    }

    /// Create a new slider node.
    pub fn slider(
        id: AccessibleNodeId,
        name: &str,
        value: f64,
        min: f64,
        max: f64,
        bounds: Rect,
    ) -> Self {
        let mut node = Self::new(id, AccessibleRole::Slider, bounds);
        node.name = Some(name.to_string());
        node.value = Some(value.to_string());
        node.value_min = Some(min);
        node.value_max = Some(max);
        node.focusable = true;
        node.tab_index = 0;
        node
    }

    /// Check if this node contains the given point.
    pub fn contains_point(&self, point: Point) -> bool {
        self.visible && self.bounds.contains_point(&point)
    }

    /// Set the node's name.
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    /// Set the node's description.
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }

    /// Set the node as focusable.
    pub fn with_focusable(mut self, focusable: bool) -> Self {
        self.focusable = focusable;
        if focusable && self.tab_index < 0 {
            self.tab_index = 0;
        }
        self
    }

    /// Set a custom property.
    pub fn with_property(mut self, key: &str, value: &str) -> Self {
        self.properties.insert(key.to_string(), value.to_string());
        self
    }
}

/// Unique identifier for an accessible node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct AccessibleNodeId(pub u64);

impl AccessibleNodeId {
    /// Create a new node ID.
    pub const fn new(id: u64) -> Self {
        Self(id)
    }
}

/// Checked state for checkboxes and radio buttons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckedState {
    /// Not checked.
    Unchecked,
    /// Checked.
    Checked,
    /// Mixed/indeterminate state.
    Mixed,
}

/// Live region type for dynamic content updates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LiveRegion {
    /// No live region.
    #[default]
    Off,
    /// Polite - announce when user is idle.
    Polite,
    /// Assertive - announce immediately.
    Assertive,
}

/// Accessibility tree for hit testing and navigation.
#[derive(Debug, Default)]
pub struct AccessibilityTree {
    /// All nodes in the tree.
    nodes: HashMap<AccessibleNodeId, AccessibleNode>,
    /// Root node ID.
    root: Option<AccessibleNodeId>,
    /// Next available node ID.
    next_id: u64,
    /// Current focus node.
    focus: Option<AccessibleNodeId>,
    /// Focus order (computed lazily).
    focus_order: Vec<AccessibleNodeId>,
    /// Whether focus order needs recomputation.
    focus_order_dirty: bool,
}

impl AccessibilityTree {
    /// Create a new empty accessibility tree.
    pub fn new() -> Self {
        Self::default()
    }

    /// Generate a new unique node ID.
    pub fn next_id(&mut self) -> AccessibleNodeId {
        let id = AccessibleNodeId::new(self.next_id);
        self.next_id += 1;
        id
    }

    /// Set the root node.
    pub fn set_root(&mut self, id: AccessibleNodeId) {
        self.root = Some(id);
    }

    /// Get the root node.
    pub fn root(&self) -> Option<&AccessibleNode> {
        self.root.and_then(|id| self.nodes.get(&id))
    }

    /// Get a node by ID.
    pub fn get(&self, id: AccessibleNodeId) -> Option<&AccessibleNode> {
        self.nodes.get(&id)
    }

    /// Get a mutable node by ID.
    pub fn get_mut(&mut self, id: AccessibleNodeId) -> Option<&mut AccessibleNode> {
        self.nodes.get_mut(&id)
    }

    /// Insert a node into the tree.
    pub fn insert(&mut self, node: AccessibleNode) {
        self.focus_order_dirty = true;
        self.nodes.insert(node.id, node);
    }

    /// Remove a node from the tree.
    pub fn remove(&mut self, id: AccessibleNodeId) -> Option<AccessibleNode> {
        self.focus_order_dirty = true;
        self.nodes.remove(&id)
    }

    /// Get the number of nodes in the tree.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Clear all nodes from the tree.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.root = None;
        self.focus = None;
        self.focus_order.clear();
        self.focus_order_dirty = false;
    }

    /// Get the currently focused node.
    pub fn focused(&self) -> Option<&AccessibleNode> {
        self.focus.and_then(|id| self.nodes.get(&id))
    }

    /// Get the currently focused node ID.
    pub fn focused_id(&self) -> Option<AccessibleNodeId> {
        self.focus
    }

    /// Set focus to a node.
    pub fn set_focus(&mut self, id: AccessibleNodeId) -> bool {
        if let Some(node) = self.nodes.get(&id) {
            if node.focusable && node.enabled && node.visible {
                // Clear old focus
                if let Some(old_id) = self.focus {
                    if let Some(old_node) = self.nodes.get_mut(&old_id) {
                        old_node.focused = false;
                    }
                }

                // Set new focus
                if let Some(new_node) = self.nodes.get_mut(&id) {
                    new_node.focused = true;
                    self.focus = Some(id);
                    return true;
                }
            }
        }
        false
    }

    /// Clear focus.
    pub fn clear_focus(&mut self) {
        if let Some(id) = self.focus.take() {
            if let Some(node) = self.nodes.get_mut(&id) {
                node.focused = false;
            }
        }
    }

    /// Move focus to the next focusable element.
    pub fn focus_next(&mut self) -> Option<AccessibleNodeId> {
        self.ensure_focus_order();

        if self.focus_order.is_empty() {
            return None;
        }

        let current_idx = self
            .focus
            .and_then(|id| self.focus_order.iter().position(|&fid| fid == id));

        let next_idx = match current_idx {
            Some(idx) => (idx + 1) % self.focus_order.len(),
            None => 0,
        };

        let next_id = self.focus_order[next_idx];
        self.set_focus(next_id);
        Some(next_id)
    }

    /// Move focus to the previous focusable element.
    pub fn focus_previous(&mut self) -> Option<AccessibleNodeId> {
        self.ensure_focus_order();

        if self.focus_order.is_empty() {
            return None;
        }

        let current_idx = self
            .focus
            .and_then(|id| self.focus_order.iter().position(|&fid| fid == id));

        let prev_idx = match current_idx {
            Some(idx) if idx > 0 => idx - 1,
            Some(_) => self.focus_order.len() - 1,
            None => self.focus_order.len() - 1,
        };

        let prev_id = self.focus_order[prev_idx];
        self.set_focus(prev_id);
        Some(prev_id)
    }

    /// Ensure focus order is computed.
    fn ensure_focus_order(&mut self) {
        if self.focus_order_dirty {
            self.compute_focus_order();
        }
    }

    /// Compute focus order based on tab indices and DOM order.
    fn compute_focus_order(&mut self) {
        let mut focusable: Vec<_> = self
            .nodes
            .values()
            .filter(|n| n.focusable && n.enabled && n.visible && n.tab_index >= 0)
            .map(|n| (n.id, n.tab_index, n.bounds.y, n.bounds.x))
            .collect();

        // Sort by tab_index (positive first, then by position)
        focusable.sort_by(|a, b| {
            match (a.1, b.1) {
                (0, 0) => {
                    // Both natural order - sort by position (top-to-bottom, left-to-right)
                    a.2.partial_cmp(&b.2)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal))
                }
                (0, _) => std::cmp::Ordering::Greater, // 0 comes after positive
                (_, 0) => std::cmp::Ordering::Less,
                _ => a.1.cmp(&b.1), // Compare positive tab indices
            }
        });

        self.focus_order = focusable.into_iter().map(|(id, _, _, _)| id).collect();
        self.focus_order_dirty = false;
    }

    /// Get the focus order.
    pub fn get_focus_order(&mut self) -> &[AccessibleNodeId] {
        self.ensure_focus_order();
        &self.focus_order
    }
}

/// Hit tester for finding accessible elements at a point.
#[derive(Debug, Default)]
pub struct HitTester {
    /// The accessibility tree to test against.
    tree: AccessibilityTree,
}

impl HitTester {
    /// Create a new hit tester.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a hit tester with an existing tree.
    pub fn with_tree(tree: AccessibilityTree) -> Self {
        Self { tree }
    }

    /// Get the underlying tree.
    pub fn tree(&self) -> &AccessibilityTree {
        &self.tree
    }

    /// Get mutable access to the tree.
    pub fn tree_mut(&mut self) -> &mut AccessibilityTree {
        &mut self.tree
    }

    /// Find the deepest accessible element at the given point.
    pub fn hit_test(&self, point: Point) -> Option<&AccessibleNode> {
        self.hit_test_id(point).and_then(|id| self.tree.get(id))
    }

    /// Find the ID of the deepest accessible element at the given point.
    pub fn hit_test_id(&self, point: Point) -> Option<AccessibleNodeId> {
        let root_id = self.tree.root?;
        self.hit_test_recursive(root_id, point)
    }

    /// Recursive hit test implementation.
    fn hit_test_recursive(
        &self,
        node_id: AccessibleNodeId,
        point: Point,
    ) -> Option<AccessibleNodeId> {
        let node = self.tree.get(node_id)?;

        if !node.contains_point(point) {
            return None;
        }

        // Check children in reverse order (last drawn = topmost)
        for &child_id in node.children.iter().rev() {
            if let Some(hit_id) = self.hit_test_recursive(child_id, point) {
                return Some(hit_id);
            }
        }

        // Return this node if no child was hit
        Some(node_id)
    }

    /// Find all accessible elements at the given point (from topmost to root).
    pub fn hit_test_all(&self, point: Point) -> Vec<&AccessibleNode> {
        let ids = self.hit_test_all_ids(point);
        ids.iter().filter_map(|&id| self.tree.get(id)).collect()
    }

    /// Find all element IDs at the given point.
    pub fn hit_test_all_ids(&self, point: Point) -> Vec<AccessibleNodeId> {
        let mut results = Vec::new();
        if let Some(root_id) = self.tree.root {
            self.hit_test_all_recursive(root_id, point, &mut results);
        }
        results.reverse(); // Topmost first
        results
    }

    /// Recursive hit test that collects all hits.
    fn hit_test_all_recursive(
        &self,
        node_id: AccessibleNodeId,
        point: Point,
        results: &mut Vec<AccessibleNodeId>,
    ) {
        let Some(node) = self.tree.get(node_id) else {
            return;
        };

        if !node.contains_point(point) {
            return;
        }

        results.push(node_id);

        for &child_id in &node.children {
            self.hit_test_all_recursive(child_id, point, results);
        }
    }

    /// Find the first focusable element at the given point.
    pub fn hit_test_focusable(&self, point: Point) -> Option<&AccessibleNode> {
        let ids = self.hit_test_all_ids(point);
        ids.into_iter()
            .filter_map(|id| self.tree.get(id))
            .find(|node| node.focusable && node.enabled)
    }

    /// Find all elements with a specific role at the given point.
    pub fn hit_test_role(&self, point: Point, role: AccessibleRole) -> Vec<&AccessibleNode> {
        self.hit_test_all(point)
            .into_iter()
            .filter(|node| node.role == role)
            .collect()
    }
}

/// Builder for constructing accessibility trees.
#[derive(Debug)]
pub struct AccessibilityTreeBuilder {
    tree: AccessibilityTree,
    current_parent: Option<AccessibleNodeId>,
}

impl AccessibilityTreeBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            tree: AccessibilityTree::new(),
            current_parent: None,
        }
    }

    /// Add a root node.
    pub fn root(mut self, role: AccessibleRole, bounds: Rect) -> Self {
        let id = self.tree.next_id();
        let node = AccessibleNode::new(id, role, bounds);
        self.tree.insert(node);
        self.tree.set_root(id);
        self.current_parent = Some(id);
        self
    }

    /// Add a child node to the current parent.
    pub fn child(mut self, role: AccessibleRole, bounds: Rect) -> (Self, AccessibleNodeId) {
        let id = self.tree.next_id();
        let mut node = AccessibleNode::new(id, role, bounds);
        node.parent = self.current_parent;

        if let Some(parent_id) = self.current_parent {
            if let Some(parent) = self.tree.get_mut(parent_id) {
                parent.children.push(id);
            }
        }

        self.tree.insert(node);
        (self, id)
    }

    /// Add a child and descend into it.
    pub fn push_child(mut self, role: AccessibleRole, bounds: Rect) -> Self {
        let id = self.tree.next_id();
        let mut node = AccessibleNode::new(id, role, bounds);
        node.parent = self.current_parent;

        if let Some(parent_id) = self.current_parent {
            if let Some(parent) = self.tree.get_mut(parent_id) {
                parent.children.push(id);
            }
        }

        self.tree.insert(node);
        self.current_parent = Some(id);
        self
    }

    /// Pop back to the parent.
    pub fn pop(mut self) -> Self {
        if let Some(current) = self.current_parent {
            if let Some(node) = self.tree.get(current) {
                self.current_parent = node.parent;
            }
        }
        self
    }

    /// Configure the current node.
    pub fn configure<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut AccessibleNode),
    {
        if let Some(id) = self.current_parent {
            if let Some(node) = self.tree.get_mut(id) {
                f(node);
            }
        }
        self
    }

    /// Build the tree.
    pub fn build(self) -> AccessibilityTree {
        self.tree
    }
}

impl Default for AccessibilityTreeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // AccessibleNode tests
    #[test]
    fn test_accessible_node_new() {
        let id = AccessibleNodeId::new(1);
        let bounds = Rect::new(0.0, 0.0, 100.0, 50.0);
        let node = AccessibleNode::new(id, AccessibleRole::Button, bounds);

        assert_eq!(node.id, id);
        assert_eq!(node.role, AccessibleRole::Button);
        assert_eq!(node.bounds, bounds);
        assert!(!node.focusable);
        assert!(node.enabled);
        assert!(node.visible);
    }

    #[test]
    fn test_accessible_node_button() {
        let id = AccessibleNodeId::new(1);
        let bounds = Rect::new(0.0, 0.0, 100.0, 50.0);
        let node = AccessibleNode::button(id, "Click me", bounds);

        assert_eq!(node.name, Some("Click me".to_string()));
        assert_eq!(node.role, AccessibleRole::Button);
        assert!(node.focusable);
        assert_eq!(node.tab_index, 0);
    }

    #[test]
    fn test_accessible_node_checkbox() {
        let id = AccessibleNodeId::new(1);
        let bounds = Rect::new(0.0, 0.0, 20.0, 20.0);

        let unchecked = AccessibleNode::checkbox(id, "Accept terms", false, bounds);
        assert_eq!(unchecked.checked, Some(CheckedState::Unchecked));

        let checked = AccessibleNode::checkbox(id, "Accept terms", true, bounds);
        assert_eq!(checked.checked, Some(CheckedState::Checked));
    }

    #[test]
    fn test_accessible_node_text_input() {
        let id = AccessibleNodeId::new(1);
        let bounds = Rect::new(0.0, 0.0, 200.0, 30.0);
        let node = AccessibleNode::text_input(id, "Username", "john_doe", bounds);

        assert_eq!(node.name, Some("Username".to_string()));
        assert_eq!(node.value, Some("john_doe".to_string()));
        assert_eq!(node.role, AccessibleRole::TextInput);
    }

    #[test]
    fn test_accessible_node_heading() {
        let id = AccessibleNodeId::new(1);
        let bounds = Rect::new(0.0, 0.0, 200.0, 40.0);
        let node = AccessibleNode::heading(id, "Welcome", 1, bounds);

        assert_eq!(node.name, Some("Welcome".to_string()));
        assert_eq!(node.role, AccessibleRole::Heading);
        assert_eq!(node.level, Some(1));
    }

    #[test]
    fn test_accessible_node_heading_level_clamp() {
        let id = AccessibleNodeId::new(1);
        let bounds = Rect::new(0.0, 0.0, 200.0, 40.0);

        // Level 0 should clamp to 1
        let h0 = AccessibleNode::heading(id, "H0", 0, bounds);
        assert_eq!(h0.level, Some(1));

        // Level 10 should clamp to 6
        let h10 = AccessibleNode::heading(id, "H10", 10, bounds);
        assert_eq!(h10.level, Some(6));
    }

    #[test]
    fn test_accessible_node_slider() {
        let id = AccessibleNodeId::new(1);
        let bounds = Rect::new(0.0, 0.0, 200.0, 20.0);
        let node = AccessibleNode::slider(id, "Volume", 50.0, 0.0, 100.0, bounds);

        assert_eq!(node.name, Some("Volume".to_string()));
        assert_eq!(node.value, Some("50".to_string()));
        assert_eq!(node.value_min, Some(0.0));
        assert_eq!(node.value_max, Some(100.0));
        assert_eq!(node.role, AccessibleRole::Slider);
    }

    #[test]
    fn test_accessible_node_contains_point() {
        let id = AccessibleNodeId::new(1);
        let bounds = Rect::new(10.0, 10.0, 100.0, 50.0);
        let node = AccessibleNode::new(id, AccessibleRole::Generic, bounds);

        assert!(node.contains_point(Point::new(50.0, 30.0)));
        assert!(node.contains_point(Point::new(10.0, 10.0))); // Edge
        assert!(!node.contains_point(Point::new(5.0, 30.0)));
        assert!(!node.contains_point(Point::new(120.0, 30.0)));
    }

    #[test]
    fn test_accessible_node_invisible_not_contains() {
        let id = AccessibleNodeId::new(1);
        let bounds = Rect::new(0.0, 0.0, 100.0, 50.0);
        let mut node = AccessibleNode::new(id, AccessibleRole::Generic, bounds);
        node.visible = false;

        assert!(!node.contains_point(Point::new(50.0, 25.0)));
    }

    #[test]
    fn test_accessible_node_builder_pattern() {
        let id = AccessibleNodeId::new(1);
        let bounds = Rect::new(0.0, 0.0, 100.0, 50.0);
        let node = AccessibleNode::new(id, AccessibleRole::Button, bounds)
            .with_name("Submit")
            .with_description("Submit the form")
            .with_focusable(true)
            .with_property("aria-pressed", "false");

        assert_eq!(node.name, Some("Submit".to_string()));
        assert_eq!(node.description, Some("Submit the form".to_string()));
        assert!(node.focusable);
        assert_eq!(
            node.properties.get("aria-pressed"),
            Some(&"false".to_string())
        );
    }

    // AccessibilityTree tests
    #[test]
    fn test_tree_new() {
        let tree = AccessibilityTree::new();
        assert!(tree.is_empty());
        assert!(tree.root().is_none());
    }

    #[test]
    fn test_tree_insert_and_get() {
        let mut tree = AccessibilityTree::new();
        let id = tree.next_id();
        let node = AccessibleNode::new(
            id,
            AccessibleRole::Generic,
            Rect::new(0.0, 0.0, 100.0, 100.0),
        );
        tree.insert(node);
        tree.set_root(id);

        assert_eq!(tree.len(), 1);
        assert!(tree.get(id).is_some());
        assert!(tree.root().is_some());
    }

    #[test]
    fn test_tree_remove() {
        let mut tree = AccessibilityTree::new();
        let id = tree.next_id();
        let node = AccessibleNode::new(
            id,
            AccessibleRole::Generic,
            Rect::new(0.0, 0.0, 100.0, 100.0),
        );
        tree.insert(node);

        let removed = tree.remove(id);
        assert!(removed.is_some());
        assert!(tree.is_empty());
    }

    #[test]
    fn test_tree_clear() {
        let mut tree = AccessibilityTree::new();
        let id1 = tree.next_id();
        let id2 = tree.next_id();
        tree.insert(AccessibleNode::new(
            id1,
            AccessibleRole::Generic,
            Rect::default(),
        ));
        tree.insert(AccessibleNode::new(
            id2,
            AccessibleRole::Generic,
            Rect::default(),
        ));

        tree.clear();
        assert!(tree.is_empty());
        assert!(tree.root().is_none());
    }

    #[test]
    fn test_tree_focus() {
        let mut tree = AccessibilityTree::new();
        let id = tree.next_id();
        let mut node = AccessibleNode::button(id, "Button", Rect::new(0.0, 0.0, 100.0, 50.0));
        node.focusable = true;
        tree.insert(node);

        assert!(tree.set_focus(id));
        assert_eq!(tree.focused_id(), Some(id));

        let focused = tree.focused().unwrap();
        assert!(focused.focused);
    }

    #[test]
    fn test_tree_focus_non_focusable() {
        let mut tree = AccessibilityTree::new();
        let id = tree.next_id();
        let node = AccessibleNode::new(
            id,
            AccessibleRole::Generic,
            Rect::new(0.0, 0.0, 100.0, 50.0),
        );
        tree.insert(node);

        assert!(!tree.set_focus(id));
        assert!(tree.focused().is_none());
    }

    #[test]
    fn test_tree_clear_focus() {
        let mut tree = AccessibilityTree::new();
        let id = tree.next_id();
        let mut node = AccessibleNode::button(id, "Button", Rect::new(0.0, 0.0, 100.0, 50.0));
        node.focusable = true;
        tree.insert(node);
        tree.set_focus(id);

        tree.clear_focus();
        assert!(tree.focused().is_none());
        assert!(!tree.get(id).unwrap().focused);
    }

    #[test]
    fn test_tree_focus_next() {
        let mut tree = AccessibilityTree::new();

        let id1 = tree.next_id();
        let mut node1 = AccessibleNode::button(id1, "First", Rect::new(0.0, 0.0, 100.0, 50.0));
        node1.focusable = true;
        tree.insert(node1);

        let id2 = tree.next_id();
        let mut node2 = AccessibleNode::button(id2, "Second", Rect::new(0.0, 60.0, 100.0, 50.0));
        node2.focusable = true;
        tree.insert(node2);

        // First focus_next should focus the first element
        let first = tree.focus_next();
        assert!(first.is_some());

        // Second focus_next should focus the second element
        let second = tree.focus_next();
        assert!(second.is_some());
        assert_ne!(first, second);
    }

    #[test]
    fn test_tree_focus_previous() {
        let mut tree = AccessibilityTree::new();

        let id1 = tree.next_id();
        let mut node1 = AccessibleNode::button(id1, "First", Rect::new(0.0, 0.0, 100.0, 50.0));
        node1.focusable = true;
        tree.insert(node1);

        let id2 = tree.next_id();
        let mut node2 = AccessibleNode::button(id2, "Second", Rect::new(0.0, 60.0, 100.0, 50.0));
        node2.focusable = true;
        tree.insert(node2);

        // Set focus to second
        tree.set_focus(id2);

        // focus_previous should move to first
        let prev = tree.focus_previous();
        assert_eq!(prev, Some(id1));
    }

    // HitTester tests
    #[test]
    fn test_hit_tester_new() {
        let tester = HitTester::new();
        assert!(tester.tree().is_empty());
    }

    #[test]
    fn test_hit_test_single_node() {
        let mut tree = AccessibilityTree::new();
        let id = tree.next_id();
        let node = AccessibleNode::new(
            id,
            AccessibleRole::Button,
            Rect::new(10.0, 10.0, 100.0, 50.0),
        );
        tree.insert(node);
        tree.set_root(id);

        let tester = HitTester::with_tree(tree);

        // Hit inside
        let result = tester.hit_test(Point::new(50.0, 30.0));
        assert!(result.is_some());
        assert_eq!(result.unwrap().id, id);

        // Miss outside
        let miss = tester.hit_test(Point::new(0.0, 0.0));
        assert!(miss.is_none());
    }

    #[test]
    fn test_hit_test_nested_nodes() {
        let mut tree = AccessibilityTree::new();

        // Parent node
        let parent_id = tree.next_id();
        let mut parent = AccessibleNode::new(
            parent_id,
            AccessibleRole::Generic,
            Rect::new(0.0, 0.0, 200.0, 200.0),
        );

        // Child node (inside parent)
        let child_id = tree.next_id();
        let mut child = AccessibleNode::new(
            child_id,
            AccessibleRole::Button,
            Rect::new(50.0, 50.0, 100.0, 100.0),
        );
        child.parent = Some(parent_id);

        parent.children.push(child_id);
        tree.insert(parent);
        tree.insert(child);
        tree.set_root(parent_id);

        let tester = HitTester::with_tree(tree);

        // Hit inside child should return child
        let child_hit = tester.hit_test(Point::new(100.0, 100.0));
        assert!(child_hit.is_some());
        assert_eq!(child_hit.unwrap().id, child_id);

        // Hit inside parent but outside child should return parent
        let parent_hit = tester.hit_test(Point::new(10.0, 10.0));
        assert!(parent_hit.is_some());
        assert_eq!(parent_hit.unwrap().id, parent_id);
    }

    #[test]
    fn test_hit_test_all() {
        let mut tree = AccessibilityTree::new();

        let parent_id = tree.next_id();
        let mut parent = AccessibleNode::new(
            parent_id,
            AccessibleRole::Generic,
            Rect::new(0.0, 0.0, 200.0, 200.0),
        );

        let child_id = tree.next_id();
        let mut child = AccessibleNode::new(
            child_id,
            AccessibleRole::Button,
            Rect::new(50.0, 50.0, 100.0, 100.0),
        );
        child.parent = Some(parent_id);
        parent.children.push(child_id);

        tree.insert(parent);
        tree.insert(child);
        tree.set_root(parent_id);

        let tester = HitTester::with_tree(tree);

        // Hit inside child should return both child and parent
        let all_hits = tester.hit_test_all(Point::new(100.0, 100.0));
        assert_eq!(all_hits.len(), 2);
        // Topmost first
        assert_eq!(all_hits[0].id, child_id);
        assert_eq!(all_hits[1].id, parent_id);
    }

    #[test]
    fn test_hit_test_focusable() {
        let mut tree = AccessibilityTree::new();

        let parent_id = tree.next_id();
        let mut parent = AccessibleNode::new(
            parent_id,
            AccessibleRole::Generic,
            Rect::new(0.0, 0.0, 200.0, 200.0),
        );

        let child_id = tree.next_id();
        let mut child =
            AccessibleNode::button(child_id, "Button", Rect::new(50.0, 50.0, 100.0, 100.0));
        child.parent = Some(parent_id);
        child.focusable = true;
        parent.children.push(child_id);

        tree.insert(parent);
        tree.insert(child);
        tree.set_root(parent_id);

        let tester = HitTester::with_tree(tree);

        let focusable = tester.hit_test_focusable(Point::new(100.0, 100.0));
        assert!(focusable.is_some());
        assert_eq!(focusable.unwrap().id, child_id);
    }

    #[test]
    fn test_hit_test_role() {
        let mut tree = AccessibilityTree::new();

        let parent_id = tree.next_id();
        let mut parent = AccessibleNode::new(
            parent_id,
            AccessibleRole::Generic,
            Rect::new(0.0, 0.0, 200.0, 200.0),
        );

        let button_id = tree.next_id();
        let mut button = AccessibleNode::new(
            button_id,
            AccessibleRole::Button,
            Rect::new(50.0, 50.0, 100.0, 100.0),
        );
        button.parent = Some(parent_id);
        parent.children.push(button_id);

        tree.insert(parent);
        tree.insert(button);
        tree.set_root(parent_id);

        let tester = HitTester::with_tree(tree);

        let buttons = tester.hit_test_role(Point::new(100.0, 100.0), AccessibleRole::Button);
        assert_eq!(buttons.len(), 1);
        assert_eq!(buttons[0].role, AccessibleRole::Button);

        let generic = tester.hit_test_role(Point::new(100.0, 100.0), AccessibleRole::Generic);
        assert_eq!(generic.len(), 1);
    }

    // AccessibilityTreeBuilder tests
    #[test]
    fn test_builder_basic() {
        let tree = AccessibilityTreeBuilder::new()
            .root(AccessibleRole::Generic, Rect::new(0.0, 0.0, 800.0, 600.0))
            .build();

        assert_eq!(tree.len(), 1);
        assert!(tree.root().is_some());
    }

    #[test]
    fn test_builder_with_children() {
        let tree = AccessibilityTreeBuilder::new()
            .root(AccessibleRole::Generic, Rect::new(0.0, 0.0, 800.0, 600.0))
            .push_child(AccessibleRole::Button, Rect::new(10.0, 10.0, 100.0, 50.0))
            .pop()
            .push_child(
                AccessibleRole::TextInput,
                Rect::new(10.0, 70.0, 200.0, 30.0),
            )
            .pop()
            .build();

        assert_eq!(tree.len(), 3);

        let root = tree.root().unwrap();
        assert_eq!(root.children.len(), 2);
    }

    #[test]
    fn test_builder_configure() {
        let tree = AccessibilityTreeBuilder::new()
            .root(AccessibleRole::Generic, Rect::new(0.0, 0.0, 800.0, 600.0))
            .push_child(AccessibleRole::Button, Rect::new(10.0, 10.0, 100.0, 50.0))
            .configure(|node| {
                node.name = Some("Submit".to_string());
                node.focusable = true;
            })
            .pop()
            .build();

        let root = tree.root().unwrap();
        let child_id = root.children[0];
        let child = tree.get(child_id).unwrap();

        assert_eq!(child.name, Some("Submit".to_string()));
        assert!(child.focusable);
    }

    // CheckedState tests
    #[test]
    fn test_checked_state_equality() {
        assert_eq!(CheckedState::Checked, CheckedState::Checked);
        assert_ne!(CheckedState::Checked, CheckedState::Unchecked);
        assert_ne!(CheckedState::Mixed, CheckedState::Checked);
    }

    // LiveRegion tests
    #[test]
    fn test_live_region_default() {
        let region = LiveRegion::default();
        assert_eq!(region, LiveRegion::Off);
    }

    // AccessibleNodeId tests
    #[test]
    fn test_accessible_node_id() {
        let id1 = AccessibleNodeId::new(1);
        let id2 = AccessibleNodeId::new(1);
        let id3 = AccessibleNodeId::new(2);

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_accessible_node_id_default() {
        let id = AccessibleNodeId::default();
        assert_eq!(id.0, 0);
    }
}
