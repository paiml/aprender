//! Widget tree diffing for efficient UI updates.
//!
//! This module provides algorithms for comparing two widget trees and computing
//! the minimal set of operations needed to transform one into the other.
//!
//! # Algorithm
//!
//! The diffing algorithm works in two phases:
//!
//! 1. **Matching**: Match widgets between old and new trees by key or position
//! 2. **Reconciliation**: Generate operations for differences
//!
//! Widgets are matched by:
//! - Explicit key (if set via `key` attribute)
//! - Type + position (fallback)

use crate::widget::TypeId;
use std::collections::HashMap;

/// A key used to identify widgets across renders.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum WidgetKey {
    /// Explicit string key
    String(String),
    /// Auto-generated index key
    Index(usize),
}

impl WidgetKey {
    /// Create a string key.
    #[must_use]
    pub fn string(s: impl Into<String>) -> Self {
        Self::String(s.into())
    }

    /// Create an index key.
    #[must_use]
    pub const fn index(i: usize) -> Self {
        Self::Index(i)
    }
}

/// A node in the widget tree for diffing.
#[derive(Debug, Clone)]
pub struct DiffNode {
    /// Widget type
    pub type_id: TypeId,
    /// Optional key for matching
    pub key: Option<String>,
    /// Widget properties hash (for detecting changes)
    pub props_hash: u64,
    /// Child nodes
    pub children: Vec<Self>,
    /// Position in parent's children list
    pub index: usize,
}

impl DiffNode {
    /// Create a new diff node.
    #[must_use]
    pub const fn new(type_id: TypeId, props_hash: u64) -> Self {
        Self {
            type_id,
            key: None,
            props_hash,
            children: Vec::new(),
            index: 0,
        }
    }

    /// Set the key for this node.
    #[must_use]
    pub fn with_key(mut self, key: impl Into<String>) -> Self {
        self.key = Some(key.into());
        self
    }

    /// Set the index of this node.
    #[must_use]
    pub const fn with_index(mut self, index: usize) -> Self {
        self.index = index;
        self
    }

    /// Add a child node.
    pub fn add_child(&mut self, child: Self) {
        self.children.push(child);
    }

    /// Add a child node with fluent API.
    #[must_use]
    pub fn with_child(mut self, child: Self) -> Self {
        self.children.push(child);
        self
    }
}

/// Operation to apply during reconciliation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffOp {
    /// Insert a new widget at path
    Insert {
        /// Path to parent (indices from root)
        path: Vec<usize>,
        /// Index to insert at
        index: usize,
        /// Type of widget to insert
        type_id: TypeId,
        /// Props hash of the new widget
        props_hash: u64,
    },
    /// Remove widget at path
    Remove {
        /// Path to widget to remove
        path: Vec<usize>,
    },
    /// Update widget properties at path
    Update {
        /// Path to widget to update
        path: Vec<usize>,
        /// New props hash
        new_props_hash: u64,
    },
    /// Move widget from one position to another
    Move {
        /// Old path
        from_path: Vec<usize>,
        /// New path
        to_path: Vec<usize>,
    },
    /// Replace widget with different type
    Replace {
        /// Path to widget to replace
        path: Vec<usize>,
        /// New type ID
        new_type_id: TypeId,
        /// New props hash
        new_props_hash: u64,
    },
}

/// Result of diffing two widget trees.
#[derive(Debug, Clone, Default)]
pub struct DiffResult {
    /// List of operations to apply
    pub operations: Vec<DiffOp>,
}

impl DiffResult {
    /// Create an empty diff result.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if there are no changes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    /// Get the number of operations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// Add an operation.
    pub fn push(&mut self, op: DiffOp) {
        self.operations.push(op);
    }
}

/// Widget tree differ.
#[derive(Debug, Default)]
pub struct TreeDiffer {
    /// Current path during traversal
    current_path: Vec<usize>,
}

impl TreeDiffer {
    /// Create a new tree differ.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute the diff between two widget trees.
    #[must_use]
    pub fn diff(&mut self, old: &DiffNode, new: &DiffNode) -> DiffResult {
        let mut result = DiffResult::new();
        self.current_path.clear();
        self.diff_node(old, new, &mut result);
        result
    }

    fn diff_node(&mut self, old: &DiffNode, new: &DiffNode, result: &mut DiffResult) {
        // Check if type changed - need full replacement
        if old.type_id != new.type_id {
            result.push(DiffOp::Replace {
                path: self.current_path.clone(),
                new_type_id: new.type_id,
                new_props_hash: new.props_hash,
            });
            return;
        }

        // Check if props changed
        if old.props_hash != new.props_hash {
            result.push(DiffOp::Update {
                path: self.current_path.clone(),
                new_props_hash: new.props_hash,
            });
        }

        // Diff children
        self.diff_children(&old.children, &new.children, result);
    }

    fn diff_children(
        &mut self,
        old_children: &[DiffNode],
        new_children: &[DiffNode],
        result: &mut DiffResult,
    ) {
        // Build key -> index maps for keyed children
        let old_keyed: HashMap<&str, usize> = old_children
            .iter()
            .enumerate()
            .filter_map(|(i, c)| c.key.as_deref().map(|k| (k, i)))
            .collect();

        let _new_keyed: HashMap<&str, usize> = new_children
            .iter()
            .enumerate()
            .filter_map(|(i, c)| c.key.as_deref().map(|k| (k, i)))
            .collect();

        // Track which old children have been matched
        let mut old_matched = vec![false; old_children.len()];
        let mut new_matched = vec![false; new_children.len()];

        // Phase 1: Match by key
        for (new_idx, new_child) in new_children.iter().enumerate() {
            if let Some(key) = &new_child.key {
                if let Some(&old_idx) = old_keyed.get(key.as_str()) {
                    old_matched[old_idx] = true;
                    new_matched[new_idx] = true;

                    // Check if moved
                    if old_idx != new_idx {
                        let mut from_path = self.current_path.clone();
                        from_path.push(old_idx);
                        let mut to_path = self.current_path.clone();
                        to_path.push(new_idx);
                        result.push(DiffOp::Move { from_path, to_path });
                    }

                    // Recursively diff
                    self.current_path.push(new_idx);
                    self.diff_node(&old_children[old_idx], new_child, result);
                    self.current_path.pop();
                }
            }
        }

        // Phase 2: Match unkeyed children by type + position
        let old_unkeyed: Vec<usize> = old_children
            .iter()
            .enumerate()
            .filter(|(i, c)| c.key.is_none() && !old_matched[*i])
            .map(|(i, _)| i)
            .collect();

        let new_unkeyed: Vec<usize> = new_children
            .iter()
            .enumerate()
            .filter(|(i, c)| c.key.is_none() && !new_matched[*i])
            .map(|(i, _)| i)
            .collect();

        // Match by type
        let mut old_unkeyed_used = vec![false; old_unkeyed.len()];
        for new_pos in &new_unkeyed {
            let new_child = &new_children[*new_pos];
            let mut found = false;

            for (old_pos_idx, old_pos) in old_unkeyed.iter().enumerate() {
                if old_unkeyed_used[old_pos_idx] {
                    continue;
                }
                let old_child = &old_children[*old_pos];

                if old_child.type_id == new_child.type_id {
                    old_unkeyed_used[old_pos_idx] = true;
                    old_matched[*old_pos] = true;
                    new_matched[*new_pos] = true;
                    found = true;

                    // Recursively diff
                    self.current_path.push(*new_pos);
                    self.diff_node(old_child, new_child, result);
                    self.current_path.pop();
                    break;
                }
            }

            if !found {
                // New child with no match - insert
                new_matched[*new_pos] = true;
                result.push(DiffOp::Insert {
                    path: self.current_path.clone(),
                    index: *new_pos,
                    type_id: new_child.type_id,
                    props_hash: new_child.props_hash,
                });

                // Recursively handle new children's children
                self.current_path.push(*new_pos);
                self.insert_subtree(new_child, result);
                self.current_path.pop();
            }
        }

        // Phase 3: Remove unmatched old children (in reverse order)
        for (i, matched) in old_matched.iter().enumerate().rev() {
            if !matched {
                let mut path = self.current_path.clone();
                path.push(i);
                result.push(DiffOp::Remove { path });
            }
        }

        // Phase 4: Insert remaining new children
        for (i, matched) in new_matched.iter().enumerate() {
            if !matched {
                let new_child = &new_children[i];
                result.push(DiffOp::Insert {
                    path: self.current_path.clone(),
                    index: i,
                    type_id: new_child.type_id,
                    props_hash: new_child.props_hash,
                });

                // Recursively handle new children's children
                self.current_path.push(i);
                self.insert_subtree(new_child, result);
                self.current_path.pop();
            }
        }
    }

    fn insert_subtree(&mut self, node: &DiffNode, result: &mut DiffResult) {
        for (i, child) in node.children.iter().enumerate() {
            result.push(DiffOp::Insert {
                path: self.current_path.clone(),
                index: i,
                type_id: child.type_id,
                props_hash: child.props_hash,
            });

            self.current_path.push(i);
            self.insert_subtree(child, result);
            self.current_path.pop();
        }
    }
}

/// Convenience function to diff two trees.
#[must_use]
pub fn diff_trees(old: &DiffNode, new: &DiffNode) -> DiffResult {
    let mut differ = TreeDiffer::new();
    differ.diff(old, new)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_type_id<T: 'static>() -> TypeId {
        TypeId::of::<T>()
    }

    #[test]
    fn test_widget_key_string() {
        let key = WidgetKey::string("test");
        assert_eq!(key, WidgetKey::String("test".to_string()));
    }

    #[test]
    fn test_widget_key_index() {
        let key = WidgetKey::index(42);
        assert_eq!(key, WidgetKey::Index(42));
    }

    #[test]
    fn test_diff_node_new() {
        let type_id = make_type_id::<u32>();
        let node = DiffNode::new(type_id, 123);

        assert_eq!(node.type_id, type_id);
        assert_eq!(node.props_hash, 123);
        assert!(node.key.is_none());
        assert!(node.children.is_empty());
    }

    #[test]
    fn test_diff_node_with_key() {
        let type_id = make_type_id::<u32>();
        let node = DiffNode::new(type_id, 123).with_key("my-key");

        assert_eq!(node.key, Some("my-key".to_string()));
    }

    #[test]
    fn test_diff_node_with_child() {
        let type_id = make_type_id::<u32>();
        let child = DiffNode::new(type_id, 456);
        let parent = DiffNode::new(type_id, 123).with_child(child);

        assert_eq!(parent.children.len(), 1);
        assert_eq!(parent.children[0].props_hash, 456);
    }

    #[test]
    fn test_diff_result_empty() {
        let result = DiffResult::new();
        assert!(result.is_empty());
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_diff_identical_trees() {
        let type_id = make_type_id::<u32>();
        let old = DiffNode::new(type_id, 123);
        let new = DiffNode::new(type_id, 123);

        let result = diff_trees(&old, &new);
        assert!(result.is_empty());
    }

    #[test]
    fn test_diff_props_changed() {
        let type_id = make_type_id::<u32>();
        let old = DiffNode::new(type_id, 123);
        let new = DiffNode::new(type_id, 456);

        let result = diff_trees(&old, &new);
        assert_eq!(result.len(), 1);
        assert!(matches!(
            &result.operations[0],
            DiffOp::Update { path, new_props_hash: 456 } if path.is_empty()
        ));
    }

    #[test]
    fn test_diff_type_changed() {
        let old_type = make_type_id::<u32>();
        let new_type = make_type_id::<String>();
        let old = DiffNode::new(old_type, 123);
        let new = DiffNode::new(new_type, 123);

        let result = diff_trees(&old, &new);
        assert_eq!(result.len(), 1);
        assert!(matches!(
            &result.operations[0],
            DiffOp::Replace { path, new_type_id, .. } if path.is_empty() && *new_type_id == new_type
        ));
    }

    #[test]
    fn test_diff_child_added() {
        let type_id = make_type_id::<u32>();
        let child_type = make_type_id::<String>();

        let old = DiffNode::new(type_id, 123);
        let new = DiffNode::new(type_id, 123).with_child(DiffNode::new(child_type, 456));

        let result = diff_trees(&old, &new);
        assert_eq!(result.len(), 1);
        assert!(matches!(
            &result.operations[0],
            DiffOp::Insert { path, index: 0, type_id: t, .. } if path.is_empty() && *t == child_type
        ));
    }

    #[test]
    fn test_diff_child_removed() {
        let type_id = make_type_id::<u32>();
        let child_type = make_type_id::<String>();

        let old = DiffNode::new(type_id, 123).with_child(DiffNode::new(child_type, 456));
        let new = DiffNode::new(type_id, 123);

        let result = diff_trees(&old, &new);
        assert_eq!(result.len(), 1);
        assert!(matches!(
            &result.operations[0],
            DiffOp::Remove { path } if *path == vec![0]
        ));
    }

    #[test]
    fn test_diff_keyed_children_reordered() {
        let type_id = make_type_id::<u32>();

        let old = DiffNode::new(type_id, 0)
            .with_child(DiffNode::new(type_id, 1).with_key("a"))
            .with_child(DiffNode::new(type_id, 2).with_key("b"));

        let new = DiffNode::new(type_id, 0)
            .with_child(DiffNode::new(type_id, 2).with_key("b"))
            .with_child(DiffNode::new(type_id, 1).with_key("a"));

        let result = diff_trees(&old, &new);

        // Should have move operations
        assert!(result
            .operations
            .iter()
            .any(|op| matches!(op, DiffOp::Move { .. })));
    }

    #[test]
    fn test_diff_keyed_child_updated() {
        let type_id = make_type_id::<u32>();

        let old = DiffNode::new(type_id, 0).with_child(DiffNode::new(type_id, 1).with_key("item"));
        let new = DiffNode::new(type_id, 0).with_child(DiffNode::new(type_id, 2).with_key("item"));

        let result = diff_trees(&old, &new);

        assert_eq!(
            result
                .operations
                .iter()
                .filter(|op| matches!(op, DiffOp::Update { .. }))
                .count(),
            1
        );
    }

    #[test]
    fn test_diff_nested_changes() {
        let type_id = make_type_id::<u32>();

        let old = DiffNode::new(type_id, 0)
            .with_child(DiffNode::new(type_id, 1).with_child(DiffNode::new(type_id, 2)));

        let new = DiffNode::new(type_id, 0)
            .with_child(DiffNode::new(type_id, 1).with_child(DiffNode::new(type_id, 3)));

        let result = diff_trees(&old, &new);

        // Should have update at path [0, 0]
        assert_eq!(
            result
                .operations
                .iter()
                .filter(|op| matches!(op, DiffOp::Update { path, .. } if *path == vec![0, 0]))
                .count(),
            1
        );
    }

    #[test]
    fn test_diff_multiple_children_mixed() {
        let type_id = make_type_id::<u32>();
        let string_type = make_type_id::<String>();

        let old = DiffNode::new(type_id, 0)
            .with_child(DiffNode::new(type_id, 1))
            .with_child(DiffNode::new(string_type, 2))
            .with_child(DiffNode::new(type_id, 3));

        let new = DiffNode::new(type_id, 0)
            .with_child(DiffNode::new(type_id, 1))
            .with_child(DiffNode::new(type_id, 4)); // Changed type and removed one

        let result = diff_trees(&old, &new);

        // Should have remove operations for removed children
        assert!(result
            .operations
            .iter()
            .any(|op| matches!(op, DiffOp::Remove { .. })));
    }

    #[test]
    fn test_tree_differ_reuse() {
        let type_id = make_type_id::<u32>();
        let mut differ = TreeDiffer::new();

        let old1 = DiffNode::new(type_id, 1);
        let new1 = DiffNode::new(type_id, 2);
        let result1 = differ.diff(&old1, &new1);

        let old2 = DiffNode::new(type_id, 3);
        let new2 = DiffNode::new(type_id, 3);
        let result2 = differ.diff(&old2, &new2);

        assert_eq!(result1.len(), 1);
        assert!(result2.is_empty());
    }

    #[test]
    fn test_diff_empty_to_tree() {
        let type_id = make_type_id::<u32>();

        let old = DiffNode::new(type_id, 0);
        let new = DiffNode::new(type_id, 0)
            .with_child(DiffNode::new(type_id, 1))
            .with_child(DiffNode::new(type_id, 2));

        let result = diff_trees(&old, &new);

        assert_eq!(
            result
                .operations
                .iter()
                .filter(|op| matches!(op, DiffOp::Insert { .. }))
                .count(),
            2
        );
    }

    #[test]
    fn test_diff_tree_to_empty() {
        let type_id = make_type_id::<u32>();

        let old = DiffNode::new(type_id, 0)
            .with_child(DiffNode::new(type_id, 1))
            .with_child(DiffNode::new(type_id, 2));
        let new = DiffNode::new(type_id, 0);

        let result = diff_trees(&old, &new);

        assert_eq!(
            result
                .operations
                .iter()
                .filter(|op| matches!(op, DiffOp::Remove { .. }))
                .count(),
            2
        );
    }

    #[test]
    fn test_diff_deeply_nested() {
        let type_id = make_type_id::<u32>();

        let old = DiffNode::new(type_id, 0).with_child(
            DiffNode::new(type_id, 1)
                .with_child(DiffNode::new(type_id, 2).with_child(DiffNode::new(type_id, 3))),
        );

        let new = DiffNode::new(type_id, 0).with_child(
            DiffNode::new(type_id, 1)
                .with_child(DiffNode::new(type_id, 2).with_child(DiffNode::new(type_id, 99))),
        );

        let result = diff_trees(&old, &new);

        // Should update the deeply nested node
        assert_eq!(result
            .operations
            .iter()
            .filter(|op| {
                matches!(op, DiffOp::Update { path, new_props_hash: 99 } if *path == vec![0, 0, 0])
            })
            .count(), 1);
    }

    #[test]
    fn test_diff_op_debug() {
        let op = DiffOp::Insert {
            path: vec![0, 1],
            index: 2,
            type_id: make_type_id::<u32>(),
            props_hash: 123,
        };
        let debug_str = format!("{op:?}");
        assert!(debug_str.contains("Insert"));
    }

    #[test]
    fn test_diff_result_push() {
        let mut result = DiffResult::new();
        result.push(DiffOp::Remove { path: vec![0] });
        assert_eq!(result.len(), 1);
        assert!(!result.is_empty());
    }
}
