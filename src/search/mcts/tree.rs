//! Search tree structure for MCTS.
//!
//! This module contains the tree data structure that stores nodes
//! and handles transposition detection.

use std::collections::HashMap;

use super::node::{Node, NodeId};
use super::traits::{Action, State};

/// The search tree structure
#[derive(Debug)]
pub struct SearchTree<S: State, A: Action> {
    /// All nodes indexed by NodeId
    nodes: Vec<Node<S, A>>,
    /// Map from state hash to node id for deduplication
    state_map: HashMap<u64, NodeId>,
    /// Root node id
    pub(crate) root_id: NodeId,
}

impl<S: State, A: Action> SearchTree<S, A> {
    /// Create a new search tree with the given root state
    #[must_use]
    pub fn new(root_state: S, root_actions: Vec<A>) -> Self {
        let root = Node::root(root_state.clone(), root_actions);
        let state_hash = root_state.state_hash();
        let mut state_map = HashMap::new();
        state_map.insert(state_hash, NodeId::new(0));

        Self { nodes: vec![root], state_map, root_id: NodeId::new(0) }
    }

    /// Get the root node
    #[must_use]
    pub fn root(&self) -> &Node<S, A> {
        &self.nodes[self.root_id.0]
    }

    /// Get a node by id
    #[must_use]
    pub fn get(&self, id: NodeId) -> Option<&Node<S, A>> {
        self.nodes.get(id.0)
    }

    /// Get a mutable node by id
    pub fn get_mut(&mut self, id: NodeId) -> Option<&mut Node<S, A>> {
        self.nodes.get_mut(id.0)
    }

    /// Add a child node, returning its id
    pub fn add_child(
        &mut self,
        parent_id: NodeId,
        state: S,
        action: A,
        untried_actions: Vec<A>,
        prior: f64,
    ) -> NodeId {
        let state_hash = state.state_hash();

        // Check for transposition (same state reached via different path)
        if let Some(&existing_id) = self.state_map.get(&state_hash) {
            // Add as child but reuse existing node's stats
            if let Some(parent) = self.nodes.get_mut(parent_id.0) {
                if !parent.children.contains(&existing_id) {
                    parent.children.push(existing_id);
                }
            }
            return existing_id;
        }

        let child_id = NodeId::new(self.nodes.len());
        let child = Node::child(child_id, state, action, parent_id, untried_actions, prior);

        self.nodes.push(child);
        self.state_map.insert(state_hash, child_id);

        if let Some(parent) = self.nodes.get_mut(parent_id.0) {
            parent.children.push(child_id);
        }

        child_id
    }

    /// Get number of nodes in the tree
    #[must_use]
    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    /// Get all children of a node
    #[must_use]
    pub fn children(&self, id: NodeId) -> Vec<&Node<S, A>> {
        self.nodes
            .get(id.0)
            .map(|n| n.children.iter().filter_map(|&cid| self.get(cid)).collect())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // Simple test state for unit tests
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct TestState {
        value: i32,
        terminal: bool,
    }

    impl State for TestState {
        fn is_terminal(&self) -> bool {
            self.terminal
        }
    }

    // Simple test action
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct TestAction {
        delta: i32,
    }

    impl Action for TestAction {
        fn name(&self) -> &'static str {
            "test_action"
        }
    }

    #[test]
    fn test_search_tree_creation() {
        let state = TestState { value: 0, terminal: false };
        let actions = vec![TestAction { delta: 1 }, TestAction { delta: -1 }];
        let tree = SearchTree::new(state.clone(), actions);

        assert_eq!(tree.size(), 1);
        assert_eq!(tree.root().state, state);
    }

    #[test]
    fn test_search_tree_add_child() {
        let state = TestState { value: 0, terminal: false };
        let actions = vec![TestAction { delta: 1 }];
        let mut tree = SearchTree::new(state, actions);

        let child_state = TestState { value: 1, terminal: false };
        let child_action = TestAction { delta: 1 };
        let child_id =
            tree.add_child(NodeId::new(0), child_state.clone(), child_action, vec![], 0.5);

        assert_eq!(tree.size(), 2);
        let child = tree.get(child_id).expect("key should exist");
        assert_eq!(child.state, child_state);
    }

    #[test]
    fn test_tree_children() {
        let state = TestState { value: 0, terminal: false };
        let mut tree = SearchTree::new(state.clone(), vec![]);

        let child1 = TestState { value: 1, terminal: false };
        let child2 = TestState { value: 2, terminal: false };

        tree.add_child(NodeId::new(0), child1, TestAction { delta: 1 }, vec![], 0.5);
        tree.add_child(NodeId::new(0), child2, TestAction { delta: 2 }, vec![], 0.5);

        let children = tree.children(NodeId::new(0));
        assert_eq!(children.len(), 2);
    }

    #[test]
    fn test_transposition_table() {
        let state = TestState { value: 0, terminal: false };
        let mut tree = SearchTree::new(state.clone(), vec![]);

        // Add same state via two different actions
        let child_state = TestState { value: 1, terminal: false };

        let id1 = tree.add_child(
            NodeId::new(0),
            child_state.clone(),
            TestAction { delta: 1 },
            vec![],
            0.5,
        );

        // Same state again
        let id2 = tree.add_child(NodeId::new(0), child_state, TestAction { delta: 1 }, vec![], 0.5);

        // Should return same node id (transposition)
        assert_eq!(id1, id2);
        // Tree size should be 2 (root + one child)
        assert_eq!(tree.size(), 2);
    }

    proptest! {
        #[test]
        fn test_tree_size_increases_monotonically(num_children in 1usize..10) {
            let state = TestState { value: 0, terminal: false };
            let mut tree = SearchTree::new(state.clone(), vec![]);

            let mut prev_size = tree.size();

            for i in 0..num_children {
                let child_state = TestState { value: i as i32, terminal: false };
                tree.add_child(
                    NodeId::new(0),
                    child_state,
                    TestAction { delta: 1 },
                    vec![],
                    0.5,
                );

                // Size should increase or stay same (transposition)
                prop_assert!(tree.size() >= prev_size);
                prev_size = tree.size();
            }
        }
    }
}
