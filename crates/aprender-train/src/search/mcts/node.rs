//! Node types and statistics for MCTS.
//!
//! This module contains the node representation, statistics tracking,
//! and UCB1/PUCT score calculations.

use super::traits::{Action, State};
use super::Reward;

/// Unique identifier for nodes in the search tree
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

impl NodeId {
    /// Creates a new NodeId
    #[must_use]
    pub const fn new(id: usize) -> Self {
        Self(id)
    }

    /// Returns the underlying id value
    #[must_use]
    pub const fn value(&self) -> usize {
        self.0
    }
}

/// Statistics for a node in the search tree
#[derive(Debug, Clone)]
pub struct NodeStats {
    /// Total visits to this node
    pub visits: usize,
    /// Total accumulated reward
    pub total_reward: f64,
    /// Mean reward (total_reward / visits)
    pub mean_reward: f64,
    /// Prior probability from policy network
    pub prior: f64,
}

impl Default for NodeStats {
    fn default() -> Self {
        Self { visits: 0, total_reward: 0.0, mean_reward: 0.0, prior: 1.0 }
    }
}

impl NodeStats {
    /// Update statistics with a new reward
    pub fn update(&mut self, reward: Reward) {
        contract_pre_update!();
        self.visits += 1;
        self.total_reward += reward;
        self.mean_reward = self.total_reward / self.visits as f64;
    }

    /// Calculate UCB1 score
    #[must_use]
    pub fn ucb1(&self, parent_visits: usize, c: f64) -> f64 {
        if self.visits == 0 {
            return f64::INFINITY;
        }
        let exploitation = self.mean_reward;
        let exploration = c * ((parent_visits as f64).max(1.0).ln() / self.visits as f64).sqrt();
        exploitation + exploration
    }

    /// Calculate PUCT score (Polynomial Upper Confidence Trees) for policy-guided search
    #[must_use]
    pub fn puct(&self, parent_visits: usize, c: f64) -> f64 {
        let exploitation = self.mean_reward;
        let exploration =
            c * self.prior * (parent_visits as f64).sqrt() / (1.0 + self.visits as f64);
        exploitation + exploration
    }
}

/// A node in the search tree
#[derive(Debug, Clone)]
pub struct Node<S: State, A: Action> {
    /// Unique identifier
    pub id: NodeId,
    /// State at this node
    pub state: S,
    /// Action that led to this node (None for root)
    pub action: Option<A>,
    /// Parent node id (None for root)
    pub parent: Option<NodeId>,
    /// Child node ids
    pub children: Vec<NodeId>,
    /// Statistics for this node
    pub stats: NodeStats,
    /// Whether this node is fully expanded
    pub expanded: bool,
    /// Untried actions from this state
    pub untried_actions: Vec<A>,
}

impl<S: State, A: Action> Node<S, A> {
    /// Create a new root node
    #[must_use]
    pub fn root(state: S, untried_actions: Vec<A>) -> Self {
        Self {
            id: NodeId::new(0),
            state,
            action: None,
            parent: None,
            children: Vec::new(),
            stats: NodeStats::default(),
            expanded: false,
            untried_actions,
        }
    }

    /// Create a new child node
    #[must_use]
    pub fn child(
        id: NodeId,
        state: S,
        action: A,
        parent: NodeId,
        untried_actions: Vec<A>,
        prior: f64,
    ) -> Self {
        Self {
            id,
            state,
            action: Some(action),
            parent: Some(parent),
            children: Vec::new(),
            stats: NodeStats { prior, ..Default::default() },
            expanded: false,
            untried_actions,
        }
    }

    /// Returns true if this node is a leaf (no children)
    #[must_use]
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Returns true if this node is fully expanded
    #[must_use]
    pub fn is_fully_expanded(&self) -> bool {
        self.expanded && self.untried_actions.is_empty()
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

    // ========================================
    // ENT-090: Core types tests
    // ========================================

    #[test]
    fn test_node_id_creation() {
        let id = NodeId::new(42);
        assert_eq!(id.value(), 42);
    }

    #[test]
    fn test_node_stats_default() {
        let stats = NodeStats::default();
        assert_eq!(stats.visits, 0);
        assert_eq!(stats.total_reward, 0.0);
        assert_eq!(stats.mean_reward, 0.0);
        assert_eq!(stats.prior, 1.0);
    }

    #[test]
    fn test_node_stats_update() {
        let mut stats = NodeStats::default();
        stats.update(1.0);
        assert_eq!(stats.visits, 1);
        assert_eq!(stats.total_reward, 1.0);
        assert_eq!(stats.mean_reward, 1.0);

        stats.update(0.0);
        assert_eq!(stats.visits, 2);
        assert_eq!(stats.total_reward, 1.0);
        assert_eq!(stats.mean_reward, 0.5);
    }

    #[test]
    fn test_node_root_creation() {
        let state = TestState { value: 0, terminal: false };
        let actions = vec![TestAction { delta: 1 }];
        let node = Node::root(state.clone(), actions);

        assert_eq!(node.id, NodeId::new(0));
        assert_eq!(node.state, state);
        assert!(node.action.is_none());
        assert!(node.parent.is_none());
        assert!(node.children.is_empty());
        assert!(!node.expanded);
    }

    #[test]
    fn test_node_child_creation() {
        let state = TestState { value: 1, terminal: false };
        let action = TestAction { delta: 1 };
        let node =
            Node::child(NodeId::new(1), state.clone(), action.clone(), NodeId::new(0), vec![], 0.5);

        assert_eq!(node.id, NodeId::new(1));
        assert_eq!(node.state, state);
        assert_eq!(node.action, Some(action));
        assert_eq!(node.parent, Some(NodeId::new(0)));
        assert_eq!(node.stats.prior, 0.5);
    }

    #[test]
    fn test_node_is_leaf() {
        let state = TestState { value: 0, terminal: false };
        let node: Node<TestState, TestAction> = Node::root(state, vec![]);
        assert!(node.is_leaf());
    }

    // ========================================
    // ENT-091: UCB1/UCT tests
    // ========================================

    #[test]
    fn test_ucb1_unvisited_node() {
        let stats = NodeStats::default();
        let score = stats.ucb1(10, std::f64::consts::SQRT_2);
        assert!(score.is_infinite());
    }

    #[test]
    fn test_ucb1_visited_node() {
        let mut stats = NodeStats::default();
        stats.update(0.5);
        let score = stats.ucb1(10, std::f64::consts::SQRT_2);

        // Should be exploitation + exploration
        // 0.5 + sqrt(2) * sqrt(ln(10) / 1) ≈ 0.5 + 2.14 = 2.64
        assert!(score > 0.5);
        assert!(score < 5.0);
    }

    #[test]
    fn test_ucb1_more_visits_lower_exploration() {
        let mut stats1 = NodeStats::default();
        stats1.visits = 10;
        stats1.total_reward = 5.0;
        stats1.mean_reward = 0.5;

        let mut stats2 = NodeStats::default();
        stats2.visits = 100;
        stats2.total_reward = 50.0;
        stats2.mean_reward = 0.5;

        let score1 = stats1.ucb1(1000, std::f64::consts::SQRT_2);
        let score2 = stats2.ucb1(1000, std::f64::consts::SQRT_2);

        // More visits should have lower exploration bonus
        assert!(score1 > score2);
    }

    #[test]
    fn test_puct_with_prior() {
        let mut stats = NodeStats::default();
        stats.prior = 0.5;
        stats.update(0.3);

        let score = stats.puct(100, 2.0);

        // PUCT = mean_reward + c * prior * sqrt(parent_visits) / (1 + visits)
        // = 0.3 + 2.0 * 0.5 * sqrt(100) / 2 = 0.3 + 5.0 = 5.3
        assert!((score - 5.3).abs() < 0.01);
    }

    // ========================================
    // ENT-095: Property tests
    // ========================================

    proptest! {
        #[test]
        fn test_node_stats_update_invariants(rewards in prop::collection::vec(0.0f64..=1.0, 1..100)) {
            let mut stats = NodeStats::default();

            for r in &rewards {
                stats.update(*r);
            }

            prop_assert_eq!(stats.visits, rewards.len());
            prop_assert!((stats.total_reward - rewards.iter().sum::<f64>()).abs() < 1e-10);
            prop_assert!((stats.mean_reward - rewards.iter().sum::<f64>() / rewards.len() as f64).abs() < 1e-10);
        }

        #[test]
        fn test_ucb1_exploration_decreases_with_visits(parent_visits in 10usize..1000, c in 0.1f64..5.0) {
            let mut stats1 = NodeStats::default();
            stats1.visits = 10;
            stats1.mean_reward = 0.5;

            let mut stats2 = NodeStats::default();
            stats2.visits = 100;
            stats2.mean_reward = 0.5;

            let ucb1 = stats1.ucb1(parent_visits, c);
            let ucb2 = stats2.ucb1(parent_visits, c);

            // More visits should lead to lower UCB (less exploration bonus)
            prop_assert!(ucb1 > ucb2, "UCB1 with fewer visits should be higher");
        }

        #[test]
        fn test_ucb1_higher_reward_higher_score(parent_visits in 10usize..1000, c in 0.1f64..5.0) {
            let mut stats1 = NodeStats::default();
            stats1.visits = 50;
            stats1.mean_reward = 0.3;

            let mut stats2 = NodeStats::default();
            stats2.visits = 50;
            stats2.mean_reward = 0.7;

            let ucb1 = stats1.ucb1(parent_visits, c);
            let ucb2 = stats2.ucb1(parent_visits, c);

            // Higher reward should lead to higher UCB (same visits)
            prop_assert!(ucb2 > ucb1, "Higher reward should give higher UCB");
        }

        #[test]
        fn test_puct_prior_increases_exploration(prior in 0.1f64..0.9) {
            let mut stats1 = NodeStats::default();
            stats1.visits = 10;
            stats1.mean_reward = 0.5;
            stats1.prior = prior;

            let mut stats2 = NodeStats::default();
            stats2.visits = 10;
            stats2.mean_reward = 0.5;
            stats2.prior = prior * 2.0;

            let puct1 = stats1.puct(100, 2.0);
            let puct2 = stats2.puct(100, 2.0);

            // Higher prior should give higher PUCT
            prop_assert!(puct2 > puct1, "Higher prior should give higher PUCT");
        }
    }
}
