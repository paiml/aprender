//! Trait definitions for MCTS components.
//!
//! This module defines the core traits that must be implemented
//! to use MCTS for a particular domain.

use std::hash::Hash;

use super::Reward;

/// Trait for states in the search space (e.g., partial AST)
pub trait State: Clone + Eq + Hash {
    /// Returns true if this is a terminal state (complete code)
    fn is_terminal(&self) -> bool;

    /// Returns a hash of this state for deduplication
    fn state_hash(&self) -> u64 {
        use std::hash::Hasher;
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

/// Trait for actions in the search space (e.g., AST transformations)
pub trait Action: Clone + Eq + Hash {
    /// Returns the name/identifier of this action
    fn name(&self) -> &str;

    /// Returns the prior probability of this action (for policy network guidance)
    fn prior(&self) -> f64 {
        1.0 // Uniform prior by default
    }
}

/// Trait for defining the state space (how states transition)
pub trait StateSpace<S: State, A: Action> {
    /// Apply an action to a state, returning the new state
    fn apply(&self, state: &S, action: &A) -> S;

    /// Evaluate a terminal state, returning the reward (0.0 to 1.0)
    fn evaluate(&self, state: &S) -> Reward;

    /// Clone the state space (for parallel simulations)
    fn clone_space(&self) -> Box<dyn StateSpace<S, A> + Send + Sync>;
}

/// Trait for defining the action space (available actions from a state)
pub trait ActionSpace<S: State, A: Action> {
    /// Returns all legal actions from the given state
    fn legal_actions(&self, state: &S) -> Vec<A>;

    /// Returns true if there are no legal actions from this state
    fn is_empty(&self, state: &S) -> bool {
        self.legal_actions(state).is_empty()
    }
}

/// Trait for policy networks that guide the search
pub trait PolicyNetwork<S: State, A: Action>: Send + Sync {
    /// Returns (action, prior probability) pairs for the given state
    fn predict(&self, state: &S) -> Vec<(A, f64)>;

    /// Returns the value estimate for a state (optional, for AlphaZero-style)
    fn value(&self, _state: &S) -> f64 {
        0.5 // Neutral value by default
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_state_trait_implementation() {
        let state = TestState { value: 5, terminal: false };
        assert!(!state.is_terminal());

        let terminal = TestState { value: 10, terminal: true };
        assert!(terminal.is_terminal());
    }

    #[test]
    fn test_action_trait_implementation() {
        let action = TestAction { delta: 1 };
        assert_eq!(action.name(), "test_action");
        assert_eq!(action.prior(), 1.0);
    }
}
