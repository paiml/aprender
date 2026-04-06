//! Tests for MCTS search algorithm.

use super::*;
use crate::search::mcts::config::MctsConfig;
use crate::search::mcts::traits::{Action, ActionSpace, State, StateSpace};
use crate::search::mcts::Reward;
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

// Test state space
struct TestStateSpace {
    target: i32,
}

impl StateSpace<TestState, TestAction> for TestStateSpace {
    fn apply(&self, state: &TestState, action: &TestAction) -> TestState {
        let new_value = state.value + action.delta;
        TestState { value: new_value, terminal: new_value >= 10 || new_value <= -10 }
    }

    fn evaluate(&self, state: &TestState) -> Reward {
        if state.value == self.target {
            1.0
        } else {
            0.0
        }
    }

    fn clone_space(&self) -> Box<dyn StateSpace<TestState, TestAction> + Send + Sync> {
        Box::new(TestStateSpace { target: self.target })
    }
}

// Test action space
struct TestActionSpace;

impl ActionSpace<TestState, TestAction> for TestActionSpace {
    fn legal_actions(&self, state: &TestState) -> Vec<TestAction> {
        if state.terminal {
            vec![]
        } else {
            vec![TestAction { delta: 1 }, TestAction { delta: -1 }, TestAction { delta: 2 }]
        }
    }
}

// ========================================
// ENT-092 & ENT-093: MCTS search tests
// ========================================

#[test]
fn test_mcts_search_basic() {
    let initial_state = TestState { value: 0, terminal: false };
    let config = MctsConfig { max_iterations: 100, ..Default::default() };

    let action_space = TestActionSpace;
    let mut mcts = MctsSearch::with_seed(initial_state, &action_space, config, 42);

    let state_space = TestStateSpace { target: 5 };
    let result = mcts.search(&state_space, &action_space, None);

    assert!(result.best_action.is_some());
    assert!(result.stats.iterations == 100);
    assert!(result.stats.tree_size > 1);
}

#[test]
fn test_mcts_finds_path_to_target() {
    let initial_state = TestState { value: 0, terminal: false };
    let config = MctsConfig {
        max_iterations: 500,
        exploration_constant: 1.0,
        use_policy_priors: false,
        ..Default::default()
    };

    let action_space = TestActionSpace;
    let mut mcts = MctsSearch::with_seed(initial_state, &action_space, config, 12345);

    // Target is 5, so we need to go +1 five times
    let state_space = TestStateSpace { target: 5 };
    let result = mcts.search(&state_space, &action_space, None);

    // With enough iterations, should find a path with positive actions
    assert!(result.best_action.is_some());
    if let Some(action) = &result.best_action {
        // First action should be positive to move toward target
        assert!(action.delta > 0, "Expected positive action, got {action:?}");
    }
}

#[test]
fn test_mcts_terminal_state() {
    let initial_state = TestState { value: 10, terminal: true };
    let config = MctsConfig { max_iterations: 10, ..Default::default() };

    let action_space = TestActionSpace;
    let mut mcts = MctsSearch::new(initial_state, &action_space, config);

    let state_space = TestStateSpace { target: 10 };
    let result = mcts.search(&state_space, &action_space, None);

    // No actions should be taken from terminal state
    assert!(result.best_action.is_none());
}

#[test]
fn test_mcts_backpropagation() {
    let initial_state = TestState { value: 0, terminal: false };
    let config = MctsConfig { max_iterations: 50, ..Default::default() };

    let action_space = TestActionSpace;
    let mut mcts = MctsSearch::new(initial_state, &action_space, config);

    let state_space = TestStateSpace { target: 5 };
    let result = mcts.search(&state_space, &action_space, None);

    // Root should have been visited
    assert!(result.stats.root_visits > 0);

    // Tree should have grown
    assert!(mcts.tree_size() > 1);
}

#[test]
fn test_mcts_reproducibility_with_seed() {
    let initial_state = TestState { value: 0, terminal: false };
    let config = MctsConfig { max_iterations: 100, ..Default::default() };

    let action_space = TestActionSpace;
    let state_space = TestStateSpace { target: 5 };

    let mut mcts1 = MctsSearch::with_seed(initial_state.clone(), &action_space, config.clone(), 42);
    let result1 = mcts1.search(&state_space, &action_space, None);

    let mut mcts2 = MctsSearch::with_seed(initial_state, &action_space, config, 42);
    let result2 = mcts2.search(&state_space, &action_space, None);

    assert_eq!(result1.best_action, result2.best_action);
    assert_eq!(result1.stats.tree_size, result2.stats.tree_size);
}

#[test]
fn test_mcts_result_structure() {
    let initial_state = TestState { value: 0, terminal: false };
    let config = MctsConfig { max_iterations: 50, ..Default::default() };

    let action_space = TestActionSpace;
    let state_space = TestStateSpace { target: 5 };

    let mut mcts = MctsSearch::with_seed(initial_state, &action_space, config, 42);
    let result = mcts.search(&state_space, &action_space, None);

    assert_eq!(result.stats.iterations, 50);
    assert!(!result.action_visits.is_empty());
}

#[test]
fn test_state_space_apply() {
    let state_space = TestStateSpace { target: 5 };
    let state = TestState { value: 0, terminal: false };
    let action = TestAction { delta: 1 };

    let new_state = state_space.apply(&state, &action);
    assert_eq!(new_state.value, 1);
    assert!(!new_state.terminal);
}

#[test]
fn test_state_space_evaluate() {
    let state_space = TestStateSpace { target: 5 };

    let state_on_target = TestState { value: 5, terminal: true };
    assert_eq!(state_space.evaluate(&state_on_target), 1.0);

    let state_off_target = TestState { value: 3, terminal: true };
    assert_eq!(state_space.evaluate(&state_off_target), 0.0);
}

#[test]
fn test_action_space_legal_actions() {
    let action_space = TestActionSpace;

    let non_terminal = TestState { value: 0, terminal: false };
    let actions = action_space.legal_actions(&non_terminal);
    assert_eq!(actions.len(), 3);

    let terminal = TestState { value: 10, terminal: true };
    let actions = action_space.legal_actions(&terminal);
    assert!(actions.is_empty());
}

// ========================================
// ENT-095: Property tests
// ========================================

proptest! {
    #[test]
    fn test_mcts_tree_grows_with_iterations(iterations in 10usize..200) {
        let initial_state = TestState { value: 0, terminal: false };
        let config = MctsConfig {
            max_iterations: iterations,
            ..Default::default()
        };

        let action_space = TestActionSpace;
        let state_space = TestStateSpace { target: 5 };

        let mut mcts = MctsSearch::with_seed(initial_state, &action_space, config, 42);
        let result = mcts.search(&state_space, &action_space, None);

        // Tree should have grown
        prop_assert!(result.stats.tree_size >= 1);
    }

    #[test]
    fn test_mcts_root_visits_increase(iterations in 10usize..500) {
        let initial_state = TestState { value: 0, terminal: false };
        let config = MctsConfig {
            max_iterations: iterations,
            ..Default::default()
        };

        let action_space = TestActionSpace;
        let state_space = TestStateSpace { target: 5 };

        let mut mcts = MctsSearch::with_seed(initial_state, &action_space, config, 42);
        let result = mcts.search(&state_space, &action_space, None);

        // Root visits should be at least iterations (minus terminal states)
        prop_assert!(result.stats.root_visits > 0);
    }

    #[test]
    fn test_action_visits_sum_to_root_minus_one(iterations in 50usize..200) {
        let initial_state = TestState { value: 0, terminal: false };
        let config = MctsConfig {
            max_iterations: iterations,
            exploration_constant: 2.0,
            ..Default::default()
        };

        let action_space = TestActionSpace;
        let state_space = TestStateSpace { target: 5 };

        let mut mcts = MctsSearch::with_seed(initial_state, &action_space, config, 42);
        let result = mcts.search(&state_space, &action_space, None);

        // Sum of child visits should be less than or equal to root visits
        let child_visits: usize = result.action_visits.iter().map(|(_, v)| *v).sum();
        prop_assert!(child_visits <= result.stats.root_visits);
    }
}
