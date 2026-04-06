//! MCTS search result.

use super::stats::MctsStats;
use crate::search::mcts::traits::{Action, State};

/// Result of an MCTS search
#[derive(Debug)]
pub struct MctsResult<S: State, A: Action> {
    /// Best action found
    pub best_action: Option<A>,
    /// Expected reward of best action
    pub expected_reward: f64,
    /// Visit counts for all root children
    pub action_visits: Vec<(A, usize)>,
    /// Search statistics
    pub stats: MctsStats,
    /// The resulting state after best action (if any)
    pub resulting_state: Option<S>,
}
