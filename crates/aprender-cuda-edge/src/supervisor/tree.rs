//! Supervision tree with restart strategies.
//!
//! [`SupervisionTree`] manages a set of workers and implements
//! one-for-one, one-for-all, and rest-for-one restart policies.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use super::strategy::{SupervisionStrategy, SupervisorAction};

/// A record of when a worker was restarted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestartRecord {
    /// Index of the worker that was restarted.
    pub worker_index: usize,
    /// Monotonic timestamp (seconds since supervision tree creation).
    pub timestamp_secs: u64,
}

/// A supervision tree managing GPU workers with a restart strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupervisionTree {
    /// The restart strategy.
    pub strategy: SupervisionStrategy,
    /// Number of workers being supervised.
    pub worker_count: usize,
    /// Maximum restarts allowed within the time window.
    pub max_restarts: u32,
    /// Time window for restart budgeting (seconds).
    pub window_secs: u64,
    /// History of recent restarts.
    restart_history: VecDeque<RestartRecord>,
}

impl SupervisionTree {
    /// Create a new supervision tree.
    #[must_use]
    pub fn new(strategy: SupervisionStrategy, worker_count: usize) -> Self {
        Self {
            strategy,
            worker_count,
            max_restarts: 5,
            window_secs: 60,
            restart_history: VecDeque::new(),
        }
    }

    /// Set the maximum restarts allowed within the time window.
    #[must_use]
    pub fn with_max_restarts(mut self, max: u32, window_secs: u64) -> Self {
        self.max_restarts = max;
        self.window_secs = window_secs;
        self
    }

    /// Handle a worker crash, returning the appropriate action.
    ///
    /// Prunes old restart records, checks the budget, and then computes
    /// which workers to restart based on the strategy.
    #[allow(clippy::cast_possible_truncation)]
    pub fn handle_crash(
        &mut self,
        crashed_index: usize,
        current_time_secs: u64,
    ) -> SupervisorAction {
        self.prune_old_restarts(current_time_secs);

        // Check restart budget (restart_history.len() is bounded by max_restarts which is u32)
        if self.restart_history.len() as u32 >= self.max_restarts {
            return SupervisorAction::Escalate;
        }

        let indices = match self.strategy {
            SupervisionStrategy::OneForOne => vec![crashed_index],
            SupervisionStrategy::OneForAll => (0..self.worker_count).collect(),
            SupervisionStrategy::RestForOne => (crashed_index..self.worker_count).collect(),
        };

        // Record the restart
        self.restart_history.push_back(RestartRecord {
            worker_index: crashed_index,
            timestamp_secs: current_time_secs,
        });

        SupervisorAction::Restart(indices)
    }

    /// Remove restart records older than the time window.
    pub fn prune_old_restarts(&mut self, current_time_secs: u64) {
        let cutoff = current_time_secs.saturating_sub(self.window_secs);
        while self
            .restart_history
            .front()
            .is_some_and(|r| r.timestamp_secs < cutoff)
        {
            self.restart_history.pop_front();
        }
    }

    /// Returns the number of restarts in the current window.
    #[must_use]
    pub fn restart_count(&self) -> usize {
        self.restart_history.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_for_one_restarts_only_crashed() {
        let mut tree = SupervisionTree::new(SupervisionStrategy::OneForOne, 4);
        let action = tree.handle_crash(2, 0);
        assert_eq!(action, SupervisorAction::Restart(vec![2]));
    }

    #[test]
    fn one_for_all_restarts_all_workers() {
        let mut tree = SupervisionTree::new(SupervisionStrategy::OneForAll, 4);
        let action = tree.handle_crash(1, 0);
        assert_eq!(action, SupervisorAction::Restart(vec![0, 1, 2, 3]));
    }

    #[test]
    fn rest_for_one_restarts_from_crashed_onward() {
        let mut tree = SupervisionTree::new(SupervisionStrategy::RestForOne, 5);
        let action = tree.handle_crash(2, 0);
        assert_eq!(action, SupervisorAction::Restart(vec![2, 3, 4]));
    }

    #[test]
    fn escalate_when_budget_exhausted() {
        let mut tree =
            SupervisionTree::new(SupervisionStrategy::OneForOne, 4).with_max_restarts(2, 60);

        // Use up the budget
        tree.handle_crash(0, 0);
        tree.handle_crash(1, 1);

        // Third crash should escalate
        let action = tree.handle_crash(2, 2);
        assert_eq!(action, SupervisorAction::Escalate);
    }

    #[test]
    fn prune_allows_new_restarts() {
        let mut tree =
            SupervisionTree::new(SupervisionStrategy::OneForOne, 4).with_max_restarts(2, 60);

        tree.handle_crash(0, 0);
        tree.handle_crash(1, 30);

        // At t=70, the first restart (t=0) is outside the 60s window
        let action = tree.handle_crash(2, 70);
        assert_eq!(action, SupervisorAction::Restart(vec![2]));
    }

    #[test]
    fn restart_count_tracks_history() {
        let mut tree = SupervisionTree::new(SupervisionStrategy::OneForOne, 4);
        assert_eq!(tree.restart_count(), 0);
        tree.handle_crash(0, 0);
        assert_eq!(tree.restart_count(), 1);
        tree.handle_crash(1, 1);
        assert_eq!(tree.restart_count(), 2);
    }

    #[test]
    fn rest_for_one_last_worker_restarts_only_self() {
        let mut tree = SupervisionTree::new(SupervisionStrategy::RestForOne, 3);
        let action = tree.handle_crash(2, 0);
        assert_eq!(action, SupervisorAction::Restart(vec![2]));
    }
}
