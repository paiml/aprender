//! Supervision strategies for GPU worker crash recovery.
//!
//! Implements Erlang/OTP-inspired supervision strategies adapted for GPU
//! worker processes: one-for-one, one-for-all, and rest-for-one restart
//! policies.

use serde::{Deserialize, Serialize};

/// Restart strategy applied when a supervised GPU worker crashes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SupervisionStrategy {
    /// Restart only the crashed worker. Other workers continue unaffected.
    OneForOne,
    /// Restart all workers when any single worker crashes.
    OneForAll,
    /// Restart the crashed worker and all workers started after it.
    RestForOne,
}

impl SupervisionStrategy {
    /// Returns true if this strategy affects only the crashed worker.
    #[must_use]
    pub fn is_isolated(&self) -> bool {
        matches!(self, Self::OneForOne)
    }
}

/// Action taken by a supervisor after evaluating a crash.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SupervisorAction {
    /// Restart the specified worker indices.
    Restart(Vec<usize>),
    /// Escalate — the restart budget is exhausted.
    Escalate,
}

/// Action taken after a health check evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HealthAction {
    /// Worker is healthy — no action required.
    Healthy,
    /// Worker missed heartbeats — restart it.
    RestartWorker,
    /// GPU temperature is critical — throttle workload.
    Throttle,
    /// GPU is unrecoverable — shut down the worker.
    Shutdown,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::disallowed_methods)]
mod tests {
    use super::*;

    #[test]
    fn one_for_one_is_isolated() {
        assert!(SupervisionStrategy::OneForOne.is_isolated());
    }

    #[test]
    fn one_for_all_is_not_isolated() {
        assert!(!SupervisionStrategy::OneForAll.is_isolated());
    }

    #[test]
    fn rest_for_one_is_not_isolated() {
        assert!(!SupervisionStrategy::RestForOne.is_isolated());
    }

    #[test]
    fn supervisor_action_restart_contains_indices() {
        let action = SupervisorAction::Restart(vec![0, 2, 3]);
        if let SupervisorAction::Restart(indices) = action {
            assert_eq!(indices, vec![0, 2, 3]);
        } else {
            panic!("expected Restart");
        }
    }

    #[test]
    fn supervisor_action_escalate() {
        let action = SupervisorAction::Escalate;
        assert_eq!(action, SupervisorAction::Escalate);
    }

    #[test]
    fn health_action_variants_distinct() {
        let actions = [
            HealthAction::Healthy,
            HealthAction::RestartWorker,
            HealthAction::Throttle,
            HealthAction::Shutdown,
        ];
        for (i, a) in actions.iter().enumerate() {
            for (j, b) in actions.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn strategy_roundtrip_serde() {
        let strategy = SupervisionStrategy::RestForOne;
        let json = serde_json::to_string(&strategy).unwrap();
        let back: SupervisionStrategy = serde_json::from_str(&json).unwrap();
        assert_eq!(back, strategy);
    }
}
