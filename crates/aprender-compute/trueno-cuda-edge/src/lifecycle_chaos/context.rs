//! CUDA context lifecycle chaos scenarios.
//!
//! [`ChaosScenario`] enumerates 8 GPU context lifecycle failure modes
//! that tests should exercise to ensure robustness.

use serde::{Deserialize, Serialize};

/// A GPU context lifecycle failure scenario to inject.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChaosScenario {
    /// Destroy a context twice.
    DoubleDestroy,
    /// Use a context after it has been destroyed.
    UseAfterDestroy,
    /// Create a context without ever destroying it (leak).
    LeakedContext,
    /// Destroy contexts in reverse creation order.
    ReverseDestructionOrder,
    /// Destroy contexts in random order.
    RandomDestructionOrder,
    /// Create more contexts than the GPU supports.
    ContextExhaustion,
    /// Attempt operations on a context from a different thread.
    CrossThreadAccess,
    /// Force a GPU reset while contexts are active.
    DeviceResetDuringUse,
}

impl ChaosScenario {
    /// Returns all 8 chaos scenarios.
    #[must_use]
    pub fn all() -> &'static [Self] {
        &[
            Self::DoubleDestroy,
            Self::UseAfterDestroy,
            Self::LeakedContext,
            Self::ReverseDestructionOrder,
            Self::RandomDestructionOrder,
            Self::ContextExhaustion,
            Self::CrossThreadAccess,
            Self::DeviceResetDuringUse,
        ]
    }

    /// Human-readable description of this scenario.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::DoubleDestroy => "destroy a context twice",
            Self::UseAfterDestroy => "use a context after destruction",
            Self::LeakedContext => "create a context without destroying it",
            Self::ReverseDestructionOrder => "destroy contexts in reverse order",
            Self::RandomDestructionOrder => "destroy contexts in random order",
            Self::ContextExhaustion => "exhaust context creation limit",
            Self::CrossThreadAccess => "access context from wrong thread",
            Self::DeviceResetDuringUse => "reset device with active contexts",
        }
    }
}

impl std::fmt::Display for ChaosScenario {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description())
    }
}

/// Configuration for lifecycle chaos testing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleChaosConfig {
    /// Which scenarios to run.
    pub scenarios: Vec<ChaosScenario>,
    /// Number of contexts to create for exhaustion tests.
    pub max_contexts: u32,
    /// Whether to capture GPU memory snapshots.
    pub capture_memory_snapshots: bool,
}

impl Default for LifecycleChaosConfig {
    fn default() -> Self {
        Self {
            scenarios: ChaosScenario::all().to_vec(),
            max_contexts: 64,
            capture_memory_snapshots: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_scenarios_has_8_entries() {
        assert_eq!(ChaosScenario::all().len(), 8);
    }

    #[test]
    fn scenarios_are_unique() {
        let all = ChaosScenario::all();
        for (i, a) in all.iter().enumerate() {
            for (j, b) in all.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "duplicate scenario at indices {i} and {j}");
                }
            }
        }
    }

    #[test]
    fn display_is_description() {
        for scenario in ChaosScenario::all() {
            assert_eq!(scenario.to_string(), scenario.description());
        }
    }

    #[test]
    fn default_config_includes_all_scenarios() {
        let cfg = LifecycleChaosConfig::default();
        assert_eq!(cfg.scenarios.len(), 8);
        assert_eq!(cfg.max_contexts, 64);
        assert!(cfg.capture_memory_snapshots);
    }
}
