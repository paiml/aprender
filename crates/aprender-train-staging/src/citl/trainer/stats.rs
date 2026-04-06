//! Decision statistics types for CITL trainer

#![allow(clippy::field_reassign_with_default)]

use super::{CompilationOutcome, DecisionTrace};

/// Session data for a single compilation
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields used for future session replay/analysis
pub(crate) struct Session {
    /// Session ID
    pub(crate) id: String,
    /// Decision traces
    pub(crate) decisions: Vec<DecisionTrace>,
    /// Compilation outcome
    pub(crate) outcome: CompilationOutcome,
    /// Optional fix diff (if error was fixed)
    pub(crate) fix_diff: Option<String>,
}

/// Statistics for a decision type across sessions
#[derive(Debug, Clone, Default)]
pub struct DecisionStats {
    /// Times seen in successful sessions
    pub success_count: u32,
    /// Times seen in failed sessions
    pub fail_count: u32,
    /// Total successful sessions
    pub total_success: u32,
    /// Total failed sessions
    pub total_fail: u32,
}

impl DecisionStats {
    /// Calculate Tarantula suspiciousness score
    ///
    /// Suspiciousness = (fail_freq) / (fail_freq + success_freq)
    /// where fail_freq = fail_count / total_fail
    /// and success_freq = success_count / total_success
    #[must_use]
    pub fn tarantula_score(&self) -> f32 {
        if self.total_fail == 0 || self.fail_count == 0 {
            return 0.0;
        }

        let fail_freq = self.fail_count as f32 / self.total_fail as f32;
        let success_freq = if self.total_success > 0 {
            self.success_count as f32 / self.total_success as f32
        } else {
            0.0
        };

        if fail_freq + success_freq < f32::EPSILON {
            0.0
        } else {
            fail_freq / (fail_freq + success_freq)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decision_stats_tarantula() {
        let mut stats = DecisionStats::default();
        stats.success_count = 2;
        stats.fail_count = 8;
        stats.total_success = 10;
        stats.total_fail = 10;

        // fail_freq = 8/10 = 0.8
        // success_freq = 2/10 = 0.2
        // suspiciousness = 0.8 / (0.8 + 0.2) = 0.8
        assert!((stats.tarantula_score() - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_decision_stats_tarantula_no_failures() {
        let stats =
            DecisionStats { success_count: 5, fail_count: 0, total_success: 5, total_fail: 0 };
        assert_eq!(stats.tarantula_score(), 0.0);
    }

    #[test]
    fn test_decision_stats_tarantula_only_failures() {
        let stats =
            DecisionStats { success_count: 0, fail_count: 5, total_success: 0, total_fail: 5 };
        // fail_freq = 1.0, success_freq = 0.0
        // suspiciousness = 1.0 / 1.0 = 1.0
        assert_eq!(stats.tarantula_score(), 1.0);
    }
}

#[cfg(test)]
mod prop_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_tarantula_score_bounded(
            success in 0u32..100,
            fail in 0u32..100,
            total_success in 1u32..100,
            total_fail in 1u32..100
        ) {
            let stats = DecisionStats {
                success_count: success.min(total_success),
                fail_count: fail.min(total_fail),
                total_success,
                total_fail,
            };

            let score = stats.tarantula_score();
            prop_assert!(score >= 0.0);
            prop_assert!(score <= 1.0);
        }
    }
}
