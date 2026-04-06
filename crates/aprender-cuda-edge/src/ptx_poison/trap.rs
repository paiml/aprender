//! PTX poison trap result tracking.
//!
//! After applying mutations to PTX source and running tests, this module
//! tracks which mutations were detected (killed) vs. which survived.

use serde::{Deserialize, Serialize};

use super::mutator::PtxMutator;

/// Result of running tests against a single mutation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MutantResult {
    /// The mutation was detected (test failed as expected).
    Killed,
    /// The mutation was not detected (test passed — bad coverage).
    Survived,
    /// The mutation caused a compilation error (not counted).
    CompileError,
    /// The mutation caused a timeout (killed by default).
    Timeout,
}

impl MutantResult {
    /// Returns true if this mutation was effectively killed.
    #[must_use]
    pub fn is_killed(&self) -> bool {
        matches!(self, Self::Killed | Self::Timeout)
    }
}

/// Configuration for PTX poison trap testing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PtxPoisonTrapConfig {
    /// Mutators to apply.
    pub mutators: Vec<PtxMutator>,
    /// Timeout per test run in milliseconds.
    pub timeout_ms: u64,
    /// Whether to continue after first surviving mutation.
    pub continue_on_survivor: bool,
}

impl Default for PtxPoisonTrapConfig {
    fn default() -> Self {
        Self {
            mutators: super::mutator::default_mutators(),
            timeout_ms: 5000,
            continue_on_survivor: true,
        }
    }
}

/// Report from a PTX poison trap session.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PoisonTrapReport {
    /// Results for each mutation attempted.
    pub results: Vec<(PtxMutator, MutantResult)>,
    /// Total PTX source files processed.
    pub sources_processed: u32,
}

impl PoisonTrapReport {
    /// Calculate the mutation score (killed / total applicable).
    ///
    /// Compile errors are excluded from the denominator.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn mutation_score(&self) -> f64 {
        let applicable: Vec<_> = self
            .results
            .iter()
            .filter(|(_, r)| !matches!(r, MutantResult::CompileError))
            .collect();

        if applicable.is_empty() {
            return 1.0;
        }

        let killed = applicable.iter().filter(|(_, r)| r.is_killed()).count();
        killed as f64 / applicable.len() as f64
    }

    /// Returns all surviving mutations.
    #[must_use]
    pub fn survivors(&self) -> Vec<&PtxMutator> {
        self.results
            .iter()
            .filter(|(_, r)| matches!(r, MutantResult::Survived))
            .map(|(m, _)| m)
            .collect()
    }

    /// Returns true if all mutations were killed.
    #[must_use]
    pub fn all_killed(&self) -> bool {
        self.survivors().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mutant_result_is_killed() {
        assert!(MutantResult::Killed.is_killed());
        assert!(MutantResult::Timeout.is_killed());
        assert!(!MutantResult::Survived.is_killed());
        assert!(!MutantResult::CompileError.is_killed());
    }

    #[test]
    fn mutation_score_all_killed() {
        let report = PoisonTrapReport {
            results: vec![
                (PtxMutator::FlipAddSub, MutantResult::Killed),
                (PtxMutator::FlipMulDiv, MutantResult::Killed),
            ],
            sources_processed: 1,
        };
        assert!((report.mutation_score() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn mutation_score_excludes_compile_errors() {
        let report = PoisonTrapReport {
            results: vec![
                (PtxMutator::FlipAddSub, MutantResult::Killed),
                (PtxMutator::FlipMulDiv, MutantResult::CompileError),
            ],
            sources_processed: 1,
        };
        // Only 1 applicable, 1 killed → 100%
        assert!((report.mutation_score() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn mutation_score_with_survivor() {
        let report = PoisonTrapReport {
            results: vec![
                (PtxMutator::FlipAddSub, MutantResult::Killed),
                (PtxMutator::FlipMulDiv, MutantResult::Survived),
            ],
            sources_processed: 1,
        };
        assert!((report.mutation_score() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn survivors_list() {
        let report = PoisonTrapReport {
            results: vec![
                (PtxMutator::FlipAddSub, MutantResult::Killed),
                (PtxMutator::FlipMulDiv, MutantResult::Survived),
                (PtxMutator::RemoveBarrier, MutantResult::Survived),
            ],
            sources_processed: 1,
        };
        let survivors = report.survivors();
        assert_eq!(survivors.len(), 2);
    }

    #[test]
    fn all_killed_true_when_no_survivors() {
        let report = PoisonTrapReport {
            results: vec![
                (PtxMutator::FlipAddSub, MutantResult::Killed),
                (PtxMutator::FlipMulDiv, MutantResult::Timeout),
                (PtxMutator::RemoveBarrier, MutantResult::CompileError),
            ],
            sources_processed: 1,
        };
        assert!(report.all_killed());
    }

    #[test]
    fn all_killed_false_when_survivors_exist() {
        let report = PoisonTrapReport {
            results: vec![
                (PtxMutator::FlipAddSub, MutantResult::Killed),
                (PtxMutator::FlipMulDiv, MutantResult::Survived),
            ],
            sources_processed: 1,
        };
        assert!(!report.all_killed());
    }

    #[test]
    fn config_default_has_all_mutators() {
        let config = PtxPoisonTrapConfig::default();
        assert_eq!(config.mutators.len(), 8);
        assert_eq!(config.timeout_ms, 5000);
        assert!(config.continue_on_survivor);
    }

    #[test]
    fn mutation_score_empty_results() {
        let report = PoisonTrapReport {
            results: vec![],
            sources_processed: 0,
        };
        // Empty is 100% by definition
        assert!((report.mutation_score() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn mutation_score_all_compile_errors() {
        let report = PoisonTrapReport {
            results: vec![
                (PtxMutator::FlipAddSub, MutantResult::CompileError),
                (PtxMutator::FlipMulDiv, MutantResult::CompileError),
            ],
            sources_processed: 1,
        };
        // No applicable mutations, returns 1.0
        assert!((report.mutation_score() - 1.0).abs() < f64::EPSILON);
    }
}
