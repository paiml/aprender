//! Soft Jidoka: Stop vs LogAndContinue
//!
//! Per spec §5.1.1: The Stop-the-Line Paradox
//!
//! In a renderfarm, if one bucket fails, you re-render it. In a test suite,
//! a "hard stop" prevents collection of data from other independent blocks.
//!
//! Solution: Distinguish between:
//! - Instrumentation Failures → Hard Stop (can't trust data)
//! - Test Failures → Log & Continue (taint the block, collect others)

use super::BlockId;
use std::collections::HashSet;

/// Jidoka response type (Kaizen: Stop vs Continue)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JidokaAction {
    /// Hard stop - pull the Andon cord (instrumentation failure)
    Stop,
    /// Soft stop - log and continue (test failure, taint the block)
    LogAndContinue,
    /// Warning only - no action needed
    Warn,
}

/// Coverage Jidoka violations with severity classification
#[derive(Debug, Clone)]
pub enum CoverageViolation {
    /// Block executed but not instrumented (CRITICAL - stop)
    UninstrumentedExecution {
        /// The block that was executed without instrumentation
        block_id: BlockId,
    },
    /// Counter overflow (>u64::MAX executions) (WARNING - continue)
    CounterOverflow {
        /// The block whose counter overflowed
        block_id: BlockId,
    },
    /// Impossible edge taken (dead code executed) (CRITICAL - stop)
    ImpossibleEdge {
        /// Source block
        from: BlockId,
        /// Target block
        to: BlockId,
    },
    /// Coverage regression detected (WARNING - continue)
    CoverageRegression {
        /// Expected coverage percentage
        expected: f64,
        /// Actual coverage percentage
        actual: f64,
    },
}

impl CoverageViolation {
    /// Classify violation severity (Soft Jidoka)
    #[must_use]
    pub fn action(&self) -> JidokaAction {
        match self {
            // Instrumentation bugs = hard stop (can't trust data)
            Self::UninstrumentedExecution { .. } | Self::ImpossibleEdge { .. } => {
                JidokaAction::Stop
            }

            // Test failures = log and continue (collect other blocks)
            Self::CounterOverflow { .. } | Self::CoverageRegression { .. } => {
                JidokaAction::LogAndContinue
            }
        }
    }

    /// Get the affected block, if any
    #[must_use]
    pub fn affected_block(&self) -> Option<BlockId> {
        match self {
            Self::UninstrumentedExecution { block_id } | Self::CounterOverflow { block_id } => {
                Some(*block_id)
            }
            Self::ImpossibleEdge { from, .. } => Some(*from),
            Self::CoverageRegression { .. } => None,
        }
    }

    /// Get a human-readable description
    #[must_use]
    pub fn description(&self) -> String {
        match self {
            Self::UninstrumentedExecution { block_id } => {
                format!("Block {} executed but not instrumented", block_id.as_u32())
            }
            Self::CounterOverflow { block_id } => {
                format!("Counter overflow for block {}", block_id.as_u32())
            }
            Self::ImpossibleEdge { from, to } => {
                format!(
                    "Impossible edge {} -> {} executed",
                    from.as_u32(),
                    to.as_u32()
                )
            }
            Self::CoverageRegression { expected, actual } => {
                format!(
                    "Coverage regression: expected {:.1}%, got {:.1}%",
                    expected, actual
                )
            }
        }
    }
}

/// Tainted block tracker (for Soft Jidoka)
///
/// Tracks blocks that encountered non-fatal violations during coverage collection.
/// These blocks are still included in the report but marked as suspect.
#[derive(Debug, Default)]
pub struct TaintedBlocks {
    /// Blocks that encountered non-fatal violations
    tainted: HashSet<BlockId>,
    /// Violation log for each tainted block
    violations: Vec<(BlockId, CoverageViolation)>,
    /// All violations (including those without specific blocks)
    all_violations: Vec<CoverageViolation>,
}

impl TaintedBlocks {
    /// Create a new empty tracker
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark block as tainted (Soft Jidoka)
    pub fn taint(&mut self, block: BlockId, violation: CoverageViolation) {
        let _ = self.tainted.insert(block);
        self.violations.push((block, violation.clone()));
        self.all_violations.push(violation);
    }

    /// Record a violation without a specific block
    pub fn record_violation(&mut self, violation: CoverageViolation) {
        if let Some(block) = violation.affected_block() {
            self.taint(block, violation);
        } else {
            self.all_violations.push(violation);
        }
    }

    /// Check if block is tainted
    #[must_use]
    pub fn is_tainted(&self, block: BlockId) -> bool {
        self.tainted.contains(&block)
    }

    /// Get the number of tainted blocks
    #[must_use]
    pub fn tainted_count(&self) -> usize {
        self.tainted.len()
    }

    /// Get the total number of violations (may exceed tainted blocks)
    #[must_use]
    pub fn violation_count(&self) -> usize {
        self.all_violations.len()
    }

    /// Get all tainted blocks
    #[must_use]
    pub fn tainted_blocks(&self) -> Vec<BlockId> {
        self.tainted.iter().copied().collect()
    }

    /// Get violations for a specific block
    #[must_use]
    pub fn violations_for(&self, block: BlockId) -> Vec<&CoverageViolation> {
        self.violations
            .iter()
            .filter(|(b, _)| *b == block)
            .map(|(_, v)| v)
            .collect()
    }

    /// Get all violations
    #[must_use]
    pub fn all_violations(&self) -> &[CoverageViolation] {
        &self.all_violations
    }

    /// Clear all tainted blocks and violations
    pub fn clear(&mut self) {
        self.tainted.clear();
        self.violations.clear();
        self.all_violations.clear();
    }
}
