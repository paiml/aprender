//! PTX bug types and classifications.

use serde::{Deserialize, Serialize};
use std::fmt;

/// PTX bug severity classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BugSeverity {
    /// P0: Correctness bugs - silent wrong results
    Critical,
    /// P1: Major performance bugs - 10-100x slowdown
    High,
    /// P2: Minor performance bugs - 2-10x slowdown
    Medium,
    /// False positive detection - error handling issues
    FalsePositive,
}

impl fmt::Display for BugSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Critical => write!(f, "P0-CRITICAL"),
            Self::High => write!(f, "P1-HIGH"),
            Self::Medium => write!(f, "P2-MEDIUM"),
            Self::FalsePositive => write!(f, "FALSE-POSITIVE"),
        }
    }
}

/// PTX bug classification
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PtxBugClass {
    /// P0: Shared memory accessed with 64-bit register (should be 32-bit)
    SharedMemU64Addressing,
    /// P0: Loop branches to END label instead of START
    LoopBranchToEnd,
    /// P0: Missing barrier sync between shared memory write and read
    MissingBarrierSync,
    /// P0: Early thread exit before barrier in loop - causes CUDA error 700 (PARITY-114)
    EarlyExitBeforeBarrier,
    /// P1: Accumulator not updated in-place in loop
    NonInPlaceLoopAccumulator,
    /// P1: Register spills to local memory
    RegisterSpills,
    /// P1: High register pressure (>64 regs) reduces occupancy
    HighRegisterPressure,
    /// P1: Predicate register overflow (>8 predicates)
    PredicateOverflow,
    /// P1: Placeholder/incomplete code detected
    PlaceholderCode,
    /// P1: Empty loop body (no actual computation)
    EmptyLoopBody,
    /// P1: Missing thread bounds check
    MissingBoundsCheck,
    /// P2: Redundant register moves
    RedundantMoves,
    /// P2: Unoptimized memory access pattern
    UnoptimizedMemoryPattern,
    /// P2: Dead code after unconditional branch or ret
    DeadCode,
    /// False Positive: Invalid PTX syntax accepted
    InvalidSyntaxAccepted,
    /// False Positive: Missing kernel entry point
    MissingEntryPoint,
}

impl PtxBugClass {
    /// Get the severity of this bug class
    #[must_use]
    pub fn severity(&self) -> BugSeverity {
        match self {
            Self::SharedMemU64Addressing
            | Self::LoopBranchToEnd
            | Self::MissingBarrierSync
            | Self::EarlyExitBeforeBarrier => BugSeverity::Critical,

            Self::NonInPlaceLoopAccumulator
            | Self::RegisterSpills
            | Self::HighRegisterPressure
            | Self::PredicateOverflow
            | Self::PlaceholderCode
            | Self::EmptyLoopBody
            | Self::MissingBoundsCheck => BugSeverity::High,

            Self::RedundantMoves | Self::UnoptimizedMemoryPattern | Self::DeadCode => {
                BugSeverity::Medium
            }

            Self::InvalidSyntaxAccepted | Self::MissingEntryPoint => BugSeverity::FalsePositive,
        }
    }

    /// Get a short code for this bug class
    #[must_use]
    pub fn code(&self) -> &'static str {
        match self {
            Self::SharedMemU64Addressing => "SHARED_U64",
            Self::LoopBranchToEnd => "LOOP_BRANCH_END",
            Self::MissingBarrierSync => "MISSING_BARRIER",
            Self::EarlyExitBeforeBarrier => "EARLY_EXIT_BARRIER",
            Self::NonInPlaceLoopAccumulator => "NON_INPLACE_ACCUM",
            Self::RegisterSpills => "REG_SPILLS",
            Self::HighRegisterPressure => "HIGH_REG_PRESSURE",
            Self::PredicateOverflow => "PRED_OVERFLOW",
            Self::PlaceholderCode => "PLACEHOLDER_CODE",
            Self::EmptyLoopBody => "EMPTY_LOOP",
            Self::MissingBoundsCheck => "NO_BOUNDS_CHECK",
            Self::RedundantMoves => "REDUNDANT_MOV",
            Self::UnoptimizedMemoryPattern => "UNOPT_MEM",
            Self::DeadCode => "DEAD_CODE",
            Self::InvalidSyntaxAccepted => "INVALID_SYNTAX",
            Self::MissingEntryPoint => "NO_ENTRY",
        }
    }
}

impl fmt::Display for PtxBugClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.code())
    }
}

/// A detected PTX bug with location and fix suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PtxBug {
    /// Bug classification
    pub class: PtxBugClass,
    /// Line number (1-indexed, 0 if unknown)
    pub line: usize,
    /// The offending PTX instruction
    pub instruction: String,
    /// Human-readable explanation
    pub message: String,
    /// Suggested fix
    pub fix: Option<String>,
}

impl PtxBug {
    /// Get the severity of this bug
    #[must_use]
    pub fn severity(&self) -> BugSeverity {
        self.class.severity()
    }
}

/// Result of PTX bug hunting analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PtxBugReport {
    /// Kernel name if detected
    pub kernel_name: Option<String>,
    /// List of detected bugs
    pub bugs: Vec<PtxBug>,
    /// Total lines analyzed
    pub lines_analyzed: usize,
    /// Analysis mode (strict or normal)
    pub strict_mode: bool,
}

impl PtxBugReport {
    /// Check if PTX passed all validations (no critical bugs)
    #[must_use]
    pub fn is_valid(&self) -> bool {
        !self.bugs.iter().any(|b| b.severity() == BugSeverity::Critical)
    }

    /// Check if there are any bugs at all
    #[must_use]
    pub fn has_bugs(&self) -> bool {
        !self.bugs.is_empty()
    }

    /// Count bugs by severity
    #[must_use]
    pub fn count_by_severity(&self, severity: BugSeverity) -> usize {
        self.bugs.iter().filter(|b| b.severity() == severity).count()
    }

    /// Check for specific bug class
    #[must_use]
    pub fn has_bug(&self, class: &PtxBugClass) -> bool {
        self.bugs.iter().any(|b| &b.class == class)
    }

    /// Get bugs by class
    #[must_use]
    pub fn bugs_of_class(&self, class: &PtxBugClass) -> Vec<&PtxBug> {
        self.bugs.iter().filter(|b| &b.class == class).collect()
    }

    /// Format as bug report (bashrs style)
    #[must_use]
    pub fn format_report(&self) -> String {
        let mut output = String::new();

        output.push_str(
            "╔══════════════════════════════════════════════════════════════════════════════╗\n",
        );
        output.push_str(
            "║                         PTX BUG HUNTING REPORT                                ║\n",
        );
        output.push_str(
            "╚══════════════════════════════════════════════════════════════════════════════╝\n\n",
        );

        if let Some(name) = &self.kernel_name {
            output.push_str(&format!("Kernel: {}\n", name));
        }
        output.push_str(&format!("PTX Lines Analyzed: {}\n\n", self.lines_analyzed));

        let critical = self.count_by_severity(BugSeverity::Critical);
        let high = self.count_by_severity(BugSeverity::High);
        let medium = self.count_by_severity(BugSeverity::Medium);
        let false_pos = self.count_by_severity(BugSeverity::FalsePositive);

        // P0 Critical
        output.push_str(&format!("P0 CRITICAL BUGS: {}\n", critical));
        if critical > 0 {
            output.push_str("──────────────────\n");
            for (i, bug) in
                self.bugs.iter().filter(|b| b.severity() == BugSeverity::Critical).enumerate()
            {
                output.push_str(&format!("  BUG-{:03}: {}\n", i + 1, bug.class));
                if bug.line > 0 {
                    output.push_str(&format!("    Line {}: {}\n", bug.line, bug.instruction));
                }
                output.push_str(&format!("    Impact: {}\n", bug.message));
                if let Some(fix) = &bug.fix {
                    output.push_str(&format!("    Fix: {}\n", fix));
                }
                output.push('\n');
            }
        }

        // P1 High
        output.push_str(&format!("\nP1 HIGH BUGS: {}\n", high));
        if high > 0 {
            output.push_str("─────────────────\n");
            for bug in self.bugs.iter().filter(|b| b.severity() == BugSeverity::High) {
                output.push_str(&format!("  {}: {}\n", bug.class, bug.message));
            }
        }

        // P2 Medium
        output.push_str(&format!("\nP2 MEDIUM BUGS: {}\n", medium));
        if medium > 0 {
            output.push_str("─────────────────\n");
            for bug in self.bugs.iter().filter(|b| b.severity() == BugSeverity::Medium) {
                output.push_str(&format!("  {}: {}\n", bug.class, bug.message));
            }
        }

        // False positives
        output.push_str(&format!("\nFALSE POSITIVES DETECTED: {}\n", false_pos));

        // Summary
        output.push_str("\nSUMMARY\n═══════\n");
        output.push_str(&format!("  Total Bugs: {}\n", self.bugs.len()));
        output.push_str(&format!("  P0 Critical: {}", critical));
        if critical > 0 {
            output.push_str(" ← BLOCKS RELEASE");
        }
        output.push('\n');
        output.push_str(&format!("  P1 High: {}\n", high));
        output.push_str(&format!("  P2 Medium: {}\n", medium));

        output
    }
}
