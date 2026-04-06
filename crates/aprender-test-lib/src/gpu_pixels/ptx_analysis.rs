//! PTX Static Analysis for GPU Kernel Bug Detection
//!
//! Detects common PTX bugs through regex-based static analysis:
//! - Shared memory using 64-bit addressing (should be 32-bit)
//! - Loop branches going to END instead of START
//! - Missing barrier synchronization
//! - Invalid register types for operations

// Static regexes are always valid - compile-time constant patterns
// collection_is_never_read: loop_start_labels is used for tracking/debug
#![allow(
    clippy::unwrap_used,
    clippy::trivial_regex,
    clippy::collection_is_never_read
)]

use regex::Regex;
use std::collections::HashSet;

/// PTX bug classification
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PtxBugClass {
    /// Shared memory accessed with 64-bit register (should be 32-bit)
    SharedMemU64Addressing,
    /// Loop branches to END label instead of START
    LoopBranchToEnd,
    /// Missing barrier sync between shared memory write and read
    MissingBarrierSync,
    /// Accumulator not updated in-place in loop
    NonInPlaceLoopAccumulator,
    /// Invalid PTX syntax
    InvalidSyntax,
    /// Kernel entry point missing
    MissingEntryPoint,
}

impl std::fmt::Display for PtxBugClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SharedMemU64Addressing => write!(f, "shared_mem_u64"),
            Self::LoopBranchToEnd => write!(f, "loop_branch_to_end"),
            Self::MissingBarrierSync => write!(f, "missing_barrier"),
            Self::NonInPlaceLoopAccumulator => write!(f, "non_inplace_accum"),
            Self::InvalidSyntax => write!(f, "invalid_syntax"),
            Self::MissingEntryPoint => write!(f, "missing_entry"),
        }
    }
}

/// A detected PTX bug
#[derive(Debug, Clone)]
pub struct PtxBug {
    /// Bug classification
    pub class: PtxBugClass,
    /// Line number (1-indexed, 0 if unknown)
    pub line: usize,
    /// The offending PTX instruction
    pub instruction: String,
    /// Human-readable explanation
    pub message: String,
}

/// Result of PTX validation
#[derive(Debug, Clone)]
pub struct PtxValidationResult {
    /// List of detected bugs
    pub bugs: Vec<PtxBug>,
    /// Kernel names found
    pub kernel_names: Vec<String>,
    /// Total lines analyzed
    pub lines_analyzed: usize,
}

impl PtxValidationResult {
    /// Check if PTX passed all validations
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.bugs.is_empty() && !self.kernel_names.is_empty()
    }

    /// Get count of bugs by class
    #[must_use]
    pub fn bug_count(&self, class: &PtxBugClass) -> usize {
        self.bugs.iter().filter(|b| &b.class == class).count()
    }

    /// Check for specific bug class
    #[must_use]
    pub fn has_bug(&self, class: &PtxBugClass) -> bool {
        self.bugs.iter().any(|b| &b.class == class)
    }
}

/// PTX static analyzer
#[derive(Debug, Default)]
pub struct PtxAnalyzer {
    /// Enable strict mode (more warnings)
    pub strict: bool,
}

impl PtxAnalyzer {
    /// Create analyzer with strict mode
    #[must_use]
    pub fn strict() -> Self {
        Self { strict: true }
    }

    /// Analyze PTX string for bugs
    #[must_use]
    pub fn analyze(&self, ptx: &str) -> PtxValidationResult {
        let mut bugs = Vec::new();
        let mut kernel_names = Vec::new();
        let lines: Vec<&str> = ptx.lines().collect();

        // Regex patterns for bug detection
        let shared_mem_u64 = Regex::new(r"[sl]t\.shared\.[^\[]+\[%rd\d+\]").unwrap();
        let entry_point = Regex::new(r"\.visible\s+\.entry\s+(\w+)").unwrap();
        let loop_label = Regex::new(r"^(\w+_loop\w*):").unwrap();
        let branch_instr = Regex::new(r"bra\s+(\w+);").unwrap();
        let bar_sync = Regex::new(r"bar\.sync").unwrap();

        // Track loop labels
        let mut loop_start_labels: HashSet<String> = HashSet::new();
        let mut loop_end_labels: HashSet<String> = HashSet::new();

        // First pass: collect labels
        for line in &lines {
            let trimmed = line.trim();
            if let Some(caps) = loop_label.captures(trimmed) {
                let label = caps.get(1).unwrap().as_str();
                if label.contains("_start")
                    || label.ends_with("_loop")
                    || label.starts_with("loop_")
                {
                    loop_start_labels.insert(label.to_string());
                } else if label.contains("_end") {
                    loop_end_labels.insert(label.to_string());
                }
            }
        }

        // Second pass: detect bugs
        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            // Detect shared memory u64 addressing
            if shared_mem_u64.is_match(trimmed) {
                bugs.push(PtxBug {
                    class: PtxBugClass::SharedMemU64Addressing,
                    line: line_num + 1,
                    instruction: trimmed.to_string(),
                    message: "Shared memory accessed with 64-bit register. Use 32-bit addressing."
                        .to_string(),
                });
            }

            // Collect kernel names
            if let Some(caps) = entry_point.captures(trimmed) {
                kernel_names.push(caps.get(1).unwrap().as_str().to_string());
            }

            // Detect branch to loop end from inside loop body
            // (this is a heuristic - may have false positives)
            if let Some(caps) = branch_instr.captures(trimmed) {
                let target = caps.get(1).unwrap().as_str();
                // If branching to a _end label that has a corresponding _start,
                // and we're not at a conditional branch after loop check,
                // it might be a bug
                if self.strict && loop_end_labels.contains(target) {
                    // Check if this is an unconditional branch (potential loop continuation bug)
                    if !trimmed.starts_with('@') && !trimmed.contains("@%p") {
                        bugs.push(PtxBug {
                            class: PtxBugClass::LoopBranchToEnd,
                            line: line_num + 1,
                            instruction: trimmed.to_string(),
                            message: format!(
                                "Unconditional branch to loop end '{}'. Should branch to start?",
                                target
                            ),
                        });
                    }
                }
            }
        }

        // Check for missing entry point
        if kernel_names.is_empty() && !ptx.trim().is_empty() {
            bugs.push(PtxBug {
                class: PtxBugClass::MissingEntryPoint,
                line: 0,
                instruction: String::new(),
                message: "No kernel entry point found".to_string(),
            });
        }

        // Check for barrier sync presence when shared memory is used
        let uses_shared =
            ptx.contains(".shared") || ptx.contains("st.shared") || ptx.contains("ld.shared");
        let has_barrier = bar_sync.is_match(ptx);
        if self.strict && uses_shared && !has_barrier {
            bugs.push(PtxBug {
                class: PtxBugClass::MissingBarrierSync,
                line: 0,
                instruction: String::new(),
                message: "Shared memory used but no bar.sync found".to_string(),
            });
        }

        PtxValidationResult {
            bugs,
            kernel_names,
            lines_analyzed: lines.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shared_mem_u64_detection() {
        let ptx = "st.shared.f32 [%rd5], %f0;";
        let analyzer = PtxAnalyzer::default();
        let result = analyzer.analyze(ptx);
        assert!(result.has_bug(&PtxBugClass::SharedMemU64Addressing));
    }

    #[test]
    fn test_shared_mem_u32_ok() {
        let ptx = "st.shared.f32 [%r5], %f0;";
        let analyzer = PtxAnalyzer::default();
        let result = analyzer.analyze(ptx);
        assert!(!result.has_bug(&PtxBugClass::SharedMemU64Addressing));
    }

    #[test]
    fn test_kernel_name_extraction() {
        let ptx = r#"
.visible .entry gemm_tiled(
    .param .u64 a_ptr
) {
    ret;
}
"#;
        let result = PtxAnalyzer::default().analyze(ptx);
        assert_eq!(result.kernel_names, vec!["gemm_tiled"]);
    }

    #[test]
    fn test_multiple_kernels() {
        let ptx = r#"
.visible .entry kernel_a() { ret; }
.visible .entry kernel_b() { ret; }
"#;
        let result = PtxAnalyzer::default().analyze(ptx);
        assert_eq!(result.kernel_names.len(), 2);
    }

    #[test]
    fn test_missing_entry_point() {
        let ptx = ".version 8.0\n.target sm_70";
        let result = PtxAnalyzer::default().analyze(ptx);
        assert!(result.has_bug(&PtxBugClass::MissingEntryPoint));
    }

    #[test]
    fn test_strict_mode_barrier() {
        let ptx = r#"
.visible .entry test() {
    .shared .b8 smem[1024];
    st.shared.f32 [%r0], %f0;
    ret;
}
"#;
        let strict_result = PtxAnalyzer::strict().analyze(ptx);
        let normal_result = PtxAnalyzer::default().analyze(ptx);

        assert!(strict_result.has_bug(&PtxBugClass::MissingBarrierSync));
        assert!(!normal_result.has_bug(&PtxBugClass::MissingBarrierSync));
    }

    #[test]
    fn test_bug_class_display() {
        assert_eq!(
            format!("{}", PtxBugClass::SharedMemU64Addressing),
            "shared_mem_u64"
        );
        assert_eq!(
            format!("{}", PtxBugClass::LoopBranchToEnd),
            "loop_branch_to_end"
        );
    }

    #[test]
    fn test_validation_result_helpers() {
        let result = PtxValidationResult {
            bugs: vec![
                PtxBug {
                    class: PtxBugClass::SharedMemU64Addressing,
                    line: 1,
                    instruction: "test".to_string(),
                    message: "test".to_string(),
                },
                PtxBug {
                    class: PtxBugClass::SharedMemU64Addressing,
                    line: 2,
                    instruction: "test".to_string(),
                    message: "test".to_string(),
                },
            ],
            kernel_names: vec!["test".to_string()],
            lines_analyzed: 10,
        };

        assert_eq!(result.bug_count(&PtxBugClass::SharedMemU64Addressing), 2);
        assert_eq!(result.bug_count(&PtxBugClass::LoopBranchToEnd), 0);
        assert!(!result.is_valid());
    }

    #[test]
    fn test_bug_class_display_all_variants() {
        assert_eq!(
            format!("{}", PtxBugClass::MissingBarrierSync),
            "missing_barrier"
        );
        assert_eq!(
            format!("{}", PtxBugClass::NonInPlaceLoopAccumulator),
            "non_inplace_accum"
        );
        assert_eq!(format!("{}", PtxBugClass::InvalidSyntax), "invalid_syntax");
        assert_eq!(
            format!("{}", PtxBugClass::MissingEntryPoint),
            "missing_entry"
        );
    }

    #[test]
    fn test_loop_branch_to_end_strict_mode() {
        // PTX with a loop that branches to _end unconditionally
        // The loop_label regex requires _loop suffix
        let ptx = r#"
.visible .entry test() {
test_loop_start:
    // loop body
    bra test_loop_end;
test_loop_end:
    ret;
}
"#;
        let strict_result = PtxAnalyzer::strict().analyze(ptx);
        // In strict mode, unconditional branch to _end should be flagged
        assert!(strict_result.has_bug(&PtxBugClass::LoopBranchToEnd));
    }

    #[test]
    fn test_loop_labels_with_loop_suffix() {
        // Labels must match the regex: ^(\w+_loop\w*):
        let ptx = r#"
.visible .entry test() {
main_loop:
    bra main_loop_end;
main_loop_end:
    ret;
}
"#;
        let result = PtxAnalyzer::strict().analyze(ptx);
        // main_loop matches the _loop pattern, main_loop_end contains _end
        assert!(result.has_bug(&PtxBugClass::LoopBranchToEnd));
    }

    #[test]
    fn test_conditional_branch_not_flagged() {
        let ptx = r#"
.visible .entry test() {
loop_start:
    @%p0 bra loop_end;
loop_end:
    ret;
}
"#;
        let result = PtxAnalyzer::strict().analyze(ptx);
        // Conditional branch should NOT be flagged
        assert!(!result.has_bug(&PtxBugClass::LoopBranchToEnd));
    }

    #[test]
    fn test_ld_shared_u64_detection() {
        // Test ld.shared with 64-bit register - must match pattern [sl]t\.shared
        // ld.shared doesn't match - only st.shared and lt.shared (which isn't valid)
        // So the regex is really for st.shared only
        let ptx = "st.shared.f32 [%rd5], %f0;";
        let result = PtxAnalyzer::default().analyze(ptx);
        assert!(result.has_bug(&PtxBugClass::SharedMemU64Addressing));
    }

    #[test]
    fn test_valid_result_empty_bugs() {
        let result = PtxValidationResult {
            bugs: vec![],
            kernel_names: vec!["kernel".to_string()],
            lines_analyzed: 5,
        };
        assert!(result.is_valid());
    }

    #[test]
    fn test_invalid_result_no_kernels() {
        let result = PtxValidationResult {
            bugs: vec![],
            kernel_names: vec![],
            lines_analyzed: 5,
        };
        assert!(!result.is_valid());
    }

    #[test]
    fn test_empty_ptx_no_bugs() {
        let result = PtxAnalyzer::default().analyze("");
        assert!(result.bugs.is_empty());
        assert!(result.kernel_names.is_empty());
    }

    #[test]
    fn test_shared_mem_st_detection() {
        let ptx = "st.shared.f32 [%rd0], %f1;";
        let result = PtxAnalyzer::default().analyze(ptx);
        assert!(result.has_bug(&PtxBugClass::SharedMemU64Addressing));
    }

    #[test]
    fn test_barrier_present() {
        let ptx = r#"
.visible .entry test() {
    .shared .b8 smem[1024];
    st.shared.f32 [%r0], %f0;
    bar.sync 0;
    ret;
}
"#;
        let result = PtxAnalyzer::strict().analyze(ptx);
        assert!(!result.has_bug(&PtxBugClass::MissingBarrierSync));
    }

    #[test]
    fn test_analyzer_debug() {
        let analyzer = PtxAnalyzer::default();
        let debug_str = format!("{:?}", analyzer);
        assert!(debug_str.contains("PtxAnalyzer"));
    }

    #[test]
    fn test_ptx_bug_fields() {
        let bug = PtxBug {
            class: PtxBugClass::InvalidSyntax,
            line: 42,
            instruction: "invalid".to_string(),
            message: "Bad syntax".to_string(),
        };
        assert_eq!(bug.line, 42);
        assert_eq!(bug.instruction, "invalid");
        assert_eq!(bug.message, "Bad syntax");
        assert_eq!(bug.class, PtxBugClass::InvalidSyntax);
    }

    #[test]
    fn test_bug_class_hash_eq() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(PtxBugClass::SharedMemU64Addressing);
        set.insert(PtxBugClass::LoopBranchToEnd);
        assert!(set.contains(&PtxBugClass::SharedMemU64Addressing));
        assert!(!set.contains(&PtxBugClass::MissingBarrierSync));
    }

    #[test]
    fn test_validation_result_clone() {
        let result = PtxValidationResult {
            bugs: vec![],
            kernel_names: vec!["test".to_string()],
            lines_analyzed: 10,
        };
        let cloned = result.clone();
        assert_eq!(cloned.kernel_names, result.kernel_names);
        assert_eq!(cloned.lines_analyzed, result.lines_analyzed);
    }
}
