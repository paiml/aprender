use regex::Regex;
use std::collections::HashSet;

use super::super::super::types::{PtxBug, PtxBugClass};
use super::super::PtxBugAnalyzer;
use trueno_gpu::ptx::optimize::barrier_safety;

impl PtxBugAnalyzer {
    /// Detect loop branches to END instead of START
    pub(in crate::ptx::bugs::analyzer) fn detect_loop_branch_to_end(
        &self,
        _ptx: &str,
        lines: &[&str],
    ) -> Vec<PtxBug> {
        let mut bugs = Vec::new();

        if !self.strict {
            return bugs;
        }

        // Collect loop labels
        let loop_label =
            Regex::new(r"^(\w+(?:_loop|loop_)\w*):").expect("invariant: regex pattern is valid");
        let branch_instr =
            Regex::new(r"^\s*bra\s+(\w+);").expect("invariant: regex pattern is valid");

        let mut loop_start_labels: HashSet<String> = HashSet::new();
        let mut loop_end_labels: HashSet<String> = HashSet::new();

        // First pass: collect labels
        for line in lines {
            let trimmed = line.trim();
            if let Some(caps) = loop_label.captures(trimmed) {
                let label = caps.get(1).expect("invariant: capture group exists").as_str();
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

        // Second pass: detect unconditional branches to end labels
        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if let Some(caps) = branch_instr.captures(trimmed) {
                let target = caps.get(1).expect("invariant: capture group exists").as_str();
                // Unconditional branch (not @%p prefixed) to _end label
                if loop_end_labels.contains(target) && !trimmed.starts_with('@') {
                    bugs.push(PtxBug {
                        class: PtxBugClass::LoopBranchToEnd,
                        line: line_num + 1,
                        instruction: trimmed.to_string(),
                        message: format!(
                            "Unconditional branch to loop end '{}'. Should branch to start?",
                            target
                        ),
                        fix: Some(format!(
                            "Change target from {} to corresponding _start label",
                            target
                        )),
                    });
                }
            }
        }

        bugs
    }

    /// Detect early thread exit before barrier in loop (PARITY-114)
    ///
    /// This is the root cause of CUDA error 700 (illegal instruction) when
    /// some threads in a warp exit early via `bra exit` before reaching a
    /// `bar.sync` instruction. The remaining threads hang waiting at the barrier.
    ///
    /// Uses trueno-gpu's `barrier_safety` analyzer for consistent detection.
    pub(in crate::ptx::bugs::analyzer) fn detect_early_exit_before_barrier(
        &self,
        ptx: &str,
    ) -> Vec<PtxBug> {
        let mut bugs = Vec::new();

        // Only check in strict mode (matches MissingBarrierSync behavior)
        if !self.strict {
            return bugs;
        }

        // Use the authoritative barrier_safety analyzer from trueno-gpu
        let result = barrier_safety::analyze(ptx);

        for violation in result.violations {
            let kind = match violation.kind {
                barrier_safety::ViolationKind::EarlyExitBeforeBarrier => {
                    "Unconditional early exit before barrier"
                }
                barrier_safety::ViolationKind::ConditionalExitBeforeBarrier => {
                    "Conditional early exit may cause thread divergence at barrier"
                }
                barrier_safety::ViolationKind::MissingBarrierAfterSharedAccess => {
                    continue; // Already handled by detect_missing_barrier_sync
                }
            };

            bugs.push(PtxBug {
                class: PtxBugClass::EarlyExitBeforeBarrier,
                line: violation.line,
                instruction: violation.instruction,
                message: format!(
                    "PARITY-114: {} - causes CUDA error 700. {}",
                    kind, violation.context
                ),
                fix: Some(
                    "Move bounds check AFTER loop body. Use predicated loads (store 0 first) \
                     so all threads participate in bar.sync regardless of bounds."
                        .to_string(),
                ),
            });
        }

        bugs
    }

    /// Detect empty loop body (P1)
    /// A loop that branches back without doing any computation
    pub(in crate::ptx::bugs::analyzer) fn detect_empty_loop_body(
        &self,
        _ptx: &str,
        lines: &[&str],
    ) -> Vec<PtxBug> {
        let mut bugs = Vec::new();
        let label_pattern = Regex::new(r"^(\w+):$").expect("invariant: regex pattern is valid");
        let branch_pattern = Regex::new(r"^\s*(?:@%\w+\s+)?bra\s+(\w+);")
            .expect("invariant: regex pattern is valid");

        for (i, line) in lines.iter().enumerate() {
            let line = line.trim();
            if let Some(label_caps) = label_pattern.captures(line) {
                let label = label_caps.get(1).expect("invariant: capture group exists").as_str();
                if scan_empty_loop(lines, i, label, &branch_pattern) {
                    bugs.push(PtxBug {
                        class: PtxBugClass::EmptyLoopBody,
                        line: i + 1,
                        instruction: format!("Loop '{}' at line {}", label, i + 1),
                        message: "Loop body contains no computation - may be placeholder code"
                            .to_string(),
                        fix: Some("Implement loop body or remove empty loop".to_string()),
                    });
                }
            }
        }

        bugs
    }

    /// Detect dead code (P2)
    /// Code after unconditional ret or bra that can never execute
    pub(in crate::ptx::bugs::analyzer) fn detect_dead_code(
        &self,
        _ptx: &str,
        lines: &[&str],
    ) -> Vec<PtxBug> {
        let mut bugs = Vec::new();

        let unconditional_ret = Regex::new(r"^\s*ret;").expect("invariant: regex pattern is valid");
        let unconditional_bra =
            Regex::new(r"^\s*bra\s+\w+;").expect("invariant: regex pattern is valid"); // No @%p prefix
        let label_pattern = Regex::new(r"^\w+:$").expect("invariant: regex pattern is valid");

        let mut after_unconditional = false;
        let mut unconditional_line = 0;

        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            // Skip empty lines and comments
            if trimmed.is_empty() || trimmed.starts_with("//") {
                continue;
            }

            // Check if this is a label (reachable code)
            if label_pattern.is_match(trimmed) {
                after_unconditional = false;
                continue;
            }

            // Check if this is closing brace
            if trimmed == "}" {
                after_unconditional = false;
                continue;
            }

            // Check if we're after an unconditional jump
            if after_unconditional {
                bugs.push(PtxBug {
                    class: PtxBugClass::DeadCode,
                    line: line_num + 1,
                    instruction: trimmed.to_string(),
                    message: format!(
                        "Dead code: unreachable after unconditional jump at line {}",
                        unconditional_line + 1
                    ),
                    fix: Some("Remove unreachable code or add label".to_string()),
                });
                // Only report once per dead code block
                after_unconditional = false;
                continue;
            }

            // Check for unconditional ret
            if unconditional_ret.is_match(trimmed) {
                after_unconditional = true;
                unconditional_line = line_num;
            }

            // Check for unconditional bra (not predicated)
            if unconditional_bra.is_match(trimmed) && !trimmed.starts_with('@') {
                after_unconditional = true;
                unconditional_line = line_num;
            }
        }

        bugs
    }
}

const COMPUTE_OPS: &[&str] = &[
    "add.", "sub.", "mul.", "div.", "fma.", "mad.", "ld.", "st.", "cvt.", "mov.", "setp.", "and.",
    "or.", "xor.", "shl.", "shr.", "min.", "max.", "abs.", "neg.", "rcp.", "sqrt.", "rsqrt.",
    "sin.", "cos.", "ex2.", "lg2.",
];

/// Returns `true` if the line is empty or a comment (should be skipped in analysis).
fn is_skip_line(line: &str) -> bool {
    line.is_empty() || line.starts_with("//")
}

/// Returns `true` if the line contains a PTX compute operation.
fn has_compute_op(line: &str) -> bool {
    COMPUTE_OPS.iter().any(|op| line.contains(op))
}

/// Returns `true` if the line is a loop-end label (e.g. `some_end:` or `LOOP_END:`).
fn is_end_label(line: &str) -> bool {
    line.ends_with(':') && (line.contains("_end") || line.contains("END"))
}

/// If the line is a branch instruction, returns the branch target label.
fn branch_target<'a>(line: &'a str, branch_re: &Regex) -> Option<&'a str> {
    branch_re
        .captures(line)
        .map(|caps| caps.get(1).expect("invariant: capture group exists").as_str())
}

/// Scan forward from a label to determine if a loop body is empty (no compute ops).
fn scan_empty_loop(lines: &[&str], start: usize, label: &str, branch_re: &Regex) -> bool {
    let mut has_computation = false;
    let mut found_back_edge = false;

    for inner_line in lines.iter().skip(start + 1).take(20) {
        let inner = inner_line.trim();
        if is_skip_line(inner) {
            continue;
        }
        if has_compute_op(inner) {
            has_computation = true;
        }
        if branch_target(inner, branch_re) == Some(label) {
            found_back_edge = true;
            break;
        }
        if is_end_label(inner) {
            break;
        }
    }

    found_back_edge && !has_computation
}
