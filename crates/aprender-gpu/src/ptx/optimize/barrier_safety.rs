//! Barrier Safety Analysis (PARITY-114 Prevention)
//!
//! Static analysis to detect early-exit-before-barrier patterns that cause
//! CUDA error 700 (thread divergence at barriers).
//!
//! ## Five Whys Root Cause
//!
//! `ptxas` validates syntax but NOT semantics. Early-exit-before-barrier is
//! syntactically valid PTX but causes runtime hangs when some threads exit
//! while others wait at `bar.sync`.
//!
//! ## Detection Strategy
//!
//! 1. Find all `bar.sync` instructions
//! 2. Find all unconditional `bra exit` or `ret` instructions
//! 3. Check if any exit is reachable before a barrier in a loop
//!
//! ## cuda-tile-behavior.md References
//!
//! - Section 4.1: Barrier synchronization requirements
//! - Falsification tests #81-90: Barrier safety validation

use std::collections::HashSet;

/// Barrier safety analysis result
#[derive(Debug, Clone, PartialEq)]
pub struct BarrierSafetyResult {
    /// Whether the PTX is barrier-safe
    pub is_safe: bool,
    /// List of violations found
    pub violations: Vec<BarrierViolation>,
    /// Number of barriers found
    pub barrier_count: usize,
    /// Number of exit points found
    pub exit_count: usize,
}

/// A barrier safety violation
#[derive(Debug, Clone, PartialEq)]
pub struct BarrierViolation {
    /// Line number (1-indexed) where violation occurs
    pub line: usize,
    /// Type of violation
    pub kind: ViolationKind,
    /// The offending instruction
    pub instruction: String,
    /// Context: what loop/block contains this
    pub context: String,
}

/// Types of barrier safety violations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViolationKind {
    /// Unconditional exit before barrier in a loop
    EarlyExitBeforeBarrier,
    /// Conditional exit that could cause divergence
    ConditionalExitBeforeBarrier,
    /// Missing barrier after shared memory access
    MissingBarrierAfterSharedAccess,
}

impl std::fmt::Display for ViolationKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EarlyExitBeforeBarrier => write!(f, "PARITY-114: Early exit before barrier"),
            Self::ConditionalExitBeforeBarrier => {
                write!(f, "PARITY-114: Conditional exit may cause divergence")
            }
            Self::MissingBarrierAfterSharedAccess => {
                write!(f, "Missing barrier after shared memory access")
            }
        }
    }
}

/// Analyze PTX for barrier safety violations
///
/// # Arguments
///
/// * `ptx` - PTX source code
///
/// # Returns
///
/// Analysis result with any violations found
///
/// # Example
///
/// ```
/// use trueno_gpu::ptx::optimize::barrier_safety::analyze;
///
/// let ptx = "..."; // PTX source
/// let result = analyze(ptx);
/// assert!(result.is_safe, "PTX should be barrier-safe: {:?}", result.violations);
/// ```
#[must_use]
pub fn analyze(ptx: &str) -> BarrierSafetyResult {
    let lines: Vec<&str> = ptx.lines().collect();
    let has_barriers = ptx.contains("bar.sync") || ptx.contains("bar.arrive");

    // First pass: identify loop structure
    let (loop_labels, loop_end_labels) = identify_loop_labels(&lines, ptx);

    // Second pass: analyze for violations
    let mut state = AnalysisState::default();
    for (idx, line) in lines.iter().enumerate() {
        analyze_line(
            line.trim(),
            idx + 1,
            has_barriers,
            &loop_labels,
            &loop_end_labels,
            &mut state,
        );
    }

    BarrierSafetyResult {
        is_safe: state.violations.is_empty(),
        violations: state.violations,
        barrier_count: state.barrier_count,
        exit_count: state.exit_count,
    }
}

/// Mutable state accumulated during the analysis pass.
#[derive(Default)]
struct AnalysisState {
    violations: Vec<BarrierViolation>,
    barrier_count: usize,
    exit_count: usize,
    in_loop: bool,
    loop_start_line: usize,
    barrier_seen_in_current_loop: bool,
}

/// First pass: identify labels that form loops (have a branch back to them).
fn identify_loop_labels(lines: &[&str], ptx: &str) -> (HashSet<String>, HashSet<String>) {
    let mut loop_labels = HashSet::new();
    let mut loop_end_labels = HashSet::new();

    for line in lines {
        let trimmed = line.trim();
        if trimmed.ends_with(':') && !trimmed.starts_with('.') && !trimmed.contains("exit") {
            let label = trimmed.trim_end_matches(':').to_string();
            let has_back_branch =
                ptx.contains(&format!("bra {};", label)) || ptx.contains(&format!("bra {}", label));
            if has_back_branch {
                loop_end_labels.insert(format!("{}_end", label));
                loop_end_labels.insert(format!("{}_done", label));
                loop_labels.insert(label);
            }
        }
    }

    // Known end/done patterns from common kernel templates
    for known in
        ["k_tile_end", "kv_loop_end", "loop_end", "sb_loop_done", "sub_block_done", "k_block_done"]
    {
        loop_end_labels.insert(known.to_string());
    }

    (loop_labels, loop_end_labels)
}

/// Analyze a single PTX line for barrier safety violations.
fn analyze_line(
    trimmed: &str,
    line_num: usize,
    has_barriers: bool,
    loop_labels: &HashSet<String>,
    loop_end_labels: &HashSet<String>,
    state: &mut AnalysisState,
) {
    // Track barrier count
    if trimmed.contains("bar.sync") || trimmed.contains("bar.arrive") {
        state.barrier_count += 1;
        if state.in_loop {
            state.barrier_seen_in_current_loop = true;
        }
    }

    // Track loop entry/exit via labels
    if trimmed.ends_with(':') && !trimmed.starts_with('.') {
        let label = trimmed.trim_end_matches(':');
        if loop_labels.contains(label) {
            state.in_loop = true;
            state.loop_start_line = line_num;
            state.barrier_seen_in_current_loop = false;
        }
        if loop_end_labels.contains(label) {
            state.in_loop = false;
        }
    }

    // Detect exit instructions
    if trimmed.contains("bra exit") {
        state.exit_count += 1;
        check_exit_violation(trimmed, line_num, has_barriers, state);
    }

    // Count ret as exit but don't flag it (ret is at function end, not loop)
    if trimmed == "ret;" {
        state.exit_count += 1;
    }
}

/// Check if an exit instruction violates barrier safety (PARITY-114).
fn check_exit_violation(
    trimmed: &str,
    line_num: usize,
    has_barriers: bool,
    state: &mut AnalysisState,
) {
    if !has_barriers || !state.in_loop || state.barrier_seen_in_current_loop {
        return;
    }
    let kind = if trimmed.starts_with('@') {
        ViolationKind::ConditionalExitBeforeBarrier
    } else {
        ViolationKind::EarlyExitBeforeBarrier
    };
    state.violations.push(BarrierViolation {
        line: line_num,
        kind,
        instruction: trimmed.to_string(),
        context: format!("loop starting at line {}", state.loop_start_line),
    });
}

/// Validate PTX is barrier-safe, returning an error if not
///
/// # Arguments
///
/// * `ptx` - PTX source code
///
/// # Returns
///
/// Ok(()) if safe, Err with violation details if not
pub fn validate(ptx: &str) -> Result<(), String> {
    let result = analyze(ptx);
    if result.is_safe {
        Ok(())
    } else {
        let mut msg = String::from("Barrier safety violations found:\n");
        for v in &result.violations {
            msg.push_str(&format!(
                "  Line {}: {} - {}\n    Context: {}\n",
                v.line, v.kind, v.instruction, v.context
            ));
        }
        Err(msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// PARITY-114: Safe PTX with barrier inside loop
    #[test]
    fn test_barrier_safe_ptx() {
        let ptx = r#"
.entry kernel() {
    mov.u32 %r0, %tid.x;
    setp.lt.u32 %p0, %r0, 32;

loop_start:
    ld.shared.f32 %f0, [%r0];
    bar.sync 0;
    st.shared.f32 [%r0], %f0;
    bra loop_start;

loop_start_end:
    @!%p0 bra exit;
    st.global.f32 [%r1], %f0;
exit:
    ret;
}
"#;
        let result = analyze(ptx);
        assert!(result.is_safe, "Should be safe: {:?}", result.violations);
        assert_eq!(result.barrier_count, 1);
    }

    /// PARITY-114: Unsafe PTX with early exit before barrier
    #[test]
    fn test_barrier_unsafe_early_exit() {
        let ptx = r#"
.entry kernel() {
    mov.u32 %r0, %tid.x;
    setp.lt.u32 %p0, %r0, 32;

loop_start:
    @!%p0 bra exit;
    ld.shared.f32 %f0, [%r0];
    bar.sync 0;
    st.shared.f32 [%r0], %f0;
    bra loop_start;

loop_start_end:
done:
    ret;
}
"#;
        let result = analyze(ptx);
        assert!(!result.is_safe, "Should detect early exit");
        assert_eq!(result.violations.len(), 1);
        assert_eq!(result.violations[0].kind, ViolationKind::ConditionalExitBeforeBarrier);
    }

    /// Test unconditional early exit
    #[test]
    fn test_unconditional_early_exit() {
        let ptx = r#"
.entry kernel() {
loop_start:
    bra exit;
    bar.sync 0;
    bra loop_start;

loop_start_end:
done:
    ret;
}
"#;
        let result = analyze(ptx);
        assert!(!result.is_safe);
        assert_eq!(result.violations[0].kind, ViolationKind::EarlyExitBeforeBarrier);
    }

    /// Test validate function
    #[test]
    fn test_validate_returns_error() {
        let unsafe_ptx = r#"
.entry kernel() {
loop_start:
    bra exit;
    bar.sync 0;
    bra loop_start;

loop_start_end:
done:
    ret;
}
"#;
        let result = validate(unsafe_ptx);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("PARITY-114"));
    }

    /// Test no false positives for exit after loop
    #[test]
    fn test_exit_after_loop_ok() {
        let ptx = r#"
.entry kernel() {
k_tile_loop:
    bar.sync 0;
    ld.shared.f32 %f0, [%r0];
    bra k_tile_loop;

k_tile_end:
    @!%p0 bra exit;
    st.global.f32 [%r1], %f0;
done:
    ret;
}
"#;
        let result = analyze(ptx);
        assert!(result.is_safe, "Exit after loop should be OK: {:?}", result.violations);
    }

    /// Test kv_loop pattern (attention kernels)
    #[test]
    fn test_kv_loop_pattern() {
        let ptx = r#"
.entry attention() {
kv_loop:
    bar.sync 0;
    wmma.mma.sync.aligned.row.col.m16n16k16.f32.f16.f16.f32 ...;
    bra kv_loop;

kv_loop_end:
    @!%p_valid bra exit;
    st.global.f32 [%out], %f0;
done:
    ret;
}
"#;
        let result = analyze(ptx);
        assert!(result.is_safe, "KV loop pattern should be safe");
    }

    /// Warp-only kernels (no barriers) are always safe
    /// This covers LayerNorm warp_shuffle, RMSNorm, and other shuffle-based kernels
    #[test]
    fn test_warp_only_kernel_safe() {
        // Simulates a warp shuffle reduction kernel with conditional exit in loop
        // No bar.sync means no barrier divergence possible
        let ptx = r#"
.entry rmsnorm() {
    mov.u32 %r0, %tid.x;
    setp.lt.u32 %p0, %r0, 32;

sum_loop:
    @!%p1 bra exit;
    ld.global.f32 %f0, [%addr];
    shfl.sync.down.b32 %f1, %f0, 16, 0x1f, 0xffffffff;
    add.f32 %f0, %f0, %f1;
    add.u32 %idx, %idx, 32;
    setp.lt.u32 %p1, %idx, %n;
    bra sum_loop;

sum_loop_end:
    st.global.f32 [%out], %f0;
exit:
    ret;
}
"#;
        let result = analyze(ptx);
        assert!(result.is_safe, "Warp-only kernel should be safe: {:?}", result.violations);
        assert_eq!(result.barrier_count, 0, "No barriers in warp-only kernel");
    }

    /// Kernels with conditional exit and NO barriers are always safe
    #[test]
    fn test_no_barrier_conditional_exit_safe() {
        let ptx = r#"
.entry kernel() {
loop:
    @%p0 bra exit;
    ld.global.f32 %f0, [%r0];
    bra loop;

loop_end:
exit:
    ret;
}
"#;
        let result = analyze(ptx);
        assert!(result.is_safe, "No-barrier kernel with conditional exit should be safe");
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Any PTX with barrier after all exits in loop is safe
        #[test]
        fn barrier_after_exits_is_safe(loop_body_len in 1usize..10) {
            // Generate safe pattern: operations, then barrier, then exits
            let mut ptx = String::from(".entry test() {\nloop:\n");
            for i in 0..loop_body_len {
                ptx.push_str(&format!("    mov.u32 %r{}, 0;\n", i));
            }
            ptx.push_str("    bar.sync 0;\n");
            ptx.push_str("    bra loop;\nloop_end:\nexit:\n    ret;\n}\n");

            let result = analyze(&ptx);
            prop_assert!(result.is_safe, "Generated safe PTX should pass: {}", ptx);
        }

        /// PTX with no loops is always safe (no barrier divergence possible)
        #[test]
        fn no_loops_always_safe(num_exits in 0usize..5) {
            let mut ptx = String::from(".entry test() {\n");
            for _ in 0..num_exits {
                ptx.push_str("    @%p0 bra exit;\n");
            }
            ptx.push_str("exit:\n    ret;\n}\n");

            let result = analyze(&ptx);
            prop_assert!(result.is_safe, "No-loop PTX should be safe");
        }

        /// Barrier count matches actual bar.sync instructions
        #[test]
        fn barrier_count_accurate(num_barriers in 0usize..5) {
            let mut ptx = String::from(".entry test() {\n");
            for i in 0..num_barriers {
                ptx.push_str(&format!("    bar.sync {};\n", i % 16));
            }
            ptx.push_str("    ret;\n}\n");

            let result = analyze(&ptx);
            prop_assert_eq!(result.barrier_count, num_barriers);
        }
    }
}
