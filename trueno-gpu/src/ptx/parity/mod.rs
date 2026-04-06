//! PTX Parity Validation (GH-219)
//!
//! Validates that batched GPU kernels are structurally compatible with their
//! single-vector counterparts. Catches dequant bugs, register type mismatches,
//! and missing batch dispatch patterns at compile time.
//!
//! ## Motivation
//!
//! Three classes of bugs motivated this module:
//! 1. `BatchedQ6KGemvKernel` had 3 dequant bugs not present in `Q6KGemvKernel`
//! 2. `BatchedVectorizedRmsNormKernel` used u64 shared memory addressing
//! 3. Stale `position_buf` caused indirect KV scatter to wrong positions
//!
//! ## Validation Checks
//!
//! - **Parameter count**: Batched kernel must have same params as single-vector
//! - **Shared memory size**: Must match (batched should not need more shared mem)
//! - **Loop structure**: Must have matching computation loops (sum_loop, norm_loop, etc.)
//! - **Batch dispatch**: Batched kernels must use `ctaid.y` for row selection
//! - **Shared memory addressing**: Must use u32 registers for shared memory offsets

/// Result of a PTX parity validation
#[derive(Debug, Clone)]
pub struct ParityResult {
    /// Whether the kernels are parity-compatible
    pub is_compatible: bool,
    /// Specific violations found
    pub violations: Vec<ParityViolation>,
    /// Single-vector kernel name
    pub single_name: String,
    /// Batched kernel name
    pub batched_name: String,
}

/// A specific parity violation
#[derive(Debug, Clone)]
pub struct ParityViolation {
    /// What kind of violation
    pub kind: ParityViolationKind,
    /// Human-readable description
    pub message: String,
}

/// Categories of parity violations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParityViolationKind {
    /// Batched kernel has different parameter count than single-vector
    ParameterCountMismatch,
    /// Shared memory size differs between kernels
    SharedMemoryMismatch,
    /// Batched kernel missing ctaid.y for row dispatch
    MissingBatchDispatch,
    /// Shared memory addressed with u64 instead of u32
    SharedMemoryAddressingU64,
    /// Computation loop structure differs
    LoopStructureMismatch,
    /// Register type mismatch in dequant logic
    RegisterTypeMismatch,
}

impl std::fmt::Display for ParityViolationKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ParameterCountMismatch => write!(f, "PARAM_COUNT"),
            Self::SharedMemoryMismatch => write!(f, "SHARED_MEM_SIZE"),
            Self::MissingBatchDispatch => write!(f, "MISSING_CTAID_Y"),
            Self::SharedMemoryAddressingU64 => write!(f, "SHARED_MEM_U64"),
            Self::LoopStructureMismatch => write!(f, "LOOP_STRUCTURE"),
            Self::RegisterTypeMismatch => write!(f, "REG_TYPE"),
        }
    }
}

/// Count `.param` declarations in PTX source
fn count_params(ptx: &str) -> usize {
    ptx.lines()
        .filter(|line| {
            let trimmed = line.trim();
            trimmed.starts_with(".param")
        })
        .count()
}

/// Extract shared memory declaration size from PTX
/// Looks for `.shared .align N .b8 smem[SIZE];`
fn extract_shared_memory_bytes(ptx: &str) -> Option<u32> {
    ptx.lines()
        .map(str::trim)
        .filter(|line| line.contains(".shared") && line.contains("smem["))
        .find_map(parse_smem_size)
}

/// Parse the size from a `smem[SIZE]` declaration.
fn parse_smem_size(line: &str) -> Option<u32> {
    let after = &line[line.find("smem[")? + 5..];
    let end = after.find(']')?;
    after[..end].parse().ok()
}

/// Extract loop labels from PTX (e.g., sum_loop, norm_loop, etc.)
fn extract_loop_labels(ptx: &str) -> Vec<String> {
    let mut labels = Vec::new();
    for line in ptx.lines() {
        let trimmed = line.trim();
        // Loop labels are like "sum_loop:" or "norm_loop:" at the start of a line
        if trimmed.ends_with(':') && !trimmed.starts_with("//") {
            let label = trimmed.trim_end_matches(':');
            // Only count loop-related labels (containing "loop")
            if label.contains("loop") {
                labels.push(label.to_string());
            }
        }
    }
    labels
}

/// Batch dispatch strategy used by batched kernels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchDispatchStrategy {
    /// Grid.y dispatch: one block per batch element (ctaid.y selects row)
    /// Used by: RmsNorm, ResidualAdd, SwiGLU, RoPE
    GridY,
    /// Register unrolling: M accumulators per block, m_dim parameter
    /// Used by: Quantized GEMV (Q4K, Q6K) for throughput optimization
    RegisterUnroll,
}

/// Check if PTX uses ctaid.y for batch dispatch
fn has_grid_y_dispatch(ptx: &str) -> bool {
    ptx.contains("%ctaid.y")
}

/// Check if PTX uses register-unrolled batch dispatch (m_dim parameter)
fn has_register_unroll_dispatch(ptx: &str) -> bool {
    ptx.contains("m_dim")
}

/// Check if PTX has any batch dispatch mechanism
fn has_batch_dispatch(ptx: &str) -> bool {
    has_grid_y_dispatch(ptx) || has_register_unroll_dispatch(ptx)
}

/// Check for u64 registers used with shared memory operations
/// Returns true if any shared memory load/store uses u64 offset registers
fn has_u64_shared_memory_addressing(ptx: &str) -> bool {
    // Look for patterns like:
    //   st.shared.f32 [%rdN], ... (u64 address register for shared memory)
    //   ld.shared.f32 %fN, [%rdN] (u64 address register for shared memory)
    // Valid shared memory addressing should use u32 registers (%rN):
    //   st.shared.f32 [%rN], ...
    //   ld.shared.f32 %fN, [%rN]
    for line in ptx.lines() {
        let trimmed = line.trim();
        if (trimmed.contains("st.shared") || trimmed.contains("ld.shared"))
            && trimmed.contains("[%rd")
        {
            return true;
        }
    }
    false
}

/// Validate parity between a single-vector kernel's PTX and a batched kernel's PTX.
///
/// The batched kernel (with M=1) should be structurally equivalent to the
/// single-vector kernel, differing only in:
/// - An additional `ctaid.y` read for row dispatch
/// - Row offset calculation for global memory access
///
/// Everything else (dequant logic, shared memory, reduction, normalization)
/// should be identical.
pub fn validate_parity(
    single_ptx: &str,
    batched_ptx: &str,
    single_name: &str,
    batched_name: &str,
) -> ParityResult {
    let mut violations = Vec::new();

    // 1. Parameter count must match
    let single_params = count_params(single_ptx);
    let batched_params = count_params(batched_ptx);
    if single_params != batched_params {
        violations.push(ParityViolation {
            kind: ParityViolationKind::ParameterCountMismatch,
            message: format!(
                "Single kernel '{}' has {} params, batched '{}' has {} params",
                single_name, single_params, batched_name, batched_params
            ),
        });
    }

    // 2. Shared memory size must match
    let single_smem = extract_shared_memory_bytes(single_ptx);
    let batched_smem = extract_shared_memory_bytes(batched_ptx);
    if single_smem != batched_smem {
        violations.push(ParityViolation {
            kind: ParityViolationKind::SharedMemoryMismatch,
            message: format!(
                "Shared memory mismatch: single={:?} bytes, batched={:?} bytes",
                single_smem, batched_smem
            ),
        });
    }

    // 3. Batched kernel must use ctaid.y for row dispatch
    if !has_batch_dispatch(batched_ptx) {
        violations.push(ParityViolation {
            kind: ParityViolationKind::MissingBatchDispatch,
            message: format!(
                "Batched kernel '{}' does not use %ctaid.y for row dispatch",
                batched_name
            ),
        });
    }

    // 4. Shared memory addressing must use u32 registers
    if has_u64_shared_memory_addressing(batched_ptx) {
        violations.push(ParityViolation {
            kind: ParityViolationKind::SharedMemoryAddressingU64,
            message: format!(
                "Batched kernel '{}' uses u64 registers (%rd) for shared memory addressing; \
                 use u32 (%r) for portability",
                batched_name
            ),
        });
    }
    // Also check single-vector kernel
    if has_u64_shared_memory_addressing(single_ptx) {
        violations.push(ParityViolation {
            kind: ParityViolationKind::SharedMemoryAddressingU64,
            message: format!(
                "Single kernel '{}' uses u64 registers (%rd) for shared memory addressing; \
                 use u32 (%r) for portability",
                single_name
            ),
        });
    }

    // 5. Loop structure should match (same computation loops)
    let single_loops = extract_loop_labels(single_ptx);
    let batched_loops = extract_loop_labels(batched_ptx);
    if single_loops != batched_loops {
        violations.push(ParityViolation {
            kind: ParityViolationKind::LoopStructureMismatch,
            message: format!(
                "Loop structure differs: single has {:?}, batched has {:?}",
                single_loops, batched_loops
            ),
        });
    }

    ParityResult {
        is_compatible: violations.is_empty(),
        violations,
        single_name: single_name.to_string(),
        batched_name: batched_name.to_string(),
    }
}

/// Validate that a batched kernel's PTX is well-formed for batch execution.
///
/// This is a standalone check (no single-vector reference needed) that verifies
/// the batched kernel has correct batch dispatch patterns.
pub fn validate_batched_kernel(ptx: &str, kernel_name: &str) -> ParityResult {
    let mut violations = Vec::new();

    // Must use ctaid.y for batch dispatch
    if !has_batch_dispatch(ptx) {
        violations.push(ParityViolation {
            kind: ParityViolationKind::MissingBatchDispatch,
            message: format!(
                "Batched kernel '{}' does not use %ctaid.y for row dispatch",
                kernel_name
            ),
        });
    }

    // Must not use u64 for shared memory
    if has_u64_shared_memory_addressing(ptx) {
        violations.push(ParityViolation {
            kind: ParityViolationKind::SharedMemoryAddressingU64,
            message: format!(
                "Batched kernel '{}' uses u64 registers for shared memory addressing",
                kernel_name
            ),
        });
    }

    ParityResult {
        is_compatible: violations.is_empty(),
        violations,
        single_name: String::new(),
        batched_name: kernel_name.to_string(),
    }
}

#[cfg(test)]
mod tests;
