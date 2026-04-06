//! KernelParity trait implementations for all batched/single-vector kernel pairs
//!
//! GH-219: Each batched kernel is paired with its single-vector reference.
//! This enables compile-time structural validation that batched kernels
//! maintain correctness invariants.

use super::{
    BatchedQ4KGemvKernel, BatchedQ6KGemvKernel, BatchedResidualAddKernel, BatchedRopeKernel,
    BatchedSwigluKernel, BatchedVectorizedRmsNormKernel, FusedSwigluKernel, KernelParity,
    Q4KGemvKernel, Q6KGemvKernel, ResidualAddKernel, RopeKernel, VectorizedRmsNormKernel,
};

// ============================================================================
// 1. BatchedVectorizedRmsNormKernel ↔ VectorizedRmsNormKernel
// ============================================================================

impl KernelParity for BatchedVectorizedRmsNormKernel {
    type SingleVector = VectorizedRmsNormKernel;

    fn single_vector_reference(&self) -> Self::SingleVector {
        VectorizedRmsNormKernel::new(self.hidden_size).with_epsilon(self.epsilon)
    }
}

// ============================================================================
// 2. BatchedQ4KGemvKernel ↔ Q4KGemvKernel
// ============================================================================

impl KernelParity for BatchedQ4KGemvKernel {
    type SingleVector = Q4KGemvKernel;

    fn single_vector_reference(&self) -> Self::SingleVector {
        Q4KGemvKernel::new(self.k, self.n)
    }
}

// ============================================================================
// 3. BatchedQ6KGemvKernel ↔ Q6KGemvKernel
// ============================================================================

impl KernelParity for BatchedQ6KGemvKernel {
    type SingleVector = Q6KGemvKernel;

    fn single_vector_reference(&self) -> Self::SingleVector {
        Q6KGemvKernel::new(self.k, self.n)
    }
}

// ============================================================================
// 4. BatchedResidualAddKernel ↔ ResidualAddKernel
// ============================================================================

impl KernelParity for BatchedResidualAddKernel {
    type SingleVector = ResidualAddKernel;

    fn single_vector_reference(&self) -> Self::SingleVector {
        ResidualAddKernel::new(self.n)
    }
}

// ============================================================================
// 5. BatchedRopeKernel ↔ RopeKernel
// ============================================================================

impl KernelParity for BatchedRopeKernel {
    type SingleVector = RopeKernel;

    fn single_vector_reference(&self) -> Self::SingleVector {
        RopeKernel::new(self.num_heads, self.head_dim, self.theta)
    }
}

// ============================================================================
// 6. BatchedSwigluKernel ↔ FusedSwigluKernel
// ============================================================================

impl KernelParity for BatchedSwigluKernel {
    type SingleVector = FusedSwigluKernel;

    fn single_vector_reference(&self) -> Self::SingleVector {
        FusedSwigluKernel::new(self.n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::Kernel;

    // Typical transformer model dimensions for testing
    const HIDDEN_1536: u32 = 1536; // Qwen2.5-1.5B
    const HIDDEN_3584: u32 = 3584; // Qwen2.5-7B
    const INTERMEDIATE_4864: u32 = 4864; // Qwen2.5-1.5B FFN intermediate
    const NUM_HEADS_12: u32 = 12; // Qwen2.5-1.5B
    const HEAD_DIM_128: u32 = 128; // Qwen2.5-1.5B/7B
    const ROPE_THETA: f32 = 1_000_000.0; // Qwen2.5

    // ====================================================================
    // GH-219: PTX Parity Tests — Batched vs Single-Vector
    // ====================================================================

    /// GH-219: BatchedVectorizedRmsNormKernel must use ctaid.y for row dispatch
    #[test]
    fn test_parity_rmsnorm_batch_dispatch() {
        let kernel = BatchedVectorizedRmsNormKernel::new(HIDDEN_1536, 1);
        let result = kernel.validate_batch_dispatch();
        assert!(
            result.is_compatible,
            "BatchedVectorizedRmsNormKernel missing ctaid.y: {:?}",
            result.violations
        );
    }

    /// GH-219: BatchedVectorizedRmsNormKernel must not use u64 for shared memory
    #[test]
    fn test_parity_rmsnorm_no_u64_shared_mem() {
        let kernel = BatchedVectorizedRmsNormKernel::new(HIDDEN_1536, 1);
        let ptx = kernel.emit_ptx();
        // Check that shared memory stores use u32 registers
        for line in ptx.lines() {
            let trimmed = line.trim();
            if (trimmed.contains("st.shared") || trimmed.contains("ld.shared"))
                && trimmed.contains("[%rd")
            {
                panic!(
                    "GH-219: BatchedVectorizedRmsNormKernel uses u64 (%rd) for shared memory: {}",
                    trimmed
                );
            }
        }
    }

    /// GH-219: BatchedVectorizedRmsNormKernel shared mem matches single-vector
    #[test]
    fn test_parity_rmsnorm_shared_memory_size() {
        let batched = BatchedVectorizedRmsNormKernel::new(HIDDEN_1536, 1);
        let single = batched.single_vector_reference();
        let batched_ptx = batched.emit_ptx();
        let single_ptx = single.emit_ptx();

        // Both should declare the same shared memory size
        let batched_smem = extract_smem(&batched_ptx);
        let single_smem = extract_smem(&single_ptx);
        assert_eq!(
            batched_smem, single_smem,
            "Shared memory size mismatch: single={:?}, batched={:?}",
            single_smem, batched_smem
        );
    }

    /// GH-219: BatchedVectorizedRmsNormKernel full parity check (7B dimensions)
    #[test]
    fn test_parity_rmsnorm_7b() {
        let kernel = BatchedVectorizedRmsNormKernel::new(HIDDEN_3584, 1);
        let result = kernel.validate_batch_dispatch();
        assert!(
            result.is_compatible,
            "7B BatchedRmsNorm parity: {:?}",
            result.violations
        );
    }

    /// GH-219: BatchedQ4KGemvKernel uses register-unrolled dispatch (m_dim param)
    #[test]
    fn test_parity_q4k_gemv_batch_dispatch() {
        let kernel = BatchedQ4KGemvKernel::new(HIDDEN_1536, HIDDEN_1536, 1);
        let result = kernel.validate_batch_dispatch();
        assert!(
            result.is_compatible,
            "BatchedQ4KGemvKernel missing batch dispatch: {:?}",
            result.violations
        );
        // Verify it uses register-unroll strategy (m_dim param), not grid.y
        let ptx = kernel.emit_ptx();
        assert!(
            ptx.contains("m_dim"),
            "Q4K batched should use register-unrolled dispatch via m_dim parameter"
        );
    }

    /// GH-219: BatchedQ6KGemvKernel uses register-unrolled dispatch (m_dim param)
    #[test]
    fn test_parity_q6k_gemv_batch_dispatch() {
        let kernel = BatchedQ6KGemvKernel::new(HIDDEN_1536, HIDDEN_1536, 1);
        let result = kernel.validate_batch_dispatch();
        assert!(
            result.is_compatible,
            "BatchedQ6KGemvKernel missing batch dispatch: {:?}",
            result.violations
        );
        // Verify it uses register-unroll strategy (m_dim param), not grid.y
        let ptx = kernel.emit_ptx();
        assert!(
            ptx.contains("m_dim"),
            "Q6K batched should use register-unrolled dispatch via m_dim parameter"
        );
    }

    /// GH-219: BatchedResidualAddKernel must use ctaid.y for row dispatch
    #[test]
    fn test_parity_residual_add_batch_dispatch() {
        let kernel = BatchedResidualAddKernel::new(HIDDEN_1536, 1);
        let result = kernel.validate_batch_dispatch();
        assert!(
            result.is_compatible,
            "BatchedResidualAddKernel missing ctaid.y: {:?}",
            result.violations
        );
    }

    /// GH-219: BatchedRopeKernel must use ctaid.y for row dispatch
    #[test]
    fn test_parity_rope_batch_dispatch() {
        let kernel = BatchedRopeKernel::new(NUM_HEADS_12, HEAD_DIM_128, 1, ROPE_THETA);
        let result = kernel.validate_batch_dispatch();
        assert!(
            result.is_compatible,
            "BatchedRopeKernel missing ctaid.y: {:?}",
            result.violations
        );
    }

    /// GH-219: BatchedSwigluKernel must use ctaid.y for row dispatch
    #[test]
    fn test_parity_swiglu_batch_dispatch() {
        let kernel = BatchedSwigluKernel::new(INTERMEDIATE_4864, 1);
        let result = kernel.validate_batch_dispatch();
        assert!(
            result.is_compatible,
            "BatchedSwigluKernel missing ctaid.y: {:?}",
            result.violations
        );
    }

    /// GH-219: All 6 batched kernels have a valid batch dispatch mechanism
    #[test]
    fn test_all_batched_kernels_have_batch_dispatch() {
        let kernels: Vec<(&str, Box<dyn Kernel>, &str)> = vec![
            (
                "BatchedVectorizedRmsNormKernel",
                Box::new(BatchedVectorizedRmsNormKernel::new(HIDDEN_1536, 1)),
                "grid_y",
            ),
            (
                "BatchedQ4KGemvKernel",
                Box::new(BatchedQ4KGemvKernel::new(HIDDEN_1536, HIDDEN_1536, 1)),
                "register_unroll",
            ),
            (
                "BatchedQ6KGemvKernel",
                Box::new(BatchedQ6KGemvKernel::new(HIDDEN_1536, HIDDEN_1536, 1)),
                "register_unroll",
            ),
            (
                "BatchedResidualAddKernel",
                Box::new(BatchedResidualAddKernel::new(HIDDEN_1536, 1)),
                "grid_y",
            ),
            (
                "BatchedRopeKernel",
                Box::new(BatchedRopeKernel::new(
                    NUM_HEADS_12,
                    HEAD_DIM_128,
                    1,
                    ROPE_THETA,
                )),
                "grid_y",
            ),
            (
                "BatchedSwigluKernel",
                Box::new(BatchedSwigluKernel::new(INTERMEDIATE_4864, 1)),
                "grid_y",
            ),
        ];

        let mut failures = Vec::new();
        for (name, kernel, expected_strategy) in &kernels {
            let ptx = kernel.emit_ptx();
            let has_grid_y = ptx.contains("%ctaid.y");
            let has_m_dim = ptx.contains("m_dim");
            let has_any = has_grid_y || has_m_dim;

            if !has_any {
                failures.push(format!(
                    "{} missing batch dispatch (no ctaid.y or m_dim)",
                    name
                ));
            }

            // Verify correct strategy
            match *expected_strategy {
                "grid_y" => {
                    if !has_grid_y {
                        failures.push(format!(
                            "{} expected grid_y dispatch but missing ctaid.y",
                            name
                        ));
                    }
                }
                "register_unroll" => {
                    if !has_m_dim {
                        failures.push(format!(
                            "{} expected register_unroll but missing m_dim",
                            name
                        ));
                    }
                }
                _ => {}
            }
        }

        assert!(
            failures.is_empty(),
            "GH-219: Batched kernel dispatch validation failures:\n{}",
            failures.join("\n")
        );
    }

    /// GH-219: No batched kernel uses u64 registers for shared memory access
    #[test]
    fn test_no_u64_shared_memory_in_batched_kernels() {
        let kernels: Vec<(&str, Box<dyn Kernel>)> = vec![
            (
                "BatchedVectorizedRmsNormKernel",
                Box::new(BatchedVectorizedRmsNormKernel::new(HIDDEN_1536, 1)),
            ),
            (
                "BatchedQ4KGemvKernel",
                Box::new(BatchedQ4KGemvKernel::new(HIDDEN_1536, HIDDEN_1536, 1)),
            ),
            (
                "BatchedQ6KGemvKernel",
                Box::new(BatchedQ6KGemvKernel::new(HIDDEN_1536, HIDDEN_1536, 1)),
            ),
        ];

        let mut failures = Vec::new();
        for (name, kernel) in &kernels {
            let ptx = kernel.emit_ptx();
            for line in ptx.lines() {
                let trimmed = line.trim();
                if (trimmed.contains("st.shared") || trimmed.contains("ld.shared"))
                    && trimmed.contains("[%rd")
                {
                    failures.push(format!(
                        "{}: u64 shared mem address: {}",
                        name,
                        trimmed.trim()
                    ));
                }
            }
        }

        assert!(
            failures.is_empty(),
            "GH-219: Batched kernels with u64 shared memory addressing:\n{}",
            failures.join("\n")
        );
    }

    /// Helper: extract shared memory size from PTX
    fn extract_smem(ptx: &str) -> Option<u32> {
        for line in ptx.lines() {
            let trimmed = line.trim();
            if trimmed.contains(".shared") && trimmed.contains("smem[") {
                if let Some(start) = trimmed.find("smem[") {
                    let after = &trimmed[start + 5..];
                    if let Some(end) = after.find(']') {
                        if let Ok(size) = after[..end].parse::<u32>() {
                            return Some(size);
                        }
                    }
                }
            }
        }
        None
    }
}
