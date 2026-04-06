/// Property tests for kernel builders (TRUENO-SPEC-014 TASK-011)
use super::*;
use proptest::prelude::*;

proptest! {
    /// All GEMM naive kernels produce valid PTX regardless of dimensions
    #[test]
    fn gemm_naive_always_valid(m in 16u32..512, n in 16u32..512, k in 16u32..512) {
        let kernel = GemmKernel::naive(m, n, k);
        let ptx = kernel.emit_ptx();

        // Must have PTX header
        prop_assert!(ptx.contains(".version"), "Missing PTX version");
        prop_assert!(ptx.contains(".target"), "Missing target");
        prop_assert!(ptx.contains(".entry"), "Missing entry point");

        // Must have kernel parameters
        prop_assert!(ptx.contains(".param"), "Missing parameters");
        prop_assert!(ptx.contains("a_ptr"), "Missing A matrix pointer");
        prop_assert!(ptx.contains("b_ptr"), "Missing B matrix pointer");
        prop_assert!(ptx.contains("c_ptr"), "Missing C matrix pointer");
    }

    /// All GEMM tiled kernels produce valid PTX with shared memory
    #[test]
    fn gemm_tiled_uses_shared_memory(m in 32u32..256, n in 32u32..256, k in 32u32..256, tile in 8u32..32) {
        let kernel = GemmKernel::tiled(m, n, k, tile);
        let ptx_kernel = kernel.build_ptx();

        // Tiled GEMM must use shared memory
        prop_assert!(ptx_kernel.shared_memory_bytes() > 0, "Tiled GEMM should use shared memory");
    }

    /// All Softmax kernels produce valid PTX
    #[test]
    fn softmax_always_valid(seq_len in 64u32..8192) {
        let kernel = SoftmaxKernel::new(seq_len);
        let ptx = kernel.emit_ptx();

        // Must have PTX header
        prop_assert!(ptx.contains(".version"), "Missing PTX version");
        prop_assert!(ptx.contains(".entry"), "Missing entry point");
        prop_assert!(ptx.contains("softmax"), "Missing softmax kernel name");
    }

    /// All LayerNorm kernels produce valid PTX
    #[test]
    fn layernorm_always_valid(hidden_size in 64u32..4096) {
        let kernel = LayerNormKernel::new(hidden_size);
        let ptx = kernel.emit_ptx();

        // Must have PTX header
        prop_assert!(ptx.contains(".version"), "Missing PTX version");
        prop_assert!(ptx.contains(".entry"), "Missing entry point");
    }

    /// All Attention kernels produce valid PTX
    #[test]
    fn attention_always_valid(
        seq_len in 64u32..2048,
        head_dim in 32u32..128,
    ) {
        let kernel = AttentionKernel::new(seq_len, head_dim);
        let ptx = kernel.emit_ptx();

        // Must have PTX header
        prop_assert!(ptx.contains(".version"), "Missing PTX version");
        prop_assert!(ptx.contains(".entry"), "Missing entry point");
    }

    /// Kernel names are deterministic
    #[test]
    fn kernel_names_deterministic(m in 16u32..512, n in 16u32..512, k in 16u32..512) {
        let kernel1 = GemmKernel::naive(m, n, k);
        let kernel2 = GemmKernel::naive(m, n, k);

        prop_assert_eq!(kernel1.name(), kernel2.name(), "Kernel names should be deterministic");
    }

    /// PTX emission produces consistent structure
    #[test]
    fn ptx_emission_consistent_structure(m in 16u32..256, n in 16u32..256, k in 16u32..256) {
        let kernel = GemmKernel::naive(m, n, k);
        let ptx = kernel.emit_ptx();

        // Verify consistent structure regardless of dimensions
        prop_assert!(ptx.contains(".version 8.0"), "Must have version 8.0");
        prop_assert!(ptx.contains(".target sm_70"), "Must target sm_70 baseline");
        prop_assert!(ptx.contains(".address_size 64"), "Must use 64-bit addresses");
        prop_assert!(ptx.contains("ret;"), "Must have return statement");
    }
}

/// Edge case tests (not random)
#[test]
fn test_minimum_dimensions() {
    // Test smallest valid dimensions
    let kernel = GemmKernel::naive(1, 1, 1);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".entry"), "Should handle 1x1x1");
}

#[test]
fn test_large_dimensions() {
    // Test large but reasonable dimensions
    let kernel = GemmKernel::naive(4096, 4096, 4096);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".entry"), "Should handle 4096x4096");
}

#[test]
fn test_non_power_of_two() {
    // Test non-power-of-two dimensions
    let kernel = GemmKernel::naive(127, 255, 63);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".entry"), "Should handle non-power-of-two");
}

// =========================================================================
// Batched GEMM Property Tests (Issue #71)
// =========================================================================

proptest! {
    /// All batched GEMM naive kernels produce valid PTX regardless of dimensions
    #[test]
    fn batched_gemm_naive_always_valid(
        batch in 1u32..16,
        m in 16u32..256,
        n in 16u32..256,
        k in 16u32..256
    ) {
        let kernel = BatchedGemmKernel::naive(batch, m, n, k);
        let ptx = kernel.emit_ptx();

        prop_assert!(ptx.contains(".version"), "Missing PTX version");
        prop_assert!(ptx.contains(".entry"), "Missing entry point");
        prop_assert!(ptx.contains(".param .u32 batch"), "Missing batch parameter");
        prop_assert!(ptx.contains("%ctaid.z"), "Missing batch indexing via ctaid.z");
    }

    /// All batched GEMM tiled kernels produce valid PTX with shared memory
    #[test]
    fn batched_gemm_tiled_always_valid(
        batch in 1u32..8,
        m in 32u32..128,
        n in 32u32..128,
        k in 32u32..128,
        tile in 8u32..17
    ) {
        let kernel = BatchedGemmKernel::tiled(batch, m, n, k, tile);
        let ptx = kernel.emit_ptx();
        let ptx_kernel = kernel.build_ptx();

        prop_assert!(ptx.contains(".entry"), "Missing entry point");
        prop_assert!(ptx.contains("bar.sync"), "Missing barrier synchronization");
        prop_assert!(ptx_kernel.shared_memory_bytes() > 0, "Should use shared memory");
    }

    /// All 4D batched GEMM kernels produce valid PTX
    #[test]
    fn batched_4d_gemm_always_valid(
        batch in 1u32..8,
        heads in 1u32..16,
        m in 32u32..128,
        n in 32u32..128,
        k in 16u32..64
    ) {
        let kernel = Batched4DGemmKernel::new(batch, heads, m, n, k);
        let ptx = kernel.emit_ptx();

        prop_assert!(ptx.contains(".version"), "Missing PTX version");
        prop_assert!(ptx.contains(".entry"), "Missing entry point");
        prop_assert!(ptx.contains(".param .u32 batch"), "Missing batch parameter");
        prop_assert!(ptx.contains(".param .u32 heads"), "Missing heads parameter");
        prop_assert!(ptx.contains("%ctaid.z"), "Missing batch*heads indexing");
    }
}

#[test]
fn test_batched_gemm_minimum_batch() {
    let kernel = BatchedGemmKernel::naive(1, 32, 32, 32);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".entry"), "Should handle batch=1");
}

#[test]
fn test_batched_4d_gemm_attention_pattern() {
    // Typical transformer attention: batch=2, heads=8, seq_len=512, head_dim=64
    let kernel = Batched4DGemmKernel::new(2, 8, 512, 512, 64);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".entry"), "Should handle attention pattern");
    assert!(
        ptx.contains("bar.sync"),
        "Should have barriers for tiled compute"
    );
}
