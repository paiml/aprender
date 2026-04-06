//! Tests for layernorm kernels

use super::*;

#[test]
fn test_precise_rmsnorm_kernel_name() {
    let kernel = PreciseRmsNormKernel::new(1536);
    assert_eq!(kernel.name(), "rmsnorm_precise");
}

#[test]
fn test_precise_rmsnorm_ptx_generation() {
    let kernel = PreciseRmsNormKernel::new(1536).with_epsilon(1e-6);
    let ptx = kernel.emit_ptx();

    // Verify parameters
    assert!(ptx.contains(".param .u64 input_ptr"));
    assert!(ptx.contains(".param .u64 output_ptr"));
    assert!(ptx.contains(".param .u64 gamma_ptr"));

    // Verify kernel name
    assert!(ptx.contains("rmsnorm_precise"), "Missing kernel name");

    // Verify warp shuffle for reduction
    assert!(ptx.contains("shfl"), "Missing warp shuffle for reduction");

    // Verify rsqrt for initial approximation
    assert!(ptx.contains("rsqrt"), "Missing rsqrt instruction");

    // Verify multiplication for Newton-Raphson refinement
    assert!(ptx.contains("mul.f32"), "Missing mul.f32 for refinement");
}

#[test]
fn test_layernorm_kernel_name() {
    let kernel = LayerNormKernel::new(768);
    assert_eq!(kernel.name(), "layernorm_warp_shuffle");

    let kernel_shared = LayerNormKernel::new(768).without_warp_shuffle();
    assert_eq!(kernel_shared.name(), "layernorm_shared");
}

#[test]
fn test_layernorm_with_epsilon() {
    let kernel = LayerNormKernel::new(768).with_epsilon(1e-6);
    assert!((kernel.epsilon - 1e-6).abs() < 1e-10);
}

#[test]
fn test_layernorm_without_affine() {
    let kernel = LayerNormKernel::new(768).without_affine();
    assert!(!kernel.affine);
}

#[test]
fn test_layernorm_ptx_generation() {
    let kernel = LayerNormKernel::new(768);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".param .u64 input_ptr"));
    assert!(ptx.contains(".param .u64 output_ptr"));
    assert!(ptx.contains(".param .u64 gamma_ptr"));
    assert!(ptx.contains(".param .u64 beta_ptr"));
    assert!(ptx.contains(".param .u32 hidden_size"));
    assert!(ptx.contains(".param .u32 batch_size"));
}

#[test]
fn test_layernorm_warp_shuffle_ptx() {
    let kernel = LayerNormKernel::new(32);
    let ptx = kernel.emit_ptx();

    // Verify warp shuffle operations for reduction
    assert!(ptx.contains("shfl") || ptx.contains("shfl.down"));

    // Verify division for mean/variance
    assert!(ptx.contains("div.rn.f32")); // div requires rounding mode for floats

    // Verify rsqrt for normalization
    assert!(ptx.contains("rsqrt.f32") || ptx.contains("rsqrt"));

    // Verify memory operations
    assert!(ptx.contains("ld.global.f32"));
    assert!(ptx.contains("st.global.f32"));
}

#[test]
fn test_layernorm_shared_memory_ptx() {
    let kernel = LayerNormKernel::new(256).without_warp_shuffle();
    let ptx = kernel.emit_ptx();

    // Verify shared memory usage
    assert!(ptx.contains("ld.shared.f32") || ptx.contains("ld.f32"));
    assert!(ptx.contains("st.shared.f32") || ptx.contains("st.f32"));

    // Verify barrier synchronization
    assert!(ptx.contains("bar"));

    // Verify rsqrt and division
    assert!(ptx.contains("rsqrt.f32") || ptx.contains("rsqrt"));
    assert!(ptx.contains("div.rn.f32")); // div requires rounding mode for floats
}

#[test]
fn test_layernorm_kernel_variants() {
    let warp_kernel = LayerNormKernel::new(32);
    let shared_kernel = LayerNormKernel::new(256).without_warp_shuffle();

    // Both should produce valid PTX
    let warp_ptx = warp_kernel.emit_ptx();
    let shared_ptx = shared_kernel.emit_ptx();

    assert!(!warp_ptx.is_empty());
    assert!(!shared_ptx.is_empty());

    // Verify different kernel names in output
    assert!(warp_ptx.contains("layernorm_warp_shuffle"));
    assert!(shared_ptx.contains("layernorm_shared"));
}

#[test]
fn test_layernorm_numerical_operations() {
    let kernel = LayerNormKernel::new(32);
    let ptx = kernel.emit_ptx();

    // Should have subtraction (for x - mean)
    assert!(ptx.contains("sub.f32"));

    // Should have multiplication (for scaling, (x-mean)^2)
    assert!(ptx.contains("mul.f32"));

    // Should have addition (for variance + epsilon, gamma*x + beta)
    assert!(ptx.contains("add.f32"));
}

#[test]
fn test_layernorm_without_affine_ptx() {
    let kernel_affine = LayerNormKernel::new(32);
    let kernel_no_affine = LayerNormKernel::new(32).without_affine();

    let ptx_affine = kernel_affine.emit_ptx();
    let ptx_no_affine = kernel_no_affine.emit_ptx();

    // Both should be valid
    assert!(!ptx_affine.is_empty());
    assert!(!ptx_no_affine.is_empty());

    // Affine version should load gamma/beta pointers
    assert!(ptx_affine.contains("gamma_ptr"));
    assert!(ptx_affine.contains("beta_ptr"));
}

#[test]
fn test_layernorm_default_config() {
    let kernel = LayerNormKernel::new(768);

    assert_eq!(kernel.hidden_size, 768);
    assert!((kernel.epsilon - 1e-5).abs() < 1e-10);
    assert!(kernel.affine);
    assert!(kernel.use_warp_shuffle);
}

#[test]
fn test_rmsnorm_kernel_name() {
    let kernel = RmsNormKernel::new(2048);
    assert_eq!(kernel.name(), "rmsnorm");
}

#[test]
fn test_rmsnorm_ptx_generation() {
    let kernel = RmsNormKernel::new(2048);
    let ptx = kernel.emit_ptx();

    // Verify parameters
    assert!(ptx.contains(".param .u64 input_ptr"));
    assert!(ptx.contains(".param .u64 output_ptr"));
    assert!(ptx.contains(".param .u64 gamma_ptr"));

    // Verify warp shuffle for reduction
    assert!(ptx.contains("shfl"));

    // Verify rsqrt for 1/sqrt(rms)
    assert!(ptx.contains("rsqrt.f32") || ptx.contains("rsqrt"));
}

#[test]
fn test_rmsnorm_with_epsilon() {
    let kernel = RmsNormKernel::new(2048).with_epsilon(1e-6);
    assert!((kernel.epsilon - 1e-6).abs() < 1e-10);
}

#[test]
fn test_rmsnorm_ptx_valid_syntax() {
    let kernel = RmsNormKernel::new(2048).with_epsilon(1e-5);
    let ptx = kernel.emit_ptx();

    // Print first 200 lines for debugging
    for (i, line) in ptx.lines().enumerate().take(200) {
        eprintln!("{:4}: {}", i + 1, line);
    }

    // Basic structure checks
    assert!(ptx.contains(".entry rmsnorm"));
    assert!(ptx.contains("ret;"));
}

// ===== VectorizedRmsNormKernel Tests =====

#[test]
fn test_vectorized_rmsnorm_kernel_new() {
    let kernel = VectorizedRmsNormKernel::new(2048);
    assert_eq!(kernel.hidden_size, 2048);
    assert!((kernel.epsilon - 1e-5).abs() < 1e-10);
}

#[test]
fn test_vectorized_rmsnorm_kernel_name() {
    let kernel = VectorizedRmsNormKernel::new(1024);
    assert_eq!(kernel.name(), "rmsnorm_vectorized");
}

#[test]
fn test_vectorized_rmsnorm_with_epsilon() {
    let kernel = VectorizedRmsNormKernel::new(2048).with_epsilon(1e-6);
    assert!((kernel.epsilon - 1e-6).abs() < 1e-10);
}

#[test]
fn test_vectorized_rmsnorm_ptx_generation() {
    let kernel = VectorizedRmsNormKernel::new(2048);
    let ptx = kernel.emit_ptx();

    // Verify kernel entry point
    assert!(
        ptx.contains(".entry rmsnorm_vectorized"),
        "Should have rmsnorm_vectorized entry"
    );

    // Verify parameters
    assert!(
        ptx.contains(".param .u64 input_ptr"),
        "Should have input_ptr"
    );
    assert!(
        ptx.contains(".param .u64 output_ptr"),
        "Should have output_ptr"
    );
    assert!(
        ptx.contains(".param .u64 gamma_ptr"),
        "Should have gamma_ptr"
    );
}

#[test]
fn test_vectorized_rmsnorm_warp_operations() {
    let kernel = VectorizedRmsNormKernel::new(1024);
    let ptx = kernel.emit_ptx();

    // Verify warp shuffle for reduction
    assert!(
        ptx.contains("shfl.sync") || ptx.contains("shfl."),
        "Should have shfl for warp reduction"
    );

    // Verify rsqrt for 1/sqrt(rms)
    assert!(
        ptx.contains("rsqrt.f32") || ptx.contains("rsqrt"),
        "Should have rsqrt for RMSNorm"
    );
}

#[test]
fn test_vectorized_rmsnorm_shared_memory() {
    let kernel = VectorizedRmsNormKernel::new(2048);
    let ptx_kernel = kernel.build_ptx();

    // Vectorized kernel uses shared memory for inter-warp reduction
    assert!(
        ptx_kernel.shared_memory_bytes() > 0,
        "Vectorized RMSNorm should use shared memory"
    );
}

#[test]
fn test_vectorized_rmsnorm_various_sizes() {
    // Test common hidden sizes
    for hidden_size in [256, 512, 1024, 2048, 4096] {
        let kernel = VectorizedRmsNormKernel::new(hidden_size);
        assert_eq!(kernel.hidden_size, hidden_size);

        let ptx = kernel.emit_ptx();
        assert!(!ptx.is_empty());
        assert!(ptx.contains(".entry"));
        assert!(ptx.contains("ret;"));
    }
}

#[test]
fn test_vectorized_rmsnorm_numerical_ops() {
    let kernel = VectorizedRmsNormKernel::new(1024);
    let ptx = kernel.emit_ptx();

    // Verify numerical operations
    assert!(ptx.contains("mul.f32"), "Should have multiplication");
    assert!(ptx.contains("add.f32"), "Should have addition");
}

// ===== BatchedVectorizedRmsNormKernel Tests =====

#[test]
fn test_batched_vectorized_rmsnorm_kernel_new() {
    let kernel = BatchedVectorizedRmsNormKernel::new(2048, 8);
    assert_eq!(kernel.hidden_size, 2048);
    assert_eq!(kernel.batch_size, 8);
    assert!((kernel.epsilon - 1e-5).abs() < 1e-10);
}

#[test]
fn test_batched_vectorized_rmsnorm_kernel_name() {
    let kernel = BatchedVectorizedRmsNormKernel::new(1024, 4);
    assert_eq!(kernel.name(), "batched_rmsnorm_vectorized");
}

#[test]
fn test_batched_vectorized_rmsnorm_with_epsilon() {
    let kernel = BatchedVectorizedRmsNormKernel::new(2048, 4).with_epsilon(1e-6);
    assert!((kernel.epsilon - 1e-6).abs() < 1e-10);
}

#[test]
fn test_batched_vectorized_rmsnorm_ptx_generation() {
    let kernel = BatchedVectorizedRmsNormKernel::new(2048, 4);
    let ptx = kernel.emit_ptx();

    // Verify kernel entry point
    assert!(
        ptx.contains(".entry batched_rmsnorm_vectorized"),
        "Should have batched_rmsnorm_vectorized entry"
    );

    // Verify parameters
    assert!(
        ptx.contains(".param .u64 input_ptr"),
        "Should have input_ptr"
    );
    assert!(
        ptx.contains(".param .u64 output_ptr"),
        "Should have output_ptr"
    );
    assert!(
        ptx.contains(".param .u64 gamma_ptr"),
        "Should have gamma_ptr"
    );
}

#[test]
fn test_batched_vectorized_rmsnorm_batch_sizes() {
    // Test various batch sizes
    for batch_size in [1, 2, 4, 8, 16] {
        let kernel = BatchedVectorizedRmsNormKernel::new(1024, batch_size);
        assert_eq!(kernel.batch_size, batch_size);

        let ptx = kernel.emit_ptx();
        assert!(!ptx.is_empty());
        assert!(ptx.contains(".entry"));
    }
}

#[test]
fn test_batched_vectorized_rmsnorm_hidden_sizes() {
    // Test common hidden sizes
    for hidden_size in [256, 512, 1024, 2048, 4096] {
        let kernel = BatchedVectorizedRmsNormKernel::new(hidden_size, 4);
        assert_eq!(kernel.hidden_size, hidden_size);

        let ptx = kernel.emit_ptx();
        assert!(!ptx.is_empty());
        assert!(ptx.contains(".entry"));
    }
}

#[test]
fn test_batched_vectorized_rmsnorm_warp_operations() {
    let kernel = BatchedVectorizedRmsNormKernel::new(1024, 4);
    let ptx = kernel.emit_ptx();

    // Verify warp shuffle for reduction
    assert!(
        ptx.contains("shfl.sync") || ptx.contains("shfl."),
        "Should have shfl for warp reduction"
    );

    // Verify rsqrt for normalization
    assert!(
        ptx.contains("rsqrt.f32") || ptx.contains("rsqrt"),
        "Should have rsqrt"
    );
}

#[test]
fn test_batched_vectorized_rmsnorm_shared_memory() {
    let kernel = BatchedVectorizedRmsNormKernel::new(2048, 4);
    let ptx_kernel = kernel.build_ptx();

    // Batched kernel uses shared memory
    assert!(
        ptx_kernel.shared_memory_bytes() > 0,
        "Batched RMSNorm should use shared memory"
    );
}

#[test]
fn test_batched_vectorized_rmsnorm_memory_ops() {
    let kernel = BatchedVectorizedRmsNormKernel::new(1024, 8);
    let ptx = kernel.emit_ptx();

    // Verify memory operations
    assert!(ptx.contains("ld.global"), "Should have global loads");
    assert!(ptx.contains("st.global"), "Should have global stores");
}

#[test]
fn test_batched_vectorized_rmsnorm_barrier_sync() {
    let kernel = BatchedVectorizedRmsNormKernel::new(2048, 4);
    let ptx = kernel.emit_ptx();

    // Verify barrier synchronization for shared memory
    assert!(
        ptx.contains("bar.sync"),
        "Should have barrier synchronization"
    );
}

// =========================================================================
// GH-280: Per-Head RMSNorm kernel tests (Qwen3 QK normalization)
// =========================================================================

#[test]
fn test_per_head_rmsnorm_kernel_name() {
    let kernel = PerHeadRmsNormKernel::new(128, 32);
    assert_eq!(kernel.name(), "per_head_rmsnorm");
}

#[test]
fn test_per_head_rmsnorm_defaults() {
    let kernel = PerHeadRmsNormKernel::new(128, 32);
    assert_eq!(kernel.head_dim, 128);
    assert_eq!(kernel.num_heads, 32);
    assert!((kernel.epsilon - 1e-6).abs() < 1e-10);
}

#[test]
fn test_per_head_rmsnorm_with_epsilon() {
    let kernel = PerHeadRmsNormKernel::new(128, 32).with_epsilon(1e-5);
    assert!((kernel.epsilon - 1e-5).abs() < 1e-10);
}

#[test]
fn test_per_head_rmsnorm_ptx_generation() {
    let kernel = PerHeadRmsNormKernel::new(128, 32).with_epsilon(1e-6);
    let ptx = kernel.emit_ptx();

    // Verify kernel entry point
    assert!(
        ptx.contains(".entry per_head_rmsnorm"),
        "Should have per_head_rmsnorm entry"
    );

    // Verify parameters
    assert!(
        ptx.contains(".param .u64 input_ptr"),
        "Should have input_ptr"
    );
    assert!(
        ptx.contains(".param .u64 output_ptr"),
        "Should have output_ptr"
    );
    assert!(
        ptx.contains(".param .u64 gamma_ptr"),
        "Should have gamma_ptr"
    );

    // Verify per-head dispatch via blockIdx.x
    assert!(ptx.contains("ctaid.x"), "Should use ctaid.x for head index");

    // Verify warp shuffle for reduction
    assert!(
        ptx.contains("shfl.sync") || ptx.contains("shfl."),
        "Should have shfl for warp reduction"
    );

    // Verify rsqrt
    assert!(ptx.contains("rsqrt"), "Should have rsqrt for normalization");
}

#[test]
fn test_per_head_rmsnorm_no_shared_memory() {
    let kernel = PerHeadRmsNormKernel::new(128, 32);
    let ptx_kernel = kernel.build_ptx();

    // Single-warp kernel uses warp shuffle, no shared memory
    assert_eq!(
        ptx_kernel.shared_memory_bytes(),
        0,
        "Per-head RMSNorm should not use shared memory"
    );
}

#[test]
fn test_per_head_rmsnorm_memory_ops() {
    let kernel = PerHeadRmsNormKernel::new(128, 32);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains("ld.global"), "Should have global loads");
    assert!(ptx.contains("st.global"), "Should have global stores");
}

#[test]
fn test_per_head_rmsnorm_qwen3_q_config() {
    // Qwen3-8B Q projection: 32 heads, head_dim=128
    let kernel = PerHeadRmsNormKernel::new(128, 32).with_epsilon(1e-6);
    let ptx = kernel.emit_ptx();
    assert!(!ptx.is_empty());
    assert!(ptx.contains(".entry"));
}

#[test]
fn test_per_head_rmsnorm_qwen3_k_config() {
    // Qwen3-8B K projection: 8 KV heads, head_dim=128
    let kernel = PerHeadRmsNormKernel::new(128, 8).with_epsilon(1e-6);
    let ptx = kernel.emit_ptx();
    assert!(!ptx.is_empty());
    assert!(ptx.contains(".entry"));
}

#[test]
fn test_per_head_rmsnorm_various_head_dims() {
    for head_dim in [64, 128, 256] {
        let kernel = PerHeadRmsNormKernel::new(head_dim, 16);
        let ptx = kernel.emit_ptx();
        assert!(!ptx.is_empty());
        assert!(ptx.contains(".entry"));
    }
}
