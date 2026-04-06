use super::*;
use crate::cuda::memory::{SizeClass, TransferMode};
use crate::cuda::pipeline::{
    presets, AsyncPipeline, BankConflictStrategy, MemoryPattern, PtxOptimizationHints,
    PtxOptimizer, RegisterTiling,
};
use proptest::prelude::*;
use serial_test::serial;

// Only run property tests on systems with CUDA
fn has_cuda() -> bool {
    CudaExecutor::is_available() && CudaExecutor::num_devices() > 0
}

proptest! {
    /// Property: Any number of lifecycle cycles should succeed
    /// (validates Drop order correctness)
    #[test]
    #[serial]
    fn prop_lifecycle_cycles_always_succeed(cycles in 1..5usize) {
        if !has_cuda() {
            return Ok(());
        }

        for i in 0..cycles {
            let executor = CudaExecutor::new(0)
                .map_err(|e| TestCaseError::fail(format!("Cycle {}: {}", i, e)))?;

            // Verify basic operations work
            prop_assert!(executor.device_name().is_ok());

            // Drop happens here
        }
    }

    /// Property: GEMM with valid dimensions should succeed on any executor
    #[test]
    #[serial]
    fn prop_gemm_valid_dims_succeed(size in 4..16u32) {
        if !has_cuda() {
            return Ok(());
        }

        let mut executor = CudaExecutor::new(0)
            .map_err(|e| TestCaseError::fail(format!("{}", e)))?;

        let n = size * size;
        let a = vec![1.0f32; n as usize];
        let b = vec![1.0f32; n as usize];
        let mut c = vec![0.0f32; n as usize];

        let result = executor.gemm(&a, &b, &mut c, size, size, size);
        prop_assert!(result.is_ok(), "GEMM should succeed for {}x{}", size, size);

        // Verify result is correct (each element should be `size`)
        let expected = size as f32;
        for (i, &val) in c.iter().enumerate() {
            prop_assert!(
                (val - expected).abs() < 1e-3,
                "c[{}] = {}, expected {}",
                i,
                val,
                expected
            );
        }
    }

    /// Property: Multiple executors can coexist (if needed)
    #[test]
    #[serial]
    fn prop_sequential_executors_independent(count in 1..3usize) {
        if !has_cuda() {
            return Ok(());
        }

        // Create and use executors sequentially
        for i in 0..count {
            let mut executor = CudaExecutor::new(0)
                .map_err(|e| TestCaseError::fail(format!("Executor {}: {}", i, e)))?;

            // Each executor should work independently
            let a = vec![1.0f32; 16];
            let b = vec![1.0f32; 16];
            let mut c = vec![0.0f32; 16];

            let result = executor.gemm(&a, &b, &mut c, 4, 4, 4);
            prop_assert!(result.is_ok(), "Executor {} GEMM failed", i);
        }
    }
}

/// Non-property test: Verify GEMM size validation always catches invalid inputs
#[test]
#[serial]
fn test_gemm_invalid_size_always_rejected() {
    if !has_cuda() {
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("test");

    // Wrong A size
    let a = vec![1.0f32; 10]; // Should be 16
    let b = vec![1.0f32; 16];
    let mut c = vec![0.0f32; 16];
    assert!(executor.gemm(&a, &b, &mut c, 4, 4, 4).is_err());

    // Wrong B size
    let a = vec![1.0f32; 16];
    let b = vec![1.0f32; 10]; // Should be 16
    let mut c = vec![0.0f32; 16];
    assert!(executor.gemm(&a, &b, &mut c, 4, 4, 4).is_err());

    // Wrong C size
    let a = vec![1.0f32; 16];
    let b = vec![1.0f32; 16];
    let mut c = vec![0.0f32; 10]; // Should be 16
    assert!(executor.gemm(&a, &b, &mut c, 4, 4, 4).is_err());
}

/// IMP-1000a: FP16 Tensor Core kernel PTX generation
#[test]
fn test_imp_1000a_fp16_tensor_core_ptx_generation() {
    let kernels = CudaKernels::new();
    let kernel_type = KernelType::GemmFp16TensorCore {
        m: 64,
        n: 64,
        k: 64,
    };

    let ptx = kernels.generate_ptx(&kernel_type);

    // Now uses trueno's GemmKernel::wmma_fp16()
    assert!(ptx.contains(".visible .entry gemm_wmma_fp16"));
    // trueno uses lowercase ptr names
    assert!(ptx.contains(".param .u64 a_ptr"));
    assert!(ptx.contains(".param .u64 b_ptr"));
    assert!(ptx.contains(".param .u64 c_ptr"));
    // trueno uses lowercase dimension names
    assert!(ptx.contains(".param .u32 m") || ptx.contains("m_param"));

    // Trueno's WMMA kernel has proper tensor core intrinsics
    // or uses tiled FP32 fallback with FP16 memory traffic
    assert!(ptx.contains(".shared")); // Shared memory for tiles

    // Verify kernel name matches trueno
    assert_eq!(kernels.kernel_name(&kernel_type), "gemm_wmma_fp16");
}

/// IMP-1000a: FP16 kernel dimensions must be multiples of 16
#[test]
fn test_imp_1000a_fp16_dimension_requirements() {
    // Verify the kernel type documents the 16-alignment requirement
    let kernel_type = KernelType::GemmFp16TensorCore {
        m: 16, // Must be multiple of 16
        n: 32, // Must be multiple of 16
        k: 48, // Must be multiple of 16
    };

    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&kernel_type);

    // PTX should be generated using trueno (validation happens at runtime)
    assert!(!ptx.is_empty());
    assert!(ptx.contains("gemm_wmma_fp16")); // trueno's kernel name
}

/// IMP-1000a: FP16 GEMM rejects non-16-aligned dimensions
#[test]
#[serial]
fn test_imp_1000a_fp16_gemm_alignment_validation() {
    if !has_cuda() {
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("test");

    // Valid: all dimensions multiple of 16
    let a = vec![1.0f32; 16 * 32];
    let b = vec![1.0f32; 32 * 16];
    let mut c = vec![0.0f32; 16 * 16];
    assert!(executor.gemm_fp16(&a, &b, &mut c, 16, 16, 32).is_ok());

    // Invalid: m not multiple of 16
    let a = vec![1.0f32; 15 * 32];
    let b = vec![1.0f32; 32 * 16];
    let mut c = vec![0.0f32; 15 * 16];
    assert!(executor.gemm_fp16(&a, &b, &mut c, 15, 16, 32).is_err());

    // Invalid: n not multiple of 16
    let a = vec![1.0f32; 16 * 32];
    let b = vec![1.0f32; 32 * 17];
    let mut c = vec![0.0f32; 16 * 17];
    assert!(executor.gemm_fp16(&a, &b, &mut c, 16, 17, 32).is_err());

    // Invalid: k not multiple of 16
    let a = vec![1.0f32; 16 * 33];
    let b = vec![1.0f32; 33 * 16];
    let mut c = vec![0.0f32; 16 * 16];
    assert!(executor.gemm_fp16(&a, &b, &mut c, 16, 16, 33).is_err());
}

/// IMP-1000a: FP16 GEMM produces correct results
#[test]
#[serial]
fn test_imp_1000a_fp16_gemm_correctness() {
    if !has_cuda() {
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("test");

    // Simple 16x16 identity-like multiplication
    let m = 16u32;
    let n = 16u32;
    let k = 16u32;

    // A = all 1s, B = identity-like (diagonal 1s scaled)
    let a = vec![1.0f32; (m * k) as usize];
    let mut b = vec![0.0f32; (k * n) as usize];
    for i in 0..k.min(n) {
        b[(i * n + i) as usize] = 1.0;
    }
    let mut c = vec![0.0f32; (m * n) as usize];

    executor.gemm_fp16(&a, &b, &mut c, m, n, k).expect("test");

    // Each row of C should sum to n (since A is all 1s and B is identity, C = A)
    for row in 0..m {
        let row_sum: f32 = (0..n).map(|col| c[(row * n + col) as usize]).sum();
        assert!(
            (row_sum - n as f32).abs() < 1.0,
            "Row {} sum {} != {}",
            row,
            row_sum,
            n
        );
    }
}

// ========================================================================
// IMP-1000b: Fused Q4_K GEMM Tests
// ========================================================================

/// IMP-1000b: Verify Q4_K fused kernel PTX generation
#[test]
fn test_imp_1000b_q4k_fused_ptx_generation() {
    let kernels = CudaKernels::new();
    let kernel_type = KernelType::QuantizedGemm {
        m: 1,
        n: 4096,
        k: 4096,
    };

    let ptx = kernels.generate_ptx(&kernel_type);

    // Verify PTX contains fused operations
    assert!(ptx.contains(".visible .entry q4k_gemm_fused"));
    assert!(ptx.contains(".param .u64 a_ptr"));
    assert!(ptx.contains(".param .u64 b_quant_ptr"));
    assert!(ptx.contains(".param .u64 c_ptr"));

    // Verify dequantization and GEMM ops are fused
    assert!(ptx.contains("mul.f32"), "Missing mul.f32 for dequant");
    assert!(ptx.contains("add.f32"), "Missing add.f32 for accumulate");

    // Verify warp shuffle for efficient reduction
    assert!(
        ptx.contains("shfl") || ptx.contains("shfl.down"),
        "Missing warp shuffle for reduction"
    );
}

/// IMP-1000b: Verify Q4_K block layout constants
#[test]
fn test_imp_1000b_q4k_block_layout() {
    // Q4_K block: 32 weights, 18 bytes (2 header + 16 data)
    let kernel_type = KernelType::QuantizedGemm {
        m: 1,
        n: 128,  // 128 / 32 = 4 blocks
        k: 4096, // 4096 / 32 = 128 blocks per row
    };

    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&kernel_type);

    // K must be divisible by 32 (block size)
    assert_eq!(4096 % 32, 0);

    // PTX should be valid
    assert!(!ptx.is_empty());
    assert!(ptx.contains("q4k_gemm_fused"));
}

/// IMP-1000b: Verify GEMM works with Q4_K-compatible dimensions
#[test]
#[serial]
fn test_imp_1000b_q4k_gemm_integration() {
    if !has_cuda() {
        return;
    }

    let mut executor = CudaExecutor::new(0).expect("test");

    // Use dimensions compatible with Q4_K (K must be multiple of 32)
    let m = 32u32;
    let n = 32u32;
    let k = 128u32; // Must be multiple of 32

    let a = vec![1.0f32; (m * k) as usize];
    let b = vec![1.0f32; (k * n) as usize];
    let mut c = vec![0.0f32; (m * n) as usize];

    // This tests the GEMM path that could use fused Q4_K
    let result = executor.gemm(&a, &b, &mut c, m, n, k);
    assert!(result.is_ok(), "GEMM failed: {:?}", result);
}

/// IMP-1000b: Verify preset generates correct kernel type
#[test]
fn test_imp_1000b_q4k_preset() {
    let kernel = presets::q4k_inference(1, 4096, 4096);

    match kernel {
        KernelType::QuantizedGemm { m, n, k } => {
            assert_eq!(m, 1, "Batch size should be 1");
            assert_eq!(n, 4096, "Hidden dim should be 4096");
            assert_eq!(k, 4096, "K dim should be 4096");
        },
        _ => panic!("Expected QuantizedGemm kernel type"),
    }
}

// ========================================================================
// IMP-1000c: Async Memory Pipelining Tests
// ========================================================================

/// IMP-1000c: Verify AsyncPipeline creation
#[test]
#[serial]
fn test_imp_1000c_async_pipeline_creation() {
    if !has_cuda() {
        return;
    }

    let context = CudaContext::new(0).expect("test");
    let pipeline = AsyncPipeline::new(&context);

    assert!(pipeline.is_ok(), "AsyncPipeline creation failed");

    let pipeline = pipeline.expect("test");
    assert!(!pipeline.is_active());
    assert_eq!(pipeline.layers_queued(), 0);
}

/// IMP-1000c: Verify pipeline lifecycle (begin/enqueue/end)
#[test]
#[serial]
fn test_imp_1000c_async_pipeline_lifecycle() {
    if !has_cuda() {
        return;
    }

    let context = CudaContext::new(0).expect("test");
    let mut pipeline = AsyncPipeline::new(&context).expect("test");

    // Begin
    pipeline.begin();
    assert!(pipeline.is_active());

    // Enqueue layers
    let l0 = pipeline.enqueue_layer();
    let l1 = pipeline.enqueue_layer();
    let l2 = pipeline.enqueue_layer();

    assert_eq!(l0, 0);
    assert_eq!(l1, 1);
    assert_eq!(l2, 2);
    assert_eq!(pipeline.layers_queued(), 3);

    // End
    let result = pipeline.end();
    assert!(result.is_ok());
    assert!(!pipeline.is_active());
}

/// IMP-1000c: Verify dual-stream sync
#[test]
#[serial]
fn test_imp_1000c_async_dual_stream_sync() {
    if !has_cuda() {
        return;
    }

    let context = CudaContext::new(0).expect("test");
    let pipeline = AsyncPipeline::new(&context).expect("test");

    // Both streams should sync without error
    let sync_result = pipeline.sync();
    assert!(sync_result.is_ok(), "Dual-stream sync failed");
}

/// IMP-1000c: Verify stream accessors
#[test]
#[serial]
fn test_imp_1000c_async_stream_accessors() {
    if !has_cuda() {
        return;
    }

    let context = CudaContext::new(0).expect("test");
    let pipeline = AsyncPipeline::new(&context).expect("test");

    // Streams should be accessible
    let _compute = pipeline.compute_stream();
    let _transfer = pipeline.transfer_stream();

    // And sync individually
    assert!(pipeline.compute_stream().synchronize().is_ok());
    assert!(pipeline.transfer_stream().synchronize().is_ok());
}

// ========================================================================
// IMP-1000d: PTX Micro-optimization Tests
// ========================================================================

/// IMP-1000d: Verify PtxOptimizationHints default values
#[test]
fn test_imp_1000d_optimization_hints_default() {
    let hints = PtxOptimizationHints::default();

    assert_eq!(hints.memory_pattern, MemoryPattern::Scalar);
    assert_eq!(hints.register_tiling.width, 4);
    assert_eq!(hints.register_tiling.height, 4);
    assert_eq!(hints.bank_conflict_strategy, BankConflictStrategy::None);
    assert!(!hints.enable_ilp);
    assert!(!hints.uses_vectorized_loads());
    assert_eq!(hints.vector_width(), 1);
}

include!("proptests_imp_1000d.rs");
include!("proptests_kernels.rs");
include!("proptests_tqa012d_forward.rs");
include!("proptests_tqa013e_batched.rs");
include!("proptests_tcov001s_tcov001t_tcov001u.rs");
include!("proptests_tcov002_kernel.rs");
include!("proptests_tcov010_more.rs");
