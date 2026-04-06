//! Additional coverage tests: UnsafeMockKernel, barrier safety for all kernel types,
//! PTX structure validation, and kernel API coverage.

use super::*;

// =========================================================================
// Additional Coverage Tests - More Kernel Types
// =========================================================================

/// Mock kernel that produces barrier-unsafe PTX for testing
struct UnsafeMockKernel;

impl Kernel for UnsafeMockKernel {
    fn name(&self) -> &str {
        "unsafe_mock_kernel"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("unsafe_mock_kernel")
    }

    fn emit_ptx(&self) -> String {
        r#".version 8.0
.target sm_70
.address_size 64

.visible .entry unsafe_mock_kernel() {
loop_start:
    bra exit;
    bar.sync 0;
    bra loop_start;

loop_start_end:
exit:
    ret;
}
"#
        .to_string()
    }
}

/// Test UnsafeMockKernel::build_ptx() coverage
#[test]
fn test_unsafe_mock_kernel_build_ptx() {
    let kernel = UnsafeMockKernel;
    let ptx_kernel = kernel.build_ptx();
    assert!(ptx_kernel.shared_memory_bytes() == 0);
}

/// Test UnsafeMockKernel::as_module()
#[test]
fn test_unsafe_mock_kernel_as_module() {
    let kernel = UnsafeMockKernel;
    let module = kernel.as_module();
    let ptx = module.emit();
    assert!(ptx.contains(".version 8.0"));
}

/// Test UnsafeMockKernel::analyze_barrier_safety() detects violations
#[test]
fn test_unsafe_mock_kernel_analyze() {
    let kernel = UnsafeMockKernel;
    let result = kernel.analyze_barrier_safety();
    assert!(!result.is_safe);
    assert!(!result.violations.is_empty());
}

/// Test UnsafeMockKernel::validate_barrier_safety() returns Err
#[test]
fn test_unsafe_mock_kernel_validate() {
    let kernel = UnsafeMockKernel;
    let result = kernel.validate_barrier_safety();
    assert!(result.is_err());
}

/// Test emit_ptx_validated panics for unsafe PTX
#[test]
#[should_panic(expected = "PARITY-114")]
fn test_emit_ptx_validated_panics() {
    let kernel = UnsafeMockKernel;
    let _ = kernel.emit_ptx_validated();
}

/// Test CoalescedGemvKernel barrier safety
#[test]
fn test_barrier_safety_coalesced_gemv() {
    let kernel = CoalescedGemvKernel::new(1024, 4096);
    assert!(kernel.analyze_barrier_safety().is_safe);
}

/// Test TransposeKernel barrier safety
#[test]
fn test_barrier_safety_transpose() {
    let kernel = TransposeKernel::new(1024, 1024);
    assert!(kernel.analyze_barrier_safety().is_safe);
}

/// Test ResidualAddKernel barrier safety
#[test]
fn test_barrier_safety_residual_add() {
    let kernel = ResidualAddKernel::new(4096);
    assert!(kernel.analyze_barrier_safety().is_safe);
}

/// Test FusedSwigluKernel barrier safety
#[test]
fn test_barrier_safety_fused_swiglu() {
    let kernel = FusedSwigluKernel::new(4096);
    assert!(kernel.analyze_barrier_safety().is_safe);
}

/// Test RopeKernel barrier safety
#[test]
fn test_barrier_safety_rope() {
    let kernel = RopeKernel::new(8, 64, 10000.0);
    assert!(kernel.analyze_barrier_safety().is_safe);
}

/// Test ElementwiseMulKernel barrier safety
#[test]
fn test_barrier_safety_elementwise_mul() {
    let kernel = ElementwiseMulKernel::new(4096);
    assert!(kernel.analyze_barrier_safety().is_safe);
}

/// Test PreciseRmsNormKernel emits valid PTX
#[test]
fn test_precise_rmsnorm_ptx() {
    let kernel = PreciseRmsNormKernel::new(512);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
    assert!(ptx.contains(".entry"));
}

/// Test RmsNormKernel barrier safety
#[test]
fn test_barrier_safety_rmsnorm() {
    let kernel = RmsNormKernel::new(512);
    assert!(kernel.analyze_barrier_safety().is_safe);
}

/// Test VectorizedRmsNormKernel emits valid PTX
#[test]
fn test_vectorized_rmsnorm_ptx() {
    let kernel = VectorizedRmsNormKernel::new(512);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
    assert!(ptx.contains(".entry"));
}

/// Test GemvKernel barrier safety
#[test]
fn test_barrier_safety_gemv() {
    let kernel = GemvKernel::new(1024, 4096);
    assert!(kernel.analyze_barrier_safety().is_safe);
}

/// Test BiasActivationKernel barrier safety
#[test]
fn test_barrier_safety_bias_activation() {
    let kernel = BiasActivationKernel::new(1024, 64).with_relu();
    assert!(kernel.analyze_barrier_safety().is_safe);
}

/// Test IncrementalAttentionKernel barrier safety
#[test]
fn test_barrier_safety_incremental_attention() {
    let kernel = IncrementalAttentionKernel::new(2048, 64, 8);
    assert!(kernel.analyze_barrier_safety().is_safe);
}

/// Test BatchedGemmKernel barrier safety
#[test]
fn test_barrier_safety_batched_gemm() {
    let kernel = BatchedGemmKernel::naive(4, 64, 64, 64);
    assert!(kernel.analyze_barrier_safety().is_safe);

    let kernel_tiled = BatchedGemmKernel::tiled(4, 64, 64, 64, 16);
    assert!(kernel_tiled.analyze_barrier_safety().is_safe);
}

/// Test Batched4DGemmKernel barrier safety
#[test]
fn test_barrier_safety_batched_4d_gemm() {
    let kernel = Batched4DGemmKernel::new(2, 8, 64, 64, 32);
    assert!(kernel.analyze_barrier_safety().is_safe);
}

/// Test LongRowSoftmaxKernel barrier safety
#[test]
fn test_barrier_safety_long_row_softmax() {
    let kernel = LongRowSoftmaxKernel::new(8192);
    assert!(kernel.analyze_barrier_safety().is_safe);
}

/// Test elementwise kernels are barrier-safe
#[test]
fn test_barrier_safety_elementwise_kernels() {
    assert!(ReluKernel::new(1024).analyze_barrier_safety().is_safe);
    assert!(GeluKernel::new(1024).analyze_barrier_safety().is_safe);
    assert!(SiluKernel::new(1024).analyze_barrier_safety().is_safe);
    assert!(ScaleKernel::new(1024).analyze_barrier_safety().is_safe);
}

/// Test validate_barrier_safety returns Ok for safe kernels
#[test]
fn test_validate_barrier_safety_all_pass() {
    let kernels: Vec<Box<dyn Kernel>> = vec![
        Box::new(GemmKernel::naive(32, 32, 32)),
        Box::new(SoftmaxKernel::new(128)),
        Box::new(LayerNormKernel::new(64)),
    ];
    for kernel in kernels {
        assert!(kernel.validate_barrier_safety().is_ok());
    }
}

/// Test as_module returns correct structure
#[test]
fn test_as_module_structure() {
    let kernel = GemmKernel::naive(32, 32, 32);
    let module = kernel.as_module();
    let ptx = module.emit();
    assert!(ptx.contains(".version 8.0"));
    assert!(ptx.contains(".target sm_70"));
    assert!(ptx.contains(".address_size 64"));
}

/// Test build_ptx for tiled kernel uses shared memory
#[test]
fn test_build_ptx_shared_memory() {
    let kernel = GemmKernel::tiled(64, 64, 64, 16);
    let ptx_kernel = kernel.build_ptx();
    assert!(ptx_kernel.shared_memory_bytes() > 0);
}

/// Test analyze_barrier_safety returns counts
#[test]
fn test_analyze_barrier_safety_counts() {
    let kernel = GemmKernel::tiled(64, 64, 64, 16);
    let result = kernel.analyze_barrier_safety();
    assert!(result.barrier_count > 0);
    assert!(result.exit_count > 0);
    assert!(result.is_safe);
    assert!(result.violations.is_empty());
}

/// Test emit_ptx_validated for multiple kernels
#[test]
fn test_emit_ptx_validated_multiple() {
    let kernels: Vec<Box<dyn Kernel>> =
        vec![Box::new(GemmKernel::naive(32, 32, 32)), Box::new(SoftmaxKernel::new(128))];
    for kernel in kernels {
        let ptx = kernel.emit_ptx_validated();
        assert!(!ptx.is_empty());
    }
}

/// Test kernel name consistency
#[test]
fn test_kernel_name_consistency() {
    let kernel = GemmKernel::naive(128, 128, 128);
    let n1 = kernel.name();
    let n2 = kernel.name();
    assert_eq!(n1, n2);
    assert!(!n1.is_empty());
}

/// Test emit_ptx produces valid output each time
#[test]
fn test_emit_ptx_validity() {
    let kernel = GemmKernel::naive(64, 64, 64);
    let ptx1 = kernel.emit_ptx();
    let ptx2 = kernel.emit_ptx();
    // Both outputs should be valid PTX (may differ in register ordering)
    assert!(ptx1.contains(".version 8.0"));
    assert!(ptx2.contains(".version 8.0"));
    assert!(ptx1.contains(".entry"));
    assert!(ptx2.contains(".entry"));
}

/// Test validate error message format
#[test]
fn test_validate_error_format() {
    let unsafe_ptx = r#"
.version 8.0
.target sm_70
.address_size 64

.visible .entry test() {
loop_start:
    bra exit;
    bar.sync 0;
    bra loop_start;

loop_start_end:
exit:
    ret;
}
"#;
    let result = crate::ptx::optimize::barrier_safety::validate(unsafe_ptx);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.contains("PARITY-114"));
}

/// Test ArgMaxKernel barrier safety
#[test]
fn test_barrier_safety_argmax() {
    let kernel = ArgMaxKernel::new(4096);
    assert!(kernel.analyze_barrier_safety().is_safe);
}

/// Test ArgMaxFinalKernel barrier safety
#[test]
fn test_barrier_safety_argmax_final() {
    let kernel = ArgMaxFinalKernel::new(128);
    assert!(kernel.analyze_barrier_safety().is_safe);
}

/// Test Conv1dKernel barrier safety
#[test]
fn test_barrier_safety_conv1d() {
    let kernel = Conv1dKernel::new(3, 128, 256, 3, 1);
    assert!(kernel.analyze_barrier_safety().is_safe);
}

/// Test TiledConv1dKernel barrier safety
#[test]
fn test_barrier_safety_tiled_conv1d() {
    let kernel = TiledConv1dKernel::whisper_conv1();
    assert!(kernel.analyze_barrier_safety().is_safe);
}

/// Test optimizer kernels barrier safety
#[test]
fn test_barrier_safety_optimizer() {
    assert!(AdamWStepKernel::new(4096).analyze_barrier_safety().is_safe);
    assert!(AdamStepKernel::new(4096).analyze_barrier_safety().is_safe);
    assert!(GradientClipKernel::new(4096).analyze_barrier_safety().is_safe);
}

/// Test quantize kernels barrier safety
#[test]
fn test_barrier_safety_quantize() {
    assert!(Q4KGemvKernel::new(4096, 4096).analyze_barrier_safety().is_safe);
    assert!(Q6KGemvKernel::new(4096, 4096).analyze_barrier_safety().is_safe);
    assert!(Q8QuantizeKernel::new(4096).analyze_barrier_safety().is_safe);
}

/// Test FusedQKVKernel barrier safety
#[test]
fn test_barrier_safety_fused_qkv() {
    let kernel = FusedQKVKernel::new(512, 64);
    assert!(kernel.analyze_barrier_safety().is_safe);
}

/// Test FusedGateUpKernel barrier safety
#[test]
fn test_barrier_safety_fused_gate_up() {
    let kernel = FusedGateUpKernel::new(512, 2048);
    assert!(kernel.analyze_barrier_safety().is_safe);
}

/// Test FusedGemmBiasGeluKernel barrier safety
#[test]
fn test_barrier_safety_fused_gemm_bias_gelu() {
    let kernel = FusedGemmBiasGeluKernel::new(512, 2048, 512);
    assert!(kernel.analyze_barrier_safety().is_safe);
}

/// Test TransformerBlockMegakernel barrier safety
#[test]
fn test_barrier_safety_megakernel() {
    let kernel = TransformerBlockMegakernel::new(512, 2048, 8);
    assert!(kernel.analyze_barrier_safety().is_safe);
}

/// Test PersistentDecoderKernel emits valid PTX
#[test]
fn test_persistent_ptx() {
    let kernel = PersistentDecoderKernel::new(512, 12, 2048);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"));
    assert!(ptx.contains(".entry"));
}

/// Test Lz4 kernels barrier safety
#[test]
fn test_barrier_safety_lz4() {
    assert!(Lz4WarpCompressKernel::new(4).analyze_barrier_safety().is_safe);
    assert!(Lz4WarpDecompressKernel::new(4).analyze_barrier_safety().is_safe);
}

/// Test batched elementwise kernels
#[test]
fn test_barrier_safety_batched_elementwise() {
    assert!(BatchedSoftmaxKernel::new(4, 1024).analyze_barrier_safety().is_safe);
    assert!(BatchedScaleKernel::new(1024).analyze_barrier_safety().is_safe);
}

/// Test FusedResidualRmsNormKernel barrier safety
#[test]
fn test_barrier_safety_fused_residual_rmsnorm() {
    let kernel = FusedResidualRmsNormKernel::new(512);
    assert!(kernel.analyze_barrier_safety().is_safe);
}

/// Test KvCacheScatterKernel barrier safety
#[test]
fn test_barrier_safety_kv_cache_scatter() {
    let kernel = KvCacheScatterKernel::new(32, 64, 2048);
    assert!(kernel.analyze_barrier_safety().is_safe);
}

/// Test KvCacheScatterIndirectKernel barrier safety
#[test]
fn test_barrier_safety_kv_cache_scatter_indirect() {
    let kernel = KvCacheScatterIndirectKernel::new(32, 64, 2048);
    assert!(kernel.analyze_barrier_safety().is_safe);
}
