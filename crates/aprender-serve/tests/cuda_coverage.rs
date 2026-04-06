//! EXTREME TDD Coverage Tests for cuda.rs
//!
//! These tests focus on increasing coverage of the cuda module
//! without requiring CUDA runtime to be available.
//!
//! Test categories:
//! 1. Struct construction and field access
//! 2. PTX generation for all kernel types
//! 3. Memory pool operations
//! 4. Pinned buffer operations
//! 5. Transfer mode handling
//! 6. Weight quantization type detection
//! 7. Presets module
//! 8. Error path coverage

#![cfg(feature = "cuda")]

use realizar::cuda::{
    presets, CudaKernels, GpuMemoryPool, IndexedLayerWeights, KernelType, PinnedHostBuffer,
    PoolStats, SizeClass, StagingBufferPool, StagingPoolStats, TransferMode, TransformerWorkspace,
    WeightQuantType,
};

// ============================================================================
// KernelType Construction Tests
// ============================================================================

#[test]
fn test_kernel_type_gemm_naive_construction() {
    let kernel = KernelType::GemmNaive {
        m: 64,
        n: 128,
        k: 256,
    };
    match kernel {
        KernelType::GemmNaive { m, n, k } => {
            assert_eq!(m, 64);
            assert_eq!(n, 128);
            assert_eq!(k, 256);
        },
        _ => panic!("Expected GemmNaive"),
    }
}

#[test]
fn test_kernel_type_gemm_tiled_construction() {
    let kernel = KernelType::GemmTiled {
        m: 32,
        n: 64,
        k: 128,
        tile_size: 16,
    };
    match kernel {
        KernelType::GemmTiled { m, n, k, tile_size } => {
            assert_eq!(m, 32);
            assert_eq!(n, 64);
            assert_eq!(k, 128);
            assert_eq!(tile_size, 16);
        },
        _ => panic!("Expected GemmTiled"),
    }
}

#[test]
fn test_kernel_type_gemm_tensor_core_construction() {
    let kernel = KernelType::GemmTensorCore {
        m: 16,
        n: 16,
        k: 16,
    };
    match kernel {
        KernelType::GemmTensorCore { m, n, k } => {
            assert_eq!(m, 16);
            assert_eq!(n, 16);
            assert_eq!(k, 16);
        },
        _ => panic!("Expected GemmTensorCore"),
    }
}

#[test]
fn test_kernel_type_gemv_construction() {
    let kernel = KernelType::Gemv { k: 4096, n: 2048 };
    match kernel {
        KernelType::Gemv { k, n } => {
            assert_eq!(k, 4096);
            assert_eq!(n, 2048);
        },
        _ => panic!("Expected Gemv"),
    }
}

#[test]
fn test_kernel_type_coalesced_gemv_construction() {
    let kernel = KernelType::CoalescedGemv { k: 2048, n: 1024 };
    match kernel {
        KernelType::CoalescedGemv { k, n } => {
            assert_eq!(k, 2048);
            assert_eq!(n, 1024);
        },
        _ => panic!("Expected CoalescedGemv"),
    }
}

#[test]
fn test_kernel_type_softmax_construction() {
    let kernel = KernelType::Softmax { dim: 32000 };
    match kernel {
        KernelType::Softmax { dim } => {
            assert_eq!(dim, 32000);
        },
        _ => panic!("Expected Softmax"),
    }
}

#[test]
fn test_kernel_type_layer_norm_construction() {
    let kernel = KernelType::LayerNorm {
        hidden_size: 4096,
        epsilon: 1e-5,
        affine: true,
    };
    match kernel {
        KernelType::LayerNorm {
            hidden_size,
            epsilon,
            affine,
        } => {
            assert_eq!(hidden_size, 4096);
            assert!((epsilon - 1e-5).abs() < 1e-10);
            assert!(affine);
        },
        _ => panic!("Expected LayerNorm"),
    }
}

#[test]
fn test_kernel_type_attention_construction() {
    let kernel = KernelType::Attention {
        seq_len: 2048,
        head_dim: 64,
        causal: true,
    };
    match kernel {
        KernelType::Attention {
            seq_len,
            head_dim,
            causal,
        } => {
            assert_eq!(seq_len, 2048);
            assert_eq!(head_dim, 64);
            assert!(causal);
        },
        _ => panic!("Expected Attention"),
    }
}

#[test]
fn test_kernel_type_multi_head_attention_construction() {
    let kernel = KernelType::MultiHeadAttention {
        seq_len: 512,
        head_dim: 64,
        n_heads: 32,
        causal: false,
    };
    match kernel {
        KernelType::MultiHeadAttention {
            seq_len,
            head_dim,
            n_heads,
            causal,
        } => {
            assert_eq!(seq_len, 512);
            assert_eq!(head_dim, 64);
            assert_eq!(n_heads, 32);
            assert!(!causal);
        },
        _ => panic!("Expected MultiHeadAttention"),
    }
}

#[test]
fn test_kernel_type_attention_tensor_core_construction() {
    let kernel = KernelType::AttentionTensorCore {
        seq_len: 1024,
        head_dim: 128,
        n_heads: 16,
        causal: true,
    };
    match kernel {
        KernelType::AttentionTensorCore {
            seq_len,
            head_dim,
            n_heads,
            causal,
        } => {
            assert_eq!(seq_len, 1024);
            assert_eq!(head_dim, 128);
            assert_eq!(n_heads, 16);
            assert!(causal);
        },
        _ => panic!("Expected AttentionTensorCore"),
    }
}

#[test]
fn test_kernel_type_quantized_gemm_construction() {
    let kernel = KernelType::QuantizedGemm {
        m: 1,
        n: 4096,
        k: 4096,
    };
    match kernel {
        KernelType::QuantizedGemm { m, n, k } => {
            assert_eq!(m, 1);
            assert_eq!(n, 4096);
            assert_eq!(k, 4096);
        },
        _ => panic!("Expected QuantizedGemm"),
    }
}

#[test]
fn test_kernel_type_quantized_gemm_ggml_construction() {
    let kernel = KernelType::QuantizedGemmGgml {
        m: 1,
        n: 2560,
        k: 2560,
    };
    match kernel {
        KernelType::QuantizedGemmGgml { m, n, k } => {
            assert_eq!(m, 1);
            assert_eq!(n, 2560);
            assert_eq!(k, 2560);
        },
        _ => panic!("Expected QuantizedGemmGgml"),
    }
}

#[test]
fn test_kernel_type_q5k_quantized_gemm_construction() {
    let kernel = KernelType::Q5KQuantizedGemm {
        m: 1,
        n: 4096,
        k: 4096,
    };
    match kernel {
        KernelType::Q5KQuantizedGemm { m, n, k } => {
            assert_eq!(m, 1);
            assert_eq!(n, 4096);
            assert_eq!(k, 4096);
        },
        _ => panic!("Expected Q5KQuantizedGemm"),
    }
}

#[test]
fn test_kernel_type_q6k_quantized_gemm_construction() {
    let kernel = KernelType::Q6KQuantizedGemm {
        m: 1,
        n: 4096,
        k: 4096,
    };
    match kernel {
        KernelType::Q6KQuantizedGemm { m, n, k } => {
            assert_eq!(m, 1);
            assert_eq!(n, 4096);
            assert_eq!(k, 4096);
        },
        _ => panic!("Expected Q6KQuantizedGemm"),
    }
}

#[test]
fn test_kernel_type_incremental_attention_construction() {
    let kernel = KernelType::IncrementalAttention {
        max_seq_len: 4096,
        head_dim: 64,
        n_heads: 32,
        n_kv_heads: 4,
        indirect: true,
    };
    match kernel {
        KernelType::IncrementalAttention {
            max_seq_len,
            head_dim,
            n_heads,
            n_kv_heads,
            indirect,
        } => {
            assert_eq!(max_seq_len, 4096);
            assert_eq!(head_dim, 64);
            assert_eq!(n_heads, 32);
            assert_eq!(n_kv_heads, 4);
            assert!(indirect);
        },
        _ => panic!("Expected IncrementalAttention"),
    }
}

#[test]
fn test_kernel_type_rms_norm_construction() {
    let kernel = KernelType::RmsNorm {
        hidden_size: 2048,
        epsilon: 1e-6,
    };
    match kernel {
        KernelType::RmsNorm {
            hidden_size,
            epsilon,
        } => {
            assert_eq!(hidden_size, 2048);
            assert!((epsilon - 1e-6).abs() < 1e-10);
        },
        _ => panic!("Expected RmsNorm"),
    }
}

#[test]
fn test_kernel_type_rope_construction() {
    let kernel = KernelType::Rope {
        num_heads: 32,
        head_dim: 64,
        theta: 10000.0,
    };
    match kernel {
        KernelType::Rope {
            num_heads,
            head_dim,
            theta,
        } => {
            assert_eq!(num_heads, 32);
            assert_eq!(head_dim, 64);
            assert!((theta - 10000.0).abs() < 0.01);
        },
        _ => panic!("Expected Rope"),
    }
}

#[test]
fn test_kernel_type_silu_construction() {
    let kernel = KernelType::Silu { n: 4096 };
    match kernel {
        KernelType::Silu { n } => {
            assert_eq!(n, 4096);
        },
        _ => panic!("Expected Silu"),
    }
}

#[test]
fn test_kernel_type_gelu_construction() {
    let kernel = KernelType::Gelu { n: 4096 };
    match kernel {
        KernelType::Gelu { n } => {
            assert_eq!(n, 4096);
        },
        _ => panic!("Expected Gelu"),
    }
}

// ============================================================================
// CudaKernels PTX Generation Tests
// ============================================================================

#[test]
fn test_cuda_kernels_default() {
    let kernels = CudaKernels::default();
    let ptx = kernels.generate_ptx(&KernelType::Softmax { dim: 128 });
    assert!(!ptx.is_empty());
    assert!(ptx.contains(".version"));
}

#[test]
fn test_cuda_kernels_new() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Softmax { dim: 256 });
    assert!(!ptx.is_empty());
}

#[test]
fn test_ptx_gemm_optimized() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::GemmOptimized {
        m: 32,
        n: 32,
        k: 32,
        tile_size: 16,
        reg_block: 4,
    };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
    assert!(ptx.contains("gemm"));
}

#[test]
fn test_ptx_gemm_bias_activation() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::GemmBiasActivation {
        m: 64,
        n: 64,
        k: 64,
        activation: 2,
    };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_bias_activation() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::BiasActivation {
        n: 1024,
        bias_size: 64,
        activation: 1,
    };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
    assert!(ptx.contains("bias_activation"));
}

#[test]
fn test_ptx_gemm_fp16_tensor_core() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::GemmFp16TensorCore {
        m: 128,
        n: 128,
        k: 128,
    };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_q4k_gemv() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::Q4KGemv { k: 2560, n: 2560 };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
    assert!(ptx.contains("q4k_gemv"));
}

#[test]
fn test_ptx_tiled_q4k_gemv() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::TiledQ4KGemv {
        k: 2560,
        n: 2560,
        outputs_per_block: 4,
    };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_chunked_tiled_q4k_gemv() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::ChunkedTiledQ4KGemv {
        k: 10240,
        n: 2560,
        outputs_per_block: 4,
    };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_coalesced_q4k_gemv() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::CoalescedQ4KGemv { k: 2560, n: 2560 };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_vectorized_q4k_gemv() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::VectorizedQ4KGemv { k: 2560, n: 2560 };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_dp4a_q4k_gemv() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::Dp4aQ4KGemv { k: 2560, n: 2560 };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_dp4a_simd_q4k_gemv() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::Dp4aSIMDQ4KGemv { k: 2560, n: 2560 };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_q5k_gemv() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::Q5KGemv { k: 2560, n: 2560 };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_q6k_gemv() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::Q6KGemv { k: 2560, n: 2560 };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_coalesced_q6k_gemv() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::CoalescedQ6KGemv { k: 2560, n: 2560 };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_batched_q6k_gemv() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::BatchedQ6KGemv {
        k: 2560,
        n: 2560,
        m: 8,
    };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_fp16_q4k_gemv() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::Fp16Q4KGemv { k: 2560, n: 2560 };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_q8_0_gemv() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::Q8_0Gemv { k: 2048, n: 2048 };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_q5_0_gemv() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::Q5_0Gemv { k: 2048, n: 2048 };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_q4_0_gemv() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::Q4_0Gemv { k: 2048, n: 2048 };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_q4_1_gemv() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::Q4_1Gemv { k: 2048, n: 2048 };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_multi_warp_attention() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::MultiWarpAttention {
        max_seq_len: 4096,
        head_dim: 64,
        n_heads: 32,
        n_kv_heads: 4,
        num_warps_per_head: 4,
        indirect: false,
    };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_kv_cache_scatter() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::KvCacheScatter {
        num_kv_heads: 8,
        head_dim: 128,
        max_len: 4096,
    };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_kv_cache_scatter_indirect() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::KvCacheScatterIndirect {
        num_kv_heads: 8,
        head_dim: 128,
        max_len: 4096,
    };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_vectorized_rms_norm() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::VectorizedRmsNorm {
        hidden_size: 4096,
        epsilon: 1e-5,
    };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_batched_vectorized_rms_norm() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::BatchedVectorizedRmsNorm {
        hidden_size: 4096,
        batch_size: 8,
        epsilon: 1e-5,
    };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_batched_rope() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::BatchedRope {
        num_heads: 32,
        head_dim: 64,
        batch_size: 8,
        theta: 10000.0,
    };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_batched_residual_add() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::BatchedResidualAdd {
        n: 4096,
        batch_size: 8,
    };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_batched_swiglu() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::BatchedSwiglu {
        n: 11008,
        batch_size: 8,
    };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_residual_add() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::ResidualAdd { n: 4096 };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_fused_residual_rms_norm() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::FusedResidualRmsNorm {
        hidden_size: 4096,
        epsilon: 1e-5,
    };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_fused_rms_norm_q4k_gemv() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::FusedRmsNormQ4KGemv {
        k: 4096,
        n: 4096,
        epsilon: 1e-5,
    };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_fused_gate_up_q4k_gemv() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::FusedGateUpQ4KGemv { k: 4096, n: 11008 };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_elementwise_mul() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::ElementwiseMul { n: 4096 };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_fused_swiglu() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::FusedSwiglu { n: 11008 };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_fused_qkv() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::FusedQKV {
        hidden_size: 4096,
        kv_dim: 512,
    };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_fused_gate_up() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::FusedGateUp {
        hidden_size: 4096,
        intermediate_size: 11008,
    };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_rope_indirect() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::RopeIndirect {
        num_heads: 32,
        head_dim: 64,
        theta: 10000.0,
    };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_rope_neox() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::RopeNeox {
        num_heads: 32,
        head_dim: 64,
        theta: 10000.0,
    };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_rope_neox_indirect() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::RopeNeoxIndirect {
        num_heads: 32,
        head_dim: 64,
        theta: 10000.0,
    };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_true_dp4a_q4k_gemv() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::TrueDp4aQ4KGemv { k: 2560, n: 2560 };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_tensor_core_q4k_gemm() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::TensorCoreQ4KGemm {
        m: 16,
        k: 2560,
        n: 2560,
    };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_batched_q4k_gemv() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::BatchedQ4KGemv {
        m: 8,
        k: 2560,
        n: 2560,
    };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_multi_warp_batched_q4k_gemv() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::MultiWarpBatchedQ4KGemv {
        k: 2560,
        n: 2560,
        warps: 4,
    };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_q8_quantize() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::Q8Quantize { n: 4096 };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_q4k_q8_dot() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::Q4KQ8Dot { k: 2560, n: 2560 };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_packed_dp4a_q4k_q8() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::PackedDp4aQ4KQ8 { k: 2560, n: 2560 };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_argmax() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::ArgMax { length: 32000 };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_argmax_final() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::ArgMaxFinal { num_blocks: 128 };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

#[test]
fn test_ptx_fused_q4_q8_dot() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::FusedQ4Q8Dot { n: 2560 };
    let ptx = kernels.generate_ptx(&kernel);
    assert!(ptx.contains(".version"));
}

// ============================================================================
// Kernel Name Tests
// ============================================================================

#[test]
fn test_kernel_name_gemv() {
    let kernels = CudaKernels::new();
    assert_eq!(
        kernels.kernel_name(&KernelType::Gemv { k: 1, n: 1 }),
        "gemv_warp_reduce"
    );
}

#[test]
fn test_kernel_name_coalesced_gemv() {
    let kernels = CudaKernels::new();
    assert_eq!(
        kernels.kernel_name(&KernelType::CoalescedGemv { k: 1, n: 1 }),
        "gemv_coalesced"
    );
}

#[test]
fn test_kernel_name_incremental_attention_direct() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::IncrementalAttention {
        max_seq_len: 4096,
        head_dim: 64,
        n_heads: 32,
        n_kv_heads: 4,
        indirect: false,
    };
    assert_eq!(kernels.kernel_name(&kernel), "incremental_attention");
}

#[test]
fn test_kernel_name_incremental_attention_indirect() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::IncrementalAttention {
        max_seq_len: 4096,
        head_dim: 64,
        n_heads: 32,
        n_kv_heads: 4,
        indirect: true,
    };
    assert_eq!(
        kernels.kernel_name(&kernel),
        "incremental_attention_indirect"
    );
}

#[test]
fn test_kernel_name_multi_warp_attention_direct() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::MultiWarpAttention {
        max_seq_len: 4096,
        head_dim: 64,
        n_heads: 32,
        n_kv_heads: 4,
        num_warps_per_head: 4,
        indirect: false,
    };
    assert_eq!(kernels.kernel_name(&kernel), "multi_warp_attention");
}

#[test]
fn test_kernel_name_multi_warp_attention_indirect() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::MultiWarpAttention {
        max_seq_len: 4096,
        head_dim: 64,
        n_heads: 32,
        n_kv_heads: 4,
        num_warps_per_head: 4,
        indirect: true,
    };
    assert_eq!(
        kernels.kernel_name(&kernel),
        "multi_warp_attention_indirect"
    );
}

#[test]
fn test_kernel_name_attention_non_causal() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::Attention {
        seq_len: 512,
        head_dim: 64,
        causal: false,
    };
    assert_eq!(kernels.kernel_name(&kernel), "flash_attention");
}

#[test]
fn test_kernel_name_attention_tensor_core_non_causal() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::AttentionTensorCore {
        seq_len: 512,
        head_dim: 64,
        n_heads: 8,
        causal: false,
    };
    assert_eq!(kernels.kernel_name(&kernel), "flash_attention_tensor_core");
}

#[test]
fn test_kernel_name_attention_tensor_core_causal() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::AttentionTensorCore {
        seq_len: 512,
        head_dim: 64,
        n_heads: 8,
        causal: true,
    };
    assert_eq!(
        kernels.kernel_name(&kernel),
        "flash_attention_tensor_core_causal"
    );
}

// ============================================================================
// SizeClass Tests
// ============================================================================

#[test]
fn test_size_class_constants() {
    assert_eq!(SizeClass::CLASSES.len(), 9);
    assert_eq!(SizeClass::CLASSES[0], 4096);
    assert_eq!(SizeClass::CLASSES[8], 268_435_456);
}

#[test]
fn test_size_class_for_exact_match_first() {
    let class = SizeClass::for_size(4096);
    assert_eq!(class.map(|c| c.bytes()), Some(4096));
}

#[test]
fn test_size_class_for_size_between_classes() {
    let class = SizeClass::for_size(5000);
    assert_eq!(class.map(|c| c.bytes()), Some(16384));
}

#[test]
fn test_size_class_for_zero() {
    let class = SizeClass::for_size(0);
    assert_eq!(class.map(|c| c.bytes()), Some(4096));
}

#[test]
fn test_size_class_for_very_small() {
    let class = SizeClass::for_size(1);
    assert_eq!(class.map(|c| c.bytes()), Some(4096));
}

// ============================================================================
// GpuMemoryPool Tests
// ============================================================================

#[test]
fn test_gpu_memory_pool_default() {
    let pool = GpuMemoryPool::default();
    assert_eq!(pool.stats().total_allocated, 0);
}

#[test]
fn test_gpu_memory_pool_with_max_size() {
    let pool = GpuMemoryPool::with_max_size(1024 * 1024 * 1024);
    assert_eq!(pool.max_size(), 1024 * 1024 * 1024);
}

#[test]
fn test_gpu_memory_pool_try_get_empty() {
    // try_get returns None on empty pool and increments miss counter
    let mut pool = GpuMemoryPool::new();

    assert!(pool.try_get(4096).is_none());
    assert_eq!(pool.stats().pool_misses, 1);
    assert_eq!(pool.stats().pool_hits, 0);

    // Multiple misses accumulate
    assert!(pool.try_get(16384).is_none());
    assert!(pool.try_get(65536).is_none());
    assert_eq!(pool.stats().pool_misses, 3);
}

#[test]
fn test_gpu_memory_pool_deallocation_saturating() {
    let mut pool = GpuMemoryPool::new();
    pool.record_allocation(1000);
    pool.record_deallocation(2000); // More than allocated
    assert_eq!(pool.stats().total_allocated, 0); // Should saturate at 0
}

// ============================================================================
// GpuMemoryPool Additional Tests
// ============================================================================

#[test]
fn test_gpu_memory_pool_clear() {
    let mut pool = GpuMemoryPool::new();
    pool.record_allocation(1024 * 1024);

    // Clear should not affect allocation tracking (just free buffers)
    pool.clear();
    assert_eq!(pool.stats().free_buffers, 0);
}

#[test]
fn test_gpu_memory_pool_has_capacity() {
    let mut pool = GpuMemoryPool::with_max_size(1000);

    // Initially has full capacity
    assert!(pool.has_capacity(500));
    assert!(pool.has_capacity(1000));
    assert!(!pool.has_capacity(1001));

    // After recording allocation
    pool.record_allocation(600);
    assert!(pool.has_capacity(400));
    assert!(!pool.has_capacity(401));
}

#[test]
fn test_gpu_memory_pool_stats_hit_rate_zero_operations() {
    let pool = GpuMemoryPool::new();
    let stats = pool.stats();

    // No operations should give 0.0 hit rate
    assert_eq!(stats.hit_rate, 0.0);
}

#[test]
fn test_gpu_memory_pool_try_get_too_large() {
    let mut pool = GpuMemoryPool::new();

    // Size larger than max size class (268_435_456) should return None
    let result = pool.try_get(500_000_000);
    assert!(result.is_none());
}

// ============================================================================
// PoolStats Tests
// ============================================================================

#[test]
fn test_pool_stats_estimated_savings_zero_hits() {
    let stats = PoolStats {
        total_allocated: 0,
        peak_usage: 0,
        pool_hits: 0,
        pool_misses: 10,
        hit_rate: 0.0,
        free_buffers: 0,
    };
    assert_eq!(stats.estimated_savings_bytes(), 0);
}

#[test]
fn test_pool_stats_clone() {
    let stats = PoolStats {
        total_allocated: 1000,
        peak_usage: 2000,
        pool_hits: 10,
        pool_misses: 5,
        hit_rate: 0.667,
        free_buffers: 3,
    };
    #[allow(clippy::redundant_clone)]
    let cloned = stats.clone();
    // Verify clone has same values
    assert_eq!(cloned.total_allocated, stats.total_allocated);
    assert_eq!(cloned.pool_hits, stats.pool_hits);
}

// ============================================================================
// PinnedHostBuffer Tests
// ============================================================================

#[test]
fn test_pinned_host_buffer_empty() {
    let buf: PinnedHostBuffer<f32> = PinnedHostBuffer::new(0);
    assert!(buf.is_empty());
    assert_eq!(buf.len(), 0);
    assert_eq!(buf.size_bytes(), 0);
}

#[test]
fn test_pinned_host_buffer_large() {
    let buf: PinnedHostBuffer<f32> = PinnedHostBuffer::new(1_000_000);
    assert_eq!(buf.len(), 1_000_000);
    assert_eq!(buf.size_bytes(), 4_000_000);
}

#[test]
fn test_pinned_host_buffer_i32() {
    let buf: PinnedHostBuffer<i32> = PinnedHostBuffer::new(100);
    assert_eq!(buf.len(), 100);
    assert_eq!(buf.size_bytes(), 400);
}

#[test]
fn test_pinned_host_buffer_u8() {
    let buf: PinnedHostBuffer<u8> = PinnedHostBuffer::new(256);
    assert_eq!(buf.len(), 256);
    assert_eq!(buf.size_bytes(), 256);
}

#[test]
fn test_pinned_host_buffer_default_values() {
    let buf: PinnedHostBuffer<f32> = PinnedHostBuffer::new(10);
    for &val in buf.as_slice() {
        assert_eq!(val, 0.0);
    }
}

// ============================================================================
// StagingBufferPool Tests
// ============================================================================

#[test]
fn test_staging_buffer_pool_default() {
    let pool = StagingBufferPool::default();
    let stats = pool.stats();
    assert_eq!(stats.total_allocated, 0);
    assert_eq!(stats.pool_hits, 0);
}

#[test]
fn test_staging_buffer_pool_with_max_size() {
    let pool = StagingBufferPool::with_max_size(256 * 1024 * 1024);
    let stats = pool.stats();
    assert_eq!(stats.total_allocated, 0);
}

#[test]
fn test_staging_buffer_pool_clear() {
    let mut pool = StagingBufferPool::new();
    let buf = pool.get(1024);
    pool.put(buf);
    assert!(pool.stats().free_buffers > 0);
    pool.clear();
    assert_eq!(pool.stats().free_buffers, 0);
    assert_eq!(pool.stats().total_allocated, 0);
}

// ============================================================================
// StagingPoolStats Tests
// ============================================================================

#[test]
fn test_staging_pool_stats_clone() {
    let stats = StagingPoolStats {
        total_allocated: 1000,
        peak_usage: 2000,
        pool_hits: 5,
        pool_misses: 3,
        free_buffers: 2,
        hit_rate: 0.625,
    };
    #[allow(clippy::redundant_clone)]
    let cloned = stats.clone();
    // Verify clone has same values
    assert_eq!(cloned.total_allocated, stats.total_allocated);
    assert_eq!(cloned.hit_rate, stats.hit_rate);
}

// ============================================================================
// TransferMode Tests
// ============================================================================

#[test]
fn test_transfer_mode_requires_pinned() {
    assert!(!TransferMode::Pageable.requires_pinned());
    assert!(TransferMode::Pinned.requires_pinned());
    assert!(TransferMode::ZeroCopy.requires_pinned());
    assert!(TransferMode::Async.requires_pinned());
}

#[test]
fn test_transfer_mode_estimated_speedup() {
    assert_eq!(TransferMode::Pageable.estimated_speedup(), 1.0);
    assert!(TransferMode::Pinned.estimated_speedup() > 1.0);
    assert!(TransferMode::ZeroCopy.estimated_speedup() > TransferMode::Pinned.estimated_speedup());
    assert!(TransferMode::Async.estimated_speedup() > 1.0);
}

#[test]
fn test_transfer_mode_default() {
    let mode = TransferMode::default();
    assert_eq!(mode, TransferMode::Pageable);
}

#[test]
fn test_transfer_mode_eq() {
    assert_eq!(TransferMode::Pinned, TransferMode::Pinned);
    assert_ne!(TransferMode::Pinned, TransferMode::Pageable);
}

#[test]
fn test_transfer_mode_copy() {
    let mode = TransferMode::Async;
    let copied = mode;
    assert_eq!(copied, TransferMode::Async);
}

// ============================================================================
// WeightQuantType Tests
// ============================================================================

#[test]
fn test_weight_quant_type_explicit_construction() {
    // PMAT-232: WeightQuantType has no Default — must be explicit
    let qtype = WeightQuantType::Q4K;
    assert_eq!(qtype, WeightQuantType::Q4K);
}

#[test]
fn test_weight_quant_type_bytes_per_superblock() {
    assert_eq!(WeightQuantType::Q4K.bytes_per_superblock(), 144);
    assert_eq!(WeightQuantType::Q5K.bytes_per_superblock(), 176);
    assert_eq!(WeightQuantType::Q6K.bytes_per_superblock(), 210);
    assert_eq!(WeightQuantType::Q8_0.bytes_per_superblock(), 34 * 8);
    assert_eq!(WeightQuantType::Q5_0.bytes_per_superblock(), 22 * 8);
    assert_eq!(WeightQuantType::Q4_0.bytes_per_superblock(), 18 * 8);
    assert_eq!(WeightQuantType::Q4_1.bytes_per_superblock(), 20 * 8);
}

#[test]
fn test_weight_quant_type_bytes_per_block() {
    assert_eq!(WeightQuantType::Q4K.bytes_per_block(), 18);
    assert_eq!(WeightQuantType::Q5K.bytes_per_block(), 22);
    assert_eq!(WeightQuantType::Q6K.bytes_per_block(), 26);
    assert_eq!(WeightQuantType::Q8_0.bytes_per_block(), 34);
    assert_eq!(WeightQuantType::Q5_0.bytes_per_block(), 22);
    assert_eq!(WeightQuantType::Q4_0.bytes_per_block(), 18);
    assert_eq!(WeightQuantType::Q4_1.bytes_per_block(), 20);
}

#[test]
fn test_weight_quant_type_from_ggml_type() {
    assert_eq!(
        WeightQuantType::from_ggml_type(2),
        Some(WeightQuantType::Q4_0)
    );
    assert_eq!(
        WeightQuantType::from_ggml_type(3),
        Some(WeightQuantType::Q4_1)
    );
    assert_eq!(
        WeightQuantType::from_ggml_type(6),
        Some(WeightQuantType::Q5_0)
    );
    assert_eq!(
        WeightQuantType::from_ggml_type(8),
        Some(WeightQuantType::Q8_0)
    );
    assert_eq!(
        WeightQuantType::from_ggml_type(12),
        Some(WeightQuantType::Q4K)
    );
    assert_eq!(
        WeightQuantType::from_ggml_type(13),
        Some(WeightQuantType::Q5K)
    );
    assert_eq!(
        WeightQuantType::from_ggml_type(14),
        Some(WeightQuantType::Q6K)
    );
    assert_eq!(WeightQuantType::from_ggml_type(99), None);
}

#[test]
fn test_weight_quant_type_matches_size_q4k() {
    // Q4K: 144 bytes per 256 elements
    // For 1024 rows, 256 cols: 1024 * 1 superblocks = 1024 * 144 = 147456 bytes
    assert!(WeightQuantType::Q4K.matches_size(147456, 1024, 256));
    assert!(!WeightQuantType::Q4K.matches_size(147456 + 1, 1024, 256));
}

#[test]
fn test_weight_quant_type_matches_size_q4_0() {
    // Q4_0: 18 bytes per 32 elements
    // For 1024 rows, 32 cols: 1024 * 1 blocks = 1024 * 18 = 18432 bytes
    assert!(WeightQuantType::Q4_0.matches_size(18432, 1024, 32));
}

#[test]
fn test_weight_quant_type_from_size_q4k() {
    // Q4K: 144 bytes per 256 elements
    let qtype = WeightQuantType::from_size(144 * 10, 10, 256);
    assert_eq!(qtype, Some(WeightQuantType::Q4K));
}

#[test]
fn test_weight_quant_type_from_size_q6k() {
    // Q6K: 210 bytes per 256 elements (checked first due to CORRECTNESS-002)
    let qtype = WeightQuantType::from_size(210 * 10, 10, 256);
    assert_eq!(qtype, Some(WeightQuantType::Q6K));
}

#[test]
fn test_weight_quant_type_from_size_q5k() {
    // Q5K: 176 bytes per 256 elements
    let qtype = WeightQuantType::from_size(176 * 10, 10, 256);
    assert_eq!(qtype, Some(WeightQuantType::Q5K));
}

#[test]
fn test_weight_quant_type_from_size_q4_0() {
    // Q4_0: 18 bytes per 32 elements
    let qtype = WeightQuantType::from_size(18 * 32, 32, 32);
    assert_eq!(qtype, Some(WeightQuantType::Q4_0));
}

#[test]
fn test_weight_quant_type_from_size_unknown() {
    let qtype = WeightQuantType::from_size(12345, 10, 256);
    assert_eq!(qtype, None);
}

#[test]
fn test_weight_quant_type_eq() {
    assert_eq!(WeightQuantType::Q4K, WeightQuantType::Q4K);
    assert_ne!(WeightQuantType::Q4K, WeightQuantType::Q6K);
}

// ============================================================================
// IndexedLayerWeights Tests
// ============================================================================

/// Test helper: creates IndexedLayerWeights with all-zero pointers/sizes and Q4K qtypes.
/// PMAT-232: No Default — every field explicit.
fn test_indexed_layer_weights_zeros() -> IndexedLayerWeights {
    IndexedLayerWeights {
        attn_q_ptr: 0,
        attn_q_len: 0,
        attn_q_qtype: WeightQuantType::Q4K,
        attn_k_ptr: 0,
        attn_k_len: 0,
        attn_k_qtype: WeightQuantType::Q4K,
        attn_v_ptr: 0,
        attn_v_len: 0,
        attn_v_qtype: WeightQuantType::Q4K,
        attn_output_ptr: 0,
        attn_output_len: 0,
        attn_output_qtype: WeightQuantType::Q4K,
        ffn_gate_ptr: 0,
        ffn_gate_len: 0,
        ffn_gate_qtype: WeightQuantType::Q4K,
        ffn_up_ptr: 0,
        ffn_up_len: 0,
        ffn_up_qtype: WeightQuantType::Q4K,
        ffn_down_ptr: 0,
        ffn_down_len: 0,
        ffn_down_qtype: WeightQuantType::Q4K,
        attn_norm_ptr: 0,
        attn_norm_len: 0,
        ffn_norm_ptr: 0,
        ffn_norm_len: 0,
        attn_q_bias_ptr: 0,
        attn_q_bias_len: 0,
        attn_k_bias_ptr: 0,
        attn_k_bias_len: 0,
        attn_v_bias_ptr: 0,
        attn_v_bias_len: 0,
        attn_q_norm_ptr: 0,
        attn_q_norm_len: 0,
        attn_k_norm_ptr: 0,
        attn_k_norm_len: 0,
    }
}

#[test]
fn test_indexed_layer_weights_explicit_construction() {
    // PMAT-232: No Default — every field must be explicit
    let weights = test_indexed_layer_weights_zeros();
    assert_eq!(weights.attn_q_ptr, 0);
    assert_eq!(weights.attn_q_len, 0);
    assert_eq!(weights.attn_q_qtype, WeightQuantType::Q4K);
}

#[test]
fn test_indexed_layer_weights_construction() {
    let weights = IndexedLayerWeights {
        attn_q_ptr: 0x1000,
        attn_q_len: 4096,
        attn_q_qtype: WeightQuantType::Q5_0,
        attn_k_ptr: 0x2000,
        attn_k_len: 512,
        attn_k_qtype: WeightQuantType::Q5_0,
        attn_v_ptr: 0x3000,
        attn_v_len: 512,
        attn_v_qtype: WeightQuantType::Q6K,
        attn_output_ptr: 0x4000,
        attn_output_len: 4096,
        attn_output_qtype: WeightQuantType::Q4K,
        ffn_gate_ptr: 0x5000,
        ffn_gate_len: 11008,
        ffn_gate_qtype: WeightQuantType::Q4K,
        ffn_up_ptr: 0x6000,
        ffn_up_len: 11008,
        ffn_up_qtype: WeightQuantType::Q4K,
        ffn_down_ptr: 0x7000,
        ffn_down_len: 4096,
        ffn_down_qtype: WeightQuantType::Q6K,
        attn_norm_ptr: 0x8000,
        attn_norm_len: 4096,
        ffn_norm_ptr: 0x9000,
        ffn_norm_len: 4096,
        attn_q_bias_ptr: 0,
        attn_q_bias_len: 0,
        attn_k_bias_ptr: 0,
        attn_k_bias_len: 0,
        attn_v_bias_ptr: 0,
        attn_v_bias_len: 0,
        attn_q_norm_ptr: 0,
        attn_q_norm_len: 0,
        attn_k_norm_ptr: 0,
        attn_k_norm_len: 0,
    };

    assert_eq!(weights.attn_q_ptr, 0x1000);
    assert_eq!(weights.attn_q_qtype, WeightQuantType::Q5_0);
    assert_eq!(weights.ffn_down_qtype, WeightQuantType::Q6K);
}

#[test]
fn test_indexed_layer_weights_clone() {
    let weights = test_indexed_layer_weights_zeros();
    #[allow(clippy::redundant_clone)]
    let cloned = weights.clone();
    // Verify clone has same values as original
    assert_eq!(cloned.attn_q_ptr, weights.attn_q_ptr);
}

// ============================================================================
// TransformerWorkspace Tests
// ============================================================================

#[test]
fn test_transformer_workspace_default() {
    let workspace = TransformerWorkspace::default();
    assert!(!workspace.initialized);
    assert_eq!(workspace.hidden_dim, 0);
    assert_eq!(workspace.batch_size, 0);
    assert!(workspace.hidden_buf1.is_none());
}

// ============================================================================
// Presets Module Tests
// ============================================================================

#[test]
fn test_presets_llama_attention() {
    let kernel = presets::llama_attention(2048, 64);
    match kernel {
        KernelType::Attention {
            seq_len,
            head_dim,
            causal,
        } => {
            assert_eq!(seq_len, 2048);
            assert_eq!(head_dim, 64);
            assert!(causal);
        },
        _ => panic!("Expected Attention"),
    }
}

#[test]
fn test_presets_ffn_gemm() {
    let kernel = presets::ffn_gemm(1, 4096, 11008);
    match kernel {
        KernelType::GemmTiled { m, n, k, tile_size } => {
            assert_eq!(m, 1);
            assert_eq!(n, 11008);
            assert_eq!(k, 4096);
            assert_eq!(tile_size, 32);
        },
        _ => panic!("Expected GemmTiled"),
    }
}

#[test]
fn test_presets_q4k_inference() {
    let kernel = presets::q4k_inference(1, 4096, 4096);
    match kernel {
        KernelType::QuantizedGemm { m, n, k } => {
            assert_eq!(m, 1);
            assert_eq!(n, 4096);
            assert_eq!(k, 4096);
        },
        _ => panic!("Expected QuantizedGemm"),
    }
}

#[test]
fn test_presets_q4k_ggml_inference() {
    let kernel = presets::q4k_ggml_inference(1, 2560, 2560);
    match kernel {
        KernelType::QuantizedGemmGgml { m, n, k } => {
            assert_eq!(m, 1);
            assert_eq!(n, 2560);
            assert_eq!(k, 2560);
        },
        _ => panic!("Expected QuantizedGemmGgml"),
    }
}

#[test]
fn test_presets_rmsnorm() {
    let kernel = presets::rmsnorm(4096);
    match kernel {
        KernelType::LayerNorm {
            hidden_size,
            epsilon,
            affine,
        } => {
            assert_eq!(hidden_size, 4096);
            assert!((epsilon - 1e-6).abs() < 1e-10);
            assert!(!affine);
        },
        _ => panic!("Expected LayerNorm"),
    }
}

#[test]
fn test_presets_multi_head_attention() {
    let kernel = presets::multi_head_attention(1024, 64, 32);
    match kernel {
        KernelType::MultiHeadAttention {
            seq_len,
            head_dim,
            n_heads,
            causal,
        } => {
            assert_eq!(seq_len, 1024);
            assert_eq!(head_dim, 64);
            assert_eq!(n_heads, 32);
            assert!(causal);
        },
        _ => panic!("Expected MultiHeadAttention"),
    }
}

#[test]
fn test_presets_phi2_multi_head_attention() {
    let kernel = presets::phi2_multi_head_attention(2048);
    match kernel {
        KernelType::MultiHeadAttention {
            seq_len,
            head_dim,
            n_heads,
            causal,
        } => {
            assert_eq!(seq_len, 2048);
            assert_eq!(head_dim, 80);
            assert_eq!(n_heads, 32);
            assert!(causal);
        },
        _ => panic!("Expected MultiHeadAttention"),
    }
}

#[test]
fn test_presets_tensor_core_attention() {
    let kernel = presets::tensor_core_attention(1024, 64, 16);
    match kernel {
        KernelType::AttentionTensorCore {
            seq_len,
            head_dim,
            n_heads,
            causal,
        } => {
            assert_eq!(seq_len, 1024);
            assert_eq!(head_dim, 64);
            assert_eq!(n_heads, 16);
            assert!(causal);
        },
        _ => panic!("Expected AttentionTensorCore"),
    }
}

#[test]
fn test_presets_llama_tensor_core_attention() {
    let kernel = presets::llama_tensor_core_attention(4096);
    match kernel {
        KernelType::AttentionTensorCore {
            seq_len,
            head_dim,
            n_heads,
            causal,
        } => {
            assert_eq!(seq_len, 4096);
            assert_eq!(head_dim, 128);
            assert_eq!(n_heads, 32);
            assert!(causal);
        },
        _ => panic!("Expected AttentionTensorCore"),
    }
}

// ============================================================================
// CudaKernels::cuda_likely_available Tests
// ============================================================================

#[test]
fn test_cuda_likely_available_returns_bool() {
    // This should not panic regardless of system configuration
    let _available = CudaKernels::cuda_likely_available();
}

// ============================================================================
// Debug and Clone Trait Tests
// ============================================================================

#[test]
fn test_kernel_type_debug() {
    let kernel = KernelType::Softmax { dim: 128 };
    let debug_str = format!("{:?}", kernel);
    assert!(debug_str.contains("Softmax"));
    assert!(debug_str.contains("128"));
}

#[test]
fn test_kernel_type_clone() {
    let kernel = KernelType::GemmNaive { m: 4, n: 4, k: 4 };
    #[allow(clippy::redundant_clone)]
    let cloned = kernel.clone();
    // Verify cloned value matches original by using both
    match (&kernel, &cloned) {
        (
            KernelType::GemmNaive {
                m: m1,
                n: n1,
                k: k1,
            },
            KernelType::GemmNaive {
                m: m2,
                n: n2,
                k: k2,
            },
        ) => {
            assert_eq!(m1, m2);
            assert_eq!(n1, n2);
            assert_eq!(k1, k2);
        },
        _ => panic!("Clone failed"),
    }
}

#[test]
fn test_gpu_memory_pool_debug() {
    let pool = GpuMemoryPool::new();
    let debug_str = format!("{:?}", pool);
    assert!(debug_str.contains("GpuMemoryPool"));
}

#[test]
fn test_pool_stats_debug() {
    let stats = PoolStats {
        total_allocated: 1000,
        peak_usage: 2000,
        pool_hits: 10,
        pool_misses: 5,
        hit_rate: 0.667,
        free_buffers: 3,
    };
    let debug_str = format!("{:?}", stats);
    assert!(debug_str.contains("PoolStats"));
}

#[test]
fn test_pinned_host_buffer_debug() {
    let buf: PinnedHostBuffer<f32> = PinnedHostBuffer::new(10);
    let debug_str = format!("{:?}", buf);
    assert!(debug_str.contains("PinnedHostBuffer"));
}

#[test]
fn test_staging_buffer_pool_debug() {
    let pool = StagingBufferPool::new();
    let debug_str = format!("{:?}", pool);
    assert!(debug_str.contains("StagingBufferPool"));
}

#[test]
fn test_transfer_mode_debug() {
    let mode = TransferMode::ZeroCopy;
    let debug_str = format!("{:?}", mode);
    assert!(debug_str.contains("ZeroCopy"));
}

#[test]
fn test_weight_quant_type_debug() {
    let qtype = WeightQuantType::Q6K;
    let debug_str = format!("{:?}", qtype);
    assert!(debug_str.contains("Q6K"));
}

#[test]
fn test_indexed_layer_weights_debug() {
    let weights = test_indexed_layer_weights_zeros();
    let debug_str = format!("{:?}", weights);
    assert!(debug_str.contains("IndexedLayerWeights"));
}
