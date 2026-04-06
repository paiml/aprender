use super::*;
use crate::cuda::memory::{GpuBufferHandle, SizeClass, TransferMode};
use crate::cuda::pipeline::{
    presets, BankConflictStrategy, MemoryPattern, PtxOptimizationHints, PtxOptimizer,
    RegisterTiling,
};
use serial_test::serial;

/// Helper to create zeroed `ValidatedLayerWeights` for tests.
/// PMAT-232: `Default` was intentionally removed from `IndexedLayerWeights`
/// to enforce explicit construction from GGUF metadata in production code.
/// Tests that only need a dummy/zeroed struct use this helper instead.
/// Uses `new_unchecked` to bypass validation (these are negative tests).
fn test_zeroed_layer_weights() -> ValidatedLayerWeights {
    ValidatedLayerWeights::new_unchecked(IndexedLayerWeights {
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
    })
}

#[test]
fn test_cuda_kernels_creation() {
    let kernels = CudaKernels::new();
    // Verify the struct was created (ZST is valid)
    let _ = kernels.generate_ptx(&KernelType::Softmax { dim: 128 });
}

#[test]
fn test_gemm_naive_ptx_generation() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::GemmNaive {
        m: 128,
        n: 128,
        k: 128,
    });

    assert!(ptx.contains(".version"));
    assert!(ptx.contains(".visible .entry"));
    assert!(ptx.contains("gemm"));
}

#[test]
fn test_gemm_tiled_ptx_generation() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::GemmTiled {
        m: 1024,
        n: 1024,
        k: 1024,
        tile_size: 32,
    });

    assert!(ptx.contains(".version"));
    assert!(ptx.contains("gemm"));
    assert!(ptx.contains(".shared"));
}

#[test]
fn test_softmax_ptx_generation() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Softmax { dim: 4096 });

    assert!(ptx.contains(".version"));
    assert!(ptx.contains("softmax"));
    assert!(ptx.contains("shfl")); // Warp shuffle
}

#[test]
fn test_layernorm_ptx_generation() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::LayerNorm {
        hidden_size: 4096,
        epsilon: 1e-5,
        affine: true,
    });

    assert!(ptx.contains(".version"));
    assert!(ptx.contains("layernorm"));
}

#[test]
fn test_attention_ptx_generation() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Attention {
        seq_len: 2048,
        head_dim: 64,
        causal: true,
    });

    assert!(ptx.contains(".version"));
    assert!(ptx.contains("flash_attention") || ptx.contains("attention"));
}

#[test]
fn test_quantized_gemm_ptx_generation() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::QuantizedGemm {
        m: 1,
        n: 4096,
        k: 4096,
    });

    assert!(ptx.contains(".version"));
    assert!(ptx.contains("q4k") || ptx.contains("gemm"));
}

// =========================================================================
// PARITY-041: GGML Q4_K Super-Block Format Tests
// =========================================================================

#[test]
fn test_parity041_ggml_kernel_ptx_generation() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::QuantizedGemmGgml {
        m: 1,
        n: 4096,
        k: 4096,
    });

    // Verify PTX is generated
    assert!(
        ptx.contains(".version"),
        "PTX should have version directive"
    );
    assert!(
        ptx.contains("q4k_gemm_ggml"),
        "PTX should contain GGML kernel name"
    );
}

#[test]
fn test_parity041_ggml_kernel_name() {
    let kernels = CudaKernels::new();
    let name = kernels.kernel_name(&KernelType::QuantizedGemmGgml {
        m: 1,
        n: 4096,
        k: 4096,
    });
    assert_eq!(name, "q4k_gemm_ggml");
}

#[test]
fn test_parity041_ggml_preset() {
    let kernel = presets::q4k_ggml_inference(1, 4096, 4096);
    match kernel {
        KernelType::QuantizedGemmGgml { m, n, k } => {
            assert_eq!(m, 1);
            assert_eq!(n, 4096);
            assert_eq!(k, 4096);
        },
        _ => panic!("Expected QuantizedGemmGgml"),
    }
}

#[test]
fn test_parity041_ggml_vs_simplified_different_kernels() {
    let kernels = CudaKernels::new();

    let simplified = kernels.generate_ptx(&KernelType::QuantizedGemm {
        m: 1,
        n: 2560,
        k: 2560,
    });

    let ggml = kernels.generate_ptx(&KernelType::QuantizedGemmGgml {
        m: 1,
        n: 2560,
        k: 2560,
    });

    // Both should be valid PTX but different kernel names
    assert!(simplified.contains("q4k_gemm_fused"));
    assert!(ggml.contains("q4k_gemm_ggml"));

    // GGML kernel should be different (super-block format)
    assert_ne!(simplified.len(), ggml.len());
}

#[test]
fn test_parity041_ggml_phi2_dimensions() {
    // phi-2 model dimensions: hidden=2560, intermediate=10240
    let kernels = CudaKernels::new();

    // FFN up projection: [batch, 2560] @ [2560, 10240]
    let up_proj = kernels.generate_ptx(&KernelType::QuantizedGemmGgml {
        m: 1,
        n: 10240,
        k: 2560,
    });
    assert!(up_proj.contains(".version"));

    // FFN down projection: [batch, 10240] @ [10240, 2560]
    let down_proj = kernels.generate_ptx(&KernelType::QuantizedGemmGgml {
        m: 1,
        n: 2560,
        k: 10240,
    });
    assert!(down_proj.contains(".version"));
}

#[test]
fn test_parity041_ggml_super_block_alignment() {
    // k must be divisible by 256 for super-blocks (256 values per super-block)
    let kernels = CudaKernels::new();

    // k=4096 is divisible by 256 (16 super-blocks)
    let ptx = kernels.generate_ptx(&KernelType::QuantizedGemmGgml {
        m: 32,
        n: 2560,
        k: 4096,
    });
    assert!(ptx.contains(".version"));

    // k=2560 is divisible by 256 (10 super-blocks)
    let ptx2 = kernels.generate_ptx(&KernelType::QuantizedGemmGgml {
        m: 1,
        n: 4096,
        k: 2560,
    });
    assert!(ptx2.contains(".version"));
}

// =========================================================================
// PARITY-042: Pinned Memory Tests
// =========================================================================

#[test]
fn test_parity042_pinned_host_buffer_creation() {
    let buf: PinnedHostBuffer<f32> = PinnedHostBuffer::new(1024);
    assert_eq!(buf.len(), 1024);
    assert_eq!(buf.size_bytes(), 1024 * 4);
    assert!(!buf.is_empty());
    // Note: is_pinned() returns false until trueno-gpu adds cuMemAllocHost
    // This is expected behavior for the fallback implementation
}

#[test]
fn test_parity042_pinned_buffer_copy() {
    let mut buf: PinnedHostBuffer<f32> = PinnedHostBuffer::new(100);
    let src: Vec<f32> = (0..100).map(|i| i as f32).collect();
    buf.copy_from_slice(&src);

    let slice = buf.as_slice();
    assert_eq!(slice[0], 0.0);
    assert_eq!(slice[50], 50.0);
    assert_eq!(slice[99], 99.0);
}

#[test]
fn test_parity042_pinned_buffer_mutable() {
    let mut buf: PinnedHostBuffer<f32> = PinnedHostBuffer::new(10);
    let slice = buf.as_mut_slice();
    slice[0] = 42.0;
    slice[9] = 99.0;

    assert_eq!(buf.as_slice()[0], 42.0);
    assert_eq!(buf.as_slice()[9], 99.0);
}

#[test]
fn test_parity042_staging_buffer_pool_basic() {
    let mut pool = StagingBufferPool::new();

    // First allocation - should be a miss
    let buf1 = pool.get(1024);
    assert!(buf1.len() >= 1024);

    let stats = pool.stats();
    assert_eq!(stats.pool_misses, 1);
    assert_eq!(stats.pool_hits, 0);

    // Return to pool
    pool.put(buf1);

    // Second allocation - should be a hit (same size class)
    let buf2 = pool.get(1024);
    let stats = pool.stats();
    assert_eq!(stats.pool_hits, 1);
    assert!(buf2.len() >= 1024);
}

#[test]
fn test_parity042_staging_pool_hit_rate() {
    let mut pool = StagingBufferPool::new();

    // Allocate and return several buffers
    for _ in 0..5 {
        let buf = pool.get(2048);
        pool.put(buf);
    }

    // Now get again - should all be hits
    for _ in 0..5 {
        let buf = pool.get(2048);
        pool.put(buf);
    }

    let stats = pool.stats();
    assert!(
        stats.hit_rate > 0.4,
        "Hit rate should be > 40%: {:.2}",
        stats.hit_rate
    );
}

#[test]
fn test_parity042_staging_pool_clear() {
    let mut pool = StagingBufferPool::new();

    // Allocate some buffers
    let buf1 = pool.get(1024);
    let buf2 = pool.get(2048);
    pool.put(buf1);
    pool.put(buf2);

    assert!(pool.stats().free_buffers > 0);

    // Clear pool
    pool.clear();
    assert_eq!(pool.stats().free_buffers, 0);
}

#[test]
fn test_parity042_transfer_mode_properties() {
    assert!(!TransferMode::Pageable.requires_pinned());
    assert!(TransferMode::Pinned.requires_pinned());
    assert!(TransferMode::ZeroCopy.requires_pinned());
    assert!(TransferMode::Async.requires_pinned());

    assert_eq!(TransferMode::Pageable.estimated_speedup(), 1.0);
    assert!(TransferMode::Pinned.estimated_speedup() > 1.0);
    assert!(TransferMode::ZeroCopy.estimated_speedup() > TransferMode::Pinned.estimated_speedup());
}

#[test]
fn test_parity042_transfer_mode_default() {
    let mode = TransferMode::default();
    assert_eq!(mode, TransferMode::Pageable);
}

// PARITY-043: Multi-Head Attention Parallelization Tests

#[test]
fn test_parity043_multi_head_attention_kernel_type() {
    let kernels = CudaKernels::new();

    // Non-causal variant - now uses trueno's FlashAttention kernel
    let kernel = KernelType::MultiHeadAttention {
        seq_len: 512,
        head_dim: 64,
        n_heads: 32,
        causal: false,
    };
    assert_eq!(kernels.kernel_name(&kernel), "flash_attention");

    // Causal variant (for autoregressive models)
    let causal_kernel = KernelType::MultiHeadAttention {
        seq_len: 512,
        head_dim: 64,
        n_heads: 32,
        causal: true,
    };
    assert_eq!(
        kernels.kernel_name(&causal_kernel),
        "flash_attention_causal"
    );
}

#[test]
fn test_parity043_multi_head_attention_ptx_generation() {
    let kernels = CudaKernels::new();

    let kernel = KernelType::MultiHeadAttention {
        seq_len: 128,
        head_dim: 64,
        n_heads: 8,
        causal: false,
    };

    let ptx = kernels.generate_ptx(&kernel);

    // Verify PTX structure (now using trueno's FlashAttention kernel)
    assert!(ptx.contains(".version 8.0"));
    assert!(ptx.contains(".target sm_70"));
    assert!(ptx.contains(".visible .entry flash_attention"));
    // trueno uses lowercase ptr names
    assert!(ptx.contains(".param .u64 q_ptr"));
    assert!(ptx.contains(".param .u64 k_ptr"));
    assert!(ptx.contains(".param .u64 v_ptr"));
    assert!(ptx.contains(".param .u64 o_ptr"));
    assert!(ptx.contains(".param .u32 seq_len"));
    assert!(ptx.contains(".param .u32 head_dim"));
    assert!(ptx.contains(".param .u32 num_heads"));

    // Verify shared memory (trueno uses .b8 smem array)
    assert!(ptx.contains(".shared"));

    // Verify block indices are used for head/tile selection
    assert!(ptx.contains("%ctaid.x")); // Q tile block
    assert!(ptx.contains("%ctaid.y")); // head index
}

#[test]
fn test_parity043_multi_head_attention_causal_ptx() {
    let kernels = CudaKernels::new();

    let kernel = KernelType::MultiHeadAttention {
        seq_len: 128,
        head_dim: 64,
        n_heads: 8,
        causal: true,
    };

    let ptx = kernels.generate_ptx(&kernel);

    // Verify causal kernel name (trueno uses flash_attention_causal)
    assert!(ptx.contains(".visible .entry flash_attention_causal"));

    // Trueno's causal masking uses setp.lt comparison for Q vs KV block
    // The causal skip happens in the kv_loop via branch
    assert!(ptx.contains("setp.lt.u32")); // Causal comparison
    assert!(ptx.contains("kv_loop")); // KV block loop with causal skip
}

include!("tests_multi_head_attention.rs");
include!("tests_cuda_vs_wgpu.rs");
include!("tests_gemm_fused.rs");
include!("tests_cov001_q6k.rs");
include!("tests_cov001_weight.rs");
