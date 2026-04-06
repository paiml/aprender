
#[test]
fn test_tcov001s_q4_0_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Q4_0Gemv { k: 4096, n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001t_q4_1_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Q4_1Gemv { k: 4096, n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001u_incremental_attention_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::IncrementalAttention {
        max_seq_len: 2048,
        head_dim: 64,
        n_heads: 32,
        n_kv_heads: 8,
        indirect: false,
    });
    assert!(ptx.contains(".version"));

    // Test with indirect=true
    let ptx_indirect = kernels.generate_ptx(&KernelType::IncrementalAttention {
        max_seq_len: 2048,
        head_dim: 64,
        n_heads: 32,
        n_kv_heads: 8,
        indirect: true,
    });
    assert!(ptx_indirect.contains(".version"));
}

#[test]
fn test_tcov001v_multi_warp_attention_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::MultiWarpAttention {
        max_seq_len: 2048,
        head_dim: 64,
        n_heads: 32,
        n_kv_heads: 8,
        num_warps_per_head: 4,
        indirect: false,
    });
    assert!(ptx.contains(".version"));

    // Test with indirect=true
    let ptx_indirect = kernels.generate_ptx(&KernelType::MultiWarpAttention {
        max_seq_len: 2048,
        head_dim: 64,
        n_heads: 32,
        n_kv_heads: 8,
        num_warps_per_head: 4,
        indirect: true,
    });
    assert!(ptx_indirect.contains(".version"));
}

#[test]
fn test_tcov001w_kv_cache_scatter_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::KvCacheScatter {
        num_kv_heads: 8,
        head_dim: 64,
        max_len: 2048,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001x_kv_cache_scatter_indirect_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::KvCacheScatterIndirect {
        num_kv_heads: 8,
        head_dim: 64,
        max_len: 2048,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001y_rmsnorm_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::RmsNorm {
        hidden_size: 4096,
        epsilon: 1e-6,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001z_vectorized_rmsnorm_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::VectorizedRmsNorm {
        hidden_size: 4096,
        epsilon: 1e-6,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001aa_batched_vectorized_rmsnorm_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::BatchedVectorizedRmsNorm {
        hidden_size: 4096,
        batch_size: 4,
        epsilon: 1e-6,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001ab_precise_rmsnorm_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::PreciseRmsNorm {
        hidden_size: 4096,
        epsilon: 1e-6,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001ac_batched_rope_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::BatchedRope {
        num_heads: 32,
        head_dim: 64,
        batch_size: 4,
        theta: 10000.0,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001ad_batched_residual_add_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::BatchedResidualAdd {
        n: 4096,
        batch_size: 4,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001ae_batched_swiglu_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::BatchedSwiglu {
        n: 4096,
        batch_size: 4,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001af_residual_add_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::ResidualAdd { n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001ag_fused_residual_rmsnorm_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::FusedResidualRmsNorm {
        hidden_size: 4096,
        epsilon: 1e-6,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001ah_fused_rmsnorm_q4k_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::FusedRmsNormQ4KGemv {
        k: 4096,
        n: 4096,
        epsilon: 1e-6,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001ai_fused_gate_up_q4k_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::FusedGateUpQ4KGemv { k: 4096, n: 11008 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001aj_packed_dp4a_q4k_q8_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::PackedDp4aQ4KQ8 { k: 4096, n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001ak_true_dp4a_q4k_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::TrueDp4aQ4KGemv { k: 4096, n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001al_q4kq8_dot_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Q4KQ8Dot { k: 4096, n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001am_q8_quantize_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Q8Quantize { n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001an_batched_q4k_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::BatchedQ4KGemv {
        k: 4096,
        n: 4096,
        m: 4,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001ao_multi_warp_batched_q4k_gemv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::MultiWarpBatchedQ4KGemv {
        k: 4096,
        n: 4096,
        warps: 4,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001ap_tensor_core_q4k_gemm_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::TensorCoreQ4KGemm {
        m: 1,
        n: 4096,
        k: 4096,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001aq_precise_rope_neox_indirect_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::PreciseRopeNeoxIndirect {
        num_heads: 32,
        head_dim: 64,
        theta: 10000.0,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001ar_rope_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Rope {
        num_heads: 32,
        head_dim: 64,
        theta: 10000.0,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001as_rope_indirect_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::RopeIndirect {
        num_heads: 32,
        head_dim: 64,
        theta: 10000.0,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001at_rope_neox_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::RopeNeox {
        num_heads: 32,
        head_dim: 64,
        theta: 10000.0,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001au_rope_neox_indirect_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::RopeNeoxIndirect {
        num_heads: 32,
        head_dim: 64,
        theta: 10000.0,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001av_precise_rope_neox_indirect_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::PreciseRopeNeoxIndirect {
        num_heads: 32,
        head_dim: 64,
        theta: 10000.0,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001aw_silu_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Silu { n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001ax_gelu_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Gelu { n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001ay_elementwise_mul_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::ElementwiseMul { n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001az_fused_swiglu_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::FusedSwiglu { n: 4096 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001ba_fused_qkv_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::FusedQKV {
        hidden_size: 4096,
        kv_dim: 4096,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001bb_fused_gate_up_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::FusedGateUp {
        hidden_size: 4096,
        intermediate_size: 11008,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001bc_argmax_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::ArgMax { length: 32000 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001bd_argmax_final_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::ArgMaxFinal { num_blocks: 128 });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001be_gemm_optimized_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::GemmOptimized {
        m: 64,
        n: 64,
        k: 64,
        tile_size: 32,
        reg_block: 4,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001bf_q5k_quantized_gemm_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Q5KQuantizedGemm {
        m: 1,
        n: 4096,
        k: 4096,
    });
    assert!(ptx.contains(".version"));
}

#[test]
fn test_tcov001bg_q6k_quantized_gemm_ptx() {
    let kernels = CudaKernels::new();
    let ptx = kernels.generate_ptx(&KernelType::Q6KQuantizedGemm {
        m: 1,
        n: 4096,
        k: 4096,
    });
    assert!(ptx.contains(".version"));
}
