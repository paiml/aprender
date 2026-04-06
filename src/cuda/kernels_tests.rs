use super::*;

// =========================================================================
// KernelType Tests
// =========================================================================

#[test]
fn test_kernel_type_gemm_naive() {
    let kernel = KernelType::GemmNaive {
        m: 64,
        n: 64,
        k: 64,
    };
    let debug_str = format!("{:?}", kernel);
    assert!(debug_str.contains("GemmNaive"));
    assert!(debug_str.contains("64"));
}

#[test]
fn test_kernel_type_gemm_tiled() {
    let kernel = KernelType::GemmTiled {
        m: 128,
        n: 256,
        k: 64,
        tile_size: 32,
    };
    let cloned = kernel.clone();
    if let KernelType::GemmTiled { m, n, k, tile_size } = cloned {
        assert_eq!(m, 128);
        assert_eq!(n, 256);
        assert_eq!(k, 64);
        assert_eq!(tile_size, 32);
    } else {
        panic!("Clone failed");
    }
}

#[test]
fn test_kernel_type_softmax() {
    let kernel = KernelType::Softmax { dim: 4096 };
    let debug_str = format!("{:?}", kernel);
    assert!(debug_str.contains("Softmax"));
    assert!(debug_str.contains("4096"));
}

#[test]
fn test_kernel_type_layer_norm() {
    let kernel = KernelType::LayerNorm {
        hidden_size: 2048,
        epsilon: 1e-6,
        affine: true,
    };
    let cloned = kernel.clone();
    if let KernelType::LayerNorm {
        hidden_size,
        epsilon,
        affine,
    } = cloned
    {
        assert_eq!(hidden_size, 2048);
        assert!((epsilon - 1e-6).abs() < 1e-10);
        assert!(affine);
    } else {
        panic!("Clone failed");
    }
}

#[test]
fn test_kernel_type_attention() {
    let kernel = KernelType::Attention {
        seq_len: 512,
        head_dim: 64,
        causal: true,
    };
    let debug_str = format!("{:?}", kernel);
    assert!(debug_str.contains("Attention"));
    assert!(debug_str.contains("512"));
    assert!(debug_str.contains("true"));
}

#[test]
fn test_kernel_type_multi_head_attention() {
    let kernel = KernelType::MultiHeadAttention {
        seq_len: 1024,
        head_dim: 128,
        n_heads: 32,
        causal: true,
    };
    let cloned = kernel.clone();
    if let KernelType::MultiHeadAttention {
        seq_len,
        head_dim,
        n_heads,
        causal,
    } = cloned
    {
        assert_eq!(seq_len, 1024);
        assert_eq!(head_dim, 128);
        assert_eq!(n_heads, 32);
        assert!(causal);
    } else {
        panic!("Clone failed");
    }
}

#[test]
fn test_kernel_type_gemv() {
    let kernel = KernelType::Gemv { k: 4096, n: 4096 };
    let debug_str = format!("{:?}", kernel);
    assert!(debug_str.contains("Gemv"));
}

#[test]
fn test_kernel_type_coalesced_gemv() {
    let kernel = KernelType::CoalescedGemv { k: 4096, n: 11008 };
    let cloned = kernel.clone();
    if let KernelType::CoalescedGemv { k, n } = cloned {
        assert_eq!(k, 4096);
        assert_eq!(n, 11008);
    } else {
        panic!("Clone failed");
    }
}

#[test]
fn test_kernel_type_quantized_gemm() {
    let kernel = KernelType::QuantizedGemm {
        m: 1,
        n: 4096,
        k: 4096,
    };
    let debug_str = format!("{:?}", kernel);
    assert!(debug_str.contains("QuantizedGemm"));
}

#[test]
fn test_kernel_type_q4k_gemv() {
    let kernel = KernelType::Q4KGemv { k: 4096, n: 4096 };
    let cloned = kernel.clone();
    if let KernelType::Q4KGemv { k, n } = cloned {
        assert_eq!(k, 4096);
        assert_eq!(n, 4096);
    } else {
        panic!("Clone failed");
    }
}

#[test]
fn test_kernel_type_incremental_attention() {
    let kernel = KernelType::IncrementalAttention {
        max_seq_len: 256,
        head_dim: 64,
        n_heads: 32,
        n_kv_heads: 4,
        indirect: false,
    };
    let debug_str = format!("{:?}", kernel);
    assert!(debug_str.contains("IncrementalAttention"));
}

#[test]
fn test_kernel_type_rms_norm() {
    let kernel = KernelType::RmsNorm {
        hidden_size: 4096,
        epsilon: 1e-6,
    };
    let cloned = kernel.clone();
    if let KernelType::RmsNorm {
        hidden_size,
        epsilon,
    } = cloned
    {
        assert_eq!(hidden_size, 4096);
        assert!((epsilon - 1e-6).abs() < 1e-10);
    } else {
        panic!("Clone failed");
    }
}

#[test]
fn test_kernel_type_rope() {
    let kernel = KernelType::Rope {
        num_heads: 32,
        head_dim: 128,
        theta: 10000.0,
    };
    let debug_str = format!("{:?}", kernel);
    assert!(debug_str.contains("Rope"));
    assert!(debug_str.contains("128"));
}

#[test]
fn test_kernel_type_residual_add() {
    let kernel = KernelType::ResidualAdd { n: 4096 };
    let cloned = kernel.clone();
    if let KernelType::ResidualAdd { n } = cloned {
        assert_eq!(n, 4096);
    } else {
        panic!("Clone failed");
    }
}

#[test]
fn test_kernel_type_silu() {
    let kernel = KernelType::Silu { n: 11008 };
    let debug_str = format!("{:?}", kernel);
    assert!(debug_str.contains("Silu"));
}

#[test]
fn test_kernel_type_gelu() {
    let kernel = KernelType::Gelu { n: 4096 };
    let cloned = kernel.clone();
    if let KernelType::Gelu { n } = cloned {
        assert_eq!(n, 4096);
    } else {
        panic!("Clone failed");
    }
}

#[test]
fn test_kernel_type_argmax() {
    let kernel = KernelType::ArgMax { length: 32000 };
    let debug_str = format!("{:?}", kernel);
    assert!(debug_str.contains("ArgMax"));
    assert!(debug_str.contains("32000"));
}

// =========================================================================
// CudaKernels Tests
// =========================================================================

#[test]
fn test_cuda_kernels_new() {
    let kernels = CudaKernels::new();
    // Just verify it creates successfully
    let _ = kernels;
}

#[test]
fn test_cuda_kernels_default() {
    let kernels = CudaKernels::default();
    // Verify default works same as new
    let _ = kernels;
}

#[test]
fn test_cuda_likely_available() {
    // Just verify the function runs without panic
    let result = CudaKernels::cuda_likely_available();
    // On a system with CUDA, this should be true
    // On a system without CUDA, this should be false
    // We can't assert the value, just that it returns a bool
    let _ = result;
}

// =========================================================================
// kernel_name Tests
// =========================================================================

#[test]
fn test_kernel_name_gemm_naive() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::GemmNaive { m: 1, n: 1, k: 1 };
    assert_eq!(kernels.kernel_name(&kernel), "gemm_naive");
}

#[test]
fn test_kernel_name_gemm_tiled() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::GemmTiled {
        m: 1,
        n: 1,
        k: 1,
        tile_size: 32,
    };
    assert_eq!(kernels.kernel_name(&kernel), "gemm_tiled");
}

#[test]
fn test_kernel_name_gemm_tensor_core() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::GemmTensorCore {
        m: 16,
        n: 16,
        k: 16,
    };
    assert_eq!(kernels.kernel_name(&kernel), "gemm_tensor_core");
}

#[test]
fn test_kernel_name_gemv() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::Gemv { k: 4096, n: 4096 };
    assert_eq!(kernels.kernel_name(&kernel), "gemv_warp_reduce");
}

#[test]
fn test_kernel_name_coalesced_gemv() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::CoalescedGemv { k: 4096, n: 4096 };
    assert_eq!(kernels.kernel_name(&kernel), "gemv_coalesced");
}

#[test]
fn test_kernel_name_softmax() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::Softmax { dim: 4096 };
    assert_eq!(kernels.kernel_name(&kernel), "softmax_warp_shuffle");
}

#[test]
fn test_kernel_name_layer_norm() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::LayerNorm {
        hidden_size: 4096,
        epsilon: 1e-5,
        affine: true,
    };
    assert_eq!(kernels.kernel_name(&kernel), "layernorm_warp_shuffle");
}

#[test]
fn test_kernel_name_attention_causal() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::Attention {
        seq_len: 512,
        head_dim: 64,
        causal: true,
    };
    assert_eq!(kernels.kernel_name(&kernel), "flash_attention_causal");
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
fn test_kernel_name_multi_head_attention_causal() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::MultiHeadAttention {
        seq_len: 512,
        head_dim: 64,
        n_heads: 32,
        causal: true,
    };
    assert_eq!(kernels.kernel_name(&kernel), "flash_attention_causal");
}

#[test]
fn test_kernel_name_attention_tensor_core_causal() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::AttentionTensorCore {
        seq_len: 512,
        head_dim: 64,
        n_heads: 32,
        causal: true,
    };
    assert_eq!(
        kernels.kernel_name(&kernel),
        "flash_attention_tensor_core_causal"
    );
}

#[test]
fn test_kernel_name_attention_tensor_core_non_causal() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::AttentionTensorCore {
        seq_len: 512,
        head_dim: 64,
        n_heads: 32,
        causal: false,
    };
    assert_eq!(kernels.kernel_name(&kernel), "flash_attention_tensor_core");
}

#[test]
fn test_kernel_name_quantized_gemm() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::QuantizedGemm {
        m: 1,
        n: 4096,
        k: 4096,
    };
    assert_eq!(kernels.kernel_name(&kernel), "q4k_gemm_fused");
}

#[test]
fn test_kernel_name_quantized_gemm_ggml() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::QuantizedGemmGgml {
        m: 1,
        n: 4096,
        k: 256,
    };
    assert_eq!(kernels.kernel_name(&kernel), "q4k_gemm_ggml");
}

#[test]
fn test_kernel_name_q4k_gemv() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::Q4KGemv { k: 4096, n: 4096 };
    assert_eq!(kernels.kernel_name(&kernel), "q4k_gemv_warp_reduce");
}

#[test]
fn test_kernel_name_q5k_gemv() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::Q5KGemv { k: 4096, n: 4096 };
    assert_eq!(kernels.kernel_name(&kernel), "q5k_gemv_warp_reduce");
}

#[test]
fn test_kernel_name_q6k_gemv() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::Q6KGemv { k: 4096, n: 4096 };
    assert_eq!(kernels.kernel_name(&kernel), "q6k_gemv_warp_reduce");
}

#[test]
fn test_kernel_name_rms_norm() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::RmsNorm {
        hidden_size: 4096,
        epsilon: 1e-6,
    };
    assert_eq!(kernels.kernel_name(&kernel), "rmsnorm");
}

#[test]
fn test_kernel_name_vectorized_rms_norm() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::VectorizedRmsNorm {
        hidden_size: 4096,
        epsilon: 1e-6,
    };
    assert_eq!(kernels.kernel_name(&kernel), "rmsnorm_vectorized");
}

include!("kernels_tests_kernel_name.rs");
