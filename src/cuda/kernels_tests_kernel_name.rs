
#[test]
fn test_kernel_name_residual_add() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::ResidualAdd { n: 4096 };
    assert_eq!(kernels.kernel_name(&kernel), "residual_add");
}

#[test]
fn test_kernel_name_silu() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::Silu { n: 4096 };
    assert_eq!(kernels.kernel_name(&kernel), "silu");
}

#[test]
fn test_kernel_name_gelu() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::Gelu { n: 4096 };
    assert_eq!(kernels.kernel_name(&kernel), "gelu");
}

#[test]
fn test_kernel_name_rope() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::Rope {
        num_heads: 32,
        head_dim: 128,
        theta: 10000.0,
    };
    assert_eq!(kernels.kernel_name(&kernel), "rope");
}

#[test]
fn test_kernel_name_rope_indirect() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::RopeIndirect {
        num_heads: 32,
        head_dim: 128,
        theta: 10000.0,
    };
    assert_eq!(kernels.kernel_name(&kernel), "rope_indirect");
}

#[test]
fn test_kernel_name_argmax() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::ArgMax { length: 32000 };
    assert_eq!(kernels.kernel_name(&kernel), "argmax_block_reduce");
}

#[test]
fn test_kernel_name_argmax_final() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::ArgMaxFinal { num_blocks: 125 };
    assert_eq!(kernels.kernel_name(&kernel), "argmax_final_reduce");
}

#[test]
fn test_kernel_name_incremental_attention_direct() {
    let kernels = CudaKernels::new();
    let kernel = KernelType::IncrementalAttention {
        max_seq_len: 256,
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
        max_seq_len: 256,
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
