
// =========================================================================
// T-COV-010: Additional kernel_name branch coverage (verify non-empty)
// =========================================================================

#[test]
fn test_tcov010_more_kernel_names() {
    let kernels = CudaKernels::new();

    // Verify a variety of kernel types return non-empty names
    let kernel_types: Vec<KernelType> = vec![
        KernelType::FusedRmsNormQ4KGemv {
            k: 4096,
            n: 4096,
            epsilon: 1e-6,
        },
        KernelType::FusedGateUpQ4KGemv { k: 4096, n: 11008 },
        KernelType::PackedDp4aQ4KQ8 { k: 4096, n: 4096 },
        KernelType::TrueDp4aQ4KGemv { k: 4096, n: 4096 },
        KernelType::Q4KQ8Dot { k: 4096, n: 4096 },
        KernelType::Q8Quantize { n: 4096 },
        KernelType::BatchedQ4KGemv {
            k: 4096,
            n: 4096,
            m: 4,
        },
        KernelType::MultiWarpBatchedQ4KGemv {
            k: 4096,
            n: 4096,
            warps: 4,
        },
        KernelType::TensorCoreQ4KGemm {
            m: 1,
            n: 4096,
            k: 4096,
        },
        KernelType::Rope {
            num_heads: 32,
            head_dim: 64,
            theta: 10000.0,
        },
        KernelType::RopeIndirect {
            num_heads: 32,
            head_dim: 64,
            theta: 10000.0,
        },
        KernelType::RopeNeox {
            num_heads: 32,
            head_dim: 64,
            theta: 10000.0,
        },
        KernelType::RopeNeoxIndirect {
            num_heads: 32,
            head_dim: 64,
            theta: 10000.0,
        },
        KernelType::PreciseRopeNeoxIndirect {
            num_heads: 32,
            head_dim: 64,
            theta: 10000.0,
        },
        KernelType::Silu { n: 4096 },
        KernelType::Gelu { n: 4096 },
        KernelType::ElementwiseMul { n: 4096 },
        KernelType::FusedSwiglu { n: 4096 },
        KernelType::FusedQKV {
            hidden_size: 4096,
            kv_dim: 4096,
        },
        KernelType::FusedGateUp {
            hidden_size: 4096,
            intermediate_size: 11008,
        },
        KernelType::ArgMax { length: 32000 },
        KernelType::ArgMaxFinal { num_blocks: 128 },
    ];

    for kernel_type in kernel_types {
        let name = kernels.kernel_name(&kernel_type);
        assert!(
            !name.is_empty(),
            "KernelType {:?} should have non-empty name",
            kernel_type
        );
    }
}
