
#[test]
fn test_cov001_weight_quant_type_detection() {
    // Test from_ggml_type
    assert!(matches!(
        WeightQuantType::from_ggml_type(12),
        Some(WeightQuantType::Q4K)
    ));
    assert!(matches!(
        WeightQuantType::from_ggml_type(13),
        Some(WeightQuantType::Q5K)
    ));
    assert!(matches!(
        WeightQuantType::from_ggml_type(14),
        Some(WeightQuantType::Q6K)
    ));
    assert!(matches!(
        WeightQuantType::from_ggml_type(8),
        Some(WeightQuantType::Q8_0)
    ));
    assert!(matches!(
        WeightQuantType::from_ggml_type(6),
        Some(WeightQuantType::Q5_0)
    ));
    assert!(matches!(
        WeightQuantType::from_ggml_type(2),
        Some(WeightQuantType::Q4_0)
    ));
    assert!(matches!(
        WeightQuantType::from_ggml_type(3),
        Some(WeightQuantType::Q4_1)
    ));
    assert!(WeightQuantType::from_ggml_type(999).is_none());

    // Test bytes_per_superblock
    assert_eq!(WeightQuantType::Q4K.bytes_per_superblock(), 144);
    assert_eq!(WeightQuantType::Q5K.bytes_per_superblock(), 176);
    assert_eq!(WeightQuantType::Q6K.bytes_per_superblock(), 210);

    // Test bytes_per_block
    assert_eq!(WeightQuantType::Q8_0.bytes_per_block(), 34);
    assert_eq!(WeightQuantType::Q4_0.bytes_per_block(), 18);

    // Test matches_size
    let q4k = WeightQuantType::Q4K;
    assert!(q4k.matches_size(144, 1, 256)); // 1 row, 256 cols = 1 superblock
}

#[test]
fn test_cov001_ptx_optimization_hints() {
    let max_throughput = PtxOptimizationHints::max_throughput();
    assert!(max_throughput.uses_vectorized_loads());
    assert_eq!(max_throughput.vector_width(), 4);
    assert_eq!(max_throughput.shared_mem_padding(), 1);

    let low_latency = PtxOptimizationHints::low_latency();
    assert!(!low_latency.uses_vectorized_loads());
    assert_eq!(low_latency.vector_width(), 1);
    assert_eq!(low_latency.shared_mem_padding(), 0);

    let balanced = PtxOptimizationHints::balanced();
    assert!(balanced.uses_vectorized_loads());
    assert_eq!(balanced.vector_width(), 2);
}

#[test]
fn test_cov001_ptx_optimizer() {
    let hints = PtxOptimizationHints::max_throughput();
    let optimizer = PtxOptimizer::new(hints);

    // Test summary generation
    let summary = optimizer.summary();
    assert!(summary.contains("PtxOptimizer"));

    // Test padded row calculation
    assert_eq!(optimizer.padded_shared_mem_row(32), 33);

    // Test register estimation
    let regs = optimizer.estimated_registers();
    assert!(regs > 0);

    // Test high register pressure detection
    let _high_pressure = optimizer.is_high_register_pressure();
}

#[test]
fn test_cov001_register_tiling() {
    let large = RegisterTiling::large();
    assert_eq!(large.registers_needed(), 64);

    let medium = RegisterTiling::medium();
    assert_eq!(medium.registers_needed(), 16);

    let small = RegisterTiling::small();
    assert_eq!(small.registers_needed(), 4);
}

#[test]
fn test_cov001_memory_pattern() {
    let scalar = MemoryPattern::Scalar;
    let vec2 = MemoryPattern::Vector2;
    let vec4 = MemoryPattern::Vector4;

    // Just ensure they can be compared
    assert_ne!(scalar, vec2);
    assert_ne!(vec2, vec4);
}

#[test]
fn test_cov001_bank_conflict_strategy() {
    let none = BankConflictStrategy::None;
    let padding = BankConflictStrategy::Padding;
    let xor = BankConflictStrategy::Xor;

    assert_ne!(none, padding);
    assert_ne!(padding, xor);
}

#[test]
fn test_cov001_presets_coverage() {
    // Test all preset functions
    let _llama_attn = presets::llama_attention(2048, 64);
    let _ffn = presets::ffn_gemm(1, 4096, 11008);
    let _q4k = presets::q4k_inference(1, 4096, 4096);
    let _q4k_ggml = presets::q4k_ggml_inference(1, 4096, 4096);
    let _rmsnorm = presets::rmsnorm(4096);
    let _mha = presets::multi_head_attention(2048, 64, 32);
    let _phi2_mha = presets::phi2_multi_head_attention(2048);
    let _tc_attn = presets::tensor_core_attention(2048, 64, 32);
    let _llama_tc = presets::llama_tensor_core_attention(2048);
}

#[test]
fn test_cov001_kernel_type_kernel_names() {
    let kernels = CudaKernels::new();

    // Test all kernel type names
    let types = [
        KernelType::GemmNaive { m: 1, n: 1, k: 1 },
        KernelType::GemmTiled {
            m: 1,
            n: 1,
            k: 1,
            tile_size: 32,
        },
        KernelType::Softmax { dim: 128 },
        KernelType::LayerNorm {
            hidden_size: 256,
            epsilon: 1e-5,
            affine: true,
        },
        KernelType::Attention {
            seq_len: 16,
            head_dim: 64,
            causal: true,
        },
        KernelType::QuantizedGemm {
            m: 1,
            n: 256,
            k: 256,
        },
        KernelType::QuantizedGemmGgml {
            m: 1,
            n: 256,
            k: 256,
        },
        KernelType::Q4KGemv { k: 256, n: 256 },
        KernelType::Q5KGemv { k: 256, n: 256 },
        KernelType::Q6KGemv { k: 256, n: 256 },
    ];

    for kt in types {
        let name = kernels.kernel_name(&kt);
        assert!(!name.is_empty(), "Kernel name should not be empty");
    }
}

