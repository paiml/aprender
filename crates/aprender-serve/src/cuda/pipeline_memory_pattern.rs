
// ============================================================================
// Tests (PMAT-802: T-COV-95)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // MemoryPattern Tests
    // =========================================================================

    #[test]
    fn test_memory_pattern_default() {
        let pattern = MemoryPattern::default();
        assert_eq!(pattern, MemoryPattern::Scalar);
    }

    #[test]
    fn test_memory_pattern_eq() {
        assert_eq!(MemoryPattern::Scalar, MemoryPattern::Scalar);
        assert_eq!(MemoryPattern::Vector2, MemoryPattern::Vector2);
        assert_eq!(MemoryPattern::Vector4, MemoryPattern::Vector4);
        assert_ne!(MemoryPattern::Scalar, MemoryPattern::Vector2);
    }

    #[test]
    fn test_memory_pattern_clone_copy() {
        let pattern = MemoryPattern::Vector4;
        let cloned = pattern;
        assert_eq!(cloned, MemoryPattern::Vector4);
    }

    #[test]
    fn test_memory_pattern_debug() {
        let debug_str = format!("{:?}", MemoryPattern::Vector2);
        assert!(debug_str.contains("Vector2"));
    }

    // =========================================================================
    // RegisterTiling Tests
    // =========================================================================

    #[test]
    fn test_register_tiling_default() {
        let tiling = RegisterTiling::default();
        assert_eq!(tiling.width, 4);
        assert_eq!(tiling.height, 4);
    }

    #[test]
    fn test_register_tiling_large() {
        let tiling = RegisterTiling::large();
        assert_eq!(tiling.width, 8);
        assert_eq!(tiling.height, 8);
    }

    #[test]
    fn test_register_tiling_medium() {
        let tiling = RegisterTiling::medium();
        assert_eq!(tiling.width, 4);
        assert_eq!(tiling.height, 4);
    }

    #[test]
    fn test_register_tiling_small() {
        let tiling = RegisterTiling::small();
        assert_eq!(tiling.width, 2);
        assert_eq!(tiling.height, 2);
    }

    #[test]
    fn test_register_tiling_registers_needed() {
        assert_eq!(RegisterTiling::small().registers_needed(), 4);
        assert_eq!(RegisterTiling::medium().registers_needed(), 16);
        assert_eq!(RegisterTiling::large().registers_needed(), 64);
    }

    #[test]
    fn test_register_tiling_clone_copy() {
        let tiling = RegisterTiling {
            width: 3,
            height: 5,
        };
        let cloned = tiling;
        assert_eq!(cloned.width, 3);
        assert_eq!(cloned.height, 5);
    }

    #[test]
    fn test_register_tiling_eq() {
        let a = RegisterTiling {
            width: 4,
            height: 4,
        };
        let b = RegisterTiling::medium();
        assert_eq!(a, b);
    }

    #[test]
    fn test_register_tiling_debug() {
        let debug_str = format!("{:?}", RegisterTiling::large());
        assert!(debug_str.contains("8"));
    }

    // =========================================================================
    // BankConflictStrategy Tests
    // =========================================================================

    #[test]
    fn test_bank_conflict_strategy_default() {
        let strategy = BankConflictStrategy::default();
        assert_eq!(strategy, BankConflictStrategy::None);
    }

    #[test]
    fn test_bank_conflict_strategy_eq() {
        assert_eq!(BankConflictStrategy::None, BankConflictStrategy::None);
        assert_eq!(BankConflictStrategy::Padding, BankConflictStrategy::Padding);
        assert_eq!(BankConflictStrategy::Xor, BankConflictStrategy::Xor);
        assert_ne!(BankConflictStrategy::None, BankConflictStrategy::Padding);
    }

    #[test]
    fn test_bank_conflict_strategy_clone_copy() {
        let strategy = BankConflictStrategy::Xor;
        let cloned = strategy;
        assert_eq!(cloned, BankConflictStrategy::Xor);
    }

    #[test]
    fn test_bank_conflict_strategy_debug() {
        let debug_str = format!("{:?}", BankConflictStrategy::Padding);
        assert!(debug_str.contains("Padding"));
    }

    // =========================================================================
    // PtxOptimizationHints Tests
    // =========================================================================

    #[test]
    fn test_ptx_hints_default() {
        let hints = PtxOptimizationHints::default();
        assert_eq!(hints.memory_pattern, MemoryPattern::Scalar);
        assert_eq!(hints.register_tiling, RegisterTiling::default());
        assert_eq!(hints.bank_conflict_strategy, BankConflictStrategy::None);
        assert!((hints.target_occupancy - 0.0).abs() < f32::EPSILON);
        assert!(!hints.enable_ilp);
        assert_eq!(hints.shared_mem_preference, 0);
    }

    #[test]
    fn test_ptx_hints_max_throughput() {
        let hints = PtxOptimizationHints::max_throughput();
        assert_eq!(hints.memory_pattern, MemoryPattern::Vector4);
        assert_eq!(hints.register_tiling, RegisterTiling::large());
        assert_eq!(hints.bank_conflict_strategy, BankConflictStrategy::Padding);
        assert!((hints.target_occupancy - 0.75).abs() < 0.001);
        assert!(hints.enable_ilp);
    }

    #[test]
    fn test_ptx_hints_low_latency() {
        let hints = PtxOptimizationHints::low_latency();
        assert_eq!(hints.memory_pattern, MemoryPattern::Scalar);
        assert_eq!(hints.register_tiling, RegisterTiling::small());
        assert_eq!(hints.bank_conflict_strategy, BankConflictStrategy::None);
        assert!((hints.target_occupancy - 1.0).abs() < 0.001);
        assert!(!hints.enable_ilp);
    }

    #[test]
    fn test_ptx_hints_balanced() {
        let hints = PtxOptimizationHints::balanced();
        assert_eq!(hints.memory_pattern, MemoryPattern::Vector2);
        assert_eq!(hints.register_tiling, RegisterTiling::medium());
        assert!((hints.target_occupancy - 0.5).abs() < 0.001);
        assert!(hints.enable_ilp);
    }

    #[test]
    fn test_ptx_hints_uses_vectorized_loads() {
        assert!(!PtxOptimizationHints::low_latency().uses_vectorized_loads());
        assert!(PtxOptimizationHints::balanced().uses_vectorized_loads());
        assert!(PtxOptimizationHints::max_throughput().uses_vectorized_loads());
    }

    #[test]
    fn test_ptx_hints_vector_width() {
        let scalar = PtxOptimizationHints {
            memory_pattern: MemoryPattern::Scalar,
            ..Default::default()
        };
        let vec2 = PtxOptimizationHints {
            memory_pattern: MemoryPattern::Vector2,
            ..Default::default()
        };
        let vec4 = PtxOptimizationHints {
            memory_pattern: MemoryPattern::Vector4,
            ..Default::default()
        };

        assert_eq!(scalar.vector_width(), 1);
        assert_eq!(vec2.vector_width(), 2);
        assert_eq!(vec4.vector_width(), 4);
    }

    #[test]
    fn test_ptx_hints_shared_mem_padding() {
        let no_padding = PtxOptimizationHints {
            bank_conflict_strategy: BankConflictStrategy::None,
            ..Default::default()
        };
        let with_padding = PtxOptimizationHints {
            bank_conflict_strategy: BankConflictStrategy::Padding,
            ..Default::default()
        };
        let xor = PtxOptimizationHints {
            bank_conflict_strategy: BankConflictStrategy::Xor,
            ..Default::default()
        };

        assert_eq!(no_padding.shared_mem_padding(), 0);
        assert_eq!(with_padding.shared_mem_padding(), 1);
        assert_eq!(xor.shared_mem_padding(), 0);
    }

    #[test]
    fn test_ptx_hints_clone() {
        let hints = PtxOptimizationHints::max_throughput();
        let cloned = hints.clone();
        assert_eq!(cloned.memory_pattern, MemoryPattern::Vector4);
    }

    #[test]
    fn test_ptx_hints_debug() {
        let hints = PtxOptimizationHints::balanced();
        let debug_str = format!("{:?}", hints);
        assert!(debug_str.contains("PtxOptimizationHints"));
    }

    // =========================================================================
    // PtxOptimizer Tests
    // =========================================================================

    #[test]
    fn test_ptx_optimizer_new() {
        let hints = PtxOptimizationHints::max_throughput();
        let optimizer = PtxOptimizer::new(hints);
        assert_eq!(optimizer.hints().memory_pattern, MemoryPattern::Vector4);
    }

    #[test]
    fn test_ptx_optimizer_summary() {
        let optimizer = PtxOptimizer::new(PtxOptimizationHints::max_throughput());
        let summary = optimizer.summary();
        assert!(summary.contains("vec=4"));
        assert!(summary.contains("tile=8x8"));
        assert!(summary.contains("Padding"));
        assert!(summary.contains("ilp=true"));
    }

    #[test]
    fn test_ptx_optimizer_padded_shared_mem_row() {
        let optimizer_padding = PtxOptimizer::new(PtxOptimizationHints {
            bank_conflict_strategy: BankConflictStrategy::Padding,
            ..Default::default()
        });
        let optimizer_none = PtxOptimizer::new(PtxOptimizationHints::default());

        assert_eq!(optimizer_padding.padded_shared_mem_row(32), 33);
        assert_eq!(optimizer_none.padded_shared_mem_row(32), 32);
    }

    #[test]
    fn test_ptx_optimizer_estimated_registers() {
        let small = PtxOptimizer::new(PtxOptimizationHints::low_latency());
        let large = PtxOptimizer::new(PtxOptimizationHints::max_throughput());

        // Low latency: base(16) + small tile(4) + no ILP(0) = 20
        assert_eq!(small.estimated_registers(), 20);
        // Max throughput: base(16) + large tile(64) + ILP(64) = 144
        assert_eq!(large.estimated_registers(), 144);
    }

    #[test]
    fn test_ptx_optimizer_is_high_register_pressure() {
        let low = PtxOptimizer::new(PtxOptimizationHints::low_latency());
        let high = PtxOptimizer::new(PtxOptimizationHints::max_throughput());

        assert!(!low.is_high_register_pressure()); // 20 <= 64
        assert!(high.is_high_register_pressure()); // 144 > 64
    }

    // =========================================================================
    // Presets Tests
    // =========================================================================

    #[test]
    fn test_preset_llama_attention() {
        let kernel = presets::llama_attention(512, 128);
        match kernel {
            KernelType::Attention {
                seq_len,
                head_dim,
                causal,
            } => {
                assert_eq!(seq_len, 512);
                assert_eq!(head_dim, 128);
                assert!(causal);
            },
            _ => panic!("Expected Attention kernel"),
        }
    }

    #[test]
    fn test_preset_ffn_gemm() {
        let kernel = presets::ffn_gemm(1, 4096, 11008);
        match kernel {
            KernelType::GemmTiled { m, n, k, tile_size } => {
                assert_eq!(m, 1);
                assert_eq!(n, 11008);
                assert_eq!(k, 4096);
                assert_eq!(tile_size, 32);
            },
            _ => panic!("Expected GemmTiled kernel"),
        }
    }

    #[test]
    fn test_preset_q4k_inference() {
        let kernel = presets::q4k_inference(4, 4096, 4096);
        match kernel {
            KernelType::QuantizedGemm { m, n, k } => {
                assert_eq!(m, 4);
                assert_eq!(n, 4096);
                assert_eq!(k, 4096);
            },
            _ => panic!("Expected QuantizedGemm kernel"),
        }
    }

    #[test]
    fn test_preset_q4k_ggml_inference() {
        let kernel = presets::q4k_ggml_inference(1, 4096, 256);
        match kernel {
            KernelType::QuantizedGemmGgml { m, n, k } => {
                assert_eq!(m, 1);
                assert_eq!(n, 4096);
                assert_eq!(k, 256);
            },
            _ => panic!("Expected QuantizedGemmGgml kernel"),
        }
    }

    #[test]
    fn test_preset_rmsnorm() {
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
            _ => panic!("Expected LayerNorm kernel"),
        }
    }

    #[test]
    fn test_preset_multi_head_attention() {
        let kernel = presets::multi_head_attention(1024, 64, 16);
        match kernel {
            KernelType::MultiHeadAttention {
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
            _ => panic!("Expected MultiHeadAttention kernel"),
        }
    }

    #[test]
    fn test_preset_phi2_multi_head_attention() {
        let kernel = presets::phi2_multi_head_attention(512);
        match kernel {
            KernelType::MultiHeadAttention {
                seq_len,
                head_dim,
                n_heads,
                causal,
            } => {
                assert_eq!(seq_len, 512);
                assert_eq!(head_dim, 80); // phi-2 specific
                assert_eq!(n_heads, 32); // phi-2 specific
                assert!(causal);
            },
            _ => panic!("Expected MultiHeadAttention kernel"),
        }
    }

    #[test]
    fn test_preset_tensor_core_attention() {
        let kernel = presets::tensor_core_attention(256, 128, 32);
        match kernel {
            KernelType::AttentionTensorCore {
                seq_len,
                head_dim,
                n_heads,
                causal,
            } => {
                assert_eq!(seq_len, 256);
                assert_eq!(head_dim, 128);
                assert_eq!(n_heads, 32);
                assert!(causal);
            },
            _ => panic!("Expected AttentionTensorCore kernel"),
        }
    }

    #[test]
    fn test_preset_llama_tensor_core_attention() {
        let kernel = presets::llama_tensor_core_attention(2048);
        match kernel {
            KernelType::AttentionTensorCore {
                seq_len,
                head_dim,
                n_heads,
                causal,
            } => {
                assert_eq!(seq_len, 2048);
                assert_eq!(head_dim, 128); // Llama specific
                assert_eq!(n_heads, 32); // Llama specific
                assert!(causal);
            },
            _ => panic!("Expected AttentionTensorCore kernel"),
        }
    }
}
