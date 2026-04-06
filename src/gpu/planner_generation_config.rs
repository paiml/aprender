
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_config_default() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_tokens, 100);
        assert_eq!(config.vocab_size, 32000);
        assert!(config.use_greedy_path()); // 32000 > 8192
    }

    #[test]
    fn test_generation_config_small_vocab() {
        let config = GenerationConfig {
            vocab_size: 1000,
            ..Default::default()
        };
        assert!(!config.use_greedy_path()); // 1000 < 8192
    }

    #[test]
    fn test_planner_basic_flow() {
        let config = GenerationConfig {
            max_tokens: 3,
            vocab_size: 1000,
            ..Default::default()
        };
        let mut planner = BatchPlanner::new(config);

        // Start with prompt
        let step = planner.start_with_prompt(&[1, 2, 3]);
        assert!(matches!(step, GenerationStep::ProcessPrompt { .. }));

        // Generate first token
        let step = planner.plan_next(Some(100));
        assert!(matches!(step, GenerationStep::GenerateToken { .. }));
        assert!(!planner.is_done());

        // Generate second token
        let step = planner.plan_next(Some(101));
        assert!(matches!(step, GenerationStep::GenerateToken { .. }));

        // Generate third token - should complete
        let step = planner.plan_next(Some(102));
        assert!(matches!(step, GenerationStep::Done { .. }));
        assert!(planner.is_done());

        assert_eq!(planner.tokens(), &[1, 2, 3, 100, 101, 102]);
        assert_eq!(planner.generated_count(), 3);
    }

    #[test]
    fn test_planner_stop_token() {
        let config = GenerationConfig {
            max_tokens: 100,
            stop_token: Some(999),
            ..Default::default()
        };
        let mut planner = BatchPlanner::new(config);

        let _ = planner.start_with_prompt(&[1]);
        let _ = planner.plan_next(Some(50));

        // Hit stop token
        let step = planner.plan_next(Some(999));
        assert!(matches!(step, GenerationStep::Done { .. }));
        assert!(planner.is_done());
    }

    #[test]
    fn test_planner_greedy_optimization() {
        // Large vocab - should use greedy
        let config = GenerationConfig {
            vocab_size: 32000,
            ..Default::default()
        };
        let mut planner = BatchPlanner::new(config);

        let _ = planner.start_with_prompt(&[1]);
        let step = planner.plan_next(Some(100));

        if let GenerationStep::GenerateToken {
            use_greedy_optimization,
            ..
        } = step
        {
            assert!(use_greedy_optimization);
        } else {
            panic!("Expected GenerateToken");
        }

        // Small vocab - should not use greedy
        let config = GenerationConfig {
            vocab_size: 1000,
            ..Default::default()
        };
        let mut planner = BatchPlanner::new(config);

        let _ = planner.start_with_prompt(&[1]);
        let step = planner.plan_next(Some(100));

        if let GenerationStep::GenerateToken {
            use_greedy_optimization,
            ..
        } = step
        {
            assert!(!use_greedy_optimization);
        } else {
            panic!("Expected GenerateToken");
        }
    }

    #[test]
    fn test_block_forward_plan_mha() {
        let config = GpuModelConfig {
            vocab_size: 32000,
            hidden_dim: 768,
            num_heads: 12,
            num_kv_heads: 12, // MHA: same as num_heads
            num_layers: 12,
            intermediate_dim: 3072,
            eps: 1e-5,
            rope_theta: 10000.0,
            explicit_head_dim: None,
            layer_types: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            linear_conv_kernel_dim: None,
            constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
        };

        let plan = BlockForwardPlan::from_config(&config, 0, false);
        assert!(!plan.is_gqa());
        assert_eq!(plan.heads_per_kv, 1);
        assert_eq!(plan.head_dim, 64); // 768 / 12
        assert!(!plan.use_swiglu);
    }

    #[test]
    fn test_block_forward_plan_gqa() {
        let config = GpuModelConfig {
            vocab_size: 32000,
            hidden_dim: 2048,
            num_heads: 32,
            num_kv_heads: 4, // GQA: 32/4 = 8 Q heads per KV head
            num_layers: 22,
            intermediate_dim: 5632,
            eps: 1e-5,
            rope_theta: 10000.0,
            explicit_head_dim: None,
            layer_types: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            linear_conv_kernel_dim: None,
            constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
        };

        let plan = BlockForwardPlan::from_config(&config, 5, true);
        assert!(plan.is_gqa());
        assert_eq!(plan.heads_per_kv, 8);
        assert_eq!(plan.head_dim, 64); // 2048 / 32
        assert!(plan.use_swiglu);
    }

    #[test]
    fn test_plan_sampling_greedy() {
        assert_eq!(plan_sampling(None, None, None), SamplingStrategy::Greedy);
        assert_eq!(
            plan_sampling(Some(1.0), None, None),
            SamplingStrategy::Greedy
        );
        assert_eq!(
            plan_sampling(Some(0.0), None, None),
            SamplingStrategy::Greedy
        );
    }

    #[test]
    fn test_plan_sampling_temperature() {
        assert_eq!(
            plan_sampling(Some(0.7), None, None),
            SamplingStrategy::Temperature { temp: 0.7 }
        );
    }

    #[test]
    fn test_plan_sampling_top_p() {
        assert_eq!(
            plan_sampling(None, None, Some(0.9)),
            SamplingStrategy::TopP { p: 0.9 }
        );
    }

    #[test]
    fn test_plan_sampling_top_k() {
        assert_eq!(
            plan_sampling(None, Some(50), None),
            SamplingStrategy::TopK { k: 50 }
        );
    }

    #[test]
    fn test_plan_sampling_priority() {
        // Temperature takes priority
        assert!(matches!(
            plan_sampling(Some(0.7), Some(50), Some(0.9)),
            SamplingStrategy::Temperature { .. }
        ));

        // Then top_p
        assert!(matches!(
            plan_sampling(None, Some(50), Some(0.9)),
            SamplingStrategy::TopP { .. }
        ));
    }

    #[test]
    fn test_plan_lm_head_path_small_vocab() {
        let path = plan_lm_head_path(1000, 768, 100_000_000);
        assert_eq!(path, LmHeadPath::Gpu);
    }

    #[test]
    fn test_plan_lm_head_path_large_vocab() {
        let path = plan_lm_head_path(32000, 768, 100_000_000);
        assert_eq!(path, LmHeadPath::CpuTransposed);
    }

    #[test]
    fn test_plan_lm_head_path_exceeds_buffer() {
        // Small vocab but exceeds buffer limit
        let path = plan_lm_head_path(5000, 768, 1_000_000);
        assert_eq!(path, LmHeadPath::CpuTransposed);
    }
}
