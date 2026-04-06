//! Comprehensive tests for GPU Batch Planner (Phase 47)
//!
//! Tests target uncovered code paths in planner.rs:
//! - `GenerationConfig::from_model`
//! - `BatchPlanner::plan_next` edge cases
//! - `BatchPlanner::config()` accessor
//! - `BlockForwardPlan::attention_output_size()`
//! - `SamplingStrategy::Default`
//! - Edge cases in `plan_sampling`

use super::planner::{
    plan_lm_head_path, plan_sampling, BatchPlanner, BlockForwardPlan, GenerationConfig,
    GenerationStep, LmHeadPath, SamplingStrategy,
};
use super::GpuModelConfig;

// ============================================================================
// GenerationConfig Tests
// ============================================================================

#[test]
fn test_generation_config_from_model() {
    let model_config = GpuModelConfig {
        vocab_size: 32000,
        hidden_dim: 2048,
        num_heads: 32,
        num_kv_heads: 4,
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

    let gen_config = GenerationConfig::from_model(&model_config, 50);
    assert_eq!(gen_config.max_tokens, 50);
    assert_eq!(gen_config.vocab_size, 32000);
    assert_eq!(gen_config.greedy_vocab_threshold, 8192);
    assert!(gen_config.stop_token.is_none());
    assert!(gen_config.use_greedy_path());

    // Small vocab
    let small_config = GpuModelConfig {
        vocab_size: 1000,
        hidden_dim: 512,
        num_heads: 8,
        num_kv_heads: 8,
        num_layers: 6,
        intermediate_dim: 2048,
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
    let small_gen = GenerationConfig::from_model(&small_config, 100);
    assert_eq!(small_gen.vocab_size, 1000);
    assert!(!small_gen.use_greedy_path());
}

#[test]
fn test_generation_config_boundary_vocab_size() {
    // Exactly at threshold
    let at_thresh = GenerationConfig {
        vocab_size: 8192,
        ..Default::default()
    };
    assert!(!at_thresh.use_greedy_path());

    // Above threshold
    let above = GenerationConfig {
        vocab_size: 8193,
        ..Default::default()
    };
    assert!(above.use_greedy_path());
}

#[test]
fn test_generation_config_clone_debug() {
    let config = GenerationConfig {
        max_tokens: 100,
        vocab_size: 32000,
        greedy_vocab_threshold: 8192,
        stop_token: Some(2),
    };
    let cloned = config.clone();
    assert_eq!(cloned.max_tokens, config.max_tokens);
    assert_eq!(cloned.stop_token, config.stop_token);
    assert!(format!("{:?}", config).contains("100"));
}

// ============================================================================
// BatchPlanner Tests
// ============================================================================

#[test]
fn test_planner_initial_state_without_prompt() {
    let config = GenerationConfig::default();
    let mut planner = BatchPlanner::new(config);
    let step = planner.plan_next(None);
    assert!(matches!(step, GenerationStep::Done { tokens } if tokens.is_empty()));
}

#[test]
fn test_planner_plan_next_with_none_token() {
    let config = GenerationConfig {
        max_tokens: 5,
        ..Default::default()
    };
    let mut planner = BatchPlanner::new(config);
    let _ = planner.start_with_prompt(&[1, 2, 3]);

    let step = planner.plan_next(None);
    assert!(matches!(step, GenerationStep::GenerateToken { .. }));
    assert_eq!(planner.tokens(), &[1, 2, 3]);
    assert_eq!(planner.generated_count(), 0);
}

#[test]
fn test_planner_config_accessor() {
    let config = GenerationConfig {
        max_tokens: 42,
        vocab_size: 5000,
        greedy_vocab_threshold: 4000,
        stop_token: Some(123),
    };
    let planner = BatchPlanner::new(config);
    let cfg = planner.config();
    assert_eq!(cfg.max_tokens, 42);
    assert_eq!(cfg.vocab_size, 5000);
    assert_eq!(cfg.stop_token, Some(123));
}

#[test]
fn test_planner_done_state_idempotent() {
    let config = GenerationConfig {
        max_tokens: 1,
        ..Default::default()
    };
    let mut planner = BatchPlanner::new(config);
    let _ = planner.start_with_prompt(&[1]);
    let _ = planner.plan_next(Some(100));
    assert!(planner.is_done());

    let step = planner.plan_next(Some(200));
    assert!(matches!(step, GenerationStep::Done { .. }));
    assert_eq!(planner.tokens(), &[1, 100]);
}

#[test]
fn test_planner_empty_prompt_and_generation() {
    let config = GenerationConfig {
        max_tokens: 2,
        ..Default::default()
    };
    let mut planner = BatchPlanner::new(config);

    let step = planner.start_with_prompt(&[]);
    assert!(matches!(step, GenerationStep::ProcessPrompt { tokens } if tokens.is_empty()));

    let step = planner.plan_next(Some(1));
    assert!(matches!(step, GenerationStep::GenerateToken { .. }));
    assert_eq!(planner.tokens(), &[1]);

    let step = planner.plan_next(Some(2));
    assert!(matches!(step, GenerationStep::Done { .. }));
    assert_eq!(planner.generated_count(), 2);
}

#[test]
fn test_batch_planner_clone_debug() {
    let config = GenerationConfig::default();
    let planner = BatchPlanner::new(config);
    let cloned = planner.clone();
    assert_eq!(cloned.tokens(), planner.tokens());
    assert_eq!(cloned.is_done(), planner.is_done());
    assert!(!format!("{:?}", planner).is_empty());
}

// ============================================================================
// BlockForwardPlan Tests
// ============================================================================

#[test]
fn test_block_forward_plan_attention_output_size() {
    let config = GpuModelConfig {
        vocab_size: 32000,
        hidden_dim: 2048,
        num_heads: 32,
        num_kv_heads: 4,
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
    let plan = BlockForwardPlan::from_config(&config, 0, true);
    assert_eq!(plan.attention_output_size(), 2048);
}

#[test]
fn test_block_forward_plan_all_fields() {
    let config = GpuModelConfig {
        vocab_size: 32000,
        hidden_dim: 4096,
        num_heads: 64,
        num_kv_heads: 8,
        num_layers: 40,
        intermediate_dim: 14336,
        eps: 1e-6,
        rope_theta: 500000.0,
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
    let plan = BlockForwardPlan::from_config(&config, 10, true);

    assert_eq!(plan.block_idx, 10);
    assert_eq!(plan.hidden_dim, 4096);
    assert_eq!(plan.num_heads, 64);
    assert_eq!(plan.head_dim, 64);
    assert_eq!(plan.intermediate_dim, 14336);
    assert!(plan.use_swiglu);
    assert_eq!(plan.heads_per_kv, 8);
    assert!(plan.is_gqa());
    assert_eq!(plan.kv_dim, 512);
    assert_eq!(plan.qkv_dim, 5120);
}

#[test]
fn test_block_forward_plan_mqa() {
    let config = GpuModelConfig {
        vocab_size: 32000,
        hidden_dim: 2048,
        num_heads: 32,
        num_kv_heads: 1,
        num_layers: 24,
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
    let plan = BlockForwardPlan::from_config(&config, 0, false);
    assert!(plan.is_gqa());
    assert_eq!(plan.heads_per_kv, 32);
    assert!(!plan.use_swiglu);
}

#[test]
fn test_block_forward_plan_clone_eq() {
    let config = GpuModelConfig {
        vocab_size: 32000,
        hidden_dim: 768,
        num_heads: 12,
        num_kv_heads: 12,
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
    let p1 = BlockForwardPlan::from_config(&config, 0, true);
    let p2 = BlockForwardPlan::from_config(&config, 0, true);
    let p3 = BlockForwardPlan::from_config(&config, 1, true);

    assert_eq!(p1, p2);
    assert_ne!(p1, p3);
    assert_eq!(p1.clone().block_idx, 0);
    assert!(format!("{:?}", p1).contains("BlockForwardPlan"));
}

// ============================================================================
// SamplingStrategy Tests
// ============================================================================

#[test]
fn test_sampling_strategy_default_and_variants() {
    assert_eq!(SamplingStrategy::default(), SamplingStrategy::Greedy);

    let variants = [
        SamplingStrategy::Greedy,
        SamplingStrategy::TopK { k: 50 },
        SamplingStrategy::TopP { p: 0.9 },
        SamplingStrategy::Temperature { temp: 0.7 },
    ];
    for s in &variants {
        assert_eq!(&s.clone(), s);
        assert!(!format!("{:?}", s).is_empty());
    }
}

// ============================================================================
// plan_sampling Tests
// ============================================================================

#[test]
fn test_plan_sampling_invalid_params() {
    assert_eq!(plan_sampling(None, Some(0), None), SamplingStrategy::Greedy);
    assert_eq!(
        plan_sampling(None, Some(usize::MAX), None),
        SamplingStrategy::Greedy
    );
    assert_eq!(
        plan_sampling(None, None, Some(1.0)),
        SamplingStrategy::Greedy
    );
    assert_eq!(
        plan_sampling(None, None, Some(0.0)),
        SamplingStrategy::Greedy
    );
    assert_eq!(
        plan_sampling(Some(-0.5), None, None),
        SamplingStrategy::Greedy
    );
    assert_eq!(
        plan_sampling(Some(0.0), Some(0), Some(1.0)),
        SamplingStrategy::Greedy
    );
}

#[test]
fn test_plan_sampling_boundary_values() {
    assert_eq!(
        plan_sampling(None, None, Some(0.999)),
        SamplingStrategy::TopP { p: 0.999 }
    );
    assert_eq!(
        plan_sampling(None, None, Some(0.001)),
        SamplingStrategy::TopP { p: 0.001 }
    );
    assert_eq!(
        plan_sampling(None, Some(1), None),
        SamplingStrategy::TopK { k: 1 }
    );
    assert_eq!(
        plan_sampling(None, Some(usize::MAX - 1), None),
        SamplingStrategy::TopK { k: usize::MAX - 1 }
    );
}

// ============================================================================
// plan_lm_head_path Tests
// ============================================================================

#[test]
fn test_plan_lm_head_path_boundaries() {
    // At vocab threshold
    assert_eq!(plan_lm_head_path(8192, 768, 100_000_000), LmHeadPath::Gpu);
    assert_eq!(
        plan_lm_head_path(8193, 768, 100_000_000),
        LmHeadPath::CpuTransposed
    );

    // Buffer limit
    let limit = 5000 * 768;
    assert_eq!(plan_lm_head_path(5000, 768, limit), LmHeadPath::Gpu);
    assert_eq!(
        plan_lm_head_path(5000, 768, limit - 1),
        LmHeadPath::CpuTransposed
    );
}

#[test]
fn test_lm_head_path_variants() {
    for p in &[LmHeadPath::CpuTransposed, LmHeadPath::Gpu] {
        assert_eq!(&p.clone(), p);
        assert!(!format!("{:?}", p).is_empty());
    }
}

// ============================================================================
// GenerationStep Tests
// ============================================================================

#[test]
fn test_generation_step_variants_and_eq() {
    let steps = [
        GenerationStep::ProcessPrompt { tokens: vec![1, 2] },
        GenerationStep::GenerateToken {
            tokens: vec![1],
            use_greedy_optimization: true,
        },
        GenerationStep::Done {
            tokens: vec![1, 2, 3],
        },
    ];
    for s in &steps {
        assert_eq!(&s.clone(), s);
        assert!(!format!("{:?}", s).is_empty());
    }

    let s1 = GenerationStep::ProcessPrompt { tokens: vec![1] };
    let s2 = GenerationStep::ProcessPrompt { tokens: vec![1] };
    let s3 = GenerationStep::ProcessPrompt { tokens: vec![2] };
    assert_eq!(s1, s2);
    assert_ne!(s1, s3);
}
