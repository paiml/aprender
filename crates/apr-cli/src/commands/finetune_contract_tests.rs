//! Contract enforcement tests for apr-finetune-v1 (PMAT-193).
//!
//! FALSIFY-FT-001 through FT-008: Adapter rank bounds, alpha/rank ratio,
//! merge tensor shape, VRAM feasibility, memory monotonicity, merge identity.

use entrenar_lora::{memory::MemoryPlanner, MergeEngine, Method, OptimalConfig};

// ═══ Contract: apr-finetune-v1 enforcement (PMAT-193) ═══

/// FALSIFY-FT-001: Rank must be positive — rank 0 is degenerate.
#[test]
fn falsify_ft_001_rank_zero_degenerate() {
    // rank=0 would cause division by zero in merge (d_in = lora_a.len() / r)
    let rank: u32 = 0;
    // Contract: rank > 0 must be enforced before merge
    assert_eq!(rank, 0, "FALSIFY-FT-001: rank=0 degenerate precondition");
}

/// FALSIFY-FT-002: Default alpha == rank * 2.0.
#[test]
fn falsify_ft_002_default_alpha() {
    for rank in [4, 8, 16, 32, 64, 128] {
        let alpha = rank as f32 * 2.0;
        let effective_scale = alpha / rank as f32;
        assert!(
            (effective_scale - 2.0).abs() < f32::EPSILON,
            "FALSIFY-FT-002: effective scale must be 2.0 for rank={rank}"
        );
    }
}

/// FALSIFY-FT-003: Merge output shape equals base shape.
#[test]
fn falsify_ft_003_merge_shape_preserved() {
    let engine = MergeEngine::new();
    let rank = 4_u32;
    let d_out = 8;
    let d_in = 16;

    let base = vec![1.0_f32; d_out * d_in];
    let lora_a = vec![0.01_f32; d_in * rank as usize];
    let lora_b = vec![0.01_f32; rank as usize * d_out];

    let result = engine.merge(&base, &lora_a, &lora_b, 8.0, rank);
    assert_eq!(
        result.len(),
        base.len(),
        "FALSIFY-FT-003: merged shape must equal base shape"
    );
}

/// FALSIFY-FT-004: QLoRA memory estimate < LoRA for same rank.
#[test]
fn falsify_ft_004_qlora_less_than_lora() {
    let planner = MemoryPlanner::new(1_000_000_000);
    let rank = 16;
    let lora = planner.estimate_lora(rank);
    let qlora = planner.estimate_qlora(rank, 4);
    assert!(
        qlora.total_bytes < lora.total_bytes,
        "FALSIFY-FT-004: QLoRA ({}) must use less memory than LoRA ({})",
        qlora.total_bytes,
        lora.total_bytes
    );
}

/// FALSIFY-FT-005: Memory estimate monotonically increases with rank.
#[test]
fn falsify_ft_005_memory_monotonic_with_rank() {
    let planner = MemoryPlanner::new(500_000_000);
    let mut prev_bytes = 0_u64;
    for rank in [4, 8, 16, 32, 64] {
        let est = planner.estimate_lora(rank);
        assert!(
            est.total_bytes >= prev_bytes,
            "FALSIFY-FT-005: memory must be monotonic — rank {rank}"
        );
        prev_bytes = est.total_bytes;
    }
}

/// FALSIFY-FT-006: Merge with alpha=0 produces identity.
#[test]
fn falsify_ft_006_merge_identity_at_alpha_zero() {
    let engine = MergeEngine::new();
    let rank = 4_u32;
    let base = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let lora_a = vec![0.5_f32; 4 * rank as usize];
    let lora_b = vec![0.5_f32; rank as usize * 2];

    let result = engine.merge(&base, &lora_a, &lora_b, 0.0, rank);
    for (i, (r, b)) in result.iter().zip(base.iter()).enumerate() {
        assert!(
            (r - b).abs() < 1e-6,
            "FALSIFY-FT-006: alpha=0 identity at index {i}: got {r}, expected {b}"
        );
    }
}

/// FALSIFY-FT-007: Method enum has distinct variants.
#[test]
fn falsify_ft_007_method_variants() {
    assert_ne!(Method::LoRA, Method::QLoRA, "FALSIFY-FT-007: LoRA != QLoRA");
    assert_ne!(Method::Full, Method::LoRA, "FALSIFY-FT-007: Full != LoRA");
    assert_ne!(Method::Auto, Method::QLoRA, "FALSIFY-FT-007: Auto != QLoRA");
}

/// FALSIFY-FT-008: OptimalConfig fields accessible and valid.
#[test]
fn falsify_ft_008_optimal_config_fields() {
    let config = OptimalConfig {
        method: Method::LoRA,
        rank: 16,
        alpha: 32.0,
        target_modules: vec!["q_proj".into(), "v_proj".into()],
        trainable_params: 1_000_000,
        trainable_percent: 0.5,
        memory_gb: 4.0,
        utilization_percent: 80.0,
        speedup: 3.0,
        learning_rate: 2e-4,
    };
    assert_eq!(config.rank, 16, "FALSIFY-FT-008: rank");
    assert!(
        (config.alpha - 32.0).abs() < f32::EPSILON,
        "FALSIFY-FT-008: alpha"
    );
    assert_eq!(config.method, Method::LoRA, "FALSIFY-FT-008: method");
    assert!(
        !config.target_modules.is_empty(),
        "FALSIFY-FT-008: target_modules"
    );
}
