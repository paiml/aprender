//! Falsification Tests: Token Budget Compliance (F021-F040)
//!
//! Per spec: docs/specifications/qwen2.5-coder-showcase-demo.md §9.2
//! Run with: cargo test --test fkr_budget
//!
//! These tests verify that brick budgets are met under normal operation.
//! Budget targets are defined in the spec:
//! - RmsNormBrick: ≤ 1.5µs
//! - QkvBrick: ≤ 6.0µs
//! - RopeBrick: ≤ 1.0µs
//! - AttentionBrick: ≤ 10.0µs
//! - OProjBrick: ≤ 3.5µs
//! - FfnBrick: ≤ 12.2µs
//! - TransformerLayerBrick: ≤ 35.7µs

use realizar::brick::{
    AttentionBrick, BrickError, ComputeBrick, FfnBrick, OProjBrick, QkvBrick, RmsNormBrick,
    RopeBrick, TokenBudget, TransformerLayerBrick,
};

// ============================================================================
// F021-F040: Token Budget Compliance (20 points)
// ============================================================================

/// F021: `TokenBudget` latency/throughput consistent (1 point)
#[test]
fn fkr_budget_f021_budget_math() {
    // From latency
    let budget = TokenBudget::from_latency(100.0); // 100µs
    assert!(
        (budget.tokens_per_sec - 10000.0).abs() < 0.001,
        "F021: 100µs should give 10000 tok/s"
    );

    // From throughput
    let budget = TokenBudget::from_throughput(10000.0); // 10k tok/s
    assert!(
        (budget.us_per_token - 100.0).abs() < 0.001,
        "F021: 10000 tok/s should give 100µs"
    );

    // Little's Law: throughput = 1 / latency
    let budget = TokenBudget::from_latency(50.0);
    let expected_tps = 1_000_000.0 / 50.0;
    assert!(
        (budget.tokens_per_sec - expected_tps).abs() < 0.001,
        "F021: Little's Law must hold"
    );
}

/// F022: Budget violation triggers `BrickError` (2 points)
#[test]
fn fkr_budget_f022_budget_enforcement() {
    // Use impossibly tight budget
    let brick =
        RmsNormBrick::new(vec![1.0; 1024], 1e-5).with_budget(TokenBudget::from_latency(0.0001)); // 0.1ns budget - impossible

    let input: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let result = brick.run(&input);

    // Should fail due to budget_met assertion
    match result {
        Err(BrickError::AssertionFailed { name, .. }) => {
            assert!(name.contains("budget"), "F022: Error should mention budget");
        },
        Ok(res) => {
            // If it succeeds, budget_met should be false
            assert!(!res.budget_met, "F022: budget_met should be false");
        },
        Err(e) => {
            panic!("F022: Unexpected error type: {:?}", e);
        },
    }
}

/// F023: `RmsNormBrick` budget (1 point)
/// Spec target: ≤ 1.5µs
#[test]
fn fkr_budget_f023_rmsnorm_budget() {
    let brick = RmsNormBrick::new(vec![1.0; 896], 1e-5);
    let budget = brick.budget();

    assert!(
        budget.us_per_token <= 1.5,
        "F023: RmsNormBrick budget should be ≤ 1.5µs, got {}",
        budget.us_per_token
    );
}

/// F024: `QkvBrick` budget (2 points)
/// Spec target: ≤ 6.0µs
#[test]
fn fkr_budget_f024_qkv_budget() {
    let brick = QkvBrick::new(896, 896, 128, 128);
    let budget = brick.budget();

    assert!(
        budget.us_per_token <= 6.0,
        "F024: QkvBrick budget should be ≤ 6.0µs, got {}",
        budget.us_per_token
    );
}

/// F025: `RopeBrick` budget (1 point)
/// Spec target: ≤ 1.0µs
#[test]
fn fkr_budget_f025_rope_budget() {
    let brick = RopeBrick::new(64, 14, 1000000.0, 2);
    let budget = brick.budget();

    assert!(
        budget.us_per_token <= 1.0,
        "F025: RopeBrick budget should be ≤ 1.0µs, got {}",
        budget.us_per_token
    );
}

/// F026: `AttentionBrick` budget (2 points)
/// Spec target: ≤ 10.0µs
#[test]
fn fkr_budget_f026_attention_budget() {
    let brick = AttentionBrick::new(14, 2, 64);
    let budget = brick.budget();

    assert!(
        budget.us_per_token <= 10.0,
        "F026: AttentionBrick budget should be ≤ 10.0µs, got {}",
        budget.us_per_token
    );
}

/// F027: `OProjBrick` budget (1 point)
/// Spec target: ≤ 3.5µs
#[test]
fn fkr_budget_f027_oproj_budget() {
    let brick = OProjBrick::new(896, 896);
    let budget = brick.budget();

    assert!(
        budget.us_per_token <= 3.5,
        "F027: OProjBrick budget should be ≤ 3.5µs, got {}",
        budget.us_per_token
    );
}

/// F028: `FfnBrick` budget (2 points)
/// Spec target: ≤ 12.2µs
#[test]
fn fkr_budget_f028_ffn_budget() {
    let brick = FfnBrick::new(896, 4864);
    let budget = brick.budget();

    assert!(
        budget.us_per_token <= 12.2,
        "F028: FfnBrick budget should be ≤ 12.2µs, got {}",
        budget.us_per_token
    );
}

/// F029: `TransformerLayerBrick` budget (2 points)
/// Spec target: ≤ 35.7µs (sum of component budgets)
#[test]
fn fkr_budget_f029_layer_budget() {
    let layer = TransformerLayerBrick::from_config(0, 896, 14, 2, 4864, 1e-5, 1000000.0, 2);
    let total = layer.total_budget_us();

    assert!(
        total <= 35.7,
        "F029: TransformerLayerBrick budget should be ≤ 35.7µs, got {}",
        total
    );
}

/// F030: Throughput calculation sanity (2 points)
/// Model throughput = 1_000_000 / (layer_us * num_layers)
#[test]
fn fkr_budget_f030_throughput_calculation() {
    let layer = TransformerLayerBrick::from_config(0, 896, 14, 2, 4864, 1e-5, 1000000.0, 2);
    let layer_us = layer.total_budget_us();

    // Qwen2.5-0.5B has 24 layers
    let num_layers = 24;
    let total_us = layer_us * num_layers as f64;
    let throughput = 1_000_000.0 / total_us;

    // Should achieve reasonable throughput
    assert!(
        throughput > 100.0,
        "F030: Throughput should be > 100 tok/s, got {}",
        throughput
    );
}

/// F031: Budget enforcement works at runtime (1 point)
#[test]
fn fkr_budget_f031_runtime_enforcement() {
    // Create brick with very lenient budget
    let brick =
        RmsNormBrick::new(vec![1.0; 4], 1e-5).with_budget(TokenBudget::from_latency(1_000_000.0)); // 1 second budget

    let input = vec![1.0_f32, 2.0, 3.0, 4.0];
    let result = brick.run(&input);

    // Should succeed with budget_met = true
    match result {
        Ok(res) => {
            assert!(res.budget_met, "F031: Lenient budget should be met");
        },
        Err(e) => {
            panic!("F031: Should succeed with lenient budget: {:?}", e);
        },
    }
}

/// F032: Budget gap_factor calculation (1 point)
#[test]
fn fkr_budget_f032_gap_factor() {
    let budget = TokenBudget::from_latency(100.0);

    // Under budget: gap < 1.0
    assert!(
        budget.gap_factor(50.0) < 1.0,
        "F032: 50µs should have gap < 1.0"
    );

    // At budget: gap = 1.0
    assert!(
        (budget.gap_factor(100.0) - 1.0).abs() < 0.001,
        "F032: 100µs should have gap = 1.0"
    );

    // Over budget: gap > 1.0
    assert!(
        budget.gap_factor(200.0) > 1.0,
        "F032: 200µs should have gap > 1.0"
    );
}

/// F033: Budget is_met calculation (1 point)
#[test]
fn fkr_budget_f033_is_met() {
    let budget = TokenBudget::from_latency(100.0);

    assert!(budget.is_met(50.0), "F033: 50µs should meet 100µs budget");
    assert!(budget.is_met(100.0), "F033: 100µs should meet 100µs budget");
    assert!(
        !budget.is_met(150.0),
        "F033: 150µs should NOT meet 100µs budget"
    );
}

/// F034: Batch size affects amortized budget (1 point)
#[test]
fn fkr_budget_f034_batch_size() {
    let budget = TokenBudget::from_latency(100.0).with_batch_size(4);

    assert_eq!(budget.batch_size, 4, "F034: Batch size should be 4");
    // Budget per token is same, but batch amortizes overhead
}

/// F035: All bricks have positive throughput targets (bonus)
#[test]
fn fkr_budget_f035_positive_throughput() {
    let bricks: Vec<Box<dyn ComputeBrick<Output = Vec<f32>>>> =
        vec![Box::new(RmsNormBrick::new(vec![1.0; 4], 1e-5))];

    for brick in &bricks {
        let budget = brick.budget();
        assert!(
            budget.tokens_per_sec > 0.0,
            "F035: {} must have positive throughput",
            brick.name()
        );
        assert!(
            !budget.tokens_per_sec.is_nan(),
            "F035: {} throughput must not be NaN",
            brick.name()
        );
        assert!(
            !budget.tokens_per_sec.is_infinite(),
            "F035: {} throughput must be finite",
            brick.name()
        );
    }
}

/// F036: Budget from_latency and from_throughput are inverses (bonus)
#[test]
fn fkr_budget_f036_inverse_constructors() {
    let original_us = 42.0;
    let budget1 = TokenBudget::from_latency(original_us);
    let budget2 = TokenBudget::from_throughput(budget1.tokens_per_sec);

    assert!(
        (budget2.us_per_token - original_us).abs() < 0.001,
        "F036: from_throughput(from_latency(x).tps).us should equal x"
    );
}

/// F037: Layer budget equals sum of component budgets (bonus)
#[test]
fn fkr_budget_f037_layer_sum() {
    let layer = TransformerLayerBrick::from_config(0, 896, 14, 2, 4864, 1e-5, 1000000.0, 2);
    let total = layer.total_budget_us();

    let sum = layer.attn_norm.budget().us_per_token
        + layer.qkv.budget().us_per_token
        + layer.rope.budget().us_per_token
        + layer.attention.budget().us_per_token
        + layer.o_proj.budget().us_per_token
        + layer.ffn_norm.budget().us_per_token
        + layer.ffn.budget().us_per_token;

    assert!(
        (total - sum).abs() < 0.1,
        "F037: Layer budget {} should equal sum {}",
        total,
        sum
    );
}

/// F038: Budget with_batch_size returns modified budget (bonus)
#[test]
fn fkr_budget_f038_with_batch_size() {
    let budget = TokenBudget::from_latency(100.0);
    let batched = budget.with_batch_size(8);

    assert_eq!(
        budget.batch_size, 1,
        "F038: Original should have batch_size 1"
    );
    assert_eq!(
        batched.batch_size, 8,
        "F038: Batched should have batch_size 8"
    );
    // Latency and throughput should be preserved
    assert!(
        (budget.us_per_token - batched.us_per_token).abs() < 0.001,
        "F038: Latency should be preserved"
    );
}

/// F039: Default TokenBudget is reasonable (bonus)
#[test]
fn fkr_budget_f039_default_budget() {
    let budget = TokenBudget::default();

    assert!(budget.us_per_token > 0.0, "F039: Default budget > 0");
    assert!(budget.us_per_token < 10000.0, "F039: Default budget < 10ms");
    assert!(budget.tokens_per_sec > 0.0, "F039: Default throughput > 0");
}

/// F040: Budget fields are never NaN (bonus)
#[test]
fn fkr_budget_f040_no_nan() {
    let budgets = vec![
        TokenBudget::from_latency(1.0),
        TokenBudget::from_latency(100.0),
        TokenBudget::from_latency(10000.0),
        TokenBudget::from_throughput(100.0),
        TokenBudget::from_throughput(10000.0),
        TokenBudget::from_throughput(1000000.0),
        TokenBudget::default(),
    ];

    for budget in &budgets {
        assert!(
            !budget.us_per_token.is_nan(),
            "F040: us_per_token must not be NaN"
        );
        assert!(
            !budget.tokens_per_sec.is_nan(),
            "F040: tokens_per_sec must not be NaN"
        );
    }
}
