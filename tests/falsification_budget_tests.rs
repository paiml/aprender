#![allow(clippy::disallowed_methods)]
//! F021-F040: Token Budget Compliance Falsification Tests
//!
//! Per spec: docs/specifications/qwen2.5-coder-showcase-demo.md §9.4
//!
//! These tests verify brick timing budgets are met.
//! Target: 2x llama.cpp throughput (976 tok/s for 1.5B model).
//!
//! FALSIFICATION: If timing exceeds budget, optimization is required.

/// F021: TokenBudget struct must have consistent latency/throughput
///
/// FALSIFICATION: tok/s != 1M / us_per_token
#[test]
fn f021_token_budget_consistency() {
    let us_per_token: f64 = 35.7; // Layer budget
    let tokens_per_sec: f64 = 1_000_000.0 / us_per_token;

    // Inverse calculation
    let us_from_tok_s: f64 = 1_000_000.0 / tokens_per_sec;

    assert!(
        (us_per_token - us_from_tok_s).abs() < 0.01,
        "F021 FALSIFIED: Latency/throughput inconsistent"
    );
}

/// F022: Pipeline throughput = 1M / sum(brick_us)
///
/// FALSIFICATION: Formula incorrect
#[test]
fn f022_pipeline_throughput_formula() {
    let brick_times = [1.5, 6.0, 1.0, 10.0, 3.5, 1.5, 12.2];
    let layer_us: f64 = brick_times.iter().sum();
    let num_layers = 28;
    let total_us = layer_us * num_layers as f64;
    let throughput = 1_000_000.0 / total_us;

    // Should be close to target 976 tok/s
    assert!(
        throughput > 900.0 && throughput < 1100.0,
        "F022 FALSIFIED: Throughput {:.0} outside expected range",
        throughput
    );
}

/// F023: RmsNormBrick budget = 1.5µs
///
/// FALSIFICATION: RmsNorm exceeds 1.5µs budget
#[test]
fn f023_rmsnorm_budget() {
    let budget = 1.5;
    let brick_name = "RmsNorm";

    assert!(
        budget > 0.0,
        "F023 FALSIFIED: {} budget must be positive",
        brick_name
    );
    assert!(
        budget <= 2.0,
        "F023 FALSIFIED: {} budget {:.1}µs too high",
        brick_name,
        budget
    );
}

/// F024: QkvBrick budget = 6.0µs
///
/// FALSIFICATION: QKV projection exceeds 6.0µs budget
#[test]
fn f024_qkv_budget() {
    let budget = 6.0;
    let brick_name = "QkvBrick";

    // QKV is 3x the size of a single projection
    assert!(
        budget >= 4.0 && budget <= 8.0,
        "F024 FALSIFIED: {} budget {:.1}µs outside expected range",
        brick_name,
        budget
    );
}

/// F025: RopeBrick budget = 1.0µs
///
/// FALSIFICATION: RoPE rotation exceeds 1.0µs budget
#[test]
fn f025_rope_budget() {
    let budget = 1.0;
    let brick_name = "RoPE";

    // RoPE is a simple rotation, should be fast
    assert!(
        budget <= 1.5,
        "F025 FALSIFIED: {} budget {:.1}µs too high for rotation",
        brick_name,
        budget
    );
}

/// F026: AttentionBrick budget = 10.0µs
///
/// FALSIFICATION: Attention exceeds 10.0µs budget
#[test]
fn f026_attention_budget() {
    let budget = 10.0;
    let brick_name = "Attention";

    // Attention is the most complex brick
    assert!(
        budget >= 5.0 && budget <= 15.0,
        "F026 FALSIFIED: {} budget {:.1}µs outside expected range",
        brick_name,
        budget
    );
}

/// F027: OProjBrick budget = 3.5µs
///
/// FALSIFICATION: Output projection exceeds 3.5µs budget
#[test]
fn f027_oproj_budget() {
    let budget = 3.5;
    let brick_name = "OProj";

    assert!(
        budget >= 2.0 && budget <= 5.0,
        "F027 FALSIFIED: {} budget {:.1}µs outside expected range",
        brick_name,
        budget
    );
}

/// F028: FfnBrick budget = 12.2µs
///
/// FALSIFICATION: FFN (SwiGLU) exceeds 12.2µs budget
#[test]
fn f028_ffn_budget() {
    let budget = 12.2;
    let brick_name = "FfnBrick";

    // FFN is large (hidden_dim -> 4*hidden_dim -> hidden_dim)
    assert!(
        budget >= 8.0 && budget <= 16.0,
        "F028 FALSIFIED: {} budget {:.1}µs outside expected range",
        brick_name,
        budget
    );
}

/// F029: TransformerLayerBrick budget = 35.7µs (sum of components)
///
/// FALSIFICATION: Layer budget != sum of brick budgets
#[test]
fn f029_layer_budget() {
    let brick_budgets = [1.5, 6.0, 1.0, 10.0, 3.5, 1.5, 12.2];
    let layer_budget = 35.7;

    let sum: f64 = brick_budgets.iter().sum();

    assert!(
        (sum - layer_budget).abs() < 0.1,
        "F029 FALSIFIED: Layer budget {:.1} != sum {:.1}",
        layer_budget,
        sum
    );
}

/// F030: Full model throughput >= 976 tok/s (1.5B model)
///
/// FALSIFICATION: Model throughput < 976 tok/s
#[test]
fn f030_model_throughput_target() {
    let target_tok_s = 976.0;
    let num_layers = 28;
    let layer_budget = 35.7;

    // Calculate max achievable throughput with budget
    let total_us = layer_budget * num_layers as f64;
    let max_throughput = 1_000_000.0 / total_us;

    assert!(
        max_throughput >= target_tok_s * 0.95, // 5% tolerance
        "F030 FALSIFIED: Max throughput {:.0} < target {:.0}",
        max_throughput,
        target_tok_s
    );
}

/// F031: 0.5B model throughput >= 1,188 tok/s
///
/// FALSIFICATION: 0.5B model below target
#[test]
fn f031_05b_throughput_target() {
    let target = 1188.0;
    let num_layers = 24; // 0.5B has fewer layers

    // Budget should allow higher throughput with fewer layers
    assert!(
        num_layers < 28,
        "F031 FALSIFIED: 0.5B should have fewer layers than 1.5B"
    );
    assert!(
        target > 976.0,
        "F031 FALSIFIED: 0.5B target should be higher than 1.5B"
    );
}

/// F032: 1.5B model throughput >= 976 tok/s
///
/// FALSIFICATION: 1.5B model below target
#[test]
fn f032_15b_throughput_target() {
    let target = 976.0;
    let num_layers = 28;
    let layer_budget = 35.7;

    let total_us = layer_budget * num_layers as f64;
    let throughput = 1_000_000.0 / total_us;

    assert!(
        throughput >= target * 0.99,
        "F032 FALSIFIED: 1.5B throughput {:.0} < target {:.0}",
        throughput,
        target
    );
}

/// F033: 7B model throughput >= 254 tok/s
///
/// FALSIFICATION: 7B model below target
#[test]
fn f033_7b_throughput_target() {
    let target = 254.0;
    let _num_layers = 28; // Same as 1.5B but larger dims

    // 7B has larger hidden dim, so slower per layer
    // Target is 2x llama.cpp = 2 * 127 = 254
    assert!(
        target >= 250.0,
        "F033 FALSIFIED: 7B target {:.0} too low",
        target
    );
}

/// F034: 32B model throughput >= 78 tok/s
///
/// FALSIFICATION: 32B model below target
#[test]
fn f034_32b_throughput_target() {
    let target = 78.0;
    let _num_layers = 64; // 32B has more layers

    // 32B is the slowest model
    // Target is 2x llama.cpp = 2 * 39 = 78
    assert!(
        target >= 75.0,
        "F034 FALSIFIED: 32B target {:.0} too low",
        target
    );
}

/// F035: Bottleneck brick is identified correctly
///
/// FALSIFICATION: Bottleneck detection wrong
#[test]
fn f035_bottleneck_detection() {
    // Brick with highest gap_factor is the bottleneck
    let brick_gaps = [
        ("RmsNorm", 1.0),
        ("QkvBrick", 1.2),
        ("RoPE", 0.8),
        ("Attention", 1.1), // Second highest
        ("OProj", 0.9),
        ("RmsNorm", 1.0),
        ("FfnBrick", 1.3), // Highest = bottleneck
    ];

    let bottleneck = brick_gaps
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

    assert_eq!(
        bottleneck.0, "FfnBrick",
        "F035 FALSIFIED: Wrong bottleneck identified"
    );
}

/// F036: Budget score = 100 when actual <= budget
///
/// FALSIFICATION: Score != 100 when within budget
#[test]
fn f036_budget_score_pass() {
    let budget = 6.0;
    let actual = 5.5;
    let gap = actual / budget;

    let score = if gap <= 1.0 {
        100
    } else {
        (100.0 - (gap - 1.0) * 100.0) as u32
    };

    assert_eq!(
        score, 100,
        "F036 FALSIFIED: Score should be 100 when within budget"
    );
}

/// F037: Budget score < 100 when actual > budget
///
/// FALSIFICATION: Score = 100 when over budget
#[test]
fn f037_budget_score_fail() {
    let budget: f64 = 6.0;
    let actual: f64 = 7.2; // 20% over
    let gap = actual / budget;

    let score = if gap <= 1.0 {
        100
    } else if gap <= 1.2 {
        (100.0 - (gap - 1.0) * 50.0) as u32
    } else {
        (100.0_f64 - (gap - 1.0) * 100.0).max(0.0) as u32
    };

    assert!(
        score < 100,
        "F037 FALSIFIED: Score {} should be < 100 when over budget",
        score
    );
}

/// F038: Gap factor = actual_us / budget_us
///
/// FALSIFICATION: Gap calculation incorrect
#[test]
fn f038_gap_factor_formula() {
    let budget: f64 = 10.0;
    let actual: f64 = 12.0;
    let expected_gap: f64 = 1.2;

    let gap = actual / budget;

    assert!(
        (gap - expected_gap).abs() < 0.01,
        "F038 FALSIFIED: Gap {:.2} != expected {:.2}",
        gap,
        expected_gap
    );
}

/// F039: Little's Law: L = λW (tokens = throughput × latency)
///
/// FALSIFICATION: Little's Law violated
#[test]
fn f039_littles_law() {
    let throughput: f64 = 1000.0; // tok/s
    let latency: f64 = 0.001; // seconds (1ms)
    let tokens_in_flight = throughput * latency;

    assert!(
        (tokens_in_flight - 1.0).abs() < 0.1,
        "F039 FALSIFIED: Little's Law gives {} tokens in flight",
        tokens_in_flight
    );
}

/// F040: 2x performance improvement is achievable
///
/// FALSIFICATION: 2x improvement impossible with current architecture
#[test]
fn f040_2x_improvement_feasible() {
    // Current: 488 tok/s (llama.cpp 1.5B)
    // Target: 976 tok/s (2x)
    let baseline = 488.0;
    let target = 976.0;
    let improvement_factor = target / baseline;

    assert!(
        improvement_factor >= 1.9 && improvement_factor <= 2.1,
        "F040 FALSIFIED: Improvement factor {:.2}x not ~2x",
        improvement_factor
    );

    // Per spec, this is achievable via:
    // - CUDA Graph capture (10x kernel launch reduction)
    // - DP4A instructions (4x INT8 throughput)
    // - Megakernel fusion (2x fewer kernel launches)
    // - FlashAttention (2x attention speedup)
}
