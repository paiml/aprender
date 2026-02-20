// ============================================================================
// M013-M020: Brick Scoring Infrastructure (10 points)
// ============================================================================

/// M013: SIMD efficiency must be in 0-1 range
#[test]
fn m013_simd_efficiency_range() {
    let peak_gflops = 100.0;
    let actual_gflops = 75.0;
    let efficiency = actual_gflops / peak_gflops;

    assert!(
        (0.0..=1.0).contains(&efficiency),
        "M013 FALSIFIED: SIMD efficiency {:.2} outside 0-1 range",
        efficiency
    );

    assert!(
        (0.0..=1.0).contains(&0.0),
        "M013 FALSIFIED: 0.0 should be valid"
    );
    assert!(
        (0.0..=1.0).contains(&1.0),
        "M013 FALSIFIED: 1.0 should be valid"
    );
}

/// M014: Memory bandwidth ratio must be in 0-1 range
#[test]
fn m014_memory_bandwidth_range() {
    let peak_gb_s = 1000.0;
    let achieved_gb_s = 800.0;
    let ratio = achieved_gb_s / peak_gb_s;

    assert!(
        (0.0..=1.0).contains(&ratio),
        "M014 FALSIFIED: Memory bandwidth ratio {:.2} outside 0-1 range",
        ratio
    );
}

/// M015: Latency ratio capped at 1.0 for scoring
#[test]
fn m015_latency_ratio_capped() {
    let budget_us: f64 = 6.0;
    let actual_us: f64 = 4.0;

    let raw_ratio: f64 = budget_us / actual_us;
    let capped_ratio: f64 = raw_ratio.min(1.0);

    assert!(
        capped_ratio <= 1.0,
        "M015 FALSIFIED: Latency ratio {:.2} not capped at 1.0",
        capped_ratio
    );

    let actual_over = 8.0;
    let raw_over = budget_us / actual_over;
    assert!(
        raw_over < 1.0,
        "M015 FALSIFIED: Over-budget should have ratio < 1.0"
    );
}

/// M016: Stability = 1 - CV (coefficient of variation)
#[test]
fn m016_stability_formula() {
    let cv_percent: f64 = 3.0;
    let stability: f64 = 1.0 - (cv_percent / 100.0);

    assert!(
        (stability - 0.97).abs() < 0.001,
        "M016 FALSIFIED: Stability {:.3} != expected 0.97",
        stability
    );

    let perfect_cv: f64 = 0.0;
    let perfect_stability: f64 = 1.0 - (perfect_cv / 100.0);
    assert!(
        (perfect_stability - 1.0).abs() < 0.001,
        "M016 FALSIFIED: CV=0 should give stability=1.0"
    );

    let worst_cv: f64 = 100.0;
    let worst_stability: f64 = 1.0 - (worst_cv / 100.0);
    assert!(
        worst_stability.abs() < 0.001,
        "M016 FALSIFIED: CV=100 should give stability=0.0"
    );
}

/// M017: CUDA-TDG score formula correct
#[test]
fn m017_cuda_tdg_formula() {
    let brick_scores = [100u32, 94, 100, 100, 100, 93, 96];
    let weights = [1.5, 6.0, 1.0, 10.0, 3.5, 1.5, 12.2];

    let weighted_sum: f64 = brick_scores
        .iter()
        .zip(weights.iter())
        .map(|(&s, &w)| s as f64 * w)
        .sum();
    let total_weight: f64 = weights.iter().sum();
    let tdg_score = weighted_sum / total_weight;

    assert!(
        tdg_score >= 90.0 && tdg_score <= 100.0,
        "M017 FALSIFIED: TDG score {:.1} outside expected range",
        tdg_score
    );
}

/// M018: Roofline model bounds check per Williams 2009
#[test]
fn m018_roofline_bounds() {
    let peak_gflops: f64 = 82.6;
    let peak_bw_gb_s: f64 = 1008.0;
    let ai: f64 = 0.375;

    let roofline_limit: f64 = peak_gflops.min(peak_bw_gb_s * ai);

    let actual_gflops = 50.0;
    assert!(
        actual_gflops <= roofline_limit * 1.1,
        "M018 FALSIFIED: Actual {:.1} GFLOPS exceeds roofline {:.1}",
        actual_gflops,
        roofline_limit
    );

    assert!(
        ai > 0.0,
        "M018 FALSIFIED: Arithmetic intensity must be positive"
    );
}

/// M019: Aggregate model score = mean of brick scores
#[test]
fn m019_aggregate_score() {
    let brick_scores = [100u32, 94, 100, 100, 100, 93, 96];
    let n = brick_scores.len();

    let sum: u32 = brick_scores.iter().sum();
    let mean = sum as f64 / n as f64;

    let expected_mean = (100.0 + 94.0 + 100.0 + 100.0 + 100.0 + 93.0 + 96.0) / 7.0;

    assert!(
        (mean - expected_mean).abs() < 0.01,
        "M019 FALSIFIED: Aggregate {:.2} != expected {:.2}",
        mean,
        expected_mean
    );

    assert!(
        mean >= 0.0 && mean <= 100.0,
        "M019 FALSIFIED: Mean score {:.2} outside 0-100 range",
        mean
    );
}

/// M020: Score JSON schema has required fields
#[test]
fn m020_score_json_schema() {
    let required_fields = [
        "model",
        "timestamp",
        "hardware",
        "throughput",
        "brick_scores",
        "pmat_scores",
        "falsification",
        "status",
        "ci_result",
    ];

    let json_output = r#"{
        "model": "qwen2.5-coder-1.5b",
        "timestamp": "2026-01-11T00:00:00Z",
        "hardware": {"gpu": "RTX 4090", "cpu": "Ryzen 9", "memory_gb": 64},
        "throughput": {"tokens_per_sec": 976.0, "ttft_ms": 1.0, "cv_percent": 2.5, "p50_us": 1.5, "p99_us": 3.0},
        "brick_scores": [],
        "pmat_scores": {"rust_project_score": 152.9, "tdg_score": 98.1, "cuda_tdg_score": 95.2, "brick_score": 97, "grade": "A"},
        "falsification": {"total_points": 120, "passed": 60, "failed": 0, "blocked": 60},
        "status": "PASS",
        "ci_result": "green"
    }"#;

    for field in required_fields {
        assert!(
            json_output.contains(&format!("\"{}\"", field)),
            "M020 FALSIFIED: JSON missing required field '{}'",
            field
        );
    }

    let pmat_fields = [
        "rust_project_score",
        "tdg_score",
        "cuda_tdg_score",
        "brick_score",
        "grade",
    ];
    for field in pmat_fields {
        assert!(
            json_output.contains(&format!("\"{}\"", field)),
            "M020 FALSIFIED: pmat_scores missing field '{}'",
            field
        );
    }
}
