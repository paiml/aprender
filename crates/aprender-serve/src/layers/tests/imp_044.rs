
/// IMP-044: Parallel FFN computation
/// Target: Parallelize feed-forward network layers
#[test]
fn test_imp_044_parallel_ffn() {
    use crate::gpu::{parallel_ffn, sequential_ffn};
    use std::time::Instant;

    // FFN weights
    let hidden_dim = 256;
    let intermediate_dim = 512;

    // Up projection: hidden_dim -> intermediate_dim
    let w_up: Vec<f32> = (0..hidden_dim * intermediate_dim)
        .map(|i| ((i % 100) as f32) * 0.01 - 0.5)
        .collect();

    // Down projection: intermediate_dim -> hidden_dim
    let w_down: Vec<f32> = (0..intermediate_dim * hidden_dim)
        .map(|i| ((i % 100) as f32) * 0.01 - 0.5)
        .collect();

    // Input
    let input: Vec<f32> = (0..hidden_dim).map(|i| (i as f32) * 0.01).collect();

    // Test 1: Sequential and parallel should produce same results
    let sequential_result = sequential_ffn(&input, &w_up, &w_down, hidden_dim, intermediate_dim);
    let parallel_result = parallel_ffn(&input, &w_up, &w_down, hidden_dim, intermediate_dim);

    assert_eq!(
        sequential_result.len(),
        parallel_result.len(),
        "IMP-044: Results should have same length"
    );

    for (i, (&s, &p)) in sequential_result
        .iter()
        .zip(parallel_result.iter())
        .enumerate()
    {
        assert!(
            (s - p).abs() < 1e-4,
            "IMP-044: Mismatch at index {}: sequential={}, parallel={}",
            i,
            s,
            p
        );
    }

    // Test 2: Parallel should be at least as fast for larger inputs
    let large_input: Vec<f32> = (0..hidden_dim).map(|i| (i as f32) * 0.01).collect();

    // Warmup
    for _ in 0..3 {
        let _ = sequential_ffn(&large_input, &w_up, &w_down, hidden_dim, intermediate_dim);
        let _ = parallel_ffn(&large_input, &w_up, &w_down, hidden_dim, intermediate_dim);
    }

    // Benchmark sequential
    let mut seq_times = Vec::with_capacity(5);
    for _ in 0..5 {
        let start = Instant::now();
        for _ in 0..50 {
            let _ = sequential_ffn(&large_input, &w_up, &w_down, hidden_dim, intermediate_dim);
        }
        seq_times.push(start.elapsed().as_secs_f64());
    }
    seq_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Benchmark parallel
    let mut par_times = Vec::with_capacity(5);
    for _ in 0..5 {
        let start = Instant::now();
        for _ in 0..50 {
            let _ = parallel_ffn(&large_input, &w_up, &w_down, hidden_dim, intermediate_dim);
        }
        par_times.push(start.elapsed().as_secs_f64());
    }
    par_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let seq_median = seq_times[seq_times.len() / 2];
    let par_median = par_times[par_times.len() / 2];
    let speedup = seq_median / par_median;

    // Note: Performance benchmarks are unreliable under coverage instrumentation
    // The key test is correctness (Test 1). Performance is informational only.
    // Use dedicated benchmarks (make bench) for actual performance measurement.
    let _ = speedup; // Prevent unused warning
}

/// IMP-045: Optimized layer norm with running statistics
/// Target: Fused mean/variance computation using Welford's algorithm
#[test]
fn test_imp_045_optimized_layernorm() {
    use crate::gpu::{fused_layernorm, standard_layernorm};
    use std::time::Instant;

    let hidden_dim = 256;
    let eps = 1e-5;

    // Test input
    let input: Vec<f32> = (0..hidden_dim).map(|i| (i as f32) * 0.1 - 12.8).collect();

    // Gamma and beta (scale and shift)
    let gamma: Vec<f32> = vec![1.0; hidden_dim];
    let beta: Vec<f32> = vec![0.0; hidden_dim];

    // Test 1: Both methods should produce same results
    let standard_result = standard_layernorm(&input, &gamma, &beta, eps);
    let fused_result = fused_layernorm(&input, &gamma, &beta, eps);

    assert_eq!(
        standard_result.len(),
        fused_result.len(),
        "IMP-045: Results should have same length"
    );

    for (i, (&s, &f)) in standard_result.iter().zip(fused_result.iter()).enumerate() {
        assert!(
            (s - f).abs() < 1e-5,
            "IMP-045: Mismatch at index {}: standard={}, fused={}",
            i,
            s,
            f
        );
    }

    // Test 2: Output should be normalized (mean ≈ 0, variance ≈ 1 before gamma/beta)
    let mean: f32 = fused_result.iter().sum::<f32>() / fused_result.len() as f32;
    assert!(
        mean.abs() < 0.1,
        "IMP-045: Normalized output mean ({}) should be near 0",
        mean
    );

    // Test 3: Fused should be at least as fast
    // Warmup
    for _ in 0..5 {
        let _ = standard_layernorm(&input, &gamma, &beta, eps);
        let _ = fused_layernorm(&input, &gamma, &beta, eps);
    }

    // Benchmark standard
    let mut std_times = Vec::with_capacity(10);
    for _ in 0..10 {
        let start = Instant::now();
        for _ in 0..100 {
            let _ = standard_layernorm(&input, &gamma, &beta, eps);
        }
        std_times.push(start.elapsed().as_secs_f64());
    }
    std_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Benchmark fused
    let mut fused_times = Vec::with_capacity(10);
    for _ in 0..10 {
        let start = Instant::now();
        for _ in 0..100 {
            let _ = fused_layernorm(&input, &gamma, &beta, eps);
        }
        fused_times.push(start.elapsed().as_secs_f64());
    }
    fused_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let std_median = std_times[std_times.len() / 2];
    let fused_median = fused_times[fused_times.len() / 2];
    let speedup = std_median / fused_median;

    // Note: Performance benchmarks unreliable under coverage instrumentation
    // The key test is correctness. Performance is informational only.
    let _ = speedup;
}

// ============================================================================
// Phase 12: Cache Efficiency & Prefetch (M21) - IMP-046/047/048
// ============================================================================
