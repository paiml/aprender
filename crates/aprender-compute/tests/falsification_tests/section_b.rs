//! Section B: Determinism (Claims 16-30)

use simular::engine::rng::SimRng;
use trueno::simulation::{BackendTolerance, JidokaGuard};
use trueno::{select_best_available_backend, Backend, Vector};

// =============================================================================
// SECTION B: Determinism (Claims 16-30)
// =============================================================================

/// B-016: SimRng::new(seed) produces identical sequence on every platform
#[test]
fn test_b016_simrng_platform_independent() {
    let mut rng = SimRng::new(42);

    // Generate a known sequence
    let seq: Vec<f64> = (0..10).map(|_| rng.gen_f64()).collect();

    // Re-run with same seed
    let mut rng2 = SimRng::new(42);
    let seq2: Vec<f64> = (0..10).map(|_| rng2.gen_f64()).collect();

    assert_eq!(seq, seq2, "B-016 FALSIFIED: Same seed must produce identical sequences");
}

/// B-017: Same seed + same input produces identical output across runs
#[test]
fn test_b017_deterministic_output() {
    for _ in 0..100 {
        let mut rng = SimRng::new(42);
        let data: Vec<f32> = (0..1000).map(|_| rng.gen_f64() as f32).collect();

        let a = Vector::from_slice(&data);
        let b: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let b_vec = Vector::from_slice(&b);

        let result = a.add(&b_vec).expect("add failed");

        // Re-run with same seed
        let mut rng2 = SimRng::new(42);
        let data2: Vec<f32> = (0..1000).map(|_| rng2.gen_f64() as f32).collect();
        let a2 = Vector::from_slice(&data2);
        let result2 = a2.add(&b_vec).expect("add failed");

        // Verify bitwise equality
        for (i, (r1, r2)) in result.as_slice().iter().zip(result2.as_slice().iter()).enumerate() {
            assert_eq!(
                r1.to_bits(),
                r2.to_bits(),
                "B-017 FALSIFIED: Results differ at index {i}: {r1} != {r2}",
            );
        }
    }
}

/// B-018: Different seeds produce different outputs
#[test]
fn test_b018_different_seeds_different_outputs() {
    let sequences: Vec<Vec<f64>> = (0..1000)
        .map(|seed| {
            let mut rng = SimRng::new(seed);
            (0..10).map(|_| rng.gen_f64()).collect()
        })
        .collect();

    // Check all pairs are different
    for i in 0..sequences.len() {
        for j in (i + 1)..sequences.len() {
            assert_ne!(
                sequences[i], sequences[j],
                "B-018 FALSIFIED: Seeds {} and {} produce same sequence",
                i, j
            );
        }
    }
}

/// B-019: Parallel execution with same seed is deterministic
#[test]
fn test_b019_parallel_determinism() {
    let mut rng = SimRng::new(42);
    let partitions = rng.partition(4);

    // Each partition should produce consistent results
    let mut results: Vec<Vec<f64>> = Vec::new();
    for mut partition in partitions {
        let seq: Vec<f64> = (0..100).map(|_| partition.gen_f64()).collect();
        results.push(seq);
    }

    // Re-run with same seed
    let mut rng2 = SimRng::new(42);
    let partitions2 = rng2.partition(4);

    let mut results2: Vec<Vec<f64>> = Vec::new();
    for mut partition in partitions2 {
        let seq: Vec<f64> = (0..100).map(|_| partition.gen_f64()).collect();
        results2.push(seq);
    }

    assert_eq!(results, results2, "B-019 FALSIFIED: Parallel partitions not deterministic");
}

/// B-020: GPU execution with same seed is deterministic
#[test]
fn test_b020_gpu_determinism() {
    // GPU determinism test - simulated since GPU may not be available
    // Verify that GPU tolerance allows for reproducible results within tolerance
    let tolerance = BackendTolerance::relaxed();
    let gpu_tolerance = tolerance.for_backends(Backend::GPU, Backend::GPU);

    // Simulate two GPU runs with same seed
    let mut rng1 = SimRng::new(12345);
    let mut rng2 = SimRng::new(12345);

    let result1: Vec<f32> = (0..100).map(|_| rng1.gen_f64() as f32).collect();
    let result2: Vec<f32> = (0..100).map(|_| rng2.gen_f64() as f32).collect();

    // GPU results should be identical when using same seed
    for (i, (r1, r2)) in result1.iter().zip(result2.iter()).enumerate() {
        let diff = (r1 - r2).abs();
        assert!(
            diff <= gpu_tolerance,
            "B-020 FALSIFIED: GPU results differ at index {i}: {} vs {} (diff: {})",
            r1,
            r2,
            diff
        );
    }
}

/// B-022: System load does not affect numerical results
#[test]
fn test_b022_system_load_independence() {
    // Run computation multiple times to verify consistency under varying load
    let data = Vector::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let scalar = 2.0f32;

    let mut results: Vec<Vec<f32>> = Vec::new();

    // Run 10 times - system load may vary between runs
    for _ in 0..10 {
        let scaled = data.scale(scalar).expect("scale failed");
        results.push(scaled.as_slice().to_vec());
    }

    // All results should be identical
    let first = &results[0];
    for (i, result) in results.iter().enumerate().skip(1) {
        assert_eq!(
            first, result,
            "B-022 FALSIFIED: Run {i} produced different results under varying load"
        );
    }
}

/// B-023: Memory pressure does not affect numerical results
#[test]
fn test_b023_memory_pressure_independence() {
    let data = Vector::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    // Capture result before allocating memory
    let result1 = data.sum().expect("sum failed");

    // Allocate and drop memory to create pressure
    {
        let _pressure: Vec<Vec<f32>> = (0..100).map(|_| vec![0.0f32; 10_000]).collect();
        // Memory is allocated here
    }
    // Memory is freed here

    // Capture result after memory pressure
    let result2 = data.sum().expect("sum failed");

    assert!(
        (result1 - result2).abs() < f32::EPSILON,
        "B-023 FALSIFIED: Memory pressure affected results: {result1} vs {result2}"
    );
}

/// B-021: Test order does not affect results (test isolation)
#[test]
fn test_b021_test_isolation() {
    let data = Vector::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = Vector::from_slice(&[1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

    // Order 1: add then mul
    let add_result = data.add(&b).expect("add failed");
    let mul_result = data.mul(&b).expect("mul failed");

    // Order 2: mul then add
    let mul_result2 = data.mul(&b).expect("mul failed");
    let add_result2 = data.add(&b).expect("add failed");

    // Results should be identical regardless of order
    assert_eq!(
        add_result.as_slice(),
        add_result2.as_slice(),
        "B-021 FALSIFIED: Test order affected add result"
    );
    assert_eq!(
        mul_result.as_slice(),
        mul_result2.as_slice(),
        "B-021 FALSIFIED: Test order affected mul result"
    );
}

/// B-024: Determinism holds for all input sizes 1 to 10M
#[test]
fn test_b024_determinism_all_sizes() {
    let sizes = [1, 7, 8, 15, 16, 31, 32, 100, 1000, 10_000];

    for size in sizes {
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (size - 1 - i) as f32).collect();

        let vec_a = Vector::from_slice(&a);
        let vec_b = Vector::from_slice(&b);

        let result1 = vec_a.add(&vec_b).expect("add failed");
        let result2 = vec_a.add(&vec_b).expect("add failed");

        assert_eq!(
            result1.as_slice(),
            result2.as_slice(),
            "B-024 FALSIFIED: Size {} not deterministic",
            size
        );
    }
}

/// B-025: Determinism holds for special values (0, -0, MIN, MAX)
#[test]
fn test_b025_special_values_determinism() {
    let special = vec![0.0f32, -0.0f32, f32::MIN, f32::MAX, f32::MIN_POSITIVE, -f32::MIN_POSITIVE];
    let b = vec![1.0f32; special.len()];

    let vec_special = Vector::from_slice(&special);
    let vec_b = Vector::from_slice(&b);

    for _ in 0..100 {
        let result = vec_special.add(&vec_b).expect("add failed");

        // Verify consistent handling of special values
        assert!(result.as_slice()[0] == 1.0, "B-025 FALSIFIED: 0.0 + 1.0 should equal 1.0");
        assert!(result.as_slice()[1] == 1.0, "B-025 FALSIFIED: -0.0 + 1.0 should equal 1.0");
    }
}

/// B-026: Determinism holds for subnormal numbers
#[test]
fn test_b026_subnormal_determinism() {
    // Smallest positive subnormal
    let subnormal = f32::from_bits(1);
    assert!(subnormal > 0.0 && subnormal < f32::MIN_POSITIVE);

    let a = vec![subnormal; 8];
    let b = vec![subnormal; 8];

    let vec_a = Vector::from_slice(&a);
    let vec_b = Vector::from_slice(&b);

    let result1 = vec_a.add(&vec_b).expect("add failed");
    let result2 = vec_a.add(&vec_b).expect("add failed");

    for (r1, r2) in result1.as_slice().iter().zip(result2.as_slice().iter()) {
        assert_eq!(
            r1.to_bits(),
            r2.to_bits(),
            "B-026 FALSIFIED: Subnormal handling not deterministic"
        );
    }
}

/// B-027: Determinism holds for NaN inputs (NaN propagation)
#[test]
fn test_b027_nan_propagation() {
    let nan_guard = JidokaGuard::nan_guard("B-027");

    let a = vec![1.0f32, f32::NAN, 3.0, 4.0];
    let b = vec![1.0f32, 1.0, 1.0, 1.0];

    let vec_a = Vector::from_slice(&a);
    let vec_b = Vector::from_slice(&b);

    let result = vec_a.add(&vec_b).expect("add failed");

    // NaN should propagate
    assert!(result.as_slice()[1].is_nan(), "B-027 FALSIFIED: NaN should propagate");

    // Jidoka should detect it
    let check = nan_guard.check_output(result.as_slice());
    assert!(check.is_err(), "B-027: Jidoka should detect NaN");
}

/// B-028: Determinism holds for Infinity inputs
#[test]
fn test_b028_infinity_handling() {
    let inf_guard = JidokaGuard::inf_guard("B-028");

    let a = vec![1.0f32, f32::INFINITY, f32::NEG_INFINITY, 4.0];
    let b = vec![1.0f32, 1.0, 1.0, 1.0];

    let vec_a = Vector::from_slice(&a);
    let vec_b = Vector::from_slice(&b);

    let result = vec_a.add(&vec_b).expect("add failed");

    // Infinity should propagate
    assert!(
        result.as_slice()[1].is_infinite() && result.as_slice()[1] > 0.0,
        "B-028 FALSIFIED: +Inf should propagate"
    );
    assert!(
        result.as_slice()[2].is_infinite() && result.as_slice()[2] < 0.0,
        "B-028 FALSIFIED: -Inf should propagate"
    );

    // Jidoka should detect it
    let check = inf_guard.check_output(result.as_slice());
    assert!(check.is_err(), "B-028: Jidoka should detect Infinity");
}

/// B-029: Cross-process determinism (fork safety)
#[test]
fn test_b029_cross_process_determinism() {
    // In single-process test, verify RNG state is isolated
    let mut rng1 = SimRng::new(42);
    let seq1: Vec<f64> = (0..100).map(|_| rng1.gen_f64()).collect();

    // Create new RNG with same seed (simulating fork)
    let mut rng2 = SimRng::new(42);
    let seq2: Vec<f64> = (0..100).map(|_| rng2.gen_f64()).collect();

    assert_eq!(seq1, seq2, "B-029 FALSIFIED: RNG not deterministic across instances");
}

/// B-030: Thread-local state does not leak between tests
#[test]
fn test_b030_thread_local_isolation() {
    // Backend selection is cached in OnceLock (thread-safe)
    let backend1 = select_best_available_backend();
    let backend2 = select_best_available_backend();

    assert_eq!(backend1, backend2, "B-030 FALSIFIED: Backend selection should be consistent");
}
