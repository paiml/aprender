// =========================================================================
// FALSIFY-SA: sampling-algorithms-v1.yaml contract (realizar sampling)
//
// Five-Whys (PMAT-354):
//   Why 1: realizar had 40+ sampling tests but zero FALSIFY-SA-* tests
//   Why 2: unit tests verify edge cases/shapes, not mathematical invariants
//   Why 3: no mapping from sampling-algorithms-v1.yaml to realizar test names
//   Why 4: realizar predates the provable-contracts YAML convention
//   Why 5: sampling was "obviously correct" (argmax, top-k filter, nucleus)
//
// References:
//   - provable-contracts/contracts/sampling-algorithms-v1.yaml
//   - Holtzman et al. (2020) "The Curious Case of Neural Text Degeneration"
// =========================================================================

use crate::generate::*;
use crate::tensor::Tensor;

/// FALSIFY-SA-001: Greedy = argmax — sample_greedy must return index of max logit
///
/// Contract: sample_greedy(logits) == argmax(logits)
#[test]
fn falsify_sa_001_greedy_is_argmax() {
    let test_cases: Vec<(Vec<f32>, usize)> = vec![
        // Simple ascending — last is max
        (vec![1.0, 2.0, 3.0, 4.0, 5.0], 4),
        // Simple descending — first is max
        (vec![5.0, 4.0, 3.0, 2.0, 1.0], 0),
        // Max in the middle
        (vec![1.0, 2.0, 100.0, 3.0, 4.0], 2),
        // All equal — first occurrence
        (vec![7.0, 7.0, 7.0], 0),
        // Single element
        (vec![42.0], 0),
        // Negative logits
        (vec![-5.0, -1.0, -3.0, -0.5, -10.0], 3),
        // Large vocabulary
        {
            let mut logits = vec![0.0; 32000];
            logits[12345] = 100.0;
            (logits, 12345)
        },
    ];

    for (i, (logits_data, expected_idx)) in test_cases.iter().enumerate() {
        let logits =
            Tensor::from_vec(vec![logits_data.len()], logits_data.clone()).expect("test tensor");
        let result = sample_greedy(&logits).expect("greedy should succeed");
        assert_eq!(
            result, *expected_idx,
            "FALSIFIED SA-001 case {i}: greedy returned {result}, expected {expected_idx}"
        );
    }
}

/// FALSIFY-SA-002: Top-K cardinality — only top-k tokens may be selected
///
/// Contract: sample_top_k(logits, k, rng) ∈ {indices of top-k logits}
#[test]
fn falsify_sa_002_top_k_cardinality() {
    // Logits with clear ordering: index 3 (10.0) > 0 (5.0) > 4 (3.0) > 1 (1.0) > 2 (0.1)
    let logits_data = vec![5.0, 1.0, 0.1, 10.0, 3.0];
    let logits = Tensor::from_vec(vec![5], logits_data).expect("test tensor");
    let k = 3;

    // Top-3 indices by logit value: {3, 0, 4}
    let top_k_set: std::collections::HashSet<usize> = [3, 0, 4].iter().copied().collect();

    // Sweep across the full rng_value range to exercise all branches
    for rng_step in 0..100 {
        let rng_value = rng_step as f32 / 100.0;
        let result = sample_top_k(&logits, k, rng_value).expect("top_k should succeed");
        assert!(
            top_k_set.contains(&result),
            "FALSIFIED SA-002: top_k(k={k}, rng={rng_value}) returned {result}, \
             not in top-{k} set {:?}",
            top_k_set
        );
    }
}

/// FALSIFY-SA-003: Top-P cumulative — only tokens within nucleus may be selected
///
/// Contract: selected token ∈ {minimal set where cumsum(sorted_probs) >= p}
#[test]
fn falsify_sa_003_top_p_cumulative() {
    // Logits designed so softmax produces well-separated probabilities
    // [10.0, 1.0, 0.0, -1.0, -10.0] → softmax ≈ [0.9998, 0.000123, 4.5e-5, 1.7e-5, 2.1e-9]
    let logits_data = vec![10.0, 1.0, 0.0, -1.0, -10.0];
    let logits = Tensor::from_vec(vec![5], logits_data).expect("test tensor");

    // With p=0.01, only the top token (index 0) should be in the nucleus
    // because softmax(10.0) ≈ 0.9998 > 0.01
    for rng_step in 0..100 {
        let rng_value = rng_step as f32 / 100.0;
        let result = sample_top_p(&logits, 0.01, rng_value).expect("top_p should succeed");
        assert_eq!(
            result, 0,
            "FALSIFIED SA-003: top_p(p=0.01, rng={rng_value}) returned {result}, \
             expected 0 (only token above nucleus threshold)"
        );
    }

    // With p=1.0 (full distribution), any token may be selected — just verify no error
    for rng_step in 0..20 {
        let rng_value = rng_step as f32 / 20.0;
        let result = sample_top_p(&logits, 1.0, rng_value);
        assert!(
            result.is_ok(),
            "FALSIFIED SA-003: top_p(p=1.0) should always succeed"
        );
    }
}

/// FALSIFY-SA-004: Temperature identity — temperature=1.0 preserves logits
///
/// Contract: apply_temperature(logits, 1.0) == logits
#[test]
fn falsify_sa_004_temperature_identity() {
    let test_cases: Vec<Vec<f32>> = vec![
        vec![1.0, 2.0, 3.0],
        vec![-100.0, 0.0, 100.0],
        vec![0.0; 10],
        vec![42.0],
        vec![1e-6, 1e6],
    ];

    for (i, logits_data) in test_cases.iter().enumerate() {
        let logits =
            Tensor::from_vec(vec![logits_data.len()], logits_data.clone()).expect("test tensor");
        let result = apply_temperature(&logits, 1.0).expect("temp=1.0 should succeed");

        for (j, (&original, &scaled)) in logits.data().iter().zip(result.data().iter()).enumerate()
        {
            assert!(
                (original - scaled).abs() < 1e-6,
                "FALSIFIED SA-004 case {i}[{j}]: temp=1.0 changed {original} to {scaled}"
            );
        }
    }
}

/// FALSIFY-SA-004b: Temperature scaling — temperature divides logits
///
/// Contract: apply_temperature(logits, T) = logits / T
#[test]
fn falsify_sa_004b_temperature_scaling() {
    let logits_data = vec![2.0, 4.0, 6.0, 8.0];
    let logits = Tensor::from_vec(vec![4], logits_data).expect("test tensor");

    for &temp in &[0.5_f32, 2.0, 0.1, 10.0] {
        let result = apply_temperature(&logits, temp).expect("should succeed");
        for (j, (&original, &scaled)) in logits.data().iter().zip(result.data().iter()).enumerate()
        {
            let expected = original / temp;
            let diff = (scaled - expected).abs();
            assert!(
                diff < 1e-5,
                "FALSIFIED SA-004b: temp={temp}, logits[{j}]={original} → {scaled}, expected {expected}"
            );
        }
    }
}
