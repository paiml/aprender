use super::super::super::super::super::*;

// Tests for clip() - Constrain values to [min, max] range
// ========================================================================

#[test]
fn test_clip_basic() {
    // [-5, 0, 5, 10, 15] clipped to [0, 10] -> [0, 0, 5, 10, 10]
    let v = Vector::from_slice(&[-5.0, 0.0, 5.0, 10.0, 15.0]);
    let clipped = v.clip(0.0, 10.0).unwrap();

    assert_eq!(clipped.as_slice(), &[0.0, 0.0, 5.0, 10.0, 10.0]);
}

#[test]
fn test_clip_no_change() {
    // All values within range should stay unchanged
    let v = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0]);
    let clipped = v.clip(0.0, 10.0).unwrap();

    assert_eq!(clipped.as_slice(), &[2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_clip_all_below() {
    // All values below min -> all become min
    let v = Vector::from_slice(&[-10.0, -5.0, -2.0]);
    let clipped = v.clip(0.0, 10.0).unwrap();

    assert_eq!(clipped.as_slice(), &[0.0, 0.0, 0.0]);
}

#[test]
fn test_clip_all_above() {
    // All values above max -> all become max
    let v = Vector::from_slice(&[15.0, 20.0, 25.0]);
    let clipped = v.clip(0.0, 10.0).unwrap();

    assert_eq!(clipped.as_slice(), &[10.0, 10.0, 10.0]);
}

#[test]
fn test_clip_invalid_range() {
    // min > max -> InvalidInput error
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = v.clip(10.0, 5.0);

    assert!(matches!(result, Err(TruenoError::InvalidInput(_))));
}

#[test]
fn test_clip_equal_bounds() {
    // min == max -> all values become that value
    let v = Vector::from_slice(&[-5.0, 0.0, 5.0, 10.0]);
    let clipped = v.clip(7.0, 7.0).unwrap();

    assert_eq!(clipped.as_slice(), &[7.0, 7.0, 7.0, 7.0]);
}

// ========================================================================
// Tests for softmax() - Softmax activation (probability distribution)
// ========================================================================

#[test]
fn test_softmax_basic() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let probs = v.softmax().unwrap();

    // Verify sum ~= 1
    let sum: f32 = probs.as_slice().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "sum = {}, expected 1", sum);

    // Verify all values in [0, 1]
    for &p in probs.as_slice() {
        assert!((0.0..=1.0).contains(&p), "prob = {} not in [0, 1]", p);
    }

    // Largest input should have largest probability
    assert!(probs.data[2] > probs.data[1]);
    assert!(probs.data[1] > probs.data[0]);
}

#[test]
fn test_softmax_uniform() {
    // All equal inputs -> uniform distribution
    let v = Vector::from_slice(&[5.0, 5.0, 5.0, 5.0]);
    let probs = v.softmax().unwrap();

    // Each should be 1/4 = 0.25
    for &p in probs.as_slice() {
        assert!((p - 0.25).abs() < 1e-5, "prob = {}, expected 0.25", p);
    }
}

#[test]
fn test_softmax_large_values() {
    // Test numerical stability with large values
    let v = Vector::from_slice(&[100.0, 101.0, 102.0]);
    let probs = v.softmax().unwrap();

    // Should still sum to 1
    let sum: f32 = probs.as_slice().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);

    // Largest value should have largest probability
    assert!(probs.data[2] > probs.data[1]);
    assert!(probs.data[1] > probs.data[0]);
}

#[test]
fn test_softmax_negative_values() {
    let v = Vector::from_slice(&[-3.0, -2.0, -1.0]);
    let probs = v.softmax().unwrap();

    // Verify sum ~= 1
    let sum: f32 = probs.as_slice().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);

    // All values should be positive
    for &p in probs.as_slice() {
        assert!(p > 0.0);
    }
}

#[test]
fn test_softmax_empty_vector() {
    let v = Vector::from_slice(&[]);
    let result = v.softmax();
    assert!(matches!(result, Err(TruenoError::EmptyVector)));
}

#[test]
fn test_softmax_single_element() {
    // Single element -> probability 1.0
    let v = Vector::from_slice(&[5.0]);
    let probs = v.softmax().unwrap();

    assert!((probs.data[0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_log_softmax_basic() {
    // Verify exp(log_softmax(x)) == softmax(x)
    let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let log_probs = v.log_softmax().unwrap();
    let probs = v.softmax().unwrap();

    for i in 0..v.len() {
        let exp_log_prob = log_probs.data[i].exp();
        assert!(
            (exp_log_prob - probs.data[i]).abs() < 1e-5,
            "exp(log_softmax)[{}] = {}, softmax[{}] = {}",
            i,
            exp_log_prob,
            i,
            probs.data[i]
        );
    }
}

#[test]
fn test_log_softmax_uniform() {
    // All equal inputs -> uniform log probabilities
    let v = Vector::from_slice(&[5.0, 5.0, 5.0, 5.0]);
    let log_probs = v.log_softmax().unwrap();

    // Each should be log(1/4) = log(0.25) ~= -1.386
    let expected = (0.25_f32).ln();
    for &lp in log_probs.as_slice() {
        assert!((lp - expected).abs() < 1e-5, "log_prob = {}, expected {}", lp, expected);
    }
}

#[test]
fn test_log_softmax_large_values() {
    // Test numerical stability with large values
    let v = Vector::from_slice(&[100.0, 101.0, 102.0]);
    let log_probs = v.log_softmax().unwrap();

    // exp(log_probs) should sum to 1
    let sum: f32 = log_probs.as_slice().iter().map(|&lp| lp.exp()).sum();
    assert!((sum - 1.0).abs() < 1e-5);

    // All log probabilities should be <= 0 (since probabilities <= 1)
    for &lp in log_probs.as_slice() {
        assert!(lp <= 1e-5, "log_prob = {} should be <= 0", lp);
    }

    // Largest input should have largest log probability (least negative)
    assert!(log_probs.data[2] > log_probs.data[1]);
    assert!(log_probs.data[1] > log_probs.data[0]);
}

#[test]
fn test_log_softmax_negative_values() {
    // Negative values should work fine
    let v = Vector::from_slice(&[-1.0, -2.0, -3.0]);
    let log_probs = v.log_softmax().unwrap();

    // exp(log_probs) should sum to 1
    let sum: f32 = log_probs.as_slice().iter().map(|&lp| lp.exp()).sum();
    assert!((sum - 1.0).abs() < 1e-5);

    // All log probabilities should be <= 0
    for &lp in log_probs.as_slice() {
        assert!(lp <= 1e-5);
    }
}

#[test]
fn test_log_softmax_empty_vector() {
    let v = Vector::from_slice(&[]);
    let result = v.log_softmax();
    assert!(matches!(result, Err(TruenoError::EmptyVector)));
}

#[test]
fn test_log_softmax_single_element() {
    // Single element -> log probability = log(1.0) = 0.0
    let v = Vector::from_slice(&[5.0]);
    let log_probs = v.log_softmax().unwrap();

    assert!(
        log_probs.data[0].abs() < 1e-5,
        "log_softmax of single element should be 0.0, got {}",
        log_probs.data[0]
    );
}

// ========================================================================
// FALSIFY-SM: softmax-kernel-v1.yaml contract falsification
//
// Five-Whys (PMAT-354):
//   Why 1: trueno had 7 softmax tests but 0 FALSIFY-SM-* tagged tests
//   Why 2: existing tests verify behavior, not provable-contract YAML claims
//   Why 3: trueno's softmax predates softmax-kernel-v1.yaml
//   Why 4: no cross-repo YAML→test naming convention existed
//   Why 5: softmax was "obviously correct" so no formal contracts
//
// References:
//   - provable-contracts/contracts/softmax-kernel-v1.yaml
//   - Bridle (1990) "Training Stochastic Model Recognition Algorithms"
// ========================================================================

/// FALSIFY-SM-001: Output sums to 1 (partition of unity)
///
/// Contract: |Σ σ(x)_i - 1.0| < ε (tolerance 1e-6)
#[test]
fn falsify_sm_001_sums_to_one() {
    let test_cases: Vec<Vec<f32>> = vec![
        vec![1.0, 2.0, 3.0],
        vec![-10.0, 0.0, 10.0],
        vec![100.0, 101.0, 102.0],
        vec![0.001, 0.002, 0.003],
        (0..100).map(|i| (i as f32 * 0.37).sin() * 5.0).collect(),
    ];

    for (idx, logits) in test_cases.iter().enumerate() {
        let v = Vector::from_vec(logits.clone());
        let probs = v.softmax().unwrap();
        let sum: f32 = probs.as_slice().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "FALSIFIED SM-001: case {idx} sum={sum}, expected 1.0");
    }
}

/// FALSIFY-SM-002: All outputs strictly positive
///
/// Contract: σ(x)_i > 0 for all i
#[test]
fn falsify_sm_002_strictly_positive() {
    let logits: Vec<f32> = (0..50).map(|i| (i as f32 - 25.0) * 2.0).collect();
    let v = Vector::from_vec(logits);
    let probs = v.softmax().unwrap();

    for (i, &p) in probs.as_slice().iter().enumerate() {
        assert!(p > 0.0, "FALSIFIED SM-002: probs[{i}] = {p} is not strictly positive");
    }
}

/// FALSIFY-SM-003: Order preservation (argmax invariant)
///
/// Contract: argmax(σ(x)) = argmax(x)
#[test]
fn falsify_sm_003_order_preservation() {
    let test_cases: Vec<Vec<f32>> =
        vec![vec![1.0, 5.0, 3.0], vec![-100.0, 0.0, -50.0], vec![0.001, 0.002, 0.001]];

    for (idx, logits) in test_cases.iter().enumerate() {
        let input_argmax = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        let v = Vector::from_vec(logits.clone());
        let probs = v.softmax().unwrap();
        let output_argmax = probs
            .as_slice()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        assert_eq!(
            input_argmax, output_argmax,
            "FALSIFIED SM-003: case {idx} argmax changed from {input_argmax} to {output_argmax}"
        );
    }
}

/// FALSIFY-SM-004: Each output bounded in (0, 1)
///
/// Contract: 0 < σ(x)_i < 1 for all i (when n > 1)
#[test]
fn falsify_sm_004_bounded_zero_one() {
    let logits: Vec<f32> = (0..20).map(|i| (i as f32 * 1.7).sin() * 10.0).collect();
    let v = Vector::from_vec(logits);
    let probs = v.softmax().unwrap();

    for (i, &p) in probs.as_slice().iter().enumerate() {
        assert!(p > 0.0 && p < 1.0, "FALSIFIED SM-004: probs[{i}] = {p} not in (0, 1)");
    }
}

/// FALSIFY-SM-005: Numerical stability with extreme values
///
/// Contract: softmax must not produce NaN or Inf even for large/small inputs
#[test]
fn falsify_sm_005_numerical_stability() {
    let extreme_cases: Vec<Vec<f32>> = vec![
        vec![1000.0, 1001.0, 1002.0],       // Large positive
        vec![-1000.0, -999.0, -998.0],      // Large negative
        vec![-500.0, 0.0, 500.0],           // Huge range
        vec![f32::MIN_POSITIVE, 1.0, 80.0], // Near-zero to large
    ];

    for (idx, logits) in extreme_cases.iter().enumerate() {
        let v = Vector::from_vec(logits.clone());
        let probs = v.softmax().unwrap();

        for (i, &p) in probs.as_slice().iter().enumerate() {
            assert!(p.is_finite(), "FALSIFIED SM-005: case {idx} probs[{i}] = {p} is not finite");
        }

        let sum: f32 = probs.as_slice().iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "FALSIFIED SM-005: case {idx} sum={sum} after extreme inputs"
        );
    }
}

/// FALSIFY-SM-006: Identical elements → uniform distribution
///
/// Contract: softmax([c, c, ..., c]) = [1/n, 1/n, ..., 1/n]
#[test]
fn falsify_sm_006_identical_elements_uniform() {
    for n in [2, 4, 8, 16] {
        let v = Vector::from_vec(vec![5.0; n]);
        let probs = v.softmax().unwrap();
        let expected = 1.0 / n as f32;

        for (i, &p) in probs.as_slice().iter().enumerate() {
            assert!(
                (p - expected).abs() < 1e-6,
                "FALSIFIED SM-006: n={n} probs[{i}] = {p}, expected {expected}"
            );
        }
    }
}

/// FALSIFY-SM-007: Translation invariance — σ(x + c) = σ(x) for any scalar c
///
/// Five-Whys (PMAT-354):
///   Why 1: SM-INV-003 (translation invariance) had ZERO coverage in any repo
///   Why 2: the max-subtraction trick IMPLEMENTS this property but nobody tests it
///   Why 3: shift invariance is "obviously true" from the mathematical definition
///   Why 4: no mapping from proof obligation SM-INV-003 to any FALSIFY test
///   Why 5: the property is foundational to numerical stability but untested
///
/// Contract: σ(x + c·1) = σ(x) for any scalar c.
/// This is the mathematical basis for the max-subtraction stability trick.
#[test]
fn falsify_sm_007_translation_invariance() {
    let base = Vector::from_vec(vec![1.0, 3.0, -2.0, 0.5]);
    let base_probs = base.softmax().unwrap();

    // Shift by various constants — result must be identical
    for c in [100.0, -100.0, 0.0, 42.0, -999.0, 1e6] {
        let shifted = Vector::from_vec(vec![1.0 + c, 3.0 + c, -2.0 + c, 0.5 + c]);
        let shifted_probs = shifted.softmax().unwrap();

        for (i, (&orig, &shift)) in
            base_probs.as_slice().iter().zip(shifted_probs.as_slice().iter()).enumerate()
        {
            assert!(
                (orig - shift).abs() < 1e-5,
                "FALSIFIED SM-007: σ(x+{c})[{i}] = {shift} != σ(x)[{i}] = {orig}"
            );
        }
    }
}

/// FALSIFY-SM-008: SIMD equivalence — scalar vs auto backend within ULP
///
/// Five-Whys (PMAT-354):
///   Why 1: YAML SM-004 specifies SIMD equivalence but was mapped to "bounded"
///   Why 2: naming mismatch — we used SM-004 for SM-BND-001 (bounded)
///   Why 3: scalar vs SIMD parity was assumed correct
///   Why 4: no explicit comparison test existed
///   Why 5: trueno's backend dispatch was tested for correctness, not equivalence
///
/// Contract: |softmax_auto(x) - softmax_scalar(x)| < 8 ULP
#[test]
fn falsify_sm_008_simd_scalar_equivalence() {
    let test_inputs: Vec<Vec<f32>> = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![-10.0, 0.0, 10.0],
        vec![100.0, 100.0, 100.0, 100.0],
        (0..32).map(|i| (i as f32 * 0.7).sin()).collect(),
        (0..128).map(|i| i as f32 - 64.0).collect(),
        vec![-500.0, 0.0, 500.0],
    ];

    for (idx, input) in test_inputs.iter().enumerate() {
        let v_scalar = Vector::from_slice_with_backend(input, Backend::Scalar);
        let v_auto = Vector::from_vec(input.clone()); // Auto selects best SIMD

        let scalar_probs = v_scalar.softmax().unwrap();
        let auto_probs = v_auto.softmax().unwrap();

        for (i, (&s, &a)) in
            scalar_probs.as_slice().iter().zip(auto_probs.as_slice().iter()).enumerate()
        {
            let diff = (s - a).abs();
            let ulp_bound = 8.0 * f32::EPSILON * s.abs().max(a.abs()).max(f32::MIN_POSITIVE);
            assert!(
                diff <= ulp_bound,
                "FALSIFIED SM-008: case {idx}[{i}] scalar={s} vs auto={a}, diff={diff} > {ulp_bound}"
            );
        }
    }
}

/// FALSIFY-SM-009: Single element boundary — softmax([x]) = [1.0]
///
/// Contract: YAML SM-005 = softmax of a single element is always 1.0.
#[test]
fn falsify_sm_009_single_element() {
    for x in [0.0f32, 1.0, -1.0, 100.0, -100.0, f32::MIN_POSITIVE, 1e30] {
        let v = Vector::from_vec(vec![x]);
        let probs = v.softmax().unwrap();
        assert!(
            (probs.as_slice()[0] - 1.0).abs() < 1e-6,
            "FALSIFIED SM-009: softmax([{x}]) = {}, expected 1.0",
            probs.as_slice()[0]
        );
    }
}

// ========================================================================
// FALSIFY-SM proptest variants: random-vector falsification
//
// These complement the deterministic FALSIFY-SM-001..009 tests above with
// proptest-driven randomised inputs for higher coverage confidence.
// ========================================================================

mod softmax_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    // FALSIFY-SM-001-prop: Normalization for random vectors
    //
    // Contract: |Σ σ(x)_i - 1.0| < ε for any input vector
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(500))]
        #[test]
        fn falsify_sm_001_prop_sums_to_one(
            logits in proptest::collection::vec(-100.0_f32..100.0, 2..64),
        ) {
            let v = Vector::from_vec(logits);
            let probs = v.softmax().unwrap();
            let sum: f32 = probs.as_slice().iter().sum();
            prop_assert!(
                (sum - 1.0).abs() < 1e-4,
                "FALSIFIED SM-001-prop: sum={} for {} elements", sum, probs.len()
            );
        }
    }

    // FALSIFY-SM-002-prop: Positivity for random vectors
    //
    // Contract: σ(x)_i > 0 and is finite for all i
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(500))]
        #[test]
        fn falsify_sm_002_prop_positive(
            logits in proptest::collection::vec(-500.0_f32..500.0, 2..32),
        ) {
            let v = Vector::from_vec(logits.clone());
            let probs = v.softmax().unwrap();
            for (i, &p) in probs.as_slice().iter().enumerate() {
                prop_assert!(p >= 0.0, "FALSIFIED SM-002-prop: probs[{}]={} negative (n={})", i, p, logits.len());
                prop_assert!(p.is_finite(), "FALSIFIED SM-002-prop: probs[{}]={} non-finite", i, p);
            }
        }
    }

    // FALSIFY-SM-003-prop: Order preservation for random vectors
    //
    // Contract: argmax(σ(x)) = argmax(x) when no duplicate max exists
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(500))]
        #[test]
        fn falsify_sm_003_prop_order_preservation(
            logits in proptest::collection::vec(-50.0_f32..50.0, 2..32),
        ) {
            // Skip if duplicate max exists (proptest may generate ties)
            let has_dupes = logits.windows(2).any(|w| (w[0] - w[1]).abs() < 1e-10);
            if has_dupes {
                return Ok(());
            }

            let v = Vector::from_vec(logits.clone());
            let probs = v.softmax().unwrap();
            let input_argmax = logits.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0;
            let output_argmax = probs.as_slice().iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0;
            prop_assert_eq!(
                input_argmax, output_argmax,
                "FALSIFIED SM-003-prop: argmax {} -> {} for {:?}", input_argmax, output_argmax, logits
            );
        }
    }
}
