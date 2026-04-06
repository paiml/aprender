use super::super::*;

// ========================================================================
// PMAT-017: AttentionOp Tests
// ========================================================================

#[test]
fn test_attention_op_basic() {
    // Simple 2x2 attention (seq_len=2, kv_seq_len=2, head_dim=2)
    let op = AttentionOp::self_attention(2, 2);

    // Q = [[1, 0], [0, 1]]
    let q = vec![1.0, 0.0, 0.0, 1.0];
    // K = [[1, 0], [0, 1]]
    let k = vec![1.0, 0.0, 0.0, 1.0];
    // V = [[1, 2], [3, 4]]
    let v = vec![1.0, 2.0, 3.0, 4.0];

    let result = op.execute((q, k, v), Backend::Scalar).unwrap();

    // Output should be [seq_len * head_dim] = 4 elements
    assert_eq!(result.len(), 4);

    // Each row should be a weighted sum of V rows
    // Q[0]·K[0] = 1, Q[0]·K[1] = 0 → softmax → [~0.73, ~0.27]
    // Output[0] ≈ 0.73 * [1,2] + 0.27 * [3,4]
    assert!(result[0] > 0.0 && result[0] < 3.0);
    assert!(result[1] > 0.0 && result[1] < 4.0);
}

#[test]
fn test_attention_op_simd_dot() {
    // Test the SIMD dot product directly
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let result = AttentionOp::simd_dot(&a, &b);
    assert!((result - 36.0).abs() < 0.001); // 1+2+3+4+5+6+7+8 = 36
}

#[test]
fn test_attention_op_softmax_row() {
    let mut scores = vec![1.0, 2.0, 3.0];
    AttentionOp::simd_softmax_row(&mut scores);

    // Sum should be 1.0
    let sum: f32 = scores.iter().sum();
    assert!((sum - 1.0).abs() < 0.001);

    // Values should be increasing
    assert!(scores[0] < scores[1]);
    assert!(scores[1] < scores[2]);
}

#[test]
fn test_attention_op_dimension_validation() {
    let op = AttentionOp::new(2, 3, 4);

    // Wrong Q size
    let result = op.execute((vec![0.0; 4], vec![0.0; 12], vec![0.0; 12]), Backend::Scalar);
    assert!(result.is_err());

    // Wrong K size
    let result = op.execute((vec![0.0; 8], vec![0.0; 8], vec![0.0; 12]), Backend::Scalar);
    assert!(result.is_err());
}

#[test]
fn test_attention_op_single_position() {
    // Single query position attending to 3 key positions
    let op = AttentionOp::new(1, 3, 4);

    let q = vec![1.0, 0.0, 0.0, 0.0]; // [1, 4]
    let k = vec![
        1.0, 0.0, 0.0, 0.0, // K[0]
        0.0, 1.0, 0.0, 0.0, // K[1]
        0.0, 0.0, 1.0, 0.0, // K[2]
    ];
    let v = vec![
        1.0, 0.0, 0.0, 0.0, // V[0]
        0.0, 1.0, 0.0, 0.0, // V[1]
        0.0, 0.0, 1.0, 0.0, // V[2]
    ];

    let result = op.execute((q, k, v), Backend::Scalar).unwrap();
    assert_eq!(result.len(), 4);

    // Q·K[0] = 1, Q·K[1] = 0, Q·K[2] = 0
    // After softmax: [~0.58, ~0.21, ~0.21] (approx)
    // Output ≈ 0.58*V[0] + 0.21*V[1] + 0.21*V[2]
    // Should have higher weight on first component
    assert!(result[0] > result[1]);
}

// =========================================================================
// FALSIFY-ATT: attention-kernel-v1.yaml contract (trueno AttentionOp)
//
// Five-Whys (PMAT-354):
//   Why 1: trueno had 5 attention unit tests but zero FALSIFY-ATT-* tests
//   Why 2: unit tests verify shapes/dot/softmax, not mathematical invariants
//   Why 3: no mapping from attention-kernel-v1.yaml to trueno test names
//   Why 4: trueno predates the provable-contracts YAML convention
//   Why 5: attention was "obviously correct" (textbook formula)
//
// References:
//   - provable-contracts/contracts/attention-kernel-v1.yaml
//   - Vaswani et al. (2017) "Attention Is All You Need"
// =========================================================================

/// FALSIFY-ATT-001: Weight normalization — softmax rows sum to 1.0
#[test]
fn falsify_att_001_weight_normalization() {
    // Use softmax_row directly on various score vectors
    for scores_input in [
        vec![1.0, 2.0, 3.0],
        vec![0.0, 0.0, 0.0, 0.0],
        vec![-10.0, 10.0],
        vec![100.0, -100.0, 0.0, 50.0],
        vec![0.1, 0.2, 0.3, 0.4, 0.5],
    ] {
        let mut scores = scores_input.clone();
        AttentionOp::simd_softmax_row(&mut scores);
        let sum: f32 = scores.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "FALSIFIED ATT-001: softmax sum = {sum}, expected 1.0 (input: {scores_input:?})"
        );
    }
}

/// FALSIFY-ATT-005: Weights bounded — all softmax outputs in (0, 1)
#[test]
fn falsify_att_005_weights_bounded() {
    for scores_input in [
        vec![1.0, 2.0, 3.0],
        vec![-100.0, 0.0, 100.0],
        vec![0.0, 0.0, 0.0],
        vec![50.0, 50.0, 50.0, 50.0],
    ] {
        let mut scores = scores_input.clone();
        AttentionOp::simd_softmax_row(&mut scores);
        for (i, &w) in scores.iter().enumerate() {
            // Note: weights can be 0.0 due to f32 exp() underflow for extreme inputs
            assert!(w >= 0.0, "FALSIFIED ATT-005: weight[{i}] = {w} < 0 (input: {scores_input:?})");
            assert!(
                w <= 1.0 + 1e-6,
                "FALSIFIED ATT-005: weight[{i}] = {w} > 1 (input: {scores_input:?})"
            );
        }
    }
}

/// FALSIFY-ATT-002: Output convexity — output bounded by V range
#[test]
fn falsify_att_002_output_convexity() {
    let op = AttentionOp::self_attention(3, 4);
    let q: Vec<f32> = (0..12).map(|i| (i as f32 * 0.37).sin()).collect();
    let k: Vec<f32> = (0..12).map(|i| (i as f32 * 0.73).cos()).collect();
    let v: Vec<f32> = vec![
        2.0, -3.0, 5.0, 1.0, // V[0]
        -1.0, 4.0, -2.0, 7.0, // V[1]
        3.0, 0.0, -4.0, 6.0, // V[2]
    ];

    let result = op.execute((q, k, v.clone()), Backend::Scalar).unwrap();

    // Output is [3 * 4] = 12 elements
    for i in 0..3 {
        for d in 0..4 {
            let out_val = result[i * 4 + d];
            let v_col_min = (0..3).map(|j| v[j * 4 + d]).fold(f32::INFINITY, f32::min);
            let v_col_max = (0..3).map(|j| v[j * 4 + d]).fold(f32::NEG_INFINITY, f32::max);

            assert!(
                out_val >= v_col_min - 1e-4 && out_val <= v_col_max + 1e-4,
                "FALSIFIED ATT-002: output[{i}][{d}] = {out_val} outside V range [{v_col_min}, {v_col_max}]"
            );
        }
    }
}

/// FALSIFY-ATT-003: Scaling factor — uses 1/√d_k
#[test]
fn falsify_att_003_scaling_factor() {
    // AttentionOp stores scale = 1/√head_dim. Verify for known values.
    for &hd in &[4, 8, 16, 32, 64, 128] {
        let op = AttentionOp::self_attention(1, hd);
        let expected_scale = 1.0 / (hd as f32).sqrt();
        assert!(
            (op.scale - expected_scale).abs() < 1e-6,
            "FALSIFIED ATT-003: scale for d_k={hd} is {}, expected {expected_scale}",
            op.scale
        );
    }
}

mod att_proptest_falsify {
    use super::super::super::*;
    use proptest::prelude::*;

    // FALSIFY-ATT-001-prop: Softmax normalization for random inputs
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn falsify_att_001_prop_weight_normalization(
            n in 2..=16usize,
            seed in 0..1000u32,
        ) {
            let mut scores: Vec<f32> = (0..n)
                .map(|i| ((i as f32 + seed as f32) * 0.37).sin() * 10.0)
                .collect();
            AttentionOp::simd_softmax_row(&mut scores);
            let sum: f32 = scores.iter().sum();
            prop_assert!(
                (sum - 1.0).abs() < 1e-4,
                "FALSIFIED ATT-001-prop: softmax sum = {}, expected 1.0 (n={})",
                sum, n
            );
        }
    }

    // FALSIFY-ATT-005-prop: Softmax outputs bounded in (0, 1)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn falsify_att_005_prop_weights_bounded(
            n in 2..=16usize,
            scale in 0.1f32..100.0,
            seed in 0..1000u32,
        ) {
            let mut scores: Vec<f32> = (0..n)
                .map(|i| ((i as f32 + seed as f32) * 0.73).cos() * scale)
                .collect();
            AttentionOp::simd_softmax_row(&mut scores);
            for (i, &w) in scores.iter().enumerate() {
                // Note: w can be 0.0 due to f32 exp() underflow for extreme inputs
                prop_assert!(
                    (0.0..=1.0 + 1e-6).contains(&w),
                    "FALSIFIED ATT-005-prop: weight[{}] = {} outside [0,1] (n={}, scale={})",
                    i, w, n, scale
                );
            }
        }
    }

    // FALSIFY-ATT-002-prop: Output convexity for random V
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn falsify_att_002_prop_output_convexity(
            seed in 0..1000u32,
        ) {
            let seq = 3;
            let hd = 4;
            let op = AttentionOp::self_attention(seq, hd);

            let q: Vec<f32> = (0..seq * hd)
                .map(|i| ((i as f32 + seed as f32) * 0.37).sin())
                .collect();
            let k: Vec<f32> = (0..seq * hd)
                .map(|i| ((i as f32 + seed as f32) * 0.73).cos())
                .collect();
            let v: Vec<f32> = (0..seq * hd)
                .map(|i| ((i as f32 + seed as f32) * 1.23).sin() * 5.0)
                .collect();

            let result = op.execute((q, k, v.clone()), Backend::Scalar).unwrap();

            for d in 0..hd {
                let v_min = (0..seq).map(|j| v[j * hd + d]).fold(f32::INFINITY, f32::min);
                let v_max = (0..seq).map(|j| v[j * hd + d]).fold(f32::NEG_INFINITY, f32::max);

                for i in 0..seq {
                    let out = result[i * hd + d];
                    prop_assert!(
                        out >= v_min - 1e-4 && out <= v_max + 1e-4,
                        "FALSIFIED ATT-002-prop: output[{}][{}] = {} outside V range [{}, {}]",
                        i, d, out, v_min, v_max
                    );
                }
            }
        }
    }
}

#[test]
fn test_compute_brick_run() {
    let brick =
        ComputeBrick::new(DotOp::new(4)).budget_tok_per_sec(1_000_000.0).backend(Backend::Scalar);

    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let result = brick.run((a, b)).unwrap();

    assert!((result.output - 70.0).abs() < 0.001);
    assert_eq!(result.tokens_processed, 4);
    assert!(result.tokens_per_sec > 0.0);
}

#[test]
fn test_compute_brick_verify() {
    let brick = ComputeBrick::new(DotOp::new(4)).assert_finite().assert_bounds(-1000.0, 1000.0);

    let verification = brick.verify();
    assert!(verification.is_valid());
    assert_eq!(verification.assertion_results.len(), 2);
}

#[test]
fn test_compute_brick_no_assertions() {
    let brick = ComputeBrick::new(DotOp::new(4));
    let verification = brick.verify();
    assert!(!verification.is_valid()); // Should fail Popperian requirement
}

#[test]
fn test_brick_layer() {
    let dot_brick = ComputeBrick::new(DotOp::new(100)).budget_tok_per_sec(50_000.0);

    let add_brick = ComputeBrick::new(AddOp::new(100)).budget_tok_per_sec(30_000.0); // Bottleneck

    let layer = BrickLayer::new().with_brick(&dot_brick).with_brick(&add_brick);

    assert!((layer.throughput_ceiling() - 30_000.0).abs() < 1.0);
    assert_eq!(layer.bottleneck(), Some("add"));
}

#[test]
fn test_backend_display() {
    assert_eq!(format!("{}", Backend::Avx2), "AVX2");
    assert_eq!(format!("{}", Backend::Cuda), "CUDA");
    assert_eq!(format!("{}", Backend::Scalar), "Scalar");
}

#[test]
fn test_budget_utilization() {
    let budget = TokenBudget::from_latency(100.0);
    assert!((budget.utilization(50.0) - 0.5).abs() < 0.001); // 50% used
    assert!((budget.utilization(100.0) - 1.0).abs() < 0.001); // 100% used
    assert!((budget.utilization(150.0) - 1.5).abs() < 0.001); // 150% over
}
