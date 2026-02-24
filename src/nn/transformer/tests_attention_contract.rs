// =========================================================================
// FALSIFY-ATT: attention-kernel-v1.yaml contract (aprender attention)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had 40+ attention unit tests but zero FALSIFY-ATT-* tests
//   Why 2: unit tests verify shapes/params, not mathematical invariants
//   Why 3: no mapping from attention-kernel-v1.yaml to aprender test names
//   Why 4: aprender predates the provable-contracts YAML convention
//   Why 5: attention was "obviously correct" (textbook Vaswani formula)
//
// References:
//   - provable-contracts/contracts/attention-kernel-v1.yaml
//   - Vaswani et al. (2017) "Attention Is All You Need"
// =========================================================================

use super::*;

/// FALSIFY-ATT-001: Weight normalization — each attention weight row sums to 1.0
///
/// Contract: Σ_j softmax(QK^T/√d_k)_{ij} = 1 for all i
#[test]
fn falsify_att_001_weight_normalization() {
    // Q: [batch=1, heads=1, seq=3, d_k=4]
    let q = Tensor::new(
        &[
            1.0, 0.5, -0.3, 0.8, -1.0, 0.2, 0.7, -0.5, 0.4, -0.6, 0.3, 0.9,
        ],
        &[1, 1, 3, 4],
    );
    let k = Tensor::new(
        &[
            0.3, -0.7, 1.0, 0.2, -0.5, 0.8, 0.1, -0.3, 0.6, -0.1, 0.4, 0.9,
        ],
        &[1, 1, 3, 4],
    );
    let v = Tensor::new(
        &[
            2.0, -3.0, 5.0, 1.0, -1.0, 4.0, -2.0, 7.0, 3.0, 0.0, -4.0, 6.0,
        ],
        &[1, 1, 3, 4],
    );

    let (_output, attn_weights) = scaled_dot_product_attention(&q, &k, &v, None, 0.0, false);

    // attn_weights shape: [1, 1, 3, 3] — each of 3 rows should sum to 1
    let data = attn_weights.data();
    for i in 0..3 {
        let row_sum: f32 = (0..3).map(|j| data[i * 3 + j]).sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-5,
            "FALSIFIED ATT-001: attn_weights row {i} sum = {row_sum}, expected 1.0"
        );
    }
}

/// FALSIFY-ATT-002: Output convexity — output bounded by min/max of V columns
///
/// Contract: min_j(V[j][d]) ≤ output[i][d] ≤ max_j(V[j][d])
#[test]
fn falsify_att_002_output_convexity() {
    let seq_len = 3;
    let d_v = 4;
    let v_data: Vec<f32> = vec![
        2.0, -3.0, 5.0, 1.0, -1.0, 4.0, -2.0, 7.0, 3.0, 0.0, -4.0, 6.0,
    ];

    let q = Tensor::new(
        &[
            1.0, 0.5, -0.3, 0.8, -1.0, 0.2, 0.7, -0.5, 0.4, -0.6, 0.3, 0.9,
        ],
        &[1, 1, seq_len, d_v],
    );
    let k = Tensor::new(
        &[
            0.3, -0.7, 1.0, 0.2, -0.5, 0.8, 0.1, -0.3, 0.6, -0.1, 0.4, 0.9,
        ],
        &[1, 1, seq_len, d_v],
    );
    let v = Tensor::new(&v_data, &[1, 1, seq_len, d_v]);

    let (output, _) = scaled_dot_product_attention(&q, &k, &v, None, 0.0, false);
    let out_data = output.data();

    for i in 0..seq_len {
        for d in 0..d_v {
            let out_val = out_data[i * d_v + d];

            let v_col_min = (0..seq_len)
                .map(|j| v_data[j * d_v + d])
                .fold(f32::INFINITY, f32::min);
            let v_col_max = (0..seq_len)
                .map(|j| v_data[j * d_v + d])
                .fold(f32::NEG_INFINITY, f32::max);

            assert!(
                out_val >= v_col_min - 1e-4 && out_val <= v_col_max + 1e-4,
                "FALSIFIED ATT-002: output[{i}][{d}] = {out_val} outside V column [{v_col_min}, {v_col_max}]"
            );
        }
    }
}

/// FALSIFY-ATT-003: Scaling factor — uses 1/√d_k
///
/// Verify by comparing against manual reference with known scaling.
#[test]
fn falsify_att_003_scaling_factor() {
    // Single query position, 2 key positions, d_k = 4
    // Q[0] = [1,0,0,0], K[0] = [1,0,0,0], K[1] = [0,1,0,0]
    let q = Tensor::new(&[1.0, 0.0, 0.0, 0.0], &[1, 1, 1, 4]);
    let k = Tensor::new(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], &[1, 1, 2, 4]);
    let v = Tensor::new(&[10.0, 20.0, 30.0, 40.0], &[1, 1, 2, 2]);

    let (output, attn_weights) = scaled_dot_product_attention(&q, &k, &v, None, 0.0, false);

    // With 1/√4 = 0.5 scaling:
    // scores = [dot(Q,K0)*0.5, dot(Q,K1)*0.5] = [0.5, 0.0]
    // softmax([0.5, 0.0]) = [exp(0.5)/(exp(0.5)+1), 1/(exp(0.5)+1)]
    let e_half = 0.5_f32.exp();
    let w0 = e_half / (e_half + 1.0);
    let w1 = 1.0 / (e_half + 1.0);

    let weights = attn_weights.data();
    assert!(
        (weights[0] - w0).abs() < 1e-5,
        "FALSIFIED ATT-003: weight[0] = {}, expected {w0} (1/√d_k scaling)",
        weights[0]
    );
    assert!(
        (weights[1] - w1).abs() < 1e-5,
        "FALSIFIED ATT-003: weight[1] = {}, expected {w1} (1/√d_k scaling)",
        weights[1]
    );

    // Verify output matches weighted V
    let ref_out_0 = w0 * 10.0 + w1 * 30.0;
    let ref_out_1 = w0 * 20.0 + w1 * 40.0;
    let out_data = output.data();
    assert!(
        (out_data[0] - ref_out_0).abs() < 1e-4,
        "FALSIFIED ATT-003: output[0] = {}, expected {ref_out_0}",
        out_data[0]
    );
    assert!(
        (out_data[1] - ref_out_1).abs() < 1e-4,
        "FALSIFIED ATT-003: output[1] = {}, expected {ref_out_1}",
        out_data[1]
    );
}

/// FALSIFY-ATT-005: Weights bounded — all attention weights in (0, 1)
///
/// Contract: 0 < attn_{ij} < 1 for all i,j when m >= 2
#[test]
fn falsify_att_005_weights_bounded() {
    let q = Tensor::new(
        &[
            1.0, 0.5, -0.3, 0.8, -1.0, 0.2, 0.7, -0.5, 0.4, -0.6, 0.3, 0.9,
        ],
        &[1, 1, 3, 4],
    );
    let k = Tensor::new(
        &[
            0.3, -0.7, 1.0, 0.2, -0.5, 0.8, 0.1, -0.3, 0.6, -0.1, 0.4, 0.9,
        ],
        &[1, 1, 3, 4],
    );
    let v = Tensor::new(
        &[
            2.0, -3.0, 5.0, 1.0, -1.0, 4.0, -2.0, 7.0, 3.0, 0.0, -4.0, 6.0,
        ],
        &[1, 1, 3, 4],
    );

    let (_, attn_weights) = scaled_dot_product_attention(&q, &k, &v, None, 0.0, false);
    let data = attn_weights.data();

    // 3x3 attention matrix, all weights should be in (0, 1)
    for i in 0..9 {
        let w = data[i];
        assert!(w > 0.0, "FALSIFIED ATT-005: weight[{i}] = {w} not > 0");
        assert!(w < 1.0, "FALSIFIED ATT-005: weight[{i}] = {w} not < 1");
    }
}

mod att_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-ATT-001-prop: Weight normalization for random Q/K
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn falsify_att_001_prop_weight_normalization(
            seq in 2..=6usize,
            d_k in (1..=4u32).prop_map(|d| (d * 2) as usize),
            seed in 0..1000u32,
        ) {
            let q_data: Vec<f32> = (0..seq * d_k)
                .map(|i| ((i as f32 + seed as f32) * 0.37).sin())
                .collect();
            let k_data: Vec<f32> = (0..seq * d_k)
                .map(|i| ((i as f32 + seed as f32) * 0.73).cos())
                .collect();
            let v_data = vec![1.0; seq * d_k];

            let q = Tensor::new(&q_data, &[1, 1, seq, d_k]);
            let k = Tensor::new(&k_data, &[1, 1, seq, d_k]);
            let v = Tensor::new(&v_data, &[1, 1, seq, d_k]);

            let (_, attn_weights) = scaled_dot_product_attention(&q, &k, &v, None, 0.0, false);
            let data = attn_weights.data();

            for i in 0..seq {
                let row_sum: f32 = (0..seq).map(|j| data[i * seq + j]).sum();
                prop_assert!(
                    (row_sum - 1.0).abs() < 1e-4,
                    "FALSIFIED ATT-001-prop: row {} sum = {}, expected 1.0",
                    i, row_sum
                );
            }
        }
    }

    /// FALSIFY-ATT-005-prop: Weights bounded in (0, 1)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn falsify_att_005_prop_weights_bounded(
            seq in 2..=6usize,
            seed in 0..1000u32,
        ) {
            let d_k = 4;
            let q_data: Vec<f32> = (0..seq * d_k)
                .map(|i| ((i as f32 + seed as f32) * 0.37).sin())
                .collect();
            let k_data: Vec<f32> = (0..seq * d_k)
                .map(|i| ((i as f32 + seed as f32) * 0.73).cos())
                .collect();
            let v_data = vec![1.0; seq * d_k];

            let q = Tensor::new(&q_data, &[1, 1, seq, d_k]);
            let k = Tensor::new(&k_data, &[1, 1, seq, d_k]);
            let v = Tensor::new(&v_data, &[1, 1, seq, d_k]);

            let (_, attn_weights) = scaled_dot_product_attention(&q, &k, &v, None, 0.0, false);
            for (i, &w) in attn_weights.data().iter().enumerate() {
                prop_assert!(
                    w > 0.0 && w < 1.0 + 1e-6,
                    "FALSIFIED ATT-005-prop: weight[{}] = {} outside (0,1)",
                    i, w
                );
            }
        }
    }

    /// FALSIFY-ATT-002-prop: Output convexity for random V
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn falsify_att_002_prop_output_convexity(
            seed in 0..1000u32,
        ) {
            let seq = 3;
            let d_v = 4;
            let q_data: Vec<f32> = (0..seq * d_v)
                .map(|i| ((i as f32 + seed as f32) * 0.37).sin())
                .collect();
            let k_data: Vec<f32> = (0..seq * d_v)
                .map(|i| ((i as f32 + seed as f32) * 0.73).cos())
                .collect();
            let v_data: Vec<f32> = (0..seq * d_v)
                .map(|i| ((i as f32 + seed as f32) * 1.23).sin() * 5.0)
                .collect();

            let q = Tensor::new(&q_data, &[1, 1, seq, d_v]);
            let k = Tensor::new(&k_data, &[1, 1, seq, d_v]);
            let v = Tensor::new(&v_data, &[1, 1, seq, d_v]);

            let (output, _) = scaled_dot_product_attention(&q, &k, &v, None, 0.0, false);
            let out = output.data();

            for d in 0..d_v {
                let v_min = (0..seq).map(|j| v_data[j * d_v + d]).fold(f32::INFINITY, f32::min);
                let v_max = (0..seq).map(|j| v_data[j * d_v + d]).fold(f32::NEG_INFINITY, f32::max);

                for i in 0..seq {
                    let val = out[i * d_v + d];
                    prop_assert!(
                        val >= v_min - 1e-4 && val <= v_max + 1e-4,
                        "FALSIFIED ATT-002-prop: output[{}][{}] = {} outside V [{}, {}]",
                        i, d, val, v_min, v_max
                    );
                }
            }
        }
    }
}

/// FALSIFY-ATT-002b: Uniform V identity — output equals V when all V rows identical
#[test]
fn falsify_att_002b_uniform_v_identity() {
    let v_row = [1.0, 2.0, 3.0, 4.0];
    let v_data: Vec<f32> = v_row.iter().copied().cycle().take(12).collect();

    let q = Tensor::new(
        &[
            1.0, 0.5, -0.3, 0.8, -1.0, 0.2, 0.7, -0.5, 0.4, -0.6, 0.3, 0.9,
        ],
        &[1, 1, 3, 4],
    );
    let k = Tensor::new(
        &[
            0.3, -0.7, 1.0, 0.2, -0.5, 0.8, 0.1, -0.3, 0.6, -0.1, 0.4, 0.9,
        ],
        &[1, 1, 3, 4],
    );
    let v = Tensor::new(&v_data, &[1, 1, 3, 4]);

    let (output, _) = scaled_dot_product_attention(&q, &k, &v, None, 0.0, false);
    let out_data = output.data();

    for i in 0..3 {
        for d in 0..4 {
            let diff = (out_data[i * 4 + d] - v_row[d]).abs();
            assert!(
                diff < 1e-5,
                "FALSIFIED ATT-002: uniform V output[{i}][{d}] = {}, expected {}",
                out_data[i * 4 + d],
                v_row[d]
            );
        }
    }
}
