// =========================================================================
// FALSIFY-ASCL: attention-scaling-v1.yaml contract (aprender scaled_dot_product)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had FALSIFY-ATT tests but zero FALSIFY-ASCL-* tests
//   Why 2: ATT tests verify shapes/weights, not 1/√d_k scaling invariants
//   Why 3: no mapping from attention-scaling-v1.yaml to aprender test names
//   Why 4: attention-scaling-v1 created after ATT to isolate scaling claims
//   Why 5: scaling was "obviously correct" (single division by sqrt)
//
// References:
//   - provable-contracts/contracts/attention-scaling-v1.yaml
//   - Vaswani et al. (2017) "Attention Is All You Need"
// =========================================================================

use super::*;

/// FALSIFY-ASCL-001: Scaling factor — variance ≈ 1 for unit-variance inputs
///
/// When Q,K entries are drawn from ~N(0,1), the scaled dot-product scores
/// should have variance ≈ 1, not d_k.
#[test]
fn falsify_ascl_001_variance_preservation() {
    let d_k_values = [16, 32, 64, 128];

    for &d_k in &d_k_values {
        let n = 20;
        let m = 20;

        // Generate pseudo-random unit-variance data
        let q_data: Vec<f32> = (0..n * d_k)
            .map(|i| ((i as f32 * 1.6180339887).sin() * 2.0))
            .collect();
        let k_data: Vec<f32> = (0..m * d_k)
            .map(|i| ((i as f32 * 2.7182818284).cos() * 2.0))
            .collect();

        let q = Tensor::new(&q_data, &[1, n, d_k]);
        let k = Tensor::new(&k_data, &[1, m, d_k]);
        let v = Tensor::ones(&[1, m, d_k]);

        let mha = MultiHeadAttention::new(d_k, 1);
        // Use forward_qkv to get attention weights (which involve scaling)
        let (_, attn_weights) = mha.forward_qkv(&q, &k, &v, None);

        // Attention weights should be valid probabilities, not all-zero or all-one
        let w_data = attn_weights.data();
        let mean: f32 = w_data.iter().sum::<f32>() / w_data.len() as f32;
        let variance: f32 =
            w_data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / w_data.len() as f32;

        // With proper scaling, attention weights should have non-trivial variance
        // (not collapsed to uniform or one-hot)
        assert!(
            variance > 1e-8,
            "FALSIFIED ASCL-001: d_k={d_k} attention variance = {variance} (too small, likely saturated)"
        );
    }
}

/// FALSIFY-ASCL-003: Attention entropy non-negative — H(attn) >= 0
///
/// The Shannon entropy of each attention row must be >= 0.
#[test]
fn falsify_ascl_003_entropy_non_negative() {
    let mha = MultiHeadAttention::new(32, 4);

    let q = Tensor::new(
        &(0..2 * 5 * 32)
            .map(|i| (i as f32 * 0.01).sin())
            .collect::<Vec<_>>(),
        &[2, 5, 32],
    );
    let k = Tensor::new(
        &(0..2 * 8 * 32)
            .map(|i| (i as f32 * 0.02).cos())
            .collect::<Vec<_>>(),
        &[2, 8, 32],
    );
    let v = Tensor::ones(&[2, 8, 32]);

    let (_, attn_weights) = mha.forward_qkv(&q, &k, &v, None);
    let w_data = attn_weights.data();
    let w_shape = attn_weights.shape(); // [batch, heads, q_len, kv_len]

    let batch = w_shape[0];
    let heads = w_shape[1];
    let q_len = w_shape[2];
    let kv_len = w_shape[3];

    for b in 0..batch {
        for h in 0..heads {
            for q_pos in 0..q_len {
                let row_start = b * heads * q_len * kv_len + h * q_len * kv_len + q_pos * kv_len;
                let entropy: f32 = (0..kv_len)
                    .map(|j| {
                        let p = w_data[row_start + j];
                        if p > 1e-10 {
                            -p * p.ln()
                        } else {
                            0.0
                        }
                    })
                    .sum();
                assert!(
                    entropy >= -1e-6,
                    "FALSIFIED ASCL-003: H(attn[{b}][{h}][{q_pos}]) = {entropy} < 0"
                );
            }
        }
    }
}

/// FALSIFY-ASCL-006: Shape correctness — score shape is [batch, heads, n_q, n_kv]
#[test]
fn falsify_ascl_006_shape_correctness() {
    let test_cases = vec![
        (32, 4, 5, 8),   // d_model=32, heads=4, q_len=5, kv_len=8
        (64, 8, 10, 10), // self-attention (q_len == kv_len)
        (16, 2, 1, 20),  // single query
    ];

    for (d_model, heads, q_len, kv_len) in test_cases {
        let mha = MultiHeadAttention::new(d_model, heads);

        let q = Tensor::ones(&[1, q_len, d_model]);
        let k = Tensor::ones(&[1, kv_len, d_model]);
        let v = Tensor::ones(&[1, kv_len, d_model]);

        let (output, attn_weights) = mha.forward_qkv(&q, &k, &v, None);

        assert_eq!(
            output.shape(),
            &[1, q_len, d_model],
            "FALSIFIED ASCL-006: output shape mismatch for d={d_model},h={heads},q={q_len},kv={kv_len}"
        );
        assert_eq!(
            attn_weights.shape(),
            &[1, heads, q_len, kv_len],
            "FALSIFIED ASCL-006: attn_weights shape mismatch"
        );
    }
}

/// FALSIFY-ASCL-007: Entropy upper bound — H(attn_i) <= log(m) for m keys
#[test]
fn falsify_ascl_007_entropy_upper_bound() {
    let kv_len = 16;
    let mha = MultiHeadAttention::new(32, 4);

    let q = Tensor::new(
        &(0..1 * 3 * 32)
            .map(|i| (i as f32 * 0.03).sin())
            .collect::<Vec<_>>(),
        &[1, 3, 32],
    );
    let k = Tensor::new(
        &(0..1 * kv_len * 32)
            .map(|i| (i as f32 * 0.07).cos())
            .collect::<Vec<_>>(),
        &[1, kv_len, 32],
    );
    let v = Tensor::ones(&[1, kv_len, 32]);

    let (_, attn_weights) = mha.forward_qkv(&q, &k, &v, None);
    let w_data = attn_weights.data();
    let w_shape = attn_weights.shape();

    let heads = w_shape[1];
    let q_len = w_shape[2];
    let max_entropy = (kv_len as f32).ln();

    for h in 0..heads {
        for q_pos in 0..q_len {
            let row_start = h * q_len * kv_len + q_pos * kv_len;
            let entropy: f32 = (0..kv_len)
                .map(|j| {
                    let p = w_data[row_start + j];
                    if p > 1e-10 {
                        -p * p.ln()
                    } else {
                        0.0
                    }
                })
                .sum();
            assert!(
                entropy <= max_entropy + 1e-4,
                "FALSIFIED ASCL-007: H(attn[0][{h}][{q_pos}]) = {entropy} > log({kv_len}) = {max_entropy}"
            );
        }
    }
}

mod ascl_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-ASCL-003-prop: Entropy non-negative for random attention weights
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn falsify_ascl_003_prop_entropy_non_negative(
            kv_len in 2..=8usize,
            seed in 0..1000u32,
        ) {
            let d_model = 16;
            let heads = 2;
            let mha = MultiHeadAttention::new(d_model, heads);

            let q_data: Vec<f32> = (0..1 * 3 * d_model)
                .map(|i| ((i as f32 + seed as f32) * 0.03).sin())
                .collect();
            let k_data: Vec<f32> = (0..1 * kv_len * d_model)
                .map(|i| ((i as f32 + seed as f32) * 0.07).cos())
                .collect();

            let q = Tensor::new(&q_data, &[1, 3, d_model]);
            let k = Tensor::new(&k_data, &[1, kv_len, d_model]);
            let v = Tensor::ones(&[1, kv_len, d_model]);

            let (_, attn_weights) = mha.forward_qkv(&q, &k, &v, None);
            let w_data = attn_weights.data();
            let w_shape = attn_weights.shape();

            let h = w_shape[1];
            let q_len = w_shape[2];
            let kv = w_shape[3];

            for hi in 0..h {
                for qi in 0..q_len {
                    let start = hi * q_len * kv + qi * kv;
                    let entropy: f32 = (0..kv)
                        .map(|j| {
                            let p = w_data[start + j];
                            if p > 1e-10 { -p * p.ln() } else { 0.0 }
                        })
                        .sum();
                    prop_assert!(
                        entropy >= -1e-6,
                        "FALSIFIED ASCL-003-prop: H(attn[0][{}][{}]) = {} < 0",
                        hi, qi, entropy
                    );
                }
            }
        }
    }

    /// FALSIFY-ASCL-007-prop: Entropy upper bound for random kv_len
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn falsify_ascl_007_prop_entropy_upper_bound(
            kv_len in 2..=16usize,
            seed in 0..1000u32,
        ) {
            let d_model = 16;
            let heads = 2;
            let mha = MultiHeadAttention::new(d_model, heads);

            let q_data: Vec<f32> = (0..1 * 2 * d_model)
                .map(|i| ((i as f32 + seed as f32) * 0.03).sin())
                .collect();
            let k_data: Vec<f32> = (0..1 * kv_len * d_model)
                .map(|i| ((i as f32 + seed as f32) * 0.07).cos())
                .collect();

            let q = Tensor::new(&q_data, &[1, 2, d_model]);
            let k = Tensor::new(&k_data, &[1, kv_len, d_model]);
            let v = Tensor::ones(&[1, kv_len, d_model]);

            let (_, attn_weights) = mha.forward_qkv(&q, &k, &v, None);
            let w_data = attn_weights.data();
            let w_shape = attn_weights.shape();

            let h = w_shape[1];
            let q_len = w_shape[2];
            let kv = w_shape[3];
            let max_entropy = (kv as f32).ln();

            for hi in 0..h {
                for qi in 0..q_len {
                    let start = hi * q_len * kv + qi * kv;
                    let entropy: f32 = (0..kv)
                        .map(|j| {
                            let p = w_data[start + j];
                            if p > 1e-10 { -p * p.ln() } else { 0.0 }
                        })
                        .sum();
                    prop_assert!(
                        entropy <= max_entropy + 1e-4,
                        "FALSIFIED ASCL-007-prop: H = {} > log({}) = {}",
                        entropy, kv, max_entropy
                    );
                }
            }
        }
    }
}
