//! Attention autograd operations: scaled dot-product attention
//!
//! Uses CUDA GEMM for Q@K^T and Attn@V operations when available.

use crate::autograd::{BackwardOp, Tensor};
use ndarray::Array1;
use std::cell::RefCell;
use std::rc::Rc;

// Import matmul_compute from sibling module for GPU-accelerated matrix operations
use super::matmul::{matmul_compute, transpose};

/// Scaled Dot-Product Attention (GPU-accelerated)
///
/// Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
///
/// Parameters:
/// - q: Query matrix (seq_len x d_k, stored flattened)
/// - k: Key matrix (seq_len x d_k, stored flattened)
/// - v: Value matrix (seq_len x d_v, stored flattened)
/// - seq_len: Sequence length
/// - d_k: Dimension of queries and keys
/// - d_v: Dimension of values
///
/// Returns: Tensor of shape (seq_len x d_v, stored flattened)
pub fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seq_len: usize,
    d_k: usize,
    _k_seq_len: usize, // Kept for API compatibility, assumes same as seq_len
    d_v: usize,
) -> Tensor {
    let scale = (d_k as f32).sqrt();

    // Step 1: Compute Q @ K^T (seq_len x seq_len) using GPU GEMM
    // Q is (seq_len, d_k), K is (seq_len, d_k), K^T is (d_k, seq_len)
    // Result: (seq_len, d_k) @ (d_k, seq_len) = (seq_len, seq_len)
    let q_slice = q.data().as_slice().unwrap_or(&[]);
    let k_slice = k.data().as_slice().unwrap_or(&[]);
    let k_t = transpose(k_slice, seq_len, d_k); // K^T: (d_k, seq_len)
    let mut scores = matmul_compute(q_slice, &k_t, seq_len, d_k, seq_len);

    // Apply scaling
    for score in &mut scores {
        *score /= scale;
    }

    // Step 2: Apply softmax row-wise (CPU for numerical stability)
    let mut attention_weights = vec![0.0; seq_len * seq_len];
    for i in 0..seq_len {
        let row_start = i * seq_len;
        let row_end = row_start + seq_len;
        let row = &scores[row_start..row_end];

        // Softmax for numerical stability
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
        let sum_exp: f32 = exp_vals.iter().sum();

        for (j, &exp_val) in exp_vals.iter().enumerate() {
            attention_weights[row_start + j] = exp_val / sum_exp;
        }
    }

    // Step 3: Compute attention_weights @ V (seq_len x d_v) using GPU GEMM
    // attention_weights is (seq_len, seq_len), V is (seq_len, d_v)
    // Result: (seq_len, seq_len) @ (seq_len, d_v) = (seq_len, d_v)
    let v_slice = v.data().as_slice().unwrap_or(&[]);
    let output_data = matmul_compute(&attention_weights, v_slice, seq_len, seq_len, d_v);

    let requires_grad = q.requires_grad() || k.requires_grad() || v.requires_grad();
    let mut result = Tensor::new(Array1::from(output_data), requires_grad);

    if requires_grad {
        let q_clone = q.clone();
        let k_clone = k.clone();
        let v_clone = v.clone();
        let backward_op = Rc::new(AttentionBackward {
            q: q_clone,
            k: k_clone,
            v: v_clone,
            attention_weights: Array1::from(attention_weights),
            seq_len,
            d_k,
            d_v,
            scale,
            result_grad: result.grad_cell(),
        });
        result.set_backward_op(backward_op);
    }

    result
}

struct AttentionBackward {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attention_weights: Array1<f32>,
    seq_len: usize,
    d_k: usize,
    d_v: usize,
    scale: f32,
    result_grad: Rc<RefCell<Option<Array1<f32>>>>,
}

impl BackwardOp for AttentionBackward {
    fn backward(&self) {
        if let Some(grad_output) = self.result_grad.borrow().as_ref() {
            let seq_len = self.seq_len;
            let d_k = self.d_k;
            let d_v = self.d_v;
            let grad_out_slice = grad_output.as_slice().unwrap_or(&[]);
            let attn_slice = self.attention_weights.as_slice().unwrap_or(&[]);

            // Gradient w.r.t. V: attention_weights^T @ grad_output
            // attention_weights is (seq_len, seq_len), grad_output is (seq_len, d_v)
            // attention_weights^T is (seq_len, seq_len)
            // Result: (seq_len, seq_len) @ (seq_len, d_v) = (seq_len, d_v)
            if self.v.requires_grad() {
                let attn_t = transpose(attn_slice, seq_len, seq_len);
                let grad_v = matmul_compute(&attn_t, grad_out_slice, seq_len, seq_len, d_v);
                self.v.accumulate_grad(Array1::from(grad_v));
            }

            // Gradient w.r.t. attention_weights: grad_output @ V^T
            // grad_output is (seq_len, d_v), V is (seq_len, d_v), V^T is (d_v, seq_len)
            // Result: (seq_len, d_v) @ (d_v, seq_len) = (seq_len, seq_len)
            let v_slice = self.v.data().as_slice().unwrap_or(&[]);
            let v_t = transpose(v_slice, seq_len, d_v);
            let grad_attention_weights =
                matmul_compute(grad_out_slice, &v_t, seq_len, d_v, seq_len);

            // Gradient through softmax (row-wise) - must be CPU for numerical stability
            let mut grad_scores = vec![0.0; seq_len * seq_len];
            for i in 0..seq_len {
                let row_start = i * seq_len;
                for j in 0..seq_len {
                    let idx = row_start + j;
                    let p_j = attn_slice[idx];

                    // Softmax gradient: p_j * (grad_j - sum_k(p_k * grad_k))
                    let mut sum_pk_gradk = 0.0;
                    for k in 0..seq_len {
                        let k_idx = row_start + k;
                        sum_pk_gradk += attn_slice[k_idx] * grad_attention_weights[k_idx];
                    }

                    grad_scores[idx] = p_j * (grad_attention_weights[idx] - sum_pk_gradk);
                }
            }

            // Gradient through scaling
            for g in &mut grad_scores {
                *g /= self.scale;
            }

            // Gradient w.r.t. Q: grad_scaled @ K
            // grad_scaled is (seq_len, seq_len), K is (seq_len, d_k)
            // Result: (seq_len, seq_len) @ (seq_len, d_k) = (seq_len, d_k)
            if self.q.requires_grad() {
                let k_slice = self.k.data().as_slice().unwrap_or(&[]);
                let grad_q = matmul_compute(&grad_scores, k_slice, seq_len, seq_len, d_k);
                self.q.accumulate_grad(Array1::from(grad_q));
            }

            // Gradient w.r.t. K: grad_scaled^T @ Q
            // grad_scaled is (seq_len, seq_len), grad_scaled^T is (seq_len, seq_len)
            // Q is (seq_len, d_k)
            // Result: (seq_len, seq_len) @ (seq_len, d_k) = (seq_len, d_k)
            if self.k.requires_grad() {
                let grad_t = transpose(&grad_scores, seq_len, seq_len);
                let q_slice = self.q.data().as_slice().unwrap_or(&[]);
                let grad_k = matmul_compute(&grad_t, q_slice, seq_len, seq_len, d_k);
                self.k.accumulate_grad(Array1::from(grad_k));
            }

            // Continue backward through the graph
            if let Some(op) = self.q.backward_op() {
                op.backward();
            }
            if let Some(op) = self.k.backward_op() {
                op.backward();
            }
            if let Some(op) = self.v.backward_op() {
                op.backward();
            }
        }
    }
}

// =========================================================================
// FALSIFY-ATT: attention-kernel-v1.yaml contract (entrenar attention)
//
// Five-Whys (PMAT-354):
//   Why 1: entrenar had zero attention tests
//   Why 2: attention was added for GPU GEMM acceleration, tested via model-level e2e
//   Why 3: no mapping from attention-kernel-v1.yaml to entrenar test names
//   Why 4: entrenar predates the provable-contracts YAML convention
//   Why 5: scaled dot-product attention was "obviously correct"
//
// References:
//   - provable-contracts/contracts/attention-kernel-v1.yaml
//   - Vaswani et al. (2017) "Attention Is All You Need"
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    /// FALSIFY-ATT-001: Weight normalization (indirect) — uniform V → output equals V
    ///
    /// If all V rows are identical [c, c, ...], any convex combination gives [c, c, ...].
    /// This implies the weights summed to 1.0.
    #[test]
    fn falsify_att_001_weight_normalization_via_uniform_v() {
        let seq_len = 3;
        let d_k = 4;
        let d_v = 4;
        let v_row = vec![2.0, -1.0, 3.0, 0.5];
        let v_data: Vec<f32> = v_row.iter().copied().cycle().take(seq_len * d_v).collect();

        let q = Tensor::new(
            Array1::from(vec![1.0, 0.5, -0.3, 0.8, -1.0, 0.2, 0.7, -0.5, 0.4, -0.6, 0.3, 0.9]),
            false,
        );
        let k = Tensor::new(
            Array1::from(vec![0.3, -0.7, 1.0, 0.2, -0.5, 0.8, 0.1, -0.3, 0.6, -0.1, 0.4, 0.9]),
            false,
        );
        let v = Tensor::new(Array1::from(v_data), false);

        let output = attention(&q, &k, &v, seq_len, d_k, seq_len, d_v);
        let out_data = output.data();
        let out_slice = out_data.as_slice().expect("contiguous");

        for i in 0..seq_len {
            for d in 0..d_v {
                let diff = (out_slice[i * d_v + d] - v_row[d]).abs();
                assert!(
                    diff < 1e-4,
                    "FALSIFIED ATT-001: output[{i}][{d}] = {}, expected {} (uniform V → weights sum to 1)",
                    out_slice[i * d_v + d],
                    v_row[d]
                );
            }
        }
    }

    /// FALSIFY-ATT-002: Output convexity — output bounded by min/max of V columns
    ///
    /// Contract: min_j(V[j][d]) ≤ output[i][d] ≤ max_j(V[j][d])
    #[test]
    fn falsify_att_002_output_convexity() {
        let seq_len = 3;
        let d_k = 4;
        let d_v = 4;
        let v_data = vec![2.0, -3.0, 5.0, 1.0, -1.0, 4.0, -2.0, 7.0, 3.0, 0.0, -4.0, 6.0];

        let q = Tensor::new(
            Array1::from(vec![1.0, 0.5, -0.3, 0.8, -1.0, 0.2, 0.7, -0.5, 0.4, -0.6, 0.3, 0.9]),
            false,
        );
        let k = Tensor::new(
            Array1::from(vec![0.3, -0.7, 1.0, 0.2, -0.5, 0.8, 0.1, -0.3, 0.6, -0.1, 0.4, 0.9]),
            false,
        );
        let v = Tensor::new(Array1::from(v_data.clone()), false);

        let output = attention(&q, &k, &v, seq_len, d_k, seq_len, d_v);
        let out_data = output.data();
        let out_slice = out_data.as_slice().expect("contiguous");

        for i in 0..seq_len {
            for d in 0..d_v {
                let out_val = out_slice[i * d_v + d];

                let v_col_min =
                    (0..seq_len).map(|j| v_data[j * d_v + d]).fold(f32::INFINITY, f32::min);
                let v_col_max =
                    (0..seq_len).map(|j| v_data[j * d_v + d]).fold(f32::NEG_INFINITY, f32::max);

                assert!(
                    out_val >= v_col_min - 1e-4 && out_val <= v_col_max + 1e-4,
                    "FALSIFIED ATT-002: output[{i}][{d}] = {out_val} outside V column [{v_col_min}, {v_col_max}]"
                );
            }
        }
    }

    /// FALSIFY-ATT-003: Scaling factor — uses 1/√d_k not 1/d_k
    ///
    /// With d_k=1, both scalings are identical (1/√1 = 1/1 = 1).
    /// With d_k=4, 1/√4 = 0.5 but 1/4 = 0.25 — outputs differ.
    /// We verify by comparing attention output against a manual reference.
    #[test]
    fn falsify_att_003_scaling_factor() {
        let seq_len = 2;
        let d_k = 4;
        let d_v = 2;

        let q_data = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let k_data = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let v_data = vec![10.0, 20.0, 30.0, 40.0];

        let q = Tensor::new(Array1::from(q_data.clone()), false);
        let k = Tensor::new(Array1::from(k_data.clone()), false);
        let v = Tensor::new(Array1::from(v_data.clone()), false);

        let output = attention(&q, &k, &v, seq_len, d_k, seq_len, d_v);
        let out_slice = output.data().as_slice().expect("contiguous").to_vec();

        // Manual reference with correct 1/√d_k scaling
        let scale = (d_k as f32).sqrt(); // 2.0
                                         // Q[0] = [1,0,0,0], K[0] = [1,0,0,0], K[1] = [0,0,1,0]
                                         // scores[0] = [dot(Q0,K0)/scale, dot(Q0,K1)/scale] = [1.0/2.0, 0.0/2.0] = [0.5, 0.0]
        let s00 = 1.0 / scale;
        let s01 = 0.0 / scale;
        let max0 = s00.max(s01);
        let e00 = (s00 - max0).exp();
        let e01 = (s01 - max0).exp();
        let sum0 = e00 + e01;
        let w00 = e00 / sum0;
        let w01 = e01 / sum0;
        let ref_out_0_0 = w00 * v_data[0] + w01 * v_data[2];
        let ref_out_0_1 = w00 * v_data[1] + w01 * v_data[3];

        assert!(
            (out_slice[0] - ref_out_0_0).abs() < 1e-4,
            "FALSIFIED ATT-003: output[0][0] = {}, reference = {ref_out_0_0} (1/√d_k scaling)",
            out_slice[0]
        );
        assert!(
            (out_slice[1] - ref_out_0_1).abs() < 1e-4,
            "FALSIFIED ATT-003: output[0][1] = {}, reference = {ref_out_0_1} (1/√d_k scaling)",
            out_slice[1]
        );
    }

    /// FALSIFY-ATT-005: Single position — softmax of single score is 1.0, output = V
    #[test]
    fn falsify_att_005_single_position() {
        let seq_len = 1;
        let d_k = 4;
        let d_v = 4;
        let v_data = vec![7.0, -3.0, 2.5, 11.0];

        let q = Tensor::new(Array1::from(vec![1.0, 0.0, 0.0, 0.0]), false);
        let k = Tensor::new(Array1::from(vec![0.5, 0.5, 0.5, 0.5]), false);
        let v = Tensor::new(Array1::from(v_data.clone()), false);

        let output = attention(&q, &k, &v, seq_len, d_k, seq_len, d_v);
        let out_slice = output.data().as_slice().expect("contiguous").to_vec();

        for (d, (&out_val, &v_val)) in out_slice.iter().zip(v_data.iter()).enumerate() {
            let diff = (out_val - v_val).abs();
            assert!(
                diff < 1e-5,
                "FALSIFIED ATT-005: single position output[{d}] = {out_val}, expected V[{d}] = {v_val}"
            );
        }
    }

    /// ENC-002: Verify attention is bidirectional (no causal mask).
    ///
    /// In causal attention, position 0 cannot attend to position 1+.
    /// In bidirectional attention, every position attends to every position.
    /// We verify by checking that changing a later token affects earlier outputs.
    #[test]
    fn enc_002_attention_is_bidirectional() {
        let seq_len = 3;
        let d_k = 4;
        let d_v = 4;

        let q_data = vec![1.0, 0.5, -0.3, 0.8, -1.0, 0.2, 0.7, -0.5, 0.4, -0.6, 0.3, 0.9];
        let k_data_a = vec![0.3, -0.7, 1.0, 0.2, -0.5, 0.8, 0.1, -0.3, 0.6, -0.1, 0.4, 0.9];
        let v_data = vec![10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        // Run A: original K
        let q_a = Tensor::new(Array1::from(q_data.clone()), false);
        let k_a = Tensor::new(Array1::from(k_data_a.clone()), false);
        let v_a = Tensor::new(Array1::from(v_data.clone()), false);
        let out_a = attention(&q_a, &k_a, &v_a, seq_len, d_k, seq_len, d_v);
        let slice_a = out_a.data().as_slice().expect("contiguous").to_vec();

        // Run B: modify K at position 2 (last token)
        let mut k_data_b = k_data_a;
        k_data_b[8] = 99.0; // K[2][0] = 99.0 (was 0.6)
        let q_b = Tensor::new(Array1::from(q_data), false);
        let k_b = Tensor::new(Array1::from(k_data_b), false);
        let v_b = Tensor::new(Array1::from(v_data), false);
        let out_b = attention(&q_b, &k_b, &v_b, seq_len, d_k, seq_len, d_v);
        let slice_b = out_b.data().as_slice().expect("contiguous").to_vec();

        // Position 0's output MUST change — it attends bidirectionally to position 2
        let diff_pos0: f32 = (0..d_v).map(|d| (slice_a[d] - slice_b[d]).abs()).sum();
        assert!(
            diff_pos0 > 1e-3,
            "ENC-002 FAILED: position 0 output unchanged when K[2] modified \
             (diff={diff_pos0}). Attention has causal mask — encoder requires bidirectional."
        );
    }

    mod att_proptest_falsify {
        use super::*;
        use proptest::prelude::*;

        // FALSIFY-ATT-002-prop: Output convexity for random V
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(100))]

            #[test]
            fn falsify_att_002_prop_output_convexity(
                seed in 0..1000u32,
            ) {
                let seq = 3;
                let d = 4;

                let q_data: Vec<f32> = (0..seq * d)
                    .map(|i| ((i as f32 + seed as f32) * 0.37).sin())
                    .collect();
                let k_data: Vec<f32> = (0..seq * d)
                    .map(|i| ((i as f32 + seed as f32) * 0.73).cos())
                    .collect();
                let v_data: Vec<f32> = (0..seq * d)
                    .map(|i| ((i as f32 + seed as f32) * 1.23).sin() * 5.0)
                    .collect();

                let q = Tensor::new(Array1::from(q_data), false);
                let k = Tensor::new(Array1::from(k_data), false);
                let v = Tensor::new(Array1::from(v_data.clone()), false);

                let output = attention(&q, &k, &v, seq, d, seq, d);
                let out_slice = output.data().as_slice().expect("contiguous").to_vec();

                for dim in 0..d {
                    let v_min = (0..seq).map(|j| v_data[j * d + dim]).fold(f32::INFINITY, f32::min);
                    let v_max = (0..seq).map(|j| v_data[j * d + dim]).fold(f32::NEG_INFINITY, f32::max);

                    for i in 0..seq {
                        let val = out_slice[i * d + dim];
                        prop_assert!(
                            val >= v_min - 1e-4 && val <= v_max + 1e-4,
                            "FALSIFIED ATT-002-prop: output[{}][{}] = {} outside V [{}, {}]",
                            i, dim, val, v_min, v_max
                        );
                    }
                }
            }
        }

        // FALSIFY-ATT-001-prop: Uniform V -> output equals V (weights sum to 1)
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(100))]

            #[test]
            fn falsify_att_001_prop_uniform_v(
                seq in 2..=5usize,
                seed in 0..1000u32,
            ) {
                let d = 4;
                let v_row: Vec<f32> = (0..d)
                    .map(|i| ((i as f32 + seed as f32) * 1.23).sin() * 5.0)
                    .collect();
                let v_data: Vec<f32> = v_row.iter().copied().cycle().take(seq * d).collect();

                let q_data: Vec<f32> = (0..seq * d)
                    .map(|i| ((i as f32 + seed as f32) * 0.37).sin())
                    .collect();
                let k_data: Vec<f32> = (0..seq * d)
                    .map(|i| ((i as f32 + seed as f32) * 0.73).cos())
                    .collect();

                let q = Tensor::new(Array1::from(q_data), false);
                let k = Tensor::new(Array1::from(k_data), false);
                let v = Tensor::new(Array1::from(v_data), false);

                let output = attention(&q, &k, &v, seq, d, seq, d);
                let out_slice = output.data().as_slice().expect("contiguous").to_vec();

                for i in 0..seq {
                    for dim in 0..d {
                        let diff = (out_slice[i * d + dim] - v_row[dim]).abs();
                        prop_assert!(
                            diff < 1e-4,
                            "FALSIFIED ATT-001-prop: output[{}][{}] = {}, expected {} (uniform V)",
                            i, dim, out_slice[i * d + dim], v_row[dim]
                        );
                    }
                }
            }
        }
    }
}
