//! Property-based tests for attention operations

use super::test_utils::finite_difference;
use crate::autograd::{attention, backward, Tensor};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn prop_attention_backward_gradient_check_q(
        q in prop::collection::vec(-2.0f32..2.0, 4..12),
        k in prop::collection::vec(-2.0f32..2.0, 4..12),
        v in prop::collection::vec(-2.0f32..2.0, 4..12)
    ) {
        // Use small dimensions for attention to keep test fast
        let seq_len = 2;
        let d_k = 2;
        let d_v = 2;
        let total_qk = seq_len * d_k;
        let total_v = seq_len * d_v;

        let q_vec: Vec<f32> = q.into_iter().take(total_qk).collect();
        let k_vec: Vec<f32> = k.into_iter().take(total_qk).collect();
        let v_vec: Vec<f32> = v.into_iter().take(total_v).collect();

        // Skip if we don't have enough elements
        if q_vec.len() < total_qk || k_vec.len() < total_qk || v_vec.len() < total_v {
            return Ok(());
        }

        let q_tensor = Tensor::from_vec(q_vec.clone(), true);
        let k_tensor = Tensor::from_vec(k_vec.clone(), false);
        let v_tensor = Tensor::from_vec(v_vec.clone(), false);

        let mut output = attention(&q_tensor, &k_tensor, &v_tensor, seq_len, d_k, seq_len, d_v);

        backward(&mut output, Some(ndarray::Array1::ones(seq_len * d_v)));

        let analytical = q_tensor.grad().expect("gradient should be available");
        let numerical = finite_difference(
            |q_val| {
                let qt = Tensor::from_vec(q_val.to_vec(), false);
                let kt = Tensor::from_vec(k_vec.clone(), false);
                let vt = Tensor::from_vec(v_vec.clone(), false);
                let att = attention(&qt, &kt, &vt, seq_len, d_k, seq_len, d_v);
                att.data().sum()
            },
            &q_vec,
            1e-3,
        );

        for i in 0..total_qk {
            let diff = (analytical[i] - numerical[i]).abs();
            prop_assert!(diff < 0.2,
                "Attention gradient (Q) mismatch at index {}: analytical={}, numerical={}, diff={}",
                i, analytical[i], numerical[i], diff);
        }
    }

    #[test]
    fn prop_attention_backward_gradient_check_k(
        q in prop::collection::vec(-2.0f32..2.0, 4..12),
        k in prop::collection::vec(-2.0f32..2.0, 4..12),
        v in prop::collection::vec(-2.0f32..2.0, 4..12)
    ) {
        let seq_len = 2;
        let d_k = 2;
        let d_v = 2;
        let total_qk = seq_len * d_k;
        let total_v = seq_len * d_v;

        let q_vec: Vec<f32> = q.into_iter().take(total_qk).collect();
        let k_vec: Vec<f32> = k.into_iter().take(total_qk).collect();
        let v_vec: Vec<f32> = v.into_iter().take(total_v).collect();

        if q_vec.len() < total_qk || k_vec.len() < total_qk || v_vec.len() < total_v {
            return Ok(());
        }

        let q_tensor = Tensor::from_vec(q_vec.clone(), false);
        let k_tensor = Tensor::from_vec(k_vec.clone(), true);
        let v_tensor = Tensor::from_vec(v_vec.clone(), false);

        let mut output = attention(&q_tensor, &k_tensor, &v_tensor, seq_len, d_k, seq_len, d_v);

        backward(&mut output, Some(ndarray::Array1::ones(seq_len * d_v)));

        let analytical = k_tensor.grad().expect("gradient should be available");
        let numerical = finite_difference(
            |k_val| {
                let qt = Tensor::from_vec(q_vec.clone(), false);
                let kt = Tensor::from_vec(k_val.to_vec(), false);
                let vt = Tensor::from_vec(v_vec.clone(), false);
                let att = attention(&qt, &kt, &vt, seq_len, d_k, seq_len, d_v);
                att.data().sum()
            },
            &k_vec,
            1e-3,
        );

        for i in 0..total_qk {
            let diff = (analytical[i] - numerical[i]).abs();
            prop_assert!(diff < 0.2,
                "Attention gradient (K) mismatch at index {}: analytical={}, numerical={}, diff={}",
                i, analytical[i], numerical[i], diff);
        }
    }

    #[test]
    fn prop_attention_backward_gradient_check_v(
        q in prop::collection::vec(-2.0f32..2.0, 4..12),
        k in prop::collection::vec(-2.0f32..2.0, 4..12),
        v in prop::collection::vec(-2.0f32..2.0, 4..12)
    ) {
        let seq_len = 2;
        let d_k = 2;
        let d_v = 2;
        let total_qk = seq_len * d_k;
        let total_v = seq_len * d_v;

        let q_vec: Vec<f32> = q.into_iter().take(total_qk).collect();
        let k_vec: Vec<f32> = k.into_iter().take(total_qk).collect();
        let v_vec: Vec<f32> = v.into_iter().take(total_v).collect();

        if q_vec.len() < total_qk || k_vec.len() < total_qk || v_vec.len() < total_v {
            return Ok(());
        }

        let q_tensor = Tensor::from_vec(q_vec.clone(), false);
        let k_tensor = Tensor::from_vec(k_vec.clone(), false);
        let v_tensor = Tensor::from_vec(v_vec.clone(), true);

        let mut output = attention(&q_tensor, &k_tensor, &v_tensor, seq_len, d_k, seq_len, d_v);

        backward(&mut output, Some(ndarray::Array1::ones(seq_len * d_v)));

        let analytical = v_tensor.grad().expect("gradient should be available");
        let numerical = finite_difference(
            |v_val| {
                let qt = Tensor::from_vec(q_vec.clone(), false);
                let kt = Tensor::from_vec(k_vec.clone(), false);
                let vt = Tensor::from_vec(v_val.to_vec(), false);
                let att = attention(&qt, &kt, &vt, seq_len, d_k, seq_len, d_v);
                att.data().sum()
            },
            &v_vec,
            1e-3,
        );

        for i in 0..total_v {
            let diff = (analytical[i] - numerical[i]).abs();
            prop_assert!(diff < 0.2,
                "Attention gradient (V) mismatch at index {}: analytical={}, numerical={}, diff={}",
                i, analytical[i], numerical[i], diff);
        }
    }
}
