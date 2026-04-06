//! Tests for trueno-tensor: tensor ops, einsum, and provable contracts.

use crate::{batch_matmul, einsum, einsum_nary, matmul, outer, trace, Tensor};

type R = Result<(), Box<dyn std::error::Error>>;

// ── Tensor basics ──────────────────────────────────────────────────

#[test]
fn test_tensor_new() -> R {
    let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    assert_eq!(t.shape(), &[2, 3]);
    assert_eq!(t.ndim(), 2);
    assert_eq!(t.len(), 6);
    assert!(!t.is_empty());
    Ok(())
}

#[test]
fn test_tensor_zeros() {
    let t = Tensor::zeros(vec![3, 4]);
    assert_eq!(t.len(), 12);
    assert!(t.data().iter().all(|&x| x == 0.0));
}

#[test]
fn test_tensor_get_set() {
    let mut t = Tensor::zeros(vec![2, 3]);
    t.set(&[1, 2], 42.0);
    assert!((t.get(&[1, 2]) - 42.0).abs() < 1e-10);
    assert!((t.get(&[0, 0]) - 0.0).abs() < 1e-10);
}

#[test]
fn test_tensor_reshape() -> R {
    let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let r = t.reshape(vec![3, 2])?;
    assert_eq!(r.shape(), &[3, 2]);
    assert_eq!(r.data(), t.data());
    Ok(())
}

#[test]
fn test_tensor_reshape_mismatch() {
    let t = Tensor::zeros(vec![2, 3]);
    assert!(t.reshape(vec![2, 4]).is_err());
}

#[test]
fn test_tensor_data_length_mismatch() {
    let result = Tensor::new(vec![2, 3], vec![1.0, 2.0]);
    assert!(result.is_err());
}

#[test]
fn test_tensor_transpose_2d() -> R {
    let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let tr = t.transpose(&[1, 0]);
    assert_eq!(tr.shape(), &[3, 2]);
    assert!((tr.get(&[0, 0]) - 1.0).abs() < 1e-10);
    assert!((tr.get(&[0, 1]) - 4.0).abs() < 1e-10);
    assert!((tr.get(&[1, 0]) - 2.0).abs() < 1e-10);
    assert!((tr.get(&[2, 1]) - 6.0).abs() < 1e-10);
    Ok(())
}

// ── Matrix multiply via einsum ─────────────────────────────────────

#[test]
fn test_matmul_2x3_3x2() -> R {
    let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let b = Tensor::new(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0])?;
    let c = matmul(&a, &b)?;
    assert_eq!(c.shape(), &[2, 2]);
    assert!((c.get(&[0, 0]) - 58.0).abs() < 1e-4);
    assert!((c.get(&[0, 1]) - 64.0).abs() < 1e-4);
    assert!((c.get(&[1, 0]) - 139.0).abs() < 1e-4);
    assert!((c.get(&[1, 1]) - 154.0).abs() < 1e-4);
    Ok(())
}

#[test]
fn test_matmul_identity() -> R {
    let eye = Tensor::new(vec![3, 3], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])?;
    let a = Tensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let c = matmul(&eye, &a)?;
    assert_eq!(c.shape(), &[3, 2]);
    for i in 0..6 {
        assert!((c.data()[i] - a.data()[i]).abs() < 1e-6);
    }
    Ok(())
}

// ── Einsum variations ──────────────────────────────────────────────

#[test]
fn test_einsum_outer_product() -> R {
    let a = Tensor::new(vec![3], vec![1.0, 2.0, 3.0])?;
    let b = Tensor::new(vec![2], vec![4.0, 5.0])?;
    let c = outer(&a, &b)?;
    assert_eq!(c.shape(), &[3, 2]);
    assert!((c.get(&[0, 0]) - 4.0).abs() < 1e-6);
    assert!((c.get(&[1, 1]) - 10.0).abs() < 1e-6);
    assert!((c.get(&[2, 0]) - 12.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn test_einsum_dot_product() -> R {
    let a = Tensor::new(vec![3], vec![1.0, 2.0, 3.0])?;
    let b = Tensor::new(vec![3], vec![4.0, 5.0, 6.0])?;
    let c = einsum("i,i->", &a, &b)?;
    assert_eq!(c.shape(), &[] as &[usize]);
    assert!((c.data()[0] - 32.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn test_einsum_trace() -> R {
    let a = Tensor::new(vec![3, 3], vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0])?;
    let t = trace(&a)?;
    assert!((t - 6.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn test_einsum_batch_matmul() -> R {
    let a = Tensor::new(vec![2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])?;
    let b = Tensor::new(vec![2, 2, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0])?;
    let c = batch_matmul(&a, &b)?;
    assert_eq!(c.shape(), &[2, 2, 2]);
    for i in 0..8 {
        assert!((c.data()[i] - a.data()[i]).abs() < 1e-6);
    }
    Ok(())
}

#[test]
fn test_einsum_3d_contraction() -> R {
    let a = Tensor::new(vec![2, 3, 4], (0..24).map(|i| i as f32).collect())?;
    let b = Tensor::new(vec![3, 4, 5], (0..60).map(|i| i as f32).collect())?;
    let c = einsum("ijk,jkl->il", &a, &b)?;
    assert_eq!(c.shape(), &[2, 5]);

    let mut expected = 0.0f32;
    for j in 0..3 {
        for k in 0..4 {
            expected += a.get(&[0, j, k]) * b.get(&[j, k, 0]);
        }
    }
    assert!((c.get(&[0, 0]) - expected).abs() < 1e-2);
    Ok(())
}

// ── Contract: matmul associativity ─────────────────────────────────

#[test]
fn test_contract_matmul_associativity() -> R {
    let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let b = Tensor::new(vec![3, 4], (1..=12).map(|i| i as f32).collect())?;
    let c = Tensor::new(vec![4, 2], (1..=8).map(|i| i as f32).collect())?;

    let ab = matmul(&a, &b)?;
    let ab_c = matmul(&ab, &c)?;

    let bc = matmul(&b, &c)?;
    let a_bc = matmul(&a, &bc)?;

    assert_eq!(ab_c.shape(), a_bc.shape());
    for i in 0..ab_c.len() {
        assert!(
            (ab_c.data()[i] - a_bc.data()[i]).abs() < 1e-1,
            "associativity violated at {i}: {} vs {}",
            ab_c.data()[i],
            a_bc.data()[i]
        );
    }
    Ok(())
}

// ── Contract: transpose involution ─────────────────────────────────

#[test]
fn test_contract_transpose_involution() -> R {
    let a = Tensor::new(vec![3, 4], (0..12).map(|i| i as f32).collect())?;
    let att = a.transpose(&[1, 0]).transpose(&[1, 0]);
    assert_eq!(att.shape(), a.shape());
    for i in 0..a.len() {
        assert!((att.data()[i] - a.data()[i]).abs() < 1e-10);
    }
    Ok(())
}

// ── Error handling ─────────────────────────────────────────────────

#[test]
fn test_einsum_no_arrow() {
    let a = Tensor::zeros(vec![2, 3]);
    let b = Tensor::zeros(vec![3, 2]);
    assert!(einsum("ij,jk", &a, &b).is_err());
}

#[test]
fn test_einsum_dimension_mismatch() {
    let a = Tensor::zeros(vec![2, 3]);
    let b = Tensor::zeros(vec![4, 2]);
    assert!(einsum("ij,jk->ik", &a, &b).is_err());
}

#[test]
fn test_einsum_label_count_mismatch() {
    let a = Tensor::zeros(vec![2, 3]);
    let b = Tensor::zeros(vec![3, 2]);
    assert!(einsum("ijk,jk->ik", &a, &b).is_err());
}

#[test]
fn test_trace_non_square() {
    let a = Tensor::zeros(vec![2, 3]);
    assert!(trace(&a).is_err());
}

// ── Proptest ───────────────────────────────────────────────────────

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_matmul_zero_row(n in 2..8usize, k in 2..8usize) {
            let a = Tensor::zeros(vec![n, k]);
            let b = Tensor::new(vec![k, n], (0..k*n).map(|i| i as f32).collect())
                .map_err(|e| TestCaseError::Fail(format!("{e}").into()))?;
            let c = matmul(&a, &b)
                .map_err(|e| TestCaseError::Fail(format!("{e}").into()))?;
            for &v in c.data() {
                prop_assert!((v).abs() < 1e-6);
            }
        }

        #[test]
        fn prop_reshape_preserves_data(m in 1..6usize, n in 1..6usize) {
            let data: Vec<f32> = (0..m*n).map(|i| i as f32).collect();
            let t = Tensor::new(vec![m, n], data.clone())
                .map_err(|e| TestCaseError::Fail(format!("{e}").into()))?;
            let r = t.reshape(vec![n, m])
                .map_err(|e| TestCaseError::Fail(format!("{e}").into()))?;
            prop_assert_eq!(r.data(), t.data());
        }
    }
}

// ── einsum_nary tests ──────────────────────────────────────────────

#[test]
fn test_einsum_nary_two_inputs() -> R {
    // einsum_nary with 2 inputs should match einsum binary
    let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let b = Tensor::new(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0])?;
    let r1 = einsum("ij,jk->ik", &a, &b)?;
    let r2 = einsum_nary("ij,jk->ik", &[&a, &b])?;
    for (v1, v2) in r1.data().iter().zip(r2.data().iter()) {
        assert!((v1 - v2).abs() < 1e-5);
    }
    Ok(())
}

#[test]
fn test_einsum_nary_three_inputs() -> R {
    // (A @ B) @ C = A @ (B @ C) for matmul chain
    // A=2×3, B=3×4, C=4×2
    let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let b =
        Tensor::new(vec![3, 4], vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0])?;
    let c = Tensor::new(vec![4, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0])?;

    let result = einsum_nary("ij,jk,kl->il", &[&a, &b, &c])?;
    assert_eq!(result.shape(), &[2, 2]);

    // Verify by manual computation: AB first, then (AB)C
    let ab = einsum("ij,jk->ik", &a, &b)?;
    let expected = einsum("ij,jk->ik", &ab, &c)?;
    for (v1, v2) in result.data().iter().zip(expected.data().iter()) {
        assert!((v1 - v2).abs() < 1e-5, "nary mismatch: {v1} vs {v2}");
    }
    Ok(())
}

#[test]
fn test_einsum_nary_single_input() -> R {
    // Single input: trace-like operation "ii->"
    let a = Tensor::new(vec![3, 3], vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0])?;
    let result = einsum_nary("ij->ij", &[&a])?;
    assert_eq!(result.shape(), &[3, 3]);
    assert!((result.get(&[0, 0]) - 1.0).abs() < 1e-5);
    assert!((result.get(&[1, 1]) - 2.0).abs() < 1e-5);
    assert!((result.get(&[2, 2]) - 3.0).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_einsum_nary_single_transpose() -> R {
    // "ij->ji" should transpose
    let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let result = einsum_nary("ij->ji", &[&a])?;
    assert_eq!(result.shape(), &[3, 2]);
    assert!((result.get(&[0, 0]) - 1.0).abs() < 1e-5);
    assert!((result.get(&[0, 1]) - 4.0).abs() < 1e-5);
    assert!((result.get(&[1, 0]) - 2.0).abs() < 1e-5);
    assert!((result.get(&[2, 0]) - 3.0).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_einsum_nary_wrong_count() {
    let a = Tensor::zeros(vec![2, 2]);
    let result = einsum_nary("ij,jk->ik", &[&a]);
    assert!(result.is_err());
}

#[test]
fn test_einsum_nary_four_inputs() -> R {
    // Chain of 4 matmuls: A(2×2) @ B(2×2) @ C(2×2) @ D(2×2)
    let a = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])?;
    let b = Tensor::new(vec![2, 2], vec![0.0, 1.0, 1.0, 0.0])?;
    let c = Tensor::new(vec![2, 2], vec![1.0, 1.0, 0.0, 1.0])?;
    let d = Tensor::new(vec![2, 2], vec![2.0, 0.0, 0.0, 2.0])?;

    let result = einsum_nary("ij,jk,kl,lm->im", &[&a, &b, &c, &d])?;

    // Manual: AB = [[2,1],[4,3]], ABC = [[2,3],[4,7]], ABCD = [[4,6],[8,14]]
    let ab = einsum("ij,jk->ik", &a, &b)?;
    let abc = einsum("ij,jk->ik", &ab, &c)?;
    let expected = einsum("ij,jk->ik", &abc, &d)?;

    for (v1, v2) in result.data().iter().zip(expected.data().iter()) {
        assert!((v1 - v2).abs() < 1e-4, "4-input mismatch: {v1} vs {v2}");
    }
    Ok(())
}

// ============================================================================
// POPPERIAN FALSIFICATION: Adversarial edge cases (GH #152)
// ============================================================================

#[test]
fn test_falsify_tensor_scalar() -> R {
    let t = Tensor::new(vec![], vec![42.0])?;
    assert_eq!(t.ndim(), 0);
    assert_eq!(t.len(), 1);
    assert!((t.data()[0] - 42.0).abs() < 1e-10);
    Ok(())
}

#[test]
fn test_falsify_tensor_1_element() -> R {
    let t = Tensor::new(vec![1, 1, 1], vec![7.0])?;
    assert_eq!(t.ndim(), 3);
    assert_eq!(t.len(), 1);
    assert!((t.get(&[0, 0, 0]) - 7.0).abs() < 1e-10);
    Ok(())
}

#[test]
fn test_falsify_matmul_1x1() -> R {
    let a = Tensor::new(vec![1, 1], vec![3.0])?;
    let b = Tensor::new(vec![1, 1], vec![7.0])?;
    let c = matmul(&a, &b)?;
    assert_eq!(c.shape(), &[1, 1]);
    assert!((c.data()[0] - 21.0).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_falsify_trace_1x1() -> R {
    let a = Tensor::new(vec![1, 1], vec![99.0])?;
    let t = trace(&a)?;
    assert!((t - 99.0).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_falsify_outer_single_elements() -> R {
    let a = Tensor::new(vec![1], vec![3.0])?;
    let b = Tensor::new(vec![1], vec![5.0])?;
    let c = outer(&a, &b)?;
    assert_eq!(c.shape(), &[1, 1]);
    assert!((c.data()[0] - 15.0).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_falsify_einsum_identity_contraction() -> R {
    // "ij,jk->ik" with identity matrices should give identity
    let eye = Tensor::new(vec![3, 3], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])?;
    let c = einsum("ij,jk->ik", &eye, &eye)?;
    assert_eq!(c.shape(), &[3, 3]);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (c.get(&[i, j]) - expected).abs() < 1e-5,
                "I×I[{i},{j}]={}, expected {expected}",
                c.get(&[i, j])
            );
        }
    }
    Ok(())
}

#[test]
fn test_falsify_transpose_3d() -> R {
    let t = Tensor::new(vec![2, 3, 4], (0..24).map(|i| i as f32).collect())?;
    let tr = t.transpose(&[2, 0, 1]);
    assert_eq!(tr.shape(), &[4, 2, 3]);
    // Double transpose back
    let back = tr.transpose(&[1, 2, 0]);
    assert_eq!(back.shape(), &[2, 3, 4]);
    for i in 0..24 {
        assert!(
            (back.data()[i] - t.data()[i]).abs() < 1e-10,
            "3D transpose roundtrip failed at {i}"
        );
    }
    Ok(())
}

#[test]
fn test_falsify_einsum_nary_empty_error() {
    let result = einsum_nary("->", &[]);
    assert!(result.is_err(), "Empty input list should error");
}

#[test]
fn test_falsify_matmul_zero_inner_dim() -> R {
    // A=2×0, B=0×3 → C=2×3 of zeros (valid but degenerate)
    let a = Tensor::new(vec![2, 0], vec![])?;
    let b = Tensor::new(vec![0, 3], vec![])?;
    let c = matmul(&a, &b)?;
    assert_eq!(c.shape(), &[2, 3]);
    for &v in c.data() {
        assert!(v.abs() < 1e-10, "Zero inner dim should give zeros: {v}");
    }
    Ok(())
}

#[test]
fn test_falsify_batch_matmul_single_batch() -> R {
    // Single batch should match regular matmul
    let a = Tensor::new(vec![1, 2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let b = Tensor::new(vec![1, 3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0])?;
    let c = batch_matmul(&a, &b)?;
    assert_eq!(c.shape(), &[1, 2, 2]);
    assert!((c.get(&[0, 0, 0]) - 58.0).abs() < 1e-3);
    assert!((c.get(&[0, 1, 1]) - 154.0).abs() < 1e-3);
    Ok(())
}
