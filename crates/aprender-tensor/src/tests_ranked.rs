//! Tests for the rank-typed tensor (#3150). The compile-fail half of the
//! contract is in `tests/rank_typed_ui.rs`.

use crate::{
    einsum, matmul, ColMajor, Layout, Matrix, RankedTensor, RowMajor, Tensor, TensorError,
};
use proptest::prelude::*;

type R = Result<(), TensorError>;

fn ramp(n: usize) -> Vec<f32> {
    (0..n).map(|i| i as f32 * 0.5 - 3.0).collect()
}

// ── the migrated hot path: `matmul` now runs on `Matrix::matmul` ────────────

proptest! {
    /// The typed kernel returns what the einsum route returned, bit for bit on
    /// these small integer-valued inputs, including zero-sized dimensions.
    #[test]
    fn matmul_matches_the_einsum_route(m in 0..7usize, k in 0..7usize, n in 0..7usize) {
        let a = Tensor::new(vec![m, k], ramp(m * k)).expect("a");
        let b = Tensor::new(vec![k, n], ramp(k * n).into_iter().rev().collect()).expect("b");
        let typed = matmul(&a, &b).expect("typed");
        let reference = einsum("ij,jk->ik", &a, &b).expect("einsum");
        prop_assert_eq!(typed.shape(), reference.shape());
        prop_assert_eq!(typed.data(), reference.data());
    }
}

/// Callers matching on the error see the same variant and text as before.
#[test]
fn matmul_errors_are_the_einsum_errors() {
    let rank3 = Tensor::zeros(vec![2, 2, 2]);
    let m22 = Tensor::zeros(vec![2, 2]);
    let m32 = Tensor::zeros(vec![3, 2]);
    for (a, b) in [(&rank3, &m22), (&m22, &rank3), (&m22, &m32)] {
        let typed = format!("{:?}", matmul(a, b).expect_err("typed must fail"));
        let reference = format!(
            "{:?}",
            einsum("ij,jk->ik", a, b).expect_err("einsum must fail")
        );
        assert_eq!(typed, reference);
    }
}

#[test]
fn matrix_matmul_known_values_and_transpose() -> R {
    let a = Matrix::new([2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let b = Matrix::new([3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0])?;
    let c = a.matmul(&b)?;
    assert_eq!(c.shape(), [2, 2]);
    assert_eq!(c.data(), &[58.0, 64.0, 139.0, 154.0]);
    // (AB)ᵀ = BᵀAᵀ
    assert_eq!(c.t(), b.t().matmul(&a.t())?);
    Ok(())
}

// ── dynamic boundary ────────────────────────────────────────────────────────

#[test]
fn from_dynamic_checks_rank_and_round_trips() -> R {
    let t = Tensor::new(vec![2, 3], ramp(6))?;
    assert!(matches!(
        RankedTensor::<3>::from_dynamic(&t),
        Err(TensorError::RankMismatch {
            expected: 3,
            got: 2
        })
    ));
    let m = Matrix::from_dynamic(&t)?;
    let back = m.into_dynamic();
    assert_eq!(back.shape(), t.shape());
    assert_eq!(back.data(), t.data());
    assert_eq!(back.get(&[1, 2]), t.get(&[1, 2]));
    Ok(())
}

#[test]
fn new_rejects_a_wrong_length() {
    assert!(matches!(
        RankedTensor::<2>::new([2, 3], vec![0.0; 5]),
        Err(TensorError::DataLengthMismatch {
            len: 5,
            product: 6,
            ..
        })
    ));
}

// ── rank-changing operations ────────────────────────────────────────────────

#[test]
fn reshape_unsqueeze_squeeze_move_the_rank_in_the_type() -> R {
    let v = RankedTensor::<1>::new([6], ramp(6))?;
    let m: Matrix = v.clone().reshape([2, 3])?;
    assert_eq!(m.get([1, 0]), v.get([3]));
    let t: RankedTensor<3> = m.clone().unsqueeze(1)?;
    assert_eq!(t.shape(), [2, 1, 3]);
    assert_eq!(t.get([1, 0, 2]), m.get([1, 2]));
    let back: Matrix = t.squeeze(1)?;
    assert_eq!(back, m);
    let front: RankedTensor<3> = m.clone().unsqueeze(0)?;
    assert_eq!(front.shape(), [1, 2, 3]);
    let end: RankedTensor<3> = m.clone().unsqueeze(2)?;
    assert_eq!(end.shape(), [2, 3, 1]);
    let scalar = RankedTensor::<0>::new([], vec![4.0])?;
    assert_eq!(scalar.unsqueeze(0)?.shape(), [1]);
    Ok(())
}

#[test]
fn rank_changes_reject_bad_axes() {
    let m = Matrix::zeros([2, 3]);
    assert!(m.clone().unsqueeze(3).is_err(), "axis past the end");
    assert!(m.clone().squeeze(0).is_err(), "axis of size 2");
    let t = RankedTensor::<3>::zeros([2, 1, 3]);
    assert!(t.squeeze(3).is_err(), "axis out of range");
    assert!(m.reshape([4, 2]).is_err(), "6 elements into 8");
}

// ── layout ──────────────────────────────────────────────────────────────────

#[test]
fn col_major_indexing_and_the_two_ways_out() -> R {
    // The matrix [[1, 2, 3], [4, 5, 6]] stored column by column.
    let cm = RankedTensor::<2, ColMajor>::new([2, 3], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])?;
    assert_eq!(cm.get([0, 2]), 3.0);
    assert_eq!(cm.get([1, 0]), 4.0);

    // Same logical matrix, row-major storage.
    let rm = cm.to_row_major();
    assert_eq!(rm.shape(), [2, 3]);
    assert_eq!(rm.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // The GGUF boundary: same bytes, reversed shape, i.e. the transpose.
    let apr = cm.clone().into_apr();
    assert_eq!(apr.shape(), [3, 2]);
    assert_eq!(apr.data(), cm.data());
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(apr.get([j, i]), cm.get([i, j]));
        }
    }
    assert_eq!(apr, rm.t());
    Ok(())
}

/// Symbolic dimension names from the contract (`hidden`, `heads*head_dim`, ...)
/// get distinct sizes so a swapped pair cannot pass.
fn dim_value(name: &str) -> usize {
    name.bytes()
        .fold(7usize, |h, b| (h * 31 + usize::from(b)) % 997)
        + 2
}

fn parse_shape(value: &str) -> Vec<String> {
    value
        .trim()
        .trim_matches('\'')
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split(',')
        .map(|s| s.trim().to_string())
        .collect()
}

/// The markers are wired to `contracts/tensor-layout-v1.yaml`: the format
/// names match, and for every `transpose: 'true'` row, `into_apr` turns the
/// GGUF shape into the APR shape the contract states.
#[test]
fn layout_markers_follow_the_tensor_layout_contract() -> R {
    let contract = include_str!("../../../contracts/tensor-layout-v1.yaml");
    let lines: Vec<&str> = contract.lines().collect();
    let layout_of = |format: &str| {
        let start = lines
            .iter()
            .position(|l| l.trim_end() == format!("  {format}:"))
            .unwrap_or_else(|| panic!("formats.{format} missing from the contract"));
        lines[start + 1]
            .trim()
            .strip_prefix("layout: ")
            .unwrap_or_else(|| panic!("formats.{format}.layout missing"))
            .to_string()
    };
    assert_eq!(layout_of("gguf"), ColMajor::NAME);
    assert_eq!(layout_of("apr"), RowMajor::NAME);
    assert_eq!(layout_of("safetensors"), RowMajor::NAME);

    let mut checked = 0;
    for (i, line) in lines.iter().enumerate() {
        let Some(gguf) = line.trim().strip_prefix("gguf_shape: ") else {
            continue;
        };
        let apr = lines[i + 1]
            .trim()
            .strip_prefix("apr_shape: ")
            .expect("apr_shape follows");
        let transpose = lines[i + 2]
            .trim()
            .strip_prefix("transpose: ")
            .expect("transpose follows");
        let (gguf, apr) = (parse_shape(gguf), parse_shape(apr));
        if transpose.trim_matches('\'') != "true" || gguf.len() != 2 {
            continue;
        }
        let ne = [dim_value(&gguf[0]), dim_value(&gguf[1])];
        let t = RankedTensor::<2, ColMajor>::from_gguf(ne, ramp(ne[0] * ne[1]))?;
        let expect = [dim_value(&apr[0]), dim_value(&apr[1])];
        assert_eq!(
            t.into_apr().shape(),
            expect,
            "contract row gguf {gguf:?} -> apr {apr:?}"
        );
        checked += 1;
    }
    assert!(
        checked >= 5,
        "parsed only {checked} transpose rows; fix the parser before trusting this"
    );
    Ok(())
}
