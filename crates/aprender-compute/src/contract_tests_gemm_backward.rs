//! CPU-reference falsification tests for the GEMM backward contracts
//! (`gemm-backward-tiled-v1`, `fp16-cublas-gemm-v1`). The GPU kernels
//! themselves are not exercised here; these pin the math they must match.

use crate::activations::{f16_to_f32, f32_to_f16};
use crate::Matrix;

fn fill(n: usize, seed: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 + seed) * 0.37).sin()).collect()
}

fn naive_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] = (0..k).map(|p| a[i * k + p] * b[p * n + j]).sum();
        }
    }
    c
}

fn naive_transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = a[r * cols + c];
        }
    }
    t
}

fn rel_err(got: &[f32], want: &[f32]) -> f32 {
    let num: f32 = got.iter().zip(want).map(|(g, w)| (g - w) * (g - w)).sum();
    let den: f32 = want.iter().map(|w| w * w).sum();
    (num / den.max(1e-30)).sqrt()
}

fn mat(rows: usize, cols: usize, data: &[f32]) -> Matrix<f32> {
    Matrix::from_vec(rows, cols, data.to_vec()).expect("valid dims")
}

/// dW = X^T . dY and dX = dY . W^T agree with a naive reference (relative bound).
#[test]
fn test_gemm_backward_matches_naive_reference() {
    let (m, k, n) = (37, 29, 41);
    let x = fill(m * k, 1.0);
    let dy = fill(m * n, 2.0);
    let w = fill(k * n, 3.0);

    let dw = mat(m, k, &x).transpose().matmul(&mat(m, n, &dy)).expect("dW");
    let dw_ref = naive_matmul(&naive_transpose(&x, m, k), &dy, k, m, n);
    assert!(rel_err(dw.as_slice(), &dw_ref) < 1e-5, "dW diverges from naive");

    let dx = mat(m, n, &dy).matmul(&mat(k, n, &w).transpose()).expect("dX");
    let dx_ref = naive_matmul(&dy, &naive_transpose(&w, k, n), m, n, k);
    assert!(rel_err(dx.as_slice(), &dx_ref) < 1e-5, "dX diverges from naive");
}

/// (A^T)^T == A across tile-straddling shapes.
#[test]
fn test_transpose_involution_across_tile_sizes() {
    for (r, c) in [(1, 1), (7, 5), (16, 16), (17, 33), (64, 3)] {
        let a = fill(r * c, 4.0);
        let tt = mat(r, c, &a).transpose().transpose();
        assert_eq!(tt.as_slice(), a.as_slice(), "{r}x{c}");
    }
}

/// fp16 GEMM bound: inputs rounded to fp16 stay within 1e-2 of the fp32 result.
#[test]
fn test_fp16_gemm_within_bound_of_fp32() {
    let (m, k, n) = (32, 64, 32);
    let a = fill(m * k, 5.0);
    let b = fill(k * n, 6.0);
    let to_half =
        |v: &[f32]| -> Vec<f32> { v.iter().map(|&x| f16_to_f32(f32_to_f16(x))).collect() };

    let full = naive_matmul(&a, &b, m, k, n);
    let half = naive_matmul(&to_half(&a), &to_half(&b), m, k, n);
    let max_abs = full.iter().zip(&half).map(|(f, h)| (f - h).abs()).fold(0.0f32, f32::max);
    assert!(max_abs < 1e-2, "fp16 GEMM max abs error {max_abs} >= 1e-2");
}
