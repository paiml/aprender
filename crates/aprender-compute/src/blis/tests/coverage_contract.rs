//! FALSIFY-BLIS contract enforcement tests (blis-gemm-v1).
//!
//! Targets the top coverage gaps: gemm_blis, elementwise, norms,
//! softmax, gemv. Exercises 0%-covered SIMD paths.

use super::super::*;
use crate::blis;

// ═══ Contract: blis-gemm-v1 — GEMM correctness ═══

#[test]
fn falsify_blis_001_gemm_standard_sizes() {
    for (m, n, k) in [(32, 32, 32), (64, 64, 64), (128, 64, 32), (16, 48, 24)] {
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];
        let mut c = vec![0.0f32; m * n];
        blis::gemm_blis(m, n, k, &a, &b, &mut c, None).unwrap();
        for (i, &v) in c.iter().enumerate() {
            assert!((v - k as f32).abs() < 1.0, "001: c[{i}]={v}, exp {k}");
        }
    }
}

#[test]
fn falsify_blis_002_gemm_non_aligned() {
    for (m, n, k) in [(7, 13, 5), (3, 1, 11), (1, 1, 1), (17, 23, 9)] {
        let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 * 0.1).collect();
        let mut c_blis = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        blis::gemm_blis(m, n, k, &a, &b, &mut c_blis, None).unwrap();
        reference::gemm_reference(m, n, k, &a, &b, &mut c_ref).unwrap();
        for i in 0..m * n {
            let diff = (c_blis[i] - c_ref[i]).abs();
            assert!(diff < 0.1, "002: [{i}] blis={} ref={}", c_blis[i], c_ref[i]);
        }
    }
}

#[test]
fn falsify_blis_003_elementwise_add() {
    for len in [1, 7, 8, 15, 16, 31, 32, 100, 255, 256, 1000] {
        let a: Vec<f32> = (0..len).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..len).map(|i| (len - i) as f32).collect();
        let result = elementwise::add_alloc(&a, &b);
        assert_eq!(result.len(), len, "003: length mismatch for {len}");
        for i in 0..len {
            assert!((result[i] - len as f32).abs() < 1e-5, "003: add[{i}]={}", result[i]);
        }
    }
}

#[test]
fn falsify_blis_004_rms_norm() {
    let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let w = vec![1.0f32; 8];
    let eps = 1e-5;
    let result = norms::rms_norm_alloc(&x, &w, eps);
    assert_eq!(result.len(), 8, "004: output length");
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "004: rms_norm[{i}] not finite");
    }
    assert!(result.iter().map(|x| x.abs()).sum::<f32>() > 0.0, "004: all zeros");
}

#[test]
fn falsify_blis_005_gemv() {
    // gemv(k, n, a, b, c): a[k] is the vector, b[k×n] is the matrix, c[n] is output
    // c[j] = sum_i(a[i] * b[i*n + j])
    for (k, n) in [(8, 16), (32, 32), (64, 128), (3, 4), (1, 1)] {
        let a: Vec<f32> = (0..k).map(|i| (i + 1) as f32 * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 10) as f32 * 0.01).collect();
        let mut c = vec![0.0f32; n];
        gemv::gemv(k, n, &a, &b, &mut c);
        for j in 0..n {
            let expected: f32 = (0..k).map(|i| a[i] * b[i * n + j]).sum();
            assert!((c[j] - expected).abs() < 0.01, "005: [{j}] got={} exp={expected}", c[j]);
        }
    }
}

#[test]
fn falsify_blis_006_gemm_large_avx_path() {
    let m = 128;
    let n = 128;
    let k = 128;
    let a: Vec<f32> = (0..m * k).map(|i| ((i * 7 + 3) % 100) as f32 * 0.01).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i * 11 + 5) % 100) as f32 * 0.01).collect();
    let mut c = vec![0.0f32; m * n];
    let mut c_ref = vec![0.0f32; m * n];
    blis::gemm_blis(m, n, k, &a, &b, &mut c, None).unwrap();
    reference::gemm_reference(m, n, k, &a, &b, &mut c_ref).unwrap();
    let max_diff: f32 =
        c.iter().zip(c_ref.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_diff < 0.5, "006: large GEMM max_diff={max_diff}");
}

#[test]
fn falsify_blis_007_elementwise_add_inplace() {
    for len in [1, 8, 32, 255, 1024] {
        let mut a: Vec<f32> = (0..len).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..len).map(|_| 1.0).collect();
        elementwise::add_inplace(&mut a, &b).unwrap();
        for i in 0..len {
            assert!((a[i] - (i as f32 + 1.0)).abs() < 1e-5, "007: [{i}]={}", a[i]);
        }
    }
}

#[test]
fn falsify_blis_008_layer_norm() {
    let x: Vec<f32> = (0..64).map(|i| (i as f32) * 0.5 - 16.0).collect();
    let w = vec![1.0f32; 64];
    let b = vec![0.0f32; 64];
    let eps = 1e-5;
    let result = norms::layer_norm_alloc(&x, &w, &b, eps);
    assert_eq!(result.len(), 64, "008: output length");
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "008: layer_norm[{i}] not finite");
    }
    let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
    assert!(mean.abs() < 0.1, "008: mean={mean}");
}

#[test]
fn falsify_blis_009_softmax() {
    let x: Vec<f32> = (0..32).map(|i| (i as f32) * 0.5 - 8.0).collect();
    let result = softmax::softmax_1d_alloc(&x);
    assert_eq!(result.len(), 32, "009: output length");
    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "009: softmax sum={sum}");
    for (i, &v) in result.iter().enumerate() {
        assert!(v >= 0.0, "009: softmax[{i}]={v} < 0");
    }
}

#[test]
fn falsify_blis_010_relu() {
    let input: Vec<f32> = (-50..50).map(|i| i as f32 * 0.1).collect();
    let result = elementwise::relu_alloc(&input);
    for (&r, &x) in result.iter().zip(input.iter()) {
        if x < 0.0 {
            assert_eq!(r, 0.0, "010: relu({x}) should be 0");
        } else {
            assert_eq!(r, x, "010: relu({x}) should be {x}");
        }
    }
}

#[test]
fn falsify_blis_011_fused_mul_add() {
    let a: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let b: Vec<f32> = vec![2.0f32; 64];
    let c: Vec<f32> = vec![1.0f32; 64];
    let mut out = vec![0.0f32; 64];
    elementwise::fused_mul_add(&a, &b, &c, &mut out).unwrap();
    for i in 0..64 {
        let expected = a[i] * b[i] + c[i];
        assert!((out[i] - expected).abs() < 1e-5, "011: [{i}] got={} exp={expected}", out[i]);
    }
}

#[test]
fn falsify_blis_012_scale_inplace() {
    let mut data: Vec<f32> = (0..100).map(|i| i as f32).collect();
    elementwise::scale_inplace(&mut data, 0.5);
    for i in 0..100 {
        assert!((data[i] - i as f32 * 0.5).abs() < 1e-5, "012: [{i}]={}", data[i]);
    }
}
