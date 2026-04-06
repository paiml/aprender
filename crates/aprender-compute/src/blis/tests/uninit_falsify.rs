//! FALSIFY tests for uninit allocation correctness (CGP-DBUF).
//!
//! These tests verify that the `Vec::with_capacity + set_len` pattern
//! produces identical results to the original `vec![0.0; n]` pattern.
//!
//! If any test fails, the uninit optimization is UNSOUND and must be reverted.

use crate::{Matrix, Vector};

// ========================================================================
// FALSIFY-UNINIT-001: Vector::sqrt uninit matches zero-init
// ========================================================================

#[test]
fn falsify_uninit_001_sqrt_correctness() {
    let data = vec![0.0, 1.0, 4.0, 9.0, 16.0, 0.25, 100.0, f32::INFINITY];
    let v = Vector::from_slice(&data);
    let result = v.sqrt().unwrap();

    let expected: Vec<f32> = data.iter().map(|x| x.sqrt()).collect();
    assert_eq!(result.as_slice(), &expected[..], "FALSIFY-UNINIT-001: sqrt output mismatch");
}

#[test]
fn falsify_uninit_001b_sqrt_large() {
    let n = 200_000;
    let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
    let v = Vector::from_slice(&data);
    let result = v.sqrt().unwrap();

    for (i, (&got, &x)) in result.as_slice().iter().zip(data.iter()).enumerate() {
        let expected = x.sqrt();
        assert!(
            (got - expected).abs() < 1e-6 || (got.is_nan() && expected.is_nan()),
            "FALSIFY-UNINIT-001b: sqrt[{i}] got {got}, expected {expected}"
        );
    }
}

// ========================================================================
// FALSIFY-UNINIT-002: Vector::recip uninit matches zero-init
// ========================================================================

#[test]
fn falsify_uninit_002_recip_correctness() {
    let data = vec![1.0, 2.0, 4.0, 0.5, 10.0, 0.1, -3.0];
    let v = Vector::from_slice(&data);
    let result = v.recip().unwrap();

    let expected: Vec<f32> = data.iter().map(|x| 1.0 / x).collect();
    for (i, (&got, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-6,
            "FALSIFY-UNINIT-002: recip[{i}] got {got}, expected {exp}"
        );
    }
}

// ========================================================================
// FALSIFY-UNINIT-003: blis::softmax uninit matches reference
// ========================================================================

#[test]
fn falsify_uninit_003_softmax_sums_to_one() {
    let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = crate::blis::softmax::softmax_1d_alloc(&logits);

    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "FALSIFY-UNINIT-003: softmax sum={sum}");

    for (i, &v) in result.iter().enumerate() {
        assert!(v > 0.0, "FALSIFY-UNINIT-003: softmax[{i}]={v} <= 0");
    }

    for i in 1..result.len() {
        assert!(result[i] >= result[i - 1], "FALSIFY-UNINIT-003: not monotone at {i}");
    }
}

#[test]
fn falsify_uninit_003b_softmax_large_avx2() {
    // n=1024 exercises AVX2 path (chunks of 32)
    let n = 1024;
    let logits: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 5.0).collect();
    let result = crate::blis::softmax::softmax_1d_alloc(&logits);

    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4, "FALSIFY-UNINIT-003b: sum={sum}");
    assert!(result.iter().all(|v| *v >= 0.0 && v.is_finite()));
}

#[test]
fn falsify_uninit_003c_softmax_deterministic() {
    let logits: Vec<f32> = (0..256).map(|i| (i as f32) * 0.1 - 12.8).collect();
    let r1 = crate::blis::softmax::softmax_1d_alloc(&logits);
    let r2 = crate::blis::softmax::softmax_1d_alloc(&logits);
    assert_eq!(r1, r2, "FALSIFY-UNINIT-003c: softmax non-deterministic");
}

// ========================================================================
// FALSIFY-UNINIT-004: Matrix::matvec uninit matches reference
// ========================================================================

#[test]
fn falsify_uninit_004_matvec_small() {
    let m = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let v = Vector::from_slice(&[7.0, 8.0]);
    let result = m.matvec(&v).unwrap();
    assert_eq!(result.as_slice(), &[23.0, 53.0, 83.0], "FALSIFY-UNINIT-004");
}

#[test]
fn falsify_uninit_004b_matvec_large_parallel() {
    // >=4096 rows triggers parallel path
    let rows = 4096;
    let cols = 128;
    let data: Vec<f32> = (0..rows * cols).map(|i| ((i % 17) as f32) * 0.1).collect();
    let v_data: Vec<f32> = (0..cols).map(|i| ((i % 7) as f32) * 0.2).collect();

    let m = Matrix::from_vec(rows, cols, data.clone()).unwrap();
    let v = Vector::from_slice(&v_data);
    let result = m.matvec(&v).unwrap();

    for r in 0..rows {
        let mut expected = 0.0f32;
        for c in 0..cols {
            expected += data[r * cols + c] * v_data[c];
        }
        let got = result.as_slice()[r];
        assert!(
            (got - expected).abs() < expected.abs() * 1e-5 + 1e-6,
            "FALSIFY-UNINIT-004b: matvec[{r}] got={got}, expected={expected}"
        );
    }
}

// ========================================================================
// FALSIFY-UNINIT-005: Q4K GEMV uninit is deterministic
// ========================================================================

#[test]
fn falsify_uninit_005_q4k_gemv_deterministic() {
    use crate::backends::q4k::matmul_q4k_f32;

    let in_dim = 256;
    let out_dim = 4;
    let num_blocks_per_row = (in_dim + 255) / 256;
    let row_bytes = num_blocks_per_row * 144;
    let q4k_data = vec![0x55u8; out_dim * row_bytes];
    let input = vec![1.0f32; in_dim];

    let r1 = matmul_q4k_f32(&q4k_data, &input, out_dim, in_dim);
    let r2 = matmul_q4k_f32(&q4k_data, &input, out_dim, in_dim);

    assert_eq!(r1, r2, "FALSIFY-UNINIT-005: Q4K non-deterministic");
    for (i, &v) in r1.iter().enumerate() {
        assert!(v.is_finite(), "FALSIFY-UNINIT-005: Q4K output[{i}]={v}");
    }
}

// ========================================================================
// FALSIFY-UNINIT-006: AttentionOp output bounded by V range
// ========================================================================

#[test]
fn falsify_uninit_006_attention_bounded() {
    use crate::brick::{AttentionOp, ComputeBackend, ComputeOp};

    let seq_len = 4;
    let kv_seq_len = 4;
    let head_dim = 8;

    let op = AttentionOp::new(seq_len, kv_seq_len, head_dim);

    let q = vec![1.0f32; seq_len * head_dim];
    let k = vec![1.0f32; kv_seq_len * head_dim];
    let v: Vec<f32> = (0..kv_seq_len * head_dim).map(|i| (i as f32) * 0.1).collect();

    let backend = ComputeBackend::default();
    let output = op.execute((q, k, v.clone()), backend).unwrap();

    let v_min = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let v_max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    for (i, &o) in output.iter().enumerate() {
        assert!(
            o >= v_min - 1e-5 && o <= v_max + 1e-5,
            "FALSIFY-UNINIT-006: output[{i}]={o} outside [{v_min}, {v_max}]"
        );
    }
}

// ========================================================================
// FALSIFY-UNINIT-007: FusedQKV SIMD dot matches scalar reference
// ========================================================================

#[test]
fn falsify_uninit_007_fused_qkv_simd_vs_scalar() {
    use crate::brick::{ComputeBackend, ComputeOp, FusedQKVOp, FusedQKVWeights};

    let hidden = 8;
    let num_heads = 2;
    let kv_heads = 1;
    let kv_dim = hidden / num_heads * kv_heads;

    let op = FusedQKVOp::new(hidden, num_heads, kv_heads);

    let w_q: Vec<f32> = (0..hidden * hidden).map(|i| ((i % 11) as f32) * 0.1 - 0.5).collect();
    let w_k: Vec<f32> = (0..kv_dim * hidden).map(|i| ((i % 7) as f32) * 0.1 - 0.3).collect();
    let w_v: Vec<f32> = (0..kv_dim * hidden).map(|i| ((i % 13) as f32) * 0.1 - 0.6).collect();
    let x: Vec<f32> = (0..hidden).map(|i| ((i % 5) as f32) * 0.2).collect();

    let weights =
        FusedQKVWeights { q_weight: w_q.clone(), k_weight: w_k.clone(), v_weight: w_v.clone() };
    let backend = ComputeBackend::default();

    let (q, k, v) = op.execute((x.clone(), weights), backend).unwrap();

    // Scalar reference
    for i in 0..hidden {
        let expected: f32 = (0..hidden).map(|j| x[j] * w_q[i * hidden + j]).sum();
        assert!(
            (q[i] - expected).abs() < 1e-4,
            "FALSIFY-UNINIT-007: Q[{i}] got={}, expected={expected}",
            q[i]
        );
    }
    for i in 0..kv_dim {
        let exp_k: f32 = (0..hidden).map(|j| x[j] * w_k[i * hidden + j]).sum();
        let exp_v: f32 = (0..hidden).map(|j| x[j] * w_v[i * hidden + j]).sum();
        assert!((k[i] - exp_k).abs() < 1e-4, "FALSIFY-UNINIT-007: K[{i}]");
        assert!((v[i] - exp_v).abs() < 1e-4, "FALSIFY-UNINIT-007: V[{i}]");
    }
}

// ========================================================================
// FALSIFY-SIMD-001: AVX2 axpy in attention matches scalar reference
// ========================================================================

#[test]
fn falsify_simd_001_axpy_correctness() {
    use crate::brick::AttentionOp;

    // head_dim=128 (typical LLM): exercises full AVX2 path (128/8=16 iterations)
    let n = 128;
    let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
    let alpha = 0.75f32;

    let mut out_simd = vec![1.0f32; n]; // non-zero initial values
    AttentionOp::simd_axpy(alpha, &x, &mut out_simd);

    // Scalar reference
    let mut out_ref = vec![1.0f32; n];
    for i in 0..n {
        out_ref[i] += alpha * x[i];
    }

    for i in 0..n {
        assert!(
            (out_simd[i] - out_ref[i]).abs() < 1e-5,
            "FALSIFY-SIMD-001: axpy[{i}] got={}, expected={}",
            out_simd[i],
            out_ref[i]
        );
    }
}

#[test]
fn falsify_simd_001b_axpy_remainder() {
    use crate::brick::AttentionOp;

    // head_dim=17: tests remainder path (17 % 8 = 1)
    let n = 17;
    let x: Vec<f32> = (0..n).map(|i| ((i * 3 + 1) as f32) * 0.1).collect();
    let alpha = -0.5f32;

    let mut out = vec![2.0f32; n];
    AttentionOp::simd_axpy(alpha, &x, &mut out);

    for i in 0..n {
        let expected = 2.0 + alpha * x[i];
        assert!(
            (out[i] - expected).abs() < 1e-5,
            "FALSIFY-SIMD-001b: axpy[{i}] got={}, expected={expected}",
            out[i]
        );
    }
}

// ========================================================================
// FALSIFY-PARALLEL-001: Parallel transpose matches serial at boundary
// ========================================================================

#[test]
fn falsify_parallel_001_transpose_boundary() {
    // 1000×1000 = 1M elements: right at PARALLEL_THRESHOLD=1M.
    // Both parallel and serial paths must produce identical results.
    let rows = 1000;
    let cols = 1000;
    let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.001).collect();

    let mut result = vec![0.0f32; rows * cols];
    crate::blis::transpose::transpose(rows, cols, &data, &mut result).unwrap();

    // Verify: result[j*rows+i] == data[i*cols+j]
    for i in 0..rows.min(100) {
        for j in 0..cols.min(100) {
            let expected = data[i * cols + j];
            let got = result[j * rows + i];
            assert!(
                (got - expected).abs() < 1e-6,
                "FALSIFY-PARALLEL-001: transpose[{j},{i}] got={got}, expected={expected}"
            );
        }
    }
}

// ========================================================================
// FALSIFY-PARALLEL-002: Parallel matvec matches serial at boundary
// ========================================================================

#[test]
fn falsify_parallel_002_matvec_boundary() {
    // 2048 rows: right at PARALLEL_THRESHOLD=2048 for matvec.
    let rows = 2048;
    let cols = 64;
    let data: Vec<f32> = (0..rows * cols).map(|i| ((i % 31) as f32) * 0.1).collect();
    let v_data: Vec<f32> = (0..cols).map(|i| ((i % 11) as f32) * 0.2).collect();

    let m = Matrix::from_vec(rows, cols, data.clone()).unwrap();
    let v = Vector::from_slice(&v_data);
    let result = m.matvec(&v).unwrap();

    // Scalar reference
    for r in 0..rows {
        let expected: f32 = (0..cols).map(|c| data[r * cols + c] * v_data[c]).sum();
        let got = result.as_slice()[r];
        assert!(
            (got - expected).abs() < expected.abs() * 1e-5 + 1e-5,
            "FALSIFY-PARALLEL-002: matvec[{r}] got={got}, expected={expected}"
        );
    }
}

// ========================================================================
// FALSIFY-PARALLEL-003: Parallel transpose deterministic
// ========================================================================

#[test]
fn falsify_parallel_003_transpose_deterministic() {
    let rows = 1024;
    let cols = 1024;
    let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.001).collect();

    let mut r1 = vec![0.0f32; rows * cols];
    let mut r2 = vec![0.0f32; rows * cols];
    crate::blis::transpose::transpose(rows, cols, &data, &mut r1).unwrap();
    crate::blis::transpose::transpose(rows, cols, &data, &mut r2).unwrap();
    assert_eq!(r1, r2, "FALSIFY-PARALLEL-003: transpose non-deterministic");
}
