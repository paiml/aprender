//! AVX2 parity, colmajor, non-aligned, and parallel tests.

use super::super::*;

/// Test AVX2 matmul with large dimensions (exercises full SIMD paths)
#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_large_matrix_mul() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("Skipping AVX2 large matrix test - CPU doesn't support AVX2+FMA");
        return;
    }

    let in_dim = 4096; // 16 super-blocks
    let out_dim = 32;

    // Build Q4K test data with realistic values
    let mut q4k_data = Vec::new();
    for row in 0..out_dim {
        for _sb in 0..(in_dim / 256) {
            // d ~ 0.1, dmin ~ 0.05
            q4k_data.extend_from_slice(&[0x66, 0x2E]); // d
            q4k_data.extend_from_slice(&[0x66, 0x2A]); // dmin
                                                       // Varied scales based on row
            let scale_val = (row as u8 % 16) | (((row + 1) as u8 % 16) << 4);
            q4k_data.extend_from_slice(&[scale_val; 12]);
            // Varied quantized values
            for i in 0..128 {
                let low = ((row + i) % 16) as u8;
                let high = ((row + i + 3) % 16) as u8;
                q4k_data.push(low | (high << 4));
            }
        }
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.001 - 2.0).collect();

    let scalar_output = matmul_q4k_f32(&q4k_data, &input, out_dim, in_dim);
    let dispatch_output = matmul_q4k_f32_dispatch(&q4k_data, &input, out_dim, in_dim);

    for (i, (scalar, dispatch)) in scalar_output.iter().zip(dispatch_output.iter()).enumerate() {
        let diff = (scalar - dispatch).abs();
        let rel_diff = if scalar.abs() > 1e-6 { diff / scalar.abs() } else { diff };
        assert!(
            rel_diff < 1e-4 || diff < 1e-4,
            "Row {}: AVX2 vs scalar divergence: {} vs {} (d={}, rel={})",
            i,
            dispatch,
            scalar,
            diff,
            rel_diff
        );
    }
}

/// Test colmajor AVX2 path with realistic dimensions
#[cfg(target_arch = "x86_64")]
#[test]
#[allow(deprecated)]
fn test_avx2_colmajor_large() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("Skipping AVX2 colmajor test - CPU doesn't support AVX2+FMA");
        return;
    }

    let in_dim = 2048; // 8 super-blocks
    let out_dim = 16;

    let mut q4k_data = Vec::new();
    for row in 0..out_dim {
        for sb in 0..(in_dim / 256) {
            q4k_data.extend_from_slice(&[0x66, 0x2E]); // d
            q4k_data.extend_from_slice(&[0x33, 0x2A]); // dmin
            let scale_val = ((row + sb) as u8 % 16) | (((row + sb + 1) as u8 % 16) << 4);
            q4k_data.extend_from_slice(&[scale_val; 12]);
            for i in 0..128 {
                q4k_data.push(((i % 16) | (((i + 1) % 16) << 4)) as u8);
            }
        }
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.002 - 1.0).collect();

    let output = matmul_q4k_f32_colmajor(&q4k_data, &input, out_dim, in_dim);
    let output_dispatch = matmul_q4k_f32_colmajor_dispatch(&q4k_data, &input, out_dim, in_dim);

    assert_eq!(output.len(), out_dim);
    assert_eq!(output_dispatch.len(), out_dim);

    for (i, (base, dispatched)) in output.iter().zip(output_dispatch.iter()).enumerate() {
        let diff = (base - dispatched).abs();
        assert!(
            diff < 1e-3 || (diff / base.abs()) < 1e-4,
            "Row {}: colmajor mismatch: {} vs {} (diff={})",
            i,
            base,
            dispatched,
            diff
        );
    }
}

/// Test non-aligned dimensions (exercises scalar remainder handling)
#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_non_aligned_dimensions() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("Skipping AVX2 non-aligned test - CPU doesn't support AVX2+FMA");
        return;
    }

    // Non-aligned: 768 = 3 super-blocks (not power of 2)
    let in_dim = 768;
    let out_dim = 7; // Odd number

    let mut q4k_data = Vec::new();
    for row in 0..out_dim {
        for _sb in 0..(in_dim / 256) {
            q4k_data.extend_from_slice(&[0x66, 0x2E]);
            q4k_data.extend_from_slice(&[0x66, 0x2A]);
            let scale_val = (row as u8 % 16) | (((row + 1) as u8 % 16) << 4);
            q4k_data.extend_from_slice(&[scale_val; 12]);
            for i in 0..128 {
                q4k_data.push(((i % 16) | (((i + 5) % 16) << 4)) as u8);
            }
        }
    }

    let input: Vec<f32> = (0..in_dim).map(|i| ((i as f32) * 0.003).sin()).collect();

    let scalar_output = matmul_q4k_f32(&q4k_data, &input, out_dim, in_dim);
    let dispatch_output = matmul_q4k_f32_dispatch(&q4k_data, &input, out_dim, in_dim);

    assert_eq!(scalar_output.len(), out_dim);
    assert_eq!(dispatch_output.len(), out_dim);

    for (i, (scalar, dispatch)) in scalar_output.iter().zip(dispatch_output.iter()).enumerate() {
        let diff = (scalar - dispatch).abs();
        let rel_diff = if scalar.abs() > 1e-6 { diff / scalar.abs() } else { diff };
        // FMA operations can have ordering differences, allow 1e-5 relative error
        assert!(
            rel_diff < 1e-5 || diff < 1e-2,
            "Row {}: non-aligned AVX2 mismatch: {} vs {} (diff={}, rel={})",
            i,
            scalar,
            dispatch,
            diff,
            rel_diff
        );
    }
}

/// Test parallel SIMD execution (exercises compute_chunk_q4k_avx2)
#[cfg(all(target_arch = "x86_64", feature = "parallel"))]
#[test]
fn test_parallel_avx2_large_batch() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("Skipping parallel AVX2 test - CPU doesn't support AVX2+FMA");
        return;
    }

    // Large enough to trigger parallel path (>1000 rows)
    let in_dim = 1024;
    let out_dim = 2048; // Large output dim for parallel execution

    let mut q4k_data = Vec::new();
    for row in 0..out_dim {
        for _sb in 0..(in_dim / 256) {
            q4k_data.extend_from_slice(&[0x66, 0x2E]);
            q4k_data.extend_from_slice(&[0x33, 0x2A]);
            let scale_val = ((row % 256) as u8) | (((row / 256) % 16) as u8 * 16);
            q4k_data.extend_from_slice(&[scale_val; 12]);
            for i in 0..128 {
                q4k_data.push(((i * row) % 256) as u8);
            }
        }
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.001).collect();

    let output = matmul_q4k_f32_colmajor_dispatch(&q4k_data, &input, out_dim, in_dim);

    assert_eq!(output.len(), out_dim);
    for (i, val) in output.iter().enumerate() {
        assert!(val.is_finite(), "Row {}: parallel AVX2 produced non-finite: {}", i, val);
    }
}
