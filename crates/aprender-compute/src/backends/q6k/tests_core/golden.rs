//! Golden Vector Tests: Q6K scalar reference vs dispatch/SIMD paths

use super::super::*;

/// Golden Test: Q6K scalar == dispatch for random input
#[test]
fn test_golden_q6k_scalar_vs_dispatch() {
    // Realistic LLM dimensions
    let in_dim = 512; // 2 super-blocks
    let out_dim = 8;

    let mut q6k_data = Vec::new();
    for row in 0..out_dim {
        for sb in 0..(in_dim / 256) {
            // ql: varied 4-bit low values
            for i in 0..128 {
                let low = ((row + sb + i) % 16) as u8;
                let high = ((row + sb + i + 3) % 16) as u8;
                q6k_data.push(low | (high << 4));
            }
            // qh: varied 2-bit high values
            for i in 0..64 {
                let vals = [
                    ((row + i) % 4) as u8,
                    ((row + i + 1) % 4) as u8,
                    ((row + i + 2) % 4) as u8,
                    ((row + i + 3) % 4) as u8,
                ];
                q6k_data.push(vals[0] | (vals[1] << 2) | (vals[2] << 4) | (vals[3] << 6));
            }
            // scales: varied signed 8-bit
            for i in 0..16 {
                q6k_data.push(((row * 7 + sb * 3 + i) % 64) as u8);
            }
            // d ~ 0.1
            q6k_data.extend_from_slice(&[0x66, 0x2E]);
        }
    }

    // Sinusoidal input
    let input: Vec<f32> = (0..in_dim).map(|i| ((i as f32) * 0.019).sin() * 0.4).collect();

    let scalar_output = matmul_q6k_f32_scalar(&q6k_data, &input, out_dim, in_dim);
    let dispatch_output = matmul_q6k_f32_dispatch(&q6k_data, &input, out_dim, in_dim);

    assert_eq!(scalar_output.len(), dispatch_output.len());
    let mut max_abs_error = 0.0f32;

    for (i, (scalar, dispatch)) in scalar_output.iter().zip(dispatch_output.iter()).enumerate() {
        let abs_error = (scalar - dispatch).abs();
        max_abs_error = max_abs_error.max(abs_error);

        // Scalar and dispatch should match closely (minor FMA ordering differences)
        assert!(
            abs_error < 2e-4,
            "Row {}: scalar={}, dispatch={}, diff={}",
            i,
            scalar,
            dispatch,
            abs_error
        );
    }

    eprintln!("[Golden Q6K Test] max_abs_error={:.6}", max_abs_error);
}

/// Golden Test: Q6K colmajor path consistency
#[test]
#[allow(deprecated)]
fn test_golden_q6k_colmajor_consistency() {
    let in_dim = 512;
    let out_dim = 4;

    let mut q6k_data = Vec::new();
    for row in 0..out_dim {
        for sb in 0..2 {
            // ql
            for i in 0..128 {
                q6k_data.push(((row * 5 + sb * 13 + i) % 256) as u8);
            }
            // qh
            for i in 0..64 {
                q6k_data.push(((row * 7 + sb * 11 + i * 2) % 256) as u8);
            }
            // scales
            for i in 0..16 {
                q6k_data.push(((row + sb + i) % 128) as u8);
            }
            // d ~ 0.5
            q6k_data.extend_from_slice(&[0x00, 0x38]);
        }
    }

    let input: Vec<f32> = (0..in_dim).map(|i| ((i as f32) * 0.011 + 0.3).cos() * 0.5).collect();

    let colmajor_output = matmul_q6k_f32_colmajor(&q6k_data, &input, out_dim, in_dim);
    let colmajor_dispatch = matmul_q6k_f32_colmajor_dispatch(&q6k_data, &input, out_dim, in_dim);

    assert_eq!(colmajor_output.len(), colmajor_dispatch.len());
    for (i, (base, dispatch)) in colmajor_output.iter().zip(colmajor_dispatch.iter()).enumerate() {
        let diff = (base - dispatch).abs();
        assert!(
            diff < 1e-4,
            "Row {}: colmajor base={}, dispatch={}, diff={}",
            i,
            base,
            dispatch,
            diff
        );
    }
}

/// Edge case: maximum 6-bit values (63)
#[test]
fn test_golden_q6k_max_quant_values() {
    let in_dim = 256;
    let out_dim = 2;

    let mut q6k_data = Vec::new();
    for _ in 0..out_dim {
        // ql: all 0xF (low nibble = 15)
        q6k_data.extend_from_slice(&[0xFFu8; 128]);
        // qh: all 0xFF (all 2-bit high = 3), so value = 15 + 3*16 = 63
        q6k_data.extend_from_slice(&[0xFFu8; 64]);
        // scales: positive
        q6k_data.extend_from_slice(&[0x3Fu8; 16]); // scale = 63
                                                   // d = 1.0
        q6k_data.extend_from_slice(&[0x00, 0x3C]);
    }

    let input = vec![1.0f32; in_dim];
    let scalar_output = matmul_q6k_f32_scalar(&q6k_data, &input, out_dim, in_dim);
    let dispatch_output = matmul_q6k_f32_dispatch(&q6k_data, &input, out_dim, in_dim);

    for (i, (scalar, dispatch)) in scalar_output.iter().zip(dispatch_output.iter()).enumerate() {
        assert!(
            scalar.is_finite() && dispatch.is_finite(),
            "Row {}: max values should produce finite output",
            i
        );
        let diff = (scalar - dispatch).abs();
        assert!(
            diff < 1e-4,
            "Row {}: max quant scalar={}, dispatch={}, diff={}",
            i,
            scalar,
            dispatch,
            diff
        );
    }
}

/// Edge case: alternating positive/negative scales
#[test]
fn test_golden_q6k_alternating_scales() {
    let in_dim = 256;
    let out_dim = 2;

    let mut q6k_data = Vec::new();
    for _ in 0..out_dim {
        // ql: mid-range values
        q6k_data.extend_from_slice(&[0x77u8; 128]); // 7, 7 repeated
                                                    // qh: zeros (full value = 7)
        q6k_data.extend_from_slice(&[0x00u8; 64]);
        // scales: alternating +32, -32
        for i in 0..16 {
            if i % 2 == 0 {
                q6k_data.push(0x20); // +32
            } else {
                q6k_data.push(0xE0); // -32 (as signed i8)
            }
        }
        // d = 0.5
        q6k_data.extend_from_slice(&[0x00, 0x38]);
    }

    let input = vec![1.0f32; in_dim];
    let scalar_output = matmul_q6k_f32_scalar(&q6k_data, &input, out_dim, in_dim);
    let dispatch_output = matmul_q6k_f32_dispatch(&q6k_data, &input, out_dim, in_dim);

    for (i, (scalar, dispatch)) in scalar_output.iter().zip(dispatch_output.iter()).enumerate() {
        let diff = (scalar - dispatch).abs();
        assert!(
            diff < 1e-4,
            "Row {}: alternating scales scalar={}, dispatch={}, diff={}",
            i,
            scalar,
            dispatch,
            diff
        );
    }
}

/// Large scale test for SIMD path coverage
#[cfg(target_arch = "x86_64")]
#[test]
fn test_golden_q6k_large_simd() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("Skipping Q6K SIMD test - no AVX2+FMA");
        return;
    }

    let in_dim = 2048; // 8 super-blocks
    let out_dim = 32;

    let mut q6k_data = Vec::new();
    for row in 0..out_dim {
        for sb in 0..(in_dim / 256) {
            for i in 0..128 {
                let val = ((row * 3 + sb * 7 + i) % 256) as u8;
                q6k_data.push(val);
            }
            for i in 0..64 {
                let val = ((row * 5 + sb * 11 + i * 2) % 256) as u8;
                q6k_data.push(val);
            }
            for i in 0..16 {
                q6k_data.push(((row + sb + i) % 64) as u8);
            }
            q6k_data.extend_from_slice(&[0x66, 0x2E]);
        }
    }

    let input: Vec<f32> = (0..in_dim).map(|i| ((i as f32) * 0.007 - 1.0).tanh()).collect();

    let scalar_output = matmul_q6k_f32_scalar(&q6k_data, &input, out_dim, in_dim);
    let dispatch_output = matmul_q6k_f32_dispatch(&q6k_data, &input, out_dim, in_dim);

    let mut max_rel_error = 0.0f32;
    for (i, (scalar, dispatch)) in scalar_output.iter().zip(dispatch_output.iter()).enumerate() {
        let abs_error = (scalar - dispatch).abs();
        let rel_error = if scalar.abs() > 1e-6 { abs_error / scalar.abs() } else { abs_error };
        max_rel_error = max_rel_error.max(rel_error);

        assert!(
            rel_error < 1e-4 || abs_error < 1e-4,
            "Row {}: large SIMD scalar={}, dispatch={}, rel_err={:.6}",
            i,
            scalar,
            dispatch,
            rel_error
        );
    }

    eprintln!("[Golden Q6K Large SIMD] max_rel_error={:.6}", max_rel_error);
}
