//! Golden Vector Tests (Section 12.4: Q4K fused matmul ~ dequant+f32_matmul)
//! and compute_chunk_scalar direct tests.

use super::gemv::compute_chunk_q4k_scalar;
use super::*;

/// Helper: naive f32 matrix-vector multiplication
fn matmul_f32_naive(weights: &[f32], input: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; out_dim];
    for row in 0..out_dim {
        let mut sum = 0.0f32;
        for col in 0..in_dim {
            sum += weights[row * in_dim + col] * input[col];
        }
        output[row] = sum;
    }
    output
}

/// Golden Vector Test: Q4K matmul ~ dequant + f32 matmul
///
/// This test verifies the invariant from Section 12.4:
/// matmul_q4k_f32(W, x) ~ matmul(dequant_q4k_to_f32(W), x) within e
///
/// Quantization introduces error, so we use a relaxed tolerance (5%).
#[test]
fn test_golden_vector_q4k_matmul_vs_dequant() {
    use crate::backends::q4k::dequantize_q4k_to_f32;

    // Realistic dimensions for LLM layers
    let in_dim = 512; // 2 super-blocks
    let out_dim = 8;

    // Build Q4K test data with realistic distribution
    let mut q4k_data = Vec::new();
    for row in 0..out_dim {
        for sb in 0..(in_dim / 256) {
            // d ~ 0.1, dmin ~ 0.05 (realistic for normalized weights)
            q4k_data.extend_from_slice(&[0x66, 0x2E]); // d
            q4k_data.extend_from_slice(&[0x66, 0x2A]); // dmin
                                                       // Varied scales based on position
            let scale_base = ((row * 7 + sb * 3) % 16) as u8;
            for i in 0..12 {
                q4k_data.push(scale_base + (i as u8 % 4));
            }
            // Varied quantized values (4-bit, so 0-15)
            for i in 0..128 {
                let low = ((row + sb + i) % 16) as u8;
                let high = ((row + sb + i + 5) % 16) as u8;
                q4k_data.push(low | (high << 4));
            }
        }
    }

    // Random-ish input vector (sinusoidal distribution)
    let input: Vec<f32> = (0..in_dim).map(|i| ((i as f32) * 0.017).sin() * 0.5).collect();

    // Method 1: Fused Q4K matmul
    let fused_output = matmul_q4k_f32(&q4k_data, &input, out_dim, in_dim);

    // Method 2: Dequantize + f32 matmul
    let total_elements = in_dim * out_dim;
    let dequantized_weights = dequantize_q4k_to_f32(&q4k_data, total_elements);
    let reference_output = matmul_f32_naive(&dequantized_weights, &input, out_dim, in_dim);

    // Verify Golden Invariant: error within 5% or absolute 0.01
    assert_eq!(fused_output.len(), reference_output.len());
    let mut max_rel_error = 0.0f32;
    let mut max_abs_error = 0.0f32;

    for (i, (fused, reference)) in fused_output.iter().zip(reference_output.iter()).enumerate() {
        let abs_error = (fused - reference).abs();
        let rel_error =
            if reference.abs() > 1e-6 { abs_error / reference.abs() } else { abs_error };
        max_rel_error = max_rel_error.max(rel_error);
        max_abs_error = max_abs_error.max(abs_error);

        assert!(
            rel_error < 0.05 || abs_error < 0.01,
            "Golden invariant violated at row {}: fused={}, reference={}, \
             rel_error={:.4}%, abs_error={:.6}",
            i,
            fused,
            reference,
            rel_error * 100.0,
            abs_error
        );
    }

    // Report max errors for visibility
    eprintln!(
        "[Golden Q4K Test] max_rel_error={:.4}%, max_abs_error={:.6}",
        max_rel_error * 100.0,
        max_abs_error
    );
}

/// Golden Vector Test: dispatch path also satisfies invariant
#[test]
fn test_golden_vector_q4k_dispatch_vs_dequant() {
    use crate::backends::q4k::dequantize_q4k_to_f32;

    // Larger dimensions to exercise SIMD paths
    let in_dim = 1024; // 4 super-blocks
    let out_dim = 16;

    let mut q4k_data = Vec::new();
    for row in 0..out_dim {
        for sb in 0..(in_dim / 256) {
            q4k_data.extend_from_slice(&[0x00, 0x3C]); // d ~ 1.0
            q4k_data.extend_from_slice(&[0x00, 0x38]); // dmin ~ 0.5
            for i in 0..12 {
                q4k_data.push(((row + sb + i) % 64) as u8);
            }
            for i in 0..128 {
                let low = ((row * 3 + sb * 7 + i) % 16) as u8;
                let high = ((row * 5 + sb * 11 + i * 2) % 16) as u8;
                q4k_data.push(low | (high << 4));
            }
        }
    }

    let input: Vec<f32> = (0..in_dim).map(|i| ((i as f32) * 0.013 + 0.5).cos() * 0.3).collect();

    // Dispatch (may use AVX2/SIMD)
    let dispatch_output = matmul_q4k_f32_dispatch(&q4k_data, &input, out_dim, in_dim);

    // Reference: dequantize + f32
    let total_elements = in_dim * out_dim;
    let dequantized = dequantize_q4k_to_f32(&q4k_data, total_elements);
    let reference_output = matmul_f32_naive(&dequantized, &input, out_dim, in_dim);

    let mut max_rel_error = 0.0f32;
    for (i, (dispatch, reference)) in
        dispatch_output.iter().zip(reference_output.iter()).enumerate()
    {
        let abs_error = (dispatch - reference).abs();
        let rel_error =
            if reference.abs() > 1e-6 { abs_error / reference.abs() } else { abs_error };
        max_rel_error = max_rel_error.max(rel_error);

        assert!(
            rel_error < 0.05 || abs_error < 0.01,
            "Golden invariant violated (dispatch) at row {}: \
             dispatch={}, reference={}, rel_error={:.4}%",
            i,
            dispatch,
            reference,
            rel_error * 100.0
        );
    }

    eprintln!("[Golden Q4K Dispatch Test] max_rel_error={:.4}%", max_rel_error * 100.0);
}

/// Edge case: zero input vector should produce zero output
#[test]
fn test_golden_vector_zero_input() {
    let in_dim = 256;
    let out_dim = 4;

    let mut q4k_data = Vec::new();
    for _row in 0..out_dim {
        q4k_data.extend_from_slice(&[0x66, 0x2E]); // d
        q4k_data.extend_from_slice(&[0x00, 0x00]); // dmin = 0 (important for zero output)
        q4k_data.extend_from_slice(&[0x01u8; 12]);
        q4k_data.extend_from_slice(&[0x55u8; 128]); // Non-zero weights
    }

    let input = vec![0.0f32; in_dim];
    let output = matmul_q4k_f32(&q4k_data, &input, out_dim, in_dim);

    // With dmin=0 and all-zero input, output should be near zero
    for (i, val) in output.iter().enumerate() {
        assert!(val.abs() < 1e-6, "Zero input should give ~zero output, got {} at row {}", val, i);
    }
}

/// Edge case: uniform input vector
#[test]
fn test_golden_vector_uniform_input() {
    use crate::backends::q4k::dequantize_q4k_to_f32;

    let in_dim = 256;
    let out_dim = 2;

    let mut q4k_data = Vec::new();
    for row in 0..out_dim {
        q4k_data.extend_from_slice(&[0x00, 0x3C]); // d ~ 1.0
        q4k_data.extend_from_slice(&[0x00, 0x00]); // dmin = 0
        q4k_data.extend_from_slice(&[0x01u8; 12]);
        // Uniform quantized weights
        q4k_data.extend_from_slice(&[((row + 1) * 0x11) as u8; 128]);
    }

    let input = vec![1.0f32; in_dim]; // All ones
    let fused_output = matmul_q4k_f32(&q4k_data, &input, out_dim, in_dim);

    let total_elements = in_dim * out_dim;
    let dequantized = dequantize_q4k_to_f32(&q4k_data, total_elements);
    let reference_output = matmul_f32_naive(&dequantized, &input, out_dim, in_dim);

    for (i, (fused, reference)) in fused_output.iter().zip(reference_output.iter()).enumerate() {
        let rel_error = if reference.abs() > 1e-6 {
            (fused - reference).abs() / reference.abs()
        } else {
            (fused - reference).abs()
        };
        assert!(
            rel_error < 0.05,
            "Uniform input failed at row {}: fused={}, ref={}, err={:.2}%",
            i,
            fused,
            reference,
            rel_error * 100.0
        );
    }
}

#[test]
fn test_parallel_dispatch_large_matrix() {
    // Test parallel path: total_work >= 8_000_000
    // Use 4096 x 2048 = 8_388_608 ops (triggers parallel)
    let out_dim = 4096;
    let in_dim = 2048; // Must be multiple of 256 (SUPER_BLOCK_SIZE)
    let total_work = out_dim * in_dim;
    assert!(total_work >= 8_000_000, "Test must trigger parallel path");

    let num_superblocks_per_row = (in_dim + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let row_bytes = num_superblocks_per_row * SUPER_BLOCK_BYTES;
    let total_bytes = out_dim * row_bytes;

    // Create deterministic test data
    let mut q4k_data = vec![0u8; total_bytes];
    for row in 0..out_dim {
        for sb in 0..num_superblocks_per_row {
            let offset = row * row_bytes + sb * SUPER_BLOCK_BYTES;
            // d = 1.0 as f16
            q4k_data[offset] = 0x00;
            q4k_data[offset + 1] = 0x3C;
            // dmin = 0.0
            q4k_data[offset + 2] = 0x00;
            q4k_data[offset + 3] = 0x00;
            // scales = 1 for all
            for i in 0..12 {
                q4k_data[offset + 4 + i] = 0x01;
            }
            // qs = predictable pattern
            for i in 0..128 {
                q4k_data[offset + 16 + i] = ((row + sb + i) % 16) as u8;
            }
        }
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i % 10) as f32 * 0.1).collect();

    // Call dispatch - should use parallel path
    let result = matmul_q4k_f32_dispatch(&q4k_data, &input, out_dim, in_dim);

    // Verify dimensions and finiteness
    assert_eq!(result.len(), out_dim);
    for (i, &val) in result.iter().enumerate() {
        assert!(val.is_finite(), "Result[{}] is not finite: {}", i, val);
    }

    // Compare a few rows against scalar for consistency
    let scalar_result = matmul_q4k_f32_scalar(&q4k_data, &input, out_dim, in_dim);
    for i in (0..out_dim).step_by(512) {
        let diff = (result[i] - scalar_result[i]).abs();
        let tol = scalar_result[i].abs() * 0.01 + 1e-5;
        assert!(
            diff < tol,
            "Parallel vs scalar mismatch at row {}: parallel={}, scalar={}, diff={}",
            i,
            result[i],
            scalar_result[i],
            diff
        );
    }
}

#[test]
#[allow(deprecated)]
fn test_parallel_colmajor_large_matrix() {
    // Test colmajor path
    // ne0 = output dimension (rows), ne1 = input dimension (columns)
    // Input must have length ne1
    let ne0 = 2048; // output dimension (rows), must be multiple of 256
    let ne1 = 4096; // input dimension (columns)

    let blocks_per_col = (ne0 + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let col_bytes = blocks_per_col * SUPER_BLOCK_BYTES;
    let total_bytes = ne1 * col_bytes;

    let mut q4k_data = vec![0u8; total_bytes];
    for col in 0..ne1 {
        for sb in 0..blocks_per_col {
            let offset = col * col_bytes + sb * SUPER_BLOCK_BYTES;
            // d = 0.5 as f16
            q4k_data[offset] = 0x00;
            q4k_data[offset + 1] = 0x38;
            // dmin = 0.0
            q4k_data[offset + 2] = 0x00;
            q4k_data[offset + 3] = 0x00;
            // scales
            for i in 0..12 {
                q4k_data[offset + 4 + i] = 0x02;
            }
            // qs
            for i in 0..128 {
                q4k_data[offset + 16 + i] = ((col ^ sb ^ i) % 16) as u8;
            }
        }
    }

    // Input must have length ne1 (input dimension)
    let input: Vec<f32> = (0..ne1).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();

    // Use colmajor dispatch
    let result = matmul_q4k_f32_colmajor_dispatch(&q4k_data, &input, ne0, ne1);

    // Output has ne0 elements
    assert_eq!(result.len(), ne0);
    for (i, &val) in result.iter().enumerate() {
        assert!(val.is_finite(), "Result[{}] is not finite: {}", i, val);
    }
}

#[test]
fn test_compute_chunk_scalar_small() {
    // Directly test compute_chunk_q4k_scalar
    let in_dim = 256;
    let out_dim = 4;
    let num_blocks_per_row = 1;
    let row_bytes = SUPER_BLOCK_BYTES;

    let mut q4k_data = vec![0u8; out_dim * row_bytes];
    for row in 0..out_dim {
        let offset = row * row_bytes;
        // d = 1.0 as f16
        q4k_data[offset] = 0x00;
        q4k_data[offset + 1] = 0x3C;
        // dmin = 0.0
        q4k_data[offset + 2] = 0x00;
        q4k_data[offset + 3] = 0x00;
        // scales = 1
        for i in 0..12 {
            q4k_data[offset + 4 + i] = 0x01;
        }
        // qs = all zeros (simplest case)
        for i in 0..128 {
            q4k_data[offset + 16 + i] = 0x00;
        }
    }

    let input = vec![1.0f32; in_dim];
    let mut chunk = vec![0.0f32; out_dim];

    compute_chunk_q4k_scalar(
        &q4k_data,
        &input,
        &mut chunk,
        0,
        out_dim,
        in_dim,
        num_blocks_per_row,
        row_bytes,
    );

    // With qs=0, d=1, scales=1, dmin=0, result should be negative
    // Each element: d * scale * 0 - dmin * min = 0 - 0 = 0
    // Actually with all zeros in qs and dmin=0, output should be 0
    for (i, &val) in chunk.iter().enumerate() {
        assert!(val.is_finite(), "Chunk[{}] is not finite: {}", i, val);
    }
}
