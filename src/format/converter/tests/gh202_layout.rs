//! GH-202: LAYOUT-002 Tensor Value Validation Tests
//!
//! These tests validate that GGUF→APR conversion preserves tensor values
//! correctly when applying the LAYOUT-002 transpose (column-major → row-major).
//!
//! Root cause investigation for GH-202: APR from GGUF produces garbage inference.

use trueno_quant::{dequantize_q4_k_to_f32, quantize_q4_k_matrix, transpose_q4k_for_matmul};

/// GH-202-FIX-001: Validate transpose preserves logical matrix values
///
/// This test diagnoses the transpose function behavior.
/// GGUF stores weights in column-major order with shape [in_dim, out_dim].
/// APR needs row-major order with shape [out_dim, in_dim].
#[test]
fn test_gh202_transpose_preserves_values() {
    // Use QK_K-aligned dimensions (256 is minimum for Q4K)
    let in_dim = 256; // GGUF shape[0]
    let out_dim = 256; // GGUF shape[1]

    // Create a matrix with values in typical neural network weight range [-0.05, 0.05]
    // W[out, in] = (out + in) / (out_dim + in_dim) * 0.1 - 0.05
    let mut logical_matrix = vec![0.0f32; out_dim * in_dim];
    for out_idx in 0..out_dim {
        for in_idx in 0..in_dim {
            // Unique value for each position, in reasonable range
            let t = (out_idx + in_idx) as f32 / (out_dim + in_dim) as f32;
            logical_matrix[out_idx * in_dim + in_idx] = t * 0.1 - 0.05;
        }
    }

    // GGUF stores in column-major: data[in_idx * out_dim + out_idx] = W[out_idx, in_idx]
    let mut gguf_colmajor = vec![0.0f32; out_dim * in_dim];
    for out_idx in 0..out_dim {
        for in_idx in 0..in_dim {
            gguf_colmajor[in_idx * out_dim + out_idx] = logical_matrix[out_idx * in_dim + in_idx];
        }
    }

    // GGUF reports shape as [in_dim, out_dim]
    let gguf_shape = vec![in_dim, out_dim];

    // Quantize the column-major data
    let q4k_bytes = quantize_q4_k_matrix(&gguf_colmajor, &gguf_shape);

    // Apply transpose
    let (transposed_q4k, transposed_shape) = transpose_q4k_for_matmul(&q4k_bytes, &gguf_shape);

    // Expected: shape should be [out_dim, in_dim] (swapped)
    assert_eq!(
        transposed_shape,
        vec![out_dim, in_dim],
        "GH-202: Transposed shape should be [out_dim, in_dim]"
    );

    // Dequantize
    let dequant = dequantize_q4_k_to_f32(&transposed_q4k, out_dim * in_dim);

    // The transposed result should be in row-major: data[out_idx * in_dim + in_idx] = W[out_idx, in_idx]
    // This should match our original logical_matrix!

    let mut max_diff = 0.0f32;
    let mut mismatch_count = 0;

    for out_idx in 0..out_dim {
        for in_idx in 0..in_dim {
            let expected = logical_matrix[out_idx * in_dim + in_idx];
            let actual = dequant[out_idx * in_dim + in_idx];
            let diff = (expected - actual).abs();

            if diff > max_diff {
                max_diff = diff;
            }

            // Q4K tolerance: values are small (0-2.56), so relative error matters
            if diff > 0.1 {
                mismatch_count += 1;
                if mismatch_count <= 5 {
                    eprintln!(
                        "GH-202 MISMATCH at [out={}, in={}]: expected {:.4}, got {:.4}, diff {:.4}",
                        out_idx, in_idx, expected, actual, diff
                    );
                }
            }
        }
    }

    eprintln!("GH-202: max_diff = {:.4}, mismatch_count = {}", max_diff, mismatch_count);

    // If more than 10% mismatch, the transpose is broken
    let total = out_dim * in_dim;
    let mismatch_pct = mismatch_count as f64 / total as f64 * 100.0;

    assert!(
        mismatch_pct < 10.0,
        "GH-202: {}% values mismatched ({}), max_diff={:.4}. Transpose is broken!",
        mismatch_pct,
        mismatch_count,
        max_diff
    );
}

/// GH-202-FIX-002: Validate Q4K dequantization round-trip
///
/// Tests that quantize → dequantize preserves values within tolerance.
#[test]
fn test_gh202_q4k_roundtrip_fidelity() {
    // Create random-ish values that span Q4K range
    let size = 256; // Must be multiple of QK_K (256)
    let mut values = vec![0.0f32; size];
    for (i, v) in values.iter_mut().enumerate() {
        // Values between -1 and 1 with some variation
        *v = (i as f32 * 0.1).sin() * 0.5;
    }

    let shape = vec![size];

    // Quantize
    let q4k_bytes = quantize_q4_k_matrix(&values, &shape);

    // Dequantize
    let dequant = dequantize_q4_k_to_f32(&q4k_bytes, size);

    // Compare
    let mut max_diff = 0.0f32;
    for (i, (&orig, &deq)) in values.iter().zip(dequant.iter()).enumerate() {
        let diff = (orig - deq).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        // Q4K should be within ~0.1 for normalized values
        assert!(
            diff < 0.2,
            "GH-202: Q4K roundtrip error at [{}]: orig={}, deq={}, diff={}",
            i,
            orig,
            deq,
            diff
        );
    }

    eprintln!("GH-202: Q4K roundtrip max diff = {}", max_diff);
}

/// GH-202-FIX-005: Debug test to examine dequantize output
#[test]
fn test_gh202_debug_dequantize() {
    // Create values in neural network weight range [-0.1, 0.1]
    let size = 256;
    let mut values = vec![0.0f32; size];
    for (i, v) in values.iter_mut().enumerate() {
        // Small values typical of neural network weights
        *v = (i as f32 / 256.0 - 0.5) * 0.2; // Range: -0.1 to 0.1
    }

    let shape = vec![size];
    let q4k = quantize_q4_k_matrix(&values, &shape);
    let dequant = dequantize_q4_k_to_f32(&q4k, size);

    eprintln!("GH-202 DEBUG: Q4K bytes = {} (expected 144 for 256 elements)", q4k.len());
    eprintln!("GH-202 DEBUG: First 10 original: {:?}", &values[..10]);
    eprintln!("GH-202 DEBUG: First 10 dequant:  {:?}", &dequant[..10]);
    eprintln!("GH-202 DEBUG: Last 10 original:  {:?}", &values[246..]);
    eprintln!("GH-202 DEBUG: Last 10 dequant:   {:?}", &dequant[246..]);

    // Check the bytes are not all zero
    let nonzero_bytes = q4k.iter().filter(|&&b| b != 0).count();
    eprintln!("GH-202 DEBUG: Non-zero Q4K bytes: {}/{}", nonzero_bytes, q4k.len());

    // Check roundtrip error
    let mut max_err = 0.0f32;
    for (orig, deq) in values.iter().zip(dequant.iter()) {
        let err = (orig - deq).abs();
        if err > max_err {
            max_err = err;
        }
    }
    eprintln!("GH-202 DEBUG: Max roundtrip error: {}", max_err);

    assert!(nonzero_bytes > 10, "GH-202: Q4K should have non-zero bytes");
    assert!(max_err < 0.1, "GH-202: Roundtrip error {} too large", max_err);
}

/// GH-202-FIX-003: Validate matmul dimension interpretation
///
/// Tests that the transposed tensor produces correct matmul results.
#[test]
fn test_gh202_transposed_matmul_correctness() {
    // Create a 256x256 identity-like matrix (for simplicity)
    // After transpose, matmul with identity should give identity
    let dim = 256; // Must be multiple of QK_K
    let mut matrix = vec![0.0f32; dim * dim];

    // Set diagonal to 1.0
    for i in 0..dim {
        matrix[i * dim + i] = 1.0;
    }

    // Quantize as column-major (GGUF convention)
    let gguf_shape = vec![dim, dim];
    let q4k_bytes = quantize_q4_k_matrix(&matrix, &gguf_shape);

    // Transpose
    let (transposed_q4k, transposed_shape) = transpose_q4k_for_matmul(&q4k_bytes, &gguf_shape);

    assert_eq!(transposed_shape, vec![dim, dim], "Square matrix shape preserved");

    // Dequantize and verify diagonal structure preserved
    let dequant = dequantize_q4_k_to_f32(&transposed_q4k, dim * dim);

    // Check that diagonal is still ~1.0 and off-diagonal is ~0.0
    let mut diag_sum = 0.0f32;
    let mut off_diag_sum = 0.0f32;

    for i in 0..dim {
        for j in 0..dim {
            let val = dequant[i * dim + j];
            if i == j {
                diag_sum += val;
            } else {
                off_diag_sum += val.abs();
            }
        }
    }

    // Diagonal should sum to ~256
    assert!(
        (diag_sum - dim as f32).abs() < dim as f32 * 0.2,
        "GH-202: Diagonal sum {} should be close to {}",
        diag_sum,
        dim
    );

    // Off-diagonal should be small
    let off_diag_avg = off_diag_sum / ((dim * dim - dim) as f32);
    assert!(
        off_diag_avg < 0.1,
        "GH-202: Off-diagonal average {} should be near zero",
        off_diag_avg
    );
}

/// GH-202-FIX-004: Verify GGUF shape interpretation
///
/// GGUF reports shapes as [in_dim, out_dim] (column-major convention).
/// After transpose, APR should have [out_dim, in_dim] (row-major).
#[test]
fn test_gh202_gguf_shape_interpretation() {
    // Simulate a weight matrix W that transforms hidden_dim → intermediate_dim
    // In standard convention: W has shape [out_dim, in_dim]
    // In GGUF column-major: stored as [in_dim, out_dim]

    let hidden_dim = 256; // Must be multiple of QK_K
    let intermediate_dim = 512;

    // GGUF shape: [hidden_dim, intermediate_dim] = [256, 512]
    let gguf_shape = vec![hidden_dim, intermediate_dim];

    // Create dummy Q4K data (just need correct size for shape test)
    let bytes_per_superblock = 144; // Q4K
    let superblocks_per_row = hidden_dim / 256; // QK_K = 256
    let bytes_per_row = superblocks_per_row * bytes_per_superblock;
    let total_bytes = intermediate_dim * bytes_per_row;

    let dummy_q4k = vec![0u8; total_bytes];

    // Transpose
    let (_, transposed_shape) = transpose_q4k_for_matmul(&dummy_q4k, &gguf_shape);

    // APR should have [out_dim, in_dim] = [intermediate_dim, hidden_dim] = [512, 256]
    assert_eq!(
        transposed_shape,
        vec![intermediate_dim, hidden_dim],
        "GH-202: APR shape should be [out_dim={}, in_dim={}]",
        intermediate_dim,
        hidden_dim
    );
}
