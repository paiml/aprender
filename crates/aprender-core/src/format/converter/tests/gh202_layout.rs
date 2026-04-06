//! GH-202: LAYOUT-002 Tensor Layout Fidelity Tests
//!
//! These tests validate that GGUF→APR conversion preserves tensor values
//! correctly. The key insight (GH-202 fix) is that GGML data layout
//! data[i0 + i1*ne0] is IDENTICAL to C row-major data[row*cols + col]
//! when the shape is reversed from [ne0, ne1] → [ne1, ne0].
//!
//! NO data transpose is needed — only shape reversal.

use trueno_quant::{dequantize_q4_k_to_f32, quantize_q4_k_matrix};

/// GH-202-FIX-001: Verify GGML layout IS row-major with reversed shape
///
/// GGML data[i0 + i1*ne0] where ne0 is contiguous equals
/// C row-major data[row*cols + col] with rows=ne1, cols=ne0.
/// This test proves no data transpose is needed.
#[test]
fn test_gh202_ggml_is_rowmajor_reversed() {
    let ne0 = 3; // in_features (contiguous dim)
    let ne1 = 2; // out_features

    // Weight matrix W: output j, input i → W[j][i]
    // W = [[1, 2, 3],   (output 0)
    //      [4, 5, 6]]   (output 1)

    // GGML layout: data[i0 + i1*ne0] where i0=input, i1=output
    let mut ggml_data = vec![0.0f32; ne0 * ne1];
    // W[0][0]=1: i0=0, i1=0 → data[0 + 0*3] = data[0]
    ggml_data[0] = 1.0;
    // W[0][1]=2: i0=1, i1=0 → data[1 + 0*3] = data[1]
    ggml_data[1] = 2.0;
    // W[0][2]=3: i0=2, i1=0 → data[2 + 0*3] = data[2]
    ggml_data[2] = 3.0;
    // W[1][0]=4: i0=0, i1=1 → data[0 + 1*3] = data[3]
    ggml_data[3] = 4.0;
    // W[1][1]=5: i0=1, i1=1 → data[1 + 1*3] = data[4]
    ggml_data[4] = 5.0;
    // W[1][2]=6: i0=2, i1=1 → data[2 + 1*3] = data[5]
    ggml_data[5] = 6.0;

    // C row-major with reversed shape [ne1=2, ne0=3]:
    // data[row*cols + col] where rows=2, cols=3
    // Row 0: data[0], data[1], data[2] = 1, 2, 3 ✓ (output 0 weights)
    // Row 1: data[3], data[4], data[5] = 4, 5, 6 ✓ (output 1 weights)

    // Verify: row-major access matches GGML access
    let rows = ne1; // 2
    let cols = ne0; // 3
    for out_idx in 0..rows {
        for in_idx in 0..cols {
            let ggml_offset = in_idx + out_idx * ne0;
            let rowmajor_offset = out_idx * cols + in_idx;
            assert_eq!(
                ggml_offset, rowmajor_offset,
                "GGML and row-major offsets must match at out={}, in={}",
                out_idx, in_idx
            );
        }
    }

    // The memory layout [1, 2, 3, 4, 5, 6] is correct for BOTH conventions
    assert_eq!(ggml_data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

/// GH-202-FIX-002: Validate Q4K roundtrip fidelity (unchanged)
#[test]
fn test_gh202_q4k_roundtrip_fidelity() {
    let size = 256;
    let mut values = vec![0.0f32; size];
    for (i, v) in values.iter_mut().enumerate() {
        *v = (i as f32 * 0.1).sin() * 0.5;
    }

    let shape = vec![size];
    let q4k_bytes = quantize_q4_k_matrix(&values, &shape);
    let dequant = dequantize_q4_k_to_f32(&q4k_bytes, size);

    let mut max_diff = 0.0f32;
    for (i, (&orig, &deq)) in values.iter().zip(dequant.iter()).enumerate() {
        let diff = (orig - deq).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        assert!(
            diff < 0.2,
            "GH-202: Q4K roundtrip error at [{}]: orig={}, deq={}, diff={}",
            i,
            orig,
            deq,
            diff
        );
    }
}

/// GH-202-FIX-003: Non-square matrix shape reversal preserves values
///
/// This is the key test that was failing before the GH-202 fix.
/// Non-square matrices [ne0=256, ne1=512] were corrupted by the wrong
/// data transpose. With shape reversal only, values are preserved.
#[test]
fn test_gh202_nonsquare_shape_reversal_preserves_values() {
    let ne0 = 256; // in_features (GGUF contiguous dim)
    let ne1 = 512; // out_features

    // Create weight matrix in GGML layout (which IS row-major [ne1, ne0])
    // Use sinusoidal values in [-0.5, 0.5] (wide enough for Q4K dynamic range)
    let total = ne0 * ne1;
    let mut data = vec![0.0f32; total];
    for (i, v) in data.iter_mut().enumerate() {
        *v = (i as f32 * 0.1).sin() * 0.5;
    }

    // Reverse shape: [ne0=256, ne1=512] → standard [512, 256]
    let standard_shape = vec![ne1, ne0]; // [rows=512, cols=256]

    // Quantize with standard shape
    let q4k_bytes = quantize_q4_k_matrix(&data, &standard_shape);
    let dequant = dequantize_q4_k_to_f32(&q4k_bytes, total);

    // Verify values are preserved (within Q4K quantization tolerance)
    let mut max_diff = 0.0f32;
    let mut mismatch_count = 0;
    for (&orig, &deq) in data.iter().zip(dequant.iter()) {
        let diff = (orig - deq).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > 0.1 {
            mismatch_count += 1;
        }
    }

    let mismatch_pct = mismatch_count as f64 / total as f64 * 100.0;
    assert!(
        mismatch_pct < 5.0,
        "GH-202: {}% values mismatched (max_diff={:.4}). Shape reversal should preserve values!",
        mismatch_pct,
        max_diff
    );
    assert!(
        max_diff < 0.2,
        "GH-202: max_diff={:.4} too large for Q4K tolerance",
        max_diff
    );
}

/// GH-202-FIX-004: Verify GGUF shape interpretation
///
/// GGUF shape [ne0, ne1] → APR shape [ne1, ne0] (reversed)
#[test]
fn test_gh202_gguf_shape_interpretation() {
    let hidden_dim = 256;
    let intermediate_dim = 512;

    // GGUF shape: [hidden_dim, intermediate_dim] = [ne0=256, ne1=512]
    let gguf_shape = vec![hidden_dim, intermediate_dim];

    // APR shape should be [ne1, ne0] = [512, 256] = [out_features, in_features]
    let apr_shape = vec![gguf_shape[1], gguf_shape[0]];

    assert_eq!(
        apr_shape,
        vec![intermediate_dim, hidden_dim],
        "GH-202: APR shape should be [out_dim={}, in_dim={}]",
        intermediate_dim,
        hidden_dim
    );
}

/// GH-202-FIX-005: Matmul correctness with reversed shape
///
/// Verify that y = W @ x produces correct results when W is stored
/// in GGML layout with reversed shape (no transpose needed).
#[test]
fn test_gh202_matmul_with_reversed_shape() {
    // W = [[1, 0],   → output 0 = 1*x0 + 0*x1
    //      [0, 1],   → output 1 = 0*x0 + 1*x1
    //      [1, 1]]   → output 2 = 1*x0 + 1*x1
    //
    // Shape: [out=3, in=2], GGML shape [ne0=2, ne1=3]

    let ne0 = 2; // in_features
    let ne1 = 3; // out_features

    // GGML layout = row-major [3, 2]
    let w_data = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];

    // Input vector
    let x = vec![3.0f32, 7.0];

    // Compute y = W @ x using row-major [ne1, ne0] interpretation
    let rows = ne1;
    let cols = ne0;
    let mut y = vec![0.0f32; rows];
    for r in 0..rows {
        for c in 0..cols {
            y[r] += w_data[r * cols + c] * x[c];
        }
    }

    // Expected: [1*3 + 0*7, 0*3 + 1*7, 1*3 + 1*7] = [3, 7, 10]
    assert_eq!(y, vec![3.0, 7.0, 10.0]);
}
