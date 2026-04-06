
#[test]
fn test_q4k_parallel_matvec_into_larger_buffer_pk14() {
    let in_dim = 256;
    let out_dim = 32;
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];
    let mut output = vec![999.0f32; 64]; // Larger than needed

    let result =
        fused_q4k_parallel_matvec_into(&weights, &activations, in_dim, out_dim, &mut output);
    assert!(result.is_ok());

    // First out_dim elements should be updated
    for val in &output[..out_dim] {
        assert!(val.is_finite());
        assert_ne!(*val, 999.0); // Should have been overwritten
    }

    // Rest should remain unchanged
    for val in &output[out_dim..] {
        assert_eq!(*val, 999.0);
    }
}

// ============================================================================
// Tests for fused_q5k_parallel_matvec
// ============================================================================

#[test]
fn test_q5k_parallel_matvec_basic_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let weights = generate_q5k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];

    let result = fused_q5k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_ok());

    let output = result.expect("test value should be present");
    assert_eq!(output.len(), out_dim);
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_q5k_parallel_matvec_large_pk14() {
    let in_dim = 512;
    let out_dim = 512;
    let weights = generate_q5k_weights(out_dim, in_dim);
    let activations = vec![0.5f32; in_dim];

    let result = fused_q5k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_ok());

    let output = result.expect("test value should be present");
    assert_eq!(output.len(), out_dim);
}

#[test]
fn test_q5k_parallel_matvec_single_row_pk14() {
    let in_dim = 256;
    let out_dim = 1;
    let weights = generate_q5k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];

    let result = fused_q5k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), 1);
}

#[test]
fn test_q5k_parallel_matvec_weight_error_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let weights = vec![0u8; 10]; // Too small
    let activations = vec![1.0f32; in_dim];

    let result = fused_q5k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Q5_K"));
}

#[test]
fn test_q5k_parallel_matvec_activation_error_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let weights = generate_q5k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; 128]; // Wrong size

    let result = fused_q5k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_err());
}

// ============================================================================
// Tests for fused_q5k_parallel_matvec_into
// ============================================================================

#[test]
fn test_q5k_parallel_matvec_into_basic_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let weights = generate_q5k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];
    let mut output = vec![0.0f32; out_dim];

    let result =
        fused_q5k_parallel_matvec_into(&weights, &activations, in_dim, out_dim, &mut output);
    assert!(result.is_ok());

    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_q5k_parallel_matvec_into_large_pk14() {
    let in_dim = 512;
    let out_dim = 256;
    let weights = generate_q5k_weights(out_dim, in_dim);
    let activations = vec![0.5f32; in_dim];
    let mut output = vec![0.0f32; out_dim];

    let result =
        fused_q5k_parallel_matvec_into(&weights, &activations, in_dim, out_dim, &mut output);
    assert!(result.is_ok());
}

#[test]
fn test_q5k_parallel_matvec_into_output_error_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let weights = generate_q5k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];
    let mut output = vec![0.0f32; 32]; // Too small

    let result =
        fused_q5k_parallel_matvec_into(&weights, &activations, in_dim, out_dim, &mut output);
    assert!(result.is_err());
}

// ============================================================================
// Tests for fused_q6k_parallel_matvec
// ============================================================================

#[test]
fn test_q6k_parallel_matvec_basic_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let weights = generate_q6k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];

    let result = fused_q6k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_ok());

    let output = result.expect("test value should be present");
    assert_eq!(output.len(), out_dim);
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_q6k_parallel_matvec_large_pk14() {
    let in_dim = 512;
    let out_dim = 512;
    let weights = generate_q6k_weights(out_dim, in_dim);
    let activations = vec![0.5f32; in_dim];

    let result = fused_q6k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_ok());

    let output = result.expect("test value should be present");
    assert_eq!(output.len(), out_dim);
}

#[test]
fn test_q6k_parallel_matvec_single_row_pk14() {
    let in_dim = 256;
    let out_dim = 1;
    let weights = generate_q6k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];

    let result = fused_q6k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), 1);
}

#[test]
fn test_q6k_parallel_matvec_weight_error_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let weights = vec![0u8; 10]; // Too small
    let activations = vec![1.0f32; in_dim];

    let result = fused_q6k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Q6_K"));
}

#[test]
fn test_q6k_parallel_matvec_activation_error_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let weights = generate_q6k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; 128]; // Wrong size

    let result = fused_q6k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_err());
}

// ============================================================================
// Tests for fused_q6k_parallel_matvec_into (with TCB tiling)
// ============================================================================

#[test]
fn test_q6k_parallel_matvec_into_basic_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let weights = generate_q6k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];
    let mut output = vec![0.0f32; out_dim];

    let result =
        fused_q6k_parallel_matvec_into(&weights, &activations, in_dim, out_dim, &mut output);
    assert!(result.is_ok());

    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_q6k_parallel_matvec_into_midi_tile_boundary_pk14() {
    let in_dim = 256;
    let out_dim = 128; // 2 midi-tiles
    let weights = generate_q6k_weights(out_dim, in_dim);
    let activations = vec![0.5f32; in_dim];
    let mut output = vec![0.0f32; out_dim];

    let result =
        fused_q6k_parallel_matvec_into(&weights, &activations, in_dim, out_dim, &mut output);
    assert!(result.is_ok());
}

#[test]
fn test_q6k_parallel_matvec_into_partial_midi_tile_pk14() {
    let in_dim = 256;
    let out_dim = 200; // 3 midi-tiles + 8 remainder
    let weights = generate_q6k_weights(out_dim, in_dim);
    let activations = vec![0.25f32; in_dim];
    let mut output = vec![0.0f32; out_dim];

    let result =
        fused_q6k_parallel_matvec_into(&weights, &activations, in_dim, out_dim, &mut output);
    assert!(result.is_ok());
}

#[test]
fn test_q6k_parallel_matvec_into_output_error_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let weights = generate_q6k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];
    let mut output = vec![0.0f32; 32]; // Too small

    let result =
        fused_q6k_parallel_matvec_into(&weights, &activations, in_dim, out_dim, &mut output);
    assert!(result.is_err());
}

// ============================================================================
// Tests for fused_q4k_q8k_parallel_matvec_into (TCB tiling with micro-tiles)
// ============================================================================

#[test]
fn test_q4k_q8k_parallel_matvec_into_basic_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let weights = generate_q4k_weights(out_dim, in_dim);
    let (q8k_scales, q8k_quants) = generate_q8k_activations(in_dim);
    let mut output = vec![0.0f32; out_dim];

    let result = fused_q4k_q8k_parallel_matvec_into(
        &weights,
        &q8k_scales,
        &q8k_quants,
        in_dim,
        out_dim,
        &mut output,
    );
    assert!(result.is_ok());

    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_q4k_q8k_parallel_matvec_into_micro_tile_boundary_pk14() {
    let in_dim = 256;
    let out_dim = 8; // 2 micro-tiles of 4 rows
    let weights = generate_q4k_weights(out_dim, in_dim);
    let (q8k_scales, q8k_quants) = generate_q8k_activations(in_dim);
    let mut output = vec![0.0f32; out_dim];

    let result = fused_q4k_q8k_parallel_matvec_into(
        &weights,
        &q8k_scales,
        &q8k_quants,
        in_dim,
        out_dim,
        &mut output,
    );
    assert!(result.is_ok());
}

#[test]
fn test_q4k_q8k_parallel_matvec_into_with_remainder_pk14() {
    let in_dim = 256;
    let out_dim = 70; // 17 micro-tiles + 2 remainder rows
    let weights = generate_q4k_weights(out_dim, in_dim);
    let (q8k_scales, q8k_quants) = generate_q8k_activations(in_dim);
    let mut output = vec![0.0f32; out_dim];

    let result = fused_q4k_q8k_parallel_matvec_into(
        &weights,
        &q8k_scales,
        &q8k_quants,
        in_dim,
        out_dim,
        &mut output,
    );
    assert!(result.is_ok());
}

#[test]
fn test_q4k_q8k_parallel_matvec_into_single_row_pk14() {
    let in_dim = 256;
    let out_dim = 1; // Less than one micro-tile
    let weights = generate_q4k_weights(out_dim, in_dim);
    let (q8k_scales, q8k_quants) = generate_q8k_activations(in_dim);
    let mut output = vec![0.0f32; out_dim];

    let result = fused_q4k_q8k_parallel_matvec_into(
        &weights,
        &q8k_scales,
        &q8k_quants,
        in_dim,
        out_dim,
        &mut output,
    );
    assert!(result.is_ok());
}

#[test]
fn test_q4k_q8k_parallel_matvec_into_3_rows_pk14() {
    let in_dim = 256;
    let out_dim = 3; // Less than one micro-tile (remainder only)
    let weights = generate_q4k_weights(out_dim, in_dim);
    let (q8k_scales, q8k_quants) = generate_q8k_activations(in_dim);
    let mut output = vec![0.0f32; out_dim];

    let result = fused_q4k_q8k_parallel_matvec_into(
        &weights,
        &q8k_scales,
        &q8k_quants,
        in_dim,
        out_dim,
        &mut output,
    );
    assert!(result.is_ok());
}

#[test]
fn test_q4k_q8k_parallel_matvec_into_large_pk14() {
    let in_dim = 512;
    let out_dim = 256; // Multiple midi-tiles
    let weights = generate_q4k_weights(out_dim, in_dim);
    let (q8k_scales, q8k_quants) = generate_q8k_activations(in_dim);
    let mut output = vec![0.0f32; out_dim];

    let result = fused_q4k_q8k_parallel_matvec_into(
        &weights,
        &q8k_scales,
        &q8k_quants,
        in_dim,
        out_dim,
        &mut output,
    );
    assert!(result.is_ok());
}

#[test]
fn test_q4k_q8k_parallel_matvec_into_weight_error_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let weights = vec![0u8; 10]; // Too small
    let (q8k_scales, q8k_quants) = generate_q8k_activations(in_dim);
    let mut output = vec![0.0f32; out_dim];

    let result = fused_q4k_q8k_parallel_matvec_into(
        &weights,
        &q8k_scales,
        &q8k_quants,
        in_dim,
        out_dim,
        &mut output,
    );
    assert!(result.is_err());
}

#[test]
fn test_q4k_q8k_parallel_matvec_into_output_error_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let weights = generate_q4k_weights(out_dim, in_dim);
    let (q8k_scales, q8k_quants) = generate_q8k_activations(in_dim);
    let mut output = vec![0.0f32; 32]; // Too small

    let result = fused_q4k_q8k_parallel_matvec_into(
        &weights,
        &q8k_scales,
        &q8k_quants,
        in_dim,
        out_dim,
        &mut output,
    );
    assert!(result.is_err());
}
