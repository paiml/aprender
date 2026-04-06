
// ============================================================================
// Tests for fused_q4k_q8k_ffn_up_gate_into
// ============================================================================

#[test]
fn test_q4k_q8k_ffn_up_gate_into_basic_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let up_weights = generate_q4k_weights(out_dim, in_dim);
    let gate_weights = generate_q4k_weights(out_dim, in_dim);
    let (q8k_scales, q8k_quants) = generate_q8k_activations(in_dim);
    let mut up_output = vec![0.0f32; out_dim];
    let mut gate_output = vec![0.0f32; out_dim];

    let result = fused_q4k_q8k_ffn_up_gate_into(
        &up_weights,
        &gate_weights,
        &q8k_scales,
        &q8k_quants,
        in_dim,
        out_dim,
        &mut up_output,
        &mut gate_output,
    );
    assert!(result.is_ok());

    for val in &up_output {
        assert!(val.is_finite());
    }
    for val in &gate_output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_q4k_q8k_ffn_up_gate_into_large_pk14() {
    let in_dim = 512;
    let out_dim = 256; // Multiple midi-tiles
    let up_weights = generate_q4k_weights(out_dim, in_dim);
    let gate_weights = generate_q4k_weights(out_dim, in_dim);
    let (q8k_scales, q8k_quants) = generate_q8k_activations(in_dim);
    let mut up_output = vec![0.0f32; out_dim];
    let mut gate_output = vec![0.0f32; out_dim];

    let result = fused_q4k_q8k_ffn_up_gate_into(
        &up_weights,
        &gate_weights,
        &q8k_scales,
        &q8k_quants,
        in_dim,
        out_dim,
        &mut up_output,
        &mut gate_output,
    );
    assert!(result.is_ok());
}

#[test]
fn test_q4k_q8k_ffn_up_gate_into_partial_midi_tile_pk14() {
    let in_dim = 256;
    let out_dim = 100; // 1 midi-tile + 36 remainder
    let up_weights = generate_q4k_weights(out_dim, in_dim);
    let gate_weights = generate_q4k_weights(out_dim, in_dim);
    let (q8k_scales, q8k_quants) = generate_q8k_activations(in_dim);
    let mut up_output = vec![0.0f32; out_dim];
    let mut gate_output = vec![0.0f32; out_dim];

    let result = fused_q4k_q8k_ffn_up_gate_into(
        &up_weights,
        &gate_weights,
        &q8k_scales,
        &q8k_quants,
        in_dim,
        out_dim,
        &mut up_output,
        &mut gate_output,
    );
    assert!(result.is_ok());
}

#[test]
fn test_q4k_q8k_ffn_up_gate_into_up_weight_error_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let up_weights = vec![0u8; 10]; // Too small
    let gate_weights = generate_q4k_weights(out_dim, in_dim);
    let (q8k_scales, q8k_quants) = generate_q8k_activations(in_dim);
    let mut up_output = vec![0.0f32; out_dim];
    let mut gate_output = vec![0.0f32; out_dim];

    let result = fused_q4k_q8k_ffn_up_gate_into(
        &up_weights,
        &gate_weights,
        &q8k_scales,
        &q8k_quants,
        in_dim,
        out_dim,
        &mut up_output,
        &mut gate_output,
    );
    assert!(result.is_err());
}

#[test]
fn test_q4k_q8k_ffn_up_gate_into_gate_weight_error_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let up_weights = generate_q4k_weights(out_dim, in_dim);
    let gate_weights = vec![0u8; 10]; // Too small
    let (q8k_scales, q8k_quants) = generate_q8k_activations(in_dim);
    let mut up_output = vec![0.0f32; out_dim];
    let mut gate_output = vec![0.0f32; out_dim];

    let result = fused_q4k_q8k_ffn_up_gate_into(
        &up_weights,
        &gate_weights,
        &q8k_scales,
        &q8k_quants,
        in_dim,
        out_dim,
        &mut up_output,
        &mut gate_output,
    );
    assert!(result.is_err());
}

#[test]
fn test_q4k_q8k_ffn_up_gate_into_up_output_error_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let up_weights = generate_q4k_weights(out_dim, in_dim);
    let gate_weights = generate_q4k_weights(out_dim, in_dim);
    let (q8k_scales, q8k_quants) = generate_q8k_activations(in_dim);
    let mut up_output = vec![0.0f32; 32]; // Too small
    let mut gate_output = vec![0.0f32; out_dim];

    let result = fused_q4k_q8k_ffn_up_gate_into(
        &up_weights,
        &gate_weights,
        &q8k_scales,
        &q8k_quants,
        in_dim,
        out_dim,
        &mut up_output,
        &mut gate_output,
    );
    assert!(result.is_err());
}

#[test]
fn test_q4k_q8k_ffn_up_gate_into_gate_output_error_pk14() {
    let in_dim = 256;
    let out_dim = 64;
    let up_weights = generate_q4k_weights(out_dim, in_dim);
    let gate_weights = generate_q4k_weights(out_dim, in_dim);
    let (q8k_scales, q8k_quants) = generate_q8k_activations(in_dim);
    let mut up_output = vec![0.0f32; out_dim];
    let mut gate_output = vec![0.0f32; 32]; // Too small

    let result = fused_q4k_q8k_ffn_up_gate_into(
        &up_weights,
        &gate_weights,
        &q8k_scales,
        &q8k_quants,
        in_dim,
        out_dim,
        &mut up_output,
        &mut gate_output,
    );
    assert!(result.is_err());
}

// LAYOUT-002: Backward-compat alias tests DELETED (2026-02-03)
// ONE WAY ONLY: Use fused_q{4,5,6}k_parallel_matvec* functions directly

// ============================================================================
// Determinism and consistency tests
// ============================================================================

#[test]
fn test_q4k_parallel_matvec_deterministic_pk14() {
    let in_dim = 256;
    let out_dim = 512; // Parallel path
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![0.5f32; in_dim];

    // Run multiple times and check determinism
    let result1 = fused_q4k_parallel_matvec(&weights, &activations, in_dim, out_dim).expect("test value should be present");
    let result2 = fused_q4k_parallel_matvec(&weights, &activations, in_dim, out_dim).expect("test value should be present");
    let result3 = fused_q4k_parallel_matvec(&weights, &activations, in_dim, out_dim).expect("test value should be present");

    for i in 0..out_dim {
        assert_eq!(
            result1[i], result2[i],
            "Mismatch at index {} between run 1 and 2",
            i
        );
        assert_eq!(
            result2[i], result3[i],
            "Mismatch at index {} between run 2 and 3",
            i
        );
    }
}

#[test]
fn test_q4k_parallel_vs_tiled_consistency_pk14() {
    let in_dim = 256;
    let out_dim = 128; // Sequential path for parallel version
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];

    let tiled_result =
        fused_q4k_tiled_matvec(&weights, &activations, in_dim, out_dim, None).expect("test value should be present");
    let parallel_result =
        fused_q4k_parallel_matvec(&weights, &activations, in_dim, out_dim).expect("test value should be present");

    // Both should produce the same output
    for i in 0..out_dim {
        let diff = (tiled_result[i] - parallel_result[i]).abs();
        assert!(
            diff < 1e-5,
            "Mismatch at index {}: tiled={}, parallel={}, diff={}",
            i,
            tiled_result[i],
            parallel_result[i],
            diff
        );
    }
}

#[test]
fn test_q4k_matvec_vs_matvec_into_consistency_pk14() {
    let in_dim = 256;
    let out_dim = 128;
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![1.0f32; in_dim];

    let alloc_result = fused_q4k_parallel_matvec(&weights, &activations, in_dim, out_dim).expect("test value should be present");

    let mut into_result = vec![0.0f32; out_dim];
    fused_q4k_parallel_matvec_into(&weights, &activations, in_dim, out_dim, &mut into_result)
        .expect("test value should be present");

    for i in 0..out_dim {
        let diff = (alloc_result[i] - into_result[i]).abs();
        assert!(
            diff < 1e-6,
            "Mismatch at index {}: alloc={}, into={}, diff={}",
            i,
            alloc_result[i],
            into_result[i],
            diff
        );
    }
}

// ============================================================================
// Edge case tests for dimension handling
// ============================================================================

#[test]
fn test_q4k_multiple_superblocks_per_row_pk14() {
    let in_dim = 768; // 3 super-blocks per row (768 / 256 = 3)
    let out_dim = 32;
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![0.1f32; in_dim];

    let result = fused_q4k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_ok());

    let output = result.expect("test value should be present");
    assert_eq!(output.len(), out_dim);
}

#[test]
fn test_q4k_non_multiple_of_qkk_in_dim_pk14() {
    // in_dim not a multiple of QK_K (256) - uses div_ceil
    let in_dim = 300; // ceil(300/256) = 2 super-blocks
    let out_dim = 16;
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![0.5f32; in_dim];

    let result = fused_q4k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_ok());
}

#[test]
fn test_q5k_multiple_superblocks_pk14() {
    let in_dim = 512; // 2 super-blocks
    let out_dim = 64;
    let weights = generate_q5k_weights(out_dim, in_dim);
    let activations = vec![0.25f32; in_dim];

    let result = fused_q5k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_ok());
}

#[test]
fn test_q6k_multiple_superblocks_pk14() {
    let in_dim = 512; // 2 super-blocks
    let out_dim = 64;
    let weights = generate_q6k_weights(out_dim, in_dim);
    let activations = vec![0.25f32; in_dim];

    let result = fused_q6k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_ok());
}

// ============================================================================
// Large scale tests (for parallel path coverage)
// ============================================================================

#[test]
fn test_q4k_large_parallel_execution_pk14() {
    let in_dim = 1024; // 4 super-blocks
    let out_dim = 1024; // Well above 256 threshold - uses parallel path
    let weights = generate_q4k_weights(out_dim, in_dim);
    let activations = vec![0.01f32; in_dim];

    let result = fused_q4k_parallel_matvec(&weights, &activations, in_dim, out_dim);
    assert!(result.is_ok());

    let output = result.expect("test value should be present");
    assert_eq!(output.len(), out_dim);

    // All outputs should be finite
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_q4k_q8k_large_parallel_pk14() {
    let in_dim = 512;
    let out_dim = 512;
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
fn test_ffn_up_gate_large_parallel_pk14() {
    let in_dim = 512;
    let out_dim = 512;
    let up_weights = generate_q4k_weights(out_dim, in_dim);
    let gate_weights = generate_q4k_weights(out_dim, in_dim);
    let (q8k_scales, q8k_quants) = generate_q8k_activations(in_dim);
    let mut up_output = vec![0.0f32; out_dim];
    let mut gate_output = vec![0.0f32; out_dim];

    let result = fused_q4k_q8k_ffn_up_gate_into(
        &up_weights,
        &gate_weights,
        &q8k_scales,
        &q8k_quants,
        in_dim,
        out_dim,
        &mut up_output,
        &mut gate_output,
    );
    assert!(result.is_ok());

    for val in &up_output {
        assert!(val.is_finite());
    }
    for val in &gate_output {
        assert!(val.is_finite());
    }
}
