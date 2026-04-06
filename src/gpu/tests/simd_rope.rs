
#[test]
fn test_simd_rope_matches_scalar() {
    let seq_len = 2;
    let head_dim = 4;
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let theta = 10000.0;

    let scalar_out = scalar_rope(&input, seq_len, head_dim, theta);
    let simd_out = simd_rope(&input, seq_len, head_dim, theta);

    for (s, si) in scalar_out.iter().zip(simd_out.iter()) {
        assert!((s - si).abs() < 1e-4, "SIMD rope should match scalar");
    }
}

// ============================================================================
// Coverage Tests - Batch/FFN Operations
// ============================================================================

#[test]
fn test_batch_embed_basic() {
    let vocab_size = 10;
    let hidden_dim = 4;
    let embedding_table: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|i| i as f32 * 0.1)
        .collect();
    let tokens = vec![0, 2, 5];

    let output = batch_embed(&embedding_table, &tokens, hidden_dim);

    assert_eq!(output.len(), 3 * hidden_dim);
    // Token 0 should have embedding [0.0, 0.1, 0.2, 0.3]
    assert!((output[0] - 0.0).abs() < 1e-5);
    assert!((output[1] - 0.1).abs() < 1e-5);
}

#[test]
fn test_batch_embed_single_token() {
    let embedding_table = vec![1.0, 2.0, 3.0, 4.0]; // vocab_size=1, hidden_dim=4
    let tokens = vec![0];

    let output = batch_embed(&embedding_table, &tokens, 4);

    assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_sequential_ffn_basic() {
    let hidden_dim = 4;
    let intermediate_dim = 8;
    let input = vec![1.0; hidden_dim];
    let up_weight = vec![0.1; hidden_dim * intermediate_dim];
    let down_weight = vec![0.1; intermediate_dim * hidden_dim];

    let output = sequential_ffn(
        &input,
        &up_weight,
        &down_weight,
        hidden_dim,
        intermediate_dim,
    );

    assert_eq!(output.len(), hidden_dim);
    assert!(output.iter().all(|v| v.is_finite()));
}

#[test]
fn test_parallel_ffn_basic() {
    let hidden_dim = 4;
    let intermediate_dim = 8;
    let input = vec![1.0; hidden_dim];
    let up_weight = vec![0.1; hidden_dim * intermediate_dim];
    let down_weight = vec![0.1; intermediate_dim * hidden_dim];

    let output = parallel_ffn(
        &input,
        &up_weight,
        &down_weight,
        hidden_dim,
        intermediate_dim,
    );

    assert_eq!(output.len(), hidden_dim);
    assert!(output.iter().all(|v| v.is_finite()));
}

// ============================================================================
// Coverage Tests - LayerNorm
// ============================================================================

#[test]
fn test_standard_layernorm_basic() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0; 4];
    let beta = vec![0.0; 4];
    let eps = 1e-5;

    let output = standard_layernorm(&input, &gamma, &beta, eps);

    assert_eq!(output.len(), 4);
    // Should be normalized (mean ~0, std ~1)
    let mean: f32 = output.iter().sum::<f32>() / output.len() as f32;
    assert!(mean.abs() < 1e-4, "Mean should be ~0 after layernorm");
}
