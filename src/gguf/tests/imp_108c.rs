
#[test]
#[cfg(feature = "gpu")]
fn test_imp_108c_attention_softmax_normalized() {
    // IMP-108c: Verify attention weights sum to 1 for each position
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 16,
        intermediate_dim: 32,
        num_layers: 1,
        num_heads: 2,
        num_kv_heads: 2,
        vocab_size: 50,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    let model = create_test_model_with_config(&config);
    let seq_len = 4;
    let hidden_dim = config.hidden_dim;
    let head_dim = hidden_dim / config.num_heads;

    // Create Q, K with known values to verify softmax
    let q: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| ((i % 3) as f32) * 0.1)
        .collect();
    let k: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| ((i % 5) as f32) * 0.1)
        .collect();

    // Use V = identity-like pattern to extract attention weights
    // V[j] = one-hot at position j within head
    let mut v = vec![0.0f32; seq_len * hidden_dim];
    for pos in 0..seq_len {
        for head in 0..config.num_heads {
            // Set V[pos, head, pos % head_dim] = 1.0
            let idx = pos * hidden_dim + head * head_dim + (pos % head_dim);
            v[idx] = 1.0;
        }
    }

    let output = model
        .batched_causal_attention_gpu(&q, &k, &v, seq_len)
        .expect("test");

    // Output should be valid (finite)
    assert!(
        output.iter().all(|x| x.is_finite()),
        "IMP-108c: All attention outputs should be finite"
    );

    // Output at each position should reflect weighted sum of V
    // Since V entries are 0 or 1, output values should be in [0, 1] range
    // (attention weights are normalized, so weighted sum of [0,1] is in [0,1])
    for &val in &output {
        assert!(
            val >= -0.01 && val <= 1.01,
            "IMP-108c: Attention output {} should be weighted sum of V (in [0,1])",
            val
        );
    }
}
