
// =============================================================================
// Edge Cases
// =============================================================================

#[test]
fn test_phase34_embed_different_dimensions() {
    for hidden_dim in [32, 64, 128, 256] {
        let config = GGUFConfig {
            architecture: "llama".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
            hidden_dim,
            intermediate_dim: hidden_dim * 2,
            num_layers: 1,
            num_heads: hidden_dim / 16,
            num_kv_heads: hidden_dim / 16,
            vocab_size: 100,
            context_length: 256,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
            explicit_head_dim: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        let model = create_test_model_with_config(&config);
        let embeddings = model.embed(&[0, 1]);

        assert_eq!(embeddings.len(), 2 * hidden_dim);
    }
}

#[test]
fn test_phase34_fused_matmul_all_qtypes_comprehensive() {
    // Comprehensive test covering all qtypes in a single test
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 256,
        intermediate_dim: 512,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    let model = create_test_model_with_config(&config);
    let in_dim = 256;
    let out_dim = 512;
    let input = vec![1.0f32; in_dim];

    // Test each qtype
    let qtypes = [
        ("Q4_0", create_q4_0_weight(in_dim, out_dim)),
        ("Q8_0", create_q8_0_weight(in_dim, out_dim)),
        ("Q4_1", create_q4_1_weight(in_dim, out_dim)),
        ("Q5_0", create_q5_0_weight(in_dim, out_dim)),
        ("Q4_K", create_q4_k_weight(in_dim, out_dim)),
        ("Q5_K", create_q5_k_weight(in_dim, out_dim)),
        ("Q6_K", create_q6_k_weight(in_dim, out_dim)),
    ];

    for (name, weight) in qtypes {
        let result = model.fused_matmul(&input, &weight);
        assert!(
            result.is_ok(),
            "{} fused_matmul failed: {:?}",
            name,
            result.err()
        );
        let output = result.expect("test value should be present");
        assert_eq!(output.len(), out_dim, "{} output size mismatch", name);
        assert!(
            output.iter().all(|x| x.is_finite()),
            "{} produced non-finite values",
            name
        );
    }
}
