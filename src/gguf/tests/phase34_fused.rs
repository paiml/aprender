
#[test]
fn test_phase34_fused_matmul_q4_1() {
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
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

    let in_dim = 64;
    let out_dim = 128;
    let input = vec![1.0f32; in_dim];
    let weight = create_q4_1_weight(in_dim, out_dim);

    let result = model.fused_matmul(&input, &weight);
    assert!(
        result.is_ok(),
        "Q4_1 fused_matmul failed: {:?}",
        result.err()
    );
    assert_eq!(result.expect("test value should be present").len(), out_dim);
}

#[test]
fn test_phase34_fused_matmul_q4_1_multi_seq() {
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
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

    let in_dim = 64;
    let out_dim = 128;
    let seq_len = 2;
    let input = vec![1.0f32; in_dim * seq_len];
    let weight = create_q4_1_weight(in_dim, out_dim);

    let result = model.fused_matmul(&input, &weight);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), out_dim * seq_len);
}

// =============================================================================
// Fused Matmul Q5_0 Tests
// =============================================================================

fn create_q5_0_weight(in_dim: usize, out_dim: usize) -> OwnedQuantizedTensor {
    // Q5_0: 22 bytes per 32 elements (2 scale + 4 high bits + 16 quants)
    let num_elements = in_dim * out_dim;
    let num_blocks = num_elements.div_ceil(32);
    let byte_size = num_blocks * 22;

    let mut data = Vec::with_capacity(byte_size);
    for _ in 0..num_blocks {
        let scale = half::f16::from_f32(0.1);
        data.extend_from_slice(&scale.to_le_bytes());
        data.extend([0u8; 4]); // high bits
        data.extend([0x00u8; 16]); // quants
    }

    OwnedQuantizedTensor {
        data,
        in_dim,
        out_dim,
        qtype: GGUF_TYPE_Q5_0,
    }
}

#[test]
fn test_phase34_fused_matmul_q5_0() {
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
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

    let in_dim = 64;
    let out_dim = 128;
    let input = vec![1.0f32; in_dim];
    let weight = create_q5_0_weight(in_dim, out_dim);

    let result = model.fused_matmul(&input, &weight);
    assert!(
        result.is_ok(),
        "Q5_0 fused_matmul failed: {:?}",
        result.err()
    );
    assert_eq!(result.expect("test value should be present").len(), out_dim);
}

#[test]
fn test_phase34_fused_matmul_q5_0_multi_seq() {
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
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

    let in_dim = 64;
    let out_dim = 128;
    let seq_len = 2;
    let input = vec![1.0f32; in_dim * seq_len];
    let weight = create_q5_0_weight(in_dim, out_dim);

    let result = model.fused_matmul(&input, &weight);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), out_dim * seq_len);
}

// =============================================================================
// Fused Matmul K-Quant Tests (Q4_K, Q5_K, Q6_K)
// =============================================================================

fn create_q4_k_weight(in_dim: usize, out_dim: usize) -> OwnedQuantizedTensor {
    // Q4_K: 144 bytes per 256 elements
    let num_elements = in_dim * out_dim;
    let num_super_blocks = num_elements.div_ceil(256);
    let byte_size = num_super_blocks * 144;

    OwnedQuantizedTensor {
        data: vec![0u8; byte_size],
        in_dim,
        out_dim,
        qtype: GGUF_TYPE_Q4_K,
    }
}

fn create_q5_k_weight(in_dim: usize, out_dim: usize) -> OwnedQuantizedTensor {
    // Q5_K: 176 bytes per 256 elements
    let num_elements = in_dim * out_dim;
    let num_super_blocks = num_elements.div_ceil(256);
    let byte_size = num_super_blocks * 176;

    OwnedQuantizedTensor {
        data: vec![0u8; byte_size],
        in_dim,
        out_dim,
        qtype: GGUF_TYPE_Q5_K,
    }
}

fn create_q6_k_weight(in_dim: usize, out_dim: usize) -> OwnedQuantizedTensor {
    // Q6_K: 210 bytes per 256 elements
    let num_elements = in_dim * out_dim;
    let num_super_blocks = num_elements.div_ceil(256);
    let byte_size = num_super_blocks * 210;

    OwnedQuantizedTensor {
        data: vec![0u8; byte_size],
        in_dim,
        out_dim,
        qtype: GGUF_TYPE_Q6_K,
    }
}

#[test]
fn test_phase34_fused_matmul_q4_k_single() {
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
    let weight = create_q4_k_weight(in_dim, out_dim);

    let result = model.fused_matmul(&input, &weight);
    assert!(
        result.is_ok(),
        "Q4_K fused_matmul failed: {:?}",
        result.err()
    );
    assert_eq!(result.expect("test value should be present").len(), out_dim);
}

#[test]
fn test_phase34_fused_matmul_q4_k_multi() {
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
    let seq_len = 2;
    let input = vec![1.0f32; in_dim * seq_len];
    let weight = create_q4_k_weight(in_dim, out_dim);

    let result = model.fused_matmul(&input, &weight);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), out_dim * seq_len);
}

#[test]
fn test_phase34_fused_matmul_q5_k() {
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
    let weight = create_q5_k_weight(in_dim, out_dim);

    let result = model.fused_matmul(&input, &weight);
    assert!(
        result.is_ok(),
        "Q5_K fused_matmul failed: {:?}",
        result.err()
    );
}

#[test]
fn test_phase34_fused_matmul_q5_k_multi() {
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
    let seq_len = 2;
    let input = vec![1.0f32; in_dim * seq_len];
    let weight = create_q5_k_weight(in_dim, out_dim);

    let result = model.fused_matmul(&input, &weight);
    assert!(result.is_ok());
    assert_eq!(result.expect("test value should be present").len(), out_dim * seq_len);
}

#[test]
fn test_phase34_fused_matmul_q6_k() {
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
    let weight = create_q6_k_weight(in_dim, out_dim);

    let result = model.fused_matmul(&input, &weight);
    assert!(
        result.is_ok(),
        "Q6_K fused_matmul failed: {:?}",
        result.err()
    );
}

#[test]
fn test_phase34_fused_matmul_q6_k_multi() {
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
    let seq_len = 2;
    let input = vec![1.0f32; in_dim * seq_len];
    let weight = create_q6_k_weight(in_dim, out_dim);

    let result = model.fused_matmul(&input, &weight);
    assert!(result.is_ok());
}

// =============================================================================
// Unsupported QType Test
// =============================================================================

#[test]
fn test_phase34_fused_matmul_unsupported_qtype() {
    let config = GGUFConfig {
        architecture: "llama".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
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

    let in_dim = 64;
    let out_dim = 128;
    let input = vec![1.0f32; in_dim];

    // Use a fake/unsupported qtype
    let weight = OwnedQuantizedTensor {
        data: vec![0u8; 1024],
        in_dim,
        out_dim,
        qtype: 255, // Invalid
    };

    let result = model.fused_matmul(&input, &weight);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("255") || err.contains("Unsupported") || err.contains("supports"),
        "Error: {}",
        err
    );
}
