
// ============================================================================
// qkv_matmul_into() Tests
// ============================================================================

#[test]
fn test_qkv_matmul_into_fused() {
    let model = create_test_model(256, 100);
    let qkv_weight = create_q4k_test_data(256, 3 * 256);
    let qkv = OwnedQKVWeights::Fused(qkv_weight);
    let input = vec![1.0f32; 256];
    let mut output = vec![0.0f32; 3 * 256];

    let result = model.qkv_matmul_into(&input, &qkv, &mut output);
    assert!(result.is_ok());
}

#[test]
fn test_qkv_matmul_into_separate() {
    let model = create_test_model(256, 100);
    let q = create_q4k_test_data(256, 256);
    let k = create_q4k_test_data(256, 64);
    let v = create_q4k_test_data(256, 64);
    let qkv = OwnedQKVWeights::Separate { q, k, v };
    let input = vec![1.0f32; 256];
    let mut output = vec![0.0f32; 384];

    let result = model.qkv_matmul_into(&input, &qkv, &mut output);
    assert!(result.is_ok());
}

// ============================================================================
// layer_norm() Tests
// ============================================================================

#[test]
fn test_layer_norm_basic() {
    let model = create_test_model(64, 100);
    let input = vec![1.0f32; 64];
    let weight = vec![1.0f32; 64];

    let output = model.layer_norm(&input, &weight, None, 1e-5);
    assert_eq!(output.len(), 64);
}

#[test]
fn test_layer_norm_with_bias() {
    let model = create_test_model(64, 100);
    let input = vec![1.0f32; 64];
    let weight = vec![1.0f32; 64];
    let bias = vec![0.5f32; 64];

    let output = model.layer_norm(&input, &weight, Some(&bias), 1e-5);
    assert_eq!(output.len(), 64);
}

// ============================================================================
// add_bias() Tests
// ============================================================================

#[test]
fn test_add_bias() {
    let model = create_test_model(64, 100);
    let mut input = vec![1.0f32; 4];
    let bias = vec![0.5f32; 4];

    model.add_bias(&mut input, &bias);
    assert!(input.iter().all(|&x| (x - 1.5).abs() < f32::EPSILON));
}

#[test]
fn test_add_bias_zeros() {
    let model = create_test_model(64, 100);
    let mut input = vec![2.0f32; 4];
    let bias = vec![0.0f32; 4];

    model.add_bias(&mut input, &bias);
    assert!(input.iter().all(|&x| (x - 2.0).abs() < f32::EPSILON));
}

// ============================================================================
// gelu() Tests
// ============================================================================

#[test]
fn test_gelu_zeros() {
    let model = create_test_model(64, 100);
    let mut input = vec![0.0f32; 4];
    model.gelu(&mut input);
    // GELU(0) = 0
    assert!(input.iter().all(|&x| x.abs() < 1e-6));
}

#[test]
fn test_gelu_positive() {
    let model = create_test_model(64, 100);
    let mut input = vec![1.0f32; 4];
    model.gelu(&mut input);
    // GELU(1) ≈ 0.841
    assert!(input.iter().all(|&x| (x - 0.841).abs() < 0.01));
}

#[test]
fn test_gelu_negative() {
    let model = create_test_model(64, 100);
    let mut input = vec![-1.0f32; 4];
    model.gelu(&mut input);
    // GELU(-1) ≈ -0.159
    assert!(input.iter().all(|&x| (x - (-0.159)).abs() < 0.01));
}

// ============================================================================
// fused_rmsnorm_qkv_matmul() Tests
// ============================================================================

#[test]
fn test_fused_rmsnorm_qkv_matmul_fused() {
    let model = create_test_model(256, 100);
    let qkv_weight = create_q4k_test_data(256, 3 * 256);
    let qkv = OwnedQKVWeights::Fused(qkv_weight);
    let input = vec![1.0f32; 256];
    let norm_weight = vec![1.0f32; 256];

    let result = model.fused_rmsnorm_qkv_matmul(&input, &norm_weight, 1e-5, &qkv);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 3 * 256);
}

#[test]
fn test_fused_rmsnorm_qkv_matmul_separate() {
    let model = create_test_model(256, 100);
    let q = create_q4k_test_data(256, 256);
    let k = create_q4k_test_data(256, 64);
    let v = create_q4k_test_data(256, 64);
    let qkv = OwnedQKVWeights::Separate { q, k, v };
    let input = vec![1.0f32; 256];
    let norm_weight = vec![1.0f32; 256];

    let result = model.fused_rmsnorm_qkv_matmul(&input, &norm_weight, 1e-5, &qkv);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 384);
}

// ============================================================================
// fused_rmsnorm_lm_head() Tests
// ============================================================================

#[test]
fn test_fused_rmsnorm_lm_head_q4k() {
    let model = create_test_model(256, 100);
    let input = vec![1.0f32; 256];

    let result = model.fused_rmsnorm_lm_head(&input);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.len(), 100); // vocab_size
}

// ============================================================================
// fused_rmsnorm_ffn_up_gate() Tests
// ============================================================================

#[test]
fn test_fused_rmsnorm_ffn_up_gate_q4_0() {
    let model = create_test_model(64, 100);
    let up_weight = create_q4_0_test_data(64, 256);
    let gate_weight = create_q4_0_test_data(64, 256);
    let input = vec![1.0f32; 64];
    let norm_weight = vec![1.0f32; 64];

    let result =
        model.fused_rmsnorm_ffn_up_gate(&input, &norm_weight, 1e-5, &up_weight, &gate_weight);
    assert!(result.is_ok());
    let (up_out, gate_out) = result.unwrap();
    assert_eq!(up_out.len(), 256);
    assert_eq!(gate_out.len(), 256);
}

#[test]
fn test_fused_rmsnorm_ffn_up_gate_q4k_fallback() {
    let model = create_test_model(256, 100);
    let up_weight = create_q4k_test_data(256, 1024);
    let gate_weight = create_q4k_test_data(256, 1024);
    let input = vec![1.0f32; 256];
    let norm_weight = vec![1.0f32; 256];

    let result =
        model.fused_rmsnorm_ffn_up_gate(&input, &norm_weight, 1e-5, &up_weight, &gate_weight);
    assert!(result.is_ok());
    let (up_out, gate_out) = result.unwrap();
    assert_eq!(up_out.len(), 1024);
    assert_eq!(gate_out.len(), 1024);
}

// ============================================================================
// qkv_matmul_q8k_into() Tests
// ============================================================================

#[test]
fn test_qkv_matmul_q8k_into_fused() {
    let model = create_test_model(256, 100);
    let qkv_weight = create_q4k_test_data(256, 3 * 256);
    let qkv = OwnedQKVWeights::Fused(qkv_weight);
    let input = vec![1.0f32; 256];
    let mut output = vec![0.0f32; 3 * 256];
    let scales = vec![1.0f32; 8];
    let quants = vec![0i8; 256];

    // Currently falls back to regular qkv_matmul_into
    let result = model.qkv_matmul_q8k_into(&input, &qkv, &mut output, &scales, &quants);
    assert!(result.is_ok());
}

// ============================================================================
// Edge Case Tests
// ============================================================================

// ============================================================================
// GH-242: F32 / F16 fused_matmul tests (type 0 / type 1)
// ============================================================================

/// Create F32 weight tensor — raw little-endian f32 bytes
fn create_f32_test_data(in_dim: usize, out_dim: usize) -> OwnedQuantizedTensor {
    let mut data = Vec::with_capacity(out_dim * in_dim * 4);
    for row in 0..out_dim {
        for col in 0..in_dim {
            // Identity-like: 0.1 on diagonal, 0.0 elsewhere
            let val = if row == col { 0.1f32 } else { 0.0f32 };
            data.extend_from_slice(&val.to_le_bytes());
        }
    }
    OwnedQuantizedTensor {
        data,
        in_dim,
        out_dim,
        qtype: GGUF_TYPE_F32,
    }
}

/// Create F16 weight tensor — raw little-endian f16 bytes
fn create_f16_test_data(in_dim: usize, out_dim: usize) -> OwnedQuantizedTensor {
    let mut data = Vec::with_capacity(out_dim * in_dim * 2);
    for row in 0..out_dim {
        for col in 0..in_dim {
            let val = if row == col { 0.1f32 } else { 0.0f32 };
            let bits = half::f16::from_f32(val).to_bits();
            data.extend_from_slice(&bits.to_le_bytes());
        }
    }
    OwnedQuantizedTensor {
        data,
        in_dim,
        out_dim,
        qtype: GGUF_TYPE_F16,
    }
}

/// Create a model with F32 weights (GH-242 regression test)
fn create_f32_test_model(hidden_dim: usize, vocab_size: usize) -> OwnedQuantizedModel {
    let config = test_config(hidden_dim, vocab_size);
    let intermediate_dim = config.intermediate_dim;

    let qkv_weight = create_f32_test_data(hidden_dim, 3 * hidden_dim);
    let attn_output_weight = create_f32_test_data(hidden_dim, hidden_dim);
    let ffn_up_weight = create_f32_test_data(hidden_dim, intermediate_dim);
    let ffn_down_weight = create_f32_test_data(intermediate_dim, hidden_dim);

    let layer = crate::gguf::OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0f32; hidden_dim],
        attn_norm_bias: None,
        qkv_weight: OwnedQKVWeights::Fused(qkv_weight),
        qkv_bias: None,
        attn_output_weight,
        attn_output_bias: None,
        ffn_up_weight,
        ffn_up_bias: None,
        ffn_down_weight,
        ffn_down_bias: None,
        ffn_gate_weight: None,
        ffn_gate_bias: None,
        ffn_norm_weight: Some(vec![1.0f32; hidden_dim]),
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    };

    let lm_head_weight = create_f32_test_data(hidden_dim, vocab_size);

    OwnedQuantizedModel {
        config,
        token_embedding: vec![0.1f32; vocab_size * hidden_dim],
        position_embedding: None,
        layers: vec![layer],
        encoder_layers: vec![],
        encoder_output_norm_weight: None,
        encoder_output_norm_bias: None,
        output_norm_weight: vec![1.0f32; hidden_dim],
        output_norm_bias: None,
        lm_head_weight,
        lm_head_bias: None,
        #[cfg(feature = "cuda")]
        cuda_executor: None,
        #[cfg(feature = "cuda")]
        cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "cuda")]
        cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
    }
}

#[test]
fn test_fused_matmul_f32_single_token() {
    // GH-242: F32 weights (type 0) must not error in fused_matmul
    let model = create_f32_test_model(64, 100);
    let input = vec![1.0f32; 64];
    let weight = create_f32_test_data(64, 32);
    let result = model.fused_matmul(&input, &weight);
    assert!(result.is_ok(), "F32 fused_matmul failed: {:?}", result.err());
    assert_eq!(result.unwrap().len(), 32);
}

#[test]
fn test_fused_matmul_f32_multi_token() {
    // GH-242: F32 with seq_len > 1
    let model = create_f32_test_model(64, 100);
    let input = vec![1.0f32; 64 * 3]; // 3 tokens
    let weight = create_f32_test_data(64, 32);
    let result = model.fused_matmul(&input, &weight);
    assert!(result.is_ok(), "F32 multi-token fused_matmul failed: {:?}", result.err());
    assert_eq!(result.unwrap().len(), 32 * 3);
}

#[test]
fn test_fused_matmul_f16_single_token() {
    // GH-242: F16 weights (type 1) must not error in fused_matmul
    let model = create_f32_test_model(64, 100);
    let input = vec![1.0f32; 64];
    let weight = create_f16_test_data(64, 32);
    let result = model.fused_matmul(&input, &weight);
    assert!(result.is_ok(), "F16 fused_matmul failed: {:?}", result.err());
    assert_eq!(result.unwrap().len(), 32);
}

#[test]
fn test_fused_matmul_f32_correctness() {
    // GH-242: Verify F32 produces correct output (identity-like weight)
    let model = create_f32_test_model(8, 16);
    let mut input = vec![0.0f32; 8];
    input[0] = 5.0;
    input[1] = 3.0;
    let weight = create_f32_test_data(8, 8); // 0.1 on diagonal
    let result = model.fused_matmul(&input, &weight).unwrap();
    // output[0] = input[0] * 0.1 = 0.5, output[1] = input[1] * 0.1 = 0.3
    assert!((result[0] - 0.5).abs() < 1e-5, "Expected 0.5, got {}", result[0]);
    assert!((result[1] - 0.3).abs() < 1e-5, "Expected 0.3, got {}", result[1]);
}

#[test]
fn test_fused_matmul_into_f32() {
    // GH-242: fused_matmul_into delegates F32 to fused_matmul
    let model = create_f32_test_model(64, 100);
    let input = vec![1.0f32; 64];
    let weight = create_f32_test_data(64, 32);
    let mut output = vec![0.0f32; 32];
    let result = model.fused_matmul_into(&input, &weight, &mut output);
    assert!(result.is_ok(), "F32 fused_matmul_into failed: {:?}", result.err());
}

#[test]
fn test_embed_max_token() {
    let model = create_test_model(64, 1000);
    let embeddings = model.embed(&[999]);
    assert_eq!(embeddings.len(), 64);
}

#[test]
fn test_embed_into_larger_buffer() {
    let model = create_test_model(64, 100);
    let mut output = vec![9.9f32; 128]; // Buffer larger than needed
    model.embed_into(0, &mut output);
    // First hidden_dim elements should be filled
    assert!(output[..64].iter().all(|&x| (x - 0.1).abs() < f32::EPSILON));
}
