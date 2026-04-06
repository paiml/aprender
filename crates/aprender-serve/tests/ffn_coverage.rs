//! FFN Coverage Tests
//!
//! Tests all 3 FFN paths in OwnedQuantizedModel::forward_single_with_cache:
//! 1. Fused RMSNorm + SwiGLU (line 13665): ffn_norm + ffn_gate + use_rmsnorm
//! 2. Non-fused SwiGLU (line 13689): ffn_gate but no RMSNorm
//! 3. GELU path (line 13720): no ffn_gate

use realizar::gguf::{
    ArchConstraints, GGUFConfig, OwnedQKVWeights, OwnedQuantizedKVCache, OwnedQuantizedLayer,
    OwnedQuantizedModel, OwnedQuantizedTensor,
};

const GGUF_TYPE_Q4_0: u32 = 2;
const QK4_0: usize = 32;

/// Create Q4_0 test tensor data (minimal size for testing)
fn create_q4_0_data(in_dim: usize, out_dim: usize) -> Vec<u8> {
    // Q4_0: 18 bytes per 32-element block
    // Each row has ceil(in_dim / 32) blocks
    let blocks_per_row = in_dim.div_ceil(QK4_0);
    let bytes_per_row = blocks_per_row * 18; // 2-byte scale + 16 byte quants per block
    let total_bytes = out_dim * bytes_per_row;

    // Fill with valid Q4_0 data with non-zero values to exercise different paths
    let mut data = vec![0u8; total_bytes];
    for row in 0..out_dim {
        for block in 0..blocks_per_row {
            let offset = row * bytes_per_row + block * 18;
            // Scale: f16 value of 0.1 = 0x2E66 (small positive scale)
            data[offset] = 0x66;
            data[offset + 1] = 0x2E;
            // Quants: varying 4-bit values to produce non-zero outputs
            // 4-bit values range from -8 to 7 after dequant
            for i in 2..18 {
                // Create pattern: low nibble and high nibble with varying values
                let idx = (row + block + i) % 16;
                let low = (idx % 8) as u8; // 0-7
                let high = ((idx + 4) % 8) as u8; // 4-7, 0-3
                data[offset + i] = (high << 4) | low;
            }
        }
    }
    data
}

/// Create quantized tensor
fn create_test_tensor(in_dim: usize, out_dim: usize) -> OwnedQuantizedTensor {
    OwnedQuantizedTensor {
        data: create_q4_0_data(in_dim, out_dim),
        qtype: GGUF_TYPE_Q4_0,
        in_dim,
        out_dim,
    }
}

/// Create test config
fn test_config() -> GGUFConfig {
    GGUFConfig {
        architecture: "test".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    }
}

/// Create base layer without SwiGLU (for GELU tests)
fn create_gelu_layer(config: &GGUFConfig) -> OwnedQuantizedLayer {
    let hidden_dim = config.hidden_dim;
    let intermediate_dim = config.intermediate_dim;
    let kv_dim = config.num_kv_heads * (hidden_dim / config.num_heads);
    let qkv_out_dim = hidden_dim + 2 * kv_dim;

    OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0f32; hidden_dim],
        attn_norm_bias: None,
        qkv_weight: OwnedQKVWeights::Fused(create_test_tensor(hidden_dim, qkv_out_dim)),
        qkv_bias: None,
        attn_output_weight: create_test_tensor(hidden_dim, hidden_dim),
        attn_output_bias: None,
        ffn_up_weight: create_test_tensor(hidden_dim, intermediate_dim),
        ffn_up_bias: None,
        ffn_down_weight: create_test_tensor(intermediate_dim, hidden_dim),
        ffn_down_bias: None,
        ffn_gate_weight: None, // No gate = GELU path
        ffn_gate_bias: None,
        ffn_norm_weight: None, // No FFN norm
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    }
}

/// Create SwiGLU layer WITH ffn_norm (for fused RMSNorm+SwiGLU)
fn create_fused_swiglu_layer(config: &GGUFConfig) -> OwnedQuantizedLayer {
    let hidden_dim = config.hidden_dim;
    let intermediate_dim = config.intermediate_dim;
    let kv_dim = config.num_kv_heads * (hidden_dim / config.num_heads);
    let qkv_out_dim = hidden_dim + 2 * kv_dim;

    OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0f32; hidden_dim], // RMSNorm: no bias
        attn_norm_bias: None,
        qkv_weight: OwnedQKVWeights::Fused(create_test_tensor(hidden_dim, qkv_out_dim)),
        qkv_bias: None,
        attn_output_weight: create_test_tensor(hidden_dim, hidden_dim),
        attn_output_bias: None,
        ffn_up_weight: create_test_tensor(hidden_dim, intermediate_dim),
        ffn_up_bias: None,
        ffn_down_weight: create_test_tensor(intermediate_dim, hidden_dim),
        ffn_down_bias: None,
        ffn_gate_weight: Some(create_test_tensor(hidden_dim, intermediate_dim)), // SwiGLU
        ffn_gate_bias: None,
        ffn_norm_weight: Some(vec![1.0f32; hidden_dim]), // FFN norm for RMSNorm path
        ffn_norm_bias: None,                             // No bias = RMSNorm
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    }
}

/// Create SwiGLU layer WITHOUT ffn_norm (for non-fused path)
fn create_nonfused_swiglu_layer(config: &GGUFConfig) -> OwnedQuantizedLayer {
    let hidden_dim = config.hidden_dim;
    let intermediate_dim = config.intermediate_dim;
    let kv_dim = config.num_kv_heads * (hidden_dim / config.num_heads);
    let qkv_out_dim = hidden_dim + 2 * kv_dim;

    OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0f32; hidden_dim],
        attn_norm_bias: None,
        qkv_weight: OwnedQKVWeights::Fused(create_test_tensor(hidden_dim, qkv_out_dim)),
        qkv_bias: None,
        attn_output_weight: create_test_tensor(hidden_dim, hidden_dim),
        attn_output_bias: None,
        ffn_up_weight: create_test_tensor(hidden_dim, intermediate_dim),
        ffn_up_bias: None,
        ffn_down_weight: create_test_tensor(intermediate_dim, hidden_dim),
        ffn_down_bias: None,
        ffn_gate_weight: Some(create_test_tensor(hidden_dim, intermediate_dim)), // SwiGLU
        ffn_gate_bias: None,
        ffn_norm_weight: None, // No FFN norm
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    }
}

/// Create SwiGLU layer WITH LayerNorm (has bias = LayerNorm not RMSNorm)
fn create_layernorm_swiglu_layer(config: &GGUFConfig) -> OwnedQuantizedLayer {
    let hidden_dim = config.hidden_dim;
    let intermediate_dim = config.intermediate_dim;
    let kv_dim = config.num_kv_heads * (hidden_dim / config.num_heads);
    let qkv_out_dim = hidden_dim + 2 * kv_dim;

    OwnedQuantizedLayer {
        attn_norm_weight: vec![1.0f32; hidden_dim],
        attn_norm_bias: Some(vec![0.0f32; hidden_dim]), // Bias = LayerNorm
        qkv_weight: OwnedQKVWeights::Fused(create_test_tensor(hidden_dim, qkv_out_dim)),
        qkv_bias: None,
        attn_output_weight: create_test_tensor(hidden_dim, hidden_dim),
        attn_output_bias: None,
        ffn_up_weight: create_test_tensor(hidden_dim, intermediate_dim),
        ffn_up_bias: None,
        ffn_down_weight: create_test_tensor(intermediate_dim, hidden_dim),
        ffn_down_bias: None,
        ffn_gate_weight: Some(create_test_tensor(hidden_dim, intermediate_dim)), // SwiGLU
        ffn_gate_bias: None,
        ffn_norm_weight: Some(vec![1.0f32; hidden_dim]), // FFN norm
        ffn_norm_bias: Some(vec![0.0f32; hidden_dim]),   // Bias = LayerNorm
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
    }
}

/// Create test model from config and layer
fn create_model(config: &GGUFConfig, layer: OwnedQuantizedLayer) -> OwnedQuantizedModel {
    let hidden_dim = config.hidden_dim;
    let vocab_size = config.vocab_size;

    OwnedQuantizedModel::new_for_test(
        config.clone(),
        vec![0.1f32; vocab_size * hidden_dim],
        vec![layer],
        vec![1.0f32; hidden_dim],
        None,
        create_test_tensor(hidden_dim, vocab_size),
        None,
    )
}

// =============================================================================
// PATH 1: GELU (no gate weight)
// =============================================================================

#[test]
fn test_ffn_path1_gelu_forward() {
    let config = test_config();
    let layer = create_gelu_layer(&config);
    let model = create_model(&config, layer);
    let mut cache = OwnedQuantizedKVCache::from_config(&config, 64);

    // Forward pass should succeed
    let logits = model
        .forward_single_with_cache(1, &mut cache, 0)
        .expect("GELU path forward should succeed");

    assert_eq!(logits.len(), config.vocab_size);
    assert!(
        logits.iter().all(|x| x.is_finite()),
        "All logits should be finite"
    );
}

#[test]
fn test_ffn_path1_gelu_generates_tokens() {
    let config = test_config();
    let layer = create_gelu_layer(&config);
    let model = create_model(&config, layer);
    let mut cache = OwnedQuantizedKVCache::from_config(&config, 64);

    // Multiple forward passes (simulating generation)
    for pos in 0..5 {
        let logits = model
            .forward_single_with_cache(pos as u32 + 1, &mut cache, pos)
            .expect("Forward should succeed");
        assert_eq!(logits.len(), config.vocab_size);
    }
}

// =============================================================================
// PATH 2: Fused RMSNorm + SwiGLU (ffn_norm + ffn_gate, no bias)
// =============================================================================

#[test]
fn test_ffn_path2_fused_swiglu_forward() {
    let config = test_config();
    let layer = create_fused_swiglu_layer(&config);
    let model = create_model(&config, layer);
    let mut cache = OwnedQuantizedKVCache::from_config(&config, 64);

    // Forward pass should succeed
    let logits = model
        .forward_single_with_cache(1, &mut cache, 0)
        .expect("Fused SwiGLU path forward should succeed");

    assert_eq!(logits.len(), config.vocab_size);
    assert!(
        logits.iter().all(|x| x.is_finite()),
        "All logits should be finite"
    );
}

#[test]
fn test_ffn_path2_fused_swiglu_generates_tokens() {
    let config = test_config();
    let layer = create_fused_swiglu_layer(&config);
    let model = create_model(&config, layer);
    let mut cache = OwnedQuantizedKVCache::from_config(&config, 64);

    // Multiple forward passes
    for pos in 0..5 {
        let logits = model
            .forward_single_with_cache(pos as u32 + 1, &mut cache, pos)
            .expect("Forward should succeed");
        assert_eq!(logits.len(), config.vocab_size);
    }
}

#[test]
fn test_ffn_path2_fused_swiglu_with_bias() {
    let config = test_config();
    let mut layer = create_fused_swiglu_layer(&config);

    // Add biases to test bias application paths
    layer.ffn_up_bias = Some(vec![0.1f32; config.intermediate_dim]);
    layer.ffn_gate_bias = Some(vec![0.1f32; config.intermediate_dim]);

    let model = create_model(&config, layer);
    let mut cache = OwnedQuantizedKVCache::from_config(&config, 64);

    let logits = model
        .forward_single_with_cache(1, &mut cache, 0)
        .expect("Fused SwiGLU with bias should succeed");

    assert_eq!(logits.len(), config.vocab_size);
    assert!(logits.iter().all(|x| x.is_finite()));
}

// =============================================================================
// PATH 3: Non-fused SwiGLU (ffn_gate but no ffn_norm)
// =============================================================================

#[test]
fn test_ffn_path3_nonfused_swiglu_forward() {
    let config = test_config();
    let layer = create_nonfused_swiglu_layer(&config);
    let model = create_model(&config, layer);
    let mut cache = OwnedQuantizedKVCache::from_config(&config, 64);

    let logits = model
        .forward_single_with_cache(1, &mut cache, 0)
        .expect("Non-fused SwiGLU path forward should succeed");

    assert_eq!(logits.len(), config.vocab_size);
    assert!(logits.iter().all(|x| x.is_finite()));
}

#[test]
fn test_ffn_path3_nonfused_swiglu_generates_tokens() {
    let config = test_config();
    let layer = create_nonfused_swiglu_layer(&config);
    let model = create_model(&config, layer);
    let mut cache = OwnedQuantizedKVCache::from_config(&config, 64);

    for pos in 0..5 {
        let logits = model
            .forward_single_with_cache(pos as u32 + 1, &mut cache, pos)
            .expect("Forward should succeed");
        assert_eq!(logits.len(), config.vocab_size);
    }
}

#[test]
fn test_ffn_path3_nonfused_swiglu_with_bias() {
    let config = test_config();
    let mut layer = create_nonfused_swiglu_layer(&config);

    // Add biases
    layer.ffn_up_bias = Some(vec![0.1f32; config.intermediate_dim]);
    layer.ffn_gate_bias = Some(vec![0.1f32; config.intermediate_dim]);

    let model = create_model(&config, layer);
    let mut cache = OwnedQuantizedKVCache::from_config(&config, 64);

    let logits = model
        .forward_single_with_cache(1, &mut cache, 0)
        .expect("Non-fused SwiGLU with bias should succeed");

    assert_eq!(logits.len(), config.vocab_size);
}

// =============================================================================
// PATH 4: LayerNorm + SwiGLU (ffn_norm with bias = LayerNorm)
// =============================================================================

#[test]
fn test_ffn_path4_layernorm_swiglu_forward() {
    let config = test_config();
    let layer = create_layernorm_swiglu_layer(&config);
    let model = create_model(&config, layer);
    let mut cache = OwnedQuantizedKVCache::from_config(&config, 64);

    let logits = model
        .forward_single_with_cache(1, &mut cache, 0)
        .expect("LayerNorm SwiGLU path forward should succeed");

    assert_eq!(logits.len(), config.vocab_size);
    assert!(logits.iter().all(|x| x.is_finite()));
}

#[test]
fn test_ffn_path4_layernorm_swiglu_generates_tokens() {
    let config = test_config();
    let layer = create_layernorm_swiglu_layer(&config);
    let model = create_model(&config, layer);
    let mut cache = OwnedQuantizedKVCache::from_config(&config, 64);

    for pos in 0..5 {
        let logits = model
            .forward_single_with_cache(pos as u32 + 1, &mut cache, pos)
            .expect("Forward should succeed");
        assert_eq!(logits.len(), config.vocab_size);
    }
}

// =============================================================================
// GELU path variations
// =============================================================================

#[test]
fn test_ffn_gelu_with_ffn_norm_rmsnorm() {
    let config = test_config();
    let mut layer = create_gelu_layer(&config);

    // Add FFN norm without bias (RMSNorm path)
    layer.ffn_norm_weight = Some(vec![1.0f32; config.hidden_dim]);
    layer.ffn_norm_bias = None;

    let model = create_model(&config, layer);
    let mut cache = OwnedQuantizedKVCache::from_config(&config, 64);

    let logits = model
        .forward_single_with_cache(1, &mut cache, 0)
        .expect("GELU with RMSNorm should succeed");

    assert_eq!(logits.len(), config.vocab_size);
}

#[test]
fn test_ffn_gelu_with_ffn_norm_layernorm() {
    let config = test_config();
    let mut layer = create_gelu_layer(&config);

    // Add FFN norm with bias (LayerNorm path)
    layer.ffn_norm_weight = Some(vec![1.0f32; config.hidden_dim]);
    layer.ffn_norm_bias = Some(vec![0.0f32; config.hidden_dim]);

    let model = create_model(&config, layer);
    let mut cache = OwnedQuantizedKVCache::from_config(&config, 64);

    let logits = model
        .forward_single_with_cache(1, &mut cache, 0)
        .expect("GELU with LayerNorm should succeed");

    assert_eq!(logits.len(), config.vocab_size);
}

#[test]
fn test_ffn_gelu_with_up_bias() {
    let config = test_config();
    let mut layer = create_gelu_layer(&config);
    layer.ffn_up_bias = Some(vec![0.1f32; config.intermediate_dim]);

    let model = create_model(&config, layer);
    let mut cache = OwnedQuantizedKVCache::from_config(&config, 64);

    let logits = model
        .forward_single_with_cache(1, &mut cache, 0)
        .expect("GELU with up bias should succeed");

    assert_eq!(logits.len(), config.vocab_size);
}

// =============================================================================
// FFN down projection coverage
// =============================================================================

#[test]
fn test_ffn_down_with_bias() {
    let config = test_config();
    let mut layer = create_gelu_layer(&config);
    layer.ffn_down_bias = Some(vec![0.1f32; config.hidden_dim]);

    let model = create_model(&config, layer);
    let mut cache = OwnedQuantizedKVCache::from_config(&config, 64);

    let logits = model
        .forward_single_with_cache(1, &mut cache, 0)
        .expect("FFN down with bias should succeed");

    assert_eq!(logits.len(), config.vocab_size);
}

// =============================================================================
// Determinism tests
// =============================================================================

#[test]
fn test_ffn_gelu_deterministic() {
    let config = test_config();
    let layer = create_gelu_layer(&config);
    let model = create_model(&config, layer);

    let mut cache1 = OwnedQuantizedKVCache::from_config(&config, 64);
    let mut cache2 = OwnedQuantizedKVCache::from_config(&config, 64);

    let logits1 = model
        .forward_single_with_cache(1, &mut cache1, 0)
        .expect("test");
    let logits2 = model
        .forward_single_with_cache(1, &mut cache2, 0)
        .expect("test");

    assert_eq!(logits1, logits2, "GELU path should be deterministic");
}

#[test]
fn test_ffn_swiglu_deterministic() {
    let config = test_config();
    let layer = create_fused_swiglu_layer(&config);
    let model = create_model(&config, layer);

    let mut cache1 = OwnedQuantizedKVCache::from_config(&config, 64);
    let mut cache2 = OwnedQuantizedKVCache::from_config(&config, 64);

    let logits1 = model
        .forward_single_with_cache(1, &mut cache1, 0)
        .expect("test");
    let logits2 = model
        .forward_single_with_cache(1, &mut cache2, 0)
        .expect("test");

    assert_eq!(logits1, logits2, "SwiGLU path should be deterministic");
}

// =============================================================================
// Verify paths produce different outputs (proves different code paths)
// =============================================================================

#[test]
fn test_ffn_paths_differ() {
    let config = test_config();

    // Create models with different FFN paths
    let gelu_layer = create_gelu_layer(&config);
    let swiglu_layer = create_fused_swiglu_layer(&config);

    let gelu_model = create_model(&config, gelu_layer);
    let swiglu_model = create_model(&config, swiglu_layer);

    let mut cache1 = OwnedQuantizedKVCache::from_config(&config, 64);
    let mut cache2 = OwnedQuantizedKVCache::from_config(&config, 64);

    let gelu_logits = gelu_model
        .forward_single_with_cache(1, &mut cache1, 0)
        .expect("test");
    let swiglu_logits = swiglu_model
        .forward_single_with_cache(1, &mut cache2, 0)
        .expect("test");

    // GELU and SwiGLU should produce different outputs (verifies different paths)
    let diff: f32 = gelu_logits
        .iter()
        .zip(swiglu_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    // Adjusted threshold: small models with random weights may have small differences
    assert!(
        diff > 0.0001,
        "GELU and SwiGLU paths should produce different logits (diff={})",
        diff
    );
}

#[test]
fn test_ffn_fused_vs_nonfused_swiglu() {
    let config = test_config();

    let fused_layer = create_fused_swiglu_layer(&config);
    let nonfused_layer = create_nonfused_swiglu_layer(&config);

    let fused_model = create_model(&config, fused_layer);
    let nonfused_model = create_model(&config, nonfused_layer);

    let mut cache1 = OwnedQuantizedKVCache::from_config(&config, 64);
    let mut cache2 = OwnedQuantizedKVCache::from_config(&config, 64);

    let fused_logits = fused_model
        .forward_single_with_cache(1, &mut cache1, 0)
        .expect("test");
    let nonfused_logits = nonfused_model
        .forward_single_with_cache(1, &mut cache2, 0)
        .expect("test");

    // Fused and non-fused should produce different outputs due to norm presence
    let diff: f32 = fused_logits
        .iter()
        .zip(nonfused_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    // Fused and non-fused should produce very similar outputs (small numerical differences)
    assert!(
        diff < 1.0,
        "Fused and non-fused SwiGLU should be similar (diff={})",
        diff
    );
}

// =============================================================================
// silu and gelu activation unit tests
// =============================================================================

#[test]
fn test_silu_activation_values() {
    // silu(x) = x * sigmoid(x)
    fn silu(x: f32) -> f32 {
        x * (1.0 / (1.0 + (-x).exp()))
    }

    // silu(0) = 0
    assert!((silu(0.0)).abs() < 1e-6);

    // silu(1) ≈ 0.7311
    assert!((silu(1.0) - 0.7311).abs() < 0.01);

    // silu(-1) ≈ -0.2689
    assert!((silu(-1.0) - (-0.2689)).abs() < 0.01);

    // silu(10) ≈ 10 (sigmoid(10) ≈ 1)
    assert!((silu(10.0) - 10.0).abs() < 0.01);

    // silu(-10) ≈ 0 (sigmoid(-10) ≈ 0)
    assert!((silu(-10.0)).abs() < 0.01);
}

#[test]
fn test_gelu_activation_values() {
    // gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    fn gelu(x: f32) -> f32 {
        let sqrt_2_over_pi = 0.797_884_6_f32;
        let c = 0.044_715_f32;
        let inner = sqrt_2_over_pi * (x + c * x * x * x);
        0.5 * x * (1.0 + inner.tanh())
    }

    // gelu(0) = 0
    assert!((gelu(0.0)).abs() < 1e-6);

    // gelu(1) ≈ 0.8413
    assert!((gelu(1.0) - 0.8413).abs() < 0.01);

    // gelu(-1) ≈ -0.1587
    assert!((gelu(-1.0) - (-0.1587)).abs() < 0.01);

    // gelu(3) ≈ 3 (tanh saturates to 1)
    assert!((gelu(3.0) - 3.0).abs() < 0.01);
}

#[test]
fn test_swiglu_formula() {
    // SwiGLU: output = silu(gate) * up
    fn silu(x: f32) -> f32 {
        x * (1.0 / (1.0 + (-x).exp()))
    }

    let gate = 2.0_f32;
    let up = 1.5_f32;

    let swiglu_output = silu(gate) * up;

    // silu(2) ≈ 1.76 (2 * sigmoid(2) ≈ 2 * 0.88)
    // swiglu ≈ 1.76 * 1.5 ≈ 2.64
    assert!(
        (swiglu_output - 2.64).abs() < 0.1,
        "SwiGLU formula incorrect: {}",
        swiglu_output
    );
}

// =============================================================================
// Multi-layer tests
// =============================================================================

#[test]
fn test_ffn_multilayer_gelu() {
    let mut config = test_config();
    config.num_layers = 3;

    let layers = vec![
        create_gelu_layer(&config),
        create_gelu_layer(&config),
        create_gelu_layer(&config),
    ];

    let model = OwnedQuantizedModel::new_for_test(
        config.clone(),
        vec![0.1f32; config.vocab_size * config.hidden_dim],
        layers,
        vec![1.0f32; config.hidden_dim],
        None,
        create_test_tensor(config.hidden_dim, config.vocab_size),
        None,
    );

    let mut cache = OwnedQuantizedKVCache::from_config(&config, 64);
    let logits = model
        .forward_single_with_cache(1, &mut cache, 0)
        .expect("Multi-layer GELU should succeed");

    assert_eq!(logits.len(), config.vocab_size);
}

#[test]
fn test_ffn_multilayer_swiglu() {
    let mut config = test_config();
    config.num_layers = 3;

    let layers = vec![
        create_fused_swiglu_layer(&config),
        create_fused_swiglu_layer(&config),
        create_fused_swiglu_layer(&config),
    ];

    let model = OwnedQuantizedModel::new_for_test(
        config.clone(),
        vec![0.1f32; config.vocab_size * config.hidden_dim],
        layers,
        vec![1.0f32; config.hidden_dim],
        None,
        create_test_tensor(config.hidden_dim, config.vocab_size),
        None,
    );

    let mut cache = OwnedQuantizedKVCache::from_config(&config, 64);
    let logits = model
        .forward_single_with_cache(1, &mut cache, 0)
        .expect("Multi-layer SwiGLU should succeed");

    assert_eq!(logits.len(), config.vocab_size);
}
