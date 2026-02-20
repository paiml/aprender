
/// Build APR with Q6_K quantized tensors (GGUF-style names)
///
/// Q6_K format: 256-element super-blocks, 210 bytes each.
/// Layout: ql (128B) + qh (64B) + scales (16B) + d (f16, 2B) = 210 bytes.
/// Uses GGUF naming: `token_embd.weight`, `blk.0.attn_q.weight`, etc.
#[must_use]
pub fn build_pygmy_apr_q6k() -> Vec<u8> {
    build_kquant_gguf_apr("pygmy-q6k", 210, |w, name, shape, data| {
        w.add_q6k_raw_tensor(name, shape, data);
    })
}

// ============================================================================
// Encryption Pygmy Builders (feature-gated)
// ============================================================================

/// Build a minimal encrypted APR model in memory
///
/// Creates an APR file encrypted with password "test_password"
#[cfg(feature = "format-encryption")]
#[must_use]
pub fn build_pygmy_apr_encrypted(password: &str) -> Vec<u8> {
    use crate::format::{save_encrypted, ModelType, SaveOptions};
    use serde::{Deserialize, Serialize};
    use tempfile::NamedTempFile;

    #[derive(Debug, Serialize, Deserialize)]
    struct PygmyModel {
        weights: Vec<f32>,
        bias: f32,
    }

    let model = PygmyModel {
        weights: vec![0.1, 0.2, 0.3, 0.4],
        bias: 0.5,
    };

    let temp = NamedTempFile::with_suffix(".apr").expect("Create temp file");
    save_encrypted(
        &model,
        ModelType::Custom,
        temp.path(),
        SaveOptions::default(),
        password,
    )
    .expect("Save encrypted");

    std::fs::read(temp.path()).expect("Read encrypted file")
}

/// Build encrypted APR with default test password
#[cfg(feature = "format-encryption")]
#[must_use]
pub fn build_pygmy_apr_encrypted_default() -> Vec<u8> {
    build_pygmy_apr_encrypted("pygmy_test_password_123")
}

// ============================================================================
// Signing Pygmy Builders (feature-gated)
// ============================================================================

/// Build a minimal signed APR model in memory
///
/// Creates an APR file with Ed25519 signature
#[cfg(feature = "format-signing")]
#[must_use]
pub fn build_pygmy_apr_signed() -> (Vec<u8>, ed25519_dalek::VerifyingKey) {
    use crate::format::{save_signed, ModelType, SaveOptions};
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;
    use serde::{Deserialize, Serialize};
    use tempfile::NamedTempFile;

    #[derive(Debug, Serialize, Deserialize)]
    struct PygmyModel {
        weights: Vec<f32>,
        bias: f32,
    }

    let model = PygmyModel {
        weights: vec![0.1, 0.2, 0.3, 0.4],
        bias: 0.5,
    };

    // Generate signing key
    let signing_key = SigningKey::generate(&mut OsRng);
    let verifying_key = signing_key.verifying_key();

    let temp = NamedTempFile::with_suffix(".apr").expect("Create temp file");
    save_signed(
        &model,
        ModelType::Custom,
        temp.path(),
        SaveOptions::default(),
        &signing_key,
    )
    .expect("Save signed");

    let data = std::fs::read(temp.path()).expect("Read signed file");
    (data, verifying_key)
}

/// Generate a test signing key pair
#[cfg(feature = "format-signing")]
#[must_use]
pub fn generate_test_signing_key() -> ed25519_dalek::SigningKey {
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;
    SigningKey::generate(&mut OsRng)
}

// ============================================================================
// Quantization Pygmy Builders (feature-gated extras)
// ============================================================================

/// Build pygmy data for Q8_0 quantization testing
#[cfg(feature = "format-quantize")]
#[must_use]
pub fn build_pygmy_quantize_data() -> Vec<f32> {
    // Create data that exercises quantization edge cases
    let mut data = Vec::with_capacity(64);

    // Normal range values
    for i in 0..32 {
        data.push((i as f32 - 16.0) / 16.0);
    }

    // Edge case values
    data.push(0.0); // Zero
    data.push(1.0); // Max normal
    data.push(-1.0); // Min normal
    data.push(0.5); // Mid positive
    data.push(-0.5); // Mid negative
                     // Small values to fill remaining capacity
    data.extend(std::iter::repeat(0.001).take(27));

    data
}

/// Build pygmy Q8_0 quantized block
#[cfg(feature = "format-quantize")]
#[must_use]
pub fn build_pygmy_q8_block() -> crate::format::quantize::QuantizedBlock {
    use crate::format::quantize::{quantize as quantize_data, QuantType};

    let data = build_pygmy_quantize_data();
    quantize_data(&data, &[64], QuantType::Q8_0).expect("Quantize Q8_0")
}

/// Build pygmy Q4_0 quantized block
#[cfg(feature = "format-quantize")]
#[must_use]
pub fn build_pygmy_q4_block() -> crate::format::quantize::QuantizedBlock {
    use crate::format::quantize::{quantize as quantize_data, QuantType};

    let data = build_pygmy_quantize_data();
    quantize_data(&data, &[64], QuantType::Q4_0).expect("Quantize Q4_0")
}

// ============================================================================
// GH-194: GGUF-Style Naming Builders (Weight Tying Support)
// ============================================================================

/// Configuration for GGUF-style pygmy model (different tensor naming)
#[derive(Debug, Clone)]
pub struct GgufPygmyConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Use weight tying (token_embd.weight for both embedding and lm_head)
    pub weight_tying: bool,
}

impl Default for GgufPygmyConfig {
    fn default() -> Self {
        Self {
            vocab_size: 8,
            hidden_size: 4,
            num_layers: 1,
            weight_tying: true, // GGUF models commonly use weight tying
        }
    }
}

/// Build APR with GGUF-style tensor naming
///
/// GGUF uses different tensor names than HuggingFace:
/// - `token_embd.weight` instead of `model.embed_tokens.weight`
/// - `blk.N.attn_q.weight` instead of `model.layers.N.self_attn.q_proj.weight`
/// - `output.weight` for lm_head (or weight-tied to token_embd)
///
/// This is critical for GH-194: realizaer must find lm_head via `token_embd.weight`
/// when weight tying is used.
#[must_use]
pub fn build_pygmy_apr_gguf_names() -> Vec<u8> {
    build_pygmy_apr_gguf_names_with_config(GgufPygmyConfig::default())
}

/// Build APR with GGUF-style naming and custom config
#[must_use]
pub fn build_pygmy_apr_gguf_names_with_config(config: GgufPygmyConfig) -> Vec<u8> {
    let mut metadata = AprV2Metadata::new("pygmy-gguf");
    metadata.architecture = Some("qwen2".to_string());
    metadata.hidden_size = Some(config.hidden_size);
    metadata.vocab_size = Some(config.vocab_size);
    metadata.num_layers = Some(config.num_layers);
    metadata
        .custom
        .insert("naming".to_string(), serde_json::json!("gguf"));
    metadata.custom.insert(
        "weight_tying".to_string(),
        serde_json::json!(config.weight_tying),
    );

    let mut writer = AprV2Writer::new(metadata);
    let h = config.hidden_size;
    let v = config.vocab_size;

    // Token embedding: GGUF uses `token_embd.weight` (NOT model.embed_tokens.weight)
    writer.add_f32_tensor("token_embd.weight", vec![v, h], &gen_embed_data(v * h));

    // Layer tensors with GGUF naming
    for layer_idx in 0..config.num_layers {
        let norm_data: Vec<f32> = vec![1.0; h];
        writer.add_f32_tensor(
            format!("blk.{layer_idx}.attn_norm.weight"),
            vec![h],
            &norm_data,
        );
        writer.add_f32_tensor(
            format!("blk.{layer_idx}.ffn_norm.weight"),
            vec![h],
            &norm_data,
        );

        // Attention: blk.N.attn_{q,k,v,output}.weight
        let qkvo_data = gen_weight_data(h * h);
        for suffix in &["attn_q", "attn_k", "attn_v", "attn_output"] {
            writer.add_f32_tensor(
                format!("blk.{layer_idx}.{suffix}.weight"),
                vec![h, h],
                &qkvo_data,
            );
        }

        // MLP: blk.N.ffn_{gate,up,down}.weight
        let intermediate = h * 2;
        let gate_up_data = gen_weight_data(intermediate * h);
        let down_data = gen_weight_data(h * intermediate);

        writer.add_f32_tensor(
            format!("blk.{layer_idx}.ffn_gate.weight"),
            vec![intermediate, h],
            &gate_up_data,
        );
        writer.add_f32_tensor(
            format!("blk.{layer_idx}.ffn_up.weight"),
            vec![intermediate, h],
            &gate_up_data,
        );
        writer.add_f32_tensor(
            format!("blk.{layer_idx}.ffn_down.weight"),
            vec![h, intermediate],
            &down_data,
        );
    }

    // Output norm: output_norm.weight
    let norm_data: Vec<f32> = vec![1.0; h];
    writer.add_f32_tensor("output_norm.weight", vec![h], &norm_data);

    // LM head (output projection)
    if !config.weight_tying {
        writer.add_f32_tensor("output.weight", vec![v, h], &gen_embed_data(v * h));
    }
    // Weight tying: NO separate output.weight tensor
    // realizar must use token_embd.weight (transposed) for lm_head

    writer.write().unwrap_or_default()
}

/// Build APR with HuggingFace-style naming and weight tying
///
/// HuggingFace naming with weight tying uses `model.embed_tokens.weight`
/// for both embedding lookup and lm_head (tied weights).
#[must_use]
pub fn build_pygmy_apr_hf_names_tied() -> Vec<u8> {
    let config = PygmyConfig::default();
    let h = config.hidden_size;
    let v = config.vocab_size;

    let mut metadata = AprV2Metadata::new("pygmy-hf-tied");
    metadata.architecture = Some("qwen2".to_string());
    metadata.hidden_size = Some(h);
    metadata.vocab_size = Some(v);
    metadata.num_layers = Some(config.num_layers);
    metadata
        .custom
        .insert("naming".to_string(), serde_json::json!("huggingface"));
    metadata
        .custom
        .insert("weight_tying".to_string(), serde_json::json!(true));

    let mut writer = AprV2Writer::new(metadata);

    // Token embedding: model.embed_tokens.weight
    writer.add_f32_tensor(
        "model.embed_tokens.weight",
        vec![v, h],
        &gen_embed_data(v * h),
    );

    // Layer tensors with HuggingFace naming
    for layer_idx in 0..config.num_layers {
        let norm_data: Vec<f32> = vec![1.0; h];
        writer.add_f32_tensor(
            format!("model.layers.{layer_idx}.input_layernorm.weight"),
            vec![h],
            &norm_data,
        );
        writer.add_f32_tensor(
            format!("model.layers.{layer_idx}.post_attention_layernorm.weight"),
            vec![h],
            &norm_data,
        );

        // Attention
        let qkvo_data = gen_weight_data(h * h);
        for suffix in &["q_proj", "k_proj", "v_proj", "o_proj"] {
            writer.add_f32_tensor(
                format!("model.layers.{layer_idx}.self_attn.{suffix}.weight"),
                vec![h, h],
                &qkvo_data,
            );
        }

        // MLP
        let intermediate = h * 2;
        let gate_up_data = gen_weight_data(intermediate * h);
        let down_data = gen_weight_data(h * intermediate);

        writer.add_f32_tensor(
            format!("model.layers.{layer_idx}.mlp.gate_proj.weight"),
            vec![intermediate, h],
            &gate_up_data,
        );
        writer.add_f32_tensor(
            format!("model.layers.{layer_idx}.mlp.up_proj.weight"),
            vec![intermediate, h],
            &gate_up_data,
        );
        writer.add_f32_tensor(
            format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
            vec![h, intermediate],
            &down_data,
        );
    }

    // Final norm
    let norm_data: Vec<f32> = vec![1.0; h];
    writer.add_f32_tensor("model.norm.weight", vec![h], &norm_data);

    // NO lm_head.weight - weight tying uses model.embed_tokens.weight
    // realizaer must find lm_head via model.embed_tokens.weight or embed_tokens.weight

    writer.write().unwrap_or_default()
}
