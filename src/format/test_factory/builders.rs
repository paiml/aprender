//! Pygmy model builder functions and configuration types.
//!
//! Contains `PygmyConfig`, `GgufPygmyConfig`, and all `build_pygmy_*` functions
//! for creating minimal valid model files in memory.

use crate::format::v2::{AprV2Metadata, AprV2Writer};

// ============================================================================
// SafeTensors Pygmy Builder
// ============================================================================

/// Build a minimal valid SafeTensors file in memory
///
/// Creates a SafeTensors file with:
/// - 2 F32 tensors (embedding + weight)
/// - Small dimensions (vocab=8, hidden=4)
/// - Valid JSON header format
#[must_use]
pub fn build_pygmy_safetensors() -> Vec<u8> {
    build_pygmy_safetensors_with_config(PygmyConfig::default())
}

/// Configuration for pygmy model generation
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)] // Config struct legitimately needs multiple flags
pub struct PygmyConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Number of attention heads (None = derive from hidden_size)
    pub num_heads: Option<usize>,
    /// Number of key/value heads for GQA (None = same as num_heads, i.e. MHA)
    pub num_kv_heads: Option<usize>,
    /// Include embedding tensor
    pub include_embedding: bool,
    /// Include norm tensors
    pub include_norms: bool,
    /// Include attention tensors
    pub include_attention: bool,
    /// Include MLP tensors
    pub include_mlp: bool,
    /// Include attention biases (Qwen2-style)
    pub include_bias: bool,
    /// Simulate tied embeddings (omit lm_head, let import synthesize it)
    pub tied_embeddings: bool,
}

impl Default for PygmyConfig {
    fn default() -> Self {
        Self {
            vocab_size: 8,
            hidden_size: 4,
            num_layers: 1,
            num_heads: None,
            num_kv_heads: None,
            include_embedding: true,
            include_norms: true,
            include_attention: true,
            include_mlp: true,
            include_bias: false,
            tied_embeddings: false,
        }
    }
}

impl PygmyConfig {
    /// Create minimal config (smallest valid model)
    #[must_use]
    pub fn minimal() -> Self {
        Self {
            vocab_size: 4,
            hidden_size: 2,
            num_layers: 1,
            include_embedding: true,
            include_norms: false,
            include_attention: false,
            include_mlp: false,
            ..Default::default()
        }
    }

    /// Create config with only embedding and LM head
    #[must_use]
    pub fn embedding_only() -> Self {
        Self {
            vocab_size: 8,
            hidden_size: 4,
            num_layers: 0,
            include_embedding: true,
            include_norms: false,
            include_attention: false,
            include_mlp: false,
            ..Default::default()
        }
    }

    /// Create LLaMA-style config
    #[must_use]
    pub fn llama_style() -> Self {
        Self {
            vocab_size: 16,
            hidden_size: 8,
            num_layers: 1,
            include_embedding: true,
            include_norms: true,
            include_attention: true,
            include_mlp: true,
            ..Default::default()
        }
    }

    /// ROSETTA-003: Create Qwen2-style GQA config with attention biases.
    ///
    /// Tests the GQA path where K/V have fewer heads than Q:
    /// - num_heads=4, num_kv_heads=2 -> head_dim=2, kv_dim=4
    /// - Q shape: [8, 8], K/V shape: [4, 8]
    /// - Includes Q/K/V biases (Qwen2-style)
    /// - 2 layers for multi-layer coverage
    #[must_use]
    pub fn qwen2_gqa() -> Self {
        Self {
            vocab_size: 16,
            hidden_size: 8,
            num_layers: 2,
            num_heads: Some(4),
            num_kv_heads: Some(2),
            include_embedding: true,
            include_norms: true,
            include_attention: true,
            include_mlp: true,
            include_bias: true,
            tied_embeddings: false,
        }
    }

    /// ROSETTA-003: Qwen2 GQA with tied embeddings (no explicit lm_head).
    ///
    /// Simulates HuggingFace models where `lm_head.weight` is omitted because
    /// it's tied to `embed_tokens.weight`. The import pipeline should synthesize it.
    #[must_use]
    pub fn qwen2_gqa_tied() -> Self {
        Self {
            tied_embeddings: true,
            ..Self::qwen2_gqa()
        }
    }

    /// Realistic-dimension config for testing `infer_model_config_from_tensors()`.
    ///
    /// All other PygmyConfig constructors use tiny hidden sizes (2-8) that never
    /// trigger the head_dim candidate path in config inference (requires
    /// `hidden_size % head_dim == 0` where head_dim in {64, 128, 96, 80}).
    ///
    /// This config uses hidden_size=128 with 2 attention heads (head_dim=64)
    /// and 1 KV head (GQA 2:1), exercising the full inference pipeline
    /// through the harness round-trip path.
    #[must_use]
    pub fn realistic() -> Self {
        Self {
            vocab_size: 256,
            hidden_size: 128,
            num_layers: 2,
            num_heads: Some(2),
            num_kv_heads: Some(1),
            include_embedding: true,
            include_norms: true,
            include_attention: true,
            include_mlp: true,
            include_bias: false,
            tied_embeddings: false,
        }
    }

    /// Effective number of attention heads
    #[must_use]
    pub fn effective_num_heads(&self) -> usize {
        self.num_heads
            .unwrap_or_else(|| (self.hidden_size / 64).max(1))
    }

    /// Effective number of key/value heads (GQA)
    #[must_use]
    pub fn effective_num_kv_heads(&self) -> usize {
        self.num_kv_heads
            .unwrap_or_else(|| self.effective_num_heads())
    }

    /// Head dimension
    #[must_use]
    pub fn head_dim(&self) -> usize {
        let nh = self.effective_num_heads();
        if nh > 0 {
            self.hidden_size / nh
        } else {
            self.hidden_size
        }
    }

    /// Key/Value dimension (for GQA, smaller than hidden_size)
    #[must_use]
    pub fn kv_dim(&self) -> usize {
        self.effective_num_kv_heads() * self.head_dim()
    }

    /// GH-197 FIX: Generate config.json that matches the tensors this config creates.
    ///
    /// This ensures round-trip testing uses consistent config values instead of
    /// relying on inference which can fail with incorrect defaults.
    /// ROSETTA-003: Uses explicit num_heads/num_kv_heads for GQA support.
    #[must_use]
    pub fn to_config_json(&self) -> String {
        let num_attention_heads = self.effective_num_heads();
        let num_key_value_heads = self.effective_num_kv_heads();
        let intermediate_size = self.hidden_size * 4;

        format!(
            r#"{{
  "architectures": ["Qwen2ForCausalLM"],
  "hidden_size": {},
  "num_hidden_layers": {},
  "num_attention_heads": {},
  "num_key_value_heads": {},
  "vocab_size": {},
  "intermediate_size": {},
  "max_position_embeddings": 2048,
  "rms_norm_eps": 1e-06,
  "rope_theta": 10000.0,
  "model_type": "qwen2"
}}"#,
            self.hidden_size,
            self.num_layers,
            num_attention_heads,
            num_key_value_heads,
            self.vocab_size,
            intermediate_size
        )
    }
}

/// Build SafeTensors with custom config
#[must_use]
pub fn build_pygmy_safetensors_with_config(config: PygmyConfig) -> Vec<u8> {
    // Build tensor metadata and data
    let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();

    // Token embedding: [vocab_size, hidden_size]
    if config.include_embedding {
        let embed_data: Vec<f32> = (0..config.vocab_size * config.hidden_size)
            .map(|i| ((i % 100) as f32 - 50.0) / 1000.0)
            .collect();
        tensors.push((
            "model.embed_tokens.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
            embed_data,
        ));
    }

    // Layer tensors
    for layer_idx in 0..config.num_layers {
        // Input layernorm
        if config.include_norms {
            let norm_data: Vec<f32> = vec![1.0; config.hidden_size];
            tensors.push((
                format!("model.layers.{layer_idx}.input_layernorm.weight"),
                vec![config.hidden_size],
                norm_data.clone(),
            ));
            tensors.push((
                format!("model.layers.{layer_idx}.post_attention_layernorm.weight"),
                vec![config.hidden_size],
                norm_data,
            ));
        }

        // Attention: Q, K, V, O projections
        // ROSETTA-003: GQA support - K/V may have fewer heads than Q
        if config.include_attention {
            let kv_dim = config.kv_dim();

            // Q and O: [hidden_size, hidden_size]
            let q_data: Vec<f32> = (0..config.hidden_size * config.hidden_size)
                .map(|i| ((i % 200) as f32 - 100.0) / 2000.0)
                .collect();
            tensors.push((
                format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
                vec![config.hidden_size, config.hidden_size],
                q_data.clone(),
            ));
            tensors.push((
                format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
                vec![config.hidden_size, config.hidden_size],
                q_data,
            ));

            // K and V: [kv_dim, hidden_size] (may differ from Q for GQA)
            let kv_data: Vec<f32> = (0..kv_dim * config.hidden_size)
                .map(|i| ((i % 200) as f32 - 100.0) / 2000.0)
                .collect();
            tensors.push((
                format!("model.layers.{layer_idx}.self_attn.k_proj.weight"),
                vec![kv_dim, config.hidden_size],
                kv_data.clone(),
            ));
            tensors.push((
                format!("model.layers.{layer_idx}.self_attn.v_proj.weight"),
                vec![kv_dim, config.hidden_size],
                kv_data,
            ));

            // Biases (Qwen2-style)
            if config.include_bias {
                let q_bias: Vec<f32> = (0..config.hidden_size)
                    .map(|i| (i as f32) / 1000.0)
                    .collect();
                tensors.push((
                    format!("model.layers.{layer_idx}.self_attn.q_proj.bias"),
                    vec![config.hidden_size],
                    q_bias,
                ));
                let kv_bias: Vec<f32> = (0..kv_dim).map(|i| (i as f32) / 1000.0).collect();
                tensors.push((
                    format!("model.layers.{layer_idx}.self_attn.k_proj.bias"),
                    vec![kv_dim],
                    kv_bias.clone(),
                ));
                tensors.push((
                    format!("model.layers.{layer_idx}.self_attn.v_proj.bias"),
                    vec![kv_dim],
                    kv_bias,
                ));
            }
        }

        // MLP: gate, up, down projections
        if config.include_mlp {
            let intermediate = config.hidden_size * 2;
            let gate_up_data: Vec<f32> = (0..intermediate * config.hidden_size)
                .map(|i| ((i % 200) as f32 - 100.0) / 2000.0)
                .collect();
            let down_data: Vec<f32> = (0..config.hidden_size * intermediate)
                .map(|i| ((i % 200) as f32 - 100.0) / 2000.0)
                .collect();

            tensors.push((
                format!("model.layers.{layer_idx}.mlp.gate_proj.weight"),
                vec![intermediate, config.hidden_size],
                gate_up_data.clone(),
            ));
            tensors.push((
                format!("model.layers.{layer_idx}.mlp.up_proj.weight"),
                vec![intermediate, config.hidden_size],
                gate_up_data,
            ));
            tensors.push((
                format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
                vec![config.hidden_size, intermediate],
                down_data,
            ));
        }
    }

    // Final norm
    if config.include_norms && config.num_layers > 0 {
        let norm_data: Vec<f32> = vec![1.0; config.hidden_size];
        tensors.push((
            "model.norm.weight".to_string(),
            vec![config.hidden_size],
            norm_data,
        ));
    }

    // LM head: [vocab_size, hidden_size]
    // ROSETTA-003: Omit lm_head when tied_embeddings=true (HuggingFace convention)
    if config.include_embedding && !config.tied_embeddings {
        let lm_head_data: Vec<f32> = (0..config.vocab_size * config.hidden_size)
            .map(|i| ((i % 100) as f32 - 50.0) / 1000.0)
            .collect();
        tensors.push((
            "lm_head.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
            lm_head_data,
        ));
    }

    // Build SafeTensors format
    build_safetensors_bytes(&tensors)
}

/// Build SafeTensors bytes from tensor list
fn build_safetensors_bytes(tensors: &[(String, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
    use std::collections::BTreeMap;

    // Calculate tensor data offsets
    let mut current_offset = 0usize;
    let mut tensor_info: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    let mut all_data = Vec::new();

    for (name, shape, data) in tensors {
        let byte_size = data.len() * 4; // f32 = 4 bytes

        // SafeTensors format: {"dtype": "F32", "shape": [...], "data_offsets": [start, end]}
        tensor_info.insert(
            name.clone(),
            serde_json::json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [current_offset, current_offset + byte_size]
            }),
        );

        // Append tensor data (little-endian f32)
        for &val in data {
            all_data.extend_from_slice(&val.to_le_bytes());
        }

        current_offset += byte_size;
    }

    // Add __metadata__ for completeness
    tensor_info.insert(
        "__metadata__".to_string(),
        serde_json::json!({"format": "pt", "pygmy": "true"}),
    );

    // Serialize header
    let header_json = serde_json::to_string(&tensor_info).unwrap_or_default();
    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;

    // Build final file: [header_len: u64] + [header: JSON] + [tensor_data]
    let mut result = Vec::with_capacity(8 + header_bytes.len() + all_data.len());
    result.extend_from_slice(&header_len.to_le_bytes());
    result.extend_from_slice(header_bytes);
    result.extend_from_slice(&all_data);

    result
}

// ============================================================================
// GH-205: F16 SafeTensors Builder for Passthrough Testing
// ============================================================================

/// Build a minimal valid F16 SafeTensors file in memory
///
/// Creates a SafeTensors file with F16 (half-precision) tensors for testing
/// the GH-205 F16 passthrough fix. The F32 values are converted to F16.
#[must_use]
pub fn build_pygmy_safetensors_f16() -> Vec<u8> {
    build_pygmy_safetensors_f16_with_config(PygmyConfig::minimal())
}

/// Build F16 SafeTensors with custom config
#[must_use]
pub fn build_pygmy_safetensors_f16_with_config(config: PygmyConfig) -> Vec<u8> {
    // Build tensor metadata and data (same as F32 version, but stored as F16)
    let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();

    // Token embedding: [vocab_size, hidden_size]
    if config.include_embedding {
        let embed_data: Vec<f32> = (0..config.vocab_size * config.hidden_size)
            .map(|i| ((i % 100) as f32 - 50.0) / 1000.0)
            .collect();
        tensors.push((
            "model.embed_tokens.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
            embed_data,
        ));
    }

    // LM head: [vocab_size, hidden_size]
    if config.include_embedding && !config.tied_embeddings {
        let lm_head_data: Vec<f32> = (0..config.vocab_size * config.hidden_size)
            .map(|i| ((i % 100) as f32 - 50.0) / 1000.0)
            .collect();
        tensors.push((
            "lm_head.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
            lm_head_data,
        ));
    }

    build_safetensors_bytes_f16(&tensors)
}

/// Build F16 SafeTensors bytes from tensor list
///
/// Converts F32 values to F16 format for storage.
fn build_safetensors_bytes_f16(tensors: &[(String, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
    use std::collections::BTreeMap;

    // Calculate tensor data offsets
    let mut current_offset = 0usize;
    let mut tensor_info: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    let mut all_data = Vec::new();

    for (name, shape, data) in tensors {
        let byte_size = data.len() * 2; // f16 = 2 bytes

        // SafeTensors format: {"dtype": "F16", "shape": [...], "data_offsets": [start, end]}
        tensor_info.insert(
            name.clone(),
            serde_json::json!({
                "dtype": "F16",
                "shape": shape,
                "data_offsets": [current_offset, current_offset + byte_size]
            }),
        );

        // Append tensor data (little-endian f16)
        for &val in data {
            all_data.extend_from_slice(&f32_to_f16_bits(val).to_le_bytes());
        }

        current_offset += byte_size;
    }

    // Add __metadata__ for completeness
    tensor_info.insert(
        "__metadata__".to_string(),
        serde_json::json!({"format": "pt", "pygmy": "true", "dtype": "F16"}),
    );

    // Serialize header
    let header_json = serde_json::to_string(&tensor_info).unwrap_or_default();
    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;

    // Build final file: [header_len: u64] + [header: JSON] + [tensor_data]
    let mut result = Vec::with_capacity(8 + header_bytes.len() + all_data.len());
    result.extend_from_slice(&header_len.to_le_bytes());
    result.extend_from_slice(header_bytes);
    result.extend_from_slice(&all_data);

    result
}

/// Convert f32 to f16 bits (IEEE 754 half-precision)
fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007fffff;

    if exp == 255 {
        // Inf or NaN
        if mant != 0 {
            sign | 0x7e00 // NaN
        } else {
            sign | 0x7c00 // Inf
        }
    } else if exp > 142 {
        // Overflow to Inf
        sign | 0x7c00
    } else if exp < 113 {
        // Underflow to zero (or denormal, which we skip for simplicity)
        sign
    } else {
        // Normal number
        let new_exp = ((exp - 127 + 15) as u16) << 10;
        let new_mant = (mant >> 13) as u16;
        sign | new_exp | new_mant
    }
}

// ============================================================================
// APR v2 Pygmy Builder
// ============================================================================

/// Build a minimal valid APR v2 file in memory
///
/// Creates an APR v2 file with:
/// - Valid header with magic "APR2"
/// - Metadata with architecture info
/// - F32 tensors (embedding + weight)
/// - 64-byte alignment
#[must_use]
pub fn build_pygmy_apr() -> Vec<u8> {
    build_pygmy_apr_with_config(PygmyConfig::default())
}

/// Build APR v2 with custom config
#[must_use]
pub fn build_pygmy_apr_with_config(config: PygmyConfig) -> Vec<u8> {
    let mut metadata = AprV2Metadata::new("pygmy");
    metadata.architecture = Some("llama".to_string());
    metadata.hidden_size = Some(config.hidden_size);
    metadata.vocab_size = Some(config.vocab_size);
    metadata.num_layers = Some(config.num_layers);
    metadata
        .custom
        .insert("pygmy".to_string(), serde_json::json!(true));

    let mut writer = AprV2Writer::new(metadata);

    // Token embedding
    if config.include_embedding {
        let embed_data: Vec<f32> = (0..config.vocab_size * config.hidden_size)
            .map(|i| ((i % 100) as f32 - 50.0) / 1000.0)
            .collect();
        writer.add_f32_tensor(
            "model.embed_tokens.weight",
            vec![config.vocab_size, config.hidden_size],
            &embed_data,
        );
    }

    // Layer tensors
    for layer_idx in 0..config.num_layers {
        if config.include_norms {
            let norm_data: Vec<f32> = vec![1.0; config.hidden_size];
            writer.add_f32_tensor(
                format!("model.layers.{layer_idx}.input_layernorm.weight"),
                vec![config.hidden_size],
                &norm_data,
            );
            writer.add_f32_tensor(
                format!("model.layers.{layer_idx}.post_attention_layernorm.weight"),
                vec![config.hidden_size],
                &norm_data,
            );
        }

        if config.include_attention {
            let qkvo_data: Vec<f32> = (0..config.hidden_size * config.hidden_size)
                .map(|i| ((i % 200) as f32 - 100.0) / 2000.0)
                .collect();
            for suffix in &["q_proj", "k_proj", "v_proj", "o_proj"] {
                writer.add_f32_tensor(
                    format!("model.layers.{layer_idx}.self_attn.{suffix}.weight"),
                    vec![config.hidden_size, config.hidden_size],
                    &qkvo_data,
                );
            }
        }

        if config.include_mlp {
            let intermediate = config.hidden_size * 2;
            let gate_up_data: Vec<f32> = (0..intermediate * config.hidden_size)
                .map(|i| ((i % 200) as f32 - 100.0) / 2000.0)
                .collect();
            let down_data: Vec<f32> = (0..config.hidden_size * intermediate)
                .map(|i| ((i % 200) as f32 - 100.0) / 2000.0)
                .collect();

            writer.add_f32_tensor(
                format!("model.layers.{layer_idx}.mlp.gate_proj.weight"),
                vec![intermediate, config.hidden_size],
                &gate_up_data,
            );
            writer.add_f32_tensor(
                format!("model.layers.{layer_idx}.mlp.up_proj.weight"),
                vec![intermediate, config.hidden_size],
                &gate_up_data,
            );
            writer.add_f32_tensor(
                format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
                vec![config.hidden_size, intermediate],
                &down_data,
            );
        }
    }

    // Final norm
    if config.include_norms && config.num_layers > 0 {
        let norm_data: Vec<f32> = vec![1.0; config.hidden_size];
        writer.add_f32_tensor("model.norm.weight", vec![config.hidden_size], &norm_data);
    }

    // LM head
    if config.include_embedding {
        let lm_head_data: Vec<f32> = (0..config.vocab_size * config.hidden_size)
            .map(|i| ((i % 100) as f32 - 50.0) / 1000.0)
            .collect();
        writer.add_f32_tensor(
            "lm_head.weight",
            vec![config.vocab_size, config.hidden_size],
            &lm_head_data,
        );
    }

    writer.write().unwrap_or_default()
}

/// Build APR with Q8 quantized tensors
#[must_use]
pub fn build_pygmy_apr_q8() -> Vec<u8> {
    let config = PygmyConfig::default();
    let mut metadata = AprV2Metadata::new("pygmy");
    metadata.architecture = Some("llama".to_string());
    metadata.hidden_size = Some(config.hidden_size);
    metadata.vocab_size = Some(config.vocab_size);

    let mut writer = AprV2Writer::new(metadata);

    // Embedding (F32 - lookup table)
    let embed_data: Vec<f32> = (0..config.vocab_size * config.hidden_size)
        .map(|i| ((i % 100) as f32 - 50.0) / 1000.0)
        .collect();
    writer.add_f32_tensor(
        "model.embed_tokens.weight",
        vec![config.vocab_size, config.hidden_size],
        &embed_data,
    );

    // Layer 0 attention (Q8)
    let qkvo_data: Vec<f32> = (0..config.hidden_size * config.hidden_size)
        .map(|i| ((i % 200) as f32 - 100.0) / 2000.0)
        .collect();
    for suffix in &["q_proj", "k_proj", "v_proj", "o_proj"] {
        writer.add_q8_tensor(
            format!("model.layers.0.self_attn.{suffix}.weight"),
            vec![config.hidden_size, config.hidden_size],
            &qkvo_data,
        );
    }

    // Norms (F32)
    let norm_data: Vec<f32> = vec![1.0; config.hidden_size];
    writer.add_f32_tensor(
        "model.layers.0.input_layernorm.weight",
        vec![config.hidden_size],
        &norm_data,
    );
    writer.add_f32_tensor("model.norm.weight", vec![config.hidden_size], &norm_data);

    writer.write().unwrap_or_default()
}

/// Build APR with Q4 quantized tensors
#[must_use]
pub fn build_pygmy_apr_q4() -> Vec<u8> {
    let config = PygmyConfig::default();
    let mut metadata = AprV2Metadata::new("pygmy");
    metadata.architecture = Some("llama".to_string());
    metadata.hidden_size = Some(32); // Q4 block size alignment
    metadata.vocab_size = Some(config.vocab_size);

    let mut writer = AprV2Writer::new(metadata);

    // Embedding (F32)
    let embed_data: Vec<f32> = (0..config.vocab_size * config.hidden_size)
        .map(|i| ((i % 100) as f32 - 50.0) / 1000.0)
        .collect();
    writer.add_f32_tensor(
        "model.embed_tokens.weight",
        vec![config.vocab_size, config.hidden_size],
        &embed_data,
    );

    // Layer 0 attention (Q4) - need at least 32 elements for Q4 blocks
    let q4_size = 32; // Q4 block size
    let q4_data: Vec<f32> = (0..q4_size * q4_size)
        .map(|i| ((i % 200) as f32 - 100.0) / 2000.0)
        .collect();
    for suffix in &["q_proj", "k_proj", "v_proj", "o_proj"] {
        writer.add_q4_tensor(
            format!("model.layers.0.self_attn.{suffix}.weight"),
            vec![q4_size, q4_size],
            &q4_data,
        );
    }

    // Norms (F32)
    let norm_data: Vec<f32> = vec![1.0; q4_size];
    writer.add_f32_tensor(
        "model.layers.0.input_layernorm.weight",
        vec![q4_size],
        &norm_data,
    );
    writer.add_f32_tensor("model.norm.weight", vec![q4_size], &norm_data);

    writer.write().unwrap_or_default()
}

/// Build APR with F16 tensors
#[must_use]
pub fn build_pygmy_apr_f16() -> Vec<u8> {
    let config = PygmyConfig::default();
    let mut metadata = AprV2Metadata::new("pygmy");
    metadata.architecture = Some("llama".to_string());
    metadata.hidden_size = Some(config.hidden_size);
    metadata.vocab_size = Some(config.vocab_size);

    let mut writer = AprV2Writer::new(metadata);

    // Token embedding (F16)
    let embed_data: Vec<f32> = (0..config.vocab_size * config.hidden_size)
        .map(|i| ((i % 100) as f32 - 50.0) / 1000.0)
        .collect();
    writer.add_f16_tensor(
        "model.embed_tokens.weight",
        vec![config.vocab_size, config.hidden_size],
        &embed_data,
    );

    // Layer 0 attention (F16)
    let qkvo_data: Vec<f32> = (0..config.hidden_size * config.hidden_size)
        .map(|i| ((i % 200) as f32 - 100.0) / 2000.0)
        .collect();
    for suffix in &["q_proj", "k_proj", "v_proj", "o_proj"] {
        writer.add_f16_tensor(
            format!("model.layers.0.self_attn.{suffix}.weight"),
            vec![config.hidden_size, config.hidden_size],
            &qkvo_data,
        );
    }

    // Norms (F16)
    let norm_data: Vec<f32> = vec![1.0; config.hidden_size];
    writer.add_f16_tensor(
        "model.layers.0.input_layernorm.weight",
        vec![config.hidden_size],
        &norm_data,
    );
    writer.add_f16_tensor("model.norm.weight", vec![config.hidden_size], &norm_data);

    writer.write().unwrap_or_default()
}

// ============================================================================
// K-Quant Pygmy Builders (Q4K / Q6K)
// ============================================================================

/// Build APR with Q4_K quantized tensors (GGUF-style names)
///
/// Q4_K format: 256-element super-blocks, 144 bytes each.
/// Layout: d (f16, 2B) + dmin (f16, 2B) + scales (12B) + qs (128B) = 144 bytes.
/// Uses GGUF naming: `token_embd.weight`, `blk.0.attn_q.weight`, etc.
#[must_use]
pub fn build_pygmy_apr_q4k() -> Vec<u8> {
    let mut metadata = AprV2Metadata::new("pygmy-q4k");
    metadata.architecture = Some("qwen2".to_string());
    metadata.hidden_size = Some(256);
    metadata.vocab_size = Some(8);
    metadata.num_layers = Some(1);
    metadata
        .custom
        .insert("naming".to_string(), serde_json::json!("gguf"));

    let mut writer = AprV2Writer::new(metadata);

    // Token embedding (F32 -- embedding lookup tables stay unquantized)
    let embed_data: Vec<f32> = (0..8 * 256)
        .map(|i| ((i % 100) as f32 - 50.0) / 1000.0)
        .collect();
    writer.add_f32_tensor("token_embd.weight", vec![8, 256], &embed_data);

    // Layer 0 attention weights (Q4K): 256 elements = 1 super-block = 144 bytes
    let q4k_block = vec![0u8; 144];
    for suffix in &["attn_q", "attn_k", "attn_v", "attn_output"] {
        writer.add_q4k_raw_tensor(
            format!("blk.0.{suffix}.weight"),
            vec![256, 1],
            q4k_block.clone(),
        );
    }

    // Norms (F32)
    let norm_data: Vec<f32> = vec![1.0; 256];
    writer.add_f32_tensor("blk.0.attn_norm.weight", vec![256], &norm_data);
    writer.add_f32_tensor("output_norm.weight", vec![256], &norm_data);

    // Output (F32)
    let output_data: Vec<f32> = (0..8 * 256)
        .map(|i| ((i % 100) as f32 - 50.0) / 1000.0)
        .collect();
    writer.add_f32_tensor("output.weight", vec![8, 256], &output_data);

    writer.write().unwrap_or_default()
}

/// Build APR with Q6_K quantized tensors (GGUF-style names)
///
/// Q6_K format: 256-element super-blocks, 210 bytes each.
/// Layout: ql (128B) + qh (64B) + scales (16B) + d (f16, 2B) = 210 bytes.
/// Uses GGUF naming: `token_embd.weight`, `blk.0.attn_q.weight`, etc.
#[must_use]
pub fn build_pygmy_apr_q6k() -> Vec<u8> {
    let mut metadata = AprV2Metadata::new("pygmy-q6k");
    metadata.architecture = Some("qwen2".to_string());
    metadata.hidden_size = Some(256);
    metadata.vocab_size = Some(8);
    metadata.num_layers = Some(1);
    metadata
        .custom
        .insert("naming".to_string(), serde_json::json!("gguf"));

    let mut writer = AprV2Writer::new(metadata);

    // Token embedding (F32)
    let embed_data: Vec<f32> = (0..8 * 256)
        .map(|i| ((i % 100) as f32 - 50.0) / 1000.0)
        .collect();
    writer.add_f32_tensor("token_embd.weight", vec![8, 256], &embed_data);

    // Layer 0 attention weights (Q6K): 256 elements = 1 super-block = 210 bytes
    let q6k_block = vec![0u8; 210];
    for suffix in &["attn_q", "attn_k", "attn_v", "attn_output"] {
        writer.add_q6k_raw_tensor(
            format!("blk.0.{suffix}.weight"),
            vec![256, 1],
            q6k_block.clone(),
        );
    }

    // Norms (F32)
    let norm_data: Vec<f32> = vec![1.0; 256];
    writer.add_f32_tensor("blk.0.attn_norm.weight", vec![256], &norm_data);
    writer.add_f32_tensor("output_norm.weight", vec![256], &norm_data);

    // Output (F32)
    let output_data: Vec<f32> = (0..8 * 256)
        .map(|i| ((i % 100) as f32 - 50.0) / 1000.0)
        .collect();
    writer.add_f32_tensor("output.weight", vec![8, 256], &output_data);

    writer.write().unwrap_or_default()
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

    // Token embedding: GGUF uses `token_embd.weight` (NOT model.embed_tokens.weight)
    // Shape: [vocab_size, hidden_size] in GGUF (row-major)
    let embed_data: Vec<f32> = (0..config.vocab_size * config.hidden_size)
        .map(|i| ((i % 100) as f32 - 50.0) / 1000.0)
        .collect();
    writer.add_f32_tensor(
        "token_embd.weight",
        vec![config.vocab_size, config.hidden_size],
        &embed_data,
    );

    // Layer tensors with GGUF naming
    for layer_idx in 0..config.num_layers {
        // Input layernorm: blk.N.attn_norm.weight
        let norm_data: Vec<f32> = vec![1.0; config.hidden_size];
        writer.add_f32_tensor(
            format!("blk.{layer_idx}.attn_norm.weight"),
            vec![config.hidden_size],
            &norm_data,
        );
        writer.add_f32_tensor(
            format!("blk.{layer_idx}.ffn_norm.weight"),
            vec![config.hidden_size],
            &norm_data,
        );

        // Attention: blk.N.attn_{q,k,v,output}.weight
        let qkvo_data: Vec<f32> = (0..config.hidden_size * config.hidden_size)
            .map(|i| ((i % 200) as f32 - 100.0) / 2000.0)
            .collect();
        for suffix in &["attn_q", "attn_k", "attn_v", "attn_output"] {
            writer.add_f32_tensor(
                format!("blk.{layer_idx}.{suffix}.weight"),
                vec![config.hidden_size, config.hidden_size],
                &qkvo_data,
            );
        }

        // MLP: blk.N.ffn_{gate,up,down}.weight
        let intermediate = config.hidden_size * 2;
        let gate_up_data: Vec<f32> = (0..intermediate * config.hidden_size)
            .map(|i| ((i % 200) as f32 - 100.0) / 2000.0)
            .collect();
        let down_data: Vec<f32> = (0..config.hidden_size * intermediate)
            .map(|i| ((i % 200) as f32 - 100.0) / 2000.0)
            .collect();

        writer.add_f32_tensor(
            format!("blk.{layer_idx}.ffn_gate.weight"),
            vec![intermediate, config.hidden_size],
            &gate_up_data,
        );
        writer.add_f32_tensor(
            format!("blk.{layer_idx}.ffn_up.weight"),
            vec![intermediate, config.hidden_size],
            &gate_up_data,
        );
        writer.add_f32_tensor(
            format!("blk.{layer_idx}.ffn_down.weight"),
            vec![config.hidden_size, intermediate],
            &down_data,
        );
    }

    // Output norm: output_norm.weight
    let norm_data: Vec<f32> = vec![1.0; config.hidden_size];
    writer.add_f32_tensor("output_norm.weight", vec![config.hidden_size], &norm_data);

    // LM head (output projection)
    if config.weight_tying {
        // Weight tying: NO separate output.weight tensor
        // realizaer must use token_embd.weight (transposed) for lm_head
        // This is the GH-194 bug scenario
    } else {
        // No weight tying: separate output.weight tensor
        let lm_head_data: Vec<f32> = (0..config.vocab_size * config.hidden_size)
            .map(|i| ((i % 100) as f32 - 50.0) / 1000.0)
            .collect();
        writer.add_f32_tensor(
            "output.weight",
            vec![config.vocab_size, config.hidden_size],
            &lm_head_data,
        );
    }

    writer.write().unwrap_or_default()
}

/// Build APR with HuggingFace-style naming and weight tying
///
/// HuggingFace naming with weight tying uses `model.embed_tokens.weight`
/// for both embedding lookup and lm_head (tied weights).
#[must_use]
pub fn build_pygmy_apr_hf_names_tied() -> Vec<u8> {
    let config = PygmyConfig::default();
    let mut metadata = AprV2Metadata::new("pygmy-hf-tied");
    metadata.architecture = Some("qwen2".to_string());
    metadata.hidden_size = Some(config.hidden_size);
    metadata.vocab_size = Some(config.vocab_size);
    metadata.num_layers = Some(config.num_layers);
    metadata
        .custom
        .insert("naming".to_string(), serde_json::json!("huggingface"));
    metadata
        .custom
        .insert("weight_tying".to_string(), serde_json::json!(true));

    let mut writer = AprV2Writer::new(metadata);

    // Token embedding: model.embed_tokens.weight
    let embed_data: Vec<f32> = (0..config.vocab_size * config.hidden_size)
        .map(|i| ((i % 100) as f32 - 50.0) / 1000.0)
        .collect();
    writer.add_f32_tensor(
        "model.embed_tokens.weight",
        vec![config.vocab_size, config.hidden_size],
        &embed_data,
    );

    // Layer tensors with HuggingFace naming
    for layer_idx in 0..config.num_layers {
        // Norms
        let norm_data: Vec<f32> = vec![1.0; config.hidden_size];
        writer.add_f32_tensor(
            format!("model.layers.{layer_idx}.input_layernorm.weight"),
            vec![config.hidden_size],
            &norm_data,
        );
        writer.add_f32_tensor(
            format!("model.layers.{layer_idx}.post_attention_layernorm.weight"),
            vec![config.hidden_size],
            &norm_data,
        );

        // Attention
        let qkvo_data: Vec<f32> = (0..config.hidden_size * config.hidden_size)
            .map(|i| ((i % 200) as f32 - 100.0) / 2000.0)
            .collect();
        for suffix in &["q_proj", "k_proj", "v_proj", "o_proj"] {
            writer.add_f32_tensor(
                format!("model.layers.{layer_idx}.self_attn.{suffix}.weight"),
                vec![config.hidden_size, config.hidden_size],
                &qkvo_data,
            );
        }

        // MLP
        let intermediate = config.hidden_size * 2;
        let gate_up_data: Vec<f32> = (0..intermediate * config.hidden_size)
            .map(|i| ((i % 200) as f32 - 100.0) / 2000.0)
            .collect();
        let down_data: Vec<f32> = (0..config.hidden_size * intermediate)
            .map(|i| ((i % 200) as f32 - 100.0) / 2000.0)
            .collect();

        writer.add_f32_tensor(
            format!("model.layers.{layer_idx}.mlp.gate_proj.weight"),
            vec![intermediate, config.hidden_size],
            &gate_up_data,
        );
        writer.add_f32_tensor(
            format!("model.layers.{layer_idx}.mlp.up_proj.weight"),
            vec![intermediate, config.hidden_size],
            &gate_up_data,
        );
        writer.add_f32_tensor(
            format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
            vec![config.hidden_size, intermediate],
            &down_data,
        );
    }

    // Final norm
    let norm_data: Vec<f32> = vec![1.0; config.hidden_size];
    writer.add_f32_tensor("model.norm.weight", vec![config.hidden_size], &norm_data);

    // NO lm_head.weight - weight tying uses model.embed_tokens.weight
    // realizaer must find lm_head via model.embed_tokens.weight or embed_tokens.weight

    writer.write().unwrap_or_default()
}
