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

/// Generate embedding-style tensor data: values in [-0.05, 0.05] range (SafeTensors variant)
///
/// NOTE: This duplicates gen_embed_data from builders_part_02.rs because
/// builders.rs is compiled before the include!() of part_02. Both are
/// private helpers generating the same deterministic pattern.
fn gen_st_embed_data(count: usize) -> Vec<f32> {
    (0..count)
        .map(|i| ((i % 100) as f32 - 50.0) / 1000.0)
        .collect()
}

/// Generate weight-style tensor data: values in [-0.05, 0.05] range (SafeTensors variant)
fn gen_st_weight_data(count: usize) -> Vec<f32> {
    (0..count)
        .map(|i| ((i % 200) as f32 - 100.0) / 2000.0)
        .collect()
}

/// Build SafeTensors with custom config
#[must_use]
pub fn build_pygmy_safetensors_with_config(config: PygmyConfig) -> Vec<u8> {
    // Build tensor metadata and data
    let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    let h = config.hidden_size;
    let v = config.vocab_size;

    // Token embedding: [vocab_size, hidden_size]
    if config.include_embedding {
        tensors.push((
            "model.embed_tokens.weight".to_string(),
            vec![v, h],
            gen_st_embed_data(v * h),
        ));
    }

    // Layer tensors
    for layer_idx in 0..config.num_layers {
        // Input layernorm
        if config.include_norms {
            let norm_data: Vec<f32> = vec![1.0; h];
            tensors.push((
                format!("model.layers.{layer_idx}.input_layernorm.weight"),
                vec![h],
                norm_data.clone(),
            ));
            tensors.push((
                format!("model.layers.{layer_idx}.post_attention_layernorm.weight"),
                vec![h],
                norm_data,
            ));
        }

        // Attention: Q, K, V, O projections
        // ROSETTA-003: GQA support - K/V may have fewer heads than Q
        if config.include_attention {
            let kv_dim = config.kv_dim();

            // Q and O: [hidden_size, hidden_size]
            let q_data = gen_st_weight_data(h * h);
            tensors.push((
                format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
                vec![h, h],
                q_data.clone(),
            ));
            tensors.push((
                format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
                vec![h, h],
                q_data,
            ));

            // K and V: [kv_dim, hidden_size] (may differ from Q for GQA)
            let kv_data = gen_st_weight_data(kv_dim * h);
            tensors.push((
                format!("model.layers.{layer_idx}.self_attn.k_proj.weight"),
                vec![kv_dim, h],
                kv_data.clone(),
            ));
            tensors.push((
                format!("model.layers.{layer_idx}.self_attn.v_proj.weight"),
                vec![kv_dim, h],
                kv_data,
            ));

            // Biases (Qwen2-style)
            if config.include_bias {
                let q_bias: Vec<f32> = (0..h).map(|i| (i as f32) / 1000.0).collect();
                tensors.push((
                    format!("model.layers.{layer_idx}.self_attn.q_proj.bias"),
                    vec![h],
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
            let intermediate = h * 2;
            let gate_up_data = gen_st_weight_data(intermediate * h);
            let down_data = gen_st_weight_data(h * intermediate);

            tensors.push((
                format!("model.layers.{layer_idx}.mlp.gate_proj.weight"),
                vec![intermediate, h],
                gate_up_data.clone(),
            ));
            tensors.push((
                format!("model.layers.{layer_idx}.mlp.up_proj.weight"),
                vec![intermediate, h],
                gate_up_data,
            ));
            tensors.push((
                format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
                vec![h, intermediate],
                down_data,
            ));
        }
    }

    // Final norm
    if config.include_norms && config.num_layers > 0 {
        tensors.push((
            "model.norm.weight".to_string(),
            vec![h],
            vec![1.0; h],
        ));
    }

    // LM head: [vocab_size, hidden_size]
    // ROSETTA-003: Omit lm_head when tied_embeddings=true (HuggingFace convention)
    if config.include_embedding && !config.tied_embeddings {
        tensors.push((
            "lm_head.weight".to_string(),
            vec![v, h],
            gen_st_embed_data(v * h),
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

include!("quant_variant.rs");
include!("gguf_pygmy_config.rs");
