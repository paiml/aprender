//! Qwen2-0.5B-Instruct Model Implementation
//!
//! This module provides a complete Qwen2 model for inference, assembling
//! primitives from the `nn` module into a decoder-only transformer.
//!
//! # Architecture (Bai et al., 2023)
//!
//! ```text
//! Qwen2-0.5B-Instruct:
//! ├── hidden_size: 896
//! ├── num_attention_heads: 14 (query heads)
//! ├── num_kv_heads: 2 (grouped query attention)
//! ├── num_layers: 24
//! ├── intermediate_size: 4864 (FFN)
//! ├── vocab_size: 151936
//! ├── max_seq_len: 32768
//! └── rope_theta: 1,000,000
//! ```
//!
//! # Example
//!
//! ```ignore
//! use aprender::models::Qwen2Model;
//! use aprender::demo::Qwen2Config;
//!
//! let config = Qwen2Config::qwen2_0_5b_instruct();
//! let model = Qwen2Model::new(&config);
//! assert_eq!(model.num_layers(), 24);
//! ```
//!
//! **Note:** Inference (`forward`/`generate`) is handled exclusively by `realizar`.
//! This module provides model construction, weight loading, and introspection only.
//!
//! # References
//!
//! - Bai et al. (2023). "Qwen Technical Report"
//! - Ainslie et al. (2023). "GQA: Training Generalized Multi-Query Transformer Models"
//! - Su et al. (2021). "`RoFormer`: Enhanced Transformer with Rotary Position Embedding"
//! - Zhang & Sennrich (2019). "Root Mean Square Layer Normalization"

use crate::autograd::Tensor;
use crate::demo::Qwen2Config;
use crate::nn::{GroupedQueryAttention, Linear, Module, RMSNorm, RotaryPositionEmbedding};

// ============================================================================
// Embedding Layer
// ============================================================================

/// Token embedding lookup table.
///
/// Maps token IDs to dense vectors.
#[derive(Debug)]
pub struct Embedding {
    /// Weight matrix [`vocab_size`, `hidden_size`]
    weight: Tensor,
    vocab_size: usize,
    hidden_size: usize,
}

impl Embedding {
    /// Create a new embedding layer.
    #[must_use]
    pub fn new(vocab_size: usize, hidden_size: usize) -> Self {
        // Initialize with small random values
        let data: Vec<f32> = (0..vocab_size * hidden_size)
            .map(|i| {
                // Deterministic pseudo-random initialization
                (i as f32 * 0.0001).sin() * 0.02
            })
            .collect();

        Self {
            weight: Tensor::new(&data, &[vocab_size, hidden_size]),
            vocab_size,
            hidden_size,
        }
    }

    /// Create a placeholder embedding with minimal memory allocation.
    ///
    /// Used for lazy initialization when loading pre-trained weights.
    /// Uses 1-element tensor instead of `vocab_size` * `hidden_size`.
    ///
    /// **IMPORTANT**: This layer will NOT work for inference until
    /// `set_weight()` is called with real weights.
    #[must_use]
    pub fn placeholder(vocab_size: usize, hidden_size: usize) -> Self {
        Self {
            weight: Tensor::new(&[0.0], &[1]),
            vocab_size,
            hidden_size,
        }
    }

    /// Look up embeddings for token IDs into a pre-allocated buffer.
    pub fn forward_into(&self, input_ids: &[u32], output: &mut [f32]) {
        for (s, &token_id) in input_ids.iter().enumerate() {
            let token_idx = token_id as usize;
            if token_idx >= self.vocab_size {
                // Out of vocabulary - zeros already in buffer if initialized
                continue;
            }

            let src_offset = token_idx * self.hidden_size;
            let dst_offset = s * self.hidden_size;

            output[dst_offset..dst_offset + self.hidden_size]
                .copy_from_slice(&self.weight.data()[src_offset..src_offset + self.hidden_size]);
        }
    }

    /// Look up embeddings for token IDs.
    #[must_use]
    pub fn forward(&self, input_ids: &[u32]) -> Tensor {
        let batch_size = 1;
        let mut output = vec![0.0f32; batch_size * input_ids.len() * self.hidden_size];
        self.forward_into(input_ids, &mut output);
        Tensor::new(&output, &[batch_size, input_ids.len(), self.hidden_size])
    }

    /// Set weights from external tensor.
    pub fn set_weight(&mut self, weight: Tensor) {
        self.weight = weight;
    }

    /// Get weight tensor reference.
    #[must_use]
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
}

// ============================================================================
// Qwen2 MLP (SwiGLU)
// ============================================================================

/// Qwen2 MLP with `SwiGLU` activation.
///
/// ```text
/// output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
/// ```
#[derive(Debug)]
#[allow(clippy::struct_field_names)] // Standard ML naming convention
pub struct Qwen2MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Qwen2MLP {
    /// Create a new Qwen2 MLP layer.
    #[must_use]
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            gate_proj: Linear::new(hidden_size, intermediate_size),
            up_proj: Linear::new(hidden_size, intermediate_size),
            down_proj: Linear::new(intermediate_size, hidden_size),
        }
    }

    /// Create a placeholder MLP with minimal memory allocation.
    ///
    /// Used for lazy initialization when loading pre-trained weights.
    #[must_use]
    pub fn placeholder(hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            gate_proj: Linear::placeholder(hidden_size, intermediate_size),
            up_proj: Linear::placeholder(hidden_size, intermediate_size),
            down_proj: Linear::placeholder(intermediate_size, hidden_size),
        }
    }

    /// Forward pass with `SwiGLU` activation.
    #[must_use]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let gate = self.gate_proj.forward(x);
        let gate_activated = silu(&gate);
        let up = self.up_proj.forward(x);
        let hidden = elementwise_mul(&gate_activated, &up);
        self.down_proj.forward(&hidden)
    }

    /// Get mutable reference to gate projection layer.
    pub fn gate_proj_mut(&mut self) -> &mut Linear {
        &mut self.gate_proj
    }

    /// Get mutable reference to up projection layer.
    pub fn up_proj_mut(&mut self) -> &mut Linear {
        &mut self.up_proj
    }

    /// Get mutable reference to down projection layer.
    pub fn down_proj_mut(&mut self) -> &mut Linear {
        &mut self.down_proj
    }
}

// ============================================================================
// Qwen2 Decoder Layer
// ============================================================================

/// Single Qwen2 decoder layer.
///
/// ```text
/// residual = x
/// x = input_layernorm(x)
/// x = self_attn(x, x, x) + residual
///
/// residual = x
/// x = post_attention_layernorm(x)
/// x = mlp(x) + residual
/// ```
#[derive(Debug)]
pub struct Qwen2DecoderLayer {
    self_attn: GroupedQueryAttention,
    mlp: Qwen2MLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl Qwen2DecoderLayer {
    /// Create a new decoder layer.
    #[must_use]
    pub fn new(config: &Qwen2Config) -> Self {
        Self {
            self_attn: GroupedQueryAttention::new(
                config.hidden_size,
                config.num_attention_heads,
                config.num_kv_heads,
            ),
            mlp: Qwen2MLP::new(config.hidden_size, config.intermediate_size),
            input_layernorm: RMSNorm::new(&[config.hidden_size]),
            post_attention_layernorm: RMSNorm::new(&[config.hidden_size]),
        }
    }

    /// Create a placeholder decoder layer with minimal memory allocation.
    ///
    /// Used for lazy initialization when loading pre-trained weights.
    #[must_use]
    pub fn placeholder(config: &Qwen2Config) -> Self {
        Self {
            self_attn: GroupedQueryAttention::placeholder(
                config.hidden_size,
                config.num_attention_heads,
                config.num_kv_heads,
            ),
            mlp: Qwen2MLP::placeholder(config.hidden_size, config.intermediate_size),
            input_layernorm: RMSNorm::placeholder(&[config.hidden_size]),
            post_attention_layernorm: RMSNorm::placeholder(&[config.hidden_size]),
        }
    }

    /// Forward pass through the decoder layer.
    #[must_use]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        _position_ids: &[usize],
        _rope: &RotaryPositionEmbedding,
        _attention_mask: Option<&Tensor>,
    ) -> Tensor {
        // Self-attention with pre-norm
        // Note: Attention mask handling is simplified - using None for now
        // Full implementation would reshape mask for multi-head attention
        let residual = hidden_states.clone();
        let hidden = self.input_layernorm.forward(hidden_states);
        let (attn_output, _attn_weights) = self.self_attn.forward_self(&hidden, None);
        let hidden = add_tensors(&residual, &attn_output);

        // MLP with pre-norm
        let residual = hidden.clone();
        let hidden = self.post_attention_layernorm.forward(&hidden);
        let mlp_output = self.mlp.forward(&hidden);
        add_tensors(&residual, &mlp_output)
    }

    /// Get mutable reference to self-attention layer.
    pub fn self_attn_mut(&mut self) -> &mut GroupedQueryAttention {
        &mut self.self_attn
    }

    /// Get mutable reference to MLP layer.
    pub fn mlp_mut(&mut self) -> &mut Qwen2MLP {
        &mut self.mlp
    }

    /// Get mutable reference to input layernorm.
    pub fn input_layernorm_mut(&mut self) -> &mut RMSNorm {
        &mut self.input_layernorm
    }

    /// Get mutable reference to post-attention layernorm.
    pub fn post_attention_layernorm_mut(&mut self) -> &mut RMSNorm {
        &mut self.post_attention_layernorm
    }
}

// ============================================================================
// KV Cache
// ============================================================================

/// Key-Value cache for efficient autoregressive generation.
#[derive(Debug)]
pub struct KVCache {
    /// Cached keys per layer: [batch, `num_kv_heads`, `cached_len`, `head_dim`]
    pub keys: Vec<Option<Tensor>>,
    /// Cached values per layer
    pub values: Vec<Option<Tensor>>,
    /// Number of cached positions
    pub cached_len: usize,
}

impl KVCache {
    /// Create a new empty KV cache.
    #[must_use]
    pub fn new(num_layers: usize) -> Self {
        Self {
            keys: vec![None; num_layers],
            values: vec![None; num_layers],
            cached_len: 0,
        }
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        for k in &mut self.keys {
            *k = None;
        }
        for v in &mut self.values {
            *v = None;
        }
        self.cached_len = 0;
    }
}

// ============================================================================
// Qwen2 Model
// ============================================================================

/// Complete Qwen2 model for inference.
///
/// Assembles embedding, decoder layers, and LM head into a complete model.
#[derive(Debug)]
pub struct Qwen2Model {
    /// Token embeddings [`vocab_size`, `hidden_size`]
    embed_tokens: Embedding,
    /// Decoder layers
    layers: Vec<Qwen2DecoderLayer>,
    /// Final `RMSNorm`
    norm: RMSNorm,
    /// Language model head [`hidden_size`, `vocab_size`]
    lm_head: Linear,
    /// Rotary position embeddings (used by realizar for inference)
    #[allow(dead_code)]
    rope: RotaryPositionEmbedding,
    /// Model configuration
    config: Qwen2Config,
    /// KV cache for generation
    kv_cache: Option<KVCache>,
    /// Training mode flag
    training: bool,
}

impl Qwen2Model {
    /// Create a new Qwen2 model from configuration.
    ///
    /// Weights are initialized randomly. Use `load()` to load pre-trained weights.
    #[must_use]
    pub fn new(config: &Qwen2Config) -> Self {
        let head_dim = config.hidden_size / config.num_attention_heads;

        Self {
            embed_tokens: Embedding::new(config.vocab_size, config.hidden_size),
            layers: (0..config.num_layers)
                .map(|_| Qwen2DecoderLayer::new(config))
                .collect(),
            norm: RMSNorm::new(&[config.hidden_size]),
            lm_head: Linear::new(config.hidden_size, config.vocab_size),
            rope: RotaryPositionEmbedding::with_base(
                head_dim,
                config.max_seq_len,
                config.rope_theta as f32,
            ),
            config: config.clone(),
            kv_cache: None,
            training: false,
        }
    }

    /// Create an uninitialized Qwen2 model with minimal memory allocation.
    ///
    /// The model is not ready for inference until weights are loaded.
    #[must_use]
    pub fn new_uninitialized(config: &Qwen2Config) -> Self {
        let head_dim = config.hidden_size / config.num_attention_heads;

        Self {
            embed_tokens: Embedding::placeholder(config.vocab_size, config.hidden_size),
            layers: (0..config.num_layers)
                .map(|_| Qwen2DecoderLayer::placeholder(config))
                .collect(),
            norm: RMSNorm::placeholder(&[config.hidden_size]),
            lm_head: Linear::placeholder(config.hidden_size, config.vocab_size),
            rope: RotaryPositionEmbedding::with_base(
                head_dim,
                config.max_seq_len,
                config.rope_theta as f32,
            ),
            config: config.clone(),
            kv_cache: None,
            training: false,
        }
    }

    /// Get model configuration.
    #[must_use]
    pub fn config(&self) -> &Qwen2Config {
        &self.config
    }

    /// Set model to evaluation mode (no dropout).
    pub fn eval(&mut self) {
        self.training = false;
    }

    /// Set model to training mode.
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Enable KV cache for efficient generation.
    pub fn enable_cache(&mut self) {
        self.kv_cache = Some(KVCache::new(self.config.num_layers));
    }

    /// Disable KV cache.
    pub fn disable_cache(&mut self) {
        self.kv_cache = None;
    }

    /// Clear KV cache.
    pub fn clear_cache(&mut self) {
        if let Some(ref mut cache) = self.kv_cache {
            cache.clear();
        }
    }

    /// Get number of layers.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    // ========================================================================
    // Weight Introspection Methods (Section A: Model Loading)
    // ========================================================================

    /// Get list of weight names following `HuggingFace` convention.
    ///
    /// Returns names like:
    /// - `model.embed_tokens.weight`
    /// - `model.layers.0.self_attn.q_proj.weight`
    /// - `model.norm.weight`
    /// - `lm_head.weight`
    #[must_use]
    pub fn weight_names(&self) -> Vec<String> {
        let mut names = Vec::new();

        // Embedding
        names.push("model.embed_tokens.weight".to_string());

        // Decoder layers
        for i in 0..self.layers.len() {
            let prefix = format!("model.layers.{i}");

            // Self-attention projections
            names.push(format!("{prefix}.self_attn.q_proj.weight"));
            names.push(format!("{prefix}.self_attn.k_proj.weight"));
            names.push(format!("{prefix}.self_attn.v_proj.weight"));
            names.push(format!("{prefix}.self_attn.o_proj.weight"));

            // MLP
            names.push(format!("{prefix}.mlp.gate_proj.weight"));
            names.push(format!("{prefix}.mlp.up_proj.weight"));
            names.push(format!("{prefix}.mlp.down_proj.weight"));

            // Layer norms
            names.push(format!("{prefix}.input_layernorm.weight"));
            names.push(format!("{prefix}.post_attention_layernorm.weight"));
        }

        // Final norm
        names.push("model.norm.weight".to_string());

        // LM head
        names.push("lm_head.weight".to_string());

        names
    }

    /// Get weight shapes as a map from name to shape.
    #[must_use]
    pub fn weight_info(&self) -> std::collections::HashMap<String, Vec<usize>> {
        use std::collections::HashMap;
        let mut info = HashMap::new();

        let h = self.config.hidden_size;
        let v = self.config.vocab_size;
        let i = self.config.intermediate_size;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = h / num_heads;
        let kv_dim = num_kv_heads * head_dim;

        // Embedding: [vocab_size, hidden_size]
        info.insert("model.embed_tokens.weight".to_string(), vec![v, h]);

        // Per-layer weights
        for layer_idx in 0..self.layers.len() {
            let prefix = format!("model.layers.{layer_idx}");

            // Attention projections
            info.insert(format!("{prefix}.self_attn.q_proj.weight"), vec![h, h]);
            info.insert(format!("{prefix}.self_attn.k_proj.weight"), vec![kv_dim, h]);
            info.insert(format!("{prefix}.self_attn.v_proj.weight"), vec![kv_dim, h]);
            info.insert(format!("{prefix}.self_attn.o_proj.weight"), vec![h, h]);

            // MLP
            info.insert(format!("{prefix}.mlp.gate_proj.weight"), vec![i, h]);
            info.insert(format!("{prefix}.mlp.up_proj.weight"), vec![i, h]);
            info.insert(format!("{prefix}.mlp.down_proj.weight"), vec![h, i]);

            // Norms
            info.insert(format!("{prefix}.input_layernorm.weight"), vec![h]);
            info.insert(format!("{prefix}.post_attention_layernorm.weight"), vec![h]);
        }

        // Final norm
        info.insert("model.norm.weight".to_string(), vec![h]);

        // LM head
        info.insert("lm_head.weight".to_string(), vec![v, h]);

        info
    }

    /// Extract accessible weights as a map from name to f32 data.
    ///
    /// Returns a map suitable for serialization to `SafeTensors` format.
    /// Note: Currently returns weights from components with public accessors.
    /// Full weight export will be enabled when nn modules expose weight accessors.
    #[must_use]
    pub fn weights(&self) -> std::collections::HashMap<String, Vec<f32>> {
        use std::collections::HashMap;
        let mut weights = HashMap::new();

        // Embedding weights (direct access via our Embedding struct)
        weights.insert(
            "model.embed_tokens.weight".to_string(),
            self.embed_tokens.weight.data().to_vec(),
        );

        // Note: lm_head and norm weights require nn::Linear and nn::RMSNorm
        // to expose weight() accessors. For now, return embedding only.
        // This is sufficient for weight loading tests.

        weights
    }

    /// Get total number of parameters in the model.
    #[must_use]
    pub fn num_parameters(&self) -> usize {
        let info = self.weight_info();
        info.values()
            .map(|shape| shape.iter().product::<usize>())
            .sum()
    }

    // ========================================================================
    // Mutable Accessors for Weight Loading
    // ========================================================================

    /// Get mutable reference to embedding layer.
    pub fn embed_tokens_mut(&mut self) -> &mut Embedding {
        &mut self.embed_tokens
    }

    /// Get mutable reference to decoder layer at index.
    pub fn layer_mut(&mut self, idx: usize) -> Option<&mut Qwen2DecoderLayer> {
        self.layers.get_mut(idx)
    }

    /// Get mutable reference to final norm layer.
    pub fn norm_mut(&mut self) -> &mut RMSNorm {
        &mut self.norm
    }

    /// Get mutable reference to language model head.
    pub fn lm_head_mut(&mut self) -> &mut Linear {
        &mut self.lm_head
    }

    /// Get reference to language model head (for testing/inspection).
    #[must_use]
    pub fn lm_head(&self) -> &Linear {
        &self.lm_head
    }

    // ========================================================================
    // SafeTensors Loading (Section A: Model Loading)
    // ========================================================================

    /// Load weights from `SafeTensors` format.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to .safetensors file
    ///
    /// # Returns
    ///
    /// Number of weights loaded
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be read or weights don't match.
    pub fn load_from_safetensors(&mut self, path: &std::path::Path) -> Result<usize, String> {
        use crate::serialization::safetensors::MappedSafeTensors;

        // Use mmap for zero-copy loading (per Native Library Mandate)
        let mapped = MappedSafeTensors::open(path)?;
        let mut loaded_count = 0;

        // Helper to load a tensor by name
        let load_tensor = |name: &str| -> Result<Tensor, String> {
            let meta = mapped
                .get_metadata(name)
                .ok_or_else(|| format!("Weight '{name}' not found in SafeTensors file"))?;
            let data = mapped.get_tensor(name)?;
            Ok(Tensor::new(&data, &meta.shape))
        };

        // Load embedding weights
        if let Ok(t) = load_tensor("model.embed_tokens.weight") {
            self.embed_tokens.set_weight(t);
            loaded_count += 1;
        }

        // Load decoder layer weights
        for i in 0..self.layers.len() {
            let prefix = format!("model.layers.{i}");
            let layer = self.layers.get_mut(i).ok_or("Layer index out of bounds")?;

            // Attention projections
            if let Ok(t) = load_tensor(&format!("{prefix}.self_attn.q_proj.weight")) {
                layer.self_attn_mut().q_proj_mut().set_weight(t);
                loaded_count += 1;
            }
            if let Ok(t) = load_tensor(&format!("{prefix}.self_attn.k_proj.weight")) {
                layer.self_attn_mut().k_proj_mut().set_weight(t);
                loaded_count += 1;
            }
            if let Ok(t) = load_tensor(&format!("{prefix}.self_attn.v_proj.weight")) {
                layer.self_attn_mut().v_proj_mut().set_weight(t);
                loaded_count += 1;
            }
            if let Ok(t) = load_tensor(&format!("{prefix}.self_attn.o_proj.weight")) {
                layer.self_attn_mut().out_proj_mut().set_weight(t);
                loaded_count += 1;
            }

            // MLP projections
            if let Ok(t) = load_tensor(&format!("{prefix}.mlp.gate_proj.weight")) {
                layer.mlp_mut().gate_proj_mut().set_weight(t);
                loaded_count += 1;
            }
            if let Ok(t) = load_tensor(&format!("{prefix}.mlp.up_proj.weight")) {
                layer.mlp_mut().up_proj_mut().set_weight(t);
                loaded_count += 1;
            }
            if let Ok(t) = load_tensor(&format!("{prefix}.mlp.down_proj.weight")) {
                layer.mlp_mut().down_proj_mut().set_weight(t);
                loaded_count += 1;
            }

            // Layer norms
            if let Ok(t) = load_tensor(&format!("{prefix}.input_layernorm.weight")) {
                layer.input_layernorm_mut().set_weight(t);
                loaded_count += 1;
            }
            if let Ok(t) = load_tensor(&format!("{prefix}.post_attention_layernorm.weight")) {
                layer.post_attention_layernorm_mut().set_weight(t);
                loaded_count += 1;
            }
        }

        // Final norm
        if let Ok(t) = load_tensor("model.norm.weight") {
            self.norm.set_weight(t);
            loaded_count += 1;
        }

        // LM head
        // Note: Qwen2 uses weight tying - lm_head shares weights with embed_tokens
        if let Ok(t) = load_tensor("lm_head.weight") {
            self.lm_head.set_weight(t);
            loaded_count += 1;
        } else if let Ok(t) = load_tensor("model.embed_tokens.weight") {
            // Weight tying fallback: use embed_tokens.weight for lm_head
            // This is common in Qwen2 and many transformer models
            self.lm_head.set_weight(t);
            loaded_count += 1;
        }

        Ok(loaded_count)
    }

    /// Load model from `SafeTensors` file.
    ///
    /// Creates a new model with the given config and loads weights from file.
    pub fn from_safetensors(config: &Qwen2Config, path: &std::path::Path) -> Result<Self, String> {
        let mut model = Self::new(config);
        model.load_from_safetensors(path)?;
        Ok(model)
    }

    /// Load weights from APR v2 format file.
    ///
    /// Per Native Library Mandate (Spec §2.4): Uses mmap via `bundle::MappedFile`
    /// for zero-copy tensor access. This is the REQUIRED approach for APR files.
    ///
    /// Note: APR canonical names don't have the "model." prefix (it's stripped
    /// during import per format/converter.rs). We look for names without prefix.
    ///
    /// # Returns
    ///
    /// Number of weights loaded
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be read or weights don't match.
    pub fn load_from_apr(&mut self, path: &std::path::Path) -> Result<usize, String> {
        use crate::bundle::MappedFile;
        use crate::format::v2::AprV2ReaderRef;

        // Use mmap for zero-copy loading (per Native Library Mandate)
        let mapped = MappedFile::open(path).map_err(|e| format!("mmap failed: {e}"))?;
        // Use AprV2ReaderRef for zero-copy - does NOT copy the mmap data!
        let reader = AprV2ReaderRef::from_bytes(mapped.as_slice())
            .map_err(|e| format!("APR parse failed: {e}"))?;

        let mut loaded_count = 0;

        // Helper to load a tensor by name
        // APR uses canonical names without "model." prefix
        let load_tensor = |name: &str| -> Result<Tensor, String> {
            let entry = reader
                .get_tensor(name)
                .ok_or_else(|| format!("Weight '{name}' not found in APR file"))?;
            let data = reader
                .get_f32_tensor(name)
                .ok_or_else(|| format!("Failed to read f32 data for '{name}'"))?;
            Ok(Tensor::new(&data, &entry.shape))
        };

        // Load embedding weights (APR uses "embed_tokens.weight" not "model.embed_tokens.weight")
        if let Ok(t) = load_tensor("embed_tokens.weight") {
            self.embed_tokens.set_weight(t);
            loaded_count += 1;
        }

        // Load decoder layer weights (APR uses "layers.N" not "model.layers.N")
        for i in 0..self.layers.len() {
            let prefix = format!("layers.{i}");
            let layer = self.layers.get_mut(i).ok_or("Layer index out of bounds")?;

            // Attention projections
            if let Ok(t) = load_tensor(&format!("{prefix}.self_attn.q_proj.weight")) {
                layer.self_attn_mut().q_proj_mut().set_weight(t);
                loaded_count += 1;
            }
            if let Ok(t) = load_tensor(&format!("{prefix}.self_attn.k_proj.weight")) {
                layer.self_attn_mut().k_proj_mut().set_weight(t);
                loaded_count += 1;
            }
            if let Ok(t) = load_tensor(&format!("{prefix}.self_attn.v_proj.weight")) {
                layer.self_attn_mut().v_proj_mut().set_weight(t);
                loaded_count += 1;
            }
            if let Ok(t) = load_tensor(&format!("{prefix}.self_attn.o_proj.weight")) {
                layer.self_attn_mut().out_proj_mut().set_weight(t);
                loaded_count += 1;
            }

            // MLP projections
            if let Ok(t) = load_tensor(&format!("{prefix}.mlp.gate_proj.weight")) {
                layer.mlp_mut().gate_proj_mut().set_weight(t);
                loaded_count += 1;
            }
            if let Ok(t) = load_tensor(&format!("{prefix}.mlp.up_proj.weight")) {
                layer.mlp_mut().up_proj_mut().set_weight(t);
                loaded_count += 1;
            }
            if let Ok(t) = load_tensor(&format!("{prefix}.mlp.down_proj.weight")) {
                layer.mlp_mut().down_proj_mut().set_weight(t);
                loaded_count += 1;
            }

            // Layer norms
            if let Ok(t) = load_tensor(&format!("{prefix}.input_layernorm.weight")) {
                layer.input_layernorm_mut().set_weight(t);
                loaded_count += 1;
            }
            if let Ok(t) = load_tensor(&format!("{prefix}.post_attention_layernorm.weight")) {
                layer.post_attention_layernorm_mut().set_weight(t);
                loaded_count += 1;
            }
        }

        // Final norm (APR uses "norm.weight" not "model.norm.weight")
        if let Ok(t) = load_tensor("norm.weight") {
            self.norm.set_weight(t);
            loaded_count += 1;
        }

        // LM head (this one doesn't have "model." prefix even in SafeTensors)
        // Note: Qwen2 uses weight tying - lm_head shares weights with embed_tokens
        if let Ok(t) = load_tensor("lm_head.weight") {
            self.lm_head.set_weight(t);
            loaded_count += 1;
        } else {
            // Weight tying: use embed_tokens.weight for lm_head
            // This is common in Qwen2 and many transformer models
            if let Ok(t) = load_tensor("embed_tokens.weight") {
                self.lm_head.set_weight(t);
                loaded_count += 1;
            }
        }

        Ok(loaded_count)
    }

    /// Load model from APR v2 format file.
    ///
    /// Creates a new model with the given config and loads weights from file.
    pub fn from_apr(config: &Qwen2Config, path: &std::path::Path) -> Result<Self, String> {
        let mut model = Self::new(config);
        model.load_from_apr(path)?;
        Ok(model)
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// `SiLU` (Swish) activation: x * sigmoid(x)
/// Uses SIMD-accelerated Tensor ops instead of naive iterators.
fn silu(x: &Tensor) -> Tensor {
    // SiLU(x) = x * sigmoid(x)
    x.mul(&x.sigmoid())
}

/// Element-wise multiplication (SIMD-accelerated).
fn elementwise_mul(a: &Tensor, b: &Tensor) -> Tensor {
    a.mul(b)
}

/// Element-wise addition (SIMD-accelerated).
fn add_tensors(a: &Tensor, b: &Tensor) -> Tensor {
    a.add(b)
}

#[cfg(test)]
mod tests;
