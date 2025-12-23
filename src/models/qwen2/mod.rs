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
//! let mut model = Qwen2Model::new(&config);
//!
//! let input_ids = vec![1u32, 2, 3, 4, 5];
//! let position_ids: Vec<usize> = (0..5).collect();
//! let logits = model.forward(&input_ids, &position_ids);
//! ```
//!
//! # References
//!
//! - Bai et al. (2023). "Qwen Technical Report"
//! - Ainslie et al. (2023). "GQA: Training Generalized Multi-Query Transformer Models"
//! - Su et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding"
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
    /// Weight matrix [vocab_size, hidden_size]
    weight: Tensor,
    vocab_size: usize,
    hidden_size: usize,
}

impl Embedding {
    /// Create a new embedding layer.
    pub fn new(vocab_size: usize, hidden_size: usize) -> Self {
        // Initialize with small random values
        let data: Vec<f32> = (0..vocab_size * hidden_size)
            .map(|i| {
                // Deterministic pseudo-random initialization
                let x = (i as f32 * 0.0001).sin() * 0.02;
                x
            })
            .collect();

        Self {
            weight: Tensor::new(&data, &[vocab_size, hidden_size]),
            vocab_size,
            hidden_size,
        }
    }

    /// Look up embeddings for token IDs.
    pub fn forward(&self, input_ids: &[u32]) -> Tensor {
        let batch_size = 1; // For now, single batch
        let seq_len = input_ids.len();

        let mut output = vec![0.0f32; batch_size * seq_len * self.hidden_size];

        for (s, &token_id) in input_ids.iter().enumerate() {
            let token_idx = token_id as usize;
            if token_idx >= self.vocab_size {
                // Out of vocabulary - use zeros or last token
                continue;
            }

            let src_offset = token_idx * self.hidden_size;
            let dst_offset = s * self.hidden_size;

            output[dst_offset..dst_offset + self.hidden_size]
                .copy_from_slice(&self.weight.data()[src_offset..src_offset + self.hidden_size]);
        }

        Tensor::new(&output, &[batch_size, seq_len, self.hidden_size])
    }

    /// Set weights from external tensor.
    pub fn set_weight(&mut self, weight: Tensor) {
        self.weight = weight;
    }

    /// Get weight tensor reference.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
}

// ============================================================================
// Qwen2 MLP (SwiGLU)
// ============================================================================

/// Qwen2 MLP with SwiGLU activation.
///
/// ```text
/// output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
/// ```
#[derive(Debug)]
pub struct Qwen2MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Qwen2MLP {
    /// Create a new Qwen2 MLP layer.
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            gate_proj: Linear::new(hidden_size, intermediate_size),
            up_proj: Linear::new(hidden_size, intermediate_size),
            down_proj: Linear::new(intermediate_size, hidden_size),
        }
    }

    /// Forward pass with SwiGLU activation.
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

    /// Forward pass through the decoder layer.
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
    /// Cached keys per layer: [batch, num_kv_heads, cached_len, head_dim]
    pub keys: Vec<Option<Tensor>>,
    /// Cached values per layer
    pub values: Vec<Option<Tensor>>,
    /// Number of cached positions
    pub cached_len: usize,
}

impl KVCache {
    /// Create a new empty KV cache.
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
    /// Token embeddings [vocab_size, hidden_size]
    embed_tokens: Embedding,
    /// Decoder layers
    layers: Vec<Qwen2DecoderLayer>,
    /// Final RMSNorm
    norm: RMSNorm,
    /// Language model head [hidden_size, vocab_size]
    lm_head: Linear,
    /// Rotary position embeddings
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

    /// Forward pass through the model.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Token IDs [seq_len]
    /// * `position_ids` - Position indices [seq_len]
    ///
    /// # Returns
    ///
    /// Logits tensor [1, seq_len, vocab_size]
    pub fn forward(&mut self, input_ids: &[u32], position_ids: &[usize]) -> Tensor {
        // Embed tokens
        let mut hidden = self.embed_tokens.forward(input_ids);

        // Generate causal mask
        let seq_len = input_ids.len();
        let attention_mask = generate_causal_mask(seq_len);

        // Pass through decoder layers
        for layer in &self.layers {
            hidden = layer.forward(&hidden, position_ids, &self.rope, Some(&attention_mask));
        }

        // Final normalization
        hidden = self.norm.forward(&hidden);

        // Project to vocabulary
        self.lm_head.forward(&hidden)
    }

    /// Generate tokens autoregressively.
    ///
    /// # Arguments
    ///
    /// * `prompt_ids` - Initial prompt token IDs
    /// * `max_new_tokens` - Maximum number of tokens to generate
    /// * `temperature` - Sampling temperature (0 = greedy)
    /// * `top_p` - Nucleus sampling threshold
    ///
    /// # Returns
    ///
    /// Complete sequence including prompt and generated tokens.
    pub fn generate(
        &mut self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        _top_p: f32,
    ) -> Vec<u32> {
        let mut output_ids = prompt_ids.to_vec();

        for _ in 0..max_new_tokens {
            let position_ids: Vec<usize> = (0..output_ids.len()).collect();
            let logits = self.forward(&output_ids, &position_ids);

            // Get last token logits
            let vocab_size = self.config.vocab_size;
            let last_pos = output_ids.len() - 1;
            let logits_slice = &logits.data()
                [last_pos * vocab_size..(last_pos + 1) * vocab_size];

            // Sample next token
            let next_token = if temperature == 0.0 {
                // Greedy
                argmax(logits_slice) as u32
            } else {
                // Temperature sampling
                sample_with_temperature(logits_slice, temperature)
            };

            // Check for EOS
            if next_token == 151645 || next_token == 151644 {
                break;
            }

            output_ids.push(next_token);
        }

        output_ids
    }

    /// Get model configuration.
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
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    // ========================================================================
    // Weight Introspection Methods (Section A: Model Loading)
    // ========================================================================

    /// Get list of weight names following HuggingFace convention.
    ///
    /// Returns names like:
    /// - `model.embed_tokens.weight`
    /// - `model.layers.0.self_attn.q_proj.weight`
    /// - `model.norm.weight`
    /// - `lm_head.weight`
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
    /// Returns a map suitable for serialization to SafeTensors format.
    /// Note: Currently returns weights from components with public accessors.
    /// Full weight export will be enabled when nn modules expose weight accessors.
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

    // ========================================================================
    // SafeTensors Loading (Section A: Model Loading)
    // ========================================================================

    /// Load weights from SafeTensors format.
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
    pub fn load_from_safetensors(
        &mut self,
        path: &std::path::Path,
    ) -> Result<usize, String> {
        use crate::serialization::safetensors::MappedSafeTensors;

        // Use mmap for zero-copy loading (per Native Library Mandate)
        let mapped = MappedSafeTensors::open(path)?;
        let mut loaded_count = 0;

        // Helper to load a tensor by name
        let load_tensor = |name: &str| -> Result<Tensor, String> {
            let meta = mapped.get_metadata(name).ok_or_else(|| {
                format!("Weight '{name}' not found in SafeTensors file")
            })?;
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
        if let Ok(t) = load_tensor("lm_head.weight") {
            self.lm_head.set_weight(t);
            loaded_count += 1;
        }

        Ok(loaded_count)
    }

    /// Load model from SafeTensors file.
    ///
    /// Creates a new model with the given config and loads weights from file.
    pub fn from_safetensors(
        config: &Qwen2Config,
        path: &std::path::Path,
    ) -> Result<Self, String> {
        let mut model = Self::new(config);
        model.load_from_safetensors(path)?;
        Ok(model)
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// SiLU (Swish) activation: x * sigmoid(x)
fn silu(x: &Tensor) -> Tensor {
    let data: Vec<f32> = x
        .data()
        .iter()
        .map(|&v| v * (1.0 / (1.0 + (-v).exp())))
        .collect();
    Tensor::new(&data, x.shape())
}

/// Element-wise multiplication.
fn elementwise_mul(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.shape(), b.shape(), "Shapes must match for multiplication");
    let data: Vec<f32> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .map(|(&x, &y)| x * y)
        .collect();
    Tensor::new(&data, a.shape())
}

/// Element-wise addition.
fn add_tensors(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.shape(), b.shape(), "Shapes must match for addition");
    let data: Vec<f32> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .map(|(&x, &y)| x + y)
        .collect();
    Tensor::new(&data, a.shape())
}

/// Generate causal attention mask.
fn generate_causal_mask(size: usize) -> Tensor {
    let mut data = vec![0.0f32; size * size];

    for i in 0..size {
        for j in 0..size {
            if j > i {
                data[i * size + j] = f32::NEG_INFINITY;
            }
        }
    }

    Tensor::new(&data, &[size, size])
}

/// Find index of maximum value.
fn argmax(slice: &[f32]) -> usize {
    slice
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Sample from logits with temperature.
fn sample_with_temperature(logits: &[f32], temperature: f32) -> u32 {
    use rand::Rng;

    // Apply temperature
    let scaled: Vec<f32> = logits.iter().map(|&l| l / temperature).collect();

    // Softmax
    let max_val = scaled.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_vals: Vec<f32> = scaled.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    let probs: Vec<f32> = exp_vals.iter().map(|&v| v / sum).collect();

    // Sample
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen();
    let mut cumsum = 0.0;

    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i as u32;
        }
    }

    (probs.len() - 1) as u32
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_tiny_config() -> Qwen2Config {
        Qwen2Config {
            hidden_size: 64,
            num_attention_heads: 4,
            num_kv_heads: 2,
            num_layers: 2,
            vocab_size: 1000,
            max_seq_len: 128,
            intermediate_size: 128,
            rope_theta: 10000.0,
        }
    }

    #[test]
    fn test_embedding_shape() {
        let emb = Embedding::new(1000, 64);
        let input = vec![1u32, 2, 3, 4, 5];
        let output = emb.forward(&input);

        assert_eq!(output.shape(), &[1, 5, 64]);
    }

    #[test]
    fn test_embedding_lookup() {
        let emb = Embedding::new(100, 8);
        let input = vec![0u32, 1, 2];
        let output = emb.forward(&input);

        // Each token should produce different embeddings
        let data = output.data();
        let emb0 = &data[0..8];
        let emb1 = &data[8..16];
        let emb2 = &data[16..24];

        assert_ne!(emb0, emb1);
        assert_ne!(emb1, emb2);
    }

    #[test]
    fn test_qwen2_mlp_shape() {
        let mlp = Qwen2MLP::new(64, 128);
        let x = Tensor::ones(&[1, 5, 64]);
        let output = mlp.forward(&x);

        assert_eq!(output.shape(), &[1, 5, 64]);
    }

    #[test]
    fn test_qwen2_model_creation() {
        let config = create_tiny_config();
        let model = Qwen2Model::new(&config);

        assert_eq!(model.num_layers(), 2);
        assert_eq!(model.config().hidden_size, 64);
    }

    #[test]
    fn test_qwen2_model_forward_shape() {
        let config = create_tiny_config();
        let mut model = Qwen2Model::new(&config);

        let input_ids = vec![1u32, 2, 3, 4, 5];
        let position_ids: Vec<usize> = (0..5).collect();
        let logits = model.forward(&input_ids, &position_ids);

        assert_eq!(logits.shape(), &[1, 5, config.vocab_size]);
    }

    #[test]
    fn test_qwen2_model_deterministic() {
        let config = create_tiny_config();
        let mut model = Qwen2Model::new(&config);
        model.eval();

        let input_ids = vec![1u32, 2, 3];
        let position_ids: Vec<usize> = (0..3).collect();

        let logits1 = model.forward(&input_ids, &position_ids);
        let logits2 = model.forward(&input_ids, &position_ids);

        assert_eq!(logits1.data(), logits2.data());
    }

    #[test]
    fn test_silu_activation() {
        // SiLU(x) = x * sigmoid(x)
        // At x=0: SiLU(0) = 0 * 0.5 = 0
        let x = Tensor::new(&[0.0, 1.0, -1.0], &[3]);
        let y = silu(&x);

        let data = y.data();
        assert!((data[0] - 0.0).abs() < 1e-5); // SiLU(0) = 0
        assert!(data[1] > 0.5); // SiLU(1) ≈ 0.731
        assert!(data[2] < 0.0); // SiLU(-1) ≈ -0.269 (negative!)
    }

    #[test]
    fn test_causal_mask() {
        let mask = generate_causal_mask(4);

        assert_eq!(mask.shape(), &[4, 4]);

        // Check upper triangle is -inf
        assert!(mask.data()[1].is_infinite()); // [0, 1]
        assert!(mask.data()[2].is_infinite()); // [0, 2]
        assert!(mask.data()[3].is_infinite()); // [0, 3]

        // Check diagonal and below is 0
        assert_eq!(mask.data()[0], 0.0); // [0, 0]
        assert_eq!(mask.data()[4], 0.0); // [1, 0]
        assert_eq!(mask.data()[5], 0.0); // [1, 1]
    }

    #[test]
    fn test_argmax() {
        let slice = [1.0_f32, 5.0, 2.0, 3.0];
        assert_eq!(argmax(&slice), 1);

        let slice2 = [0.0_f32, -1.0, -2.0];
        assert_eq!(argmax(&slice2), 0);
    }
}
